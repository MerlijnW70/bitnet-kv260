`timescale 1ns/1ps
module tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rstn;
    reg  [127:0] s_tdata;
    reg          s_tvalid, s_tlast;
    wire         s_tready;
    wire  [31:0] m_tdata;
    wire   [3:0] m_tkeep;
    wire         m_tvalid, m_tlast;
    reg          m_tready;
    reg   [31:0] ctrl, params, mval;
    wire  [31:0] status;

    ternary_glue_axi dut (
        .aclk(clk), .aresetn(rstn),
        .s_axis_tdata(s_tdata), .s_axis_tkeep(16'hffff), .s_axis_tvalid(s_tvalid), .s_axis_tready(s_tready), .s_axis_tlast(s_tlast),
        .m_axis_tdata(m_tdata), .m_axis_tkeep(m_tkeep), .m_axis_tvalid(m_tvalid), .m_axis_tready(m_tready), .m_axis_tlast(m_tlast),
        .ctrl(ctrl), .params(params), .mval(mval), .status(status)
    );

    reg [127:0] ain   [0:6911];
    reg  [31:0] ah    [0:6911];
    reg [127:0] bin   [0:1727];
    reg  [31:0] bq    [0:1727];
    reg  [31:0] words [0:3];
    reg  [31:0] got   [0:6911];
    integer ngot, nlast, wrong, total, faults, k, n_set;
    reg stalls;
    integer t_first, t_last;
    reg [31:0] done_status;

    always @(posedge clk) m_tready <= stalls ? ($random % 4 != 0) : 1'b1;
    reg m_tvalid_q, m_tready_q, m_tlast_q;
    reg [31:0] m_tdata_q;
    always @(posedge clk) begin
        if (m_tvalid && m_tready) begin
            got[ngot] = m_tdata;
            if (m_tlast) nlast = ngot;
            t_last = $time;
            ngot = ngot + 1;
        end
        m_tvalid_q <= m_tvalid; m_tready_q <= m_tready; m_tdata_q <= m_tdata; m_tlast_q <= m_tlast;
        if (m_tvalid_q && !m_tready_q && !m_tvalid && rstn && ctrl[0]) begin
            faults = faults + 1; $display("FAULT at %0t: m_axis_tvalid dropped before its handshake", $time);
        end
        if (m_tvalid_q && !m_tready_q && m_tvalid && (m_tdata !== m_tdata_q || m_tlast !== m_tlast_q) && rstn && ctrl[0]) begin
            faults = faults + 1; $display("FAULT at %0t: a held beat changed before its handshake", $time);
        end
        if (m_tkeep !== 4'b1111) begin faults = faults + 1; $display("FAULT at %0t: m_axis_tkeep %b", $time, m_tkeep); end
    end

    task send(input sel, input integer count, input gaps);
        integer k; reg go;
        begin
            k = 0; go = 0;
            while (k < count) begin
                if (!go) go = gaps ? ($random % 3 != 0) : 1'b1;
                s_tvalid = go;
                s_tdata  = sel ? bin[k] : ain[k];
                s_tlast  = (k == count - 1);
                @(posedge clk);
                if (go && s_tready) begin
                    if (k == 0) t_first = $time;
                    k = k + 1; go = 0;
                end
                #1;
            end
            s_tvalid = 0; s_tlast = 0; s_tdata = 0;
        end
    endtask

    task start_run(input mode, input [15:0] n);
        begin
            ctrl = {n, 14'b0, mode, 1'b1};
            @(posedge clk); #1;
        end
    endtask

    task finish_run;
        begin
            wait (status[31] == 1'b1); @(posedge clk); #1;
            done_status = status;
            ctrl = 0; @(posedge clk); #1; @(posedge clk); #1;
        end
    endtask

    task load_set(input string name);
        begin
            $readmemh({"vectors/glue_", name, "_words.hex"}, words);
            n_set = words[0];
            params = words[1];
            mval = words[2];
            $readmemh({"vectors/glue_", name, "_a_in.hex"}, ain, 0, n_set - 1);
            $readmemh({"vectors/glue_", name, "_a_h.hex"}, ah, 0, n_set - 1);
            $readmemh({"vectors/glue_", name, "_b_in.hex"}, bin, 0, (n_set + 3) / 4 - 1);
            $readmemh({"vectors/glue_", name, "_b_q.hex"}, bq, 0, (n_set + 3) / 4 - 1);
        end
    endtask

    task check_a(input string name, input integer n);
        integer i, bad;
        begin
            bad = 0;
            for (i = 0; i < n; i = i + 1) begin
                total = total + 1;
                if (got[i] !== ah[i]) begin
                    bad = bad + 1;
                    if (bad <= 5) $display("  %0s element %0d: fabric h %08x expected %08x (in %032x)", name, i, got[i], ah[i], ain[i]);
                end
            end
            if (ngot != n) begin bad = bad + 1; $display("  %0s: %0d beats out, expected %0d", name, ngot, n); end
            if (nlast != n - 1) begin bad = bad + 1; $display("  %0s: tlast on beat %0d, expected %0d", name, nlast, n - 1); end
            if (done_status[30:16] != n) begin bad = bad + 1; $display("  %0s: elements done %0d, expected %0d", name, done_status[30:16], n); end
            if (done_status[14:0] != words[3][14:0]) begin bad = bad + 1; $display("  %0s: hmax %0d, expected %0d", name, done_status[14:0], words[3]); end
            if (status !== {17'h10000, words[3][14:0]}) begin bad = bad + 1; $display("  %0s: status after run dropped %08x, expected idle with hmax kept", name, status); end
            wrong = wrong + bad;
            $display("%0s pass A: %0d elements, %0d beats out, tlast on %0d, hmax %0d, status while done %08x, %0d wrong", name, n, ngot, nlast, done_status[14:0], done_status, bad);
            $fflush;
        end
    endtask

    task check_b(input string name, input integer n);
        integer i, bad, beats;
        begin
            bad = 0;
            beats = (n + 3) / 4;
            for (i = 0; i < beats; i = i + 1) begin
                total = total + 4;
                if (got[i] !== bq[i]) begin
                    bad = bad + 1;
                    if (bad <= 5) $display("  %0s beat %0d: fabric q %08x expected %08x (in %032x)", name, i, got[i], bq[i], bin[i]);
                end
            end
            if (ngot != beats) begin bad = bad + 1; $display("  %0s: %0d beats out, expected %0d", name, ngot, beats); end
            if (nlast != beats - 1) begin bad = bad + 1; $display("  %0s: tlast on beat %0d, expected %0d", name, nlast, beats - 1); end
            if (done_status[30:16] != 4 * beats) begin bad = bad + 1; $display("  %0s: elements done %0d, expected %0d", name, done_status[30:16], 4 * beats); end
            if (done_status[14:0] != words[3][14:0]) begin bad = bad + 1; $display("  %0s: hmax %0d after pass B, expected %0d kept", name, done_status[14:0], words[3]); end
            wrong = wrong + bad;
            $display("%0s pass B: %0d elements, %0d beats out, tlast on %0d, status while done %08x, %0d wrong", name, n, ngot, nlast, done_status, bad);
            $fflush;
        end
    endtask

    task pass_a(input string name, input integer n, input gaps, input stall);
        begin
            ngot = 0; nlast = -1; stalls = stall;
            start_run(0, n); send(0, n, gaps); finish_run;
            stalls = 0;
            check_a(name, n);
        end
    endtask

    task pass_b(input string name, input integer n, input gaps, input stall);
        begin
            ngot = 0; nlast = -1; stalls = stall;
            start_run(1, n); send(1, (n + 3) / 4, gaps); finish_run;
            stalls = 0;
            check_b(name, n);
        end
    endtask

    task both(input string name, input gaps, input stall);
        begin
            load_set(name);
            pass_a(name, n_set, gaps, stall);
            pass_b(name, n_set, gaps, stall);
        end
    endtask

    integer e;
    string ext_name;
    initial begin
        rstn = 0; s_tvalid = 0; s_tlast = 0; s_tdata = 0; ctrl = 0; params = 0; mval = 0; stalls = 0;
        ngot = 0; nlast = -1; wrong = 0; total = 0; faults = 0;
        #22; rstn = 1;
        repeat (100) @(posedge clk); #1;
        if (status !== 32'h80000000) begin wrong = wrong + 1; $display("status after reset %08x, expected 80000000", status); end

        both("small8", 1, 1);
        load_set("one1");
        pass_a("one1", 1, 0, 0);
        both("one", 0, 0);

        for (e = 0; e < 23; e = e + 1) begin
            ext_name = $sformatf("ext%0d", e);
            both(ext_name, e % 2, e % 3 == 0);
        end

        ngot = 0; nlast = -1;
        start_run(0, 0); finish_run;
        if (ngot != 0 || done_status[31] != 1'b1 || done_status[30:16] != 0) begin wrong = wrong + 1; $display("zero: %0d beats out, status %08x", ngot, done_status); end
        else $display("zero: N = 0 done at once, status while done %08x, 0 wrong", done_status);
        $fflush;

        load_set("small8");
        ngot = 0; nlast = -1;
        start_run(0, 8); send(0, 8, 0);
        wait (status[31] == 1'b1); @(posedge clk); #1;
        for (k = 0; k < 8; k = k + 1) begin
            total = total + 1;
            if (got[k] !== ah[k]) begin wrong = wrong + 1; $display("held: element %0d: fabric h %08x expected %08x", k, got[k], ah[k]); end
        end
        ctrl = {16'd4, 14'b0, 1'b1, 1'b1};
        s_tvalid = 1; s_tdata = bin[0]; s_tlast = 1;
        repeat (50) begin
            @(posedge clk); #1;
            if (s_tready) begin wrong = wrong + 1; $display("held: s_axis_tready high with run held"); end
        end
        s_tvalid = 0; s_tdata = 0; s_tlast = 0;
        if (status[31] != 1'b1 || status[30:16] != 8 || ngot != 8) begin wrong = wrong + 1; $display("held: status %08x, %0d beats out, expected idle with 8 done", status, ngot); end
        else $display("held: run kept high with a new ctrl word: nothing started, status %08x, 0 wrong", status);
        ctrl = 0; @(posedge clk); #1; @(posedge clk); #1;
        $fflush;

        load_set("rand");
        pass_a("rand", n_set, 0, 0);
        $display("rand pass A: %0d clocks from the first beat taken to the last beat out (%0d elements)", (t_last - t_first) / 10, n_set);
        $fflush;
        pass_b("rand", n_set, 0, 0);
        $display("rand pass B: %0d clocks from the first beat taken to the last beat out (%0d elements)", (t_last - t_first) / 10, n_set);
        $fflush;
        pass_a("rand-gaps", n_set, 1, 1);
        pass_b("rand-gaps", n_set, 1, 1);

        both("tok0", 0, 0);

        load_set("rand");
        ngot = 0; nlast = -1;
        start_run(0, n_set); send(0, 1000, 0);
        repeat (20) @(posedge clk); #1;
        ctrl = 0; @(posedge clk); #1; @(posedge clk); #1;
        if (status[31] != 1'b1 || status[30:16] != 0) begin wrong = wrong + 1; $display("status after abort %08x, expected idle with 0 done", status); end
        else $display("abort: %0d beats out before the drop, status after %08x", ngot, status);
        pass_a("after-abort", n_set, 0, 0);
        pass_b("after-abort", n_set, 1, 1);

        $display("checked %0d elements, %0d wrong, %0d protocol faults", total, wrong, faults);
        $finish;
    end
endmodule
