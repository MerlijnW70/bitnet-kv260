`timescale 1ns/1ps
`ifndef SLICES
 `define SLICES 2
`endif
module tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rstn;
    reg  [127:0] s_tdata;
    reg          s_tvalid, s_tlast;
    wire         s_tready;
    wire  [31:0] m_tdata;
    wire         m_tvalid, m_tlast;
    reg          m_tready;
    reg   [31:0] ctrl;
    wire  [31:0] status;

    ternary_matvec #(.SLICES(`SLICES)) dut (
        .clk(clk), .rstn(rstn),
        .s_axis_tdata(s_tdata), .s_axis_tvalid(s_tvalid), .s_axis_tready(s_tready), .s_axis_tlast(s_tlast),
        .m_axis_tdata(m_tdata), .m_axis_tvalid(m_tvalid), .m_axis_tready(m_tready), .m_axis_tlast(m_tlast),
        .ctrl(ctrl), .status(status)
    );

    reg [127:0] act [0:1739];
    reg [127:0] wts [0:2047];
    reg  [31:0] yall[0:63];
    reg  [31:0] ys  [0:63];
    reg  [31:0] got [0:63];
    integer ngot, nlast, wrong, total, faults;
    reg stalls;
    integer t_first, t_last;

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
        if (m_tvalid && !m_tready && s_tready && !dut.room) begin
            faults = faults + 1; $display("FAULT at %0t: s_axis_tready high while the output queue has no room", $time);
        end
        m_tvalid_q <= m_tvalid; m_tready_q <= m_tready; m_tdata_q <= m_tdata; m_tlast_q <= m_tlast;
        if (m_tvalid_q && !m_tready_q && !m_tvalid && rstn && ctrl[0]) begin
            faults = faults + 1; $display("FAULT at %0t: m_axis_tvalid dropped before its handshake", $time);
        end
        if (m_tvalid_q && !m_tready_q && m_tvalid && (m_tdata !== m_tdata_q || m_tlast !== m_tlast_q) && rstn && ctrl[0]) begin
            faults = faults + 1; $display("FAULT at %0t: a held sum changed before its handshake", $time);
        end
    end

    task send(input sel, input integer base, input integer count, input gaps);
        integer k; reg go;
        begin
            k = 0; go = 0;
            while (k < count) begin
                if (!go) go = gaps ? ($random % 3 != 0) : 1'b1;
                s_tvalid = go;
                s_tdata  = sel ? wts[base + k] : act[base + k];
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

    task start_run(input mode, input [15:0] n, input [6:0] b, input [2:0] batch);
        begin
            ctrl = {n, 5'b0, batch[1:0] - 2'd1, b, mode, 1'b1};
            @(posedge clk); #1;
        end
    endtask

    reg [31:0] done_status;
    task finish_run;
        begin
            wait (status[31] == 1'b1); @(posedge clk); #1;
            done_status = status;
            ctrl = 0; @(posedge clk); #1; @(posedge clk); #1;
        end
    endtask

    task expect_beats(input [95:0] name, input integer want);
        begin
            total = total + 1;
            if (done_status[26:16] != want) begin
                wrong = wrong + 1;
                $display("  %0s: activation beats stored %0d, expected %0d", name, done_status[26:16], want);
            end
        end
    endtask

    task expect_neurons(input [95:0] name, input integer want);
        begin
            total = total + 1;
            if (done_status[15:0] != want) begin
                wrong = wrong + 1;
                $display("  %0s: neurons done %0d, expected %0d", name, done_status[15:0], want);
            end
        end
    endtask

    task expect_batch(input integer nneur, input integer b, input integer vecs);
        integer n, v;
        begin
            for (n = 0; n < nneur; n = n + 1)
                for (v = 0; v < b; v = v + 1)
                    ys[n * b + v] = yall[n * vecs + v];
        end
    endtask

    task check(input [95:0] name, input integer n);
        integer i, bad;
        begin
            bad = 0;
            for (i = 0; i < n; i = i + 1) begin
                total = total + 1;
                if (got[i] !== ys[i]) begin
                    bad = bad + 1;
                    if (bad <= 5) $display("  %0s sum %0d: fabric %08x expected %08x", name, i, got[i], ys[i]);
                end
            end
            if (ngot != n) begin bad = bad + 1; $display("  %0s: %0d sums out, expected %0d", name, ngot, n); end
            if (nlast != n - 1) begin bad = bad + 1; $display("  %0s: tlast on sum %0d, expected %0d", name, nlast, n - 1); end
            wrong = wrong + bad;
            $display("%0s: %0d sums out, tlast on %0d, status while done %08x, %0d wrong", name, ngot, nlast, done_status, bad);
        end
    endtask

    task load_set(input [95:0] set);
        begin
            if (set == "rand32") begin
                $readmemh("vectors/rand32_act.hex", act, 0, 159);
                $readmemh("vectors/rand32_w.hex", wts, 0, 32 * 32 - 1);
                $readmemh("vectors/rand32_y.hex", yall, 0, 31);
            end else if (set == "k6912") begin
                $readmemh("vectors/k6912_act.hex", act, 0, 434);
                $readmemh("vectors/k6912_w.hex", wts, 0, 8 * 87 - 1);
                $readmemh("vectors/k6912_y.hex", yall, 0, 7);
            end else if (set == "batch32") begin
                $readmemh("vectors/batch32_act.hex", act, 0, 639);
                $readmemh("vectors/batch32_w.hex", wts, 0, 8 * 32 - 1);
                $readmemh("vectors/batch32_y.hex", yall, 0, 31);
            end else if (set == "bk6912") begin
                $readmemh("vectors/bk6912_act.hex", act, 0, 1739);
                $readmemh("vectors/bk6912_w.hex", wts, 0, 4 * 87 - 1);
                $readmemh("vectors/bk6912_y.hex", yall, 0, 15);
            end else begin
                $readmemh("vectors/gate64_act.hex", act, 0, 159);
                $readmemh("vectors/gate64_w.hex", wts, 0, 64 * 32 - 1);
                $readmemh("vectors/gate64_y.hex", yall, 0, 63);
            end
        end
    endtask

    integer b;
    initial begin
        rstn = 0; s_tvalid = 0; s_tlast = 0; s_tdata = 0; ctrl = 0; stalls = 0;
        ngot = 0; nlast = -1; wrong = 0; total = 0; faults = 0;
        $display("SLICES = %0d", `SLICES);
        #22; rstn = 1; #10;
        if (status !== 32'h80000000) begin wrong = wrong + 1; $display("status after reset %08x, expected 80000000", status); end

        load_set("rand32"); expect_batch(32, 1, 1);
        start_run(0, 0, 7'd0, 1); send(0, 0, 160, 1); finish_run;
        expect_beats("rand32", 160);
        ngot = 0; nlast = -1; stalls = 1;
        start_run(1, 32, 7'd0, 1); send(1, 0, 32 * 32, 1); finish_run;
        stalls = 0;
        expect_neurons("rand32", 32);
        if (status !== 32'h80000000) begin wrong = wrong + 1; $display("status after run dropped %08x, expected 80000000", status); end
        check("rand32", 32);

        load_set("gate64"); expect_batch(64, 1, 1);
        start_run(0, 0, 7'd32, 1); send(0, 0, 160, 0); finish_run;
        expect_beats("gate64", 160);
        ngot = 0; nlast = -1;
        start_run(1, 64, 7'd32, 1); send(1, 0, 64 * 32, 0); finish_run;
        check("gate64", 64);
        $display("gate64: %0d clocks from the first weight beat taken to the last sum taken (2048 beats)", (t_last - t_first) / 10);

        ngot = 0; nlast = -1;
        start_run(1, 1, 7'd32, 1); send(1, 0, 32, 0); finish_run;
        check("one", 1);

        ngot = 0; nlast = -1;
        start_run(1, 64, 7'd32, 1); send(1, 0, 30 * 32 + 17, 0);
        ctrl = 0; @(posedge clk); #1; @(posedge clk); #1;
        if (status !== 32'h80000000) begin wrong = wrong + 1; $display("status after abort %08x, expected 80000000", status); end
        ngot = 0; nlast = -1;
        start_run(1, 64, 7'd32, 1); send(1, 0, 64 * 32, 0); finish_run;
        check("after-abort", 64);

        load_set("k6912"); expect_batch(8, 1, 1);
        start_run(0, 0, 7'd87, 1); send(0, 0, 435, 1); finish_run;
        expect_beats("k6912", 435);
        ngot = 0; nlast = -1; stalls = 1;
        start_run(1, 8, 7'd87, 1); send(1, 0, 8 * 87, 1); finish_run;
        stalls = 0;
        expect_neurons("k6912", 8);
        check("k6912", 8);

        ngot = 0; nlast = -1;
        start_run(1, 8, 7'd87, 1); send(1, 0, 8 * 87, 0); finish_run;
        check("k6912-full", 8);
        $display("k6912: %0d clocks from the first weight beat taken to the last sum taken (696 beats)", (t_last - t_first) / 10);

        load_set("batch32");
        for (b = 1; b <= 4; b = b + 1) begin
            expect_batch(8, b, 4);
            start_run(0, 0, 7'd32, b[2:0]); send(0, 0, 160 * b, (b == 4)); finish_run;
            expect_beats("batch32-load", 160 * b);
            ngot = 0; nlast = -1; stalls = (b == 4);
            start_run(1, 8, 7'd32, b[2:0]); send(1, 0, 8 * 32, (b == 4)); finish_run;
            stalls = 0;
            expect_neurons("batch32", 8);
            check(b == 1 ? "batch32-b1" : b == 2 ? "batch32-b2" : b == 3 ? "batch32-b3" : "batch32-b4", 8 * b);
        end

        expect_batch(8, 4, 4);
        ngot = 0; nlast = -1;
        start_run(1, 8, 7'd32, 3'd4); send(1, 0, 8 * 32, 0); finish_run;
        check("b32-b4-full", 32);
        $display("batch32 batch 4: %0d clocks from the first weight beat taken to the last sum taken (256 beats)", (t_last - t_first) / 10);
        ngot = 0; nlast = -1;
        start_run(1, 8, 7'd32, 3'd4); send(1, 0, 5 * 32 + 9, 0);
        ctrl = 0; @(posedge clk); #1; @(posedge clk); #1;
        if (status !== 32'h80000000) begin wrong = wrong + 1; $display("status after a batch abort %08x, expected 80000000", status); end
        ngot = 0; nlast = -1;
        start_run(1, 8, 7'd32, 3'd4); send(1, 0, 8 * 32, 0); finish_run;
        check("b32-reabort", 32);

        load_set("bk6912"); expect_batch(4, 4, 4);
        start_run(0, 0, 7'd87, 3'd4); send(0, 0, 1740, 1); finish_run;
        expect_beats("bk6912-load", 1740);
        ngot = 0; nlast = -1; stalls = 1;
        start_run(1, 4, 7'd87, 3'd4); send(1, 0, 4 * 87, 1); finish_run;
        stalls = 0;
        expect_neurons("bk6912", 4);
        check("bk6912", 16);

        $display("checked %0d values, %0d wrong, %0d protocol faults", total, wrong, faults);
        $finish;
    end
endmodule
