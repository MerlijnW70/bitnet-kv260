`timescale 1ns/1ps
`default_nettype none

module tb_attn_fx_axi;
    reg clk = 0, rstn = 0;
    reg [127:0] s_tdata = 0;
    reg s_tvalid = 0;
    wire s_tready;
    wire [31:0] m_tdata;
    wire m_tvalid, m_tlast;
    reg m_tready = 0;
    wire [31:0] status;

    attn_fx_axi dut (.clk(clk), .rstn(rstn),
        .s_axis_tdata(s_tdata), .s_axis_tvalid(s_tvalid), .s_axis_tready(s_tready),
        .m_axis_tdata(m_tdata), .m_axis_tvalid(m_tvalid), .m_axis_tready(m_tready), .m_axis_tlast(m_tlast),
        .status(status));

    always #5 clk = ~clk;

    integer fd, r, h, j, g, t, T, layers, bad_top, bad_sum, bad_acc, bad_last, positions, tmp, got_n;
    reg [8*8-1:0] tag;
    reg signed [63:0] want;
    reg [15:0] qf [0:19];
    reg signed [23:0] tops [0:19];
    reg signed [7:0] q [0:20*128-1];
    reg [15:0] k16s [0:5*1100-1];
    reg [15:0] v16s [0:5*1100-1];
    reg signed [7:0] ks [0:5*1100*128-1];
    reg signed [7:0] vs [0:5*1100*128-1];
    reg [31:0] got [0:5139];
    reg        got_last [0:5139];
    reg [31:0] seed;

    always @(posedge clk) m_tready <= ($random(seed) & 3) != 0;

    task send(input [127:0] beat);
    begin
        s_tdata = beat;
        s_tvalid = 1;
        @(posedge clk);
        while (!s_tready) @(posedge clk);
        #1;
        s_tvalid = 0;
        if (($random(seed) & 7) == 0) begin @(posedge clk); #1; end
    end
    endtask

    task collect(input integer n);
    begin
        got_n = 0;
        while (got_n < n) begin
            @(posedge clk);
            if (m_tvalid && m_tready) begin
                got[got_n] = m_tdata;
                got_last[got_n] = m_tlast;
                got_n = got_n + 1;
            end
        end
    end
    endtask

    task run_pass(input b);
        reg [127:0] beat;
        reg [383:0] qfb;
        reg [511:0] topb;
    begin
        beat = 0;
        beat[15:0] = 16'hA77E;
        beat[16] = b;
        beat[47:32] = T;
        send(beat);
        qfb = 0;
        for (h = 0; h < 20; h = h + 1) qfb[16*h +: 16] = qf[h];
        for (j = 0; j < 3; j = j + 1) send(qfb[128*j +: 128]);
        topb = 0;
        for (h = 0; h < 20; h = h + 1) topb[24*h +: 24] = tops[h];
        for (j = 0; j < 4; j = j + 1) send(topb[128*j +: 128]);
        for (j = 0; j < 20 * 128; j = j + 1) begin
            beat[8*(j % 16) +: 8] = q[j];
            if (j % 16 == 15) send(beat);
        end
        for (t = 0; t < T; t = t + 1) begin
            beat = 0;
            for (g = 0; g < 5; g = g + 1) beat[16*g +: 16] = k16s[t * 5 + g];
            send(beat);
            beat = 0;
            for (g = 0; g < 5; g = g + 1) beat[16*g +: 16] = v16s[t * 5 + g];
            send(beat);
            for (j = 0; j < 5 * 128; j = j + 1) begin
                beat[8*(j % 16) +: 8] = ks[(t * 5 * 128) + j];
                if (j % 16 == 15) send(beat);
            end
            for (j = 0; j < 5 * 128; j = j + 1) begin
                beat[8*(j % 16) +: 8] = vs[(t * 5 * 128) + j];
                if (j % 16 == 15) send(beat);
            end
        end
    end
    endtask

    initial begin
        seed = 32'd20260919;
        layers = 0; bad_top = 0; bad_sum = 0; bad_acc = 0; bad_last = 0; positions = 0;
        repeat (3) @(posedge clk);
        #1 rstn = 1;
        fd = $fopen("attn_layers.txt", "r");
        r = $fscanf(fd, "%s", tag);
        while (tag != "END") begin
            r = $fscanf(fd, "%d", T);
            r = $fscanf(fd, "%s", tag);
            for (h = 0; h < 20; h = h + 1) begin r = $fscanf(fd, "%d", tmp); qf[h] = tmp; end
            r = $fscanf(fd, "%s", tag);
            for (j = 0; j < 20 * 128; j = j + 1) begin r = $fscanf(fd, "%d", tmp); q[j] = tmp; end
            for (t = 0; t < T; t = t + 1) begin
                r = $fscanf(fd, "%s", tag);
                for (g = 0; g < 5; g = g + 1) begin r = $fscanf(fd, "%d", tmp); k16s[t * 5 + g] = tmp; end
                for (g = 0; g < 5; g = g + 1) begin r = $fscanf(fd, "%d", tmp); v16s[t * 5 + g] = tmp; end
                for (j = 0; j < 5 * 128; j = j + 1) begin r = $fscanf(fd, "%d", tmp); ks[t * 5 * 128 + j] = tmp; end
                for (j = 0; j < 5 * 128; j = j + 1) begin r = $fscanf(fd, "%d", tmp); vs[t * 5 * 128 + j] = tmp; end
            end
            positions = positions + T;
            for (h = 0; h < 20; h = h + 1) tops[h] = 0;
            fork
                run_pass(0);
                collect(20);
            join
            if (!got_last[19]) bad_last = bad_last + 1;
            r = $fscanf(fd, "%s", tag);
            for (h = 0; h < 20; h = h + 1) begin
                r = $fscanf(fd, "%d", want);
                tops[h] = got[h][23:0];
                if ($signed(got[h]) != want) begin
                    bad_top = bad_top + 1;
                    if (bad_top < 5) $display("layer %0d head %0d top %0d want %0d", layers, h, $signed(got[h]), want);
                end
            end
            fork
                run_pass(1);
                collect(5140);
            join
            if (!got_last[5139]) bad_last = bad_last + 1;
            r = $fscanf(fd, "%s", tag);
            for (h = 0; h < 20; h = h + 1) begin
                r = $fscanf(fd, "%d", want);
                if (got[h] != want[31:0]) begin
                    bad_sum = bad_sum + 1;
                    if (bad_sum < 5) $display("layer %0d head %0d sum %0d want %0d", layers, h, got[h], want);
                end
            end
            r = $fscanf(fd, "%s", tag);
            for (j = 0; j < 20 * 128; j = j + 1) begin
                r = $fscanf(fd, "%d", want);
                if ($signed({got[20 + 2 * j + 1][7:0], got[20 + 2 * j]}) != want) begin
                    bad_acc = bad_acc + 1;
                    if (bad_acc < 5) $display("layer %0d entry %0d acc %0d want %0d", layers, j, $signed({got[20 + 2 * j + 1][7:0], got[20 + 2 * j]}), want);
                end
            end
            layers = layers + 1;
            r = $fscanf(fd, "%s", tag);
        end
        $display("RESULT %0d layers, %0d positions: top %0d wrong, sum %0d wrong, acc %0d wrong, tlast %0d wrong",
                 layers, positions, bad_top, bad_sum, bad_acc, bad_last);
        $finish;
    end
endmodule
