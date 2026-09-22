`timescale 1ns/1ps
`default_nettype none

module tb_attn_host;
    parameter GROUPS = 16;
    parameter SLOTS  = 1;
    parameter ROWS   = 6;
    parameter BEATS  = "host16.beats";
    parameter WANT   = "host16.want";

    localparam HEADS = GROUPS * SLOTS;
    localparam DIM   = 16 * ROWS;
    localparam OUTA  = HEADS;
    localparam OUTB  = HEADS + 2 * HEADS * DIM;

    reg clk = 0, rstn = 0;
    reg [127:0] s_tdata = 0;
    reg s_tvalid = 0;
    wire s_tready;
    wire [31:0] m_tdata;
    wire m_tvalid, m_tlast;
    reg m_tready = 0;
    wire [31:0] status;

    attn_fx_axi #(.GROUPS(GROUPS), .SLOTS(SLOTS), .ROWS(ROWS)) dut (.clk(clk), .rstn(rstn),
        .s_axis_tdata(s_tdata), .s_axis_tvalid(s_tvalid), .s_axis_tready(s_tready),
        .m_axis_tdata(m_tdata), .m_axis_tvalid(m_tvalid), .m_axis_tready(m_tready), .m_axis_tlast(m_tlast),
        .status(status));

    always #5 clk = ~clk;

    integer fb, fw, r, h, j, T, layers, bad_top, bad_sum, bad_acc, positions, got_n, pas;
    reg [8*8-1:0] tag;
    reg signed [63:0] want;
    reg [31:0] got [0:OUTB-1];
    reg [31:0] gota [0:OUTA-1];
    reg [127:0] beat;
    reg [31:0] seed;

    always @(posedge clk) m_tready <= ($random(seed) & 3) != 0;

    task send(input [127:0] b);
    begin
        s_tdata = b;
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
                got_n = got_n + 1;
            end
        end
    end
    endtask

    task stream_pass(input integer nbeats);
        integer i;
    begin
        for (i = 0; i < nbeats; i = i + 1) begin
            r = $fscanf(fb, "%h", beat);
            send(beat);
        end
    end
    endtask

    integer hdrbeats, posbeats;
    initial begin
        seed = 32'd20260923;
        layers = 0; bad_top = 0; bad_sum = 0; bad_acc = 0; positions = 0;
        hdrbeats = 1 + (HEADS * 16 + 127) / 128 + (HEADS * 24 + 127) / 128 + HEADS * ROWS;
        posbeats = 2 * ((GROUPS + 7) / 8) + 2 * ROWS * GROUPS;
        repeat (3) @(posedge clk);
        #1 rstn = 1;
        fb = $fopen(BEATS, "r");
        fw = $fopen(WANT, "r");
        if (fb == 0 || fw == 0) begin $display("cannot open the vectors"); $finish; end
        r = $fscanf(fb, "%s", tag);
        while (tag != "END") begin
            r = $fscanf(fb, "%d", T);
            positions = positions + T;
            for (pas = 0; pas < 2; pas = pas + 1) begin
                r = $fscanf(fb, "%s", tag);
                r = $fscanf(fb, "%d", j);
                fork
                    stream_pass(hdrbeats + T * posbeats);
                    collect(pas ? OUTB : OUTA);
                join
                if (!pas) for (h = 0; h < OUTA; h = h + 1) gota[h] = got[h];
            end
            r = $fscanf(fw, "%s", tag);
            for (h = 0; h < HEADS; h = h + 1) begin
                r = $fscanf(fw, "%d", want);
                if ($signed(gota[h]) != want) begin
                    bad_top = bad_top + 1;
                    if (bad_top < 4) $display("layer %0d head %0d top %0d want %0d", layers, h, $signed(gota[h]), want);
                end
            end
            r = $fscanf(fw, "%s", tag);
            for (h = 0; h < HEADS; h = h + 1) begin
                r = $fscanf(fw, "%d", want);
                if (got[h] != want[31:0]) begin
                    bad_sum = bad_sum + 1;
                    if (bad_sum < 4) $display("layer %0d head %0d sum %0d want %0d", layers, h, got[h], want);
                end
            end
            r = $fscanf(fw, "%s", tag);
            for (j = 0; j < HEADS * DIM; j = j + 1) begin
                r = $fscanf(fw, "%d", want);
                if ($signed({got[HEADS + 2 * j + 1][7:0], got[HEADS + 2 * j]}) != want) begin
                    bad_acc = bad_acc + 1;
                    if (bad_acc < 4) $display("layer %0d entry %0d acc %0d want %0d", layers, j,
                                              $signed({got[HEADS + 2 * j + 1][7:0], got[HEADS + 2 * j]}), want);
                end
            end
            layers = layers + 1;
            r = $fscanf(fb, "%s", tag);
        end
        $display("HOSTFMT %0d groups x %0d of %0d: %0d layers, %0d positions: top %0d wrong, sum %0d wrong, acc %0d wrong",
                 GROUPS, SLOTS, DIM, layers, positions, bad_top, bad_sum, bad_acc);
        $finish;
    end
endmodule
`default_nettype wire
