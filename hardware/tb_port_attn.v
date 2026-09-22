`timescale 1ns/1ps
`default_nettype none

/* The block inside the port wrapper the bitstream actually instantiates, driven the way the
 * runtime drives it: the select bit set through gpo, the header and the cache streamed in, the
 * answers taken out of m_axis. tb_attn_host covers attn_fx_axi alone; this covers the mux. */
module tb_port_attn;
    parameter GROUPS = 16;
    parameter SLOTS  = 1;
    parameter ROWS   = 6;
    parameter BEATSF = "host16.beats";
    parameter WANT   = "host16.want";

    localparam HEADS = GROUPS * SLOTS;
    localparam DIM   = 16 * ROWS;
    localparam OUTA  = HEADS;
    localparam OUTB  = HEADS + 2 * HEADS * DIM;

    reg clk = 0, rstn = 0;
    reg [127:0] s_tdata = 0;
    reg s_tvalid = 0;
    reg s_tlast = 0;
    wire s_tready;
    wire [31:0] m_tdata;
    wire m_tvalid, m_tlast;
    reg m_tready = 0;
    reg [31:0] gpo = 0;
    wire [31:0] gpi;

    ternary_port_axi #(.GROUPS(GROUPS), .SLOTS(SLOTS), .ROWS(ROWS)) dut (
        .aclk(clk), .aresetn(rstn),
        .s_axis_tdata(s_tdata), .s_axis_tkeep(16'hFFFF), .s_axis_tvalid(s_tvalid),
        .s_axis_tready(s_tready), .s_axis_tlast(s_tlast),
        .m_axis_tdata(m_tdata), .m_axis_tkeep(), .m_axis_tvalid(m_tvalid),
        .m_axis_tready(m_tready), .m_axis_tlast(m_tlast),
        .gpo(gpo), .gpi(gpi));

    always #5 clk = ~clk;

    integer fb, fw, r, h, j, T, layers, bad_sum, bad_acc, positions, got_n, pas, stuck;
    reg [8*8-1:0] tag;
    reg signed [63:0] want;
    reg [31:0] got [0:OUTB-1];
    reg [127:0] beat;
    reg [31:0] seed;

    always @(posedge clk) m_tready <= ($random(seed) & 3) != 0;

    task send(input [127:0] b, input last);
    begin
        s_tdata = b;
        s_tvalid = 1;
        s_tlast = last;
        @(posedge clk);
        while (!s_tready) @(posedge clk);
        #1;
        s_tvalid = 0;
        s_tlast = 0;
    end
    endtask

    task collect(input integer n);
    begin
        got_n = 0;
        stuck = 0;
        while (got_n < n && stuck < 2000000) begin
            @(posedge clk);
            stuck = stuck + 1;
            if (m_tvalid && m_tready) begin
                got[got_n] = m_tdata;
                got_n = got_n + 1;
                stuck = 0;
            end
        end
        if (got_n < n) $display("STALLED after %0d of %0d words, gpi 0x%08x (state %0d)",
                                got_n, n, gpi, gpi[31:29]);
    end
    endtask

    task stream(input integer nbeats);
        integer i;
    begin
        for (i = 0; i < nbeats; i = i + 1) begin
            r = $fscanf(fb, "%h", beat);
            send(beat, i == nbeats - 1);
        end
    end
    endtask

    integer hdrbeats, posbeats;
    initial begin
        seed = 32'd20260923;
        layers = 0; bad_sum = 0; bad_acc = 0; positions = 0;
        hdrbeats = 1 + (HEADS * 16 + 127) / 128 + (HEADS * 24 + 127) / 128 + HEADS * ROWS;
        posbeats = 2 * ((GROUPS + 7) / 8) + 2 * ROWS * GROUPS;
        repeat (3) @(posedge clk);
        #1 rstn = 1;
        gpo = 32'h00008000;
        fb = $fopen(BEATSF, "r");
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
                    begin
                        stream(hdrbeats);
                        stream(T * posbeats);
                    end
                    collect(pas ? OUTB : OUTA);
                join
            end
            r = $fscanf(fw, "%s", tag);
            for (h = 0; h < HEADS; h = h + 1) r = $fscanf(fw, "%d", want);
            r = $fscanf(fw, "%s", tag);
            for (h = 0; h < HEADS; h = h + 1) begin
                r = $fscanf(fw, "%d", want);
                if (got[h] != want[31:0]) bad_sum = bad_sum + 1;
            end
            r = $fscanf(fw, "%s", tag);
            for (j = 0; j < HEADS * DIM; j = j + 1) begin
                r = $fscanf(fw, "%d", want);
                if ($signed({got[HEADS + 2 * j + 1][7:0], got[HEADS + 2 * j]}) != want)
                    bad_acc = bad_acc + 1;
            end
            layers = layers + 1;
            r = $fscanf(fb, "%s", tag);
        end
        $display("PORT %0dx%0d of %0d: %0d layers, %0d positions: sum %0d wrong, acc %0d wrong",
                 GROUPS, SLOTS, DIM, layers, positions, bad_sum, bad_acc);
        $finish;
    end
endmodule
`default_nettype wire
