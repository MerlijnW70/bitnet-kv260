`timescale 1ns/1ps
`default_nettype none

module attn_fx_axi #(
    parameter GROUPS = 5,
    parameter SLOTS  = 4,
    parameter ROWS   = 8,
    parameter EXPF_HEX = "attn_expf.hex",
    parameter EXPI_HEX = "attn_expi.hex"
) (
    input  wire         clk,
    input  wire         rstn,
    input  wire [127:0] s_axis_tdata,
    input  wire         s_axis_tvalid,
    output wire         s_axis_tready,
    output reg  [31:0]  m_axis_tdata,
    output reg          m_axis_tvalid,
    input  wire         m_axis_tready,
    output reg          m_axis_tlast,
    output wire [31:0]  status
);
    localparam HEADS = GROUPS * SLOTS;
    localparam DIM   = 16 * ROWS;
    localparam QFB   = (HEADS * 16 + 127) / 128;
    localparam TOPB  = (HEADS * 24 + 127) / 128;
    localparam QROWS = HEADS * ROWS;
    localparam HDRB  = QFB + TOPB + QROWS;
    localparam SCB   = (GROUPS + 7) / 8;
    localparam BEATS = 2 * SCB + 2 * ROWS * GROUPS;
    localparam OUTA  = HEADS;
    localparam OUTB  = HEADS + 2 * HEADS * DIM;
    localparam [2:0] S_IDLE = 0, S_HDR = 1, S_CFG = 2, S_DATA = 3, S_DRAIN = 4, S_OUT = 5;
    localparam [15:0] MAGIC = 16'hA77E;

    reg [2:0]   state;
    reg         pass_b;
    reg [15:0]  positions;
    reg [15:0]  hbeat;
    reg [QFB*128-1:0]  qf_bits;
    reg [TOPB*128-1:0] top_bits;
    reg [5:0]   cfg;
    reg [22:0]  data_left;
    reg [5:0]   drain;
    reg [15:0]  ototal;

    reg          core_start;
    reg          qf_we, top_we, qrow_we;
    reg  [4:0]   qf_head, top_head, qrow_head;
    reg  [15:0]  qf_val;
    reg  [23:0]  top_val;
    reg  [2:0]   qrow_idx;
    reg  [127:0] qrow_val;
    reg  [4:0]   qh;
    reg  [2:0]   qi;
    wire         core_ready;
    reg  [4:0]   rd_head;
    reg  [6:0]   rd_idx;
    wire [39:0]  rd_acc;
    wire [23:0]  rd_top;
    wire [31:0]  rd_sum;

    wire in_valid = (state == S_DATA) && !core_start && s_axis_tvalid && data_left != 0;
    assign s_axis_tready = (state == S_IDLE) || (state == S_HDR)
                        || (state == S_DATA && !core_start && core_ready && data_left != 0);
    wire take = s_axis_tvalid && s_axis_tready;
    reg [15:0] sent;
    assign status = {state, pass_b, sent[13:0], 1'b0, fidx[12:0]};

    attn_fx_v3 #(.GROUPS(GROUPS), .SLOTS(SLOTS), .ROWS(ROWS),
                 .EXPF_HEX(EXPF_HEX), .EXPI_HEX(EXPI_HEX)) core (
        .clk(clk), .rstn(rstn), .start(core_start), .pass_b(pass_b),
        .q_we(1'b0), .q_head(5'd0), .q_idx(7'd0), .q_val(8'd0),
        .qrow_we(qrow_we), .qrow_head(qrow_head), .qrow_idx(qrow_idx), .qrow_val(qrow_val),
        .qf_we(qf_we), .qf_head(qf_head), .qf_val(qf_val),
        .top_we(top_we), .top_head(top_head), .top_val(top_val),
        .in_data(s_axis_tdata), .in_valid(in_valid), .in_ready(core_ready),
        .rd_head(rd_head), .rd_idx(rd_idx), .rd_acc(rd_acc), .rd_top(rd_top), .rd_sum(rd_sum));

    reg [15:0] fidx;
    reg        a_valid, a_last, a_sum, a_high;
    reg        b_valid, b_last;
    reg [31:0] b_data;
    reg  [4:0]  oh;
    reg  [6:0]  od;
    reg         ohigh;
    wire [31:0] word = !pass_b ? {{8{rd_top[23]}}, rd_top}
                     : a_sum ? rd_sum
                     : a_high ? {{24{rd_acc[39]}}, rd_acc[39:32]} : rd_acc[31:0];
    wire out_stall = m_axis_tvalid && !m_axis_tready;
    wire out_done = (fidx == ototal) && !a_valid && !b_valid && !m_axis_tvalid;

    always @(posedge clk) begin
        core_start <= 0;
        qf_we <= 0;
        top_we <= 0;
        qrow_we <= 0;
        if (!rstn) begin
            state <= S_IDLE;
            pass_b <= 0;
            positions <= 0;
        end else case (state)
            S_IDLE: if (take && s_axis_tdata[15:0] == MAGIC) begin
                pass_b <= s_axis_tdata[16];
                positions <= s_axis_tdata[47:32];
                hbeat <= 1;
                qh <= 0;
                qi <= 0;
                state <= S_HDR;
            end
            S_HDR: if (take) begin
                if (hbeat <= QFB) qf_bits[128 * (hbeat - 1) +: 128] <= s_axis_tdata;
                else if (hbeat <= QFB + TOPB) top_bits[128 * (hbeat - QFB - 1) +: 128] <= s_axis_tdata;
                else begin
                    qrow_we <= 1;
                    qrow_head <= qh;
                    qrow_idx <= qi;
                    qrow_val <= s_axis_tdata;
                    if (qi == ROWS - 1) begin qi <= 3'd0; qh <= qh + 5'd1; end
                    else qi <= qi + 3'd1;
                end
                if (hbeat == HDRB) begin
                    cfg <= 0;
                    state <= S_CFG;
                end
                hbeat <= hbeat + 1;
            end
            S_CFG: begin
                qf_we <= 1;
                qf_head <= cfg;
                qf_val <= qf_bits[16 * cfg +: 16];
                top_we <= pass_b;
                top_head <= cfg;
                top_val <= top_bits[24 * cfg +: 24];
                if (cfg == HEADS - 1) begin
                    core_start <= 1;
                    data_left <= positions * BEATS[22:0];
                    state <= S_DATA;
                end
                cfg <= cfg + 1;
            end
            S_DATA: begin
                if (take) data_left <= data_left - 1;
                if (data_left == 0 || (take && data_left == 1)) begin
                    drain <= 6'd40;
                    state <= S_DRAIN;
                end
            end
            S_DRAIN: begin
                drain <= drain - 1;
                if (drain == 0) begin
                    ototal <= pass_b ? OUTB[15:0] : OUTA[15:0];
                    state <= S_OUT;
                end
            end
            S_OUT: if (out_done) state <= S_IDLE;
            default: state <= S_IDLE;
        endcase
    end

    always @(posedge clk) begin
        if (m_axis_tvalid && m_axis_tready) sent <= sent + 1;
        if (!rstn || state != S_OUT) begin
            fidx <= 0;
            oh <= 0;
            od <= 0;
            ohigh <= 0;
            a_valid <= 0;
            b_valid <= 0;
            if (state == S_DRAIN) sent <= 0;
            if (!rstn || state != S_OUT) begin
                m_axis_tvalid <= 0;
                m_axis_tlast <= 0;
            end
        end else if (!out_stall) begin
            if (b_valid) begin
                m_axis_tdata <= b_data;
                m_axis_tlast <= b_last;
                m_axis_tvalid <= 1;
            end else begin
                m_axis_tvalid <= 0;
            end
            b_data <= word;
            b_valid <= a_valid;
            b_last <= a_last;
            if (fidx < ototal) begin
                rd_head <= (fidx < OUTA) ? fidx[4:0] : oh;
                rd_idx  <= (fidx < OUTA) ? 7'd0 : od;
                a_sum <= fidx < OUTA;
                a_high <= (fidx >= OUTA) && ohigh;
                a_valid <= 1;
                a_last <= fidx == ototal - 1;
                fidx <= fidx + 1;
                if (fidx >= OUTA) begin
                    if (ohigh) begin
                        ohigh <= 1'b0;
                        if (od == DIM - 1) begin od <= 7'd0; oh <= oh + 5'd1; end
                        else od <= od + 7'd1;
                    end else ohigh <= 1'b1;
                end
            end else begin
                a_valid <= 0;
            end
        end
    end
endmodule
`default_nettype wire
