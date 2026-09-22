`timescale 1ns/1ps
`default_nettype none

module attn_fx_v3 #(
    parameter GROUPS = 5,
    parameter SLOTS  = 4,
    parameter ROWS   = 8,
    parameter EXPF_HEX = "attn_expf.hex",
    parameter EXPI_HEX = "attn_expi.hex"
) (
    input  wire         clk,
    input  wire         rstn,
    input  wire         start,
    input  wire         pass_b,
    input  wire         q_we,
    input  wire [4:0]   q_head,
    input  wire [6:0]   q_idx,
    input  wire [7:0]   q_val,
    input  wire         qrow_we,
    input  wire [4:0]   qrow_head,
    input  wire [2:0]   qrow_idx,
    input  wire [127:0] qrow_val,
    input  wire         qf_we,
    input  wire [4:0]   qf_head,
    input  wire [15:0]  qf_val,
    input  wire         top_we,
    input  wire [4:0]   top_head,
    input  wire [23:0]  top_val,
    input  wire [127:0] in_data,
    input  wire         in_valid,
    output wire         in_ready,
    input  wire [4:0]   rd_head,
    input  wire [6:0]   rd_idx,
    output wire [39:0]  rd_acc,
    output wire [23:0]  rd_top,
    output wire [31:0]  rd_sum
);
    localparam FRAC  = 12;
    localparam CUT   = 20;
    localparam HEADS = GROUPS * SLOTS;
    localparam CELLS = ROWS * GROUPS;
    localparam SCB   = (GROUPS + 7) / 8;
    localparam BEATS = 2 * SCB + 2 * CELLS;

    reg [15:0]        qf   [0:HEADS-1];
    reg signed [23:0] top  [0:HEADS-1];
    reg [31:0]        sum  [0:HEADS-1];
    reg [19:0]        u    [0:HEADS-1];
    reg [15:0]        k16  [0:GROUPS-1];
    reg [15:0]        v16  [0:GROUPS-1];

    reg        mode_b;
    reg [9:0]  clearing;
    reg [9:0]  beat;
    assign in_ready = clearing == 0;
    wire take = in_valid && in_ready;

    reg         a_k, a_v, a_first, a_last;
    reg [4:0]   a_g;
    reg [2:0]   a_r;
    reg [9:0]   a_cell;
    reg [9:0]   bcell;
    reg [4:0]   bg;
    reg [2:0]   br;
    reg [127:0] a_data;
    integer gi;
    always @(posedge clk) begin
        a_k <= 0;
        a_v <= 0;
        if (!rstn || start) begin
            beat <= 0;
            bcell <= 0;
            bg <= 0;
            br <= 0;
            clearing <= start ? CELLS[9:0] : 10'd0;
            mode_b <= pass_b;
        end else begin
            if (clearing != 0) clearing <= clearing - 1;
            if (take) begin
                beat <= (beat == BEATS - 1) ? 10'd0 : beat + 1;
                if (beat < SCB) begin
                    for (gi = 0; gi < 8; gi = gi + 1)
                        if (beat * 8 + gi < GROUPS) k16[beat * 8 + gi] <= in_data[16*gi +: 16];
                end else if (beat < 2 * SCB) begin
                    for (gi = 0; gi < 8; gi = gi + 1)
                        if ((beat - SCB) * 8 + gi < GROUPS) v16[(beat - SCB) * 8 + gi] <= in_data[16*gi +: 16];
                end else begin
                    a_k <= beat < 2 * SCB + CELLS;
                    a_v <= mode_b && !(beat < 2 * SCB + CELLS);
                    a_g <= bg;
                    a_r <= br;
                    a_cell <= bcell;
                    a_first <= br == 0;
                    a_last <= br == ROWS - 1;
                    bcell <= (bcell == CELLS - 1) ? 10'd0 : bcell + 10'd1;
                    if (br == ROWS - 1) begin
                        br <= 3'd0;
                        bg <= (bg == GROUPS - 1) ? 5'd0 : bg + 5'd1;
                    end else br <= br + 3'd1;
                end
                a_data <= in_data;
            end
        end
    end

    reg [4:0] kv, kfirst, klast;
    reg [4:0] kg [0:5];
    reg [8:0] sv;
    reg [4:0] sg [0:8];
    integer ti;
    always @(posedge clk) begin
        kv     <= {kv[3:0], a_k};
        kfirst <= {kfirst[3:0], a_first};
        klast  <= {klast[3:0], a_k && a_last};
        kg[0]  <= a_g;
        for (ti = 1; ti < 6; ti = ti + 1) kg[ti] <= kg[ti-1];
        sv     <= {sv[7:0], klast[4]};
        sg[0]  <= kg[5];
        for (ti = 1; ti < 9; ti = ti + 1) sg[ti] <= sg[ti-1];
        if (!rstn || start) begin
            kv <= 0;
            klast <= 0;
            sv <= 0;
        end
    end

    reg [19:0] expi [0:CUT];
    initial $readmemh(EXPI_HEX, expi);

    wire [39:0] rdm [0:SLOTS*16-1];
    wire signed [23:0] slot_s  [0:SLOTS-1];
    wire [19:0]        slot_w  [0:SLOTS-1];
    wire [35:0]        slot_uw [0:SLOTS-1];
    genvar gs, gj;
    generate
        for (gs = 0; gs < SLOTS; gs = gs + 1) begin : slot
            (* rom_style = "block" *) reg [19:0] expf [0:(1<<FRAC)-1];
            initial $readmemh(EXPF_HEX, expf);

            (* ram_style = "distributed" *) reg [127:0] qrow [0:CELLS-1];
            always @(posedge clk)
                if (qrow_we && qrow_head % SLOTS == gs)
                    qrow[(qrow_head / SLOTS) * ROWS + qrow_idx] <= qrow_val;
                else if (q_we && q_head % SLOTS == gs)
                    qrow[(q_head / SLOTS) * ROWS + q_idx[6:4]][8*q_idx[3:0] +: 8] <= q_val;
            wire [127:0] qsel = qrow[a_cell];

            (* use_dsp = "yes" *) reg signed [15:0] p1 [0:15];
            reg signed [16:0] p2 [0:7];
            reg signed [17:0] p3 [0:3];
            reg signed [18:0] p4 [0:1];
            reg signed [19:0] p5;
            reg signed [23:0] dot;
            integer j;
            always @(posedge clk) begin
                for (j = 0; j < 16; j = j + 1)
                    p1[j] <= $signed(a_data[8*j +: 8]) * $signed(qsel[8*j +: 8]);
                for (j = 0; j < 8; j = j + 1) p2[j] <= p1[2*j] + p1[2*j+1];
                for (j = 0; j < 4; j = j + 1) p3[j] <= p2[2*j] + p2[2*j+1];
                for (j = 0; j < 2; j = j + 1) p4[j] <= p3[2*j] + p3[2*j+1];
                p5 <= p4[0] + p4[1];
                if (kv[4]) dot <= kfirst[4] ? p5 : dot + p5;
            end

            wire [4:0] hs0 = sg[0] * SLOTS + gs;
            wire [4:0] hs2 = sg[2] * SLOTS + gs;
            reg signed [40:0] m1;
            reg signed [57:0] m2;
            reg signed [57:0] sh;
            reg signed [23:0] s;
            reg signed [24:0] dd;
            reg        cut, cut2;
            reg [19:0] ei;
            reg [39:0] wf;
            reg [19:0] w;
            reg [35:0] uw;
            wire [FRAC-1:0] fa = (dd < 0) ? {FRAC{1'b0}} : dd[FRAC-1:0];
            reg [19:0] ef;
            always @(posedge clk) ef <= expf[fa];
            reg [15:0] qf_r;
            always @(posedge clk) qf_r <= qf[kg[5] * SLOTS + gs];
            always @(posedge clk) begin
                m1 <= dot * $signed({1'b0, k16[kg[5]]});
                m2 <= m1 * $signed({1'b0, qf_r});
                sh = m2 >>> (36 - FRAC);
                if (sh > 58'sd8388607) s <= 24'sd8388607;
                else if (sh < -58'sd8388608) s <= -24'sd8388608;
                else s <= sh[23:0];
                dd <= $signed({top[hs2][23], top[hs2]}) - $signed({s[23], s});
                cut <= (dd >= 0) && ((dd >>> FRAC) > CUT);
                ei <= expi[(dd < 0) ? 5'd0 : ((dd >>> FRAC) > CUT ? 5'd0 : dd[FRAC+4:FRAC])];
                cut2 <= cut;
                wf <= ef * ei;
                w <= cut2 ? 20'd0 : wf[39:20];
                uw <= w * v16[sg[6]];
            end
            assign slot_s[gs] = s;
            assign slot_w[gs] = w;
            assign slot_uw[gs] = uw;

            for (gj = 0; gj < 16; gj = gj + 1) begin : lane
                (* ram_style = "distributed" *) reg signed [39:0] mem [0:CELLS-1];
                (* use_dsp = "yes" *) reg signed [28:0] pv;
                reg signed [28:0] pv2;
                reg [9:0] addr, addr2;
                reg       wr, wr2;
                always @(posedge clk) begin
                    pv <= $signed({1'b0, u[a_g * SLOTS + gs]}) * $signed(a_data[8*gj +: 8]);
                    addr <= a_cell;
                    wr <= clearing == 0 && a_v;
                    pv2 <= pv;
                    addr2 <= addr;
                    wr2 <= wr;
                    if (clearing != 0) mem[clearing - 10'd1] <= 40'sd0;
                    else if (wr2) mem[addr2] <= mem[addr2] + pv2;
                end
                assign rdm[gs * 16 + gj] = mem[(rd_head / SLOTS) * ROWS + rd_idx[6:4]];
            end
        end
    endgenerate

    integer h;
    always @(posedge clk) begin
        if (qf_we) qf[qf_head] <= qf_val;
        for (h = 0; h < HEADS; h = h + 1) begin
            if (start && !pass_b) top[h] <= 24'h800000;
            else if (top_we && top_head == h) top[h] <= top_val;
            else if (sv[3] && !mode_b && sg[2] == h / SLOTS && slot_s[h % SLOTS] > top[h]) top[h] <= slot_s[h % SLOTS];
            if (start) sum[h] <= 0;
            else if (sv[7] && mode_b && sg[6] == h / SLOTS) sum[h] <= sum[h] + slot_w[h % SLOTS];
            if (sv[8] && sg[7] == h / SLOTS) u[h] <= slot_uw[h % SLOTS][35:16];
        end
    end

    assign rd_acc = rdm[(rd_head % SLOTS) * 16 + rd_idx[3:0]];
    assign rd_top = top[rd_head];
    assign rd_sum = sum[rd_head];
endmodule
`default_nettype wire
