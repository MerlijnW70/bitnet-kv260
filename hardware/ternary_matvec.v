`timescale 1ns/1ps
module ternary_matvec #(
    parameter SLICES = 2,
    parameter ULTRA  = 1
) (
    input              clk,
    input              rstn,
    input      [127:0] s_axis_tdata,
    input              s_axis_tvalid,
    output             s_axis_tready,
    input              s_axis_tlast,
    output reg  [31:0] m_axis_tdata,
    output reg         m_axis_tvalid,
    input              m_axis_tready,
    output reg         m_axis_tlast,
    input       [31:0] ctrl,
    output      [31:0] status
);
    localparam LANES      = 80;
    localparam SUBS       = 5;
    localparam MAX_BEATS  = 87;
    localparam BATCH_MAX  = 4;
    localparam FIFO_DEPTH = 8;
    localparam LOGS       = (SLICES == 1) ? 0 : (SLICES == 2) ? 1 : 2;

    localparam [2:0] S_IDLE = 3'd0, S_LOAD = 3'd1, S_COMPUTE = 3'd2, S_DRAIN = 3'd3, S_DONE = 3'd4;
    reg  [2:0]  state;
    wire        run     = ctrl[0];
    wire        mode    = ctrl[1];
    wire [6:0]  b_field = ctrl[8:2];
    wire [6:0]  b_ctrl  = (b_field == 7'd0) ? 7'd32 : b_field;
    wire [2:0]  batch_ctrl = {1'b0, ctrl[10:9]} + 3'd1;
    wire [15:0] n_ctrl  = ctrl[31:16];

    reg  [15:0] n_neurons;
    reg  [6:0]  b_last;
    reg  [2:0]  batch;
    reg  [1:0]  p_last;
    reg  [15:0] neurons_done;
    reg  [10:0] act_beat;
    reg  [2:0]  sub;
    reg  [1:0]  vec;
    (* max_fanout = 128 *)
    reg  [6:0]  beat;
    reg  [1:0]  pass;
    reg  [15:0] neuron;
    reg         out_eon;

    (* max_fanout = 128 *)
    reg  room;
    wire advance   = room;
    wire last_pass = (pass == p_last);
    assign s_axis_tready = (state == S_LOAD) || (state == S_COMPUTE && advance && last_pass);
    wire take  = s_axis_tvalid && s_axis_tready;
    wire fire  = (state == S_COMPUTE) && s_axis_tvalid && advance;
    wire abort = (state != S_IDLE) && !run;
    wire idle  = (state == S_DONE) || (state == S_IDLE && !run);
    assign status = {idle, (state == S_LOAD || state == S_COMPUTE || state == S_DRAIN),
                     state, act_beat, neurons_done};

    function [255:0] dmask(input integer t, input integer b);
        integer i, j, v, c;
        begin
            dmask = 256'd0;
            for (i = 0; i < 256; i = i + 1) begin
                v = i; c = 0;
                for (j = 0; j < 5; j = j + 1) begin
                    if (j == t) c = v % 3;
                    v = v / 3;
                end
                dmask[i] = (c >> b) & 1;
            end
        end
    endfunction
    localparam [255:0] M0 = dmask(0, 0), M1 = dmask(0, 1);
    localparam [255:0] M2 = dmask(1, 0), M3 = dmask(1, 1);
    localparam [255:0] M4 = dmask(2, 0), M5 = dmask(2, 1);
    localparam [255:0] M6 = dmask(3, 0), M7 = dmask(3, 1);
    localparam [255:0] M8 = dmask(4, 0), M9 = dmask(4, 1);
    wire [LANES*2-1:0] wcode;
    genvar gs, gi, gj;
    generate
        for (gj = 0; gj < 16; gj = gj + 1) begin : dec
            wire [7:0] byt = s_axis_tdata[8*gj +: 8];
            assign wcode[10*gj + 0] = M0[byt];
            assign wcode[10*gj + 1] = M1[byt];
            assign wcode[10*gj + 2] = M2[byt];
            assign wcode[10*gj + 3] = M3[byt];
            assign wcode[10*gj + 4] = M4[byt];
            assign wcode[10*gj + 5] = M5[byt];
            assign wcode[10*gj + 6] = M6[byt];
            assign wcode[10*gj + 7] = M7[byt];
            assign wcode[10*gj + 8] = M8[byt];
            assign wcode[10*gj + 9] = M9[byt];
        end
    endgenerate

    wire [8:0]      waddr = {vec, beat};
    wire            store = (state == S_LOAD) && take;
    wire [SUBS-1:0] wr_en;
    generate
        for (gj = 0; gj < SUBS; gj = gj + 1) begin : wsel
            assign wr_en[gj] = store && (sub == gj);
        end
    endgenerate

    always @(posedge clk) begin
        if (!rstn) begin
            state <= S_IDLE; n_neurons <= 16'd0; neurons_done <= 16'd0;
            b_last <= 7'd31; batch <= 3'd1; p_last <= 2'd0;
            act_beat <= 11'd0; sub <= 3'd0; vec <= 2'd0; beat <= 7'd0; pass <= 2'd0; neuron <= 16'd0;
        end else if (abort) begin
            state <= S_IDLE; neurons_done <= 16'd0;
            act_beat <= 11'd0; sub <= 3'd0; vec <= 2'd0; beat <= 7'd0; pass <= 2'd0; neuron <= 16'd0;
        end else begin
            if (m_axis_tvalid && m_axis_tready && out_eon) neurons_done <= neurons_done + 16'd1;
            case (state)
                S_IDLE: if (run) begin
                    n_neurons <= n_ctrl; neurons_done <= 16'd0;
                    b_last <= b_ctrl - 7'd1; batch <= batch_ctrl;
                    p_last <= (batch_ctrl - 3'd1) >> LOGS;
                    act_beat <= 11'd0; sub <= 3'd0; vec <= 2'd0; beat <= 7'd0; pass <= 2'd0; neuron <= 16'd0;
                    if (!mode)             state <= S_LOAD;
                    else if (n_ctrl == 0)  state <= S_DONE;
                    else                   state <= S_COMPUTE;
                end
                S_LOAD: if (take) begin
                    act_beat <= act_beat + 11'd1;
                    if (sub == SUBS - 1) begin
                        sub <= 3'd0;
                        if (beat == b_last) begin
                            beat <= 7'd0;
                            if ({1'b0, vec} == batch - 3'd1) state <= S_DONE;
                            else vec <= vec + 2'd1;
                        end else beat <= beat + 7'd1;
                    end else sub <= sub + 3'd1;
                end
                S_COMPUTE: if (fire) begin
                    if (last_pass) begin
                        pass <= 2'd0;
                        if (beat == b_last) begin
                            beat <= 7'd0; neuron <= neuron + 16'd1;
                            if (neuron == n_neurons - 1) state <= S_DRAIN;
                        end else beat <= beat + 7'd1;
                    end else pass <= pass + 2'd1;
                end
                S_DRAIN: if (m_axis_tvalid && m_axis_tready && m_axis_tlast) state <= S_DONE;
                S_DONE: ;
                default: state <= S_IDLE;
            endcase
        end
    end

    reg          v1, lastb1, lastn1;
    reg  [1:0]   pass1;
    reg  [LANES*2-1:0] w1;
    always @(posedge clk) begin
        if (!rstn || abort) v1 <= 1'b0;
        else if (advance)   v1 <= fire;
        if (advance) begin
            w1     <= wcode;
            pass1  <= pass;
            lastb1 <= (beat == b_last);
            lastn1 <= (neuron == n_neurons - 1);
        end
    end
    reg          v2, lastb2, lastn2; reg [1:0] pass2;
    reg          v3, lastb3, lastn3; reg [1:0] pass3;
    reg          v4, lastb4, lastn4; reg [1:0] pass4;
    reg          v5, lastb5, lastn5; reg [1:0] pass5;
    always @(posedge clk) begin
        if (!rstn || abort) begin v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0; v5 <= 1'b0; end
        else if (advance) begin v2 <= v1; v3 <= v2; v4 <= v3; v5 <= v4; end
        if (advance) begin
            pass2  <= pass1;  pass3  <= pass2;  pass4  <= pass3;  pass5  <= pass4;
            lastb2 <= lastb1; lastb3 <= lastb2; lastb4 <= lastb3; lastb5 <= lastb4;
            lastn2 <= lastn1; lastn3 <= lastn2; lastn4 <= lastn3; lastn5 <= lastn4;
        end
    end

    reg  [31:0] acc [0:BATCH_MAX-1];
    wire [SLICES*32-1:0] accsum_flat;
    wire [SLICES*2-1:0]  vidx_flat;
    wire [SLICES-1:0]    push_flat, eon_flat, eor_flat;
    generate
    for (gs = 0; gs < SLICES; gs = gs + 1) begin : sl
        wire [1:0] vsel   = (pass << LOGS) + gs;
        wire       act_on = ({1'b0, vsel} < batch);
        wire [8:0] raddr  = {vsel, beat};
        wire [SUBS*128-1:0] aword;
        for (gj = 0; gj < SUBS; gj = gj + 1) begin : word
            ternary_matvec_ram #(.ULTRA((gs == 0) ? 0 : ULTRA)) ram (
                .clk(clk), .we(wr_en[gj]), .waddr(waddr), .wdata(s_axis_tdata),
                .ren(advance && act_on), .raddr(raddr), .rdout(aword[128*gj +: 128]));
        end

        wire [LANES*8-1:0] term;
        wire [LANES-1:0]   neg;
        for (gi = 0; gi < LANES; gi = gi + 1) begin : lane
            wire [1:0] c = w1[2*gi +: 2];
            wire [7:0] a = aword[8*gi +: 8];
            assign term[8*gi +: 8] = {8{~c[0]}} & (a ^ {8{~c[1]}});
            assign neg[gi] = ~c[0] & ~c[1];
        end
        wire [40*9-1:0] l1;
        ternary_matvec_pairs #(.N(80), .W(8), .SIGNED(1)) L1 (.x(term), .y(l1));
        wire [10*4-1:0] p1;
        for (gi = 0; gi < 10; gi = gi + 1) begin : pop
            assign p1[4*gi +: 4] = pop8(neg[8*gi +: 8]);
        end
        reg [40*9-1:0] s2;
        reg [10*4-1:0] p2;
        always @(posedge clk) if (advance) begin s2 <= l1; p2 <= p1; end

        wire [20*10-1:0] l2;
        wire [10*11-1:0] l3;
        ternary_matvec_pairs #(.N(40), .W(9),  .SIGNED(1)) L2 (.x(s2), .y(l2));
        ternary_matvec_pairs #(.N(20), .W(10), .SIGNED(1)) L3 (.x(l2), .y(l3));
        wire [5*5-1:0] q2;
        wire [3*6-1:0] q3;
        ternary_matvec_pairs #(.N(10), .W(4), .SIGNED(0)) P2 (.x(p2), .y(q2));
        ternary_matvec_pairs #(.N(5),  .W(5), .SIGNED(0)) P3 (.x(q2), .y(q3));
        reg [10*11-1:0] s3;
        reg [3*6-1:0]   p3;
        always @(posedge clk) if (advance) begin s3 <= l3; p3 <= q3; end

        wire [5*12-1:0] l4;
        wire [3*13-1:0] l5;
        ternary_matvec_pairs #(.N(10), .W(11), .SIGNED(1)) L4 (.x(s3), .y(l4));
        ternary_matvec_pairs #(.N(5),  .W(12), .SIGNED(1)) L5 (.x(l4), .y(l5));
        wire [2*7-1:0] q4;
        ternary_matvec_pairs #(.N(3), .W(6), .SIGNED(0)) P4 (.x(p3), .y(q4));
        reg [3*13-1:0] s4;
        reg [2*7-1:0]  p4;
        always @(posedge clk) if (advance) begin s4 <= l5; p4 <= q4; end

        wire [2*14-1:0] l6;
        wire [14:0]     l7;
        ternary_matvec_pairs #(.N(3), .W(13), .SIGNED(1)) L6 (.x(s4), .y(l6));
        ternary_matvec_pairs #(.N(2), .W(14), .SIGNED(1)) L7 (.x(l6), .y(l7));
        wire [7:0] q5;
        ternary_matvec_pairs #(.N(2), .W(7), .SIGNED(0)) P5 (.x(p4), .y(q5));
        wire [14:0] bsum = l7 + {7'b0, q5};
        reg  [14:0] s5;
        always @(posedge clk) if (advance) s5 <= bsum;

        wire [1:0] vidx = (pass5 << LOGS) + gs;
        assign vidx_flat[2*gs +: 2]   = vidx;
        assign accsum_flat[32*gs +: 32] = acc[vidx] + {{17{s5[14]}}, s5};
        assign push_flat[gs] = ({1'b0, vidx} <  batch);
        assign eon_flat[gs]  = ({1'b0, vidx} == batch - 3'd1);
        assign eor_flat[gs]  = eon_flat[gs] && lastn5;
    end
    endgenerate

    integer ai, si;
    always @(posedge clk) begin
        if (!rstn || abort) begin
            for (ai = 0; ai < BATCH_MAX; ai = ai + 1) acc[ai] <= 32'd0;
        end else if (advance && v5) begin
            for (si = 0; si < SLICES; si = si + 1)
                acc[vidx_flat[2*si +: 2]] <= lastb5 ? 32'd0 : accsum_flat[32*si +: 32];
        end
    end

    wire       sum_done = advance && v5 && lastb5;
    reg [33:0] fifo [0:FIFO_DEPTH-1];
    reg  [3:0] wp, rp;
    reg  [2:0] widx;
    reg  [2:0] n_push;
    integer si2;
    integer pi;
    always @* begin
        n_push = 3'd0;
        for (pi = 0; pi < SLICES; pi = pi + 1) n_push = n_push + {2'b0, push_flat[pi]};
    end
    always @(posedge clk) begin
        if (!rstn || abort) wp <= 4'd0;
        else if (sum_done) begin
            for (si2 = 0; si2 < SLICES; si2 = si2 + 1) begin
                widx = wp[2:0] + si2;
                if (push_flat[si2]) fifo[widx] <= {eor_flat[si2], eon_flat[si2], accsum_flat[32*si2 +: 32]};
            end
            wp <= wp + {1'b0, n_push};
        end
    end
    wire [3:0] fill    = wp - rp;
    wire       out_free = !m_axis_tvalid || m_axis_tready;
    wire       pop      = out_free && (fill != 4'd0);
    wire [3:0] wp_next  = wp + (sum_done ? {1'b0, n_push} : 4'd0);
    wire [3:0] rp_next  = rp + (pop ? 4'd1 : 4'd0);
    always @(posedge clk) begin
        if (!rstn || abort) begin
            rp <= 4'd0; room <= 1'b1; out_eon <= 1'b0;
            m_axis_tvalid <= 1'b0; m_axis_tlast <= 1'b0; m_axis_tdata <= 32'd0;
        end else begin
            room <= ((wp_next - rp_next) <= (FIFO_DEPTH - SLICES));
            if (out_free) begin
                if (fill != 4'd0) begin
                    m_axis_tdata  <= fifo[rp[2:0]][31:0];
                    out_eon       <= fifo[rp[2:0]][32];
                    m_axis_tlast  <= fifo[rp[2:0]][33];
                    m_axis_tvalid <= 1'b1;
                    rp <= rp + 4'd1;
                end else m_axis_tvalid <= 1'b0;
            end
        end
    end

    function [3:0] pop8(input [7:0] b);
        integer k;
        begin
            pop8 = 4'd0;
            for (k = 0; k < 8; k = k + 1) pop8 = pop8 + {3'b0, b[k]};
        end
    endfunction
endmodule

module ternary_matvec_ram #(parameter ULTRA = 0) (
    input              clk,
    input              we,
    input      [8:0]   waddr,
    input      [127:0] wdata,
    input              ren,
    input      [8:0]   raddr,
    output     [127:0] rdout
);
    generate
        if (ULTRA) begin : u
            (* ram_style = "ultra" *) reg [127:0] mem [0:511];
            reg [127:0] dout;
            always @(posedge clk) if (we)  mem[waddr] <= wdata;
            always @(posedge clk) if (ren) dout <= mem[raddr];
            assign rdout = dout;
        end else begin : b
            (* ram_style = "block" *) reg [127:0] mem [0:511];
            reg [127:0] dout;
            always @(posedge clk) if (we)  mem[waddr] <= wdata;
            always @(posedge clk) if (ren) dout <= mem[raddr];
            assign rdout = dout;
        end
    endgenerate
endmodule

module ternary_matvec_pairs #(parameter N = 2, parameter W = 8, parameter SIGNED = 1) (
    input  [N*W-1:0]               x,
    output [((N+1)/2)*(W+1)-1:0]   y
);
    genvar i;
    generate
        for (i = 0; i < N/2; i = i + 1) begin : p
            wire [W-1:0] a = x[(2*i)*W +: W];
            wire [W-1:0] b = x[(2*i+1)*W +: W];
            if (SIGNED) begin : s
                assign y[i*(W+1) +: W+1] = {a[W-1], a} + {b[W-1], b};
            end else begin : u
                assign y[i*(W+1) +: W+1] = {1'b0, a} + {1'b0, b};
            end
        end
        if (N % 2) begin : odd
            wire [W-1:0] a = x[(N-1)*W +: W];
            if (SIGNED) begin : s
                assign y[(N/2)*(W+1) +: W+1] = {a[W-1], a};
            end else begin : u
                assign y[(N/2)*(W+1) +: W+1] = {1'b0, a};
            end
        end
    endgenerate
endmodule
