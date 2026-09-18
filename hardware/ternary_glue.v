`timescale 1ns/1ps
module ternary_glue (
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
    input       [31:0] params,
    input       [31:0] mval,
    output      [31:0] status
);
    localparam CHAINS  = 16;
    localparam SPACING = 32 / CHAINS;
    localparam LOGS    = $clog2(SPACING);
    localparam LOGC    = $clog2(CHAINS);
    localparam DEPTH   = 128;
    localparam [1:0] S_IDLE = 2'd0, S_RUN = 2'd1, S_DONE = 2'd2;

    reg  [1:0]  state;
    wire        run       = ctrl[0];
    wire        mode_ctrl = ctrl[1];
    wire [15:0] n_ctrl    = ctrl[31:16];
    reg         mode;
    reg  [15:0] n_elems;
    reg  [15:0] n_beats;
    reg  [4:0]  s_g, s_u, s_S, s_T, s_V;
    reg  [15:0] m_val;
    reg  [15:0] taken;
    reg  [15:0] acc_beats;
    reg  [14:0] done;
    reg  [15:0] hmax;
    reg  [7:0]  free;
    wire abort    = (state != S_IDLE) && !run;
    wire starting = (state == S_IDLE) && run;
    wire clearing = !rstn || abort || starting;
    wire idle     = (state == S_DONE) || (state == S_IDLE && !run);
    assign status = {idle, done, 1'b0, hmax[14:0]};

    reg [4:0] fc;
    always @(posedge clk) if (!rstn) fc <= 5'd0; else fc <= fc + 5'd1;

    always @(posedge clk) begin
        if (!rstn) begin
            state <= S_IDLE; mode <= 1'b0; n_elems <= 16'd0; n_beats <= 16'd0;
            s_g <= 5'd0; s_u <= 5'd0; s_S <= 5'd0; s_T <= 5'd0; s_V <= 5'd0; m_val <= 16'd0;
        end else if (abort) begin
            state <= S_IDLE;
        end else case (state)
            S_IDLE: if (run) begin
                mode <= mode_ctrl; n_elems <= n_ctrl;
                n_beats <= mode_ctrl ? ((n_ctrl + 16'd3) >> 2) : n_ctrl;
                s_g <= params[4:0]; s_u <= params[9:5]; s_S <= params[14:10]; s_T <= params[19:15]; s_V <= params[24:20];
                m_val <= mval[15:0];
                state <= (n_ctrl == 16'd0) ? S_DONE : S_RUN;
            end
            S_RUN: if (m_axis_tvalid && m_axis_tready && m_axis_tlast) state <= S_DONE;
            S_DONE: ;
            default: state <= S_IDLE;
        endcase
    end

    reg          validA, validB, validC;
    reg  [127:0] beatA;
    reg  [127:0] hB;
    reg  [30:0]  rgB, auB;
    reg  [14:0]  agB;
    reg          negB;
    reg  [1:0]   idxB;
    reg  [15:0]  xC, yC, bC;
    reg  [14:0]  gC;
    reg          negC;
    reg  [1:0]   idxC;

    assign s_axis_tready = (state == S_RUN) && !validA && (acc_beats != n_beats);
    wire take_in   = s_axis_tvalid && s_axis_tready;
    wire a_move    = validA && !validB;
    wire slot_any  = (&fc[LOGS-1:0]);
    wire credit_ok = (mode && idxC != 2'd0) || (free != 8'd0);
    wire c_take    = slot_any && validC && (taken != n_elems) && credit_ok;
    wire b_move    = validB && (!validC || c_take);
    wire reserve   = c_take && (!mode || idxC == 2'd0);
    wire last_in   = (taken == n_elems - 16'd1);
    wire [31:0] hsel = hB[32 * idxB +: 32];
    wire [30:0] sh_a = rgB >> s_g;
    wire [30:0] sh_b = auB >> s_u;

    always @(posedge clk) begin
        if (clearing) begin
            validA <= 1'b0; validB <= 1'b0; validC <= 1'b0; idxB <= 2'd0; idxC <= 2'd0;
            acc_beats <= 16'd0; taken <= 16'd0;
        end else begin
            if (take_in) begin beatA <= s_axis_tdata; validA <= 1'b1; acc_beats <= acc_beats + 16'd1; end
            else if (a_move) validA <= 1'b0;
            if (a_move) begin
                validB <= 1'b1; idxB <= 2'd0;
                hB   <= beatA;
                rgB  <= beatA[31] ? 31'd0 : beatA[30:0];
                auB  <= beatA[63] ? (31'd0 - beatA[62:32]) : beatA[62:32];
                agB  <= beatA[79] ? (15'd0 - beatA[78:64]) : beatA[78:64];
                negB <= beatA[63] ^ beatA[79];
            end
            if (b_move) begin
                validC <= 1'b1;
                if (!mode) begin
                    xC <= sh_a[15:0]; yC <= sh_a[15:0]; bC <= sh_b[15:0]; gC <= agB; negC <= negB; idxC <= 2'd0;
                    validB <= 1'b0;
                end else begin
                    xC <= hsel[31] ? (16'd0 - hsel[15:0]) : hsel[15:0]; yC <= m_val;
                    bC <= 16'd0; gC <= 15'd0; negC <= hsel[31]; idxC <= idxB;
                    idxB <= idxB + 2'd1;
                    if (idxB == 2'd3) validB <= 1'b0;
                end
            end else if (c_take) validC <= 1'b0;
            if (c_take) taken <= taken + 16'd1;
        end
    end

    wire [CHAINS-1:0] slot_now, res_valid, res_last, res_neg;
    wire [17:0]       res_bus [0:CHAINS-1];
    wire [4:0]        s1 = mode ? 5'd14 : s_S;
    genvar c;
    generate
        for (c = 0; c < CHAINS; c = c + 1) begin : ch
            glue_chain #(.OFFSET(SPACING * c)) chain (
                .clk(clk), .clear(clearing), .fc(fc), .mode(mode), .s1(s1), .s2(s_T), .s3(s_V),
                .load(slot_now[c] && c_take), .load_last(last_in), .load_neg(negC),
                .load_x(xC), .load_y(yC), .load_b(bC), .load_g(gC),
                .slot_now(slot_now[c]), .res(res_bus[c]),
                .res_valid(res_valid[c]), .res_last(res_last[c]), .res_neg(res_neg[c])
            );
        end
    endgenerate

    wire [4:0]      doff = fc - (mode ? 5'd22 : 5'd4);
    wire            hit  = (doff[LOGS-1:0] == {LOGS{1'b0}});
    wire [LOGC-1:0] sel  = doff[4:LOGS];
    reg [17:0] resR;
    reg        rvalR, rlastR, rnegR;
    always @(posedge clk) begin
        resR <= res_bus[sel]; rlastR <= res_last[sel]; rnegR <= res_neg[sel];
        rvalR <= hit && res_valid[sel] && !clearing;
    end
    wire [15:0] vp   = resR[15:0];
    wire [16:0] hA   = rnegR ? (17'd0 - {1'b0, vp}) : {1'b0, vp};
    reg  [31:0] hR;
    reg  [17:0] QR;
    reg         rvalQ, rlastQ, rnegQ;
    always @(posedge clk) begin
        hR <= {{15{hA[16]}}, hA};
        QR <= {1'b0, resR[17:1]} + {17'd0, resR[0]};
        rvalQ <= rvalR && !clearing; rlastQ <= rlastR; rnegQ <= rnegR;
        if (!rstn) hmax <= 16'd0;
        else if (starting && !mode_ctrl) hmax <= 16'd0;
        else if (rvalR && !mode && vp > hmax) hmax <= vp;
    end
    wire [6:0]  qmag = (QR > 18'd127) ? 7'd127 : QR[6:0];
    wire [7:0]  qB   = rnegQ ? (8'd0 - {1'b0, qmag}) : {1'b0, qmag};
    reg  [1:0]  pos;
    reg  [31:0] pk, pkB;
    always @* begin pkB = pk; pkB[8 * pos +: 8] = qB; end
    wire        pushB = rvalQ && ((pos == 2'd3) || rlastQ);
    wire        push  = rvalQ && (!mode || pushB);
    wire [32:0] push_data = mode ? {rlastQ, pkB} : {rlastQ, hR};
    always @(posedge clk) begin
        if (clearing) begin pos <= 2'd0; pk <= 32'd0; end
        else if (rvalQ && mode) begin
            if (pushB) begin pos <= 2'd0; pk <= 32'd0; end
            else begin pk <= pkB; pos <= pos + 2'd1; end
        end
    end

    reg  [32:0] fmem [0:DEPTH-1];
    reg  [7:0]  wptr, rptr;
    wire        fempty   = (wptr == rptr);
    wire        out_free = !m_axis_tvalid || m_axis_tready;
    wire        pop      = out_free && !fempty;
    wire [32:0] head     = fmem[rptr[6:0]];
    always @(posedge clk) if (push) fmem[wptr[6:0]] <= push_data;
    always @(posedge clk) begin
        if (clearing) begin
            wptr <= 8'd0; rptr <= 8'd0; free <= 8'd128; done <= 15'd0;
            m_axis_tvalid <= 1'b0; m_axis_tlast <= 1'b0; m_axis_tdata <= 32'd0;
        end else begin
            if (push) wptr <= wptr + 8'd1;
            if (out_free) begin
                if (!fempty) begin
                    m_axis_tdata <= head[31:0]; m_axis_tlast <= head[32]; m_axis_tvalid <= 1'b1;
                    rptr <= rptr + 8'd1;
                end else m_axis_tvalid <= 1'b0;
            end
            free <= free + {7'd0, pop} - {7'd0, reserve};
            if (m_axis_tvalid && m_axis_tready) done <= done + (mode ? 15'd4 : 15'd1);
        end
    end
endmodule

module glue_chain #(parameter OFFSET = 0) (
    input         clk,
    input         clear,
    input  [4:0]  fc,
    input         mode,
    input  [4:0]  s1, s2, s3,
    input         load, load_last, load_neg,
    input  [15:0] load_x, load_y, load_b,
    input  [14:0] load_g,
    output        slot_now,
    output [17:0] res,
    output        res_valid, res_last, res_neg
);
    wire [4:0] ph1 = fc - OFFSET[4:0];
    wire [4:0] ph2 = ph1 + 5'd9;
    wire [4:0] ph3 = ph1 + 5'd18;
    assign slot_now = (ph1 == 5'd31);

    reg  [15:0] x1, y1;
    wire        p1;
    wire [17:0] hold1;
    mul16 m1 (.clk(clk), .clr(ph1 == 5'd31), .x(x1[0]), .y(y1[0]), .p(p1));
    glue_capture cap1 (.clk(clk), .ph(ph1), .s(s1), .p(p1), .hold(hold1));
    always @(posedge clk) begin
        if (ph1 == 5'd31) begin x1 <= load ? load_x : 16'd0; y1 <= load ? load_y : 16'd0; end
        else begin x1 <= {1'b0, x1[15:1]}; y1 <= {1'b0, y1[15:1]}; end
    end

    reg  [15:0] b1, b2;
    reg  [14:0] g1, g2, g3, g4;
    always @(posedge clk) begin
        if (ph1 == 5'd31) begin b1 <= load ? load_b : 16'd0; g1 <= load ? load_g : 15'd0; end
        if (ph1 == 5'd22) b2 <= b1;
        if (ph1 == 5'd13) begin g2 <= g1; g3 <= g2; g4 <= g3; end
    end

    reg  [15:0] x2, y2;
    wire        p2;
    wire [17:0] hold2;
    mul16 m2 (.clk(clk), .clr(ph2 == 5'd31), .x(x2[0]), .y(y2[0]), .p(p2));
    glue_capture cap2 (.clk(clk), .ph(ph2), .s(s2), .p(p2), .hold(hold2));
    always @(posedge clk) begin
        if (ph2 == 5'd31) begin x2 <= hold1[15:0]; y2 <= b2; end
        else begin x2 <= {1'b0, x2[15:1]}; y2 <= {1'b0, y2[15:1]}; end
    end

    reg  [15:0] x3, y3;
    wire        p3;
    wire [17:0] hold3;
    mul16 m3 (.clk(clk), .clr(ph3 == 5'd31), .x(x3[0]), .y(y3[0]), .p(p3));
    glue_capture cap3 (.clk(clk), .ph(ph3), .s(s3), .p(p3), .hold(hold3));
    always @(posedge clk) begin
        if (ph3 == 5'd31) begin x3 <= hold2[15:0]; y3 <= {1'b0, g4}; end
        else begin x3 <= {1'b0, x3[15:1]}; y3 <= {1'b0, y3[15:1]}; end
    end

    reg [2:0] f1, f2, f3, f4, f5, f6, fb2;
    always @(posedge clk) begin
        if (clear) begin
            f1 <= 3'd0; f2 <= 3'd0; f3 <= 3'd0; f4 <= 3'd0; f5 <= 3'd0; f6 <= 3'd0; fb2 <= 3'd0;
        end else begin
            if (ph1 == 5'd31) f1 <= {load, load && load_last, load && load_neg};
            if (ph1 == 5'd4) begin f2 <= f1; f3 <= f2; f4 <= f3; f5 <= f4; f6 <= f5; end
            if (ph1 == 5'd22) fb2 <= f1;
        end
    end
    assign res = mode ? hold1 : hold3;
    assign {res_valid, res_last, res_neg} = mode ? fb2 : f6;
endmodule

module glue_capture (
    input             clk,
    input      [4:0]  ph,
    input      [4:0]  s,
    input             p,
    output reg [17:0] hold
);
    wire [4:0] u        = ph - 5'd5;
    wire [5:0] hi       = {1'b0, s} + 6'd17;
    wire       cur      = ({1'b0, u} >= {1'b0, s}) && ({1'b0, u} <= hi);
    wire       fill     = ({1'b1, u} <= hi);
    wire       din      = cur ? p : 1'b0;
    wire       complete = ({1'b0, u} == hi) || ({1'b1, u} == hi);
    reg  [17:0] r;
    always @(posedge clk) begin
        if (cur || fill) r <= {din, r[17:1]};
        if (complete) hold <= {din, r[17:1]};
    end
endmodule
