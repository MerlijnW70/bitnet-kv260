`timescale 1ns/1ps
module ternary_port_axi #(
    parameter SLICES = 2,
    parameter ULTRA  = 1,
    parameter EXPF_HEX = "attn_expf.hex",
    parameter EXPI_HEX = "attn_expi.hex"
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis:m_axis, ASSOCIATED_RESET aresetn" *)
    input          aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input          aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TDATA" *)
    input  [127:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TKEEP" *)
    input   [15:0] s_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TVALID" *)
    input          s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TREADY" *)
    output         s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TLAST" *)
    input          s_axis_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TDATA" *)
    output  [31:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TKEEP" *)
    output   [3:0] m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TVALID" *)
    output         m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TREADY" *)
    input          m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TLAST" *)
    output         m_axis_tlast,

    input   [31:0] gpo,
    output  [31:0] gpi
);
    wire sel = gpo[15];

    wire        mv_s_ready, at_s_ready;
    wire [31:0] mv_m_data, at_m_data;
    wire        mv_m_valid, at_m_valid, mv_m_last, at_m_last;
    wire [31:0] mv_status, at_status;

    ternary_matvec #(.SLICES(SLICES), .ULTRA(ULTRA)) engine (
        .clk(aclk), .rstn(aresetn),
        .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid && !sel), .s_axis_tready(mv_s_ready),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(mv_m_data), .m_axis_tvalid(mv_m_valid), .m_axis_tready(m_axis_tready && !sel),
        .m_axis_tlast(mv_m_last),
        .ctrl({gpo[31:16], 1'b0, gpo[14:0]}), .status(mv_status)
    );

    attn_fx_axi #(.EXPF_HEX(EXPF_HEX), .EXPI_HEX(EXPI_HEX)) attention (
        .clk(aclk), .rstn(aresetn),
        .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid && sel), .s_axis_tready(at_s_ready),
        .m_axis_tdata(at_m_data), .m_axis_tvalid(at_m_valid), .m_axis_tready(m_axis_tready && sel),
        .m_axis_tlast(at_m_last), .status(at_status)
    );

    assign s_axis_tready = sel ? at_s_ready : mv_s_ready;
    assign m_axis_tdata  = sel ? at_m_data  : mv_m_data;
    assign m_axis_tvalid = sel ? at_m_valid : mv_m_valid;
    assign m_axis_tlast  = sel ? at_m_last  : mv_m_last;
    assign m_axis_tkeep  = 4'b1111;
    assign gpi           = sel ? at_status  : mv_status;
endmodule
