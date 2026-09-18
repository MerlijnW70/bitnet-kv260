set part xck26-sfvc784-2LV-c
set board xilinx.com:kv260_som:part0:2.0
set E 4
set mhz 250
set glue_mhz 250
set name ffn
set burst 64
set sburst 256
set swidth 128
set ports hp
set A 2
set hw [file dirname [file normalize [info script]]]
if {[llength $argv] % 2} { error "arguments come in name value pairs: $argv" }
foreach {k v} $argv {
    if {![info exists $k]} { error "unknown argument $k" }
    set $k $v
}
array set portgp {HPC0 0 HPC1 1 HP0 2 HP1 3 HP2 4 HP3 5}
switch -- $ports {
    hp     { set engport {HP0 HP1 HP2 HP3}; set glueport HPC0 }
    spread { set engport {HP0 HP2 HPC0 HPC1}; set glueport HP1 }
    default { error "unknown ports $ports" }
}
proc aclkpin {p} { return "ps/saxi[string tolower $p]_fpd_aclk" }
set two [expr {$glue_mhz != $mhz}]
set here [file dirname [file normalize [info script]]]
puts "=== start: [clock format [clock seconds]]"
puts "=== name $name  burst $burst  sburst $sburst  swidth $swidth  ports $ports: engines [join $engport { }], glue $glueport"
create_project -force -part $part ${name}_pl [file join $here work_$name]
set_property board_part $board [current_project]
add_files [list \
    [file join $hw ternary_matvec.v] \
    [file join $hw trit_decode_lut.v] \
    [file join $hw ternary_matvec_axi.v] \
    [file join $hw mul16.v] \
    [file join $hw ternary_glue.v] \
    [file join $hw ternary_glue_axi.v] \
    [file join $here attn_fx_v3.v] \
    [file join $here attn_fx_axi.v] \
    [file join $here ternary_port_axi.v]]
puts "=== attention on the first $A of $E engine ports"

proc ipdef {name} { return [lindex [get_ipdefs -quiet xilinx.com:ip:${name}:*] end] }

create_bd_design design_1
set ps [create_bd_cell -type ip -vlnv [ipdef zynq_ultra_ps_e] ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset 1} $ps
set pscfg [list \
    CONFIG.PSU__USE__M_AXI_GP0 1 CONFIG.PSU__USE__M_AXI_GP1 0 CONFIG.PSU__USE__M_AXI_GP2 0 \
    CONFIG.PSU__FPGA_PL0_ENABLE 1 CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $mhz]
if {$two} { lappend pscfg CONFIG.PSU__FPGA_PL1_ENABLE 1 CONFIG.PSU__CRL_APB__PL1_REF_CTRL__FREQMHZ $glue_mhz }
foreach p [concat $engport [list $glueport]] {
    set gp $portgp($p)
    lappend pscfg CONFIG.PSU__USE__S_AXI_GP${gp} 1 CONFIG.PSU__SAXIGP${gp}__DATA_WIDTH 128
}
set_property -dict $pscfg $ps
puts "=== pl_clk0 asked: $mhz MHz; PS reports ACT_FREQMHZ [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ $ps]"
if {$two} { puts "=== pl_clk1 asked: $glue_mhz MHz; PS reports ACT_FREQMHZ [get_property CONFIG.PSU__CRL_APB__PL1_REF_CTRL__ACT_FREQMHZ $ps]" }

set rst [create_bd_cell -type ip -vlnv [ipdef proc_sys_reset] rst]

set periph [create_bd_cell -type ip -vlnv [ipdef smartconnect] periph]
set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI [expr {2 * $E + 3}] CONFIG.NUM_CLKS [expr {$two ? 2 : 1}]] $periph
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins periph/S00_AXI]

set clkpins [list ps/pl_clk0 ps/maxihpm0_fpd_aclk rst/slowest_sync_clk periph/aclk]
set rstpins [list rst/peripheral_aresetn periph/aresetn]
for {set i 0} {$i < $E} {incr i} {
    set gpio [create_bd_cell -type ip -vlnv [ipdef axi_gpio] gpio$i]
    set_property -dict [list CONFIG.C_GPIO_WIDTH 32 CONFIG.C_IS_DUAL 1 CONFIG.C_GPIO2_WIDTH 32 CONFIG.C_ALL_OUTPUTS 1 CONFIG.C_ALL_INPUTS_2 1] $gpio

    set dma [create_bd_cell -type ip -vlnv [ipdef axi_dma] dma$i]
    set_property -dict [list \
        CONFIG.c_include_sg 0 CONFIG.c_sg_length_width 26 CONFIG.c_addr_width 40 \
        CONFIG.c_include_mm2s 1 CONFIG.c_m_axi_mm2s_data_width 128 CONFIG.c_m_axis_mm2s_tdata_width 128 \
        CONFIG.c_include_mm2s_dre 0 CONFIG.c_mm2s_burst_size $burst \
        CONFIG.c_include_s2mm 1 CONFIG.c_m_axi_s2mm_data_width $swidth CONFIG.c_s_axis_s2mm_tdata_width 32 \
        CONFIG.c_include_s2mm_dre 0 CONFIG.c_s2mm_burst_size $sburst] $dma

    if {$i < $A} {
        set eng [create_bd_cell -type module -reference ternary_port_axi eng$i]
        set_property -dict [list CONFIG.EXPF_HEX [file join $here attn_expf.hex] CONFIG.EXPI_HEX [file join $here attn_expi.hex]] $eng
    } else {
        set eng [create_bd_cell -type module -reference ternary_matvec_axi eng$i]
    }

    set p [lindex $engport $i]
    set mem [create_bd_cell -type ip -vlnv [ipdef smartconnect] mem$i]
    set_property -dict [list CONFIG.NUM_SI 2 CONFIG.NUM_MI 1] $mem
    connect_bd_intf_net [get_bd_intf_pins dma$i/M_AXI_MM2S] [get_bd_intf_pins mem$i/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins dma$i/M_AXI_S2MM] [get_bd_intf_pins mem$i/S01_AXI]
    connect_bd_intf_net [get_bd_intf_pins mem$i/M00_AXI] [get_bd_intf_pins ps/S_AXI_${p}_FPD]

    connect_bd_intf_net [get_bd_intf_pins periph/M[format %02d [expr {2 * $i}]]_AXI] [get_bd_intf_pins gpio$i/S_AXI]
    connect_bd_intf_net [get_bd_intf_pins periph/M[format %02d [expr {2 * $i + 1}]]_AXI] [get_bd_intf_pins dma$i/S_AXI_LITE]

    set rsi [create_bd_cell -type ip -vlnv [ipdef axis_register_slice] rsi$i]
    set_property -dict [list CONFIG.REG_CONFIG 8] $rsi
    set rso [create_bd_cell -type ip -vlnv [ipdef axis_register_slice] rso$i]
    set_property -dict [list CONFIG.REG_CONFIG 8] $rso
    connect_bd_intf_net [get_bd_intf_pins dma$i/M_AXIS_MM2S] [get_bd_intf_pins rsi$i/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins rsi$i/M_AXIS] [get_bd_intf_pins eng$i/s_axis]
    connect_bd_intf_net [get_bd_intf_pins eng$i/m_axis] [get_bd_intf_pins rso$i/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins rso$i/M_AXIS] [get_bd_intf_pins dma$i/S_AXIS_S2MM]

    connect_bd_net [get_bd_pins gpio$i/gpio_io_o] [get_bd_pins eng$i/gpo]
    connect_bd_net [get_bd_pins eng$i/gpi] [get_bd_pins gpio$i/gpio2_io_i]

    lappend clkpins [aclkpin $p] gpio$i/s_axi_aclk \
        dma$i/s_axi_lite_aclk dma$i/m_axi_mm2s_aclk dma$i/m_axi_s2mm_aclk \
        mem$i/aclk rsi$i/aclk rso$i/aclk eng$i/aclk
    lappend rstpins gpio$i/s_axi_aresetn dma$i/axi_resetn mem$i/aresetn rsi$i/aresetn rso$i/aresetn eng$i/aresetn
}

set ggpio_a [create_bd_cell -type ip -vlnv [ipdef axi_gpio] ggpio_a]
set_property -dict [list CONFIG.C_GPIO_WIDTH 32 CONFIG.C_IS_DUAL 1 CONFIG.C_GPIO2_WIDTH 32 CONFIG.C_ALL_OUTPUTS 1 CONFIG.C_ALL_INPUTS_2 1] $ggpio_a
set ggpio_b [create_bd_cell -type ip -vlnv [ipdef axi_gpio] ggpio_b]
set_property -dict [list CONFIG.C_GPIO_WIDTH 32 CONFIG.C_IS_DUAL 1 CONFIG.C_GPIO2_WIDTH 32 CONFIG.C_ALL_OUTPUTS 1 CONFIG.C_ALL_OUTPUTS_2 1] $ggpio_b
set gdma [create_bd_cell -type ip -vlnv [ipdef axi_dma] gdma]
set_property -dict [list \
    CONFIG.c_include_sg 0 CONFIG.c_sg_length_width 26 CONFIG.c_addr_width 40 \
    CONFIG.c_include_mm2s 1 CONFIG.c_m_axi_mm2s_data_width 128 CONFIG.c_m_axis_mm2s_tdata_width 128 \
    CONFIG.c_include_mm2s_dre 0 CONFIG.c_mm2s_burst_size $burst \
    CONFIG.c_include_s2mm 1 CONFIG.c_m_axi_s2mm_data_width $swidth CONFIG.c_s_axis_s2mm_tdata_width 32 \
    CONFIG.c_include_s2mm_dre 0 CONFIG.c_s2mm_burst_size $sburst] $gdma
set glue [create_bd_cell -type module -reference ternary_glue_axi glue]
set gmem [create_bd_cell -type ip -vlnv [ipdef smartconnect] gmem]
set_property -dict [list CONFIG.NUM_SI 2 CONFIG.NUM_MI 1] $gmem
connect_bd_intf_net [get_bd_intf_pins gdma/M_AXI_MM2S] [get_bd_intf_pins gmem/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins gdma/M_AXI_S2MM] [get_bd_intf_pins gmem/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins gmem/M00_AXI] [get_bd_intf_pins ps/S_AXI_${glueport}_FPD]
connect_bd_intf_net [get_bd_intf_pins periph/M[format %02d [expr {2 * $E}]]_AXI] [get_bd_intf_pins ggpio_a/S_AXI]
connect_bd_intf_net [get_bd_intf_pins periph/M[format %02d [expr {2 * $E + 1}]]_AXI] [get_bd_intf_pins ggpio_b/S_AXI]
connect_bd_intf_net [get_bd_intf_pins periph/M[format %02d [expr {2 * $E + 2}]]_AXI] [get_bd_intf_pins gdma/S_AXI_LITE]
set grsi [create_bd_cell -type ip -vlnv [ipdef axis_register_slice] grsi]
set_property -dict [list CONFIG.REG_CONFIG 8] $grsi
set grso [create_bd_cell -type ip -vlnv [ipdef axis_register_slice] grso]
set_property -dict [list CONFIG.REG_CONFIG 8] $grso
connect_bd_intf_net [get_bd_intf_pins gdma/M_AXIS_MM2S] [get_bd_intf_pins grsi/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins grsi/M_AXIS] [get_bd_intf_pins glue/s_axis]
connect_bd_intf_net [get_bd_intf_pins glue/m_axis] [get_bd_intf_pins grso/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins grso/M_AXIS] [get_bd_intf_pins gdma/S_AXIS_S2MM]
connect_bd_net [get_bd_pins ggpio_a/gpio_io_o] [get_bd_pins glue/ctrl]
connect_bd_net [get_bd_pins glue/status] [get_bd_pins ggpio_a/gpio2_io_i]
connect_bd_net [get_bd_pins ggpio_b/gpio_io_o] [get_bd_pins glue/params]
connect_bd_net [get_bd_pins ggpio_b/gpio2_io_o] [get_bd_pins glue/mval]
set gclkpins [list [aclkpin $glueport] ggpio_a/s_axi_aclk ggpio_b/s_axi_aclk \
    gdma/s_axi_lite_aclk gdma/m_axi_mm2s_aclk gdma/m_axi_s2mm_aclk \
    gmem/aclk grsi/aclk grso/aclk glue/aclk]
set grstpins [list ggpio_a/s_axi_aresetn ggpio_b/s_axi_aresetn gdma/axi_resetn gmem/aresetn grsi/aresetn grso/aresetn glue/aresetn]

if {$two} {
    set rstg [create_bd_cell -type ip -vlnv [ipdef proc_sys_reset] rstg]
    connect_bd_net [get_bd_pins [concat [list ps/pl_clk1 rstg/slowest_sync_clk periph/aclk1] $gclkpins]]
    connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rstg/ext_reset_in]
    connect_bd_net [get_bd_pins [concat [list rstg/peripheral_aresetn] $grstpins]]
} else {
    set clkpins [concat $clkpins $gclkpins]
    set rstpins [concat $rstpins $grstpins]
}
connect_bd_net [get_bd_pins $clkpins]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst/ext_reset_in]
connect_bd_net [get_bd_pins $rstpins]

for {set i 0} {$i < $E} {incr i} {
    assign_bd_address -target_address_space [get_bd_addr_spaces ps/Data] \
        -offset [format 0x%08X [expr {0xA0000000 + $i * 0x10000}]] -range 0x10000 [get_bd_addr_segs gpio$i/S_AXI/Reg]
    assign_bd_address -target_address_space [get_bd_addr_spaces ps/Data] \
        -offset [format 0x%08X [expr {0xA0040000 + $i * 0x10000}]] -range 0x10000 [get_bd_addr_segs dma$i/S_AXI_LITE/Reg]
}
assign_bd_address -target_address_space [get_bd_addr_spaces ps/Data] -offset 0xA0080000 -range 0x10000 [get_bd_addr_segs ggpio_a/S_AXI/Reg]
assign_bd_address -target_address_space [get_bd_addr_spaces ps/Data] -offset 0xA0090000 -range 0x10000 [get_bd_addr_segs ggpio_b/S_AXI/Reg]
assign_bd_address -target_address_space [get_bd_addr_spaces ps/Data] -offset 0xA00A0000 -range 0x10000 [get_bd_addr_segs gdma/S_AXI_LITE/Reg]
assign_bd_address
for {set i 0} {$i < $E} {incr i} {
    set p [lindex $engport $i]
    set gp $portgp($p)
    foreach ch {Data_MM2S Data_S2MM} {
        foreach seg [list ${p}_QSPI ${p}_LPS_OCM] {
            set r [catch {exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces dma$i/$ch] [get_bd_addr_segs ps/SAXIGP${gp}/$seg]} excl]
            puts "=== exclude $seg from dma$i/$ch: [expr {$r ? $excl : "done"}]"
        }
    }
}
foreach ch {Data_MM2S Data_S2MM} {
    foreach seg [list ${glueport}_QSPI ${glueport}_LPS_OCM] {
        set r [catch {exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces gdma/$ch] [get_bd_addr_segs ps/SAXIGP$portgp($glueport)/$seg]} excl]
        puts "=== exclude $seg from gdma/$ch: [expr {$r ? $excl : "done"}]"
    }
}
foreach sp [get_bd_addr_spaces] {
    foreach seg [get_bd_addr_segs -of_objects $sp] {
        puts "=== address: [get_property PATH $sp] -> [get_property NAME $seg] offset [get_property OFFSET $seg] range [get_property RANGE $seg]"
    }
}
puts "=== pl_clk0: [get_property CONFIG.FREQ_HZ [get_bd_pins ps/pl_clk0]] Hz"
if {$two} { puts "=== pl_clk1: [get_property CONFIG.FREQ_HZ [get_bd_pins ps/pl_clk1]] Hz" }

regenerate_bd_layout
validate_bd_design
save_bd_design

set wrap [make_wrapper -files [get_files design_1.bd] -top]
add_files -norecurse $wrap
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
launch_runs synth_1 -jobs 8
wait_on_run synth_1
puts "=== synth status: [get_property STATUS [get_runs synth_1]]"
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
puts "=== impl status: [get_property STATUS [get_runs impl_1]]  progress: [get_property PROGRESS [get_runs impl_1]]"

set every 0
foreach log [glob -nocomplain [file join $here work_$name ${name}_pl.runs * runme.log]] {
    set f [open $log r]
    while {[gets $f line] >= 0} {
        if {[string match "CRITICAL WARNING*" $line]} {
            incr every
            puts "=== [file tail [file dirname $log]]: $line"
        }
    }
    close $f
}
puts "=== critical warnings in every run of this build: $every"
foreach run {synth_1 impl_1} {
    set log [file join $here work_$name ${name}_pl.runs $run runme.log]
    set n 0
    if {[file exists $log]} {
        set f [open $log r]
        while {[gets $f line] >= 0} { if {[string match "CRITICAL WARNING*" $line]} { incr n; puts "=== $run: $line" } }
        close $f
    }
    puts "=== critical warnings in $run: $n"
}

open_run impl_1
report_utilization -file [file join $here $name-utilization.txt]
report_utilization -hierarchical -hierarchical_depth 4 -file [file join $here $name-utilization-hier.txt]
report_timing_summary -max_paths 3 -file [file join $here $name-timing.txt]
puts "=== WNS: [get_property STATS.WNS [get_runs impl_1]] ns  TNS: [get_property STATS.TNS [get_runs impl_1]] ns  WHS: [get_property STATS.WHS [get_runs impl_1]] ns  THS: [get_property STATS.THS [get_runs impl_1]] ns"
foreach clk [get_clocks] {
    puts "=== clock: [get_property PERIOD $clk] ns period on $clk"
}
proc lutff {rep} {
    set l ?; set f ?
    regexp {CLB LUTs\*?\s*\|\s*(\d+)} $rep -> l
    regexp {CLB Registers\s*\|\s*(\d+)} $rep -> f
    return "LUTs $l FFs $f"
}
puts "=== totals: [lutff [report_utilization -return_string]]"
foreach {label pat} {engine0 *eng0/inst/engine glue *glue/inst/glue rsi0 *rsi0/inst rso0 *rso0/inst dma0 *dma0/U0 mem0 *mem0/inst gdma *gdma/U0 gmem *gmem/inst grsi *grsi/inst grso *grso/inst gpio0 *gpio0/U0 ggpio_a *ggpio_a/U0 ggpio_b *ggpio_b/U0 periph *periph/inst} {
    set c [get_cells -hierarchical -filter "NAME =~ $pat"]
    if {[llength $c] == 1} { puts "=== $label ($c): [lutff [report_utilization -cells $c -return_string]]" } else { puts "=== $label: [llength $c] cells match $pat" }
}
set muls [lsort [get_cells -hierarchical -filter {NAME =~ *chain/m1 || NAME =~ *chain/m2 || NAME =~ *chain/m3}]]
puts "=== mul16 pieces: [llength $muls]"
if {[llength $muls] > 0} { puts "=== mul16 ([lindex $muls 0]): [lutff [report_utilization -cells [get_cells [lindex $muls 0]] -return_string]]" }
puts "=== engine0: [report_utilization -cells [get_cells -hierarchical -filter {NAME =~ *eng0/inst/engine}] -return_string]"
puts "=== glue: [report_utilization -cells [get_cells -hierarchical -filter {NAME =~ *glue/inst/glue}] -return_string]"
puts "=== dma0: [report_utilization -cells [get_cells -hierarchical -filter {NAME =~ *dma0/U0}] -return_string]"

set bit [glob -nocomplain [file join $here work_$name ${name}_pl.runs impl_1 *.bit]]
puts "=== bitstream: $bit"
write_cfgmem -force -format bin -interface SMAPx32 -disablebitswap -loadbit "up 0x0 $bit" [file join $here $name.bit.bin]
puts "=== wrote $name.bit.bin"
puts "=== end: [clock format [clock seconds]]"
