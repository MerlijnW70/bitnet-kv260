set part xck26-sfvc784-2LV-c
set groups 5
set slots 4
set rows 8
set mhz 250
if {[llength $argv] % 2} { error "arguments come in name value pairs: $argv" }
foreach {k v} $argv { if {![info exists $k]} { error "unknown argument $k" }; set $k $v }
set hw [file dirname [file normalize [info script]]]
set period [expr {1000.0 / $mhz}]
create_project -in_memory -part $part
add_files [list [file join $hw attn_fx_v3.v] [file join $hw attn_fx_axi.v]]
set_property generic [list GROUPS=$groups SLOTS=$slots ROWS=$rows] [current_fileset]
set xdc [file join $hw fit-attn.xdc]
set fh [open $xdc w]
puts $fh "create_clock -period $period -name clk \[get_ports clk\]"
close $fh
synth_design -top attn_fx_axi -part $part -mode out_of_context -flatten_hierarchy rebuilt
read_xdc -mode out_of_context $xdc
opt_design -quiet
report_utilization -file [file join $hw fit-util-${groups}x${slots}x${rows}.txt]
set p [get_timing_paths -max_paths 1 -nworst 1 -setup -from [all_registers] -to [all_registers]]
puts "=== shape ${groups}x${slots} of [expr {16*$rows}] at ${mhz} MHz: reg-to-reg slack [get_property SLACK $p] ns"
puts [report_timing -of_objects $p -return_string]
