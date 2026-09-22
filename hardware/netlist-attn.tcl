set part xck26-sfvc784-2LV-c
set groups 16
set slots 1
set rows 6
set mhz 250
if {[llength $argv] % 2} { error "arguments come in name value pairs: $argv" }
foreach {k v} $argv { if {![info exists $k]} { error "unknown argument $k" }; set $k $v }
set hw [file dirname [file normalize [info script]]]
create_project -in_memory -part $part
add_files [list [file join $hw attn_fx_v3.v] [file join $hw attn_fx_axi.v]]
set_property generic [list GROUPS=$groups SLOTS=$slots ROWS=$rows] [current_fileset]
synth_design -top attn_fx_axi -part $part -mode out_of_context -flatten_hierarchy rebuilt
set out [file join $hw attn_net_${groups}x${slots}x${rows}.v]
write_verilog -force -mode funcsim $out
puts "=== wrote $out"
