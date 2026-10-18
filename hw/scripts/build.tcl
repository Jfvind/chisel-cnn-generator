# =============================================================================
# build.tcl — Vivado batch build for the PYNQ-Z2:  RTL from ShellGen  ->  .bit + .hwh
#
#   vivado -mode batch -source hw/scripts/build.tcl -tclargs <rtl_dir> <out_dir> [name] [jobs] [deploy]
#
#   rtl_dir  directory written by ShellGen
#   out_dir  where the project, reports and the overlay end up (hw/vivado is git-ignored)
#   name     overlay name (default: accel) -> <out_dir>/overlay/<name>.bit/.hwh
#   jobs     parallel implementation jobs (default: 4)
#   deploy   1 = copy to the board and run the smoke test (hw/scripts/deploy.py)
#
# Block design (identical for every core, only accel_top changes):
#   PS7 M_AXI_GP0 --AXI-Lite--> accel/s_axi            (registers)
#                  \----------> axi_dma_0/S_AXI_LITE   (DMA control, used by PYNQ)
#   axi_dma_0 M_AXIS_MM2S ----> accel/s_axis           (DDR -> core)
#   accel/m_axis -------------> axi_dma_0 S_AXIS_S2MM  (core -> DDR, needs TLAST)
#   axi_dma_0 M_AXI_MM2S/S2MM -> PS7 S_AXI_HP0          (DDR)
#
# Steps: args -> project + board files -> block design -> wrapper -> impl ->
#        timing gate + reports -> export .bit/.hwh/manifest -> optional deploy
# =============================================================================

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file normalize "$script_dir/../.."]

# Arguments land in argv (from -tclargs)
set rtl_dir [file normalize [lindex $argv 0]]
set out_dir [file normalize [lindex $argv 1]]
set name    [expr {[llength $argv] > 2 ? [lindex $argv 2] : "accel"}]
set jobs    [expr {[llength $argv] > 3 ? [lindex $argv 3] : 4}]
set deploy  [expr {[llength $argv] > 4 ? [lindex $argv 4] : 0}]

if {$rtl_dir eq "" || ![file exists "$rtl_dir/shell_params.tcl"]} {
    error "rtl_dir '$rtl_dir' has no shell_params.tcl -- run ShellGen first"
}
source "$rtl_dir/shell_params.tcl";     # defines CORE, STREAM_W, ADDR_W, FCLK_MHZ
puts "### Building '$name': core=$CORE stream=${STREAM_W}b fclk=${FCLK_MHZ}MHz"

set proj_dir    "$out_dir/project"
set report_dir  "$out_dir/reports"
set overlay_dir "$out_dir/overlay"
file mkdir $report_dir $overlay_dir

# --- Project ---
set_param board.repoPaths [list "$repo_root/hw/boards"];      # using board files from the repo
create_project -force $name $proj_dir -part xc7z020clg400-1;    # -force: overwriting last project
set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]

add_files -norecurse [glob "$rtl_dir/*.sv" "$rtl_dir/*.v"];     # AccelShell.sv + accel_top.v
update_compile_order -fileset sources_1

# --- Block design ---
create_bd_design $name

# Zynq PS. Apply_board_preset sets the DDR, MIO and clocks.
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
    [get_bd_cells ps7]
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $FCLK_MHZ \
]   [get_bd_cells ps7]
# GP0 = ARM -> Fabric (registers). HP0 = Fabric -> DDR (for DMA). FCLK0 = our set clock

# Our design as a reference to the generated Verilog wrapper
create_bd_cell -type module -reference accel_top accel

# AXI DMA
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma axi_dma_0
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_m_axis_mm2s_tdata_width $STREAM_W \
    CONFIG.c_s_axis_s2mm_tdata_width $STREAM_W \
    CONFIG.c_mm2s_burst_size {16} \
    CONFIG.c_s2mm_burst_size {16} \
]   [get_bd_cells axi_dma_0]
# length_width 26: max transfer 64MB

# AXI-LITE connections
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/ps7/M_AXI_GP0} Slave {/accel/s_axi} ddr_seg {Auto} \
    intc_ip {New AXI Interconnect} master_apm {0}} [get_bd_intf_pins accel/s_axi]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/ps7/M_AXI_GP0} Slave {/axi_dma_0/S_AXI_LITE} ddr_seg {Auto} \
    intc_ip {Auto} master_apm {0}} [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

# DMA memory side -> DDR via HP0
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/axi_dma_0/M_AXI_MM2S} Slave {/ps7/S_AXI_HP0} ddr_seg {Auto} \
    intc_ip {New AXI Interconnect} master_apm {0}} [get_bd_intf_pins ps7/S_AXI_HP0]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/axi_dma_0/M_AXI_S2MM} Slave {/ps7/S_AXI_HP0} ddr_seg {Auto} \
    intc_ip {Auto} master_apm {0}} [get_bd_intf_pins axi_dma_0/M_AXI_S2MM]

# Streams
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins accel/s_axis]
connect_bd_intf_net [get_bd_intf_pins accel/m_axis] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# If automation doesn't wire accel's clock/reset, here it's manual
if {[llength [get_bd_nets -quiet -of_objects [get_bd_pins accel/aclk]]] == 0} {
    connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins accel/aclk]
}
if {[llength [get_bd_nets -quiet -of_objects [get_bd_pins accel/aresetn]]] == 0} {
    connect_bd_net [get_bd_pins [lindex [get_bd_cells -filter {VLNV =~ *proc_sys_reset*}] 0]/peripheral_aresetn] \
    [get_bd_pins accel/aresetn]
}

assign_bd_address;      # gives accel and DMA addresses in GP0
validate_bd_design;     # fails here on width, clock or connection mistakes
save_bd_design

# HDL wrapper around block design, becomes top module
set bd_file [get_files "$name.bd"]
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper
set_property top ${name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

#Synthesis + impl + bitstream
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "Implementation failed -- see $proj_dir/$name.runs/impl_1/runme.log"
}

# Reports + timing gate
open_run impl_1
report_timing_summary -file "$report_dir/timing_summary.rpt"
report_utilization    -file "$report_dir/utilization.rpt"
report_utilization -hierarchical -file "$report_dir/utilization_hier.rpt";   # Usage per module. Used in evaluation of synthesis
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]];  # worst negative slack
puts "### WNS = $wns ns @ ${FCLK_MHZ} MHz"
if {$wns < 0} {
    error "Timing not met (WNS: $wns ns). Lower FCLK_MHZ or pipeline the core."
}

# --- Export: PYNQ needs <name>.bit and <name>.hwh
set bit [glob "$proj_dir/$name.runs/impl_1/${name}_wrapper.bit"]
set hwh [glob -nocomplain "$proj_dir/$name.gen/sources_1/bd/$name/hw_handoff/$name.hwh" \
                          "$proj_dir/$name.srcs/sources_1/bd/$name/hw_handoff/$name.hwh"]
if {[llength $hwh] == 0} { error "No .hwh found for $name" }
file copy -force $bit                        "$overlay_dir/$name.bit"
file copy -force [lindex $hwh 0]             "$overlay_dir/$name.hwh"
file copy -force "$rtl_dir/hw_manifest.json" "$overlay_dir/hw_manifest.json"
puts "### Overlay ready in $overlay_dir"

# --- Optional: deploy + run on board ---
if {$deploy} {
    set py [expr {[info exists ::env(PYTHON)] ? $::env(PYTHON) : "python"}]
    exec $py "$script_dir/deploy.py" --overlay "$overlay_dir/$name.bit" >@stdout 2>@stderr <@stdin
}