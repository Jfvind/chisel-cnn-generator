package shell

import chisel3._

// =============================================================================
// Axi.scala — AXI port bundles used by the shell.
//
// AxiLiteSlaveIO : AXI4-Lite slave port seen from the slave (5 channels: AW, W, B, AR, R).
//                  Field names are the lower-case AXI signal names, so a port declared as
//                  `val s_axi = IO(new AxiLiteSlaveIO(...))` becomes s_axi_awaddr, s_axi_wdata, ...
//                  which is exactly what Vivado needs to infer the interface.
// AxisIO         : AXI4-Stream seen from the master (tdata, tvalid, tready, tlast).
//                  Use Flipped(new AxisIO(w)) for a slave port.
// StreamBeat     : one beat of a stream inside the design (data + last, no handshake).
// =============================================================================

class AxiLiteSlaveIO(val addrW: Int, val dataW: Int = 32) extends Bundle {
    // Address write channel (M -> S, meaning Input is sent from Master to Slave and output from S to M)
    val awaddr  = Input(UInt(addrW.W))           // register address
    val awprot  = Input(UInt(3.W))               // protection bits: default, not used
    val awvalid = Input(Bool())                  // awaddr is valid
    val awready = Output(Bool())                 // Slave: "ready for address"
    // Write data channel (M -> S)
    val wdata  = Input(UInt(dataW.W))            // send data
    val wstrb  = Input(UInt((dataW / 8).W))      // one bit per byte lane: which bytes to write
    val wvalid = Input(Bool())
    val wready = Output(Bool())
    // Write response channel (S -> M)
    val bresp  = Output(UInt(2.W))               // 00 = OK, 10 = SLVERR, 11 = DECERR
    val bvalid = Output(Bool())                  // Slave: "response is valid"
    val bready = Input(Bool())                   // Master: "ready to receive response"
    // Address read channel (M -> S)
    val araddr  = Input(UInt(addrW.W))
    val arprot  = Input(UInt(3.W))
    val arvalid = Input(Bool())
    val arready = Output(Bool())
    // Read data channel (S -> M)
    val rdata  = Output(UInt(dataW.W))
    val rresp  = Output(UInt(2.W))
    val rvalid = Output(Bool())
    val rready = Input(Bool())
}

// AXI4-Stream seen from the master, which pushes the data
class AxisIO(val dataW: Int) extends Bundle {
    val tdata  = Output(UInt(dataW.W))  // payload
    val tvalid = Output(Bool())         // master has a beat (many beats form a packet)
    val tready = Input(Bool())          // slave accepts data (backpressure if not)
    val tlast  = Output(Bool())         // last beat of a packet
}

// One beat in the design. The flow of data without handshaking.
class StreamBeat(val dataW: Int) extends Bundle {
    val data = UInt(dataW.W)            // No direction, as Decoupled() assigns it
    val last = Bool()
}