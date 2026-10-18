package shell

import chisel3._
import chisel3.util._

// =============================================================================
// AxiLiteSlave.scala — generic AXI4-Lite slave protocol engine (32-bit data,
// one outstanding transaction per direction). Knows nothing about the register map.
//
// RegWrite     : bundle (addr, data, strb) for a completed write.
// AxiLiteSlave : module. Outputs a one-cycle Valid(RegWrite) per completed write;
//                for reads it exposes rdAddr and captures rdData (driven
//                combinationally by the parent) at the AR handshake.
//
// Protocol rules it must obey:
//   - AW and W accepted independently, in any order or in the same cycle.
//   - Write performed when both have arrived; BVALID held until BREADY.
//   - No new AW/W accepted while a write response is pending.
//   - RVALID held until RREADY; no new AR accepted meanwhile.
//   - Always responds OKAY (00).
// =============================================================================

// Handed to the parent by the slave after a write
class RegWrite(val addrW: Int) extends Bundle {
    val addr = UInt(addrW.W)
    val data = UInt(32.W)
    val strb = UInt(4.W)      // Byte enables, e.g. strb = b0011 only data[15:0] is replaced at the addr 
}

class AxiLiteSlave(addrW: Int) extends Module {
    val io = IO(new Bundle {
        val axi    = new AxiLiteSlaveIO(addrW)              // The bus
        val wr     = Output(Valid(new RegWrite(addrW)))     // valid one-cycle pulse per completed write
        val rdAddr = Output(UInt(addrW.W))                  // telling parent which register is read
        val rdData = Input(UInt(32.W))                      // Parent answers with data
    })

    // ======= Write Logic ==========
    // AW and W can arrive in any order, so each needs a register flag
    val awFull = RegInit(false.B)       // Valid address but no write yet
    val awAddr = Reg(UInt(addrW.W))
    val wFull  = RegInit(false.B)       // Data accepted
    val wData  = Reg(UInt(32.W))
    val wStrb  = Reg(UInt(4.W))
    val bValid = RegInit(false.B)       // Awaiting ready from Master

    // ready when holding/flag register is empty/false and no response to the master is pending
    io.axi.awready := !awFull && !bValid
    io.axi.wready  := !wFull && !bValid

    when(io.axi.awvalid && io.axi.awready) { // AW successful handshake
        awFull := true.B
        awAddr := io.axi.awaddr
    }
    when(io.axi.wvalid && io.axi.wready) { // W successful handshake
        wFull := true.B
        wData := io.axi.wdata
        wStrb := io.axi.wstrb
    }

    val doWrite = awFull && wFull     // write when both address and data are present
    io.wr.valid := doWrite
    io.wr.bits.addr := awAddr
    io.wr.bits.data := wData
    io.wr.bits.strb := wStrb
    when(doWrite) {
        awFull := false.B
        wFull  := false.B
        bValid := true.B             // Data valid, waiting on master
    }

    io.axi.bvalid := bValid
    io.axi.bresp  := 0.U             // always OK
    when(bValid && io.axi.bready) { bValid := false.B }  // Master confirms that the response is received

    // ======== Read Logic ========
    val rValid = RegInit(false.B)     // read data waiting for master
    val rData  = Reg(UInt(32.W))

    io.axi.arready := !rValid        // accepting new address when last read is done
    io.rdAddr := io.axi.araddr
    when(io.axi.arvalid && io.axi.arready) { // AR handshake
        rValid := true.B
        rData  := io.rdData
    }
    when(rValid && io.axi.rready) { rValid := false.B } // R handshake: data has been transferred

    io.axi.rvalid := rValid         // demanded high until rready, by AXI protocol
    io.axi.rdata  := rData
    io.axi.rresp  := 0.U
}