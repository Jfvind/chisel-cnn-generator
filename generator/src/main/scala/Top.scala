package generator

import chisel3._
import chisel3.util._

class Top() extends Module {
    val io = IO(new Bundle {
        val in  = Input(UInt(1.W))
        val out = Output(UInt(1.W))
    })

    io.out := io.in
}


/**
 * Verilog generation.
 * Usage: sbt "runMain generator.TopGen"
 */
object TopGen extends App {
  emitVerilog(new Top(), Array("--target-dir", "generated"))
}