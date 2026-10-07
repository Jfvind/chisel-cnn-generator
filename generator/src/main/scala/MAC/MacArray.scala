package mac

import chisel3._
import chisel3.util._

class MacArray(width: Int, activations: Int) extends Module {
    val sumMaxSize = width * 2 + log2Ceil(activations)

    val io = IO(new Bundle {
        val acts    = Input(Vec(activations, UInt(width.W)))
        val weights = Input(Vec(activations, SInt(width.W)))
        val bias    = Input(SInt(width.W))
        val sum     = Output(SInt(sumMaxSize.W))
    })

    val products = io.acts.zip(io.weights).map { case (a, w) => (a*w).pad(sumMaxSize) } // Generates af Seq of acts*weights for each value pair and pads to shared width

    io.sum := VecInit(products).reduceTree(_ + _) + io.bias
}