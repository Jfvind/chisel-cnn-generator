package mac

import chisel3._
import chisel3.util._

class MacArray(width: Int, prevLayerSize: Int) extends Module {
    val sumMaxSize = width * 2 + log2Ceil(prevLayerSize)

    val io = IO(new Bundle {
        val acts    = Input(Vec(prevLayerSize, SInt(width.W)))
        val weights = Input(Vec(prevLayerSize, SInt(width.W)))
        val bias    = Input(SInt(width.W))
        val sum     = Output(SInt((width*2 + log2Ceil(prevLayerSize)).W))
    })

    val products = io.acts.zip(io.weights).map { case (a, w) => (a*w).pad(sumMaxSize) } // Generates af Seq of acts*weights for each value pair and pads to shared width

    io.sum := VecInit(products).reduceTree(_ + _) + io.bias
}