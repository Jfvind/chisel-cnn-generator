package mlp

import chisel3._
import chisel3.util._

import shell.StreamBeat

// The output side of MlpCore: sends the nOut logits as stream beats (saturated to streamW, TLAST on the last
// beat of the batch) and tracks the argmax while they leave; RESULT = the class of the last image.
class OutputArgmax(nOut: Int, accW: Int, streamW: Int) extends Module {
    val io = IO(new Bundle {
        val logits   = Input(Vec(nOut, SInt(accW.W)))
        val valid    = Input(Bool())                       // the logits are ready to send
        val endBatch = Input(Bool())                       // this image is the batch's last: TLAST on its last beat
        val start    = Input(Bool())                       // CTRL start: restart at beat 0
        val out      = Decoupled(new StreamBeat(streamW))
        val lastFire = Output(Bool())                      // the image's last beat leaves this cycle
        val result   = Output(UInt(32.W))
    })

    val oReg       = RegInit(0.U(log2Ceil(nOut max 2).W))      // output beat index
    val bestValReg = Reg(SInt(accW.W))
    val bestIdxReg = Reg(UInt(32.W))
    val resultReg  = RegInit(0.U(32.W))
    val cur        = MlpCore.pick(io.logits, oReg)
    val isBetter   = oReg === 0.U || cur > bestValReg          // Only greater. lowest index wins ties
    val lastBeat   = oReg === (nOut - 1).U                     // Last score of this image

    io.out.valid     := io.valid
    io.out.bits.data := saturate(cur)
    io.out.bits.last := lastBeat && io.endBatch
    io.lastFire      := io.out.fire && lastBeat
    io.result        := resultReg

    when(io.out.fire) {
        when(isBetter) { bestValReg := cur; bestIdxReg := oReg }
        oReg := Mux(lastBeat, 0.U, oReg + 1.U)
        when(lastBeat) { resultReg := Mux(isBetter, oReg, bestIdxReg) }   // RESULT valid the cycle after done
    }
    when(io.start) { oReg := 0.U }                             // last: overrides the update above

    // Two's complement in streamW bits, clamped when the logit does not fit
    def saturate(s: SInt): UInt = {
        if (streamW >= accW) s.pad(streamW).asUInt
        else {
            val hi = ((BigInt(1) << (streamW - 1)) - 1).S(accW.W)
            val lo = (-(BigInt(1) << (streamW - 1))).S(accW.W)
            val clamped = Mux(s > hi, hi, Mux(s < lo, lo, s)).asUInt
            clamped(streamW - 1, 0)
        }
    }
}
