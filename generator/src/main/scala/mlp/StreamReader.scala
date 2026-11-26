package mlp

import chisel3._
import chisel3.util._

import shell.StreamBeat

// The input side of MlpCore: unpacks K pixels per clock from stream words of ppb pixels (pixel 0 in the
// lowest bits) and checks the packet length.
// Packet check: TLAST must come with the last word of the batch's last image (expectEnd).
//   too few words (TLAST early) -> short: the rest of this image is zeros and the batch ends here.
//   too many words (no TLAST)   -> the extra words are drained until TLAST, never reaching the core.
//   Both raise err until CTRL start.
class StreamReader(streamW: Int, bits: Int, k: Int, ppb: Int) extends Module {
    val io = IO(new Bundle {
        val in         = Flipped(Decoupled(new StreamBeat(streamW)))
        val start      = Input(Bool())                  // CTRL start: drop a half-used word, clear the packet check
        val req        = Input(Bool())                  // the controller takes a group of K pixels this cycle if valid
        val endOfImage = Input(Bool())                  // this group is the image's last; the rest of the word is padding
        val expectEnd  = Input(Bool())                  // this word must carry TLAST
        val pixels     = Output(Vec(k, UInt(bits.W)))
        val valid      = Output(Bool())
        val short      = Output(Bool())                 // TLAST came early: the batch ends with this image
        val err        = Output(Bool())                 // packet length wrong (sticky until start)
    })

    val gpw         = ppb / k                                   // K-pixel groups per word: 4 / 4 = 1
    val wordReg     = Reg(UInt(streamW.W))                      // the word being unpacked
    val haveReg     = RegInit(0.U(log2Ceil(gpw + 1).W))         // groups left in wordReg
    val shortPktReg = RegInit(false.B)
    val longPktReg  = RegInit(false.B)
    val drainingReg = RegInit(false.B)                          // discarding extra words until TLAST

    val inValid   = io.in.valid && !drainingReg                 // the core never sees words that are being drained
    val zeroFill  = shortPktReg && haveReg === 0.U              // after an early TLAST: missing pixels are 0
    val grp       = Mux(haveReg =/= 0.U, wordReg, Mux(zeroFill, 0.U, io.in.bits.data))
    val takeWord  = io.req && haveReg === 0.U && !shortPktReg
    io.in.ready  := takeWord || drainingReg
    io.valid     := haveReg =/= 0.U || inValid || zeroFill
    io.pixels    := MlpCore.fields(grp, k, bits)
    io.short     := shortPktReg
    io.err       := shortPktReg || longPktReg

    val wordIn = takeWord && io.in.valid                        // a word for the core was taken this cycle
    when(wordIn) {
        when(io.in.bits.last && !io.expectEnd) { shortPktReg := true.B }                     // too few words
        when(!io.in.bits.last && io.expectEnd) { longPktReg := true.B; drainingReg := true.B }  // too many words
    }
    when(drainingReg && io.in.valid && io.in.bits.last) { drainingReg := false.B }              // extra words gone

    when(io.req && io.valid) {                                  // a group is issued
        when(haveReg === 0.U && !zeroFill) {                    // took group 0 straight from the stream
            wordReg := io.in.bits.data >> (k * bits)
            haveReg := Mux(io.endOfImage, 0.U, (gpw - 1).U)
        }.elsewhen(haveReg =/= 0.U) {
            wordReg := wordReg >> (k * bits)
            haveReg := Mux(io.endOfImage, 0.U, haveReg - 1.U)
        }
    }

    when(io.start) {                                            // last: overrides the updates above
        haveReg     := 0.U                                      // drops a half-used word; wordReg is then unused
        shortPktReg := false.B
        longPktReg  := false.B
        drainingReg := false.B
    }
}
