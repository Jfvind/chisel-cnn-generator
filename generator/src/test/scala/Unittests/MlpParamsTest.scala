package mlp

import chisel3._
import circt.stage.ChiselStage
import model.LayerParams
import org.scalatest.flatspec.AnyFlatSpec

class MlpParamsTest extends AnyFlatSpec {
    // MNIST shape with dummy values: 784 -> 128 -> 10
    def layer(index: Int, in: Int, out: Int) =
        LayerParams(index, Seq.fill(in)(Seq.fill(out)(0)), Seq.fill(out)(0), 0)
    val mnist = Seq(layer(0, 784, 128), layer(2, 128, 10))

    "MlpParams" should "derive sizes and widths from the layers" in {
        val p = MlpParams(mnist)
        assert(p.nIn == 784 && p.nOut == 10)
        assert(p.accBits(mnist(0)) == 26)       // 8 + 8 + log2Ceil(784)
        assert(p.scoreBits == 23)               // 8 + 8 + log2Ceil(128)
        assert(p.romDepth(mnist(0)) == 784 * 128)
        assert(p.romFile(mnist(1)) == "l2_w.mem")
    }

    it should "pad the last read when mnc does not divide the inputs" in {
        val p = MlpParams(mnist, mnc = 3)
        assert(p.words(784) == 262)             // 261 full reads + 1 read with one real input
        assert(p.romDepth(mnist(0)) == 128 * 262)
        assert(p.romWidth == 24)
    }

    it should "elaborate the datapath IO" in {
        class Dummy(p: MlpParams) extends MlpDatapath(p) {
            io.in.en   := false.B
            io.in.addr := 0.U
            io.done    := false.B
            io.scores  := VecInit(Seq.fill(p.nOut)(0.S(p.scoreBits.W)))
            io.ovf     := 0.U
        }
        val sv = ChiselStage.emitSystemVerilog(new Dummy(MlpParams(mnist, mnc = 4)))
        assert(sv.contains("io_scores_9"))      // 10 scores
        assert(sv.contains("io_in_data_3"))     // 4 lanes
    }
}