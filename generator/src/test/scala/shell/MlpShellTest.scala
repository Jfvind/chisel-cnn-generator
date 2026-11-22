package shell

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.io.Source
import scala.util.Using

//==================================================================================================
// MlpShellTest.scala - #15: the MNIST MLP inside the AXI shell, bit exact against the integer reference.
//
// One export in generated/quantized/ (make reference) is used for everything: network.json and the
// integer weights build the hardware, test_inputs.csv (64 images x 784 pixels) and expected_outputs.csv
// (64 x 10 scores) are the vectors. python/tests recompute the vectors in Python as an independent check.
//==================================================================================================

class MlpShellTest extends AnyFlatSpec with ChiselScalatestTester {
    behavior of "AccelShell with the MLP core"

    val vectorDir = "generated/quantized" // realtive to generator/, where sbt runs

    // one row per line, comma-seperated integers
    def readCsv(name: String): Seq[Seq[Long]] = Using.resource(Source.fromFile(s"$vectorDir/$name")) { src =>
        src.getLines().map(_.trim).filter(_.nonEmpty).map(_.split(",").map(_.trim.toLong).toSeq).toSeq
    }

    val inputs = readCsv("test_inputs.csv") // 64 rows x 784 pixels
    val expectedOutputs = readCsv("expected_outputs.csv") // 64 rows x 10 scores

    // The hardware, built from the same export as the test vectors.
    val cfg = AccelConfig.resolve(AccelConfig(
        core            = "mlp",
        streamW         = 32,
        quantizedDir    = vectorDir))

    val net = AccelConfig.network(cfg)

    // --- Bit-exact through the shell ---
    // Default 2 images (one at full speed, one with backpressure); all 64 with:  sbt "-Dall=1" "testOnly shell.MlpShellTest"
    val nImages = if (sys.props.get("all").contains("1")) inputs.length else 2
    val seed    = sys.props.get("seed").map(_.toLong).getOrElse(System.nanoTime())

    // CYCLES at full speed (see the MlpCore header): K = 4 inputs per clock (4 pixels per 32-bit word,
    // two lanes per DSP), sum over layers of (ceil(inputs / K) + 6) + outputs - 1 = (196 + 6) + (32 + 6) + 9 = 249
    val K = 4
    val fullSpeedCycles = net.layers.map(l => (l.inFeatures + K - 1) / K + 6).sum + net.layers.last.outFeatures - 1

    // s_axis words: cfg.pixelsPerBeat pixels per word, pixel 0 in the lowest bits.
    // 784 pixels --> 196 words. Image 0, pixels 200-203 = 0, 0, 84, 185 --> word 50 = 0xB9540000
    val pixelBits = net.quantization.activationBits    // 8
    def pack(pixels: Seq[Long]): Seq[BigInt] =
        pixels.grouped(cfg.pixelsPerBeat).map(group => group.zipWithIndex.map { case (p, j) => BigInt(p) << (pixelBits * j) }.sum).toSeq 

    // m_axis beats are 32-bit two's complement: 0xFFFFF9C3 means -1597
    def toSigned(raw: BigInt): Long = (if (raw >= (BigInt(1) << 31)) raw - (BigInt(1) << 32) else raw).toLong

    // index of the largest score; indexOf finds the first, so the lowest index wins a tie
    def argmax(xs: Seq[Long]): Int = xs.indexOf(xs.max)

    it should s"match Python bit-exactly through the AXI shell for $nImages images (seed $seed)" in {
        val t0 = System.nanoTime()
        test(new AccelShellInner(cfg)) { dut =>
            println(f"MlpShellTest: setup (elaboration + simulator) ${(System.nanoTime() - t0) / 1e9}%.1f s")
            val bus = new AxiDriver(dut, new scala.util.Random(seed))
            bus.idle()
            for (k <- 0 until nImages) {
                val stall = k % 2 == 1                                   // every 2nd image with random backpressure
                bus.write(RegMap.CTRL, 1)                                // start: clears done, CYCLES and OVF_COUNT
                val (raw, lasts, firstIn, lastOut) = bus.stream(pack(inputs(k)), stall, outBeats = cfg.nOut)
                val got  = raw.map(toSigned)
                val want = expectedOutputs(k)
                assert(got == want, s"image $k: scores differ\n  Python:   ${want.mkString(", ")}\n  hardware: ${got.mkString(", ")}")
                assert(lasts == Seq(cfg.nOut - 1), s"image $k: TLAST must be on the last beat only, got it on beats $lasts")
                assert(bus.read(RegMap.RESULT) == argmax(want), s"image $k: RESULT is not the index of the largest score")
                val status = bus.read(RegMap.STATUS)                     // bit1 = done, bit2 = packet error
                assert((status & 2) == 2, s"image $k: STATUS.done not set")
                assert((status & 4) == 0, s"image $k: STATUS packet error: the core got too few or too many stream words")
                if (net.quantization.shiftPolicy == "worst") assert(bus.read(RegMap.OVF_COUNT) == 0, s"image $k: OVF_COUNT must be 0 with shift policy worst")
                val cycles = bus.read(RegMap.CYCLES)
                assert(cycles == lastOut - firstIn, s"image $k: CYCLES register $cycles, testbench measured ${lastOut - firstIn}")
                if (!stall) assert(cycles == fullSpeedCycles, s"image $k: CYCLES $cycles at full speed, expected $fullSpeedCycles")
            }
        }
        println(f"MlpShellTest: $nImages images in ${(System.nanoTime() - t0) / 1e9}%.1f s")
    }

}