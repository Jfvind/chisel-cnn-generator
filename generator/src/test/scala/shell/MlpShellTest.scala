package shell

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.io.Source
import scala.util.Using

//==================================================================================================
// MlpShellTest.scala - #15: the MINST MLP inside the AXI shell, bit exact against Python reference.
//
// Python's test vectors in generated/quantized/: test_inputs.csv (64 images x 784 pixels),
// expected_outputs.csv (64 x 10 scores) and network.json (the widths and shift policy they
// were made with). The hardware is built from the samme settings, so it must match exactly.
//==================================================================================================

class MlpShellTest extends AnyFlatSpec with ChiselScalatestTester {
    behavior of "AccelShell with the MLP core"

    val vectorDir = "generated/quantized" // realtive to generator/, where sbt runs

    // network.json is flat, so a pattern seach is enough: "key": value or "key": "value"
    val json = Using.resource(Source.fromFile(s"$vectorDir/network.json"))(_.mkString)
    def jsonValue(key: String): String = 
        ("\"" + key + "\"\\s*:\\s*\"?(\\w+)").r.findFirstMatchIn(json)
            .getOrElse(throw new IllegalArgumentException(s"$key missing in network.json")).group(1)

    // one row per line, comma-seperated integers
    def readCsv(name: String): Seq[Seq[Long]] = Using.resource(Source.fromFile(s"$vectorDir/$name")) { src =>
        src.getLines().map(_.trim).filter(_.nonEmpty).map(_.split(",").map(_.trim.toLong).toSeq).toSeq
    }

    val inputs = readCsv("test_inputs.csv") // 64 rows x 784 pixels
    val expectedOutputs = readCsv("expected_outputs.csv") // 64 rows x 10 scores

    // The hardware, built from the float weights with the same settings as the export
    val cfg = AccelConfig.resolve(AccelConfig(
        core            = "mlp",
        streamW         = 32,
        weightBiasDir   = "../python/models/csv",
        weightBits      = jsonValue("weight_bits").toInt,
        actBits         = jsonValue("activation_bits").toInt,
        shiftPolicy     = jsonValue("shift_policy")))
    
    val net = AccelConfig.network(cfg)

    // Test if the network's weights & biases are the same as the Python reference model.
    it should "use the same integer weights as Python's export" in {
        assert(net.layers(0).weights.map(_.map(_.toLong)) == readCsv("0_weight.csv"), "the weights in layer 1 are not the same - rerun make reference")
        assert(net.layers(0).bias.map(_.toLong) == readCsv("0_bias.csv").flatten, "the biases in layer 1 are not the same - rerun make reference")
        assert(net.layers(1).weights.map(_.map(_.toLong)) == readCsv("2_weight.csv"), "the weights in layer 2 are not the same - rerun make reference")
        assert(net.layers(1).bias.map(_.toLong) == readCsv("2_bias.csv").flatten, "the biases in layer 2 are not the same - rerun make reference")
    }

    // --- Bit-exact through the shell ---
    // Default 8 images; all 64 with:  sbt "-Dall=1" "testOnly shell.MlpShellTest"
    val nImages = if (sys.props.get("all").contains("1")) inputs.length else 8
    val seed    = sys.props.get("seed").map(_.toLong).getOrElse(System.nanoTime())

    // CYCLES at full speed (see the MlpCore header): sum over layers of (inputs + 4) + outputs - 1 = 929 for MNIST
    val fullSpeedCycles = net.layers.map(_.inFeatures + 4).sum + net.layers.last.outFeatures - 1

    // m_axis beats are 32-bit two's complement: 0xFFFFF9C3 means -1597
    def toSigned(raw: BigInt): Long = (if (raw >= (BigInt(1) << 31)) raw - (BigInt(1) << 32) else raw).toLong

    // index of the largest score; indexOf finds the first, so the lowest index wins a tie
    def argmax(xs: Seq[Long]): Int = xs.indexOf(xs.max)

    it should s"match Python bit-exactly through the AXI shell for $nImages images (seed $seed)" in {
        val t0 = System.nanoTime()
        test(new AccelShellInner(cfg)) { dut =>
            val bus = new AxiDriver(dut, new scala.util.Random(seed))
            bus.idle()
            for (k <- 0 until nImages) {
                val stall = k % 4 == 3                                   // every 4th image with random backpressure
                bus.write(RegMap.CTRL, 1)                                // start: clears done, CYCLES and OVF_COUNT
                val (raw, lasts, firstIn, lastOut) = bus.stream(inputs(k).map(BigInt(_)), stall, outBeats = cfg.nOut)
                val got  = raw.map(toSigned)
                val want = expectedOutputs(k)
                assert(got == want, s"image $k: scores differ\n  Python:   ${want.mkString(", ")}\n  hardware: ${got.mkString(", ")}")
                assert(lasts == Seq(cfg.nOut - 1), s"image $k: TLAST must be on the last beat only, got it on beats $lasts")
                assert(bus.read(RegMap.RESULT) == argmax(want), s"image $k: RESULT is not the index of the largest score")
                assert((bus.read(RegMap.STATUS) & 2) == 2, s"image $k: STATUS.done not set")
                if (cfg.shiftPolicy == "worst") assert(bus.read(RegMap.OVF_COUNT) == 0, s"image $k: OVF_COUNT must be 0 with shift policy worst")
                val cycles = bus.read(RegMap.CYCLES)
                assert(cycles == lastOut - firstIn, s"image $k: CYCLES register $cycles, testbench measured ${lastOut - firstIn}")
                if (!stall) assert(cycles == fullSpeedCycles, s"image $k: CYCLES $cycles at full speed, expected $fullSpeedCycles")
            }
        }
        println(f"MlpShellTest: $nImages images in ${(System.nanoTime() - t0) / 1e9}%.1f s")
    }

}