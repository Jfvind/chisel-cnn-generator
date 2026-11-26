package mlp

import org.scalatest.flatspec.AnyFlatSpec

import model.{LayerParams, NetworkModel, QuantizationConfig}
import shell.AccelConfig

/*
Checks how Parallelism picks inputs per clock and the multiplier form for the MNIST shape, 784 -> 128 -> 10 at
8 bits on a 32-bit stream (4 pixels per word, so inputsPerClock 1, 2 or 4). The 128 lanes need 64 x K
multipliers packed and 128 x K single, which gives these settings against a DSP budget:

    packed: K = 1 -> 64,  K = 2 -> 128, K = 4 -> 256
    single: K = 1 -> 128, K = 2 -> 256, K = 4 -> 512

The last test resolves the committed export in generated/quantized/, as ShellGen does, and checks that the
choice lands in the config that the manifest and CFG_HASH are written from.
*/

class ParallelismTest extends AnyFlatSpec {
    behavior of "Parallelism"

    def mnistShape(bits: Int = 8): NetworkModel = NetworkModel(Seq(
        LayerParams(0, Seq.fill(784)(Seq.fill(128)(0)), Seq.fill(128)(0), 0),
        LayerParams(1, Seq.fill(128)(Seq.fill(10)(0)), Seq.fill(10)(0), 0)
    ), QuantizationConfig(bits, bits))

    def config(inputsPerClock: Int = 0, multiplier: String = "auto", dspBudget: Int = 0, pixelsPerBeat: Int = 4) =
        AccelConfig(core = "mlp", nIn = 784, nOut = 10, pixelsPerBeat = pixelsPerBeat,
                    inputsPerClock = inputsPerClock, multiplier = multiplier, dspBudget = dspBudget)

    val network = mnistShape()

    it should "keep packed with 4 inputs per clock when nothing is set" in {
        assert(Parallelism.choose(config(), network) == Parallelism.Choice(4, packed = true))
        assert(Parallelism.multipliers(128, 4, packed = true) == 256)
    }

    it should "stop the automatic inputs per clock at 2 for single multipliers" in {
        assert(Parallelism.choose(config(multiplier = "single"), network) == Parallelism.Choice(2, packed = false))
    }

    it should "use a fixed inputs per clock as given" in {
        assert(Parallelism.choose(config(inputsPerClock = 1), network) == Parallelism.Choice(1, packed = true))
        assert(Parallelism.choose(config(inputsPerClock = 4, multiplier = "single"), network) == Parallelism.Choice(4, packed = false))
    }

    it should "pick the fastest setting that fits a DSP budget, packed before single" in {
        assert(Parallelism.choose(config(dspBudget = 220), network) == Parallelism.Choice(2, packed = true))
        assert(Parallelism.choose(config(dspBudget = 128), network) == Parallelism.Choice(2, packed = true))
        assert(Parallelism.choose(config(dspBudget = 100), network) == Parallelism.Choice(1, packed = true))
        assert(Parallelism.choose(config(dspBudget = 512), network) == Parallelism.Choice(4, packed = true))
        assert(Parallelism.choose(config(dspBudget = 220, multiplier = "single"), network) == Parallelism.Choice(1, packed = false))
    }

    it should "not use packed multipliers above 8 bits" in {
        assert(Parallelism.choose(config(pixelsPerBeat = 2), mnistShape(12)) == Parallelism.Choice(2, packed = false))
        assertThrows[IllegalArgumentException](Parallelism.choose(config(multiplier = "packed", pixelsPerBeat = 2), mnistShape(12)))
    }

    it should "reject settings that cannot be built" in {
        val tooMany  = intercept[IllegalArgumentException](Parallelism.choose(config(inputsPerClock = 4, dspBudget = 220), network))
        assert(tooMany.getMessage.contains("256 multipliers"), tooMany.getMessage)
        val tooSmall = intercept[IllegalArgumentException](Parallelism.choose(config(dspBudget = 10), network))
        assert(tooSmall.getMessage.contains("64 multipliers"), tooSmall.getMessage)
        val notDivisor = intercept[IllegalArgumentException](Parallelism.choose(config(inputsPerClock = 3), network))
        assert(notDivisor.getMessage.contains("1, 2, 4"), notDivisor.getMessage)
        assertThrows[IllegalArgumentException](config(multiplier = "dsp"))
    }

    it should "write the choice into the resolved config, so the manifest and CFG_HASH name the hardware" in {
        val resolved = AccelConfig.resolve(AccelConfig(core = "mlp", quantizedDir = "generated/quantized"))
        assert(resolved.inputsPerClock == 4 && resolved.multiplier == "packed")
        assert(AccelConfig.resolve(resolved) == resolved, "resolving twice changes nothing")
        val budget = AccelConfig.resolve(AccelConfig(core = "mlp", quantizedDir = "generated/quantized", dspBudget = 220))
        assert(budget.inputsPerClock == 2 && budget.multiplier == "packed")
        assert(budget.cfgHash != resolved.cfgHash, "other hardware -> other CFG_HASH")
    }
}
