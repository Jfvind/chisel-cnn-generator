package mlp

import mac.MacPair
import model.NetworkModel
import shell.AccelConfig

/*
How many multipliers the MLP core builds and how it uses them. MlpCore has one lane per neuron of the widest
layer, and every clock each lane multiplies inputsPerClock inputs by their weights, so a layer with n inputs
takes ceil(n / inputsPerClock) clocks. A multiplier takes one of two forms: "packed" lets two neighbouring lanes
share one multiplier per input (MacPair, possible when 3 x bits fits the 25-bit DSP48E1 input, so up to 8 bits),
"single" gives every lane its own (Mac). The core therefore builds

    ceil(lanes / 2) x inputsPerClock   multipliers when packed
    lanes x inputsPerClock             multipliers when single

The PYNQ-Z2 has 220 DSP48E1 slices; Vivado builds the multipliers beyond that from LUTs.

inputsPerClock has to divide the pixels per stream word, so a group of inputs never spans two words. Each of the
two settings is either fixed by the user (AccelConfig.inputsPerClock > 0, AccelConfig.multiplier "packed" or
"single") or chosen here. Without a DSP budget the choice is packed whenever the widths allow it, and the largest
inputsPerClock of at most 4 (packed) or 2 (single) that divides the pixels per word. With a budget it is the
fastest setting whose multipliers fit the budget, packed before single when both are equally fast. A fixed
setting that does not fit the budget is an error, so a build that cannot fit stops before Vivado runs.

    multiplier = auto, inputsPerClock = auto, dspBudget = 0     MNIST 784-128-10: packed, 4 per clock, 256 multipliers
    multiplier = auto, inputsPerClock = auto, dspBudget = 220   MNIST 784-128-10: packed, 2 per clock, 128 multipliers
*/

object Parallelism {
    case class Choice(inputsPerClock: Int, packed: Boolean) {
        def multiplier: String = if (packed) "packed" else "single"
    }

    val Multipliers = Set("auto", "packed", "single")

    def lanes(network: NetworkModel): Int = network.layers.map(_.outFeatures).max

    def multipliers(lanes: Int, inputsPerClock: Int, packed: Boolean): Int =
        (if (packed) (lanes + 1) / 2 else lanes) * inputsPerClock

    def choose(config: AccelConfig, network: NetworkModel): Choice = {
        val bits          = network.quantization.activationBits
        val laneCount     = lanes(network)
        val pixelsPerBeat = config.pixelsPerBeat
        val divisors      = (1 to pixelsPerBeat).filter(pixelsPerBeat % _ == 0)

        val packedOptions = config.multiplier match {
            case "auto"   => if (MacPair.fits(bits)) Seq(true, false) else Seq(false)
            case "packed" =>
                require(MacPair.fits(bits), s"multiplier packed needs at most 8-bit weights (3 x bits <= 25), got $bits bits")
                Seq(true)
            case "single" => Seq(false)
        }
        val inputOptions =
            if (config.inputsPerClock > 0) {
                require(divisors.contains(config.inputsPerClock),
                    s"inputsPerClock ${config.inputsPerClock} must divide the $pixelsPerBeat pixels per stream word " +
                    s"(${config.streamW}-bit words, $bits-bit pixels): one of ${divisors.mkString(", ")}")
                Seq(config.inputsPerClock)
            } else divisors

        if (config.dspBudget == 0) {
            val packed = packedOptions.head
            val inputs =
                if (config.inputsPerClock > 0) config.inputsPerClock
                else divisors.filter(_ <= (if (packed) 4 else 2)).max
            Choice(inputs, packed)
        } else {
            val candidates = for (packed <- packedOptions; inputs <- inputOptions) yield Choice(inputs, packed)
            val cost       = (choice: Choice) => multipliers(laneCount, choice.inputsPerClock, choice.packed)
            val fitting    = candidates.filter(cost(_) <= config.dspBudget)
            require(fitting.nonEmpty, {
                val cheapest = candidates.minBy(cost)
                s"No setting of $laneCount lanes fits a DSP budget of ${config.dspBudget}: the smallest, " +
                s"${cheapest.multiplier} with ${cheapest.inputsPerClock} input(s) per clock, needs ${cost(cheapest)} multipliers"
            })
            fitting.maxBy(choice => (choice.inputsPerClock, choice.packed))
        }
    }
}
