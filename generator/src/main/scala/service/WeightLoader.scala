package service

import java.io.File

import model.{LayerParams, NetworkModel, QuantizationConfig}

/**
 * Loads float weights and biases exported by python/scripts/train_model.py and quantizes them.
 * Expects files named <index>_weight.csv and <index>_bias.csv in the given directory.
 */
object WeightLoader {
    private val WeightFile = """(\d+)_weight\.csv""".r

    /** One trained layer as train_model.py wrote it: float weights [input][output] and bias. */
    case class FloatLayer(index: Int, weights: Seq[Seq[Double]], bias: Seq[Double])

    def readFloatLayers(dir: String): Seq[FloatLayer] = {
        val folder = new File(dir)
        require(folder.isDirectory, s"Weight directory not found: ${folder.getAbsolutePath}")
        val indices = folder.listFiles().map(_.getName).collect { case WeightFile(i) => i.toInt }.sorted
        require(indices.nonEmpty, s"No weight files found in $dir")
        indices.toSeq.map { i =>
            val weights = Csv.read(new File(folder, s"${i}_weight.csv").toPath)(_.toDouble)
            val bias    = Csv.read(new File(folder, s"${i}_bias.csv").toPath)(_.toDouble).flatten
            require(weights.nonEmpty && bias.nonEmpty && weights.forall(_.length == bias.length),
                s"Layer $i: weight columns do not match bias length ${bias.length}")
            FloatLayer(i, weights, bias)
        }
    }

    def load(dir: String, config: QuantizationConfig = QuantizationConfig()): NetworkModel = {
        // Scale of activations: input integers represent pixels in 0.0..1.0.
        var actScale = config.activationMax.toDouble

        val layers = readFloatLayers(dir).map { case FloatLayer(i, weights, bias) =>
            config.accumulatorBits(weights.length)

            val qWeights = weights.map(_.map(w => quantizeWeight(w, config)))
            // The MAC adds bias to act * weight, so bias uses their product scale.
            val qBias = bias.map { b =>
                val scaled = b * actScale * config.weightScale
                require(scaled.isFinite, s"Layer $i: bias must be finite")
                val rounded = math.round(scaled)
                require(rounded >= -config.biasLimit && rounded < config.biasLimit, s"Layer $i: bias does not fit ${config.biasBits} bits")
                rounded.toInt
            }
            val shift = activationShift(qWeights, qBias, config)

            actScale = math.scalb(actScale * config.weightScale, -shift)
            LayerParams(i, qWeights, qBias, shift)
        }

        NetworkModel(layers, config)
    }

    def quantizeWeight(x: Double, config: QuantizationConfig = QuantizationConfig()): Int = {
        val scale = config.weightScale
        require(x.isFinite && math.abs(x) <= 1.0, s"Weight $x is outside [-1, 1] and would be clamped")
        math.round(x * scale).toInt.min(scale - 1)
    }

    val TypicalSigmas = 3 // "typical" policy: how far above the mean sum a sum may reach before it clamps

    /**
     * Shift that brings a hidden layer's MAC sums into the unsigned activation range.
     *   "worst"   (default): smallest shift so the worst-case positive sum fits, so nothing ever clamps.
     *   "typical": smallest shift so a typical sum fits; larger sums clamp (OVF_COUNT counts them).
     *              Estimated from the weights only, inputs assumed uniform over 0..actMax:
     *              typical_j = actMax/2 * sum(w) + b + TypicalSigmas * actMax/sqrt(12) * sqrt(sum(w^2)).
     * The typical policy keeps 2-3 more bits of precision; which one is more accurate is measured
     * with `make reference SHIFT_POLICY=...`.
     */
    def activationShift(weights: Seq[Seq[Int]], bias: Seq[Int], config: QuantizationConfig = QuantizationConfig()): Int = {
        val actMax = config.activationMax
        val fits: Int => Boolean = config.shiftPolicy match {
            case "worst" =>
                val maxSum = bias.indices.map(j => BigInt(actMax) * weights.map(row => BigInt(row(j).max(0))).sum + bias(j)).max
                s => (maxSum >> s) <= actMax
            case "typical" =>
                val typical = bias.indices.map { j =>
                    val col = weights.map(_(j).toDouble)
                    actMax / 2.0 * col.sum + bias(j) + TypicalSigmas * actMax / math.sqrt(12) * math.sqrt(col.map(w => w * w).sum)
                }.max
                s => typical / math.pow(2, s) <= actMax
        }
        (0 until 63).find(fits).getOrElse(
            throw new IllegalArgumentException("Activation shift exceeds the reference arithmetic width"))
    }
}
