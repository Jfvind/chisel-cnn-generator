package service

import java.io.File
import scala.io.Source
import scala.util.Using

import model.{LayerParams, NetworkModel, QuantizationConfig}

/**
 * Loads float weights and biases exported by python/scripts/train_model.py and quantizes them.
 * Expects files named <index>_weight.csv and <index>_bias.csv in the given directory.
 */
object WeightLoader {
    private val WeightFile = """(\d+)_weight\.csv""".r

    def load(dir: String, config: QuantizationConfig = QuantizationConfig()): NetworkModel = {
        val folder = new File(dir)
        require(folder.isDirectory, s"Weight directory not found: ${folder.getAbsolutePath}")

        val indices = folder.listFiles().map(_.getName).collect { case WeightFile(i) => i.toInt }.sorted
        require(indices.nonEmpty, s"No weight files found in $dir")

        // Scale of activations: input integers represent pixels in 0.0..1.0.
        var actScale = config.activationMax.toDouble

        val layers = indices.toSeq.map { i =>
            val weights = readCsv(new File(folder, s"${i}_weight.csv"))
            val bias    = readCsv(new File(folder, s"${i}_bias.csv")).flatten
            require(weights.nonEmpty && bias.nonEmpty && weights.forall(_.length == bias.length),
                s"Layer $i: weight columns do not match bias length ${bias.length}")
            config.accumulatorBits(weights.length)

            val qWeights = weights.map(_.map(w => quantizeWeight(w, config.weightBits)))
            // The MAC adds bias to act * weight, so bias uses their product scale.
            val qBias = bias.map { b =>
                val scaled = b * actScale * config.weightScale
                require(scaled.isFinite, s"Layer $i: bias must be finite")
                val rounded = math.round(scaled)
                val limit = 1L << (config.biasBits - 1)
                require(rounded >= -limit && rounded < limit, s"Layer $i: bias does not fit ${config.biasBits} bits")
                rounded.toInt
            }
            val shift = activationShift(qWeights, qBias, config.activationBits)

            actScale = math.scalb(actScale * config.weightScale, -shift)
            LayerParams(i, qWeights, qBias, shift)
        }

        layers.zip(layers.drop(1)).foreach { case (a, b) =>
            require(a.outFeatures == b.inFeatures,
                s"Layer ${a.index} outputs ${a.outFeatures} but layer ${b.index} expects ${b.inFeatures}")
        }

        NetworkModel(layers, config)
    }

    def quantizeWeight(x: Double, bits: Int = 8): Int = {
        val scale = QuantizationConfig(weightBits = bits).weightScale
        require(x.isFinite && math.abs(x) <= 1.0, s"Weight $x is outside [-1, 1] and would be clamped")
        math.round(x * scale).toInt.min(scale - 1)
    }

    /**
     * Smallest shift so the worst-case positive sum fits the activation width.
     */
    def activationShift(weights: Seq[Seq[Int]], bias: Seq[Int], activationBits: Int = 8): Int = {
        val actMax = QuantizationConfig(activationBits = activationBits).activationMax
        val maxSum = bias.indices.map(j => BigInt(actMax) * weights.map(row => BigInt(row(j).max(0))).sum + bias(j)).max
        (0 until 63).find(s => (maxSum >> s) <= actMax).getOrElse(
            throw new IllegalArgumentException("Activation shift exceeds the reference arithmetic width"))
    }

    private def readCsv(file: File): Seq[Seq[Double]] = {
        require(file.isFile, s"Missing file: ${file.getAbsolutePath}")
        Using.resource(Source.fromFile(file)) { src =>
            src.getLines().map(_.trim).filter(_.nonEmpty).map(_.split(",").map(_.trim.toDouble).toSeq).toSeq
        }
    }
}
