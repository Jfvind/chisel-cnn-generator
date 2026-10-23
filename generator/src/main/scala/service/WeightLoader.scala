package service

import java.io.File
import scala.io.Source
import scala.util.Using

import model.{LayerParams, NetworkModel}
import model.NetworkModel.Bits

/**
 * Loads weights and biases exported by python/train_model.py and quantizes them.
 * Expects files named <index>_weight.csv and <index>_bias.csv in the given directory.
 */
object WeightLoader {
    val WeightScale = 1 << (Bits - 1) // real weight in [-1, 1] -> signed Bits-bit
    val ActMax      = (1 << Bits) - 1 // largest unsigned Bits-bit activation

    private val WeightFile = """(\d+)_weight\.csv""".r

    def load(dir: String): NetworkModel = {
        val folder = new File(dir)
        require(folder.isDirectory, s"Weight directory not found: ${folder.getAbsolutePath}")

        val indices = folder.listFiles().map(_.getName).collect { case WeightFile(i) => i.toInt }.sorted

        // Scale of the activations going into the current layer: input pixels 0..255 represent 0.0..1.0
        var actScale = ActMax.toDouble

        val layers = indices.toSeq.map { i =>
            val weights = readCsv(new File(folder, s"${i}_weight.csv"))
            val bias    = readCsv(new File(folder, s"${i}_bias.csv")).flatten
            require(weights.forall(_.length == bias.length),
                s"Layer $i: weight columns do not match bias length ${bias.length}")

            val qWeights = weights.map(_.map(quantizeWeight))
            // The MAC adds bias to act * weight, so bias must sit at actScale * WeightScale
            val qBias    = bias.map(b => math.round(b * actScale * WeightScale).toInt)
            val shift    = activationShift(qWeights, qBias)

            actScale = actScale * WeightScale / (1 << shift)
            LayerParams(i, qWeights, qBias, shift)
        }

        layers.zip(layers.drop(1)).foreach { case (a, b) =>
            require(a.outFeatures == b.inFeatures,
                s"Layer ${a.index} outputs ${a.outFeatures} but layer ${b.index} expects ${b.inFeatures}")
        }

        NetworkModel(layers)
    }

    def quantizeWeight(x: Double): Int = {
        require(math.abs(x) <= 1.0, s"Weight $x is outside [-1, 1] and would be clamped")
        math.round(x * WeightScale).toInt.min(WeightScale - 1) // 1.0 -> 127, not 128
    }

    val TypicalSigmas = 3 // how far above the mean sum a "typical" sum may reach before it clamps

    /**
     * Smallest shift so a typical MAC sum fits in an unsigned Bits-bit activation; larger sums clamp.
     * Estimated from the weights only: inputs are assumed uniform over 0..ActMax, so per neuron j
     * typical_j = mean + TypicalSigmas * std = ActMax/2 * sum(w) + b + TypicalSigmas * ActMax/sqrt(12) * sqrt(sum(w^2)).
     */
    def activationShift(weights: Seq[Seq[Int]], bias: Seq[Int]): Int = {
        val typical = bias.indices.map { j =>
            val col = weights.map(_(j).toDouble)
            ActMax / 2.0 * col.sum + bias(j) + TypicalSigmas * ActMax / math.sqrt(12) * math.sqrt(col.map(w => w * w).sum)
        }.max
        Iterator.from(0).find(s => typical / (1L << s) <= ActMax).get
    }

    private def readCsv(file: File): Seq[Seq[Double]] = {
        require(file.isFile, s"Missing file: ${file.getAbsolutePath}")
        Using.resource(Source.fromFile(file)) { src =>
            src.getLines().map(_.trim).filter(_.nonEmpty).map(_.split(",").map(_.trim.toDouble).toSeq).toSeq
        }
    }
}
