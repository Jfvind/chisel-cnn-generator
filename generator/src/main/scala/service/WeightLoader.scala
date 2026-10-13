package service

import java.io.File
import scala.io.Source
import scala.util.Using

import model.{LayerParams, NetworkModel}

/**
 * Loads weights and biases exported by python/train_model.py.
 * Expects files named <index>_weight.csv and <index>_bias.csv in the given directory.
 */
object WeightLoader {
    private val WeightFile = """(\d+)_weight\.csv""".r

    def load(dir: String): NetworkModel = {
        val folder = new File(dir)
        require(folder.isDirectory, s"Weight directory not found: ${folder.getAbsolutePath}")

        val indices = folder.listFiles().map(_.getName).collect { case WeightFile(i) => i.toInt }.sorted

        val layers = indices.toSeq.map { i =>
            val weights = readCsv(new File(folder, s"${i}_weight.csv"))
            val bias    = readCsv(new File(folder, s"${i}_bias.csv")).flatten
            require(weights.forall(_.length == bias.length),
                s"Layer $i: weight columns do not match bias length ${bias.length}")
            LayerParams(i, weights, bias)
        }

        layers.zip(layers.drop(1)).foreach { case (a, b) =>
            require(a.outFeatures == b.inFeatures,
                s"Layer ${a.index} outputs ${a.outFeatures} but layer ${b.index} expects ${b.inFeatures}")
        }

        NetworkModel(layers)
    }

    private def readCsv(file: File): Seq[Seq[Double]] = {
        require(file.isFile, s"Missing file: ${file.getAbsolutePath}")
        Using.resource(Source.fromFile(file)) { src =>
            src.getLines().map(_.trim).filter(_.nonEmpty).map(_.split(",").map(_.trim.toDouble).toSeq).toSeq
        }
    }
}
