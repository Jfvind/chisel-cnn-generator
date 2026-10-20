package service

import java.nio.file.{Files, Paths}
import model.{NetworkModel, QuantizationConfig}

object WeightExporter {
    def write(network: NetworkModel, directory: String): Unit = {
        require(network.layers.nonEmpty, "Cannot export an empty network")
        val config = network.quantization
        val layers = network.layers.zipWithIndex.map { case (layer, position) =>
            val bits = config.accumulatorBits(layer.inFeatures)
            require(layer.bias.nonEmpty && layer.weights.forall(_.size == layer.outFeatures), "Invalid layer shape")
            require(layer.shift >= 0 && layer.shift < 63, "Invalid activation shift")
            require(layer.weights.flatten.forall(w => w >= -config.weightScale && w < config.weightScale), "Weight exceeds configured width")
            val biasLimit = 1L << (config.biasBits - 1)
            require(layer.bias.forall(b => b.toLong >= -biasLimit && b.toLong < biasLimit), "Bias exceeds configured width")
            for (j <- layer.bias.indices) {
                val bound = BigInt(config.activationMax) * layer.weights.map(row => BigInt(row(j)).abs).sum + BigInt(layer.bias(j)).abs
                require(bound < (BigInt(1) << (bits - 1)), s"Layer ${layer.index} can overflow its MAC width")
            }
            val activation = if (position == network.layers.size - 1) "identity" else "relu"
            s"""    {"index":${layer.index},"in_features":${layer.inFeatures},"out_features":${layer.outFeatures},"accumulator_bits":$bits,"multiplier":1,"shift":${layer.shift},"activation":"$activation"}"""
        }
        val metadata = s"""{
  "schema_version": 1,
  "input_bits": ${config.activationBits},
  "activation_bits": ${config.activationBits},
  "weight_bits": ${config.weightBits},
  "bias_bits": ${config.biasBits},
  "weight_signed": true,
  "activation_signed": false,
  "parameter_rounding": "nearest_ties_positive",
  "shift_rounding": "truncate",
  "activation_overflow": "wrap",
  "accumulator_overflow": "reject",
  "layers": [
${layers.mkString(",\n")}
  ]
}
"""
        val output = Paths.get(directory)
        Files.createDirectories(output)
        for (layer <- network.layers) {
            Files.writeString(output.resolve(s"${layer.index}_weight.csv"), layer.weights.map(_.mkString(",")).mkString("\n") + "\n")
            Files.writeString(output.resolve(s"${layer.index}_bias.csv"), layer.bias.mkString("\n") + "\n")
        }
        Files.writeString(output.resolve("network.json"), metadata)
        // A new parameter export invalidates previously generated reference vectors.
        Seq("layers.csv", "test_inputs.csv", "expected_outputs.csv", "test_vectors.json").foreach(name => Files.deleteIfExists(output.resolve(name)))
    }

    def main(args: Array[String]): Unit = {
        val allowed = Set("--input", "--output", "--weight-bits", "--activation-bits", "--bias-bits")
        require(args.length % 2 == 0 && args.grouped(2).forall(pair => allowed(pair(0))), "Expected --input/--output/--weight-bits/--activation-bits/--bias-bits value pairs")
        val options = args.grouped(2).map(pair => pair(0) -> pair(1)).toMap
        require(options.size * 2 == args.length, "Duplicate exporter option")
        val config = QuantizationConfig(
            options.getOrElse("--weight-bits", "8").toInt,
            options.getOrElse("--activation-bits", "8").toInt,
            options.getOrElse("--bias-bits", "32").toInt)
        val network = WeightLoader.load(options.getOrElse("--input", "../python/models/csv"), config)
        write(network, options.getOrElse("--output", "generated/quantized"))
        println(s"$config\n$network")
    }
}
