package service

import java.nio.file.{Files, Path}

import model.{LayerParams, NetworkModel, QuantizationConfig}

/*
Reads a quantized export (network.json plus <index>_weight.csv and <index>_bias.csv, written by
WeightExporter) back into a NetworkModel. The export is the one place a network is quantized: the
hardware generator (shell.AccelConfig.network) and the reference (reference.Evaluate) both read it
here, so the bitstream, the test vectors and the board test are built from exactly the same files.
It also lets either of them run on an export made earlier, without the float CSVs or the
quantization step that produced it.

It checks the same contract as load_network in python/scripts/integer_reference.py: supported
policies and widths, strictly increasing layer indices, parameter shapes and ranges, relu on the
hidden layers and identity on the last, and a conservative MAC bound where
activationMax * sum(|weights|) + |bias| must fit the accumulator. Input and activation widths must be
equal, since the Scala scheme ties them, and NetworkModel itself checks that the layer sizes chain.
A missing shift_policy is read as "worst", the policy every export had before the field existed.
*/

object QuantizedReader {
    private val requiredPolicies = Seq(
        "weight_signed" -> ujson.Bool(true), "activation_signed" -> ujson.Bool(false),
        "parameter_rounding" -> ujson.Str("nearest_ties_positive"), "shift_rounding" -> ujson.Str("truncate"),
        "activation_overflow" -> ujson.Str("saturate"), "accumulator_overflow" -> ujson.Str("reject"))

    def load(directory: Path): NetworkModel = {
        val json = ujson.read(Files.readString(directory.resolve("network.json")))
        require(integerField(json, "schema_version") == 1, "Unsupported schema_version")
        for ((key, expected) <- requiredPolicies)
            require(json.obj.get(key).contains(expected), s"Unsupported or missing $key: ${json.obj.get(key)}")
        require(integerField(json, "input_bits") == integerField(json, "activation_bits"),
            "input_bits must equal activation_bits")

        val quantization = QuantizationConfig(integerField(json, "weight_bits"), integerField(json, "activation_bits"),
            integerField(json, "bias_bits"), json.obj.get("shift_policy").map(_.str).getOrElse("worst"))

        val layerEntries = json("layers").arr.toSeq
        require(layerEntries.nonEmpty, "Expected a nonempty layer list")
        val layers = layerEntries.zipWithIndex.map { case (entry, position) =>
            val index        = integerField(entry, "index")
            val inputCount   = integerField(entry, "in_features")
            val outputCount  = integerField(entry, "out_features")
            val shift        = integerField(entry, "shift")
            val activation   = if (position == layerEntries.length - 1) "identity" else "relu"
            require(integerField(entry, "multiplier") == 1, "Only shift-only rescaling is supported")
            require(integerField(entry, "accumulator_bits") == quantization.accumulatorBits(inputCount), s"Layer $index: accumulator_bits does not match the widths")
            require(entry("activation").str == activation, s"Layer $index: expected activation $activation")
            require(shift >= 0 && shift < 63, s"Layer $index: invalid shift $shift")

            val weights = Csv.read(directory.resolve(s"${index}_weight.csv"))(_.toInt)
            val bias    = Csv.read(directory.resolve(s"${index}_bias.csv"))(_.toInt).flatten
            require(weights.length == inputCount && weights.forall(_.length == outputCount) && bias.length == outputCount,
                s"Layer $index: parameter shapes do not match $inputCount x $outputCount")
            require(weights.flatten.forall(weight => weight >= -quantization.weightScale && weight < quantization.weightScale),
                s"Layer $index: weight exceeds ${quantization.weightBits} bits")
            require(bias.forall(value => value >= -quantization.biasLimit && value < quantization.biasLimit),
                s"Layer $index: bias exceeds ${quantization.biasBits} bits")
            for (output <- 0 until outputCount) {
                val bound = quantization.activationMax.toLong * weights.map(row => math.abs(row(output).toLong)).sum + math.abs(bias(output).toLong)
                require(bound < quantization.accumulatorLimit(inputCount), s"Layer $index: MAC accumulator can overflow")
            }
            LayerParams(index, weights, bias, shift)
        }
        val indices = layers.map(_.index)
        require(indices == indices.distinct.sorted, "Layer indices must be strictly increasing")
        NetworkModel(layers, quantization)
    }

    private def integerField(json: ujson.Value, key: String): Int = json.objOpt.flatMap(_.get(key)).flatMap(_.numOpt) match {
        case Some(number) if number.isWhole => number.toInt
        case other => throw new IllegalArgumentException(s"Invalid or missing $key: $other")
    }
}
