package model

/**
 * Quantized parameters for a single fully connected layer.
 * weights is laid out as [inFeatures][outFeatures], matching the CSV export in train_model.py.
 * Weight/activation/bias widths belong to NetworkModel.quantization.
 * Bias uses the act * weight scale; shift rescales hidden MAC sums.
 */
case class LayerParams(
    index: Int,
    weights: Seq[Seq[Int]],
    bias: Seq[Int],
    shift: Int
) {
    def inFeatures: Int  = weights.length
    def outFeatures: Int = bias.length
}

case class QuantizationConfig(weightBits: Int = 8, activationBits: Int = 8, biasBits: Int = 32) {
    require(weightBits >= 2 && weightBits <= 16, "Weight width must be 2..16 bits")
    require(activationBits >= 1 && activationBits <= 16, "Activation width must be 1..16 bits")
    require(biasBits >= 2 && biasBits <= 32, "Bias width must be 2..32 bits")

    val weightScale: Int = 1 << (weightBits - 1)
    val activationMax: Int = (1 << activationBits) - 1

    def accumulatorBits(inputs: Int): Int = {
        require(inputs > 0, "Layers must have at least one input")
        val bits = weightBits + activationBits + chisel3.util.log2Ceil(inputs)
        require(bits <= 63, "Accumulator width must fit signed int64 reference arithmetic")
        bits
    }
}

/**
 * The full network as an ordered sequence of quantized layers.
 */
case class NetworkModel(layers: Seq[LayerParams], quantization: QuantizationConfig = QuantizationConfig()) {
    override def toString: String =
        layers.map(l => s"Layer ${l.index}: ${l.inFeatures} -> ${l.outFeatures}, shift ${l.shift}").mkString("\n")
}
