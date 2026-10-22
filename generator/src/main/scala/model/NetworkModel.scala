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

/**
 * The full network as an ordered sequence of quantized layers.
 */
case class NetworkModel(layers: Seq[LayerParams], quantization: QuantizationConfig = QuantizationConfig()) {
    override def toString: String =
        layers.map(l => s"Layer ${l.index}: ${l.inFeatures} -> ${l.outFeatures}, shift ${l.shift}").mkString("\n")
}
