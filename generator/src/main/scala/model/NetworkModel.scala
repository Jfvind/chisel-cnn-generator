package model

/**
 * Quantized parameters for a single fully connected layer.
 * weights is laid out as [inFeatures][outFeatures], matching the CSV export in train_model.py.
 * Real value = quantized value / 2^shift (shared by weights and bias in this layer).
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
 * The full network as an ordered sequence of layers, quantized to `bits`-bit signed integers.
 */
case class NetworkModel(layers: Seq[LayerParams], bits: Int) {
    override def toString: String =
        layers.map(l => s"Layer ${l.index}: ${l.inFeatures} -> ${l.outFeatures}, shift ${l.shift}").mkString("\n")
}
