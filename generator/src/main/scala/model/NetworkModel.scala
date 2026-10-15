package model

/**
 * Quantized parameters for a single fully connected layer.
 * weights is laid out as [inFeatures][outFeatures], matching the CSV export in train_model.py.
 * weights are signed Bits-bit, bias is wider (it sits at the scale of act * weight),
 * and shift is what the Activation module shifts the MAC sum by to fit it in an unsigned Bits-bit act.
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

object NetworkModel {
    /** Bit width of weights (signed) and activations (unsigned). */
    val Bits = 8
}

/**
 * The full network as an ordered sequence of quantized layers.
 */
case class NetworkModel(layers: Seq[LayerParams]) {
    override def toString: String =
        layers.map(l => s"Layer ${l.index}: ${l.inFeatures} -> ${l.outFeatures}, shift ${l.shift}").mkString("\n")
}
