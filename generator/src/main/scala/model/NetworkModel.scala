package model

/**
 * Parameters for a single fully connected layer.
 * weights is laid out as [inFeatures][outFeatures], matching the CSV export in train_model.py.
 */
case class LayerParams(
    index: Int,
    weights: Seq[Seq[Double]],
    bias: Seq[Double]
) {
    def inFeatures: Int  = weights.length
    def outFeatures: Int = bias.length
}

/**
 * The full network as an ordered sequence of layers.
 */
case class NetworkModel(layers: Seq[LayerParams]) {
    override def toString: String =
        layers.map(l => s"Layer ${l.index}: ${l.inFeatures} -> ${l.outFeatures}").mkString("\n")
}
