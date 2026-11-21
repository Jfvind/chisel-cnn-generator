package reference

import java.io.{BufferedInputStream, DataInputStream, FileInputStream, InputStream}
import java.nio.file.{Files, Path}
import java.util.zip.GZIPInputStream

/*
Reads the MNIST test set from the IDX files torchvision downloads into python/data/MNIST/raw/, raw
or gzipped, so the Scala reference can measure integer accuracy and pick test vectors without going
through Python. An IDX file starts with two zero bytes, a type byte (0x08 for unsigned bytes) and the
number of dimensions, followed by one big-endian int32 per dimension and then the data. The images
come back in dataset order as arrays of 784 raw bytes; Java bytes are signed, so a pixel's value
0..255 is byte & 0xff.
*/

object Mnist {
    case class Split(images: Seq[Array[Byte]], labels: Seq[Int])

    def test(rawDirectory: Path): Split = {
        val images = readIdx(rawDirectory, "t10k-images-idx3-ubyte")
        val labels = readIdx(rawDirectory, "t10k-labels-idx1-ubyte")
        require(images.dimensions.length == 3 && labels.dimensions.length == 1 && images.dimensions.head == labels.dimensions.head,
            "MNIST image and label files do not match")
        Split(images.data.grouped(images.dimensions(1) * images.dimensions(2)).toSeq, labels.data.map(_ & 0xff).toSeq)
    }

    private case class IdxFile(dimensions: Seq[Int], data: Array[Byte])

    private def readIdx(directory: Path, name: String): IdxFile = {
        val rawFile  = directory.resolve(name)
        val gzipFile = directory.resolve(s"$name.gz")
        val stream: InputStream =
            if (Files.isRegularFile(rawFile)) new FileInputStream(rawFile.toFile)
            else if (Files.isRegularFile(gzipFile)) new GZIPInputStream(new FileInputStream(gzipFile.toFile))
            else throw new IllegalArgumentException(s"Missing $rawFile (or .gz)")
        val input = new DataInputStream(new BufferedInputStream(stream))
        try {
            val magic = input.readInt()
            require((magic >>> 8) == 0x08, s"$name: not an unsigned-byte IDX file (magic 0x${magic.toHexString})")
            val dimensions = Seq.fill(magic & 0xff)(input.readInt())
            val data = new Array[Byte](dimensions.product)
            input.readFully(data)
            IdxFile(dimensions, data)
        } finally input.close()
    }
}
