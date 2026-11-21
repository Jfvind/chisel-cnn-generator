package service

import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters._

/*
Reads and writes comma-separated files of numbers, one row per line. The float CSVs from
train_model.py (read by WeightLoader), the integer CSVs of the export (written by WeightExporter, read
by QuantizedReader) and the test vectors (written by reference.Evaluate) all share this
layout, and one reader and one writer keep them byte for byte alike. That matters because
parameter_sha256 hashes the exported CSVs and Python compares the vector files byte for byte. Values
are trimmed before parsing, so "1, 2" and "1,2" read the same, a missing file fails with its full
path, and every written line, the last one included, ends with a newline.
*/
object Csv {
    def read[T](file: Path)(parse: String => T): Seq[Seq[T]] = {
        require(Files.isRegularFile(file), s"Missing file: ${file.toAbsolutePath}")
        Files.readAllLines(file).asScala.toSeq.map(_.trim).filter(_.nonEmpty).map(_.split(",").toSeq.map(value => parse(value.trim)))
    }

    def write(file: Path, rows: Seq[Seq[Any]]): Unit =
        Files.writeString(file, rows.map(_.mkString(",")).mkString("", "\n", "\n"))
}
