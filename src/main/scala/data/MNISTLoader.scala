
package data

import java.io.DataInputStream
import java.io.BufferedInputStream
import java.io.FileInputStream
import org.apache.spark.sql.SparkSession

object MNISTLoader {

  def readImages(path: String): Array[Array[Byte]] = {
    val in = new DataInputStream(
      new BufferedInputStream(new FileInputStream(path))
    )

    try {
      val magic = in.readInt()
      val count = in.readInt()
      val rows = in.readInt()
      val cols = in.readInt()

      require(magic == 2051, "Invalid MNIST image file")
      require(rows == 28 && cols == 28, "Expected 28x28 images")

      Array.fill(count) {
        val image = new Array[Byte](rows * cols)
        in.readFully(image)
        image
      }
    } finally {
      in.close()
    }
  }

  def readLabels(path: String): Array[Byte] = {
    val in = new DataInputStream(
      new BufferedInputStream(new FileInputStream(path))
    )

    try {
      val magic = in.readInt()
      val count = in.readInt()

      require(magic == 2049, "Invalid MNIST label file")

      val labels = new Array[Byte](count)
      in.readFully(labels)
      labels
    } finally {
      in.close()
    }
  }

  def main(args: Array[String]): Unit = {
    val basePath = "data/raw/mnist"

    val images = readImages(
      s"$basePath/train-images.idx3-ubyte"
    )

    val labels = readLabels(
      s"$basePath/train-labels.idx1-ubyte"
    )

    require(images.length == labels.length,
      "Image and label counts do not match")

    require(labels.forall(label => label >= 0 && label <= 9),
      "Invalid digit label found")

    val spark = SparkSession.builder()
      .appName("MNISTLoader")
      .master("local[*]")
      .getOrCreate()

    try {
      val sample = images.take(10).zip(labels.take(10))
        .map { case (image, label) =>
          (image.map(_ & 0xff).toSeq, label.toInt)
        }

      val df = spark.createDataFrame(sample)
        .toDF("pixels", "label")

      df.show(10, truncate = false)

      println(s"Training images loaded: ${images.length}")
      println(s"Training labels loaded: ${labels.length}")
      println(s"Image dimensions: 28 x 28")
      println("MNIST loading completed successfully!")
    } finally {
      spark.stop()
    }
  }
}
