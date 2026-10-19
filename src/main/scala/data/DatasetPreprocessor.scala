
package data

import java.io._
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, StandardCopyOption}
import java.util.Map

import net.razorvine.pickle.Unpickler

object DatasetPreprocessor {

  private val RawDir = "data/raw"
  private val OutputDir = "data/processed"

  private def ensureParent(path: String): Unit = {
    val parent = new File(path).getAbsoluteFile.getParentFile
    if (parent != null) parent.mkdirs()
  }

  private def writeAtomically(
      outputPath: String
  )(writeContent: PrintWriter => Unit): Unit = {
    val output = new File(outputPath).getAbsoluteFile
    ensureParent(outputPath)

    if (output.exists() && output.length() > 0) {
      println(s"Skipping existing file: $outputPath")
      return
    }

    val temporary = File.createTempFile(
      "dataset-",
      ".tmp",
      output.getParentFile
    )

    val writer = new PrintWriter(
      new BufferedWriter(
        new OutputStreamWriter(
          new FileOutputStream(temporary),
          StandardCharsets.UTF_8
        )
      )
    )

    try {
      writeContent(writer)
      writer.flush()

      if (writer.checkError()) {
        throw new IOException(s"Failed writing $outputPath")
      }

      writer.close()

      try {
        Files.move(
          temporary.toPath,
          output.toPath,
          StandardCopyOption.ATOMIC_MOVE
        )
      } catch {
        case _: java.nio.file.AtomicMoveNotSupportedException =>
          Files.move(temporary.toPath, output.toPath)
      }

      println(s"Created: $outputPath")
    } finally {
      writer.close()
      if (temporary.exists()) temporary.delete()
    }
  }

  def saveMNIST(
      imagePath: String,
      labelPath: String,
      outputPath: String
  ): Unit = {
    writeAtomically(outputPath) { out =>
      val images = new DataInputStream(
        new BufferedInputStream(new FileInputStream(imagePath))
      )
      val labels = new DataInputStream(
        new BufferedInputStream(new FileInputStream(labelPath))
      )

      try {
        require(images.readInt() == 2051, "Invalid MNIST image file")
        require(labels.readInt() == 2049, "Invalid MNIST label file")

        val count = images.readInt()
        val rows = images.readInt()
        val cols = images.readInt()
        val labelCount = labels.readInt()
        val pixelCount = rows * cols

        require(count == labelCount, "MNIST image-label count mismatch")

        out.println(
          "sample_id,label," +
            (0 until pixelCount).map(i => s"pixel_$i").mkString(",")
        )

        val pixels = new Array[Byte](pixelCount)

        for (i <- 0 until count) {
          val label = labels.readUnsignedByte()
          images.readFully(pixels)

          val features = pixels.iterator
            .map(p => (p & 0xff).toDouble / 255.0)
            .mkString(",")

          out.println(s"mnist_$i,$label,$features")
        }

        println(s"Processed $count MNIST samples")
      } finally {
        images.close()
        labels.close()
      }
    }
  }

  private def getField(
      map: Map[Object, Object],
      name: String
  ): Object = {
    val direct = map.get(name.asInstanceOf[Object])

    if (direct != null) {
      direct
    } else {
      val byteKey = name.getBytes(StandardCharsets.UTF_8)
      val byteValue = map.get(byteKey.asInstanceOf[Object])

      if (byteValue != null) {
        byteValue
      } else {
        throw new IllegalArgumentException(
          s"Missing CIFAR field: $name"
        )
      }
    }
  }

  private def flattenBytes(value: Object): Array[Byte] = {
    value match {
      case bytes: Array[Byte] =>
        bytes

      case rows: Array[Array[Byte]] =>
        rows.iterator.flatMap(_.iterator).toArray

      case list: java.util.List[_] =>
        list.toArray.iterator.flatMap {
          case b: java.lang.Number =>
            Iterator(b.byteValue())

          case row: Array[Byte] =>
            row.iterator

          case other =>
            throw new IllegalArgumentException(
              s"Unsupported CIFAR list element: ${other.getClass.getName}"
            )
        }.toArray

      case other =>
        throw new IllegalArgumentException(
          "CIFAR pixel data could not be decoded as bytes. " +
            "The pickle may contain a NumPy ndarray requiring a " +
            "compatible NumPy unpickling constructor. Found: " +
            other.getClass.getName
        )
    }
  }

  def saveCIFAR(
      inputPath: String,
      outputPath: String,
      batchName: String
  ): Unit = {
    writeAtomically(outputPath) { out =>
      val unpickler = new Unpickler()
      val input = new FileInputStream(inputPath)

      try {
        val obj = unpickler.load(input)

        val map = obj match {
          case m: java.util.Map[_, _] =>
            m.asInstanceOf[Map[Object, Object]]

          case other =>
            throw new IllegalArgumentException(
              s"Unexpected CIFAR pickle root: ${other.getClass.getName}"
            )
        }

        val labelsObject =
          try {
            getField(map, "labels")
          } catch {
            case _: IllegalArgumentException =>
              getField(map, "fine_labels")
          }

        val labels = labelsObject match {
          case list: java.util.List[_] =>
            list

          case array: Array[_] =>
            java.util.Arrays.asList(array: _*)

          case other =>
            throw new IllegalArgumentException(
              s"Unsupported CIFAR labels: ${other.getClass.getName}"
            )
        }

        val data = flattenBytes(getField(map, "data"))
        val pixelsPerImage = 3072
        val count = labels.size()

        require(
          data.length == count * pixelsPerImage,
          s"Expected ${count * pixelsPerImage} pixel bytes, got ${data.length}"
        )

        out.println(
          "sample_id,label," +
            (0 until pixelsPerImage).map(i => s"pixel_$i").mkString(",")
        )

        for (i <- 0 until count) {
          val label = labels.get(i).toString.toInt
          val start = i * pixelsPerImage

          val features = data
            .slice(start, start + pixelsPerImage)
            .iterator
            .map(b => (b & 0xff).toDouble / 255.0)
            .mkString(",")

          out.println(s"${batchName}_$i,$label,$features")
        }

        println(s"Processed $count CIFAR-10 samples from $batchName")
      } finally {
        input.close()
        unpickler.close()
      }
    }
  }

  def main(args: Array[String]): Unit = {
    val mnistRoot = s"$RawDir/mnist"
    val cifarRoot = s"$RawDir/cifar-10-batches-py"

    saveMNIST(
      s"$mnistRoot/train-images.idx3-ubyte",
      s"$mnistRoot/train-labels.idx1-ubyte",
      s"$OutputDir/mnist/train.csv"
    )

    saveMNIST(
      s"$mnistRoot/t10k-images.idx3-ubyte",
      s"$mnistRoot/t10k-labels.idx1-ubyte",
      s"$OutputDir/mnist/test.csv"
    )

    for (i <- 1 to 5) {
      saveCIFAR(
        s"$cifarRoot/data_batch_$i",
        s"$OutputDir/cifar10/train_batch_$i.csv",
        s"cifar_train_$i"
      )
    }

    saveCIFAR(
      s"$cifarRoot/test_batch",
      s"$OutputDir/cifar10/test.csv",
      "cifar_test"
    )

    println("Dataset preprocessing finished.")
  }
}
