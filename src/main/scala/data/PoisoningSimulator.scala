
package data

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets

object PoisoningSimulator {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("PoisoningSimulator")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    val datasets = Seq(
      ("data/processed/mnist/train.csv",
       "mnist/train.csv"),
      ("data/processed/cifar10/train_batch_1.csv",
       "cifar10/train_batch_1.csv"),
      ("data/processed/cifar10/train_batch_2.csv",
       "cifar10/train_batch_2.csv"),
      ("data/processed/cifar10/train_batch_3.csv",
       "cifar10/train_batch_3.csv"),
      ("data/processed/cifar10/train_batch_4.csv",
       "cifar10/train_batch_4.csv"),
      ("data/processed/cifar10/train_batch_5.csv",
       "cifar10/train_batch_5.csv")
    )

    try {
      datasets.foreach { case (inputPath, outputPath) =>
        println(s"Processing: $inputPath")

        val input = spark.read
          .option("header", "true")
          .csv(inputPath)

        val poisoned = input
          .withColumn("original_label", col("label").cast("int"))
          .withColumn("is_poisoned", rand(42L) < 0.20)
          .withColumn(
            "label",
            when(
              col("is_poisoned"),
              (col("original_label") + 1) % 10
            ).otherwise(col("original_label"))
          )

        val file = Paths.get(
          "data/processed/poisoned", outputPath
        )

        Files.createDirectories(file.getParent)

        val writer = Files.newBufferedWriter(
          file, StandardCharsets.UTF_8
        )

        var total = 0
        var poisonedCount = 0

        try {
          writer.write(poisoned.columns.mkString(",") + "\n")

          val rows = poisoned.toLocalIterator()

          while (rows.hasNext) {
            val row = rows.next()

            writer.write(
              row.toSeq.map {
                case null => ""
                case value => value.toString
              }.mkString(",") + "\n"
            )

            total += 1
            if (row.getAs[Boolean]("is_poisoned")) {
              poisonedCount += 1
            }
          }
        } finally {
          writer.close()
        }

        println(s"Samples: $total")
        println(s"Poisoned: $poisonedCount")
        println(s"Saved: $file")
      }
    } finally {
      spark.stop()
    }
  }
}
