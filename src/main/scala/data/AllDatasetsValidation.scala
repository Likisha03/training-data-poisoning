
package data

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import java.io.PrintWriter
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets

object AllDatasetsValidation {

  case class Result(
      dataset: String,
      file: String,
      rows: Long,
      features: Int,
      invalidLabels: Long,
      invalidIds: Long,
      invalidPixels: Long,
      pixelMin: String,
      pixelMax: String,
      status: String
  )

  def validate(
      spark: SparkSession,
      dataset: String,
      file: String,
      expectedFeatures: Int
  ): Result = {

    if (!Files.exists(Paths.get(file))) {
      return Result(dataset, file, 0, 0, -1, -1, -1,
        "-", "-", "MISSING FILE")
    }

    try {
      val df = spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .csv(file)

      val columns = df.columns
      val pixels = columns.filter(_.startsWith("pixel_"))

      if (!columns.contains("sample_id") ||
          !columns.contains("label") ||
          pixels.length != expectedFeatures) {
        return Result(dataset, file, 0, pixels.length, -1, -1, -1,
          "-", "-", "INVALID STRUCTURE")
      }

      println(s"  Counting rows in $file ...")
      val rows = df.count()

      println("  Checking labels and IDs ...")
      val badLabels = df.filter(
        col("label").isNull ||
        col("label").cast("int").isNull ||
        col("label").cast("int") < 0 ||
        col("label").cast("int") > 9
      ).count()

      val badIds = df.filter(
        col("sample_id").isNull ||
        length(trim(col("sample_id"))) === 0
      ).count()

      // Validate pixel columns in groups of 128.
      // Each group is checked in one aggregate action.
      var badPixels = 0L
      var globalMin = Double.PositiveInfinity
      var globalMax = Double.NegativeInfinity

      pixels.grouped(128).foreach { group =>
        println(s"  Checking pixel columns ${group.head} to ${group.last} ...")

        val stats = group.flatMap { name =>
          Seq(
            min(col(name).cast("double")).as(s"${name}_min"),
            max(col(name).cast("double")).as(s"${name}_max"),
            sum(
              when(
                col(name).isNull ||
                col(name).cast("double").isNull ||
                isnan(col(name).cast("double")) ||
                col(name).cast("double") < 0.0 ||
                col(name).cast("double") > 1.0,
                1L
              ).otherwise(0L)
            ).as(s"${name}_bad")
          )
        }

        val row = df.agg(stats.head, stats.tail: _*).first()

        group.indices.foreach { i =>
          val offset = i * 3
          val minValue = row.getAs[java.lang.Double](offset)
          val maxValue = row.getAs[java.lang.Double](offset + 1)
          val badValue = row.getAs[java.lang.Long](offset + 2)

          if (minValue != null && !minValue.isNaN &&
              !minValue.isInfinite) {
            globalMin = math.min(globalMin, minValue.doubleValue())
          }

          if (maxValue != null && !maxValue.isNaN &&
              !maxValue.isInfinite) {
            globalMax = math.max(globalMax, maxValue.doubleValue())
          }

          if (badValue != null) {
            badPixels += badValue.longValue()
          }
        }
      }

      val status =
        if (rows == 0) "EMPTY FILE"
        else if (badLabels > 0 || badIds > 0 || badPixels > 0)
          "CHECK REQUIRED"
        else "PASS"

      Result(
        dataset, file, rows, pixels.length, badLabels, badIds,
        badPixels,
        if (globalMin.isInfinite) "-" else f"$globalMin%.4f",
        if (globalMax.isInfinite) "-" else f"$globalMax%.4f",
        status
      )

    } catch {
      case e: Exception =>
        println(s"  ERROR: ${e.getMessage}")
        Result(dataset, file, 0, 0, -1, -1, -1,
          "-", "-", "ERROR")
    }
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("AllDatasetsValidation")
      .master("local[2]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    try {
      val files = Seq(
        ("MNIST", "data/processed/mnist/train.csv", 784),
        ("MNIST", "data/processed/mnist/test.csv", 784),
        ("CIFAR-10", "data/processed/cifar10/train_batch_1.csv", 3072),
        ("CIFAR-10", "data/processed/cifar10/train_batch_2.csv", 3072),
        ("CIFAR-10", "data/processed/cifar10/train_batch_3.csv", 3072),
        ("CIFAR-10", "data/processed/cifar10/train_batch_4.csv", 3072),
        ("CIFAR-10", "data/processed/cifar10/train_batch_5.csv", 3072),
        ("CIFAR-10", "data/processed/cifar10/test.csv", 3072)
      )

      val results = files.map { case (dataset, file, features) =>
        println(s"\nValidating $dataset: $file")
        validate(spark, dataset, file, features)
      }

      println("\n========== FINAL SUMMARY ==========")
      println("DATASET | ROWS | FEATURES | BAD LABELS | BAD IDS | BAD PIXELS | MIN | MAX | STATUS")

      results.foreach { r =>
        println(
          s"${r.dataset} | ${r.rows} | ${r.features} | " +
          s"${r.invalidLabels} | ${r.invalidIds} | ${r.invalidPixels} | " +
          s"${r.pixelMin} | ${r.pixelMax} | ${r.status}"
        )
      }

      val report = Paths.get("data/processed/validation_report.csv")
      Files.createDirectories(report.getParent)

      val writer = new PrintWriter(
        Files.newBufferedWriter(report, StandardCharsets.UTF_8)
      )

      try {
        writer.println(
          "dataset,file,rows,features,invalid_labels,invalid_ids," +
          "invalid_pixels,pixel_min,pixel_max,status"
        )

        results.foreach { r =>
          val fields = Seq(
            r.dataset, r.file, r.rows.toString, r.features.toString,
            r.invalidLabels.toString, r.invalidIds.toString,
            r.invalidPixels.toString, r.pixelMin, r.pixelMax, r.status
          )
          writer.println(fields.map(v => "\"" + v + "\"").mkString(","))
        }
      } finally {
        writer.close()
      }

      println(s"\nReport saved: ${report.toAbsolutePath}")
      println(s"Files checked: ${results.size}")
      println(s"Passed: ${results.count(_.status == "PASS")}")
      println(s"Needs attention: ${results.count(_.status != "PASS")}")

    } finally {
      spark.stop()
    }
  }
}
