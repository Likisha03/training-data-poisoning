
package data

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object PreprocessingValidation {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("PreprocessingValidation")
      .master("local[2]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    try {
      val df = spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .csv("data/processed/cifar10/train_batch_1.csv")
        .limit(1000)

      val rowCount = df.count()
      val pixelColumns = df.columns.filter(_.startsWith("pixel_"))

      println(s"Rows checked: $rowCount")
      println(s"Pixel features: ${pixelColumns.length}")
      println(s"Label column exists: ${df.columns.contains("label")}")
      println(s"Sample ID column exists: ${df.columns.contains("sample_id")}")

      val invalidLabels = df.filter(
        col("label").cast("int").isNull ||
        col("label").cast("int") < 0 ||
        col("label").cast("int") > 9
      ).count()

      println(s"Invalid labels: $invalidLabels")

      val checkedPixels = Seq(
        "pixel_0",
        "pixel_1",
        "pixel_1535",
        "pixel_3071"
      )

      val pixelStats = df.agg(
        min(col("pixel_0").cast("double")).as("pixel_0_min"),
        max(col("pixel_0").cast("double")).as("pixel_0_max"),
        min(col("pixel_1").cast("double")).as("pixel_1_min"),
        max(col("pixel_1").cast("double")).as("pixel_1_max"),
        min(col("pixel_1535").cast("double")).as("pixel_1535_min"),
        max(col("pixel_1535").cast("double")).as("pixel_1535_max"),
        min(col("pixel_3071").cast("double")).as("pixel_3071_min"),
        max(col("pixel_3071").cast("double")).as("pixel_3071_max")
      )

      println("Pixel value ranges for representative features:")
      pixelStats.show(false)

      val missingPixels = df.filter(
        checkedPixels.map { name =>
          col(name).isNull || col(name).cast("double").isNull
        }.reduce(_ || _)
      ).count()

      println(s"Rows with missing/invalid checked pixels: $missingPixels")
      println("Validation test finished.")
    } finally {
      spark.stop()
    }
  }
}
