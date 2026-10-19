
package data

import org.apache.spark.sql.SparkSession

object DatasetLoader {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("TrainingDataPoisoning")
      .master("local[*]")
      .getOrCreate()

    try {
      val sample = spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .csv("data/processed/cifar10/train_batch_1.csv")
        .limit(100)

      println(s"Sample rows: ${sample.count()}")
      println(s"Total columns: ${sample.columns.length}")

      sample.select("sample_id", "label").show(5, truncate = false)

    } finally {
      spark.stop()
    }
  }
}
