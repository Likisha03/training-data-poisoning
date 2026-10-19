
package data

import org.apache.spark.sql.SparkSession

object SparkTest {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("TrainingDataPoisoning")
      .master("local[*]")
      .getOrCreate()

    val data = Seq(
      ("sample1", 0),
      ("sample2", 1),
      ("sample3", 0)
    )

    val df = spark.createDataFrame(data).toDF("sample", "label")

    df.show()

    println("Spark is working successfully!")

    spark.stop()
  }
}
