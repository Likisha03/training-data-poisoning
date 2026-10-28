
package spectral

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import scala.util.Random
import scala.jdk.CollectionConverters._

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.storage.StorageLevel

object SpectralSignatures {

  private def dot(a: Array[Double], b: Array[Double]): Double =
    a.indices.map(i => a(i) * b(i)).sum

  private def normalize(a: Array[Double]): Array[Double] = {
    val n = math.sqrt(dot(a, a))
    if (n < 1e-15) Array.fill(a.length)(0.0) else a.map(_ / n)
  }

  private def mean(rows: Array[Array[Double]]): Array[Double] = {
    require(rows.nonEmpty, "Cannot calculate mean of empty data")
    val result = Array.fill(rows.head.length)(0.0)
    rows.foreach(r => r.indices.foreach(i => result(i) += r(i)))
    result.indices.foreach(i => result(i) /= rows.length)
    result
  }

  private def leadingDirection(
      rows: Array[Array[Double]], iterations: Int, seed: Long
  ): Array[Double] = {
    val random = new Random(seed)
    var v = normalize(Array.fill(rows.head.length)(random.nextDouble() - 0.5))
    var i = 0

    while (i < iterations) {
      val next = Array.fill(v.length)(0.0)
      rows.foreach { r =>
        val p = dot(r, v)
        v.indices.foreach(j => next(j) += r(j) * p)
      }
      val updated = normalize(next)
      if (dot(updated, updated) < 1e-15) return v
      v = updated
      i += 1
    }
    v
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("SpectralSignatures")
      .master("local[*]")
      .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
      .config("spark.sql.shuffle.partitions", "8")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    var data: DataFrame = null
    var scored: DataFrame = null

    try {
      val input = args.headOption.getOrElse("data/processed/poisoned/mnist/train.csv")
      val output = args.lift(1).getOrElse("results/spectral_results.csv")
      val quantile = args.lift(2).map(_.toDouble).getOrElse(0.95)
      val iterations = args.lift(3).map(_.toInt).getOrElse(50)
      val dataset = if (input.toLowerCase.contains("mnist")) "MNIST" else "CIFAR-10"

      require(quantile > 0 && quantile < 1, "Quantile must be between 0 and 1")
      require(iterations > 0, "Iterations must be positive")

      println(s"Input file: $input")
      println(s"Dataset: $dataset")

      val raw = spark.read.option("header", "true")
        .option("inferSchema", "true").csv(input)

      require(raw.columns.contains("sample_id") && raw.columns.contains("label"),
        "Dataset must contain sample_id and label")

      val features = raw.columns.filterNot(
        Set("sample_id", "label", "original_label", "is_poisoned")
      )
      require(features.nonEmpty, "No feature columns found")

      data = new VectorAssembler().setInputCols(features)
        .setOutputCol("features").setHandleInvalid("error").transform(raw)
        .select(
          col("sample_id").cast(StringType).as("sample_id"),
          col("label").cast(IntegerType).as("label"),
          col("features"),
          (if (raw.columns.contains("is_poisoned"))
            col("is_poisoned").cast(BooleanType)
          else lit(null).cast(BooleanType)).as("is_poisoned")
        ).persist(StorageLevel.MEMORY_AND_DISK)

      val count = data.count()
      require(count > 0, "Dataset is empty")
      require(data.filter(col("sample_id").isNull || col("label").isNull)
        .limit(1).count() == 0, "Null sample_id or label")
      require(data.select("sample_id").distinct().count() == count,
        "sample_id values must be unique within the input file")

      println(s"Samples: $count | Features: ${features.length}")

      val models = data.select("label", "features").rdd
        .map(r => (r.getInt(0), r.getAs[Vector](1).toArray))
        .groupByKey().collect().sortBy(_._1)
        .map { case (label, values) =>
          val rows = values.toArray
          val m = mean(rows)
          val centered = rows.map(r => r.indices.map(i => r(i) - m(i)).toArray)
          println(s"Prepared class $label: ${rows.length} samples")
          label -> (m, leadingDirection(centered, iterations, label.toLong + 12345L))
        }.toMap

      val broadcast = spark.sparkContext.broadcast(models)
      val schema = StructType(Seq(
        StructField("sample_id", StringType, false),
        StructField("label", IntegerType, false),
        StructField("suspicion_score", DoubleType, false)
      ))

      scored = spark.createDataFrame(
        data.select("sample_id", "label", "features").rdd.map { r =>
          val x = r.getAs[Vector](2).toArray
          val (m, v) = broadcast.value(r.getInt(1))
          val projection = dot(x.indices.map(i => x(i) - m(i)).toArray, v)
          Row(r.getString(0), r.getInt(1), projection * projection)
        }, schema
      ).persist(StorageLevel.MEMORY_AND_DISK)

      require(scored.count() == count, "Score count mismatch")

      val threshold = scored.stat.approxQuantile(
        "suspicion_score", Array(quantile), 0.001
      )(0)

      val results = scored.withColumn(
        "is_flagged", col("suspicion_score") >= lit(threshold)
      ).join(data.select("sample_id", "is_poisoned"), Seq("sample_id"), "left")
        .persist(StorageLevel.MEMORY_AND_DISK)

      try {
        val evaluation = results.filter(col("is_poisoned").isNotNull)

        if (evaluation.limit(1).count() > 0) {
          val m = evaluation.agg(
            sum(when(col("is_flagged") && col("is_poisoned"), 1L).otherwise(0L)).as("tp"),
            sum(when(col("is_flagged") && !col("is_poisoned"), 1L).otherwise(0L)).as("fp"),
            sum(when(!col("is_flagged") && col("is_poisoned"), 1L).otherwise(0L)).as("fn"),
            sum(when(!col("is_flagged") && !col("is_poisoned"), 1L).otherwise(0L)).as("tn")
          ).first()

          val tp = m.getAs[Long]("tp")
          val fp = m.getAs[Long]("fp")
          val fn = m.getAs[Long]("fn")
          val tn = m.getAs[Long]("tn")
          val total = tp + fp + fn + tn

          val precision = if (tp + fp == 0) 0.0 else tp.toDouble / (tp + fp)
          val recall = if (tp + fn == 0) 0.0 else tp.toDouble / (tp + fn)
          val specificity = if (tn + fp == 0) 0.0 else tn.toDouble / (tn + fp)
          val accuracy = if (total == 0) 0.0 else (tp + tn).toDouble / total
          val f1 = if (precision + recall == 0) 0.0
            else 2 * precision * recall / (precision + recall)
          val fpr = if (fp + tn == 0) 0.0 else fp.toDouble / (fp + tn)
          val fnr = if (fn + tp == 0) 0.0 else fn.toDouble / (fn + tp)

          val positives = evaluation.filter(col("is_poisoned") === true).count()
          val negatives = evaluation.filter(col("is_poisoned") === false).count()
          val auc = if (positives > 0 && negatives > 0) {
            new BinaryClassificationEvaluator()
              .setLabelCol("truth_label")
              .setRawPredictionCol("suspicion_score")
              .setMetricName("areaUnderROC")
              .evaluate(evaluation.withColumn(
                "truth_label", col("is_poisoned").cast(DoubleType)
              ))
          } else Double.NaN

          println("\n========== SPECTRAL SIGNATURES RESULTS ==========")
          println(s"Dataset: $dataset")
          println(s"Total samples: $count")
          println(s"TP: $tp | FP: $fp | FN: $fn | TN: $tn")
          println(f"Accuracy:     ${accuracy * 100}%.2f%%")
          println(f"Precision:    ${precision * 100}%.2f%%")
          println(f"Recall:       ${recall * 100}%.2f%%")
          println(f"F1-score:     ${f1 * 100}%.2f%%")
          println(f"Specificity:  ${specificity * 100}%.2f%%")
          println(f"False Positive Rate: ${fpr * 100}%.2f%%")
          println(f"False Negative Rate: ${fnr * 100}%.2f%%")
          if (auc.isNaN) println("ROC-AUC: Cannot calculate (only one class present)")
          else println(f"ROC-AUC:      $auc%.6f")
          println("=================================================")
        } else println("No is_poisoned labels available for evaluation.")
      } finally results.unpersist()

      val path = Paths.get(output)
      if (path.getParent != null) Files.createDirectories(path.getParent)

      val header = "dataset,sample_id,label,suspicion_score,is_flagged,is_poisoned"
      val currentIds = data.select("sample_id").collect().map(_.getString(0)).toSet

      val existing = if (Files.exists(path)) {
        Files.readAllLines(path, StandardCharsets.UTF_8).asScala
          .filterNot(_ == header)
          .filter { line =>
            val fields = line.split(",", 3)
            fields.length >= 2 &&
              (fields(0).replace("\"", "") != dataset ||
                !currentIds.contains(fields(1).replace("\"", "")))
          }.toSeq
      } else Seq.empty[String]

      val writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)
      try {
        writer.write(header + "\n")
        existing.foreach(line => writer.write(line + "\n"))

        val iterator = scored.withColumn(
          "is_flagged", col("suspicion_score") >= lit(threshold)
        ).join(data.select("sample_id", "is_poisoned"), Seq("sample_id"), "left")
          .select("sample_id", "label", "suspicion_score", "is_flagged", "is_poisoned")
          .toLocalIterator()

        while (iterator.hasNext) {
          val r = iterator.next()
          val truth = if (r.isNullAt(4)) "" else r.getBoolean(4).toString
          writer.write(
            s"$dataset,${r.getString(0)},${r.getInt(1)}," +
              s"${r.getDouble(2)},${r.getBoolean(3)},$truth\n"
          )
        }
      } finally writer.close()

      println(s"\nThreshold: $threshold")
      println(s"Quantile: $quantile | Iterations: $iterations")
      println(s"CSV updated: $output")

      println("\nUpdating Excel workbook...")
      val process = new ProcessBuilder(
        "python", "update_excel.py", input, dataset,
        threshold.toString, quantile.toString, iterations.toString
      ).inheritIO().start()

      require(process.waitFor() == 0,
        "Excel update failed. Check the Python error above.")
      println("Excel workbook updated successfully.")

    } finally {
      if (scored != null) scored.unpersist()
      if (data != null) data.unpersist()
      spark.stop()
    }
  }
}
