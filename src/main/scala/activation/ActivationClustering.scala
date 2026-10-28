
package activation

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters._

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{PCA, StandardScaler, VectorAssembler}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

object ActivationClustering {

  def main(args: Array[String]): Unit = {

    System.setProperty("dev.ludovic.netlib.blas.allowNative", "false")
    System.setProperty("dev.ludovic.netlib.lapack.allowNative", "false")
    System.setProperty("dev.ludovic.netlib.arpack.allowNative", "false")
    System.setProperty("com.github.fommil.netlib.BLAS",
      "com.github.fommil.netlib.F2jBLAS")
    System.setProperty("com.github.fommil.netlib.LAPACK",
      "com.github.fommil.netlib.F2jLAPACK")

    val spark = SparkSession.builder()
      .appName("ActivationClustering")
      .master("local[*]")
      .config("spark.sql.shuffle.partitions", "8")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    var data: DataFrame = null
    var scored: DataFrame = null

    try {
      val input = args.headOption.getOrElse(
        "data/processed/poisoned/mnist/train.csv")
      val output = args.lift(1).getOrElse(
        "results/activation/activation_results.csv")
      val k = args.lift(2).map(_.toInt).getOrElse(2)
      val pcaComponents = args.lift(3).map(_.toInt).getOrElse(20)
      val maxIterations = args.lift(4).map(_.toInt).getOrElse(10)

      require(k >= 2, "K-means cluster count must be at least 2")
      require(pcaComponents >= 1, "PCA components must be positive")
      require(maxIterations >= 1, "K-means iterations must be positive")

      val lowerInput = input.toLowerCase
      val dataset =
        if (lowerInput.contains("mnist")) "MNIST"
        else if (lowerInput.contains("cifar")) "CIFAR-10"
        else throw new IllegalArgumentException(
          "Input path must contain 'mnist' or 'cifar'.")

      println(s"Input file: $input")
      println(s"Dataset: $dataset")

      val raw = spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(input)

      require(
        raw.columns.contains("sample_id") &&
        raw.columns.contains("label") &&
        raw.columns.contains("is_poisoned"),
        "Input must contain sample_id, label, and is_poisoned.")

      val excluded =
        Set("sample_id", "label", "original_label", "is_poisoned")
      val featureColumns = raw.columns.filterNot(excluded)

      require(featureColumns.nonEmpty, "No feature columns found.")

      val assembled = new VectorAssembler()
        .setInputCols(featureColumns)
        .setOutputCol("raw_features")
        .setHandleInvalid("error")
        .transform(raw)

      data = assembled.select(
        col("sample_id").cast(StringType).as("sample_id"),
        col("label").cast(IntegerType).as("label"),
        col("raw_features"),
        col("is_poisoned").cast(BooleanType).as("is_poisoned")
      ).persist(StorageLevel.MEMORY_AND_DISK)

      val total = data.count()
      require(total > 0, "Input dataset is empty.")

      require(
        data.filter(
          col("sample_id").isNull ||
          col("label").isNull ||
          col("is_poisoned").isNull
        ).limit(1).count() == 0,
        "Null sample_id, label, or is_poisoned found.")

      require(
        data.select("sample_id").distinct().count() == total,
        "sample_id values must be unique within the input file.")

      println(s"Samples: $total | Features: ${featureColumns.length}")

      val outputRows = ArrayBuffer[Row]()

      val classLabels = data.select("label")
        .distinct()
        .collect()
        .map(_.getInt(0))
        .sorted

      classLabels.foreach { classLabel =>
        val classData = data
          .filter(col("label") === classLabel)
          .select("sample_id", "label", "raw_features", "is_poisoned")
          .persist(StorageLevel.MEMORY_AND_DISK)

        try {
          val classCount = classData.count()

          if (classCount >= k) {
            val actualComponents = math.min(
              pcaComponents,
              math.min(featureColumns.length, classCount - 1).toInt)

            val scaler = new StandardScaler()
              .setInputCol("raw_features")
              .setOutputCol("scaled_features")
              .setWithMean(false)
              .setWithStd(true)

            val scaled = scaler.fit(classData).transform(classData)

            val pca = new PCA()
              .setInputCol("scaled_features")
              .setOutputCol("pca_features")
              .setK(actualComponents)

            val reduced = pca.fit(scaled)
              .transform(scaled)
              .persist(StorageLevel.MEMORY_AND_DISK)

            try {
              val kmeans = new KMeans()
                .setK(k)
                .setSeed(42L)
                .setFeaturesCol("pca_features")
                .setPredictionCol("cluster")
                .setMaxIter(maxIterations)

              val model = kmeans.fit(reduced)
              val predictions = model.transform(reduced)

              val clusterCounts = predictions
                .groupBy("cluster")
                .count()
                .collect()
                .map(r => r.getInt(0) -> r.getLong(1))
                .toMap

              val suspiciousCluster = clusterCounts.minBy {
                case (clusterId, size) => (size, clusterId)
              }._1

              val centers = model.clusterCenters

              println(
                s"Class $classLabel: samples=$classCount, " +
                s"PCA=$actualComponents, clusters=$clusterCounts, " +
                s"suspiciousCluster=$suspiciousCluster")

              val iterator = predictions.select(
                "sample_id", "label", "cluster",
                "pca_features", "is_poisoned"
              ).toLocalIterator()

              while (iterator.hasNext) {
                val r = iterator.next()

                // Keep the existing ID; combined CSV IDs are already unique.
                val sampleId = r.getString(0)
                val label = r.getInt(1)
                val cluster = r.getInt(2)
                val features =
                  r.getAs[org.apache.spark.ml.linalg.Vector](3)
                val poisoned = r.getBoolean(4)

                val center = centers(cluster)
                var squaredDistance = 0.0
                var i = 0

                while (i < features.size) {
                  val difference = features(i) - center(i)
                  squaredDistance += difference * difference
                  i += 1
                }

                val distance = math.sqrt(squaredDistance)
                val distancePart = distance / (1.0 + distance)

                val suspicionScore =
                  (if (cluster == suspiciousCluster) 0.5 else 0.0) +
                  0.5 * distancePart

                val flagged = cluster == suspiciousCluster

                outputRows += Row(
                  dataset, sampleId, label, cluster,
                  suspicionScore, flagged, poisoned)
              }
            } finally {
              reduced.unpersist()
            }
          } else {
            println(s"Skipping class $classLabel: only $classCount samples.")
          }
        } finally {
          classData.unpersist()
        }
      }

      val schema = StructType(Seq(
        StructField("dataset", StringType, nullable = false),
        StructField("sample_id", StringType, nullable = false),
        StructField("label", IntegerType, nullable = false),
        StructField("cluster", IntegerType, nullable = false),
        StructField("suspicion_score", DoubleType, nullable = false),
        StructField("is_flagged", BooleanType, nullable = false),
        StructField("is_poisoned", BooleanType, nullable = false)
      ))

      scored = spark.createDataFrame(
        spark.sparkContext.parallelize(outputRows.toSeq), schema
      ).persist(StorageLevel.MEMORY_AND_DISK)

      val confusion = scored.agg(
        sum(when(col("is_flagged") && col("is_poisoned"), 1L)
          .otherwise(0L)).as("tp"),
        sum(when(col("is_flagged") && !col("is_poisoned"), 1L)
          .otherwise(0L)).as("fp"),
        sum(when(!col("is_flagged") && col("is_poisoned"), 1L)
          .otherwise(0L)).as("fn"),
        sum(when(!col("is_flagged") && !col("is_poisoned"), 1L)
          .otherwise(0L)).as("tn")
      ).first()

      def getCount(name: String): Long =
        Option(confusion.getAs[java.lang.Long](name))
          .map(_.longValue()).getOrElse(0L)

      val tp = getCount("tp")
      val fp = getCount("fp")
      val fn = getCount("fn")
      val tn = getCount("tn")
      val evaluated = tp + fp + fn + tn

      val precision =
        if (tp + fp == 0) 0.0 else tp.toDouble / (tp + fp)
      val recall =
        if (tp + fn == 0) 0.0 else tp.toDouble / (tp + fn)
      val accuracy =
        if (evaluated == 0) 0.0 else (tp + tn).toDouble / evaluated
      val specificity =
        if (tn + fp == 0) 0.0 else tn.toDouble / (tn + fp)
      val f1 =
        if (precision + recall == 0) 0.0
        else 2 * precision * recall / (precision + recall)
      val fpr =
        if (fp + tn == 0) 0.0 else fp.toDouble / (fp + tn)
      val fnr =
        if (fn + tp == 0) 0.0 else fn.toDouble / (fn + tp)

      val classCounts = scored.groupBy("is_poisoned")
        .count()
        .collect()
        .map(r => r.getAs[Boolean]("is_poisoned") -> r.getLong(1))
        .toMap

      val auc =
        if (classCounts.getOrElse(true, 0L) > 0 &&
            classCounts.getOrElse(false, 0L) > 0) {
          val scoreAndLabels = scored.select(
            "suspicion_score", "is_poisoned"
          ).rdd.map { r =>
            (r.getDouble(0), if (r.getBoolean(1)) 1.0 else 0.0)
          }
          new BinaryClassificationMetrics(scoreAndLabels).areaUnderROC()
        } else Double.NaN

      println("\n=== ACTIVATION CLUSTERING RESULTS ===")
      println(s"Dataset: $dataset")
      println(s"Processed samples: ${scored.count()}")
      println(s"TP=$tp | FP=$fp | FN=$fn | TN=$tn")
      println(f"Accuracy:    ${accuracy * 100}%.2f%%")
      println(f"Precision:   ${precision * 100}%.2f%%")
      println(f"Recall:      ${recall * 100}%.2f%%")
      println(f"F1-score:    ${f1 * 100}%.2f%%")
      println(f"Specificity: ${specificity * 100}%.2f%%")
      println(f"FPR:         ${fpr * 100}%.2f%%")
      println(f"FNR:         ${fnr * 100}%.2f%%")
      println(f"ROC-AUC:     $auc%.4f")

      val path = Paths.get(output)
      if (path.getParent != null) Files.createDirectories(path.getParent)

      val header =
        "dataset,sample_id,label,cluster,suspicion_score,is_flagged,is_poisoned"
      val currentIds = outputRows.map(_.getString(1)).toSet

      val existing =
        if (Files.exists(path)) {
          Files.readAllLines(path, StandardCharsets.UTF_8)
            .asScala
            .filterNot(_.trim == header)
            .filter { line =>
              val fields = line.split(",", -1)
              fields.length >= 2 && !(
                fields(0).replace("\"", "") == dataset &&
                currentIds.contains(fields(1).replace("\"", "")))
            }
            .map { line =>
              val fields = line.split(",", -1)
              if (fields.length == 6)
                fields.take(4).mkString(",") + ",0.0," +
                  fields.drop(4).mkString(",")
              else line
            }
            .toSeq
        } else Seq.empty[String]

      val writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)
      try {
        writer.write(header + "\n")
        existing.foreach(line => writer.write(line + "\n"))
        outputRows.foreach { r =>
          writer.write(
            s"${r.getString(0)},${r.getString(1)},${r.getInt(2)}," +
            s"${r.getInt(3)},${r.getDouble(4)},${r.getBoolean(5)}," +
            s"${r.getBoolean(6)}\n")
        }
      } finally {
        writer.close()
      }

      println(s"Activation Clustering CSV updated: $output")

      val updater = new ProcessBuilder("python", "update_excel.py")
        .inheritIO()
        .start()
      val exitCode = updater.waitFor()

      if (exitCode != 0)
        println(s"WARNING: Excel updater failed with exit code $exitCode")
      else
        println("Excel reports refreshed successfully.")

    } finally {
      if (scored != null) scored.unpersist()
      if (data != null) data.unpersist()
      spark.stop()
    }
  }
}
