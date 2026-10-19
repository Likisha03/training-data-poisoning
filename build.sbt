
ThisBuild / scalaVersion := "2.13.18"
ThisBuild / version := "0.1.0"
ThisBuild / name := "training-data-poisoning"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "4.2.0",
  "org.apache.spark" %% "spark-sql" % "4.2.0",
  "org.apache.spark" %% "spark-mllib" % "4.2.0",
  "net.razorvine" % "pickle" % "1.4"
)
