
ThisBuild / scalaVersion := "2.13.18"
ThisBuild / version := "0.1.0"

name := "training-data-poisoning"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "4.2.0",
  "org.apache.spark" %% "spark-sql" % "4.2.0",
  "org.apache.spark" %% "spark-mllib" % "4.2.0",
  "net.razorvine" % "pickle" % "1.4"
)

Compile / run / fork := true

Compile / run / javaOptions ++= Seq(
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
  "--add-opens=java.base/java.io=ALL-UNNAMED",
  "--add-opens=java.base/java.net=ALL-UNNAMED",
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/java.util=ALL-UNNAMED",
  "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
  "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
  "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
  "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
  "-Dlog4j.configurationFile=log4j2.properties",
  "-Dlog4j2.statusLoggerLevel=OFF",
  "-Ddev.ludovic.netlib.blas.allowNative=false",
  "-Ddev.ludovic.netlib.lapack.allowNative=false",
  "-Ddev.ludovic.netlib.arpack.allowNative=false"
)
