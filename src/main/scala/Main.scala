import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by tust on 02.09.2016.
  */
object Main {

  def main(args: Array[String]): Unit = {


    val numberOfOutput = 1
    val conf = new SparkConf().setAppName("DiplomaSparkProject").setMaster("local")
    val sc = new SparkContext(conf)

    val data = sc.textFile("D:\\procom_train.txt")
    val parsedData = data.map(s => Vectors.dense(s.split('\t').map(_.toDouble))).cache()
    // Cluster the data into two classes using KMeans

    val inputData = data.map(s => Vectors.dense(s.split('\t').dropRight(numberOfOutput).map(_.toDouble))).cache()
    val numberOfInput = inputData.first().size
    val outputData = data.map(s => Vectors.dense(s.split('\t').drop(numberOfInput).map(_.toDouble))).cache()

    //inputData.foreach{println}

    //outputData.foreach{println}
    val summary: MultivariateStatisticalSummary = Statistics.colStats(inputData)

    println(summary.max)

    val mat: RowMatrix = new RowMatrix(parsedData)


    // Get its size.
    val m = mat.numRows()
    val n = mat.numCols()


    val numClusters = 2 // Value of K in Kmeans
    val numIterations = 100
    val clusters = KMeans.train(parsedData, numClusters, numIterations)
    val cost = clusters.computeCost(parsedData)
    //  println("cost = " + cost)

    // println("PD:")
    //  parsedData.foreach{println}

    val NamesandData = data.map(s => (s.split('\t')(0), Vectors.dense(s.split('\t').drop(1).map(_.toDouble)))).cache()

    //  println("ND:")
    //  NamesandData.foreach{println}

    //
    // Print out a list of the clusters and each point of the clusters
    //val groupedClusters = NamesandData.groupBy{rdd => clusters.predict(rdd._2)}.collect()
    val groupedClusters = parsedData.groupBy { rdd => clusters.predict(rdd) }.collect()

    // println(groupedClusters.length);

    // println(groupedClusters(0)._2);
    // groupedClusters.foreach { println}

  }
}