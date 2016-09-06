import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.mllib.util.MLUtils
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


    //read data from csv file just input
    val inputData = parsedData.map(s => Vectors.dense(s.toArray.dropRight(numberOfOutput))).cache()
    val numberOfInput = inputData.first().size

    //read data from csv just

    val outputData = parsedData.map(s => Vectors.dense(s.toArray.dropRight(numberOfInput))).cache()

    //inputData.foreach{println}

    //outputData.foreach{println}
    val summary: MultivariateStatisticalSummary = Statistics.colStats(inputData)

    val inputDataABS = inputData.map(s => Vectors.dense(s.toArray.map(el => Math.abs(el))))

   // inputDataABS.foreach{println}

    val summaryABS: MultivariateStatisticalSummary = Statistics.colStats(inputDataABS)



    //println("max:" + summary.max)
    //println("min:" + summary.min)
    //println("maxABS: " + summaryABS.max)
    //println(summary.max)


    //scaling(normalization) data
    val inputDataNormal = inputData.map(s => Vectors.dense((s.toArray, summaryABS.max.toArray).zipped.map(_/_)))

    //inputDataNormal.foreach(println)


    val mat: RowMatrix = new RowMatrix(parsedData)


    // Get its size.
    //val m = mat.numRows()
    //val n = mat.numCols()


    val numClusters = 2 // Value of K in Kmeans
    val numIterations = 20
    val clusters = KMeans.train(inputDataNormal, numClusters, numIterations)
    val cost = clusters.computeCost(inputDataNormal)
    //  println("cost = " + cost)

    // println("PD:")
    //  parsedData.foreach{println}

//    val NamesandData = data.map(s => (s.split('\t')(0), Vectors.dense(s.split('\t').drop(1).map(_.toDouble)))).cache()

    //  println("ND:")
    //  NamesandData.foreach{println}

    //
    // Print out a list of the clusters and each point of the clusters
    //val groupedClusters = NamesandData.groupBy{rdd => clusters.predict(rdd._2)}.collect()
    val groupedClusters = inputDataNormal.groupBy { rdd => clusters.predict(rdd) }.collect()

    val inputDataWithClusterIndex = inputDataNormal.map(s => Vectors.dense(s.toArray :+ clusters.predict(s).toDouble))

  val inputDataWithAdditionalColumn = inputDataWithClusterIndex.map(s => Vectors.dense(s.toArray :+ s.toArray(el =>)))



    println(inputDataWithClusterIndex.getClass)
    inputDataWithClusterIndex.foreach(println)

    // println(groupedClusters.length);

    // println(groupedClusters(0)._2);
    // groupedClusters.foreach { println}

  }
}