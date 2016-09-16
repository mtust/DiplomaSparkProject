import java.io._

import org.apache.spark.mllib.clustering.{KMeans, StreamingKMeans}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by tust on 02.09.2016.
  */
object Main {

  def main(args: Array[String]): Unit = {

    val numberOfOutput = 1
    val conf = new SparkConf().setAppName("DiplomaSparkProject").setMaster("local").set("spark.executor.memory", "2g")
    val sc = new SparkContext(conf)

    val dataUseWithHeader = sc.textFile("sample_test.csv")
   val  headerUse = dataUseWithHeader.first()
    val dataUse = dataUseWithHeader.filter(row => row != headerUse)
    val dataTrainWithHeader = sc.textFile("my_sample_train.csv")
    val  headerTrain = dataTrainWithHeader.first()
    val dataTrain = dataTrainWithHeader.filter(row => row != headerTrain)


    //val data = useData
    val parseDataUse = dataUse.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()
    val parseDataTrain = dataTrain.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

   // val parsedData = parseDataTrain.union(parseDataUse)


    val inputDataTrain = parseDataTrain.map(s => Vectors.dense(s.toArray.dropRight(numberOfOutput))).cache()
    val numberOfInput = inputDataTrain.first().size
    val outputDataTrain = parseDataTrain.map(s => Vectors.dense(s.toArray.dropRight(numberOfInput))).cache()
    val inputDataUse = parseDataUse.map(s => Vectors.dense(s.toArray.dropRight(numberOfOutput))).cache()
    val outputDataUse = parseDataUse.map(s => Vectors.dense(s.toArray.dropRight(numberOfInput))).cache()

    println(inputDataTrain.count())
    println(inputDataUse.count())

    // Cluster the data into two classes using KMeans/
    //read data from csv file just input
  //  val inputData = parsedData.map(s => Vectors.dense(s.toArray.dropRight(numberOfOutput))).cache()


    //read data from csv just

  //  val outputData = parsedData.map(s => Vectors.dense(s.toArray.dropRight(numberOfInput))).cache()


    //outputData.foreach{println}
    //  val summary: MultivariateStatisticalSummary = Statistics.colStats(inputData)

    val inputDataTrainABS = inputDataTrain.map(s => Vectors.dense(s.toArray.map(el => Math.abs(el))))

    val inputDataUseABS = inputDataUse.map(s => Vectors.dense(s.toArray.map(el => Math.abs(el))))
    // inputDataABS.foreach{println}

    val summaryTrainABS: MultivariateStatisticalSummary = Statistics.colStats(inputDataTrainABS)
    val summaryUseABS: MultivariateStatisticalSummary = Statistics.colStats(inputDataUseABS)



    //println("max:" + summary.max)
    //println("min:" + summary.min)
    //println("maxABS: " + summaryABS.max)
    //println(summary.max)


    //scaling(normalization) data
    val inputDataTrainNormal = inputDataTrain.map(s => Vectors.dense((s.toArray, summaryTrainABS.max.toArray).zipped.map(_ / _)))

    val inputDataUseNormal = inputDataUse.map(s => Vectors.dense((s.toArray, summaryUseABS.max.toArray).zipped.map(_ / _)))




    //inputDataNormal.foreach(println)


    //inputDataNormal.foreach(println)


    //  val mat: RowMatrix = new RowMatrix(parsedData)


    // Get its size.
    //val m = mat.numRows()
    //val n = mat.numCols()



    val numClusters = 2 // Value of K in Kmeans
    val numIterations = 20
    val clusters = KMeans.train(inputDataTrainNormal, numClusters, numIterations)


    val inputData = inputDataTrainNormal.union(inputDataUseNormal)

    //val cost = clusters.computeCost(inputDataTrainNormal)
    //  println("cost = " + cost)

    // println("PD:")
    //  parsedData.foreach{println}

    //    val NamesandData = data.map(s => (s.split('\t')(0), Vectors.dense(s.split('\t').drop(1).map(_.toDouble)))).cache()

    //  println("ND:")
    //  NamesandData.foreach{println}


    //val groupedClusters = inputDataNormal.groupBy { rdd => clusters.predict(rdd) }.collect()


    val inputDataWithClusterIndex = inputData.map(s => Vectors.dense(s.toArray :+ clusters.predict(s).toDouble))



    val normalizationData = normalization(inputDataWithClusterIndex)
//
//
//
//
    val additionalColumn = arrayToScalingNormalizaionVector(inputDataWithClusterIndex.
      map(s => magnitude(s.toArray)).collect())
//
//    //additionalColumn.foreach(println)
//
    val inputDataWithAdditionalColumn = normalizationData.zipWithIndex().map(s => Vectors.dense(s._1.toArray :+ additionalColumn.apply(s._2.toInt)))
//
//
//
//
//    //change it before add vector
//    //    inputDataWithAdditionalColumn.foreach(println)
//
//
    val normalizationSecondStepData = normalization(inputDataWithAdditionalColumn)

    println(normalizationSecondStepData.count())


    normalizationSecondStepData.foreach(println)





    // println(groupedClusters.length);

    // println(groupedClusters(0)._2);
    // groupedClusters.foreach { println}

  }

  def magnitude(x: Array[Double]): Double = {
    math.sqrt(x map (i => i * i) sum)
  }

  def arrayToScalingNormalizaionVector(x: Array[Double]): Array[Double] = {
    x.map(el => el / x.max)
  }

  def normalization(rddVecors: RDD[Vector]): RDD[Vector] = {
    rddVecors.map(s => Vectors.dense(s.toArray.map(el => el / magnitude(s.toArray))))
  }



}