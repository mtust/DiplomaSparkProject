
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.classification.{NaiveBayes}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
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
    //    val dataUseWithHeader = sc.textFile("procom_use.txt")
    val headerUse = dataUseWithHeader.first()
    val dataUse = dataUseWithHeader.filter(row => row != headerUse)
    val dataTrainWithHeader = sc.textFile("my_sample_train.csv")
    //    val dataTrainWithHeader = sc.textFile("procom_train.txt")
    val headerTrain = dataTrainWithHeader.first()
    val dataTrain = dataTrainWithHeader.filter(row => row != headerTrain)


    //val data = useData
    val parseDataUse = dataUse.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()
    val parseDataTrain = dataTrain.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

    // val parsedData = parseDataTrain.union(parseDataUse)


    val inputDataTrain = parseDataTrain.map(s => Vectors.dense(s.toArray.dropRight(numberOfOutput))).cache()
    val numberOfInput = inputDataTrain.first().size
    val outputDataTrain = parseDataTrain.map(s => Vectors.dense(s.toArray.drop(numberOfInput))).cache()
    val inputDataUse = parseDataUse.map(s => Vectors.dense(s.toArray.dropRight(numberOfOutput))).cache()
    val outputDataUse = parseDataUse.map(s => Vectors.dense(s.toArray.drop(numberOfInput))).cache()





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


    //    inputData.foreach(println)

    val inputDataWithClusterIndex = inputData.map(s => Vectors.dense(s.toArray :+ clusters.predict(s).toDouble))

    //inputDataWithClusterIndex.foreach(println)


    val normalizationData = normalization(inputDataWithClusterIndex)
    //
    //
    //
    //
    val additionalColumn = arrayToScalingNormalizaionVector(inputDataWithClusterIndex.
      map(s => magnitude(s.toArray)).collect())
    //
    //additionalColumn.foreach(println)
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

    val k: Integer = 2
    val n = normalizationSecondStepData.partitions.size
    val rdds = (0 until n) // Create Seq of partitions numbers
      .grouped(n / k) // group it into fixed sized buckets
      .map(idxs => (idxs.head, idxs.last)) // Take the first and the last idx
      .map {
      case (min, max) => normalizationSecondStepData.mapPartitionsWithIndex(
        // If partition in [min, max] range keep its iterator
        // otherwise return empty-one
        (i, iter) => if (i >= min & i <= max) iter else Iterator()
      )
    }

    val normalizationSecondStepDataTrain = rdds.next()
    val normalizationSecondStepDataUse = rdds.next()






    val trainDataWithOutput = normalizationSecondStepDataTrain.repartition(1).zip(outputDataTrain)
    //
    ////

    val inputDataTrainWithR1 = trainDataWithOutput.filter { case (x, y) => moreThanHalf(x, y) }
      .map(_._1).map(s => Vectors.dense(s.toArray :+ 1.toDouble))
    val inputDataTrainWithR2 = trainDataWithOutput.filter { case (x, y) => lessThanHalf(x, y) }
      .map(_._1).map(s => Vectors.dense(s.toArray :+ 0.toDouble)).repartition(1)

    val inputDataTrainWithR1R = inputDataTrainWithR1.repartition(1)
    val inputDataTrainWithR2R = inputDataTrainWithR2.repartition(1)


    val testFromUse = normalizationSecondStepDataUse.repartition(1).zip(outputDataUse)
      .map{case (input, output) => Vectors.dense(input.toArray ++ output.toArray)}
//        inputDataTrainWithR1R.saveAsTextFile("trainClass1")
//        inputDataTrainWithR2R.saveAsTextFile("trainClass2")
    //    normalizationSecondStepDataUse.saveAsTextFile("use")

//    testFromUse.saveAsTextFile("test")





//    val parseDataInputDataTrainWithR1R = inputDataTrainWithR1.zipWithIndex().map { case (line, i) => LabeledPoint(i.toDouble,line) }
//    val parseDataNormalizationSecondStepDataUse = normalizationSecondStepDataUse.zipWithIndex().map { case (line, i) => LabeledPoint(i.toDouble,line) }

    val tdr = trainDataWithOutput.map{ case (input, output) => Vectors.dense(input.toArray ++ output.toArray)}
    val parseDataTrainWithOutput = tdr
      .zipWithIndex().map { case (line, i) => LabeledPoint(i.toDouble,line) }
    val parseDataTestFromUse = testFromUse.zipWithIndex().map { case (line, i) => LabeledPoint(i.toDouble,line) }
//    val sparkSession =  SparkSession.builder().getOrCreate()
//    val dataset = sparkSession.createDataset(parseData)

    //   normalizationSecondStepData.foreach(println)

//    val model = NaiveBayes.train(parseDataTrainWithOutput, lambda = 1.0, modelType = "multinomial")
//    val predictionAndLabel = parseDataTestFromUse.map(p => (model.predict(p.features), p.label))
//    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / parseDataTestFromUse.count()
//
//
//    print(accuracy)



//    val spark = SparkSession
//      .builder()
//      .getOrCreate()
//
//    import spark.implicits._
//
//    // Apply the schema to the RDD
//    val df = spark.sparkContext.textFile("test/part-00000").map(_.split(","))
//      .toDF()
//
//    df.show()

//    val layers = Array[Int](4, 5, 4, 3)
//    // create the trainer and set its parameters
//    val trainer = new MultilayerPerceptronClassifier()
//      .setLayers(layers)
//      .setBlockSize(128)
//      .setSeed(1234L)
//      .setMaxIter(100)
//    // train the model
//    val model = trainer.fit(df)
//    // compute precision on the test set
//    val result = model.transform(df)
//    val predictionAndLabels = result.select("prediction", "label")
//    val evaluator = new MulticlassClassificationEvaluator()
//      .setMetricName("precision")
//    println("Precision:" + evaluator.evaluate(predictionAndLabels))


    // Save and load model


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

  def lessThanHalf(x: Vector, y: Vector): Boolean = {
    y.apply(0) < 0.5;
  }

  def moreThanHalf(x: Vector, y: Vector): Boolean = {
    y.apply(0) >= 0.5;
  }




}