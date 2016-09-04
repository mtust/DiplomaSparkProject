import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.util.StatCounter;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by tust on 03.09.2016.
 */
public class MainJava {

    private static final Logger LOGGER = Logger.getLogger(MainJava.class);

    public static void main(String[] args){




        SparkConf conf = new SparkConf().setAppName("DiplomaSparkProject").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(conf);



        List<Double> testData = IntStream.range(-5, 10).mapToDouble(d -> d).collect(ArrayList::new, ArrayList::add, ArrayList::addAll);



        JavaDoubleRDD rdd = jsc.parallelizeDoubles(testData);

        StatCounter statCounter = rdd.stats();
        LOGGER.info("Count:    " + statCounter.count());
        LOGGER.info("Min:      " + statCounter.min());
        LOGGER.info("Max:      " + statCounter.max());
        LOGGER.info("Sum:      " + statCounter.sum());
        LOGGER.info("Mean:     " + statCounter.mean());
        LOGGER.info("Variance: " + statCounter.variance());
        LOGGER.info("Stdev:    " + statCounter.stdev());

    }

}
