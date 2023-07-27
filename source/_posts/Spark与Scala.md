---
title: Spark与Scala
tags: [ Scala, Spark ]
categories: [ Computer Technology ]
top: false
comments: true
lang: en
toc: true
excerpt: 使用Scala语言编写Spark程序
swiper: false
swiperDesc: 使用Scala语言编写Spark程序
tocOpen: true
onlyTitle: false
share: true
copyright: true
donate: true
bgImgTransition: fade
bgImgDelay: 180000
prismjs: default
mathjax: false
imgTop: ture
date: 2023-07-27 09:13:17
updated: 2023-07-27 09:13:17
swiperImg:
bgImg:
img: https://s1.imagehub.cc/images/2023/07/27/2023-07-27-10.35.38.png
---

# Spark与Scala

Scala是一种通用的编程语言，旨在融合面向对象编程和函数式编程的特性。它运行在Java虚拟机（JVM）上，因此可以与Java库和框架无缝交互，并且可以利用Java生态系统中的各种工具和库。Scala的设计目标之一是让编程更具表现力和简洁，同时提供强大的类型系统和静态类型检查。

Apache Spark是一个开源的大数据处理框架，它旨在处理大规模数据集的分布式计算。Spark提供了许多用于处理数据的高级API，例如批处理（Spark Core）和流处理（Spark Streaming），还支持SQL查询（Spark SQL）、图处理（GraphX）和机器学习（MLlib）等功能。Spark的一个关键特点是其内存计算能力，它允许在内存中进行数据处理，从而大大提高了性能。

Scala与Spark之间有着密切的关系。事实上，Spark最初是用Scala编写的，并且Scala一直是Spark主要的编程语言。这是因为Scala语言具有与Spark的分布式计算模型和函数式编程特性非常契合的特点。Scala的函数式编程范式特性使得Spark的代码可以更具表现力和易读性，从而更容易编写和维护复杂的分布式计算任务。

Spark提供了Scala API以及其他语言（如Java、Python和R）的API，但Scala API通常被认为是最原生和最灵活的。许多Spark的核心开发人员和社区成员都喜欢使用Scala来编写Spark应用程序，因为Scala的语法和功能可以更好地利用Spark提供的高级特性，使得开发更加高效。

总结：Scala是Spark的主要编程语言，它与Spark之间有着紧密的关系，Scala的设计特点与Spark的分布式计算模型和函数式编程特性非常契合，使得使用Scala编写Spark应用程序更加方便和高效。

## 下载

{% link Scala官网, https://www.scala-lang.org/, https://www.scala-lang.org/resources/img/frontpage/scala-spiral.png %}

{% link Spark官网, https://spark.apache.org/, https://spark.apache.org/images/spark-logo-rev.svg %}

## 使用IDEA构建一个Spark Scala项目

1. 下载Scala

2. 在IDEA中设置使用Scala语言
   添加maven依赖
    ```xml
    <dependencies>
        <!-- https://mvnrepository.com/artifact/org.apache.spark/spark-core -->
        <dependency>
            <groupId>org.apache.spark</groupId>
            <artifactId>spark-core_2.11</artifactId>
            <version>2.4.3</version>
        </dependency>
        <!-- https://mvnrepository.com/artifact/org.apache.spark/spark-sql -->
        <dependency>
            <groupId>org.apache.spark</groupId>
            <artifactId>spark-sql_2.11</artifactId>
            <version>2.4.3</version>
        </dependency>
    </dependencies>
    ```
   
3. 将Scala添加到全局库中

4. 下载插件Scala并重启IDEA

## 案例解析

```scala
import com.hankcs.hanlp.HanLP
import com.hankcs.hanlp.seg.common.Term
import org.apache.log4j.{Level, Logger}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import java.util

object SearchEngineLogAnalysis {
 def main(args: Array[String]): Unit = {
   Logger
     .getLogger("org.apache.spark")
     .setLevel(Level.ERROR)
   val conf = new SparkConf()
     .setAppName("SearchEngineLogAnalysis")
     .setMaster("local[*]")
   val sparkContext = new SparkContext(config = conf)
   val file = "/Volumes/KeQing/Documents/IntelliJ IDEA Ultimate/SparkTraning/src/main/resources/reduced.txt"
   val outputPath1 = "/Volumes/KeQing/Documents/IntelliJ IDEA Ultimate/SparkTraning/src/main/resources/output/queryWordsWordCountRDD"
   val outputPath2 = "/Volumes/KeQing/Documents/IntelliJ IDEA Ultimate/SparkTraning/src/main/resources/output/userQueryAnalysisRDD"
   val outputPath3 = "/Volumes/KeQing/Documents/IntelliJ IDEA Ultimate/SparkTraning/src/main/resources/output/timeQueryCountAnalysisRDD"
   val rddSougouRecord = sparkContext.textFile(file)
     .filter(line => line != null && line.trim.split("\\s+").length == 6)
     .mapPartitions(it => {
       it.map(line => {
         val contents = line.trim.split("\\s+")
         SogouRecord(
           contents(0),
           contents(1),
           contents(2).replaceAll("[\\[|\\]]", ""),
           contents(3).toInt,
           contents(4).toInt,
           contents(5)
         )
       })
     })
   rddSougouRecord.persist(StorageLevel.MEMORY_AND_DISK).count()

   val queryWordsWordCountRDD = rddSougouRecord.mapPartitions(it => {
     it.flatMap(record => {
       val terms: util.List[Term] = HanLP.segment(record.queryWords.trim)
       import scala.collection.JavaConverters._
       terms.asScala.map(_.word)
     })
   }).map((_, 1)).reduceByKey(_ + _).sortBy(_._2, ascending = false)

   queryWordsWordCountRDD.coalesce(numPartitions = 1).saveAsTextFile(outputPath1)

   val userQueryAnalysisRDD = rddSougouRecord.mapPartitions(it => {
     it.map(record => {
       ((record.userId, record.queryWords), 1)
     })
   }).reduceByKey(_ + _).sortBy(_._2, ascending = false)
   userQueryAnalysisRDD.coalesce(1).saveAsTextFile(outputPath2)

   val timeQueryCountAnalysisRDD = rddSougouRecord.mapPartitions(it => {
     it.map(record => {
       (record.queryTime.substring(0, 2), 1)
     })
   }).reduceByKey(_ + _).sortBy(_._2, ascending = false)
   timeQueryCountAnalysisRDD.coalesce(1).saveAsTextFile(outputPath3)

   rddSougouRecord.unpersist()
   sparkContext.stop()
 }
}
```

### 逐步分析

```scala
   Logger
     .getLogger("org.apache.spark")
     .setLevel(Level.ERROR)
```

设置`org.apache.spark`的日志输出级别，可以在测试阶段更加方便的观察控制台的输出结果

```scala
   val conf = new SparkConf()
     .setAppName("SearchEngineLogAnalysis")
     .setMaster("local[*]")
```
这段代码是使用Scala语言创建一个Spark应用程序的配置对象（SparkConf），并对其进行一些设置。

1. `val conf = new SparkConf()`: 首先，创建了一个名为`conf`的不可变（val）变量，并通过`new SparkConf()`来实例化一个新的SparkConf对象。SparkConf是用于配置Spark应用程序的类。

2. `.setAppName("SearchEngineLogAnalysis")`: 使用`.setAppName()`方法为Spark应用程序设置一个名称，这里将应用程序的名称设置为"SearchEngineLogAnalysis"。应用程序的名称将在Spark集群中显示，用于标识不同的应用程序。

3. `.setMaster("local[*]")`: 使用`.setMaster()`方法设置Spark应用程序的主节点URL或本地模式。在这里，将主节点URL设置为"local[*]"，表示在本地模式下运行Spark应用程序，并使用所有可用的CPU内核。

综合起来，这段代码的作用是创建一个Spark应用程序的配置对象，设置了应用程序的名称为`SearchEngineLogAnalysis`，并将其配置为在本地模式下运行，并使用所有可用的CPU内核进行并行处理。请注意，在本地模式下运行的Spark应用程序将不会连接到Spark集群，而是在本地计算机上以单机模式运行，适用于开发和测试阶段。在实际生产环境中，你需要将`.setMaster()`方法的参数设置为连接到Spark集群的主节点URL，以便在集群上分布式运行你的Spark应用程序。

```scala
val sparkContext = new SparkContext(config = conf)
```
这段代码用于创建一个SparkContext对象，它是Spark应用程序与Spark集群之间的主要接口。

1. `val sparkContext`: 这是一个不可变（val）变量，用于存储SparkContext对象。

2. `new SparkContext(config = conf)`: 这是创建SparkContext对象的语句。通过`new SparkContext()`来实例化一个新的SparkContext对象，并通过`config = conf`参数将之前创建的SparkConf对象`conf`传递给SparkContext构造函数。

在这里，我们使用之前配置好的SparkConf对象`conf`来初始化SparkContext。SparkConf对象包含了应用程序的各种配置设置，而SparkContext则使用这些配置在Spark集群上执行任务。

需要注意的是，创建SparkContext对象是Spark早期版本（2.x及之前）的做法。从Spark 2.0版本开始，推荐使用SparkSession来代替SparkContext。SparkSession是一个更高级的接口，它封装了SparkContext和SQLContext，并且提供了更方便的功能，例如对DataFrame和Dataset的支持。

如果在使用Spark 2.x及之后版本，建议改用SparkSession来初始化Spark应用程序，示例如下：

```scala
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder()
  .appName("SearchEngineLogAnalysis")
  .master("local[*]")
  .getOrCreate()
```

使用`SparkSession.builder()`来构建SparkSession对象，并使用`.appName()`方法设置应用程序名称，`.master()`方法设置主节点URL或本地模式。最后，通过`.getOrCreate()`方法来获取或创建一个SparkSession对象。使用SparkSession之后，可以通过`.sparkContext`属性来访问底层的SparkContext对象，以便进行一些底层操作。

```scala
   val rddSougouRecord = sparkContext.textFile(file)
     .filter(line => line != null && line.trim.split("\\s+").length == 6)
     .mapPartitions(it => {
       it.map(line => {
         val contents = line.trim.split("\\s+")
         SogouRecord(
           contents(0),
           contents(1),
           contents(2).replaceAll("[\\[|\\]]", ""),
           contents(3).toInt,
           contents(4).toInt,
           contents(5)
         )
       })
     })
```
这段代码是使用Spark的RDD（Resilient Distributed Dataset） API来读取一个文本文件，并根据特定的规则将每行数据转换为自定义的SogouRecord对象。

1. `val rddSougouRecord = sparkContext.textFile(file)`: 这行代码读取文本文件并创建了一个RDD（Resilient Distributed Dataset）。`file`是指定的文件路径。`textFile`方法将文件的每一行作为RDD的一个元素。

2. `.filter(line => line != null && line.trim.split("\\s+").length == 6)`: 在上一步创建的RDD上调用`filter`方法，过滤掉一些不符合条件的行。`filter`方法接受一个函数作为参数，这里使用了一个Lambda表达式。Lambda表达式中的条件`line != null && line.trim.split("\\s+").length == 6`判断是否满足过滤条件：行不为null且按空格拆分后有6个元素。

3. `.mapPartitions(it => { ... })`: 这是一个转换操作，使用`mapPartitions`方法对每个分区（partition）中的数据进行转换。`mapPartitions`方法也接受一个函数作为参数，该函数的输入是每个分区的迭代器（iterator），输出是转换后的迭代器。

4. `it.map(line => { ... })`: 在`mapPartitions`方法的函数体内，对每个分区的数据（即每行文本）进行转换。使用`map`方法对每行文本应用一个函数，将文本按空格拆分，并根据自定义的规则构造`SogouRecord`对象。

5. `val contents = line.trim.split("\\s+")`: 将当前行去除首尾空格后按空格进行拆分，将结果保存在`contents`数组中。

6. `SogouRecord(...)`：根据拆分后的`contents`数组的内容构造`SogouRecord`对象。`SogouRecord`是一个自定义的类，它包含了6个字段，对应着每行文本中拆分后的6个元素。

最后，整个代码的作用是从给定的文件中读取数据，根据特定的条件过滤掉无效的行，然后按照自定义的规则将每行文本转换为`SogouRecord`对象，并最终得到一个包含`SogouRecord`对象的RDD `rddSougouRecord`。

```scala
rddSougouRecord.persist(StorageLevel.MEMORY_AND_DISK).count()
```
这行代码对之前的RDD `rddSougouRecord` 进行了持久化（缓存）操作，并计算了RDD中元素的数量。

1. `rddSougouRecord.persist(StorageLevel.MEMORY_AND_DISK)`: 这部分代码对RDD `rddSougouRecord` 进行了持久化操作。在Spark中，RDD是一个不可变的分布式数据集，当对一个RDD执行一系列的转换操作时，每个转换都会生成一个新的RDD，而原始RDD则不会受到影响。持久化（缓存）操作的目的是将RDD的数据缓存在内存或磁盘中，这样可以避免在后续的操作中重复计算，提高计算效率。

   `persist`方法接受一个参数，即存储级别（StorageLevel）。在这里，使用了`StorageLevel.MEMORY_AND_DISK`存储级别，表示将数据缓存在内存中，如果内存空间不足，则溢写到磁盘。这样可以在内存充足时快速访问数据，并在内存不足时保留数据在磁盘上，以便后续使用。

2. `.count()`: 这部分代码对持久化后的RDD执行了`count`操作，用于计算RDD中元素的数量。`count`是一个动作（action）操作，会触发实际的计算并返回RDD中元素的数量。

综合起来，这行代码的作用是对RDD `rddSougouRecord` 进行持久化操作，将数据缓存在内存或磁盘中，然后计算RDD中元素的数量。持久化操作使得在后续对`rddSougouRecord`执行其他转换或动作操作时，可以从缓存中快速访问数据，而不必每次都重新计算。

```scala
   val queryWordsWordCountRDD = rddSougouRecord.mapPartitions(it => {
     it.flatMap(record => {
       val terms: util.List[Term] = HanLP.segment(record.queryWords.trim)
       import scala.collection.JavaConverters._
       terms.asScala.map(_.word)
     })
   }).map((_, 1)).reduceByKey(_ + _).sortBy(_._2, ascending = false)
```
这段代码是对之前持久化的RDD `rddSougouRecord` 进行一系列转换操作，最终得到一个包含查询词（queryWords）及其出现频率的RDD `queryWordsWordCountRDD`。

1. `val queryWordsWordCountRDD = rddSougouRecord.mapPartitions(it => { ... })`: 这行代码首先对持久化的RDD `rddSougouRecord` 进行了 `mapPartitions` 转换操作。`mapPartitions` 方法对 RDD 的每个分区进行操作，并返回一个新的 RDD，它的每个元素都是转换后的结果。这里使用了一个 Lambda 表达式来处理每个分区的数据。

2. `it.flatMap(record => { ... })`: 在 `mapPartitions` 中，对每个分区的元素进行 `flatMap` 转换操作。`flatMap` 方法类似于 `map` 方法，但是它可以将每个输入元素映射为零个或多个输出元素。在这里，`flatMap` 用于将每个 `SogouRecord` 对象的 `queryWords` 字段拆分为单个词语。

3. `val terms: util.List[Term] = HanLP.segment(record.queryWords.trim)`: 这行代码使用了 HanLP 中文分词库对每个 `queryWords` 进行中文分词，并将结果保存在 `terms` 变量中。`HanLP.segment` 方法返回一个 `util.List[Term]` 对象，其中包含了分词后的词语。

4. `import scala.collection.JavaConverters._`: 这行代码导入了 Scala 与 Java 集合之间的转换工具，用于将 Java 集合转换为 Scala 集合，方便后续的处理。

5. `terms.asScala.map(_.word)`: 在前面的步骤中，将 `queryWords` 分词后得到的结果是一个 Java 集合，这里使用 `asScala` 方法将其转换为 Scala 集合，并使用 `map` 方法提取每个 `Term` 对象的 `word` 字段，即每个词语，形成一个包含所有词语的新的 Scala 集合。

6. `.map((_, 1))`: 这一步是对每个词语添加一个计数值 1，将每个词语映射为一个二元组 `(词语, 1)`。

7. `.reduceByKey(_ + _)`: 接着使用 `reduceByKey` 方法，将相同的词语进行分组并求和。即将相同词语的计数值相加，得到每个词语的出现频率。

8. `.sortBy(_._2, ascending = false)`: 最后使用 `sortBy` 方法，按照词语出现频率（即二元组的第二个元素）进行降序排序，得到包含查询词及其出现频率的 RDD `queryWordsWordCountRDD`。

综合起来，这段代码的作用是将每个 `SogouRecord` 对象的 `queryWords` 字段进行中文分词，统计查询词的出现频率，并按频率降序排列，得到一个包含查询词及其出现频率的 RDD `queryWordsWordCountRDD`。

```scala
queryWordsWordCountRDD.coalesce(numPartitions = 1).saveAsTextFile(outputPath1)
```
这段代码是将之前得到的RDD `queryWordsWordCountRDD` 进行一系列操作，并将结果保存为文本文件。

1. `queryWordsWordCountRDD.coalesce(numPartitions = 1)`: 这部分代码使用了 `coalesce` 方法对RDD进行重新分区。`coalesce` 方法可以将RDD的分区数量减少到指定的数量。在这里，`numPartitions = 1` 表示将RDD合并为一个分区，也就是将所有数据合并到一个分区中。

2. `.saveAsTextFile(outputPath1)`: 接着使用 `saveAsTextFile` 方法将RDD保存为文本文件。`saveAsTextFile` 方法将RDD中的每个元素（每行文本）写入到文本文件中。`outputPath1` 是保存文本文件的输出路径。

综合起来，这段代码的作用是将之前计算得到的 `queryWordsWordCountRDD` 重新分区为一个分区，并将结果保存为文本文件。在文本文件中，每个查询词及其出现频率将以文本行的形式存储。注意，由于使用了 `coalesce(1)`，结果文件将只有一个分区，并且所有数据都会写入同一个文件中。如果数据量很大，可能会造成单个文件过大的问题。如果想要分多个文件保存，可以调整 `coalesce` 方法中的分区数为需要的值。

```scala
   rddSougouRecord.unpersist()
   sparkContext.stop()
```
这段代码是对之前持久化的RDD `rddSougouRecord` 进行解除持久化，并停止SparkContext对象，从而终止Spark应用程序的执行。

1. `rddSougouRecord.unpersist()`: 这行代码调用了 `unpersist` 方法，用于解除RDD的持久化。在之前的代码中，我们对 `rddSougouRecord` 进行了持久化（缓存）操作，将数据缓存到内存或磁盘中，以便后续的操作可以快速访问数据。而 `unpersist` 方法的作用就是解除持久化，即释放RDD的缓存，从而释放内存或磁盘空间。在某些情况下，当RDD不再被频繁使用时，可以手动调用 `unpersist` 方法来释放资源，以避免占用过多的存储空间。

2. `sparkContext.stop()`: 这行代码调用了 `stop` 方法，用于停止SparkContext对象，从而终止Spark应用程序的执行。`SparkContext` 是Spark应用程序与Spark集群之间的主要接口，调用 `stop` 方法将会关闭与集群的连接，并释放资源。在应用程序执行完成后，通常会调用 `stop` 方法来优雅地终止Spark应用程序，释放集群资源，避免资源浪费。

综合起来，这段代码的作用是解除之前对RDD的持久化，释放缓存的资源，并停止SparkContext对象，从而正常终止Spark应用程序的执行。

## 使用Spark SQL案例解析

```scala
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

object SogouAnalysisSQL {
  def main(args: Array[String]): Unit = {
    // 设置输出的日志级别
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    val spark = SparkSession.builder()
      .appName("SogouAnalysisSQL")
      .master("local[*]")
      .getOrCreate()

    val file = "/Volumes/KeQing/Documents/IntelliJ IDEA Ultimate/SparkTraning/src/main/resources/reduced.txt"

    val fileRDD = spark.sparkContext.textFile(file)
      .filter(line => line != null && line.trim.split("\\s+").length == 6)

    val schema = StructType(Array(
      StructField("queryTime", StringType, false),
      StructField("userId", StringType, false),
      StructField("queryWords", StringType, false),
      StructField("resultRank", IntegerType, false),
      StructField("clickRank", IntegerType, false),
      StructField("clickUrl", StringType, false)
    ))

    val rddRow = fileRDD.mapPartitions(it => {
      it.map(line => {
        val contents = line.trim.split("\\s+")
        try {
          // 封装的字段的时候，顺序要与定义的schema顺序完全一致
          Row(
            contents(0),
            contents(1),
            contents(2).replaceAll("[\\[|\\]]", ""),
            contents(3).toInt,
            contents(4).toInt,
            contents(5)
          )
        } catch {
          case e: Exception => Row("error", "error", "error", 0, 0, "error")
        }
      })
    })

    val df = spark.createDataFrame(rddRow, schema)

    df.createOrReplaceTempView("sogou_view")

    val sql =
      """
        |select * from sogou_view where clickUrl like 'www%'
        |limit 3
        |""".stripMargin

    val sqlResult = spark.sql(sql)
    //    sqlResult.show(truncate = false)

    val querySQL =
      """
        |select userId,queryWords,count(*) as query_count from sogou_view
        |group by userId,queryWords
        |order by userid,query_count desc
        |""".stripMargin

    val queryResult = spark.sql(querySQL)

    //    queryResult.show(truncate = false)

    val timeSQL =
      """
        |select substring(queryTime,0,5) as query_time,count(*) as query_count from sogou_view
        |group by substring(queryTime,0,5)
        |order by query_time,query_count desc
        |""".stripMargin

    val timeResult = spark.sql(timeSQL)
    timeResult.show(truncate = false)
    timeResult.rdd.coalesce(1).saveAsTextFile("/Volumes/KeQing/Documents/IntelliJ IDEA Ultimate/SparkTraning/src/main/resources/output/timeResult")
    spark.stop()
  }
}
```

### 分析

```scala
    val schema = StructType(Array(
      StructField("queryTime", StringType, false),
      StructField("userId", StringType, false),
      StructField("queryWords", StringType, false),
      StructField("resultRank", IntegerType, false),
      StructField("clickRank", IntegerType, false),
      StructField("clickUrl", StringType, false)
    ))
```
这段代码定义了一个Spark SQL的Schema，用于描述数据的结构和字段类型。Spark SQL是Spark的模块，它提供了用于处理结构化数据的高级数据处理接口。

1. `val schema = StructType(Array(...))`: 这行代码创建了一个`StructType`对象，该对象表示了数据的结构。`StructType`是Spark SQL中用于表示数据结构的类，它包含一个包含了多个`StructField`的数组。

2. `StructField("queryTime", StringType, false)`: 这行代码创建了一个`StructField`对象，表示数据的一个字段。`StructField`接受三个参数：字段名称（"queryTime"）、字段类型（StringType）和是否可空（false）。

3. `StructField("userId", StringType, false)`: 同上，创建了一个表示"userId"字段的`StructField`对象。

4. `StructField("queryWords", StringType, false)`: 同上，创建了一个表示"queryWords"字段的`StructField`对象。

5. `StructField("resultRank", IntegerType, false)`: 同上，创建了一个表示"resultRank"字段的`StructField`对象。这里字段类型是`IntegerType`，表示整数类型。

6. `StructField("clickRank", IntegerType, false)`: 同上，创建了一个表示"clickRank"字段的`StructField`对象。

7. `StructField("clickUrl", StringType, false)`: 同上，创建了一个表示"clickUrl"字段的`StructField`对象。

综合起来，这段代码定义了一个包含六个字段的Schema，每个字段都有一个名称、一个字段类型和一个是否可空的标志。这个Schema描述了一个数据集，其中包含了查询时间（queryTime）、用户ID（userId）、查询词（queryWords）、结果排名（resultRank）、点击排名（clickRank）和点击URL（clickUrl）等六个字段。通过定义Schema，我们可以将RDD转换为DataFrame，从而利用Spark SQL的高级功能进行数据处理和查询。

```scala
    val rddRow = fileRDD.mapPartitions(it => {
      it.map(line => {
        val contents = line.trim.split("\\s+")
        try {
          // 封装的字段的时候，顺序要与定义的schema顺序完全一致
          Row(
            contents(0),
            contents(1),
            contents(2).replaceAll("[\\[|\\]]", ""),
            contents(3).toInt,
            contents(4).toInt,
            contents(5)
          )
        } catch {
          case e: Exception => Row("error", "error", "error", 0, 0, "error")
        }
      })
    })
```
这段代码对之前的RDD `fileRDD` 进行了一系列转换操作，将每行文本数据转换为Spark SQL的`Row`对象，并得到一个新的RDD `rddRow`。

1. `val rddRow = fileRDD.mapPartitions(it => { ... })`: 这行代码首先对RDD `fileRDD` 进行了 `mapPartitions` 转换操作。`mapPartitions` 方法对RDD的每个分区进行操作，并返回一个新的RDD，它的每个元素都是转换后的结果。这里使用了一个 Lambda 表达式来处理每个分区的数据。

2. `it.map(line => { ... })`: 在 `mapPartitions` 中，对每个分区的元素（即每行文本）进行转换操作。

3. `val contents = line.trim.split("\\s+")`: 将当前行去除首尾空格后按空格进行拆分，将结果保存在 `contents` 数组中。

4. `try { ... } catch { ... }`: 这是一个异常处理结构，用于捕获可能的异常。在这里，代码尝试根据之前定义的Schema封装每行数据为`Row`对象。如果成功封装，则返回封装好的`Row`对象；如果出现异常（例如数组越界或类型转换失败），则将异常捕获，并返回一个包含错误信息的`Row`对象。

5. `Row(...)`: 在 `try` 块中，根据之前定义的Schema构造`Row`对象。`Row`对象是Spark SQL中的一种数据结构，它用于表示一行数据，其中的参数按照之前定义的Schema的顺序对应各个字段。

6. `contents(2).replaceAll("[\\[|\\]]", "")`: 这行代码对第三个字段（contents(2)）进行处理，去除其中的方括号 "[ ]"。

7. `contents(3).toInt, contents(4).toInt`: 这行代码将第四个字段（contents(3)）和第五个字段（contents(4)）转换为整数类型。

8. `case e: Exception => Row("error", "error", "error", 0, 0, "error")`: 在 `catch` 块中，捕获异常，并返回一个包含错误信息的`Row`对象。如果出现异常，则用"error"字符串代替无法解析的字段，并将数值类型字段设为0。

综合起来，这段代码的作用是将每行文本数据转换为Spark SQL的`Row`对象，并根据之前定义的Schema对字段进行类型转换和处理。如果能成功转换，则得到一个包含正确数据的`Row`对象；如果出现异常，则返回一个包含错误信息的`Row`对象。最终得到的RDD `rddRow` 包含了经过处理的`Row`对象，可以用于构建DataFrame并使用Spark SQL的高级功能进行数据处理和查询。

```scala
    val df = spark.createDataFrame(rddRow, schema)
```
这段代码使用之前得到的RDD `rddRow` 和之前定义的Schema `schema` 来创建一个DataFrame。DataFrame是Spark SQL中的一个分布式数据表，它可以使用结构化的数据和Schema进行处理和查询。

1. `val df = spark.createDataFrame(rddRow, schema)`: 这行代码调用了SparkSession对象的 `createDataFrame` 方法，用于创建一个DataFrame。`createDataFrame` 方法接受两个参数：第一个参数是之前创建的RDD `rddRow`，第二个参数是之前定义的Schema `schema`。

2. `rddRow`: 这是之前转换得到的RDD，其中的每个元素都是一个Spark SQL的`Row`对象。该RDD包含了处理后的数据，每个`Row`对象按照之前定义的Schema的顺序对应各个字段。

3. `schema`: 这是之前定义的Schema，它描述了DataFrame中每个字段的名称和类型。

综合起来，这段代码的作用是利用之前处理得到的`rddRow` 和定义好的 `schema`，创建一个DataFrame `df`。DataFrame是一个结构化的、可分布式处理的数据表，现在我们可以使用`df` 来使用Spark SQL的高级功能进行数据处理和查询。DataFrame提供了更方便的数据处理接口，可以使用SQL语句或DataFrame API来进行数据分析、过滤、聚合、连接等操作。

```scala
    df.createOrReplaceTempView("sogou_view")
```
这段代码在Spark SQL中将DataFrame `df` 注册为一个临时视图（Temporary View），使其可以在当前SparkSession中被查询。临时视图是一种临时的数据表，仅在当前SparkSession的生命周期内有效，通常用于执行SQL查询。

1. `df.createOrReplaceTempView("sogou_view")`: 这行代码调用了DataFrame `df` 的 `createOrReplaceTempView` 方法，用于创建或替换一个临时视图。`createOrReplaceTempView` 方法接受一个字符串参数，表示视图的名称。在这里，视图名称被设置为"sogou_view"。

2. "sogou_view": 这是临时视图的名称。通过此名称，可以在当前SparkSession中使用SQL语句查询DataFrame `df` 的数据。

综合起来，这段代码的作用是在当前SparkSession中将DataFrame `df` 注册为一个临时视图，使得我们可以在后续的操作中使用SQL语句查询临时视图的数据。通过创建临时视图，我们可以使用更熟悉的SQL语法来进行数据查询和分析。视图只在当前SparkSession的生命周期内有效，不会被持久化到存储系统中。如果需要在不同的SparkSession中访问相同的视图，可以使用`createGlobalTempView`方法，创建一个全局临时视图。

```scala
    val timeSQL =
      """
        |select substring(queryTime,0,5) as query_time,count(*) as query_count from sogou_view
        |group by substring(queryTime,0,5)
        |order by query_time,query_count desc
        |""".stripMargin

    val timeResult = spark.sql(timeSQL)
    timeResult.show(truncate = false)
```
这段代码使用Spark SQL查询之前创建的临时视图 `sogou_view`，并统计查询时间（`queryTime`字段）的每年月份（截取前5个字符）的查询次数。然后按照查询时间（年月份）进行分组，并按查询次数降序排序，最后将查询结果打印输出。

1. `val timeSQL = """ ... """`: 这是一个包含SQL查询语句的多行字符串。在Scala中，可以使用三个双引号 `""" ... """` 来定义多行字符串，其中的换行符和缩进都会被保留。这个SQL查询语句用于统计查询时间（`queryTime`字段）的每年月份的查询次数，并按照年月份进行分组和排序。

2. `select substring(queryTime,0,5) as query_time, count(*) as query_count from sogou_view group by substring(queryTime,0,5) order by query_time, query_count desc`: 这是SQL查询语句的主体部分。在这里，使用`substring`函数截取`queryTime`字段的前5个字符（即年月份），并将其命名为`query_time`。然后使用`count(*)`函数对查询结果进行计数，得到每个年月份的查询次数。接着使用`group by`子句对结果按照年月份进行分组，最后使用`order by`子句按照年月份和查询次数降序排序。

3. `val timeResult = spark.sql(timeSQL)`: 这行代码使用`spark.sql`方法执行之前定义的SQL查询语句，并将结果保存在`timeResult`中。`spark.sql`方法接受一个SQL查询字符串作为参数，并返回一个DataFrame，其中包含了查询结果的数据。

4. `timeResult.show(truncate = false)`: 最后，使用`show`方法打印输出查询结果。`show`方法用于显示DataFrame的内容，默认显示前20行，并且截断字段内容以便于显示。通过设置`truncate = false`，可以禁用字段内容的截断，以便完整显示字段的内容。

综合起来，这段代码的作用是执行SQL查询，统计每个查询时间（年月份）的查询次数，并按照年月份进行分组和降序排序，然后将查询结果打印输出。