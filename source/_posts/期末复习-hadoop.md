---
title: 期末复习-hadoop
tags: [ 'Hadoop' ]
categories: [ 'Final Review' ]
top: false
comments: true
lang: en
toc: true
excerpt: Hadoop概念、指令、编程
swiper: false
swiperDesc: Hadoop概念、指令、编程
tocOpen: true
onlyTitle: false
share: true
copyright: true
donate: true
bgImgTransition: fade
bgImgDelay: 180000
prismjs: true
mathjax: false
imgTop: ture
date: 2023-07-07 21:01:48
updated: 2023-07-07 21:01:48
swiperImg:
bgImg:
img: https://i.postimg.cc/YCKLjNg9/2023-07-08-12-47-35.png

---

# Hadoop

## Hadoop基本概念

1. 奠定大数据技术的理论基石的Google公司的三篇论文包括：GFS、MapReduce、BigTable，被称为Google公司的“三驾马车”。分别对应了Hadoop的HDFS、MapReduce、Hbase。
2. Hadoop的运行模式包括：单机（本地）模式，伪分布式模式，完全分布式模式
3. Hadoop能够使用户轻松开发和运行处理大数据的应用程序，它的优点是：处理能力强，实现了高效性和高可靠性；在集群间分配数据，高扩展性；能够保存多个数据副本，高容错性。
4. 机架感知机制：同时兼顾节点之间的通信速度和提高数据的容错能力。
5. 虚拟机是一种虚拟化技术，它能够在现有的操作系统上运行一个或多个操作系统。
6. Hadoop的安装方式包括：单机模式，伪分布式模式和完全分布式模式三种。
7. 集群是指一组相互独立的、通过高速网络互联的计算机，它们构成了一个组，并以单一系统的模式加以管理。
8. 分布式系统是指将多台服务器集中在一起， 每台服务器都实现总体中的不同业务，做不同的事情。

在linux的命令行输入：

```shell
# 命令，能够启动所有Hadoop相关的服务进程。
stop-all.sh
# 命令，能够单独启动HDFS服务。
start-dfs.sh
# 查看在当前系统中启动了几个java进程的信息的
jps
# 实现显示本机IP 
ifconfig
```

## HDFS功能

1. HDFS是基于流数据模式访问和处理超大文件的需求而开发的，具有高容错、高可靠性、高可扩展性、高吞吐率等特征，适合的读写任务是一次写入，多次读。
2. 在HDFS中通过分块来实现数据的分布式存储功能，一个数据块会在多个此类节点中进行冗余备份。
3. NameNode是HDFS的管理节点，负责管理和维护HDFS的命名空间和管理DataNode上的数据块，它可以接受客户端的文件操作请求。所管理命名空间的信息文件有两个：fsimage和edits。
4. SecondaryNameNode，其主要职责是定期把NameNode上的fsimage和edits下载并合并生成新的fsimage。
5. 在Hadoop2.2以上版本中，HDFS默认Block Size的大小是128m。
6. HDFS的缺点是：不适合处理低延迟的数据访问，不适合多用户写入及任意修改文件，不适合处理大量的小文件。
7. 高可用性：是指通过在同一集群中运行两个NameNode的方式来防止由于单个NameNode故障引起的整个HDFS集群无法使用。
8. 配额：是指可以设置HDFS在某个目录下文件数量与目录数量之和最大值。
9. 联邦：是指在HDFS中有多个NameNode或NameSpace是联合的，相互独立，各自分工，管理自己的区域。
10. 在core-site.xml修改fs.trash.interval选项来设置已删除文件在回收站中的保留时间。

HDFS Shell命令

```shell
# 创建目录（文件夹）
hdfs dfs -mkdir
```

HDFS API

* getBlockSize：获得指定文件的数据块大小
* getLength：实现返回文件长度

## YARN的功能

1. Hadoop2.0相比较Hadoop1.0增加了YARN模块以实现对上层应用提供统一的资源管理和调度功能。
2. Container是YARN中的资源抽象，它封装了某个节点上的内存、CPU、磁盘、网络等多维度资源。
3. YARN提供先入先出调度器，容器调度器，公平调度器三种调度机制。

## MapReduce的功能

1. MapReduce的特点：易于编程，分布式程序可以在大量廉价的PC上运行；良好的扩展性，可以通过增加集群中计算机数来提高计算能力；高容错性，当一台主机出故障计算任务可以转移到其他节点上。
2. MapReduce在对大数据的计算过程至少分为Map和Reduce两个部分。
3. MapReduce在Map步骤后对数据进行分区，排序，溢写，合并的步骤可以统称为 Shuffle 过程。Shuffle功能根据其功能和在MapReduce过程中的步骤，可以分为Map
   Shuffle和Reduce Shuffle。
4. Hadoop的内置类型IntWritable用来定义整形数据，与Java的int类型对应。
5. Hadoop的内置类型 LongWritable 用来定义长整型型数据，与Java的long类型对应。
6. 序列化是一种将内存中的Java对象转化为其他可存储文件或可跨计算机传输数据流的技术。
7. Hadoop序列化实现了org.apche.hadoop.io.Writable接口，其中包含write()和readFields()两个方法。
8. LineRecordReader 类实现了MapReduce从文本文件中逐行读取文本内容，生成<Key,Value>格式数据。
9. MapReduce通过分片对数据进行逻辑划分，在进行Map计算时所划分的处理数据对MapTask一一对应。所保存的HDFS的数据块（Block）与MapReduce所处理的数据切片（split）大小关系是相等、整数倍、几分之一的关系。
10. 在Map阶段的分区步骤实现了根据Key值对数据进行划分，使其分配给对应ReduceTask。

## 简答题

1. 简述大数据理念的四个特征：<br/>
   数据体量大：大数据的数据量从 TB级别跃升到PB级别。<br/>
   数据类型多：大数据的数据类型包括前文提到的网络日志、视频、图片、地理位置信息等。<br/>
   处理速度快：1秒定律。这是大数据技术与传统数据挖掘技术的本质区别。<br/>
   价值密度低：以视频为例，在连续不间断的视频监控过程中，可能有用的数据仅仅有一两秒。
2. 简述Hadoop对 Google公司三篇论文思想的实现的功能：<br/>
   HDFS是分布式文件系统，实现了分布式计算中数据存储管理功能，它是对GFS论文思想的实现。<br/>
   MapReduce是用于大规模数据集（大于1TB）的并行运算编程模型，它是对MapReduce论文思想的实现。<br/>
   HBase是一个分布式的、面向列的开源数据库。它是在Hadoop之上提供了类似于 BigTable的能力，是对BigTable论文思想的实现。
3. 简述对HDFS进行访问和操作的三种方式：<br/>
   通过Web界面进行HDFS操作，使用浏览器访问HDFS对应IP端口为50070的页面对HDFS进行访问和操作。<br/>
   HDFS命令是由一系列类使用Linux
   Shell的命令组成的，命令分为操作命令，管理命令和其他命令。可以通过Linux或远程控制段的命令行输入Shell名称来对HDFS进行访问和操作。<br/>
   Hadoop提供了多种HDFS的访问接口，Hadoop Java API位于“org.apache.hadoop.fs”包中，能够实现通过Java语言对HDFS进行文件操作和文件系统管理的访问功能。
4. 简述HDFS的三个优点：<br/>
   处理超大文件：HDFS能够处理TB级甚至PB级的数据。<br/>
   支持流式数据访问：HDFS
   设计建立在“一次写入，多次读取”的基础上，意味着一个数据集一旦生成，就会被复制分发到不同的存储节点，然后响应各种数据分析任务请求。<br/>
   低成本运行：HDFS可运行在低廉的商用硬件集群上。
5. 简述MapReduce的四个特点：<br/>
   易于编程：用它的一些简单接口，就可以完成一个分布式程序，<br/>
   良好的扩展性：当计算资源不能得到满足的时候，可以通过简单地增加计算机来扩展它的计算能力。<br/>
   高容错性：一台主机出问题了，它可以把上面的计算任务转移到另外一个节点上运行，不至于使这个任务运行失败，而且这个过程不需要人工干预，完全由MapReduce在内部完成。<br/>
   能对PB级以上海量数据进行离线处理：MapReduce适合离线处理而不适合实时处理比如毫秒级别地返回一个结果，MapReduce很难做到。
6. 简述Hadoop的三种安装模式：<br/>
   单机模式：指Hadoop运行在一台主机上，按默认配置以非分布式模式运行一个独的Java进程。单机模式是Hadoop的默认模式。<br/>
   伪分布式模式：指Hadoop运行在一台主机上，使用多个Java进程，模仿完全分布式的各类节点。伪分布式模式具备完全分布式的所有功能，常用调试程序。<br/>
   完全分布式模式：也叫集群模式，是将 Hadoop运行在多台主机中，各个主机按照相关配置运行相立的Hadoop守护进程。完全分布式模式是真正的分布式环境，可用于实际的生产环境。

## 易错概念

1. Hadoop是Java开发的，所以 MapReduce只支持Java语言编写。
2. Mapreduce的input split就是一个block。
3. 客户端上传的数据首先发送给NameNode，根据NameNode所管理的DataNode的分工再逐个发送给DataNode。
4. 为了实现环形缓冲区功能，Hadoop需要配置特殊结构的环形内存芯片。
5. MapReduce是Java语言开发的，因此其数据类型也沿用了Java的数据类型。
6. 如果NameNode意外终止，SecondaryNameNode会接替它使集群继续工作。
7. 为了保证数据的稳定性，数据切片（Split）的大小必须是数据块（block）的一样大。
8. 高可用性HA通过在一个集群中同时处于Active状态的两个NameNode来保证HDFS能够正常运行。
9. 数据块的大小（BlockSize）是不可以修改的。
10. 在MapReduce计算中为了提高数据计算速度，当单个MapTask完成后数据会直接被拉取到Reduce阶段进行后续计算。
11. 由于会出现相同的Key在不同的ReduceTask进行统计的情况，因此在Reduce阶段最后需要对所有节点所生成所有文件进行合并。

## HDFS Shell命令操作

1. 在HDFS的根目录中上创建一个目录结构，为`new/folder`
   ```shell
   hdfs dfs -mkdir -p /new/folder
   ```
2. 在HDFS根目录的`new`目录中创建文件`1.txt`
   ```shell
   hdfs dfs -touchz /new/1.txt
   ```
3. 将用户目录下的`context.txt`内容追加到HDFS文件`/new/1.txt`中
   ```shell
   hdfs dfs -appendToFile context.txt /new/1.txt
   ```
4. 查看HDFS文件`/new/1.txt`的文件内容
   ```shell
   hdfs dfs -cat /new/1.txt
   ```
5. 查看HDFS文件目录`/new`所占的磁盘空间大小
   ```shell
   hdfs dfs -du /new
   ```
6. 删除HDFS文件目录`/new`
   ```shell
   hdfs dfs -rm /new
   ```
7. 在HDFS的根目录中上创建一个目录结构，为`dir1/dir2/dir3`
   ```shell
   hdfs dfs -mkdir -p /dirl/dir2/dir3
   ```
8. 将Linux用户根目录中的文件`localFile.txt`复制到HDFS根目录中保存为`hdfsFile.txt`
   ```shell
   hdfs dfs -copyFromLocal localFile.txt /hdfsFile.txt
   ```
   ```shell
   hdfs dfs -put localFile.txt /hdfsFile.txt
   ```
9. 显示HDFS中级联列出根目录下的所有目录和文件
   ```shell
   hdfs dfs -ls -R /
   ```
10. 在HDFS根目录`dir1`目录中创建文件`1.txt`
   ```shell
   hdfs dfs -touchz /dir1/1.txt
   ```
11. 将用户目录下的`code2.txt`内容追加到HDFS文件`/dir1/1.txt`中
   ```shell
   hdfs dfs -appendToFile code2.txt /dir1/1.txt
   ```
12. 将dir1目录中创建文件`1.txt`下载到Linux的本地根目录中，保存为文件`downloadFile.txt`
   ```shell
   hdfs dfs -copyToLocal / dir1/1.txt downloadFile.txt
   ```
   ```shell
   hdfs dfs -get /dir1/1.txt downloadFile.txt
   ```

### HDFS API编程

1. 已知HDFS的IP地址为`192.168.44.128`，访问端口为`9000`，用户名为`admin`。编程实现调用`FileSystem`
   类自带的`copyFromLocalFile`方法上传Linux系统中本地文件`/home/admin/upload.txt`到HDFS文件夹`/user/admin/`
   中保存为文件`1.txt`
   ```java
   package Exam;
   
   import java.net.URI;
   
   import org.apache.hadoop.conf.Configuration;
   import org.apache.hadoop.fs.FileSystem;
   import org.apache.hadoop.fs.Path;
   
   public class App {
       public static void main(String[] args) throws Exception {
           URI uri = new URI("hdfs://192.168.44.128:9000");
           Configuration conf = new Configuration();
           FileSystem fs = FileSystem.get(uri, conf, "admin");
           Path src = new Path("/home/admin/upload.txt");
           Path dst = new Path("/user/admin/1.txt");
           fs.copyFromLocalFile(src, dst);
           fs.close();
       }
   }
   ```

2. 已知HDFS的IP地址为`192.168.44.128`，访问端口为`9000`，用户名为`admin`。编程实现调用`FileSystem`类自带的`copyToLocalFile`
   方法下载HDFS文件夹`/user/admin/`中文件`1.txt`到Linux系统中本地目录`/home/admin/`保存为`download.txt`
   ```java
   package Exam;
   
   import java.net.URI;
   
   import org.apache.hadoop.conf.Configuration;
   import org.apache.hadoop.fs.FileSystem;
   import org.apache.hadoop.fs.Path;
   
   public class App {
       public static void main(String[] args) throws Exception {
           URI uri = new URI("hdfs://192.168.44.128:9000");
           Configuration conf = new Configuration();
           FileSystem fs = FileSystem.get(uri, conf, "admin");
           Path src = new Path("/user/admin/1.txt");
           Path dst = new Path("/home/admin/download.txt");
           fs.copyToLocalFile(src, dst);
           fs.close();
       }
   }
   ```

## MapReduce程序设计题

### 练习1

在某企业管理系统的数据库中所保存员工信息表EMP的结构如下所示：

| EMPNO | ENAME | JOB | MGR  | HIREDATE  | SAL   | COMM  | DEPTNO |
|-------|-------|-----|------|-----------|-------|-------|--------|
| 3641  | 石金泉   | 管理  | 0001 | 2009/3/4  | 20000 | 50000 | 1      |
| 6984  | 高丽    | 行政  | 5684 | 2009/3/7  | 3500  | 4000  | 1      |
| 7397  | 门亮    | 开发  | 7396 | 2009/5/11 | 10000 | 25000 | 2      |
| 7398  | 李贺    | 开发  | 7396 | 2009/9/22 | 10000 | 25000 | 2      |
| 8049  | 李加里   | 销售  | 3641 | 2009/3/21 | 8000  | 50000 | 3      |
| 8331  | 孙立宪   | 销售  | 8049 | 2009/4/29 | 6000  | 40000 | 3      |
| 9486  | 李明礼   | 生产  | 3641 | 2009/3/25 | 7000  | 9000  | 4      |
| 9696  | 张顺涛   | 生产  | 9486 | 2009/4/20 | 4000  | 5000  | 4      |

其中表中各字段代表的含义：

| 字段       | 含义   |
|----------|------|
| EMPNO    | 员工号  |
| ENAME    | 员工姓名 |
| JOB      | 岗位   |
| MGR      | 直属领导 |
| HIREDATE | 雇佣时间 |
| SAL      | 月薪   |
| COMM     | 奖金   |
| DEPTNO   | 部门号  |

将数据从数据库中导出，保存为文件emp.csv以作为统计每个部门员工月薪总额的基础数据，文件其内容为：

> 3641,石金泉,管理,0001,2009/3/4,20000,50000,1
>
> 6984,高丽,行政,5684,2009/3/7,3500,4000,1
>
> 7397,门亮,开发,7396,2009/5/11,10000,25000,2
>
> 7398,李贺,开发,7396,2009/9/22,10000,25000,2
>
> 8049,李加里,销售,3641,2009/3/21,8000,50000,3
>
> 8331,孙立宪,销售,8049,2009/4/29,6000,40000,3
>
> 9486,李明礼,生产,3641,2009/3/25,7000,9000,4
>
> 9696,张顺涛,生产,9486,2009/4/20,4000,5000,4

将`emp.csv`上传到HDFS目录`hdfs://192.168.44.128:9000/user/input/`

以MapReduce方式实现根据`emp.csv`中所包含的员工信息统计部门月薪总额的功能，以文字的方式分析求每个部门月工资总额数据的Map和Reduce步骤各完成的功能以及输入，输出数据类型。
根据注释描述分别填写Main类，Map类和Reduce类对应java代码。要求将计算的结果保存在HDFS目录`hdfs://192.168.44.128:9000/user/output/`
中

#### 程序分析

Map阶段输入类型为`Key LongWritable : Value Text`分别表示文本偏移量，当前行员工的信息

Map所实现的功能是根据逗号从当前行文本内容中提取出当前员工的部门编号和月薪

输出`key IntWritable : Value IntWritable`分别表示部门编号，当前行员工的月薪

Reduce阶段输入类型`Key IntWritable : Value IntWritable`分别表示部门编号，当前部门的员工月薪集合

Reduce所实现的功能是将当前部门的所有员工月薪进行累加得到整个部门的月薪总额

输出`Key IntWritable : Value IntWritable`分别表示部门编号，当前部门的月薪总额

#### 程序

1. ExamMapper Class
   ```java
   package MapReduceExam1;
   
   import java.io.IOException;
   
   import org.apache.hadoop.io.IntWritable;
   import org.apache.hadoop.io.LongWritable;
   import org.apache.hadoop.io.Text;
   import org.apache.hadoop.mapreduce.Mapper;
   
   public class ExamMapper extends Mapper<LongWritable, Text, IntWritable, IntWritable> {
       @Override
       protected void map(LongWritable key1, Text value1, Context context)
               throws IOException, InterruptedException {
           // 读取的一行文本内容
           String data = value1.toString();
           // 将字符串以逗号为标记点进行分割
           String[] words = data.split(",");
           // 输出< key,Value >
           context.write(new IntWritable(Integer.parseInt(words[7])), new IntWritable(Integer.parseInt(words[5])));
       }
   }
   ```

2. ExamReducer Class
   ```java
   package MapReduceExam1;
   
   import java.io.IOException;
   
   import org.apache.hadoop.io.IntWritable;
   import org.apache.hadoop.mapreduce.Reducer;
   
   public class ExamReducer extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {
       @Override
       protected void reduce(IntWritable key3, Iterable<IntWritable> value3, Context context)
               throws IOException, InterruptedException {
           int total = 0;
           for (IntWritable value : value3) {
               total += value.get();
           }
           context.write(key3, new IntWritable(total));
       }
   }
   ```

3. ExamMain Class
   ```java
   package MapReduceExam1;
   
   import org.apache.hadoop.conf.Configuration;
   import org.apache.hadoop.fs.Path;
   import org.apache.hadoop.io.IntWritable;
   import org.apache.hadoop.mapreduce.Job;
   import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
   import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
   
   public class ExamMain {
       public static void main(String[] args) throws Exception {
           Job job = Job.getInstance(new Configuration());
           // 设置Client的类
           job.setJarByClass(ExamMain.class);
           // 设置Map的类和输入类型
           job.setMapperClass(ExamMapper.class);
           job.setMapOutputKeyClass(IntWritable.class);
           job.setMapOutputValueClass(IntWritable.class);
           // 设置reduce的类和输入类型
           job.setReducerClass(ExamReducer.class);
           job.setOutputKeyClass(IntWritable.class);
           job.setOutputValueClass(IntWritable.class);
   
           FileInputFormat.setInputPaths(job, new Path("hdfs://192.168.44.128:9000/user/input/Employee/EMP.csv"));
           FileOutputFormat.setOutputPath(job, new Path("hdfs://192.168.44.128:9000/user/output/SalaryTotal"));
           job.waitForCompletion(true);
       }
   }
   ```

### 练习1

在某图书馆图书管理系统的数据库中所保存图书信息表BookInfo的结构如下所示：

| BOOKID | BNAME  | ROOM | TYPE | DATE      | PRICE |
|--------|--------|------|------|-----------|-------|
| 0001   | 老人与海   | 1    | 文学   | 2009/3/4  | 35    |
| 0002   | 时尚杂志   | 2    | 期刊杂志 | 2009/3/7  | 20    |
| 0003   | C语言    | 3    | 计算机  | 2009/5/11 | 32    |
| 0004   | 红楼梦    | 1    | 文学   | 2009/9/22 | 50    |
| 0005   | JAVA开发 | 3    | 计算机  | 2009/3/21 | 24    |
| 0006   | 计算机基础  | 3    | 计算机  | 2009/4/29 | 25    |
| 0007   | 西游记    | 1    | 文学   | 2009/3/25 | 40    |
| 0008   | 读者文摘   | 2    | 期刊杂志 | 2009/4/20 | 8     |

其中表中各字段代表的含义：

| 字段     | 含义   |
|--------|------|
| BOOKID | 图书编号 |
| BNAME  | 书名   |
| ROOM   | 所在房间 |
| TYPE   | 图书类型 |
| DATE   | 购入时间 |
| PRICE  | 价格   |

将数据从数据库中导出，保存为文件book.csv以作为统计每类图书总价格的基础数据，文件其内容为：

> 0001,老人与海,1,文学,2009/3/4,35
>
> 0002,时尚杂志,2,期刊杂志,2009/3/7,20
>
> 0003,C语言,3 计算机,2009/5/11,32
>
> 0004,红楼梦,1,文学,2009/9/22,50
>
> 0005,JAVA开发,3,计算机,2009/3/21,24
>
> 0006,计算机基础,3,计算机,2009/4/29,25
>
> 0007,西游记,1,文学,2009/3/25,40
>
> 0008,读者文摘,2,期刊杂志,2009/4/20,8

将`book.csv`上传到HDFS目录`hdfs://192.168.44.128:9000/user/input/`
以MapReduce方式实现根据book.csv中所包含的图书信息统计每类图书的总价格的功能，以文字的方式分析求每大类图书价格的Map和Reduce步骤各完成的功能以及输入，输出数据类型。根据注释描述分别填写Main类，Map类和Reduce类对应java代码。要求将计算的结果保存在HDFS目录`hdfs://192.168.44.128:9000/user/output/`
中。

#### 程序分析

Map阶段输入类型为`Key LongWritable : Value Text`分别表示文本偏移量，当前行图书的信息

Map所实现的功能是根据逗号从当前行文本内容中提取出当前图书的类型和价格

输出`key Text : Value IntWritable`分别表示当前行图书的类型和价格

Reduce阶段输入类型为`key Text : Value IntWritable`分别表示当前行图书的类型，当前类所有图书的价格集合

Reduce所实现的功能是将当前类型的所有图书价格进行累加得的价格总额

输出`key Text : Value IntWritable`分别表示图书类型，当前类型图书的价格总额

#### 程序

1. ExamMapper Class
   ```java
   package MapReduceExam2;
   
   import java.io.IOException;
   
   import org.apache.hadoop.io.IntWritable;
   import org.apache.hadoop.io.LongWritable;
   import org.apache.hadoop.io.Text;
   import org.apache.hadoop.mapreduce.Mapper;
   
   public class ExamMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
       @Override
       protected void map(LongWritable key1, Text value1, Context context) throws IOException, InterruptedException {
           String data = value1.toString();
           // 将字符串以逗号为标记点进行分割，保存如字符串数组中
           String[] words = data.split(",");
           context.write(new Text(words[3]), new IntWritable(Integer.parseInt(words[5])));
       }
   }
   ```
2. ExamReducer Class
   ```java
   package MapReduceExam2;
   
   import java.io.IOException;
   
   import org.apache.hadoop.io.IntWritable;
   import org.apache.hadoop.io.Text;
   import org.apache.hadoop.mapreduce.Reducer;
   
   public class ExamReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
       @Override
       protected void reduce(Text key3, Iterable<IntWritable> value3, Context context)
               throws IOException, InterruptedException {
           int total = 0;
           for (IntWritable value : value3) {
               total += value.get();
           }
           context.write(key3, new IntWritable(total));
       }
   }
   ```
3. ExamMain Class
   ```java
   package MapReduceExam2;
   
   import org.apache.hadoop.conf.Configuration;
   import org.apache.hadoop.fs.Path;
   import org.apache.hadoop.io.IntWritable;
   import org.apache.hadoop.io.Text;
   import org.apache.hadoop.mapreduce.Job;
   import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
   import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
   
   public class ExamMain {
       public static void main(String[] args) throws Exception {
           // 创建一个job作业实例
           Job job = Job.getInstance(new Configuration());
           // 设置作业的启动类
           job.setJarByClass(ExamMain.class);
           // 设置Map的类和输入类型
           job.setMapperClass(ExamMapper.class);
           job.setMapOutputKeyClass(Text.class);
           job.setMapOutputValueClass(IntWritable.class);
           // 设置reduce的类和输入类型
           job.setReducerClass(ExamReducer.class);
           job.setOutputKeyClass(Text.class);
           job.setOutputValueClass(IntWritable.class);
   
           FileInputFormat.setInputPaths(job, new Path("hdfs://192.168.44.128:9000/ExamData/book.csv"));
           FileOutputFormat.setOutputPath(job, new Path("hdfs://192.168.44.128:9000/ExamOutput/"));
           job.waitForCompletion(true);
       }
   }
   ```