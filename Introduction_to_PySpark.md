# Introduction to PySpark

## 結構

* Spark 把資料切開到不同的 nodes 上，每個 node 只有一部分的資料，所以**可以做平行計算**
  * Cluster 包含了一個 master node 和剩下的 slaves nodes (workers)
  * master manages splitting up the data and the computations.
  * slaves 實際執行運算，運算完後把結果傳回給 master
* Spark 的資料結構是 **Resilient Distributed Dataset (RDD)**
  * 但是 RDD 不好直接操作，所以有個基於 RDD 的 Spark DataFrame
  * Spark DataFrame 就像是 SQL 的表格一樣

## SparkContext

* 要使用 Spark DataFrame 前，要先用 `SparkContext()` 建立 SparkSession 物件
  * SparkContext 就像是和 cluster 建立連結，而連結建立以後 SparkSession 就是操作介面
* SparkContext:
  * 通常簡寫為 sc
  * 用 `sc.version` 查看 Spark 版本
  * 可以在類別的建構子的參數中指定 cluster 屬性，`SparkConf()` 建構子可以建立一個物件保存這些屬性
* 同時建立許多 SparkSessions 和 SparkContexts 會造成問題，所以用 `SparkSession.builder.getOrCreate()` 傳回目前的 SparkSession ，若是不存在時就建立一個新的

```python
# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession
# Create my_spark
my_spark = SparkSession.builder.getOrCreate()
```

* SparkSession
  * `catalog` 屬性列出 cluster 中的所有 data，這個 `catalog` 屬性有許多方法可以呼叫，例如：`spark.catalog.listTables()` 傳回 cluster 中所有表格的名字

```python
# Print the tables in the catalog
print(spark.catalog.listTables())
```

## Spark 的基本操作
* 可以用 SQL queries 來查詢表格

```python
from pyspark.sql import SparkSession
spark.sql(query) # spark 是 SparkSession 的物件，query 是 SQL 查詢命令組成的字串
spark.sql(query).show() # 顯示查詢結果
spark.sql(query).toPandas() # 把查詢結果轉換成 Pandas DataFrame
```

* `Spark_DF = spark.createDataFrame(pandas_df)` 把 Pandas DataFrame 轉成 Spark DataFrame
  * 但是轉完之後 SparkSession 並**不能**存取，要先把轉完後的 DataFrame 註冊到 SparkSession 之後才能用 (catalog 中要有，才可以用)
  * 註冊的方式是  `Spark_DF.createOrReplaceTempView("表格的名字")`
    * 用 `Spark_DF.createTempView("表格的名字")` 也可以，但是差別在於前者在表格不存在時，會建立 temporary table，若表格已經存在了，就只是更新表格
* `spark.read.csv(file_path, header=True)` 是從檔案中讀入 Spark DataFrame
  * `header=True` 表示用 csv 中的第一列當 column names
* 在 Spark 中要做 column-wise 的操作時是使用 `spark_DF.withColumn('新欄位的名字', 新欄位的值)` 方法，新的欄位必須是 Column 類別的物件
  * `spark_DF.colName` 可以產生 Column 類別的物件
  * `df = spark_DF.withColumn("newCol", spark_DF.oldCol 運算)`
* `Spark.table('表格的名字')` 會用表格來產生 Spark DataFrame ，其中表格必須是要用 `spark.catalog.listTables()` 查詢時能看到的
* Spark DataFrame 是**不可變的 immutable**，所有對 DataFrame 做的動作都是傳回一個新的 DataFrame
  * 把原先的 DataFrame 覆寫掉 `df = df.withColumn("newCol", df.oldCol 做某運算)`
  * 把原先的 column 覆寫掉 `df = df.withColumn("oldCol", df.oldCol 做某運算)`
* `spark_DF.fliter(condition)`
  * 相當於 SQL 中的 `WHERE`，是用來篩選符合條件的資料
  * condition 可以是經 spark 操作後的**布林判斷**，也可以是 SQL 的 **WHERE 子句的字串**
    * 例如:

```python
flights.filter(flights.air_time > 120).show() # 是用布林判斷
flights.filter("air_time > 120").show() # 是用 WHERE 子句的字串

# Filter flights with a SQL string
long_flights1 = flights.filter("distance > 1000")
# Filter flights with a boolean column
long_flights2 = flights.filter(flights.distance > 1000)
```

* `spark_DF.select('欄位名字')`
  * 相當於 SQL 中的 `SELECT`
  * 參數可以用**欄位的名字的字串**或是用 **spark Column 類別的物件**
    * 如果用的是 spark Column 類別的物件，那可以對 column 做運算
    * 如果參數用的是欄位的名字的字串，是無法對欄位做運算的，要改用 `spark_DF.selectExpr('SQL 命令的字串')` 才可以用 SQL 的方式對欄位做運算
    * 例如:

```python
flights.select(flights.air_time/60) # 用 select() 又要對欄位做運算，那只能用 spark 的 Column 物件
flights.selectExpr("air_time/60 as duration_hrs") # 要用 SQL 命令對欄位做運算，就要改用 selectExpr()
```

  * `spark_DF.select()` 和 `spark_DF.withColumn()` 功能很像
    * `spark_DF.select()` 傳回的是**選擇欄位**
    * `spark_DF.withColumn()` 傳回的是**全部的欄位**

* `spark_column_obj.alias()`
  * 相當於 SQL 裡的 `AS`
  * 例如: `flights.select((flights.air_time/60).alias("duration_hrs"))`
* `spark_DF.groupBy()` 會產生 GroupedData 物件
  * 如果有參數時，就相當於 SQL 的 `GROUP BY`
  * Aggregation function (像是 `.max()`, `.min()`, `.count()`, `.avg()` 等等) 是 GroupedData 物件的方法
  * 例如: `spark_DF.groupBy().min("col").show()` 注意 Aggregation function 中放的是欄位的名字的字串
  * 如果用 `spark_DF.groupBy().agg( F.函數(欄位名字) )` 的話，就可以使用定義在 `pyspark.sql.functions` 的 Aggregation function，像是 `F.stddev()` 只是記得要先 `import pyspark.sql.functions as F`
* `spark_DF1.join(spark_DF2, on='欄位名字', how='leftouter')`
  * 依照指定的欄位來結合兩個表格

* 在 `payspark.ml` 中最重要的兩個模組是 Transformer 和 Estimator
  * `Transformer.transform()` 把輸入的 DataFrame 變成另一個新的 DataFrame
  * `Estimator.fit()` 也是輸入 DataFrame 但是傳回 model 物件

* `spark_DF.withColumnRenamed('舊欄位名字', '新欄位名字')` 可以幫欄位改名字

## Spark Machine Learning

* 用 spark 做 ML 的話，只能處理**數值的資料**
  * 所以要先用 `spark_DF.withColumn( spark_DF.欄位.cast('型態') )` 來對欄位做型態轉換
* categorical feature 要先建立 `StringIndexer` 然後再建立 `OneHotEncoder` 變成數值的才能用 spark 做 ML

```python
pyspark.ml.feature
欄位名字_indexer = StringIndexer(inputCol="欄位名字", outputCol="欄位名字_index")
欄位名字_encoder = OneHotEncoder(inputCol="欄位名字_index", outputCol="欄位名字_fact")
```

* Spark 需要把所有的 feature 的欄位變成**單一的一個欄位**，然後才能做 model
  * 其實就是把每一個 row 的全部欄位用 `VectorAssembler` 變成一個大大的 vector

```python
vec_assembler = VectorAssembler(inputCols=[欄位列表], outputCol="features")
```

* 要把的動作加到 Pipeline 裡面

```python
# Import Pipeline
from pyspark.ml import Pipeline
# Make the pipeline
spark_DF_pipe = Pipeline(stages=[欄位1_indexer, 欄位1_encoder, 欄位2_indexer, 欄位2_encoder, vec_assembler])
piped_data = spark_DF_pipe.fit(spark_DF).transform(spark_DF)
```

* 都做完後才能做 train test split

```python
training, test = piped_data.randomSplit([train 要多少比例, test 要多少比例]) # 比例是介於 0 ~ 1
```

* Logistic regression 是 classification 的一種，是用來預測屬於某個事件的機率，所以我們要提供一個 cutoff 或是叫做 threshold 來判斷預測的結果是否屬於某個事件

```python
# Import LogisticRegression
from pyspark.ml.classification import LogisticRegression
# Create a LogisticRegression Estimator
lr = LogisticRegression()
```

* Hyperparameter 是我們提供的數值，不是由 model 學到的
* k-fold cross validation: 把 training data 分成 k 等分 (payspark 預設是分 3 份)，其中 k-1 份拿來做訓練，訓練完後拿剩下的那一份求誤差。這個動作對每一份 data 都做一次，會得到 k 個誤差值，cross validation error 就是這 k 個誤差值的平均

* 建立 evaluator

```python
# Import the evaluation submodule
import pyspark.ml.evaluation as evals
# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")
```

* 建立 grid


`payspark.ml.tuning.ParamGridBuilder()` 建立 grid 物件，用 `grid.addGrid(參數, 參數的範圍)` 把 hyperparameter 加到 grid 中，再用 `grid.build()` 不加參數建立 grid

```python
# Import the tuning submodule
import pyspark.ml.tuning as tune
# Create the parameter grid
grid = tune.ParamGridBuilder()
# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])
# Build the grid
grid = grid.build()
```

* 建立 CrossValidator 然後做 model fitting 來求出最佳的 hyperparameters，再用最佳的 hyperparameters 來對 test data 做預測

```python
# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)
# Fit cross validation models
models = cv.fit(training)
# Extract the best model
best_lr = models.bestModel
# Use the model to predict the test set
test_results = best_lr.transform(test)
# Evaluate the predictions
print(evaluator.evaluate(test_results))
```
