# spark-LSH
Locality Sensitive Hashing (LSH) using Spark for clustering





```
df = spark.read.format("csv").option("header",True) .load("data.csv")
df.show(truncate=False)
+---+--------------------------------------------------------------------------+
|id |Text                                                                      |
+---+--------------------------------------------------------------------------+
|1  |This is an example. Let's see if it can produce hashes close to next line.|
|2  |This is another example. it should produce hashes close to previous line. |
|3  |here we are using a different text and saying hello world!                |
|4  |hello world! is different here because we are using it first.             |
+---+--------------------------------------------------------------------------+
```


```
LSH_pipeline = build_LSH_pipeline(inputCol="Text", outputCol="hashes", n_ngram=3, seedValue=110, p=5)

train_df = df.select("Text")
trained_LSHModel = LSH_pipeline.fit(train_df)
trained_LSHModel.write().overwrite().save(MODEL_PATH)
```



```
saved_LSHModel = PipelineModel.load(MODEL_PATH)

df = saved_LSHModel.transform(df)
df.show(truncate=False)

+---+--------------------------------------------------------------------------+------------------------------------------------+
|id |Text                                                                      |hashes                                          |
+---+--------------------------------------------------------------------------+------------------------------------------------+
|1  |This is an example. Let's see if it can produce hashes close to next line.|[41879091, 6320997, 5784988, 724070, 18556512]  |
|2  |This is another example. it should produce hashes close to previous line. |[41879091, 13769064, 3753495, 724070, 18556512] |
|3  |here we are using a different text and saying hello world!                |[21235875, 124956191, 5784988, 31714247, 415965]|
|4  |hello world! is different here because we are using it first.             |[21235875, 21217131, 1722002, 31714247, 415965] |
+---+--------------------------------------------------------------------------+------------------------------------------------+
```
