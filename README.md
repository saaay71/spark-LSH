# Locality Sensitive Hashing (LSH) using Spark for Clustering

to be updated.




```
df = spark.read.format("csv").option("header",True) .load("data.csv")
df.show(truncate=False)
+---+------------------------------------------------------------------------------+
|id |Text                                                                          |
+---+------------------------------------------------------------------------------+
|1  |This is an example. Let's see if it can produce hashes close to the next line.|
|2  |This is another example. It should produce hashes close to the previous line. |
|3  |here we are using a different text and saying hello world!                    |
|4  |hello world! is different here because we are using and saying them first.    |
+---+------------------------------------------------------------------------------+
```


```
LSH_pipeline = build_LSH_pipeline(inputCol="Text", outputCol="hashes", n_ngram=3, seedValue=110, p=5)

train_df = df.select("Text")
trained_LSHModel = LSH_pipeline.fit(train_df)
trained_LSHModel.write().overwrite().save(MODEL_PATH)
```



```
saved_LSHModel = PipelineModel.load(MODEL_PATH)

df = saved_LSHModel.transform(df).select("id", "Text", "hashes")
df.show(truncate=False)

+---+------------------------------------------------------------------------------+-------------------------------------------------+
|id |Text                                                                          |hashes                                           |
+---+------------------------------------------------------------------------------+-------------------------------------------------+
|1  |This is an example. Let's see if it can produce hashes close to the next line.|[21235875, 6320997, 3753495, 724070, 415965]     |
|2  |This is another example. It should produce hashes close to the previous line. |[21235875, 13769064, 1722002, 724070, 415965]    |
|3  |here we are using a different text and saying hello world!                    |[31557483, 47151896, 5784988, 21384188, 870083]  |
|4  |hello world! is different here because we are using and saying them first.    |[31557483, 39703829, 5784988, 21384188, 18556512]|
+---+------------------------------------------------------------------------------+-------------------------------------------------+
```
