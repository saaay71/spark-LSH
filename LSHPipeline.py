import pyspark.sql.functions as F
from pyspark.sql import types as T
from pyspark.ml.feature import NGram, CountVectorizer, MinHashLSH
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable


class TextSpliter(Transformer, HasInputCol, HasOutputCol,
                  DefaultParamsReadable, DefaultParamsWritable):
    pattern = Param(Params._dummy(), "pattern",
                    "a string representing a regular expression. The regex string should be a Java regular expression.",
                    typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(TextSpliter, self).__init__()
        self.pattern = Param(self, "pattern", "")
        self._setDefault(pattern='')
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setPattern(self, value):
        return self._set(pattern=value)

    def getPattern(self):
        return self.getOrDefault(self.pattern)

    # Required in Spark >= 3.0
    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    # Required in Spark >= 3.0
    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    def _transform(self, dataset):
        output_col = self.getOutputCol()
        input_col = dataset[self.getInputCol()]
        return dataset.withColumn(output_col, F.split(input_col, self.getPattern()))


class NoneZeroVectorRemover(Transformer, HasInputCol, HasOutputCol,
                            DefaultParamsReadable, DefaultParamsWritable):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(NoneZeroVectorRemover, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    # Required in Spark >= 3.0
    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    # Required in Spark >= 3.0
    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    #     @udf(returnType=T.BooleanType())
    #     def isNoneZeroVector(vector):
    #         return vector.numNonzeros() > 0
    # return dataset.withColumn(self.getOutputCol(), F.col(self.getInputCol())).filter(isNoneZeroVector(F.col(self.getInputCol())))

    def _transform(self, dataset):
        def isNoneZeroVector(vector):
            return vector.numNonzeros() > 0

        output_col = self.getOutputCol()
        input_col = dataset[self.getInputCol()]
        return dataset.withColumn(output_col, input_col).filter(F.udf(isNoneZeroVector, T.BooleanType())(input_col))


class HashValuesToArray(Transformer, HasInputCol, HasOutputCol,
                        DefaultParamsReadable, DefaultParamsWritable):
    numHashes = Param(Params._dummy(), "numHashes", "numHashes",
                      typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(HashValuesToArray, self).__init__()
        self.numHashes = Param(self, "numHashes", "")
        self._setDefault(numHashes=1)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setNumHashes(self, value):
        return self._set(numHashes=value)

    def getNumHashes(self):
        return self.getOrDefault(self.numHashes)

    # Required in Spark >= 3.0
    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    # Required in Spark >= 3.0
    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    def _transform(self, dataset):
        output_col = self.getOutputCol()
        input_col = dataset[self.getInputCol()]
        n = self.getNumHashes()
        return dataset.withColumn(output_col, F.array(
            [vector_to_array(input_col.getItem(i)).getItem(0).cast("int") for i in range(n)]))


class ConcatNgrams(Transformer, HasInputCol, HasOutputCol,
                   DefaultParamsReadable, DefaultParamsWritable):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(ConcatNgrams, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    # Required in Spark >= 3.0
    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    # Required in Spark >= 3.0
    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    def _transform(self, dataset):
        output_col = self.getOutputCol()
        #       input_col = dataset[self.getInputCol()]
        input_cols = self.getInputCol().split(",")
        concat_cols = [col for col in input_cols]
        return dataset.withColumn(output_col, F.concat(*concat_cols)).drop(*concat_cols)


def build_NGram_pipeline(inputCol="TextCol", outputCol="ngrams_output",
                         n_ngram=2):  # n_ngram = [2, 3, 4]
    if isinstance(n_ngram, list):
        ngramStages = []
        ngramOutputCols = []
        for n in n_ngram:
            ngramOutputCol = "{}_{}_ngrams".format(outputCol, str(n))
            ngramOutputCols += [ngramOutputCol]
            ngramStages += [NGram().setN(n).setInputCol(inputCol).setOutputCol(ngramOutputCol)]

        concatNgrams = ConcatNgrams().setInputCol(",".join(ngramOutputCols)).setOutputCol(outputCol)

        ngramStages += [concatNgrams]

        ngram_pipeline = Pipeline().setStages(ngramStages)

        return ngram_pipeline
    # if n_ngram is not provided as a list then return a pipeline consisting of one NGram() with N set to n_ngram
    return Pipeline().setStages([NGram().setN(n_ngram).setInputCol(inputCol).setOutputCol(outputCol)])


def build_LSH_pipeline(inputCol="TextCol", outputCol="output1",
                       pattern="",
                       n_ngram=5,
                       vocabSize=2 ** 25, minTF=1, minDF=1, maxDF=2 ** 63 - 1,
                       seedValue=110, p=4
                       ):
    textSpliter = TextSpliter().setInputCol(inputCol).setOutputCol("{}_textSplited".format(outputCol)).setPattern(
        pattern)

    ngram_pipeline = build_NGram_pipeline(inputCol=textSpliter.getOutputCol(), outputCol="{}_ngrams".format(outputCol),
                                          n_ngram=n_ngram)

    countVectorizer = CountVectorizer() \
        .setInputCol(ngram_pipeline.getStages()[-1].getOutputCol()).setOutputCol("{}_countVector".format(outputCol)) \
        .setVocabSize(vocabSize) \
        .setMinTF(minTF) \
        .setMinDF(minDF) \
        .setMaxDF(maxDF)

    noneZeroVectorRemover = NoneZeroVectorRemover().setInputCol(countVectorizer.getOutputCol()).setOutputCol(
        "{}_noneZeroVectorRemoved".format(outputCol))

    minHashLSH = MinHashLSH().setNumHashTables(p).setSeed(seedValue) \
        .setInputCol(noneZeroVectorRemover.getOutputCol()).setOutputCol("{}_hashValue".format(outputCol))

    hashValuesToArray = HashValuesToArray().setNumHashes(p) \
        .setInputCol(minHashLSH.getOutputCol()).setOutputCol("{}".format(outputCol))

    pipelineStages = []
    if n_ngram == 0:
        countVectorizer.setInputCol(textSpliter.getOutputCol())
        pipelineStages = [textSpliter, countVectorizer, noneZeroVectorRemover, minHashLSH, hashValuesToArray]
    else:
        pipelineStages = [textSpliter, ngram_pipeline, countVectorizer, noneZeroVectorRemover, minHashLSH,
                          hashValuesToArray]
    LSH_pipeline = Pipeline().setStages(pipelineStages)

    return LSH_pipeline
