from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import os.path
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

"""Create Spark context with Spark configuration."""
conf = SparkConf().setAppName("Practica 4. Lidia Sanchez Merida.")
sc = SparkContext(conf=conf)

"""Create a Spark session to create a new dataframe"""
ss = SparkSession.builder.appName("Practica 4. Lidia Sanchez Merida.").getOrCreate()

def is_df(df_file):
    """Checks if the dataset file exists"""
    if (os.path.exists(df_file) and os.path.isfile(df_file)):
        df = ss.read.csv(df_file, header=True, sep=",", inferSchema=True)
        return df
    else:
        raise Exception("ERROR. The file doesn't exists.")

def preprocess_df(df, selected_columns, label_column):
    """Preprocesses the dataframe in order to train several models. First we add
        a feature column in which all predicted columns are together. Then we add
        a label column with the indexes of each class."""
    assembler_features = VectorAssembler(inputCols=selected_columns, outputCol="features")
    label_indexes = StringIndexer(inputCol = 'class', outputCol = 'label')
    stages = []
    stages += [assembler_features]
    stages += [label_indexes]
    # Add both columns to the a new dataframe
    pipeline = Pipeline(stages = stages)
    pipeline_model = pipeline.fit(df)
    preprocessed_df = pipeline_model.transform(df)
    cols = ['label', 'features'] + df.columns
    preprocessed_df = preprocessed_df.select(cols)
    #preprocessed_df.printSchema()
    return preprocessed_df

def binomial_logistic_regression(train, test, iters):
    """Binomial Logistic Regression"""
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=iters)
    lrModel = lr.fit(train)
    """Summary of the model"""
    trainingSummary = lrModel.summary
    predictions = lrModel.transform(test)
    """ROC"""
    print('Area under ROC:', round(trainingSummary.areaUnderROC*100, 3), '%')

    """Confusion matrix"""
    # True Positives
    tp = predictions[(predictions.label == 1) & (predictions.prediction == 1)].count()
    # True Negatives
    tn = predictions[(predictions.label == 0) & (predictions.prediction == 0)].count()
    # False Positives
    fp = predictions[(predictions.label == 0) & (predictions.prediction == 1)].count()
    # False Negatives
    fn = predictions[(predictions.label == 1) & (predictions.prediction == 0)].count()
    total = tp+tn+fp+fn
    print('Confusion matrix')
    print([[tn,fn], [fp, tp]])
    
    """Accuracy"""
    accuracy = float(tp+tn)/float(tp+tn+fp+fn)
    print('Accuracy:',(round(accuracy*100,3)),'%')

    """Kappa"""
    # Probability observed
    po = float(tp+tn)/total
    # Probability expected
    pe = float(((tn+fp)*(tn+fn))+((fn+tp)*(fp+tp)))/(total*total)
    # Cohen's kappa Coefficient
    kappa = (float(po-pe)/(1-pe))
    print('Kappa Coefficient: ',(round(kappa*100,3)),'%')

if __name__ == "__main__":
  my_df = is_df("./filtered.small.training")
  """Preprocess the df"""
  my_cols = ["PSSM_r1_2_F", "PSSM_r1_-2_F", "PSSM_r2_1_I", "PSSM_r1_3_F", "PSSM_r1_-1_S", "PSSM_r2_3_M"]
  label_col = ["class"]
  preproc_df = preprocess_df(my_df, my_cols, label_col)
  preproc_df.printSchema()
  """Get the train (70%) and test (30%) dataset"""
  train, test = preproc_df.randomSplit([0.7, 0.3], seed = 2020)
  """Binomial Logistic Regression"""
  binomial_logistic_regression(train, test, 10)