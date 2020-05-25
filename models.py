from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import os.path
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import MinMaxScaler

"""Create Spark context with Spark configuration."""
conf = SparkConf().setAppName("Practica 4. Lidia Sanchez Merida.")
sc = SparkContext(conf=conf)

"""Create a Spark session to create a new dataframe"""
ss = SparkSession.builder.appName("Practica 4. Lidia Sanchez Merida.").getOrCreate()

def is_df(df_file):
    """Checks if the dataset file exists in order to read it and get the dataframe."""
    if (os.path.exists(df_file) and os.path.isfile(df_file)):
        df = ss.read.csv(df_file, header=True, sep=",", inferSchema=True)
        return df
    else:
        raise Exception("ERROR. The file doesn't exists.")

def preprocess_df(df, selected_columns, label_column):
    """Preprocesses the dataframe in order to train the models. First we add
        a feature column in which all predicted columns are together. Then we add
        a label column with the indexes of each class."""
    assembler_features = VectorAssembler(inputCols=selected_columns, outputCol="features")
    label_indexes = StringIndexer(inputCol = 'class', outputCol = 'label')
    label_indexes = label_indexes.setHandleInvalid("skip")
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

def scale_features(df):
    """Normalize features in range [0,1] with MinMax scale
        by adding the column 'scaledFeatures' """
    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
    scalerModel = scaler.fit(df)
    scaledData = scalerModel.transform(df)
    return scaledData

def evaluate_model(model_summary, predictions, file):
    """Evaluates a model by using its summary and predictions in order to get
        the area under the curve ROC, accuracy, Kappa coefficient and the values of
        the confusion matrix. All this data will be stored in a csv file."""
    # ROC
    roc = round(model_summary.areaUnderROC*100, 3)
    # Values of confusion matrix: true positives, true negatives
    tp = predictions[(predictions['class'] == 1) & (predictions['prediction'] == 1)].count()
    tn = predictions[(predictions['class'] == 0) & (predictions['prediction'] == 0)].count()
    fp = predictions[(predictions['class'] == 0) & (predictions['prediction'] == 1)].count()
    fn = predictions[(predictions['class'] == 1) & (predictions['prediction'] == 0)].count()
    total = tp+tn+fp+fn
    # Accuracy
    accuracy = float(tp+tn)/float(tp+tn+fp+fn)
    accuracy = round(accuracy*100,3)
    # Kappa Coefficient
    prob_observed = float(tp+tn)/total
    prob_expected = float(((tn+fp)*(tn+fn))+((fn+tp)*(fp+tp)))/(total*total)
    kappa = (float(prob_observed-prob_expected)/(1-prob_expected))
    kappa = round(kappa*100,3)
    """Store the results as a dataframe in a csv file"""
    results = [(str(roc), str(accuracy), str(kappa), str(tn), str(fn), str(fp), str(tp))]
    schema = StructType([
        StructField('ROC', StringType(), False),
        StructField('Accuracy', StringType(), False),
        StructField('Kappa', StringType(), False),
        StructField('TN', StringType(), False),
        StructField('FN', StringType(), False),
        StructField('FP', StringType(), False),
        StructField('TP', StringType(), False),
    ])
    results_df = ss.createDataFrame(results, schema)
    results_df.write.csv(file, header=True, mode="overwrite")

def binomial_logistic_regression(train, test, iters, regularization):
    """Binomial Logistic Regression model in which it uses cross validation to
        get the best lambda value to train the model."""
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', 
        maxIter=iters, elasticNetParam=regularization)
    # Cross validation for regParam
    grid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01, 0.001, 0.0001]).build()
    evaluator = BinaryClassificationEvaluator()
    cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)
    cv_model = cv.fit(train)
    best_model = cv_model.bestModel
    """Train with the best lambda for the specified regularization and iters"""
    best_lambda = best_model._java_obj.getRegParam()
    lr = LogisticRegression(featuresCol = 'scaledFeatures', labelCol = 'label', 
        maxIter=iters, regParam=best_lambda, elasticNetParam=regularization)
    lrModel = lr.fit(train)
    """Summary of the model and predictions"""
    trainingSummary = lrModel.summary
    predictions = lrModel.transform(test)
    
    return [trainingSummary, predictions]

if __name__ == "__main__":
    my_df = is_df("./filteredC.small.training")
    # My 6 columns + Class column
    my_cols = ["PSSM_r1_2_F", "PSSM_r1_-2_F", "PSSM_r2_1_I", "PSSM_r1_3_F", "PSSM_r1_-1_S", "PSSM_r2_3_M"]
    label_col = ["class"]
    preproc_df = preprocess_df(my_df, my_cols, label_col)
    scaled_df = scale_features(preproc_df)
    """Get the train (70%) and test (30%) dataset"""
    train, test = scaled_df.randomSplit([0.7, 0.3], seed = 2020)
    
    """Binomial Logistic Regression models"""
    results_ridge = binomial_logistic_regression(train, test, 10000, 0.0)
    evaluate_model(results_ridge[0], results_ridge[1], 'blg.ridge')
    results_lasso = binomial_logistic_regression(train, test, 10000, 1.0)
    evaluate_model(results_lasso[0], results_lasso[1], 'blg.lasso')