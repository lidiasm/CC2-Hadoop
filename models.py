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
from pyspark.ml.classification import NaiveBayes
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier

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

def balanced_classes(df, file):
    """Gets the number of positive and negative labels in order to know if the 
        classes are balanced."""
    pos = df[df['class'] == 1].count()
    neg = df[df['class'] == 0].count()
    """Store the results as a dataframe in a csv file"""
    results = [(str(pos), str(neg))]
    schema = StructType([
        StructField('Positives', StringType(), False),
        StructField('Negatives', StringType(), False)
    ])
    results_df = ss.createDataFrame(results, schema)
    results_df.write.csv(file, header=True, mode="overwrite")

def preprocess_df(df, selected_columns, label_column):
    """Preprocesses the dataframe in order to train the models. First we add
        a feature column in which all predicted columns are together. Then we add
        a label column with the indexes of each class."""
    assembler_features = VectorAssembler(inputCols=selected_columns, outputCol="features")
    label_indexes = StringIndexer(inputCol = 'class', outputCol = 'label')
    # Avoid null issues
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
    scaler_model = scaler.fit(df)
    scaled_data = scaler_model.transform(df)
    return scaled_data

def under_sampling(df):
    """Undersamples the dataframe in order to balance the classes. To do that 
        the negative class will be reduced to the number of samples of the positive class."""
    # Get two dataframes with the negative and the positive class
    df0 = df[df['class'] == 0]
    df1 = df[df['class'] == 1]
    fr = float(df1.count()) / float(df0.count())
    # Sample the negative class
    new_df0 = df0.sample(withReplacement=False, fraction=fr, seed=2020)
    # Join the positive and the sampled negative dataframes
    balanced_df = new_df0.union(df1)
    return balanced_df

def evaluate_model(predictions, file):
    """Evaluates a model by using its predictions in order to get
        the area under the curve ROC, accuracy, Kappa coefficient and the values of
        the confusion matrix. All this data will be stored in a csv file."""
    # ROC
    #evaluator = BinaryClassificationEvaluator()
    #roc = round(evaluator.evaluate(predictions)*100, 3)
    roc = 0
    
    # Confusion matrix
    """Creates (prediction, label) pairs in order to use MulticlassMetrics"""
    predictionAndLabel = predictions.select("prediction", "label").rdd
    # Generate confusion matrix
    metrics = MulticlassMetrics(predictionAndLabel)
    cnf_matrix = metrics.confusionMatrix()
    cnf_matrix_list = cnf_matrix.toArray().tolist()
    tn = int(cnf_matrix_list[0][0])
    fn = int(cnf_matrix_list[1][0])
    fp = int(cnf_matrix_list[0][1])
    tp = int(cnf_matrix_list[1][1])
    total = tn + fn + fp + tp
    
    # Kappa Coefficient
    prob_observed = float(tp+tn)/total
    prob_expected = float(((tn+fp)*(tn+fn))+((fn+tp)*(fp+tp)))/(total*total)
    kappa = (float(prob_observed-prob_expected)/(1-prob_expected))
    kappa = round(kappa*100,3)
    
     # Accuracy
    accuracy = round(metrics.accuracy*100, 3)
    
    """Store the results as a dataframe in a csv file"""
    results = [(str(roc), str(accuracy), str(kappa), str(tn), str(fn), str(fp), str(tp))]
    schema = StructType([
        StructField('ROC', StringType(), False),
        StructField('Accuracy', StringType(), False),
        StructField('Kappa', StringType(), False),
        StructField('TN', StringType(), False),
        StructField('FN', StringType(), False),
        StructField('FP', StringType(), False),
        StructField('TP', StringType(), False)
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
    lr_model = lr.fit(train)
    """Summary of the model and predictions"""
    #summary = lr_model.summary
    predictions = lr_model.transform(test)
    
    return predictions

def naive_bayes(train, test):
    """Naive Bayes model. It uses cross validation to calculate the best smoothing
        value to train the model."""
    nb = NaiveBayes(modelType="multinomial", featuresCol='scaledFeatures')
    grid = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 0.5, 1.0]).build()
    evaluator = BinaryClassificationEvaluator()
    cv = CrossValidator(estimator=nb, estimatorParamMaps=grid, evaluator=evaluator)
    cv_model = cv.fit(train)
    best_model = cv_model.bestModel
    best_smooth = best_model._java_obj.getSmoothing()
    """Training with the best smoothing value"""
    best_nb = NaiveBayes(smoothing=best_smooth, modelType="multinomial", featuresCol='scaledFeatures')
    nb_model = best_nb.fit(train)
    predictions = nb_model.transform(test)
    
    return predictions

def decision_tree(train, test, imp, depth):
    """Decision Tree model in which it uses one decision tree in order to train
        a model. The maximum number of nodes can be specified (max 30)."""
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="scaledFeatures", 
        impurity=imp, maxDepth=depth, seed=2020)
    dt_model = dt.fit(train)
    predictions = dt_model.transform(test)
    
    return predictions

def random_forest(train, test, imp, depth, n_trees):
    """Random Forest model in which we specify the impurity metric, the maximum
        number of nodes in a branch (max 30) and the maximum number of trees."""
    rf = RandomForestClassifier(labelCol="label", featuresCol="scaledFeatures",
            maxDepth=depth, impurity=imp, seed=2020, numTrees=n_trees)
    rf_model = rf.fit(train)
    predictions = rf_model.transform(test)
    return predictions

def multilayer_perceptron(train, test, iters, layers):
    """Multilayer perceptron based on a feedforward neural network in which the
        number of iterations can be specified as well as the layers. The first layer
        has the number of features and the last the number of classes."""
    nn = MultilayerPerceptronClassifier(maxIter=iters, layers=layers, seed=2020)
    nn_model = nn.fit(train)
    predictions = nn_model.transform(test)
    return predictions

if __name__ == "__main__":
    my_df = is_df("./filteredC.small.training")
    # Balanced classes
    balanced_classes(my_df, './original.df.balanced.classes')
    # My 6 columns + Class column
    my_cols = ["PSSM_r1_2_F", "PSSM_r1_-2_F", "PSSM_r2_1_I", "PSSM_r1_3_F", "PSSM_r1_-1_S", "PSSM_r2_3_M"]
    label_col = ["class"]
    preproc_df = preprocess_df(my_df, my_cols, label_col)
    scaled_df = scale_features(preproc_df)
    """Get the train (70%) and test (30%) dataset"""
    train, test = scaled_df.randomSplit([0.7, 0.3], seed = 2020)
    balanced_train = under_sampling(train)
    #balanced_classes(balanced_train, './traindf.balanced.classes')
    
    """Binomial Logistic Regression models"""
    #preds_ridge = binomial_logistic_regression(balanced_train, test, 10000, 0.0)
    #evaluate_model(preds_ridge, 'blg.ridge')
    #preds_lasso = binomial_logistic_regression(balanced_train, test, 10000, 1.0)
    #evaluate_model(preds_lasso, 'blg.lasso')
    
    """Naive Bayes models"""
    #preds_nb = naive_bayes(balanced_train, test)
    #evaluate_model(preds_nb, 'naive.bayes.multinomial')
    
    """Decision Tree models"""
    #preds_dt_gini = decision_tree(balanced_train, test, 'gini', 15)
    #evaluate_model(preds_dt_gini, 'decision.tree.gini')
    #preds_dt_entropy = decision_tree(balanced_train, test, 'entropy', 15)
    #evaluate_model(preds_dt_entropy, 'decision.tree.entropy')
    
    """Random Forest models"""
   # preds_rf = random_forest(balanced_train, test, 'entropy', 15, 20)
    #evaluate_model(preds_rf, 'random.forest')
    
    """Multilayer Perceptron models"""
    layers = [6, 128, 64, 2]
    preds_nn = multilayer_perceptron(train, test, 100, layers)
    evaluate_model(preds_nn, 'nn.perceptron')