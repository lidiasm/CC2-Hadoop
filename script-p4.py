from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession #SQLContext

"""Create Spark context with Spark configuration."""
conf = SparkConf().setAppName("Practica 4. Lidia Sanchez Merida.")
sc = SparkContext(conf=conf)

"""Create a Spark session to create a new dataframe"""
ss = SparkSession \
    .builder \
    .appName("Practica 4. Lidia Sanchez Merida.") \
    .getOrCreate()

def read_data():
    """Read the header file"""
    headers = sc.textFile("/user/datasets/ecbdl14/ECBDL14_IR2.header").collect()
    """Get the columns"""
    columns = [inp for inp in headers if "@inputs" in inp]
    """Get each column as a list element and delete '@inputs' and the first blank space"""
    list_columns = columns[0].replace('@inputs', '').replace(' ','').split(',')
    """Read data and set the columns with SQL context"""
#    sql_c = SQLContext(sc)
#    data = sql_c.read.csv("/user/datasets/ecbdl14/ECBDL14_IR2.data", header=False, inferSchema=True)
#    print(len(list_columns))
#    print(len(data.columns))
#    for c in range(0, len(data.columns)):
#        data = data.withColumnRenamed(data.columns[c], list_columns[c])
#    
#    return data

    """Read data and set the columns"""
    data = ss.read.csv("/user/datasets/ecbdl14/ECBDL14_IR2.data", header=False, inferSchema=True)
    for c in range(0, len(data.columns)):
        data = data.withColumnRenamed(data.columns[c], list_columns[c])

    return (data)

def create_new_df(df, columns):
    """Creates a new dataframe with the specified columns"""
    new_df = df.select(columns)
    new_df.write.csv('./filteredC.small.training', header=True, mode="overwrite")

if __name__ == "__main__":
    data = read_data()
    selected_columns = ["PSSM_r1_2_F", "PSSM_r1_-2_F", "PSSM_r2_1_I",
        "PSSM_r1_3_F", "PSSM_r1_-1_S", "PSSM_r2_3_M", "class"]
    create_new_df(data, selected_columns)
    sc.stop()

