#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import lit, udf
from pyspark.sql.functions import percent_rank
from pyspark.sql import Window
import pyspark.sql.functions  as F
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler,VectorIndexer


from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline

import mlflow
from mlflow import spark

import boto3
import os
sc= SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

class UdfModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, ordered_df_columns, model_artifact):
        self.ordered_df_columns = ordered_df_columns
        self.model_artifact = model_artifact

    def load_context(self, context):
        import mlflow.pyfunc
        self.spark_pyfunc = mlflow.pyfunc.load_model(context.artifacts[self.model_artifact])

    def predict(self, context, model_input):
        renamed_input = model_input.rename(
            columns={
                str(index): column_name for index, column_name
                    in list(enumerate(self.ordered_df_columns))
            }
        )
        return self.spark_pyfunc.predict(renamed_input)

def log_udf_model(artifact_path, ordered_columns, run_id):
    udf_artifact_path = f"udf-{artifact_path}"
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mlflow.pyfunc.log_model(
        artifact_path = udf_artifact_path,
        python_model = UdfModelWrapper(ordered_columns, artifact_path),
        artifacts={ artifact_path: model_uri }
    )
    return udf_artifact_path


# In[2]:


customer = sqlContext.read.parquet('hdfs://isilon.tan.lab/tpch-s1/customer.parquet')
lineitem  = sqlContext.read.parquet('hdfs://isilon.tan.lab/tpch-s1/lineitem.parquet')
order = sqlContext.read.parquet('hdfs://isilon.tan.lab/tpch-s1/order.parquet')


# In[3]:


customer = customer.dropna()
lineitem  = lineitem.dropna()
order = order.dropna()


# In[4]:


print(customer.count(),lineitem.count(), order.count())


# In[5]:



sales = order.join(customer, order.o_custkey == customer.c_custkey, how = 'inner')
sales = sales.join(lineitem, lineitem.l_orderkey == sales.o_orderkey, how = 'full')
sales = sales.where('c_mktsegment == "BUILDING"').select('l_quantity','o_orderdate')

sales = sales.groupBy('o_orderdate').agg({'l_quantity': 'sum'}) .withColumnRenamed("sum(l_quantity)", "TOTAL_SALES") .withColumnRenamed("o_orderdate", "ORDERDATE")


sales = sales.withColumn('DATE', F.unix_timestamp(sales.ORDERDATE) )             .withColumn('DAY', F.dayofmonth(sales.ORDERDATE) )             .withColumn('WDAY', F.dayofweek(sales.ORDERDATE) )              .withColumn('YDAY', F.dayofyear(sales.ORDERDATE) )              .withColumn('WEEK', F.weekofyear(sales.ORDERDATE) )

sales = sales.withColumn("rank", percent_rank().over(Window.partitionBy().orderBy("DATE")))
training = sales.where("rank <= .8").drop("rank").drop("ORDERDATE")
testing  = sales.where("rank > .8").drop("rank").drop("ORDERDATE")


# In[7]:


featuresCols = training.columns
featuresCols.remove('TOTAL_SALES')
va = VectorAssembler(inputCols = featuresCols, outputCol = "features")


# In[8]:



os.environ['AWS_ACCESS_KEY_ID']= 'AKIAIOSFODNN7EXAMPLE'
os.environ['AWS_SECRET_ACCESS_KEY']='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
os.environ['MLFLOW_S3_ENDPOINT_URL']='http://minio.tan.lab'

remote_server_uri = "http://mlflow.tan.lab"
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment("git-gbt")


with mlflow.start_run() as run:
    # https://github.com/mlflow/mlflow/issues/2141
    classifier = GBTRegressor(featuresCol="features", labelCol="TOTAL_SALES", predictionCol="prediction")
    pipeline = Pipeline(stages=[va, classifier])
    model = pipeline.fit(training)
    mlflow.set_tag("change","remove debug")
    mlflow.spark.log_model(model, "gbt-model")
    runid = run.info.run_id
    experiment_id = run.info.experiment_id
    log_udf_model("gbt-model",  ['DATE', 'DAY', 'WDAY', 'YDAY', 'WEEK'], runid)


# In[ ]:




