import time
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("scaling-test").getOrCreate()
n_exec = spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size() - 1
df = spark.read.parquet("hdfs://master:9000/user/hadoop/accidents_clean")

t0 = time.time()
sev = (F.col("Severity") >= 3).cast("int")
df.groupBy("State", "year").agg(F.count("*"), F.avg("Temperature_F"), F.avg(sev)).collect()
df.groupBy("City", "State").count().orderBy(F.desc("count")).limit(100).collect()
df.groupBy("hour", "weather_group", "season").agg(F.count("*"), F.avg("Visibility_mi"), F.avg(sev)).collect()
df.repartition(32, "State").groupBy("County", "State").agg(F.count("*"), F.avg("Humidity")).count()
print("RESULT", n_exec, round(time.time() - t0, 1))
spark.stop()
