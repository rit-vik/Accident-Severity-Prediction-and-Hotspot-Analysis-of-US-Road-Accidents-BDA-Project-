import os, time
import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

SAMPLE_FRACTION = 0.13   # about 1M rows; lower it (e.g. 0.07) if memory errors occur
SEED = 42

spark = SparkSession.builder.appName("accidents-ml").getOrCreate()
os.makedirs("results", exist_ok=True)

num_cols = ["Start_Lat", "Start_Lng", "Temperature_F", "Humidity", "Pressure_in",
            "Visibility_mi", "Wind_Speed_mph", "Precipitation_in",
            "hour", "day_of_week", "month", "is_weekend",
            "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
            "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"]
cat_cols = ["State", "weather_group", "season", "Sunrise_Sunset", "Wind_Direction"]

df = spark.read.parquet("hdfs://master:9000/user/hadoop/accidents_clean")
df = df.select(["Severity"] + num_cols + cat_cols).withColumn("label", F.col("Severity") - 1.0)
sample = df.sampleBy("Severity", {1: SAMPLE_FRACTION, 2: SAMPLE_FRACTION,
                                  3: SAMPLE_FRACTION, 4: SAMPLE_FRACTION}, seed=SEED)
sample = sample.persist(StorageLevel.MEMORY_AND_DISK)
print("Sample rows:", sample.count())

train, test = sample.randomSplit([0.8, 0.2], seed=SEED)

# class weights (inverse frequency) to handle the imbalance
counts = {r["label"]: r["count"] for r in train.groupBy("label").count().collect()}
total = sum(counts.values())
weights = {k: total / (len(counts) * v) for k, v in counts.items()}
print("Class weights:", weights)
w = F.lit(1.0)
for k, v in weights.items():
    w = F.when(F.col("label") == k, v).otherwise(w)
train = train.withColumn("w", w)

indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in cat_cols]
encoder = OneHotEncoder(inputCols=[c + "_idx" for c in cat_cols],
                        outputCols=[c + "_ohe" for c in cat_cols], handleInvalid="keep")
assembler = VectorAssembler(inputCols=num_cols + [c + "_ohe" for c in cat_cols], outputCol="features")
prep_model = Pipeline(stages=indexers + [encoder, assembler]).fit(train)

train_f = prep_model.transform(train).select("label", "w", "features").persist(StorageLevel.MEMORY_AND_DISK)
test_f = prep_model.transform(test).select("label", "features").persist(StorageLevel.MEMORY_AND_DISK)
print("Train rows:", train_f.count(), " Test rows:", test_f.count())

attrs = train_f.schema["features"].metadata["ml_attr"]["attrs"]
names = {a["idx"]: a["name"] for group in attrs.values() for a in group}
feat_names = [names[i] for i in sorted(names)]

def evaluate(name, pred, secs):
    out = {"model": name, "train_seconds": round(secs, 1)}
    for m in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
        out[m] = round(MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                          metricName=m).evaluate(pred), 4)
    cm = (pred.groupBy("label", "prediction").count().toPandas()
              .pivot(index="label", columns="prediction", values="count").fillna(0).astype(int))
    cm.index = ["Sev1", "Sev2", "Sev3", "Sev4"][:len(cm.index)]
    cm.to_csv(f"results/confusion_{name}.csv")
    print("\n==", name, out)
    print(cm.to_string())
    recall = {f"recall_sev{int(i)+1}": round(float(cm.iloc[i, i] / cm.iloc[i].sum()), 3)
              for i in range(len(cm)) if i < cm.shape[1]}
    print("Per-class recall:", recall)
    return {**out, **recall}

# majority-class baseline
maj = test_f.groupBy("label").count().orderBy(F.desc("count")).first()
base_acc = round(maj["count"] / test_f.count(), 4)
print("\nBaseline (always predict the majority class) accuracy:", base_acc)
rows = [{"model": "baseline_majority", "accuracy": base_acc}]

t = time.time()
lr = LogisticRegression(featuresCol="features", labelCol="label", weightCol="w",
                        maxIter=30, regParam=0.01, family="multinomial").fit(train_f)
rows.append(evaluate("logistic_regression", lr.transform(test_f), time.time() - t))

t = time.time()
rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="w",
                            numTrees=30, maxDepth=8, seed=SEED).fit(train_f)
rows.append(evaluate("random_forest", rf.transform(test_f), time.time() - t))

imp = sorted(zip(feat_names, rf.featureImportances.toArray()), key=lambda x: -x[1])[:15]
pd.DataFrame(imp, columns=["feature", "importance"]).to_csv("results/rf_feature_importance.csv", index=False)
print("\nTop RF features:")
for n, v in imp:
    print(f"  {n:35s} {v:.4f}")

pd.DataFrame(rows).to_csv("results/ml_metrics.csv", index=False)
spark.stop()
