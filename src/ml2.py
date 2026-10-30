import os, time
import numpy as np, pandas as pd
from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

SAMPLE_FRACTION = 0.20   # about 1.5M rows; use 0.13 if memory errors occur
SEED = 42

spark = SparkSession.builder.appName("accidents-ml-experiments").getOrCreate()
os.makedirs("results", exist_ok=True)

num_cols = ["Start_Lat", "Start_Lng", "Temperature_F", "Humidity", "Pressure_in",
            "Visibility_mi", "Wind_Speed_mph", "Precipitation_in",
            "hour", "day_of_week", "month", "is_weekend",
            "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
            "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"]
cat_cols = ["State", "weather_group", "season", "Sunrise_Sunset", "Wind_Direction"]

df = spark.read.parquet("hdfs://master:9000/user/hadoop/accidents_clean")
df = df.select(["Severity"] + num_cols + cat_cols).withColumn("label", F.col("Severity") - 1.0)
sample = df.sampleBy("Severity", {s: SAMPLE_FRACTION for s in (1, 2, 3, 4)}, seed=SEED)
sample = sample.persist(StorageLevel.MEMORY_AND_DISK)
print("Sample rows:", sample.count())
train, test = sample.randomSplit([0.8, 0.2], seed=SEED)

counts = {r["label"]: r["count"] for r in train.groupBy("label").count().collect()}
total = sum(counts.values())
inv = {k: total / (len(counts) * v) for k, v in counts.items()}
sq = {k: v ** 0.5 for k, v in inv.items()}
def weight_col(wmap):
    c = F.lit(1.0)
    for k, v in wmap.items():
        c = F.when(F.col("label") == k, v).otherwise(c)
    return c
train = train.withColumn("w_inverse", weight_col(inv)).withColumn("w_sqrt", weight_col(sq))

def confusion(pred):
    cm = np.zeros((4, 4))
    for r in pred.groupBy("label", "prediction").count().collect():
        cm[int(r["label"]), int(r["prediction"])] = r["count"]
    return cm

def metrics(cm):
    tp = np.diag(cm); support = cm.sum(axis=1); predicted = cm.sum(axis=0)
    recall = tp / np.maximum(support, 1); precision = tp / np.maximum(predicted, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    out = {"accuracy": tp.sum() / cm.sum(),
           "weighted_f1": (f1 * support).sum() / support.sum(),
           "macro_f1": f1.mean(), "balanced_acc": recall.mean()}
    out.update({f"recall_sev{i+1}": recall[i] for i in range(4)})
    out.update({f"precision_sev{i+1}": precision[i] for i in range(4)})
    return {k: round(float(v), 4) for k, v in out.items()}

rows = []
# majority-class baseline
tc = {r["label"]: r["count"] for r in test.groupBy("label").count().collect()}
cm_b = np.zeros((4, 4))
for k, v in tc.items():
    cm_b[int(k), 1] = v
rows.append({"features": "-", "weights": "-", "model": "baseline_majority", "train_seconds": 0, **metrics(cm_b)})

feature_sets = [("with_month_season", []), ("without_month_season", ["month", "season"])]
for fs_name, drop in feature_sets:
    nc = [c for c in num_cols if c not in drop]
    cc = [c for c in cat_cols if c not in drop]
    idx = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in cc]
    enc = OneHotEncoder(inputCols=[c + "_idx" for c in cc], outputCols=[c + "_ohe" for c in cc], handleInvalid="keep")
    asm = VectorAssembler(inputCols=nc + [c + "_ohe" for c in cc], outputCol="features")
    prep = Pipeline(stages=idx + [enc, asm]).fit(train)
    train_f = prep.transform(train).select("label", "w_inverse", "w_sqrt", "features").persist(StorageLevel.MEMORY_AND_DISK)
    test_f = prep.transform(test).select("label", "features").persist(StorageLevel.MEMORY_AND_DISK)
    print(f"\n#### {fs_name}: train={train_f.count()} test={test_f.count()}")
    attrs = train_f.schema["features"].metadata["ml_attr"]["attrs"]
    nm = {a["idx"]: a["name"] for g in attrs.values() for a in g}
    feat_names = [nm[i] for i in sorted(nm)]

    for w_name in ["inverse", "sqrt"]:
        wc = "w_" + w_name
        makers = {
            "logistic_regression": lambda: LogisticRegression(featuresCol="features", labelCol="label", weightCol=wc,
                                                              maxIter=30, regParam=0.01, family="multinomial"),
            "random_forest": lambda: RandomForestClassifier(featuresCol="features", labelCol="label", weightCol=wc,
                                                            numTrees=30, maxDepth=8, seed=SEED),
        }
        for m_name, make in makers.items():
            t = time.time()
            model = make().fit(train_f)
            secs = round(time.time() - t, 1)
            cm = confusion(model.transform(test_f))
            pd.DataFrame(cm.astype(int), index=["Sev1", "Sev2", "Sev3", "Sev4"],
                         columns=["P1", "P2", "P3", "P4"]).to_csv(f"results/cm_{m_name}_{fs_name}_{w_name}.csv")
            res = {"features": fs_name, "weights": w_name, "model": m_name, "train_seconds": secs, **metrics(cm)}
            rows.append(res)
            print(res)
            if m_name == "random_forest":
                imp = sorted(zip(feat_names, model.featureImportances.toArray()), key=lambda x: -x[1])[:15]
                pd.DataFrame(imp, columns=["feature", "importance"]).to_csv(
                    f"results/rf_importance_{fs_name}_{w_name}.csv", index=False)
    train_f.unpersist(); test_f.unpersist()

out = pd.DataFrame(rows)
out.to_csv("results/ml_experiments.csv", index=False)
cols = ["features", "weights", "model", "accuracy", "weighted_f1", "macro_f1", "balanced_acc",
        "recall_sev1", "recall_sev2", "recall_sev3", "recall_sev4", "train_seconds"]
print("\n===== SUMMARY =====")
print(out[cols].to_string(index=False))
spark.stop()
