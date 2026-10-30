from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("accidents-clean").getOrCreate()
src = "hdfs://master:9000/user/hadoop/accidents_parquet"
dst = "hdfs://master:9000/user/hadoop/accidents_clean"

df = spark.read.parquet(src)
n_raw = df.count()
df = df.dropDuplicates(["ID"])

# 1. Drop columns that are mostly empty, constant or free text
df = df.drop("End_Lat", "End_Lng", "Wind_Chill_F", "Description", "Street",
             "Country", "Airport_Code", "Weather_Timestamp", "Zipcode", "Source")

# 2. Remove physically impossible values (nulls are kept and imputed later)
def ok(col, lo, hi):
    c = F.col(col)
    return c.isNull() | c.between(lo, hi)

df = df.filter(ok("Temperature_F", -60, 130) & ok("Humidity", 0, 100) &
               ok("Pressure_in", 20, 35) & ok("Visibility_mi", 0, 100) &
               ok("Wind_Speed_mph", 0, 200) & ok("Precipitation_in", 0, 20) &
               (F.col("Distance_mi") >= 0))

# 3. Time features
df = (df.withColumn("hour", F.hour("Start_Time"))
        .withColumn("day_of_week", F.dayofweek("Start_Time"))   # 1 = Sunday
        .withColumn("month", F.month("Start_Time"))
        .withColumn("year", F.year("Start_Time"))
        .withColumn("is_weekend", F.col("day_of_week").isin(1, 7).cast("int"))
        .withColumn("season",
            F.when(F.col("month").isin(12, 1, 2), "Winter")
             .when(F.col("month").isin(3, 4, 5), "Spring")
             .when(F.col("month").isin(6, 7, 8), "Summer")
             .otherwise("Autumn"))
        .withColumn("duration_min",
            (F.unix_timestamp("End_Time") - F.unix_timestamp("Start_Time")) / 60))
df = df.filter((F.col("duration_min") > 0) & (F.col("duration_min") <= 1440))

# 4. Impute missing values
num_cols = ["Temperature_F", "Humidity", "Pressure_in", "Visibility_mi", "Wind_Speed_mph"]
meds = df.approxQuantile(num_cols, [0.5], 0.01)
fill = {c: m[0] for c, m in zip(num_cols, meds)}
fill["Precipitation_in"] = 0.0
df = df.fillna(fill)
df = df.fillna({"Weather_Condition": "Unknown", "Wind_Direction": "Unknown",
                "Sunrise_Sunset": "Unknown", "Civil_Twilight": "Unknown",
                "Nautical_Twilight": "Unknown", "Astronomical_Twilight": "Unknown",
                "City": "Unknown", "Timezone": "Unknown"})
df = df.withColumn("Wind_Direction", F.upper(F.col("Wind_Direction")))

# 5. Group the 100+ weather labels into a few categories
wc = F.lower(F.col("Weather_Condition"))
df = df.withColumn("weather_group",
    F.when(wc == "unknown", "Unknown")
     .when(wc.rlike("snow|sleet|ice|wintry|freezing"), "Snow/Ice")
     .when(wc.rlike("rain|drizzle|shower|thunder|storm"), "Rain/Storm")
     .when(wc.rlike("fog|mist|haze|smoke"), "Fog/Haze")
     .when(wc.rlike("cloud|overcast"), "Cloudy")
     .when(wc.rlike("clear|fair"), "Clear")
     .otherwise("Other"))

# 6. Booleans to 0/1 for MLlib
for c in ["Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
          "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"]:
    df = df.withColumn(c, F.col(c).cast("int"))

df.write.mode("overwrite").parquet(dst)

clean = spark.read.parquet(dst)
print("RAW ROWS:", n_raw)
print("CLEAN ROWS:", clean.count())
print("CLEAN COLUMNS:", len(clean.columns))
clean.groupBy("Severity").count().orderBy("Severity").show()
clean.groupBy("weather_group").count().orderBy(F.desc("count")).show()
clean.printSchema()
spark.stop()
