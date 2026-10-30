#!/bin/bash
# Runs src/scaling_job.py with 1, 2 and 3 executors (2 runs each) and writes results/scaling.csv
cd "$(dirname "$0")/.."
mkdir -p logs results
echo "requested,actual,run,seconds" > results/scaling.csv
for N in 1 2 3; do
  for RUN in 1 2; do
    OUT=$(spark-submit --conf spark.executor.instances=$N --conf spark.dynamicAllocation.enabled=false \
      --conf spark.scheduler.minRegisteredResourcesRatio=1.0 \
      --conf spark.scheduler.maxRegisteredResourcesWaitingTime=180s \
      src/scaling_job.py 2> logs/scaling_${N}_${RUN}.log | grep '^RESULT')
    echo "$N,$(echo $OUT | awk '{print $2}'),$RUN,$(echo $OUT | awk '{print $3}')" >> results/scaling.csv
  done
done
cat results/scaling.csv
