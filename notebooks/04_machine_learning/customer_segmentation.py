"""
===============================================================================
MACHINE LEARNING: Customer Segmentation (RFM Model & K-Means Clustering)
===============================================================================
Purpose:
- Reads/Computes customer RFM metrics directly from Gold Star Schema.
- Preprocesses and standardizes RFM features using PySpark ML StandardScaler.
- Evaluates optimal number of clusters (k) using Elbow Method & Silhouette Score.
- Trains K-Means clustering model.
- Analyzes cluster profiles and assigns business segment labels (VIP, Loyal, At Risk, etc.).
- Saves segmented results into Delta Table 'gold.mart_customer_segments'.
===============================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
  col,
  when,
  lit,
  avg as spark_avg,
  count as spark_count,
  countDistinct,
  datediff,
  max as spark_max,
  sum as spark_sum,
  round as spark_round,
  current_timestamp
)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import pandas as pd

# Initialize Spark Session
spark = SparkSession.builder \
  .appName("Car Big Data - Customer Segmentation ML") \
  .getOrCreate()

GOLD_SCHEMA = "gold"

print("============================================================")
print("Starting Customer Segmentation Pipeline (K-Means)")
print("============================================================")

# =============================================================================
# 1. LOAD OR COMPUTE RFM DATA (SELF-CONTAINED)
# =============================================================================
print("\n>> 1. Loading / Computing Customer RFM Data...")

# Kiểm tra xem bảng mart_customer_rfm đã có sẵn chưa
tables_in_gold = [t.name for t in spark.catalog.listTables(GOLD_SCHEMA)]

if "mart_customer_rfm" in tables_in_gold:
  print("Found existing table: gold.mart_customer_rfm. Reading from table...")
  df_rfm_raw = spark.table(f"{GOLD_SCHEMA}.mart_customer_rfm")
else:
  print("Table 'gold.mart_customer_rfm' not found. Computing RFM directly from Gold Star Schema...")
  fact_sales = spark.table(f"{GOLD_SCHEMA}.fact_sales")
  dim_customers = spark.table(f"{GOLD_SCHEMA}.dim_customers")
  
  # Lấy ngày giao dịch gần nhất toàn hệ thống làm mốc
  max_date_row = fact_sales.select(spark_max("order_date")).collect()[0][0]
  print(f"Reference Latest Date in Fact Sales: {max_date_row}")
  
  # Tính toán RFM chuẩn xác
  df_rfm_raw = (
    fact_sales
    .join(dim_customers, on="customer_key", how="inner")
    .filter(col("order_date").isNotNull())
    .groupBy(
      "customer_key",
      "customer_id",
      "first_name",
      "last_name",
      "country",
      "gender"
    )
    .agg(
      datediff(lit(max_date_row), spark_max("order_date")).alias("recency_days"),
      countDistinct("order_number").alias("frequency_orders"),
      spark_round(spark_sum("sales_amount"), 2).alias("monetary_value"),
      spark_sum("quantity").alias("total_items_bought"),
      spark_round(spark_avg("sales_amount"), 2).alias("avg_order_value")
    )
  )

# Lọc bỏ các dòng không hợp lệ
df_rfm = (
  df_rfm_raw
  .filter(
    col("recency_days").isNotNull() &
    col("frequency_orders").isNotNull() &
    col("monetary_value").isNotNull() &
    (col("monetary_value") > 0)
  )
)

total_customers = df_rfm.count()
print(f"Status: Loaded/Computed {total_customers} valid customer records.")
df_rfm.select("customer_id", "first_name", "last_name", "recency_days", "frequency_orders", "monetary_value").show(5)

# =============================================================================
# 2. FEATURE ENGINEERING & SCALING (StandardScaler)
# =============================================================================
print("\n>> 2. Feature Assembly & Scaling...")

feature_cols = ["recency_days", "frequency_orders", "monetary_value"]

assembler = VectorAssembler(
  inputCols=feature_cols,
  outputCol="features_raw"
)
df_vector = assembler.transform(df_rfm)

scaler = StandardScaler(
  inputCol="features_raw",
  outputCol="features",
  withStd=True,
  withMean=True
)
scaler_model = scaler.fit(df_vector)
df_scaled = scaler_model.transform(df_vector)

print("Status: Feature scaling completed successfully.")

# =============================================================================
# 3. EVALUATING OPTIMAL k (ELBOW METHOD & SILHOUETTE SCORE)
# =============================================================================
print("\n>> 3. Evaluating Optimal Clusters k (Range k=2 to k=6)...")

evaluator = ClusteringEvaluator(
  featuresCol="features",
  predictionCol="prediction",
  metricName="silhouette",
  distanceMeasure="squaredEuclidean"
)

k_results = []
print(f"{'k':<5} | {'Silhouette Score':<20} | {'WSSSE (Cost)':<20}")
print("-" * 50)

for k in range(2, 7):
  kmeans = KMeans(featuresCol="features", predictionCol="prediction", k=k, seed=42)
  model = kmeans.fit(df_scaled)
  preds = model.transform(df_scaled)
  
  silhouette = evaluator.evaluate(preds)
  wssse = model.summary.trainingCost
  
  k_results.append({
    "k": k,
    "silhouette": silhouette,
    "wssse": wssse,
    "model": model
  })
  print(f"{k:<5} | {silhouette:<20.4f} | {wssse:<20.2f}")
# =============================================================================
# 3.1 SAVE MODEL EVALUATION METRICS FOR POWER BI
# =============================================================================

evaluation_df = spark.createDataFrame(
  [
    (
      result["k"],
      float(result["silhouette"]),
      float(result["wssse"])
    )
    for result in k_results
  ],
  ["k", "silhouette_score", "wssse"]
)

# Add selected model information
evaluation_df = evaluation_df.withColumn(
  "selected_model",
  when(col("k") == 4, "Selected").otherwise("Tested")
)

# Save evaluation metrics to Gold Delta Table
evaluation_table = f"{GOLD_SCHEMA}.mart_ml_kmeans_evaluation"

(
  evaluation_df.write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema", "true")
  .saveAsTable(evaluation_table)
)

print(f"\nStatus: SUCCESS | Saved model evaluation metrics into {evaluation_table}")

print("\nK-Means Model Evaluation Results:")
evaluation_df.orderBy("k").show()
OPTIMAL_K = 4
best_result = next((item for item in k_results if item["k"] == OPTIMAL_K), k_results[0])
best_model = best_result["model"]
final_silhouette = best_result["silhouette"]

print(f"\n>> Selected Optimal k = {OPTIMAL_K} with Silhouette Score = {final_silhouette:.4f}")

# =============================================================================
# 4. PREDICTION & CLUSTER PROFILING
# =============================================================================
print("\n>> 4. Training Final Model & Profiling Clusters...")

final_predictions = best_model.transform(df_scaled)

cluster_profile = (
  final_predictions
  .groupBy("prediction")
  .agg(
    spark_count("*").alias("customer_count"),
    spark_round(spark_avg("recency_days"), 1).alias("avg_recency"),
    spark_round(spark_avg("frequency_orders"), 1).alias("avg_frequency"),
    spark_round(spark_avg("monetary_value"), 2).alias("avg_monetary")
  )
  .orderBy("prediction")
)

print("Cluster Profiles Summary:")
cluster_profile.show()

# Phân loại nhóm khách hàng nghiệp vụ
pdf_profile = cluster_profile.toPandas()

def assign_segment_name(row, df):
  sorted_by_monetary = df.sort_values("avg_monetary", ascending=False)["prediction"].tolist()
  sorted_by_recency = df.sort_values("avg_recency", ascending=True)["prediction"].tolist()
  
  c_id = row["prediction"]
  if c_id == sorted_by_monetary[0]:
    return "VIP Customers"
  elif len(sorted_by_monetary) > 1 and c_id == sorted_by_monetary[1]:
    return "Loyal Customers"
  elif c_id == sorted_by_recency[0] and c_id != sorted_by_monetary[0]:
    return "Potential / New Customers"
  else:
    return "At-Risk / Hibernating"

pdf_profile["segment_name"] = pdf_profile.apply(lambda row: assign_segment_name(row, pdf_profile), axis=1)
label_mapping = {row["prediction"]: row["segment_name"] for _, row in pdf_profile.iterrows()}

segment_expr = when(col("prediction") == list(label_mapping.keys())[0], list(label_mapping.values())[0])
for p_id, s_name in list(label_mapping.items())[1:]:
  segment_expr = segment_expr.when(col("prediction") == p_id, s_name)
segment_expr = segment_expr.otherwise("Regular Customers")

df_segmented = (
  final_predictions
  .withColumn("cluster_id", col("prediction"))
  .withColumn("customer_segment", segment_expr)
  .select(
    col("customer_key"),
    col("customer_id"),
    col("first_name"),
    col("last_name"),
    col("country"),
    col("gender"),
    col("recency_days"),
    col("frequency_orders"),
    col("monetary_value"),
    col("total_items_bought"),
    col("avg_order_value"),
    col("cluster_id"),
    col("customer_segment"),
    current_timestamp().alias("dwh_create_date")
  )
)

# =============================================================================
# 5. SAVE RESULTS TO DELTA TABLE
# =============================================================================
print("\n>> 5. Saving Segmented Results to Delta Table...")

output_table_name = f"{GOLD_SCHEMA}.mart_customer_segments"

(
  df_segmented.write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema", "true")
  .saveAsTable(output_table_name)
)

print(f"Status: SUCCESS | Saved {df_segmented.count()} records into {output_table_name}")

segment_distribution = (
  df_segmented
  .groupBy("customer_segment")
  .agg(
    spark_count("*").alias("total_customers"),
    spark_round(spark_avg("recency_days"), 1).alias("avg_recency_days"),
    spark_round(spark_avg("frequency_orders"), 1).alias("avg_orders"),
    spark_round(spark_avg("monetary_value"), 2).alias("avg_spend")
  )
  .orderBy(col("total_customers").desc())
)

print("\nFinal Customer Segments Business Summary:")
segment_distribution.show(truncate=False)

print("============================================================")
print("Customer Segmentation Job Completed Successfully!")
print("============================================================")