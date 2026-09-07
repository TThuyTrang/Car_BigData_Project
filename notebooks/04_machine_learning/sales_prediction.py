"""
===============================================================================
MACHINE LEARNING: Product-Level Next Month Sales Prediction
Random Forest Regressor
===============================================================================

Purpose:
- Read Gold Fact Sales and Product Dimension.
- Aggregate sales by Product and Month.
- Create historical sales features.
- Predict next month's sales for each product.
- Train Random Forest Regressor.
- Evaluate using R², MAE, RMSE.
- Extract Feature Importance.
- Save predictions and model metrics into Gold Delta tables.

Target:
- next_month_sales

Important:
- Does NOT predict sales_amount directly from price × quantity.
- Uses historical information to predict the following month.
===============================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
  col,
  sum as spark_sum,
  avg,
  countDistinct,
  lag,
  date_format,
  to_date,
  current_timestamp,
  round as spark_round
)

from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

import pandas as pd

# =============================================================================
# 0. SPARK SESSION
# =============================================================================

spark = SparkSession.builder \
  .appName("Car Big Data - Product Next Month Sales Prediction") \
  .getOrCreate()

GOLD_SCHEMA = "gold"

print("============================================================")
print("Starting Product-Level Sales Prediction")
print("Random Forest Regressor")
print("============================================================")

# =============================================================================
# 1. READ GOLD DATA
# =============================================================================

print("\n>> 1. Reading Gold Star Schema Data...")

fact_sales = spark.table(f"{GOLD_SCHEMA}.fact_sales")
dim_products = spark.table(f"{GOLD_SCHEMA}.dim_products")

# Join Fact + Product Dimension
raw_dataset = (
  fact_sales
  .join(
    dim_products,
    on="product_key",
    how="inner"
  )
  .filter(
    col("order_date").isNotNull() &
    col("sales_amount").isNotNull() &
    (col("sales_amount") > 0) &
    col("quantity").isNotNull() &
    (col("quantity") > 0) &
    col("price").isNotNull() &
    (col("price") > 0) &
    col("cost").isNotNull()
  )
  .select(
    "order_number",
    "product_key",
    "order_date",
    "quantity",
    "price",
    "sales_amount",
    "cost",
    "category",
    "product_line"
  )
)

total_records = raw_dataset.count()

print(
  f"Status: Loaded {total_records} valid sales records."
)

raw_dataset.show(5, truncate=False)

# =============================================================================
# 2. AGGREGATE BY PRODUCT AND MONTH
# =============================================================================

print("\n>> 2. Aggregating Sales by Product and Month...")

monthly_product_sales = (
  raw_dataset
  .withColumn(
    "month",
    date_format(col("order_date"), "yyyy-MM")
  )
  .groupBy(
    "product_key",
    "month",
    "category",
    "product_line"
  )
  .agg(
    spark_sum("sales_amount").alias("monthly_sales"),

    spark_sum("quantity").alias(
      "monthly_quantity"
    ),

    countDistinct("order_number").alias(
      "order_count"
    ),

    avg("price").alias(
      "avg_price"
    ),

    avg("cost").alias(
      "avg_cost"
    )
  )
  .withColumn(
    "month_date",
    to_date(col("month"), "yyyy-MM")
  )
)

print("\nProduct-Month Dataset:")

monthly_product_sales.show(
  20,
  truncate=False
)

# =============================================================================
# 3. CREATE HISTORICAL FEATURES
# =============================================================================

print("\n>> 3. Creating Historical Features...")

# Window riêng cho từng sản phẩm
product_window = (
  Window
  .partitionBy("product_key")
  .orderBy("month_date")
)

ml_dataset = (
  monthly_product_sales

  # Doanh số tháng trước
  .withColumn(
    "previous_month_sales",
    lag("monthly_sales", 1).over(product_window)
  )

  # Số lượng bán tháng trước
  .withColumn(
    "previous_month_quantity",
    lag("monthly_quantity", 1).over(product_window)
  )

  # Số đơn tháng trước
  .withColumn(
    "previous_month_orders",
    lag("order_count", 1).over(product_window)
  )

  # Giá trung bình tháng trước
  .withColumn(
    "previous_month_avg_price",
    lag("avg_price", 1).over(product_window)
  )

  # Chi phí trung bình tháng trước
  .withColumn(
    "previous_month_avg_cost",
    lag("avg_cost", 1).over(product_window)
  )

  # TARGET:
  # Doanh số của tháng tiếp theo
  .withColumn(
    "next_month_sales",
    lag("monthly_sales", -1).over(product_window)
  )
)

# =============================================================================
# 4. REMOVE INVALID TRAINING RECORDS
# =============================================================================

print("\n>> 4. Preparing ML Dataset...")

ml_dataset = (
  ml_dataset
  .filter(
    col("previous_month_sales").isNotNull() &
    col("previous_month_quantity").isNotNull() &
    col("previous_month_orders").isNotNull() &
    col("previous_month_avg_price").isNotNull() &
    col("previous_month_avg_cost").isNotNull() &
    col("next_month_sales").isNotNull()
  )
)

ml_dataset = ml_dataset.orderBy(
  "month_date",
  "product_key"
)

total_ml_records = ml_dataset.count()

print(
  f"Status: Prepared {total_ml_records} ML records."
)

print("\nSample ML Dataset:")

ml_dataset.select(
  "product_key",
  "month",
  "previous_month_sales",
  "previous_month_quantity",
  "previous_month_orders",
  "previous_month_avg_price",
  "previous_month_avg_cost",
  "next_month_sales"
).show(
  20,
  truncate=False
)

# =============================================================================
# 5. FEATURE ASSEMBLY
# =============================================================================

print("\n>> 5. Configuring Feature Vector...")

feature_columns = [
  "previous_month_sales",
  "previous_month_quantity",
  "previous_month_orders",
  "previous_month_avg_price",
  "previous_month_avg_cost"
]

assembler = VectorAssembler(
  inputCols=feature_columns,
  outputCol="features"
)

# =============================================================================
# 6. TIME-BASED TRAIN / TEST SPLIT
# =============================================================================

print("\n>> 6. Splitting Dataset by Time...")

# Lấy các tháng theo thứ tự
months = (
  ml_dataset
  .select("month_date")
  .distinct()
  .orderBy("month_date")
  .collect()
)

month_values = [
  row["month_date"]
  for row in months
]

total_months_ml = len(month_values)

train_month_count = int(
  total_months_ml * 0.8
)

train_months = month_values[
  :train_month_count
]

test_months = month_values[
  train_month_count:
]

train_data = (
  ml_dataset
  .filter(
    col("month_date").isin(train_months)
  )
)

test_data = (
  ml_dataset
  .filter(
    col("month_date").isin(test_months)
  )
)

print(
  f"Total ML months : {total_months_ml}"
)

print(
  f"Train months  : {len(train_months)}"
)

print(
  f"Test months   : {len(test_months)}"
)

print(
  f"Train records  : {train_data.count()}"
)

print(
  f"Test records  : {test_data.count()}"
)

print("\nTraining Period:")

train_data.select(
  "month_date"
).distinct().orderBy(
  "month_date"
).show(
  50,
  truncate=False
)

print("\nTesting Period:")

test_data.select(
  "month_date"
).distinct().orderBy(
  "month_date"
).show(
  50,
  truncate=False
)

# =============================================================================
# 7. RANDOM FOREST REGRESSOR
# =============================================================================

print("\n>> 7. Training Random Forest Regressor...")

rf = RandomForestRegressor(
  featuresCol="features",
  labelCol="next_month_sales",
  predictionCol="prediction",

  numTrees=60,
  maxDepth=8,
  seed=42
)

pipeline = Pipeline(
  stages=[
    assembler,
    rf
  ]
)

# TRAIN MODEL
pipeline_model = pipeline.fit(
  train_data
)

print(
  "Status: Model training completed successfully."
)

# =============================================================================
# 8. PREDICTION
# =============================================================================

print("\n>> 8. Generating Predictions...")

predictions = pipeline_model.transform(
  test_data
)

comparison = (
  predictions
  .withColumn(
    "predicted_sales",
    spark_round(
      col("prediction"),
      2
    )
  )
  .withColumn(
    "actual_sales",
    spark_round(
      col("next_month_sales"),
      2
    )
  )
  .select(
    "product_key",
    "month",
    "month_date",
    "actual_sales",
    "predicted_sales"
  )
  .orderBy(
    "month_date",
    "product_key"
  )
)

print("\nActual vs Predicted:")

comparison.show(
  30,
  truncate=False
)

# =============================================================================
# 9. MODEL EVALUATION
# =============================================================================

print("\n>> 9. Evaluating Model Performance...")

evaluator_r2 = RegressionEvaluator(
  labelCol="next_month_sales",
  predictionCol="prediction",
  metricName="r2"
)

evaluator_mae = RegressionEvaluator(
  labelCol="next_month_sales",
  predictionCol="prediction",
  metricName="mae"
)

evaluator_rmse = RegressionEvaluator(
  labelCol="next_month_sales",
  predictionCol="prediction",
  metricName="rmse"
)

r2_score = evaluator_r2.evaluate(
  predictions
)

mae_score = evaluator_mae.evaluate(
  predictions
)

rmse_score = evaluator_rmse.evaluate(
  predictions
)

print("\n==================================================")
print("    MODEL EVALUATION METRICS")
print("==================================================")

print(
  f"R-squared (R²) : {r2_score:.4f}"
)

print(
  f"MAE      : ${mae_score:,.2f}"
)

print(
  f"RMSE      : ${rmse_score:,.2f}"
)

print("==================================================")

# =============================================================================
# 10. FEATURE IMPORTANCE
# =============================================================================

print("\n>> 10. Extracting Feature Importance...")

rf_model = pipeline_model.stages[-1]

importances = rf_model.featureImportances

importance_list = []

for idx, importance in enumerate(
  importances
):

  importance_list.append(
    {
      "Feature": feature_columns[idx],
      "Importance": round(
        float(importance),
        4
      )
    }
  )

df_importance = (
  pd.DataFrame(
    importance_list
  )
  .sort_values(
    "Importance",
    ascending=False
  )
)

print(
  "\nFeature Importance Ranking:"
)

print(
  df_importance.to_string(
    index=False
  )
)

# =============================================================================
# 11. SAVE PREDICTIONS TO DELTA
# =============================================================================

print(
  "\n>> 11. Saving Predictions..."
)

output_predictions_table = (
  f"{GOLD_SCHEMA}.mart_sales_predictions"
)

(
  comparison
  .withColumn(
    "dwh_create_date",
    current_timestamp()
  )
  .write
  .format("delta")
  .mode("overwrite")
  .option(
    "overwriteSchema",
    "true"
  )
  .saveAsTable(
    output_predictions_table
  )
)

print(
  f"Status: SUCCESS | Saved predictions to "
  f"{output_predictions_table}"
)

# =============================================================================
# 12. SAVE MODEL METRICS
# =============================================================================

print(
  "\n>> 12. Saving Model Metrics..."
)

metrics_data = [
  (
    "Random Forest Regressor",
    "Next Month Sales Prediction",
    float(r2_score),
    float(mae_score),
    float(rmse_score),
    60,
    8
  )
]

df_metrics = spark.createDataFrame(
  metrics_data,
  [
    "model_name",
    "prediction_target",
    "r2_score",
    "mae",
    "rmse",
    "num_trees",
    "max_depth"
  ]
)

df_metrics = df_metrics.withColumn(
  "dwh_create_date",
  current_timestamp()
)

output_metrics_table = (
  f"{GOLD_SCHEMA}.mart_sales_model_metrics"
)

(
  df_metrics
  .write
  .format("delta")
  .mode("overwrite")
  .option(
    "overwriteSchema",
    "true"
  )
  .saveAsTable(
    output_metrics_table
  )
)

print(
  f"Status: SUCCESS | Saved metrics to "
  f"{output_metrics_table}"
)

# =============================================================================
# 13. FINAL MESSAGE
# =============================================================================

print("\n============================================================")
print("Product-Level Next Month Sales Prediction Completed!")
print("============================================================")