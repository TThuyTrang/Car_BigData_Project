# ===============================================================================
# MACHINE LEARNING: Customer Segmentation (RFM Model & K-Means Clustering)
# ===============================================================================
"""
Purpose:
  - Reads/Computes customer RFM metrics directly from Gold Star Schema.
  - Preprocesses and standardizes RFM features using PySpark ML StandardScaler.
  - Evaluates number of clusters (k) using Elbow Method & Silhouette Score.
  - Trains K-Means clustering model with 3 customer segments.
  - Analyzes cluster profiles and assigns business segment labels:
        1. VIP Customers
        2. Loyal Customers
        3. At-Risk Customers
  - Saves segmented results into Delta Table 'gold.mart_customer_segments'.
  - Saves K-Means evaluation metrics into
    'gold.mart_ml_kmeans_evaluation'.
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

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

# =============================================================================
# INITIALIZE SPARK SESSION
# =============================================================================

spark = SparkSession.builder \
    .appName("Car Big Data - Customer Segmentation ML") \
    .getOrCreate()

GOLD_SCHEMA = "gold"

print("============================================================")
print("Starting Customer Segmentation Pipeline (K-Means)")
print("============================================================")

# =============================================================================
# 1. LOAD OR COMPUTE RFM DATA
# =============================================================================

print("\n>> 1. Loading / Computing Customer RFM Data...")

# Check whether mart_customer_rfm already exists
tables_in_gold = [
    t.name for t in spark.catalog.listTables(GOLD_SCHEMA)
]

if "mart_customer_rfm" in tables_in_gold:

    print(
        "Found existing table: "
        "gold.mart_customer_rfm. Reading from table..."
    )

    df_rfm_raw = spark.table(
        f"{GOLD_SCHEMA}.mart_customer_rfm"
    )

else:

    print(
        "Table 'gold.mart_customer_rfm' not found. "
        "Computing RFM directly from Gold Star Schema..."
    )

    fact_sales = spark.table(
        f"{GOLD_SCHEMA}.fact_sales"
    )

    dim_customers = spark.table(
        f"{GOLD_SCHEMA}.dim_customers"
    )

    # -------------------------------------------------------------------------
    # Get latest transaction date
    # -------------------------------------------------------------------------

    max_date_row = (
        fact_sales
        .select(spark_max("order_date"))
        .collect()[0][0]
    )

    print(
        f"Reference Latest Date in Fact Sales: "
        f"{max_date_row}"
    )

    # -------------------------------------------------------------------------
    # Calculate RFM metrics
    # -------------------------------------------------------------------------

    df_rfm_raw = (
        fact_sales
        .join(
            dim_customers,
            on="customer_key",
            how="inner"
        )
        .filter(
            col("order_date").isNotNull()
        )
        .groupBy(
            "customer_key",
            "customer_id",
            "first_name",
            "last_name",
            "country",
            "gender"
        )
        .agg(

            # Recency:
            # Number of days since the customer's latest purchase
            datediff(
                lit(max_date_row),
                spark_max("order_date")
            ).alias("recency_days"),

            # Frequency:
            # Number of distinct orders
            countDistinct(
                "order_number"
            ).alias("frequency_orders"),

            # Monetary:
            # Total amount spent by the customer
            spark_round(
                spark_sum("sales_amount"),
                2
            ).alias("monetary_value"),

            # Total items purchased
            spark_sum(
                "quantity"
            ).alias("total_items_bought"),

            # Average order value
            spark_round(
                spark_avg("sales_amount"),
                2
            ).alias("avg_order_value")
        )
    )

# =============================================================================
# FILTER INVALID RFM RECORDS
# =============================================================================

df_rfm = (
    df_rfm_raw
    .filter(
        col("recency_days").isNotNull()
        &
        col("frequency_orders").isNotNull()
        &
        col("monetary_value").isNotNull()
        &
        (col("monetary_value") > 0)
    )
)

total_customers = df_rfm.count()

print(
    f"Status: Loaded/Computed "
    f"{total_customers} valid customer records."
)

df_rfm.select(
    "customer_id",
    "first_name",
    "last_name",
    "recency_days",
    "frequency_orders",
    "monetary_value"
).show(5)

# =============================================================================
# 2. FEATURE ENGINEERING & SCALING
# =============================================================================

print("\n>> 2. Feature Assembly & Scaling...")

# RFM features used for K-Means
feature_cols = [
    "recency_days",
    "frequency_orders",
    "monetary_value"
]

# -------------------------------------------------------------------------
# Assemble features into vector
# -------------------------------------------------------------------------

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

df_vector = assembler.transform(df_rfm)

# -------------------------------------------------------------------------
# Standardize features
# -------------------------------------------------------------------------

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

scaler_model = scaler.fit(df_vector)

df_scaled = scaler_model.transform(df_vector)

print(
    "Status: Feature scaling completed successfully."
)

# =============================================================================
# 3. EVALUATE NUMBER OF CLUSTERS
# =============================================================================

print(
    "\n>> 3. Evaluating Optimal Clusters "
    "k (Range k=2 to k=6)..."
)

# Silhouette evaluator
evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

k_results = []

print(
    f"{'k':<5} | "
    f"{'Silhouette Score':<20} | "
    f"{'WSSSE (Cost)':<20}"
)

print("-" * 50)

# Test k from 2 to 6
for k in range(2, 7):

    kmeans = KMeans(
        featuresCol="features",
        predictionCol="prediction",
        k=k,
        seed=42
    )

    model = kmeans.fit(
        df_scaled
    )

    preds = model.transform(
        df_scaled
    )

    # Silhouette Score
    silhouette = evaluator.evaluate(
        preds
    )

    # WSSSE
    wssse = model.summary.trainingCost

    k_results.append({
        "k": k,
        "silhouette": silhouette,
        "wssse": wssse,
        "model": model
    })

    print(
        f"{k:<5} | "
        f"{silhouette:<20.4f} | "
        f"{wssse:<20.2f}"
    )

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
    [
        "k",
        "silhouette_score",
        "wssse"
    ]
)

# -------------------------------------------------------------------------
# IMPORTANT:
# k = 3 is selected because the business requirement is to create
# three customer segments.
# -------------------------------------------------------------------------

evaluation_df = evaluation_df.withColumn(
    "selected_model",
    when(
        col("k") == 3,
        "Selected"
    ).otherwise(
        "Tested"
    )
)

# -------------------------------------------------------------------------
# Save evaluation metrics
# -------------------------------------------------------------------------

evaluation_table = (
    f"{GOLD_SCHEMA}.mart_ml_kmeans_evaluation"
)

(
    evaluation_df.write
    .format("delta")
    .mode("overwrite")
    .option(
        "overwriteSchema",
        "true"
    )
    .saveAsTable(
        evaluation_table
    )
)

print(
    f"\nStatus: SUCCESS | "
    f"Saved model evaluation metrics into "
    f"{evaluation_table}"
)

print(
    "\nK-Means Model Evaluation Results:"
)

evaluation_df.orderBy(
    "k"
).show()

# =============================================================================
# 3.2 SELECT FINAL MODEL
# =============================================================================

# Business requirement:
# 3 customer segments
#
# 1. VIP Customers
# 2. Loyal Customers
# 3. At-Risk Customers

OPTIMAL_K = 3

best_result = next(
    (
        item
        for item in k_results
        if item["k"] == OPTIMAL_K
    ),
    k_results[0]
)

best_model = best_result["model"]

final_silhouette = best_result[
    "silhouette"
]

print(
    f"\n>> Selected Optimal k = "
    f"{OPTIMAL_K} "
    f"with Silhouette Score = "
    f"{final_silhouette:.4f}"
)

# =============================================================================
# 4. FINAL PREDICTION & CLUSTER PROFILING
# =============================================================================

print(
    "\n>> 4. Training Final Model & "
    "Profiling Clusters..."
)

# Apply selected K-Means model
final_predictions = best_model.transform(
    df_scaled
)

# -------------------------------------------------------------------------
# Create cluster profile
# -------------------------------------------------------------------------

cluster_profile = (
    final_predictions
    .groupBy("prediction")
    .agg(

        spark_count("*").alias(
            "customer_count"
        ),

        spark_round(
            spark_avg("recency_days"),
            1
        ).alias(
            "avg_recency"
        ),

        spark_round(
            spark_avg("frequency_orders"),
            1
        ).alias(
            "avg_frequency"
        ),

        spark_round(
            spark_avg("monetary_value"),
            2
        ).alias(
            "avg_monetary"
        )
    )
    .orderBy(
        "prediction"
    )
)

print(
    "Cluster Profiles Summary:"
)

cluster_profile.show()

# =============================================================================
# 4.1 ASSIGN BUSINESS SEGMENT NAMES
# =============================================================================

pdf_profile = cluster_profile.toPandas()

def assign_segment_name(row, df):
    """
    Assign business labels to 3 customer segments.

    VIP Customers:
        Highest monetary value.

    Loyal Customers:
        Highest purchase frequency among
        the remaining clusters.

    At-Risk Customers:
        Remaining cluster with lower
        customer engagement/value.
    """

    # ---------------------------------------------------------------------
    # 1. VIP Customers
    # ---------------------------------------------------------------------
    # The cluster with the highest average monetary value
    # is classified as VIP.

    vip_cluster = (
        df
        .sort_values(
            "avg_monetary",
            ascending=False
        )
        .iloc[0]["prediction"]
    )

    # ---------------------------------------------------------------------
    # 2. Remove VIP cluster
    # ---------------------------------------------------------------------

    remaining = df[
        df["prediction"] != vip_cluster
    ].copy()

    # ---------------------------------------------------------------------
    # 3. Loyal Customers
    # ---------------------------------------------------------------------
    # Among the remaining clusters:
    # - Higher purchase frequency is preferred.
    # - More recent activity is preferred.

    loyal_cluster = (
        remaining
        .sort_values(
            [
                "avg_frequency",
                "avg_recency"
            ],
            ascending=[
                False,
                True
            ]
        )
        .iloc[0]["prediction"]
    )

    # ---------------------------------------------------------------------
    # 4. Assign labels
    # ---------------------------------------------------------------------

    c_id = row["prediction"]

    if c_id == vip_cluster:

        return "VIP Customers"

    elif c_id == loyal_cluster:

        return "Loyal Customers"

    else:

        return "At-Risk Customers"

# Apply business labels
pdf_profile["segment_name"] = (
    pdf_profile.apply(
        lambda row:
            assign_segment_name(
                row,
                pdf_profile
            ),
        axis=1
    )
)

# -------------------------------------------------------------------------
# Create cluster-to-segment mapping
# -------------------------------------------------------------------------

label_mapping = {
    row["prediction"]:
        row["segment_name"]
    for _, row in
    pdf_profile.iterrows()
}

# =============================================================================
# 4.2 CREATE SPARK SEGMENT EXPRESSION
# =============================================================================

segment_expr = None

for p_id, s_name in label_mapping.items():

    if segment_expr is None:

        segment_expr = when(
            col("prediction") == p_id,
            s_name
        )

    else:

        segment_expr = segment_expr.when(
            col("prediction") == p_id,
            s_name
        )

# Safety fallback
segment_expr = segment_expr.otherwise(
    "At-Risk Customers"
)

# =============================================================================
# 4.3 CREATE FINAL SEGMENTED DATASET
# =============================================================================

df_segmented = (
    final_predictions

    .withColumn(
        "cluster_id",
        col("prediction")
    )

    .withColumn(
        "customer_segment",
        segment_expr
    )

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

        current_timestamp().alias(
            "dwh_create_date"
        )
    )
)

# =============================================================================
# 5. SAVE RESULTS TO DELTA TABLE
# =============================================================================

print(
    "\n>> 5. Saving Segmented Results "
    "to Delta Table..."
)

output_table_name = (
    f"{GOLD_SCHEMA}.mart_customer_segments"
)

(
    df_segmented.write
    .format("delta")
    .mode("overwrite")
    .option(
        "overwriteSchema",
        "true"
    )
    .saveAsTable(
        output_table_name
    )
)

print(
    f"Status: SUCCESS | "
    f"Saved {df_segmented.count()} records "
    f"into {output_table_name}"
)

# =============================================================================
# 5.1 FINAL BUSINESS SEGMENT SUMMARY
# =============================================================================

segment_distribution = (
    df_segmented
    .groupBy(
        "customer_segment"
    )
    .agg(

        spark_count("*").alias(
            "total_customers"
        ),

        spark_round(
            spark_avg("recency_days"),
            1
        ).alias(
            "avg_recency_days"
        ),

        spark_round(
            spark_avg("frequency_orders"),
            1
        ).alias(
            "avg_orders"
        ),

        spark_round(
            spark_avg("monetary_value"),
            2
        ).alias(
            "avg_spend"
        )
    )
    .orderBy(
        col("total_customers").desc()
    )
)

print(
    "\nFinal Customer Segments "
    "Business Summary:"
)

segment_distribution.show(
    truncate=False
)

# =============================================================================
# JOB COMPLETED
# =============================================================================

print(
    "============================================================"
)

print(
    "Customer Segmentation Job "
    "Completed Successfully!"
)

print(
    "============================================================"
)

