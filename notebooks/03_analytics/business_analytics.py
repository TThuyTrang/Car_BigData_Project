"""
===============================================================================
ANALYTICS: Business Intelligence & Executive KPIs Calculation
===============================================================================
Purpose:
    - Reads Star Schema tables from Gold Layer.
    - Aggregates Core KPIs, Product Category Performance, and Sales Trends.
    - Generates customer-level RFM metrics for BI Dashboard and ML workloads.
    - Saves analytical datamarts into Gold layer for Streamlit consumption.
===============================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    countDistinct,
    date_format,
    datediff,
    lit,
    max as spark_max,
    round as spark_round,
    sum as spark_sum,
)

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Car Big Data - Business Analytics") \
    .getOrCreate()

GOLD_SCHEMA = "gold"

print("============================================================")
print("Starting Business Analytics & Aggregations")
print("============================================================")


# =============================================================================
# 1. READ GOLD LAYER TABLES
# =============================================================================
fact_sales = spark.table(f"{GOLD_SCHEMA}.fact_sales")
dim_customers = spark.table(f"{GOLD_SCHEMA}.dim_customers")
dim_products = spark.table(f"{GOLD_SCHEMA}.dim_products")


# =============================================================================
# 2. EXECUTIVE KPIS (Overall Business Performance)
# =============================================================================
print("\n>> Computing Executive KPIs...")

overall_kpis = fact_sales.agg(
    spark_round(
        spark_sum("sales_amount"), 2
    ).alias("total_revenue"),

    countDistinct(
        "order_number"
    ).alias("total_orders"),

    spark_sum(
        "quantity"
    ).alias("total_units_sold"),

    countDistinct(
        "customer_key"
    ).alias("total_active_customers"),

    spark_round(
        spark_sum("sales_amount") /
        countDistinct("order_number"),
        2
    ).alias("average_order_value")
)

print("Executive KPIs Summary:")
overall_kpis.show()


# =============================================================================
# 3. MONTHLY SALES TREND (Time-Series Aggregation)
# =============================================================================
print("\n>> Computing Monthly Sales Trends...")

monthly_sales_trend = (
    fact_sales
    .filter(col("order_date").isNotNull())
    .withColumn(
        "order_year_month",
        date_format(col("order_date"), "yyyy-MM")
    )
    .groupBy("order_year_month")
    .agg(
        spark_round(
            spark_sum("sales_amount"), 2
        ).alias("monthly_revenue"),

        countDistinct(
            "order_number"
        ).alias("order_count"),

        spark_sum(
            "quantity"
        ).alias("units_sold")
    )
    .orderBy("order_year_month")
)

(
    monthly_sales_trend.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        f"{GOLD_SCHEMA}.mart_monthly_sales_trend"
    )
)

print(
    f"Status: SUCCESS | "
    f"Saved gold.mart_monthly_sales_trend "
    f"({monthly_sales_trend.count()} months)"
)


# =============================================================================
# 4. PRODUCT CATEGORY & LINE PERFORMANCE
# =============================================================================
print("\n>> Computing Product Line & Category Performance...")

category_performance = (
    fact_sales
    .join(
        dim_products,
        on="product_key",
        how="inner"
    )
    .groupBy(
        "category",
        "product_line"
    )
    .agg(
        spark_round(
            spark_sum("sales_amount"), 2
        ).alias("total_revenue"),

        spark_sum(
            "quantity"
        ).alias("total_units_sold"),

        countDistinct(
            "order_number"
        ).alias("order_count"),

        spark_round(
            avg("cost"), 2
        ).alias("avg_product_cost")
    )
    .orderBy(
        col("total_revenue").desc()
    )
)

(
    category_performance.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        f"{GOLD_SCHEMA}.mart_category_performance"
    )
)

print(
    "Status: SUCCESS | "
    "Saved gold.mart_category_performance"
)


# =============================================================================
# 5. CUSTOMER RFM METRICS
# =============================================================================
print("\n>> Computing Customer RFM Metrics for ML & Segmentation...")

# Reference date = latest transaction date in the fact table
max_order_date = (
    fact_sales
    .select(spark_max("order_date"))
    .collect()[0][0]
)

customer_rfm = (
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
        datediff(
            lit(max_order_date),
            spark_max("order_date")
        ).alias("recency_days"),

        countDistinct(
            "order_number"
        ).alias("frequency_orders"),

        spark_round(
            spark_sum("sales_amount"),
            2
        ).alias("monetary_value"),

        spark_sum(
            "quantity"
        ).alias("total_items_bought"),

        spark_round(
            avg("sales_amount"),
            2
        ).alias("avg_order_value")
    )
)

(
    customer_rfm.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        f"{GOLD_SCHEMA}.mart_customer_rfm"
    )
)

print(
    f"Status: SUCCESS | "
    f"Saved gold.mart_customer_rfm "
    f"({customer_rfm.count()} customers)"
)


# =============================================================================
# SUMMARY
# =============================================================================
print("\n============================================================")
print("Analytics Marts Creation Completed Successfully")
print("============================================================")

display(
    spark.sql(
        "SHOW TABLES IN gold LIKE 'mart_*'"
    )
)