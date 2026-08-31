"""
===============================================================================
DATA QUALITY CHECKS: Gold Layer Validation (Star Schema Integrity)
===============================================================================
Script Purpose:
    - Asserts uniqueness of surrogate keys across Dimension tables.
    - Validates referential integrity between Fact and Dimension tables.
    - Verifies Star Schema relationships for Analytics and ML readiness.
===============================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Car Big Data - Gold Quality Checks") \
    .getOrCreate()

print("============================================================")
print("Running Gold Layer Quality Checks")
print("============================================================")


def assert_zero_records(df, check_name: str):
    """Utility function to validate that no faulty records exist."""
    error_count = df.count()
    if error_count == 0:
        print(f"[PASS] {check_name}")
    else:
        print(f"[FAIL] {check_name} -> Found {error_count} faulty record(s)!")
        df.show(5, truncate=False)


# ====================================================================
# 1. Checking 'gold.dim_customers'
# ====================================================================
print("\n--- Testing: gold.dim_customers ---")
df_dim_cust = spark.table("gold.dim_customers")

# Check 1.1: Uniqueness & Non-null of Customer Key
dup_cust_key = (
    df_dim_cust.groupBy("customer_key")
    .agg(count("*").alias("duplicate_count"))
    .filter((col("duplicate_count") > 1) | col("customer_key").isNull())
)
assert_zero_records(dup_cust_key, "dim_customers: Surrogate Key Uniqueness (customer_key)")


# ====================================================================
# 2. Checking 'gold.dim_products'
# ====================================================================
print("\n--- Testing: gold.dim_products ---")
df_dim_prd = spark.table("gold.dim_products")

# Check 2.1: Uniqueness & Non-null of Product Key
dup_prd_key = (
    df_dim_prd.groupBy("product_key")
    .agg(count("*").alias("duplicate_count"))
    .filter((col("duplicate_count") > 1) | col("product_key").isNull())
)
assert_zero_records(dup_prd_key, "dim_products: Surrogate Key Uniqueness (product_key)")


# ====================================================================
# 3. Checking 'gold.fact_sales' Referential Integrity
# ====================================================================
print("\n--- Testing: gold.fact_sales (Referential Integrity) ---")
df_fact_sales = spark.table("gold.fact_sales")

# Check 3.1: Data Model Connectivity (Orphan Keys in Fact)
orphan_records = (
    df_fact_sales.alias("f")
    .join(df_dim_cust.alias("c"), col("c.customer_key") == col("f.customer_key"), how="left")
    .join(df_dim_prd.alias("p"), col("p.product_key") == col("f.product_key"), how="left")
    .filter(col("c.customer_key").isNull() | col("p.product_key").isNull())
)
assert_zero_records(orphan_records, "fact_sales: Foreign Keys Integrity (No orphan sales records)")


print("\n============================================================")
print("Gold Layer Quality Checks Completed!")
print("============================================================")