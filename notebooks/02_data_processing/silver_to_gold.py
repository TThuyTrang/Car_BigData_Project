"""
===============================================================================
PIPELINE: Silver to Gold Layer Transformation (Star Schema)
===============================================================================
Purpose:
    - Reads clean data from Silver Delta tables.
    - Joins CRM and ERP sources to generate Dimension and Fact tables.
    - Assigns surrogate keys for analytics, reporting, and ML workloads.
    - Saves final tables as Gold Delta tables.
===============================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, coalesce, lit, row_number, when
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Car Big Data - Silver to Gold") \
    .getOrCreate()

GOLD_SCHEMA = "gold"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {GOLD_SCHEMA}")

print("============================================================")
print("Starting Gold Layer Transformation Pipeline")
print("============================================================")


# =============================================================================
# 1. DIMENSION: gold.dim_customers
# =============================================================================
print("\n>> Processing: gold.dim_customers")

ci = spark.table("silver.crm_cust_info")
ca = spark.table("silver.erp_cust_az12")
la = spark.table("silver.erp_loc_a101")

joined_cust = (
    ci.join(ca, ci.cst_key == ca.cid, how="left")
      .join(la, ci.cst_key == la.cid, how="left")
)

window_cust_key = Window.orderBy("cst_id")

gold_dim_customers = (
    joined_cust
    .withColumn("customer_key", row_number().over(window_cust_key))
    .select(
        col("customer_key"),
        col("cst_id").alias("customer_id"),
        col("cst_key").alias("customer_number"),
        col("cst_firstname").alias("first_name"),
        col("cst_lastname").alias("last_name"),
        coalesce(col("cntry"), lit("n/a")).alias("country"),
        col("cst_marital_status").alias("marital_status"),
        when(col("cst_gndr") != "n/a", col("cst_gndr"))
        .otherwise(coalesce(col("gen"), lit("n/a")))
        .alias("gender"),
        col("bdate").alias("birthdate"),
        col("cst_create_date").alias("create_date")
    )
)

(
    gold_dim_customers.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{GOLD_SCHEMA}.dim_customers")
)
print(f"Status: SUCCESS | Loaded {gold_dim_customers.count()} rows into gold.dim_customers")


# =============================================================================
# 2. DIMENSION: gold.dim_products
# =============================================================================
print("\n>> Processing: gold.dim_products")

# Filter only current active products
pn = spark.table("silver.crm_prd_info").filter(col("prd_end_dt").isNull())
pc = spark.table("silver.erp_px_cat_g1v2")

joined_prd = pn.join(pc, pn.cat_id == pc.id, how="left")

window_prd_key = Window.orderBy("prd_start_dt", "prd_key")

gold_dim_products = (
    joined_prd
    .withColumn("product_key", row_number().over(window_prd_key))
    .select(
        col("product_key"),
        col("prd_id").alias("product_id"),
        col("prd_key").alias("product_number"),
        col("prd_nm").alias("product_name"),
        col("cat_id").alias("category_id"),
        col("cat").alias("category"),
        col("subcat").alias("subcategory"),
        col("maintenance"),
        col("prd_cost").alias("cost"),
        col("prd_line").alias("product_line"),
        col("prd_start_dt").alias("start_date")
    )
)

(
    gold_dim_products.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{GOLD_SCHEMA}.dim_products")
)
print(f"Status: SUCCESS | Loaded {gold_dim_products.count()} rows into gold.dim_products")


# =============================================================================
# 3. FACT TABLE: gold.fact_sales
# =============================================================================
print("\n>> Processing: gold.fact_sales")

sd = spark.table("silver.crm_sales_details")
pr = spark.table("gold.dim_products")
cu = spark.table("gold.dim_customers")

gold_fact_sales = (
    sd.join(pr, sd.sls_prd_key == pr.product_number, how="left")
      .join(cu, sd.sls_cust_id == cu.customer_id, how="left")
      .select(
          col("sls_ord_num").alias("order_number"),
          col("product_key"),
          col("customer_key"),
          col("sls_order_dt").alias("order_date"),
          col("sls_ship_dt").alias("shipping_date"),
          col("sls_due_dt").alias("due_date"),
          col("sls_sales").alias("sales_amount"),
          col("sls_quantity").alias("quantity"),
          col("sls_price").alias("price")
      )
)

(
    gold_fact_sales.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{GOLD_SCHEMA}.fact_sales")
)
print(f"Status: SUCCESS | Loaded {gold_fact_sales.count()} rows into gold.fact_sales")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n============================================================")
print("Gold Layer Star Schema Built Successfully")
print("============================================================")
display(spark.sql(f"SHOW TABLES IN {GOLD_SCHEMA}"))