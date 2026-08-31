"""
===============================================================================
PIPELINE: Bronze to Silver Layer Transformation
===============================================================================
Purpose:
    - Reads cleansed-ready data from Bronze Delta tables.
    - Applies business transformations, deduplication, and data type casting.
    - Loads structured results into Silver Delta tables.
===============================================================================
"""

from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    abs as spark_abs,
    coalesce,
    col,
    current_date,
    current_timestamp,
    lead,
    length,
    lit,
    regexp_replace,
    row_number,
    substring,
    to_date,
    trim,
    upper,
    when,
)
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Car Big Data - Bronze to Silver") \
    .getOrCreate()

SILVER_SCHEMA = "silver"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SILVER_SCHEMA}")

print("============================================================")
print("Starting Silver Layer ETL Pipeline")
print("============================================================")


# ============================================================================
# 1. CRM CUSTOMER INFO (silver.crm_cust_info)
# ============================================================================
print("\n>> Processing: silver.crm_cust_info")

# Read bronze table and filter invalid IDs
crm_cust_info = spark.table("bronze.crm_cust_info").filter(col("cst_id").isNotNull())

# Window specification to extract the latest record per customer
window_cust = Window.partitionBy("cst_id").orderBy(col("cst_create_date").desc_nulls_last())

# Cleanse and standardize customer attributes
silver_crm_cust_info = (
    crm_cust_info
    .withColumn("flag_last", row_number().over(window_cust))
    .filter(col("flag_last") == 1)
    .select(
        col("cst_id"),
        trim(col("cst_key")).alias("cst_key"),
        trim(col("cst_firstname")).alias("cst_firstname"),
        trim(col("cst_lastname")).alias("cst_lastname"),
        when(upper(trim(col("cst_marital_status"))) == "S", "Single")
        .when(upper(trim(col("cst_marital_status"))) == "M", "Married")
        .otherwise("n/a")
        .alias("cst_marital_status"),
        when(upper(trim(col("cst_gndr"))) == "F", "Female")
        .when(upper(trim(col("cst_gndr"))) == "M", "Male")
        .otherwise("n/a")
        .alias("cst_gndr"),
        to_date(col("cst_create_date")).alias("cst_create_date"),
        current_timestamp().alias("dwh_create_date")
    )
)

# Overwrite into Silver Delta table
(
    silver_crm_cust_info.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{SILVER_SCHEMA}.crm_cust_info")
)
print(f"Status: SUCCESS | Loaded {silver_crm_cust_info.count()} rows into silver.crm_cust_info")


# ============================================================================
# 2. CRM PRODUCT INFO (silver.crm_prd_info)
# ============================================================================
print("\n>> Processing: silver.crm_prd_info")

crm_prd_info = spark.table("bronze.crm_prd_info")

# Extract category prefix and clean product keys
df_prd_prep = (
    crm_prd_info
    .withColumn("cat_id", regexp_replace(substring(col("prd_key"), 1, 5), "-", "_"))
    .withColumn("prd_key_clean", substring(col("prd_key"), 7, length(col("prd_key"))))
    .withColumn("prd_start_dt_clean", to_date(col("prd_start_dt")))
)

# Window specification to compute effective end date (SCD Type 2)
window_prd = Window.partitionBy("prd_key_clean").orderBy("prd_start_dt_clean")

silver_crm_prd_info = (
    df_prd_prep
    .withColumn("prd_end_dt_lead", lead("prd_start_dt_clean").over(window_prd))
    .select(
        col("prd_id"),
        col("cat_id"),
        col("prd_key_clean").alias("prd_key"),
        trim(col("prd_nm")).alias("prd_nm"),
        coalesce(col("prd_cost"), lit(0)).alias("prd_cost"),
        when(upper(trim(col("prd_line"))) == "M", "Mountain")
        .when(upper(trim(col("prd_line"))) == "R", "Road")
        .when(upper(trim(col("prd_line"))) == "S", "Other Sales")
        .when(upper(trim(col("prd_line"))) == "T", "Touring")
        .otherwise("n/a")
        .alias("prd_line"),
        col("prd_start_dt_clean").alias("prd_start_dt"),
        col("prd_end_dt_lead").alias("prd_end_dt"),
        current_timestamp().alias("dwh_create_date")
    )
)

(
    silver_crm_prd_info.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{SILVER_SCHEMA}.crm_prd_info")
)
print(f"Status: SUCCESS | Loaded {silver_crm_prd_info.count()} rows into silver.crm_prd_info")


# ============================================================================
# 3. CRM SALES DETAILS (silver.crm_sales_details)
# ============================================================================
print("\n>> Processing: silver.crm_sales_details")

crm_sales_details = spark.table("bronze.crm_sales_details")

# Convert integer date keys (YYYYMMDD) into standard DateType
order_dt_str = col("sls_order_dt").cast("string")
ship_dt_str = col("sls_ship_dt").cast("string")
due_dt_str = col("sls_due_dt").cast("string")

order_dt_parsed = when(
    (col("sls_order_dt") == 0) | (length(order_dt_str) != 8), None
).otherwise(to_date(order_dt_str, "yyyyMMdd"))

ship_dt_parsed = when(
    (col("sls_ship_dt") == 0) | (length(ship_dt_str) != 8), None
).otherwise(to_date(ship_dt_str, "yyyyMMdd"))

due_dt_parsed = when(
    (col("sls_due_dt") == 0) | (length(due_dt_str) != 8), None
).otherwise(to_date(due_dt_str, "yyyyMMdd"))

# Recalculate sales amount and unit price if corrupted or missing
calc_sales = when(
    col("sls_sales").isNull()
    | (col("sls_price") <= 0)
    | (col("sls_sales") != col("sls_quantity") * spark_abs(col("sls_price"))),
    col("sls_quantity") * spark_abs(col("sls_price"))
).otherwise(col("sls_sales"))

calc_price = when(
    col("sls_price").isNull() | (col("sls_price") <= 0),
    when(col("sls_quantity") == 0, None).otherwise(col("sls_sales") / col("sls_quantity"))
).otherwise(col("sls_price"))

silver_crm_sales_details = crm_sales_details.select(
    trim(col("sls_ord_num")).alias("sls_ord_num"),
    trim(col("sls_prd_key")).alias("sls_prd_key"),
    col("sls_cust_id"),
    order_dt_parsed.alias("sls_order_dt"),
    ship_dt_parsed.alias("sls_ship_dt"),
    due_dt_parsed.alias("sls_due_dt"),
    calc_sales.cast("int").alias("sls_sales"),
    col("sls_quantity"),
    calc_price.cast("int").alias("sls_price"),
    current_timestamp().alias("dwh_create_date")
)

(
    silver_crm_sales_details.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{SILVER_SCHEMA}.crm_sales_details")
)
print(f"Status: SUCCESS | Loaded {silver_crm_sales_details.count()} rows into silver.crm_sales_details")


# ============================================================================
# 4. ERP CUSTOMER AZ12 (silver.erp_cust_az12)
# ============================================================================
print("\n>> Processing: silver.erp_cust_az12")

erp_cust_az12 = spark.table("bronze.erp_cust_az12")

silver_erp_cust_az12 = erp_cust_az12.select(
    # Strip 'NAS' prefix if present
    when(col("cid").startswith("NAS"), substring(col("cid"), 4, length(col("cid"))))
    .otherwise(col("cid"))
    .alias("cid"),
    # Set future birth dates to NULL
    when(to_date(col("bdate")) > current_date(), None)
    .otherwise(to_date(col("bdate")))
    .alias("bdate"),
    # Standardize gender descriptions
    when(upper(trim(col("gen"))).isin(["F", "FEMALE"]), "Female")
    .when(upper(trim(col("gen"))).isin(["M", "MALE"]), "Male")
    .otherwise("n/a")
    .alias("gen"),
    current_timestamp().alias("dwh_create_date")
)

(
    silver_erp_cust_az12.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{SILVER_SCHEMA}.erp_cust_az12")
)
print(f"Status: SUCCESS | Loaded {silver_erp_cust_az12.count()} rows into silver.erp_cust_az12")


# ============================================================================
# 5. ERP LOCATION (silver.erp_loc_a101)
# ============================================================================
print("\n>> Processing: silver.erp_loc_a101")

erp_loc_a101 = spark.table("bronze.erp_loc_a101")

silver_erp_loc_a101 = erp_loc_a101.select(
    # Remove hyphens from customer ID
    regexp_replace(col("cid"), "-", "").alias("cid"),
    # Standardize country names and handle blanks
    when(trim(col("cntry")) == "DE", "Germany")
    .when(trim(col("cntry")).isin(["US", "USA"]), "United States")
    .when((trim(col("cntry")) == "") | col("cntry").isNull(), "n/a")
    .otherwise(trim(col("cntry")))
    .alias("cntry"),
    current_timestamp().alias("dwh_create_date")
)

(
    silver_erp_loc_a101.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{SILVER_SCHEMA}.erp_loc_a101")
)
print(f"Status: SUCCESS | Loaded {silver_erp_loc_a101.count()} rows into silver.erp_loc_a101")


# ============================================================================
# 6. ERP PRODUCT CATEGORY (silver.erp_px_cat_g1v2)
# ============================================================================
print("\n>> Processing: silver.erp_px_cat_g1v2")

erp_px_cat_g1v2 = spark.table("bronze.erp_px_cat_g1v2")

silver_erp_px_cat_g1v2 = erp_px_cat_g1v2.select(
    trim(col("id")).alias("id"),
    trim(col("cat")).alias("cat"),
    trim(col("subcat")).alias("subcat"),
    trim(col("maintenance")).alias("maintenance"),
    current_timestamp().alias("dwh_create_date")
)

(
    silver_erp_px_cat_g1v2.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{SILVER_SCHEMA}.erp_px_cat_g1v2")
)
print(f"Status: SUCCESS | Loaded {silver_erp_px_cat_g1v2.count()} rows into silver.erp_px_cat_g1v2")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n============================================================")
print("Silver Layer Ingestion Completed Successfully")
print("============================================================")
display(spark.sql(f"SHOW TABLES IN {SILVER_SCHEMA}"))