from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    DateType,
    TimestampType
)
from datetime import datetime


# ============================================================
# PROJECT: Car Big Data Analytics
# Layer: Bronze
# Purpose:
#     Load raw CRM and ERP CSV files into Bronze Delta Tables
# ============================================================


spark = SparkSession.builder \
    .appName("Car Big Data - Bronze Layer") \
    .getOrCreate()


# ============================================================
# CONFIGURATION
# ============================================================

BRONZE_SCHEMA = "bronze"

# IMPORTANT:
# Replace this with the path that you successfully used
# to read cust_info.csv earlier.

RAW_PATH = "/Workspace/Users/truongthithuytrang140205@gmail.com/Car_BigData_Project/datasets/raw_data"


# ============================================================
# CREATE BRONZE SCHEMA
# ============================================================

spark.sql(f"""
    CREATE SCHEMA IF NOT EXISTS {BRONZE_SCHEMA}
""")


print("===================================")
print("Loading Bronze Layer")
print("===================================")


# ============================================================
# FUNCTION: LOAD CSV → BRONZE DELTA TABLE
# ============================================================

def load_to_bronze(
    source_file,
    table_name,
    schema
):
    start_time = datetime.now()

    full_table_name = f"{BRONZE_SCHEMA}.{table_name}"

    print("-----------------------------------")
    print(f"Source : {source_file}")
    print(f"Target : {full_table_name}")

    try:

        # ----------------------------------------------------
        # Read CSV
        # ----------------------------------------------------

        df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "false")
            .schema(schema)
            .csv(source_file)
        )

        row_count = df.count()

        print(f"Rows   : {row_count}")

        # ----------------------------------------------------
        # Write Delta Table
        # If table already exists, overwrite it
        # ----------------------------------------------------

        (
            df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(full_table_name)
        )

        end_time = datetime.now()

        duration = (
            end_time - start_time
        ).total_seconds()

        print(f"Status : SUCCESS")
        print(f"Time   : {duration:.2f} seconds")

    except Exception as e:

        print(f"Status : FAILED")
        print(f"Error  : {str(e)}")

        raise e


# ============================================================
# CRM SCHEMAS
# ============================================================

crm_cust_info_schema = StructType([
    StructField("cst_id", IntegerType(), True),
    StructField("cst_key", StringType(), True),
    StructField("cst_firstname", StringType(), True),
    StructField("cst_lastname", StringType(), True),
    StructField("cst_marital_status", StringType(), True),
    StructField("cst_gndr", StringType(), True),
    StructField("cst_create_date", DateType(), True)
])


crm_prd_info_schema = StructType([
    StructField("prd_id", IntegerType(), True),
    StructField("prd_key", StringType(), True),
    StructField("prd_nm", StringType(), True),
    StructField("prd_cost", IntegerType(), True),
    StructField("prd_line", StringType(), True),
    StructField("prd_start_dt", TimestampType(), True),
    StructField("prd_end_dt", TimestampType(), True)
])


crm_sales_details_schema = StructType([
    StructField("sls_ord_num", StringType(), True),
    StructField("sls_prd_key", StringType(), True),
    StructField("sls_cust_id", IntegerType(), True),
    StructField("sls_order_dt", IntegerType(), True),
    StructField("sls_ship_dt", IntegerType(), True),
    StructField("sls_due_dt", IntegerType(), True),
    StructField("sls_sales", IntegerType(), True),
    StructField("sls_quantity", IntegerType(), True),
    StructField("sls_price", IntegerType(), True)
])


# ============================================================
# ERP SCHEMAS
# ============================================================

erp_loc_a101_schema = StructType([
    StructField("cid", StringType(), True),
    StructField("cntry", StringType(), True)
])


erp_cust_az12_schema = StructType([
    StructField("cid", StringType(), True),
    StructField("bdate", DateType(), True),
    StructField("gen", StringType(), True)
])


erp_px_cat_g1v2_schema = StructType([
    StructField("id", StringType(), True),
    StructField("cat", StringType(), True),
    StructField("subcat", StringType(), True),
    StructField("maintenance", StringType(), True)
])


# ============================================================
# LOAD CRM TABLES
# ============================================================

print()
print("-----------------------------------")
print("Loading CRM Tables")
print("-----------------------------------")


load_to_bronze(
    f"{RAW_PATH}/source_crm/cust_info.csv",
    "crm_cust_info",
    crm_cust_info_schema
)


load_to_bronze(
    f"{RAW_PATH}/source_crm/prd_info.csv",
    "crm_prd_info",
    crm_prd_info_schema
)


load_to_bronze(
    f"{RAW_PATH}/source_crm/sales_details.csv",
    "crm_sales_details",
    crm_sales_details_schema
)


# ============================================================
# LOAD ERP TABLES
# ============================================================

print()
print("-----------------------------------")
print("Loading ERP Tables")
print("-----------------------------------")


load_to_bronze(
    f"{RAW_PATH}/source_erp/LOC_A101.csv",
    "erp_loc_a101",
    erp_loc_a101_schema
)


load_to_bronze(
    f"{RAW_PATH}/source_erp/CUST_AZ12.csv",
    "erp_cust_az12",
    erp_cust_az12_schema
)


load_to_bronze(
    f"{RAW_PATH}/source_erp/PX_CAT_G1V2.csv",
    "erp_px_cat_g1v2",
    erp_px_cat_g1v2_schema
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("===================================")
print("Bronze Layer Loaded Successfully")
print("===================================")

print()
print("Bronze Tables:")
print("-----------------------------------")

tables = spark.sql(
    f"SHOW TABLES IN {BRONZE_SCHEMA}"
)

display(tables)