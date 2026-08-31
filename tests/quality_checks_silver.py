"""
===============================================================================
DATA QUALITY CHECKS: Silver Layer Validation
===============================================================================
Script Purpose:
    - Runs automated quality audits across all Silver tables.
    - Validates Primary Key integrity (Uniqueness, Non-Null).
    - Checks for leading/trailing whitespaces in string columns.
    - Asserts data range validity, business logic rules, and standardized categories.
===============================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, current_date, length, trim

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Car Big Data - Silver Quality Checks") \
    .getOrCreate()

print("============================================================")
print("Running Silver Layer Quality Checks")
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
# 1. Checking 'silver.crm_cust_info'
# ====================================================================
print("\n--- Testing: silver.crm_cust_info ---")
df_cust = spark.table("silver.crm_cust_info")

# Check 1.1: Primary Key integrity (NULL or Duplicates)
dup_cust = (
    df_cust.groupBy("cst_id")
    .agg(count("*").alias("count"))
    .filter((col("count") > 1) | col("cst_id").isNull())
)
assert_zero_records(dup_cust, "crm_cust_info: Primary Key Integrity (cst_id)")

# Check 1.2: Unwanted Whitespaces in cst_key
space_cust_key = df_cust.filter(col("cst_key") != trim(col("cst_key")))
assert_zero_records(space_cust_key, "crm_cust_info: No Whitespaces in cst_key")

# Check 1.3: Data Standardization (Marital Status & Gender)
print("Standardized Marital Status values:")
df_cust.select("cst_marital_status").distinct().show()

print("Standardized Gender values:")
df_cust.select("cst_gndr").distinct().show()


# ====================================================================
# 2. Checking 'silver.crm_prd_info'
# ====================================================================
print("\n--- Testing: silver.crm_prd_info ---")
df_prd = spark.table("silver.crm_prd_info")

# Check 2.1: Primary Key integrity (prd_id)
dup_prd = (
    df_prd.groupBy("prd_id")
    .agg(count("*").alias("count"))
    .filter((col("count") > 1) | col("prd_id").isNull())
)
assert_zero_records(dup_prd, "crm_prd_info: Primary Key Integrity (prd_id)")

# Check 2.2: Whitespaces in prd_nm
space_prd_nm = df_prd.filter(col("prd_nm") != trim(col("prd_nm")))
assert_zero_records(space_prd_nm, "crm_prd_info: No Whitespaces in prd_nm")

# Check 2.3: NULL or Negative Cost
invalid_cost = df_prd.filter((col("prd_cost") < 0) | col("prd_cost").isNull())
assert_zero_records(invalid_cost, "crm_prd_info: Valid Product Cost (>= 0 and NOT NULL)")

# Check 2.4: Invalid Date Orders (start_date > end_date)
invalid_prd_dates = df_prd.filter(
    col("prd_end_dt").isNotNull() & (col("prd_end_dt") < col("prd_start_dt"))
)
assert_zero_records(invalid_prd_dates, "crm_prd_info: Valid Date Sequences (start_dt <= end_dt)")

# Check 2.5: Product Line Standardization
print("Standardized Product Line values:")
df_prd.select("prd_line").distinct().show()


# ====================================================================
# 3. Checking 'silver.crm_sales_details'
# ====================================================================
print("\n--- Testing: silver.crm_sales_details ---")
df_sales = spark.table("silver.crm_sales_details")

# Check 3.1: Invalid Date Orders (order_dt > ship_dt OR order_dt > due_dt)
invalid_sales_dates = df_sales.filter(
    (col("sls_order_dt") > col("sls_ship_dt")) | 
    (col("sls_order_dt") > col("sls_due_dt"))
)
assert_zero_records(invalid_sales_dates, "crm_sales_details: Valid Chronology (order_dt <= ship/due_dt)")

# Check 3.2: Sales Amount Calculation Consistency
invalid_calc = df_sales.filter(
    (col("sls_sales") != col("sls_quantity") * col("sls_price")) |
    col("sls_sales").isNull() |
    col("sls_quantity").isNull() |
    col("sls_price").isNull() |
    (col("sls_sales") <= 0) |
    (col("sls_quantity") <= 0) |
    (col("sls_price") <= 0)
)
assert_zero_records(invalid_calc, "crm_sales_details: Formula Consistency (sales = quantity * price > 0)")


# ====================================================================
# 4. Checking 'silver.erp_cust_az12'
# ====================================================================
print("\n--- Testing: silver.erp_cust_az12 ---")
df_az = spark.table("silver.erp_cust_az12")

# Check 4.1: Out-of-range Birthdates (before 1924 or in future)
invalid_bdate = df_az.filter(
    col("bdate").isNotNull() & ((col("bdate") < "1924-01-01") | (col("bdate") > current_date()))
)
assert_zero_records(invalid_bdate, "erp_cust_az12: Realistic Birthdates (1924 <= bdate <= Today)")

# Check 4.2: Gender Standardization
print("Standardized ERP Gender values:")
df_az.select("gen").distinct().show()


# ====================================================================
# 5. Checking 'silver.erp_loc_a101'
# ====================================================================
print("\n--- Testing: silver.erp_loc_a101 ---")
df_loc = spark.table("silver.erp_loc_a101")

# Check 5.1: Country Standardization
print("Standardized Country values:")
df_loc.select("cntry").distinct().show()


# ====================================================================
# 6. Checking 'silver.erp_px_cat_g1v2'
# ====================================================================
print("\n--- Testing: silver.erp_px_cat_g1v2 ---")
df_px = spark.table("silver.erp_px_cat_g1v2")

# Check 6.1: Whitespaces in Category fields
space_px = df_px.filter(
    (col("cat") != trim(col("cat"))) |
    (col("subcat") != trim(col("subcat"))) |
    (col("maintenance") != trim(col("maintenance")))
)
assert_zero_records(space_px, "erp_px_cat_g1v2: No Whitespaces in Text Columns")

# Check 6.2: Maintenance Values Standardization
print("Standardized Maintenance values:")
df_px.select("maintenance").distinct().show()

print("\n============================================================")
print("Quality Checks Completed!")
print("============================================================")