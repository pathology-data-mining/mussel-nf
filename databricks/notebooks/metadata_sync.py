# Databricks notebook: metadata_sync
# ---------------------------------------------------------------------------
# Parameterized notebook for syncing slide inventory Parquet files into a
# Delta table via MERGE.
#
# One row per (merge_key, model) — the merge_key column (e.g. slide_id or a
# dataset-specific UUID) is the primary identifier.
#
# Expected to be triggered by the dispatcher sync scripts after each batch,
# or run manually via Databricks widgets.
#
# Parameters (Databricks job widgets / task values):
#   volume_folder     UC volume folder containing Parquet files
#                     e.g. /Volumes/your_catalog/your_schema/mussel_dispatcher
#   target_table      Delta table to MERGE INTO
#   merge_key         Column used as the row identifier in the MERGE condition
#                     default: "slide_id"
#   filename_prefix   Only pick parquet files whose name starts with this prefix.
#                     default: "" (any .parquet file)
# ---------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metadata Sync
# MAGIC Incremental MERGE of slide inventory + feature extraction status into Delta.

# COMMAND ----------

import re
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Parameters — read from Databricks widgets when running as a notebook,
# fall back to defaults when running as part of a job task.
# ---------------------------------------------------------------------------

dbutils.widgets.text(  # noqa: F821
    "volume_folder",
    "/Volumes/your_catalog/your_schema/dispatcher",
    "UC volume folder (Parquet files)",
)
dbutils.widgets.text(  # noqa: F821
    "target_table",
    "your_catalog.your_schema.slide_embeddings",
    "Target Delta table",
)
dbutils.widgets.text(  # noqa: F821
    "merge_key",
    "slide_id",
    "Merge key column (e.g. slide_id or file_id)",
)
dbutils.widgets.text(  # noqa: F821
    "filename_prefix",
    "",
    "Parquet filename prefix filter (e.g. tcga_inventory_ or impact_inventory_)",
)

volume_folder    = dbutils.widgets.get("volume_folder")     # noqa: F821
target_table     = dbutils.widgets.get("target_table")      # noqa: F821
merge_key        = dbutils.widgets.get("merge_key")         # noqa: F821
filename_prefix  = dbutils.widgets.get("filename_prefix")   # noqa: F821

print(f"volume_folder   : {volume_folder}")
print(f"target_table    : {target_table}")
print(f"merge_key       : {merge_key}")
print(f"filename_prefix : {filename_prefix!r}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Discover the latest Parquet file in the volume folder.
# Files are named  <prefix><timestamp>.parquet  by the sync scripts,
# so we sort lexicographically and take the last matching one.
# ---------------------------------------------------------------------------

all_files = dbutils.fs.ls(volume_folder)  # noqa: F821
parquet_files = [
    f.path
    for f in all_files
    if f.name.endswith(".parquet") and f.name.startswith(filename_prefix)
]

if not parquet_files:
    raise FileNotFoundError(
        f"No .parquet files matching prefix {filename_prefix!r} found in {volume_folder}"
    )

parquet_files.sort()
latest_parquet = parquet_files[-1]
print(f"Latest Parquet : {latest_parquet}  ({len(parquet_files)} file(s) found)")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Read source Parquet
# ---------------------------------------------------------------------------

source_df = spark.read.parquet(latest_parquet)  # noqa: F821
source_count = source_df.count()
print(f"Source rows    : {source_count:,}")
source_df.printSchema()

# COMMAND ----------

# ---------------------------------------------------------------------------
# Ensure target Delta table exists with correct schema.
# If the table does not exist we create it from the source; subsequent runs
# will MERGE without touching the CREATE TABLE branch.
# ---------------------------------------------------------------------------

create_sql = f"""
CREATE TABLE IF NOT EXISTS {target_table}
USING DELTA
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact'   = 'true'
)
AS SELECT * FROM parquet.`{latest_parquet}`
WHERE 1 = 0
"""

spark.sql(create_sql)  # noqa: F821
print(f"Table ensured  : {target_table}")

# ---------------------------------------------------------------------------
# Schema evolution: add any columns present in the source but missing from
# the target Delta table.
# ---------------------------------------------------------------------------

target_cols = set(spark.table(target_table).columns)  # noqa: F821
for field in source_df.schema:
    if field.name not in target_cols:
        dtype = field.dataType.simpleString()
        spark.sql(f"ALTER TABLE {target_table} ADD COLUMN IF NOT EXISTS `{field.name}` {dtype}")  # noqa: F821
        print(f"Added column   : {field.name} ({dtype})")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Register source as a temp view for the MERGE statement
# ---------------------------------------------------------------------------

source_df.createOrReplaceTempView("_metadata_source")

# COMMAND ----------

# ---------------------------------------------------------------------------
# MERGE INTO target
# Match on composite key (merge_key, model).
# When matched:            update all columns
# When not matched:        insert new row
# When not matched by src: delete (removes rows no longer in the export)
#
# UPDATE SET * / INSERT * break when the target has columns not present in
# the source Parquet (e.g. a column added directly to the table later).
# Build explicit SET and INSERT clauses from the intersection of source and
# target columns so target-only columns are preserved on UPDATE and NULL on
# INSERT.
# ---------------------------------------------------------------------------

source_col_set = set(source_df.columns)
target_col_set = {
    row["col_name"]
    for row in spark.sql(f"DESCRIBE TABLE {target_table}").collect()  # noqa: F821
    if not row["col_name"].startswith("#")
}
common_cols  = sorted(source_col_set & target_col_set)
extra_target = sorted(target_col_set - source_col_set)

if extra_target:
    print(f"Target-only columns (preserved on UPDATE, NULL on INSERT): {extra_target}")

update_set  = ",\n    ".join(f"target.{c} = source.{c}" for c in common_cols)
insert_cols = ", ".join(common_cols + extra_target)
insert_vals = ", ".join([f"source.{c}" for c in common_cols] + ["NULL"] * len(extra_target))

merge_sql = f"""
MERGE INTO {target_table} AS target
USING _metadata_source AS source
ON target.{merge_key} = source.{merge_key}
   AND target.model = source.model
WHEN MATCHED THEN
    UPDATE SET
    {update_set}
WHEN NOT MATCHED THEN
    INSERT ({insert_cols})
    VALUES ({insert_vals})
WHEN NOT MATCHED BY SOURCE THEN
    DELETE
"""

spark.sql(merge_sql)  # noqa: F821
print("MERGE complete")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Post-merge stats
# ---------------------------------------------------------------------------

result = spark.sql(f"SELECT COUNT(*) AS n_rows FROM {target_table}").collect()  # noqa: F821
n_total = result[0]["n_rows"]

status_counts = (
    spark.sql(  # noqa: F821
        f"SELECT status, COUNT(*) AS n FROM {target_table} GROUP BY status ORDER BY status"
    )
    .collect()
)

print(f"Total rows in {target_table}: {n_total:,}")
for row in status_counts:
    print(f"  status={row['status']:<10}  {row['n']:>8,}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Log run metadata to a Delta audit table (if it exists)
# ---------------------------------------------------------------------------

audit_table = target_table.rsplit(".", 1)[0] + ".metadata_sync_audit"

try:
    spark.sql(  # noqa: F821
        f"""
        CREATE TABLE IF NOT EXISTS {audit_table} (
            run_ts         TIMESTAMP,
            source_file    STRING,
            source_rows    LONG,
            target_rows    LONG,
            target_table   STRING
        ) USING DELTA
        """
    )
    spark.sql(  # noqa: F821
        f"""
        INSERT INTO {audit_table}
        VALUES (
            current_timestamp(),
            '{latest_parquet}',
            {source_count},
            {n_total},
            '{target_table}'
        )
        """
    )
    print(f"Audit row written to {audit_table}")
except Exception as exc:
    print(f"Audit write skipped: {exc}")
