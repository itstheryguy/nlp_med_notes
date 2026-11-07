# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC
# MAGIC Creates the Unity Catalog objects the rest of the project uses:
# MAGIC
# MAGIC | Object | Purpose |
# MAGIC |---|---|
# MAGIC | `<catalog>.<schema>` schema | dataset tables, predictions, and results |
# MAGIC | `<catalog>.<schema>.models` volume | fine-tuned weights: the best seed of each model, one folder per model |
# MAGIC | `<catalog>.<schema>.pipeline` volume | run folders when training runs on Databricks compute (`02_train_evaluate`) |
# MAGIC | `<catalog>.<schema>.sagemaker` external volume | run folders and best weights when training runs on SageMaker. It's the S3 prefix `sagemaker_s3_root`, so SageMaker reads and writes the same files |
# MAGIC
# MAGIC The `catalog` and `schema` widgets default to `workspace.med_notes`. When the notebook runs in the
# MAGIC `med_notes_pipeline` job, the job parameters set them (see `databricks.yml`).

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "med_notes", "Schema")
dbutils.widgets.text("sagemaker_s3_root", "", "S3 URI for the sagemaker volume (blank = skip)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
sagemaker_s3_root = dbutils.widgets.get("sagemaker_s3_root").rstrip("/")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.models")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.pipeline")
if sagemaker_s3_root:
    # Needs a Unity Catalog external location covering this path (aws-ml-infra: ml-training-bucket).
    spark.sql(f"CREATE EXTERNAL VOLUME IF NOT EXISTS {catalog}.{schema}.sagemaker LOCATION '{sagemaker_s3_root}/'")
    display(spark.sql(f"DESCRIBE VOLUME {catalog}.{schema}.sagemaker"))

display(spark.sql(f"SHOW TABLES IN {catalog}.{schema}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Hugging Face token
# MAGIC
# MAGIC The dataset and base models are public, so no token is needed to train. With one, downloads aren't
# MAGIC rate-limited, and later the best model can be pushed to the Hub for the Space. Store it in a secret scope:
# MAGIC
# MAGIC ```bash
# MAGIC databricks secrets create-scope hf
# MAGIC databricks secrets put-secret hf token
# MAGIC ```

# COMMAND ----------

try:
    dbutils.secrets.get("hf", "token")
    print("Found secret 'hf/token'. Notebooks will use it for Hugging Face downloads.")
except Exception:
    print("No 'hf/token' secret. Downloads run unauthenticated, which works but is rate-limited.")
