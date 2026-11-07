# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Report
# MAGIC
# MAGIC For one job run (default: the latest):
# MAGIC 1. **Model comparison:** mean ± std over seeds, on the test split. Overlap metrics, tier-1 audit, judge audit.
# MAGIC 2. **Audit calibration:** how well the free tier-1 grounding flag predicts what the judge finds.
# MAGIC 3. **Efficiency and cost:** training minutes per run (actual vs estimate), an optional estimated cost, and actual billed
# MAGIC    cost from `system.billing` once it lands (usually within a few hours).
# MAGIC 4. **Side by side:** one dialogue, every model's note, and the judge's verdict.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "med_notes", "Schema")
dbutils.widgets.text("job_run_id", "", "Job run ID (blank = latest)")
dbutils.widgets.text("job_id", "", "Job ID (for billing lookup)")
dbutils.widgets.text("train_usd_per_hour", "0", "Estimated $ per training hour (0 = skip)")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
spark.sql(f"USE {catalog}.{schema}")

if not spark.catalog.tableExists("runs"):
    dbutils.notebook.exit("No runs table yet.")
job_run_id = dbutils.widgets.get("job_run_id") or spark.sql(
    "SELECT job_run_id FROM runs ORDER BY ingested_at DESC LIMIT 1").first()[0]
has_judge = spark.catalog.tableExists("judge_results")
print(f"Report for job run {job_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Model comparison (test split, mean ± std over seeds)
# MAGIC `reference` is the clinician's note run through the same audits: the ceiling, and the judge's noise floor.

# COMMAND ----------

judge_cte = f"""
  judged AS (
    SELECT experiment, avg(CAST(faithful AS INT)) AS faithful_rate, avg(CAST(hallucinated AS INT)) AS hallucination_rate,
           avg(n_major_errors) AS major_errors_per_note, avg(faithfulness_score) AS judge_score
    FROM judge_results WHERE error IS NULL AND (job_run_id = '{job_run_id}' OR experiment = 'reference')
    GROUP BY experiment)""" if has_judge else """
  judged AS (SELECT CAST(NULL AS STRING) AS experiment, CAST(NULL AS DOUBLE) AS faithful_rate,
             CAST(NULL AS DOUBLE) AS hallucination_rate, CAST(NULL AS DOUBLE) AS major_errors_per_note,
             CAST(NULL AS DOUBLE) AS judge_score)"""

per_run = spark.sql(f"""
WITH tier1 AS (
    SELECT experiment, family, seed, avg(rougeL) AS rougeL, avg(bertscore_f1) AS bertscore_f1,
           avg(CAST(template_ok AS INT)) AS template_rate, avg(na_agreement) AS na_agreement,
           avg(unsupported_rate) AS unsupported_rate, avg(CAST(grounding_flag AS INT)) AS grounding_flag_rate
    FROM predictions WHERE job_run_id = '{job_run_id}' AND split = 'test'
    GROUP BY experiment, family, seed
  ),
  reference AS (
    SELECT 'reference' AS experiment, 'reference' AS family, 0 AS seed, 1.0 AS rougeL, 1.0 AS bertscore_f1,
           avg(CAST(template_ok AS INT)), avg(na_agreement), avg(unsupported_rate), avg(CAST(grounding_flag AS INT))
    FROM reference_checks WHERE split = 'test'
  ),
  {judge_cte}
SELECT t.*, j.faithful_rate, j.hallucination_rate, j.major_errors_per_note, j.judge_score, r.minutes, r.best_epoch
FROM (SELECT * FROM tier1 UNION ALL SELECT * FROM reference) t
LEFT JOIN judged j USING (experiment)
LEFT JOIN (SELECT experiment, minutes, best_epoch FROM runs WHERE job_run_id = '{job_run_id}') r USING (experiment)
""").toPandas()

metrics = ["rougeL", "bertscore_f1", "template_rate", "na_agreement", "unsupported_rate", "grounding_flag_rate",
           "faithful_rate", "hallucination_rate", "major_errors_per_note", "judge_score", "minutes"]
agg = per_run.groupby("family")[metrics].agg(["mean", "std"])
comparison = agg.xs("mean", axis=1, level=1).round(3)
for m in metrics:
    comparison[m] = [f"{mu:.3f} ± {sd:.3f}" if sd == sd and sd > 0 else (f"{mu:.3f}" if mu == mu else "")
                     for mu, sd in zip(agg[(m, "mean")], agg[(m, "std")])]
comparison.insert(0, "seeds", per_run.groupby("family")["seed"].nunique())
display(comparison.reset_index())

# COMMAND ----------

display(per_run.sort_values(["family", "seed"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Audit calibration: does the free grounding flag agree with the judge?
# MAGIC If precision and recall are high, tier 1 can screen every note in production and the judge only needs
# MAGIC to audit a sample. If they're low, the flag needs work before it can be trusted on its own.

# COMMAND ----------

if has_judge:
    display(spark.sql(f"""
        SELECT count(*) AS judged_notes,
               round(avg(CAST(hallucinated AS INT)), 3) AS judge_hallucination_rate,
               round(avg(CAST(grounding_flag AS INT)), 3) AS tier1_flag_rate,
               round(count_if(grounding_flag AND hallucinated) / nullif(count_if(grounding_flag), 0), 3) AS flag_precision,
               round(count_if(grounding_flag AND hallucinated) / nullif(count_if(hallucinated), 0), 3) AS flag_recall
        FROM judge_results
        WHERE job_run_id = '{job_run_id}' AND experiment != 'reference' AND error IS NULL
    """))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Efficiency and cost

# COMMAND ----------

usd_per_hour = float(dbutils.widgets.get("train_usd_per_hour"))
eff = spark.sql(f"""
    SELECT experiment, status, round(estimated_minutes, 1) AS estimated_minutes, round(minutes, 1) AS actual_minutes,
           round(train_minutes, 1) AS train_minutes, best_epoch, epochs_run, num_params_m
    FROM runs WHERE job_run_id = '{job_run_id}' ORDER BY experiment
""").toPandas()
display(eff)
train_min = eff["actual_minutes"].sum()
print(f"Run time inside the training session: {train_min:.1f} min")
if usd_per_hour > 0:
    print(f"≈ ${train_min / 60 * usd_per_hour:.2f} at ${usd_per_hour}/hour (estimate: excludes startup and package "
          "install; see billed cost below)")

# COMMAND ----------

# SageMaker training: what AWS billed (BillableTimeInSeconds already reflects spot savings).
if spark.catalog.tableExists("training_jobs"):
    display(spark.sql(f"""
        SELECT training_job_name, status, instance_type, spot, round(training_seconds / 60, 1) AS training_minutes,
               round(billable_seconds / 60, 1) AS billable_minutes, spot_savings_pct, estimated_usd
        FROM training_jobs WHERE job_run_id = '{job_run_id}'
    """))

# COMMAND ----------

job_id = dbutils.widgets.get("job_id")
if job_id:
    try:
        spark.sql(f"""
        CREATE OR REPLACE VIEW v_job_run_costs AS
        SELECT u.usage_metadata.job_run_id AS job_run_id, u.billing_origin_product,
               u.product_features.serverless_gpu IS NOT NULL AS is_gpu, u.sku_name,
               round(sum(u.usage_quantity), 3) AS dbus, round(sum(u.usage_quantity * p.pricing.default), 2) AS list_usd
        FROM system.billing.usage u
        JOIN system.billing.list_prices p
          ON u.sku_name = p.sku_name AND u.usage_start_time >= p.price_start_time
         AND (p.price_end_time IS NULL OR u.usage_start_time < p.price_end_time)
        WHERE u.usage_metadata.job_id = '{job_id}'
        GROUP BY 1, 2, 3, 4
        """)
        display(spark.sql(f"SELECT * FROM v_job_run_costs WHERE job_run_id = '{job_run_id}'"))
        print("Empty? Billing data usually lands within a few hours; rerun this cell later.")
    except Exception as e:
        print(f"Billing system tables not readable: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Side by side

# COMMAND ----------

dbutils.widgets.text("example_id", "", "Example ID (blank = random audited example)")
example_id = dbutils.widgets.get("example_id") or spark.sql(
    "SELECT example_id FROM audit_sample ORDER BY rand() LIMIT 1").first()[0]
src = spark.sql(f"SELECT section_header, dialogue, section_text FROM silver_dialog_notes WHERE example_id = {int(example_id)}").first()
print(f"example_id {example_id} · {src.section_header}\n\nDIALOGUE\n{src.dialogue}\n\nREFERENCE\n{src.section_text}")

judge_join = (f"LEFT JOIN judge_results j ON j.job_run_id = p.job_run_id AND j.experiment = p.experiment "
              f"AND j.example_id = p.example_id") if has_judge else ""
judge_cols = ", j.faithful, j.faithfulness_score, j.verdict_json" if has_judge else ""
display(spark.sql(f"""
    SELECT p.experiment, p.prediction, p.template_ok, p.unsupported_terms {judge_cols}
    FROM predictions p {judge_join}
    WHERE p.job_run_id = '{job_run_id}' AND p.split = 'test' AND p.example_id = {int(example_id)}
    ORDER BY p.experiment
"""))
