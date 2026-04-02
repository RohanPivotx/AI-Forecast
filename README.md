# Propensity Forecasting Pipeline

End-to-end monthly net cash-flow forecasting pipeline.  
Local-first (runs on a laptop), GCP-ready (flip one config flag to switch to GCS/BigQuery and upload results).

---

## Repository structure

```
propensity_forecasting/
├── config/
│   └── config.yaml              # all tunable knobs in one place
├── src/
│   ├── constants.py             # GROUP_COLS hierarchy definition
│   ├── etl/
│   │   ├── ingest.py            # load CSV / GCS / BigQuery → monthly net-CF
│   │   └── validate.py          # data quality checks
│   ├── features/
│   │   └── feature_builder.py   # lag/rolling/trend/calendar features + split
│   ├── forecasting/
│   │   ├── method_selector.py   # CV model selection + ensemble builders
│   │   ├── pipeline.py          # ForecastingPipeline class (orchestrator)
│   │   └── models/
│   │       ├── ensemble.py      # EnsembleRegressor / EnsembleClassifier
│   │       └── registry.py      # MODEL_CONFIGS — all base model definitions
│   └── utils/
│       ├── logger.py            # centralised logging
│       ├── metrics.py           # wmape_safe, medape_safe, batch_predict, …
│       └── parallel.py          # ProcessPoolExecutor worker + flush logic
├── scripts/
│   └── run_forecast.py          # CLI entry point (Airflow / Vertex compatible)
├── tests/
│   └── test_method_selector.py  # pytest unit tests
└── requirements.txt
```

---

## Quick start (local)

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r propensity_forecasting/requirements.txt

# 3. Place your data file in the workspace root (or update config.yaml)
#    Default: hitachi_clean_level2_level5_new.csv

# 4. Run the pipeline
cd propensity_forecasting
python scripts/run_forecast.py

# Or pass a custom config:
python scripts/run_forecast.py --config config/config.yaml
```

Outputs land in `Hitachi_global_forecasting_level5_perseries_netcf/run_YYYYMMDD_HHMMSS/`.

---

## Key config options (`config/config.yaml`)

| Section | Key | Default | Description |
|---|---|---|---|
| `data` | `data_file` | `hitachi_clean_level2_level5_new.csv` | Input CSV path |
| `data` | `source_filter` | `"OAT"` | Keep only this source value |
| `dates` | `train_cutoff` | `2025-10-31` | Last date included in training |
| `dates` | `forecast_start` | `2026-02-01` | First forecast date |
| `model` | `forecast_horizons` | `14` | Number of months to forecast |
| `model` | `enable_ensemble` | `true` | Use top-K ensemble or single best model |
| `model` | `ensemble_top_k` | `3` | How many models to combine |
| `performance` | `parallel_workers` | `4` | CPU cores for parallel training |
| `performance` | `batch_flush_threshold` | `200` | Series per RAM-flush cycle |
| `gcp` | `use_gcs` | `false` | Set `true` to enable GCS/BQ I/O |

---

## Connecting to GCP

1. Uncomment the GCP packages in `requirements.txt` and reinstall.
2. In `config.yaml`, set:
   ```yaml
   gcp:
     use_gcs: true
     project_id: my-gcp-project
     bucket_name: my-bucket
     bq_dataset: my_dataset     # optional — omit to load from GCS CSV
     bq_table: transactions      # optional
   ```
3. Authenticate:
   ```bash
   gcloud auth application-default login
   ```
4. Run the pipeline — it will read from GCS/BQ and upload the full run directory back to GCS.

---

## Running tests

```bash
cd propensity_forecasting
pytest tests/ -v
```

---

## Pipeline steps

| Step | Module | Description |
|---|---|---|
| 1 | `etl/ingest.py` | Load raw data, filter sources, aggregate to monthly net-CF, gap-fill |
| 2 | `etl/validate.py` | Column checks, date-range checks, duplicate checks |
| 3 | `features/feature_builder.py` | Label encode, per-group z-score, lag/rolling/trend features |
| 4 | `features/feature_builder.py` | Train / test / history split |
| 5 | `utils/parallel.py` | Parallel per-series ensemble training (ProcessPoolExecutor) |
| 6 | `forecasting/pipeline.py` | Batch-flush: generate forecasts + free RAM every N series |
| 7 | `forecasting/pipeline.py` | Generate forecasts for any remaining series |
| 8 | `forecasting/pipeline.py` | Forecast analysis + hierarchy rollup CSVs |
| 9 | `forecasting/pipeline.py` | Forecast vs actuals comparison reports |
| 10 | `forecasting/pipeline.py` | Test-period h=1 held-out validation |
| 11 | `forecasting/pipeline.py` | Persist config snapshot + optional GCS upload |

---

## Adding a new model

Open `src/forecasting/models/registry.py` and add one entry to `MODEL_CONFIGS`:

```python
"my_model": {
    "regressor":  MyRegressorClass,
    "classifier": MyClassifierClass,
    "reg_params": { ... },
    "clf_params": { ... },
}
```

No other file changes needed — the method selector picks it up automatically.
