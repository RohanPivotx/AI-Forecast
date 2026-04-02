"""
Pipeline Entry Point
=====================
Called directly or scheduled by Airflow / Vertex AI Pipelines.

Usage (local):
    python scripts/run_forecast.py
    python scripts/run_forecast.py --config config/config.yaml

Usage (Airflow / Vertex):
    python scripts/run_forecast.py --config gs://bucket/config/config.yaml

Environment variables (override config file values):
    FORECAST_DATA_FILE        → data.data_file
    FORECAST_OUTPUT_ROOT      → output.output_root
    FORECAST_PARALLEL_WORKERS → performance.parallel_workers
    GCP_PROJECT_ID            → gcp.project_id
    GCP_BUCKET_NAME           → gcp.bucket_name
    USE_GCS                   → gcp.use_gcs  (set to "true" to enable)
"""

import argparse
import os
import sys
import warnings

# Suppress known-harmless library warnings that flood the output
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Downcasting object dtype arrays",
    category=FutureWarning,
)

# ── Ensure the package root is on sys.path when invoked as a script ──────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

import yaml  # type: ignore

from src.forecasting.pipeline import ForecastingPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config from local path or GCS (if path starts with gs://)."""
    if config_path.startswith("gs://"):
        try:
            from google.cloud import storage  # type: ignore
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required to load config from GCS. "
                "Install it with: pip install google-cloud-storage"
            )
        parts = config_path[5:].split("/", 1)
        bucket_name, blob_name = parts[0], parts[1]
        client = storage.Client()
        content = client.bucket(bucket_name).blob(blob_name).download_as_text()
        return yaml.safe_load(content)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_env_overrides(cfg: dict) -> dict:
    """
    Apply environment-variable overrides on top of the YAML config.
    This allows CI/CD or Vertex AI jobs to inject runtime values without
    modifying the config file.
    """
    overrides = {
        "FORECAST_DATA_FILE": ("data", "data_file"),
        "FORECAST_OUTPUT_ROOT": ("output", "output_root"),
    }
    for env_key, (section, field) in overrides.items():
        val = os.environ.get(env_key)
        if val:
            cfg.setdefault(section, {})[field] = val
            logger.info(f"Env override: {env_key} → cfg[{section}][{field}] = {val}")

    # Numeric override
    workers_env = os.environ.get("FORECAST_PARALLEL_WORKERS")
    if workers_env:
        cfg.setdefault("performance", {})["parallel_workers"] = int(workers_env)
        logger.info(f"Env override: parallel_workers = {workers_env}")

    # GCP overrides
    gcp_project = os.environ.get("GCP_PROJECT_ID")
    if gcp_project:
        cfg.setdefault("gcp", {})["project_id"] = gcp_project
    gcp_bucket = os.environ.get("GCP_BUCKET_NAME")
    if gcp_bucket:
        cfg.setdefault("gcp", {})["bucket_name"] = gcp_bucket
    use_gcs = os.environ.get("USE_GCS", "").lower()
    if use_gcs in ("true", "1", "yes"):
        cfg.setdefault("gcp", {})["use_gcs"] = True

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the cash-flow forecasting pipeline."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(_PACKAGE_ROOT, "config", "config.yaml"),
        help="Path to config.yaml (local or gs://bucket/path/config.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Loading config from: {args.config}")
    cfg = load_config(args.config)
    cfg = apply_env_overrides(cfg)

    pipeline = ForecastingPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
