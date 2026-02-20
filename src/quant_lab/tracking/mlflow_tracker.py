"""MLflow experiment tracking integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from quant_lab.tracking.base_tracker import BaseTracker

logger = structlog.get_logger(__name__)


class MLflowTracker(BaseTracker):
    """Tracks experiments using MLflow."""

    def __init__(
        self,
        experiment_name: str = "quant_lab",
        tracking_uri: str = "./outputs/mlruns",
    ):
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = None

        logger.info(
            "mlflow_init",
            experiment=experiment_name,
            tracking_uri=tracking_uri,
        )

    def start_run(self, run_name: str | None = None) -> None:
        """Start a new MLflow run."""
        self._run = self._mlflow.start_run(run_name=run_name)
        logger.info("mlflow_run_started", run_id=self._run.info.run_id, name=run_name)

    def end_run(self) -> None:
        """End the current MLflow run."""
        if self._run:
            self._mlflow.end_run()
            logger.info("mlflow_run_ended", run_id=self._run.info.run_id)
            self._run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters. Truncates values > 500 chars (MLflow limit)."""
        for key, value in params.items():
            str_val = str(value)[:500]
            try:
                self._mlflow.log_param(key, str_val)
            except Exception as e:
                logger.warning("mlflow_param_error", key=key, error=str(e))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log numeric metrics."""
        for key, value in metrics.items():
            try:
                self._mlflow.log_metric(key, float(value), step=step)
            except Exception as e:
                logger.warning("mlflow_metric_error", key=key, error=str(e))

    def log_artifact(self, path: str | Path, artifact_name: str | None = None) -> None:
        """Log a file as an artifact."""
        path = str(path)
        try:
            self._mlflow.log_artifact(path)
            logger.info("mlflow_artifact_logged", path=path)
        except Exception as e:
            logger.warning("mlflow_artifact_error", path=path, error=str(e))
