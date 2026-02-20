"""Tests for tracking module (base tracker and MLflow tracker)."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_lab.tracking.base_tracker import BaseTracker


class ConcreteTracker(BaseTracker):
    """Minimal concrete implementation for testing the abstract base."""

    def __init__(self):
        self.runs = []
        self.params = {}
        self.metrics = {}
        self.artifacts = []
        self._in_run = False

    def start_run(self, run_name=None):
        self.runs.append(run_name)
        self._in_run = True

    def end_run(self):
        self._in_run = False

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            self.metrics[k] = v

    def log_artifact(self, path, artifact_name=None):
        self.artifacts.append(str(path))


class TestBaseTracker:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseTracker()

    def test_concrete_tracker_start_end(self):
        tracker = ConcreteTracker()
        tracker.start_run("test_run")
        assert tracker._in_run
        tracker.end_run()
        assert not tracker._in_run

    def test_log_params(self):
        tracker = ConcreteTracker()
        tracker.log_params({"lr": "0.001", "batch_size": "32"})
        assert tracker.params["lr"] == "0.001"

    def test_log_metrics(self):
        tracker = ConcreteTracker()
        tracker.log_metrics({"loss": 0.5, "accuracy": 0.9})
        assert tracker.metrics["loss"] == 0.5

    def test_log_artifact(self, tmp_path):
        tracker = ConcreteTracker()
        tracker.log_artifact(tmp_path / "model.pt")
        assert len(tracker.artifacts) == 1

    def test_log_config_flattens(self):
        tracker = ConcreteTracker()
        config = {
            "model": {
                "d_model": 128,
                "layers": 4,
            },
            "lr": 0.001,
        }
        tracker.log_config(config)
        assert tracker.params["model.d_model"] == "128"
        assert tracker.params["model.layers"] == "4"
        assert tracker.params["lr"] == "0.001"

    def test_flatten_nested_dict(self):
        result = BaseTracker._flatten_dict({
            "a": {"b": {"c": 1}},
            "d": 2,
        })
        assert result["a.b.c"] == "1"
        assert result["d"] == "2"


class TestMLflowTracker:
    def test_mlflow_tracker_imports(self):
        """Verify MLflowTracker can be imported."""
        from quant_lab.tracking.mlflow_tracker import MLflowTracker

        assert MLflowTracker is not None
