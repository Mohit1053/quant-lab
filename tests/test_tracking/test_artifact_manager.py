"""Tests for artifact manager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_lab.tracking.artifact_manager import ArtifactManager, ArtifactMetadata


class TestArtifactManager:
    def test_save_creates_versioned_dir(self, tmp_path):
        manager = ArtifactManager(tmp_path / "artifacts")

        # Create a dummy model file
        model_file = tmp_path / "model.pt"
        model_file.write_text("fake model")

        meta = manager.save(model_file, name="test_model", model_type="ridge")
        assert meta.version == 1
        assert meta.name == "test_model"
        assert (tmp_path / "artifacts" / "test_model" / "v001" / "model.pt").exists()

    def test_save_increments_version(self, tmp_path):
        manager = ArtifactManager(tmp_path / "artifacts")
        model_file = tmp_path / "model.pt"
        model_file.write_text("fake model")

        m1 = manager.save(model_file, name="model")
        m2 = manager.save(model_file, name="model")
        assert m1.version == 1
        assert m2.version == 2

    def test_save_with_metrics(self, tmp_path):
        manager = ArtifactManager(tmp_path / "artifacts")
        model_file = tmp_path / "model.pt"
        model_file.write_text("fake model")

        meta = manager.save(
            model_file,
            name="model",
            metrics={"sharpe": 1.5, "cagr": 0.12},
        )
        assert meta.metrics["sharpe"] == 1.5

    def test_load_metadata(self, tmp_path):
        manager = ArtifactManager(tmp_path / "artifacts")
        model_file = tmp_path / "model.pt"
        model_file.write_text("fake model")

        manager.save(model_file, name="model", model_type="transformer")

        loaded = manager.load_metadata("model", version=1)
        assert loaded.model_type == "transformer"
        assert loaded.version == 1

    def test_load_latest_metadata(self, tmp_path):
        manager = ArtifactManager(tmp_path / "artifacts")
        model_file = tmp_path / "model.pt"
        model_file.write_text("fake model")

        manager.save(model_file, name="model", notes="v1")
        manager.save(model_file, name="model", notes="v2")

        latest = manager.load_metadata("model")
        assert latest.version == 2
        assert latest.notes == "v2"

    def test_get_model_path(self, tmp_path):
        manager = ArtifactManager(tmp_path / "artifacts")
        model_file = tmp_path / "model.pt"
        model_file.write_text("fake model")

        manager.save(model_file, name="model")
        path = manager.get_model_path("model", version=1)
        assert path.exists()
        assert path.name == "model.pt"

    def test_list_versions(self, tmp_path):
        manager = ArtifactManager(tmp_path / "artifacts")
        model_file = tmp_path / "model.pt"
        model_file.write_text("fake model")

        manager.save(model_file, name="model")
        manager.save(model_file, name="model")

        versions = manager.list_versions("model")
        assert len(versions) == 2

    def test_list_versions_empty(self, tmp_path):
        manager = ArtifactManager(tmp_path / "artifacts")
        assert manager.list_versions("nonexistent") == []

    def test_compare_versions(self, tmp_path):
        manager = ArtifactManager(tmp_path / "artifacts")
        model_file = tmp_path / "model.pt"
        model_file.write_text("fake model")

        manager.save(model_file, name="model", metrics={"sharpe": 1.0})
        manager.save(model_file, name="model", metrics={"sharpe": 1.5})

        comparison = manager.compare_versions("model", 1, 2)
        assert "sharpe" in comparison
        assert comparison["sharpe"]["delta"] == 0.5

    def test_save_nonexistent_source_raises(self, tmp_path):
        manager = ArtifactManager(tmp_path / "artifacts")
        with pytest.raises(FileNotFoundError):
            manager.save(tmp_path / "nonexistent.pt", name="model")

    def test_load_nonexistent_raises(self, tmp_path):
        manager = ArtifactManager(tmp_path / "artifacts")
        with pytest.raises(FileNotFoundError):
            manager.load_metadata("nonexistent")

    def test_metadata_has_timestamp(self, tmp_path):
        manager = ArtifactManager(tmp_path / "artifacts")
        model_file = tmp_path / "model.pt"
        model_file.write_text("fake model")

        meta = manager.save(model_file, name="model")
        assert meta.created_at != ""
