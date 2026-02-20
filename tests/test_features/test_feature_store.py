"""Tests for FeatureStore."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_lab.features.feature_store import FeatureStore


class TestFeatureStore:
    def test_save_and_load(self, tmp_path, synthetic_features):
        store = FeatureStore(tmp_path)
        store.save_features(synthetic_features, name="test_features")
        loaded = store.load_features(name="test_features")
        assert len(loaded) == len(synthetic_features)
        assert list(loaded.columns) == list(synthetic_features.columns)

    def test_has_features_true(self, tmp_path, synthetic_features):
        store = FeatureStore(tmp_path)
        store.save_features(synthetic_features, name="test_features")
        assert store.has_features(name="test_features") is True

    def test_has_features_false(self, tmp_path):
        store = FeatureStore(tmp_path)
        assert store.has_features(name="nonexistent") is False

    def test_default_name(self, tmp_path, synthetic_features):
        store = FeatureStore(tmp_path)
        store.save_features(synthetic_features)
        assert store.has_features() is True
        loaded = store.load_features()
        assert len(loaded) == len(synthetic_features)

    def test_load_nonexistent_raises(self, tmp_path):
        store = FeatureStore(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.load_features(name="does_not_exist")
