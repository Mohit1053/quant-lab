"""Artifact manager for checkpoint & config versioning."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ArtifactMetadata:
    """Metadata for a saved artifact."""

    name: str
    version: int
    created_at: str = ""
    model_type: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    git_hash: str = ""
    notes: str = ""


class ArtifactManager:
    """Version-controlled artifact storage for models and configs.

    Tracks model checkpoints with metadata (metrics, config, git hash)
    to enable reproducibility and easy comparison of model versions.

    Directory layout:
        base_dir/
            {name}/
                v001/
                    model.pt (or model.pkl)
                    metadata.json
                v002/
                    ...
                latest -> v002 (symlink or metadata pointer)
    """

    def __init__(self, base_dir: str | Path = "outputs/artifacts"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        source_path: str | Path,
        name: str,
        model_type: str = "",
        metrics: dict[str, float] | None = None,
        config: dict | None = None,
        notes: str = "",
    ) -> ArtifactMetadata:
        """Save a model artifact with versioning.

        Args:
            source_path: Path to model file to archive.
            name: Artifact name (e.g., "transformer", "ridge_baseline").
            model_type: Model type identifier.
            metrics: Performance metrics to store with the artifact.
            config: Training config to store.
            notes: Free-text notes.

        Returns:
            ArtifactMetadata for the saved version.
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        artifact_dir = self.base_dir / name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Determine next version
        version = self._next_version(artifact_dir)
        version_dir = artifact_dir / f"v{version:03d}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy model file
        dest_path = version_dir / source_path.name
        shutil.copy2(source_path, dest_path)

        # Get git hash
        git_hash = self._get_git_hash()

        # Create metadata
        metadata = ArtifactMetadata(
            name=name,
            version=version,
            created_at=datetime.now().isoformat(),
            model_type=model_type,
            metrics=metrics or {},
            config=config or {},
            git_hash=git_hash,
            notes=notes,
        )

        # Save metadata
        meta_path = version_dir / "metadata.json"
        meta_path.write_text(json.dumps(asdict(metadata), indent=2, default=str))

        # Update latest pointer
        latest_path = artifact_dir / "latest.json"
        latest_path.write_text(json.dumps({"version": version, "path": str(version_dir)}))

        logger.info(
            "artifact_saved",
            name=name,
            version=version,
            path=str(version_dir),
            metrics=metrics,
        )

        return metadata

    def load_metadata(self, name: str, version: int | None = None) -> ArtifactMetadata:
        """Load metadata for a specific artifact version.

        Args:
            name: Artifact name.
            version: Specific version (None = latest).

        Returns:
            ArtifactMetadata.
        """
        artifact_dir = self.base_dir / name

        if version is None:
            latest_path = artifact_dir / "latest.json"
            if latest_path.exists():
                latest = json.loads(latest_path.read_text())
                version = latest["version"]
            else:
                raise FileNotFoundError(f"No artifacts found for '{name}'")

        version_dir = artifact_dir / f"v{version:03d}"
        meta_path = version_dir / "metadata.json"

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        data = json.loads(meta_path.read_text())
        return ArtifactMetadata(**data)

    def get_model_path(self, name: str, version: int | None = None) -> Path:
        """Get the path to a saved model file.

        Args:
            name: Artifact name.
            version: Specific version (None = latest).

        Returns:
            Path to the model file.
        """
        artifact_dir = self.base_dir / name

        if version is None:
            latest_path = artifact_dir / "latest.json"
            if latest_path.exists():
                latest = json.loads(latest_path.read_text())
                version = latest["version"]
            else:
                raise FileNotFoundError(f"No artifacts found for '{name}'")

        version_dir = artifact_dir / f"v{version:03d}"
        if not version_dir.exists():
            raise FileNotFoundError(f"Version not found: {version_dir}")

        # Find the model file (first non-json file)
        for f in version_dir.iterdir():
            if f.suffix != ".json":
                return f

        raise FileNotFoundError(f"No model file in {version_dir}")

    def list_versions(self, name: str) -> list[ArtifactMetadata]:
        """List all versions of an artifact."""
        artifact_dir = self.base_dir / name
        if not artifact_dir.exists():
            return []

        versions = []
        for version_dir in sorted(artifact_dir.iterdir()):
            meta_path = version_dir / "metadata.json"
            if meta_path.exists():
                data = json.loads(meta_path.read_text())
                versions.append(ArtifactMetadata(**data))

        return versions

    def compare_versions(
        self, name: str, v1: int, v2: int
    ) -> dict[str, dict[str, float]]:
        """Compare metrics between two versions."""
        m1 = self.load_metadata(name, v1)
        m2 = self.load_metadata(name, v2)

        all_keys = set(m1.metrics) | set(m2.metrics)
        comparison = {}
        for key in sorted(all_keys):
            comparison[key] = {
                f"v{v1:03d}": m1.metrics.get(key, float("nan")),
                f"v{v2:03d}": m2.metrics.get(key, float("nan")),
                "delta": m2.metrics.get(key, 0) - m1.metrics.get(key, 0),
            }
        return comparison

    def _next_version(self, artifact_dir: Path) -> int:
        """Determine the next version number."""
        existing = [
            int(d.name[1:])
            for d in artifact_dir.iterdir()
            if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
        ]
        return max(existing, default=0) + 1

    @staticmethod
    def _get_git_hash() -> str:
        """Get current git commit hash."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""
