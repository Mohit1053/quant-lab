"""Colab integration utilities for Google Colab Pro+ environments."""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Google Drive mount point and project directory
DRIVE_MOUNT = "/content/drive"
DRIVE_PROJECT_DIR = "/content/drive/MyDrive/quant_lab"
COLAB_REPO_DIR = "/content/quant-lab"
GITHUB_REPO = "mohit1/quant-lab"  # Update with actual username


def is_colab() -> bool:
    """Detect if running inside Google Colab."""
    return "google.colab" in sys.modules


def setup_colab(
    github_repo: str | None = None,
    mount_drive: bool = True,
    install_deps: bool = True,
) -> dict[str, str]:
    """Full Colab environment setup.

    Call this at the top of any Colab notebook:
        from quant_lab.utils.colab import setup_colab
        paths = setup_colab()

    Returns:
        Dict with paths: data_dir, output_dir, model_dir, repo_dir
    """
    if not is_colab():
        logger.info("not_colab", msg="Skipping Colab setup (local environment)")
        return {
            "data_dir": "data",
            "output_dir": "outputs",
            "model_dir": "outputs/models",
            "repo_dir": ".",
        }

    repo = github_repo or GITHUB_REPO

    # 1. Mount Google Drive
    if mount_drive:
        _mount_drive()

    # 2. Create persistent directories on Drive
    paths = _create_drive_dirs()

    # 3. Clone/update repo
    _clone_or_update_repo(repo)

    # 4. Install package
    if install_deps:
        _install_package()

    # 5. Symlink data/outputs to Drive for persistence
    _create_symlinks(paths)

    # 6. Log GPU info
    _log_gpu_info()

    logger.info("colab_setup_complete", paths=paths)
    return paths


def _mount_drive():
    """Mount Google Drive."""
    from google.colab import drive  # type: ignore
    drive.mount(DRIVE_MOUNT, force_remount=False)
    logger.info("drive_mounted", path=DRIVE_MOUNT)


def _create_drive_dirs() -> dict[str, str]:
    """Create persistent directory structure on Google Drive."""
    dirs = {
        "data_dir": f"{DRIVE_PROJECT_DIR}/data",
        "output_dir": f"{DRIVE_PROJECT_DIR}/outputs",
        "model_dir": f"{DRIVE_PROJECT_DIR}/outputs/models",
        "regime_dir": f"{DRIVE_PROJECT_DIR}/outputs/regimes",
        "backtest_dir": f"{DRIVE_PROJECT_DIR}/outputs/backtests",
        "mlruns_dir": f"{DRIVE_PROJECT_DIR}/mlruns",
    }
    for d in dirs.values():
        Path(d).mkdir(parents=True, exist_ok=True)

    # Also create data subdirs
    for sub in ["raw", "cleaned", "features", "embeddings"]:
        Path(f"{dirs['data_dir']}/{sub}").mkdir(exist_ok=True)

    return dirs


def _clone_or_update_repo(repo: str):
    """Clone repo or pull latest if already cloned."""
    repo_dir = Path(COLAB_REPO_DIR)
    if repo_dir.exists():
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=False)
        logger.info("repo_updated", path=str(repo_dir))
    else:
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo}.git", str(repo_dir)],
            check=True,
        )
        logger.info("repo_cloned", path=str(repo_dir))

    os.chdir(str(repo_dir))
    # Add src to Python path
    src_path = str(repo_dir / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _install_package():
    """Install the package in editable mode."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", "."],
        check=True,
        cwd=COLAB_REPO_DIR,
    )
    logger.info("package_installed")


def _create_symlinks(paths: dict[str, str]):
    """Symlink local data/outputs dirs to Google Drive for persistence."""
    repo_dir = Path(COLAB_REPO_DIR)

    # Symlink data/ -> Drive
    local_data = repo_dir / "data"
    if local_data.exists() and not local_data.is_symlink():
        import shutil
        shutil.rmtree(local_data, ignore_errors=True)
    if not local_data.exists():
        local_data.symlink_to(paths["data_dir"])

    # Symlink outputs/ -> Drive
    local_outputs = repo_dir / "outputs"
    if local_outputs.exists() and not local_outputs.is_symlink():
        import shutil
        shutil.rmtree(local_outputs, ignore_errors=True)
    if not local_outputs.exists():
        local_outputs.symlink_to(paths["output_dir"])

    logger.info("symlinks_created", data=str(local_data), outputs=str(local_outputs))


def _log_gpu_info():
    """Log available GPU information."""
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        bf16 = torch.cuda.is_bf16_supported()
        logger.info(
            "gpu_detected",
            name=gpu_name,
            memory_gb=f"{gpu_mem:.1f}",
            bf16_supported=bf16,
        )
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB) | BF16: {bf16}")
    else:
        logger.warning("no_gpu", msg="No GPU detected. Training will be slow.")
        print("WARNING: No GPU detected!")


def get_data_dir() -> str:
    """Get the data directory (Drive path on Colab, local otherwise)."""
    if is_colab():
        return f"{DRIVE_PROJECT_DIR}/data"
    return "data"


def get_output_dir() -> str:
    """Get the output directory (Drive path on Colab, local otherwise)."""
    if is_colab():
        return f"{DRIVE_PROJECT_DIR}/outputs"
    return "outputs"
