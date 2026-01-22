import shutil
import subprocess
import sys
from pathlib import Path

sys.path.append("../src/")

import pytest

# Project root (one level up from tests/)
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_outputs():
    """Remove TEST_DELETE_ME folders from outputs/ after all tests run."""
    yield  # Run tests first
    outputs_dir = PROJECT_ROOT / "outputs"
    if outputs_dir.exists():
        for folder in outputs_dir.iterdir():
            if folder.is_dir() and "TEST_DELETE_ME" in folder.name:
                shutil.rmtree(folder)
                print(f"Cleaned up: {folder.name}")


def test_train_loop_SR():
    """Run the training loop with SR test config."""
    result = subprocess.run(
        [sys.executable, "src/train.py", "tests/test_configs/SR_1epoch.yaml"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, f"Train failed: {result.stderr}"


def test_train_loop_SGS():
    """Run the training loop with SGS test config."""
    result = subprocess.run(
        [sys.executable, "src/train.py", "tests/test_configs/SGS_1epoch.yaml"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, f"Train failed: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])