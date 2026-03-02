"""
Pytest configuration and shared fixtures for MMR-Elites tests.
"""

from typing import Tuple

import numpy as np
import pytest

# Try to import Rust backend
try:
    import mmr_elites_rs

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


@pytest.fixture
def random_seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seed(random_seed):
    """Set numpy random seed before each test."""
    np.random.seed(random_seed)
    yield


@pytest.fixture
def small_population():
    """Small test population for unit tests."""
    np.random.seed(42)
    n, d = 100, 5
    fitness = np.random.rand(n).astype(np.float64)
    descriptors = np.random.rand(n, d).astype(np.float64)
    return fitness, descriptors


@pytest.fixture
def medium_population():
    """Medium test population for integration tests."""
    np.random.seed(42)
    n, d = 1000, 20
    fitness = np.random.rand(n).astype(np.float64)
    descriptors = np.random.rand(n, d).astype(np.float64)
    return fitness, descriptors


@pytest.fixture
def arm_task():
    """5-DOF arm task for quick tests."""
    from mmr_elites.tasks.arm import ArmTask

    return ArmTask(n_dof=5, use_highdim_descriptor=True)


@pytest.fixture
def arm_task_20dof():
    """20-DOF arm task for main experiments."""
    from mmr_elites.tasks.arm import ArmTask

    return ArmTask(n_dof=20, use_highdim_descriptor=True)


@pytest.fixture
def mmr_selector():
    """MMR selector with default parameters."""
    if not RUST_AVAILABLE:
        pytest.skip("Rust backend not available")
    return mmr_elites_rs.MMRSelector(target_k=100, lambda_val=0.5)


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "rust: marks tests requiring Rust backend")
