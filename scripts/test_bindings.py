import numpy as np
import mmr_elites_rs
import pytest

def test_zero_copy_bindings():
    k = 5
    n_samples = 10
    dim = 2

    # Create random numpy arrays
    fitness = np.random.rand(n_samples).astype(np.float64)
    descriptors = np.random.rand(n_samples, dim).astype(np.float64)

    selector = mmr_elites_rs.MMRSelector(target_k=k, lambda_val=0.5)

    try:
        selected_indices = selector.select(fitness, descriptors)
        print(f"Selected indices: {selected_indices}")

        assert isinstance(selected_indices, np.ndarray)
        assert selected_indices.dtype == np.uintp # or whatever the Rust usize maps to in numpy
        assert selected_indices.shape == (k,)
        
        # Verify no crash (implicit by reaching this point)
        print("Zero-copy bindings test passed: No segfaults and correct output format.")

    except Exception as e:
        pytest.fail(f"Zero-copy bindings test failed: {e}")

# To run this test:
# 1. Compile Rust: `maturin develop --release`
# 2. Run pytest: `pytest scripts/test_bindings.py`
