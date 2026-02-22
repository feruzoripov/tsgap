"""Unit tests for missingness simulation."""

import numpy as np
import pytest
from ts_missingness import simulate_missingness, simulate_many_rates, MissingnessSimulator


class TestMCAR:
    """Test MCAR mechanism."""
    
    def test_exact_missing_rate_2d(self):
        """MCAR should inject exact missing rate for 2D data."""
        X = np.random.randn(100, 5)
        target_rate = 0.15
        
        X_missing, mask = simulate_missingness(X, "mcar", target_rate, seed=42)
        
        actual_rate = (~mask).sum() / mask.size
        assert abs(actual_rate - target_rate) < 0.01
    
    def test_exact_missing_rate_3d(self):
        """MCAR should inject exact missing rate for 3D data."""
        X = np.random.randn(10, 50, 5)
        target_rate = 0.25
        
        X_missing, mask = simulate_missingness(X, "mcar", target_rate, seed=42)
        
        actual_rate = (~mask).sum() / mask.size
        assert abs(actual_rate - target_rate) < 0.01
    
    def test_target_dims(self):
        """MCAR should only mask specified dimensions."""
        X = np.random.randn(100, 5)
        target_dims = [0, 2]
        
        X_missing, mask = simulate_missingness(
            X, "mcar", 0.2, seed=42, target=target_dims
        )
        
        # Check that only target dims have missingness
        for d in range(5):
            if d in target_dims:
                assert (~mask[:, d]).sum() > 0
            else:
                assert (~mask[:, d]).sum() == 0


class TestMAR:
    """Test MAR mechanism."""
    
    def test_approximate_missing_rate(self):
        """MAR should inject approximately requested missing rate."""
        X = np.random.randn(100, 5)
        target_rate = 0.20
        
        X_missing, mask = simulate_missingness(
            X, "mar", target_rate, seed=42, driver_dims=[0]
        )
        
        actual_rate = (~mask).sum() / mask.size
        assert abs(actual_rate - target_rate) < 0.05  # Within 5%
    
    def test_dependency_on_driver(self):
        """MAR missingness should depend on driver dimension."""
        # Create data where dim 0 has clear high/low regions
        X = np.zeros((200, 3))
        X[:100, 0] = -2  # Low values
        X[100:, 0] = 2   # High values
        X[:, 1:] = np.random.randn(200, 2)
        
        X_missing, mask = simulate_missingness(
            X, "mar", 0.3, seed=42, driver_dims=[0], strength=3.0
        )
        
        # Check that missingness differs between regions
        missing_low = (~mask[:100]).sum()
        missing_high = (~mask[100:]).sum()
        
        # Should have different missingness rates
        assert abs(missing_low - missing_high) > 10


class TestMNAR:
    """Test MNAR mechanism."""
    
    def test_approximate_missing_rate(self):
        """MNAR should inject approximately requested missing rate."""
        X = np.random.randn(100, 5)
        target_rate = 0.15
        
        X_missing, mask = simulate_missingness(
            X, "mnar", target_rate, seed=42, mnar_mode="extreme"
        )
        
        actual_rate = (~mask).sum() / mask.size
        assert abs(actual_rate - target_rate) < 0.05

    def test_extreme_mode(self):
        """MNAR extreme mode should mask extreme values more."""
        X = np.random.randn(200, 3)
        # Add some extreme values
        X[0, 0] = 10
        X[1, 0] = -10
        
        X_missing, mask = simulate_missingness(
            X, "mnar", 0.2, seed=42, mnar_mode="extreme", strength=5.0
        )
        
        # Extreme values should be more likely missing
        assert not mask[0, 0] or not mask[1, 0]  # At least one should be missing


class TestBlockMissingness:
    """Test block missingness patterns."""
    
    def test_creates_contiguous_blocks(self):
        """Block missingness should create contiguous segments."""
        X = np.random.randn(200, 5)
        
        X_missing, mask = simulate_missingness(
            X, "mcar", 0.25, seed=42,
            block=True, block_len=10, block_density=0.8
        )
        
        # Check for contiguous missing segments
        for d in range(5):
            missing_seq = ~mask[:, d]
            # Find runs of consecutive True values
            runs = []
            current_run = 0
            for val in missing_seq:
                if val:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)
            
            # Should have some longer runs
            if runs:
                assert max(runs) >= 5  # At least some blocks


class TestReproducibility:
    """Test seed reproducibility."""
    
    def test_same_seed_same_result(self):
        """Same seed should produce identical results."""
        X = np.random.randn(100, 5)
        
        X_missing1, mask1 = simulate_missingness(X, "mcar", 0.15, seed=42)
        X_missing2, mask2 = simulate_missingness(X, "mcar", 0.15, seed=42)
        
        np.testing.assert_array_equal(mask1, mask2)
        np.testing.assert_array_equal(
            np.isnan(X_missing1), np.isnan(X_missing2)
        )
    
    def test_different_seed_different_result(self):
        """Different seeds should produce different results."""
        X = np.random.randn(100, 5)
        
        X_missing1, mask1 = simulate_missingness(X, "mcar", 0.15, seed=42)
        X_missing2, mask2 = simulate_missingness(X, "mcar", 0.15, seed=123)
        
        assert not np.array_equal(mask1, mask2)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_constant_signal(self):
        """Should handle constant signals without errors."""
        X = np.ones((100, 5))
        
        # Should not raise
        X_missing, mask = simulate_missingness(X, "mnar", 0.15, seed=42)
        assert X_missing.shape == X.shape
    
    def test_existing_nans(self):
        """Should handle existing NaNs properly."""
        X = np.random.randn(100, 5)
        X[0:10, 0] = np.nan  # 10% pre-existing NaNs in one column
        
        X_missing, mask = simulate_missingness(X, "mcar", 0.15, seed=42)
        
        # Existing NaNs should remain NaN
        assert np.isnan(X_missing[0:10, 0]).all()
    
    def test_invalid_mechanism(self):
        """Should raise error for invalid mechanism."""
        X = np.random.randn(100, 5)
        
        with pytest.raises(ValueError):
            simulate_missingness(X, "invalid", 0.15)
    
    def test_invalid_missing_rate(self):
        """Should clip out-of-range missing rates instead of raising."""
        X = np.random.randn(100, 5)
        
        # Negative rate should be clipped to 0
        X_missing, mask = simulate_missingness(X, "mcar", -0.5, seed=42)
        assert mask.all()  # All observed
        
        # Rate > 1 should be clipped to 1
        X_missing, mask = simulate_missingness(X, "mcar", 1.5, seed=42)
        assert (~mask).all()  # All missing


class TestMultiRate:
    """Test multi-rate generation."""
    
    def test_simulate_many_rates(self):
        """Should generate results for multiple rates."""
        X = np.random.randn(100, 5)
        rates = [0.05, 0.15, 0.25]
        
        results = simulate_many_rates(X, "mcar", rates, seed=42)
        
        assert len(results) == 3
        for rate in rates:
            assert rate in results
            X_missing, mask = results[rate]
            assert X_missing.shape == X.shape


class TestOOInterface:
    """Test object-oriented interface."""
    
    def test_simulator_basic(self):
        """MissingnessSimulator should work correctly."""
        X = np.random.randn(100, 5)
        
        sim = MissingnessSimulator("mcar", missing_rate=0.15, seed=42)
        X_missing, mask = sim.generate(X)
        
        assert X_missing.shape == X.shape
        actual_rate = (~mask).sum() / mask.size
        assert abs(actual_rate - 0.15) < 0.01
    
    def test_simulator_with_config(self):
        """MissingnessSimulator should accept configuration."""
        X = np.random.randn(100, 5)
        
        sim = MissingnessSimulator(
            "mar", missing_rate=0.2, seed=42,
            driver_dims=[0], strength=2.0
        )
        X_missing, mask = sim.generate(X)
        
        assert X_missing.shape == X.shape


class TestValidation:
    """Test input validation."""
    
    def test_invalid_target_dims(self):
        """Should raise error for out-of-range target dimensions."""
        X = np.random.randn(100, 5)
        
        with pytest.raises(ValueError, match="target dimensions out of range"):
            simulate_missingness(X, "mcar", 0.15, target=[10])
        
        with pytest.raises(ValueError, match="target dimensions out of range"):
            simulate_missingness(X, "mar", 0.15, driver_dims=[0], target=[-1])
    
    def test_invalid_driver_dims(self):
        """Should raise error for out-of-range driver dimensions."""
        X = np.random.randn(100, 5)
        
        with pytest.raises(ValueError, match="driver_dims out of range"):
            simulate_missingness(X, "mar", 0.15, driver_dims=[10])
        
        with pytest.raises(ValueError, match="driver_dims out of range"):
            simulate_missingness(X, "mar", 0.15, driver_dims=[-1])
    
    def test_zero_missing_rate(self):
        """Should handle zero missing rate gracefully."""
        X = np.random.randn(100, 5)
        
        X_missing, mask = simulate_missingness(X, "mcar", 0.0, seed=42)
        assert mask.all()  # All observed
        
        X_missing, mask = simulate_missingness(X, "mar", 0.0, seed=42, driver_dims=[0])
        assert mask.all()  # All observed


class TestAdditionalValidation:
    """Test additional validation and edge cases."""
    
    def test_negative_strength(self):
        """Should raise error for negative strength."""
        X = np.random.randn(100, 5)
        
        with pytest.raises(ValueError, match="strength must be >= 0"):
            simulate_missingness(X, "mar", 0.15, driver_dims=[0], strength=-1.0)
        
        with pytest.raises(ValueError, match="strength must be >= 0"):
            simulate_missingness(X, "mnar", 0.15, strength=-1.0)
    
    def test_missing_rate_clipping(self):
        """Should clip missing_rate to [0, 1]."""
        X = np.random.randn(100, 5)
        
        # Negative rate should be treated as 0
        X_missing, mask = simulate_missingness(X, "mcar", -0.1, seed=42)
        assert mask.all()  # All observed
        
        # Rate > 1 should be treated as 1
        X_missing, mask = simulate_missingness(X, "mcar", 1.5, seed=42)
        assert (~mask).sum() == mask.size  # All missing
    
    def test_mcar_full_missing(self):
        """MCAR should handle missing_rate=1.0 correctly."""
        X = np.random.randn(100, 5)
        
        X_missing, mask = simulate_missingness(X, "mcar", 1.0, seed=42)
        assert (~mask).sum() == mask.size  # All missing
        assert np.isnan(X_missing).all()


class TestFinalValidation:
    """Test final validation enhancements."""
    
    def test_invalid_direction(self):
        """Should raise error for invalid direction."""
        X = np.random.randn(100, 5)
        
        with pytest.raises(ValueError, match="direction must be"):
            simulate_missingness(
                X, "mar", 0.15, driver_dims=[0], direction="invalid"
            )
    
    def test_target_as_tuple(self):
        """Should accept target as tuple."""
        X = np.random.randn(100, 5)
        
        # Should work with tuple
        X_missing, mask = simulate_missingness(
            X, "mcar", 0.15, seed=42, target=(1, 2)
        )
        
        # Only dims 1 and 2 should have missingness
        assert (~mask[:, 0]).sum() == 0
        assert (~mask[:, 1]).sum() > 0
        assert (~mask[:, 2]).sum() > 0
        assert (~mask[:, 3]).sum() == 0
    
    def test_target_as_array(self):
        """Should accept target as numpy array."""
        X = np.random.randn(100, 5)
        
        # Should work with numpy array
        X_missing, mask = simulate_missingness(
            X, "mcar", 0.15, seed=42, target=np.array([0, 4])
        )
        
        # Only dims 0 and 4 should have missingness
        assert (~mask[:, 0]).sum() > 0
        assert (~mask[:, 1]).sum() == 0
        assert (~mask[:, 4]).sum() > 0
