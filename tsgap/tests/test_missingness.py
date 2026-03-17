"""Unit tests for missingness simulation."""

import numpy as np
import pytest
from tsgap import simulate_missingness, simulate_many_rates, MissingnessSimulator


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


class TestPatternAPI:
    """Test explicit pattern parameter."""
    
    def test_pointwise_pattern_explicit(self):
        """Should work with explicit pattern='pointwise'."""
        X = np.random.randn(100, 5)
        
        X_missing, mask = simulate_missingness(
            X, "mcar", 0.15, seed=42, pattern="pointwise"
        )
        
        actual_rate = (~mask).sum() / mask.size
        assert abs(actual_rate - 0.15) < 0.01
    
    def test_block_pattern_explicit(self):
        """Should work with explicit pattern='block'."""
        X = np.random.randn(200, 5)
        
        X_missing, mask = simulate_missingness(
            X, "mcar", 0.20, seed=42, 
            pattern="block", block_len=10, block_density=0.7
        )
        
        # Should have some contiguous blocks
        has_blocks = False
        for d in range(5):
            missing_seq = ~mask[:, d]
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
            
            if runs and max(runs) >= 3:  # At least some blocks of length >= 3
                has_blocks = True
                break
        
        assert has_blocks, "Should have at least one block of length >= 3"
    
    def test_pattern_aliases(self):
        """Should accept pattern aliases."""
        X = np.random.randn(100, 5)
        
        # Test "point" alias
        X_missing1, mask1 = simulate_missingness(
            X, "mcar", 0.15, seed=42, pattern="point"
        )
        
        # Test "scattered" alias
        X_missing2, mask2 = simulate_missingness(
            X, "mcar", 0.15, seed=42, pattern="scattered"
        )
        
        # Test "contiguous" alias for block
        X_missing3, mask3 = simulate_missingness(
            X, "mcar", 0.15, seed=42, pattern="contiguous", block_len=5
        )
        
        # All should work without errors
        assert X_missing1.shape == X.shape
        assert X_missing2.shape == X.shape
        assert X_missing3.shape == X.shape
    
    def test_backward_compatibility_block_true(self):
        """Old block=True API should still work."""
        X = np.random.randn(200, 5)
        
        # Old API
        X_missing, mask = simulate_missingness(
            X, "mcar", 0.20, seed=42,
            block=True, block_len=10, block_density=0.7
        )
        
        # Should create blocks
        actual_rate = (~mask).sum() / mask.size
        assert abs(actual_rate - 0.20) < 0.05
    
    def test_mechanism_pattern_combinations(self):
        """Should support all mechanism + pattern combinations."""
        X = np.random.randn(100, 5)
        
        combinations = [
            ("mcar", "pointwise"),
            ("mcar", "block"),
            ("mar", "pointwise"),
            ("mar", "block"),
            ("mnar", "pointwise"),
            ("mnar", "block"),
        ]
        
        for mech, patt in combinations:
            kwargs = {}
            if mech == "mar":
                kwargs["driver_dims"] = [0]
            
            X_missing, mask = simulate_missingness(
                X, mech, 0.15, seed=42, pattern=patt, **kwargs
            )
            
            assert X_missing.shape == X.shape
            assert mask.shape == X.shape
    
    def test_invalid_pattern(self):
        """Should raise error for invalid pattern."""
        X = np.random.randn(100, 5)
        
        with pytest.raises(ValueError, match="Unknown pattern"):
            simulate_missingness(X, "mcar", 0.15, pattern="invalid")


class TestExtremeRates:
    """Test calibration accuracy at extreme missing rates."""

    @pytest.mark.parametrize("rate", [0.01, 0.02, 0.05])
    def test_mcar_low_rates(self, rate):
        X = np.random.default_rng(42).standard_normal((200, 5))
        _, mask = simulate_missingness(X, "mcar", rate, seed=42)
        actual = (~mask).sum() / mask.size
        assert abs(actual - rate) < 0.01

    @pytest.mark.parametrize("rate", [0.50, 0.70, 0.90])
    def test_mcar_high_rates(self, rate):
        X = np.random.default_rng(42).standard_normal((200, 5))
        _, mask = simulate_missingness(X, "mcar", rate, seed=42)
        actual = (~mask).sum() / mask.size
        assert abs(actual - rate) < 0.01

    @pytest.mark.parametrize("rate", [0.01, 0.05, 0.50, 0.90])
    def test_mar_extreme_rates(self, rate):
        X = np.random.default_rng(42).standard_normal((200, 5))
        _, mask = simulate_missingness(X, "mar", rate, seed=42, driver_dims=[0])
        actual = (~mask).sum() / mask.size
        assert abs(actual - rate) < 0.05

    @pytest.mark.parametrize("rate", [0.01, 0.05, 0.50, 0.90])
    def test_mnar_extreme_rates(self, rate):
        X = np.random.default_rng(42).standard_normal((200, 5))
        _, mask = simulate_missingness(X, "mnar", rate, seed=42, mnar_mode="extreme")
        actual = (~mask).sum() / mask.size
        assert abs(actual - rate) < 0.05


class TestNumericalStability:
    """Test numerical stability with extreme parameters."""

    def test_high_strength_mar(self):
        X = np.random.default_rng(42).standard_normal((200, 5))
        _, mask = simulate_missingness(
            X, "mar", 0.5, seed=42, driver_dims=[0], strength=10.0
        )
        actual = (~mask).sum() / mask.size
        assert abs(actual - 0.5) < 0.1

    def test_high_strength_mnar(self):
        X = np.random.default_rng(42).standard_normal((200, 5))
        _, mask = simulate_missingness(
            X, "mnar", 0.5, seed=42, mnar_mode="extreme", strength=10.0
        )
        actual = (~mask).sum() / mask.size
        assert abs(actual - 0.5) < 0.1

    def test_extreme_data_values(self):
        X = np.random.default_rng(42).standard_normal((100, 5)) * 1000
        _, mask = simulate_missingness(
            X, "mnar", 0.2, seed=42, mnar_mode="extreme", strength=5.0
        )
        actual = (~mask).sum() / mask.size
        assert abs(actual - 0.2) < 0.1


class TestMonotonePattern:
    """Test monotone (dropout) missingness pattern."""

    def test_monotone_constraint_2d(self):
        """Once a dimension goes missing, it should stay missing."""
        X = np.random.default_rng(42).standard_normal((200, 5))

        _, mask = simulate_missingness(
            X, "mcar", 0.20, seed=42, pattern="monotone"
        )

        # For each dimension, verify monotone: no observed after first missing
        for d in range(5):
            col = mask[:, d]
            missing_positions = np.where(~col)[0]
            if len(missing_positions) > 0:
                dropout_t = missing_positions[0]
                # Everything after dropout should be missing
                assert not col[dropout_t:].any(), (
                    f"Dim {d}: found observed values after dropout at t={dropout_t}"
                )

    def test_monotone_constraint_3d(self):
        """Monotone should work with 3D data."""
        X = np.random.default_rng(42).standard_normal((5, 100, 4))

        _, mask = simulate_missingness(
            X, "mcar", 0.25, seed=42, pattern="monotone"
        )

        for n in range(5):
            for d in range(4):
                col = mask[n, :, d]
                missing_positions = np.where(~col)[0]
                if len(missing_positions) > 0:
                    dropout_t = missing_positions[0]
                    assert not col[dropout_t:].any()

    def test_monotone_preserves_approximate_rate(self):
        """Monotone should preserve approximately the same missing rate."""
        X = np.random.default_rng(42).standard_normal((200, 5))

        _, mask = simulate_missingness(
            X, "mcar", 0.20, seed=42, pattern="monotone"
        )

        actual_rate = (~mask).sum() / mask.size
        # Monotone reshuffles missingness, rate may shift somewhat
        assert abs(actual_rate - 0.20) < 0.10

    def test_monotone_with_mar(self):
        """Monotone should work with MAR mechanism."""
        X = np.random.default_rng(42).standard_normal((200, 5))

        _, mask = simulate_missingness(
            X, "mar", 0.20, seed=42, pattern="monotone", driver_dims=[0]
        )

        # Verify monotone constraint holds
        for d in range(5):
            col = mask[:, d]
            missing_positions = np.where(~col)[0]
            if len(missing_positions) > 0:
                dropout_t = missing_positions[0]
                assert not col[dropout_t:].any()

    def test_dropout_alias(self):
        """'dropout' should be an alias for monotone."""
        X = np.random.default_rng(42).standard_normal((100, 5))

        _, mask = simulate_missingness(
            X, "mcar", 0.15, seed=42, pattern="dropout"
        )

        assert mask.shape == X.shape


class TestTemporalDecayPattern:
    """Test temporal decay (degradation) missingness pattern."""

    def test_decay_increases_over_time(self):
        """Later timesteps should have more missingness than earlier ones."""
        X = np.random.default_rng(42).standard_normal((500, 5))

        _, mask = simulate_missingness(
            X, "mcar", 0.30, seed=42, pattern="decay",
            decay_rate=5.0, decay_center=0.5
        )

        T = X.shape[0]
        first_quarter = (~mask[:T // 4]).sum()
        last_quarter = (~mask[3 * T // 4:]).sum()

        assert last_quarter > first_quarter, (
            f"Last quarter ({last_quarter}) should have more missing "
            f"than first quarter ({first_quarter})"
        )

    def test_decay_preserves_total_missing(self):
        """Decay should preserve approximately the target missing count."""
        X = np.random.default_rng(42).standard_normal((200, 5))

        _, mask = simulate_missingness(
            X, "mcar", 0.20, seed=42, pattern="decay"
        )

        actual_rate = (~mask).sum() / mask.size
        assert abs(actual_rate - 0.20) < 0.02

    def test_decay_3d(self):
        """Decay should work with 3D data."""
        X = np.random.default_rng(42).standard_normal((5, 200, 4))

        _, mask = simulate_missingness(
            X, "mcar", 0.25, seed=42, pattern="decay",
            decay_rate=4.0, decay_center=0.6
        )

        assert mask.shape == X.shape
        actual_rate = (~mask).sum() / mask.size
        assert abs(actual_rate - 0.25) < 0.02

    def test_decay_with_mnar(self):
        """Decay should work with MNAR mechanism."""
        X = np.random.default_rng(42).standard_normal((200, 5))

        _, mask = simulate_missingness(
            X, "mnar", 0.20, seed=42, pattern="decay",
            mnar_mode="extreme"
        )

        assert mask.shape == X.shape

    def test_degradation_alias(self):
        """'degradation' should be an alias for decay."""
        X = np.random.default_rng(42).standard_normal((100, 5))

        _, mask = simulate_missingness(
            X, "mcar", 0.15, seed=42, pattern="degradation"
        )

        assert mask.shape == X.shape

    def test_decay_center_controls_timing(self):
        """Lower decay_center should shift missingness earlier."""
        X = np.random.default_rng(42).standard_normal((500, 5))

        _, mask_early = simulate_missingness(
            X, "mcar", 0.30, seed=42, pattern="decay",
            decay_rate=5.0, decay_center=0.3
        )
        _, mask_late = simulate_missingness(
            X, "mcar", 0.30, seed=42, pattern="decay",
            decay_rate=5.0, decay_center=0.8
        )

        T = X.shape[0]
        # With early center, first half should have more missing
        early_first_half = (~mask_early[:T // 2]).sum()
        late_first_half = (~mask_late[:T // 2]).sum()

        assert early_first_half > late_first_half


class TestMARDriverWeights:
    """Test MAR weighted multi-driver combination."""

    def test_driver_weights_basic(self):
        """MAR with driver_weights should work without errors."""
        X = np.random.default_rng(42).standard_normal((200, 5))

        _, mask = simulate_missingness(
            X, "mar", 0.20, seed=42,
            driver_dims=[0, 1], driver_weights=[0.8, 0.2]
        )

        actual_rate = (~mask).sum() / mask.size
        assert abs(actual_rate - 0.20) < 0.05

    def test_driver_weights_normalized(self):
        """Unnormalized weights should be auto-normalized."""
        X = np.random.default_rng(42).standard_normal((200, 5))

        # These should produce the same result (both normalize to [0.8, 0.2])
        _, mask1 = simulate_missingness(
            X, "mar", 0.20, seed=42,
            driver_dims=[0, 1], driver_weights=[0.8, 0.2]
        )
        _, mask2 = simulate_missingness(
            X, "mar", 0.20, seed=42,
            driver_dims=[0, 1], driver_weights=[4.0, 1.0]
        )

        np.testing.assert_array_equal(mask1, mask2)

    def test_driver_weights_single_driver_dominates(self):
        """Heavily weighted driver should dominate missingness pattern."""
        # Create data where dim 0 and dim 1 have opposite patterns
        rng = np.random.default_rng(42)
        X = rng.standard_normal((300, 4))
        X[:150, 0] = -2  # dim 0: low first half
        X[150:, 0] = 2   # dim 0: high second half
        X[:150, 1] = 2   # dim 1: high first half (opposite)
        X[150:, 1] = -2  # dim 1: low second half

        # Weight dim 0 heavily
        _, mask_d0 = simulate_missingness(
            X, "mar", 0.30, seed=42,
            driver_dims=[0, 1], driver_weights=[0.99, 0.01],
            strength=3.0
        )

        # Weight dim 1 heavily
        _, mask_d1 = simulate_missingness(
            X, "mar", 0.30, seed=42,
            driver_dims=[0, 1], driver_weights=[0.01, 0.99],
            strength=3.0
        )

        # With dim 0 dominant: more missing in second half (high values)
        missing_d0_first = (~mask_d0[:150]).sum()
        missing_d0_second = (~mask_d0[150:]).sum()

        # With dim 1 dominant: more missing in first half (high values)
        missing_d1_first = (~mask_d1[:150]).sum()
        missing_d1_second = (~mask_d1[150:]).sum()

        # The dominant driver should flip which half has more missing
        assert missing_d0_second > missing_d0_first
        assert missing_d1_first > missing_d1_second

    def test_driver_weights_length_mismatch(self):
        """Should raise error if weights length doesn't match dims."""
        X = np.random.default_rng(42).standard_normal((100, 5))

        with pytest.raises(ValueError, match="driver_weights length"):
            simulate_missingness(
                X, "mar", 0.15, seed=42,
                driver_dims=[0, 1], driver_weights=[0.5]
            )

    def test_driver_weights_negative(self):
        """Should raise error for negative weights."""
        X = np.random.default_rng(42).standard_normal((100, 5))

        with pytest.raises(ValueError, match="non-negative"):
            simulate_missingness(
                X, "mar", 0.15, seed=42,
                driver_dims=[0, 1], driver_weights=[0.5, -0.5]
            )

    def test_driver_weights_all_zero(self):
        """Should raise error if all weights are zero."""
        X = np.random.default_rng(42).standard_normal((100, 5))

        with pytest.raises(ValueError, match="must not all be zero"):
            simulate_missingness(
                X, "mar", 0.15, seed=42,
                driver_dims=[0, 1], driver_weights=[0.0, 0.0]
            )

    def test_none_weights_equals_equal_weights(self):
        """None weights should behave like equal weights (mean)."""
        X = np.random.default_rng(42).standard_normal((200, 5))

        _, mask_none = simulate_missingness(
            X, "mar", 0.20, seed=42,
            driver_dims=[0, 1]
        )
        _, mask_equal = simulate_missingness(
            X, "mar", 0.20, seed=42,
            driver_dims=[0, 1], driver_weights=[0.5, 0.5]
        )

        np.testing.assert_array_equal(mask_none, mask_equal)

    def test_driver_weights_3d(self):
        """Driver weights should work with 3D data."""
        X = np.random.default_rng(42).standard_normal((5, 100, 4))

        _, mask = simulate_missingness(
            X, "mar", 0.20, seed=42,
            driver_dims=[0, 1], driver_weights=[0.7, 0.3]
        )

        assert mask.shape == X.shape
        actual_rate = (~mask).sum() / mask.size
        assert abs(actual_rate - 0.20) < 0.05


class TestMarkovPattern:
    """Test Markov chain temporal dependence pattern."""

    def test_markov_approximate_rate(self):
        """Markov pattern should produce approximately the target rate."""
        X = np.random.default_rng(42).standard_normal((500, 5))

        _, mask = simulate_missingness(
            X, "mcar", 0.20, seed=42, pattern="markov", persist=0.8
        )

        actual_rate = (~mask).sum() / mask.size
        assert abs(actual_rate - 0.20) < 0.05

    def test_markov_creates_bursts(self):
        """Higher persist should create longer missing bursts."""
        X = np.random.default_rng(42).standard_normal((500, 5))

        def avg_burst_length(mask):
            lengths = []
            for d in range(mask.shape[-1]):
                col = ~mask[:, d] if mask.ndim == 2 else ~mask[0, :, d]
                run = 0
                for v in col:
                    if v:
                        run += 1
                    else:
                        if run > 0:
                            lengths.append(run)
                        run = 0
                if run > 0:
                    lengths.append(run)
            return np.mean(lengths) if lengths else 0

        _, mask_low = simulate_missingness(
            X, "mcar", 0.20, seed=42, pattern="markov", persist=0.3
        )
        _, mask_high = simulate_missingness(
            X, "mcar", 0.20, seed=42, pattern="markov", persist=0.9
        )

        avg_low = avg_burst_length(mask_low)
        avg_high = avg_burst_length(mask_high)

        assert avg_high > avg_low, (
            f"High persist ({avg_high:.1f}) should have longer bursts "
            f"than low persist ({avg_low:.1f})"
        )

    def test_markov_3d(self):
        """Markov pattern should work with 3D data."""
        X = np.random.default_rng(42).standard_normal((5, 200, 4))

        _, mask = simulate_missingness(
            X, "mcar", 0.25, seed=42, pattern="markov", persist=0.7
        )

        assert mask.shape == X.shape
        actual_rate = (~mask).sum() / mask.size
        assert abs(actual_rate - 0.25) < 0.10

    def test_markov_with_mar(self):
        """Markov pattern should work with MAR mechanism."""
        X = np.random.default_rng(42).standard_normal((300, 5))

        _, mask = simulate_missingness(
            X, "mar", 0.20, seed=42,
            pattern="markov", driver_dims=[0], persist=0.7
        )

        assert mask.shape == X.shape

    def test_markov_with_mnar(self):
        """Markov pattern should work with MNAR mechanism."""
        X = np.random.default_rng(42).standard_normal((300, 5))

        _, mask = simulate_missingness(
            X, "mnar", 0.20, seed=42,
            pattern="markov", mnar_mode="extreme", persist=0.7
        )

        assert mask.shape == X.shape

    def test_flickering_alias(self):
        """'flickering' should be an alias for markov."""
        X = np.random.default_rng(42).standard_normal((100, 5))

        _, mask = simulate_missingness(
            X, "mcar", 0.15, seed=42, pattern="flickering", persist=0.6
        )

        assert mask.shape == X.shape

    def test_markov_persist_zero(self):
        """persist=0 should produce no temporal dependence (pointwise-like)."""
        X = np.random.default_rng(42).standard_normal((500, 5))

        _, mask = simulate_missingness(
            X, "mcar", 0.20, seed=42, pattern="markov", persist=0.0
        )

        actual_rate = (~mask).sum() / mask.size
        assert abs(actual_rate - 0.20) < 0.05

    def test_markov_invalid_persist(self):
        """Should raise error for persist >= 1.0 or < 0."""
        X = np.random.default_rng(42).standard_normal((100, 5))

        with pytest.raises(ValueError, match="persist"):
            simulate_missingness(
                X, "mcar", 0.15, seed=42, pattern="markov", persist=1.0
            )

        with pytest.raises(ValueError, match="persist"):
            simulate_missingness(
                X, "mcar", 0.15, seed=42, pattern="markov", persist=-0.1
            )

    def test_markov_reproducible(self):
        """Same seed should produce identical Markov masks."""
        X = np.random.default_rng(42).standard_normal((200, 5))

        _, mask1 = simulate_missingness(
            X, "mcar", 0.20, seed=99, pattern="markov", persist=0.8
        )
        _, mask2 = simulate_missingness(
            X, "mcar", 0.20, seed=99, pattern="markov", persist=0.8
        )

        np.testing.assert_array_equal(mask1, mask2)
