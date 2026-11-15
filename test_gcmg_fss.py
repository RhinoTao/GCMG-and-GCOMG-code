import unittest

import numpy as np

from gcmg_fss import (
    GCMGConfig,
    MarketMaker,
    build_fss,
    simulate_gcmg,
    _sample_fss_holdings,
)


class TestFSS(unittest.TestCase):
    def test_m2_full_matrix_still_contains_every_strategy(self):
        fss = build_fss(2)
        self.assertEqual(fss.P, 4)
        self.assertEqual(fss.n_strategies, 16)
        self.assertTrue(fss.is_materialized)
        self.assertEqual(fss.actions.shape, (16, 4))
        self.assertTrue(np.all(np.isin(fss.actions, (-1, 1))))
        self.assertEqual(np.unique(fss.actions, axis=0).shape[0], 16)
        np.testing.assert_array_equal(fss.actions[0], -np.ones(4, dtype=np.int8))
        np.testing.assert_array_equal(fss.actions[-1], np.ones(4, dtype=np.int8))

    def test_m4_full_matrix_remains_materialized(self):
        fss = build_fss(4)
        self.assertEqual(fss.n_strategies, 65_536)
        self.assertEqual(fss.actions.shape, (65_536, 16))
        self.assertEqual(fss.actions.nbytes, 65_536 * 16)

    def test_m5_is_implicit_and_has_exact_full_pool_size(self):
        fss = build_fss(5)
        self.assertFalse(fss.is_materialized)
        self.assertIsNone(fss.actions)
        self.assertEqual(fss.P, 32)
        self.assertEqual(fss.n_strategies, 1 << 32)

    def test_m10_is_implicit_and_has_exact_full_pool_size(self):
        fss = build_fss(10)
        self.assertFalse(fss.is_materialized)
        self.assertIsNone(fss.actions)
        self.assertEqual(fss.P, 1024)
        self.assertEqual(fss.n_strategies, 1 << 1024)

    def test_sparse_sampling_has_distinct_strategies_within_each_agent(self):
        rng = np.random.default_rng(2026)
        unique_actions, holdings_indices, unique_ids, holdings_ids = _sample_fss_holdings(
            N=80, s=3, P=64, rng=rng
        )
        self.assertEqual(unique_actions.shape[1], 64)
        self.assertTrue(np.all(np.isin(unique_actions, (-1, 1))))
        self.assertEqual(holdings_indices.shape, (80, 3))
        self.assertEqual(holdings_ids.shape, (80, 3))
        self.assertEqual(len(np.unique(unique_actions, axis=0)), unique_actions.shape[0])
        for row in holdings_ids:
            self.assertEqual(len(set(int(x) for x in row)), 3)
        # Every agent's strategy IDs must resolve to the sampled action matrix.
        # Check every sampled ID against its corresponding action row.
        for action_row, strategy_id in zip(unique_actions, unique_ids):
            expected_bits = np.array(
                [(int(strategy_id) >> j) & 1 for j in range(64)], dtype=np.int8
            )
            np.testing.assert_array_equal(action_row, 2 * expected_bits - 1)

    def test_m10_short_simulation_runs_and_returns_sparse_scores(self):
        cfg = GCMGConfig(
            N=51, m=10, s=2, r=4, T=100, L=0,
            steps=100, seed=12, initial_score_offset=10,
        )
        result = simulate_gcmg(cfg)
        self.assertEqual(result.demand.size, 100)
        self.assertEqual(result.final_holdings.shape, (51, 2))
        self.assertEqual(result.final_scores.size, result.final_score_strategy_ids.size)
        self.assertEqual(result.sampled_strategy_actions.shape[1], 1024)
        self.assertTrue(np.all(result.price > 0))
        self.assertEqual(int(result.price_fallback.sum()), 0)

    def test_negative_r_is_accepted(self):
        cfg = GCMGConfig(N=21, m=2, s=2, r=-10, T=100, L=0, steps=10)
        result = simulate_gcmg(cfg)
        self.assertEqual(result.demand.size, 10)

    def test_market_maker_linear_step_is_unchanged(self):
        mm = MarketMaker(lam=10.0, price=100.0, mode="linear")
        new_price, price_change, simple_ret, log_ret = mm.clear(5)
        self.assertAlmostEqual(new_price, 100.5)
        self.assertAlmostEqual(price_change, 0.5)
        self.assertAlmostEqual(simple_ret, 0.005)
        self.assertAlmostEqual(log_ret, np.log(1.005))
        self.assertFalse(mm.last_fallback)

    def test_linear_boundary_uses_log_rule_for_that_step_only(self):
        mm = MarketMaker(lam=1.0, price=10.0, mode="linear")
        new_price, price_change, simple_ret, log_ret = mm.clear(-11)
        expected = 10.0 * np.exp(-11.0)
        self.assertTrue(mm.last_fallback)
        self.assertGreater(new_price, 0.0)
        self.assertAlmostEqual(new_price, expected, places=14)
        self.assertAlmostEqual(log_ret, -11.0)
        self.assertAlmostEqual(simple_ret, np.expm1(-11.0))
        self.assertAlmostEqual(price_change, expected - 10.0)

        # A subsequent non-boundary step must return to the linear rule.
        old_price = mm.price
        new_price2, price_change2, _, _ = mm.clear(1)
        self.assertFalse(mm.last_fallback)
        self.assertAlmostEqual(new_price2, old_price + 1.0)
        self.assertAlmostEqual(price_change2, 1.0)

    def test_simulation_records_price_boundary_fallback(self):
        cfg = GCMGConfig(
            N=21, m=2, s=1, r=-100, T=100, L=0,
            lam=1, x0=0.1, steps=30, seed=2,
            initial_score_offset=10, zero_score_tie="zero",
        )
        result = simulate_gcmg(cfg)
        self.assertGreater(int(result.price_fallback.sum()), 0)
        self.assertTrue(np.all(result.price > 0))

    def test_log_mode_remains_log_mode(self):
        mm = MarketMaker(lam=2.0, price=10.0, mode="log")
        new_price, impact, simple_ret, log_ret = mm.clear(-2)
        self.assertAlmostEqual(new_price, 10.0 * np.exp(-1.0))
        self.assertAlmostEqual(impact, -1.0)
        self.assertAlmostEqual(simple_ret, np.expm1(-1.0))
        self.assertAlmostEqual(log_ret, -1.0)
        self.assertFalse(mm.last_fallback)

    def test_strict_zero_tie_absorbing_state_remains(self):
        cfg = GCMGConfig(
            N=31, m=2, s=2, r=4, T=100, L=0,
            steps=100, seed=7, initial_score=0, zero_score_tie="zero",
        )
        result = simulate_gcmg(cfg)
        self.assertTrue(np.all(result.demand == 0))
        self.assertTrue(np.all(result.active_agents == 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
