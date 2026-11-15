import unittest
import numpy as np

from gcomg_fss import (
    GCOMGConfig,
    build_fss,
    random_priority_matching,
    simulate_gcomg,
)


class TestRandomPriorityMatching(unittest.TestCase):
    def test_buy_side_majority(self):
        orders = np.array([1, 1, 1, -1, -1, 0], dtype=np.int8)
        mask, M, m_small, direction = random_priority_matching(
            orders, np.random.default_rng(10)
        )
        self.assertEqual((M, m_small, direction), (3, 2, 1))
        self.assertEqual(int(mask.sum()), 2)
        self.assertTrue(np.all(orders[mask] == 1))

    def test_sell_side_majority(self):
        orders = np.array([-1, -1, -1, 1, 0], dtype=np.int8)
        mask, M, m_small, direction = random_priority_matching(
            orders, np.random.default_rng(11)
        )
        self.assertEqual((M, m_small, direction), (3, 1, -1))
        self.assertEqual(int(mask.sum()), 1)
        self.assertTrue(np.all(orders[mask] == -1))

    def test_tie_has_no_strict_priority_group(self):
        orders = np.array([1, 1, -1, -1, 0], dtype=np.int8)
        mask, M, m_small, direction = random_priority_matching(
            orders, np.random.default_rng(12)
        )
        self.assertEqual((M, m_small, direction), (2, 2, 0))
        self.assertEqual(int(mask.sum()), 0)


class TestGCOMGSimulation(unittest.TestCase):
    def test_dense_fss_score_shape_and_reproducibility(self):
        cfg = GCOMGConfig(
            N=31, m=3, s=2, r=0.0, T=20.0, steps=300,
            seed=1234, zero_score_tie="random",
        )
        space = build_fss(cfg.m)
        result1 = simulate_gcomg(cfg, space)
        result2 = simulate_gcomg(cfg, space)
        self.assertEqual(result1.final_scores.shape, (cfg.N, cfg.s))
        self.assertEqual(result1.final_holdings.shape, (cfg.N, cfg.s))
        np.testing.assert_array_equal(result1.demand, result2.demand)
        np.testing.assert_allclose(result1.final_scores, result2.final_scores)
        self.assertTrue(np.all(result1.priority_count <= result1.minority_count))
        for t, direction in enumerate(result1.majority_direction):
            if direction == 0:
                self.assertEqual(
                    result1.majority_game_agents[t], result1.active_agents[t]
                )
            else:
                self.assertEqual(
                    result1.majority_game_agents[t], 2 * result1.priority_count[t]
                )
            self.assertEqual(
                result1.minority_game_agents[t],
                result1.active_agents[t] - result1.majority_game_agents[t],
            )

    def test_implicit_fss(self):
        cfg = GCOMGConfig(
            N=15, m=5, s=2, r=0.0, T=10.0, steps=50,
            seed=22, zero_score_tie="zero",
        )
        space = build_fss(cfg.m)
        self.assertIsNone(space.actions)
        result = simulate_gcomg(cfg, space)
        self.assertEqual(result.final_scores.shape, (cfg.N, cfg.s))
        self.assertEqual(result.sampled_strategy_actions.shape[1], 1 << cfg.m)


if __name__ == "__main__":
    unittest.main(verbosity=2)
