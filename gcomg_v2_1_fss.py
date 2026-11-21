#!/usr/bin/env python3
"""GCOMGv2.1-FSS: expected-payoff scoring with optional strategy exploration.

Core conventions
----------------
* The original GCMG decision process is retained: each agent holds ``s`` FSS
  strategies, picks its highest-scoring strategy, and participates iff its best
  score is strictly greater than ``r``.
* For every held strategy R of every agent i, construct a counterfactual book:
    - if i is active, replace i's actually submitted order by R's action;
    - if i is inactive, add R's hypothetical order to the actual book.
  All other agents retain their actual orders.
* Let M_cf >= n_cf be the two side counts in that counterfactual book.  If R's
  action is on the majority side, its expected score is

      g_iR = ((n_cf - M_cf) / M_cf) * a_iR * chi_cf * pt_cf,

  where pt_cf = abs(D_cf/lambda), chi_cf = sign(D_cf/lambda - L).
  If R's action is on the minority side, or the book is balanced, g_iR = 0.
  For L=0 this reduces exactly to the user's proposed rule:
      majority: ((n_cf-M_cf)/M_cf) * pt_cf; minority: 0.
* The expected rule is the analytical average of the two-stage matching:
  n/2 agents on each side are assigned the Majority-Game payoff, and the
  remaining n/2 minority-side plus M-n/2 majority-side agents are assigned the
  Minority-Game payoff.  No random priority draw and no score redistribution
  are performed.
* Strategy scores are agent-specific (N x s), because each strategy is scored
  in its owner's counterfactual order book.
* Finite score memory: S_iR(t+1) = (1 - 1/T) S_iR(t) + g_iR(t+1).

The public information and market-price rules follow the supplied GCMG-FSS
implementation.  ``m`` is the original public-information memory length; it
is not the minority-side count n_cf.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

MAX_FSS_M = 10
DENSE_FSS_MAX_M = 4


class StrategySpace(Protocol):
    m: int
    actions: Optional[np.ndarray]


@dataclass(frozen=True)
class FSSStrategySpace:
    """Full Strategy Space descriptor; materialize only for m <= 4."""

    m: int
    actions: Optional[np.ndarray] = None

    @property
    def P(self) -> int:
        return 1 << self.m

    @property
    def n_strategies(self) -> int:
        return 1 << self.P


def build_fss(m: int, verify: bool = True) -> FSSStrategySpace:
    if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or m < 1:
        raise ValueError("m must be an integer >= 1.")
    m = int(m)
    if m > MAX_FSS_M:
        raise ValueError(f"This implementation supports 1 <= m <= {MAX_FSS_M}; got m={m}.")
    P = 1 << m
    if m > DENSE_FSS_MAX_M:
        return FSSStrategySpace(m=m, actions=None)

    n_strategies = 1 << P
    strategy_ids = np.arange(n_strategies, dtype=np.uint32)
    bit_positions = np.arange(P, dtype=np.uint32)
    bits = (strategy_ids[:, None] >> bit_positions[None, :]) & 1
    actions = (2 * bits.astype(np.int8) - 1).astype(np.int8, copy=False)
    if verify:
        if actions.shape != (n_strategies, P):
            raise RuntimeError("FSS construction failed: wrong action-matrix shape.")
        if not np.all(np.isin(actions, (-1, 1))):
            raise RuntimeError("FSS actions must be +/-1.")
        if not np.all(actions[0] == -1) or not np.all(actions[-1] == 1):
            raise RuntimeError("FSS binary encoding is inconsistent.")
    return FSSStrategySpace(m=m, actions=actions)


def _bits_to_python_ids(bits: np.ndarray) -> list[int]:
    packed = np.packbits(np.asarray(bits, dtype=np.uint8), axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little", signed=False) for row in packed]


def _sample_fss_holdings(
    N: int, s: int, P: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample s distinct strategies per agent uniformly from the implicit FSS."""
    held_bits = np.empty((N, s, P), dtype=np.uint8)
    held_ids: list[list[int]] = []
    for i in range(N):
        seen: set[bytes] = set()
        ids_i: list[int] = []
        j = 0
        while j < s:
            bit_row = rng.integers(0, 2, size=P, dtype=np.uint8)
            key = bit_row.tobytes()
            if key in seen:
                continue
            seen.add(key)
            held_bits[i, j] = bit_row
            ids_i.append(int.from_bytes(np.packbits(bit_row, bitorder="little").tobytes(), "little"))
            j += 1
        held_ids.append(ids_i)

    unique_bits, inverse = np.unique(held_bits.reshape(N * s, P), axis=0, return_inverse=True)
    unique_actions = (2 * unique_bits.astype(np.int8) - 1).astype(np.int8, copy=False)
    unique_id_list = _bits_to_python_ids(unique_bits)
    if P <= 64:
        unique_ids = np.asarray(unique_id_list, dtype=np.uint64)
        holdings_ids = np.asarray(held_ids, dtype=np.uint64)
    else:
        unique_ids = np.asarray(unique_id_list, dtype=object)
        holdings_ids = np.asarray(held_ids, dtype=object)
    return unique_actions, inverse.reshape(N, s).astype(np.int32), unique_ids, holdings_ids


@dataclass
class GCOMGv2Config:
    N: int = 501
    m: int = 3                         # public-information memory length
    s: int = 2                         # strategies held per agent
    r: float = 4.0                     # finite Grand-Canonical entry threshold
    T: float = 100.0                   # finite score-memory time scale
    L: float = 0.0
    lam: Optional[float] = None        # None -> 10*N, as in supplied program
    x0: float = 100.0
    price_mode: str = "linear"         # linear or log
    steps: int = 100_000
    seed: int = 123
    mu0: Optional[int] = None
    initial_score_offset: float = 10.0
    initial_score: Optional[float] = None
    zero_score_tie: str = "random"     # public-information tie only
    tie_tolerance: float = 1e-12
    strategy_exploration: float = 0.0  # epsilon-greedy choice among held strategies

    def resolved_lambda(self) -> float:
        value = 10.0 * self.N if self.lam is None else float(self.lam)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("lambda must be finite and positive.")
        return value

    def validate(self, n_strategies: Optional[int] = None) -> None:
        if self.N < 1 or self.m < 1 or self.s < 1:
            raise ValueError("N, m, and s must all be positive integers.")
        if not np.isfinite(self.r):
            raise ValueError("r must be finite in this finite-r version.")
        if not np.isfinite(self.T) or self.T < 1:
            raise ValueError("T must be finite and >= 1 (so alpha=1-1/T lies in [0,1)).")
        if not np.isfinite(self.L):
            raise ValueError("L must be finite.")
        if self.steps < 1:
            raise ValueError("steps must be >= 1.")
        if not np.isfinite(self.x0) or self.x0 <= 0:
            raise ValueError("x0 must be finite and positive.")
        if self.price_mode not in {"linear", "log"}:
            raise ValueError("price_mode must be 'linear' or 'log'.")
        if not np.isfinite(self.initial_score_offset) or self.initial_score_offset < 0:
            raise ValueError("initial_score_offset must be finite and non-negative.")
        if self.initial_score is not None and not np.isfinite(self.initial_score):
            raise ValueError("initial_score must be finite when specified.")
        if self.zero_score_tie not in {"random", "zero"}:
            raise ValueError("zero_score_tie must be 'random' or 'zero'.")
        if self.tie_tolerance < 0:
            raise ValueError("tie_tolerance must be non-negative.")
        if (not np.isfinite(self.strategy_exploration)
                or not 0.0 <= self.strategy_exploration <= 1.0):
            raise ValueError("strategy_exploration must lie in [0, 1].")
        self.resolved_lambda()
        if n_strategies is not None and self.s > n_strategies:
            raise ValueError("s exceeds the total number of strategies in the FSS.")
        if self.mu0 is not None and not (0 <= self.mu0 < (1 << self.m)):
            raise ValueError(f"mu0 must lie in [0, {(1 << self.m) - 1}].")


@dataclass
class MarketMaker:
    lam: float
    price: float
    mode: str = "linear"
    last_fallback: bool = field(default=False, init=False)

    def clear(self, demand: int) -> tuple[float, float, float, float]:
        if not np.isfinite(self.price) or self.price <= 0:
            raise ValueError("MarketMaker price must remain finite and positive.")
        impact = float(demand) / self.lam
        old_price = self.price
        self.last_fallback = False
        if self.mode == "linear":
            candidate = old_price + impact
            if candidate > 0:
                new_price = candidate
                price_change = new_price - old_price
                simple_ret = price_change / old_price
                log_ret = float(np.log(new_price / old_price))
            else:
                self.last_fallback = True
                with np.errstate(under="ignore"):
                    new_price = old_price * float(np.exp(impact))
                if new_price <= 0:
                    new_price = float(np.nextafter(0.0, 1.0))
                price_change = new_price - old_price
                log_ret = float(np.log(new_price) - np.log(old_price))
                simple_ret = float(np.expm1(log_ret))
        else:
            log_ret = impact
            with np.errstate(over="raise", under="ignore"):
                new_price = old_price * float(np.exp(impact))
            if not np.isfinite(new_price) or new_price <= 0:
                raise FloatingPointError("Log-price update overflowed or underflowed.")
            price_change = new_price - old_price
            simple_ret = float(np.expm1(log_ret))
        self.price = float(new_price)
        return self.price, float(price_change), float(simple_ret), float(log_ret)


def pick_best_strategies(scores: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Pick each agent's highest-score personal strategy; randomize exact ties."""
    best_scores = scores.max(axis=1)
    ties = scores == best_scores[:, None]
    tie_random = rng.random(scores.shape)
    tie_random[~ties] = -1.0
    columns = np.argmax(tie_random, axis=1)
    return columns, best_scores


def expected_gcomgv2_counterfactual_scores(
    actual_orders: np.ndarray,
    active: np.ndarray,
    candidate_actions: np.ndarray,
    lam: float,
    L: float = 0.0,
    tie_tolerance: float = 1e-12,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute expected GCOMGv2 score for every (agent, held-strategy) slot.

    ``actual_orders`` has shape (N,), with -1/0/+1. ``active`` marks agents
    whose chosen action actually entered the market. ``candidate_actions`` has
    shape (N,s), with +/-1 actions of each held strategy at the current mu.

    An active agent's candidate replaces its actual order. An inactive agent's
    candidate is hypothetically added to the actual order book. This preserves
    the counterfactual convention of the supplied GCOMG codes.

    Returns the score matrix and diagnostics containing D_cf, M_cf, n_cf,
    majority_mask, delta_x_cf, chi_cf, and pt_cf, all with shape (N,s).
    """
    orders = np.asarray(actual_orders, dtype=np.int8)
    active = np.asarray(active, dtype=bool)
    actions = np.asarray(candidate_actions, dtype=np.int8)
    if orders.ndim != 1 or active.shape != orders.shape:
        raise ValueError("actual_orders and active must be one-dimensional arrays of equal shape.")
    if actions.ndim != 2 or actions.shape[0] != orders.size:
        raise ValueError("candidate_actions must have shape (N,s).")
    if not np.all(np.isin(orders, (-1, 0, 1))) or not np.all(np.isin(actions, (-1, 1))):
        raise ValueError("Orders must be -1/0/+1 and candidate actions must be +/-1.")
    if not np.isfinite(lam) or lam <= 0 or not np.isfinite(L) or tie_tolerance < 0:
        raise ValueError("lam must be positive; L finite; tie_tolerance non-negative.")

    n_buy = int(np.count_nonzero(orders == 1))
    n_sell = int(np.count_nonzero(orders == -1))
    # Remove the agent's actual order iff it was active, then insert candidate.
    remove_buy = ((orders == 1) & active).astype(np.int64)[:, None]
    remove_sell = ((orders == -1) & active).astype(np.int64)[:, None]
    base_buy = n_buy - remove_buy
    base_sell = n_sell - remove_sell
    cf_buy = base_buy + (actions == 1)
    cf_sell = base_sell + (actions == -1)
    D_cf = cf_buy - cf_sell
    M_cf = np.maximum(cf_buy, cf_sell)
    n_cf = np.minimum(cf_buy, cf_sell)

    majority_direction = np.sign(D_cf)
    majority_mask = actions == majority_direction
    # A balanced book has no strict majority; sign(0)=0 makes mask false.
    delta_x_cf = D_cf.astype(np.float64) / float(lam)
    chi_cf = np.zeros_like(delta_x_cf, dtype=np.float64)
    excess = delta_x_cf - float(L)
    chi_cf[excess > tie_tolerance] = 1.0
    chi_cf[excess < -tie_tolerance] = -1.0
    pt_cf = np.abs(delta_x_cf)

    coefficient = np.divide(
        n_cf - M_cf,
        M_cf,
        out=np.zeros_like(delta_x_cf, dtype=np.float64),
        where=M_cf > 0,
    )
    scores = np.zeros_like(delta_x_cf, dtype=np.float64)
    scores[majority_mask] = (
        coefficient[majority_mask]
        * actions[majority_mask].astype(np.float64)
        * chi_cf[majority_mask]
        * pt_cf[majority_mask]
    )
    diagnostics = {
        "D_cf": D_cf.astype(np.int32),
        "M_cf": M_cf.astype(np.int32),
        "n_cf": n_cf.astype(np.int32),
        "majority_mask": majority_mask,
        "delta_x_cf": delta_x_cf,
        "chi_cf": chi_cf,
        "pt_cf": pt_cf,
    }
    return scores, diagnostics


def _public_signal_and_bit(
    demand: int, lam: float, L: float, tol: float, rng: np.random.Generator, tie_mode: str
) -> tuple[float, int]:
    excess = float(demand) / lam - L
    if excess > tol:
        return 1.0, 1
    if excess < -tol:
        return -1.0, 0
    bit = int(rng.integers(0, 2))
    signal = float(1 if rng.integers(0, 2) else -1) if tie_mode == "random" else 0.0
    return signal, bit


@dataclass
class GCOMGv2Result:
    config: GCOMGv2Config
    strategy_space: FSSStrategySpace
    demand: np.ndarray
    buy_orders: np.ndarray
    sell_orders: np.ndarray
    active_agents: np.ndarray
    majority_count: np.ndarray
    minority_count: np.ndarray
    counterfactual_majority_fraction: np.ndarray
    mean_counterfactual_score: np.ndarray
    mean_score: np.ndarray
    price: np.ndarray
    price_change: np.ndarray
    simple_returns: np.ndarray
    log_returns: np.ndarray
    global_info: np.ndarray
    appended_bit: np.ndarray
    public_score_signal: np.ndarray
    final_scores: np.ndarray
    final_holdings: np.ndarray
    final_score_strategy_ids: Optional[np.ndarray] = None
    sampled_strategy_actions: Optional[np.ndarray] = None
    price_fallback: Optional[np.ndarray] = None


def simulate_gcomgv2(
    config: GCOMGv2Config,
    strategy_space: Optional[FSSStrategySpace] = None,
) -> GCOMGv2Result:
    if strategy_space is None:
        strategy_space = build_fss(config.m)
    if getattr(strategy_space, "m", None) != config.m:
        raise ValueError("strategy_space.m must equal config.m.")
    P = 1 << config.m
    n_total_strategies = 1 << P
    config.validate(n_strategies=n_total_strategies)
    if P > 1024:
        raise ValueError("This implementation supports at most m=10 (P=1024 states).")

    seed_sequence = np.random.SeedSequence(config.seed)
    (strategy_seed,) = seed_sequence.spawn(1)
    rng = np.random.default_rng(strategy_seed)
    lam = config.resolved_lambda()
    alpha = 1.0 - 1.0 / config.T

    if strategy_space.actions is not None:
        full_actions = np.asarray(strategy_space.actions, dtype=np.int8)
        if full_actions.shape != (n_total_strategies, P):
            raise ValueError("Materialized FSS action matrix has an unexpected shape.")
        holdings = np.empty((config.N, config.s), dtype=np.int32)
        for i in range(config.N):
            holdings[i] = rng.choice(n_total_strategies, size=config.s, replace=False)
        used_ids = np.unique(holdings)
        candidate_actions = full_actions[used_ids]
        holdings_positions = np.searchsorted(used_ids, holdings)
        final_strategy_ids = None
    else:
        candidate_actions, holdings_positions, used_ids, holdings = _sample_fss_holdings(
            config.N, config.s, P, rng
        )
        final_strategy_ids = used_ids.copy()

    initial_score = (
        float(config.initial_score)
        if config.initial_score is not None
        else float(config.r + config.initial_score_offset)
    )
    scores = np.full((config.N, config.s), initial_score, dtype=np.float64)
    mu = int(rng.integers(P)) if config.mu0 is None else int(config.mu0)
    market_maker = MarketMaker(lam=lam, price=float(config.x0), mode=config.price_mode)

    steps = config.steps
    demand = np.zeros(steps, dtype=np.int32)
    buy_orders = np.zeros(steps, dtype=np.int32)
    sell_orders = np.zeros(steps, dtype=np.int32)
    active_agents = np.zeros(steps, dtype=np.int32)
    majority_count = np.zeros(steps, dtype=np.int32)
    minority_count = np.zeros(steps, dtype=np.int32)
    cf_majority_fraction = np.zeros(steps, dtype=np.float64)
    mean_cf_score = np.zeros(steps, dtype=np.float64)
    mean_score = np.zeros(steps, dtype=np.float64)
    price = np.zeros(steps + 1, dtype=np.float64)
    price_change = np.zeros(steps, dtype=np.float64)
    simple_returns = np.zeros(steps, dtype=np.float64)
    log_returns = np.zeros(steps, dtype=np.float64)
    global_info = np.zeros(steps, dtype=np.int32)
    appended_bit = np.zeros(steps, dtype=np.int8)
    public_score_signal = np.zeros(steps, dtype=np.float64)
    price_fallback = np.zeros(steps, dtype=bool)
    price[0] = market_maker.price
    agent_rows = np.arange(config.N)

    for t in range(steps):
        mu_t = mu
        chosen_columns, best_scores = pick_best_strategies(scores, rng)
        # Optional epsilon-greedy exploration: with probability epsilon, choose
        # a uniformly random strategy from the agent's own s held strategies.
        # This explores existing strategies; it does not flip an action directly.
        eps = config.strategy_exploration
        if eps > 0.0:
            explore = rng.random(config.N) < eps
            random_columns = rng.integers(config.s, size=config.N)
            chosen_columns = np.where(explore, random_columns, chosen_columns)
        active = best_scores > config.r
        held_actions_now = candidate_actions[holdings_positions, mu_t]  # N x s
        chosen_actions = held_actions_now[agent_rows, chosen_columns]
        actual_orders = chosen_actions * active.astype(np.int8)

        D = int(actual_orders.sum())
        n_buy = int(np.count_nonzero(actual_orders == 1))
        n_sell = int(np.count_nonzero(actual_orders == -1))
        M_actual = max(n_buy, n_sell)
        n_actual = min(n_buy, n_sell)

        # Evaluate every strategy in its own counterfactual book.
        score_increment, cf = expected_gcomgv2_counterfactual_scores(
            actual_orders=actual_orders,
            active=active,
            candidate_actions=held_actions_now,
            lam=lam,
            L=config.L,
            tie_tolerance=config.tie_tolerance,
        )
        scores = alpha * scores + score_increment

        public_signal, new_bit = _public_signal_and_bit(
            D, lam, config.L, config.tie_tolerance, rng, config.zero_score_tie
        )
        price_now, dx, simple_ret, log_ret = market_maker.clear(D)
        price_fallback[t] = market_maker.last_fallback
        mu = (2 * mu_t + new_bit) % P

        demand[t] = D
        buy_orders[t] = n_buy
        sell_orders[t] = n_sell
        active_agents[t] = int(active.sum())
        majority_count[t] = M_actual
        minority_count[t] = n_actual
        cf_majority_fraction[t] = float(cf["majority_mask"].mean())
        mean_cf_score[t] = float(score_increment.mean())
        mean_score[t] = float(scores.mean())
        price[t + 1] = price_now
        price_change[t] = dx
        simple_returns[t] = simple_ret
        log_returns[t] = log_ret
        global_info[t] = mu_t
        appended_bit[t] = new_bit
        public_score_signal[t] = public_signal

    return GCOMGv2Result(
        config=config,
        strategy_space=strategy_space,
        demand=demand,
        buy_orders=buy_orders,
        sell_orders=sell_orders,
        active_agents=active_agents,
        majority_count=majority_count,
        minority_count=minority_count,
        counterfactual_majority_fraction=cf_majority_fraction,
        mean_counterfactual_score=mean_cf_score,
        mean_score=mean_score,
        price=price,
        price_change=price_change,
        simple_returns=simple_returns,
        log_returns=log_returns,
        global_info=global_info,
        appended_bit=appended_bit,
        public_score_signal=public_score_signal,
        final_scores=scores.copy(),
        final_holdings=holdings.copy(),
        final_score_strategy_ids=final_strategy_ids,
        sampled_strategy_actions=candidate_actions.copy(),
        price_fallback=price_fallback,
    )


def autocorrelation_function(series: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("series must be one-dimensional with at least two observations.")
    if max_lag < 0 or max_lag >= x.size:
        raise ValueError("max_lag must be in [0, len(series)-1].")
    x = x - x.mean()
    if np.allclose(x, 0.0):
        out = np.full(max_lag + 1, np.nan)
        out[0] = 1.0
        return out
    n = x.size
    nfft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft)[:max_lag + 1]
    acov /= n
    return acov / acov[0] if acov[0] > 0 else np.full(max_lag + 1, np.nan)


def lag_autocorrelation(series: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(series):
        return float("nan")
    return float(autocorrelation_function(series, lag)[lag])


def summarize_result(result: GCOMGv2Result, burn_in: int = 10_000) -> dict[str, float]:
    if not (0 <= burn_in < result.demand.size - 3):
        raise ValueError("burn_in must leave at least four observations.")
    d = result.demand[burn_in:].astype(float)
    v = result.active_agents[burn_in:].astype(float)
    rlog = result.log_returns[burn_in:].astype(float)
    centered = rlog - rlog.mean()
    std = float(centered.std())
    kurt = float(np.mean((centered / std) ** 4) - 3.0) if std > 0 else float("nan")
    dstd = float(d.std())
    dkurt = float(np.mean(((d - d.mean()) / dstd) ** 4) - 3.0) if dstd > 0 else float("nan")
    return {
        "observations_after_burn_in": float(rlog.size),
        "mean_demand": float(d.mean()),
        "std_demand": dstd,
        "excess_kurtosis_demand": dkurt,
        "mean_active_agents": float(v.mean()),
        "std_active_agents": float(v.std()),
        "mean_majority_side_agents": float(result.majority_count[burn_in:].mean()),
        "mean_minority_side_agents": float(result.minority_count[burn_in:].mean()),
        "mean_counterfactual_majority_fraction": float(result.counterfactual_majority_fraction[burn_in:].mean()),
        "mean_counterfactual_score_increment": float(result.mean_counterfactual_score[burn_in:].mean()),
        "mean_strategy_score": float(result.final_scores.mean()),
        "max_strategy_score": float(result.final_scores.max()),
        "mean_log_return": float(rlog.mean()),
        "std_log_return": float(rlog.std()),
        "excess_kurtosis_log_return": kurt,
        "acf_return_lag1": lag_autocorrelation(rlog, 1),
        "acf_abs_return_lag1": lag_autocorrelation(np.abs(centered), 1),
        "acf_abs_return_lag2": lag_autocorrelation(np.abs(centered), 2),
        "acf_abs_return_lag5": lag_autocorrelation(np.abs(centered), 5),
        "acf_abs_return_lag10": lag_autocorrelation(np.abs(centered), 10),
        "acf_abs_return_lag20": lag_autocorrelation(np.abs(centered), 20),
        "acf_abs_return_lag50": lag_autocorrelation(np.abs(centered), 50),
        "acf_volume_lag1": lag_autocorrelation(v, 1),
        "acf_volume_lag5": lag_autocorrelation(v, 5),
        "acf_volume_lag10": lag_autocorrelation(v, 10),
    }


def save_csv(result: GCOMGv2Result, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "t", "mu_t", "new_bit", "demand", "buy_orders", "sell_orders",
        "active_agents", "majority_count", "minority_count",
        "counterfactual_majority_fraction", "mean_counterfactual_score",
        "mean_score", "price", "price_change", "simple_return", "log_return",
        "public_score_signal",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in range(result.demand.size):
            writer.writerow([
                t, int(result.global_info[t]), int(result.appended_bit[t]),
                int(result.demand[t]), int(result.buy_orders[t]), int(result.sell_orders[t]),
                int(result.active_agents[t]), int(result.majority_count[t]),
                int(result.minority_count[t]),
                f"{result.counterfactual_majority_fraction[t]:.12g}",
                f"{result.mean_counterfactual_score[t]:.12g}",
                f"{result.mean_score[t]:.12g}", f"{result.price[t+1]:.12g}",
                f"{result.price_change[t]:.12g}", f"{result.simple_returns[t]:.12g}",
                f"{result.log_returns[t]:.12g}", f"{result.public_score_signal[t]:.1f}",
            ])


def save_diagnostics_plot(
    result: GCOMGv2Result, path: Path, burn_in: int = 10_000, show_steps: int = 3_000
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import pi, sqrt

    if not (0 <= burn_in < result.demand.size - 3):
        raise ValueError("burn_in must leave enough observations for diagnostics.")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    nprice = min(show_steps, result.price.size)
    axes[0, 0].plot(np.arange(nprice), result.price[:nprice], linewidth=1.0)
    axes[0, 0].set(title="Market price", xlabel="t", ylabel="x(t)")
    nvol = min(show_steps, result.active_agents.size)
    axes[0, 1].plot(np.arange(nvol), result.active_agents[:nvol], linewidth=0.8)
    axes[0, 1].set(title="Trading activity", xlabel="t", ylabel="active agents")

    returns = result.log_returns[burn_in:]
    rc = returns - returns.mean()
    sd = float(rc.std())
    z = rc / sd if sd > 0 else rc
    bins = np.linspace(-6, 6, 81)
    axes[1, 0].hist(z, bins=bins, density=True, alpha=0.65, label="GCOMGv2")
    grid = np.linspace(-6, 6, 500)
    axes[1, 0].plot(grid, np.exp(-grid**2 / 2) / sqrt(2*pi), linewidth=1.5, label="Gaussian")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(title="Standardized log-return distribution", xlabel="z", ylabel="density")
    axes[1, 0].legend(frameon=False)

    max_lag = min(100, len(returns)-1)
    lags = np.arange(1, max_lag + 1)
    acf_r = autocorrelation_function(rc, max_lag)[1:]
    acf_abs = autocorrelation_function(np.abs(rc), max_lag)[1:]
    axes[1, 1].plot(lags, acf_r, label="return")
    axes[1, 1].plot(lags, acf_abs, label="|return|")
    axes[1, 1].axhline(0, linewidth=0.7)
    axes[1, 1].set(title="Autocorrelation", xlabel="lag", ylabel="ACF")
    axes[1, 1].legend(frameon=False)
    fig.suptitle(
        f"GCOMGv2-FSS: N={result.config.N}, m={result.config.m}, s={result.config.s}, "
        f"r={result.config.r:g}, T={result.config.T:g}, L={result.config.L:g}", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_price_change_distribution_plot(
    result: GCOMGv2Result, path: Path, burn_in: int = 10_000, bins: int = 80
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import pi, sqrt
    from statistics import NormalDist

    if not (0 <= burn_in < result.demand.size - 3):
        raise ValueError("burn_in must leave at least four observations.")
    dx = np.diff(result.price)[burn_in:]
    mean, std = float(dx.mean()), float(dx.std())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    if std > 0:
        _, edges, _ = axes[0].hist(dx, bins=bins, density=True, alpha=0.7, label="price increments")
        grid = np.linspace(edges[0], edges[-1], 500)
        axes[0].plot(grid, np.exp(-0.5*((grid-mean)/std)**2)/(std*sqrt(2*pi)), label="Gaussian")
    else:
        axes[0].axvline(mean, label="price increments")
    axes[0].set(title="PDF of price increments", xlabel="x(t+1)-x(t)", ylabel="density")
    axes[0].legend(frameon=False)
    ordered = np.sort(dx)
    n = len(ordered)
    indices = np.unique(np.linspace(0, n-1, min(n, 5000)).round().astype(int))
    probs = (indices + 0.5) / n
    normal = NormalDist()
    theoretical = np.fromiter((normal.inv_cdf(float(p)) for p in probs), float, count=len(probs))
    axes[1].scatter(theoretical, ordered[indices], s=7, alpha=0.55)
    if std > 0:
        q = np.linspace(theoretical.min(), theoretical.max(), 200)
        axes[1].plot(q, mean + std*q, "--", linewidth=1.5)
    axes[1].set(title="Normal Q-Q plot", xlabel="normal quantile", ylabel="price-increment quantile")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GCOMGv2-FSS expected-score simulator")
    parser.add_argument("--N", type=int, default=501)
    parser.add_argument("--m", type=int, default=3, help="public-information memory length")
    parser.add_argument("--s", type=int, default=2, help="strategies held by each agent")
    parser.add_argument("--r", type=float, default=4.0, help="finite strategy-score entry threshold")
    parser.add_argument("--T", type=float, default=100.0, help="finite score-memory time scale")
    parser.add_argument("--L", type=float, default=0.0, help="market reference threshold")
    parser.add_argument("--lambda", dest="lam", type=float, default=None, help="market depth; default 10*N")
    parser.add_argument("--x0", type=float, default=100.0)
    parser.add_argument("--price-mode", choices=("linear", "log"), default="linear")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mu0", type=int, default=None)
    parser.add_argument("--initial-score-offset", type=float, default=10.0)
    parser.add_argument("--initial-score", type=float, default=None)
    parser.add_argument("--zero-score-tie", choices=("random", "zero"), default="random")
    parser.add_argument(
        "--strategy-exploration", type=float, default=0.0,
        help="epsilon probability of choosing a random strategy among each agent's held s strategies (default 0)",
    )
    parser.add_argument("--burn-in", type=int, default=10_000)
    parser.add_argument("--outdir", type=Path, default=Path("gcomgv2_output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GCOMGv2Config(
        N=args.N, m=args.m, s=args.s, r=args.r, T=args.T, L=args.L,
        lam=args.lam, x0=args.x0, price_mode=args.price_mode, steps=args.steps,
        seed=args.seed, mu0=args.mu0, initial_score_offset=args.initial_score_offset,
        initial_score=args.initial_score, zero_score_tie=args.zero_score_tie,
        strategy_exploration=args.strategy_exploration,
    )
    space = build_fss(config.m)
    result = simulate_gcomgv2(config, space)
    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "gcomgv2_timeseries.csv"
    plot_path = args.outdir / "gcomgv2_diagnostics.png"
    dx_plot_path = args.outdir / "gcomgv2_price_change_distribution.png"
    summary_path = args.outdir / "gcomgv2_summary.json"
    save_csv(result, csv_path)
    save_diagnostics_plot(result, plot_path, burn_in=args.burn_in)
    save_price_change_distribution_plot(result, dx_plot_path, burn_in=args.burn_in)
    summary = summarize_result(result, burn_in=args.burn_in)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "parameters": asdict(config),
            "lambda_used": config.resolved_lambda(),
            "initial_score_used": (config.initial_score if config.initial_score is not None
                                    else config.r + config.initial_score_offset),
            "diagnostics": summary,
        }, f, ensure_ascii=False, indent=2)
    if space.actions is not None:
        print(f"FSS: P={space.P}, number of strategies={space.n_strategies}")
    else:
        print(f"FSS: P={space.P}, total strategies=2^{space.P}; sampled unique strategies={len(result.sampled_strategy_actions)}")
    print(f"lambda={config.resolved_lambda():g}; alpha={1-1/config.T:.6g}; redistribution=OFF")
    print("Diagnostics (after burn-in):")
    for key, value in summary.items():
        print(f"  {key:42s}: {value:.6g}")
    print(f"Saved: {csv_path}\nSaved: {plot_path}\nSaved: {dx_plot_path}\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
