#!/usr/bin/env python3
"""GCMG simulator using an implicitly represented Full Strategy Space (FSS).

The FSS contains all 2**P deterministic +/-1 maps on P=2**m public states.
For m>4 this program never constructs that astronomically large pool.  It
samples only the N*s strategies actually held by agents, uniformly from FSS,
with no duplicates within an agent's own strategy set.  Strategies may be
shared by different agents, as in independent agent-by-agent sampling.

The model dynamics remain those of the previous GCMG-FSS v2.  Linear price
updates are used normally; only a step whose linear update would make price
non-positive is handled with the logarithmic price rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
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
    """Descriptor of the complete FSS; ``actions`` is materialized only m<=4.

    The complete strategy count is 2**(2**m).  For m>4, ``actions`` is None:
    the complete pool is represented implicitly and strategies are sampled
    directly as length-P independent fair bit strings.
    """

    m: int
    actions: Optional[np.ndarray] = None

    @property
    def P(self) -> int:
        return 1 << self.m

    @property
    def n_strategies(self) -> int:
        # Python integers can represent 2**1024 and larger exactly.
        return 1 << self.P

    @property
    def is_materialized(self) -> bool:
        return self.actions is not None


def build_fss(m: int, verify: bool = True) -> FSSStrategySpace:
    """Describe the FSS; materialize its matrix only for 1<=m<=4.

    For m>=5 the full strategy pool is far too large to store.  A single
    strategy is simply a uniformly random P-bit vector, so the simulator can
    sample each agent's holdings without enumerating the full pool.
    """
    if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or m < 1:
        raise ValueError("m must be an integer >= 1.")
    m = int(m)
    if m > MAX_FSS_M:
        raise ValueError(
            f"This implementation supports 1 <= m <= {MAX_FSS_M}; got m={m}."
        )

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
            raise RuntimeError("FSS construction failed: actions must be +/-1.")
        if not np.all(actions[0] == -1) or not np.all(actions[-1] == 1):
            raise RuntimeError("FSS construction failed: binary encoding is inconsistent.")
    return FSSStrategySpace(m=m, actions=actions)


def _bits_to_python_ids(bits: np.ndarray) -> list[int]:
    """Encode rows of 0/1 bits as exact Python integer FSS strategy IDs."""
    packed = np.packbits(np.asarray(bits, dtype=np.uint8), axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), byteorder="little", signed=False) for row in packed]


def _sample_fss_holdings(
    N: int,
    s: int,
    P: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly sample s distinct FSS strategies per agent without enumerating FSS.

    Returns (unique_actions, holdings_indices, unique_strategy_ids,
    holdings_strategy_ids).  Different agents sample independently, so they
    may share strategies.  Each individual agent's s strategies are distinct.

    Rejection sampling of independent fair P-bit strings is exactly uniform
    sampling without replacement from the FSS for each agent.
    """
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

    unique_bits, inverse = np.unique(
        held_bits.reshape(N * s, P), axis=0, return_inverse=True
    )
    unique_actions = (2 * unique_bits.astype(np.int8) - 1).astype(np.int8, copy=False)
    unique_id_list = _bits_to_python_ids(unique_bits)

    # Use compact numeric dtypes whenever possible; Python ints are required
    # once a strategy ID is wider than 64 bits (m>=7).
    if P <= 64:
        unique_ids = np.asarray(unique_id_list, dtype=np.uint64)
        holdings_ids = np.asarray(held_ids, dtype=np.uint64)
    else:
        unique_ids = np.asarray(unique_id_list, dtype=object)
        holdings_ids = np.asarray(held_ids, dtype=object)

    holdings_indices = inverse.reshape(N, s).astype(np.int32, copy=False)
    return unique_actions, holdings_indices, unique_ids, holdings_ids


@dataclass
class GCMGConfig:
    # Parameters explicitly listed on lecture p.25
    N: int = 501
    m: int = 3
    s: int = 2
    r: float = 4.0
    T: float = 100.0
    L: float = 0.0

    # Price formation also contains lambda on p.25; it is included explicitly.
    lam: Optional[float] = None  # None -> 10*N, a numerical depth convention
    x0: float = 100.0
    price_mode: str = "linear"  # "linear" follows p.25; "log" follows p.14 alternative

    steps: int = 100_000
    seed: int = 123
    mu0: Optional[int] = None
    initial_score_offset: float = 10.0  # default: initialize scores at r + offset
    initial_score: Optional[float] = None  # optional absolute override (e.g. 0)
    zero_score_tie: str = "zero"  # "random" by default; or literal sign(0)=0
    tie_tolerance: float = 1e-12

    def resolved_lambda(self) -> float:
        value = 10.0 * self.N if self.lam is None else float(self.lam)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("lambda (lam) must be a finite positive number.")
        return value

    def validate(self, n_strategies: Optional[int] = None) -> None:
        if self.N < 1:
            raise ValueError("N must be >= 1.")
        if self.m < 1:
            raise ValueError("m must be >= 1.")
        if self.s < 1:
            raise ValueError("s must be >= 1.")
        if self.T < 1:
            raise ValueError("T must be >= 1 so alpha=1-1/T is in [0,1).")
        if self.steps < 1:
            raise ValueError("steps must be >= 1.")
        if self.x0 <= 0 or not np.isfinite(self.x0):
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
        self.resolved_lambda()
        if n_strategies is not None and self.s > n_strategies:
            raise ValueError(
                f"s={self.s} exceeds the available strategy count {n_strategies}."
            )
        if self.mu0 is not None and not (0 <= self.mu0 < (1 << self.m)):
            raise ValueError(f"mu0 must lie in [0, {(1 << self.m) - 1}].")




@dataclass
class MarketMaker:
    """Market maker with linear updates and a one-step boundary fallback.

    With mode='linear', the original x_new=x_old+D/lambda update is used
    whenever it gives a positive price.  If and only if that update would be
    non-positive, this single step uses x_new=x_old*exp(D/lambda).  The next
    step returns to the linear rule automatically.
    """

    lam: float
    price: float
    mode: str = "linear"
    last_fallback: bool = field(default=False, init=False)

    def clear(self, excess_demand: int) -> tuple[float, float, float, float]:
        if self.lam <= 0 or self.price <= 0 or not np.isfinite(self.price):
            raise ValueError("MarketMaker requires finite positive lambda and price.")
        impact = float(excess_demand) / self.lam
        old_price = self.price
        self.last_fallback = False

        if self.mode == "linear":
            new_price = old_price + impact
            if new_price > 0.0:
                # Keep these operations exactly as in the previous linear code.
                simple_ret = impact / old_price
                log_ret = float(np.log(new_price / old_price))
                price_change = impact
            else:
                # Boundary-only fallback to the lecture's logarithmic rule:
                # log(x_new)-log(x_old)=D/lambda.
                self.last_fallback = True
                with np.errstate(under="ignore"):
                    new_price = old_price * float(np.exp(impact))
                if new_price <= 0.0:
                    # Preserve strict positivity in extreme underflow cases.
                    new_price = float(np.nextafter(0.0, 1.0))
                    log_ret = float(np.log(new_price) - np.log(old_price))
                    simple_ret = float(np.expm1(log_ret))
                else:
                    log_ret = impact
                    simple_ret = float(np.expm1(impact))
                # This array/CSV field is the actual price increment on the
                # fallback step; ordinary linear steps remain exactly D/lambda.
                price_change = new_price - old_price
        elif self.mode == "log":
            # Explicit log mode remains unchanged from the previous version.
            log_ret = impact
            simple_ret = float(np.expm1(impact))
            new_price = old_price * float(np.exp(impact))
            price_change = impact
        else:
            raise ValueError("MarketMaker mode must be 'linear' or 'log'.")

        self.price = new_price
        return new_price, price_change, simple_ret, log_ret


@dataclass
class GCMGResult:
    config: GCMGConfig
    strategy_space: FSSStrategySpace
    demand: np.ndarray
    buy_orders: np.ndarray
    sell_orders: np.ndarray
    active_agents: np.ndarray
    price: np.ndarray
    price_change: np.ndarray
    simple_returns: np.ndarray
    log_returns: np.ndarray
    global_info: np.ndarray
    appended_bit: np.ndarray
    score_signal: np.ndarray
    final_scores: np.ndarray
    final_holdings: np.ndarray
    # For m>4, final_scores contains the exact scores only for sampled/held
    # strategies; these IDs map score rows back to the implicit full FSS.
    final_score_strategy_ids: Optional[np.ndarray] = None
    sampled_strategy_actions: Optional[np.ndarray] = None
    price_fallback: Optional[np.ndarray] = None


def _pick_best_strategies(
    scores: np.ndarray,
    holdings: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick each agent's highest-score strategy; randomize exact score ties."""
    held_scores = scores[holdings]  # (N, s)
    best_scores = held_scores.max(axis=1)
    tie_mask = held_scores == best_scores[:, None]
    tie_random = rng.random(holdings.shape)
    tie_random[~tie_mask] = -1.0
    chosen_columns = np.argmax(tie_random, axis=1)
    chosen_strategies = holdings[np.arange(holdings.shape[0]), chosen_columns]
    return chosen_strategies, best_scores


def _reconstruct_all_strategy_scores(
    actions: np.ndarray,
    global_info: np.ndarray,
    score_signal: np.ndarray,
    initial_score: float,
    alpha: float,
) -> np.ndarray:
    """Reconstruct the exact final scores of *all* FSS strategies.

    The simulation only needs to update strategies held by at least one agent,
    because holdings are quenched.  However, the lecture's score equation also
    updates unheld strategies.  Their scores therefore cannot be reconstructed
    by pure decay alone.  We use the closed-form solution

        S_R(n) = alpha**n S_R(0)
                 - sum_t alpha**(n-1-t) a_R[mu_t] q_t,

    and group the weighted score signals by public-information state.  This
    gives every FSS strategy's exact score with one matrix-vector product.
    """
    n_steps = int(global_info.size)
    powers = np.arange(n_steps - 1, -1, -1, dtype=np.int64)
    weighted_signal = np.asarray(score_signal, dtype=np.float64) * np.power(alpha, powers)
    signal_by_state = np.bincount(
        np.asarray(global_info, dtype=np.int64),
        weights=weighted_signal,
        minlength=actions.shape[1],
    )
    return (
        (alpha ** n_steps) * float(initial_score)
        - np.asarray(actions, dtype=np.float64) @ signal_by_state
    )


def simulate_gcmg(
    config: GCMGConfig,
    strategy_space: Optional[FSSStrategySpace] = None,
) -> GCMGResult:
    """Run GCMG with a complete FSS represented explicitly or by sparse sampling."""
    if strategy_space is None:
        strategy_space = build_fss(config.m)
    if getattr(strategy_space, "m", None) != config.m:
        raise ValueError("strategy_space.m must equal config.m.")

    P = 1 << config.m
    n_total_strategies = 1 << P
    config.validate(n_strategies=n_total_strategies)
    if P > 1024:
        raise ValueError("This project supports at most m=10 (P=1024 public states).")

    rng = np.random.default_rng(config.seed)
    lam = config.resolved_lambda()
    alpha = 1.0 - 1.0 / config.T

    # For m<=4, retain the previous exact RNG path and materialized matrix.
    # This preserves old seeded trajectories and every existing full-score
    # output when no price-boundary fallback is triggered.
    full_actions = strategy_space.actions
    if full_actions is not None:
        actions = np.asarray(full_actions, dtype=np.int8)
        if actions.shape != (n_total_strategies, P):
            raise ValueError(
                f"Materialized FSS actions must have shape ({n_total_strategies}, {P})."
            )
        if not np.all(np.isin(actions, (-1, 1))):
            raise ValueError("All FSS actions must be -1 or +1.")

        holdings = np.empty((config.N, config.s), dtype=np.int32)
        for i in range(config.N):
            holdings[i] = rng.choice(n_total_strategies, size=config.s, replace=False)
        used_strategy_ids = np.unique(holdings)
        candidate_actions = actions[used_strategy_ids]
        holdings_positions = np.searchsorted(used_strategy_ids, holdings)
        candidate_strategy_ids = used_strategy_ids
    else:
        # For m>=5, FSS is implicit.  Generate only the strategies actually
        # held; this costs O(N*s*P), not O(2**P * P).
        candidate_actions, holdings_positions, candidate_strategy_ids, holdings = (
            _sample_fss_holdings(config.N, config.s, P, rng)
        )

    initial_score_value = (
        float(config.initial_score)
        if config.initial_score is not None
        else config.r + config.initial_score_offset
    )
    candidate_scores = np.full(
        candidate_strategy_ids.size, initial_score_value, dtype=np.float64
    )
    mu = int(rng.integers(P)) if config.mu0 is None else int(config.mu0)
    market_maker = MarketMaker(
        lam=lam, price=float(config.x0), mode=config.price_mode
    )

    steps = config.steps
    demand = np.zeros(steps, dtype=np.int32)
    buy_orders = np.zeros(steps, dtype=np.int32)
    sell_orders = np.zeros(steps, dtype=np.int32)
    active_agents = np.zeros(steps, dtype=np.int32)
    price = np.zeros(steps + 1, dtype=np.float64)
    price_change = np.zeros(steps, dtype=np.float64)
    simple_returns = np.zeros(steps, dtype=np.float64)
    log_returns = np.zeros(steps, dtype=np.float64)
    global_info = np.zeros(steps, dtype=np.int32)
    appended_bit = np.zeros(steps, dtype=np.int8)
    score_signal_series = np.zeros(steps, dtype=np.float64)
    price_fallback = np.zeros(steps, dtype=bool)
    price[0] = market_maker.price

    for t in range(steps):
        mu_t = mu
        best_strategy, best_score = _pick_best_strategies(
            candidate_scores, holdings_positions, rng
        )

        # Grand-canonical entry rule: only agents with best score > r trade.
        active = best_score > config.r
        chosen_actions = candidate_actions[best_strategy, mu_t]
        agent_orders = chosen_actions * active.astype(np.int8)

        D = int(agent_orders.sum())
        n_buy = int(np.count_nonzero(agent_orders == 1))
        n_sell = int(np.count_nonzero(agent_orders == -1))
        q = D / lam
        excess_vs_reference = q - config.L

        if excess_vs_reference > config.tie_tolerance:
            score_signal = 1.0
            new_bit = 1
        elif excess_vs_reference < -config.tie_tolerance:
            score_signal = -1.0
            new_bit = 0
        else:
            new_bit = int(rng.integers(0, 2))
            if config.zero_score_tie == "random":
                score_signal = float(1 if rng.integers(0, 2) else -1)
            else:
                score_signal = 0.0

        # Exact online score updates for every strategy that is held by at
        # least one agent. Unheld strategies cannot affect this quenched run.
        candidate_scores = (
            alpha * candidate_scores
            - candidate_actions[:, mu_t].astype(np.float64) * score_signal
        )

        price_now, impact_or_price_change, simple_ret, log_ret = market_maker.clear(D)
        price_fallback[t] = market_maker.last_fallback

        mu = (2 * mu_t + new_bit) % P

        demand[t] = D
        buy_orders[t] = n_buy
        sell_orders[t] = n_sell
        active_agents[t] = int(active.sum())
        price[t + 1] = price_now
        price_change[t] = impact_or_price_change
        simple_returns[t] = simple_ret
        log_returns[t] = log_ret
        global_info[t] = mu_t
        appended_bit[t] = new_bit
        score_signal_series[t] = score_signal

    if full_actions is not None:
        # Preserve the old m<=4 output: exact final scores of every FSS row.
        final_scores = _reconstruct_all_strategy_scores(
            np.asarray(full_actions, dtype=np.int8),
            global_info, score_signal_series, initial_score_value, alpha,
        )
        final_score_strategy_ids = None
    else:
        # For m>=5 the complete score vector is itself astronomically large.
        # Return exact scores for all sampled strategies, with their FSS IDs.
        final_scores = candidate_scores.copy()
        final_score_strategy_ids = candidate_strategy_ids.copy()

    return GCMGResult(
        config=config,
        strategy_space=strategy_space,
        demand=demand,
        buy_orders=buy_orders,
        sell_orders=sell_orders,
        active_agents=active_agents,
        price=price,
        price_change=price_change,
        simple_returns=simple_returns,
        log_returns=log_returns,
        global_info=global_info,
        appended_bit=appended_bit,
        score_signal=score_signal_series,
        final_scores=final_scores,
        final_holdings=holdings.copy(),
        final_score_strategy_ids=final_score_strategy_ids,
        sampled_strategy_actions=candidate_actions.copy(),
        price_fallback=price_fallback,
    )


def autocorrelation_function(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Fast normalized ACF using an FFT (lags 0..max_lag)."""
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("series must be a one-dimensional array with at least 2 points")
    if max_lag < 0 or max_lag >= x.size:
        raise ValueError("max_lag must be in [0, len(series)-1]")
    x = x - x.mean()
    if np.allclose(x, 0.0):
        out = np.full(max_lag + 1, np.nan)
        out[0] = 1.0
        return out
    n = x.size
    nfft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft)[:max_lag + 1]
    acov /= n  # biased estimator; consistent across all requested lags
    if acov[0] <= 0:
        return np.full(max_lag + 1, np.nan)
    return acov / acov[0]


def lag_autocorrelation(series: np.ndarray, lag: int) -> float:
    """Pearson autocorrelation at a positive integer lag."""
    x = np.asarray(series, dtype=np.float64)
    if lag <= 0 or lag >= x.size:
        return float("nan")
    return float(autocorrelation_function(x, lag)[lag])


def summarize_result(result: GCMGResult, burn_in: int = 10_000) -> dict[str, float]:
    """Compute basic stylized-fact diagnostics after discarding burn-in."""
    if not (0 <= burn_in < result.demand.size - 3):
        raise ValueError("burn_in must be >=0 and leave at least 4 observations.")
    r = np.asarray(result.log_returns[burn_in:], dtype=np.float64)
    d = np.asarray(result.demand[burn_in:], dtype=np.float64)
    v = np.asarray(result.active_agents[burn_in:], dtype=np.float64)

    r_centered = r - r.mean()
    r_std = r_centered.std()
    if r_std > 0:
        z = r_centered / r_std
        excess_kurtosis = float(np.mean(z**4) - 3.0)
    else:
        excess_kurtosis = float("nan")

    demand_std = float(d.std())
    demand_kurtosis = (
        float(np.mean(((d - d.mean()) / demand_std) ** 4) - 3.0)
        if demand_std > 0 else float("nan")
    )
    summary: dict[str, float] = {
        "observations_after_burn_in": float(r.size),
        "mean_demand": float(d.mean()),
        "std_demand": demand_std,
        "excess_kurtosis_demand": demand_kurtosis,
        "mean_active_agents": float(v.mean()),
        "std_active_agents": float(v.std()),
        "mean_log_return": float(r.mean()),
        "std_log_return": float(r.std()),
        "excess_kurtosis_log_return": excess_kurtosis,
        "acf_return_lag1": lag_autocorrelation(r, 1),
        "acf_abs_return_lag1": lag_autocorrelation(np.abs(r_centered), 1),
        "acf_abs_return_lag2": lag_autocorrelation(np.abs(r_centered), 2),
        "acf_abs_return_lag5": lag_autocorrelation(np.abs(r_centered), 5),
        "acf_abs_return_lag10": lag_autocorrelation(np.abs(r_centered), 10),
        "acf_volume_lag1": lag_autocorrelation(v, 1),
        "acf_volume_lag2": lag_autocorrelation(v, 2),
        "acf_volume_lag5": lag_autocorrelation(v, 5),
        "acf_volume_lag10": lag_autocorrelation(v, 10),
    }
    return summary


def save_csv(result: GCMGResult, path: Path) -> None:
    """Write one row per market round; price[t+1] is after that round's order flow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "t", "mu_t", "new_bit", "demand", "buy_orders", "sell_orders",
        "active_agents", "price", "price_change", "simple_return",
        "log_return", "score_signal",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in range(result.demand.size):
            writer.writerow([
                t,
                int(result.global_info[t]),
                int(result.appended_bit[t]),
                int(result.demand[t]),
                int(result.buy_orders[t]),
                int(result.sell_orders[t]),
                int(result.active_agents[t]),
                f"{result.price[t + 1]:.12g}",
                f"{result.price_change[t]:.12g}",
                f"{result.simple_returns[t]:.12g}",
                f"{result.log_returns[t]:.12g}",
                f"{result.score_signal[t]:.1f}",
            ])


def save_diagnostics_plot(
    result: GCMGResult,
    path: Path,
    burn_in: int = 10_000,
    show_steps: int = 3_000,
) -> None:
    """Save price/activity, return-tail and autocorrelation diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import sqrt, pi

    if not (0 <= burn_in < result.demand.size - 3):
        raise ValueError("burn_in must leave enough observations for diagnostics.")

    r = result.log_returns[burn_in:]
    rc = r - r.mean()
    rstd = rc.std()
    z = rc / rstd if rstd > 0 else rc
    lags = np.arange(1, 101)
    acf_r = autocorrelation_function(rc, 100)[1:]
    acf_abs = autocorrelation_function(np.abs(rc), 100)[1:]
    volume = result.active_agents[burn_in:].astype(float)
    acf_vol = autocorrelation_function(volume, 100)[1:]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    ax = axes[0, 0]
    t_price = np.arange(min(show_steps, result.price.size))
    ax.plot(t_price, result.price[:t_price.size], linewidth=1.0)
    ax.set_title("Market price")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")

    ax = axes[0, 1]
    t_vol = np.arange(min(show_steps, result.active_agents.size))
    ax.plot(t_vol, result.active_agents[:t_vol.size], linewidth=0.8)
    ax.set_title("Trading activity (number of active agents)")
    ax.set_xlabel("t")
    ax.set_ylabel("V(t)")

    ax = axes[1, 0]
    bins = np.linspace(-6, 6, 81)
    ax.hist(z, bins=bins, density=True, alpha=0.65, label="GCMG log returns")
    grid = np.linspace(-6, 6, 500)
    normal_pdf = np.exp(-0.5 * grid**2) / sqrt(2 * pi)
    ax.plot(grid, normal_pdf, linewidth=1.6, label="Standard normal")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-5)
    ax.set_title("Standardized return distribution (log scale)")
    ax.set_xlabel("z = (r - mean) / std")
    ax.set_ylabel("Probability density")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(lags, acf_r, label="return")
    ax.plot(lags, acf_abs, label="|return|")
    ax.plot(lags, acf_vol, label="activity V(t)")
    ax.axhline(0.0, linewidth=0.7)
    ax.set_title("Autocorrelation functions")
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    ax.legend(frameon=False)

    fig.suptitle(
        f"GCMG-FSS: N={result.config.N}, m={result.config.m}, s={result.config.s}, "
        f"r={result.config.r:g}, T={result.config.T:g}, L={result.config.L:g}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_price_change_distribution_plot(
    result: GCMGResult,
    path: Path,
    burn_in: int = 10_000,
    bins: int = 80,
    max_qq_points: int = 5_000,
) -> None:
    """Save raw price-increment PDF and normal Q-Q plot (lecture p.27 style).

    The horizontal variable is the actual x[t+1]-x[t], not a standardized
    return.  The Gaussian reference uses the sample mean and standard deviation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import sqrt, pi
    from statistics import NormalDist

    if not (0 <= burn_in < result.demand.size - 3):
        raise ValueError("burn_in must be >=0 and leave at least 4 observations.")
    if bins < 5:
        raise ValueError("bins must be >= 5.")
    if max_qq_points < 100:
        raise ValueError("max_qq_points must be >= 100.")

    # In linear mode this equals D/lambda. Computing it from price also makes
    # the label correct when price_mode='log'.
    delta_x = np.diff(result.price)[burn_in:]
    delta_x = delta_x[np.isfinite(delta_x)]
    if delta_x.size < 4:
        raise ValueError("Not enough finite price increments to plot.")

    mean = float(delta_x.mean())
    std = float(delta_x.std())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) PDF of raw price increments.
    ax = axes[0]
    if std > 0:
        _, edges, _ = ax.hist(
            delta_x, bins=bins, density=True, alpha=0.7,
            label="GCMG price increments",
        )
        grid = np.linspace(float(edges[0]), float(edges[-1]), 600)
        gaussian_pdf = np.exp(-0.5 * ((grid - mean) / std) ** 2) / (std * sqrt(2 * pi))
        ax.plot(grid, gaussian_pdf, linewidth=1.7, label="Gaussian with same mean/std")
    else:
        ax.axvline(mean, linewidth=2, label="GCMG price increments")
    ax.set_title("PDF of price increments")
    ax.set_xlabel(r"Price increment $\Delta x=x_{t+1}-x_t$")
    ax.set_ylabel("Probability density")
    ax.legend(frameon=False)

    # (b) Normal Q-Q plot, using standard-library inverse normal CDF.
    ax = axes[1]
    ordered = np.sort(delta_x)
    n = ordered.size
    n_plot = min(n, max_qq_points)
    indices = np.unique(np.linspace(0, n - 1, n_plot).round().astype(int))
    sample_quantiles = ordered[indices]
    probabilities = (indices + 0.5) / n
    normal = NormalDist()
    theoretical_quantiles = np.fromiter(
        (normal.inv_cdf(float(prob)) for prob in probabilities),
        dtype=np.float64, count=probabilities.size,
    )
    ax.scatter(theoretical_quantiles, sample_quantiles, s=7, alpha=0.55,
               label="Empirical quantiles")
    if std > 0:
        qline = np.linspace(float(theoretical_quantiles.min()),
                            float(theoretical_quantiles.max()), 200)
        ax.plot(qline, mean + std * qline, "--", linewidth=1.7,
                label="Gaussian reference")
    ax.set_title("Normal Q-Q plot")
    ax.set_xlabel("Quantiles of standard normal distribution")
    ax.set_ylabel(r"Quantiles of $\Delta x$")
    ax.legend(frameon=False)

    fig.suptitle(
        f"GCMG-FSS price increments: N={result.config.N}, m={result.config.m}, "
        f"s={result.config.s}, lambda={result.config.resolved_lambda():g}, "
        f"burn-in={burn_in}", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate the GCMG with the Full Strategy Space (FSS).")
    parser.add_argument("--N", type=int, default=501, help="number of agents")
    parser.add_argument("--m", type=int, default=3, help="global-information memory length")
    parser.add_argument("--s", type=int, default=2, help="strategies held by each agent")
    parser.add_argument("--r", type=float, default=4.0, help="strategy-score participation threshold")
    parser.add_argument("--T", type=float, default=100.0, help="score memory/effective period")
    parser.add_argument("--L", type=float, default=0.0, help="market reference threshold")
    parser.add_argument("--lambda", dest="lam", type=float, default=None, help="market depth; default 10*N")
    parser.add_argument("--x0", type=float, default=100.0, help="initial price")
    parser.add_argument("--price-mode", choices=("linear", "log"), default="linear")
    parser.add_argument("--steps", type=int, default=100_000, help="number of market rounds")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--mu0", type=int, default=None, help="initial m-bit public-information state")
    parser.add_argument(
        "--initial-score-offset", type=float, default=10.0,
        help="initialize every strategy score at r + offset (default: 10)",
    )
    parser.add_argument(
        "--initial-score", type=float, default=None,
        help="absolute initial score for all strategies; overrides --initial-score-offset",
    )
    parser.add_argument(
        "--zero-score-tie", choices=("random", "zero"), default="random",
        help="score signal at D/lambda=L: literal sign(0)=0 (default) or random regularization",
    )
    parser.add_argument("--burn-in", type=int, default=10_000, help="discarded transient for diagnostics")
    parser.add_argument("--outdir", type=Path, default=Path("gcmg_output"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = GCMGConfig(
        N=args.N,
        m=args.m,
        s=args.s,
        r=args.r,
        T=args.T,
        L=args.L,
        lam=args.lam,
        x0=args.x0,
        price_mode=args.price_mode,
        steps=args.steps,
        seed=args.seed,
        mu0=args.mu0,
        initial_score_offset=args.initial_score_offset,
        initial_score=args.initial_score,
        zero_score_tie=args.zero_score_tie,
    )
    strategy_space = build_fss(config.m)
    result = simulate_gcmg(config, strategy_space)

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "gcmg_timeseries.csv"
    plot_path = args.outdir / "gcmg_diagnostics.png"
    distribution_plot_path = args.outdir / "gcmg_price_change_distribution.png"
    save_csv(result, csv_path)
    save_diagnostics_plot(result, plot_path, burn_in=args.burn_in)
    save_price_change_distribution_plot(
        result, distribution_plot_path, burn_in=args.burn_in
    )
    summary = summarize_result(result, burn_in=args.burn_in)
    summary_path = args.outdir / "gcmg_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"parameters": vars(config), "lambda_used": config.resolved_lambda(), "diagnostics": summary},
            f, ensure_ascii=False, indent=2,
        )

    if strategy_space.actions is not None:
        # Preserve the previous terminal line for the legacy m<=4 path.
        print(f"FSS: P={strategy_space.P}, number of strategies={strategy_space.n_strategies}")
    else:
        print(f"FSS: P={strategy_space.P}, total strategies=2^{strategy_space.P}; "
              f"sampled unique strategies={result.sampled_strategy_actions.shape[0]}")
    initial_score_value = (
        config.initial_score if config.initial_score is not None
        else config.r + config.initial_score_offset
    )
    print(
        f"lambda={config.resolved_lambda():g}; price_mode={config.price_mode}; "
        f"initial_score={initial_score_value:g}; zero_score_tie={config.zero_score_tie}"
    )
    tail_window = max(1, result.active_agents.size // 10)
    if np.all(result.active_agents[-tail_window:] == 0):
        print(
            "\nWARNING: no agents were active in the final 10% of the run. "
            "With L=0 and sign(0)=0, a zero-demand state can become absorbing. "
            "See README.md; compare --zero-score-tie random as an explicit alternative."
        )
    fallback_count = int(np.count_nonzero(result.price_fallback))
    if fallback_count:
        print(f"\nPrice-boundary log fallback used in {fallback_count} step(s).")
    print("\nDiagnostics (after burn-in):")
    for key, value in summary.items():
        print(f"  {key:32s}: {value:.6g}")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {distribution_plot_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
