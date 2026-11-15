#!/usr/bin/env python3
"""Grand Canonical Minority Game (GCMG) with a Reduced Strategy Space (RSS).

The market/agent dynamics are deliberately separated from the strategy-space
construction.  A future FSS implementation only needs to provide an object
with an ``actions`` array of shape (n_strategies, 2**m), containing +/-1.

Model equations follow the uploaded lecture slides, especially p.25:
    D_t = n_buy(t) - n_sell(t) = sum_i a_i(t)
    x_t - x_{t-1} = D_t / lambda
    S_R(t+1) = (1 - 1/T) S_R(t)
               - a_R[mu_t] * sign(D_t/lambda - L)
    mu_{t+1} = 2 mu_t - P H(mu_t - P/2) + H(D_t/lambda - L)

By default, scores start at r + initial_score_offset and the random
zero-score tie regularization is used.  The literal sign(0)=0 convention
remains available as ``--zero-score-tie zero`` for comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np


class StrategySpace(Protocol):
    """Interface required by the GCMG simulator.

    ``actions[R, mu]`` is the +/-1 action prescribed by strategy R in state mu.
    The simulator is agnostic about whether the strategy set is RSS or FSS.
    """

    m: int
    actions: np.ndarray


@dataclass(frozen=True)
class RSSStrategySpace:
    """One standard RSS built from a Sylvester-Walsh-Hadamard matrix.

    P = 2**m information states.  The RSS contains 2P strategies:
    P pairwise-uncorrelated strategies and their P anti-correlated partners.
    RSS is not unique; this is a conventional representative of the class.
    """

    m: int
    actions: np.ndarray

    @property
    def P(self) -> int:
        return 1 << self.m

    @property
    def n_strategies(self) -> int:
        return int(self.actions.shape[0])


def sylvester_hadamard(order: int) -> np.ndarray:
    """Return the +/-1 Sylvester Hadamard matrix of power-of-two order."""
    if order < 1 or (order & (order - 1)) != 0:
        raise ValueError("Hadamard order must be a positive power of two.")
    H = np.ones((1, 1), dtype=np.int8)
    while H.shape[0] < order:
        H = np.block([[H, H], [H, -H]]).astype(np.int8, copy=False)
    return H


def build_rss(m: int, verify: bool = True) -> RSSStrategySpace:
    """Construct an RSS of size 2P=2**(m+1) for memory length m."""
    if not isinstance(m, int) or m < 1:
        raise ValueError("m must be an integer >= 1.")
    P = 1 << m
    H = sylvester_hadamard(P)
    actions = np.concatenate((H, -H), axis=0).astype(np.int8, copy=False)

    if verify:
        gram = actions[:P].astype(np.int64) @ actions[:P].astype(np.int64).T
        if not np.array_equal(gram, P * np.eye(P, dtype=np.int64)):
            raise RuntimeError("RSS construction failed: first half is not orthogonal.")
        if not np.array_equal(actions[P:], -actions[:P]):
            raise RuntimeError("RSS construction failed: anti-correlated partners missing.")
    return RSSStrategySpace(m=m, actions=actions)


@dataclass
class GCMGConfig:
    # Parameters explicitly listed on lecture p.25
    N: int = 501
    m: int = 3
    s: int = 2
    r: float = -100.0
    T: float = 10_000.0
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
    zero_score_tie: str = "random"  # random regularization by default; or literal "zero"
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
    """Market maker implementing the lecture's linear price-impact rule."""

    lam: float
    price: float
    mode: str = "linear"

    def clear(self, excess_demand: int) -> tuple[float, float, float, float]:
        """Clear net demand and return (new_price, impact, simple_ret, log_ret).

        In linear mode, impact = D/lambda = x[t+1]-x[t].
        In log mode, impact = D/lambda = log(x[t+1])-log(x[t]).
        """
        if self.lam <= 0 or self.price <= 0:
            raise ValueError("MarketMaker requires positive lambda and price.")
        impact = float(excess_demand) / self.lam
        old_price = self.price
        if self.mode == "linear":
            new_price = old_price + impact
            if new_price > 0:
                simple_ret = impact / old_price
                log_ret = float(np.log(new_price / old_price))
            else:
                # The linear rule is undefined as a positive price process at
                # this boundary.  Use the lecture's log-price alternative for
                # this step only, leaving every valid linear step unchanged.
                log_ret = impact
                simple_ret = float(np.expm1(impact))
                new_price = old_price * float(np.exp(impact))
        elif self.mode == "log":
            log_ret = impact
            simple_ret = float(np.expm1(impact))
            new_price = old_price * float(np.exp(impact))
        else:
            raise ValueError("MarketMaker mode must be 'linear' or 'log'.")
        self.price = new_price
        return new_price, impact, simple_ret, log_ret


@dataclass
class GCMGResult:
    config: GCMGConfig
    strategy_space: StrategySpace
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


def simulate_gcmg(
    config: GCMGConfig,
    strategy_space: Optional[StrategySpace] = None,
) -> GCMGResult:
    """Run the GCMG model using the supplied strategy space (RSS by default)."""
    if strategy_space is None:
        strategy_space = build_rss(config.m)

    P = 1 << config.m
    actions = np.asarray(strategy_space.actions, dtype=np.int8)
    if getattr(strategy_space, "m", None) != config.m:
        raise ValueError("strategy_space.m must equal config.m.")
    if actions.ndim != 2 or actions.shape[1] != P:
        raise ValueError(f"actions must have shape (n_strategies, {P}).")
    if not np.all(np.isin(actions, (-1, 1))):
        raise ValueError("All strategy actions must be -1 or +1.")
    n_strategies = actions.shape[0]
    config.validate(n_strategies=n_strategies)

    rng = np.random.default_rng(config.seed)
    lam = config.resolved_lambda()
    alpha = 1.0 - 1.0 / config.T

    # Each agent independently draws s distinct strategies from the strategy pool.
    holdings = np.empty((config.N, config.s), dtype=np.int32)
    for i in range(config.N):
        holdings[i] = rng.choice(n_strategies, size=config.s, replace=False)

    # By default initialize all virtual scores at r + offset, as a startup
    # convention. An absolute initial_score can override this (e.g. 0 to
    # reproduce the all-zero initialization mentioned on lecture p.36).
    # Since the scores are equal at t=0, each agent randomizes among its s
    # held strategies; a positive offset makes all agents eligible initially.
    initial_score_value = (
        float(config.initial_score)
        if config.initial_score is not None
        else config.r + config.initial_score_offset
    )
    scores = np.full(n_strategies, initial_score_value, dtype=np.float64)
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
    price[0] = market_maker.price

    for t in range(steps):
        mu_t = mu
        best_strategy, best_score = _pick_best_strategies(scores, holdings, rng)

        # Grand-canonical entry rule: trade only when the best held strategy
        # has a score strictly greater than r (lecture p.23 / p.25).
        active = best_score > config.r
        chosen_actions = actions[best_strategy, mu_t]
        agent_orders = chosen_actions * active.astype(np.int8)

        D = int(agent_orders.sum())
        n_buy = int(np.count_nonzero(agent_orders == 1))
        n_sell = int(np.count_nonzero(agent_orders == -1))
        q = D / lam
        excess_vs_reference = q - config.L

        # Sign for score updates and the new public bit.  The lecture p.36
        # explicitly randomizes the public bit at equality.  We expose whether
        # the virtual score at equality is also randomized or set to zero.
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

        # Update all virtual strategy scores, including strategies not currently
        # held/selected by an active agent, as in the lecture's score table.
        scores = alpha * scores - actions[:, mu_t].astype(np.float64) * score_signal

        # Market-maker clearing: linear impact is the p.25 rule; log mode is
        # the alternative price representation mentioned on p.14.
        price_now, impact, simple_ret, log_ret = market_maker.clear(D)

        # Shift m-bit history left; discard the oldest bit and append new_bit.
        mu = (2 * mu_t + new_bit) % P

        demand[t] = D
        buy_orders[t] = n_buy
        sell_orders[t] = n_sell
        active_agents[t] = int(active.sum())
        price[t + 1] = price_now
        price_change[t] = impact
        simple_returns[t] = simple_ret
        log_returns[t] = log_ret
        global_info[t] = mu_t
        appended_bit[t] = new_bit
        score_signal_series[t] = score_signal

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
        final_scores=scores.copy(),
        final_holdings=holdings.copy(),
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
        f"GCMG-RSS: N={result.config.N}, m={result.config.m}, s={result.config.s}, "
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
        f"GCMG-RSS price increments: N={result.config.N}, m={result.config.m}, "
        f"s={result.config.s}, lambda={result.config.resolved_lambda():g}, "
        f"burn-in={burn_in}", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate the GCMG with RSS strategies.")
    parser.add_argument("--N", type=int, default=501, help="number of agents")
    parser.add_argument("--m", type=int, default=3, help="global-information memory length")
    parser.add_argument("--s", type=int, default=2, help="strategies held by each agent")
    parser.add_argument("--r", type=float, default=-100.0, help="strategy-score participation threshold")
    parser.add_argument("--T", type=float, default=10_000.0, help="score memory/effective period")
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
        help="score signal at D/lambda=L: random regularization (default) or literal sign(0)=0",
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
    strategy_space = build_rss(config.m)
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

    print(f"RSS: P={strategy_space.P}, number of strategies={strategy_space.n_strategies}")
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
    print("\nDiagnostics (after burn-in):")
    for key, value in summary.items():
        print(f"  {key:32s}: {value:.6g}")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {distribution_plot_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
