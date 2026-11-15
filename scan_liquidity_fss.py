#!/usr/bin/env python3
"""Scan GCMG-FSS market depth and store liquidity diagnostics in one NPZ file.

Place this script in the same directory as ``gcmg_fss.py``.  For every value
of lambda/N, the same seed sequence is used.  Only one representative seed's
full trajectories are stored; statistics are stored for every seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import gcmg_fss as model
from gcmg_fss import (
    GCMGConfig,
    build_fss,
    lag_autocorrelation,
    simulate_gcmg,
)


def _bounded_log_clear(self, excess_demand: int) -> tuple[float, float, float, float]:
    """Log-mode clearing without floating-point price-level overflow.

    In log mode, D/lambda determines every score update and log return, while
    the absolute price level never feeds back into agent decisions.  Keeping
    the internal price at one therefore leaves all quantities stored by this
    scanner unchanged.  The true log-price path is reconstructed from the
    unmodified D/lambda series after the run.  Linear mode never uses this.
    """
    if self.lam <= 0 or not np.isfinite(self.lam):
        raise ValueError("MarketMaker requires finite positive lambda.")
    if self.price <= 0 or not np.isfinite(self.price):
        raise ValueError("MarketMaker requires finite positive price.")
    impact = float(excess_demand) / self.lam
    self.last_fallback = False
    with np.errstate(over="ignore"):
        simple_ret = float(np.expm1(impact))
    self.price = 1.0
    return self.price, impact, simple_ret, impact


def _simulate_without_log_price_overflow(
    config: GCMGConfig, strategy_space,
):
    """Run log mode with bounded internal price; leave linear mode untouched."""
    if config.price_mode != "log":
        return simulate_gcmg(config, strategy_space)
    original_clear = model.MarketMaker.clear
    model.MarketMaker.clear = _bounded_log_clear
    try:
        return simulate_gcmg(config, strategy_space)
    finally:
        model.MarketMaker.clear = original_clear


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GCMG-FSS simulations for several lambda/N values and save "
            "per-seed statistics plus representative trajectories."
        )
    )
    parser.add_argument("--N", type=int, default=101, help="number of agents")
    parser.add_argument("--m", type=int, default=3, help="global-information memory length")
    parser.add_argument("--s", type=int, default=2, help="strategies held by each agent")
    parser.add_argument("--r", type=float, default=4.0, help="participation threshold")
    parser.add_argument("--T", type=float, default=100.0, help="score memory/effective period")
    parser.add_argument("--L", type=float, default=0.0, help="market reference threshold")
    parser.add_argument("--x0", type=float, default=100.0, help="initial price")
    parser.add_argument(
        "--price-mode", choices=("log", "linear"), default="log",
        help="log isolates liquidity scaling; linear also diagnoses boundary fallbacks",
    )
    parser.add_argument("--steps", type=int, default=50_000, help="market rounds per run")
    parser.add_argument("--burn-in", type=int, default=5_000, help="rounds discarded in statistics")
    parser.add_argument(
        "--lambda-over-n", type=float, nargs="+", default=[10.0, 1.0, 0.1, 0.01],
        help="positive dimensionless depths lambda/N (default: 10 1 0.1 0.01)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(123, 133)),
        help="seed sequence reused unchanged at every lambda/N",
    )
    parser.add_argument(
        "--trajectory-seed", type=int, default=None,
        help="seed whose full trajectories are stored; default: first seed",
    )
    parser.add_argument(
        "--trajectory-stride", type=int, default=1,
        help="store every kth trajectory point; the final point is always included",
    )
    parser.add_argument("--mu0", type=int, default=None, help="optional fixed initial public state")
    parser.add_argument("--initial-score-offset", type=float, default=10.0)
    parser.add_argument("--initial-score", type=float, default=None)
    parser.add_argument(
        "--zero-score-tie", choices=("random", "zero"), default="random",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("liquidity_scan.npz"),
        help="output NPZ path (the .npz suffix is added if missing)",
    )
    return parser.parse_args()


def _trajectory_indices(steps: int, stride: int) -> np.ndarray:
    indices = np.arange(0, steps + 1, stride, dtype=np.int64)
    if indices[-1] != steps:
        indices = np.append(indices, np.int64(steps))
    return indices


def main() -> None:
    args = _parse_args()
    kappas = np.asarray(args.lambda_over_n, dtype=np.float64)
    seeds = np.asarray(args.seeds, dtype=np.int64)

    if kappas.ndim != 1 or kappas.size == 0:
        raise ValueError("At least one lambda/N value is required.")
    if not np.all(np.isfinite(kappas)) or np.any(kappas <= 0):
        raise ValueError("Every lambda/N must be finite and strictly positive.")
    if seeds.ndim != 1 or seeds.size == 0:
        raise ValueError("At least one seed is required.")
    if np.unique(seeds).size != seeds.size:
        raise ValueError("Seeds must be distinct.")
    if args.trajectory_stride < 1:
        raise ValueError("trajectory_stride must be >= 1.")
    if not (0 <= args.burn_in < args.steps - 3):
        raise ValueError("burn_in must be >= 0 and leave at least four observations.")

    trajectory_seed = int(seeds[0]) if args.trajectory_seed is None else args.trajectory_seed
    if trajectory_seed not in set(int(seed) for seed in seeds):
        raise ValueError("trajectory_seed must be included in --seeds.")

    output = args.output
    if output.suffix.lower() != ".npz":
        output = Path(str(output) + ".npz")
    output.parent.mkdir(parents=True, exist_ok=True)

    n_kappas = kappas.size
    n_seeds = seeds.size
    shape = (n_kappas, n_seeds)
    lambda_values = kappas * float(args.N)

    std_log_return = np.full(shape, np.nan, dtype=np.float64)
    mean_log_return = np.full(shape, np.nan, dtype=np.float64)
    acf_return_lag1 = np.full(shape, np.nan, dtype=np.float64)
    mean_demand = np.full(shape, np.nan, dtype=np.float64)
    std_demand = np.full(shape, np.nan, dtype=np.float64)
    acf_demand_lag1 = np.full(shape, np.nan, dtype=np.float64)
    mean_active_agents = np.full(shape, np.nan, dtype=np.float64)
    fallback_rate = np.full(shape, np.nan, dtype=np.float64)

    trajectory_time = _trajectory_indices(args.steps, args.trajectory_stride)
    log_price = np.full((n_kappas, trajectory_time.size), np.nan, dtype=np.float64)
    cumulative_demand = np.full(
        (n_kappas, trajectory_time.size), np.nan, dtype=np.float64
    )

    metadata = {
        "N": args.N,
        "m": args.m,
        "s": args.s,
        "r": args.r,
        "T": args.T,
        "L": args.L,
        "x0": args.x0,
        "price_mode": args.price_mode,
        "steps": args.steps,
        "burn_in": args.burn_in,
        "mu0": args.mu0,
        "initial_score_offset": args.initial_score_offset,
        "initial_score": args.initial_score,
        "zero_score_tie": args.zero_score_tie,
        "trajectory_seed": trajectory_seed,
        "trajectory_stride": args.trajectory_stride,
        "lambda_rule": "lambda = (lambda/N) * N",
        "log_price_definition": "ln[x(t)/x(0)] reconstructed from log_returns",
        "log_overflow_handling": (
            "log mode keeps only the internal price level bounded; D/lambda, "
            "agent dynamics and reconstructed log prices are unchanged"
        ),
        "rescaled_log_price_definition": "lambda * ln[x(t)/x(0)]",
        "cumulative_demand_definition": "sum_{tau<t} D_tau",
        "fallback_rate_definition": "fraction of post-burn-in steps using price fallback",
    }

    strategy_space = build_fss(args.m)
    total_runs = n_kappas * n_seeds
    run_number = 0

    for i, (kappa, lam) in enumerate(zip(kappas, lambda_values)):
        for j, seed_value in enumerate(seeds):
            seed = int(seed_value)
            run_number += 1
            print(
                f"[{run_number}/{total_runs}] lambda/N={kappa:g}, "
                f"lambda={lam:g}, seed={seed}",
                flush=True,
            )
            config = GCMGConfig(
                N=args.N,
                m=args.m,
                s=args.s,
                r=args.r,
                T=args.T,
                L=args.L,
                lam=float(lam),
                x0=args.x0,
                price_mode=args.price_mode,
                steps=args.steps,
                seed=seed,
                mu0=args.mu0,
                initial_score_offset=args.initial_score_offset,
                initial_score=args.initial_score,
                zero_score_tie=args.zero_score_tie,
            )
            result = _simulate_without_log_price_overflow(config, strategy_space)

            returns = np.asarray(result.log_returns[args.burn_in:], dtype=np.float64)
            demand = np.asarray(result.demand[args.burn_in:], dtype=np.float64)
            active = np.asarray(result.active_agents[args.burn_in:], dtype=np.float64)
            fallback = np.asarray(result.price_fallback[args.burn_in:], dtype=bool)

            std_log_return[i, j] = float(returns.std())
            mean_log_return[i, j] = float(returns.mean())
            acf_return_lag1[i, j] = lag_autocorrelation(returns, 1)
            mean_demand[i, j] = float(demand.mean())
            std_demand[i, j] = float(demand.std())
            acf_demand_lag1[i, j] = lag_autocorrelation(demand, 1)
            mean_active_agents[i, j] = float(active.mean())
            fallback_rate[i, j] = float(fallback.mean())

            if seed == trajectory_seed:
                full_log_price = np.concatenate((
                    np.zeros(1, dtype=np.float64),
                    np.cumsum(result.log_returns, dtype=np.float64),
                ))
                full_cumulative_demand = np.concatenate((
                    np.zeros(1, dtype=np.float64),
                    np.cumsum(result.demand, dtype=np.float64),
                ))
                log_price[i] = full_log_price[trajectory_time]
                cumulative_demand[i] = full_cumulative_demand[trajectory_time]

    if not np.all(np.isfinite(log_price)) or not np.all(np.isfinite(cumulative_demand)):
        raise RuntimeError("Representative trajectories are incomplete or non-finite.")

    rescaled_log_price = lambda_values[:, None] * log_price
    rescale_error = float(np.max(np.abs(rescaled_log_price - cumulative_demand)))
    demand_spread = float(np.max(np.ptp(cumulative_demand, axis=0)))

    np.savez_compressed(
        output,
        lambda_over_n=kappas,
        lambda_values=lambda_values,
        seeds=seeds,
        std_log_return=std_log_return,
        mean_log_return=mean_log_return,
        acf_return_lag1=acf_return_lag1,
        mean_demand=mean_demand,
        std_demand=std_demand,
        acf_demand_lag1=acf_demand_lag1,
        mean_active_agents=mean_active_agents,
        fallback_rate=fallback_rate,
        trajectory_time=trajectory_time,
        trajectory_seed=np.int64(trajectory_seed),
        log_price=log_price,
        rescaled_log_price=rescaled_log_price,
        cumulative_demand=cumulative_demand,
        representative_rescale_error=np.float64(rescale_error),
        representative_demand_spread=np.float64(demand_spread),
        metadata_json=np.str_(json.dumps(metadata, ensure_ascii=False, indent=2)),
    )

    print(f"Saved: {output}")
    print(f"Representative trajectory seed: {trajectory_seed}")
    if args.price_mode == "log":
        print(f"max |lambda*log_price - cumulative_demand| = {rescale_error:.6g}")
    else:
        print(
            "lambda*log_price differs from cumulative_demand in linear mode; "
            f"diagnostic difference = {rescale_error:.6g}"
        )
    print(f"max cumulative-demand spread across lambda/N = {demand_spread:.6g}")
    if metadata["L"] == 0.0:
        print(
            "For L=0 and positive lambda, identical demand trajectories across "
            "lambda/N indicate pure price-impact scaling rather than a new agent phase."
        )


if __name__ == "__main__":
    main()
