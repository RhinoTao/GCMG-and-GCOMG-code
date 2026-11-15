#!/usr/bin/env python3
"""Scan RSS and sparse FSS in the exact Minority-Game limit stated on lecture P37.

Fixed model conditions: L=0, r=-infinity (all agents participate), and
T=+infinity (no score decay). Strategy scores start at zero.

The observable is the post-burn-in population variance of D[t^-]. Linear
prices are used except for a one-step logarithmic fallback if a linear price
candidate would be non-positive.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import gcmg_rss as rss
import gcmg_fss as fss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare RSS and sparse FSS in the exact P37 Minority-Game limit."
    )
    p.add_argument("--N", type=int, default=101)
    p.add_argument("--s", type=int, default=2)
    p.add_argument("--lambda", dest="lam", type=float, default=None,
                   help="market depth; omitted means the v2 convention lambda=10*N")
    p.add_argument("--x0", type=float, default=100.0)
    p.add_argument("--steps", type=int, default=50_000,
                   help="total market rounds per run (default: 50000)")
    p.add_argument("--burn-in", type=int, default=5_000,
                   help="discard this many initial rounds before calculating sigma^2")
    p.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789],
                   help="random seeds; default is three seeds: 123 456 789")
    p.add_argument("--rss-m-min", type=int, default=1)
    p.add_argument("--rss-m-max", type=int, default=10)
    p.add_argument("--fss-m-min", type=int, default=1)
    p.add_argument("--fss-m-max", type=int, default=10)
    p.add_argument("--mu0", type=int, default=None)
    p.add_argument("--initial-score", type=float, default=0.0,
                   help="initial virtual strategy score; P36 suggests zero (default: 0)")
    p.add_argument("--zero-score-tie", choices=("zero", "random"), default="random",
                   help="at D=0, randomize next public bit (default, P36) or append 0")
    p.add_argument("--outdir", type=Path, default=Path("sigma2_scan"))
    return p.parse_args()


def make_config(module: Any, args: argparse.Namespace, m: int, seed: int):
    return module.GCMGConfig(
        N=args.N,
        m=m,
        s=args.s,
        # Exact lecture P37 limit: L=0, r -> -infinity, T -> +infinity.
        r=-math.inf,
        T=math.inf,
        L=0.0,
        lam=args.lam,
        x0=args.x0,
        # This scan always uses linear prices, with logarithmic fallback
        # only on steps where the linear candidate would be non-positive.
        price_mode="linear",
        steps=args.steps,
        seed=seed,
        mu0=args.mu0,
        initial_score=args.initial_score,
        zero_score_tie=args.zero_score_tie,
    )


def calculate_observables(result: Any, burn_in: int, N: int) -> dict[str, float | int]:
    """Calculate sigma_D^2 and activity diagnostics after burn-in."""
    D = np.asarray(result.demand[burn_in:], dtype=np.float64)
    active = np.asarray(result.active_agents[burn_in:], dtype=np.float64)
    if D.size < 2:
        raise ValueError("Fewer than two post-burn-in rounds remain.")

    mean_D = float(np.mean(D))
    mean_D2 = float(np.mean(D * D))
    # This is exactly (1/n) sum D^2 - [(1/n) sum D]^2.
    sigma2 = max(0.0, mean_D2 - mean_D * mean_D)
    sigma = math.sqrt(sigma2)
    active_time_fraction = float(np.mean(active > 0))
    mean_active_agent_fraction = float(np.mean(active) / N)
    nonzero_demand_fraction = float(np.mean(D != 0))
    extinct = bool(np.all(active == 0))
    fallback = getattr(result, "price_fallback", None)
    price_fallback_steps = int(np.count_nonzero(fallback)) if fallback is not None else 0

    return {
        "n_postburn": int(D.size),
        "mean_D": mean_D,
        "mean_D2": mean_D2,
        "sigma2": sigma2,
        "sigma": sigma,
        "active_time_fraction": active_time_fraction,
        "mean_active_agent_fraction": mean_active_agent_fraction,
        "nonzero_demand_fraction": nonzero_demand_fraction,
        "extinct_after_burn": int(extinct),
        "price_fallback_steps_total": price_fallback_steps,
        "price_fallback_steps_after_burn": int(np.count_nonzero(fallback[burn_in:])) if fallback is not None else 0,
    }


def mean_sd_sem(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
    sem = sd / math.sqrt(arr.size) if arr.size > 1 else float("nan")
    return mean, sd, sem


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_runs(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        buckets[(row["strategy_space"], int(row["m"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (method, m), rows in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        all_sigma2 = [float(r["sigma2"]) for r in rows]
        all_sigma = [float(r["sigma"]) for r in rows]
        valid_rows = [r for r in rows if int(r["extinct_after_burn"]) == 0]
        valid_sigma2 = [float(r["sigma2"]) for r in valid_rows]
        valid_sigma = [float(r["sigma"]) for r in valid_rows]

        mean_all, sd_all, sem_all = mean_sd_sem(all_sigma2)
        mean_sigma_all, sd_sigma_all, sem_sigma_all = mean_sd_sem(all_sigma)
        if valid_rows:
            mean_valid, sd_valid, sem_valid = mean_sd_sem(valid_sigma2)
            mean_sigma_valid, sd_sigma_valid, sem_sigma_valid = mean_sd_sem(valid_sigma)
        else:
            mean_valid = sd_valid = sem_valid = float("nan")
            mean_sigma_valid = sd_sigma_valid = sem_sigma_valid = float("nan")

        summary_rows.append({
            "strategy_space": method,
            "m": m,
            "n_seeds": len(rows),
            "n_valid_seeds": len(valid_rows),
            "mean_sigma2_all_seeds": mean_all,
            "sd_sigma2_all_seeds": sd_all,
            "sem_sigma2_all_seeds": sem_all,
            "mean_sigma2_nonextinct": mean_valid,
            "sd_sigma2_nonextinct": sd_valid,
            "sem_sigma2_nonextinct": sem_valid,
            "mean_sigma_all_seeds": mean_sigma_all,
            "sd_sigma_all_seeds": sd_sigma_all,
            "sem_sigma_all_seeds": sem_sigma_all,
            "mean_sigma_nonextinct": mean_sigma_valid,
            "sd_sigma_nonextinct": sd_sigma_valid,
            "sem_sigma_nonextinct": sem_sigma_valid,
            "mean_active_time_fraction": float(np.mean([r["active_time_fraction"] for r in rows])),
            "mean_active_agent_fraction": float(np.mean([r["mean_active_agent_fraction"] for r in rows])),
            "mean_nonzero_demand_fraction": float(np.mean([r["nonzero_demand_fraction"] for r in rows])),
        })
    return summary_rows


def _fit_parabolic_vertex(m_left: int, y_left: float, m_mid: int, y_mid: float,
                           m_right: int, y_right: float) -> float | None:
    coeff = np.polyfit([m_left, m_mid, m_right], [y_left, y_mid, y_right], 2)
    a, b, _ = coeff
    if a <= 0:
        return None
    vertex = float(-b / (2.0 * a))
    if m_left <= vertex <= m_right:
        return vertex
    return None


def detect_minima(summary_rows: list[dict[str, Any]], metric: str = "sigma2") -> dict[str, Any]:
    """Find interior local minima using only m points with all seeds active.

    A three-point quadratic interpolation is reported as an exploratory
    sub-grid estimate. A candidate is marked 1-SEM-resolved only when it lies
    more than the combined standard error below both neighboring points.
    """
    results: dict[str, Any] = {}
    if metric == "sigma2":
        y_key, sem_key = "mean_sigma2_nonextinct", "sem_sigma2_nonextinct"
    elif metric == "sigma":
        y_key, sem_key = "mean_sigma_nonextinct", "sem_sigma_nonextinct"
    else:
        raise ValueError("metric must be 'sigma2' or 'sigma'")

    for method in ("RSS", "FSS"):
        rows = {int(r["m"]): r for r in summary_rows if r["strategy_space"] == method}
        eligible = {
            m: r for m, r in rows.items()
            if int(r["n_valid_seeds"]) == int(r["n_seeds"])
            and np.isfinite(float(r[y_key]))
        }
        global_min_m = min(eligible, key=lambda m: float(eligible[m][y_key])) if eligible else None
        candidates: list[dict[str, Any]] = []

        for m in sorted(eligible):
            if m - 1 not in eligible or m + 1 not in eligible:
                continue
            left, mid, right = eligible[m - 1], eligible[m], eligible[m + 1]
            yl, ym, yr = float(left[y_key]), float(mid[y_key]), float(right[y_key])
            if not (ym < yl and ym < yr):
                continue

            sem_l = float(left[sem_key])
            sem_m = float(mid[sem_key])
            sem_r = float(right[sem_key])
            left_gap = yl - ym
            right_gap = yr - ym
            left_err = math.sqrt(sem_l**2 + sem_m**2) if np.isfinite(sem_l) and np.isfinite(sem_m) else float("nan")
            right_err = math.sqrt(sem_r**2 + sem_m**2) if np.isfinite(sem_r) and np.isfinite(sem_m) else float("nan")
            resolved = bool(np.isfinite(left_err) and np.isfinite(right_err)
                            and left_gap > left_err and right_gap > right_err)
            vertex = _fit_parabolic_vertex(m - 1, yl, m, ym, m + 1, yr)
            candidates.append({
                "m_discrete": m,
                "value_at_minimum": ym,
                "quadratic_vertex_estimate": vertex,
                "is_global_minimum_among_eligible_points": bool(m == global_min_m),
                "lower_than_both_neighbors_by_more_than_combined_1_SEM": resolved,
                "left_neighbor_gap": left_gap,
                "right_neighbor_gap": right_gap,
            })

        results[method] = {
            "eligible_m_values": sorted(eligible),
            "global_minimum_m_among_eligible_points": global_min_m,
            "interior_local_minima": candidates,
            "minimum_verified": bool(candidates),
            "note": (
                "In the exact P37 limit every agent is active every round, so extinction filtering "
                "should not exclude any runs. Three seeds provide a preliminary error estimate, "
                "not a high-confidence phase-transition determination."
            ),
        }
    return {"metric": metric, "results": results}


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or not (0 <= args.burn_in < args.steps - 1):
        raise SystemExit("Require steps > burn-in + 1 and burn-in >= 0.")
    if not args.seeds:
        raise SystemExit("At least one seed is required.")
    if not (1 <= args.rss_m_min <= args.rss_m_max <= 10):
        raise SystemExit("RSS scan range must lie inside 1 <= m <= 10.")
    if not (1 <= args.fss_m_min <= args.fss_m_max <= 10):
        raise SystemExit("FSS scan range must lie inside 1 <= m <= 10.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    total = ((args.rss_m_max - args.rss_m_min + 1)
             + (args.fss_m_max - args.fss_m_min + 1)) * len(args.seeds)
    counter = 0

    for method, module, m_values, builder, simulator in (
        ("RSS", rss, range(args.rss_m_min, args.rss_m_max + 1), rss.build_rss, rss.simulate_gcmg),
        ("FSS", fss, range(args.fss_m_min, args.fss_m_max + 1), fss.build_fss, fss.simulate_gcmg),
    ):
        for m in m_values:
            print(f"\n[{method}] m={m}: constructing strategy space...", flush=True)
            strategy_space = builder(m)
            if method == "FSS" and not strategy_space.is_materialized:
                print(
                    f"  full strategies=2^{1 << m}; sparse sampling enabled; "
                    f"information states={1 << m}", flush=True
                )
            else:
                print(f"  strategies={strategy_space.n_strategies:,}; information states={1 << m}", flush=True)
            for seed in args.seeds:
                counter += 1
                print(f"  ({counter}/{total}) seed={seed}: simulating...", flush=True)
                config = make_config(module, args, m, seed)
                result = simulator(config, strategy_space)
                observables = calculate_observables(result, args.burn_in, args.N)
                row: dict[str, Any] = {
                    "strategy_space": method,
                    "m": m,
                    "seed": seed,
                    "N": args.N,
                    "s": args.s,
                    "r": "-inf",
                    "T": "inf",
                    "L": 0.0,
                    "lambda_used": config.resolved_lambda(),
                    "steps": args.steps,
                    "burn_in": args.burn_in,
                    "initial_score": args.initial_score,
                    "zero_score_tie": args.zero_score_tie,
                }
                row.update(observables)
                run_rows.append(row)
                print(
                    f"    sigma^2={observables['sigma2']:.6g}; "
                    f"active-time={observables['active_time_fraction']:.3f}; "
                    f"extinct={observables['extinct_after_burn']}",
                    flush=True,
                )

    summary_rows = aggregate_runs(run_rows)
    write_csv(args.outdir / "sigma2_runs.csv", run_rows)
    write_csv(args.outdir / "sigma2_summary.csv", summary_rows)

    minima_sigma2 = detect_minima(summary_rows, "sigma2")
    minima_sigma = detect_minima(summary_rows, "sigma")
    metadata = {
        "description": "RSS/FSS comparison in the exact P37 Minority-Game limit",
        "observable": "sigma_D^2 = mean(D[t^-]^2) - mean(D[t^-])^2, population denominator n",
        "D_definition": "pre-clearing net excess demand = buy_orders - sell_orders",
        "parameters": {
            "N": args.N, "s": args.s, "r": "-inf", "T": "inf", "L": 0.0,
            "dynamics": "exact P37 limit: L=0, r=-infinity (all agents active), T=+infinity (alpha=1); S_R[t+1]=S_R[t]-a_R[mu_t]*sign(D[t^-]/lambda)",
            "score_tie_and_history_rule": "equal-score strategy choices are unbiased random; at D=0 the score signal is exactly 0 and the next public bit is random by default (P36)",
            "initial_score": args.initial_score,
            "lambda": args.lam if args.lam is not None else 10 * args.N,
            "x0": args.x0,
            "price_rule": "linear except one-step logarithmic fallback if x + D/lambda <= 0",
            "fss_implementation": "complete FSS represented implicitly; sample only each agent's s held strategies; m=1..10",
            "steps": args.steps, "burn_in": args.burn_in,
            "seeds": args.seeds, "zero_score_tie": args.zero_score_tie,
            "rss_m_range": [args.rss_m_min, args.rss_m_max],
            "fss_m_range": [args.fss_m_min, args.fss_m_max],
        },
        "minimum_checks_sigma2": minima_sigma2,
        "minimum_checks_sigma": minima_sigma,
        "interpretation_warning": (
            "A variance minimum is a finite-size numerical feature, not by itself a proof of a "
            "thermodynamic phase transition. Check activity and repeat with more seeds / longer runs."
        ),
    }
    with (args.outdir / "sigma2_scan_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(metadata), f, ensure_ascii=False, indent=2)

    print("\n=== Minimum check for sigma_D^2 ===")
    for method, info in minima_sigma2["results"].items():
        print(f"{method}: eligible m={info['eligible_m_values']}")
        if info["interior_local_minima"]:
            for candidate in info["interior_local_minima"]:
                print(
                    f"  local minimum at m={candidate['m_discrete']}; "
                    f"value={candidate['value_at_minimum']:.6g}; "
                    f"quadratic estimate={candidate['quadratic_vertex_estimate']}; "
                    f"1-SEM-resolved={candidate['lower_than_both_neighbors_by_more_than_combined_1_SEM']}"
                )
        else:
            print("  No interior minimum verified (possibly no complete active points or a boundary minimum).")

    print("\nSaved:")
    print(f"  {args.outdir / 'sigma2_runs.csv'}")
    print(f"  {args.outdir / 'sigma2_summary.csv'}")
    print(f"  {args.outdir / 'sigma2_scan_metadata.json'}")


if __name__ == "__main__":
    main()
