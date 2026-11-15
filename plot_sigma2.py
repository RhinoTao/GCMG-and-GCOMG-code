#!/usr/bin/env python3
"""Plot the RSS/FSS sigma_D^2 versus m scan in the exact P37 limit."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot RSS/FSS excess-demand variance versus m.")
    p.add_argument("--input-dir", type=Path, default=Path("sigma2_scan"),
                   help="directory containing sigma2_runs.csv and sigma2_summary.csv")
    p.add_argument("--outdir", type=Path, default=None,
                   help="plot output directory (default: same as --input-dir)")
    p.add_argument("--show-p39-reference", action="store_true",
                   help="draw sigma=sqrt(N) (and sigma^2=N) as the El Farol P39 reference only")
    p.add_argument("--no-individual-seeds", action="store_true",
                   help="hide the individual seed markers")
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def as_float(value: str | float | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _minimum_candidates(summary: list[dict[str, str]], method: str,
                        y_key: str, sem_key: str) -> list[dict[str, Any]]:
    rows = {
        int(r["m"]): r for r in summary
        if r["strategy_space"] == method
        and int(r["n_valid_seeds"]) == int(r["n_seeds"])
        and math.isfinite(as_float(r.get(y_key)))
    }
    out = []
    for m in sorted(rows):
        if m - 1 not in rows or m + 1 not in rows:
            continue
        yl = as_float(rows[m - 1][y_key])
        ym = as_float(rows[m][y_key])
        yr = as_float(rows[m + 1][y_key])
        if ym < yl and ym < yr:
            sl = as_float(rows[m - 1].get(sem_key))
            sm = as_float(rows[m].get(sem_key))
            sr = as_float(rows[m + 1].get(sem_key))
            left_gap, right_gap = yl - ym, yr - ym
            left_err = math.sqrt(sl * sl + sm * sm) if math.isfinite(sl) and math.isfinite(sm) else float("nan")
            right_err = math.sqrt(sr * sr + sm * sm) if math.isfinite(sr) and math.isfinite(sm) else float("nan")
            resolved = bool(math.isfinite(left_err) and math.isfinite(right_err)
                            and left_gap > left_err and right_gap > right_err)
            coeff = np.polyfit([m - 1, m, m + 1], [yl, ym, yr], 2)
            a, b, _ = coeff
            vertex = float(-b / (2 * a)) if a > 0 else float("nan")
            if not (m - 1 <= vertex <= m + 1):
                vertex = float("nan")
            out.append({"m": m, "value": ym, "vertex": vertex, "resolved": resolved})
    return out


def make_plot(runs: list[dict[str, str]], summary: list[dict[str, str]],
              outpath: Path, y_mode: str, N: int | None,
              show_p39_reference: bool, show_individual: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if y_mode == "sigma2":
        y_key = "mean_sigma2_nonextinct"
        err_key = "sd_sigma2_nonextinct"
        ylabel = r"Excess-demand variance $\sigma_D^2$"
        sem_key = "sem_sigma2_nonextinct"
        title_metric = r"$\sigma_D^2$"
        reference = float(N) if N is not None else None
        reference_label = r"$N$ (El Farol P39 reference only)"
        out_name_note = "variance"
    else:
        y_key = "mean_sigma_nonextinct"
        err_key = "sd_sigma_nonextinct"
        ylabel = r"Excess-demand standard deviation $\sigma_D$"
        sem_key = "sem_sigma_nonextinct"
        title_metric = r"$\sigma_D$"
        reference = math.sqrt(N) if N is not None else None
        reference_label = r"$\sqrt{N}$ (El Farol P39 reference only)"
        out_name_note = "standard_deviation"

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    colors = {"RSS": "C0", "FSS": "C1"}
    markers = {"RSS": "o", "FSS": "s"}

    for method in ("RSS", "FSS"):
        m_values = sorted(int(r["m"]) for r in summary if r["strategy_space"] == method)
        row_by_m = {int(r["m"]): r for r in summary if r["strategy_space"] == method}
        xs, ys, yerr = [], [], []
        partial_x, partial_y = [], []
        extinct_x, extinct_y = [], []
        for m in m_values:
            row = row_by_m[m]
            n_valid = int(row["n_valid_seeds"])
            n_seeds = int(row["n_seeds"])
            if n_valid == 0:
                # Plot an extinction-contaminated estimate separately; it is not used in the minimum check.
                extinct_x.append(m)
                all_key = "mean_sigma2_all_seeds" if y_mode == "sigma2" else "mean_sigma_all_seeds"
                extinct_y.append(as_float(row.get(all_key)))
            else:
                xs.append(m)
                ys.append(as_float(row.get(y_key)))
                err = as_float(row.get(err_key))
                yerr.append(err if math.isfinite(err) else 0.0)
                if n_valid < n_seeds:
                    partial_x.append(m)
                    partial_y.append(as_float(row.get(y_key)))

        if xs:
            ax.errorbar(
                xs, ys, yerr=yerr, color=colors[method], marker=markers[method],
                linewidth=1.8, markersize=6, capsize=3, label=f"{method} (mean ± SD; non-extinct seeds)",
                zorder=3,
            )
        if partial_x:
            ax.scatter(partial_x, partial_y, marker="o", s=100, facecolors="none",
                       edgecolors=colors[method], linewidths=1.6, zorder=5,
                       label=f"{method}: incomplete seed set")
        if extinct_x:
            ax.scatter(extinct_x, extinct_y, marker="x", s=70, color=colors[method],
                       linewidths=1.8, zorder=5,
                       label=f"{method}: all seeds extinct (excluded from minimum check)")

        if show_individual:
            for run in runs:
                if run["strategy_space"] != method:
                    continue
                m = int(run["m"])
                val = as_float(run.get("sigma2" if y_mode == "sigma2" else "sigma"))
                # Small deterministic horizontal offsets prevent three seeds from hiding one another.
                seed = int(run["seed"])
                seed_index = sorted({int(x["seed"]) for x in runs}).index(seed)
                offset = (seed_index - 1) * 0.055
                if int(run["extinct_after_burn"]) == 0:
                    ax.scatter(m + offset, val, s=20, color=colors[method], alpha=0.45, zorder=2)
                else:
                    ax.scatter(m + offset, val, s=30, marker="x", color=colors[method], alpha=0.55, zorder=2)

        candidates = _minimum_candidates(summary, method, y_key, sem_key)
        if candidates:
            best = min(candidates, key=lambda c: c["value"])
            ax.axvline(best["m"], color=colors[method], linestyle="--", alpha=0.45, linewidth=1.2)
            mark = "resolved at 1 SEM" if best["resolved"] else "not resolved at 1 SEM"
            # Keep the two labels apart when RSS and FSS minima occur at
            # nearby (or identical) memory lengths.
            label_offset = (-92, 34) if method == "RSS" else (10, -42)
            ax.annotate(
                f"{method} min: m={best['m']}\n{mark}",
                xy=(best["m"], best["value"]), xytext=label_offset,
                textcoords="offset points", color=colors[method], fontsize=8,
            )

    if show_p39_reference and reference is not None:
        ax.axhline(reference, color="0.35", linestyle=":", linewidth=1.3,
                   label=reference_label)

    ax.set_xlabel("Memory length $m$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"P37 Minority Game: RSS vs FSS: {title_metric} versus memory length")
    ax.set_xticks(sorted({int(r["m"]) for r in summary}))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = args.outdir or args.input_dir
    runs_path = args.input_dir / "sigma2_runs.csv"
    summary_path = args.input_dir / "sigma2_summary.csv"
    metadata_path = args.input_dir / "sigma2_scan_metadata.json"
    if not runs_path.exists() or not summary_path.exists():
        raise SystemExit(
            f"Missing {runs_path} or {summary_path}. Run scan_sigma2.py first."
        )
    runs = read_csv(runs_path)
    summary = read_csv(summary_path)
    N = None
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        N = int(metadata["parameters"]["N"])

    outdir.mkdir(parents=True, exist_ok=True)
    make_plot(runs, summary, outdir / "sigma2_vs_m_RSS_FSS.png", "sigma2", N,
              args.show_p39_reference, not args.no_individual_seeds)
    make_plot(runs, summary, outdir / "sigma_vs_m_RSS_FSS_P39_style.png", "sigma", N,
              args.show_p39_reference, not args.no_individual_seeds)

    print("Saved:")
    print(f"  {outdir / 'sigma2_vs_m_RSS_FSS.png'}")
    print(f"  {outdir / 'sigma_vs_m_RSS_FSS_P39_style.png'}")


if __name__ == "__main__":
    main()
