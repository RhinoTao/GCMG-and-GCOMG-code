#!/usr/bin/env python3
"""Read a liquidity_scan.npz file and draw the 2x2 GCMG-FSS diagnostic figure.

Panel selection follows the price mode stored by ``scan_liquidity_fss.py``:
- log mode: panel (d) shows lambda-rescaled log-price trajectories;
- linear mode: panel (d) shows the price-boundary fallback rate.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_ARRAYS = {
    "lambda_over_n",
    "seeds",
    "std_log_return",
    "acf_return_lag1",
    "fallback_rate",
    "trajectory_time",
    "log_price",
    "rescaled_log_price",
    "metadata_json",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 2x2 liquidity diagnostics from scan_liquidity_fss.py output."
    )
    parser.add_argument("input", type=Path, help="NPZ file produced by the scan program")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="output image; default: <input stem>_2x2.png",
    )
    parser.add_argument("--dpi", type=int, default=300, help="image resolution")
    return parser.parse_args()


def _nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def _nanstd(values: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(values, axis=axis)


def _shade_burn_in(ax: plt.Axes, burn_in: int) -> None:
    if burn_in > 0:
        ax.axvspan(0, burn_in, color="0.85", alpha=0.45, linewidth=0)


def main() -> None:
    args = _parse_args()
    if args.dpi < 50:
        raise ValueError("dpi must be >= 50.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with np.load(args.input, allow_pickle=False) as data:
        missing = REQUIRED_ARRAYS.difference(data.files)
        if missing:
            raise ValueError(f"NPZ is missing required arrays: {sorted(missing)}")
        arrays = {key: np.asarray(data[key]) for key in REQUIRED_ARRAYS if key != "metadata_json"}
        metadata = json.loads(str(data["metadata_json"].item()))

    kappas = np.asarray(arrays["lambda_over_n"], dtype=np.float64)
    seeds = np.asarray(arrays["seeds"], dtype=np.int64)
    volatility = np.asarray(arrays["std_log_return"], dtype=np.float64)
    acf1 = np.asarray(arrays["acf_return_lag1"], dtype=np.float64)
    fallback = np.asarray(arrays["fallback_rate"], dtype=np.float64)
    time = np.asarray(arrays["trajectory_time"], dtype=np.int64)
    log_price = np.asarray(arrays["log_price"], dtype=np.float64)
    rescaled = np.asarray(arrays["rescaled_log_price"], dtype=np.float64)

    expected_shape = (kappas.size, seeds.size)
    for name, values in (("std_log_return", volatility), ("acf_return_lag1", acf1), ("fallback_rate", fallback)):
        if values.shape != expected_shape:
            raise ValueError(f"{name} has shape {values.shape}; expected {expected_shape}.")
    if log_price.shape != (kappas.size, time.size) or rescaled.shape != log_price.shape:
        raise ValueError("Trajectory array shapes are inconsistent with lambda/N and time.")
    if np.any(kappas <= 0) or not np.all(np.isfinite(kappas)):
        raise ValueError("Stored lambda/N values must be finite and positive.")

    output = args.output
    if output is None:
        output = args.input.with_name(args.input.stem + "_2x2.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Turbo keeps intermediate curves visible; coolwarm's midpoint is nearly white.
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.90, kappas.size))
    inverse_depth = 1.0 / kappas
    order = np.argsort(inverse_depth)
    x = inverse_depth[order]
    burn_in = int(metadata["burn_in"])

    vol_mean = _nanmean(volatility, axis=1)[order]
    vol_std = _nanstd(volatility, axis=1)[order]
    acf_mean = _nanmean(acf1, axis=1)[order]
    acf_std = _nanstd(acf1, axis=1)[order]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) Representative log-price trajectories at identical seed.
    for i, kappa in enumerate(kappas):
        ax_a.plot(
            time, log_price[i], color=colors[i], linewidth=1.15,
            label=rf"$\lambda/N={kappa:g}$",
        )
    _shade_burn_in(ax_a, burn_in)
    ax_a.set_title("(a) Price trajectories: liquidity amplification")
    ax_a.set_xlabel("Time $t$")
    ax_a.set_ylabel(r"$\ln[x(t)/x(0)]$")
    ax_a.legend(fontsize=9, frameon=False)
    ax_a.text(
        0.015, 0.975, f"seed = {int(metadata['trajectory_seed'])}",
        transform=ax_a.transAxes, ha="left", va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
    )

    # (b) Return volatility; slope-one reference represents mechanical D/lambda scaling.
    valid_vol = np.isfinite(vol_mean) & (vol_mean > 0)
    if np.any(valid_vol):
        ax_b.errorbar(
            x[valid_vol], vol_mean[valid_vol], yerr=vol_std[valid_vol],
            fmt="o-", color="#204a87", capsize=3, linewidth=1.4,
            label="Mean over seeds",
        )
        x_ref = x[valid_vol][0]
        y_ref = vol_mean[valid_vol][0]
        ax_b.plot(
            x[valid_vol], y_ref * x[valid_vol] / x_ref, "--",
            color="0.35", linewidth=1.2, label=r"$\propto N/\lambda$",
        )
        ax_b.set_yscale("log")
        ax_b.legend(frameon=False, fontsize=9)
    else:
        ax_b.text(0.5, 0.5, "No positive finite volatility", ha="center", va="center", transform=ax_b.transAxes)
    ax_b.set_xscale("log")
    ax_b.set_title("(b) Volatility amplification")
    ax_b.set_xlabel(r"Inverse relative depth $N/\lambda$")
    ax_b.set_ylabel(r"Log-return volatility $\sigma_r$")

    # (c) Signed return persistence, not absolute-return/volatility clustering.
    valid_acf = np.isfinite(acf_mean)
    if np.any(valid_acf):
        ax_c.errorbar(
            x[valid_acf], acf_mean[valid_acf], yerr=acf_std[valid_acf],
            fmt="s-", color="#a32d2d", capsize=3, linewidth=1.4,
            label=rf"$C_r(1)$, {seeds.size} seeds",
        )
        ax_c.legend(frameon=False, fontsize=9)
    else:
        ax_c.text(0.5, 0.5, "Lag-1 ACF undefined", ha="center", va="center", transform=ax_c.transAxes)
    ax_c.axhline(0.0, color="0.25", linewidth=0.9)
    ax_c.set_xscale("log")
    ax_c.set_title("(c) Directional persistence / trend indicator")
    ax_c.set_xlabel(r"Inverse relative depth $N/\lambda$")
    ax_c.set_ylabel(r"Return autocorrelation $C_r(1)$")

    # (d) A direct control in log mode; a numerical-boundary diagnostic in linear mode.
    if metadata["price_mode"] == "log":
        linestyles = ["-", "--", "-.", ":"]
        for i, kappa in enumerate(kappas):
            ax_d.plot(
                time, rescaled[i], color=colors[i], linewidth=1.2,
                linestyle=linestyles[i % len(linestyles)], alpha=0.82,
                label=rf"$\lambda/N={kappa:g}$",
            )
        _shade_burn_in(ax_d, burn_in)
        spread = float(np.max(np.ptp(rescaled, axis=0)))
        ax_d.set_title("(d) Rescaled trajectories: order-flow control")
        ax_d.set_xlabel("Time $t$")
        ax_d.set_ylabel(r"$\lambda\ln[x(t)/x(0)]$")
        ax_d.text(
            0.015, 0.975, f"max spread = {spread:.3g}",
            transform=ax_d.transAxes, ha="left", va="top", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
        )
        ax_d.legend(fontsize=8, frameon=False, ncol=2)
    else:
        fallback_mean = _nanmean(fallback, axis=1)[order]
        fallback_std = _nanstd(fallback, axis=1)[order]
        ax_d.errorbar(
            x, fallback_mean, yerr=fallback_std, fmt="^-", color="#6a3d9a",
            capsize=3, linewidth=1.4,
        )
        ax_d.set_xscale("log")
        ax_d.set_ylim(bottom=0.0)
        if np.nanmax(fallback_mean) == 0.0:
            ax_d.set_ylim(0.0, 1.0)
        ax_d.set_title("(d) Numerical price-boundary diagnostic")
        ax_d.set_xlabel(r"Inverse relative depth $N/\lambda$")
        ax_d.set_ylabel("Post-burn-in fallback rate")

    fig.suptitle(
        "GCMG-FSS liquidity scan: "
        rf"$N={metadata['N']}$, $m={metadata['m']}$, $s={metadata['s']}$, "
        rf"$r={metadata['r']:g}$, $T={metadata['T']:g}$, $L={metadata['L']:g}$; "
        f"mode={metadata['price_mode']}",
        fontsize=13,
    )
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
