#!/usr/bin/env python3
"""Create subject-level TMS evoked-potential figures from D30 sensor trials."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Some conda/pip packaged MNE builds import numba helpers that need a writable
# cache. Setting this before importing MNE keeps the script portable.
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.io
from matplotlib.patches import ConnectionPatch
from scipy.interpolate import Rbf, griddata


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tvbtoolkit.core.paths import stimulation_raw, stimulation_results  # noqa: E402

DEFAULT_MANIFEST = stimulation_raw(
    "stim_data", "python_clean_primary", "tables", "d30_reconstruction_manifest.csv"
)
DEFAULT_OUT = stimulation_results("stim_data", "tep_figures")


def _loadmat(path: Path) -> dict:
    return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)


def _abs_path(path_like: str | float | int | None) -> Path | None:
    if path_like is None or (isinstance(path_like, float) and np.isnan(path_like)):
        return None
    text = str(path_like)
    if not text or text.lower() == "nan":
        return None
    path = Path(text)
    return path if path.is_absolute() else _REPO_ROOT / path


def _bad_channels_0based(value: object) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=int)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return np.array([], dtype=int)
    bad = []
    for part in text.split(";"):
        part = part.strip()
        if part:
            bad.append(int(float(part)) - 1)
    return np.asarray(bad, dtype=int)


def _load_coregistered_sensor_locs(data_dir: Path, n_channels: int) -> tuple[np.ndarray, str]:
    """Load the electrode positions used by the D30 forward model.

    The raw ``D30_Sensorfile.mat::S`` coordinates are the original template
    sensor points. ``D30_CoregSensorfile.mat::sensreg`` and
    ``D30_Fwd_BSTSsensors.mat::Channel.Loc`` are the coregistered sensor
    positions; the latter is what the forward model stores in metres.
    """
    fwd_path = data_dir / "D30_Fwd_BSTSsensors.mat"
    if fwd_path.exists():
        channels = np.asarray(_loadmat(fwd_path)["Channel"], dtype=object).ravel()
        locs = np.asarray([np.asarray(ch.Loc, dtype=float).ravel() for ch in channels])
        return locs[:n_channels], "D30_Fwd_BSTSsensors.mat::Channel.Loc"

    coreg_path = data_dir / "D30_CoregSensorfile.mat"
    if coreg_path.exists():
        locs = np.asarray(_loadmat(coreg_path)["sensreg"], dtype=float) / 1000.0
        return locs[:n_channels], "D30_CoregSensorfile.mat::sensreg"

    sensor_path = data_dir / "D30_Sensorfile.mat"
    coords = np.asarray(_loadmat(sensor_path)["S"], dtype=float)
    coords = coords - np.nanmean(coords, axis=0, keepdims=True)
    return (coords[:n_channels] / 1000.0), "D30_Sensorfile.mat::S"


def _topomap_xy_from_3d(locs_m: np.ndarray, ch_names: list[str]) -> np.ndarray:
    """Project coregistered head coordinates into a top-view 2D map.

    Brainstorm's coregistered EEG coordinates are already in a head-like frame
    where x/y are the lateral/anterior plane and z is height. Passing these
    generic ``EEG 1..60`` points through MNE's 3D dig projection collapses this
    dataset into an artificial radial layout, so for topoplots we use the
    direct x/y head-plane projection and let MNE interpolate on that 2D montage.
    """
    xy = np.asarray(locs_m[:, :2], dtype=float)
    xy -= np.nanmean(xy, axis=0, keepdims=True)
    radius = np.nanmax(np.sqrt(np.sum(xy**2, axis=1)))
    if radius > 0:
        xy /= radius
    return xy


def _trial_average(row: pd.Series) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    d30_path = _abs_path(row["d30_structure_path"])
    dat_path = _abs_path(row["sensor_dat_path"])
    if d30_path is None or dat_path is None:
        raise FileNotFoundError("Missing D30 metadata or sensor .dat path")

    d30 = _loadmat(d30_path)["D30_structure"]
    n_channels = int(d30.Data.Nchannels)
    n_samples = int(d30.Data.Nsamples)
    n_trials = int(d30.Data.Nevents)
    dtype = np.dtype(str(d30.Data.datatype))
    times = np.asarray(d30.Data.samples.times, dtype=float).ravel()
    names = [str(name) for name in np.asarray(d30.Data.channels.name, dtype=object).ravel()]

    data = np.memmap(
        dat_path,
        mode="r",
        dtype=dtype,
        shape=(n_channels, n_samples, n_trials),
        order="F",
    )
    avg = np.asarray(data.mean(axis=2), dtype=float)

    baseline_mask = (times >= -400.0) & (times <= -1.4)
    if baseline_mask.any():
        avg = avg - avg[:, baseline_mask].mean(axis=1, keepdims=True)
    return times, avg, names, _bad_channels_0based(row.get("bad_channels_1based"))


def _peak_latencies(times: np.ndarray, avg: np.ndarray, good_channels: np.ndarray, n_peaks: int = 5) -> np.ndarray:
    post = (times >= 0.0) & (times <= 300.0)
    t_post = times[post]
    if t_post.size == 0:
        return np.array([], dtype=float)

    gfp = np.nanstd(avg[np.ix_(good_channels, post)], axis=0)
    order = np.argsort(gfp)[::-1]
    selected: list[int] = []
    min_sep_ms = 15.0
    for idx in order:
        if all(abs(t_post[idx] - t_post[prev]) >= min_sep_ms for prev in selected):
            selected.append(int(idx))
        if len(selected) == n_peaks:
            break
    selected.sort(key=lambda idx: t_post[idx])
    return t_post[selected]


def _plot_topomap(ax: plt.Axes, xy: np.ndarray, values: np.ndarray, vlim: float) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    try:
        mne.viz.plot_topomap(
            values,
            xy,
            axes=ax,
            show=False,
            sensors=True,
            contours=4,
            outlines="head",
            sphere=(0.0, 0.0, 0.0, 1.0),
            image_interp="cubic",
            extrapolate="head",
            border="mean",
            res=128,
            cmap="RdBu_r",
            vlim=(-vlim, vlim),
        )
        return
    except Exception:
        ax.clear()
        ax.set_aspect("equal")
        ax.axis("off")

    grid_x, grid_y = np.mgrid[-1.05:1.05:90j, -1.05:1.05:90j]
    mask = grid_x**2 + grid_y**2 <= 1.0
    try:
        rbf = Rbf(xy[:, 0], xy[:, 1], values, function="multiquadric", smooth=0.35)
        grid_z = rbf(grid_x, grid_y)
    except Exception:
        grid_z = griddata(xy, values, (grid_x, grid_y), method="linear")
    if grid_z is None or np.all(np.isnan(grid_z)):
        grid_z = griddata(xy, values, (grid_x, grid_y), method="nearest")
    grid_z = np.where(mask, grid_z, np.nan)
    ax.imshow(
        grid_z.T,
        origin="lower",
        extent=(-1.05, 1.05, -1.05, 1.05),
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        interpolation="bicubic",
    )
    ax.scatter(xy[:, 0], xy[:, 1], s=4, color="0.15", alpha=0.35, linewidths=0)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=0.9))
    ax.plot([0, -0.08, 0.08, 0], [1.0, 1.10, 1.10, 1.0], color="black", linewidth=0.8)


def _condition_label(row: pd.Series) -> str:
    session = "NA" if pd.isna(row["session"]) else str(int(row["session"]))
    return f"{str(row['condition']).upper()}  session {session}  trials {int(row['n_trials'])}"


def _repelled_positions(
    times_ms: np.ndarray,
    *,
    xlim: tuple[float, float],
    min_sep: float,
    pad: float,
) -> list[float]:
    """Place topomaps near their timepoints while avoiding obvious overlap."""
    lo, hi = xlim
    span = hi - lo
    targets = [min(1.0 - pad, max(pad, (float(t) - lo) / span)) for t in times_ms]
    if not targets:
        return []

    order = np.argsort(targets)
    placed = [0.0] * len(targets)
    last = -np.inf
    for idx in order:
        pos = max(targets[idx], last + min_sep)
        placed[idx] = pos
        last = pos

    overflow = placed[order[-1]] - (1.0 - pad)
    if overflow > 0:
        for idx in reversed(order):
            placed[idx] -= overflow
            overflow = max(0.0, pad - placed[idx])
            if overflow == 0:
                break
        if overflow > 0:
            for idx in order:
                placed[idx] = min(1.0 - pad, max(pad, placed[idx]))

    for prev, curr in zip(order[:-1], order[1:]):
        if placed[curr] - placed[prev] < min_sep:
            placed[curr] = min(1.0 - pad, placed[prev] + min_sep)
    return placed


def make_subject_figure(subject: str, rows: pd.DataFrame, out_dir: Path) -> list[dict[str, object]]:
    rows = rows.sort_values(["condition", "session", "record_id"], ascending=[False, True, True])
    n_records = len(rows)
    fig_height = max(4.0, 2.65 * n_records)
    fig = plt.figure(figsize=(11.5, fig_height), constrained_layout=False)
    outer = fig.add_gridspec(
        n_records,
        1,
        hspace=0.42,
    )
    fig.suptitle(f"Sensor-level TEPs ({subject})", fontsize=16, weight="bold", y=0.985)
    fig.subplots_adjust(top=0.90)

    summary_rows: list[dict[str, object]] = []
    colors = plt.cm.hsv(np.linspace(0, 1, 60, endpoint=False))

    for row_idx, (_, row) in enumerate(rows.iterrows()):
        times, avg, channel_names, bad = _trial_average(row)
        n_channels = avg.shape[0]
        bad_set = set(int(v) for v in bad if 0 <= int(v) < n_channels)
        good = np.asarray([i for i in range(n_channels) if i not in bad_set], dtype=int)
        if good.size == 0:
            good = np.arange(n_channels)

        d30_path = _abs_path(row["d30_structure_path"])
        data_dir = d30_path.parent / "Data"
        locs_m, montage_source = _load_coregistered_sensor_locs(data_dir, n_channels)
        xy = _topomap_xy_from_3d(locs_m, channel_names)

        baseline_mask = (times >= -400.0) & (times < 0.0)
        tep_mask = (times >= 0.0) & (times <= 300.0)
        plot_mask = (times >= -50.0) & (times <= 300.0)
        trace_vlim = float(np.nanpercentile(np.abs(avg[np.ix_(good, plot_mask)]), 99.0))
        if not math.isfinite(trace_vlim) or trace_vlim == 0:
            trace_vlim = 1.0

        peak_times = _peak_latencies(times, avg, good)
        topo_vlim = float(np.nanpercentile(np.abs(avg[np.ix_(good, tep_mask)]), 98.0))
        if not math.isfinite(topo_vlim) or topo_vlim == 0:
            topo_vlim = 1.0

        row_grid = outer[row_idx, 0].subgridspec(
            2,
            1,
            height_ratios=[0.95, 1.0],
            hspace=0.03,
        )
        ax_toprow = fig.add_subplot(row_grid[0, 0])
        ax_toprow.set_xlim(-50, 300)
        ax_toprow.set_ylim(0, 1)
        ax_toprow.axis("off")
        ax_tep = fig.add_subplot(row_grid[1, 0])

        for ch in range(n_channels):
            alpha = 0.22 if ch not in bad_set else 0.08
            lw = 0.75 if ch not in bad_set else 0.5
            color = colors[ch % len(colors)]
            ax_tep.plot(times[plot_mask], avg[ch, plot_mask], color=color, alpha=alpha, lw=lw)

        ax_tep.axvspan(-50, 0, color="0.94", zorder=-10)
        ax_tep.axhline(0, color="0.25", lw=0.7)
        ax_tep.axvline(0, color="0.05", lw=1.1)
        ax_tep.set_ylim(-trace_vlim * 1.08, trace_vlim * 1.08)
        ax_tep.set_xlim(-50, 300)
        ax_tep.grid(True, color="0.90", linewidth=0.6)
        ax_tep.tick_params(labelsize=8)
        ax_tep.set_ylabel("uV", fontsize=9)
        ax_tep.set_xlabel("Time from TMS pulse (ms)", fontsize=9)
        ax_tep.text(
            0.01,
            0.92,
            _condition_label(row),
            transform=ax_tep.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
        )

        line_color = "0.18"
        for t in peak_times:
            ax_tep.axvline(float(t), color=line_color, lw=1.05, alpha=0.72)

        xlim = ax_tep.get_xlim()
        topo_width = 0.082
        topo_height = 0.82
        topo_pad = topo_width * 0.55
        min_sep = topo_width * 0.88
        topo_x_fracs = _repelled_positions(
            peak_times,
            xlim=(float(xlim[0]), float(xlim[1])),
            min_sep=min_sep,
            pad=topo_pad,
        )

        for topo_idx, topo_x_frac in enumerate(topo_x_fracs):
            ax_topo = ax_toprow.inset_axes(
                [
                    topo_x_frac - topo_width / 2,
                    0.08,
                    topo_width,
                    topo_height,
                ],
                zorder=30,
            )
            if topo_idx < len(peak_times):
                t = float(peak_times[topo_idx])
                sample = int(np.argmin(np.abs(times - t)))
                values = avg[:, sample]
                _plot_topomap(ax_topo, xy[good], values[good], topo_vlim)
                ax_topo.set_title(f"{t:.0f} ms", fontsize=8)
                con = ConnectionPatch(
                    xyA=(0.5, 0.0),
                    coordsA=ax_topo.transAxes,
                    xyB=(t, trace_vlim * 1.04),
                    coordsB=ax_tep.transData,
                    color=line_color,
                    linewidth=1.25,
                    alpha=0.82,
                    zorder=20,
                    clip_on=False,
                )
                fig.add_artist(con)
                ax_tep.plot(
                    [t],
                    [trace_vlim * 1.04],
                    marker="o",
                    markersize=2.6,
                    color=line_color,
                    alpha=0.82,
                    clip_on=False,
                    zorder=21,
                )
            else:
                ax_topo.axis("off")

        summary_rows.append(
            {
                "record_id": row["record_id"],
                "subject": row["subject"],
                "condition": row["condition"],
                "session": int(row["session"]) if not pd.isna(row["session"]) else "",
                "n_trials": int(row["n_trials"]),
                "n_channels": int(n_channels),
                "bad_channels_1based": row.get("bad_channels_1based", ""),
                "topomap_montage_source": montage_source,
                "trace_ylim_uv": trace_vlim,
                "topomap_vlim_uv": topo_vlim,
                "peak_latencies_ms": ";".join(f"{float(t):.3f}" for t in peak_times),
                "peak_latency_selection": "top_5_global_field_power_peaks_0_300ms_min_15ms_apart",
                "displayed_prestimulus_window_ms": "-50..0",
                "plot_window_ms": "-50..300",
                "tep_window_ms": "0..300",
                "baseline_correction_window_ms": "-400..-1.4",
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{subject}_tep_butterfly_topomaps.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    for row in summary_rows:
        row["figure_path"] = str(png_path)
    return summary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--primary-only", action="store_true", default=True)
    parser.add_argument("--include-non-primary", action="store_false", dest="primary_only")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if args.primary_only and "is_primary" in df.columns:
        df = df[df["is_primary"].astype(bool)]
    df = df.dropna(subset=["sensor_dat_path", "d30_structure_path"])
    if args.subjects:
        wanted = {s.upper() for s in args.subjects}
        df = df[df["subject"].str.upper().isin(wanted)]

    all_summary: list[dict[str, object]] = []
    for subject, rows in df.groupby("subject", sort=True):
        print(f"Creating TEP figure for {subject} ({len(rows)} records)", flush=True)
        all_summary.extend(make_subject_figure(str(subject), rows, args.out_dir))

    summary_csv = args.out_dir / "tep_figure_summary.csv"
    pd.DataFrame(all_summary).to_csv(summary_csv, index=False)
    metadata = {
        "manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "n_records": len(all_summary),
        "n_subjects": int(df["subject"].nunique()),
        "subjects": sorted(str(s) for s in df["subject"].unique()),
        "figures": sorted(str(p) for p in args.out_dir.glob("*_tep_butterfly_topomaps.png")),
    }
    (args.out_dir / "tep_figure_summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Wrote summary to {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
