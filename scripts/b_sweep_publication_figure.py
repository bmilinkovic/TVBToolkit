"""Build the publication figure from per-b pickles produced by ``b_sweep_publication_run.py``.

The figure is a 4-column × 7-row panel (one row per b_e value) showing:

    col 0 : Spiking raster (excitatory + inhibitory subsample)
    col 1 : Population firing rate (SNN, exc & inh)
    col 2 : Single-region mean-field — E (Hz, left axis) + W_e (pA, right axis)
    col 3 : Whole-brain DK-68 — 68 regional E traces + global mean

All time axes are aligned (post-transient seconds).

Usage
-----
::

    python scripts/b_sweep_publication_figure.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
PER_B_DIR = REPO_ROOT / "notebooks" / "outputs" / "b_sweep" / "per_b"
FIG_DIR = REPO_ROOT / "notebooks" / "outputs" / "b_sweep"
FIG_DIR.mkdir(parents=True, exist_ok=True)

B_VALUES_PA: tuple[int, ...] = (5, 25, 45, 65, 85, 105, 125)

N_EXC_RASTER: int = 100
N_INH_RASTER: int = 25

COLOR_EXC = "#2B6CB0"
COLOR_INH = "#C05621"
COLOR_MF = "#2E4057"
COLOR_W = "#B4656F"

mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _load_results() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for b in B_VALUES_PA:
        pth = PER_B_DIR / f"b_{b}.pkl"
        if not pth.exists():
            raise FileNotFoundError(
                f"Missing {pth} — run scripts/b_sweep_publication_run.py first."
            )
        with open(pth, "rb") as f:
            out[b] = pickle.load(f)
    return out


def _thin_raster(
    spike_t_ms: np.ndarray,
    spike_id: np.ndarray,
    n_show: int,
    max_id: int,
    cut_ms: float,
):
    """Subsample neurons + drop transient spikes."""
    if spike_t_ms.size == 0:
        return spike_t_ms, spike_id
    spike_t_ms = np.asarray(spike_t_ms, dtype=float)
    spike_id = np.asarray(spike_id, dtype=int)
    # Drop transient
    keep_t = spike_t_ms >= cut_ms
    spike_t_ms = spike_t_ms[keep_t] - cut_ms
    spike_id = spike_id[keep_t]
    if spike_id.size == 0 or max_id == 0:
        return spike_t_ms, spike_id
    rng = np.random.default_rng(0)
    n_show = min(n_show, max_id)
    chosen = np.sort(rng.choice(max_id, size=n_show, replace=False))
    remap = -np.ones(max_id, dtype=np.int64)
    remap[chosen] = np.arange(chosen.size)
    mapped = remap[spike_id]
    mask = mapped >= 0
    return spike_t_ms[mask], mapped[mask].astype(np.int32)


def _region_cmap(n: int):
    base = plt.get_cmap("twilight")
    return [base(x) for x in np.linspace(0.05, 0.95, n)]


def build_figure() -> Path:
    results = _load_results()
    cut_ms = float(results[B_VALUES_PA[0]]["cut_transient_ms"])
    sim_ms = float(results[B_VALUES_PA[0]]["sim_duration_ms"])
    xlim = (0, (sim_ms - cut_ms) / 1000.0)

    n_rows = len(B_VALUES_PA)
    fig = plt.figure(figsize=(16.5, 2.1 * n_rows + 0.8))
    gs = GridSpec(
        n_rows, 4,
        figure=fig,
        width_ratios=[1.15, 1.15, 1.15, 1.55],
        hspace=0.36, wspace=0.32,
        left=0.055, right=0.985, top=0.945, bottom=0.065,
    )

    n_regions = results[B_VALUES_PA[0]]["wb"]["ve_hz"].shape[1]
    reg_colors = _region_cmap(n_regions)
    ax_last_per_col: list = [None] * 4

    for r, b in enumerate(B_VALUES_PA):
        res = results[b]
        snn = res["snn"]
        mf = res["mf"]
        wb = res["wb"]
        is_last = r == n_rows - 1

        # ----- Col 0: raster (raster timestamps are unshifted; we trim transient here) -----
        ax = fig.add_subplot(gs[r, 0])
        ras_exc = snn["raster_exc"]
        ras_inh = snn["raster_inh"]
        # raster_exc/inh are stored as object arrays with shape (2,)
        se_t = np.asarray(ras_exc[0], dtype=float) if ras_exc[0] is not None else np.array([])
        se_i = np.asarray(ras_exc[1], dtype=int) if ras_exc[1] is not None else np.array([], dtype=int)
        si_t = np.asarray(ras_inh[0], dtype=float) if ras_inh[0] is not None else np.array([])
        si_i_raw = np.asarray(ras_inh[1], dtype=int) if ras_inh[1] is not None else np.array([], dtype=int)
        # Inhibitory indices in legacy raster are shifted by n_exc; recover originals.
        si_i = si_i_raw - snn["n_exc"]

        se_t_kept, se_i_kept = _thin_raster(se_t, se_i, N_EXC_RASTER, snn["n_exc"], cut_ms)
        si_t_kept, si_i_kept = _thin_raster(si_t, si_i, N_INH_RASTER, snn["n_inh"], cut_ms)

        ax.scatter(se_t_kept / 1000.0, se_i_kept,
                   s=0.8, c=COLOR_EXC, alpha=0.55, rasterized=True, linewidths=0)
        if si_t_kept.size:
            ax.scatter(si_t_kept / 1000.0, si_i_kept + N_EXC_RASTER,
                       s=0.8, c=COLOR_INH, alpha=0.55, rasterized=True, linewidths=0)
        ax.axhline(N_EXC_RASTER - 0.5, color="#DDDDDD", lw=0.6, zorder=0)
        ax.set_xlim(*xlim)
        ax.set_ylim(-1, N_EXC_RASTER + N_INH_RASTER)
        ax.set_ylabel(f"$b_e$={b} pA\nneuron #", fontsize=9.5)
        if r == 0:
            ax.set_title("Spiking raster", loc="left", fontweight="bold")
        if not is_last:
            ax.set_xticklabels([])
        ax_last_per_col[0] = ax

        # ----- Col 1: population firing rate (SNN) -----
        ax = fig.add_subplot(gs[r, 1])
        t_s = snn["time_ms"] / 1000.0
        ax.plot(t_s, snn["rate_exc_hz"], color=COLOR_EXC, lw=0.9, label="Exc", alpha=0.95)
        ax.plot(t_s, snn["rate_inh_hz"], color=COLOR_INH, lw=0.9, label="Inh", alpha=0.9)
        ax.set_xlim(*xlim)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Pop. rate (Hz)", fontsize=9.5)
        if r == 0:
            ax.set_title("Population firing rate (SNN)", loc="left", fontweight="bold")
            ax.legend(loc="upper right", fontsize=8)
        if not is_last:
            ax.set_xticklabels([])
        ax_last_per_col[1] = ax

        # ----- Col 2: single-region MF E (Hz) + W_e (pA) on twin axis -----
        ax = fig.add_subplot(gs[r, 2])
        t_s = mf["time_ms"] / 1000.0
        axw = ax.twinx()
        axw.plot(t_s, mf["W_pa"], color=COLOR_W, lw=0.9, alpha=0.6, zorder=1)
        axw.set_ylabel("$W_e$ (pA)", color=COLOR_W, fontsize=9)
        axw.tick_params(axis="y", colors=COLOR_W, labelsize=7.5)
        axw.spines["top"].set_visible(False)
        ax.plot(t_s, mf["ve_hz"], color=COLOR_MF, lw=1.15, zorder=5)
        ax.set_zorder(axw.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_xlim(*xlim)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("MF $E$ (Hz)", color=COLOR_MF, fontsize=9.5)
        ax.tick_params(axis="y", colors=COLOR_MF)
        if r == 0:
            ax.set_title("Mean-field ($E$ & $W_e$) — 2nd-order, shared OU drive", loc="left", fontweight="bold")
        if not is_last:
            ax.set_xticklabels([])
        ax_last_per_col[2] = ax

        # ----- Col 3: whole-brain DK-68 regional E traces -----
        ax = fig.add_subplot(gs[r, 3])
        t_s = wb["time_ms"] / 1000.0
        ve_all = wb["ve_hz"]
        for i in range(ve_all.shape[1]):
            ax.plot(t_s, ve_all[:, i], color=reg_colors[i], lw=0.35, alpha=0.55, rasterized=True)
        ax.plot(t_s, ve_all.mean(axis=1), color="black", lw=1.3, label="global mean", zorder=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Regional $E$ (Hz)", fontsize=9.5)
        if r == 0:
            ax.set_title(f"Whole brain — DK-68 ({n_regions} regions)",
                         loc="left", fontweight="bold")
            ax.legend(loc="upper right", fontsize=8)
        if not is_last:
            ax.set_xticklabels([])
        ax_last_per_col[3] = ax

    for ax in ax_last_per_col:
        if ax is not None:
            ax.set_xlabel("Time (s)")

    fig.suptitle(
        "$b_e$ sweep — Sacha et al. 2025 Fig 4 style  |  SNN + single-region MF + WB DK-68 (all TVB Zerlaut 2nd-order)\n"
        "T=20 ms  |  22 s sim, 4 s transient cut  |  paper σ=3.5 Hz, τ_OU=5 ms  |  "
        "$v_\\mathrm{drive}$: 0.4 Hz (SNN+MF), 0.315 Hz (WB) — Fig 4a caption",
        fontsize=11, fontweight="bold", y=0.995,
    )

    pdf_path = FIG_DIR / "b_sweep_snn_mf_whole_brain.pdf"
    png_path = FIG_DIR / "b_sweep_snn_mf_whole_brain.png"
    svg_path = FIG_DIR / "b_sweep_snn_mf_whole_brain.svg"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    print("Saved:")
    for p in (pdf_path, png_path, svg_path):
        print(f"  {p}  ({p.stat().st_size/1024:.1f} KB)")

    # Print numerical summary
    print("\nSummary:")
    print(f"{'b_e':>4} | {'SNN exc':>14} | {'SNN inh':>14} | {'MF E':>10} | {'MF W':>7} | {'WB E':>10}")
    for b in B_VALUES_PA:
        r = results[b]
        snn = r["snn"]; mf = r["mf"]; wb = r["wb"]
        print(
            f"{b:>4} | "
            f"{snn['rate_exc_hz'].mean():6.2f}±{snn['rate_exc_hz'].std():5.2f} | "
            f"{snn['rate_inh_hz'].mean():6.2f}±{snn['rate_inh_hz'].std():5.2f} | "
            f"{mf['ve_hz'].mean():5.2f}±{mf['ve_hz'].std():4.2f} | "
            f"{mf['W_pa'].mean():6.1f} | "
            f"{wb['ve_hz'].mean():5.2f}±{wb['ve_hz'].std():4.2f}"
        )
    return png_path


if __name__ == "__main__":
    build_figure()
