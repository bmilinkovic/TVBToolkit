"""Discovery, loading and reproduction of the tDCS/TMS-EEG PCI artifacts.

The tDCS Stimulation dataset (external
``data_stimulation_liege/raw/stim_data/tdcs-eeg`` by default) stores, for each
patient and session, the source-space Perturbational Complexity Index (PCI)
products of the Liège/Milan "Droutine" pipeline (Casali et al., 2013):

``.../C_tDCS_<ID>/.../Session<N>/Data/Droutine_<PRE|POST>_PCI.mat``

Each ``*_PCI.mat`` holds:

- ``binJ``  : binary significant-source matrix, shape ``(n_sources, n_time)``.
- ``PCI``   : accumulating PCI over time (the scalar PCI is its final value).
- ``H``     : Bernoulli source entropy used for normalization.
- ``Norm``  : Casali normalization factor ``L * H / log2(L)``.

This module discovers those files, recovers the stored scalars, and
*reproduces* PCI directly from ``binJ`` using the toolkit's own
:mod:`tvbtoolkit.complexity.pci_casali` engine, so reproduction can be checked
against the values the original pipeline saved.

The condition tokens (``PRE``/``POST``) refer to **pre- vs post-tDCS** TMS-EEG
acquisitions; tDCS is the intervention delivered between the two TMS-EEG probes.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.io

from ..complexity.pci_casali import (
    binarise_signals,
    binarise_signals_casali,
    lz_complexity_2d,
    pci_norm_factor,
    sort_binJ,
    source_entropy,
)

# Patient folders are named ``C_tDCS_<ID>`` (e.g. C_tDCS_DSM9).
_SUBJECT_RE = re.compile(r"C_tDCS_([A-Za-z0-9]+)")
# PCI files: Droutine_PRE_PCI.mat / Droutine_Post_PCI.mat / Droutine_pre_PCI.mat.
_PCI_FILE_RE = re.compile(r"droutine_(pre|post)_pci\.mat$", re.IGNORECASE)
_SESSION_RE = re.compile(r"session[_ ]?(\d+)", re.IGNORECASE)

__all__ = [
    "PciRecord",
    "PciReproduction",
    "discover_pci_files",
    "load_pci_mat",
    "reproduce_record",
    "reproduce_all",
    "reproduction_to_dict",
    "singletrials_path",
    "load_singletrials_mat",
    "compute_route_pci",
    "compute_reconstructed_vertex_casali",
]


@dataclass(frozen=True)
class PciRecord:
    """Locator + provenance for one stored ``*_PCI.mat`` file."""

    subject: str
    condition: str  # 'pre' | 'post'  (pre-/post-tDCS)
    session: int | None
    variant: str  # 'primary' | 'pre_filter' | 'second_analysis' | 'test'
    is_primary: bool
    path: Path


@dataclass(frozen=True)
class PciReproduction:
    """Stored vs reproduced PCI for one file."""

    record: PciRecord
    n_sources: int
    n_time: int
    n_active: int
    active_fraction: float
    # stored (from the .mat)
    stored_pci: float
    stored_H: float
    stored_norm: float
    # reproduced (recomputed from binJ with the toolkit engine)
    repro_H: float
    repro_norm: float
    lz_sorted: int
    lz_unsorted: int
    pci_sorted: float
    pci_unsorted: float

    @property
    def pci_repro(self) -> float:
        """Best-matching reproduction (unsorted concatenation, see notes)."""
        return self.pci_unsorted

    @property
    def abs_err(self) -> float:
        return abs(self.pci_repro - self.stored_pci)

    @property
    def rel_err(self) -> float:
        if self.stored_pci == 0:
            return float("nan")
        return self.abs_err / abs(self.stored_pci)


def _classify_variant(path: Path) -> tuple[str, bool]:
    """Tag a file by reanalysis lineage; ``is_primary`` flags the canonical run."""
    parts = "/".join(path.parts).lower()
    if "second_analysis" in parts:
        return "second_analysis", False
    if "test" in parts:  # e.g. "Test NotMNI" (non-MNI sanity run)
        return "test", False
    if "pre_filter" in parts:  # SG12's only (canonical) lineage lives here
        return "pre_filter", True
    return "primary", True


def discover_pci_files(eeg_root: str | Path) -> list[PciRecord]:
    """Find every ``Droutine_<PRE|POST>_PCI.mat`` under ``eeg_root``.

    Parameters
    ----------
    eeg_root : path
        The ``stim_data/tdcs-eeg`` raw directory.

    Returns
    -------
    list of :class:`PciRecord`
        Sorted by subject, condition, session.
    """
    root = Path(eeg_root)
    if not root.exists():
        raise FileNotFoundError(f"EEG root not found: {root}")

    records: list[PciRecord] = []
    for path in root.rglob("*.mat"):
        m = _PCI_FILE_RE.search(path.name)
        if m is None:
            continue
        condition = m.group(1).lower()  # 'pre' | 'post'

        subj_match = _SUBJECT_RE.search(str(path))
        if subj_match is None:
            continue
        subject = subj_match.group(1)

        sess_match = _SESSION_RE.search(str(path))
        session = int(sess_match.group(1)) if sess_match else None

        variant, is_primary = _classify_variant(path)
        records.append(
            PciRecord(
                subject=subject,
                condition=condition,
                session=session,
                variant=variant,
                is_primary=is_primary,
                path=path,
            )
        )

    records.sort(key=lambda r: (r.subject, r.condition, r.session or -1, r.variant))
    return records


def load_pci_mat(path: str | Path) -> dict[str, np.ndarray | float]:
    """Load ``binJ`` and the stored PCI/H/Norm scalars from a ``*_PCI.mat``.

    Handles both classic (``scipy.io``) and HDF5 v7.3 MAT files.
    """
    path = Path(path)
    try:
        mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        binJ = np.asarray(mat["binJ"])
        stored_pci_arr = np.atleast_1d(np.asarray(mat["PCI"], dtype=float))
        stored_H = float(np.asarray(mat["H"]))
        stored_norm = float(np.asarray(mat["Norm"]))
    except NotImplementedError:  # HDF5 (v7.3)
        import h5py

        with h5py.File(path, "r") as h:
            binJ = np.asarray(h["binJ"]).T  # MATLAB column-major
            stored_pci_arr = np.atleast_1d(np.asarray(h["PCI"], dtype=float).ravel())
            stored_H = float(np.asarray(h["H"]).ravel()[0])
            stored_norm = float(np.asarray(h["Norm"]).ravel()[0])

    finite_pci = stored_pci_arr.ravel()[np.isfinite(stored_pci_arr.ravel())]
    stored_pci = float(finite_pci[-1]) if finite_pci.size else float("nan")

    return {
        "binJ": (binJ > 0).astype(np.uint8),
        # PCI is saved as an accumulating time course; the scalar is final time.
        "stored_pci": stored_pci,
        "stored_H": stored_H,
        "stored_norm": stored_norm,
    }


def reproduce_record(record: PciRecord) -> PciReproduction:
    """Recompute PCI from ``binJ`` and compare to the stored values.

    Two concatenation conventions for the 2D Lempel-Ziv complexity are reported:

    - ``sorted``   : sources ranked by activation (canonical Casali ordering).
    - ``unsorted`` : native ``binJ`` ordering — empirically the match to the
      values the Liège pipeline stored.

    ``H`` and ``Norm`` reproduce the stored scalars exactly; the small PCI
    residual is the LZ counter's concatenation convention.
    """
    data = load_pci_mat(record.path)
    binJ = data["binJ"]
    n_sources, n_time = binJ.shape
    n_active = int(binJ.sum())

    repro_H = source_entropy(binJ)
    repro_norm = pci_norm_factor(binJ)
    lz_sorted = lz_complexity_2d(sort_binJ(binJ))
    lz_unsorted = lz_complexity_2d(binJ)
    pci_sorted = lz_sorted / repro_norm if repro_norm else float("nan")
    pci_unsorted = lz_unsorted / repro_norm if repro_norm else float("nan")

    return PciReproduction(
        record=record,
        n_sources=int(n_sources),
        n_time=int(n_time),
        n_active=n_active,
        active_fraction=n_active / float(binJ.size),
        stored_pci=data["stored_pci"],
        stored_H=data["stored_H"],
        stored_norm=data["stored_norm"],
        repro_H=repro_H,
        repro_norm=repro_norm,
        lz_sorted=lz_sorted,
        lz_unsorted=lz_unsorted,
        pci_sorted=pci_sorted,
        pci_unsorted=pci_unsorted,
    )


def reproduce_all(
    eeg_root: str | Path,
    *,
    primary_only: bool = False,
    progress: bool = False,
) -> list[PciReproduction]:
    """Discover and reproduce PCI for every file under ``eeg_root``."""
    records = discover_pci_files(eeg_root)
    if primary_only:
        records = [r for r in records if r.is_primary]

    out: list[PciReproduction] = []
    for i, rec in enumerate(records, 1):
        if progress:
            print(f"  [{i:2d}/{len(records)}] {rec.subject:6s} {rec.condition:4s} "
                  f"session={rec.session} ({rec.variant})", flush=True)
        out.append(reproduce_record(rec))
    return out


def singletrials_path(record: PciRecord) -> Path | None:
    """Path to the ``*_resfile_singletrials.mat`` sibling of a PCI file (or None)."""
    sib = Path(re.sub(r"_pci\.mat$", "_resfile_singletrials.mat", str(record.path), flags=re.I))
    return sib if sib.exists() else None


def load_singletrials_mat(path: str | Path) -> dict[str, np.ndarray | None]:
    """Load continuous source products from ``*_resfile_singletrials.mat``.

    The empirical files contain ``J`` as a trial-averaged source time course
    with shape ``(n_sources, n_time)``. Many files also contain ``AllTf`` with
    shape ``(n_sources, n_bins, n_trials)``; this is returned as
    ``trials`` with shape ``(n_trials, n_sources, n_bins)`` when present.
    """
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    J = np.asarray(mat["J"], dtype=float)
    if J.ndim != 2:
        raise ValueError(f"Expected J to be 2D in {path}; got shape {J.shape}.")

    trials = None
    if "AllTf" in mat:
        alltf = np.asarray(mat["AllTf"], dtype=float)
        if alltf.ndim == 3:
            if alltf.shape[0] == J.shape[0]:
                trials = np.transpose(alltf, (2, 0, 1))
            elif alltf.shape[1] == J.shape[0]:
                trials = np.transpose(alltf, (0, 1, 2))
            else:
                raise ValueError(
                    f"Cannot orient AllTf from {path}; got shape {alltf.shape} "
                    f"for J shape {J.shape}."
                )
    return {"J": J, "trials": trials}


def _d30_structure_path(record: PciRecord) -> Path:
    return record.path.parent.parent / "D30_structure.mat"


def _load_d30_structure(record: PciRecord):
    path = _d30_structure_path(record)
    if not path.exists():
        raise FileNotFoundError(f"D30_structure.mat not found for {record.path}")
    return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)["D30_structure"]


def _load_pci_times(record: PciRecord) -> np.ndarray:
    mat = scipy.io.loadmat(record.path, squeeze_me=True, struct_as_record=False)
    return np.asarray(mat["parameters"].times, dtype=float)


def _load_boot_interval(record: PciRecord) -> np.ndarray:
    stat_path = Path(str(record.path).replace("_PCI.mat", "_resfile_statistics_bootstrap.mat"))
    if stat_path.exists():
        mat = scipy.io.loadmat(stat_path, squeeze_me=True, struct_as_record=False)
        if "bootinterval" in mat:
            return np.asarray(mat["bootinterval"], dtype=float).ravel()
    struct = _load_d30_structure(record)
    return np.asarray(struct.Data.BaseCorr.times, dtype=float).ravel()


def _nearest_time_indices(times: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    return np.asarray([int(np.argmin(np.abs(times - t))) for t in wanted], dtype=int)


def _d30_sensor_memmap(record: PciRecord, struct):
    data_dir = record.path.parent
    dat_path = data_dir / str(struct.Data.fnamedat)
    if not dat_path.exists():
        raise FileNotFoundError(f"D30 sensor .dat not found: {dat_path}")

    n_samples = int(struct.Data.Nsamples)
    n_channels = int(struct.Data.Nchannels)
    n_trials = int(struct.Data.Nevents)
    dtype = np.float32 if str(struct.Data.datatype).lower() == "float32" else np.float64

    expected = n_samples * n_channels * n_trials * np.dtype(dtype).itemsize
    actual = dat_path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"Unexpected D30 .dat size for {dat_path}: got {actual} bytes, "
            f"expected {expected}."
        )

    # D30 stores trials as MATLAB column-major (channels, samples, trials).
    return np.memmap(
        dat_path,
        dtype=dtype,
        mode="r",
        shape=(n_channels, n_samples, n_trials),
        order="F",
    )


def _reconstruct_source_window_memmap(
    record: PciRecord,
    selected_indices: np.ndarray,
    *,
    work_dir: str | Path | None = None,
) -> tuple[np.memmap, Path]:
    """Reconstruct ``(trial, source, selected_time)`` source data to a memmap."""
    sib = singletrials_path(record)
    if sib is None:
        raise FileNotFoundError(f"Missing *_resfile_singletrials.mat for {record.path}")
    single = load_singletrials_mat(sib)
    trials = single["trials"]
    if trials is None:
        raise ValueError(f"Missing AllTf in {sib}; cannot reconstruct source trials.")

    struct = _load_d30_structure(record)
    sensor = _d30_sensor_memmap(record, struct)
    n_channels, _, n_trials = sensor.shape
    if trials.shape[0] != n_trials:
        raise ValueError(
            "AllTf/sensor dimensions do not align: "
            f"AllTf as trials={trials.shape}, sensor={sensor.shape}."
        )

    if trials.shape[2] == n_channels:
        channel_idx = np.arange(n_channels, dtype=int)
    else:
        bad = np.atleast_1d(np.asarray(struct.Data.channels.Bad, dtype=int)).ravel()
        bad = bad[bad > 0]
        # MATLAB channel indices are 1-based in these structures.
        bad0 = bad - 1 if bad.size and bad.min() >= 1 else bad
        keep = np.ones(n_channels, dtype=bool)
        keep[bad0] = False
        channel_idx = np.flatnonzero(keep)
        if channel_idx.size != trials.shape[2]:
            raise ValueError(
                "AllTf/sensor channel dimensions do not align after bad-channel "
                f"masking: AllTf={trials.shape}, sensor={sensor.shape}, "
                f"bad_channels={bad.tolist()}."
            )

    n_sources = int(trials.shape[1])
    n_time = int(len(selected_indices))
    work_root = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir())
    work_root.mkdir(parents=True, exist_ok=True)
    out_path = work_root / (
        f"{record.subject}_{record.condition}_S{record.session}_source_window.dat"
    )
    out = np.memmap(
        out_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_trials, n_sources, n_time),
    )

    for k in range(n_trials):
        sensor_trial = np.asarray(sensor[channel_idx[:, None], selected_indices, k], dtype=np.float64)
        out[k] = np.asarray(trials[k], dtype=np.float64) @ sensor_trial
    out.flush()
    return out, out_path


def _binarise_reconstructed_source_memmap(
    source_trials: np.memmap,
    t_stim: int,
    *,
    n_bootstrap: int,
    alpha: float,
    seed: int | None,
) -> np.ndarray:
    """Casali bootstrap binarisation for a reconstructed source-trial memmap."""
    if source_trials.ndim != 3:
        raise ValueError(f"Expected source_trials to be 3D, got {source_trials.shape}.")
    n_trials, n_sources, n_bins = source_trials.shape
    if n_trials < 2:
        raise ValueError("Reconstructed Casali PCI requires at least two trials.")
    if not (1 <= t_stim < n_bins):
        raise ValueError(f"t_stim must be in [1, n_bins-1], got {t_stim}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    base_means = np.empty((n_trials, n_sources), dtype=np.float32)
    base_ss = np.zeros(n_sources, dtype=np.float64)
    avg_sum = np.zeros((n_sources, n_bins), dtype=np.float64)

    for k in range(n_trials):
        trial = np.asarray(source_trials[k], dtype=np.float64)
        mean = trial[:, :t_stim].mean(axis=1)
        base_means[k] = mean.astype(np.float32)
        bc = trial - mean[:, None]
        base_ss += np.square(bc[:, :t_stim]).sum(axis=1)
        avg_sum += bc

    base_sd = np.sqrt(base_ss / float(n_trials * t_stim))
    base_sd = np.where(base_sd < np.finfo(float).eps, 1.0, base_sd)
    avg = (avg_sum / float(n_trials)) / base_sd[:, None]

    rng = np.random.default_rng(seed)
    maxstat = np.empty(int(n_bootstrap), dtype=float)
    for b in range(int(n_bootstrap)):
        idx = rng.integers(0, n_trials, n_trials)
        counts = np.bincount(idx, minlength=n_trials).astype(np.float64)
        boot = np.zeros((n_sources, t_stim), dtype=np.float64)
        for k, weight in enumerate(counts):
            if weight == 0:
                continue
            base = np.asarray(source_trials[k, :, :t_stim], dtype=np.float64)
            boot += weight * (base - base_means[k, :, None])
        boot /= float(n_trials)
        boot /= base_sd[:, None]
        maxstat[b] = float(np.abs(boot).max())

    thresh = float(np.quantile(maxstat, 1.0 - alpha))
    return (np.abs(avg) > thresh).astype(np.uint8)


def compute_reconstructed_vertex_casali(
    record: PciRecord,
    *,
    n_bootstrap: int = 100,
    alpha: float = 0.01,
    seed: int | None = 0,
    work_dir: str | Path | None = None,
    keep_tensor: bool = False,
) -> dict[str, object]:
    """Reconstruct vertex-level source trials and compute Casali-style PCI.

    This rebuilds the missing source-trial tensor from the saved D30 sensor
    trials and per-trial inverse maps:

    ``AllTf[:, :, trial] @ sensor_trial``.

    Only the bootstrap baseline and saved PCI post-stimulus times are
    reconstructed. The output PCI is therefore directly comparable in shape to
    the stored vertex-level ``binJ``/PCI, although small differences from the
    historic MATLAB implementation are still expected.
    """
    struct = _load_d30_structure(record)
    times = np.asarray(struct.Data.samples.times, dtype=float)
    boot_interval = _load_boot_interval(record)
    pci_times = _load_pci_times(record)

    base_idx = np.where(
        (times >= boot_interval[0] - 1e-9) & (times <= boot_interval[1] + 1e-9)
    )[0]
    post_idx = _nearest_time_indices(times, pci_times)
    selected = np.concatenate([base_idx, post_idx])
    t_stim = int(len(base_idx))

    source_trials, tensor_path = _reconstruct_source_window_memmap(
        record, selected, work_dir=work_dir
    )
    try:
        bin_full = _binarise_reconstructed_source_memmap(
            source_trials,
            t_stim,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=seed,
        )
        bin_post = bin_full[:, t_stim:]
        stored = load_pci_mat(record.path)
        stored_binJ = np.asarray(stored["binJ"], dtype=np.uint8)
        out = {
            **asdict(record),
            "path": str(record.path),
            "n_bootstrap": int(n_bootstrap),
            "alpha": float(alpha),
            "baseline_bins": int(t_stim),
            "post_bins": int(len(post_idx)),
            "source_trial_shape": tuple(int(v) for v in source_trials.shape),
            "tensor_path": str(tensor_path) if keep_tensor else None,
            "stored_pci": float(stored["stored_pci"]),
            "reconstructed_casali_pci_sorted": _pci_from_binJ(bin_post, sort_sources=True),
            "reconstructed_casali_pci_unsorted": _pci_from_binJ(bin_post, sort_sources=False),
            "stored_active_frac": float(stored_binJ.mean()),
            "reconstructed_active_frac": float(bin_post.mean()),
            "binJ_jaccard": float(
                np.logical_and(bin_post, stored_binJ).sum()
                / max(np.logical_or(bin_post, stored_binJ).sum(), 1)
            ),
            "binJ_match_fraction": float((bin_post == stored_binJ).mean()),
        }
    finally:
        if not keep_tensor:
            try:
                del source_trials
                tensor_path.unlink(missing_ok=True)
            except Exception:
                pass
    return out


def _detect_onset(J: np.ndarray) -> int:
    """Detect stimulus onset bin from trial-averaged source currents.

    Uses the steepest rise of the (smoothed) RMS-over-sources, constrained to the
    central 20–85% of the epoch so baseline edge artifacts are not picked.
    """
    rms = np.sqrt((J ** 2).mean(axis=0))
    k = 5
    smooth = np.convolve(rms, np.ones(k) / k, mode="same")
    grad = np.gradient(smooth)
    lo, hi = int(0.20 * rms.size), int(0.85 * rms.size)
    return lo + int(np.argmax(grad[lo:hi]))


def _pci_from_binJ(binJ: np.ndarray, *, sort_sources: bool = True) -> float:
    b = binJ.astype(np.uint8)
    if sort_sources:
        b = sort_binJ(b)
    if not np.any(b):
        return 0.0
    return float(lz_complexity_2d(b) / max(pci_norm_factor(b), np.finfo(float).eps))


def compute_route_pci(
    record: PciRecord,
    *,
    n_bootstrap: int = 500,
    alpha: float = 0.01,
    nshuffles: int = 10,
    percentile: float = 100.0,
    single_trial_strategy: str | None = None,
) -> dict[str, object] | None:
    """Three-way PCI comparison for one file: empirical vs both binarise routes.

    Loads the continuous source products from the ``*_resfile_singletrials.mat``
    sibling. The legacy TVBSim route is computed from trial-averaged ``J`` for
    backward comparison. The Casali/bootstrap route is computed only when a
    trial stack contains a baseline plus a post-stimulus window; otherwise its
    PCI is reported as ``NaN`` with a status note. Returns ``None`` if the
    sibling is missing.

    Returns a dict with stored / tvbsim / casali PCI and active fractions.
    """
    sib = singletrials_path(record)
    if sib is None:
        return None

    stored = load_pci_mat(record.path)
    n_post = int(stored["binJ"].shape[1])  # type: ignore[index]
    stored_binJ = np.asarray(stored["binJ"])

    single = load_singletrials_mat(sib)
    J = np.asarray(single["J"], dtype=float)  # (n_sources, n_time)
    trials = single["trials"]
    onset = _detect_onset(J)

    # Legacy route: trial-averaged J, post window matched to stored binJ.
    tv = binarise_signals(J[np.newaxis], t_stim=onset, nshuffles=nshuffles,
                          percentile=percentile)[0]
    tv_post = tv[:, onset:onset + n_post]

    casali_status = "unavailable"
    ca_post = None
    trial_shape = None if trials is None else tuple(int(v) for v in trials.shape)
    if trials is not None and trials.shape[2] > n_post:
        t_stim_trials = int(trials.shape[2] - n_post)
        try:
            ca = binarise_signals_casali(
                trials,
                t_stim=t_stim_trials,
                n_bootstrap=n_bootstrap,
                alpha=alpha,
            )
            ca_post = ca[:, t_stim_trials:t_stim_trials + n_post]
            casali_status = "trial_bootstrap"
        except ValueError as exc:
            casali_status = f"unavailable: {exc}"
    elif trials is not None:
        casali_status = (
            "unavailable: AllTf has no identifiable baseline+post window "
            f"(trial bins={trials.shape[2]}, stored post bins={n_post})"
        )

    if ca_post is None and single_trial_strategy is not None:
        try:
            ca = binarise_signals_casali(
                J[np.newaxis],
                t_stim=onset,
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                single_trial=single_trial_strategy,
            )
            ca_post = ca[:, onset:onset + n_post]
            casali_status = f"single_trial_{single_trial_strategy}"
        except ValueError as exc:
            casali_status = f"unavailable: {exc}"

    casali_sorted = _pci_from_binJ(ca_post, sort_sources=True) if ca_post is not None else float("nan")
    casali_unsorted = _pci_from_binJ(ca_post, sort_sources=False) if ca_post is not None else float("nan")

    rec = asdict(record)
    rec["path"] = str(record.path)
    return {
        **rec,
        "onset": int(onset),
        "n_post": n_post,
        "J_shape": tuple(int(v) for v in J.shape),
        "trial_shape": trial_shape,
        "stored_pci": float(stored["stored_pci"]),
        "tvbsim_pci": _pci_from_binJ(tv_post, sort_sources=True),
        "tvbsim_pci_sorted": _pci_from_binJ(tv_post, sort_sources=True),
        "tvbsim_pci_unsorted": _pci_from_binJ(tv_post, sort_sources=False),
        "casali_pci": casali_sorted,
        "casali_pci_sorted": casali_sorted,
        "casali_pci_unsorted": casali_unsorted,
        "casali_status": casali_status,
        "stored_active_frac": float(stored_binJ.mean()),
        "tvbsim_active_frac": float(tv_post.mean()),
        "casali_active_frac": float(ca_post.mean()) if ca_post is not None else float("nan"),
    }


def reproduction_to_dict(rep: PciReproduction) -> dict[str, object]:
    """Flatten a :class:`PciReproduction` into a tabular row."""
    rec = asdict(rep.record)
    rec["path"] = str(rep.record.path)
    return {
        **rec,
        "n_sources": rep.n_sources,
        "n_time": rep.n_time,
        "n_active": rep.n_active,
        "active_fraction": rep.active_fraction,
        "stored_pci": rep.stored_pci,
        "stored_H": rep.stored_H,
        "stored_norm": rep.stored_norm,
        "repro_H": rep.repro_H,
        "repro_norm": rep.repro_norm,
        "lz_sorted": rep.lz_sorted,
        "lz_unsorted": rep.lz_unsorted,
        "pci_sorted": rep.pci_sorted,
        "pci_unsorted": rep.pci_unsorted,
        "pci_repro": rep.pci_repro,
        "abs_err": rep.abs_err,
        "rel_err": rep.rel_err,
    }
