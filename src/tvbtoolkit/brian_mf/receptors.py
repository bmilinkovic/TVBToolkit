"""Receptor helpers ported from legacy `brian_MF/receptors.py`.

Hansen receptor atlas
---------------------
The primary data source for AAL-90 receptor densities is the Hansen et al.
(2022, *Nature Neuroscience*) PET receptor atlas, parcellated from volumetric
NIfTI images into AAL-90 space using nilearn (Brain-Act pipeline).

The atlas is stored in ``data/receptors/hansen_receptors_aal90.csv``:
  - 90 rows  : AAL regions, Precentral_L → Temporal_Inf_R (no cerebellum)
  - 37 columns: individual PET tracer maps (see ``HANSEN_TRACER_NAMES``)
  - Values   : max-scaled to [0, 1] within each tracer map

Reference: Hansen, J.Y. et al. (2022). Mapping neurotransmitter systems to
the structural and functional organization of the human neocortex.
*Nature Neuroscience*, 25, 1569–1581.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

_REPO_ROOT        = Path(__file__).resolve().parents[3]
_RECEPTOR_MAT     = _REPO_ROOT / "data" / "receptors" / "serotonin_receptors_PET_DesikanKilliany68.mat"
_HANSEN_AAL90_CSV = _REPO_ROOT / "data" / "receptors" / "hansen_receptors_aal90.csv"

# ---------------------------------------------------------------------------
# Hansen receptor atlas — tracer names in column order
# ---------------------------------------------------------------------------
HANSEN_TRACER_NAMES: list[str] = [
    "H3_cban_hc8_gallezot",
    "GABAa-bz_flumazenil_hc16_norgaard",
    "CB1_FMPEPd2_hc22_laurikainen",
    "CB1_omar_hc77_normandin",
    "VAChT_feobv_hc18_aghourian_sum",
    "5HT2a_alt_hc19_savli",
    "5HTT_dasb_hc100_beliveau",
    "D2_raclopride_hc7_alakurtti",
    "5HT2a_mdl_hc3_talbot",
    "5HT2a_cimbi_hc29_beliveau",
    "5HT4_sb20_hc59_beliveau",
    "DAT_fepe2i_hc6_sasaki",
    "VAChT_feobv_hc4_tuominen",
    "MU_carfentanil_hc204_kantonen",
    "NAT_MRB_hc77_ding",
    "D2_flb457_hc37_smith",
    "A4B2_flubatine_hc30_hillmer",
    "GABAa_flumazenil_hc6_dukart",
    "NMDA_ge179_hc29_galovic",
    "D2_fallypride_hc49_jaworska",
    "5HTT_dasb_hc30_savli",
    "5HT1a_cumi_hc8_beliveau",
    "5HT1b_p943_hc22_savli",
    "VAChT_feobv_hc5_bedard_sum",
    "5HT1a_way_hc36_savli",
    "mGluR5_abp_hc22_rosaneto",
    "mGluR5_abp_hc28_dubois",
    "mGluR5_abp_hc73_smart",
    "D1_SCH23390_hc13_kaller",
    "MU_carfentanil_hc39_turtonen",
    "DAT_fpcit_hc174_dukart_spect",
    "NAT_MRB_hc10_hesse",
    "5HT1b_az_hc36_beliveau",
    "5HT1b_p943_hc65_gallezot",
    "M1_lsn_hc24_naganawa",
    "D2_flb457_hc55_sandiego",
    "5HT6_gsk_hc30_radhakrishnan",
]

# Grouped summary of what's available
HANSEN_RECEPTOR_SYSTEMS: dict[str, list[str]] = {
    "Serotonin (5-HT)": [t for t in HANSEN_TRACER_NAMES if t.startswith("5HT")],
    "Dopamine (D1/D2/DAT)": [t for t in HANSEN_TRACER_NAMES if t.startswith(("D1", "D2", "DAT"))],
    "GABA": [t for t in HANSEN_TRACER_NAMES if t.startswith("GABA")],
    "Glutamate (mGluR5/NMDA)": [t for t in HANSEN_TRACER_NAMES if t.startswith(("mGluR", "NMDA"))],
    "Cannabinoid (CB1)": [t for t in HANSEN_TRACER_NAMES if t.startswith("CB1")],
    "Opioid (MU)": [t for t in HANSEN_TRACER_NAMES if t.startswith("MU")],
    "Noradrenaline (NAT)": [t for t in HANSEN_TRACER_NAMES if t.startswith("NAT")],
    "Acetylcholine (M1/A4B2/VAChT)": [t for t in HANSEN_TRACER_NAMES if t.startswith(("M1", "A4B2", "VAChT"))],
    "Histamine (H3)": [t for t in HANSEN_TRACER_NAMES if t.startswith("H3")],
}

# ---------------------------------------------------------------------------
# DK-68 → AAL-116 approximate mapping
# ---------------------------------------------------------------------------
# For each of the 116 AAL regions (0-indexed), this table stores the index
# into the DK-68 receptor array (0-indexed, LH 0-33, RH 34-67).
# Special sentinel values:
#   -1  subcortical (Hippocampus, Amygdala, BG, Thalamus) — fixed values
#   -2  cerebellum — fixed near-zero value
#
# DK-68 LH order:
#   0  bankssts          8  isthmus_cingulate  16 pars_opercularis  24 rostral_ant_cingulate
#   1  caud_ant_cing     9  lateral_occipital  17 pars_orbitalis    25 rostral_mid_frontal
#   2  caud_mid_frontal  10 lat_orbitofrontal  18 pars_triangularis 26 superior_frontal
#   3  cuneus            11 lingual            19 pericalcarine     27 superior_parietal
#   4  entorhinal        12 med_orbitofrontal  20 postcentral       28 superior_temporal
#   5  fusiform          13 middle_temporal    21 posterior_cing    29 supramarginal
#   6  inferior_parietal 14 parahippocampal    22 precentral        30 frontal_pole
#   7  inferior_temporal 15 paracentral        23 precuneus         31 temporal_pole
#                                                                   32 transverse_temporal
#                                                                   33 insula
# RH: same order offset by +34.

_AAL116_TO_DK68: list[int] = [
    # --- Frontal (AAL 0-27) ---
    22, 56,   # 0-1   Precentral L/R
    26, 60,   # 2-3   Frontal_Sup L/R
    30, 64,   # 4-5   Frontal_Sup_Orb L/R  (frontal pole proxy)
    25, 59,   # 6-7   Frontal_Mid L/R  (rostral mid-frontal)
    10, 44,   # 8-9   Frontal_Mid_Orb L/R  (lat orbitofrontal)
    16, 50,   # 10-11 Frontal_Inf_Oper L/R
    18, 52,   # 12-13 Frontal_Inf_Tri L/R
    17, 51,   # 14-15 Frontal_Inf_Orb L/R  (pars orbitalis)
    16, 50,   # 16-17 Rolandic_Oper L/R  (pars opercularis proxy)
    26, 60,   # 18-19 Supp_Motor_Area L/R  (superior frontal proxy)
    12, 46,   # 20-21 Olfactory L/R  (medial orbitofrontal)
    26, 60,   # 22-23 Frontal_Sup_Medial L/R
    12, 46,   # 24-25 Frontal_Med_Orb L/R
    12, 46,   # 26-27 Rectus L/R  (gyrus rectus = med orbitofrontal)
    # --- Limbic / Cingulate (AAL 28-41) ---
    33, 67,   # 28-29 Insula L/R
    24, 58,   # 30-31 Cingulum_Ant L/R  (rostral ant cingulate)
     1, 35,   # 32-33 Cingulum_Mid L/R  (caudal ant cingulate)
    21, 55,   # 34-35 Cingulum_Post L/R  (posterior cingulate)
    -1, -1,   # 36-37 Hippocampus L/R  (subcortical)
    14, 48,   # 38-39 ParaHippocampal L/R
    -1, -1,   # 40-41 Amygdala L/R  (subcortical)
    # --- Occipital (AAL 42-53) ---
    19, 53,   # 42-43 Calcarine L/R  (pericalcarine)
     3, 37,   # 44-45 Cuneus L/R
    11, 45,   # 46-47 Lingual L/R
     9, 43,   # 48-49 Occipital_Sup L/R
     9, 43,   # 50-51 Occipital_Mid L/R
     9, 43,   # 52-53 Occipital_Inf L/R
    # --- Temporal (AAL 54-89) ---
     5, 39,   # 54-55 Fusiform L/R
    20, 54,   # 56-57 Postcentral L/R
    27, 61,   # 58-59 Parietal_Sup L/R
     6, 40,   # 60-61 Parietal_Inf L/R
    29, 63,   # 62-63 SupraMarginal L/R
     6, 40,   # 64-65 Angular L/R  (part of inf parietal in DK)
    23, 57,   # 66-67 Precuneus L/R
    15, 49,   # 68-69 Paracentral_Lobule L/R
    # --- Subcortical (AAL 70-77) ---
    -1, -1,   # 70-71 Caudate L/R
    -1, -1,   # 72-73 Putamen L/R
    -1, -1,   # 74-75 Pallidum L/R
    -1, -1,   # 76-77 Thalamus L/R
    # --- Temporal (continued, AAL 78-89) ---
    32, 66,   # 78-79 Heschl L/R  (transverse temporal)
    28, 62,   # 80-81 Temporal_Sup L/R
    31, 65,   # 82-83 Temporal_Pole_Sup L/R
    13, 47,   # 84-85 Temporal_Mid L/R
    31, 65,   # 86-87 Temporal_Pole_Mid L/R  (temporal pole proxy)
     7, 41,   # 88-89 Temporal_Inf L/R
    # --- Cerebellum (AAL 90-115) ---
    -2, -2,   # 90-91  Cerebelum_Crus1 L/R
    -2, -2,   # 92-93  Cerebelum_Crus2 L/R
    -2, -2,   # 94-95  Cerebelum_3 L/R
    -2, -2,   # 96-97  Cerebelum_4_5 L/R
    -2, -2,   # 98-99  Cerebelum_6 L/R
    -2, -2,   # 100-101 Cerebelum_7b L/R
    -2, -2,   # 102-103 Cerebelum_8 L/R
    -2, -2,   # 104-105 Cerebelum_9 L/R
    -2, -2,   # 106-107 Cerebelum_10 L/R
    -2, -2,   # 108-109 Vermis_1_2, Vermis_3
    -2, -2,   # 110-111 Vermis_4_5, Vermis_6
    -2, -2,   # 112-113 Vermis_7, Vermis_8
    -2, -2,   # 114-115 Vermis_9, Vermis_10
]

# Literature-based 5-HT2A density for subcortical structures, scaled to
# match the raw PET Bmax units in the DK-68 mat file (cortical range ≈ 30–63).
# Sources: Erritzoe et al. 2009 (brain-wide PET), Beliveau et al. 2017.
# Values expressed as fraction of cortical mean (~52) × cortical mean.
_SUBCORTICAL_5HT2A = {
    "hippocampus": 20.0,  # ~38 % of cortical mean; CA1-CA3 has modest 5-HT2A
    "amygdala":    25.0,  # ~48 % basolateral amygdala has notable 5-HT2A
    "caudate":     15.0,  # ~29 % low, primarily dopaminergic
    "putamen":     15.0,
    "pallidum":     8.0,  # ~15 % very low
    "thalamus":    18.0,  # ~35 % moderate in some nuclei
}
_CEREB_5HT2A = 5.0        # ~10 % of cortical mean; near-background in cerebellum


def get_5ht2a_receptors() -> np.ndarray:
    """Return legacy 5-HT2A receptor density map (Desikan-like cortical order)."""

    return np.array(
        [
            5.96,
            4.48,
            3.59,
            4.27,
            3.93,
            4.26,
            4.24,
            3.89,
            3.73,
            4.38,
            3.72,
            4.56,
            3.98,
            4.49,
            4.31,
            4.81,
            4.65,
            4.18,
            3.9,
            4.14,
            4.15,
            4.39,
            3.5,
            4.77,
            4.02,
            4.34,
            4.2,
            4.52,
            4.26,
            4.55,
            4.52,
            4.22,
            4.13,
            4.53,
            4.42,
            4.21,
            4.17,
            3.95,
            4.36,
            3.55,
            4.17,
            3.77,
            4.31,
            3.69,
            4.73,
            4.56,
            4.48,
            4.28,
            4.64,
            4.28,
            4.3,
            3.92,
            4.17,
            4.13,
            4.58,
            4.03,
            4.42,
            4.47,
            4.22,
            4.37,
            4.17,
            4.33,
            3.93,
            4.06,
            3.88,
            4.21,
            4.25,
            4.26,
        ]
    )


def get_5ht1a_receptors() -> np.ndarray:
    """Return legacy 5-HT1A receptor density map placeholder (same order as 5-HT2A)."""

    # Legacy code did not include a dedicated alternative map.
    return get_5ht2a_receptors().copy()


def get_g_k_values(
    g_k_max: float,
    g_k_min: float,
    include_5ht1a: bool = False,
    fht1a_effect: float = 1.0,
    receptors: np.ndarray | None = None,
) -> np.ndarray:
    """Map receptor densities to potassium conductance values.

    This mirrors the simple min-max normalization used in the legacy module.
    """

    rec = np.asarray(receptors) if receptors is not None else get_5ht2a_receptors()
    rec_norm = (rec - rec.min()) / (rec.max() - rec.min() + 1e-12)
    g_k = g_k_min + rec_norm * (g_k_max - g_k_min)
    if include_5ht1a:
        rec1 = get_5ht1a_receptors()
        rec1_norm = (rec1 - rec1.min()) / (rec1.max() - rec1.min() + 1e-12)
        g_k = g_k * (1.0 - fht1a_effect * rec1_norm)
    return g_k


def get_hansen_receptors_aal90(
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return the full Hansen receptor atlas in AAL-90 space.

    Loads ``data/receptors/hansen_receptors_aal90.csv``:

    - **90 rows** : AAL regions ``Precentral_L`` → ``Temporal_Inf_R``
      (cerebellum excluded).
    - **37 columns**: individual PET tracer maps (see ``HANSEN_TRACER_NAMES``).
    - **Values**   : max-scaled to [0, 1] within each tracer.

    Parameters
    ----------
    csv_path : path-like, optional
        Override path to the CSV.

    Returns
    -------
    pd.DataFrame, shape (90, 37)
        Index = AAL region name.  Columns = tracer names from
        ``HANSEN_TRACER_NAMES``.

    Examples
    --------
    >>> rec = get_hansen_receptors_aal90()
    >>> rec.shape
    (90, 37)
    >>> rec["5HT2a_cimbi_hc29_beliveau"].values   # 5-HT2A density
    array([...])
    >>> rec[["5HT2a_cimbi_hc29_beliveau", "D1_SCH23390_hc13_kaller"]]  # subset
    """
    path = Path(csv_path) if csv_path is not None else _HANSEN_AAL90_CSV
    return pd.read_csv(str(path), index_col=0)


def get_5ht2a_aal90(
    tracer: str = "cimbi",
    csv_path: str | Path | None = None,
) -> np.ndarray:
    """Return 5-HT2A receptor density in AAL-90 space (no cerebellum).

    Convenience wrapper around :func:`get_hansen_receptors_aal90`.
    Three 5-HT2A tracers are available:

    - ``"cimbi"``  — [¹¹C]Cimbi-36, Beliveau et al. 2017, n=29  **(default)**
    - ``"savli"``  — [¹⁸F]altanserin, Savli et al. 2012, n=19
    - ``"talbot"`` — [¹⁸F]MDL 100907, Talbot et al. 2012, n=3

    Values are max-scaled to [0, 1].

    Parameters
    ----------
    tracer : str
        Which 5-HT2A tracer to return.
    csv_path : path-like, optional
        Override path to the Hansen atlas CSV.

    Returns
    -------
    np.ndarray, shape (90,)
        5-HT2A density per AAL-90 region, max-scaled to [0, 1].
    """
    _TRACER_COL = {
        "cimbi":  "5HT2a_cimbi_hc29_beliveau",
        "savli":  "5HT2a_alt_hc19_savli",
        "talbot": "5HT2a_mdl_hc3_talbot",
    }
    if tracer not in _TRACER_COL:
        raise ValueError(f"tracer must be one of {list(_TRACER_COL)}; got {tracer!r}")
    df = get_hansen_receptors_aal90(csv_path)
    return df[_TRACER_COL[tracer]].values.astype(float)


def get_5ht2a_aal116(receptor_mat_path: str | Path | None = None) -> np.ndarray:
    """Return 5-HT2A receptor density mapped to AAL-116 space.

    Loads the Desikan-Killiany 68-region PET receptor atlas
    (Beliveau et al. 2017 / Hansen et al. 2022 style), extracts the 5-HT2A
    column, and remaps it to the 116-region AAL atlas using the approximate
    anatomical correspondence table ``_AAL116_TO_DK68``.

    Subcortical AAL regions (Hippocampus, Amygdala, BG, Thalamus) that have
    no DK-68 cortical counterpart are filled with literature-based PET
    estimates.  Cerebellar regions are set to near-zero.

    Parameters
    ----------
    receptor_mat_path : path-like, optional
        Path to ``serotonin_receptors_PET_DesikanKilliany68.mat``.
        Defaults to ``data/receptors/`` relative to the repo root.

    Returns
    -------
    np.ndarray, shape (116,)
        5-HT2A density per AAL-116 region in the original PET units
        (roughly [0.4, 6.0]).

    Notes
    -----
    The DK-68 → AAL-116 mapping is anatomically approximate.  Several AAL
    regions have no 1-to-1 DK equivalent and use the nearest plausible proxy
    (documented in ``_AAL116_TO_DK68``).  For final analyses, replace with a
    volumetric resample from a 5-HT2A PET volume in MNI space.
    """
    path = Path(receptor_mat_path) if receptor_mat_path is not None else _RECEPTOR_MAT
    mat = sio.loadmat(str(path))
    # receptors: shape (68, 6), column 2 is 5HT2a
    ht2a_dk68 = mat["receptors"][:, 2].astype(float)  # (68,)

    # Fixed subcortical values ordered by AAL subcortical region index
    # (AAL 36/37=Hipp, 40/41=Amyg, 70/71=Caud, 72/73=Put, 74/75=Pal, 76/77=Thal)
    _subctx_val = {
        36: _SUBCORTICAL_5HT2A["hippocampus"],
        37: _SUBCORTICAL_5HT2A["hippocampus"],
        40: _SUBCORTICAL_5HT2A["amygdala"],
        41: _SUBCORTICAL_5HT2A["amygdala"],
        70: _SUBCORTICAL_5HT2A["caudate"],
        71: _SUBCORTICAL_5HT2A["caudate"],
        72: _SUBCORTICAL_5HT2A["putamen"],
        73: _SUBCORTICAL_5HT2A["putamen"],
        74: _SUBCORTICAL_5HT2A["pallidum"],
        75: _SUBCORTICAL_5HT2A["pallidum"],
        76: _SUBCORTICAL_5HT2A["thalamus"],
        77: _SUBCORTICAL_5HT2A["thalamus"],
    }

    out = np.empty(116, dtype=float)
    for aal_i, dk_i in enumerate(_AAL116_TO_DK68):
        if dk_i >= 0:
            out[aal_i] = ht2a_dk68[dk_i]
        elif dk_i == -1:
            out[aal_i] = _subctx_val.get(aal_i, _SUBCORTICAL_5HT2A["thalamus"])
        else:  # -2 = cerebellum
            out[aal_i] = _CEREB_5HT2A
    return out


def conversion(e_na: float, e_k: float, e_l: float, g_l: float | None = None, g_na: float | None = None) -> tuple[float, float]:
    """Convert leak reversal potential into equivalent `(g_K, g_Na)` values.

    Parameters are legacy-compatible and unit-agnostic; the returned values follow
    the same unit system as `g_l`/`g_na`.
    """

    if g_l is None and g_na is None:
        raise ValueError("At least one of g_l or g_na must be provided.")
    if g_l is not None:
        g_na_eff = (g_l * (e_l - e_k)) / (e_na - e_k)
        g_k_eff = g_l - g_na_eff
    else:
        g_na_eff = float(g_na)
        g_k_eff = g_na_eff * (e_na - e_l) / (e_l - e_k)
    return float(g_k_eff), float(g_na_eff)
