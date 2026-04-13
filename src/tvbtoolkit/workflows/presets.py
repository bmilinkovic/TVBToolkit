"""Condition preset builders for common paradigms."""

from __future__ import annotations

import numpy as np

from tvbtoolkit.workflows.experiments import ConditionSpec
from tvbtoolkit.workflows.pharmacology import leak_to_conductances, receptor_to_gk_profile


def ketamine_depth_conditions() -> list[ConditionSpec]:
    """Legacy-like ketamine depth sweep: control, sub-anaesthesia, anaesthesia."""
    b_es = [5.0, 5.0, 25.0]
    tau_e_es = [5.0, 4.8, 4.5]
    tau_e_is = [5.0, 4.7, 4.5]
    names = ["control", "ketamine_subanesthesia", "ketamine_anesthesia"]
    desc = ["Awake/control", "Sub-anaesthesia ketamine", "Anaesthesia ketamine"]
    out = []
    for n, d, b, tee, tei in zip(names, desc, b_es, tau_e_es, tau_e_is):
        out.append(
            ConditionSpec(
                name=n,
                description=d,
                parameter_overrides={
                    "parameter_model": {
                        "b_e": b,
                        "tau_e_e": tee,
                        "tau_e_i": tei,
                    }
                },
            )
        )
    return out


def psilocybin_receptor_conditions(
    receptor_profiles: dict[str, np.ndarray],
    E_Na_e: float = 50.0,
    E_K_e: float = -90.0,
    E_L_e_start: float = -63.0,
    E_L_e_end: float = -61.0,
    E_Na_i: float = 50.0,
    E_K_i: float = -90.0,
    E_L_i_start: float = -65.0,
    E_L_i_end: float = -63.0,
) -> list[ConditionSpec]:
    """Build receptor-driven psilocybin gradient condition specs.

    Returns control + one condition per receptor profile.
    """
    g_k_e_start, g_na_e_start = leak_to_conductances(E_Na_e, E_K_e, E_L_e_start, g_L=10.0)
    g_k_e_end, g_na_e_end = leak_to_conductances(E_Na_e, E_K_e, E_L_e_end, g_Na=g_na_e_start)

    g_k_i_start, g_na_i_start = leak_to_conductances(E_Na_i, E_K_i, E_L_i_start, g_L=10.0)
    g_k_i_end, g_na_i_end = leak_to_conductances(E_Na_i, E_K_i, E_L_i_end, g_Na=g_na_i_start)

    specs = [
        ConditionSpec(
            name="control",
            description="No psilocybin",
            parameter_overrides={
                "parameter_model": {
                    "gK_gNa": True,
                    "g_K_e": g_k_e_start,
                    "g_Na_e": g_na_e_start,
                    "g_K_i": g_k_i_start,
                    "g_Na_i": g_na_i_start,
                }
            },
        )
    ]

    for name, receptor in receptor_profiles.items():
        g_ke = receptor_to_gk_profile(g_k_e_start, g_k_e_end, receptor)
        g_ki = receptor_to_gk_profile(g_k_i_start, g_k_i_end, receptor)
        specs.append(
            ConditionSpec(
                name=f"psilocybin_{name}",
                description=f"Psilocybin receptor profile: {name}",
                parameter_overrides={
                    "parameter_model": {
                        "gK_gNa": True,
                        "g_K_e": g_ke.tolist(),
                        "g_Na_e": g_na_e_end,
                        "g_K_i": g_ki.tolist(),
                        "g_Na_i": g_na_i_end,
                    }
                },
            )
        )
    return specs


def build_stimulation_override(stimval: float, stimtime: float, stimdur: float = 50.0, stimregion: list[int] | None = None):
    """Build legacy-compatible stimulation override dictionary."""
    if stimregion is None:
        stimregion = [8]
    return {
        "parameter_stimulus": {
            "stimval": float(stimval),
            "stimtime": float(stimtime),
            "stimdur": float(stimdur),
            "stimregion": list(stimregion),
        }
    }


def stimulation_schedule(cut_transient: float, run_sim: float, t_analysis: float = 300.0, n_stims: int = 5):
    """Legacy-like stimulation schedule used in TVBSim ketamine/psilocybin notebooks."""
    return np.linspace(cut_transient + t_analysis, run_sim - t_analysis, n_stims)


def maria_sacha_nature_conditions() -> list[ConditionSpec]:
    """Paper-style whole-brain conditions used in Maria Sacha pipeline notebook.

    Returns conditions aligned with labels in `paper_pipeline_hub`:
    `wake`, `nmda`, `gaba`, `sleep`.
    """
    return [
        ConditionSpec(
            name="wake",
            description="Awake baseline",
            parameter_overrides={
                "parameter_model": {"b_e": 5.0, "tau_e_e": 5.0, "tau_e_i": 5.0, "tau_i": 5.0}
            },
        ),
        ConditionSpec(
            name="nmda",
            description="NMDA-blockade-like state",
            parameter_overrides={"parameter_model": {"b_e": 30.0, "tau_e_e": 3.75, "tau_e_i": 3.75}},
        ),
        ConditionSpec(
            name="gaba",
            description="GABAergic-anaesthesia-like state",
            parameter_overrides={"parameter_model": {"b_e": 30.0, "tau_i": 7.0}},
        ),
        ConditionSpec(
            name="sleep",
            description="NREM-like state",
            parameter_overrides={
                "parameter_model": {"b_e": 120.0, "tau_e_e": 5.0, "tau_e_i": 5.0, "tau_i": 5.0}
            },
        ),
    ]
