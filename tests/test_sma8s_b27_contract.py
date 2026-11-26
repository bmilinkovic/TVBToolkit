"""The sensitivity bank must differ only in MCS/UWS b, not protocol details."""

import csv

import numpy as np

from experiments.sma8s_b27_full100.common import protocol_for
from experiments.sma8s_b27_full100.cutoff import (
    choose_youden, classify, uws_occupancy_crossings,
)
from experiments.sma8s_b27_full100.common import OCCUPANCIES
from experiments.sma8s_full100.common import PROTOCOL as FIRST
from experiments.sma8s_full100_second_order.common import PROTOCOL as SECOND


def test_only_two_b_values_change():
    for order, original in ((1, FIRST), (2, SECOND)):
        expected = dict(original)
        expected["version"] = f"sma8s_el62_full100_b27_u42_order{order}_v1"
        expected["b_pa_by_cohort"] = dict(original["b_pa_by_cohort"])
        expected["b_pa_by_cohort"].update(mcs=27.0, uws=42.0)
        assert protocol_for(order) == expected


def test_cutoff_counts_subjects_not_trials():
    mcs = np.array([0.33, 0.34, 0.35])
    uws = np.array([0.20, 0.21, 0.36])
    result = choose_youden(mcs, uws)
    assert result["threshold"] == 0.33
    assert classify(mcs, uws, result["threshold"])["mcs_above"] == 3
    assert result["uws_above"] == 1


def test_uws_crossing_uses_fixed_baseline_cutoff(tmp_path):
    path = tmp_path / "subject_summary.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "subject", "occupancy", "mean_pci_lz"])
        writer.writeheader()
        for subject, scores in {
            "u1": [0.20, 0.28, 0.35, 0.31],
            "u2": [0.34, 0.36, 0.37, 0.38],
        }.items():
            for occupancy, score in zip(OCCUPANCIES, scores):
                writer.writerow(dict(group="UWS", subject=subject,
                                     occupancy=occupancy, mean_pci_lz=score))
    rows, counts = uws_occupancy_crossings(path, 0.30)
    assert counts[str(OCCUPANCIES[0])]["above_cutoff"] == 1
    assert counts[str(OCCUPANCIES[0])]["crossed_from_below"] == 0
    assert counts[str(OCCUPANCIES[2])]["above_cutoff"] == 2
    assert counts[str(OCCUPANCIES[2])]["crossed_from_below"] == 1
    assert np.isclose(next(row for row in rows if row["subject"] == "u1" and
                           row["occupancy"] == OCCUPANCIES[2])["delta_pci"], 0.15)
