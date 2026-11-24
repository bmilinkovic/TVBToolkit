"""The sensitivity bank must differ only in MCS/UWS b, not protocol details."""

import numpy as np

from experiments.sma8s_b27_full100.common import protocol_for
from experiments.sma8s_b27_full100.cutoff import choose_youden, classify
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
