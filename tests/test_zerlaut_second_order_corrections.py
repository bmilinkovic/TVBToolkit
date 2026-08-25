from __future__ import annotations

import numpy as np
import pytest

from tvbtoolkit.whole_brain.legacy_engine.src import Zerlaut, Zerlaut_gK_gNa


E0 = 0.003
I0 = 0.004


def _toy_model(base_class):
    class ToySecondOrder(base_class):
        """Polynomial transfer functions with an analytic Jacobian/Hessian."""

        def TF_excitatory(self, e, i, *unused):
            x = e - E0
            y = i - I0
            return E0 + 2.0 * x + 3.0 * y + 4.0 * x**2 + 5.0 * x * y + 6.0 * y**2

        def TF_inhibitory(self, e, i, *unused):
            x = e - E0
            y = i - I0
            return I0 + 7.0 * x + 8.0 * y + 9.0 * x**2 + 10.0 * x * y + 11.0 * y**2

        @staticmethod
        def get_fluct_regime_vars(*args):
            template = np.asarray(args[0], dtype=float)
            return np.zeros_like(template), np.ones_like(template), np.ones_like(template)

    return ToySecondOrder()


@pytest.mark.parametrize(
    "base_class",
    [
        Zerlaut.Zerlaut_adaptation_second_order,
        Zerlaut_gK_gNa.Zerlaut_adaptation_second_order,
    ],
)
def test_second_order_jacobian_and_hessian_use_consistent_khz_units(base_class) -> None:
    model = _toy_model(base_class)
    c_ee, c_ei, c_ii = 2.0e-5, 3.0e-6, 4.0e-5
    state = np.array(
        [[E0], [I0], [c_ee], [c_ei], [c_ii], [0.0], [0.0], [0.0]],
        dtype=float,
    )
    derivative = model.dfun(state, np.zeros((1, 1), dtype=float)).reshape(-1)

    time_scale = float(np.asarray(model.T).reshape(-1)[0])
    n_total = float(np.asarray(model.N_tot).reshape(-1)[0])
    inhibitory_fraction = float(np.asarray(model.g).reshape(-1)[0])
    n_e = n_total * (1.0 - inhibitory_fraction)
    n_i = n_total * inhibitory_fraction

    # Analytic Hessians at (E0, I0): H_e=(8,5,12), H_i=(18,10,22).
    expected_rate_e = (0.5 * c_ee * 8.0 + c_ei * 5.0 + 0.5 * c_ii * 12.0) / time_scale
    expected_rate_i = (0.5 * c_ee * 18.0 + c_ei * 10.0 + 0.5 * c_ii * 22.0) / time_scale

    # Analytic Jacobian: dFe=(2,3), dFi=(7,8). At the expansion point F=(E,I),
    # so the drift terms vanish and only shot noise, Jacobian coupling and decay remain.
    expected_c_ee = (
        E0 * (1.0 / time_scale - E0) / n_e
        + 2.0 * c_ee * 2.0
        + 2.0 * c_ei * 3.0
        - 2.0 * c_ee
    ) / time_scale
    expected_c_ei = (
        c_ee * 2.0 + c_ei * 3.0 + c_ei * 7.0 + c_ii * 8.0 - 2.0 * c_ei
    ) / time_scale
    expected_c_ii = (
        I0 * (1.0 / time_scale - I0) / n_i
        + 2.0 * c_ii * 8.0
        + 2.0 * c_ei * 7.0
        - 2.0 * c_ii
    ) / time_scale

    np.testing.assert_allclose(
        derivative[:5],
        [expected_rate_e, expected_rate_i, expected_c_ee, expected_c_ei, expected_c_ii],
        rtol=2.0e-4,
        atol=2.0e-10,
    )
