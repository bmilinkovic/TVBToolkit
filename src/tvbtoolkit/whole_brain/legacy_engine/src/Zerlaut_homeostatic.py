"""Online inhibitory homeostasis for second-order Zerlaut AdEx models."""

# ruff: noqa: N999, RUF012

from __future__ import annotations

import numpy
from tvb.basic.neotraits.api import Final, List, NArray, Range

from . import Zerlaut, Zerlaut_gK_gNa


class _OnlineInhibitoryHomeostasis:
    """Mixin adding slow rate detectors and an I-to-E efficacy state."""

    homeostasis_on = True

    def _homeostatic_dfun(self, state_variables, coupling, local_coupling):
        # Base second-order AdEx states occupy indices 0..7.  H_i_e is a
        # dimensionless multiplier on Q_i_e; R_e and R_i are slow rate traces.
        base_state = state_variables[:8]
        h_raw = state_variables[8, :]
        rate_e_filtered = state_variables[9, :]
        rate_i_filtered = state_variables[10, :]

        h_min = numpy.asarray(self.homeostasis_min_scale)
        h_max = numpy.asarray(self.homeostasis_max_scale)
        h = numpy.clip(h_raw, h_min, h_max)
        self._Q_i_e_effective = numpy.asarray(self.Q_i_e) * h
        base_derivative = super().dfun(base_state, coupling, local_coupling)

        derivative = numpy.empty_like(state_variables)
        derivative[:8] = base_derivative
        detector_tau = numpy.asarray(self.homeostasis_detector_tau_ms)
        derivative[9] = (base_state[0, :] - rate_e_filtered) / detector_tau
        derivative[10] = (base_state[1, :] - rate_i_filtered) / detector_tau

        target = numpy.maximum(numpy.asarray(self.homeostasis_target_rate), 1e-9)
        error = (rate_e_filtered - target) / target
        inhibitory_gate = numpy.maximum(rate_i_filtered, 0.0) / target
        runaway_gate = (
            rate_e_filtered >= numpy.asarray(self.homeostasis_activation_rate)
        ).astype(float)
        up_span = numpy.clip((h_max - h) / numpy.maximum(h_max - 1.0, 1e-9), 0.0, 1.0)
        down_span = numpy.clip((h - h_min) / numpy.maximum(1.0 - h_min, 1e-9), 0.0, 1.0)
        soft_bound = numpy.where(
            error >= 0.0,
            up_span ** numpy.asarray(self.homeostasis_beta),
            down_span ** numpy.asarray(self.homeostasis_beta),
        )
        derivative[8] = (
            runaway_gate
            * inhibitory_gate
            * error
            * soft_bound
            / numpy.asarray(self.homeostasis_tau_ms)
        )
        # Numerical safeguard for a rare integrator overshoot beyond a bound.
        derivative[8] += numpy.where(h_raw < h_min, h_min - h_raw, 0.0)
        derivative[8] += numpy.where(h_raw > h_max, h_max - h_raw, 0.0)
        return derivative


class Zerlaut_adaptation_second_order_homeostatic(
    _OnlineInhibitoryHomeostasis, Zerlaut.Zerlaut_adaptation_second_order
):
    """Standard split-population AdEx with online inhibitory homeostasis."""

    homeostasis_target_rate = NArray(
        label="Homeostatic target excitatory rate [kHz]",
        default=numpy.array([0.005]),
        domain=Range(lo=0.0001, hi=0.200, step=0.0001),
    )
    homeostasis_detector_tau_ms = NArray(
        label="Rate-detector time constant [ms]",
        default=numpy.array([50.0]),
        domain=Range(lo=1.0, hi=5000.0, step=1.0),
    )
    homeostasis_tau_ms = NArray(
        label="Inhibitory-homeostasis time constant [ms]",
        default=numpy.array([2000.0]),
        domain=Range(lo=10.0, hi=100000.0, step=10.0),
    )
    homeostasis_activation_rate = NArray(
        label="Runaway-rate activation threshold [kHz]",
        default=numpy.array([0.020]),
        domain=Range(lo=0.001, hi=0.500, step=0.001),
    )
    homeostasis_beta = NArray(
        label="Soft-bound exponent",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.1),
    )
    homeostasis_min_scale = NArray(
        label="Minimum I-to-E efficacy scale",
        default=numpy.array([0.25]),
        domain=Range(lo=0.01, hi=1.0, step=0.01),
    )
    homeostasis_max_scale = NArray(
        label="Maximum I-to-E efficacy scale",
        default=numpy.array([4.0]),
        domain=Range(lo=1.0, hi=20.0, step=0.1),
    )

    state_variable_range = Final(
        label="State variable initial ranges",
        default={
            **Zerlaut.Zerlaut_adaptation_second_order.state_variable_range.default,
            "H_i_e": numpy.array([1.0, 1.0]),
            "R_e": numpy.array([0.004, 0.004]),
            "R_i": numpy.array([0.010, 0.010]),
        },
    )
    variables_of_interest = List(
        of=str,
        choices=(
            "E",
            "I",
            "C_ee",
            "C_ei",
            "C_ii",
            "W_e",
            "W_i",
            "noise",
            "H_i_e",
            "R_e",
            "R_i",
        ),
        default=("E",),
    )
    state_variables = [
        "E",
        "I",
        "C_ee",
        "C_ei",
        "C_ii",
        "W_e",
        "W_i",
        "noise",
        "H_i_e",
        "R_e",
        "R_i",
    ]
    _nvar = 11

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        return self._homeostatic_dfun(state_variables, coupling, local_coupling)


class Zerlaut_adaptation_second_order_gK_gNa_homeostatic(
    _OnlineInhibitoryHomeostasis, Zerlaut_gK_gNa.Zerlaut_adaptation_second_order
):
    """Split-leak serotonergic AdEx with online inhibitory homeostasis."""

    homeostasis_target_rate = NArray(
        label="Homeostatic target excitatory rate [kHz]",
        default=numpy.array([0.005]),
        domain=Range(lo=0.0001, hi=0.200, step=0.0001),
    )
    homeostasis_detector_tau_ms = NArray(
        label="Rate-detector time constant [ms]",
        default=numpy.array([50.0]),
        domain=Range(lo=1.0, hi=5000.0, step=1.0),
    )
    homeostasis_tau_ms = NArray(
        label="Inhibitory-homeostasis time constant [ms]",
        default=numpy.array([2000.0]),
        domain=Range(lo=10.0, hi=100000.0, step=10.0),
    )
    homeostasis_activation_rate = NArray(
        label="Runaway-rate activation threshold [kHz]",
        default=numpy.array([0.020]),
        domain=Range(lo=0.001, hi=0.500, step=0.001),
    )
    homeostasis_beta = NArray(
        label="Soft-bound exponent",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.1),
    )
    homeostasis_min_scale = NArray(
        label="Minimum I-to-E efficacy scale",
        default=numpy.array([0.25]),
        domain=Range(lo=0.01, hi=1.0, step=0.01),
    )
    homeostasis_max_scale = NArray(
        label="Maximum I-to-E efficacy scale",
        default=numpy.array([4.0]),
        domain=Range(lo=1.0, hi=20.0, step=0.1),
    )

    state_variable_range = Final(
        label="State variable initial ranges",
        default={
            **Zerlaut_gK_gNa.Zerlaut_adaptation_second_order.state_variable_range.default,
            "H_i_e": numpy.array([1.0, 1.0]),
            "R_e": numpy.array([0.004, 0.004]),
            "R_i": numpy.array([0.010, 0.010]),
        },
    )
    variables_of_interest = List(
        of=str,
        choices=(
            "E",
            "I",
            "C_ee",
            "C_ei",
            "C_ii",
            "W_e",
            "W_i",
            "noise",
            "H_i_e",
            "R_e",
            "R_i",
        ),
        default=("E",),
    )
    state_variables = [
        "E",
        "I",
        "C_ee",
        "C_ei",
        "C_ii",
        "W_e",
        "W_i",
        "noise",
        "H_i_e",
        "R_e",
        "R_i",
    ]
    _nvar = 11

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        return self._homeostatic_dfun(state_variables, coupling, local_coupling)
