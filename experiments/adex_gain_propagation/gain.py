"""Optional, baseline-preserving receiving-node gain for first-order TVB Zerlaut models.

Coupling arrays and saturation are in kHz. Matrix convention: weights[i,j] is j→i.
Import this module from a custom Zerlaut subclass or from a separate experiment.
No production defaults are modified.
"""
import numpy as np

def heterogeneous_gain(weights, stim_node, amount=1.0, hops=2):
    """Exploratory source-conditioned gain map, NOT independent physiological data.

    Two-hop network reachability is a prior, not a fitted biological susceptibility.
    Because the map depends on stimulus location, it is a *stimulation-conditioned*
    intervention, not a fixed property of each cortical region.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be square")
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative")
    if stim_node < 0 or stim_node >= len(weights):
        raise ValueError("invalid stimulation index")
    influence = weights[:, stim_node].copy()
    if hops >= 2:
        influence += (weights @ weights)[:, stim_node]
    influence /= max(float(np.max(influence)), 1e-12)
    return 1.0 + amount * influence

def bounded_input_gain(coupling, reference_coupling, gamma, saturation_khz):
    """Exactly preserve reference coupling; slope at reference is gamma.

    gamma=1 with infinite saturation returns the identity mapping.
    Does not guarantee stability of the high-rate attractor; direct TMS bypasses
    this transformation if injected downstream of the coupling pathway.
    """
    coupling = np.asarray(coupling, dtype=float)
    ref = np.asarray(reference_coupling, dtype=float)
    gain = np.asarray(gamma, dtype=float)
    if not np.all(np.isfinite(gain)) or np.any(gain < 0):
        raise ValueError("gain must be finite and nonnegative")
    if np.isinf(saturation_khz):
        return ref + gain * (coupling - ref)
    if saturation_khz <= 0:
        raise ValueError("saturation_khz must be positive")
    return ref + saturation_khz * np.tanh(gain * (coupling - ref) / saturation_khz)

class ReceivingGainMixin:
    """Mixin: put BEFORE Zerlaut_adaptation_first_order in an experimental subclass.

    For integration without rewriting production dfun, transform coupling argument
    before delegating to production model. The reference must be the actual
    pre-stimulus whole-brain coupling vector measured from a stable baseline,
    not a scalar guessed from regional mean rates.

    This does not distinguish pathways to E and I because the first-order
    production dfun feeds the same c_0 into both populations.
    """
    gain_vector = None
    reference_coupling = None
    saturation_khz = np.inf

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        if self.gain_vector is None:
            return super().dfun(state_variables, coupling, local_coupling)
        if self.reference_coupling is None:
            raise ValueError("Set reference_coupling from a baseline run")
        transformed = np.array(coupling, copy=True)
        transformed[0, :] = bounded_input_gain(
            transformed[0, :], self.reference_coupling,
            self.gain_vector, self.saturation_khz
        )
        return super().dfun(state_variables, transformed, local_coupling)
