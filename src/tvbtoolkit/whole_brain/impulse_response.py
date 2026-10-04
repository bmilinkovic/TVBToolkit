"""Paired state impulses using the native simulator and its full delay history."""
from copy import deepcopy
from dataclasses import replace
import numpy as np

from .simulation import configure_adex_simulator


def fork_checkpoint(sim, cfg, seed):
    """Reconfigure normally, then copy dynamic state (TVB deepcopy drops it).

    Restricted to the standard, non-homeostatic model and white integrator noise
    used by the native workflow. The OU drive itself is in the model state.
    """
    if cfg.zerlaut_matteo or cfg.zerlaut_gk_gna or cfg.online_inhibitory_homeostasis:
        raise ValueError("Checkpoint analysis supports the native standard model only")
    fork = configure_adex_simulator(cfg, seed)
    fork.current_state = sim.current_state.copy()
    fork.current_step = sim.current_step
    # TVB history descriptors also store arrays outside __dict__: copy the
    # complete ring buffer into an independently configured identical history.
    fork.history.buffer[:] = sim.history.buffer
    if cfg.stochastic_integrator:
        if np.any(sim.integrator.noise.ntau != 0):
            raise ValueError("Checkpoint analysis requires white integrator noise")
        fork.integrator.noise.random_stream.set_state(
            sim.integrator.noise.random_stream.get_state())
    return fork


def impulse_response(cfg, *, source=0, epsilon=1e-4, baseline_ms=8000.,
                     response_ms=300., seed=20260928, historical_pulse=False):
    """Return raw rate responses; epsilon is in model rate units (kHz).

    No significance test, activity floor, or change to model/coupling is applied.
    Independent checkpoints preserve the complete state, delayed history, and
    stochastic stream. Only the present excitatory state/history slot is changed.
    """
    if cfg.monitor_mode != "raw" or tuple(cfg.monitor_variables) != (0, 1):
        raise ValueError("Use a raw monitor with E and I: monitor_variables=(0, 1)")
    if not np.isfinite([epsilon, baseline_ms, response_ms]).all() or min(epsilon, baseline_ms, response_ms) <= 0:
        raise ValueError("Durations and epsilon must be positive")
    if cfg.zerlaut_order != 2:
        raise ValueError("This analysis requires the second-order model")
    for duration in (baseline_ms, response_ms):
        if not np.isclose(duration / cfg.dt_ms, round(duration / cfg.dt_ms)):
            raise ValueError("Durations must be integer multiples of the integration step")
    sim = configure_adex_simulator(cfg, seed)
    if sim.stimulus is not None:
        raise ValueError("Disable external stimulation: this analysis uses a state impulse only")
    if not 0 <= source < sim.number_of_nodes:
        raise ValueError("Source outside atlas")
    baseline_t, baseline = sim.run(simulation_length=baseline_ms)[0]
    if not np.isfinite(baseline).all():
        raise FloatingPointError("Nonfinite resting dynamics")

    def branch(delta):
        fork = fork_checkpoint(sim, cfg, seed)
        fork.current_state[0, source, 0] += delta
        fork.history.update(fork.current_step, fork.current_state)
        initial = fork.current_state[0, :, 0].copy()
        times, values = fork.run(simulation_length=response_ms)[0]
        return np.r_[0., times - baseline_t[-1]], np.column_stack(
            [initial, values[:, 0, :, 0].T])

    times, control = branch(0.)
    _, perturbed = branch(epsilon)
    _, half = branch(epsilon / 2)
    _, repeat = branch(0.)
    response = (perturbed - control) / epsilon
    half_response = (half - control) / (epsilon / 2)
    if not np.isfinite(response).all():
        raise FloatingPointError("Nonfinite impulse dynamics")
    peaks = np.max(np.abs(response), axis=1)
    non_source = np.arange(response.shape[0]) != source
    result = dict(time_ms=times, response=response, control_khz=control,
                perturbed_khz=perturbed, half_epsilon_response=half_response,
                baseline_time_ms=baseline_t, baseline_khz=baseline[:, 0, :, 0].T,
                baseline_inhibitory_khz=baseline[:, 1, :, 0].T,
                checkpoint_state=sim.current_state.copy(),
                delay_steps=sim.connectivity.idelays.copy(),
                source_response=response[source],
                max_abs_non_source_timecourse=np.max(np.abs(response[non_source]), axis=0),
                peak_abs_by_region=peaks,
                peak_time_ms=times[np.argmax(np.abs(response), axis=1)],
                responsive_1pct=peaks > .01,
                peak_time_above_1pct_ms=np.where(peaks > .01,
                    times[np.argmax(np.abs(response), axis=1)], np.nan),
                zero_impulse_max_error=np.max(np.abs(repeat - control)),
                halving_relative_l2=np.linalg.norm(response-half_response) /
                    max(np.linalg.norm(half_response), np.finfo(float).tiny),
                epsilon_khz=epsilon, source_index=source, seed=seed)
    if historical_pulse:
        # Exact historical TVB forcing convention: additive derivative drive,
        # 0.0003 kHz/ms = 0.3 Hz/ms, not a 0.3-Hz state displacement.
        overrides = deepcopy(cfg.parameter_overrides)
        overrides["parameter_stimulus"] = dict(stimtime=0., stimdur=10.,
            stimperiod=response_ms*10, stimval=.0003, stimregion=[source],
            stimvariables=[0], stimshape="square")
        pulse_cfg = replace(cfg, parameter_overrides=overrides)
        pulse = fork_checkpoint(sim, pulse_cfg, seed)
        pulse.history.update(pulse.current_step, pulse.current_state)
        initial = pulse.current_state[0, :, 0].copy()
        _, values = pulse.run(simulation_length=response_ms)[0]
        rates = np.column_stack([initial, values[:, 0, :, 0].T])
        # PulseTrain excludes both endpoints at this dt, as in historical TVB.
        # Save the actually applied waveform instead of claiming an exact area.
        waveform = np.asarray(pulse.stimulus.temporal.evaluate(
            np.arange(0., response_ms, cfg.dt_ms))).reshape(-1) * .3
        result.update(pulse_perturbed_khz=rates,
            pulse_delta_hz=(rates-control)*1000,
            pulse_drive_time_ms=np.arange(0., response_ms, cfg.dt_ms),
            pulse_drive_hz_per_ms=waveform,
            pulse_total_injected_hz=float(waveform.sum()*cfg.dt_ms))
    return result
