# Zerlaut second-order implementation audit and correction

**Correction date:** 2026-08-25
**Scope:** native TVBToolkit only; `/Users/borjan/CNRS/projects/TVBSim` was not
modified.

## Finding

The original TVBSim second-order Zerlaut implementation contains two distinct
problems. Before this correction, TVBToolkit fixed one of them only in the
standard leak-conductance model. The split `gK/gNa` model used by the
serotonergic and graph-Laplacian experiments inherited both problems.

## 1. Inconsistent derivative units

The state rates `E`, `I` and transfer-function outputs are represented in kHz.
The covariance source terms therefore generate `C_ee`, `C_ei`, and `C_ii` in
kHz squared. A derivative such as `dF_e/dE` must consequently be computed as

```text
(F_e(E + df) - F_e(E - df)) / (2 df)
```

when `df` is expressed in kHz. TVBSim instead divides by `2 df 1e3`; its second
derivatives divide by `(df 1e3)^2`. The numerator remains in kHz and the
covariance states are not converted to Hz squared. Thus the first derivatives
are 1,000 times smaller and Hessian terms are 1,000,000 times smaller than the
dimensionally consistent kHz calculation.

Current upstream TVB retained this scaling at the time of the audit.
TVBToolkit previously retained it in both the standard and split `gK/gNa`
implementations. This explains why the earlier local parity audit found the
first- and second-order models almost identical: that test reproduced the
implementation, including its derivative scaling, rather than independently
validating units.

## 2. Wrong transfer-function partial derivatives in covariance equations

For the master equation, the required cross terms are

```text
dC_ee/dt: 2 C_ei dF_e/dI
dC_ii/dt: 2 C_ei dF_i/dE
```

Original TVBSim instead uses `dF_i/dI` in the first equation and `dF_e/dE` in
the second. Current upstream TVB corrects these to `dF_e/dI` and `dF_i/dE`.
Before this change, TVBToolkit's standard `Zerlaut.py` already had the correct
population assignments, but `Zerlaut_gK_gNa.py` still used the two original
incorrect terms.

## Corrections made in native TVBToolkit

The following changes are now applied to both native second-order models:

| Quantity | Previous denominator | Correct denominator | Previous scale |
|---|---:|---:|---:|
| First derivative | `2 * df * 1e3` | `2 * df` | 1,000 times too small |
| Pure second derivative | `(df * 1e3) ** 2` | `df ** 2` | 1,000,000 times too small |
| Mixed second derivative | inherited the same `1e3` factors | kHz-consistent finite difference | 1,000,000 times too small |

In the split `gK/gNa` model, the covariance equations were also changed from

```text
dC_ee/dt cross term: 2 C_ei dF_i/dI   -> 2 C_ei dF_e/dI
dC_ii/dt cross term: 2 C_ei dF_e/dE   -> 2 C_ei dF_i/dE
```

No biological parameter, global coupling value, stimulus parameter,
connectome, random seed policy, or PCI calculation was changed as part of this
correction.

The native serotonergic and Momi-Laplacian protocol versions were advanced to
include `zerlaut-so-khz-v2`. Their simulation fingerprints now differ from
pre-correction runs. The four-subject occupancy runner also checks the protocol
version stored in every cached trial and refuses to reuse a pre-correction
file unless the run is explicitly overwritten.

## Impact on current experiments

The Laplacian/PCI and 5-HT2A runners set `zerlaut_order=2` and
`zerlaut_gk_gna=True`. Model selection therefore resolves to

```text
tvbtoolkit.whole_brain.legacy_engine.src.Zerlaut_gK_gNa
    .Zerlaut_adaptation_second_order
```

so runs produced before 2026-08-25 contain both the derivative-unit
inconsistency and the two wrong covariance Jacobian terms. They should not be
mixed with results generated after this correction. With `N_tot=10000`, the
steady-state change may be small in some regimes, but transient responses can
still change materially. Coupling and PCI calibration must therefore be
repeated with the corrected implementation.

## Verification

`tests/test_zerlaut_second_order_corrections.py` substitutes polynomial
transfer functions with known analytic Jacobians and Hessians. For both native
second-order classes, it checks the two firing-rate corrections and all three
covariance derivatives against their analytic values. This test would fail
under either the previous unit scaling or the previous `gK/gNa` population
swap.
