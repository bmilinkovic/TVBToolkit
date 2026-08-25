# Momi graph-Laplacian AdEx/PCI project

This isolated project applies the Frobenius-normalized graph Laplacian used by
Momi, Wang, and Griffiths to every subject's native inverse-node-volume
structural connectome, then runs the established AAL90 AdEx stimulation and
dual PCI analysis route.

The operator is

```text
D_ii = sum_j A_ij
L = (D - A) / ||D - A||_F
```

It has a positive weighted-degree diagonal, negative off-diagonal anatomical
edges, zero row sums, one zero eigenvalue per connected component, and no
negative eigenvalues. Momi's released equations apply the negative of this
matrix: `-L*x = A*x_delayed - D*x_current`. TVB therefore receives `-L`, while
all saved diagnostics and plots report the conventional `D-A` Laplacian. The
code disables legacy `nullify_diagonals` behavior so the `-D` self-term reaches
TVB.

Tract lengths are not used to reweight `A` or `L_momi`. As in Momi et al., they
remain a separate, subject-specific delay matrix:

```text
delay_ms[i,j] = tract_length_mm[i,j] / conduction_speed_m_per_s
```

Run the default control plus low- and high-edge-loss UWS comparison:

```bash
python experiments/momi_laplacian_pci/run_two_subject_robustness.py --dry-run
python experiments/momi_laplacian_pci/run_two_subject_robustness.py --workers 8
```

For a short end-to-end execution check:

```bash
python experiments/momi_laplacian_pci/run_two_subject_robustness.py \
  --smoke-test --workers 4 \
  --output-root results/momi_laplacian_pci_two_subject_smoke
```

The production robustness run uses ten matched stochastic trials for
`control:c0015`, `uws:u0001`, and `uws:u0020`, left supplementary motor area stimulation,
3-ms sampling, a 300-ms pre/post window, and both PCI-LZ and PCI-ST. Outputs
include per-trial caches, aligned regional time courses, structural/operator
QC plots, evoked time-course plots, PCI plots, a metrics CSV, and a complete
JSON manifest.

PCI-LZ uses the validated simulation route by default: 500 baseline
trial-bootstrap resamples, a global maximum statistic, and `alpha=0.01`.
Pre/post whole-block exchange is retained only as an explicitly requested
sensitivity analysis (`--pci-significance-method pre_post_swap`).
Source entropy and active fraction are reported as diagnostics; the former
0.08 rule no longer forces a calculated simulation PCI to zero. Every result
also includes the original TVBSim estimator (shared threshold, one PCI per
trial, then averaging) as a clearly labelled legacy comparison.

Use `--connectivity-mode native --coupling-strength 0.0025` to repeat the
protocol on the untransformed inverse-node-volume adjacency. Tract lengths and
all delay calculations remain subject-specific and unchanged in both modes.

The default Laplacian coupling `G=3.5` is a predeclared starting value, not a
fitted biological result. It approximately preserves the off-diagonal scale
of the existing native-weight reference `G=0.0025` for the prespecified
control (`0.0025 * ||D-A||_F = 3.52`). This comparison concerns the
unnormalized Laplacian scale and is not an exact native-adjacency operator-norm
match. Every subject within an operator receives the same `G`; no
subject-specific recoupling is performed. A coupling sensitivity run should
precede cohort-scale inference.
