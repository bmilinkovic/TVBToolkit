# PCI v6 HPC protocol

This protocol replaces all earlier serotonergic PCI simulation outputs.
Old caches are not reused: v6 has a new output root and an incompatible
simulation fingerprint.

## Fixed model and stimulation protocol

- Native inverse-node-volume structural weights, with no further subject or
  simulator normalization.
- Damage mask is the union of the matched raw and inverse-node-volume zero
  masks; matching tract lengths are zeroed.
- Subject-matched tract lengths and delays at 4 m/s.
- Left supplementary motor area (`Supp_Motor_Area_L`, zero-based AAL index 9).
- Cimbi 5-HT2A map joined by anatomical label, never by array position.
- Diagnosis-level excitatory adaptation: CNT 10, EMCS 30, MCS 55, and
  UWS/COMA 75 pA.
- Four receptor occupancies: 0, 0.25, 0.5, and 0.766.
- One hundred matched stochastic trial seeds per subject and occupancy.
- Each trial has a separately scheduled stimulation onset. Epochs are cut
  around the recorded onset and only then aligned and averaged.
- Temporal-average sampling every 3 ms (333.33 Hz).
- PCI-LZ and PCI-ST are calculated from the same 100-trial average.
- Both response analyses start at 8 ms, excluding the imposed pulse.

## Stage 1: calibrate coupling and pulse propagation

After pulling the branch on the cluster:

```bash
sbatch hpc/submit_pci_stimulus_g_calibration.sh
```

This is a seven-subject calibration, not a cohort analysis: one control plus
sedated/non-sedated pairs from EMCS, MCS, and UWS. Subjects were selected
without examining PCI outcomes, with approximately matched disconnection
within each sedation pair and increasing disconnection across diagnoses. All
seven use a shared `b_e=10 pA`, so anatomical damage is isolated before the
diagnosis-specific adaptation hypothesis is introduced.

The workflow uses ten matched trials and proceeds in stages: pulse waveform,
duration and amplitude at a reference coupling; global-coupling selection with
the winning safe pulse; then homeostasis off, pre-fitted/frozen and online at
the selected coupling. It reports both PCI-LZ and PCI-ST and saves the aligned
90-region response time courses and subject-level evoked-response figures.

## Stage 2: full production run

Set the selected global coupling explicitly and submit:

```bash
export SEROTONERGIC_COUPLING_STRENGTH=<selected_G>
sbatch hpc/submit_serotonergic_pci_full.sh
```

The job refuses to launch if the coupling value is omitted. The default v6
output root is:

```text
notebooks/outputs/serotonergic_pci_v6_dual_100trials_invnodevol_native
```

Expected simulation count: 189 subjects x 4 occupancies x 100 trials = 75,600
trial files.

The versioned analysis directory contains:

- `serotonergic_pci_subject_metrics.csv`, with PCI-LZ and PCI-ST;
- `serotonergic_pci_subject_metrics_with_rescue.csv`, with change from
  occupancy-zero baseline;
- `serotonergic_pci_lz_st_summary.{png,pdf,svg}`;
- immutable simulation and analysis manifests.

## Why old outputs cannot contaminate v6

Trial validation checks the protocol/simulation fingerprint, subject,
diagnosis, occupancy, trial seed, exact onset, stimulation label, atlas hash,
receptor-file hash, receptor-map hash, and analysis-window duration. A legacy
trial copied into the v6 location fails validation rather than being silently
accepted.
