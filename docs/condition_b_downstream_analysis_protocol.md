# Condition-B Downstream Analysis Protocol

This document defines how the condition-specific `b` downstream analyses should be performed for the DOC simulation project.

It covers:

- brain states
- LZc
- PCI

The goal is to keep the analysis logic explicit and stable before running final plotting, statistics, and interpretation.

## Core Idea For The Condition-B Branch

For these analyses, the unit of analysis is a **single noise scenario**.

Within one noise scenario, the dataset contains all subjects across the clinical conditions. The clinical conditions already differ in their simulated adaptation parameter `b`.

The condition-specific `b` mapping is:

| Condition | b value |
| --- | ---: |
| Control | 10 |
| EMCS | 30 |
| MCS | 55 |
| UWS | 75 |
| COMA | excluded from brain-states analysis |

The downstream analyses should therefore not treat `b` as a separate sweep dimension. Instead, `b` is embedded in the condition-specific simulations.

## Plain-Language Summary

For each noise scenario, discover the shared functional brain states across all non-COMA subjects. Then ask how often each clinical group occupies those shared states, and how strongly those states couple to each subject's own structural connectome.

For LZc and PCI, no shared brain-state model is needed. Complexity is computed directly per subject within each noise scenario.

## Required Input Structure

The condition-b simulation root is expected to look like:

```text
condition_b/sims/condb_doc_gradient/{noise_scenario}/{cohort}/{subject_id}/seed_*.npz
```

Example:

```text
condition_b/sims/condb_doc_gradient/private_alpha0/control/c0001/seed_000.npz
condition_b/sims/condb_doc_gradient/private_alpha0/emcs/e0001/seed_000.npz
condition_b/sims/condb_doc_gradient/private_alpha0/mcs/m0001/seed_000.npz
condition_b/sims/condb_doc_gradient/private_alpha0/uws/u0001/seed_000.npz
```

Each `.npz` file should contain the simulated activity for one subject under one noise scenario.

Expected signal arrays include:

| Key | Meaning |
| --- | --- |
| `rate` | firing-rate time series |
| `time_rate_ms` | firing-rate time vector |
| `bold` | BOLD time series |
| `time_bold_ms` | BOLD time vector |

## What Counts As A Noise Scenario?

A noise scenario is one background-input configuration.

Examples:

| File label | Plain meaning |
| --- | --- |
| `private_alpha0` | each region receives independent private noise |
| `global_alpha_025` | moderate whole-brain shared noise |
| `sc_alpha_045` | strong structurally informed shared noise |

Each noise scenario must be analysed independently.

Do not pool across different noise scenarios when fitting brain states.

The same principle applies to LZc and PCI: do not mix different noise scenarios when computing or interpreting subject-level values.

# Brain-States Analysis

## Brain-States Steps Per Noise Scenario

For each noise scenario:

1. Load all subject simulations from that noise scenario.
2. Exclude COMA subjects.
3. Keep Control, EMCS, MCS, and UWS subjects together.
4. Extract brain-state features/patterns from each included subject.
5. Concatenate the extracted patterns across all included subjects within that noise scenario.
6. Fit one shared k-means model to this pooled scenario-level matrix.
7. Assign each subject's data back onto the shared centroids.
8. For each subject and each state, compute state occupancy.
9. For each subject and each state, compute SC-FC coupling using that subject's own structural connectome.
10. Save the result for that noise scenario before moving to the next scenario.

## What Must Not Be Done

Do not include COMA subjects in the brain-states analysis.

Do not pool data across different noise scenarios.

Do not fit separate k-means models per condition.

Do not fit separate k-means models per subject.

Do not treat `b` as an independent sweep dimension in this condition-b analysis.

Do not use a group-average SC for the final subject-specific SC-FC coupling values.

## What Must Be Done

Fit one shared k-means model per noise scenario using all non-COMA subjects together.

Use the same shared centroids for Control, EMCS, MCS, and UWS within that noise scenario.

Compute occupancy per subject from that subject's assignment to the shared states.

Compute SC-FC coupling per subject using that subject's own SC matrix.

Save/checkpoint outputs separately for each noise scenario.

## Recommended Initial Run Order

Start with the three noise scenarios that match the PCI analysis:

```text
private_alpha0
global_alpha_025
sc_alpha_045
```

These correspond to:

| Scenario | Plain meaning |
| --- | --- |
| `private_alpha0` | independent regional noise baseline |
| `global_alpha_025` | moderate whole-brain shared noise |
| `sc_alpha_045` | strong structurally informed shared noise |

After these complete and are verified, decide whether to run the remaining noise scenarios.

## Expected Output Per Noise Scenario

Each noise scenario should produce a table with one row per:

```text
subject × state × domain
```

Minimum columns should include:

| Column | Meaning |
| --- | --- |
| `scenario` | noise scenario label |
| `domain` | `rate` or `bold` |
| `cohort` | source cohort folder |
| `condition` | clinical condition |
| `subject_id` | subject identifier |
| `state_rank` or `state_id` | shared brain-state index |
| `occupancy_pct` | percentage of time assigned to that state |
| `sfc_sub` | coupling between that state's FC pattern and the subject's own SC |
| `centroid_order_mode` | centroid ordering convention, if used |

## Interpretation

Within one noise scenario, the centroids represent shared functional states discovered across the whole non-COMA cohort.

Differences between clinical groups then reflect how often each group occupies those shared states, and how those states relate to each subject's own structural connectome.

This is the intended personalised brain-states analysis.

# LZc Analysis

## Core Idea

LZc is computed directly per subject. It does not require shared centroids, k-means, or pooling subjects before the metric is calculated.

For the condition-b branch, LZc should be computed independently within each noise scenario.

## LZc Steps Per Noise Scenario

For each noise scenario:

1. Load all subject simulations from that one noise scenario.
2. Exclude COMA subjects.
3. Keep Control, EMCS, MCS, and UWS.
4. Remember that each condition already has its own simulated `b` value.
5. Compute LZc separately for each subject.
6. Compute LZc separately for each signal domain, where available:
   - firing rate
   - BOLD
7. Save one row per subject, per domain, per noise scenario.
8. Run group statistics across clinical conditions within that scenario.

## What LZc Answers

Within one background-noise setting, LZc asks whether spontaneous signal complexity differs across clinical conditions, where the clinical groups already differ by their condition-specific `b` values.

## What Must Not Be Done For LZc

Do not pool across noise scenarios before computing LZc.

Do not pool subjects before computing LZc.

Do not treat `b` as a separate sweep variable in the condition-b analysis.

Do not include COMA in the main condition-b LZc statistics.

## Expected LZc Output

Minimum output should contain one row per:

```text
subject × domain × scenario
```

Minimum columns should include:

| Column | Meaning |
| --- | --- |
| `scenario` | noise scenario label |
| `domain` | `rate` or `bold` |
| `cohort` | source cohort folder |
| `condition` | clinical condition |
| `subject_id` | subject identifier |
| `lzc` | subject-level LZc value |

# PCI Analysis

## Core Idea

PCI is computed directly per subject from that subject's repeated stimulation trials. It does not require k-means or shared centroids.

For the condition-b branch, PCI should be computed independently within each PCI noise scenario.

## PCI Input Structure

The condition-b PCI simulation root is expected to look like:

```text
condition_b/sims_pci/condb_doc_gradient/{noise_scenario}/{cohort}/{subject_id}/trial_*.npz
```

Each subject should have 100 stimulation trials for each included PCI noise scenario.

The final calibrated stimulation protocol is:

| Parameter | Value |
| --- | ---: |
| stimulation amplitude | 0.00030 kHz |
| stimulation duration | 10 ms |
| stimulated region | 18 |
| trials per subject/scenario | 100 |

## PCI Steps Per Noise Scenario

For each PCI noise scenario:

1. Load all subject PCI trial folders from that one noise scenario.
2. Exclude COMA subjects.
3. Keep Control, EMCS, MCS, and UWS.
4. For each subject, load that subject's 100 stimulation trials.
5. Compute multi-trial PCI for that subject using those 100 trials.
6. Save one row per subject and per noise scenario.
7. Run group statistics across clinical conditions within that scenario.

## What PCI Answers

Within one stimulation/noise setting, PCI asks whether perturbational complexity differs across clinical conditions, where the clinical groups already differ by their condition-specific `b` values.

## What Must Not Be Done For PCI

Do not pool across noise scenarios before computing PCI.

Do not pool subjects before computing PCI.

Do not compute PCI from fewer than 100 trials for the final condition-b analysis unless explicitly marked as a diagnostic or pilot.

Do not treat `b` as a separate sweep variable in the condition-b analysis.

Do not include COMA in the main condition-b PCI statistics.

## PCI Scenarios For The First Final Pass

The complete condition-b PCI branch contains these three key scenarios:

| Scenario | Plain meaning |
| --- | --- |
| `private_alpha0` | independent regional noise baseline |
| `global_alpha_025` | moderate whole-brain shared noise |
| `sc_alpha_045` | strong structurally informed shared noise |

These should be analysed first because they match the primary comparison set across brain states, LZc, and PCI.

## Expected PCI Output

Minimum output should contain one row per:

```text
subject × scenario
```

Minimum columns should include:

| Column | Meaning |
| --- | --- |
| `scenario` | PCI noise scenario label |
| `cohort` | source cohort folder |
| `condition` | clinical condition |
| `subject_id` | subject identifier |
| `n_trials_used` | number of stimulation trials used |
| `pci_mean` | subject-level multi-trial PCI |
| `pci_trials_mean` | average single-trial PCI-like value, if saved |
| `pci_trials_std` | trial-level variability, if saved |

# Recommended Run Strategy

Run and checkpoint analyses scenario by scenario.

First-pass scenarios:

```text
private_alpha0
global_alpha_025
sc_alpha_045
```

Recommended order:

1. Brain states for `private_alpha0`.
2. Verify brain-states outputs and plots.
3. Brain states for `global_alpha_025`.
4. Brain states for `sc_alpha_045`.
5. LZc for the same three scenarios.
6. PCI for the same three scenarios.
7. Only then decide whether to expand LZc/brain-states analyses to the remaining global and SC-shaped alpha values.

Do not launch all downstream analyses monolithically unless the scripts checkpoint after every scenario and print reliable progress.
