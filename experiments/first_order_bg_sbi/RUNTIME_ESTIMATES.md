# Runtime estimates for the two-minute pilot

Local timing measured 2026-10-04/05 with the actual 90-region first-order gNa/gK
model, control c0005, b=20 pA, G=0.05, dt=0.1 ms, the existing stochastic drive,
10-ms neural monitor and 2.4-s double-gamma BOLD monitor. One worker; setup 0.008 s.
Consecutive timing blocks of the same simulator:

| Simulated seconds | Measured elapsed seconds | Elapsed/simulated |
|---:|---:|---:|
| 2.4 | 6.541 | 2.725 |
| 12 | 32.475 | 2.706 |
| 60 | 165.948 | 2.766 |

Using the longest block, the **180-second total simulation** (60 warm-up + 120
retained) costs approximately **498 s = 8.3 minutes on one local worker**.
This is an extrapolation; no full 180-second benchmark or cluster-throughput
benchmark has yet run. Excludes feature extraction, compression/I/O, inference
training, queue time and failed/retried jobs. Some brief diagnostic processes ran
during this measurement, so it is not a controlled hardware-performance study.

Estimated wall days = N_draws × 180 × 2.7658 / (86400 × simultaneous_workers).
The estimates below assume each cluster worker achieves that local rate and
ideal parallel scaling. Shared CPU/memory/filesystem limits can slow it down;
different CPUs may be faster or slower. No GPU acceleration is assumed for TVB.

| Campaign | Draws | 48 workers / one node | 192 workers / four nodes |
|---|---:|---:|---:|
| Four subjects ×10k | 40,000 | 4.8 days | 1.2 days |
| 174 usable BOLD subjects ×10k | 1,740,000 | 209 days | 52 days |
| All 179 SC subjects ×10k (upper-bound plan) | 1,790,000 | 215 days | 54 days |

Matched BOLD audit, excluding COMA: control 35/35, EMCS 18/18, MCS 72/75, UWS
49/51 finite and nonconstant. Thus **174 currently pass this basic BOLD check**;
motion/preprocessing quality and other exclusions still require review. The
whole-dataset timing is not a proposal to impute or silently discard the other
five. All clinical stages/sedation categories are included in this broad count,
unlike the non-sedated chronic-patient pilot.

The launcher defaults to `--array=0-3%1`: one 48-worker subject job at a time, so
the **4.8-day** pilot column is the relevant ideal extrapolation. `--array=0-3%4`
allows four such nodes if the allocation permits; it does not make one node
four times faster. Each subject task has its own 14-day job limit.

Run the initial 16-draw-per-subject cluster benchmark and inspect its actual
elapsed times/validity before extending to 10k. Do not change dt, neural equations
or delay implementation just to meet this estimate. Posterior training is a
separate unbenchmarked cost and is not included in these totals.
