# First-order gNa/gK, condition adaptation, 50-trial TMS

New workflow; historical launchers and model equations are unchanged. Read the
pilot report before production. This is not a guarantee of cohort-wide validity.

Locked settings are in `experiments/first_order_gnak/protocol.json`:
control/EMCS/MCS/UWS b = 5/15/30/60 pA, tau_w=500 ms, G=0.05, first-order split
leak at every occupancy (0, .25, .5, .766). COMA cannot enter the job inventory.
Pulse: left SMA, 5 ms, .35 Hz/ms into E's derivative. dt=.1 ms, saved interval=1 ms.
Each production trial is 10 seconds, first 3 seconds excluded, one requested
integer-ms random onset uniformly from 3400 through 8000 ms. Fifty independent
seeds per subject; seeds/onsets matched across occupancy. Save E/I/W and actual
timestamps; PCI epochs are time-locked to the first nonzero waveform sample.
TVB's pulse equation can start one integration step after the requested onset.
In the local dt=.1-ms test it delivered 4.9 ms of nonzero input for the nominal
5-ms pulse; the existing equation is preserved and that duration is recorded.
No sham subtraction. Production saves stimulated recordings only; paired controls
are used in the small diagnostic pilot. The baseline is part of the same trial.

Pull the committed repository on the cluster. The receptor CSV and required
helper scripts are version controlled. Separately copy the entire
shared-control-maximum normalized dataset directory to cluster storage;
participant data and simulation results are not included in Git.
The old native-invnodevol dataset without the shared divisor is rejected.
Do not use the old inference loader's independent subject normalization.

From the repository root on the cluster:

```bash
mkdir -p hpc/logs
export TVB_REPO=/home/bmilinkovic/TVBToolkit
export TVB_DATASET_ROOT=/path/to/converted_structural_invnodevol_control_max
export GNAK_SIM_OUTPUT=/path/to/new/first_order_gnak_tms50
export GNAK_PCI_OUTPUT=/path/to/new/first_order_gnak_pci
# Activate the established TVB environment first for this preflight:
python experiments/first_order_gnak/simulate.py --dataset-root "$TVB_DATASET_ROOT" --output "$GNAK_SIM_OUTPUT" --plan-only
# Review the manifest: expected 35,800 trials, no COMA.
sbatch hpc/submit_first_order_gnak_tms50.sh
# After successful completion, submit separately:
sbatch hpc/submit_first_order_gnak_pci_analysis.sh
```

The launchers source the existing environment through the absolute TVB_REPO path
(default /home/bmilinkovic/TVBToolkit), rather than relying on the shell's current
directory. Submit from that repository root because SLURM log paths remain
relative, matching the older launchers. Replace every /path/to placeholder with
an actual cluster location; no personal-data location is inferred or created.
The dataset must contain index.json and every cohort's subject NPZ files, with
their existing relative directory structure intact. The simulator resolves
repository helpers and the receptor CSV from its own installed file location.

Both jobs retain workq/1 node/1 task/48 CPUs/mem=0/14 days from the existing
serotonergic launcher. The simulator uses at most 48 worker processes; numerical
libraries have one thread each. Estimate peak memory on the actual cluster before
launching full concurrency. Logs directory must exist BEFORE sbatch.
An optional `sbatch --dependency=afterok:JOBID ...` can schedule the analysis after
the successful simulation job. No jobs are submitted automatically by this code.

Rerunning with an unchanged manifest reuses complete trial files. Code/protocol
changes require a new output directory. A RUNNING file prevents concurrent writers;
after a cluster cancellation, verify the job has stopped before removing that
specific stale lock. Failed trial metadata are retained, not silently skipped.

Analysis reads simulation files only and requires all declared trials. It writes
to a separate tree: native trial-bootstrap PCI-LZ (alpha=.10, 2000 bootstraps),
PCI-ST, and the historical TVBSim threshold with PCI calculated per trial then
averaged. The historical minimum-surrogate cutoff is retained and explicitly
labelled, but all windows use correct sample indices. Thus this is not a
reproduction of historical millisecond/indexing mistakes.

Common windows: [-400,-100) ms baseline and [0,300) ms from delivered onset.
No high-pass, no .08 entropy floor. Analysis uses the existing polyphase
anti-alias resampling from 1 to 3 ms; no additional 45-Hz filter. Both all-90 and
source-excluded outputs are saved. Native bootstrap and ST use the trial ensemble;
legacy trial values/masks, native significance masks, evoked maps and row orders
are retained alongside subject/occupancy scalar results.

The small pilot is 4 subjects x 4 occupancies x 2 seeds plus four longer-warmup
checks. It is deliberately not the production PCI run, and cannot establish rare
failure probabilities or universal 3-second equilibration. Do not interpret
matched-noise trajectory differences as PCI input or require exact pathwise
convergence in a stochastic oscillatory regime. Check recovery of rate statistics.

Pilot commands:
```bash
python experiments/first_order_gnak/simulate.py --pilot --output results/first_order_gnak_pilot_v2 --workers 6
python experiments/first_order_gnak/analyze.py --pilot-report --input results/first_order_gnak_pilot_v2 --output results/first_order_gnak_pilot_analysis_v2
python experiments/first_order_gnak/check_equivalence.py --output results/first_order_gnak_equivalence.json
```

The initial v1 pilot failed a waveform-inspection check before integration because
the equation object was inspected instead of its temporal pattern; v2 corrects
that software check. This was not a model instability. v1 is preserved for audit.

No VBI configuration or fitting code is changed by this workflow.
