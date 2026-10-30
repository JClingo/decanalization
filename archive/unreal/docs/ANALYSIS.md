# Paper analysis and data guide

## Authoritative dataset

`analysis/paper-update-2026-10-17-reruns/selection.json` selects four complete
schema-2 batches. No trials are discarded within them. `inputs/` includes YAML,
metrics JSON, completion receipts, loaded encoder/policy/decoder weights, and
3,000 gzip-compressed scene JSON records. HRL weights have update label 10000;
NF weights have label 5000. These labels are not training-transition counts.
`evidence/trial_training_manifests.json` identifies their hashes and effective
training settings. `validation.json` records condition checks and alignment.

The archive was verified byte-for-byte against its 3,069-entry manifest before
this reproduction wrapper was added. Its original code, inputs, and outputs
remain unchanged. The wrapper normalizes Windows path separators only in a
copied script and removes the machine-local Saved/AnalysisPython import override.
`prepare_inputs.py` and `package_results.py` are historical collection/packaging
tools; neither is needed by readers. The output copy's inherited manifest is
removed on success because it no longer describes regenerated files.

## Commands and outputs

Run from the repository root:

```sh
python analysis/reproduce.py --verify-only
python analysis/reproduce.py --paper-figures --output analysis/generated/my-run
```

| Paper material | Generated output relative to output directory |
| --- | --- |
| All trial/phase measurements | `trial_metrics.csv` |
| Phase means, SDs, medians, 95% intervals | `phase_summary.csv` |
| Primary Welch differences, intervals, p and Holm p | `primary_contrasts.csv` |
| Paired and baseline-change sensitivity analyses | `paired_sensitivity.csv` |
| Primary result narrative and contrast table | `RESULTS.md` |
| Reward equations, protocol, effective PPO settings | `CHANGE_INVENTORY.md` |
| Manuscript primary figure | `figures/rerun-primary-results.png` (R flag) |
| Manuscript appendix regulation figure | `figures/rerun-regulation-results.png` (R flag) |
| Python versions of outcome plots | `primary_results.*`, `regulation_results.*` |
| Legacy traversal factors 4/6/7 | `legacy-supplement/summary.csv`, `contrasts.csv`, `RESULTS.md`, `supplement_results.*` |
| Runtime versions and comparison success | `reproduction.json` |

The R script is the figure code used by the paper workspace, adapted to accept a
package directory. Python plots show the same summaries with different layout.
Reports named `MANUSCRIPT_EDITS.md` describe historical proposed edits; they are
not the current manuscript. Earlier screenshots are illustrative, not outputs
from this statistical pipeline. No raw per-frame action/trajectory reconstruction
is implied by the phase-level records.

## Data dictionary and statistics

`trial_metrics.csv` has one row per phase: `run` identifies the batch; `family`
is HRL or NF; `order` is cdr or ccc; `trial` is a zero-based trial ID; `phase` is
1, 2, or 3; `sample_count` counts recorded homeostatic history samples.

| Field | Meaning |
| --- | --- |
| `homeostatic-mae` | Mean absolute deviation of h from zero |
| `setpoint-dev` | Within-phase population SD of h (division by N) |
| `reward-total` | Sum of actual objective rewards over 2,000 decisions |
| `homeostatic-fraction-within-5` | Fraction of recorded samples within ±5 of zero |
| `homeostatic-rms` | Root mean square of h |
| `setpoint-avg` | Signed mean h |
| `loss-total`, `gain-total` | Logged total homeostatic loss and gain |
| `gains-harvested` | Pellet harvest count |

History includes seeded and event/update samples, rather than uniformly timed
samples. Each phase lasts 2,000 decisions at dt=0.05 s (100 simulation seconds).
Population SD within phases differs from the across-trial sample SD in summary
CSV files. Reward definitions differ by family; compare raw totals only within
families. Homeostatic state is restored at phase boundaries; this is not a
continuous physiological recovery trace.

Primary contrasts are CDR minus CCC, two-sided Welch tests and unadjusted 95%
intervals. Holm correction spans nine outcomes across two policies separately
for phases 2 and 3. Sensitivity methods are paired-all, paired-exact-scene, and
paired-change-from-baseline; Holm families are separate by method and phase.
Exact initial-scene matches number 230 HRL pairs and 226 NF pairs; all 250 pairs
match baseline outcomes. Neither scene records nor hashes establish complete
engine-state replay. The study evaluates selected policies, without independent
training-seed replication. Legacy traversal data use older reward/dispersion
definitions and must remain separate.

## Historical implementation audit

`analysis/implementation-audit-2026-09-14/` preserves the paper workspace's earlier
settings extraction, archive, scripts, and provenance. Its README explicitly
marks the 2025-era settings as superseded for the corrected October study.
Native exports in its ZIP help inspect Blueprint implementation history. The
current study's execution records and effective settings are in the corrected
package and `analysis/homeostatic/`; do not substitute the historical audit's
network counts, thresholds, or environment defaults for those records.
