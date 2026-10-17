# Recomputed results from the selected files

The four primary conditions each contain 250 trials (IDs 0–249), three complete phases, and logged duration 1000 throughout. Statistics below use the frozen bytes in `inputs/`. No separate dataset normalization, absolute-value transformation of signed means, or post hoc exclusions was applied.

## Primary post-intervention results

| Logged measure | HRL CDR | HRL CCC | NF CDR | NF CCC |
|---|---:|---:|---:|---:|
| Signed phase mean h | 8.597 | 20.227 | 13.846 | 6.946 |
| Reported setpoint deviation | 1216.691 | 1415.510 | 1234.173 | 1224.716 |
| Logged evaluation score | 32.675 | 27.210 | 41.862 | 40.350 |
| Signed total loss | -492.697 | -423.412 | -458.346 | -412.138 |
| Total gain | 709.277 | 650.579 | 681.192 | 638.381 |
| Logged unique-pellet count | 27.940 | 26.216 | 29.460 | 28.176 |

The score is the GameMode evaluation quantity, not the new training reward. Scores are not comparable on a common scale across HRL and NF. Gains-harvested is supplied by GetUniquePelletsCount; it should not be described as total consumption events. Repeated consumption can increase gain without increasing distinct pellet count.

For each trial we calculated phase 3 minus phase 1, then contrasted the mean changes between CDR and CCC using Welch's unequal-variance calculation. These exploratory intervals treat trial outcomes as replicate observations conditional on the selected policies. Dependence under a shared deterministic stream and the absence of independent training replication limit population claims. Trial-ID pairing is not assumed merely from matching IDs.

| Baseline-adjusted contrast, CDR minus CCC | Difference | Unadjusted 95% CI | Holm p |
|---|---:|---|---:|
| HRL signed h | -11.630 | [-16.742, -6.518] | 0.0000390 |
| HRL reported deviation | -198.819 | [-361.910, -35.728] | 0.01699 |
| HRL evaluation score | +5.465 | [3.627, 7.304] | 0.0000000659 |
| NF signed h | +6.900 | [-28.131, 41.931] | 1.0 |
| NF reported deviation | +9.456 | [-759.041, 777.954] | 1.0 |
| NF evaluation score | +1.455 | [-2.424, 5.334] | 1.0 |

Each reward family has its own 24-test Holm family: six metrics × two follow-up phases × two estimands (raw phase difference and baseline-adjusted change). This choice is explicit and differs from older project reports. Intervals are not multiplicity-adjusted. All tests, not only selected rows, appear in `primary_contrasts.csv`.

HRL's lower positive signed mean is consistent with reduced positive bias. It does not prove lower mean absolute error: positive and negative states can cancel. The deviation statistic is, in the audited function, sqrt(sum((h - mean(h))²)). Its magnitude depends on sample count. Without the number and timing of samples for each selected phase, do not relabel it standard deviation, variance, RMSE, or absolute setpoint distance. The current graph provides a mechanistic definition, but a run-specific asset link is still needed to make that definition definitive for the archived records.

NF uncertainty is wide. Failure to reject its contrast does not establish no effect, equivalence, or HRL superiority. A difference in significance between reward families does not establish a reward-family-by-intervention interaction. Because training configurations also differ, even a direct interaction for these selected policies would not isolate the reward function as the cause.

## Baseline checks

All 250 matched IDs have exactly equal baseline values for each physical summary in both reward families. Baseline scores match in 249/250 HRL rows and only 120/250 NF rows. This supports common starting scenarios but exposes an evaluation-score discrepancy. The NF raw post-phase score difference is +1.512; the baseline-adjusted difference is +1.455. The baseline subtraction does not diagnose or repair the scoring discrepancy. Full scene/seed/checkpoint records are needed before asserting exact pairing across conditions.

## Supplement: traversal-count rates during decanalization

Author-confirmed mapping on 17 October 2026: `4-plasticity` means 4 counted visits per entry, `6-plasticity` means 6, and `7-plasticity` means 7. This supersedes the older report's interpretation of the `6` filename as 7 counted visits. The primary HRL-CDR file matches the saved `0-plasticity` run byte for byte; the earlier documented interpretation of that setting is ordinary 1× counting. It remains a CDR intervention, not a CCC control.

| File / count rate | Post score | Change vs primary | Post signed h | Reported deviation |
|---|---:|---:|---:|---:|
| Primary HRL-CDR / ordinary counting | 32.675 | — | 8.597 | 1216.691 |
| HRL-CDR-4-plasticity / 4 | 29.424 | -10.0% | 13.011 | 1291.668 |
| HRL-CDR-6-plasticity / 6 | 27.891 | -14.6% | 16.386 | 1320.272 |
| HRL-CDR-7-plasticity / 7 | 27.761 | -15.2% | 19.673 | 1397.362 |

Relative to the primary HRL-CDR file, all three baseline-adjusted post-phase score reductions survive the supplementary 36-test Holm family (six metrics × two follow-up phases × three comparisons with the reference): adjusted p = 0.0132, 0.0000128, and 0.00000374. The signed-h increases have adjusted p = 1.0, 0.0510, and 0.00158. The reported-deviation increases do not survive that correction (1.0, 1.0, 0.528).

Use these experiments only as supplementary evidence that faster traversal-history accumulation can adversely affect subsequent score and regulation-related state summaries. The score evidence is stronger than the evidence for a general loss of homeostatic regulation. Do not claim every rate worsens absolute error, a graded causal dose response, or impairment during the manipulated phase itself. The comparisons concern later recanalization. No 4-versus-6 or 6-versus-7 superiority tests are reported in this package.

## Figures

`primary_results.png` / `.pdf`: HRL and NF in separate rows; signed h, reported deviation, and evaluation score across all three phases, with means and unadjusted trial-level 95% intervals. Use these as an updated analysis figure, subject to the metric/provenance qualifications above. `phase_summary.csv` also supplies SDs and medians for all six measures and seven inputs. Do not reuse old manuscript figures or tables with these new text claims.
