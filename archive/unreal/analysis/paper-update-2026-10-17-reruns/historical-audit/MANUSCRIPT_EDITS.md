# Proposed manuscript revisions

These are reviewable replacements for the specified sections of `modelingDecanalization.tex` and `appendix.tex`. They use the manuscript's terminology while narrowing claims to the available evidence. The proposed text deliberately distinguishes completed analyses from additional measurements still needed. The author-confirmed decisions and rerun requirements are recorded in OPEN_ITEMS.md. Historical results below remain provisional until the four primary conditions are rerun with schema-2 logging.

## 1. Abstract: replace the comparison and finding sentences

> We compared policies trained with a continuous homeostatic reward (HRL) and a one-sided negative-feedback reward (NF) in a navigation task with visit-dependent traversal costs. Each selected policy was evaluated over 250 trials under a canalization–decanalization–recanalization sequence and 250 trials of uninterrupted canalization. Following decanalization, the selected HRL policy had a lower positive signed mean homeostatic state and a higher logged evaluation score than its control, whereas the corresponding NF contrasts were uncertain. Supplementary experiments showed that increasing traversal-count accumulation during decanalization reduced subsequent HRL evaluation scores. These findings illustrate how temporary changes in an agent's environment can influence later behavior with fixed policy weights; they do not establish improved learning or generalize across independently trained policies.

Retain the opening motivation. Consider replacing “first such model” with a narrower description unless the literature claim is separately verified. This package does not conduct a new literature review.

## 2. Introduction: refine the hypothesis and NF rationale

Replace the paragraph beginning “In this paper, we test the hypothesis...” with:

> We test whether a temporary perturbation of resource impacts, homeostatic costs, and traversal history changes subsequent regulation relative to uninterrupted canalization. We compare a policy trained to receive an ongoing reward near the homeostatic setpoint with a policy rewarded only for improvements that end below it. Both policies are trained with discounted reinforcement learning and can, in principle, learn anticipatory actions. The comparison therefore concerns the consequences of different objectives and training configurations, rather than a contrast between an anticipatory learner and a purely reactive controller.

Throughout the introduction and discussion, describe accelerated visit counting as **environmental history**, not neural metaplasticity. Keep the theoretical analogy, but state that weights remain fixed during evaluation.

## 3. HRL Methods: replace the implemented reward equation and explanation

Retain the classical drive-reduction equation as background, explicitly distinguishing it from the implemented objective. Replace the exponential implementation equation with:

```latex
\begin{equation}
  r_t^{\mathrm{HRL}} = \Delta t\,\frac{R}{1+(h_t/S)^2},
  \qquad S=25,\quad R=1.
\end{equation}
```

> At each ordinary training decision, reward depends on the resulting signed homeostatic state. It is greatest at the setpoint, decreases symmetrically with absolute distance, and remains positive away from zero. Repeated states receive the same quality reward. With a decision interval of 0.05 simulation seconds, h = 0, ±25, and ±50 yield rewards of 0.05, 0.025, and 0.01, respectively. This objective favors sustained proximity to the setpoint; it is not equivalent to minimizing squared error. Exceeding the training boundary produces a single terminal reward of -1. The current boundary is |h| > 100.

Identify these as training rewards, not the recorded YAML evaluation score. Delete the training-start footnote asserting that negative initial h produces immediate punishment: the new HRL ordinary reward is positive.

## 4. NF Methods: replace the reward and architecture claims

```latex
\begin{equation}
  r_t^{\mathrm{NF}} =
  \begin{cases}
    \dfrac{R}{S}\max\bigl(|h_{t-1}|-|h_t|,0\bigr), & h_t<0,\\
    0, & h_t\geq 0.
  \end{cases}
\end{equation}
```

> Here h at the previous decision is cached once per reward calculation. Reward is given only when absolute distance decreases and the resulting state remains below zero. Unchanged or worsening negative states receive zero, as do transitions that reach zero or overshoot into positive values. NF failure terminates training episodes without a failure penalty. Unlike the HRL reward rate, this improvement reward is not multiplied by the decision interval. The one-sided gate specifies the rewarded transitions; it does not restrict the agent's steering actions or eliminate anticipation through discounted future reward.

Delete the obsolete five-case commented formula and the currently active punishment-based equation once this replacement is accepted. Remove “deliberately simpler control architecture” and “should not provide ... reward discounting.”

## 5. PPO Methods: replace the shared-trainer description

> We used project-local adapters to the Unreal Engine 5.5 Learning Agents PPO backend. The common adapter corrects final-transition credit, bootstraps time-limit truncations, preserves episode boundaries, centers normalized advantages, and supports negative advantages. It also corrects clipping for negative advantages and rejects nonfinite optimizer updates. HRL uses the common configuration. NF additionally uses the NFStable1 adapter, which caps the policy and critic learning rates at 3×10^-5 and 3×10^-4, limits updates per gather to eight, uses a clip parameter of 0.1 and gradient clipping at 0.25, and fixes exploration standard deviation at 0.15. NF actor updates are skipped when the sampled approximate KL exceeds 0.015; critic updates do not change the shared encoder. The first NF rollout is discarded before learning. These training differences should be considered when interpreting comparisons between the selected policies.

> The policy observes 11 values: a pellet-collision indicator, seven resource-ray proportions, anticipated gain and loss, and the current signed homeostatic state. A correction replaced a stale agent-local state observation with the live GameMode value while preserving the observation schema and its scale. The policy maintains 64 recurrent memory values and selects a continuous steering action. Policy weights are fixed during evaluation.

Replace the appendix PPO table with the HRL and effective NF columns in CHANGE_INVENTORY.md. Preserve the distinction between configured iteration limits, snapshot update labels, and actual collected transitions. Replace the historical selection statement with: “Snapshots were selected on the basis of their performance relative to other snapshots. For HRL, selection favored mean homeostatic state closer to zero and the highest mean reward; for NF, selection favored the highest mean reward. The selected snapshot update labels were 10000 and 5000, respectively.” This selection account is author-confirmed; no new candidate ranking is inferred.

## 6. Protocol: replace trial counts, phase resets, and pairing claims

> The primary analysis used HRL-CDR-10000.yml, HRL-CCC-10000.yml, NF-CDR-5000.yml, and NF-CCC-5000.yml. Each file contained 250 complete three-phase trials. The numerical filename suffixes identify the supplied datasets and are not treated as trial counts or elapsed times. The CDR sequence comprised canalization, decanalization, and recanalization; the CCC control remained canalized throughout. The supplementary traversal-count experiments were analyzed separately.

> In the revised reset procedure, the agent's initial homeostatic state is drawn once per trial from [-15,15] and restored at each phase boundary. Policy memory and the relevant environment state persist between phases, while completed trials use a full reset that clears recurrent memory, traversal state, stale perception, and resource timers. Forward velocity is initialized at the beginning of each phase. Consequently, post-intervention performance measures behavior after homeostatic state reinitialization, rather than recovery from the preceding phase's terminal state.

> The environment uses separate training and inference random seeds and a continuing seeded stream across successive trials. Baseline physical summaries agree exactly for corresponding trial IDs in the selected CDR and CCC files. Some baseline evaluation scores differ, however, and the phase summaries do not include full scenario manifests. We therefore do not infer exact cross-condition pairing solely from trial IDs.

Replace the old timing diagram with boundaries at 2000/4000/6000 decisions, corresponding to 100/200/300 s. The author confirms dt = 0.05 s and 100 s per phase. The archived duration field of 1000 is a legacy constant; it should not be interpreted as elapsed time. Rerun logging now checks 2000 reward decisions and records 100 s per phase.

In the intervention paragraph, replace pellet value 20 with the verified selected-run value (current implementation: 35), and replace the old mandatory threefold count increment with ordinary counting for the primary HRL-CDR setting if run provenance confirms the retained 0-plasticity mapping. The interventions still change actual gains/losses as well as observations; do not call them perception-only.

## 7. Outcome definitions and statistical methods

> We retained the signed mean homeostatic state, the logged deviation statistic, the evaluation score, signed total loss, total gain, and the count returned by the unique-pellet function. Signed mean h measures directional bias and is distinct from mean absolute error. The audited deviation routine returns the square root of summed squared deviations from the phase mean, without division by sample count. The logged evaluation score is accumulated by GameMode.EstimateReward and is separate from the training reward; it is not a critic prediction. Scores were kept on their recorded scale and were not compared numerically across reward families.

> We summarized all complete trials without exclusions. Exploratory CDR–CCC comparisons used unequal-variance Welch calculations for phase differences and for differences in within-trial change from baseline. We report unadjusted 95% confidence intervals and Holm-adjusted p-values, with 24 comparisons in each reward-family family. Supplementary traversal comparisons use a separate 36-test family. These calculations characterize variability across evaluations of the selected policies, not variability across independent training runs. Matching trial IDs alone was not used to impose pairing.

Before submission, reconcile the dependency structure with scenario records. These are transparent exploratory calculations, not a retrospective claim of a prespecified independent-trial design.

## 8. Replace Results and the old numerical tables

> Following decanalization, the selected HRL policy had a signed mean homeostatic state of 8.60, compared with 20.23 under continued canalization. Its mean logged evaluation score was 32.68, compared with 27.21. The baseline-adjusted score contrast was +5.47 (95% CI 3.63 to 7.30; Holm-adjusted p < 0.001), and the signed-state contrast was -11.63 (95% CI -16.74 to -6.52; Holm-adjusted p < 0.001). The logged deviation statistic was also lower. These outcomes support reduced positive state bias and a higher evaluation score for this selected HRL policy. The available aggregates do not establish a reduction in mean absolute error.

> The NF post-intervention signed means were 13.85 for CDR and 6.95 for CCC; evaluation scores were 41.86 and 40.35. The baseline-adjusted score contrast was +1.46 (95% CI -2.42 to 5.33), and the signed-state contrast was +6.90 (95% CI -28.13 to 41.93). Neither provided a clear condition difference after adjustment. These intervals do not demonstrate an absence of NF effects or equivalence between conditions.

Insert the two primary tables in RESULTS.md, replacing the old post-intervention tables, and regenerate early-phase appendix tables from phase_summary.csv / primary_contrasts.csv. Remove legacy Cohen's d and p-values unless recalculated under a clearly specified definition and experimental unit.

## 9. Add a short supplementary-results paragraph

> We separately examined faster traversal-history accumulation during the decanalized phase. The supplementary settings counted each tile entry as 4, 6, or 7 visits. Relative to the primary HRL-CDR condition, their mean post-intervention evaluation scores were lower by 10.0%, 14.6%, and 15.2%, respectively. Signed homeostatic means increased from 8.60 to 13.01, 16.39, and 19.67. All three score reductions remained detectable after the supplementary multiplicity adjustment, whereas the evidence for state and deviation changes was less consistent. Faster accumulation of traversal history can therefore have adverse subsequent effects in this model; it should not be assumed to strengthen the benefit of decanalization.

Place the supporting table in the supplement. Do not mix these trials into the four primary conditions or describe the tested settings as neural plasticity rates.

## 10. Discussion, conclusion, and global edits

Replace the strong HRL-only learning/optimality conclusion with:

> The selected HRL policy showed a more favorable post-intervention evaluation score and lower positive state bias under CDR than under uninterrupted canalization. The NF contrasts were uncertain. These findings concern fixed policies operating in environments with altered history and state; they do not demonstrate new learning during evaluation. The supplementary results further show that increasing traversal-history accumulation can reduce subsequent scores. The consequences of decanalization therefore depend on how the perturbation is implemented, as well as on the policy and training procedure.

Add to limitations:

> Only one selected policy per reward family was evaluated, and the effective training configurations differed. The recorded phase summaries cannot reconstruct absolute-error trajectories, time within a homeostatic tolerance, or the new training objectives. The evaluation score uses a separate legacy routine, and its baseline discrepancies require further investigation. The logged duration is not a measured elapsed time, and the deviation statistic depends on sample count. Claims about regulation, optimality, exploration, and adaptation should therefore be limited to the measured summaries until trajectory and provenance records are available.

Global replacements/checks:

- “reward prediction” -> “logged evaluation score,” including axes, captions, and tables.
- “mean distance” -> “signed mean homeostatic state” where based on setpoint-avg.
- “standard deviation” / “variance” -> the verified root-summed-squared-deviation label where based on setpoint-dev.
- “NF does not anticipate” -> remove; discounted PPO does not establish that restriction.
- “only HRL benefits” -> conditional finding for these policies; no direct interaction has been established.
- “optimal policy” / “escaped an attractor” / “greedy or exploratory strategy” -> hypothesis unless supported by trajectories or explicit benchmarks.
- “perception-only intervention” -> composite environment and observation intervention.
- Update abstract, Results summary, Discussion, Conclusion, figure captions, and early-phase appendix together. Remove superseded TODO comments after resolving their substantive questions, not merely because replacement prose exists.

## 11. Replacement Methods for the corrected reruns

The following supersedes the legacy measurement descriptions when the primary analyses have been regenerated from schema-2 runs. The numerical Results and abstract findings above must then be refreshed rather than carried forward without verification.

> Each evaluation trial comprised three phases of 2,000 decision steps at a fixed interval of 0.05 s, giving 100 s per phase. Evaluation reward was the sum of the outputs of the same objective callback used during training: r = 0.05/[1 + (h/25)²] for HRL, and r = max(|h_previous| − |h|, 0)/25 when h < 0, otherwise zero, for NF. Evaluation continued for the full phase even when |h| exceeded the training failure boundary. The terminal penalty and episode cutoff were applied only during training. The NF reward history was initialized at the start of each phase so that resetting the homeostatic state did not generate progress reward.

> Homeostatic dispersion was calculated as the population standard deviation, sqrt(sum((h − mean(h))²)/N), over the recorded homeostatic history. We also recorded mean absolute homeostatic deviation, root mean square homeostatic deviation, and the fraction of recorded samples within ±5 of the setpoint. Because this history includes event and update samples, these measures describe the recorded sample distribution rather than a uniformly sampled time distribution. Reward was accumulated separately once per decision.

> Each run retained the loaded policy, encoder, and decoder snapshots and their hashes, the executed asset and logger hashes, runtime parameters, and a matching effective training configuration where available. Phase records included initial scene state, recurrent policy memory, and random-stream state. Trials constitute repeated evaluations of selected policies; the analysis does not estimate variation across independently trained policies.

For the appendix, use the effective PPO values in `rerun-update/trial_training_manifests.json`; configured pre-override NF values are retained there only as provenance. Keep the supplementary traversal-rate experiments separate from the primary comparison. If using their archived results, explicitly call their reward measure the legacy evaluation score.
