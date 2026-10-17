# Resolve before submission

The consolidation and proposed edits are complete. The following are evidence gaps that prevent treating every current implementation detail as a verified property of the selected archived runs.

1. **Selected-run timing.** The paper says 6000/12000/18000 ticks. Retained phase-initialization validation uses those thresholds. Fresh saved assets instead use 2000/4000/6000. Every selected phase logs duration 1000, but that value is assigned as a constant. Obtain the actual graph/configuration or trace for each selected run before reporting phase seconds, number of homeostatic samples, or per-second rates. At dt=0.05, 2000 decision ticks would mean 100 s and 6000 would mean 300 s; neither inference is yet assigned to these archived files.

2. **Evaluation score.** Trace and validate GameMode.EstimateReward on the selected-run version, separately from TrainingEnv.GatherAgentReward. Baseline physical aggregates match exactly, but 130 NF baseline scores differ and one HRL score differs. Establish why before asserting exact condition matching or interpreting the score as a clean objective measure. Preserve the current recorded analysis; do not silently replace scores with newly inferred training rewards.

3. **Regulation metrics.** The current GetStandardDev computes sqrt(sum((h-mean(h))²)). Confirm the selected-run library hash and sample counts. Retain the descriptive raw metric meanwhile. To establish improved regulation, log timestep h and calculate mean(abs(h)), sqrt(mean(h²)), fraction within ±5, and comparable exposure durations. |mean(h)| is not an acceptable substitute.

4. **Run manifests and checkpoint identity.** Retained Winning HRL/NF snapshot matches can be recorded, but trial YAML has no asset hashes, selected snapshot hashes, RNG state, or full environment parameter record. Link the saved logs to the executed settings, including ordinary-count HRL primary intervention and NF intervention settings. Author has confirmed supplementary rates 4, 6, and 7; this replaces the older filename interpretation. The primary HRL file's byte identity with the retained 0-plasticity run is verified.

5. **Training selection account.** Record which 2026 candidates were tried and why HRL 10000 / NF 5000 were chosen. The manuscript's historical manual-search statement and Saved/CheckpointSelection's different provisional recommendations do not establish the new selection procedure. Match the effective NFStable1 configuration rather than copying the pre-override JSON.

6. **Experimental unit and scope.** Trial-level intervals are conditional summaries for selected policies. Use scene/RNG manifests to decide whether a paired or unpaired analysis is defensible. Independent training seeds are needed for algorithm-level claims. A reward-by-intervention interaction must be tested directly on a common metric and still cannot isolate reward when training recipes differ.

None of these requires altering the user's designated primary dataset scope. The supplementary traversal data remain a supporting sensitivity demonstration rather than the main experiment. No additional simulation, training, or publication was performed for this package.
