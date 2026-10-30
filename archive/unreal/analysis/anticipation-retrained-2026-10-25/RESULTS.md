# Corrected anticipation: selected policies and final evaluation

Selection followed the behavioral objectives: HRL minimizes mean absolute homeostatic state; NF maximizes actual received reward. Eight-trial screening used seed 20001. Independent 32-trial candidate confirmation used seed 30001. The final test used seed 10001, excluded from selection.

| Family | Selected candidate | Confirmation distance | Confirmation reward per trial |
|---|---|---:|---:|
| HRL | fresh | 11.3354 | 249.0332 |
| NF | preserved-winner | 27.5783 | 65.0552 |

The selected HRL policy is the fresh checkpoint at 750,000 collected transitions. The preserved NF winner outperformed the new candidates and was retained with its original weights. Full candidate scores and bootstrap comparisons are in candidate_confirmation.json and confirmation_contrasts.json.

| Family | Final condition | Trials | Mean absolute state | Mean actual reward per trial |
|---|---|---:|---:|---:|
| HRL | CDR | 250 | 115.235952 | 176.371442 |
| HRL | CCC | 250 | 14.151521 | 243.985260 |
| NF | CDR | 250 | 189.449393 | 44.045412 |
| NF | CCC | 250 | 45.376193 | 61.522151 |

All 1,000 final trials completed all three phases: 3,000 phases and 6,000,000 inference decisions. Native receipts, loaded-network hashes, family, condition, timestep and held-out test seed were checked. final_verification.json records those checks. contrasts.json gives trial-level bootstrap intervals for mean(P3-P1, CDR) minus mean(P3-P1, CCC).

The fresh NF run ended with a native communication timeout after collecting 5.38 million transitions, before its original 640-gather minimum. Its strongest saved checkpoint remained valid, and four subsequent completed screening checks had not improved it. Both refinements completed their goal-based stopping checks. Raw failure receipts and all checkpoints were preserved; the timeout is not reported as successful training completion. See training_closeout.json and PROTOCOL.md.

The existing metric uses the event/update-sampled GameMode.Setpoints history, not a uniformly timed trajectory. Reward is the actual TrainingEnv.GatherAgentReward output. Reward scales differ across HRL and NF and should be compared within each family. There is one initialization seed per underlying policy; evaluation-trial bootstrap intervals do not estimate variation across independent training runs.

The main project's inference defaults and historical results were preserved. Selected snapshots are under policies/HRL and policies/NF. To apply a policy, with play stopped, run in Unreal's Python console:

```python
import sys
sys.path.insert(0, '/Users/jingo/Work/PhD/Dissertation Project/decanalization/analysis/anticipation-retrained-2026-10-25')
import load_selected_policy
load_selected_policy.apply('HRL')  # or 'NF'
```

Open the Landscape map with play stopped before calling the helper. It saves the matching policy settings and Landscape actor overrides. The helper checks the corrected anticipation asset and all snapshot hashes, sets the matching reward family and inference mode, and updates snapshot paths and provenance. It keeps the current trial order and physical settings. Importing the helper changes nothing. The corrected BP_Agent asset and portable raw results are included in the result bundle under Saved/AnticipationRetrain20261025/reproducible-results.zip.
