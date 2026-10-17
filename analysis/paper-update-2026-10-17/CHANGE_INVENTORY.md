# Consolidated project changes

Reporting basis: inspection began at commit `4cf19d5` with the user's data changes present. During preparation the repository advanced to `2351160` ("renamed"); final provenance captures that revision. All seven analyzed source files were rechecked against their frozen hashes and remained identical. The primary HRL-CDR file has changed since the existing comparison report. Its bytes now match `Saved/TrialResults/HRL-CDR-0-plasticity.yml`; the file named `HRL-CDR-4-plasticity.yml` matches the original standard-rate run. Use the frozen files and hashes in this package, not an older report bearing the same filename.

Evidence categories below distinguish current saved assets, implemented Python behavior, historical validation, and data-supported facts. These are not interchangeable with run-specific provenance.

## Reward and state observation

| Paper / older implementation | Current implementation | Evidence and editorial consequence |
|---|---|---|
| HRL exponential improvement reward and asymmetric worsening punishment | For ordinary training decisions, r = dt * R / (1 + (h/S)^2), with S=25 and R=1; terminal failure reward -1, once | `analysis/homeostatic/patch_reward.py`, README, current TrainingEnv defaults. Replace the executed HRL equation; retain classical drive reduction only as theoretical background. Positive ongoing quality is a surrogate for time near zero, not exact squared-error minimization. |
| NF deficit reductions rewarded and worsening deficits punished | If current h < 0: max(|previous decision h| - |current h|, 0) * R/S; otherwise zero. No dt factor. NF failure reward zero | `patch_nf_reward.py`, `nf_validation.json`. Reaching zero and crossing into positive h earn zero, even if absolute error decreases. Do not call this a punishment-based thermostat or say it prevents anticipation: it is still trained by discounted PPO. |
| Reward depends on state/anticipated-net change; historical samples used for previous reward | New training reward caches state per reward decision; cache invalidated on full reset; phase transitions retain it | `patch_nf_reward.py`. Avoid equating five-tick homeostatic sampling with reward-call frequency. |
| Policy receives current h | Old graph read Agent.SetpointDistance, observed stuck at -25; repaired graph reads live GameMode.Setpoints | `patch_observation.py`, homeostatic README. Same 11-dimensional schema and scale 100; scientifically material correction to information available to the policy. |
| Evaluation reward described as the training rule or reward prediction | GameMode.EstimateReward remains a separate legacy evaluator; its output feeds TotalRewards, which StoreTrial writes as reward-total | Fresh `BP_GameMode.nodes.txt`: nodes 212, 506, 619, 739, 697, 247. Negative NF YAML scores also contradict the new nonnegative NF objective. Report a logged evaluation score, separately from training reward. Reconstructing the new objective requires trajectories. |

HRL terminal threshold is **100**, not the paper's 150, in current TrainingEnv defaults. Max Time remains 100 seconds. Training termination is distinct from evaluation scoring, which can contain large positive h values without the training episode termination behavior.

## PPO and neural network

| Setting | Paper | Current HRL/common configured value | Effective NFStable1 value |
|---|---:|---:|---:|
| Policy / critic learning rates | 1e-4 / 1e-3 | 1e-4 / 1e-3 | capped at 3e-5 / 3e-4 |
| Policy / critic batch sizes | 2056 / 4096 | 2056 / 4096 | same |
| Recurrent training window | 16 | 16 | same |
| Updates per gather | 32 | 32 | capped at 8 |
| Critic warmup | 8 | 8 | 0 |
| Clip parameter | 0.2 | 0.2 | capped at 0.1 |
| Discount / GAE lambda | 0.99 / 0.95 | 0.99 / 0.95 | same |
| Normalized advantage range | 0 to 10 | -10 to 10 | same |
| Entropy weight | 0 | 0.001 | 0 |
| Action regularization | 0.001 | 0.001 | 0 |
| Gradient clipping | disabled | enabled, 0.5 | enabled, cap 0.25 |
| Start / end trim | 1 / 0 | 0 / 0 | same |
| Exploration SD | learned | learned | fixed at 0.15 |
| Actor KL stopping | not reported | no NF-specific guard | skip update if sampled approximate KL > 0.015 |
| Critic changes encoder | yes | yes | no; features detached |
| First rollout | ordinary learning | ordinary learning | discarded, bounded decoder then published |

Unchanged recorded common settings: weight decay 1e-4, return regularization 1e-4, learning-rate multiplier 1, action surrogate weight 1, PPO seed 1234, replay capacity 1,000 episodes / 10,000 steps, configured iteration limit 1,000,000, GPU, TensorBoard and snapshot output enabled. The configured limit is not achieved training duration. The historical 1028-step episode cap should be checked against the selected-run manager settings before carrying it forward.

`homeostatic_ppo.py` credits the final transition, bootstraps truncations, respects episode boundaries, centers normalized advantages, repairs negative-advantage clipping, and rejects nonfinite losses/gradients/weights. NF uses probability-ratio PPO clipping and its own safeguards. The pre-trainer NF JSON still lists the common values: `nf_ppo.stable_config` changes them in Python. Reporting only that JSON would miss the effective NF settings.

The schema remains 11 observations, one continuous steering action, and 64 recurrent memory values in the selected configs. The prior appendix's detailed topology and parameter counts are historical recovered-network evidence; do not claim they have been freshly recounted for selected 2026 snapshots. Fixed NF decoder coordinates also distinguish stored parameters from parameters actually free to train.

Snapshot evidence: the saved manager points to `Winning HRL` encoder/policy/decoder 10000. `Winning NF` includes the copied 5000 triplet and NFStable1 files from 2026-10-16 12:08:22. Matching retained snapshot hashes are recorded in this package. Matching files establishes retained identity; the YAML itself lacks a loaded-checkpoint hash. Earlier provisional recommendations for HRL 15000 / NF 20000 are not the selected dataset identities and should not replace them.

## Environment and timing

| Setting | Existing appendix | Fresh saved-asset evidence |
|---|---|---|
| Grid-width parameter | 25 | 50; spacing remains 1000 Unreal units |
| Spawn probability | 0.05 | 0.05; one seeded Bernoulli decision per cell, rather than an exact fixed pellet count |
| Ordinary pellet impacts | [1,20] | random draw [5,35]; Resource base default 25 is replaced by the startup draw |
| Decanalized pellet impact | 20 | MaxHomeostaticImpact 35 |
| Pellet respawn interval | 0.01 s | 0.01 s |
| Passive-loss parameter | 0.05 | 0.5; README's intermediate 0.1/0.2 values are older |
| Tile base impact | 10 | 10; inverse-visit rule and stored delta factor retained |
| Traversal increment | 1 normally, 3 decanalized | Current tile class has DecanalizationRateFactor=7. The actual decanalized count update adds this factor; it is not an extra increment on top of the ordinary branch. Supplement rates 4/6/7 are author confirmed. Current default is not the primary run's rate. |
| Recanalization history | floor(n/2) | retained |
| Speed | 1000, rescaled by frame duration | BaseMaxSpeed=5000 cm/s, assigned as native CharacterMovement speed, without old multiplier chain |
| Environment seed | 1 | training seed 1; inference/trial seed 10001 |
| Trial count | 50 | 250 in each selected file and current defaults |
| Phase thresholds | 6000/12000/18000 ticks | current saved graph 2000/4000/6000 ticks; older validation uses 6000/12000/18000. Neither proves selected-run timing. |
| Logged duration | fixed 3000 | fixed 1000 in current StoreTrial preparation and every selected phase; not an elapsed-time measurement |

The corrected simulation runs at fixed dt=0.05 s (20 Hz), with homeostatic updates every five decisions (0.25 s). Thus the current passive-loss parameter corresponds to 2 units/s under that update cadence. Tick ordering is GameMode -> manager -> controller -> pawn/movement. Headless launch flags `-UseFixedTimeStep -FPS=20 -NoVSync` remove wall-clock pacing while retaining simulation delta. NullRHI alone did not establish this. Timing validation is evidence for the tested builds, not automatic proof of every selected run's duration.

## Resets, phase continuity, and reproducibility

- Full training/trial resets now clear recurrent memory via Learning Agents, custom reward/termination bookkeeping, perception caches, tile count and observed impact, resource availability and pending respawn timers, and stale movement input/forces. The first scored trial uses the same reset path as later trials.
- The later phase-initialization repair supersedes stationary reset starts: forward velocity is initialized from BaseMaxSpeed at startup, full resets, and intermediate phase boundaries. Subsequent physics/collisions can change velocity.
- A trial samples initial h once in [-15,15] and restores that same value at later phase boundaries. Training starts at -15. Phase boundaries preserve policy memory, pose, pellet state/timers and environmental history except for the specified intervention. Full trial resets clear memory. The paper's statement that memory persists across trials is obsolete.
- Phase resets mean the post-intervention phase is not recovery of the preceding physiological h trajectory. It measures performance after h is reinitialized, with other within-trial state retained.
- Training/inference seed selection occurs before BeginPlay. The stream progresses across resets; it is not rewound every trial. Identical seed values or trial IDs alone do not prove identical scenarios after trajectories diverge.
- YAML now checkpoints complete headers/phase records, uses unique run filenames, and records completion receipts. Notifications, Python runtime repairs, and accelerated execution support reliability; they are not scientific intervention factors.
- Added telemetry includes MAE, RMS error, signed mean h, fraction within ±5, transitions, throughput, and NF KL/action diagnostics. Those training telemetry values are not contained in the primary phase-summary YAML files.

Key implementation commits: `e693150` preserved runtime baseline; `7ca8ed4` continuous objective/live state/PPO; `22bc117` observation and reward work; `0aa34f9` NF reward; `6504425` separate seeds; `89abea4` trials and parameters; `4cf19d5` traversal comparisons. The detailed source files and captured asset graphs take precedence over these shorthand commit descriptions.
