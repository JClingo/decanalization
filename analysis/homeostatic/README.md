# Homeostatic regulation

The baseline project state is preserved in commit `e693150`. These changes
implement a shared, continuous homeostatic objective. Phase transitions and
full resets between trials remain unchanged. Legacy HRL/NF reward branches
remain available by disabling `UseHomeostaticReward` in BP_AgentTrainingEnv.

## Reward

Every decision receives:

```text
reward = dt * HomeostaticRewardRate / (1 + (h / RewardDistanceScale)^2)
```

The default scale is 25 homeostatic units; rate is 1 per simulated second.
This gives a bounded positive quality score, highest at zero, decreasing
smoothly with absolute error and symmetric across zero. Repeated states still
receive the same quality reward. A pellet is useful according to the resulting
error, including its overshoot, rather than its collision alone. Positive
ongoing quality avoids terminating to escape a stream of negative error costs.
This is a surrogate for time spent near zero, rather than exact squared-error
minimization. Reward scale is independent of the existing failure boundary.

`abs(h) > Max Setpoint Distance` sets termination and yields
`HomeostaticFailureReward` (default -1) once. Further calls return zero until
the manager resets the episode. Episode reward and step counts use the existing
reset bookkeeping. Scale is floored at 0.001, reward rate at zero, and failure
reward capped at zero to keep accidentally edited settings well-defined.

At dt=0.05, rewards at h=0, ±25 and ±50 are 0.05, 0.025 and 0.01.
The near-zero region is tunable through RewardDistanceScale, without changing
physical resource/tile impact, passive loss, or the failure threshold.

## PPO and telemetry

The observation graph now reads the latest GameMode.Setpoints value through the
agent's cached GameMode reference. Previously it read Agent.SetpointDistance,
which stayed at its default -25 during the production smoke rollout. This fix
preserves the 11-dimensional schema and its existing FloatScale=100, while
providing the policy with the actual signed homeostatic value.

The manager uses the project-local `homeostatic_ppo` module in this directory.
It delegates training to UE's train_ppo and replaces the advantage calculation
to include the last transition (the UE 5.5 implementation omits it), bootstrap
time-limit truncations, respect episode boundaries, and center normalized
advantages. MinimumAdvantage=-10 preserves negative feedback. Gradient norm
clipping is enabled at 0.5, entropy weight is 0.001, and start-step trimming is
zero now that reset state is valid. Learning rates, update count, memory window,
physical parameters and simulation timestep retain their baseline values.

TensorBoard adds mean absolute error, RMS error, signed mean and the fraction
of samples within ±5, plus collected transitions and throughput. The adapter
locates SetpointDistance by schema index. It converts its normalized float
observation using BP_AgentInteractor's existing FloatScale=100. If that scale
changes, update the adapter; these metrics describe observed homeostatic state.
Task metrics use collected post-trim transitions on the x axis; UE's existing
loss/reward metrics retain their optimizer-update x axis. Historical rewards
from the legacy objective are not numerically comparable to the new reward.

## Validation and operation

`validate_reward.py` invokes the actual Blueprint callback in isolated PIE,
covering unchanged anticipated net, unchanged distance, sign symmetry,
improvement, worsening overshoot, both legacy mode flags, terminal reward and
reset. `test_ppo.py` checks final-transition credit, truncation bootstrapping,
episode separation and negative normalized advantages. Logs and detailed
runtime results are in Saved/HomeostaticFix.

Use the existing project-root `start_headless_training.cmd` for a new run.
Use a fresh training run for this changed objective; old frozen policies were
optimized for different rewards. Judge performance by MAE and fraction within
±5 on complete held-out trials, including every intended phase/order. A passing
startup test establishes correct execution, not that the task has been learned.

Validation passed 14 real reward/observation cases across both mode flags,
repeated-state rewards, terminal handling and resets, plus six PPO regression
tests. The corrected production trainer completed a PPO update with 20,000
transitions collected, changing homeostatic telemetry, and a clean editor exit.
Network/external-actor packages and debugger settings were restored afterward;
only the three intended Blueprint packages retain implementation changes.
`validation.json` records the results. The smoke test was stopped; it did not
leave a long training run running or replace the saved baseline policies.

Graph authoring utilities `add_reward_variables.py`, `patch_reward.py`,
`compile_reward.py` and `configure_trainer.py` document how assets were made.
They are development tools, not required to launch training. The patch script
expects the exported variable-augmented asset at Saved/HomeostaticFix/env.variables.json.
