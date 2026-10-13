# Unreal 5.5 timing repair

This repair was made in `D:\Dev\decanalization`, starting at commit `b2ec222`,
using the installed Unreal Engine 5.5.4. The pre-existing modification to
`BP_AgentTrainingEnv.uasset` was backed up and preserved byte-for-byte.

## Simulation contract

- Simulation and policy decisions run at 20 Hz (0.05 simulated seconds per step).
  The connected manager Training Game Settings already specified 20 Hz.
  DefaultEngine.ini now supplies the matching inference timestep and maximum
  physics delta. FixedFrameRate controls simulation delta, rather than relying
  on an FPS cap to infer simulation time.
- BP_Agent assigns CharacterMovement.MaxWalkSpeed from BaseMaxSpeed each tick,
  then updates MaxSpeed. BaseMaxSpeed remains 1000 cm/s. CharacterMovement
  integrates movement. The four delta/mode/landscape multiplier nodes were
  removed from the EventGraph, and their execution subtree is disconnected.
- Look remains input per fixed decision, passed to AddControllerYawInput.
  Its effect includes PlayerController yaw scaling. No extra delta multiplier
  was introduced. The validation records the action and resulting yaw.
- PassiveLossRate is loss per homeostatic update, with one update every five
  decisions. Ordinary BasePassiveLossRate=0.2 means 0.8 units per simulated
  second at 20 Hz. Decanalization can still set the loss to zero.
- The old SecondsElapsed counter was renamed HomeostaticPeriodsElapsed,
  including its local result-function argument. One unit means five decisions
  (0.25 simulated seconds), not one second. Existing trial thresholds and
  scheduling keep their existing period-count behavior. Learning Agents
  episode time remains independent. Existing legacy log labels are not a
  promise of physical seconds; convert period counts using 0.25 seconds/unit.
- Tick prerequisites establish GameMode -> manager -> PlayerController ->
  pawn/CharacterMovement. The existing manager -> agent prerequisite remains.
  This setup assumes the single local player controller used by this project;
  dedicated-server controllers have not been validated.
- Inference skips trainer-process and PPO-communicator initialization.
  Training retains those startup paths.

## Validation and limits

The original NF-Final .bin snapshots are unavailable. Four embedded DA_Agent
network assets were exported to frozen snapshots for the comparison. Their
training timestep and provenance are unknown. Each test launches a fresh
editor process with FixedSeed; Learning Agents initializes recurrent memory to
zero on agent addition. Initial states and actions are compared explicitly.

The pre-fix rendered and headless runs used the same initial state but differed
in simulation timestep and movement. The rendered baseline reached measured
speed about 32,471 cm/s; the headless baseline used a typical speed limit near
6,667 cm/s at roughly 120 Hz. These are observations, not inferred limits.

Final compilation, traces, comparison, original hash manifest, and backups are
in `Saved/TimingFix`. See `comparison.json` for numerical results. Tests use
rendered PIE and -nullrhi PIE, not a cooked shipping executable. The completed
comparison does not establish compatibility with the missing NF-Final
checkpoint. Repeat with the intended checkpoint and a cooked executable before
using the repaired simulation for research results. Changed inference dynamics
may require retraining; do not treat old trial results as comparable without
revalidation.

The final five-Blueprint compile completed with zero errors. The two final runs
completed 200 consecutive decisions (10 simulated seconds). Actions, positions,
velocities, agent/controller yaw, h, speed limits, time, and period counters
matched exactly at every decision. Delta was 0.050000000745 seconds in both
runs. Measured maximum horizontal speed was 1000 cm/s. Both runs had a local
PlayerController and inference enabled, and had no Learning Agents errors or
tick-cycle warnings. A copy of the numerical result is tracked in
`analysis/timing/validation.json`.

Actual training was not run. Its process settings still reference a previous
Windows Intermediate/PipInstall/Scripts/python.exe location. Configure the
training Python environment before starting training. The inference guards
avoid that dependency for the comparison. The training timestep was checked
statically, rather than through an active optimizer run.

## Reproducing the comparison

Use the installed engine at `D:\Dev\UE_5.5`. Enable PythonScriptPlugin on the
command line; it need not be permanently enabled in the project.

1. `inspect_unreal.py` loads defaults and exports the four embedded snapshots.
   Run with `-run=pythonscript -script=<absolute script path> -nullrhi`.
2. `compile_unreal.py` compiles all five Blueprints and saves the three edited
   assets. Check the commandlet exit code and compiler log for errors.
3. Start the editor with
   `-ExecCmds="py D:/Dev/decanalization/analysis/timing/trace_pie.py"`
   `-EnablePlugins=PythonScriptPlugin -FixedSeed -nosound -unattended -nop4`.
   Use `-TimingLabel=fixed_visible` for rendered PIE, and add `-nullrhi`
   with `-TimingLabel=fixed_headless` for headless PIE. The runner stops after
   200 actual simulation ticks, skipping editor frames that do not advance PIE.
4. Run `compare_traces.py` from the project root. It rejects missing steps,
   requires 0.05-second deltas and 1000 cm/s limits, and compares all 200 steps.

The test runner changes snapshot paths in editor memory. Unreal can save dirty
network or level packages during these tests; preserve a backup and restore
those test-only changes afterward. Original assets from this repair are in
`Saved/TimingFix/backup/Content` with SHA-256 hashes in `manifest.json`.

`AssetAudit` is a small console adapter for
[UAssetAPI 1.1.0](https://github.com/atenfyr/UAssetAPI), used to export and stage
uncooked graph edits. `graph_asset.py` reads pin serialization using the local
UE 5.5 engine source. `patch_graphs.py` applies this specific audited graph
layout; it is not a general-purpose migration for other project versions.
All staged edits were compiled and saved by Unreal before validation.

## Episode reset repair

`BP_AgentTrainingEnv.ResetAgentEpisode` now always runs `Initialize`; its
`Episode Steps > 1` branch controls reporting only. Zero-step and one-step
episodes clear custom steps, total reward, termination and last-resource state.
`GetPreviousSetpoint` uses the seeded initial value until a second sample exists,
avoiding an index -1 read on the first reward after reset.

`BP_Agent.ResetEvent` resolves and caches the current GameMode before spawning,
stops jumping, uncrouches, clears accumulated forces and pending launch velocity,
stops movement, consumes movement input twice (pending and last input), and
clears the old resource reference before teleport overlap updates. It sets a
fresh controller rotation and then matches the pawn rotation. Random spawn and
the existing 0–180 degree yaw distribution are retained. The existing GameMode
reset clears homeostatic/traversal histories and episode counters; Learning
Agents clears recurrent memory and framework episode bookkeeping.

Reset-time source backups are under `Saved/ResetFix/backup`, including the user's
existing TrainingEnv edits. `patch_resets.py` targets exports of those current
assets, rather than rebuilding TrainingEnv from an older version. Compile with
`compile_resets.py`; runtime validation uses `validate_resets.py` via ExecCmds.
The final compile passed with zero errors and the existing Projectile collision
channel warning. Four actual manager reset calls (episode steps 0, 1, 20, 0)
passed in PIE: velocity, pending/last input, custom reward/steps/termination,
world tick count and all 64 memory values cleared; actor/controller rotations
matched with zero pitch/roll. The final runtime log had no Blueprint runtime
or Learning Agents errors. Results are in `reset_validation.json`.

This invokes the training environment's real reset listener without launching
an optimizer. Full optimizer training still needs its Python environment path
configured as described above. Test-created network and external actor saves
were preserved under `Saved/ResetFix/test_saved` and restored afterward.

## Independent trial reset

`BP_GameMode.RunTrial` now calls `AgentManager.LearningAgentsManager.ResetAllAgents`
on the completed-trial branch, after incrementing `TrialsRun` and before starting
the next trial. This replaces the direct `Reset(false)` broadcast. The training
environment callback invokes the existing full pawn/world reset with a fresh
spawn/orientation, while Learning Agents resets memory and episode bookkeeping.
The terminal save/quit branch and all phase transitions are unchanged.

`patch_trial_resets.py` targets the current GameMode export backed up under
`Saved/TrialResetFix/backup`; `compile_trial_resets.py` compiles the affected
Blueprint and its dependencies and saves GameMode. Production compilation passed
with zero errors and the existing collision-channel warning.

`validate_trial_resets.py` invoked the real `RunTrial` graph for both phase
orders, seeding counters at 6000, 12000, 18000 and 18002. Phase calls preserved
position, controller orientation, velocity, pending movement input and all 64
seeded memory values. Each completed-trial call incremented the trial count,
cleared world/episode counters, velocity, input and memory, and produced a fresh
spawn and matching pawn/controller orientation. All eight checks passed; the
runtime log contained no Blueprint runtime or Learning Agents errors. Results
are tracked in `trial_reset_validation.json`.

The test temporarily allowed editing `TotalTickCount` on instances (both the
generated property's `CPF_DisableEditOnInstance` flag and the Blueprint variable
description's corresponding 65536 bit were cleared, then compiled). This let the
test reach boundaries directly. The exact compiled production package from
before the test was restored afterward, retaining the normal property flags.
Network and external actor packages were also restored from this turn's backups;
test-created versions remain under `Saved/TrialResetFix/test_saved`. This verifies
boundary behavior in PIE, rather than running every intervening trial timestep.

## Windows trainer runtime repair

The configured Python environment started correctly, but UE's pinned PyTorch
2.4.0+cu121 failed importing `fbgemm.dll` because `libomp140.x86_64.dll` was absent.
Installed the release x64 DLL from Microsoft's official VS 14.40 CRT package
into `Intermediate/PipInstall/Lib/site-packages/torch/lib`. The package and DLL
SHA256 checks passed, and Windows verified the DLL's Microsoft signature.
PyTorch and the UE engine installation were not upgraded.

`fetch_openmp_runtime.py` repeats the repair after recreating `PipInstall`:

```powershell
& .\Intermediate\PipInstall\Scripts\python.exe .\analysis\timing\fetch_openmp_runtime.py
```

CPU and CUDA forward/backward calculations and Adam optimizer steps passed;
the actual Learning Agents `train_ppo` module imported. `training_smoke.py`
then passed 30 actual PIE training decisions with the current manager settings,
active PPO training, sampled actions and pawn movement. The first run directly
hit the user's saved breakpoint inside `BP_AgentInteractor.PerformAgentAction`;
the breakpoint was temporarily disabled for the completed smoke test and then
restored. This checks startup and action execution, not a full PPO training run.

Results, source provenance, logs and current-asset backups are in
`Saved/PythonRuntimeFix`. The four network assets and manager external actor were
restored byte-for-byte from this repair's fresh backups after the test.

## TensorBoard

TensorBoard 2.13.0 is installed in the project Python environment, with Unreal's
protobuf 4.24.0 and typing-extensions 4.8.0 pins retained. Package dependency
checks and a PyTorch SummaryWriter event write/read test passed. The existing
PPO settings already enable `UseTensorBoard`.

Dashboard: `http://127.0.0.1:6006/`. Logs are read from
`Intermediate/LearningAgents/TensorBoard/runs`; each training session gets its
own run. Reward, return, episode length, policy/critic loss and gradient metrics
are emitted after experience collection and PPO updates. Earlier sessions where
TensorBoard was missing have no recoverable event files; start a new training
session after installation. A blank dashboard before the first update is normal.

Restart the dashboard with `analysis/timing/start_tensorboard.ps1`. It starts a
hidden localhost server; process ID and server logs are under `Saved/TensorBoard`.
If the Python environment is recreated, reinstall with:

```powershell
& .\Intermediate\PipInstall\Scripts\python.exe -m pip install -r .\analysis\timing\tensorboard-requirements.txt
```

## Headless training

`start_headless_training.ps1` launches Landscape in standalone game mode using
`UnrealEditor-Cmd.exe -game -nullrhi -nosound -unattended -NoVSync`. It uses the
saved manager training settings and non-editor trainer paths; editor Blueprint
breakpoints do not pause this game process. Current network assets are backed
up before launch. The PID, arguments, logs and backups are stored under the
timestamped `Saved/HeadlessTraining` run directory. Runs remain active after the
launcher exits. TensorBoard reads their metrics from the shared runs directory.

From Explorer, double-click the project-root `start_headless_training.cmd`.
From PowerShell in the project root, run `./start_headless_training.cmd` or
`./analysis/timing/start_headless_training.ps1`. The CMD wrapper selects Windows
PowerShell and applies execution policy only to that invocation. The launcher
checks required paths, prevents a second standalone run for this project, and
waits up to 120 seconds to confirm the first PPO update. Failures include the
run log location; the CMD wrapper keeps the window open when startup fails.

## Current movement comparison (2026-10-13)

The current project was copied into `Saved/MovementAudit/TestProject`, preserving
the open source editor. Only the copy's manager was configured for inference.
Four frozen snapshots from `Training_ppo_sharedmemory_2026-10-13_12-56-03`,
optimizer step 5000, were copied and hashed under `Saved/MovementAudit/checkpoint`.
`setup_movement_audit.py` configures this isolated copy; use `-DisablePipInstall`
when launching it since inference does not require optimizer dependencies.

`trace_movement_audit.py` recorded 200 decisions with `-FixedSeed` for rendered
PIE, headless PIE, standalone headless game, and standalone headless game with
`-UseFixedTimeStep -FPS=20 -NoVSync` (the uncapped fixed-time mechanism used by
training). `compare_movement_audit.py` produced `Saved/MovementAudit/comparison.json`.
All four had identical initial position/state, 0.05-second deltas, BaseMaxSpeed
and MaxWalkSpeed of 1000 cm/s, and exactly identical positions, velocities,
actions, yaw, homeostatic state and period counters at every sampled decision.
Each traversed 9925 cm over 9.95 simulated seconds between samples 1 and 200.
Rendered PIE took 10.125 wall seconds; uncapped headless took 0.172 seconds.
Standalone world time began 0.05 seconds ahead of PIE before the first decision;
elapsed simulation time and decision-aligned state matched exactly.

This verifies current saved assets for one frozen-policy starting trial, not
every phase transition, unsaved editor modification, or optimizer update. The
earlier 5000 cm/s training smoke measurement is not reproduced by this current
inference comparison. No original project assets were modified by the audit.
