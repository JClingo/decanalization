# Running and extending the Unreal project

## Install and open

1. Install Unreal Engine 5.5 (retained audits identify 5.5.4). Use Windows for the
   reference training/evaluation workflow. Other engine versions/platforms require
   fresh validation; frozen-data analysis needs neither Windows nor Unreal.
2. Install engine plugins enabled in `ue_canalization.uproject`: LearningAgents,
   GeometryScripting, ModelingToolsEditorMode, plus Marketplace/Fab dependencies
   **DataSerialization**, **FileSDK**, and **ElectronicNodes**. Their product
   identifiers are in the project descriptor. Their implementations are not
   bundled. ElectronicNodes is an editor graph-display dependency; disabling it
   requires checking Blueprint loading. Do not disable data/file plugins without
   replacing the Blueprint nodes that depend on them.
3. PythonScriptPlugin is explicitly enabled for runtime helpers in editor/standalone
   execution. Let Unreal install its Learning Agents Python dependencies on first
   launch. This engine Python environment is separate from the analysis venv.
4. Open `ue_canalization.uproject`. Wait for imports/shaders and Python setup.
   Open `/Game/Level/Landscape`; compile the project Blueprints and resolve missing
   nodes/plugins before running. The repository does not ship engine binaries,
   `Intermediate/` Python installs, or a cooked executable.

## Evaluate the selected policies

Work on a copy or branch. Change defaults and check placed-actor overrides in the
Landscape level; placed values can supersede Blueprint class defaults.

| Asset / setting | Paper evaluation |
| --- | --- |
| `/Game/Learning/BP_AgentManager`: `Run Inference` | true |
| `/Game/Agent/BP_GameMode`: `IsTrialMode` | true |
| GameMode: `IsHRL` | true for HRL; false for NF |
| GameMode: `TrialSequenceOrder` | `cdr` or `ccc` |
| GameMode: `TrialsToRun` | 250 (use a small count for an initial smoke run) |
| GameMode: `TrainingEnvironmentSeed`, `TrialEnvironmentSeed` | 1, 10001 |
| Manager: `EncoderSnapshot`, `PolicySnapshot`, `Decoder Snapshot` | Absolute paths to the matching triplet below |
| TrainingEnv: `UseHomeostaticReward` | true |
| Primary tile `DecanalizationRateFactor` | 0 (verify scene records; traversal factors 4/6/7 are legacy) |

Print the exact absolute paths, with hash verification, from the repository root:

```sh
python analysis/list_paper_snapshots.py HRL
python analysis/list_paper_snapshots.py NF
```

Paste the three paths into the manager fields above. For inspection, use
`analysis/paper-update-2026-10-17-reruns/selection.json` to find the condition's
run ID. Its `inputs/<run-id>.metrics.json` names each loaded file under
`provenance.snapshots.<kind>.loaded-snapshot`. Resolve these paths relative to
`inputs/`, converting backslashes to your platform's separator. Load all three
files from the same run. A scoring flag alone does not choose the policy family.
The committed `Content/Python/trial_training_manifests.json` checks known hashes.
The critic is not needed for frozen inference. See the corrected package's
`CHANGE_INVENTORY.md` and runtime metadata for all physical and PPO settings.

Launch editor standalone execution from PowerShell, with your installed engine:

```powershell
.\start_headless_training.cmd -EngineExe 'C:\Program Files\Epic Games\UE_5.5\Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
```

Despite its historical name, this launcher runs the saved project configuration,
including inference/trials. It uses `-UseFixedTimeStep -FPS=20`, `-game`, `-nullrhi`,
and `-nosound`. First launch the editor once so the launcher can find
`Intermediate/PipInstall/Scripts/python.exe`. Headless launching does not cook a
shipping build. Python helpers rely on editor facilities; packaged-runtime
support is not established.

Outputs go to `Saved/TrialResults/`: YAML, `.metrics.json`, `.completed.json`, and
`.provenance/` scenes and captured weights. Check `[TRIALS_COMPLETE]`, the completion
receipt, and sidecar `status=complete`; do not infer completion from process exit
alone. A fresh batch has its own GUID. Compare its manifest and runtime settings
before treating it as a replication. The published archive is a frozen record;
engine/plugin changes may alter exact outcomes despite equal seeds.

## Train or change the experiment

Set `Run Inference=false` and `IsTrialMode=false`, then choose `IsHRL` before a
fresh run. The manager selects `analysis/homeostatic/homeostatic_ppo.py` for HRL
and `nf_ppo.py` for NF through `Content/Python/nf_training.py`. Training can update
network assets; preserve selected weights and work in a separate checkout.

Change rewards in BP_AgentTrainingEnv, observations/actions in BP_AgentInteractor,
phase protocol in BP_GameMode, and tile/resource behavior in BP_Tile and resource
assets. The files in `analysis/homeostatic/`, `analysis/timing/`, and
`analysis/trial_metrics/` document implementation and native validation. Historical
patch/validation utilities often depend on author-local paths and exported assets
under ignored Saved directories; they are not a fresh-install setup procedure.
Do not run them indiscriminately over project assets.

For new trained weights, add their hashes, training configuration, and effective
overrides to a new training manifest before evaluation. The published selection
manifest identifies the paper's policies only. Keep new data and selection
contracts separate, and adapt validation expectations rather than relabeling new
results as the frozen paper dataset. Retrain after changing schemas or objectives.

## Validation boundary

This release's data checks are automated. A fresh Windows Unreal installation and
all external plugin dependencies have not been executed in this packaging session.
The retained native validation reports document prior execution; they do not
replace checking a new installation. Missing plugins or unavailable external
assets must be resolved before claiming that the simulation reproduced the paper.
