# README

The homeostatic objective, live-state observation fix, project-local PPO adapter,
and validation results are documented in [analysis/homeostatic/README.md](analysis/homeostatic/README.md).
Start a fresh run with `start_headless_training.cmd`; evaluate mean absolute
homeostatic error and the fraction of time within ±5 in TensorBoard.

## Learning Agents

https://dev.epicgames.com/community/learning/courses/GAR/unreal-engine-learning-agents-5-5/bZnJ/unreal-engine-learning-agents-5-5

## Tensorflow

./python (to run local installs)

python -m pip install tensorboard

cd "C:\Program Files\Epic Games\UE_5.5\Engine\Binaries\ThirdParty\Python3\Win64\Scripts"
./tensorboard --logdir="C:\Users\jingo\Documents\Unreal Projects\decanalization\Intermediate\LearningAgents\Tensorboard\runs"

Runs on:
<http://localhost:6006/>

## Headless Mode (for training and trials)

executable (after quick cook):
C:\Users\jingo\Documents\Unreal Projects\decanalization\Saved\StagedBuilds\Windows\ue_canalization\Binaries\Win64

once there, in terminal:

./ue_canalization.exe Landscape -nullrhi -nosound -log -log=canalization.log


## New build things
Everything in analysis

Run metadata (inc runId)
D:\Dev\decanalization\Saved\HeadlessTraining

To kill a run
taskkill /PID <runId> /T /F
