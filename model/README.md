# Python cognitive-map model

A gridworld forager that learns where food is and how confident to be about it.
Canalization is learned belief precision. Decanalization temporarily relaxes
that precision and raises the randomness of the agent's choices. Physiology is
never changed. See `../docs/FINDINGS.md` for the results, and the paper for
the full method.

## Files

| File | Role |
|---|---|
| `world.py` | World: grid, wear, resource patches, degradation, new (rich/reliable) patches, phases. Also the imposed-narrowing prototype controller (`Trial`). |
| `mapagent.py` | Cognitive map: precision-weighted beliefs, regrowth expectation, hand-written planner, decanalization levers. |
| `policy_agent.py` | Learned option policy over the map (seek / explore / cruise / avoid); HRL and NF drive-reduction rewards. |
| `settings.py` | Shared world settings for training and evaluation. |
| `ppo_train.py` | PPO training (needs PyTorch). |
| `select_ckpt.py` | Checkpoint selection on held-out trials by each family's own objective. |
| `eval_policies.py` | Main evaluation: undisturbed, low-precision reference, decanalization cells. |
| `pool_seeds.py` | Pools evaluations across training seeds (seed = unit of inference). |
| `levers_learned.py` | Lever decomposition (entropy-only arms). |
| `eval_phases.py` | Per-phase CCC vs CDR evaluation behind Table 2 and Figure 5. |
| `eval_first_harvest.py` | When CCC and CDR agents first harvest from the new patches (same trials as `eval_phases.py`). |
| `make_figure_data.py` | Regenerates the data for Figures 2, 4, 5 and 6, prints Table 2, and builds the figure PDFs. |
| `smoke_test.py` | Quick check that the saved policies run. |
| `example-trial-scan.py` | Picks the paper's example trial (evaluation seed 5020). |
| `long-run-no-new-patches.py` | Exploratory: changing world without new patches, run to 1,200 s. |
| `path_wear_ablation.py` | Exploratory: evaluation with flat traversal costs, to test whether path wear contributes to trapping. |
| `gates.py`, `sweep.py`, `cost_sweep.py`, `confirm_dose.py` | Development stage A: imposed narrowing. |
| `map_gates.py`, `trap_diag.py`, `map_levers.py`, `map_levers2.py`, `confirm_map.py`, `surface.py` | Development: cognitive map with the hand-written planner. |

## Running

Scripts read and write `runs/` relative to this folder, which is git-ignored.
Published outputs are in `../results/`.

Install NumPy (needed for everything):

```bash
python -m pip install "numpy>=1.24"
```

Training also needs PyTorch:

```bash
python -m pip install "torch>=2.1"
```

Check that the saved policies run:

```bash
python smoke_test.py
```

**Evaluate the published policies.** Copy them into `runs/`, then run the
evaluation. The empty tag is the matched-world control; `_stable` is training
seed 0.

```bash
mkdir -p runs && cp ../results/policies/*.npz runs/
```

```bash
python eval_policies.py _stable
```

**Retrain from scratch.** Run these for each family (HRL or NF) and each
training seed: `_stable` with seed 0, and `_stable_s1` to `_stable_s3` with
seeds 1–3. This example is HRL, seed 1.

```bash
python ppo_train.py HRL 60 _stable_s1 '{"degrade":0.0,"rich_patches":0}' 1
```

```bash
python select_ckpt.py HRL _stable_s1 '{"degrade":0.0,"rich_patches":0}'
```

**Reproduce the paper's statistics and figures from the published outputs** (numpy only, seconds):

```bash
python pool_seeds.py --published
```

```bash
python make_figure_data.py
```

Both write to `runs/`; their outputs match `../results/evaluations/pooled_seeds.json` and `../results/figures/*.dat` exactly.

**Pool your own evaluations, decompose the levers, and rerun the per-phase evaluation:**

```bash
python pool_seeds.py
```

```bash
python levers_learned.py
```

```bash
python eval_phases.py
```

`make_figure_data.py` then uses these outputs in `runs/` in place of the published ones.

**Timing** on an 8-core laptop: training takes about 10 minutes per family
and seed; one evaluation takes about 10 minutes.

**Determinism:** results are deterministic given seeds and library versions.
Training used PyTorch 2.1 on CPU; the published numbers were produced with
NumPy 1.24 (training) and NumPy 2.5 (simulation and evaluation).
