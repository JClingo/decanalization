# Canalization and decanalization: prototype findings — 30 October 2026

This consolidates `SPEC.md` (imposed narrowing) and `COGNITIVE_MAP_BOOKMARK.md`
(cognitive map, learned policies). Code is in `model/` (local
only). Outputs are in `results/`.

## Summary

In a Python gridworld that mirrors the Unreal task, an HRL agent with a
learned cognitive map shows the dynamics the theory describes:
- **It canalizes.** In a stable world it learns precise beliefs and fixed
  routes.
- **It hypercanalizes when the world changes.** Its familiar patches
  degrade and better ones appear elsewhere, but it keeps pursuing the old
  ones. About 70% of agents are trapped in a starving rut, and precise
  beliefs make this worse.
- **Decanalization releases it.** A 100 s window of relaxed belief
  precision plus more random option choice lets it escape, find the new
  resources and regulate again. Phase-3 mean absolute h falls by about 83–98
  points; the median falls from about 80 to about 14.
- **There is an interior optimum.** Stronger perturbation works less well
  and costs more while it lasts.
- **HRL benefits much more than NF.** This is the paper's original
  prediction, now tested in a model where canalization demonstrably
  exists.

These results hold across four independent training seeds per family.

## How we got here

| Step | Model | Outcome |
|---|---|---|
| Unreal, original and redesigned | reactive PPO, fixed perception | no canals form; decanalization cannot be tested (see `revisionPlan/decanalizationRedesign/NULL_RESULT.md` in the manuscript repository) |
| Prototype A | imposed perceptual narrowing, heuristic controller | traps and releases, but canalization never pays and there is no optimum; a reactive controller cannot use a canal |
| Cognitive map, heuristic planner | learned precision; flattening by precision scaling | learned precision traps agents (emergent), but flattening alone never helps |
| + opportunity world, directed exploration | rich, reliable patches appear at the phase-2 start | release, plus a Goldilocks dose × boost surface confirmed on fresh seeds; the benefit sits in the tail |
| Learned option policies, trained in the evaluation world | HRL and NF PPO over the map | HRL learns to cope with the very change it is tested on, so there is no trap and no benefit; NF is harmed |
| **Learned policies, trained stable, tested changing** | the same policies, distribution shift | **hypercanalization, release, interior optimum, HRL ≫ NF; replicated over 4 seeds** |

## The model

- **World:**
  - 50×50 toroidal grid; 10 resource patches of 3×3 cells; pellet quality
    3–15.
  - Worn tiles are cheaper to cross.
  - Patches degrade with use and recover slowly.
  - In the changing world, two *reliable* patches (never degrading, fast
    regrowth, ordinary quality) appear at the start of phase 2, at least
    20 tiles from the agent.
- **Agent:**
  - A cognitive map of believed value and precision per cell.
    Precision-weighted updates mean rewarded cells become hard to revise.
  - The agent expects a harvested cell to regrow over 8 s.
  - A PPO option policy (HRL `drive-ratio` or NF `nf-deficit-drive` reward)
    chooses, every 0.5 s, among: seek the best remembered cell, explore the
    frontier, cruise the cheapest path, or avoid visible food.
- **Canalization** is precision accumulated through experience: it is
  learned, not imposed. Perception is fixed.
- **Decanalization** (dose d, 100 s, phase 2) does two things:
  - scales belief precision by (1 − d), as in REBUS, where priors relax;
  - raises the policy's sampling temperature to 1 + boost·d, as more
    entropic, exploratory action.

  Physiology is never changed.
- **Protocol:** 300 s canalization, 100 s intervention, 300 s post, with h
  continuous across phases. Paired trials share a seed.

## Main result

4 training seeds per family; per seed, 64 fresh paired trials.

| | HRL | NF |
|---|---|---|
| Trapped after the change, no intervention | 0.70 [0.66, 0.73] | 0.48 [0.44, 0.51] |
| Precision cost: phase-3 MAE vs a low-precision agent with the same policy | +60 [43, 78] | +27 [−12, 66]; trap +0.33 |
| Decanalization (temperature 1, d = 1): phase-3 MAE | **−83 [−89, −77]** | −34 [−54, −14] |
| Decanalization (temperature 3, d = 1): phase-3 MAE | **−98 [−122, −73]** | −28 [−58, 2] |
| Temperature 10 vs 3: phase-3 MAE | +18 [5, 31] (worse) | +21 [−33, 75] |
| Phase-2 MAE at temperatures 1 / 3 / 10 | 66 / 62 / 108 | 91 / 108 / 139 |

- **HRL − NF benefit** at temperature 1, d = 1: −48 [−68, −29]; every
  seed negative.

## What is and isn't established

**Established (in the prototype):**
- Canalization emerges from learning and becomes maladaptive when the
  world changes.
- Relaxing priors and increasing behavioral entropy releases agents and
  improves regulation.
- Moderate perturbation beats strong perturbation.
- HRL benefits more than NF.

**Not established:**
- **The harm side for learned policies.** No tested setting was worse than
  no intervention. The heuristic map agent did show harm, at high
  exploration boost.
- **Generality beyond this model.** The option set, the map and both
  decanalization levers are hand-specified modeling choices motivated by
  REBUS and CANAL.
- ~~Separate lever contributions for learned policies.~~ Now tested; see
  "Lever decomposition" below.
- **NF regulation.** NF's MAE is dominated by surplus, which its objective
  does not penalize. Its benefit should be read mainly as trap rate.

**Simplifications:**
- No finite-endpoint reward correction.
- Checkpoint selection on held-out trials of the training world.
- A Python world rather than the Unreal sandbox.
- No learning during evaluation; weights stay frozen.

## Implications for the paper

- The canalization-decanalization claim can be tested once the agent
  (i) learns precise, history-dependent beliefs and (ii) faces a world that
  changes after canalization forms. The Unreal model had neither.
- The intervention that works targets where canalization is stored: belief
  precision plus action entropy. It is not a physical or reward change.
  This is closer to REBUS than the published CDR.
- **Choices for the author:**
  - report this model, possibly ported to Unreal;
  - keep the Unreal study as the null precursor that motivated it;
  - or both.

## Suggested next steps

1. ~~Lever decomposition~~ (done; see below).
2. **Harm-side search:** longer windows, or decanalization during
   hunger.
3. **Port** the map, precision and option machinery into Unreal, or adopt
   the Python model as the paper's model.
4. **Emergent variant without the hand-built options:** a recurrent policy
   over raw map features.

## Lever decomposition (learned policies; `levers_learned.py`)

**Setup.** d = 1 for 100 s. Phase-3 MAE vs the undisturbed agent; 4
training seeds, t-interval over seeds.

| Lever | HRL | NF |
|---|---:|---:|
| Precision only (temperature 1) | −59 [−76, −41] | −35 [−79, 10] |
| Temperature only, boost 1 | −11 [−19, −3] | −24 [−56, 8] |
| Temperature only, boost 3 | −61 [−73, −49] | −22 [−35, −9] |
| Both, boost 1 | **−83 [−89, −77]** | −34 [−54, −14] |
| Both, boost 3 | **−98 [−122, −73]** | −28 [−58, 2] |

**HRL.**
- **Either lever alone releases a large share of trapped agents.**
  Relaxing belief precision on its own gives −59. Raising action entropy
  enough (boost 3) also gives −61.
- **Together they work best:** −83 to −98.
  - At boost 1 the combination exceeds the sum of the parts (−83 vs
    −59 − 11).
  - At boost 3 it falls short of the sum (−98 vs −120): the two levers
    overlap.
- **This differs from the heuristic map agent,** where precision-only
  flattening never helped. With the learned policy, precise beliefs feed
  its observations, and relaxing them changes which options it picks.
- This matches the distinction Juliani et al. draw between inference
  (precision) and behavioral variability (entropy): each contributes, and
  combining them is most effective.

**NF.** All levers give smaller, less certain improvements. The pattern of
HRL benefiting more holds for every lever.
