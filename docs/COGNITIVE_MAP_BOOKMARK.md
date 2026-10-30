# Bookmark: emergent cognitive-map canalization (not started)

This is a later alternative to the built-in perceptual narrowing in
`SPEC.md`. Here canalization is learned rather than imposed: precise priors
come from reward history, which is truer to the "overweighted priors" of
CANAL/REBUS.

## Sketch

- **Cognitive map.** The agent keeps a map of expected resource value
  V(tile) and a precision Π(tile).
  - Each harvest updates V by a delta rule: V ← V + η·(r − V).
  - Π grows with consistent rewards and shrinks with prediction error.
- **Perception as inference.** The perceived value of a tile combines the
  map prior and the sensory sample, weighted by precision:
  v̂ = (Π·V + Π_s·v_sensed) / (Π + Π_s).
  - High map precision makes the agent "see" what it expects, so exhausted
    ruts still look rich.
- **Action.** The agent plans over the map, by greedy value-per-cost or by
  short-horizon value iteration, and re-plans as the map updates. Sensory
  range stays fixed; narrowing emerges because high-precision regions
  dominate planning.
- **Decanalization (REBUS).** Temporarily scale map precision down
  (Π ← (1 − d)·Π), which flattens the landscape, and optionally raise η, a
  critical-period plasticity window.
- **Same world, endpoints and gates** as `SPEC.md` (G0–G3), so the imposed
  and emergent versions can be compared directly.

## Why it is deferred

- It needs a planning agent rather than a reactive policy.
- It has more free parameters: η, the precision dynamics, the planner
  horizon.
- The imposed version tests quickly whether the world's dynamics can
  produce the theory's predictions at all. If they can, the emergent version
  asks whether the agent produces canalization itself.

## Questions when it is picked up

- Is canalization emergent? Does Π concentrate on a few patches without
  being told to?
- Does the emergent version reproduce G1 and G2 from the imposed version?
- Do the two decanalization levers, precision and learning rate, differ, as
  Juliani et al. (2024) distinguish inference from plasticity?

## First build (30 October 2026): `model/mapagent.py`, `map_gates.py`

**Status:** started; the first gates are mixed. The world and protocol are
the same as `SPEC.md` (pellets 3–15).

### Agent

- **Map:** a per-cell expected value V and precision P.
  - Sightings give precision 0.05 per tick, standing on a cell 0.5/s, a
    harvest 2.
  - Precision decays with τ_P = 200 s.
  - Updates are precision-weighted: V ← (P·V + Ps·v)/(P + Ps).
- **Planning:** the agent believes a harvested cell regrows over base_delay
  (8 s), so its worth = V·(1 − e^(−Δt/8)).
  - When hungry, it heads for the best worth/(1 + distance) if that worth
    is at least v_explore, and explores otherwise.
  - Perception is fixed at 90° and 10 tiles; nothing narrows by
    construction.
- **References:**
  - LOWPREC: harvest precision 0 and τ_P = 20 s, so beliefs track the
    senses;
  - NOMAP: the reactive heuristic.
- **Decanalization:** at the window start P ← (1 − d)·P, and optionally V is
  flattened toward its mean (`flatten_values`). Inference uses (1 − d)·P
  during the window.

### Build fixes

1. With no exploration value, the agent orbited its first patch forever.
   Added `v_explore`.
2. After a harvest the planner kept targeting the empty cell underfoot.
   Added the regrowth expectation.

### Gates (32 paired trials, seeds 1000+)

| | v_explore 4 | v_explore 7 | v_explore 4, precision-only flattening |
|---|---|---|---|
| **S1** stable: MAP − LOWPREC phase-3 MAE | −0.3 [−1.2, 0.5] ✗ | −0.4 [−1.1, 0.2] ✗ | same ✗ |
| **G1** degrading: MAP − LOWPREC phase-3 trap rate | **+0.22 [0.09, 0.38] ✓** (0.31 vs 0.09) | +0.09 [−0.03, 0.25] ✗ (MAE +9 [0.4, 24]) | same ✓ |
| **G2** d = 0.5: MAP − undisturbed MAP, phase-3 MAE | +10 [−2, 29] ✗ | **+52 [16, 94]: harm** | +12 [−3, 28] ✗ |
| Doses 0.25–1 vs undisturbed, phase-3 MAE | +1 to +17 | +16 to +52 | +9 to +15 |

- The G3U check "passed" mechanically in two runs (best dose 0.75), but
  every dose was worse than no dose. The criterion should have required
  beating the undisturbed agent, and should be read as failed.
- Stable-world MAE is about 12 for both map agents, against 27 in phase 1
  for NOMAP. A map pays early; precision adds nothing.

### Reading

- **Learned precision produces trapping on its own** at v_explore 4. With no
  narrowing built in, the map agent is trapped in a degrading world 3× as
  often as an agent whose beliefs track its senses. That is emergent
  hypercanalization.
- **Precision does not pay in a stable world.** The benefit of
  canalization is still missing.
- **Flattening never helps and can harm.**
  - With value flattening, empty cells inherit the map's mean value, and
    the agent chases phantoms everywhere.
  - Precision-only flattening is milder but still not beneficial.
- **Not yet known:** whether trapped trials are caused by precise beliefs
  about the degraded rut. That check, the precision of trapped cells, needs
  doing before the G1 result is interpreted.

### Possible next steps

- **Make precision pay.** For example: more volatile sensing, where single
  sightings are noisy, so precise beliefs filter noise. Or longer travel
  between patches, so remembered routes save search.
- **Learning-rate lever.** Test decanalization as raised η, a plasticity
  window, rather than lowered precision (Juliani et al.: inference versus
  plasticity).

## Trap mechanism check (30 October 2026): `model/trap_diag.py`

**Method.**
- Map agents at v_explore 4, degrading world, CCC, 32 trials (seeds 1000+).
- Every 0.5 s, logged what the agent was pursuing: target cell, believed
  worth, precision, whether the pellet was really there, the patch's
  fertility, and distance.
- Compared trapped samples (h ≤ −100) with samples where the agent was
  hungry but not trapped.

| | MAP, trapped | MAP, hungry, not trapped | LOWPREC, trapped | LOWPREC, hungry, not trapped |
|---|---:|---:|---:|---:|
| Trials ever trapped | 13 / 32 | — | 9 / 32 | — |
| Share of time pursuing a remembered cell | 0.92 | 0.94 | 0.98 | 0.96 |
| **Target actually empty** | **0.37** | 0.18 | **0.02** | 0.02 |
| Target on a degraded patch (f < 0.5) | 0.75 | 0.73 | 0.84 | 0.72 |
| **Median target precision** | **15.5** | 11.7 | 2.2 | 2.1 |
| Believed worth / true value (median) | **6.7 / 5.3** | 9.3 / 10.5 | 7.6 / 8.2 | 10.9 / 11.4 |
| Distinct patches pursued per trial | **4.3** | 7.5 | 4.7 | 7.2 |
| Median target distance (tiles) | 1.9 | 2.6 | 1.6 | 3.0 |

**Two trap mechanisms, one of them precision-specific.**
1. **Phantom pursuit, caused by precise priors.**
   - While trapped, the precision-learning agent heads for cells that are
     actually empty 37% of the time. That is twice its untrapped rate, and
     about 18× the low-precision agent's rate (2%).
   - Its targets are more precise (15.5 vs 2.2), and it overestimates their
     worth (6.7 believed vs 5.3 true).
   - This is the hypothesized mechanism: precise beliefs about an exhausted
     rut override the evidence that it is empty.
2. **Local fixation, shared by both agents.**
   - Trapped agents of both kinds pursue fewer patches (about 4.5 vs 7.5),
     closer by (1.6–1.9 vs 2.6–3.0 tiles), and mostly degraded ones.
   - Nearby remembered cells keep scoring above the exploration threshold,
     so the agent circles a shrinking, depleted neighborhood instead of
     exploring.
   - This comes from the planner's exploration rule, not from precision.
     It explains why the low-precision agent also gets trapped (9/32).

**Implication for decanalization.**
- Lowering precision should reduce phantom pursuit.
- It cannot fix local fixation, which is what the rule comparing believed
  worth with v_explore produces.
- REBUS ties relaxed priors to more exploration (higher entropy). A faithful
  lever would therefore also make the unknown more attractive during the
  window: a raised or optimistic exploration value. That has not been
  tested. It may explain why precision-only flattening did not help.

## Exploration lever (30 October 2026): `model/map_levers.py`

**Design.**
- Decanalization now has two levers, both scaled by dose d during the
  100 s window:
  - precision reduction (`prec_lever`), with values not flattened;
  - an exploration boost: the threshold that remembered worth must beat
    rises to v_explore·(1 + boost·d).
- Map agent at v_explore 4, degrading world, 32 paired trials.
- The undisturbed agent's phase-3 MAE is 24.9 and its trap rate 0.31.

**Pre-set criteria:**
- **G2:** both levers at d = 0.5 beat the undisturbed agent.
- **G3U:** the best dose is a middle dose (0.5 or 0.75), beats the
  undisturbed agent, and beats both ends.

**Result: both fail at every boost tested (1, 2, 4).**

Phase-3 values; differences from the undisturbed agent with [95% CI]:

| | Boost 1 | Boost 2 | Boost 4 |
|---|---|---|---|
| Both levers, d = 0.25 | MAE +10 [−3, 27] | +11 [−3, 27] | +18 [0, 47] |
| Both, d = 0.5 | +14 [−4, 35] | +23 [−1, 60] | **+132** [77, 193] |
| Both, d = 0.75 | +3 [−6, 14]; trap **−0.12** [−0.25, −0.03] | +29 [1, 69] | **+301**; trap 1.0 |
| Both, d = 1 | +2 [−7, 11] | **+59** [26, 98] | **+282**; trap 1.0 |
| Exploration only, d = 0.5 | +6 [−6, 22] | +20 [1, 43] | **+170** |
| Exploration only, d = 1 | +20 [1, 43] | **+170** [114, 228] | **+313** |

**Reading.**
- **The harm side of the Goldilocks curve appears; the benefit side does
  not.**
  - Strong exploration during the window is catastrophic. At boost 4 with
    d ≥ 0.75, every trial is trapped.
  - The agent abandons known food and explores blindly while hungry: it
    heads straight ahead with only noise.
  - That is the "too much decanalization disorganizes" prediction, but no
    dose produces a net benefit.
- **The only release signal** is boost 1 at d = 0.75: trap rate −0.12
  [−0.25, −0.03]. MAE does not improve (+3), and that dose is not the best
  by the pre-set MAE rule.
- **Why escape doesn't pay here:**
  - Exploration is blind and costly. Fresh tiles plus passive loss cost
    about 7 h/s.
  - Alternatives are not better on average: patch quality is uniform at
    random, and any patch the agent uses degrades the same way.
  - The theory's benefit requires a better, unexploited alternative that
    the canalized agent fails to pursue. The world never guarantees one.

**Candidates for the benefit side** (not yet tested):
- **Heterogeneous world:** a few rich patches beyond the agent's habitual
  range, so escaping the rut has real upside.
- **Directed exploration:** decanalized exploration aims at unvisited or
  unknown regions, or at visible but unvisited food, rather than straight
  ahead.
- **Both together** test the theory's specific claim: canalized perception
  hides better options that relaxed perception reveals.

## Opportunity world and directed exploration (30 October 2026): `map_levers2.py`

**World and agent changes** (`world.py`, `mapagent.py`):
- `rich_patches = 2`, quality 25–35. They **appear at the phase-2 start**,
  at least 20 tiles from wherever the agent is then. Phase 1 is identical
  across arms, so the new layout is too.
- When present from the start, rich patches were found in phase 1 and
  simply became the rut.
- `directed_explore`: when exploring, the agent heads for the nearest
  never-seen cell at least 3 tiles away, not straight ahead.

**Run.** Map agent at v_explore 4 with precision-only flattening and both
levers during the window. Degrading world, 32 paired trials, seeds 1000+.

| Exploration | Undisturbed P3 MAE / trap | Best result vs undisturbed (95% CI) | G2 | G3U |
|---|---|---|---|---|
| Straight ahead | 22.2 / 0.09 | MAE never better (−1.6 to +7.3); trap −0.03 to −0.09 | ✗ at all boosts | ✗ |
| Directed | 30.1 / 0.12 | boost 1, d = 0.5: MAE −7.9 [−19.3, 0.3], trap **−0.12 [−0.25, −0.03]** | **✓ at boost 1**; ✗ at 0.5 and 2 | ✗ (best dose 0.25) |
| Directed, boost 2 | 30.1 / 0.12 | d = 0.75–1: trap −0.12 [−0.25, −0.03]; MAE −7 [−18, 1] | ✗ | ✗ |

**Escape endpoint:** rich-patch harvests in phases 2–3, dose minus
undisturbed.
- Straight-ahead exploration: **+5 to +13** at every dose and boost, all CIs
  above 0.
- Directed exploration: +1 to +11, most CIs above 0.

**Reading.**
- **Decanalization now reliably releases agents toward new opportunities.**
  Decanalized agents find and use the new rich patches that the undisturbed
  agent largely misses. This is the theory's core behavioral claim, shown
  for the first time in this project.
- **The regulation benefit is modest and fragile.**
  - With straight-ahead exploration, extra rich harvests do not improve
    MAE. Rich pellets (25–35) overshoot the setpoint, so finding them
    trades a deficit for an excess.
  - With directed exploration, MAE improves by about 6–8 and trapping by
    about 0.12. Only one setting clears the pre-set G2 criterion.
  - Directed exploration also worsens the undisturbed baseline (MAE 30 vs
    22), which leaves more room for improvement.
- **No inverted U.** The lowest dose often does as well as any. In this
  world even a little relaxation is enough, and directed exploration is
  never catastrophic, unlike blind exploration.
- **Trapping from precision (G1) vanishes in this world:** MAP − LOWPREC
  +0.03, n.s. Overall trap rates are low (0.09–0.12) because new food
  arrives at phase 2.

**Next candidates:**
- **Make the opportunity regulatory rather than calorie-dense:** rich =
  reliable (fast regrowth, no degradation) at ordinary quality. Escaping to
  it should then improve MAE, not cause overshoot.
- **Confirm on fresh seeds** before relying on any setting. The G2 pass is a
  single cell of a 6-config × 4-dose grid, so multiplicity applies.

## Reliable-opportunity world and fresh-seed confirmation (30 October 2026)

**World change** (`world.py`: `rich_degrade`, `rich_delay`). The two rich
patches that appear at the phase-2 start are now *reliable* rather than
calorie-dense:
- ordinary quality (8–12);
- they never degrade (`rich_degrade = 0`);
- they regrow twice as fast (`rich_delay = 0.5`).

Escaping to them should improve regulation rather than cause overshoot.

### Lever test (`map_levers2.py` with overrides → `runs/map_levers3.json`; seeds 1000+, 32 trials)

Phase-3 MAE, dose minus undisturbed [95% CI]:

| Exploration, boost | d = 0.25 | d = 0.5 | d = 0.75 | d = 1 |
|---|---:|---:|---:|---:|
| Straight, 0.5 | +0.6 | −3.4 [−8.0, 0.2] | −0.5 | +2.0 |
| Straight, 1 | +0.7 | +6.9 | −3.7 | −5.0 [−10.7, 0.0] |
| Straight, 2 | **−5.6 [−10.5, −1.9]** | −4.5 | +7.4 | **+35.1 [10.2, 64.9]** |
| Directed, 0.5 | −6.9 | −5.0 | **−9.8 [−19.8, −2.1]** | −4.1 |
| Directed, 1 | −10.3 [−23.1, 1.3] | −6.3 | **−12.5 [−24.5, −3.1]** | −7.1 |
| Directed, 2 | **−12.2 [−26.1, −1.6]** | **−9.3 [−18.0, −2.0]** | +1.3 | +28.5 [−2.3, 70.9] |

- **Inverted U** (pre-set G3U) **at directed boosts 0.5 and 1:** the best
  dose is 0.75, and it beats the undisturbed agent.
- **The harm side appears at boost 2:** strong doses worsen regulation
  (+28 to +35).
- **Precision-driven trapping (G1) returns** with straight-ahead
  exploration: MAP − LOWPREC phase-3 trap rate +0.12 [0.03, 0.25].
  Directed: +0.09 [−0.06, 0.25].

### Confirmation (`confirm_map.py`; fresh seeds 3000+, 64 paired trials)

The setting was chosen by the rule fixed beforehand (the most negative
beating cell): directed exploration, boost 1, d = 0.75. Undisturbed agent:
phase-3 MAE 46.0 (median 11.9), trap rate 0.31.

| Dose | P3 MAE (median) | MAE vs undisturbed | Trap vs undisturbed | Rich harvests vs undisturbed |
|---|---|---:|---:|---:|
| 0.25 | 34.1 (11.5) | −11.9 [−31.1, 7.3] | −0.08 [−0.20, 0.03] | −1.8 [−6.9, 3.1] |
| 0.5 | 24.1 (11.6) | −21.9 [−39.9, −5.2] | −0.09 [−0.20, 0.02] | +5.2 [−0.2, 11.0] |
| **0.75 (chosen)** | **20.6 (11.4)** | **−25.3 [−43.3, −8.8]** | **−0.14 [−0.25, −0.03]** | **+6.0 [0.2, 12.2]** |
| 1.0 | 19.5 (11.6) | −26.5 [−43.6, −12.1] | −0.19 [−0.30, −0.08] | +5.5 [0.1, 11.4] |

**Confirmed (pre-set criterion):** decanalization at the chosen setting
improves phase-3 regulation on fresh seeds. It frees trapped agents (trap
rate −0.14), and they harvest the new reliable patches more (+6).

**Qualifications.**
- **The benefit lives in the tail.** Medians barely move (about 11.5 in
  every arm). Means fall because trapped trials are rescued. This is what
  escaping an attractor looks like, but it should be reported as a shift
  in trap probability, not as a typical-trial gain.
- **The inverted U did not replicate at boost 1.** On fresh seeds the dose
  curve is monotone: d = 1 is as good as or better than 0.75. In the lever
  test the U appeared at boosts 0.5 and 1, and the harm side only at
  boost 2. The optimum likely depends jointly on dose and exploration
  boost; it needs its own confirmation across the boost axis.
- **Selection from a grid.** The setting was picked from 24 cells. The
  fresh-seed test guards against that selection, but only for this one
  setting.
- **The heuristic planner still stands in for a learned policy.**
  Precision, values and exploration rules are hand-specified.

### What is now established in the prototype (map agent, reliable-opportunity world)

1. **Learned precision traps the agent when its rut degrades**, with
   nothing narrowing perception by construction (straight-ahead
   exploration: +0.12 trap rate vs a low-precision agent). In the earlier
   world, the mechanism check traced this to precise beliefs about
   exhausted cells.
2. **Relaxing precision and opening exploration frees trapped agents**
   toward new, better resources, and improves regulation on fresh seeds.
3. **Too much decanalization harms** at the strongest exploration boost
   (lever test only, not yet confirmed).

## Dose × exploration-boost surface (30 October 2026): `surface.py`

**Setup.**
- Fresh seeds 4000+, 64 paired trials per cell.
- Reliable-opportunity world; map agent with directed exploration; both
  levers during the 100 s window.
- Intensity is defined as d·(1 + boost).
- Undisturbed agent: phase-3 MAE 29.5 (median 12.3), trap rate 0.27.

**Phase-3 MAE, cell minus undisturbed [95% CI]:**

| Boost \ dose | 0.25 | 0.5 | 0.75 | 1 |
|---|---:|---:|---:|---:|
| 0.5 | −1.4 [−15, 13] | **−10.3** [−20, −2] | −9.1 [−18, 0] | **−12.1** [−24, −1] |
| 1 | −8.6 [−18, 1] | **−10.3** [−21, −1] | −9.7 [−20, 0] | **−14.4** [−25, −6] |
| 2 | **−10.6** [−21, −2] | **−11.6** [−22, −4] | +10.8 [−8, 34] | **+26.6** [8, 46] |
| 3 | −5.3 [−17, 7] | +7.2 [−5, 19] | **+90.8** [63, 121] | **+165.9** [133, 201] |

**Phase-3 trap rate, cell minus undisturbed:**

| Boost \ dose | 0.25 | 0.5 | 0.75 | 1 |
|---|---:|---:|---:|---:|
| 0.5 | −0.03 | −0.09 | −0.06 | −0.11 |
| 1 | −0.06 | −0.11 | −0.11 | −0.09 |
| 2 | −0.11 | −0.06 | +0.05 | +0.25 |
| 3 | −0.06 | +0.16 | +0.52 | +0.72 |

**Pre-set criteria: all three pass.**
- **Benefit ✓:** 7 of 16 cells beat the undisturbed agent. The best is
  boost 1, d = 1 (intensity 2): −14.4 [−25, −6].
- **Harm ✓:** the strongest cell (boost 3, d = 1) is worse than no
  intervention: +165.9 [133, 201], trap rate +0.72.
- **Interior optimum ✓:** the best cell's intensity (2) lies strictly
  inside the grid's range (0.375–4). It beats both the weakest cell (−13.0
  [−27.0, −1.8]) and the strongest (−180 [−217, −146]).

**Reading: the Goldilocks pattern is now confirmed on fresh seeds.**
- **Too little** (intensity ≲ 0.5) does little.
- **Moderate decanalization** (intensity about 0.75–2) releases trapped
  agents and improves regulation.
- **Too much** (intensity ≳ 2.25) disorganizes behavior. The agent keeps
  exploring while starving.
  - Rich-patch harvests are highest at the strongest cells (+19 to +35).
  - Phase-2 MAE rises to 164–259.
  - The cost carries over into phase 3 (median MAE 86–158).
- The optimum is a ridge along moderate intensity, not a single dose. Dose
  and exploration boost trade off against each other.

**Qualifications.**
- **Medians barely move in the beneficial region** (about 11.5–12.5). The
  benefit is a lower trap probability, while the harm moves the median too.
- **Hand-built pieces:** the intensity index, both levers and the planner
  are specified by hand. The levers are modeling choices motivated by
  REBUS (relaxed precision, more exploration), not derived.
- **No learned policy yet:** this prototype has not tested HRL vs NF
  reward differences.

## Learned HRL and NF option policies (30 October 2026)

**Code:** `policy_agent.py` (option policy over the map), `ppo_train.py`
(PPO, torch env), `select_ckpt.py`, `eval_policies.py`.

**Policy.**
- Every 0.5 s, a 64×64 MLP picks one of four options, each executed with
  the map agent's steering rules:
  - SEEK the best remembered cell;
  - EXPLORE the frontier;
  - CRUISE the cheapest path;
  - AVOID visible food.
- Inputs (11 values): h, best remembered worth, its distance and
  precision, visible food, frontier distance, map coverage, tile cost, and
  time since the last meal.
- Rewards: HRL `drive-ratio` (D = |h|/25) and NF `nf-deficit-drive`
  (D = max(−h, 0)/25), with κ·(D_prev − D_cur) per tick. No
  finite-endpoint correction.

**Training and selection.**
- World: the reliable-opportunity degrading world, as used for evaluation.
  No intervention; seeds 10000+; 60 iterations × 16 trials.
- Checkpoints were selected on 8 held-out trials (seeds 20000+) by each
  family's own objective: HRL iteration 50 (mean |h| 13.2), NF iteration 40
  (mean deficit 0.8).
- NF learns what its objective rewards: it avoids deficits and overshoots
  freely, with mean h around +50 to +110.

**Evaluation** (fresh seeds 5000+, 64 paired trials, sampled actions).
Cells are temperature boost × dose, with the precision lever on.

| | HRL | NF |
|---|---|---|
| Undisturbed phase-3 MAE (median) / trap rate | 12.5 (11.4) / 0.02 | 162 (133) / 0.05 |
| G1: MAP − LOWPREC | trap +0.02 [0.00, 0.05]; MAE **+2.3 [1.3, 3.5] ✓** | MAE −44 [−71, −18] ✗ |
| Best cell vs undisturbed, phase-3 MAE | −0.7 [−1.9, 0.8] (no benefit) | +9.6 [−13, 34]; most cells **+26 to +34 (harm)** |
| Rich-patch harvests vs undisturbed | +5 to +14 | +5 to +13 |
| Harm at (3, 1) | none (−0.7) | +25.8 [−0.1, 54] |

- **Contrast,** HRL − NF effect at (boost 1, d 1): **−28.1 [−49.3, −7.2]**.

**Reading.**
- **The learned HRL policy is almost never trapped** (0.02). It regulates
  near the noise floor (MAE about 12), so decanalization has nothing to
  improve. Decanalized HRL agents do find the new patches more (+5 to +14
  harvests), but regulation does not change. Precise beliefs still cost a
  little (+2.3 MAE vs low precision): canalization exists but is mild.
- **Decanalization harms NF** (+26 to +34 MAE). The HRL-vs-NF contrast is
  significant, matching the paper's prediction that NF does not benefit,
  but it is driven by NF harm, not HRL benefit.
- **Why no hypercanalization:** the policy was trained in the same changing
  world it is evaluated in, including degradation and new patches at
  300 s. RL learned to cope with exactly this change.
- **Hypercanalization, in the theory, is maladaptive persistence after the
  world changes in ways the habit did not anticipate.** The next test is
  therefore distribution shift: train in a stable world, evaluate in the
  changing one.

## Distribution-shift test (started 30 October 2026)

- **Train** HRL and NF in a *stable* world (no degradation, no rich patches)
  and select checkpoints there. Tag `_stable`.
- **Evaluate** in the changing world: degradation plus reliable rich patches
  appearing at 300 s.
- The criteria are the same as in `eval_policies.py`.
- **Added prediction, fixed before running:** the stable-trained HRL
  policy, undisturbed, is trapped in phase 3 more often than the
  matched-trained one (0.02).

### Result (fresh seeds 5000+, 64 paired trials)

**Checkpoints:** HRL `_stable` iteration 55; NF `_stable` iteration 55.

| | HRL (stable-trained) | NF (stable-trained) |
|---|---|---|
| Undisturbed phase-3 MAE (median) / **trap rate** | 145 (81) / **0.67** | 219 (174) / 0.50 |
| G1: MAP − LOWPREC | trap +0.14 [0.00, 0.28]; **MAE +67 [23, 115] ✓** | **trap +0.38 [0.25, 0.50] ✓**; MAE +3 [−37, 45] |
| Cell (temperature 1, d = 1) vs undisturbed | **MAE −86 [−131, −46]; trap −0.25**; median 81 → 14 | MAE −26 [−60, 9]; trap −0.34 |
| Cell (temperature 3, d = 1) vs undisturbed | **MAE −119 [−166, −78]; trap −0.30**; median → 14 | MAE −24 [−59, 11]; trap −0.25 |
| Benefit / harm criteria | **✓** / ✗ | ✓ (2 cells, CIs near 0) / ✗ |
| Rich-patch harvests vs undisturbed | −3 to +14 | +11 to +21 |

- **Contrast** (HRL − NF effect at temperature 1, d = 1): **−60.5
  [−111.9, −10.8]**.
- **The added prediction is confirmed:** the stable-trained HRL policy is
  trapped 0.67 of the time in the changing world, against 0.02 for the
  matched-trained one.

**Reading. This is the clearest demonstration of the theory so far.**
- **Canalization formed in a stable world becomes hypercanalization when
  the world changes.**
  - Trap rates are 0.67 (HRL) and 0.50 (NF).
  - Learned precision makes it worse: HRL MAE +67 vs low precision; NF
    trap rate +0.38.
- **Decanalization frees the HRL agent and restores its regulation.**
  - Relaxed precision plus a higher sampling temperature for 100 s cuts
    phase-3 MAE by 86–119.
  - Median MAE falls from 81 to 14. Unlike the heuristic prototype, the
    benefit now reaches typical trials.
- **NF benefits much less.** Its trap rate falls (−0.17 to −0.34), but
  regulation barely improves, because NF overshoots freely. The HRL-vs-NF
  contrast is significant in the direction the paper predicted.
- **No harm side.** The strongest tested setting (temperature 3, d = 1) is
  the best. Stronger perturbation is needed to look for the Goldilocks
  ceiling with learned policies.

**Limitations.**
- One training seed per family.
- The option set and the map machinery are hand-built.
- The training and evaluation worlds differ by construction (this is the
  test).
- MAE for NF is dominated by surplus.
- No finite-endpoint reward correction.
- Selection used the training world's held-out trials.

**Next (started):** replicate across 3 training seeds per family, and
extend the temperature axis to 10 to look for the harm side.

## Seed replication (4 training seeds per family; `pool_seeds.py` → `runs/pooled_seeds.json`)

**Method.** Training seed is the unit of inference: per-seed paired-trial
effects are averaged across seeds, with a t-interval over seeds (df 3).
Each seed is evaluated on 64 trials (seeds 5000+).

| | HRL | NF |
|---|---|---|
| Undisturbed phase-3 trap rate | **0.70 [0.66, 0.73]** (0.67–0.72) | 0.48 [0.44, 0.51] |
| G1: precision cost, MAP − LOWPREC | trap +0.24 [0.12, 0.36]; **MAE +60 [43, 78]** | **trap +0.33 [0.25, 0.41]**; MAE +27 [−12, 66] |
| Decanalization, temperature 1, d = 1: MAE | **−83 [−89, −77]** (every seed −78 to −86) | −34 [−54, −14] |
| … trap rate | −0.23 [−0.34, −0.12] | −0.33 [−0.39, −0.28] |
| Temperature 3, d = 1: MAE | **−98 [−122, −73]** | −28 [−58, 2] |
| Temperature 10, d = 1 (3 seeds): MAE | −72 [−100, −44] | −9 [−14, −4] |
| Temperature 10 minus temperature 3, MAE | **+18 [5, 31]** | +21 [−33, 75] |
| Phase-2 MAE at temperatures 1 / 3 / 10 | 66 / 62 / **108** | 91 / 108 / **139** |

- **HRL − NF contrast** at temperature 1, d = 1: **−48 [−68, −29]**;
  every seed negative (−33 to −60).

**Reading (robust across seeds):**
- **Canalization learned in a stable world becomes hypercanalization when
  the world changes.** About 70% of HRL agents are trapped. Learned
  precision makes it worse.
- **Decanalization releases the agents and restores regulation.** HRL
  phase-3 MAE falls by 83–98 at moderate settings, and the effect is
  nearly identical across seeds.
- **There is an interior optimum along temperature.**
  - Temperature 10 is reliably worse than 3 (+18 [5, 31]), though still
    better than no intervention.
  - It is also much costlier during the intervention: phase-2 MAE 108 vs
    62.
  - The harm side (worse than no intervention) was not reached with
    learned policies.
- **HRL benefits much more than NF:** NF's trap rate falls, but its
  regulation improves less, since it tolerates surplus. This matches the
  paper's original hypothesis, now in a model where canalization
  demonstrably exists.
