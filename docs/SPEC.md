# Canalization as perceptual narrowing — prototype spec, 30 October 2026

Status: prototype under construction in `model/` (Python;
local only). Parameter values are starting points, set in the gates.

**Author decisions, 30 October 2026:**
- Prototype the dynamics in a fast Python gridworld first, then port what
  works to Unreal.
- Bookmark the emergent cognitive-map variant for a later test
  (`COGNITIVE_MAP_BOOKMARK.md`).

## Why

Neither the Unreal model nor its redesign produced canals
(see `revisionPlan/decanalizationRedesign/NULL_RESULT.md` in the manuscript repository). Four ingredients were missing:
- use does not make a path more attractive;
- perception never narrows;
- the rut never stops paying while still holding the agent;
- canalization lives nowhere that decanalization can reach.

This model builds the theory's mechanism in explicitly, in the spirit of
CANAL/REBUS precision weighting, and tests its consequences:
- Does rigidity cost regulation once ruts degrade?
- Does relaxing perception alone produce escape and better recovery?
- How does the outcome depend on dose?
- Does HRL differ from NF?

**Built in:** perceptual narrowing as a function of canal depth.
**Emergent:** everything downstream of it.

## World (mirrors the Unreal sandbox where possible)

- **Grid:** 50×50 tiles; distances in tile units, 1 tile = 1,000 Unreal
  units.
- **Boundary:** toroidal. This is simpler than Unreal's antipodal wrap and
  avoids its escape fault.
- **Movement:**
  - Time step dt = 0.05 s.
  - The agent moves forward at 5 tiles/s (0.25 tiles per tick).
  - It steers with a ∈ [−1, 1], turning by a·10° per tick.
- **Homeostasis:** h changes each tick by
  + pellet value − traversal cost(tile)·dt − passive loss·dt.
  - Passive loss is 2/s.
  - The setpoint is 0.
- **Wear:** each tile has an entry count n that fades with half-life
  T_n = 200 s (lazy decay at entry).
  - cost(n) = c_min + (c0 − c_min)·exp(−n/k), with c0 = 5/s, c_min = 1/s
    and k = 3.
  - Worn paths are cheap.
- **Resource patches:**
  - 10 patches of 3×3 cells, placed at random without overlap; one pellet
    per cell.
  - Each patch has a quality q drawn uniformly from [5, 35]. A pellet's
    value is q.
- **Use-dependent degradation (the rut stops paying):**
  - Each patch has a fertility f ∈ (0, 1].
  - Every harvest multiplies f by (1 − δ), with δ = 0.08.
  - f recovers toward 1 with time constant τ_f = 300 s.
  - A harvested pellet regrows after base_delay / f, with base_delay = 8 s.
  - A patch that is used hard therefore regrows slowly, and its real yield
    falls.

## Agent state: canal depth κ

```
fam   = 1 − exp(−n_tile / n0)            n0 = 3
dκ/dt = α·fam·(1 − κ) − β·κ               α = 1/20 s⁻¹, β = 1/120 s⁻¹
```

κ rises while the agent travels familiar tiles and decays slowly otherwise.
It is the depth of the valley in Waddington's picture. κ is not an
observation: the agent does not know how canalized it is.

## Perception narrows with κ

The perceived depth is κ_p = (1 − d)·κ, where d is the decanalization dose
(d = 0 outside the intervention).

- **Field of view:** θ = 90°·(1 − 0.75·κ_p), so from 90° down to 22.5°.
  There are 7 rays, evenly spread across θ.
- **Range:** L = 10·(1 − 0.7·κ_p) tiles, so from 10 down to 3.
- **Prior precision:** π = κ_p.
  - On familiar tiles (fam > 0.5), a cell of a patch the agent has
    harvested is perceived to hold value (1 − π)·v_actual + π·v_prior.
  - v_actual is 0 if the pellet has not regrown.
  - v_prior is the agent's remembered mean harvest value from that patch.
  - A precise prior therefore "sees" food in an exhausted rut.
- **Per-ray observations:**
  - the perceived value of the nearest seen pellet, divided by its distance
    relative to L;
  - the mean predicted traversal cost along the ray;
  - h/25.

## Phases and arms

**Phases:** 300 s canalization, 100 s intervention, 300 s post. h is
continuous across phases: the intervention no longer changes physiology, so
carrying h over is not a confound.

| Arm | Phase 2 | Role |
|---|---|---|
| CCC | d = 0 | control |
| CDR | d = 1 (fully relaxed perception) | decanalization |
| WIDE | d = 1 throughout | never narrows; the cost-of-rigidity reference |
| Dose | d ∈ {0.25, 0.5, 0.75}, duration ∈ {25, 50, 100 s} | dose-response (G3) |

Trials are paired: the same seed gives the same layout, start and noise
stream in every arm until the arms diverge.

## Controllers

1. **Stage A, heuristic.** A homeostatic heuristic steers toward the ray
   with the best perceived food score when h is below a threshold, and
   otherwise toward the cheapest ray. It is used to tune the world quickly.
2. **Stage B, learned.** PPO policies with the HRL `drive-ratio` and NF
   `nf-deficit-drive` rewards. They are trained across the natural range of
   κ, with fixed weights at evaluation, using the torch environment at
   a Python environment with PyTorch.

## Endpoints

- **Canal:**
  - route persistence: Jaccard of entered-tile sets between consecutive 50 s
    windows, against cross-trial chance;
  - κ over time.
- **Hypercanalization:**
  - time share on degraded patches (f < 0.5) and their surroundings;
  - time-MAE in the late part of phase 1.
- **Escape:**
  - new patches harvested in phases 2–3;
  - P1–P3 route Jaccard;
  - time from the phase-2 start to the first harvest at a patch not used in
    phase 1.
- **Benefit:** phase-3 time-MAE and drive return, CDR − CCC, paired.

## Gates (in order)

| Gate | Test | Pass |
|---|---|---|
| G0 Canal forms | CCC, phase 1 | consecutive-window route Jaccard ≥ 2× chance, and mean κ ≥ 0.5 over the last 100 s |
| G1 Hypercanalization costs | CCC vs WIDE, phase 3 | CCC time-MAE worse, paired 95% CI excluding 0; CCC spends more time on degraded patches |
| G2 Decanalization helps | CDR vs CCC, phase 3 | CDR time-MAE better, paired CI excluding 0; CDR harvests more new patches |
| G3 Dose-response | dose grid | non-flat; report the shape (an inverted U is predicted) |

G0–G2 are tuned with the heuristic, then re-run with the learned policies.
**If G0–G2 cannot pass with any reasonable parameters, the mechanism itself
does not produce the theory's dynamics, and that is the finding.**

## Stage A, run 1 (30 October 2026): heuristic controller, 32 paired trials per arm

- Code: `model/world.py`, `gates.py`
- Outputs: `results/`
- A trial (700 s) takes about 2.5 s, so a 3-arm × 32-trial gate run takes
  about 16 s on 8 cores.
- Added parameter: `prior_gain`, with π = min(1, prior_gain·κ_p).

**Base world:** pellets 5–35 as in Unreal.
- The canal forms strongly: G0 passes at 14.6× chance, with late-phase-1
  κ = 0.77.
- The heuristic overshoots, though. Crossing a 3×3 patch eats several large
  pellets, h swings to ±150 in every arm, and G1/G2 are swamped. Pellet
  values were therefore reduced to 3–15, with controller thresholds seek
  below −3 and avoid above 8.

**Results with pellets 3–15.** Phase time-MAE is given as P1 / P2 / P3;
paired 95% CIs.

| Config | G0 ratio | G1: CCC − WIDE, P3 MAE | G2: CDR − CCC, P3 MAE | CCC MAE | CDR MAE | WIDE MAE |
|---|---:|---:|---:|---|---|---|
| Full mechanism (narrowing + prior) | 9.9× | **+223** [150, 303] ✓ | **−184** [−302, −86] ✓ | 75 / 196 / 289 | 75 / 104 / 105 | 29 / 48 / 66 |
| Prior only (no narrowing) | 10.5× | **+60** [13, 127] ✓ | **−34** [−73, −4] ✓ | 42 / 81 / 126 | 42 / 67 / 92 | 29 / 48 / 66 |
| Narrowing only (no prior) | 10.4× | +35 [−76, 136] ✗ | **−60** [−117, −14] ✓ | 59 / 106 / 101 | 59 / 73 / 42 | 29 / 48 / 66 |
| Moderate narrowing (fov 0.5, range 0.4) + prior | 9.7× | +32 [−40, 115] ✗ | −29 [−63, −5] | 62 / 112 / 98 | 62 / 81 / 69 | 29 / 48 / 66 |
| Moderate narrowing + half prior | 9.5× | +17 [−10, 47] ✗ | −23 [−112, 38] ✗ | 62 / 104 / 83 | 62 / 81 / 60 | 29 / 48 / 66 |

The moderate-narrowing G2 row passes on MAE but fails the gate's
new-patch criterion.

**What this shows (heuristic stand-in, so it tunes the world rather than
supplying evidence):**
- **Routes canalize in every arm, WIDE included** (κ ≈ 0.8). That comes
  from the controller's preference for cheap, worn tiles. The arms differ
  only in whether perception narrows. The physical canal is emergent;
  perceptual canalization is the built-in part.
- **Hypercanalization appears, and it costs.**
  - With the full mechanism, the canalized agent's regulation collapses:
    phase-3 MAE 289 against 66 when perception never narrows.
  - Relaxing perception for 100 s in phase 2 recovers most of the gap
    (105), even though perception narrows again in phase 3.
- **The phantom prior drives most of the cost.**
  - A prior alone (seeing remembered food in exhausted familiar patches)
    passes G1 and G2.
  - Narrowing alone helps under decanalization but does not reliably cost
    regulation.
- **The full mechanism is too strong:** canalized agents starve rather than
  merely regulate worse. Milder settings lose G1 at 32 trials. A
  middle setting that is costly but survivable still needs to be found.
- **Time on degraded patches (f < 0.5) hardly differs across arms**
  (0.22–0.25). The cost runs through what the agent perceives (phantom food,
  invisible alternatives), not through how long it sits on degraded patches.
  The degraded-time endpoint may need redefining, e.g. as time spent
  approaching empty familiar cells.

**Next:**
1. Find a survivable middle setting: a full mechanism with a weaker prior
   or slower κ.
2. Run the dose-response grid (G3).
3. Stage B: train PPO (HRL and NF) in the chosen world and rerun the gates.
   A learned policy may avoid the overshoot that the heuristic shows.

## Stage A, run 2 (30 October 2026): tuning, confirmation and dose-response

### Tuning sweep (`sweep.py`; seeds 1000+, 32 trials)

- **Grid:** prior gain {0.6, 0.8, 1.0} × narrowing {(0.75, 0.7), (0.6, 0.5)} ×
  κ decay β {1/120, 1/60}.
- **Pre-set criterion** (pass G0–G2 with CCC phase-3 mean MAE ≤ 2.5× WIDE):
  **no setting qualified.**
- **Why:** outcomes are **bistable**. A trial either falls into a starving
  rut or regulates well, so means and CIs are dominated by a few collapsed
  trials (G1 CIs about ±120).
- The data were therefore re-expressed by **trap rate**: a phase is
  trapped if h ≤ −100. This endpoint was adopted after seeing the sweep and
  is exploratory.
- Trap rates are cleanly ordered in every setting: WIDE 3–6%, CCC 19–78%,
  CDR always below CCC.
- **Chosen setting** (`confirm_dose.py`, CHOSEN): prior gain 0.8, narrowing
  (0.6, 0.5), β = 1/120, pellets 3–15. It traps about half of trials in
  phase 1, rescues strongly, and keeps median MAE survivable.

### Confirmation (fresh seeds 2000+, 64 paired trials)

| Arm | Trap rate P1 / P2 / P3 | Median MAE P1 / P2 / P3 | Mean MAE P3 | New patches |
|---|---|---|---:|---:|
| CCC | 0.45 / 0.50 / 0.53 | 24 / 43 / 27 | 146 | 0.98 |
| CDR (d = 1, 100 s) | 0.45 / 0.30 / 0.28 | 24 / 12 / 14 | 47 | 1.39 |
| WIDE | 0.08 / 0.03 / 0.05 | 10 / 9 / 9 | 11 | 0.75 |

- **G1′ (canalization traps):** CCC − WIDE phase-3 trap rate +0.48
  [+0.36, +0.61]. ✓
- **G2′ (decanalization releases):** CDR − CCC phase-3 trap rate −0.25
  [−0.39, −0.11] ✓. Mean phase-3 MAE −99 [−147, −58]; median 14 vs 27.

### Dose-response (64 paired trials per cell)

Phase-3 trap rate, CDR-dose minus CCC, [95% CI]:

| Dose d \ duration | 25 s | 50 s | 100 s |
|---|---:|---:|---:|
| 0.25 | 0.00 [−0.14, 0.12] | −0.03 [−0.16, 0.09] | −0.06 [−0.20, 0.08] |
| 0.5 | −0.02 [−0.14, 0.11] | −0.05 [−0.17, 0.08] | **−0.20** [−0.33, −0.08] |
| 0.75 | −0.12 [−0.27, 0.02] | −0.03 [−0.14, 0.08] | −0.12 [−0.25, 0.00] |
| 1.0 | −0.05 [−0.17, 0.08] | **−0.22** [−0.36, −0.09] | **−0.25** [−0.39, −0.11] |

- **G3 (non-flat):** ✓. Release grows with dose × duration. Weak or brief
  doses do nothing; strong, long ones release about a quarter of trapped
  agents.
- **No inverted U.** The intervention never costs anything: phase-2 MAE
  falls with dose (−65 at d = 1, 100 s), because relaxed perception is
  simply the WIDE agent's accurate wide perception.

### Design gap this exposes

- **In this world narrowing is a pure handicap.** WIDE is best at every
  point, so there is no reason to canalize. The theory says canalization is
  efficient while the world is stable and turns maladaptive only when ruts
  degrade.
- **A Goldilocks curve needs a cost on both sides:**
  - **A cost of wide perception**, e.g. sensory noise or metabolic cost
    that grows with field and range. Narrowing then pays in a stable world.
  - **A cost of decanalization:** REBUS describes the decanalized state as
    high-entropy, not accurate, so relaxed perception should be noisier.
    High doses then disorganize behavior.
- Without both, the model can show that canalization traps and that
  decanalization releases, but not an optimum.

## Stage A, run 3 (30 October 2026): two-sided costs — criteria not met

### Added to `world.py` (all default off)

- `breadth_cost`: a metabolic cost of wide perception, in h/s at full
  field × range.
- `perc_noise`: log-normal noise on perceived food and cost, scaled by
  breadth.
- `dec_noise`: the same noise scaled by dose d. This models the high-entropy
  decanalized state.
- `kappa_reward`: canal depth grows with familiarity × recent harvest rate,
  instead of familiarity alone.
- WIDE is now a separate never-narrow flag: it carries no decanalization
  noise.

### Sweep (`cost_sweep.py`; seeds 1000+, 32 paired trials)

Pre-set criteria:
- **S1:** canalization pays in a stable world (degradation 0).
- **G1′:** it traps agents in a degrading world.
- **G2′:** a full dose releases them.
- **G3U:** the trap rate is lowest at d = 0.5 or 0.75, at least 0.08 below
  both ends.

| Family | Configs | S1 | G1′ | G2′ | G3U |
|---|---:|---|---|---|---|
| breadth_cost {0.5, 1, 2} × dec_noise {0.5, 1, 2} | 9 | 0/9 | 9/9 | 2/9 | 0/9 |
| + reward-driven κ, β {1/120, 1/300}, breadth_cost {1, 2} × dec_noise {1, 2} | 8 | 0/8 | 8/8 | 0/8 | 0/8 |
| perc_noise {0.5, 1, 1.5} × dec_noise {1, 2} | 6 | 0/6 | 6/6 | 1/6 | 0/6 |

- **S1 never passes.** In the stable world the canalized agent regulates
  worse than the wide one: phase-3 MAE differences +9 to +52, CIs mostly
  above 0. The one exception, breadth cost 2 (−11.7 [−84, +37]), is not
  reliable.
- **No inverted U, and G2′ becomes unreliable** once costs are added. The
  trap rate barely moves with dose.
- **G1′ always holds:** narrowing traps agents.

### Diagnosis

- **The reactive controller gets nothing from a canal.** It steers by what
  it currently sees. A narrower view only ever hides food. The canal carries
  no information the controller can use to reach food, such as a remembered
  route to a patch.
- **Costs on breadth just tax every arm.** They cannot make narrowing
  worthwhile.
- **Multiplicative perceptual noise is a weak cost.** Rays that show no food
  stay at zero, so noise only reorders rays that already show food.
- **In this formulation the prior is pure cost.** It appears only as phantom
  food, never as useful guidance.
- **The theory needs priors that pay in a stable world:** remembered routes
  to known food are efficient. They become costly only when the world
  changes. That trade-off requires an agent that navigates with its memory.

### Implication

- The imposed-narrowing model with a reactive controller can show **trapping**
  (G1′) and, without costs, **release** (G2′). It cannot show
  **adaptive canalization** or a **dose optimum**.
- Two ways forward:
  - **Route memory in the controller:** when hungry and seeing no food,
    follow worn trails toward remembered patch locations. Narrowing then
    saves perceptual cost while memory guides the agent. This is a partial
    step toward the cognitive map.
  - **The cognitive-map variant** (`COGNITIVE_MAP_BOOKMARK.md`). Priors guide
    navigation (a benefit) and mislead when the world changes (a cost).
    Decanalization by precision scaling then has a natural optimum: too
    little leaves the agent stuck; too much makes it ignore a map that was
    still useful.
- **Stage B (PPO)** might learn to use canals through recurrent memory.
  Nothing here shows that it would.
