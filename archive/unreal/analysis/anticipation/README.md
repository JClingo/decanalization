# Anticipation repair — 2026-10-25

The compiled change is in `Content/Agent/BP_Agent.uasset`. It was made in
`/Users/jingo/Work/PhD/Dissertation Project/decanalization` from the source
fingerprint in `change_manifest.json`. Unreal Engine 5.5.4 compiled the graph
with no Blueprint errors. No C++ or additional runtime plugin is required.

## Behavior

- `GetAnticipatedResourceYield` adds every resource hit's `HomeostaticImpact`
  to its running total. The old assignment returned only the last resource.
  The disconnected distance/passive-loss term is kept disconnected: travel
  loss belongs in the traversal calculation and must not be charged per pellet.
- `GetAnticipatedTraversalCosts` measures the segment inside each hit tile's
  world component bounds. It clips at the actual ray endpoints, replacing the
  tile-diagonal and half-endpoint approximation (including the disconnected
  last-index comparison). This supports partial tiles, reversed and angled
  rays, gaps, and a horizon contained inside one tile.
- Tile cost equals time inside the tile times its predicted cost rate.
  The starting tile uses `BaseHomeostaticImpact / (OverlapCount + 1)` because
  its overlap was already recorded; future tiles use `OverlapCount + 2` to
  account for entering once. The denominator is bounded below by one.
- Passive loss is counted once across the whole horizon, even when no tile
  is hit. `PassiveLossRate` is per homeostatic update, so its per-second rate
  is `PassiveLossRate / (PeriodTickLength * DeltaT)`. The old final `0.2`
  multiplier no longer scales the loss. Totals reset for each call; invalid
  zero speed or zero delta returns zero rather than dividing by zero.
- The existing ordinary net remains `gain - loss`. The decanalized event
  continues to supply zero loss, matching its existing experimental condition.

Writing `v` for the supplied `MaxSpeed`, the ordinary prediction is:

```text
gain = sum(pellet.HomeostaticImpact)
loss = sum(tile_segment_length / v * predicted_tile_cost_rate)
       + HorizonLength / v * PassiveLossRate / (PeriodTickLength * DeltaT)
net  = gain - loss
```

The estimate assumes straight travel at the supplied speed and the existing
axis-aligned cube tiles. It does not plan turns or model pellet respawns,
other agents, or walls beyond the supplied ray hit set. A capsule can overlap
an adjacent tile before its center reaches the boundary; this small boundary
interval is still approximated by the centerline ray.

## Verification

`validation.json` records **23 passing checks** executed against the real
Blueprint in Unreal, using real resource/tile collision hits and an independent
segment/box calculation. They cover pellet sums and ordering, partial and
angled tiles, reverse and parallel rays, different canalization counts,
speed, simulation timestep, homeostatic update period, empty rays, zero
inputs, repeated calls, and the net calculation. The largest absolute
numerical discrepancy was below `2.9e-7`.

One check uses three pellets worth 3, 7 and 11: gain is **21**. A separate
2,500-unit, 2.5-second ray through three tiles predicts **14.5** loss with the
specified test settings, giving **6.5** net when combined with those pellets.

The tests temporarily enable variable instance editing in memory. Those flags
were not saved; the installed asset retains the original property permissions.
GameMode, Tile and Resource assets were verified unchanged. Validation uses a
disposable project and never runs training or writes evaluation results.

## Evidence and reproduction

- `patch_anticipation.py WORKDIR` stages graph edits from the audited original
  `BP_Agent.json` and `BP_GameMode.json` UAssetAPI exports. It imports the
  existing helpers from `analysis/timing` and refuses an already migrated
  export. Use the existing `analysis/timing/AssetAudit` adapter to export/import.
- `compile_anticipation.py` compiles and saves only BP_Agent. Run it with
  Unreal's Python commandlet; a success message alone is insufficient—also
  require no Blueprint/Python errors in the compiler log.
- Run `launch_validation.py` through `-ExecCmds="py <absolute path>"` in a
  disposable full editor project with rendering enabled. Place an empty
  `ANTICIPATION_TEST_COPY` marker in that project first. The validation script
  refuses projects without that marker, writes its report to
  `Saved/AnticipationFix/runtime_validation.json`, and quits the test editor.
- Original asset, source/compiled exports, successful compiler/test logs and
  install receipt are under `Saved/AnticipationFix/2026-10-25/`.

These corrections change policy inputs and their scale. Existing frozen
results retain their original implementation; evaluate the corrected model
with an appropriately validated or retrained policy before comparing new runs.
