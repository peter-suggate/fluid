# Lane D progress: the collar, band depth, shape tolerance, body tiles (items 6–11)

Plan: `docs/plans/paddleboats-60fps-2026-11-10.md`, Phase 2 items 6–8 and
Phase 3 items 9–11. Nothing here is committed. The defaults are unchanged
until the GPU gate below decides.

## Decisions

### Two new method values (item 6)

Both are `update:"solver"` selects in `uniform-narrow-band-method.ts`. They
are parsed in `uniform-near-settled.ts` by
`lib/methods/uniform/uniform-narrow-band-collar.ts`, and **unknown values
throw** (unlike the older near-settled keys, which fall back to the default).

- **`collarSpread`**: `"neighbours"` (default, today's behaviour) or `"liquid"`.
  Under `"liquid"`, `activitySpread` reads a tile's neighbours only where the
  head census found the tile wet: `UNIFORM_STAGE_IMPORTANCE.wet`, bit 29 of
  importance word 1, which `activity()` copies into `NB_ACTIVITY_SCORES`
  before the spread. A dry tile keeps its own heat and nothing else. The same
  value makes nbFinish's arrival cap unconditional for grid-supported
  samples, `heat=min(heat,max(arrived,1.0))`. Today the cap applies only
  where the arrival tile has target heat.
- **`collarDensity`**: `"band"` (default) or `"deep"`. Under `"deep"`, a
  non-source cell whose tile's target heat is in (0, 1] (the collar) seeds 4
  samples of 2 units on the even sites 0, 3, 5, 6 at every depth. A hot tile
  keeps the band's density. `nbCollarSeedTarget(heat, depth, source)` feeds
  three places: seedCell (count, sites, mass), seedCells (count check and the
  deep flag) and the merge split in nbFinish (its `seeded` target, read with
  `nbMoveHeat(q)`). The peer's split logic itself is untouched. `"deep"`
  requires `crowdedCells "merge"`, the default; with `"retire"` both the
  parser and the constructor throw.

Every default emits the WGSL text compiled before this change, byte for byte
(see Verified).

### Why the collar's purposes still hold (by construction)

1. **Transfer support** (`nbTransferBlend`: θ·mix(min(1, mass/expected), 1, θs)).
   - *liquid*: every wet tile's target is exactly today's, so every cell with
     liquid that today's collar seeded is still seeded at the same heat.
   - *liquid*, dry tiles: a dry tile has no non-air owner, so every vertex
     has phi > 0 and no seed site passes seedSite's `distance>0.0` test. A dry
     collar tile never held a seed. Its target only did two things: it
     refreshed passing samples to heat 1, and it capped arrivals.
   - *deep*: a collar cell's mass is still 8 units (4 × 2), and nbExpectedMass
     is mass-based under merge, so coverage is unchanged. The even sites put
     2 samples in each half-cell next to every face, so every face of a collar
     cell has support within half a cell, as in today's deep band cells.
2. **No erosion of unseeded liquid.**
   - Erasure needs surface heat > 1 from the nearest sample. A tile's target
     exceeds 1 only through its own heat, because a neighbour contributes at
     most ½·2.
   - *liquid*: with the cap unconditional, a grid-supported sample ends a step
     with heat > 1 only inside a tile whose own heat is > 1. Those tiles are
     seeded, and so are their wet neighbours. This holds however far the
     sample travelled. Today a sample that crosses the collar into a tile with
     no target keeps heat 2; `"liquid"` closes that gap. Ballistic spray keeps
     its heat, as today.
   - *deep*: collar heat stays ≤ 1, so no erasure authority is added. The
     union's mass bound counts samples, `min(nbBinCount, 8)`, so 4 samples a
     cell halve what the collar can add. Through `min(bulk, …)` that bound
     can only add less liquid; it can never remove any.
   - *deep*, splits: a deep seed never splits where it was seeded (2·4 = 8,
     not > 8). In a hot shallow cell (target 8) it splits back to band density.
3. **Costs moved, not hidden.**
   - Under `"liquid"`, samples in dry tiles are no longer refreshed to heat 1,
     so they cool and retire. Exceptions: `heatExpiry "held"` keeps shallow
     samples in thin tiles, and spray is exempt.
   - A tile that turns wet during a step joins the collar one census later.
     When it heats it is seeded through activation (`activationSeeding`).
   - Both effects are volume risks, which is why the gate watches drift on
     three scenes.

### Fractional band depth (item 7): no code change needed

`particleBandDepth` already accepts steps of 0.5 in [2, 4]:
- the param has min 2, max 4, step 0.5;
- `uniformNearSettled` clamps it to that range;
- the constructor checks it;
- `NB_BAND_DEPTH` is emitted with `toFixed(2)`.

Every reader:
- **Retire** (`depth < -NB_BAND_DEPTH`, the move, via `nbMoveDepth` /
  `particleDepthMoveIn`) and the **site test** (`distance < -NB_BAND_DEPTH`,
  seedSite, via `particleDepth`) use the same threshold on the same depth, so
  no seed is retired where it lands.
- The **cell gate** (`d < -NB_BAND_DEPTH-0.5`, seedCell and seedCells) is exact
  for a level surface. For a tilted one it can skip sites up to
  (√3/2 − ½) cells above −B. That predates this change, and it can only make
  the band thinner.
- **Capacity**: `webgpu-uniform-reference.ts:1554` sizes it with
  `Math.ceil(depth)`. That over-estimates, which is safe; the file is Lane R's
  and is not changed.
- **Transfer support** (`nbExpectedMass`) does not read B.

At B = 2.5 on a cell-aligned surface, the cell layer centred at −2.5 is half
inside the band. That layer is listed by seedCells every step and fills over
frames as the per-frame salted jitter puts sites above −B. This costs list
work only. Integer B has the same effect, with an empty layer.

### Shape tolerance (item 8): no code change

It is set per scene in `lib/core/scenes.ts` through
`methodProfile.overrides`, for example
`{ rigidCoupling: "on", detailShapeTolerance: 1 }` for paddleboats. Its range
is [0, 2] h in steps of 0.05. For a single run, pass
`--values={"detailShapeTolerance":1}`. In the app the Surface Tune dial drives
it, and a value off the dial's curve shows as "Custom". The gate's `shape1` arm
is the one confirming run.

### Body tiles by cause (items 9–11): `tools/probe-census-reasons.ts`

New output:
- **`solidBody.<boat>.<hull|blade|other>.<cut|inside|motion|ring>.<wet|dry|closed>`**
  splits each solid tile. Each tile is tested against the nearest body's box
  SDF at its pose, sampled on a 5³ lattice per tile. `motion` means the tile
  is body-marked in the contact record but not reached by the shape (the
  motion bound alone). `ring` means neither marked nor reached. Tiles with no
  body within 8 h are counted as `solidBody.static.*`.
- **`solidBody.<part>.<motion|ring>.gap.*`** gives how far the shape is from
  those tiles.
- **`join.body.<marked|unmarked>[.cut|.within4h][.dry]`** counts join tiles
  that are also body-marked or cut.
- **`yHistogram.join`** and **`yHistogram.particleHeat0.<collar|carried>`**
  give tile-row histograms. The paddleboats water is 24 cells deep, so the
  rest surface lies on the face y = 6.

My hypothesis for the dry join tiles: transferRefine's join box grows by the
travel plus padding from a crossing cell in a tile's top layer, so any upward
travel joins the tile above. That would put a join peak at y = 6. If the peak
follows the collar instead, `"liquid"` removes it.

Options:
- `--report-at=12,300` reports after each listed step (the plan's step-12 and
  step-300 pair).
- `--summary` prints one `SUMMARY {label, scene, step, fine, particles,
  particleTiles, bandTiles, drift}` line per report.
- `--label=NAME` tags that line.
- `--values-file=PATH` reads the values from a file.

Report steps now request the volume probe, so drift is measured on those
steps.

## Verified, and how (CPU unless stated)

- `tests/uniform-narrow-band-collar.test.ts`, 7 tests, all pass:
  - **Parsers**: absent values give the defaults, unknown values throw (also
    through `uniformNearSettled`), deep + retire throws, and the params are
    registered.
  - **(iii) Defaults**:
    - the default fragments are byte-equal to the old literal text;
    - `narrowBandActivityWGSL === narrowBandActivityWGSLFor("neighbours")`;
    - the neighbours model equals an independent implementation of today's rule.
  - **(i) and (ii)**:
    - on 20 random grids, every wet tile's `"liquid"` target equals today's;
    - a dry tile holds only its own heat;
    - every wet neighbour of a heated tile gets at least half its heat;
    - over 40k random arrivals, heat > 1 survives only in a tile with own heat > 1.
  - **(iv)**:
    - for B from 2 to 4 in steps of 0.25–0.5, on level and random tilted
      planes, in both densities and both heats, seed counts are monotone in B;
    - every seed lies in [−B, 0], and none is retired where it lands;
    - the gate drops nothing on level planes.
  - **Deep**:
    - mass is 8 per cell;
    - the even sites put 2 next to every face;
    - the split rule holds;
    - the deep column has the same mass as the band column with fewer samples.
- **WGSL**: `tools/capture-uniform-wgsl.mts`, which now takes `VALUES=<json>`,
  ran on sparse-cm12-long-dam-break.
  - With the defaults, the dumped modules contain today's text at every
    changed site.
  - With `{"collarSpread":"liquid","collarDensity":"deep"}`, only the 3
    NB-FLIP modules differ.
  - naga validates all 4 affected modules in both configurations. A
    pre-existing naga-only "missing return after loop" quirk in
    `nbBoundaryCubicHit` and `umUnitTrace` was patched in the dump for this
    check only.
- **GPU** (1 run, `fluid-gpu-run.sh check`):
  `tests/uniform-narrow-band-collar-gpu.test.ts` runs the solver's own
  `activitySpread` entry under both spreads on a 7×6×5 grid with absent, dry
  and hot tiles. Targets and NEW flags match the CPU model exactly; 49 dry,
  cold collar tiles were exercised. Pass.
- `npm run check:types`: no errors in Lane D's files. The remaining errors are
  in peers' in-flight files:
  - `tests/rigid-motor-gpu.test.ts`
  - `tests/uniform-pressure-arena-launches.test.ts`
  - `tests/paddleboat-contacts.test.ts`

## Needs the GPU gate (not run)

```
node --import tsx tools/gate-paddleboats-collar.ts            # all arms, all scenes
node --import tsx tools/gate-paddleboats-collar.ts --dry-run  # the commands only
```

The arms are `liquid`, `deep`, `both`, `band25` (particleBandDepth 2.5) and
`shape1` (detailShapeTolerance 1). Each arm runs against a base run just
before it, on three scenes:
- paddleboats: 300 steps, bodies driven;
- letters: 120 steps at 1/60;
- channel dam: 60 steps, `--worker=dam`.

Each run goes through `tools/gpu/fluid-gpu-run.sh check`. The gate prints one
table: h tiles, particles, band tiles, drift, and the change against the
paired base. It stops at the first rc 76/77/78. There are 30 runs in all;
`--arms=` and `--scenes=` narrow it.

Item 9's body split on its own (one run):
```
tools/gpu/fluid-gpu-run.sh check node --import tsx tools/probe-census-reasons.ts --scene=paddleboats --steps=300 --report-at=12,300
```

What would decide each value:
- **`collarSpread "liquid"`**: particles and h tiles fall, and the drift on
  letters and the dam stays within the base's.
- **`collarDensity "deep"`**: collar particles fall by about half, with the
  same drift.
- **`band25`**: the same test, on letters.

Make a default change only after that gate.
