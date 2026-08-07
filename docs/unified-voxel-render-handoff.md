# Unified voxel render — handoff

Date: 2026-08-07. Nothing here is committed; all work is in the working tree.

## What landed, part 1: the primary reads voxels only

**The render path reads voxels only.** No analytic object, no terrain heightfield,
no owner. The `sceneGeometry` binding was removed from the visibility pipeline
outright — not branched around, absent.

- The per-voxel identity word is now `material id (low 16) | oct8 normal (high 16)`.
  The normal is baked at voxelization, where evaluating the field is free.
  Per-voxel bytes are unchanged: the normal occupies the sixteen bits the owner id
  used to hold.
- The three-tier shading normal ladder (owner algebra → terrain heightfield →
  certified trilinear → voxel face) is gone, replaced by one unpack plus the
  retained right-angle test against the entered face.
- The brick DDA no longer resolves an owner, so the run tracker, the sphere-trace
  promotion, the upgrade budget and the owner-suppression tests went with it.

**Measured, depth 3, 800x460, canonical-parametric + split, same lane and same
metric (`worker.log medianFrame_ms`):**

| | median frame |
|---|---|
| before (`artifacts/xctrace-hero-garden-hose-depth3`) | 25.86 ms |
| after (`artifacts/xctrace-hero-garden-hose-depth3-unified`) | 14.99 ms |

**1.73x.** Not interleaved — two captures on different days — but same lane, same
config, same metric.

## What landed, part 2: the analytic residue is gone from lighting too

The three "what is next" items are done. The shape of the work was not what the
plan expected, and the correction is the most useful thing in this document.

### The shadow tier had no voxels in it

The plan read "delete `traceScenePrimitives` from every visibility ray" as a
deletion. It was not. `traceLeafPayloadVisibility` — the leaf DDA every shadow,
AO and GI ray runs — **stepped cells and never tested one**. It resolved an owner,
dropped it, and returned MISS. Every shadow in the frame came from the analytic
candidate BVH beside it, and deleting that walk on its own would have deleted the
shadows.

That is also the answer to the question the old probe measurement posed.
`visibilityAnalyticDisabledProbe` measured *slower* than either real arm
(145.0 vs 137–139 ms) because it removed the occluders rather than the work, and
unshadowed rays then ran to full `tMax`. It was not an anomaly to explain away;
it was the shape of the bug.

So the deletion is now in the right order:

1. **`traceLeafPayloadVisibility` tests solidity** — the same `cellSolidGate` the
   primary uses, `material != 0`, one gate, no owner — and returns a HIT at
   `entry`, the ray's own distance to the face it crossed to reach the cell.
2. **`traceScenePrimitives` is off the visibility path**, and the three arms that
   ordered it (`unboundedAnalyticVisibility`, `visibilityAnalyticDisabledProbe`,
   and the bounded default) collapse to one tier with no ordering question left.
   `FLUID_SVO_DRY_SMOKE_VISIBILITY_ANALYTIC` now throws rather than silently
   selecting an arm nobody compiles.
3. **`traceTerrain` is deleted**, along with `traceTerrainHeightfield`, the
   `DryTerrainGrid` decoder, `terrainHeightAt`/`terrainGridHeightAt`/
   `terrainCeiling`/`terrainNormalAt`, `DRY_GBUFFER_FIELD_TERRAIN`, and its five
   call sites. It had already been reduced to `return missHit()`, so this is dead
   WGSL leaving the module rather than a behaviour change.
   `traceStaticSolidScene` is gone with it: its whole content was
   `traceStatic` plus the second ground, so call sites now say `traceStatic`.
   `intersectSvoTerrainHeightfield` on the TypeScript side **stays** — editor
   hover picks against the authored heightfield, which is a question about the
   document rather than about the frame.
4. **Owner suppression is narrowed rather than deleted.** The plan called
   `dryVisibilityIgnoredOwner` dead. It is not: it is how a rigid body avoids
   shadowing itself, and rigid bodies are the one producer of a non-`NONE` owner
   left. What *was* dead — and worse than dead — is its term inside
   `dryOpaqueOwnerSuppressed`, which compared an authored **record** owner id
   against a **body index** in 0..11. Two different namespaces that compared
   equal whenever the numbers collided. The variable is now
   `dryVisibilityIgnoredBody`, read only by the rigid loop, and record
   suppression names only the editor preview and thick-glass takeover.
   `nearestBodyIgnoring` is untouched, as planned.

### Measured

`hero-garden-hose`, depth 0, 800x460, canonical-parametric + split, brick
occupancy `bounds`, cone visibility, median submit-to-fence over six serialized
encodes:

| | median frame | settled hash |
|---|---|---|
| before the purge | 13.25 ms | `0x8553a29b` |
| after | 8.96 ms | `0x8553a29b` |

**The image is byte-identical and the frame is ~1.4x cheaper.** Three later runs
of the same configuration (the `quality` rung below) medianed 9.03 / 11.75 / 9.17,
so read this as ≈1.4x against a lane whose run-to-run drift reached 30% once.
There is one before-arm run, because the arms differ by a compile-time shader
change and cannot be interleaved in one process; what makes it credible is that
the before run's six samples (10.81–18.16) do not overlap the after run's
(8.58–9.09).

**Why an unchanged image gets faster is the interesting part, and it is not
proven.** The smoke lane runs cone visibility with silhouette refinement off, so
the exact path this change rewrites is never *executed* — which is exactly why the
hash did not move. But `dryLightVisibility` selects between cone and exact on a
**uniform**, not a compile flag, so the candidate-BVH walk and its 18-entry
`array<u32>` stack were compiled into the deferred-lighting kernel regardless and
priced into its register and scratch allocation. Deleting code from a branch the
frame never takes is the only mechanism that fits "byte-identical, 30% cheaper".
It is a hypothesis consistent with the measurement, not a profile — nobody has
read the register allocation.

### The exact rung now draws the scene's own shadows

`FLUID_SVO_DRY_SMOKE_CONE_TRACING=cones|exact|off` is new on the smoke lane, and
`exact` is the point of it. `artifacts/unified-voxel/depth0-cones.png` and
`depth0-exact.png` are the same frame under each: cone visibility is soft and
washed, exact is crisp, and the hard shadows fall from the pond bowl, the coping
and the fence posts — geometry with no authored record behind it, which is
precisely what the old exact arm could not shadow. Twenty-one of twenty-one
checks pass in both.

One CPU test moved from red to green on its own:
`tests/webgpu-svo-dry-scene-lighting.test.ts` "bounded hard-shadow visibility
covers opaque sources and transmissive panes". It had been asserting the contract
the shadow walk was silently failing.

## What landed, part 3: the ladder, re-measured

`SVO_RENDER_TUNING_PRESETS` was a tuning-only map with three rungs, matched in the
UI by tuning key. It is now the projection of `SVO_RENDER_QUALITY_PRESETS`, a
ladder of **pairs** — sliders plus how visibility is answered — because the two
halves are not independent: under `cones` the visibility budgets are barely
reached, and under `exact` they are the whole cost of a shadow. The PROFILE strip
sets both, so a named rung can no longer be half-applied, and the mode keeps its
own buttons under Lighting.

`reference` is the new top rung: `quality`'s sliders with the cone tier off.

**Re-measured 2026-08-07** on `hero-garden-hose` at depth 0, each rung rendered at
its own `resolutionScale` of an 800x460 request
(`FLUID_SVO_DRY_SMOKE_PRESET=<rung>`), median submit-to-fence over six serialized
encodes, three runs each for the cone rungs:

| rung | pixels | medians | median of medians |
|---|---|---|---|
| `performance` | 400x230 | 6.76 / 6.84 / 6.94 | **6.84 ms** |
| `balanced` | 576x331 | 6.96 / 8.79 / 7.07 | **7.07 ms** |
| `quality` | 800x460 | 9.03 / 11.75 / 9.17 | **9.17 ms** |
| `reference` | 800x460 | 17.05 / 17.33 | **17.2 ms** |

**The cheap end of the ladder has collapsed into the fixed-overhead floor.**
`performance` is 2.07x fewer pixels than `balanced`, a 4x4 cone prepass against
2x2, and nine reduced knobs — and it is worth **3.4%**. The ordering is real (its
slowest run, 6.94, is under `balanced`'s fastest, 6.96) and the size of it is not
a rung. Fitting the marginal cost between `balanced` and `quality` gives
~11.9 ns/pixel and a **fixed floor near 4.8 ms of a 9.2 ms frame**; `performance`
then lands ~1 ms *above* what that line predicts, so its own reductions are
costing about what they save. This agrees with the separately measured
few-pixels floor (`docs/svo-fixed-overhead-handoff.md`) and it means the next
useful rung is not a smaller frame.

`reference` costs **1.9x** `quality`, which is the honest price of one bounded
hierarchy ray per shadow and AO sample.

## What landed, part 4: the publication gate

`SVO_DRY_SCENE_REQUIRED_VALID_FIELDS` is now `topology | materialOwner` — exactly
the two lanes a ray reads: the tree it descends, and the per-voxel identity word
that decides both solidity and shading. `sceneGeometry` is out. Nothing in the
render path reads the distance lane any more; it is a derived-lighting input, and
requiring it made a lagging distance publication black out a frame that had
everything it consumed.

Today's producer finalizes all three fields in one pass
(`OCTREE_SPARSE_BRICK_SCENE_VALID_FIELDS`), so this changes no frame that renders
now. It changes which *future* publication is legal.

## What was tried and rejected

The `banded` leaf payload was made compressive again (palette interns the material
half; normals in their own lane). It works, all three payload arms render
`hero-garden-hose` to `0x8553a29b`, and it is **not the default and should stay
that way**. See [[occupancy-mask-is-not-the-win]] in memory for the numbers. Short
version:

- the 1-bit occupancy mask predicate is worth **+0.64%** — the identity lane is
  2 KB a leaf and already cache-resident, so there was no traffic to save;
- full `banded` costs **+3.5%**, and that cost is identity *indirection* (header →
  palette → normal lane) rather than the predicate;
- the arena does compress **3.13x** (52.1 → 16.7 MB), but that is not realized on
  device because the banded arm still carries both dense staging lanes.

Two things it found on the way that are worth keeping: the old whole-word palette
silently clamped to a **wrong identity** on 14 leaves (0.11%), and the banded arena
allocator had **no reader at all**, so an undersized reservation published air.
Both are fixed; the allocator now has a smoke oracle.

## What is next

**1. Profile the register-pressure claim, or replace it.** The 1.4x above is
measured; its mechanism is not. A frame capture that reads occupancy for the
deferred-lighting kernel before and after would either confirm that a never-taken
branch was pricing the whole pass, or find the real term. That answer generalises
much further than this one deletion: the same uniform-selected structure exists
for AO, GI and the prepass shortcuts.

**2. The fixed floor is the ladder's cheap end.** `performance` buys 3.4%, so
there is no useful rung below `balanced` until the ~4.8 ms floor moves.
`docs/svo-fixed-overhead-handoff.md` has the anatomy — worst-pixel march latency,
O(1) in pixels and records, sky at 0.85 ms. Until that is spent, the profile strip
has three rungs that matter, not four.

**3. Re-measure the ladder at depth 3.** Every number above is depth 0. The
frame that matters is depth 3, where the primary is 73% of the frame at 9%
occupancy and the per-pixel term is much larger relative to the floor — which is
exactly the regime where `performance` should start earning its rung again. The
lane exists (`FLUID_SVO_DRY_SMOKE_REFINEMENT=3`), it just costs a document rebuild
per run.

**4. `reference` is a rung, not yet a reference.** It renders correct hard
shadows at the voxel lattice, and at depth 0 that lattice is 6.25 mm, so the
shadow edges are visibly stepped. Whether that is the rung people want above
cones, or whether it wants a soft-shadow variant (several jittered exact rays per
sample, which the budgets already allow), is an art-direction question the shape
lab is the right place to answer.

## Known issues, unfixed

- Visible in `artifacts/unified-voxel/depth3.png`: horizontal banding on the pond
  bowl's inner wall (likely the pre-existing snapped-sample-point issue), stepped
  leaf discs, and 236/368,000 isolated sky pixels (0.064%) that are sub-pixel
  traversal holes.
- Voxel hits always report `SVO_FEATURE_SMOOTH`; the analytic normal's hard-feature
  classification had nowhere to live in sixteen bits.
- `buildSvoTerrainMaterial` still packs into a now-unread `DryParams` vec4f.
- The sculpted-terrain region of `SVO_DRY_SCENE_ARENA_LAYOUT` is still written by
  the producer and no longer read by any shader. Its offsets stay because every
  later region is placed after them, so reclaiming the bytes is an ABI change
  rather than a deletion.
- `sampleSparseScenePrimitiveCell` mirrors only the material half and has zero
  callers.
- The exact tier's shadow bias is `shadowBiasCells = 0.02`, and the bias direction
  is the *shading* normal rather than the entered face. No acne is visible on the
  hero at depth 0, but the two can differ by up to a right angle, so a finer
  lattice is where that would first show.

## Traps

- **The benchmark lane is not the renderer's default.**
  `benchmark-svo-dry-frame-gpu.ts:270` defaults `brickOccupancyMode` to `off`; the
  renderer defaults to `bounds`. Absolute milliseconds from the profiling lane are
  for a `brick-off` build.
- **A rung timed at another rung's pixel count is not a cost point.**
  `FLUID_SVO_DRY_SMOKE_PRESET` applies `resolutionScale` to the request for
  exactly this reason. Unset, the lane keeps its historical behaviour: balanced
  sliders at whatever `_WIDTH`/`_HEIGHT` asked for.
- **This lane's run-to-run drift reached 30%.** Two of the twelve rung runs
  medianed well above their siblings (`balanced` 8.79 against 6.96/7.07, `quality`
  11.75 against 9.03/9.17). Single runs 3% apart have measured nothing; the
  `performance`/`balanced` ordering above survives only because the runs were
  interleaved and the sample sets are disjoint.
- The CPU suite is red independently of this work: **123 failures** after these
  changes against **127** before them, on the same working tree, with **no new
  failing names**. Diff failing *names*, never counts — and note that
  `tests/octree-balance-elision.test.ts` flips which of two tests it fails
  between full-suite runs (123 or 124 depending on the run), so a one-name diff
  there is noise.
- Two adjacent GPU lanes are red for reasons that predate this work and do not
  touch the render path: `test:webgpu:garden-hose` fails
  `inflow-retains-source-volume` (a solver volume diagnostic),
  `test:webgpu:voxel-scene` fails a `tall-cell` parity hook whose second method
  is absent, and `test:webgpu:svo-live-voxelization` fails
  `pyramid-survives-edit-script` and `crown-carries-the-field` — the node-mip
  edit lifecycle and the canopy distance lane. None were re-measured before the
  change, so treat that attribution as reasoned rather than bisected.
