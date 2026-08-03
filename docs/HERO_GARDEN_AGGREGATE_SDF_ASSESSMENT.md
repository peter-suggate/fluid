# The hero scene's blockers, and whether an aggregate SDF can carry it

Assessment date 2026-08-03, working tree at `0d681ec` + uncommitted changes,
branch `perf/structured-cutover`.

Companion to `HERO_GARDEN_HOSE_SCENE_PLAN.md`. That document is the plan; this
one is what the code and the GPU say about it now that most of §5 phases 0–2
have actually been built. Several of its load-bearing estimates are wrong, in
the direction that makes the hard engine change *more* necessary rather than
less.

Reference image: `output/imagegen/garden-pond-hose-fill-simplified.png`.

---

## 1. What renders today

`hero-garden-hose` was rendered headless at 1280 × 720 through the production
dry-scene path:

```
FLUID_SVO_DRY_FRAME_SCENE=hero-garden-hose \
FLUID_SVO_DRY_FRAME_WIDTH=1280 FLUID_SVO_DRY_FRAME_HEIGHT=720 \
npm run benchmark:svo-silhouette-refinement
```

It works, and it is considerably further along than the plan's §6b implies —
that section reads as though the dry set is unbuilt, while five untracked
generator modules (`bonsai.ts`, `stone-set.ts`, `rosette.ts`, `swept-coping.ts`,
`hero-layout.ts`, ~4 300 lines) already place it.

| measure | value |
|---|---|
| frame, median over 16 encodes | **19.88 ms** (p95 21.52) |
| traversal | `canonical-parametric`, split shading, cone scale 0.5 |
| environment | `garden`; shadows, cone AO and GI all **on** |
| node-mip pyramid | ready, 232 pages |
| tetrahedral radiance | ready, 232 pages, **0 black** |
| structural grid | 72 × 24 × 48, brick 8, depth 4, 249 344 voxels |
| **primitives published** | **2 227** |
| rigid bodies | **0** |

Present in frame: pond vessel, coping, pebble beds, four mushroom boulders,
five stepping discs, bonsai, hose with ferrule, air-plant stand-ins, ground.
Absent: water (the scene opens dry by design, `systems.fluid: false`).

**Perf is not the blocker on the dry path.** Twenty milliseconds at 720p with
the full lighting stack resident is a comfortable starting point.

Two corrections to the plan while we are here:

- **§2 said stepping discs would be static rigid cylinders** so water parts
  around them. `scene.rigidBodies = []` (`lib/hero-garden-scene.ts:471`) — they
  are scenery, and water will pass straight through them the moment
  `water: true` is set.
- **§6b's "7.5 mm overruns a device limit" is scoped wrong.** The 65 536-vs-65 535
  one-axis dispatch (`lib/webgpu-octree-fine-levelset-topology.ts:725`) is on the
  *solver* path, reached only when `systems.fluid === true`. **The dry render
  path reaches 7.5 mm and 3 mm without touching it** — 16.6 ms and 39.5 ms
  respectively at 800 × 460 on the traced arm (`SVO_FINE_VOXEL_CAPACITY.md` §2,
  captured at ~1 004 primitives, so treat both as floors). The first render wall
  is 2.5 mm, on the payload arena. Lattice is not what stands between us and the
  reference's look.

---

## 2. The binding constraint is the primitive budget, and the plan under-counted it 4.5×

`SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES = 4_096` (`lib/svo-primitive-candidates.ts:19`).
Plan §3 G1 counted "≈ **500**" and concluded "the budget is not the problem for
anything except the canopy. Good news, and it means pebbles can stay honest
individual ellipsoids."

The scene publishes **2 227** today, and two objects hold 90 % of it:

| object | primitives | share |
|---|---:|---:|
| bonsai | 1 339 | 60 % |
| swept coping | 672 | 30 % |
| pebble beds (at 18 % of planned population) | 158 | 7 % |
| boulders, discs, hose, plants, shell | 58 | 3 % |

`layoutStatistics(heroGardenLayout())` budgets **3 106** detailed leaves for the
finished set — and does not count the coping's 672 at all. 3 106 + 672 = **3 778
of 4 096**, i.e. 92 % consumed by the authored set *before* a single
reference-density floret exists. Crossing 4 096 is a cliff, not a slope:
`canConsumeSparseVoxelPrimitiveCandidates` returns false
(`lib/webgpu-svo-dry-scene.ts:1058`), the candidate BVH is never built, and the
scene stops drawing.

### The lower, silent ceiling

`OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK = 64`
(`lib/webgpu-octree-sparse-bricks.ts:215`) is a **density per brick**, not a
total, and overflow is a **silent drop** — surplus primitives are simply not
written into that brick's voxels, so they are absent from the opacity pyramid
and the radiance atlas while still drawing in primary visibility.

`SVO_FINE_VOXEL_CAPACITY.md` §4 measured the hero scene already exceeding it at
25 mm — busiest brick 70, three bricks over — **at ~1 004 primitives**. The
scene now publishes 2 227. Contact-packed graded pebble bands are precisely the
shape that trips it, and a 1 500-leaf explicit canopy over the bonsai's volume
averages ~83 leaves per brick, which **cannot be voxelised at that lattice at
all**.

The scar is already in the tree: `lib/webgpu-sparse-scene-proxies.ts:195-214`
records three pebble-band bricks overflowing, a revision never certified
complete, and "the garden rendered with a hard-edged black slab across the
container footprint."

**This makes the aggregate primitive a correctness requirement, not a fidelity
stretch.** That is a stronger argument than the one §4 makes for it, and it
should replace it.

---

## 3. The SDF verdict: smooth-union works, displacement cannot

Plan §4 specifies the new kind as *"an ellipsoidal lobe envelope, with the
surface displaced by two octaves of domain-repeated sphere packing."*

**That construction is incompatible with this engine's march**, and no amount of
tuning fixes it.

`intersectMarchedLocal` (`lib/svo-primitive-abi.ts:747-780`) and its WGSL mirror
(`:1276-1308`) are an **unsigned sphere trace**: `t += max(|d|, 1e-8)`, with
`SVO_PRIMITIVE_MARCH_ITERATIONS = 48` (`:18`) and an acceptance band
`max(1e-6, 1e-4·t)` (`:738-740`). Both constants are shared verbatim between CPU
and GPU and pinned by `tests/webgpu-svo-primitive-exact.test.ts` at 3e-4, so the
step policy is not a free parameter.

Stepping by `|d|` is safe only where the field is Lipschitz-1. A displaced
envelope `d(p) − a·noise(p)` has `|∇d| > 1`: the step overshoots and the ray
tunnels through the detail it was meant to resolve.

A polynomial smooth-min, by contrast, is a strict **lower** bound on the true
union distance. A trace using it *understeps*, which is always safe.

### The measurement

4 000 deterministic rays at hero canopy scale — 10 mm florets, 16 mm repetition
lattice, 150 mm oblate lobe — marched with the engine's exact loop, epsilon,
iteration ceiling and rotation-invariant bounding-sphere interval, scored
against a dense fixed-step ground truth (step = floret radius ÷ 40, then 40
bisections).

| construction | rays missed | steps/ray | sphere evals/ray | worst t error |
|---|---:|---:|---:|---:|
| smin over explicit jittered spheres (1 443 of them) | 0.3 % | 7.2 | — | 1.3e-2 |
| smin, domain-repeated, 27-neighbour, jitter 0.7 | 0.3 % | 8.4 | 227 | 1.9e-2 |
| smin, domain-repeated, 8-neighbour, jitter 0.7 | 0.5 % | 8.5 | 68 | 4.5e-2 |
| smin, domain-repeated, 27-neighbour, jitter 0.3 | **0.0 %** | 7.8 | 211 | 6.0e-3 |
| **smin, domain-repeated, 8-neighbour, jitter 0.3** | **0.0 %** | **7.8** | **63** | **6.0e-3** |
| **envelope + noise displacement (plan §4)** | **58.9 %** | 12.9 | — | 3.2e-1 |

Read that last row against the others. The plan's construction loses **three
rays in five**. The recommended one loses none, and converges in **8 of the 48
iterations available** — leaving the entire march contract, the CPU/WGSL parity
test and the iteration ceiling untouched.

The residual 0.3 % on the jitter-0.7 rows is the honest signature of jitter
large enough that a sphere escapes its own repetition cell; dropping jitter to
0.3 of the period removes it entirely without visibly regularising the packing.

### The recommended kind

`smoothUnionCluster` — a smooth-min over a **jittered, domain-repeated sphere
lattice**, intersected with a lobe envelope. Not an envelope with displacement
applied to it. The distinction is the whole finding.

It serves more than the canopy: §6b(3) already observed that the buttressed root
mass, the worn boulders and possibly the pebble beds are the same operation. The
current render confirms it — the trunk shows **visible capsule seams** where
limbs meet, which is exactly what smooth-union removes.

### Cost

~63 sphere-distance evaluations per covering fragment. Calibrating against the
measured brick raster (3.09 ms for ≈ 37 M cell-steps at 83 % ALU limiter,
`SVO_RASTER_PRIMARY_RESULTS.md` §7a), a canopy at 15 % of a 1500² frame lands at
**≈ 2 ms**, or ≈ 4 ms while the primitive-raster pass keeps `cullMode: "none"`
(`lib/webgpu-svo-dry-scene.ts:5314`) and pays for both faces of every box.

That last is a one-line, ~2× saving on the exact-test count of *every* primitive
in the scene, marched or analytic. The brick raster already culls
(`lib/webgpu-svo-brick-raster.ts:31`). It should be the first thing done, before
any of this.

The trade is honest: the aggregate is roughly an order of magnitude more
expensive **per pixel** than the explicit ellipsoid set it replaces. What it buys
is §2 — staying under 4 096 records and under 64 candidates per brick — plus
detail that does not have a lattice.

---

## 4. Three engine obstacles for an aggregate kind

**a. The record holds three floats of per-kind parameters.**
`SVO_PRIMITIVE_RECORD_STRIDE_BYTES = 64`, fixed for every kind; the per-kind
budget is `dimensions.xyz` (`lib/svo-primitive-abi.ts:7-9`, packer `:328-351`).
A cluster needs lobe axes, lattice period, jitter amplitude, seed, smin radius
and octave count — it does not fit.

The precedent already exists and is clean: `terrainHeightfield` carries no
geometry in its record at all, only a **word-13 offset into the shared scene
arena** (`lib/webgpu-svo-dry-scene.ts:471-484`, arena layout `:263-288`, which
already handles a variably sized tail). Copy that pattern rather than widening
the record.

**b. The node-mip pyramid cannot see sub-voxel detail.**
Coverage is derived from a **single SDF sample at each voxel centre** through a
planar law, `primitiveFraction = clamp(0.5 − d/(2·cellRadius), 0, 1)`
(`lib/webgpu-sparse-scene-proxies.ts:733-735`). That law assumes a locally planar
surface with `|∇d| = 1`; a fissured cluster violates both and under-reports. An
under-populated page is *"indistinguishable from empty space to every consumer"*
(`lib/svo-node-mip-cpu-oracle.ts:47-58`) — geometry silently stops casting
shadows and stops occluding GI.

The fix is architecturally sanctioned, because the render and voxelisation
vocabularies are **already deliberately separate**
(`lib/webgpu-sparse-scene-proxies.ts:16-20` states the split as intent):

> Report the inflated **envelope** to the voxeliser and the node-mip oracle;
> report the full cluster SDF only to the render ABI.

That keeps the pyramid conservative — over-occluding rather than under-occluding,
which is the safe direction for shadows and AO — at the cost of a slightly soft
shadow through the canopy. That is a good trade and it is what the reference
image shows anyway.

Note this obstacle is real but *narrower* than it looks: the shipped primary and
shadow paths never consult the per-voxel owner. `traceLeafPayload` says so
outright (`lib/webgpu-svo-dry-scene.ts:3443-3444`) — analytic geometry is
authoritative in `traceScenePrimitives`. Silhouettes and normals are exact
regardless. Only the derived lighting reads voxel coverage.

The plan's **G2 ("one owner per voxel") is therefore correct in its conclusion
but wrong in its mechanism**: an aggregate does not win because it dodges owner
contention in primary visibility; it wins because one record owns a coarse
region outright in the *derived* hierarchy, and because of §2.

**c. Fourteen dispatch sites, thirteen of which fail silently.**
Adding a kind touches 14 mandatory `lib/` files (the plan's §3 estimate of "~12
lib files" is close, but the shape matters more than the count). Exactly **one**
is compile-time exhaustive — `lib/scenery-expand.ts:217`, `node satisfies never`.
The others fall through to an ellipsoid test
(`lib/svo-node-mip-cpu-oracle.ts:110`), `1e20`
(`lib/webgpu-sparse-scene-proxies.ts:612`), `vec3f(-1)` and a vertex-shader bail
(`lib/webgpu-svo-dry-scene.ts:2240-2263`), `[0,0,0]`
(`lib/scenery-sway.ts:139`), or `RAY_INVALID` (`lib/svo-primitive-abi.ts:1198`).

A kind added by hand will render in one path and be invisible in another, with no
error. Note also that `SVO_PRIMITIVE_FLAGS` is **write-only** — nothing branches
on `marchedIntersection`; the real dispatch is a hardcoded
`kind == TORUS || kind == CONE` at `lib/svo-primitive-abi.ts:1150` and `:283-285`.

Also required, and genuinely new work rather than a copy of the cone's: neither
torus nor cone is in the editor place-tool catalog
(`lib/editor-scenery.ts:229-251`, `lib/stores/ui-store.ts:9-13`), and five
independent per-kind bounds formulas must be inflated consistently by the
cluster's maximum displacement or the silhouette clips.

---

## 5. The look blockers are mostly not geometry

This is the part the plan under-weights. The current render is white-on-white and
close to formless, and none of the reasons are the canopy.

| blocker | status | anchor |
|---|---|---|
| **Terrain is not wired to the procedural materials at all** — the whole vessel is one unmodulated albedo | the porcelain branch resolves to `materialFunctionId: none`, an exact passthrough | `lib/svo-material-abi.ts:148-150`; shading branch `lib/webgpu-svo-dry-scene.ts:3730-3732`; passthrough `lib/svo-procedural-material.ts:232` |
| **Caustics are worse than off — the map is write-only** | `causticTexture` appears in no bind group; the composite has no caustic slot. Flipping the flag changes **zero pixels** | flag `lib/webgpu-water-pipeline.ts:1053`; composite bindings `:864-879` |
| Caustic shader is also physically wrong for this scene | hard-coded light direction, **flat receiver plane at y = 0.006** against a sculpted basin, and no ray-bundle Jacobian, so it washes rather than forming filaments | `lib/webgpu-water-pipeline.ts:849-857` |
| **The teal cannot survive the depth** | at 0.115 m the frozen absorption gives ~5–8 % chroma; `WATER_OPTICS` is `Object.freeze`d and inlined into WGSL at build time, with no authoring hook anywhere | `lib/webgpu-lighting.ts:34-39`, `lib/webgpu-water-pipeline.ts:1038` |
| **Spray is dead code** | `WebGPUSecondaryParticleSystem` is never instantiated; the shipped solver returns `undefined`. No plunge or impact spawn source exists | `lib/webgpu-secondary-particles.ts:526`, `lib/webgpu-uniform-eulerian.ts:1552` |
| **Camera FOV is hard-coded** | `0.72` half-tangent → 104° horizontal, plus bare `.72` literals at ~17 WGSL sites in the dry shader alone. The reference reads ~50 mm (`tanHalfV ≈ 0.24`) | `lib/webgpu-camera.ts:10`; already flagged in `lib/hero-garden-scene.ts:190-203` |
| **No exposure, no filmic curve, no DOF** | one Reinhard `x/(x+1)` + gamma 2.2, applied once | `lib/webgpu-lighting.ts:73-79` |
| No subsurface or translucency for white stone | absent; `UnifiedLightingMaterial` has no transmission field | `lib/webgpu-lighting.ts:12-23` |

Two of these are close to free and should be taken first:

1. **The terrain material.** The ternary at `lib/svo-material-abi.ts:148-150`
   sends non-porcelain terrain to `gardenTerrain` and *everything else*,
   including the hero's porcelain, to `none`. It needs a third arm, not a swap.
   The reference's plaster speckle is sub-millimetre, which the `ceramic` policy
   (3.5 /m ≈ 290 mm) does not deliver — it wants the `stone` band's frequency
   (72 /m with a 4.5 mm detail octave) or a new `plaster` row
   (`lib/svo-procedural-material.ts:65-100`). Mind the file's own aliasing
   caveat at `:33-38`: there is no filtering and no distance-driven octave count.

2. **The primitive raster's `cullMode`** (§3), which is a ~2× on the pass that
   every marched primitive will land in.

The camera is the largest single change and the one that most determines whether
the framing can match at all — the scene file already documents it as the gate.
Everything else in this table can proceed independently of it.

Also worth noting from the render: **every primitive silhouette carries a fringe
of white sparkle pixels**, and the stepping discs render glassy and see-through.
Both are defects rather than missing features, and the sparkle would be worst
exactly where a dense floret canopy puts the most silhouette in frame.

---

## 6. What "no one-off hacks" implies for sequencing

The declaration language has a specific hole this scene keeps falling into.

**Plan §4 promised each generator "gets a declarative node kind in
`SceneryNode`". None did.** All six bake to flat `SceneryNode[]` at `build()`
time; the seed and parameters are discarded and hundreds of ellipsoids are
persisted instead of `{species, seed, rail}`. `tree` (`lib/scenery-graph.ts:174`)
is the sole surviving parameterized node, and `emitTree` is a hardcoded
`if (node.kind === "tree")` at `lib/scenery-expand.ts:372`. Consequences:
re-seeding is not an edit but a factory re-run that discards user edits, and a
saved hero scene stores baked geometry rather than its own description.

**Material assignment is a regex over object names**
(`lib/svo-material-abi.ts:29-48`). `lib/voxel-scenery/stone-set.ts:282` carries
the comment *"Nothing here may be named 'mushroom'"*; the README warns the regex
is load-bearing. `SceneryMaterial` (`lib/scenery-graph.ts:80`) carries a palette
reference or a literal colour and **cannot express a surface closure at all**.

There is also **no schema versioning or migration** — one literal
`schemaVersion: "1.0.0"`, `parseScene` throws on mismatch, and saved scenes are
raw JSON in `localStorage`. A more expressive declaration language changes the
schema; today there is no way to do that without breaking every saved scene.

So the sequencing that avoids adding a seventh one-off is: **fix the registries
before adding the kind.**

1. A frozen `SCENERY_GENERATORS` catalog plus a `SceneryGeneratorNode` kind.
   This converts six existing one-offs into vocabulary rather than adding to
   them, and needs no change to the expansion contract. The house style for it is
   already stated four separate times in-tree — `EDITOR_ENTITIES`,
   `RESOURCE_PLUGIN_CATALOG`, `SCENERY_GRAPHS`, `VISUALIZATION_CATALOG`:
   colocated declaration, static frozen catalog, narrow protocol, no mutable
   `register()`, plus a catalog test that proves the freeze.
2. A `surface` field on `SceneryMaterial`, with the regex demoted to a legacy
   fallback for unmigrated nodes.
3. A single table from which the TS union, the u32 kind code, the flags, the CPU
   SDF, the WGSL constant and all five bounds formulas derive — so kind #9 is one
   entry rather than fourteen hand-edits across a surface with thirteen silent
   fallthroughs.
4. **Then** add `smoothUnionCluster` through it, which is what proves the table.

Note that `RESOURCE_PLUGIN_ARCHITECTURE.md` is shipped but is *not* the right
home for any of this: it is a lifecycle and readiness architecture — what
allocates, what compiles, what blocks which UI action. Generators and primitive
kinds are content vocabulary with no lifecycle. They want **sibling** catalogs,
not entries inside that one.

---

## 7. Corrections to `HERO_GARDEN_HOSE_SCENE_PLAN.md`

| where | claim | correction |
|---|---|---|
| §3 G1 | "~500 primitives; the budget is not the problem" | **2 227 today; 3 778 planned of 4 096.** The budget is the binding constraint |
| §3 G2 | one-owner-per-voxel is "the real limit on fine detail" | true for derived lighting only; primary visibility never reads the voxel owner (`lib/webgpu-svo-dry-scene.ts:3443-3444`). The real limit is §2's 64-candidates-per-brick |
| §4 floret canopy | envelope + two octaves of displacement | **loses 58.9 % of rays.** Use a smooth-union of repeated spheres instead |
| §4 | "each gets a declarative node kind in `SceneryNode`" | none did; all six bake |
| §2 | stepping discs as static rigid cylinders | they are scenery; `rigidBodies` is empty |
| §6b | "7.5 mm overruns a device limit" | solver path only. Dry render reaches 7.5 mm in 16.6 ms and 3 mm in 39.5 ms |
| §6b | "11 oracles, green" | 12 oracles, **1 red** — `tests/pond-vessel.test.ts:95` |
| §5 phase 4 | floret canopy is the last phase | §6b(3) already re-argued it as the backbone; §2 and §3 here make that conclusive |

---

## 8. What I would do, in order

1. **`cullMode: "none"` → single face** on the primitive-visibility pipeline
   (`lib/webgpu-svo-dry-scene.ts:5314`). One line, ~2× on every primitive's exact
   test count, and it is the pass the aggregate will land in.
2. **Wire the porcelain terrain to a procedural surface** — a third arm on
   `lib/svo-material-abi.ts:148-150` plus a `plaster` policy row. Highest return
   per line in the program; the vessel is most of the frame.
3. **Resolve the doubly-authored coping.** The crest moved from the heightfield
   into 672 scenery cones and nothing asserts the two agree; that disagreement is
   the four red tests below. Decide which authority owns the crest, then either
   fix the oracles or fix the sweep.
4. **Register a GPU smoke lane for the scene.** Nothing renders it in CI today,
   so every finding in this document decays from the moment it is written.
5. **The three registries** of §6, in that order.
6. **`smoothUnionCluster` through the new table**, envelope-to-voxeliser and
   full-SDF-to-render per §4b. Retire the bonsai's 1 339 records and the coping's
   672, which alone returns the budget to comfortable.
7. **Then the water**, which is a larger program than the plan implies: caustics
   need a consumer and a rewritten energy term, the teal needs an authoring hook,
   and the splash crown needs a spray system that is currently dead code.

Camera aperture sits outside this order because it gates the framing rather than
any single item, and because it is ~20 call sites plus their CPU inverses.

---

## 9. Loose threads

**Four red tests in the working tree**, one root cause — the coping left the
heightfield for scenery:

- `tests/pond-vessel.test.ts:95` — crest 0.30105 against an authored band of 0.355
- `tests/sculpted-terrain-render.test.ts:98`, `:132`, `:168`

**`npm run check:scenery` is green but does not cover this scene** — it walks the
eight environment preset graphs, never the hero document's own `scene.scenery`.
Plan §5 phase 0's oracle does not do what it was written to do.

**`planVoxelScene` / `VoxelScenePlan` (`lib/voxel-scene.ts:205`, `:541`) has no
production callers** — test-only. It is worth knowing because `VoxelSceneSource`
is the usual citation for "scenery is invisible to the water." The invariant does
hold, but the real proof is that the solver files never import the proxy catalog,
not that type.

**Reproducing §3.** The march experiment is a standalone script — engine loop,
epsilon, iteration ceiling and bounding-sphere interval transcribed from
`lib/svo-primitive-abi.ts`, scored against a dense-march ground truth. It is not
yet in the tree. It should become either a `tools/` benchmark or, better, a CPU
oracle test alongside `tests/svo-primitive-ray.test.ts`, so that the Lipschitz
property of any future aggregate kind is pinned rather than remembered.
