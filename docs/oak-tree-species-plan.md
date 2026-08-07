# A procedural oak, as a cutover for the hero bonsai

## Context

The hero garden's tree is `BONSAI_POND_CANOPY` — a fused multi-trunk under a wide
flat cauliflower crown, tuned against `artifacts/plate-crops/canopy.png`. The ask
is to replace it with a **realistic oak**, procedurally grown, with enough
exposed parameters to art-direct it, and — the requirement that shapes every
decision below — one that **voxelizes and SDFs correctly at a range of leaf sizes
and refines or degrades gracefully between them**.

Reference given: <https://80.lv/articles/how-to-create-realistic-3d-oak-tree>.

## Decisions taken

| question | answer |
|---|---|
| Size in scene | **~0.80 m tall** — double the bonsai. Buys a full second branch order *and* individually drawable oak leaves at the production leaf. The container went 0.6 → 1.2 m alongside it, though not for the reason first given; see concern 7. |
| Surface | **Porcelain oak** — realistic in form, ceramic in surface. The set stays monochrome; palette and value stay form parameters so a naturalistic re-palette is one edit. |
| `gate:hero-fidelity` | **Gated off through P0–P3**, reconciled at P4. Fastest iteration; the risk accepted is finding a composition problem late, which P0's preview frame is meant to blunt. |

---

## Concerns, up front

### 1. The article is an art pipeline, not an algorithm — and half of it is unavailable here

The 80.lv piece is SpeedTree → Maya (LODs) → Substance 3D Designer → UE5. Its
three load-bearing techniques do not exist in this renderer:

| The article does | Why it cannot cross over |
|---|---|
| Foliage as **branch cards** — alpha-masked planes | A plane has no interior. It cannot be voxelized, has no SDF, and the whole *overdraw* discussion is a rasteriser concern that a marched SDF renderer does not have. |
| **Leaf/bark texture atlases**, vertex-colour masks | Shading here is the voxel cell's surface plus a palette + material closure (`svoMaterialFunctionIdForEnvironmentProxy`). There are no UVs to atlas. |
| **Discrete LOD meshes** baked in Maya | Level of detail here is `detailCellSize_m` fed into the generator's own derivations, plus the SVO's opacity pyramid. There is no mesh to decimate. |

What *does* cross over is the art direction, and it is worth taking seriously
because it argues against what `procedural-tree.ts` currently does:

- **"A more massive structure and multiple trunk starts"** over a single trunk.
- **Schematic branch-level planning** — decide how many branch orders up front
  and colour-code them. That is exactly the parameter axis we want to expose.
- **Card placement optimised so "branches remain visible, not hidden by
  excessive leaves."** In our terms: the crown must have sky and shadow *through*
  it, and the skeleton must read inside the canopy. This is the same lesson
  `bonsai-canopy-pads.ts` already learned the hard way ("a canopy with no sky in
  it is a disc").
- **Bark as light/dark noise banding enhancing height detail** — reachable as a
  `domain-warp` + `worley-subtract` tape on the trunk record.
- **Variations: older/younger, broken branches** — re-seeds and sibling forms.

**So the plan below is: the article's art direction, this codebase's machinery.**
Nothing about SpeedTree is imported.

### 2. The band law decides how realistic an oak can be, and it is a function of the tree's size in frame

The project-wide law, written down in `lib/voxel-scenery/bonsai.ts`:

> A feature whose period is under about **two leaves** does not render as that
> feature; it renders as aliasing. Above about **three leaves** it renders as
> geometry.

Production runs at `SVO_ENVIRONMENT_REFINEMENT_DEPTH_DEFAULT = 3`, so the leaf is
**0.78125 mm** and the legibility floor is **2.34 mm**. Taking a real oak at 20 m
with a 1.0 m bole and scaling it to a height `H` in the scene:

| feature | real oak | H = 0.40 m (the bonsai's) | H = 0.80 m | H = 1.20 m |
|---|---|---|---|---|
| bole | 1.00 m | 20 mm — 25.6 leaves ✓ | 40 mm — 51 ✓ | 60 mm — 77 ✓ |
| primary limb | 0.30 m | 6 mm — 7.7 ✓ | 12 mm — 15 ✓ | 18 mm — 23 ✓ |
| secondary | 0.10 m | 2 mm — **2.6 ✗** | 4 mm — 5.1 ✓ | 6 mm — 7.7 ✓ |
| tertiary | 0.03 m | 0.6 mm — 0.8 ✗ | 1.2 mm — 1.5 ✗ | 1.8 mm — **2.3 ✗** |
| twig | 0.01 m | 0.2 mm — 0.3 ✗ | 0.4 mm — 0.5 ✗ | 0.6 mm — 0.8 ✗ |
| leaf blade | 0.10 m | 2 mm — **2.6 ✗** | 4 mm — 5.1 ✓ | 6 mm — 7.7 ✓ |

Read off that table:

- **At the bonsai's exact size, a faithful oak gets a trunk and one branch order
  and no individual leaves.** Everything else is under the floor. This is why
  0.40 m was not taken.
- **At the chosen 0.80 m the tree gets a bole, primaries and secondaries, and
  leaf blades at 5.1 leaves across — individually drawable.** That is the whole
  reason for the size, and it is the difference between an oak and an oak-shaped
  bonsai.
- **A third branch order never arrives** at any size this stage can hold, and
  **twigs never render at all.** Both have to be implied by the foliage's own
  granulation rather than modelled — which is precisely what the canopy tape's
  three interleaved scatter lattices exist to do.

The honest consequence: the specimen is authored at proportions *pushed coarser
than nature*, with the finest published scale floored at three leaves, and the
plate's own proportions taken over as the leaf shrinks. `bonsaiCanopyLadder` is
the pattern to copy — it is a derivation, not a set of authored numbers, which is
what makes "refines gracefully" a property rather than a hope.

### 3. The hero set is monochrome fired porcelain

`{ kind: "terrain-shell", id: "shell", materialModel: "porcelain" }` is the
scene's only declaration of what it is made of, and every prop reads it. A
brown-barked, green-leaved oak does not go in this scene; it goes in a different
scene. The recommendation is a **porcelain oak** — realistic in *form*, ceramic
in *surface* — with `barkPalette` / `canopyPalette` / `barkValue` / `canopyValue`
as form parameters so a naturalistic re-palette is one edit, not a rewrite.

### 4. There are already two tree generators, and a third needs justifying

`procedural-tree.ts` grows a niwaki (`kind: "tree"`, cones + ellipsoid pads,
sway). `bonsai.ts` grows a multi-trunk with a `field-program` canopy. The bonsai
header argues carefully for why it was not a fork of the first. The same test
applies here, and an oak passes it: a single dominant bole with strong apical
dominance, three branch orders on a phyllotactic spiral, an irregular *billowing*
crown of many masses rather than one flat plate, and — the article's explicit
direction — visible skeleton *through* the canopy. Neither existing generator
re-tunes into that.

But the new module must be **layout and form, not new machinery**: every run
reuses `swept-tube.ts` (`tapered-sweep`, one record per run, C¹ through bends),
and the foliage reuses the canopy tape construction from
`bonsai-canopy-field.ts`. No new primitive kinds, no new ops.

### 5. Budgets: the per-brick candidate ceiling, not the scene ceiling, is what bites

Measured on the hero document at depths 0 and 3 (`buildEnvironmentProxyCatalog`):

- Scene ceiling: `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES = 16_384`, and the whole
  hero garden publishes **405 records at depth 0 and 442 at depth 3**. The scene
  budget is not the constraint and will not become one.
- **Per-brick: 64 candidates, and `svo-render-tuning.ts` states that depth 2 is
  already "the measured point where the hero garden's busiest brick reaches the
  64 candidates the hierarchy binds".** And the overflow is worse than the
  bonsai's header says. It is not merely a silent drop of the surplus: per
  `webgpu-sparse-scene-proxies.ts`, three overfull bricks in the hero garden's
  pebble bands meant `completedRevision` was never stored, the live generation
  never advanced, the node-mip certification pass read zero — and **cone
  visibility failed closed over the whole octree domain, rendering a hard-edged
  black slab across the container footprint.** An oak crown of many overlapping
  foliage pads is exactly the geometry that provokes this, so the busiest-brick
  census is a correctness acceptance criterion, not a budget nicety.
- **Tape ops: 16, and `bonsaiCanopyPadProgram` already uses 16 of 16** at the
  production leaf (4 point registers, 4 field registers). The foliage tape has no
  headroom for a fourth scatter level. Three interleaved lattices is the ceiling
  inside one record; more scales come from *layout*, not from the tape.

### 6. Sway

The bonsai deliberately has none — plain primitive nodes, and only `kind: "tree"`
opts into the gust. `procedural-tree.ts` has a full sway system
(`treeSwayFor`). An oak published as generator nodes inherits *no sway*. If the
tree should move, that is a separate piece of work (the excursion budget is set
by the sparse lattice, because a swaying prop is re-posed every frame and never
re-voxelized). **Recommend: no sway in v1**, matching the bonsai, and revisit.

### 7. The container was 0.6 m tall, and raising it is right for reasons other than the one I gave

**Corrected after measurement. The premise below was wrong and the conclusion survives it.**

I originally wrote that the container box *is* the voxel domain, so an 0.80 m oak
reaching y = 1.01 m would have two fifths of itself unvoxelized. That is not what
the planner does. `planSparseSceneDomain` seeds its lattice at the container and
then takes the **union** with every proxy AABB, brick-aligned, so a prop that
overhangs the box *extends* the domain. Measured directly: adding one synthetic
proxy reaching y = 1.01 to the 0.6 m document took the domain from 15 to 21
bricks on y with the container untouched. **A tall oak under the old lid would
have rendered.**

Two related measurements also corrected me. The bonsai's true extent is
y = 0.145 … **0.7035**, not 0.608 — my first probe read `lobe` half-extents and
so missed the canopy pads, whose `field-program` extent is derived from the tape
rather than authored. So the crown was already **103 mm** over the old lid, not
8 mm.

The raise still happened, for the two reasons that survive:

- `HERO_LAYOUT_CONTAINER_BOUNDS` is this box and `tests/hero-layout.test.ts`
  asserts nothing the layout places escapes it. A 1.01 m crown does not fit under
  a 0.6 m lid. (The tree is an *authored* node, not a layout placement, which is
  how the bonsai's crown came to sit 103 mm over unremarked.)
- **The domain's height stops being an accident.** At 0.6 m the planned domain
  came out 15 bricks tall purely because that is where the bonsai's crown ended:
  the extent of the world was a property of whichever prop happened to be
  tallest, and swapping the tree re-shaped it silently.

**It costs nothing, measured.** Claimed bricks, nodes, leaves and plan voxels are
*identical* at 0.6, 1.1 and 1.2 m at both depth 0 and depth 3 — only the bounding
box grows. A brick exists because something claims it, and a dry world drops the
container claim, so the ladder scales with occupancy rather than with the box.

**The height is 1.2 m, not the 1.1 m the tree needs.** 1.1 m breaks a stated
invariant: every container dimension must be a whole number of 8-cell bricks, and
at 1.1 m the 25 mm rung is 5.5 bricks — 25 mm being `HERO_GARDEN_SOLVER_CELL_M`,
what a wet document is clamped to. The concrete symptom is that
`FLUID_SVO_DRY_SMOKE_CELL_MM=25` and `=15` abort on the brick guard. 1.2 m is
exactly 6 bricks, whole at every finer rung, and measures the same 324 solver
bricks and the same [13, 6, 8] domain as 1.1 — the rounding was already paid.

### 8. The bonsai does not refine with the leaf, so "graceful refinement" has no precedent to copy

Also measured: the bonsai publishes **exactly 154 records at depth 0 and at
depth 3** — identical. Only its canopy *tape* changes with the leaf (the
sub-floret carve gates in at a fine enough rung). The rosettes do refine
(25 → 35 records, 34 → 50); the hose never does.

So the requirement to "refine/degrade gracefully by level" is a property the oak
has to be **built** to have. It is not inherited from the object being replaced,
and the record-count-monotone assertion in the verification section below is
therefore a new contract rather than a regression guard.

---

## The approach

### Growth model: parametric recursive (Weber–Penn shaped), not space colonization

Space colonization needs an attractor cloud, produces a graph you then have to
fit sweeps to, and its structure moves unpredictably under a re-seed. A
parametric recursive model — which is what SpeedTree is underneath — gives
exactly what is being asked for:

- **Per-order parameters**: `branches`, `downAngle` (+ variance), `rotate`
  (phyllotaxis), `curve` / `curveBack` (the vase), `lengthFraction`, `taper`,
  `radiusRatio`. That is the "extensively modify it" axis, and it is the
  article's own "schematic planning with colour coding" made into data.
- **Determinism**: one integer avalanche hash of the branch path, exactly as
  `procedural-tree.ts` and `bonsai.ts` do. The sparse publication cache's static
  revision depends on it.
- **Re-seed grows a sibling**, not a different species — the README's rule 5.

### Graceful degradation is *branch-order admission*, and it is derived

This is the mechanism that answers the level requirement, and it is the heart of
the design:

> Order *k* is published only when `2 · radius_k ≥ 3 · leafSize_m`. Below that,
> its foliage is absorbed into its parent's pad rather than dropped.

Two invariants follow, and both are testable:

1. **The crown envelope is leaf-invariant.** The silhouette, the overhang and the
   total foliage volume are the same at every rung; only the *internal* structure
   refines. A tree that changes shape with the voxel is a tree that pops.
2. **Nothing is ever published under three leaves.** The finest scale either
   exists at the floor or does not exist at all.

So the ladder is:

| leaf | rung | published |
|---|---|---|
| 6.25 mm | depth 0 | bole + primaries; few large foliage masses, tape at one lattice level |
| 3.125 mm | depth 1 | + secondaries begin; two lattice levels |
| 1.5625 mm | depth 2 | secondaries fully; three lattice levels |
| 0.78125 mm | depth 3 | + the sub-floret carve; foliage at the plate's own proportion |

and the record count rises monotonically with the rung, which is the thing to
assert.

### Module layout — mirroring the bonsai's three-file split

| file | holds |
|---|---|
| `lib/voxel-scenery/oak-branching.ts` | The growth model. Pure: `OakBranchOrder[]` + a seed → a list of runs (polylines + radii) and foliage attachment frames. No nodes, no records, no scene. |
| `lib/voxel-scenery/oak-canopy-field.ts` | What one foliage mass is made of: the `field-program` tape. Generalised from `bonsai-canopy-field.ts` — an *oak* leaf plate is larger, flatter and less clumped than a cauliflower floret, so the constants differ; the construction does not. |
| `lib/voxel-scenery/oak.ts` | The species: `OakForm` (shape only), named forms, `OakSpec` (form + `at_m` + `groundHeightAt` + `lean` + `leafSize_m` + `seed`), `planOak` → `OakPlan { nodes, leafCount, bounds_m, crownBounds_m, ... }`. Emits `tapered-sweep` runs via `sweptTubeNodes` and `field-program` pads. |

Then the wiring, which is three one-liners each per the catalog's own rule:

- `SCENERY_GENERATOR_IDS` in `lib/scenery-graph.ts` — add `"oak"`.
- `SceneryGeneratorParamsByKind` + `SCENERY_GENERATORS` in
  `lib/scenery-generators.ts` — `OakGeneratorParams extends OakForm` with
  `at_m` / `lean`; `needsVessel: false`; pass `leafSize_m: request.detailCellSize_m`.
- `lib/hero-garden-scene.ts` — replace the `bonsai` generator node with an `oak`
  one at `BONSAI_AT_M`, and rename `HERO_BONSAI_FORM` → `HERO_OAK_FORM`.

`/shape-lab` picks it up **for free**: `shapeLabWorld` enumerates top-level
graph nodes and groups by id prefix, so the oak appears as a specimen with its
`params` on sliders the moment the node exists. That is the art-direction loop,
and it should be the first thing that works.

### What is *not* deleted

`bonsai.ts`, `bonsai-canopy-pads.ts`, `bonsai-canopy-field.ts` and
`tests/bonsai.test.ts` stay. `BONSAI_COURTYARD_STANDARD` and
`BONSAI_SHELF_MINIATURE` are catalogue specimens, `tools/preview/bonsai.ts`
renders them, and the species is the control arm the field-program construction
was argued against. The cutover is **the hero graph's node**, not the module.
Retiring the species is a separate call once the oak has held the frame.

---

## Open items found during implementation

### The sub-leaf grain carve does not fit this ladder, and is gated off

`oak-canopy-field.ts` carries the carve and admits it only when the leaf can draw
it *and* it fits inside the tightest lattice level's clearance. Measured, it is
admitted at **no** rung, and two independent budgets agree on why:

- **The leaf gate.** A 2.5 mm lobe grain at the bonsai's 4-voxel carve gate needs
  a 0.625 mm leaf. Production's is 0.78125 mm.
- **The clearance budget.** The carve runs on the assembled pad, where nothing
  clamps it back to a fold's saturation value, so it is only sound while it acts
  strictly inside the tightest level's clearance — 0.66 mm here, against the
  bonsai's 2.4 mm, because the oak's finest lattice is a 6 mm cell rather than a
  30 mm one. Any pit that fits is about **half a voxel deep**, and a carve that
  cannot move the surface by one voxel is not a feature.

Both exits were considered and neither works: raising the grain to 3.2 mm keeps
the 4-voxel gate but is then refused by the clearance check, and lowering the
gate to 3 leaves admits a pit too shallow to see. The honest position is that
this is the **fifth** scale on an object whose ladder reaches four, and it needs
either a finer leaf than depth 3 or a coarser finest lattice level to buy the
clearance back. The code self-gates; nothing is fudged. Revisit after the
silhouette is right.

### Clearance is the unmeasured performance risk

The oak's tightest level saturates at **0.66 mm** against the bonsai's 2.4 mm, so
sphere-trace steps near the canopy surface are roughly 3.6x shorter from depth 2
on — on what will be the largest object in the frame. This is arithmetic, not a
measurement. `OAK_FOLIAGE_CELLS_M[2]` is the lever if a frame comes back too
dear, and it should be measured before the ladder is declared affordable.

---

## Order of work

**P0a — the container, before any geometry.** Raise
`HERO_GARDEN_CONTAINER.height_m` from 0.6 m to ~1.1 m so an 0.80 m tree is
inside the voxel domain (concern 7). Re-measure the depth-0..3 page ladder and
confirm the existing scene renders unchanged. This gates the size decision, so
it goes first and its result is reported before anything is grown.

**P0b — the loop, before any tuning.** Stand up `oak-branching.ts` + a minimal
`oak.ts` emitting only the skeleton (no foliage), wire the generator id, swap the
hero node, and look at it in `/shape-lab` at depths 0–3. Nothing about proportion
is decided here; the point is that the parameter loop and the depth ladder are
live before a single number is tuned. The bonsai's own history is that two art
passes were spent tuning parameters inside functions that could not express the
target shape at all.

**P1 — the skeleton.** Branch orders, phyllotaxis, curve/vase, taper, root
flare. Reuse `sweptTubeNodes` for every run so a limb is one record with a C¹
bend and a solved envelope. Assert: run count, extent, seated on the ground, and
the order-admission ladder against the leaf.

**P2 — the crown.** Foliage mass *layout* first (how many, how big, where — the
`bonsai-canopy-pads.ts` argument: a displacement cannot put a void between two
masses, so the scale at which the crown stops being connected is geometry and
everything below it is relief). Then the tape. Oak-specific: the crown is
billowing and irregular where the bonsai's is a flat plate, and the skeleton must
stay visible through it.

**P3 — bark.** A `domain-warp` + `worley-subtract` tape on the bole, gated on the
leaf (`CANOPY_LEGIBLE_PITCH_LEAVES`-style admission). This is the article's
"noise patterns to create lighter and darker areas, enhancing the height
details," and at depth 3 a 3 mm bark fissure is 3.8 leaves — the first rung where
it is geometry rather than noise. Below that it must gate off entirely.

**P4 — variants.** `OAK_HERO`, plus at least one sibling form (the article's
"older or younger versions") to prove the parameters are a species rather than
one specimen.

---

## Verification

**Per-stage, cheap:**

- `npm run check:scenery` — the scenery audit.
- `npx tsc --noEmit` — the catalog totality is enforced by types; a missing entry
  is a compile error by construction.
- `node --import tsx --test tests/oak.test.ts` — the new oracle. Per
  `lib/voxel-scenery/README.md` it must pin: determinism (same seed → identical
  geometry), re-seeded siblinghood, leaf count against a stated budget, extent,
  ground seating, and the material closure regex.
  Oak-specific, and the ones that matter most:
  - **crown envelope invariant across depths 0–3** within a stated tolerance;
  - **record count monotone non-decreasing** with the rung;
  - **no published feature under 3 leaves** at any rung;
  - **worst-brick candidate count ≤ 64** — the silent-drop ceiling.
- `shapeLabDepthLadder({ specimenId: "oak" })` — reports records and `moved` per
  rung, which is the degradation contract read directly off the product path.

**Looking at it:**

- `/shape-lab` for parameter work — one specimen, CPU, sliders, any leaf.
- `tools/preview/oak.ts` on `heroPreviewScene` for a lit frame through the
  production dry-scene path (`tools/preview/README.md`). Never on the scene
  factory directly — that renders the porcelain garden under the default set's
  dark teal sky.

**The scene:**

- `npm run test:webgpu:hero-garden-hose` — the dry render smoke.
- `npm run test:webgpu:hero-garden-hose-x10` — the densified acceptance scene,
  which is where a record-count regression shows up first.
- `npm run test:webgpu:garden-hose` — the wet lane, to confirm raising the
  container did not cost the solver anything (P0a).
- `npm run gate:hero-fidelity` — **gated off through P0–P3 by decision**, then
  reconciled at P4: agree the frame by eye first, re-baseline the gate to it, and
  it guards against regression from there.

Note the GPU lock: the WebGPU lanes serialize through
`tools/run-webgpu-exclusive.ts` and cannot be run in parallel.

---

## Status at first playable version

Landed and measured. `/shape-lab` lists `oak · generator · oak` and expands it in
about 12 ms.

**The refinement contract holds, verified independently of the generator:**

| rung | leaf | orders | runs | masses | records | Σ share | crown box | drift |
|---|---|---|---|---|---|---|---|---|
| depth 0 | 6.25 mm | 2 | 12 | 30 | 56 | 1.000000000 | 713 x 689 x 700 mm | — |
| depth 1 | 3.125 mm | 2 | 12 | 30 | 62 | 1.000000000 | 713 x 689 x 700 mm | 0.00% |
| depth 2 | 1.5625 mm | 3 | 42 | 30 | 174 | 1.000000000 | 713 x 689 x 700 mm | 0.00% |
| depth 3 | 0.78125 mm | 3 | 42 | 30 | 174 | 1.000000000 | 713 x 689 x 700 mm | 0.00% |

Zero drift on every axis at every rung: the crown is the same crown and only its
internal structure refines. Records rise monotonically.

**The per-brick census passes with headroom.** Busiest brick 33 / 34 / 39 of the
64 the hierarchy binds at depths 0 / 2 / 3, zero overflowed bricks. Note this is
the 50 mm *base* brick; the refined-environment-leaf case that
`refineEnvironmentLeaf` exists to split has not been measured and needs the full
plan chain.

### How the invariance was actually achieved, and why the obvious fix fails

The first build drifted +36% in crown width when the third order published, and
the fix I prescribed — place a culled subtree's absorbing attachment at its
share-weighted centroid — is **not sufficient**, for a reason worth keeping: a
point plus a radius has three degrees of freedom and a cluster's extent has six.
Collapsing a subtree to its centre of mass conserves bulk and discards spread,
and `share^(1/3)` sizing cannot recover it — closing the hero's 230 mm of lost
reach would need a mass 324 mm in radius on a tree 340 mm across.

What works is a **floor on the merge**: a mass wider than 0.40 of the crown has
no room for sky beside it, and volume tiling turns that into a floor on the
*count*. Culling asks for 6 masses at depth 0 and the floor pins it at 30 — the
same set as depth 3 — which is why the invariance is exact rather than close.

The honest cost: at depth 0, **10 of 30 masses hang off secondaries the leaf does
not draw**. That is irreducible while the crown may not shrink, and it is
tolerable only because the masses are ~260 mm across on a 713 mm crown and
overlap into a connected canopy. Look at a coarse rung before trusting it.

### Known-open

- **Sweep splitting.** 42 runs become 144 records at depth 3 — `sweptTubeNodes`
  splitting on envelope waste, driven by aspect ratio (order 1 is 19:1, order 2
  is 29:1) rather than by curvature; measured sinuosity is 1.02-1.04. Raising
  `envelopeWasteBudget` to 20 roughly halves the records for ~3.4x the envelope
  volume — deliberately **not** taken, because fatter envelopes mark more bricks
  occupied and the per-brick ceiling is the binding constraint, not the record
  count.
- **The grain carve is gated off at every rung** (above).
- **Canopy march cost** is unmeasured (above).
- **A pre-existing burial**, unrelated to this work: `tests/hero-layout.test.ts`
  reports `plant/air-4/core` inside `hose/tube/run-4`. Neither the hose nor the
  plants depend on the container height or on the tree, so this is not from the
  cutover; it surfaced once the oak's own burial was fixed.
- **`tests/bonsai.test.ts` has 7 failures, all species-level and pre-existing** —
  coverage ratios, crown aspect, floret span, head count. The one test that
  genuinely depended on the hero document holding a bonsai was retargeted to
  build its own node, since the species is no longer in that scene.
