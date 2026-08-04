# One raster-visibility pipeline for the SVO renderer — handoff plan

**Goal:** scenes ~10× the complexity of `hero-garden-hose` (≈5,000 authored SDF
records) at 60 fps (≤ 16.7 ms/frame, ~1600×1240) on an M1 Max, by adapting the
SVO renderer to (1) Teardown-style cheap visibility via a raster pass, (2) the
SDF voxelization the engine already performs, with (3) analytic marching
reserved for a bounded near-field band.

**Supersedes** `docs/sdf-unified-render-plan.md` (2026-08-04). That plan's
diagnosis stands; this document corrects several of its load-bearing claims
against HEAD `58c94f4`, adds the external research, and — the biggest change —
records that most of the machinery its Phase 2 proposed to build **already
exists in production** and needs upgrading, not building.

All measured numbers trace to `docs/svo-render-frame-anatomy.html` +
`.data.js` (capture at `a3adca9`; no SVO render file has changed since) or to
a `file:line` verified at HEAD on 2026-08-04.

---

## 0. Implementation outcome (2026-08-04) — read before §1

W0–W4 are implemented. §1–§9 below are preserved as the original handoff; where
implementation contradicted them, the contradiction is recorded here and marked
inline. **Six load-bearing claims did not survive measurement.**

### What landed

| WS | Outcome |
|---|---|
| W0 | `hero-garden-hose-x10` (5,039 records, hero camera preserved, multiplier-sweepable); record/BVH/arena ceilings 4,096 → 16,384 |
| W1 | SDF raster on coverage/resolve/overflow. Hero **415.7 → 160.4 ms** (2.59×); the SDF pass **≈357 → ≈101 ms**. Depth parity exact (0 of 1,984,000 texels), overflow 0.0 %, stationary reuse engages |
| W2 | Voxelizer ABI-true (297/501 hero records); node-mip address plan growable and **total** (441/441 pages), so the silent withdrawal cliff is unreachable |
| W2b | Invalidation scatter-from-region-bounds (**17–67×** at 1,002 regions); binning off a coarse record grid (flat vs 23× linear); per-frame chunk budget with the completion contract verified non-vacuously |
| W3 | Voxel primary promoted — **flattens 10× scaling 12.2× → 2.30×**. Near-field band built but **ships disabled**. Coverage-time occlusion reject kills arena overflow at every rung |
| W4 | Tripwires on the silent rows; overflow-driven indirect draw; 10× memory audit; both broken trace lanes fixed |

### Gate results, stated plainly

- **W1's gate: met.** ≤ 200 ms, exact parity, < 1 % overflow, reuse engaged.
- **W3's gates: two missed.** Hero is ~55 ms pipelined against a ≤ 33 ms target.
  10× costs **2.22×** against a < 1.3× target. The voxel primary did the
  structural work (12.2× → 2.30×); the band contributed the last ~4 %.
- **W3's silhouette gate passes only with the band off.** At a 48 px threshold,
  crown-crop depth p99 shifts to 97.7 mm at ×10, so `nearFieldBandPixels`
  defaults to 0. §5/W3 assumed the band was the mechanism for 10×; **it is not** —
  the voxel primary is, and the band is quality/headroom.

### The six corrections

1. **§5/W5 and §6's lighting cost model is half wrong.** "Lighting stays ∝
   pixels × cone resolution — scene-size independent" holds for the *pyramid
   consumers* (world-GI cache 1.06×, deferred lighting 2.30×) and fails for the
   **cone prepass, 4.93×, which is 93 % of the lighting group at 10×**. The
   mechanism: `dryPrepassCoherentMain` rejects 47–57 % of prepass texels as
   non-homogeneous and `dryPrepassBoundaryMain` re-traces each through
   `traceScenePrimitives` — a candidate-BVH walk, O(records). Boundary *count*
   grows 1.22×; **cost per re-trace grows ≈ n^0.61**. §6's budget is unreachable
   as written, but not for the reason §8 feared.
2. **"Deferred lighting is a large pass" was a measurement artifact, twice.**
   On this TBDR GPU, `beginningOfPassWriteIndex`/`endOfPassWriteIndex` brackets
   *[vertex-stage start, fragment-stage end]*, and the tiler hoists every vertex
   stage to the front of the frame — so the **last-encoded render pass reports
   ≈ the whole frame**. Deferred lighting is really **2.70 ms at 1× / 6.21 ms at
   10×**, ~1 % of the frame. **Render-pass timestamps here are unusable;
   compute-pass timestamps are sound.** The lanes now encode this in the type
   (render passes return `duration_ms: null`).
3. **§5/W4's "Live-scene voxelizer records — overflow flag" is wrong.** It is a
   **hard throw at world construction**, and it fires *before* the record ceiling,
   from the PBR material table.
4. **§5/W4's "Brick node index mask 2²² — silent aliasing" is wrong at HEAD.**
   `svoBrickEmitMain` already returns early past the mask, so it is a silent
   **drop**. The ceiling is also 4.19 M **nodes**, not leaves (~2× fewer leaves).
5. **§5/W4's "Candidates/brick — overflow flag (image-preserving)" does not
   survive the 10× scene.** Busiest brick measured **442 against a 64 arena**,
   37 bricks dropping — those primitives were absent from the opacity pyramid and
   radiance atlas while still drawing in primary visibility. The arena is now 512,
   **decoupled** from a refinement target of 64 (sharing one constant made the
   planner build a *coarser* tree as the arena grew).
6. **§8's memory risk resolves favourably, and relocates.** At 10×: 445.6 MiB at
   the authored 25 mm, 761.4 MiB at 10 mm. Record count barely moves it — **cell
   size is the whole question**. 5 mm fails, but *not* on bytes: it fails because
   `planSvoNodeMipPyramid` returns incomplete at 25,479 pages against
   `maxTextureDimension2D` = 16,384. **W2 may commit to any voxel size down to
   10 mm; below that, raise pyramid page capacity, not memory.**

### Where the remaining time is

After W3, what still grows with record count is the octree itself (more resident
bricks, more per-pixel payload) and the cone prepass boundary tier. The
identified wall is the owner→exact hybrid: every solid voxel triggers
`svoIntersectPrimitiveExact` on an aggregate's 3-octave lattice at up to 48
iterations, and the upgrade budget is currently doing double duty as the quality
knob (budget 8 → 179/165 ms at p99 10.7 mm; budget 2 → 155/101 ms at p99 253 mm).
A cheap per-cell rejection is worth more than anything left in W3.

### The lane's noise floor — read before quoting any single number above

Four interleaved repetitions of `test:webgpu:hero-garden-hose-x10`, alternating
arms so process-launch and thermal drift hit both equally, measured a
**within-arm sd of 6.7–8.4 ms on a ~145 ms frame (±5 %), with a batch-to-batch
mean drift of ~9 ms on top**: the same binary that medians 137 ms in one batch
medians 148 ms twenty minutes later. Every frame time in this section is a
single-sample point estimate carrying that error bar, so **1.78× should be read
as ≈1.8× ± 0.1**, and no A/B smaller than ~10 ms at 10× can be resolved by this
lane without repetitions. Two runs differing by 3 % have not measured anything.

### Bounding the visibility candidate-BVH walk — implemented, and a null result

`svoVisibilityNext` opened every shadow/AO/GI ray with `traceScenePrimitives`
bounded only by the ray's own tMax, which is the identical unbounded term the
primary shed in W3. Reordering it the same way — octree first, analytic walk
seeded with the voxel hit — is implemented and on by default, with
`unboundedAnalyticVisibility` as the control arm. Both arms pass the same 21
checks, `frame-carries-radiance`, `radiance-black-pages` and `frame-determinism`
among them.

**It does not move the wall.** Four interleaved reps at 10×: control 148.68 ms,
bounded 144.69 ms, delta −3.99 ± 5.39 ms, t ≈ 0.74. The sign flips across
repetitions. Kept anyway because it is image-preserving, removes a real O(records)
term, and strictly improves the exhausted-ray path — the old order discarded the
analytic hit on every exhausted ray, and the new one returns it when it lies in
front of the walk's reach — but **it is not a performance win and should not be
cited as one.**

The mechanism behind the null result, which also corrects the premise: the big
O(records) consumer named in correction 1 is the cone prepass boundary tier, and
`dryPrepassBoundaryMain` reaches the analytic set through
`traceOpaqueScene → traceDrySolidScene → traceStaticSolidScene → traceStatic`.
It therefore **inherited W3's bound automatically**, and correction 1's mechanism
description is stale for that reason. Every `traceScenePrimitives` call site in
the shader is now bounded; `svoVisibilityNext`'s share of the term was small
because shadow and AO rays are short, so there was little left to win.

A probe arm that deletes the visibility analytic term outright
(`visibilityAnalyticDisabledProbe`) was built to bound the prize and **does not
bound it**: at 10× it measures *slower* than either real arm (145.0 vs
137–139 ms), because deleting occluders stops shadow rays terminating early and
`svoTraceVisibility` then carries them to full tMax. It changes the workload
instead of costing less of it. Kept as a flag, but it answers a different
question than the one it was built for.

### New lanes

`test:webgpu:hero-garden-hose-x10`, `test:webgpu:svo-live-voxelization`,
`test:webgpu:svo-capacity-sweep`, `benchmark:svo-record-scale`.
`FLUID_SVO_DRY_SMOKE_VISIBILITY_ANALYTIC=bounded|unbounded|probe-off` selects
the visibility ordering arm on the dry-render lane.

---

## 1. Status quo

The hero frame is **405 ms**. One pass — `Sparse voxel exact live-scene
primitive visibility`, the direct proxy raster over 501 authored SDF records —
is **346 ms (85 %)**. The whole octree brick primary
(cull → coverage → resolve → overflow) is **3.0 ms**. Cone visibility 43 ms,
deferred lighting 11 ms.

The 346 ms pass rasterizes one 36-vertex proxy box per record
(`lib/webgpu-svo-dry-scene.ts:7314`, fragment `:2613`) and sphere-traces the
record's field in the fragment. The fragment returns `@builtin(frag_depth)`
**and** contains four `discard` sites (`:2627–:2632`), so the tiler can reject
nothing: every covering proxy marches even when a nearer proxy already resolved
the pixel. A covered pixel sits under a median of 7 proxies (p99 33, max 59).

Sixty lines above it in the same file, the brick raster solves the identical
problem correctly: a coverage-only pass appends candidates into a fixed
24-entry per-pixel arena (`:5328`), one full-screen resolve walks each pixel's
list front-to-back exactly once (`:5344`), and an overflow pass re-runs the
historical fragment for the rare deep pixels (`:5364`). Capacity is a
performance parameter, never an image change. Cost: 3.0 ms for ~40k leaves.

Two hard facts bound what tuning can achieve:

- The *sound* hidden-surface fix is worth **1.75×** on marches (5.22 vs 9.11
  per covered pixel, exact CPU census over 731 pixels), bracketed above by the
  unsound hardware probe's 3.15×. Applied to 346 ms that is 110–200 ms — a
  ~6 fps frame at **1×** complexity. Per-pixel analytic marching of every
  authored record cannot reach 16.7 ms at 10× by any constant factor.
- A ~5,000-record scene does not draw at all today: `buildSvoScenePrimitives`
  throws past `SVO_SCENE_DEFAULT_MAXIMUM_PRIMITIVES = 4096`
  (`lib/svo-scene-primitives.ts:36`, throw at `:263–:265`), caught as
  `console.error` + `{state:"blocked"}` in `lib/webgpu-renderer.ts:2020`.
  (The prior plan said the candidate build silently "stops drawing" — wrong:
  the failure is loud, and it is a *second* constant,
  `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES` in `lib/svo-primitive-candidates.ts:20`,
  that guards the BVH. Both must be raised.)

So the plan is structural: visibility must stop scaling with authored record
count. The octree brick path already has the right asymptotics (per-pixel cost
bounded by the arena walk, independent of what stands behind the first hit),
and the project's stated destination is exactly that — *one pipeline: very
high resolution voxels sampled from procedural geometry*.

---

## 2. Research: how the shipped engines solve this

### 2.1 Teardown (Tuxedo Labs / Dennis Gustafsson)

Sources: [acko.net "Teardown Frame Teardown"](https://acko.net/blog/teardown-frame-teardown/),
[juandiegomontoya "Teardown Teardown"](https://juandiegomontoya.github.io/teardown_breakdown.html),
[zacxalot render techniques](https://zacxalot.github.io/rendering/9-teardown/),
[Gustafsson's blog](https://blog.voxagon.se/).

- **Per-object volumes, proxy raster.** Every shape is a dense voxel 3D
  texture (1 byte/voxel palette index). Each draw is exactly 12 triangles —
  the shape's OBB, **back faces only** so the camera can clip into volumes.
  The fragment reconstructs the ray, clamps it to the box, and marches with a
  DDA accelerated by volume mips (skipping 2×2×2 / 4×4×4 empty regions).
  Thousands of shapes per level, one draw call each.
- **Cheap visibility without conservative depth.** Writing `gl_FragDepth`
  disables early-Z, and Teardown does *not* use GL conservative-depth.
  Instead: shapes are sorted front-to-back, linear depth is written to a
  G-buffer target, and that target is **periodically copied to a second,
  sampleable texture (~8 checkpoints per frame)**. Each fragment tests its
  volume's near face against the copied depth and discards before marching.
  Manual early-Z, tuned to the tiler.
- **Lighting against one global coarse volume.** All stochastic rays (AO, sun,
  local lights, specular) trace a *separate* world occupancy volume (1-bit
  voxels packed 2×2×2 per byte, with mips, updated by ~50 small uploads per
  frame), never the per-shape volumes — lighting is resolution-bound, not
  scene-bound. Deferred, 5 MRTs, blue-noise + spatiotemporal denoise.
  OpenGL 3.3, no compute.

### 2.2 Claybook (Sebastian Aaltonen, GDC 2018)

Source: [GPU-based clay simulation & ray-tracing tech, slides](https://media.gdcvault.com/gdc2018/presentations/Aaltonen_Sebastian_GPU_Based_Clay.pdf).
Directly relevant because Claybook's content is SDF, like ours.

- **World SDF volume**: 1024×1024×512, 8-bit, narrow-band multilevel (±4
  voxels per level → 1/32-voxel precision), **rebuilt sparsely on GPU in
  8×8×8 tiles with indirect dispatch** from transformed brush volumes; mips
  dilated + 3 eikonal relaxation steps.
- **Trace**: sphere trace with mip switching; terminate when
  `D < pixelConeWidth · t` (surface smaller than the pixel — "perfect LOD").
- **Coarse prepass**: one cone per 8×8-pixel tile (0.2 ms); per-pixel rays
  spawn at the end of their tile's cone. This beat both wave-ballot
  load-balancing and overstepping in their measurements. Full visibility +
  shadows + material ≈ 4 ms at 720p on base Xbox One.
- **They evaluated our exact current path.** The "Ray-Traced SDF Meshes"
  slide: raster the SDF's bounding box, sphere-trace in the fragment, discard
  on miss, and use conservative depth (`SV_DepthLessEqual`) — "**up to 6×
  faster than SV_Depth when high overdraw**". That 6× is what our direct
  raster forfeits, and it matches our measured 85 %-of-frame loss.
- **Sparse indirection measured slower**: software virtual texturing for the
  volume was 13 % slower per trace (sphere-trace steps are fetch-bound).
  Keep dense, directly-addressed page atlases.

### 2.3 The WebGPU constraint

WGSL has **no conservative depth**. A `@depth(greater|less)` attribute on
`frag_depth` is an open proposal ([gpuweb#5342](https://github.com/gpuweb/gpuweb/issues/5342),
PR #6299 unshipped); wgpu exposes it only as a native-only extension. Writing
`frag_depth` disables early/hierarchical Z with no way to promise monotonicity.
So Claybook's one-line fix is unavailable, and the proven WebGPU-viable
substitutes are:

| Mechanism | Proven where | Character |
|---|---|---|
| Front-to-back sort + periodic depth-checkpoint copies | Teardown (shipped, GL 3.3) | Approximate early-out; CPU sort; checkpoint cadence tunes reject rate |
| Per-tile coarse cone/depth prepass bounding rays | Claybook (shipped, consoles) | Compute prepass; bounds march starts, not candidate count |
| **Per-pixel coverage/resolve candidate arena** | **This codebase — brick raster, 3.0 ms** | Exact; bounded per-pixel cost; overflow fallback preserves image |

**Decision: keep the arena.** It is already shipped here, it is exact (image
parity is testable), and it is *stronger* than either external mechanism — it
bounds candidate count per pixel and enables front-to-back early termination
in a single resolve pass. Teardown's checkpoint scheme is the fallback if
arena memory becomes the constraint; Claybook's tile prepass is the natural
add-on for bounding march starts within the resolve.

### 2.4 Mapping table — external technique ↔ this codebase

| Teardown / Claybook | Here |
|---|---|
| OBB back-face raster + in-fragment march | `SVO_SCENE_PRIMITIVE_RASTER_CONTRACT` (36 verts, back-face cull) — already identical in shape |
| Depth checkpoints / conservative depth | Coverage/resolve arena (`lib/webgpu-svo-brick-raster.ts`) — adopt for the SDF set (workstream W1) |
| Per-shape dense voxel volume + mips | Octree leaf bricks (8³, 25 mm hero voxels) + brick occupancy macro-mask + node-mip pyramid |
| Global 1-bit occupancy volume for all lighting rays | Node-mip opacity pyramid + tetra radiance atlas — same role, already incremental |
| Claybook sparse GPU SDF rebuild in 8×8×8 tiles | `lib/webgpu-sparse-scene-proxies.ts` voxelizer — same design, already in production (W2 upgrades it) |
| `D < pixelConeWidth·t` termination | `lib/svo-screen-space-termination.ts` — exists for traversal; reuse for the near-field band policy (W3) |
| Claybook sparse-VT −13 % | Keep dense page atlas addressing (verified: current atlases are direct) |

---

## 3. What already exists (correcting the prior plan)

The prior plan's Phase 2 ("make the SDF→octree voxelizer incremental…")
assumed a build. In fact:

**A complete incremental GPU SDF→octree voxelizer is in production.**
`lib/webgpu-sparse-scene-proxies.ts` (`sparseSceneProxyVoxelizationShader`
`:507`, `SparseSceneProxyVoxelizer` `:862`): dirty-region → dirty-brick
invalidation → per-brick candidate binning (64/brick) → per-voxel rebuild
(min signed distance, solid fraction, packed material|owner into dedicated
scene payload lanes) → occupancy finalize (2×2×2 macro mask + cell bbox).
All indirect-dispatched, revisioned, with exact CPU mirrors. Driven by
`OctreeSparseBrickWorld` (`lib/webgpu-octree-sparse-bricks.ts`):
signature-keyed dirty tracking (`stageSceneUpdate` `:1454`,
`stageLivePrimitiveUpdates` `:1524`), GPU topology mutation for newly covered
bricks (`lib/webgpu-sparse-brick-topology-mutation.ts`), structural finalize,
then the node-mip + tetrahedral-radiance incremental builder — **which already
reads the scene lanes** (`lib/webgpu-svo-live-derived-builder.ts:289–:303`
merges fluid + scene coverage per page).

**The voxel-resolves-the-surface primary already exists, switched off.** The
macro-hdda traversal variant `traceLeafPayloadFineInterval`
(`lib/webgpu-svo-dry-scene.ts:2140–:2144`) maps a voxel's owner id to its
primitive record and calls `primitiveHit` over just `[entry, cellExit]` — the
voxel→owner→exact-hit hybrid. The default `traceLeafPayload` (`:3804`)
deliberately ignores owners ("live analytic geometry is authoritative in
traceScenePrimitives").

**The payload lanes are safely separable.** Scene lanes never alias fluid
writes (invariant comment `lib/sparse-brick-octree.ts:624`); per-voxel
identity is `(ownerId<<16)|materialId`; 56 B/voxel across all lanes.

What is genuinely missing (this defines W2/W3):

1. **The voxelizer's shape vocabulary is not the render ABI.** It has its own
   6 shapes; adapters lossily collapse render records —
   **`smooth-union-cluster` voxelizes as its solid envelope ellipsoid**
   (`webgpu-sparse-scene-proxies.ts:130`), `round-cone` → capsule,
   `rounded-cylinder` → cylinder, `terrain-heightfield` → throw (`:100`). The
   bonsai crown — 65 % of hero distance-function evaluations — is an ellipsoid
   blob in voxel space today. The full ABI (`svoPrimitiveWGSL`,
   `lib/svo-primitive-abi.ts:2170`, incl. `svoPrimitiveDistance_m` and all
   cluster field variants) is binding-free and **already compiles and links in
   a compute pass** (`lib/webgpu-svo-brick-raster-probe.ts:155`) — a drop-in.
2. **No per-frame voxelization budget** — a publication voxelizes in one go.
   (The fluid path already has the needed pattern:
   `lib/webgpu-fluid-brick-residency.ts` worklist/retire.)
3. **Super-linear maintenance loops**: invalidation is
   O(leafCapacity × dirtyRegions) (`:658`), binning is
   O(dirtyBricks × primitiveCount) (`:704`) — fine at 501 records, hot at 5,000.
4. **Cell-centre sampling loses thin features** (≤1.5 cells, flagged in
   `lib/svo-scene-coverage.ts:129`) — conservative coverage is new work.
5. **No screen-space-error / near-field band policy.**
   `projectedSvoNodeFootprintPixels` (`lib/svo-screen-space-termination.ts:57`)
   is the right primitive with the wrong consumer.
6. **Dry scenes have no terrain voxels.** `terrainColumnHeights` +
   `terrainCellSolidFraction` (`lib/terrain.ts:218,:235`) bake terrain into
   voxels — but only through the solver's dense `solidCells`; a dry scene
   traces terrain analytically scene-wide (`traceTerrain`, `:3669`) and the
   pyramid never sees it.

---

## 4. Target architecture

One visibility architecture, one depth authority:

1. **Voxel content is the default.** Authored SDF records are voxelized
   (ABI-true, incremental, budgeted) into the octree's scene lanes and
   resolved by the existing brick coverage/resolve raster. Authored complexity
   is paid at voxelization time, amortized over frames — never per pixel per
   frame. Where crispness matters, the voxel's owner id upgrades the hit to an
   exact primitive intersection over one cell's interval (the existing hybrid).
2. **A bounded near-field analytic band.** Records whose projected voxel error
   exceeds a screen-space threshold (a small, camera-local set by
   construction) stay analytic — but drawn through the *unified*
   coverage/resolve arena (W1), never the direct fragment.
3. **Lighting stays on the node-mip pyramid** (Teardown's global-volume move,
   already built here). ABI-true voxel content upgrades lighting for free:
   today cones see cluster *envelopes*; after W2 they see the fissured
   interiors.
4. **The direct-fragment raster survives only as the overflow fallback.**

## 5. Workstreams, with complexity mapping

Effort scale: S ≈ days, M ≈ 1–2 weeks, L ≈ several weeks. Each workstream
lists its asymptotic cost model — the thing that must be true for 10× to hold.

### W0 — Acceptance scene + gates  (effort S · risk low · deps none)

- Author a ~5,000-record stress scene (`hero-garden-hose-x10` or tiled
  variant). Raise **both** 4096s so it draws at all:
  `SVO_SCENE_DEFAULT_MAXIMUM_PRIMITIVES` (`lib/svo-scene-primitives.ts:36`,
  loud RangeError) and `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES`
  (`lib/svo-primitive-candidates.ts:20`). Expect the arena-bytes throw
  (`Live scene primitive arena capacity exceeded`,
  `lib/webgpu-svo-dry-scene.ts:6497`) next; raise
  `SVO_PRIMITIVE_CANDIDATE_ARENA_SIZE_BYTES` with it.
- Wire a paired, interleaved benchmark lane. **Profiling traps (measured):**
  use the GPU-timestamp lane — the `PROFILE_SECONDS` wall lane includes
  ~11 ms/frame of Dawn `encoder.finish()`; xctrace needs `--launch` (never
  `--attach`) plus `FLUID_WEBGPU_DAWN_FEATURES=use_user_defined_labels_in_backend`
  or every pass exports unlabeled.
- **Cost model:** scene build O(n log n); this workstream only removes cliffs.
- **Gate:** the scene draws (however slowly) and the lane reports per-pass
  GPU timestamps.

### W1 — Unify the SDF raster onto coverage/resolve  (effort M–L · risk medium · deps W0)

The "Teardown-style cheap visibility" ask, using the in-house arena as the
conservative-depth substitute (§2.3).

- **Coverage:** proxy raster (same 36-vertex instances, back-face cull)
  appends `(instanceIndex, tEnter)` to a per-pixel arena via `atomicAdd`. No
  depth write, no shading, no field evaluation.
- **Resolve:** one full-screen pass: seed the running best with the depth the
  brick resolve already wrote (requires binding `hardwareDepth` as a *sampled*
  texture — the depth plane is fully authoritative by then, reversed-Z
  `greater`, but is currently only ever an attachment on this path); walk
  candidates front-to-back by `tEnter`; march each only over
  `[tEnter, best]`; stop when the next candidate begins behind the best hit.
- **Overflow:** re-run today's direct fragment for flagged pixels only —
  the existing brick-overflow pattern.
- Fold the rigid-impostor pass (12-body `rasterRigid`, trivial cost) into the
  same arena — it is the sole blocker forcing `rasterRigidActive` and thereby
  stationary-reuse off on raster-primary
  (`lib/webgpu-svo-dry-scene.ts:4978–:4979`, `:7232`).
- **Constraints discovered at HEAD:**
  - The brick resolve's `visited` set is a **u32 bitmask** (`:2694`) — hard
    cap 32 entries/pixel. Hero p99 covering proxies is **33**. Use a two-word
    mask or an extraction-sort; do not copy the pattern blind.
  - Arena memory: the brick coverage buffer is w·h·96 B ≈ **190 MB at
    1600×1240** for 24 entries. A second arena for the SDF set at 40 entries
    would be ~317 MB. Options: share one arena (bricks and SDF proxies are
    disjoint passes), size to p90 with overflow doing more work, or run at
    `resolutionScale`.
- **Cost model:** pixels × min(covering proxies, arena cap) *entry tests*, but
  marches only until first confirmed hit — cost tracks *visible* surface, not
  record count. This is the property Teardown's checkpoints approximate and
  the arena guarantees.
- **Expected effect (measured bracket):** 1.75×–3.15× on the 346 ms pass →
  110–200 ms at 1×. Not sufficient for 10× — W1 is the *structural
  prerequisite* for W3's band, and a big absolute win on its own.
- **Gate:** image parity under `tests/webgpu-svo-primitive-exact.test.ts`;
  hero SDF visibility ≤ 200 ms; overflow rate < 1 % of covered pixels;
  stationary reuse engages on raster-primary with bodies present.

### W2 — ABI-true incremental voxelization  (effort L · risk medium-high · deps none, parallel to W1)

The load-bearing bet. Upgrade `webgpu-sparse-scene-proxies.ts`, don't rebuild it.

- **Vocabulary swap:** include `svoPrimitiveWGSL` in the voxelizer and
  evaluate real records (64 B stride, cluster arena bound) instead of the
  6-shape downgrades. Kills the cluster→ellipsoid collapse; the precedent for
  ABI-in-compute is `webgpu-svo-brick-raster-probe.ts:155`.
- **Growable node-mip address plan — prerequisite, not cleanup.** Any edit
  that activates a derived page outside the address plan fixed at first
  publish sets `liveDerivedAddressPlanValid = false` and **nulls the opacity
  pyramid and radiance atlas** (`lib/webgpu-octree-sparse-bricks.ts:1576–:1591`)
  — cones silently fall back to exact traversal (the known 15× cliff, this
  time with no console warning; only `lightingVisibilityStatus` reports it).
  Incremental voxelization activates pages continuously, so W2 trips this by
  construction. The plan must grow (or re-plan with hysteresis) instead of
  invalidating.
- **Per-frame budget:** cap voxelized bricks per frame with a
  residency-style worklist (`GPUFluidBrickResidency` pattern); publications
  converge over frames like the light cache. Preserve the completion contract
  (`SPARSE_SCENE_MAINTENANCE_INCOMPLETE_OVERFLOW` — a documented regression
  once painted a black slab across the garden when overflow was mis-treated).
- **Sub-linear maintenance:** replace the O(leaves × regions) invalidation and
  O(bricks × records) binning with scatter-from-record-bounds or a coarse
  grid/BVH over records. Reuse `svoPrimitiveSweptBounds`
  (`lib/svo-primitive-motion.ts:235`) for moving-record dirty regions.
- **Conservative thin-feature coverage:** cell-centre + planar fraction loses
  ≤1.5-cell features; add conservative sampling (min over corner/face samples,
  or interval bound) where `svoClusterFeatureRadius_m` flags risk.
- **Cost model:** voxelization O(dirty voxels × binned candidates), amortized;
  steady state on a still scene is **zero**. Render cost of voxelized content
  is the brick raster's — already 3.0 ms at 40k leaves, scaling with resident
  *visible* pages, not records.
- **Gate:** still scene → zero maintenance dispatches; a teleported record
  converges in ≤ N budgeted frames (authored N); pyramid never withdraws
  across the 10× scene's full edit script; voxelized bonsai crown is fissured,
  not a blob (visual + occupancy-mask assertion).

### W3 — Voxel-resolved primary + near-field analytic band  (effort M–L · risk medium-high · deps W1+W2)

- Promote the owner→exact hybrid (`traceLeafPayloadFineInterval` mechanism) to
  the default primary so voxel hits resolve exactly where a record is
  resident; the coarse voxel is the accelerator, the analytic surface the
  authority — resolution-independence is retained where it is visible.
- **Screen-space-error policy:** per record (cluster-granular), compare
  projected voxel size (`projectedSvoNodeFootprintPixels`) against an
  *authored* threshold. Records failing it (near camera) stay fully analytic
  through W1's arena; everything else stops being drawn analytically at all.
  Band membership gets hysteresis + a hard budget (arena cap is the
  backstop). Optionally densify octree depth locally near the camera under an
  explicit page budget instead of growing the band.
- **Cost model:** frame cost ∝ pixels × visible surface + band size. The band
  is camera-local by construction — this is what makes record count ×10 flat.
- **Gate:** hero ≤ 33 ms with band active; frame cost < 1.3× when authored
  record count is scaled 10× behind an unchanged camera; no visible silhouette
  degradation at authored thresholds (side-by-side against the analytic
  reference).

### W4 — Cliffs → slopes  (effort M · risk low, breadth not depth · deps W0)

Every fixed capacity that lies between here and 10×, with its failure mode:

| Arena | Location | Value | Failure mode |
|---|---|---|---|
| Scene records | `svo-scene-primitives.ts:36` | 4,096 | loud throw |
| Candidate BVH leaves | `svo-primitive-candidates.ts:20` | 4,096 | loud (unreachable — above fires first) |
| Candidate arena bytes | `svo-primitive-candidates.ts:60` | 786,368 B | loud throw |
| Live-scene voxelizer records | `webgpu-octree-sparse-bricks.ts:225` | 4,096 | ~~overflow flag~~ **hard throw at construction** (§0.3) |
| Candidates/brick | same `:227` | 64 | ~~overflow flag (image-preserving)~~ **silent drop; 442/64 at 10×** (§0.5) |
| Mutation bricks/publication | same `:228` | 4,096 | bounded, queues |
| Per-pixel brick arena | `webgpu-svo-brick-raster.ts:61` | 24 (hard cap 32 — u32 mask) | silent-by-design overflow |
| Coverage candidate buffer | derived | ~190 MB @ 1600×1240 | silent memory pressure, **unaudited** |
| Brick node index mask | `webgpu-svo-brick-raster.ts:34` | 2²² | ~~silent aliasing past 4.19 M leaves~~ **silent drop past 4.19 M nodes** (§0.4) |
| Node-mip address plan | `webgpu-octree-sparse-bricks.ts:1576` | fixed at first publish | **silent 15× lighting cliff** (W2 prerequisite) |
| Materials | `webgpu-svo-dry-scene.ts:254` | 8,192 | loud |
| Cluster blocks | same `:292` | 1,024 | loud |
| Glass panes | `svo-scene-glass.ts:26` | 256 | loud |
| Rigid bodies | `webgpu-svo-rigid-raster.ts:12` | 12 | loud |

- Rule: overflow must degrade (more analytic → coarser voxel → overflow
  fragment) rather than stop drawing or silently alias. Add loud tripwires to
  the three silent rows.
- Also: give the brick overflow pass an overflow-driven indirect count — its
  vertex stage re-runs the full instance list every frame (0.841 ms today,
  ~8 ms at 10× brick count, for a pass that almost never writes). And the
  topology-mutation kernel is serial `@workgroup_size(1)` O(requests × depth)
  (`webgpu-sparse-brick-topology-mutation.ts:212`) — parallelize or budget it.
- **Gate:** the W0 scene renders correctly at every record count up to 10×;
  tripwires fire in a capacity-sweep test.

### W5 — Lighting and reuse at scale  (effort M · risk low · deps W1, W2)

- Quarter-rate cones (`coneLightingScale 0.25`, already the performance
  preset) + `stationaryPrimaryReuseEnabled` on raster-primary (unblocked by
  W1's impostor fold; note reuse also requires the cone prepass — cone scale
  ≠ 1). Prior measurement on garden: 47.6 → 14.7 ms combined.
- ABI-true voxel content upgrades every cone/shadow/GI ray for free (today
  they see cluster envelopes).
- **Terrain into the pyramid for dry scenes:** bake via
  `terrainColumnHeights`/`terrainCellSolidFraction` into the scene lanes —
  the one-pipeline destination. Note two dormant tripwires if terrain is ever
  expressed as a primitive record instead: any `terrain-heightfield`
  descriptor suppresses the candidate BVH scene-wide
  (`svo-scene-primitives.ts:333`) and throws in live proxy updates
  (`webgpu-sparse-scene-proxies.ts:100`).
- **Cost model:** ~~lighting stays ∝ pixels × cone resolution — scene-size
  independent, which is the whole point of the pyramid.~~ **Half refuted (§0.1).**
  True of the pyramid consumers (world-GI cache 1.06×, deferred lighting 2.30×);
  false of the **cone prepass, 4.93×**, whose boundary tier re-traces 47–57 % of
  prepass texels through `traceScenePrimitives` at O(records). That term, not
  quarter-rate cones, is W5's real subject.
- **Gate:** 10× scene ≤ 16.7 ms end-to-end; luminance error vs full-rate
  reference within the anatomy doc's thresholds (mean ~0.5 %, p95 bounded).

### Dependency graph

```
W0 ──► W1 ──► W3 ──► W5 (reuse)
  └──► W4          ▲
W2 (parallel) ─────┘──► W5 (lighting)
      └─ prerequisite: growable node-mip address plan
```

W1 and W2 are independent and can proceed in parallel; W3 needs both. W4 is
mechanical and interleaves anywhere after W0.

## 6. Frame budget at 10× (target)

| Subsystem | Today (1×) | Target (10×) | How |
|---|---|---|---|
| Octree primary incl. voxelized SDF | 3.0 ms | ≤ 5 ms | same path, more resident pages |
| Near-field analytic band | 346 ms (all 501 records) | ≤ 3 ms | W1 arena × W3 band shrink |
| Cone visibility + GI cache | 45 ms | ≤ 6 ms | quarter rate + reuse + pyramid |
| Deferred shading + rest | 11 ms | ≤ 3 ms | resolution-bound already |
| **Frame** | **405 ms** | **≤ 16.7 ms** | |

## 7. Measured dead ends — do not revisit

- **Tightening proxy boxes**: refuted (−5.2/+3.7/−24.1 ms against a 30 ms
  control spread). The pass is per-fragment-cost bound, not
  fragment-count bound.
- **Optimising the bonsai's cluster fields first**: 65 % of its evaluations
  are currently thrown away unresolved; fix occlusion before field cost.
- **Hardening the direct-fragment raster** beyond its overflow-fallback role:
  it is the path being left.
- **Sparse/virtual-texture page indirection** for trace-hot volumes:
  Claybook measured it 13 % slower; current atlases are direct — keep them so.
- **Salvaging "packet BVH" work**: nothing remains in-tree, and it was the
  fluid air-support march (reverted), never the renderer.
- **Quoting render-pass GPU timestamps on this machine** (added 2026-08-04):
  they bracket [vertex start, fragment end] on a tiler that hoists vertex stages,
  so the last-encoded render pass reports ≈ the frame wall. This produced a
  confident, entirely wrong "deferred lighting scales 10.5× with record count"
  conclusion. Compute-pass timestamps are sound; render-pass rows now report
  `duration_ms: null`. The tell is a row tracking frame-wall-minus-a-constant
  across a large frame-time change. See §0.2.

## 8. Risks

- **Band blowup on macro shots** (camera in the bonsai crown): thousands of
  records analytic at once. Mitigations: authored threshold, band budget +
  hysteresis, local octree densification shrinking the band; W1's arena cap
  is the hard backstop.
- **Address-plan growth** (W2 prerequisite) touches the most stateful part of
  the sparse world; a wrong invalidation either leaks light (truncated
  pyramid) or re-trips the silent cliff. The CPU oracle
  (`svo-node-mip-cpu-oracle.ts`) is the safety net — extend it to cover
  re-planning.
- **Voxelization churn on animated records**: bounded by the budget; worst
  case is convergence latency (visible lag of voxel content behind a fast
  mover), not frame spikes. The near-field band hides it where it is most
  visible.
- **Silhouette/AA vs analytic exactness**: the owner→exact hybrid keeps
  surfaces exact where records are resident; pure-voxel regions trade
  byte-identical resolution independence for scale. The screen-space-error
  threshold is where that trade is governed — keep it authored.
- **Memory at 10×**: 56 B/voxel across lanes, pyramid pages, radiance atlas
  (16 KB/page), ~190 MB coverage buffer, candidate arena. W4's audit must
  produce totals before W2 commits to a finest-voxel size at 10×.
- **Record stride co-location**: BVH nodes share the 64 B record stride
  (`svo-primitive-candidates.ts:262`) — widening records (per-record band
  state, dirty hashes) breaks that; prefer sidecar buffers.

## 9. Reproduction anchors

- Evidence: `docs/svo-render-frame-anatomy.html` (+ `.data.js`, regenerate via
  `tmp/hero-doc-data.mjs` from `artifacts/xctrace-hero-garden-hose-2026-08-04/`).
- Benchmarks: `tools/benchmark-svo-dry-frame-gpu.ts` (env-driven; GPU-timestamp
  lane), `tools/run-svo-dry-render-smoke.ts` (already asserts per-brick
  candidate density and warns on pyramid dropouts),
  `tools/svo-dry-frame-harness.ts`. CPU censuses: `tmp/hero-proxy-coverage.ts`,
  `tmp/hero-sdf-census.ts`.
- Key code: brick coverage/resolve/overflow + scene-primitive raster in
  `lib/webgpu-svo-dry-scene.ts`; arena/BVH `lib/svo-primitive-candidates.ts`;
  ABI `lib/svo-primitive-abi.ts`, kinds `lib/svo-primitive-kinds.ts`;
  voxelizer `lib/webgpu-sparse-scene-proxies.ts`; world owner
  `lib/webgpu-octree-sparse-bricks.ts`; derived lighting
  `lib/webgpu-svo-live-derived-builder.ts`.
- Tests to keep green: `tests/webgpu-sparse-scene-proxies.test.ts`,
  `tests/svo-dry-scene-live-publication.test.ts`,
  `tests/svo-structural-render-source.test.ts`,
  `tests/webgpu-svo-primitive-exact.test.ts`,
  `tests/svo-node-mip-cpu-oracle.test.ts`; suites
  `npm run test:webgpu:svo-dry-render`, `npm run test:webgpu:hero-garden-hose`.
