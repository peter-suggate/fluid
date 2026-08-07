# Depth-3 exact render: the 10–100x program

Goal: 10–100x the exact-shaded SVO render at refinement depth 3 on
`hero-garden-hose`. Investigation opened 2026-08-06 at HEAD `bc6d987`.

**Status: 6.8x landed, then a further 1.78x on the exact arm. The original
diagnosis in this doc was wrong and is corrected below — read §0, then §0b,
before anything else. §0b is where the frame actually was: not in the traversal
but in the smooth-union cluster's shading normal, which was 55 % of a depth-3
frame.**

## 0. What the measurements changed

This doc originally concluded that the depth-3 wall was the exact per-cell voxel
DDA (`traceLeafPayload`) plus the coverage resolve, and ranked its plan against
that. **It was the wrong target.** The DDA is *inside* the expensive pass, but
what made the pass expensive was the number of proxies feeding it.

`raster-primary` emits one 36-vertex box proxy per occupied leaf brick. At depth
3 that is **873,700 proxies for 368,000 pixels — 2.4 per pixel, 31 M vertices,
10 M triangles**, every one of them a few pixels across. Hardware hidden-surface
removal has nothing to give when the occluders are smaller than the pixels they
would reject. The raster depth sweep is an *area* law, not a pixel law:

| refinement depth | occupied bricks | `raster-primary` |
|---|---:|---:|
| 0 | 2,729 | 88.5 ms |
| 1 | 17,284 | 90.7 ms |
| 2 | 120,069 | 186.9 ms |
| 3 | 873,700 | 428.1 ms |

fitting `t ≈ 88 + 0.038·bricks^(2/3)`.

`canonical-parametric` — the megakernel, which rasterizes no proxies at all —
was already in the tree, already the secondary-ray path, and had never been
timed past depth 0.

### The result

**Re-measured 2026-08-06 on a clean lane.** The first sweep ran under external
GPU contention; see §6. Interleaved, scene-source fingerprinted, hashes constant
within every arm:

| depth | leaf bricks | `traced` | `raster` | ratio |
|---|---:|---:|---:|---:|
| 0 | 16,820 | 24.26 ms | 50.46 ms | 2.08x |
| 1 | 29,720 | 27.12 ms | 69.57 ms | 2.57x |
| 2 | 81,850 | 43.12 ms | 148.03 ms | 3.43x |
| 3 | 267,733 | **56.14 ms** | **246.12 ms** | **4.38x** |

**No crossover at any rung**, so the unconditional default holds
(`resolveSvoPrimaryTraversal`; `FLUID_SVO_PRIMARY_TRAVERSAL=raster` forces the
old arm). Three corrections to what this section used to claim:

- **The headline is 4.4x, not 6.2x.** Contention inflates the *longer* arm more
  (a 246 ms frame has ~4x the collision window of a 56 ms one), so it biased the
  ratio and not merely the precision. The old ladder read 3.4 / 3.1 / 4.6 —
  **non-monotonic**, which was the tell.
- **`raster` did not get faster; the scene got smaller.** Depth 3 now publishes
  267,733 leaf bricks, not 873,700. The area law below predicts 246.0 ms at that
  count and measured 246.12. The model is intact; only the operating point moved.
  **Distrust any absolute number here that does not carry its brick count.**
- **"`traced` is nearly flat in refinement depth" is false** — 24.26 → 56.14 is a
  2.31x rise across the ladder.

`raster`'s fixed term is still the vertex tax, paid identically at 368 k and 4 M
pixels, so more pixels *amortize* it — which is why a "proxies per pixel"
predicate steers the wrong way at production resolution and was abandoned.

Image cost: **321 of 368,000 px (0.087%) differ by >8/255**, mean abs
0.054/255, on sub-voxel silhouette edges — `raster` additionally runs
`encodeScenePrimitivePrimary` where the megakernel returns at the first voxel.
Visually indistinguishable.

### What else the differencing sweep refuted

- **The near-field band is worth nothing at depth 3.** Band 3 = 428.1 ms; band
  4096 — every record analytic, the band's own worst case — = 403.7 ms. Inside
  the ±13% lane spread and in the wrong direction. Default returned to 0. The
  earlier "byte-identical, worth 20–25 ms" reading was a depth-3 arm against a
  scene since replaced, and the saving did not survive re-measurement.
- **Lighting is 12%** (shadows+AO off: 428.1 → 378.0 ms). Compute passes total
  1.6 ms. Neither is a lever.
- **`macro` brick occupancy remains a +33% regression** — per-step mask tests
  cost more than the skipped cells save at this lattice. `bounds` (span clamp,
  no per-step test) is the surviving shape and measured −5.9%, but it moves the
  image, so it is available and not defaulted.

## 0b. The frame is the shading normal — attribution, 2026-08-07

Item 2 of §5 ("re-attribute the 56.1 ms frame") is discharged, and the answer was
not in the traversal at all. Opened by two observations from Peter: depth 3
barely costs more than depth 2, and RAW is far faster than EXACT. **They are the
same fact.**

Interleaved arms, `hero-garden-hose`, 800x460, 12 serialized encodes each,
depth-honest smoke lane:

| arm | depth 2 | depth 3 |
|---|---:|---:|
| `analytic` (EXACT) | 32.93 ms | 42.74 ms (repeat 42.34) |
| `voxel-face` (RAW) | 17.72 ms | 18.75 ms |
| depth 3 over depth 2 | — | **EXACT +29.8 %, RAW +5.8 %** |

**3.3x the leaf bricks (81,850 → 267,733) buys 5.8 % under RAW.** The primary
marches to the first solid voxel, so its cost is a per-pixel law and refinement
moves it barely — which is the honest answer to "why is depth 3 nearly free".
Nine tenths of the depth-3 *increment* (9.81 of 10.84 ms) was inside the shading
normal rather than the DDA.

Attribution, by deleting one component at a time and answering with the entered
face (`SvoDryOptimizationExperiments.analyticNormalProbe`, default off; the smoke
lane exposes it as `FLUID_SVO_DRY_SMOKE_NORMAL_PROBE`). Depth 3:

| probe — what it deletes | frame | worth |
|---|---:|---:|
| smooth-union **cluster** gradients | 19.10 ms | **−23.4 ms (−55 %)** |
| all marched-kind gradients | 16.11 ms | −26.4 ms (−62 %) |
| field-program **tape** gradients | 40.83 ms | −1.7 ms (−4 %) |
| terrain heightfield gradient | 44.38 ms | 0 (wrong sign, inside noise) |
| the fallback trilinear stencil | 42.19 ms | 0 |

The scene publishes 940 records, of which **631 are smooth-union clusters** (512
tapered-sweep, 119 lattice) and 145 are field programs; 99 % of the set is a
marched kind. A lattice cluster folds 64 jittered spheres per octave, and
`svoClusterLocalNormal` was spending **four** of those evaluations per shaded
pixel on a tetrahedral difference.

Two smaller readings from the same sweep, both real and both unspent:

- **`trilinear` (SHADED) was 28.05 ms** — cheaper than `analytic` was, because
  the stencil is eight payload loads against four tape evaluations. That is the
  "+13 % on field programs" note in `trilinear-normal-is-the-artifact` seen from
  the other side, and it is no longer true after the fix below.
- **The unused arm costs the arm that ships.** `voxel-face` with the marched
  gradient *compiled out* measured 17.59 ms against 18.75 with it compiled in and
  never called — ~6 % of dead register pressure, because the reconstruction mode
  is a runtime uniform and the compiler budgets for the worst path. Specializing
  the pipeline per surface mode would recover it.

### The fix: differentiate the fold instead of sampling it

Every leaf of a cluster field is a sphere, an ellipsoid or the convex hull of two
spheres, each with a closed-form normal, and both combinators above them pass a
gradient through **exactly**: a polynomial smooth minimum's blend weight *is* its
derivative (the `h`-derivative terms cancel identically), and the envelope `max`
passes the winner through. So the whole field differentiates in one pass.
`svoClusterLocalNormal` and its CPU twin now do that; `svoClusterSample` and the
`...Sample` twins beside each distance function are the differentiated fold.

Interleaved against `analyticNormalProbe: "cluster-tetrahedral"`, which restores
the retired four-tap path so this stays measurable:

| depth | four-tap control | differentiated | |
|---|---:|---:|---:|
| 2 | 35.47 ms | **20.42 ms** | 1.74x |
| 3 | 43.95 ms | **24.66 ms** | 1.78x |

At the size the browser actually renders (2488x1256 — a 1728x872 CSS viewport at
DPR 2, times `resolutionScale` 0.72, which is **8.5x this lane's pixels**):
depth 1 105.50 → **82.48 ms**, and the rung ladder is then 82.48 / 96.28 /
113.36 ms for depth 1 / 2 / 3, or 12.1 / 10.4 / 8.8 FPS.

Image cost: **4.93 % of pixels differ at all, 0.043 % by more than 8/255, mean
abs 0.087/255** — visually indistinguishable, and in the direction of *more*
exact, since the four taps were a difference over a step
`svoClusterFeatureRadius_m` had to guess at. The settled hash moves
(`0x66222bda` → `0x14a91de6` at depth 3); all 21 smoke checks pass on both arms,
and `tests/svo-cluster-gradient.test.ts` holds the published normal to a central
difference of the published distance, for all three fields.

Nothing outside shading moved: the voxelizer mirror
(`webgpu-sparse-scene-proxies.ts`) and the scenery builder read
`sampleSvoPrimitive(...).signedDistance_m` only, and the brick census is
identical across the pair.

**What is left of the EXACT tax**: depth 3 is now 24.66 vs RAW's 18.75 (+32 %,
was +128 %); depth 2 is 20.42 vs 17.72 (+15 %, was +86 %). Of the remaining
~5.9 ms, ~1.7 ms is the field-program tapes — forward-mode differentiation of the
tape is the same trick one level up, and is the next item if this is still the
wall.

## 1. Landed

Image-exact work, verified byte-identical (`0xfc086075` on both arms) and worth
−8.3% on the raster path before the traversal switch:

- **Fused leaf node loads.** `traceLeafPayload` issued three independent
  `svoNodeLoad` of the same 32 B record (8 scalar loads each) per leaf visit, up
  to 48 visits deep. Now one. Same for the shadow and macro-HDDA twins.
- **Span-clamped DDA cell range** (`bounds`): the walk is clamped to the
  occupied sub-box the occupancy word already publishes, not to `[0, brickSize)`.
- **Frustum-culled analytic proxies**, 8-corner conservative test transcribed
  from the brick emit kernel, failing open on a degenerate camera basis.
- **Both O(n²) coverage extractions linearised.** The brick resolve re-ran a
  full ray/AABB intersection *and* re-fetched a 32 B instance record for every
  unvisited slot on every extraction; entries are now computed once into
  registers. The scene resolve re-read storage n² times; keys are now read once.
- **Voxel-light cache releases when switched off** — it was allocating the full
  node-mip atlas extent (386 MB floored, ~2 GB unfloored at depth 3) and the
  runtime toggle only flipped a shader word.
- **Two never-bound acceleration structures no longer built** under the shipping
  primary (compact hierarchy, wide fanout — the latter's `pages × 292 B`
  micro-mip buffer is bound by no shader at all): ~42–82 MB and a ~34 ms CPU
  plan per world build.
- **Per-level derived worklist arena** instead of largest-level capacity × every
  level: −88.6% at depth 3 (607.5 → 69.4 MiB).

## 2. Structural inventory (still true, still unspent)

| structure | today | live content | shrink |
|---|---|---|---|
| `sceneMaterialOwners` | 4 B/voxel | 91.67% of bricks hold ONE identity | banded arena, ~25x — **blocked, see below** |
| `sceneGeometry` | 4 B/voxel | fallback-only under `analytic` normals | banded arena → ~60 MB |
| node record | 8 words | word 5 redundant by construction; `address.w` bits 8..31 spare | 5-word packer exists |
| leaf record | 4 words | words 2,3 (morton): zero GPU readers | 1 word |
| coverage arena | 164 B/px (40+1 u32) — *not* the 100 B/px this doc first claimed, which was the brick stride; the allocation is sized by the wider scene-primitive stride | measured crossings p90 = 9 | halve |
| BVH node | 64 B stride | words 8–15 never written | 32 B |
| primitive record | 64 B | words 12, 14, 15 have zero GPU readers | 12 B |
| opacity pyramid floor | `maximumLevels: 1` | 64x the pages the anchor intended | lifting to 2 ≈ 5x |

**Tier 1 #1 is blocked by its own producer.** The palette pack is built and
wired (`FLUID_SVO_LEAF_PAYLOAD=banded`, with the per-leaf header hoisted out of
the DDA loop so an `indexBits == 0` leaf costs one occupancy bit test per cell).
It is not defaulted because `encodeBandedLeaves` constructs each leaf's palette
by reading back **both dense lanes the rebuild pass just wrote** — so dropping
them makes the banded arena build from the absent-lane page, and `banded` today
*adds* 1.47 B/voxel instead of saving 4. Moving the producer off the dense lanes
is the prerequisite nobody had costed.

## 3. Culling that still does not exist

- **Face-visibility / buried-brick culling: nowhere.** 88–89% of terrain bricks
  are buried at every measured rung (`lib/sparse-scene-terrain-field.ts`), and
  the depth-3 smoke census reports 94,399,321 of 99,236,796 terrain voxels
  buried — all of them voxelized, stored, and asserted full by
  `terrain-coverage-solid`. A brick whose six face-neighbours are fully solid
  can never be the nearest surface.
  - Everything needed exists: `address.w` bits 8..31 are spare (every reader
    masks `&0xffu`), the producer already scans all 512 cells so an `allSolid`
    count is free, and `dryLeafForGlobalCell` is an O(depth) neighbour descent
    that can run once at build time.
  - **But weigh it against the traversal switch first.** Under `traced` the
    octree descent already skips buried leaves spatially — it stops at the first
    solid payload hit — so this is now a *memory and build-time* lever, not a
    frame-time one. Under `raster` it was worth 56–88% of the proxy set.
  - Not to be confused with `FLUID_SVO_GROUND_SHELL`, which ships off because
    deleting the leaf deletes its mip page and darkens lit ground 5.1% / the
    shadowed basin 8.9%. Culling only the *proxy* keeps payload, mip page and
    opacity resident; that blocker does not apply.
- **Record frustum culling** — landed (§1).
- Brick cull emit+scatter dispatch over `brickLeafCapacity`: **not worth
  fixing.** The capacity is `plan.leaves.length + 4,096`, so the overshoot is 64
  workgroups, not thousands, and no CPU-visible live leaf count exists.

## 4. What `traced` gives up

Audited, with the verdict for each:

| capability | verdict |
|---|---|
| rigid bodies | **parity, traced better** — folded inline; the raster arms exist *because* raster lacks the loop |
| thin/thick glass | **parity** — inline, key packed into the opaque identity |
| coverage depth seed | **no loss** — it bounds a pass `traced` does not run |
| analytic scene-primitive tier | **lost.** The 0.087% pixel difference. `traceScenePrimitives(ro,rd,0,voxel.t,…)` already exists as a tail and is already bounded by the voxel hit — restoring it is ~1 line plus a hash re-bless |
| near-field band | lost; worth nothing at depth 3 anyway |
| screen-space LOD tier | structurally unavailable under `canonical-parametric`+split, but inert today (`lodScreenSpacePixels = 0`) |
| stationary primary reuse | **traced better** — was blocked by `rasterRigidActive`; now reachable, still gated by its own default-false flag |
| `primaryLeafVisits` (48) | **real truncation risk, silently.** Both exhaustion arms are empty statement blocks and no lane counts them. A depth-3 ray crossing 873 k bricks could be dropping surfaces with no signal |

## 5. Ranked plan from here

0. ~~**Instrument the truncation budget.**~~ **CLOSED — 48 is safe and buys
   nothing.** Depth 3 canonical at `primaryLeafVisits` 48 / 96 / 256 produces
   **byte-identical settled frames** (sha256 over the raw `rgba16float` buffers,
   not just the FNV hash) and frame times within 0.1%. A positive control at 2
   visits moves both (27.75 ms, different sha), proving the override reaches the
   shader. There is **no tail past 48**.
   But the control is the useful part: 2 visits *halves* the frame, so **~28 ms
   of the 56 ms frame is leaf visits beyond the second**, all concentrated below
   48. The cost is the **bulk** of the visit distribution, not its tail — which
   is the pool a per-brick entry-rejection test draws from. Note
   `primaryLeafVisitHistogram` / `SVO_DRY_PRIMARY_VISIT_HISTOGRAM_CONTRACT` in
   `webgpu-svo-dry-scene.ts` are **declared but never implemented**; the doc
   comment reads as a live diagnostic and it is scaffolding only.
1. **Per-brick contour (entry rejection).** A conservative slab per leaf brick —
   normal from the owning primitive's analytic gradient at build time, offsets
   over solid cell *extents* so it is hole-free by construction — tested once at
   brick entry to clamp `[entry, brickExit]` and reject empty chords with zero
   DDA steps. Laine-Karras contours, fitted analytically rather than to voxels.
   Storage is free: 56 spare bits in the node record the DDA already loads
   (`links.y` is dead on a leaf; `address.w` bits 8..31 are masked off by every
   reader). Producer is `finalizeDirtyBricks`, which already scans all 512 cells
   and sits in the module that binds `primitives` and splices
   `svoPrimitiveLocalNormal`. **Must be outside the dependent loop** — that is
   the whole `bounds` (−5.9%) vs `macro` (+33%) lesson.
2. ~~**Re-attribute the 56.1 ms frame.**~~ **DONE — see §0b.** It was the
   smooth-union cluster shading normal, 55% of the frame, now differentiated in
   closed form for 1.78x. The successor items §0b leaves open are the
   field-program tape gradient (~1.7 ms of what remains) and specializing the
   pipeline on the surface-reconstruction mode (~6% of dead register pressure in
   every arm, including `voxel-face`).
3. ~~**Restore the analytic tier under `traced`.**~~ **RETIRED — Peter, asked
   directly, 2026-08-06.** The 0.087% is accepted permanently. Consequence worth
   keeping: `voxelsOnlyPrimary` is permanent, so the producer's **material-only**
   solidity predicate is now the definitive contract and the shared
   conservativeness basis for `bounds` and any contour. Bringing the tier back
   would silently break both.
4. **Unblock the banded payload** by moving `encodeBandedLeaves` off the dense
   lanes (§2). This is what decides whether depth 3 *fits*, independently of
   frame time.
5. **Buried-brick culling** (§3) — now a memory/build lever; size it against (2).
6. Structural ABI diet: 5-word nodes, 1-word leaves, halve the coverage arena.

## 6. Methodology traps (do not rediscover)

- Render-pass GPU timestamps bracket [vertex-hoist, fragment-end] on Apple
  tilers; the last-encoded pass window ≈ the whole frame. Only compute-pass
  timestamps and serial-frame aggregates are trustworthy. **Attribute by
  differencing configurations instead** — that is what found everything in §0.
- **The ±13% "lane spread" was external GPU contention, not lane noise.** The
  `mkdir` lock only excludes processes that go through the wrapper — the app, a
  dev server, or a browser WebGPU context sails past it. With the machine
  confirmed idle, a canonical depth-3 arm measures **±1.1–2.8%** over 5
  interleaved pairs. Treat >10% within-arm spread as "the lane is dirty", not as
  the floor, and **do not discard small structural wins as unmeasurable** — a
  4.7% win resolved cleanly at 5/5 pairs with non-overlapping distributions.
  Contention is also asymmetric: it inflates the longer arm more, so it biases
  *ratios*.
- **Fingerprint the scene source around every run.** The scenery is live-edited;
  one 9-pair sweep saw the settled hash move four times mid-sweep and times
  inflate to 84–126 ms with >100% spread. Void any sample whose fingerprint
  differs from its pair's.
- **A poisoned lock looks exactly like work in progress.**
  `tools/run-webgpu-smoke-isolated.ts` waits on its worker and never detaches, but
  its timeout path SIGKILLs and then `unref()`s and exits 124, deliberately
  "leaving the lock as owner evidence" — orphaning a live worker (PPID 1) that
  holds the lock forever. Read `/tmp/fluid-webgpu-exclusive.lock/owner.json`; if
  the owner pid is dead the lock is stale, and if it is alive with PPID 1 nothing
  is reading its output. `tools/webgpu-smoke-isolation.ts` already has the
  liveness check.
- The GPU-lane depth flag refines the tree only, not the scenery lattice; the
  smoke lane is the depth-honest one. Do not carry GPU-lane record-tier results
  into a plan.
- The GPU lock (`/tmp/fluid-webgpu-exclusive.lock`) never queues — serialize
  every arm.

## 7. The paired-worktree method

The scene is edited while this work happens, so absolute frame times drift
between runs and several sweeps died mid-flight on in-progress scenery. Do not
stash and do not touch the primary worktree. Instead:

1. `git worktree add --detach <control> HEAD` and `<measure>` likewise.
2. Copy the changed renderer files into `<measure>` only.
3. Copy **one frozen snapshot of the scene files into both**, so they are
   byte-identical.
4. Symlink `node_modules` into each.

The renderer becomes the only variable and the lane stops moving under you. This
is what produced the byte-identical `0xfc086075` pair proving the §1 work was
image-exact, and the clean 385.1 → 353.3 → 56.8 ms chain.
