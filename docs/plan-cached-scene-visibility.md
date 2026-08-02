# Factored direct-light visibility CPU-oracle experiment

**Status: Phase 0 complete — the measurement refutes Phase 1. Not wired.**

**Original goal:** shadow rays dominate per-pixel ray work (~85%, ~39 node visits/ray, cost
linear in light count). Replace per-pixel exact traces against the *captured snapshot* world with a
persistent world-space cache; keep dynamic occluders exact but cheap.

**Core factorization:** `visibility(pixel, light) = snapshotOracle[page, channel] × liveTrace × fluidCone`.
Nothing that moves is ever cached; the cached term never invalidates while snapshot topology holds.

Phase 0 was specified as a gate: *"if partials dominate, stop and rethink."* It does. The
numbers are below, the producer fixes found on the way are kept, and the consumer is not wired.

## Phase 0 — measured

`tools/report-svo-shadow-cpu-oracle-coverage.ts` bakes the certificates through the production
producer (`buildSvoShadowVisibleCpuOracleProofs`) and reports the share of receiver-page/light
channels that are provably clear of remote static occlusion.

A certificate does not remove the receiver's exact trace, it *shortens* it: every blocker
within `localTraceReach_m` of the receiver page is excused from the proof precisely because
the consumer promises to keep tracing that far. So the metric that matters is coverage
**against** that reach, as a share of the scene diagonal.

| scene | reach | as % of scene diagonal | certified overall | directional | point / area |
|---|---|---|---|---|---|
| hose-tank (1.70 m diag) | 0.28 m | 16% | 0.0% | 0.0% | 0.0% |
| | 0.55 m | 33% | 3.0% | 17.9% | **0.0%** |
| | 1.11 m | 65% | 5.5% | 30.1% | 0.0–1.4% |
| | 1.66 m | 98% | 7.9% | 32.8% | 0.5–5.2% |
| garden-hose (3.85 m diag) | 0.69 m | 18% | 5.7% | 11.4% | **0.0%** |
| | 1.39 m | 36% | 27.0% | 46.0% | 8.1% |
| | 2.08 m | 54% | 52.9% | 66.5% | 39.4% |

**Read:** there is no regime with both a useful hit rate and a cheap local trace. Coverage only
climbs as the mandated local trace approaches the length of the ray it was meant to replace —
at 54% of the scene diagonal the "cache" has bought a ~2× shorter ray, not a lookup. And for
point/area/rectangle fixtures — the lights whose *count* actually drives frame time, per the
8→1 observation that motivated this — coverage is 0.0% at every affordable reach.

**Why.** Page granularity is the binding constraint, in three compounding ways:
1. Blockers are page AABB covers of the geometry, not the geometry, so a 0.2 m page of a thin
   railing occludes like a 0.2 m block.
2. The receiver is a page too, so the Minkowski hull test is inflated by the receiver's own
   half-extent — the test tube is two pages wide.
3. A furnished room has something within one page of nearly every light path.

**OCCLUDED certificates cannot rescue it.** Measured separately: only 2–5% of pages are fully
solid, and a full-blockage proof needs the blocker eroded by the receiver's half-extent, which
empties a page-sized blocker entirely. Yield was 0.0% on every scene and light. A page-sized
blocker can never prove full coverage of a page-sized receiver.

## What was kept

Two real defects in the (already-existing, never-wired) producer, found by measuring it:

- `finiteLightBeamIntersectsAabb` tested the **union AABB** of receiver and emitter, which for a
  floor page under a ceiling fixture spans the whole column between them — every intervening
  page vetoed, so finite lights certified nothing at all. Replaced by
  `svoFinitePageBeamIntersectsAabb`, the exact Minkowski hull test (the hull lies inside the
  centre segment grown per axis by `max(receiverHalf, emitterHalf)`). 0.0% → 3.1% at the reach
  where it was measured.
- The local carve-out used strict AABB **containment**, which for same-level pages can only ever
  match the receiver's own page — so all 26 neighbours vetoed every proof, and the producer
  certified ~0% everywhere. Replaced by an explicit, published `localTraceReach_m` with a
  distance test (`maximumSeparation_m <= reach`) that is sound for *any* shaded point in the
  page. Default is two page diagonals: the smallest reach that owns the full 26-page
  neighbourhood, since a corner neighbour's farthest point is exactly two diagonals away.

`svoStaticShadowLocalTraceReach_m` and its WGSL twin `svoStaticShadowLocalTraceWGSL` are
compared on-device in `tests/webgpu-svo-shadow-field-cpu-oracle.test.ts` — a disagreement in either
direction is a light leak.

## Separate finding, now fixed: hose-tank overflowed the node-mip directory

While measuring: **hose-tank requests 10,361 node-mip pages against a production cap of 8,192**
(`maximumDirectoryPages = min(8192, maxTextureDimension2D)` in `webgpu-octree-sparse-bricks.ts`),
so ~1,600 base pages of snapshot geometry were dropped. `dryNodeMipAt` returns `valid = 1u` with a
**zero sample** for a non-resident page, so the cone marcher read dropped geometry as empty air
rather than falling back — a silent light/GI leak over ~20% of the scene's static pages, in the
scene the ray-work HUD was captured from.

Fixed in two parts:

- **The cap was arbitrary.** The sampled directory is two texels wide with one *row* per page,
  so the only real ceiling is `maxTextureDimension2D` — 16,384 on Apple, twice the hard-coded
  constant. Capacity now comes from `webGpuSvoNodeMipMaximumPages(device)`, with a 2,048 floor
  so a device reporting no limits stays usable. hose-tank's 10,361 pages now fit whole:
  `omittedBasePageCount` 1,608 → **0**.
- **Overflow no longer renders the wrong picture.** `plan.complete` describes the *truncated*
  plan, so it could never report the loss; publications now carry `requiredPageCount`, and the
  renderer declines to build the pyramid at all when any base page was dropped, warning with
  both numbers. That costs the cone accelerator and GI and falls back to exact traversal —
  slower, but showing the geometry that was authored.

Still open: a scene genuinely larger than the device limit now loses the accelerator entirely.
The principled answer is to coarsen the static lighting lattice by one level (8× fewer base
pages, same coverage, conservative because the mip max lane only grows) instead of dropping
pages. That needs a node-mip base cell size separate from `dry.mapping.cellSize`, which the
shader currently assumes are the same — a contained ABI change, not attempted here.

## Where the win probably is instead

The measurement points away from world-space caching of the *answer* and toward the two things
page granularity cannot fix:

1. **Raise the directory cap / coarsen the static lighting lattice** so hose-tank fits. Fixes the
   leak above, and is a prerequisite for trusting any cone result in that scene.
2. **Shared traversal entry points per tile** (Laine–Karras beam optimisation) rather than
   per-receiver caching. Shadow rays from an 8×8 tile toward one light are near-parallel and
   co-located; descending once per tile and starting per-pixel traces at the deepest enclosing
   node attacks the ~39-node descent directly, and — unlike a certificate — its benefit does not
   depend on the scene being uncluttered.
3. **Cone-trace secondary lights.** The cone path already exists and is ~36 taps; the exact path
   is the fallback. Establishing why any given receiver takes the exact fallback is cheaper to
   answer than building a cache, and would confirm or kill the 8→1 light scaling directly.

## Non-goals (unchanged)

Screen-space/temporal shadow caching (cannot causally detect moving-occluder shadows on static
surfaces); GI changes (cones already amortized); light culling/importance sampling.
