# SVO Traced-Primary Fast Path — Implementation Handoff

Status: planned, not started. Companion documents:
`docs/SVO_RASTER_PRIMARY_HANDOFF.md` (the strategic raster rework) and
`docs/SVO_VOXEL_LIGHT_CACHE_HANDOFF.md` (lighting side). This document covers
the incremental levers on the *existing traced* primary pass: a beam
prepass, a register diet, adaptive-rate rays, and (conditionally) a staged
kernel split. They are ordered by payoff-per-effort, compose with each
other, and everything built here feeds the raster plan (the beam prepass is
its HiZ seed; the unfused kernel is its fragment-shader baseline).

## 1. Baseline and thesis

`docs/SVO_GARDEN_1500_RENDER_PROFILE.md`: primary visibility is 26.95 ms at
1500×1500, pixel-linear (~13–15 ms/Mpix), **not** bandwidth bound
(3.7/78 GB/s), pinned at 22.2% occupancy / 34.6% ALU — a register-pressure
ceiling on a megakernel that fuses octree traversal + terrain secant march +
12-body rigid loop + glass loop (`traceDrySolidScene`,
`lib/webgpu-svo-dry-scene.ts:2426`).

Two consequences drive everything below:

- **Fewer/cheaper rays first.** Cost is linear in (rays × steps); a
  conservative depth prepass cuts steps, adaptive rate cuts rays. Both are
  camera-dynamic and deformation-safe because they are recomputed per frame.
- **Registers are the machine-level constraint.** Any change that lowers the
  worst-case live state of the hot kernel (f16, unfusing, specialization)
  buys occupancy across *all* of the above.

Negative results to respect (Claybook, GDC 2018 — deformable SDF world,
60 fps base Xbox One): in-kernel wave-ballot lane refill and overstepping
were both tried and rejected there ("bloats VGPR count", worse locality).
Prefer prepasses and multi-dispatch over intra-kernel load balancing. Ray
binning/reordering is for incoherent secondary rays; primaries are already
screen-coherent — do not spend effort there.

## 2. Lever 1 — beam prepass (conservative coarse t-min)

ESVO beam optimization (Laine & Karras 2010, §5): trace one conservative ray
per 8×8 tile at tile corners; full-res rays start at
`min(corner t) − safety`, skipping all empty-space descent above that depth.
Claybook's equivalent cone prepass cost 0.2 ms and left most surface pixels
converging in ≤7 steps; ESVO reports 1.15–1.4× on primaries, growing with
resolution — favorable at 2.25 Mpix.

Design:

- New small fragment/compute pass at `(⌈W/8⌉+1)×(⌈H/8⌉+1)` (corner grid)
  reusing the existing traversal WGSL with two changes: **conservative
  termination** — stop descent as soon as the current node is too small to
  be guaranteed hit by every ray in the 8×8 beam footprint (compare node
  projected size against beam width at that t) — and no payload/primitive
  intersection (topology-only; skip `traceLeafPayload`).
- Output: `r16float` (or `r32float` first, shrink later) t-min texture.
- Consumption: `traceStatic` (`lib/webgpu-svo-dry-scene.ts:2191`) reads the
  4 surrounding corners, takes the min, subtracts a safety margin scaled by
  cell size, and initializes the traversal cursor at that depth. Terrain
  secant march can use the same t-min as its lower bracket.
- Correctness invariant: the prepass may only ever *understate* depth. A
  debug assert path (behind the existing diagnostics plumbing) traces
  without the prepass for a pixel-trace probe and compares hit-t.

Gates: prepass cost ≤ 0.5 ms; primary pass step counts (pixel-trace
`mipSteps`/node-visit counters, `lib/svo-pixel-trace.ts:149`) drop ≥ 30%
median; image diff vs baseline is bit-identical (this optimization is
exact, not approximate — any diff is a bug); net primary ms −15% or better
at 1500².

Effort: ~2–3 days. Risk: LOW. The only subtlety is the conservative
termination rule; get it wrong and silhouettes clip — which the
bit-identical gate catches immediately.

## 3. Lever 2 — register diet on the hot kernel

Attacks the 22%-occupancy ceiling directly. Three independent cuts, in order
of expected effect:

1. **Unfuse the megakernel.** Move rigid bodies, terrain, and glass out of
   the per-pixel traversal shader. Rigids and terrain are rasterizable
   (proxy meshes / heightfield grid, or full-screen passes depth-tested
   against the SVO result); glass already has a raster pipeline
   (`glassRasterVertex/Fragment`, `lib/webgpu-svo-dry-scene.ts:2685`). The
   traversal kernel's register allocation stops being the max over four
   subsystems. This is also Phase D of the raster handoff — build it once,
   in whichever plan executes first.
2. **`shader-f16`.** Dawn exposes WGSL f16 on Metal; Apple's occupancy
   guidance is explicit that 16-bit values halve register footprint
   (256 half-register/thread budget shared across resident simdgroups).
   Apply to bounded-range quantities only: brick-local coordinates, DDA
   accumulators, packed G-buffer staging, normal math. Keep global ray t and
   world positions f32 (precision hazard is real at scene scale). Requires
   requesting the feature at device creation and a fallback compile.
3. **Pipeline-overridable constants** to specialize out debug/diagnostic/
   view-mode branches that currently inflate worst-case register allocation
   (the diagnostic counters were measured costing real time before —
   5X handoff §9 "diagnostic-counter stripping").

Gates (quiet-GPU `tools/profile-svo-render-xctrace.ts`, garden 1500²):
primary-pass occupancy measurably above 22.2% after each cut — occupancy is
the direct metric here, ms the confirming one. If occupancy rises but time
does not, stop and re-profile before proceeding (would indicate the latency
was hiding elsewhere, e.g. dependent texture fetch chains).

Effort: f16 + constants ~2–3 days; unfusing ~1 week. Risk: LOW–MEDIUM
(f16 precision regressions — gate with the existing image-diff lanes).

## 4. Lever 3 — adaptive-rate primary (1 ray per 2×2 on coherent tiles)

Because cost is pixel-linear, tracing a quarter of the rays where the image
is locally flat is a near-mechanical ~1.7–2× on the pass. No published
system does "ray density matched to projected voxel size" exactly (verified
gap in the literature sweep); the nearest production relatives are software
VRS and checkerboard rendering. We build the spatial variant only — no
temporal history, hence no deformation or disocclusion risk.

Design (mirrors the shipped cone-prepass machinery — this is
`dryPrepassCoherentMain`/`dryPrepassBoundaryMain`
(`lib/webgpu-svo-dry-scene.ts:1568/:1579`) applied one pass earlier):

- Classification: the beam-prepass corner grid already encodes tile
  coherence — corners agreeing in t (within a cell-scaled tolerance) and
  hitting the same brick/owner mark the tile *coherent*. Silhouettes,
  brick edges, and high-slope tiles are *boundary*.
- Coherent tiles: trace 1 ray per 2×2 quad; replicate the packed G-buffer
  pixel to the quad; write a `coarse` flag bit (G-buffer flags field,
  `lib/svo-gbuffer.ts:59`).
- Boundary tiles: full per-pixel rate via the existing path (queue +
  64-wide dispatch, same shape as the boundary re-trace pass). Expect
  10–20% of tiles at 1500² based on the cone prepass's observed
  homogeneous-quad rates; measure, don't assume.
- Downstream: the deferred lighting pass already reads the G-buffer
  per-pixel and is unchanged. Picking and pixel-trace must ignore the
  `coarse` flag (they re-trace exactly anyway).

Quality note: IDs/normals/depth do not interpolate — replication, not
filtering, and the boundary classifier is what protects edges. Thin fluid
features (droplets, sheets) are the stress case: hose-tank orbit captures
with consecutive-frame diffs are the gate, and the classifier must treat
fluid-coverage-active tiles as boundary until proven otherwise.

Gates: primary ms ≥ 35% down on garden at 1500² with boundary fraction
≤ 20%; diff PNGs show error confined to interior of coherent tiles at
sub-perceptual amplitude; no droplet flicker on hose-tank captures.

Effort: ~1 week. Risk: MEDIUM (quality). Ship behind a tuning knob like the
cone scale (`1 | 0.5-quad`), default off until the hose-tank gate passes.

## 5. Lever 4 (conditional) — staged traversal split

Only if the register diet stalls above ~30% occupancy and the raster plan
is not proceeding: split the traversal loop into 2–3 compute dispatches
(coarse node descent → compacted leaf-DDA queue → G-buffer resolve) so each
stage compiles at its own register count. Wavefront literature reports 16%
to 2×+ when per-stage register needs diverge; our unused bandwidth
(3.7/78 GB/s) makes inter-stage ray-state buffers nearly free. Keep state
minimal (`t`, node index, packed short stack). WGSL subgroups (shipped in
Dawn) provide the compaction primitives.

This is deliberately last: it is the most invasive change to the most
load-bearing shader, and both the beam prepass (fewer deep rays) and the
raster plan (no deep rays) reduce its payoff. Decide on Phase-2 numbers.

## 6. Order of execution and combined expectation

1. Beam prepass (exact, cheap, feeds everything downstream).
2. Register diet: f16 + override constants, then unfusing (shared with
   raster Phase D).
3. Adaptive rate (uses the beam prepass classifier).
4. Staged split — only on evidence per §5.

Multiplying measured-range estimates (1.2–1.4× beam, occupancy recovery
from the diet, 1.5–1.9× adaptive net of boundary cost): the 27 ms pass
plausibly lands in the **8–12 ms** range at 1500×1500, before
`resolutionScale` and before the raster rework. Each lever gates
independently; abandoning any single one does not strand the others.

## 7. Measurement protocol

Scenes: garden-svo-lighting and hose-tank at 1500×1500 (matching
`docs/SVO_GARDEN_1500_RENDER_PROFILE.md`), plus the 750²/2000² endpoints of
its resolution sweep to confirm scaling behavior. Tools:
`tools/profile-svo-render-xctrace.ts` (quiet GPU — the pass table and
occupancy are the primary metrics), `tools/benchmark-svo-dry-frame-gpu.ts`
(median/p95 + diff PNGs), pixel-trace step histograms before/after each
lever. Artifacts retained under `artifacts/render-primary-fast-path/`.
Every lever records: pass ms, occupancy, step-count histogram delta, diff
error, and a one-line verdict appended to this document's ledger.

## Ledger

| Date | Lever | Result | Verdict |
|---|---|---|---|
| — | — | — | — |

## 8. Sources

1. Laine & Karras, "Efficient Sparse Voxel Octrees", I3D 2010 — beam
   optimization: mechanism, conservative termination, 1.15–1.4× primary-ray
   speedups (Table 2), gains growing with resolution.
2. Aaltonen, "GPU-Based Clay Simulation and Ray-Tracing Tech in Claybook",
   GDC 2018 — 0.2 ms coarse prepass, ≤7-step convergence, geometric-series
   last-step refinement, and the documented rejections of overstepping and
   wave-ballot lane refill.
3. Laine, Karras, Aila, "Megakernels Considered Harmful", HPG 2013 —
   per-stage register budgets as the mechanism behind wavefront wins.
4. Apple, "Metal Compute on MacBook Pro" tech talk + GPU occupancy docs —
   256 half-register/thread budget, 16-bit register savings, constant
   address space, threadgroup sizing.
5. DICE, "Ray-Traced Reflections in Battlefield V" (GDC 2019) — variable-
   rate tracing precedent; also the evidence that binning targets
   incoherent rays only.
6. dubiousconst282, "A guide to fast voxel ray tracing using sparse
   64-trees" (2024) — modern traversal-iteration accounting (ancestor
   memoization ~2×, bitmask coalescing −21% iterations) if a
   structure-level follow-on is ever wanted.
