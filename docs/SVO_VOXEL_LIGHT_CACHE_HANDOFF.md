# SVO Voxel-Space Light Cache — Implementation Handoff

Status: **Phase 0 and Phase 1 implemented and gated (2026-08-02)**. Phases 2–4
remain planned. The implementation moves directional-light visibility from
screen-space per-frame evaluation into a persistent, runtime-populated
per-voxel cache.

This is **not pre-baked lighting or visibility**. No offline artifact is read or
published. The GPU discovers demand from the current exact split G-buffer,
deduplicates it, runs the existing live cone marcher for missing voxels, and
stores only runtime results. Source/topology/authored-primitive and relevant
march-tuning changes advance the cache epoch; rigid blockers remain live; fluid
attachment and unsupported configurations fail closed to the existing live
path. Camera motion never invalidates world-space entries.

### Phase 0/1 completion record

Quiet garden evidence is retained in
`artifacts/render-voxel-light-cache/phase0-baseline/benchmark.json` and
`artifacts/render-voxel-light-cache/phase1-cache/benchmark.json`. Both captures
are 660×662, raster-primary, split shading, one directional light, AO/GI off,
12 warmups and the same render-source fingerprint on an M1 Max/Metal backend.

| Gate | Cache off | Phase 1 warm | Verdict |
|---|---:|---:|---|
| Compact cone-visibility production pass (serialized single-pass median) | 4.009 ms | 0.044 ms (pass withheld) | **98.9% reduction; pass ≥70%** |
| Whole configured frame median | 5.046 ms | 2.949 ms | **41.6% reduction** |
| Distinct visible voxels / pixels | — | 18,209 / 436,920 | **4.2%; mapping sane** |
| Missing / queued / overflow voxels after warmup | — | 0 / 0 / 0 | **drained** |
| Cache resident memory | — | 13,824,000 B | **13.18 MiB** |
| Mean / p95 relative luminance error vs inline | 0.000773 / 0.001690 | 0.001019 / 0.003518 | boundary-localized; see PNGs |
| Maximum relative luminance error vs inline | 0.601055 | 0.250451 | **below old prepass bound** |
| Warm A→B→A orbit return | — | 0 differing packed words; 0 max error | **bit exact** |

The Phase 1 route is the default only inside its audited eligibility envelope:
split shading, node-mip/direct page table available, one shaded directional
light, AO off, no published tetrahedral GI, no fluid attachment, and sufficient
sampled-texture limits. In that tier the old screen-space cone pass is not
encoded. Missing, rejected, dirty, dynamic, or ineligible content continues
through the live cone/exact chain. `setVoxelLightCacheEnabled(false)` remains a
runtime A/B and emergency fallback switch. Raster-primary visibility is
unchanged and remains primary.

The initial `profile-svo-render-xctrace.ts` attach attempts did not acquire the
short-lived Node worker before it exited on this machine; their logs are kept
under `phase0-baseline/xctrace-attach-failed/`. The gate therefore uses a
serialized command buffer containing only the exact production compact-cone
pass, plus the existing hardware-timestamp whole-frame lane. The profiler
remains unchanged; a future Instruments capture should launch the workload
under Instruments instead of relying on late PID attachment.

## 1. Why — baseline evidence

From `artifacts/xctrace-hose-tank-render-only-2026-07-31/summary.json`
(100 frames, hose-tank, 660×662, M1 Max, GPU-contended capture):

| Pass | GPU ms/frame | Share | Occupancy | ALU |
|---|---:|---:|---:|---:|
| Sparse voxel primary visibility | 9.94 | 38.8% | 18.6% | 27.1% |
| Sparse voxel compact cone visibility | 9.29 | 36.3% | **2.5%** | 10.6% |
| Sparse voxel deferred dry lighting | 5.37 | 21.0% | 9.3% | 10.7% |
| Sparse voxel persistent world GI cache | 0.99 | 3.9% | 3.1% | 35.1% |

Lighting (cones + deferred + GI) = 15.65 ms of 25.42 ms GPU busy. The cone
pass is latency-bound (dependent node-mip pyramid taps), which is exactly the
workload a cache-lookup-plus-interpolation replaces best. Earlier measurement
(`docs/plan-cached-static-visibility.md`): shadow rays are ~85% of per-pixel
ray work, ~39 node visits/ray, cost linear in light count.

The screen-space route to cheaper lighting is exhausted: `coneLightingScale`
is already 0.5 (1/4 texels), the 0.25 and temporal-prepass variants were tried
and reverted (`docs/SVO_RENDER_5X_HANDOFF.md` §9) because screen-space
coarsening shimmers under camera motion. A world-space cache has no
camera-motion failure mode by construction.

Ceiling honesty: collapsing lighting 10× yields a ~2.2× whole-frame win;
primary visibility (9.94 ms) then dominates and is out of scope here.

## 2. The technique

Shade lighting at *surface voxel* granularity, store it in the atlas, and have
the deferred lighting pass interpolate it instead of marching cones:

1. **Populate at runtime**: for each *requested, dirty* surface voxel, run the existing
   cone march once from the voxel center and write per-light visibility into a
   new atlas lane. Static voxels compute once and persist across frames.
2. **Consume**: `dryLightingMain` replaces the screen-space prepass-plane read
   with a voxel-lane read (nearest in Phase 1; trilinear, normal-weighted and
   apron-backed from Phase 2),
   then applies the existing full-rate analytic rigid-body correction.
3. **Amortize**: steady-state cost ≈ (voxels whose lighting changed) +
   (voxels newly on screen), bounded by a per-frame budget with the live cone
   march as fallback until a cache entry fills.

Material evaluation, normals, and specular stay per-pixel — only the
illumination signal is cached coarsely. Requests are screen-driven: the split
primary pass's G-buffer names the hit voxel per pixel; dedupe that set
(~50–150K distinct voxels for ~437K pixels) and never touch the ~4.2M
non-visible voxels.

Prior art validating the shape (see §9): The Tomorrow Children (per-voxel
per-face lighting, 1/16-res evaluation + upsample, ~6 ms on PS4), Lumen
surface cache + screen probes, EA surfel GI, DDGI's leak-free lattice
interpolation, Ward 1992 irradiance gradients for sparse-point interpolation.

## 3. Inventory — what already exists and is reused

| Asset | Where | Role in this plan |
|---|---|---|
| Node-mip page machinery (8³ interior + 1 apron = 10³ physical, direct `r32uint` page table) | `lib/svo-node-mip-pyramid.ts:3`, `SVO_NODE_MIP_LAYOUT` | Page layout, slot addressing, apron fill for the new visibility lane |
| Tetrahedral radiance atlas (4× `rgb9e5` 3D textures sharing node-mip slots/directory) | `lib/svo-tetrahedral-radiance.ts`, `lib/webgpu-svo-tetrahedral-radiance.ts` | Template for "extra lane sharing the atlas directory"; later target for Phase 4 irradiance |
| Cone marcher | `dryConeVisibility`, `lib/webgpu-svo-dry-scene.ts:880/:905` | The population kernel — reused verbatim, invoked per dirty voxel instead of per prepass texel |
| Shadow shortcut + analytic rigid overlay | `prepassShadowShortcutWGSL` `lib/webgpu-svo-dry-scene.ts:1391`, `anyBodyBlockerIgnoring` `:1397`, consumed in `dryLightVisibility` `:2346` | The consumption template: cached value × live analytic body test. Rigid bodies therefore never dirty the cache |
| Boundary re-trace queue | `dryPrepassCoherentMain` `:1568`, `dryPrepassBoundaryMain` `:1579` | Pattern for the request/overflow queue and heterogeneous-voxel exact fallback |
| World GI hash cache (demand-populated, camera-independent, lighting-epoch invalidation) | `SVO_DRY_WORLD_GI_CACHE_CONTRACT` `:1024`, `dryWorldGiKey` `:1627` | Proven in-repo instance of the exact caching pattern; its invalidation triggers are the model for §5 |
| Static shadow visible proofs (page × light, CPU, conservative) | `buildSvoStaticShadowVisibleProofs` `lib/svo-static-shadow-visible-proofs.ts:287`, field in `lib/svo-static-shadow-field.ts`, GPU side `lib/webgpu-svo-static-shadow-field.ts` | **Currently unwired.** Used here as a population fast path: a (page, light) pair certified `visible` seeds visibility = 1 for the whole page without marching (see caveat §5.4) |
| Fluid coverage pyramid | `lib/svo-fluid-coverage.ts:26` | Its per-frame page deltas are the fluid dirty-set source |
| Benchmarks / profiling | `tools/benchmark-svo-dry-frame-gpu.ts`, `tools/profile-svo-render-xctrace.ts`, `lib/svo-pixel-trace.ts` | Gates in §7 |

## 4. Storage design

**New lane: per-voxel per-light visibility**, sharing node-mip page slots and
the direct page table (same virtual→physical mapping, no new directory).

- Phase 1 format: one `rg32uint` 3D texture at node-mip physical layout (10³
  per page). Channel x stores slot-zero visibility quantized to 16 bits plus
  one (zero remains the missing/rejected sentinel); channel y stores the
  16-bit runtime lighting epoch and an octahedral representative normal.
  Multi-light packing remains Phase 2 work.
- Phase 1 uses one global runtime epoch. Per-page/per-light dirty metadata
  remains a Phase 2/3 refinement; correctness currently invalidates the whole
  slot-zero epoch for authored/topology changes.
- Memory: 8 B/voxel × 1000 texels/page = 8,000 B/page. Hose-tank ceiling
  8,192 pages ≈ 65.5 MiB; garden 1,715 pages ≈ 13.7 MiB. Sits beside the
  tetra atlas's ~125 MiB ceiling — acceptable, but note it in the tuning docs.
- The physical atlas reserves the existing one-texel apron, but Phase 1 uses
  nearest sampling and does not publish apron values. Apron fill is required
  when Phase 2 enables trilinear sampling.
- Trilinear interpolation of packed `rg32uint` is manual (8 taps + unpack +
  weight). If that costs too much in the consumer, fall back to nearest tap
  first (Phase 1 gate allows it) and revisit with an `rgba8unorm`
  4-light-per-texture split only if measurements demand filterable formats.

Deliberately **not** stored: normals/materials (analytic per-pixel already),
specular (view-dependent), rigid-body occlusion (live analytic overlay).

## 5. Population, invalidation, consumption

### 5.1 Request flow (per frame)

1. After the split primary pass, a small compute pass walks the G-buffer,
   maps each hit to its level-0 voxel, and marks "requested" bits in a
   per-page bitmask (atomic OR; no dedup queue needed — the bitmask *is* the
   dedup).
2. A scheduling pass turns (requested ∧ dirty) into a bounded work queue
   (reuse the `dryPrepassResetMain`/`dryPrepassBoundaryMain` queue mechanics,
   `lib/webgpu-svo-dry-scene.ts:1565/:1579`). Budget: start at 16,384
   voxels/frame (tuning knob in `lib/svo-render-tuning.ts`); overflow stays
   queued and the consumer falls back for those voxels.
3. Population kernel (compute, 64-wide over the queue): per voxel × per
   light, `dryLightSample` + `dryConeVisibility` from the voxel center offset
   along the stored-free analytic direction (see 5.3), write packed lane +
   clear dirty bit.

### 5.2 Invalidation triggers (mirror the world GI cache's discipline —
camera motion never invalidates)

- **Lighting epoch** (light moved/added/recolored): dirty all resident pages
  for that light channel only (per-light dirty bits make this cheap).
- **Fluid**: pages whose fluid-coverage pyramid content changed this frame
  dirty themselves *and* pages within the cone-support radius along each
  light direction. First implementation: dirty the coverage-changed page set
  dilated by 1 page; measure leakage of stale shadows on hose-tank before
  getting cleverer.
- **SVO topology edits** (destruction, primitive changes): the existing scene
  epoch dirties affected pages; same channel as tetra-radiance invalidation.
- **Rigid bodies**: never dirty the cache. Their occlusion is applied live in
  the consumer via `anyBodyBlockerIgnoring`, exactly as the screen prepass
  does today. This is the single most important design carry-over.

### 5.3 Consumption (`dryLightingMain` / `dryLightVisibility`)

Insert a new tier *above* the current prepass shortcut in `dryLightVisibility`
(`lib/webgpu-svo-dry-scene.ts:2346`):

1. Resolve the pixel's voxel + fractional position via the direct page table.
2. If the 8 trilinear neighbors are cached-and-clean: manual trilinear of the
   per-light channel, weighted by `max(0, dot(N, neighborOffsetDir))`-style
   backface rejection (DDGI-lite; full Chebyshev weights only if leaks are
   observed — see §8), then × live rigid overlay × `dry.tuningRays0.y` mix,
   matching the existing shortcut's contract.
3. Else: fall through to today's path (screen prepass plane if present,
   `dryConeVisibility` live, `svoTraceVisibility` exact fallback) — the
   fallback chain already exists and stays intact. Quality is therefore never
   worse than today, only cheaper where the cache is warm.
4. **Heterogeneous guard**: voxels whose node-mip texel is "mixed"
   (`solidMaximum` high while `solidMean` low, thresholds to be tuned) are
   never trusted from the cache — permanent live fallback, the same
   philosophy as `dryPrepassBoundaryMain`'s exact re-trace.

### 5.4 Static-proof seeding (optional accelerant, Phase 2)

`buildSvoStaticShadowVisibleProofs` certificates are hard-ray, zero-aperture
(`maximumCertifiedConeApertureRadians: 0`,
`includesNodeMipSamplingSupport: false` — `lib/svo-static-shadow-visible-proofs.ts:44-48`),
so they **cannot** stand in for a cone result near occluders. They *can*
short-circuit population for fully-visible (page, light) pairs — pages the
beam test proves unoccluded get visibility = 1 written without marching,
minus the local-trace obligation (`svoStaticShadowLocalTraceReach_m`), which
the population kernel honors by still cone-marching the first page-diagonal
of distance. This finally gives the proofs a consumer; measure hit-rate with
`tools/report-svo-static-shadow-coverage.ts` before building it.

## 6. Phased plan with gates

Branch discipline: per repo policy, no stash/checkout/reset in this worktree;
work directly on a feature branch cut from `perf/structured-cutover`'s tip.

### Phase 0 — instrumentation — **complete**
- Added frame-wide GPU counters for distinct hits, misses, cache hits, queued,
  populated, rejected, overflow and resident bytes. They are copied directly
  from the demand queue by `copyVoxelLightCacheCounters`; this intentionally
  supersedes the proposed pixel-trace header because these are frame-wide
  values rather than one picked pixel's trace. Warm static `dirtyPages` is 0;
  localized page-dirty accounting remains Phase 3 work.
- Captured a quiet-GPU garden baseline with the benchmark's isolated
  production-pass and whole-frame lanes. Instruments attach failure is noted
  in the completion record above. (The 2026-07-31 capture was contended; §9 of
  the 5X handoff shows contended captures mislead.)
- **Gate passed**: 18,209 distinct visible voxels is well below 436,920 pixels.

### Phase 1 — static single-light vertical slice — **complete**
- Added the storage lane, request bitmask, bounded population kernel and
  nearest-tap consumer tier for one directional light. Fluid remains on the
  live path until Phase 3 supplies localized invalidation.
- The implementation is dynamically safe: source replacement and authored
  primitive/tuning changes advance the epoch, rigid motion is overlaid live,
  and fluid disables this Phase 1 tier. This is broader correctness coverage
  than the original static-only slice, not offline baking.
- **Gates passed** (all via `tools/benchmark-svo-dry-frame-gpu.ts` on garden):
  (a) settled-camera frames: cone-visibility pass cost drops ≥70% once warm;
  (b) diff PNGs vs. `shadingPath:"inline"` ground truth show error confined
  to shadow-boundary neighborhoods, max luminance error < the existing
  prepass path's own error vs. ground truth (measure that first — it is not
  zero); (c) orbiting camera: no frame-to-frame lighting deltas on static
  geometry (assert via consecutive-frame diff).

### Phase 2 — multi-light, area lights, trilinear, proof seeding (2–3 days)
- All 8 prepass light channels; area lights store the 2-sample average (same
  contract as today's plane). Trilinear + backface weighting replaces nearest.
- Static-proof seeding per §5.4 if the coverage report shows ≥30% of
  (page, light) pairs certifiable on garden.
- **Gate**: garden with full light rig matches Phase 1 gates; population
  burst on a 180° camera cut drains within budget in ≤ 8 frames.

### Phase 3 — fluid invalidation + AO channel (hose-tank, 3–5 days)
- Fluid dirty-set from coverage pyramid deltas (dilated by 1 page along
  light directions). AO channel: 4-cone result stored per voxel using a
  *fixed* direction basis (drop the per-pixel `featureId` jitter inside
  cached texels; keep jitter only on the live path). AO is normal-dependent —
  store the hemisphere result for the voxel's dominant normal from the
  G-buffer request, and fall back live when the pixel normal deviates > ~30°.
- **Gates** on hose-tank: (a) no visible stale-shadow trails behind moving
  water (visual + consecutive-frame diff near the fluid front); (b) total
  lighting GPU ms (population + consumer) < today's 15.65 ms contended /
  re-measured quiet baseline by ≥ 3 ms; (c) `tests/` suite green, plus new
  unit tests for pack/unpack, dirty propagation, and request dedup.

### Phase 4 — decide the endgame (measure first)
Two exits, chosen on Phase 3 numbers:
- **Retire the screen prepass** for scenes where cache hit-rate is high
  (`coneLightingScale` becomes a fallback-only knob), or
- **Irradiance caching** (Tomorrow-Children step): fold the full 8-light ×
  2-sample loop into per-voxel L1-SH irradiance stored in the tetra-radiance
  lane family, collapsing `shadeDryOpaque`'s light loop to one SH evaluation.
  Only worth it if Phase 3 shows the deferred pass (5.37 ms) is now the
  bottleneck.
- If banding on smooth surfaces appears at any phase: add Ward-style
  translational gradients (one extra `rgba16float` lane) computed during the
  same population march — this is the "gradient information" escalation
  path, not a day-one requirement.

## 7. Measurement protocol

- Scenes: garden-svo-lighting (static-heavy, best case) and hose-tank
  (fluid-heavy, worst case) at 660×662, matching existing artifacts.
- Tools: `tools/benchmark-svo-dry-frame-gpu.ts` (median/p95 + feature
  attribution + diff PNGs), `tools/profile-svo-render-xctrace.ts` (per-pass
  occupancy; expect the population pass to inherit the cone pass's low
  occupancy — that's fine, it runs on ~10³–10⁴ voxels, not 10⁵ texels),
  `tools/render-svo-optimization-report.ts` for the verdict ledger. Retain
  artifacts under `artifacts/render-voxel-light-cache/` — the 5X handoff
  lost its experiment artifacts and the report tool still flags
  `missing-evidence`; don't repeat that.
- Every phase records: warm-frame GPU ms by pass, cold-burst drain time,
  cache memory resident, diff-PNG max/mean error vs. inline ground truth.

## 8. Risks and pre-planned fallbacks

| Risk | Signal | Fallback |
|---|---|---|
| Voxel-boundary shadow blockiness | Stair-stepping in diff PNGs at shadow edges | Trilinear (Phase 2) → gradient lane (Phase 4) → shrink heterogeneous-guard thresholds so more texels go live |
| Light leaks at thin walls | Bright pixels behind single-voxel occluders | Backface weights → per-face (6-direction) storage → DDGI Chebyshev weights, in that order of cost |
| Stale shadows behind moving fluid | Trails in consecutive-frame diffs | Widen dirty dilation along light axis; worst case dirty whole light channel when fluid page-delta count exceeds a threshold |
| Population burst jank on camera cuts | Frame spikes on cuts | Budget already bounds it; degrade to prepass-plane path for unfilled voxels (never worse than today) |
| Manual trilinear of `rg32uint` too hot in consumer | Deferred pass ms grows > cone savings | Nearest tap + screen-space 3×3 bilateral (Teardown-style) instead of trilinear; or split lanes into filterable `rgba8unorm` |
| Cache management overhead eats the win (the FastAtlas/Neff failure mode) | Warm-frame total ≥ baseline | Per-shade-point work here is expensive (≤48-step cone × lights), which is the regime where caching wins in the literature; if it still loses, stop after Phase 1 — the slice is cheap by design |
| AO normal-dependence | AO differs visibly from today near curved surfaces | Keep AO screen-space permanently; the win is in the per-light channels anyway (~85% of ray work) |

## 9. Sources (most applicable first)

1. The Tomorrow Children — cascaded per-voxel per-face lighting, 1/16-res
   evaluation + upsample, ~6 ms PS4; motivation was freely-deformable
   geometry. gamedeveloper.com "Graphics Deep Dive: Cascaded voxel cone
   tracing in The Tomorrow Children"; GDC Vault 2015/2016.
2. DDGI — Majercik et al., JCGT 2019: trilinear × Chebyshev-visibility ×
   backface weights = leak-free interpolation from a coarse lattice.
3. Ward & Heckbert 1992, "Irradiance Gradients" (+ Křivánek 2005, SIGGRAPH
   2008 irradiance-caching course): gradients at sparse cache points raise
   interpolation order — the escalation path in §6 Phase 4.
4. kajiya `gi-overview.md` (Embark): candid account of a raw voxel irradiance
   cache being leak-resistant but not spatially smooth — motivates the
   trilinear/gradient/bilateral ladder in §8.
5. Teardown breakdowns (blog.voxagon.se "From Screen Space to Voxel Space";
   acko.net "Teardown Frame Teardown"): NB Teardown shades **per pixel**
   (1–2 rays + irradiance/albedo separation + bilateral/temporal denoise);
   its relevance here is the irradiance/material separation and the denoise
   ladder, not per-voxel shading.
6. Cosin Ayerbe et al., "Dynamic Voxel-Based Global Illumination," CGF
   2024/2025: current academic form of cached per-voxel-face lighting with
   static/dynamic separation.
