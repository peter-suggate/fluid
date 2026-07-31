# SVO Renderer 5× Plan — Handoff

**Goal:** ~5× GPU frame-time reduction of the sparse-voxel-octree render path
(cone-traced shadows + AO) with minimal visual degradation.
**Baseline:** 25.9 ms median → **target ≈ 5.2 ms** (garden-svo-lighting, 660×662,
Apple M1 Max / Metal, split shading, cone scale 0.5). This also happens to land the
frame inside the retired raster/SVO comparison suite's historical budget (4–8 ms
presentation p95) for the first time.

Date: 2026-07-30. Branch context: `perf/structured-cutover` (render work rides
along; the branch itself is a solver cutover). The working tree contains an
in-flight, now-measured radiance-reconstruction experiment — see §2.

---

## 1. Measured baseline (what the milliseconds actually are)

Fresh captures from today, `/tmp/svo-profile-relight-split-correct/summary.json`
(xctrace, commit `321c775` + dirty tree) and `/tmp/svo-bench/*.json`:

| Pass | GPU ms | Share | Occupancy | ALU |
|---|---:|---:|---:|---:|
| Sparse voxel primary visibility | **22.86** | 58.5% | **11.3%** | 17.1% |
| Sparse voxel cone-lighting prepass (0.5×) | **11.53** | 29.5% | 0.9% | 0.1% |
| Sparse voxel deferred dry lighting | **4.66** | 11.9% | 13.2% | 20.2% |
| Pass overlap (concurrent execution) | **−13.87** | | | |
| **GPU busy / frame** | **25.17** | 94.4% of wall | | |

Feature attribution (split, `/tmp/svo-bench/relight-split.json`): full 26.7 ms,
AO off 25.4, shadows off 23.9, **both off 17.0**. The cone system's true
incremental cost is **~14.6 ms/frame** and is non-additive — the prepass is only
elided when shadows *and* AO are both off. Primary visibility (traversal +
terrain + rigid + glass, no cones) is **~17 ms**.

Key facts that shape the plan:

- **Everything is latency-bound, not ALU-bound.** Fragment occupancy 11–13%,
  ALU 15–20%, bandwidth single-digit GB/s. The megakernels are register-heavy
  (32-entry traversal stack + wide-fanout frame stack + cone page caches in
  function-private memory) and the GPU cannot hide memory latency. This is the
  single biggest multiplier available.
- **Split shading is a fresh, large, measured win**: inline 36.5 ms → split
  25.9 ms (−29%) via 13.9 ms of pass overlap. It contradicts the older
  "Reject; keep inline" verdict in `tools/render-svo-optimization-report.ts` —
  that rejection predates the `dryPrepassShadeMain` register-lifetime split and
  the `rg32uint` target repack (`0572c5e`). The old verdict is stale.
- **The cone prepass at scale 0.5 currently buys ≈ 0 ms** (reference 36.6 vs
  reduced 36.2 on garden). At scale **0.25** it buys 20–25% with luminance error
  mean 0.0095 / p95 0.040 (vs 0.005/0.021 at 0.5). Guided-upsample fallback band
  was 0% in every fresh run.
- **All captures were GPU-contended** (WindowServer/Chrome/Codex; only 24–34%
  of our GPU time uncontended). Absolute numbers are trustworthy to ~±15%;
  every A/B in this plan must re-run on a quiet machine (see §7).

Reproduction:

```bash
# Whole-frame benchmark + A/B + attribution + quality stats (~20–60 s)
FLUID_SVO_DRY_FRAME_SCENE=garden-svo-lighting \
FLUID_SVO_DRY_FRAME_WIDTH=660 FLUID_SVO_DRY_FRAME_HEIGHT=662 \
FLUID_SVO_DRY_FRAME_CONE_SCALE=0.5 FLUID_SVO_DRY_FRAME_SHADING=split \
FLUID_SVO_DRY_FRAME_RADIANCE_RECONSTRUCTION=full-res-relight \
FLUID_SVO_DRY_FRAME_OUT=/tmp/svo-bench/run.json \
node --import tsx tools/benchmark-svo-dry-frame-gpu.ts

# Per-pass GPU counters via xctrace (~5–7 min, takes the webgpu-exclusive lock)
node --import tsx tools/profile-svo-render-xctrace.ts \
  --scene=garden-svo-lighting --resolution=660x662 --shading=split \
  --cone-scale=0.5 --radiance-reconstruction=full-res-relight \
  --out=/tmp/svo-profile-NAME
```

---

## 2. Constraints, gates, and prior decisions (do not relearn these)

1. **Scale-1 fingerprint gate.** `createSvoDrySceneFragmentWGSL(1)` must stay
   byte-identical to `svoDrySceneShader` (first test in
   `tests/webgpu-svo-cone-prepass.test.ts`). Every shader optimization must be
   an empty-string insertion at scale 1, or the gate must be explicitly
   renegotiated (WS-1 below will require renegotiating it — call this out in
   review, don't silently break it).
2. **10-storage-buffer device limit.** The dry fragment shader is at the limit
   (`docs/FRAME_PROFILER_ARCHITECTURE.md` §421-436). New per-pixel or per-node
   data must be **textures** (the pixel-trace probe already had to use an
   `r32uint` storage texture for this reason).
3. **Quality gate:** exact output preferred; otherwise ≥5% median saving with
   ≤0.05% changed pixels when localized (`render-svo-optimization-report.ts`).
   Reduced-rate paths carry error bars instead: the accepted precedent is
   mean ≤ 0.005 / p95 ≤ 0.027 relative luminance (the 0.5-scale prepass).
4. **Tint/Metal trap:** dynamically indexing a uniform array costs ~+5 ms/frame;
   constant-index lookups are pinned by test. Keep it that way.
5. **Already rejected — do not re-propose as-is** (reasons recorded in
   `render-svo-optimization-report.ts` and shader comments):
   - Screen-space termination at 64 px (proxy material/normal wrong at edges) —
     but see WS-3 for the fix that makes it viable.
   - Disabling AO while moving (contiguous error patches, pops on settle);
     reducing the cone *step budget* while moving (shifts the shadow body).
   - Cone-marcher empty-space elision (bit-exact but measured slower);
     XOR child ordering; full-range LOD blending (+35%); uniform brick size 4.
   - Static-primary coherence: *neutral* only because shadow/cone work dominated
     the frame at the time (`/tmp/svo-bench/README-ray-coherence.md`). Revisit
     after WS-2 shrinks the cone cost — it removes 6–7 ms of exact primary work.
6. **In-flight working-tree experiment** (uncommitted): the five
   `coneRadianceReconstruction` modes + `environmentBrickRefinementLevels`,
   plus `anyBodyBlockerIgnoring()` early-out. Today's benchmarks measured them
   (§1). This must be committed/landed before new work stacks on top —
   otherwise every A/B is against a moving target.

---

## 3. Workstream 0 — land what is already measured (≈1.4× alone, near-zero risk)

Do these first; they are wins sitting on the table and they stabilize the
baseline for everything after.

| Item | Action | Evidence |
|---|---|---|
| 0a | Commit the working-tree radiance-reconstruction + `anyBodyBlockerIgnoring` work; make **split shading the production default** (renderer still constructs with `shadingPath: "inline"` defaults at `lib/webgpu-renderer.ts:715`) | 36.5 → 25.9 ms measured today |
| 0b | Default **cone scale 0.5 + `full-res-relight`**; keep 0.25 as the explicit performance preset | 0.5 retains the accepted visibility error bar; 0.25 remains the measured 20–25% opt-in saving |
| 0c | **Prewarm both cone-prepass pipeline scales** so the moving-camera tier can drop to 0.25 without a compile stall (`ensureConeLightingPrepass()` caches one scale, `lib/webgpu-svo-dry-scene.ts:2125`) | measured 11.08 ms moving vs 13.63 settled; named "the follow-up that unlocks it" in `917a45d` |
| 0d | Re-evaluate `traversalMode: "canonical-parametric"` (the report's own recommended mode, never wired into the browser) against split shading; ship whichever wins | report tool's "Recommended modes" |
| 0e | Capture the missing **compact-hierarchy timing** (16 B nodes, 50% smaller, implemented, "Not selected without timing") | `lib/webgpu-svo-compact-hierarchy.ts` |
| 0f | Update the stale split-shading rejection in `render-svo-optimization-report.ts` and re-establish `artifacts/render-traversal-experiments/` + `artifacts/svo-render-experiments-20260729/` so the report tool runs again | it currently throws |

Expected after WS-0: **~18–19 ms** (0.72× of baseline from 0b alone, plus 0c on
moving frames), quality within already-accepted error bars.

---

## 4. Workstream 1 — occupancy and the primary-visibility megakernel (the big one)

Primary visibility is 22.9 ms at **11% occupancy**. The shader simultaneously
holds: the 32-entry canonical LIFO stack, the 12-deep wide-fanout frame stack,
brick-DDA state, terrain-march state, the 12-body rigid loop, glass state, and
13 always-maintained diagnostic counters. Register pressure is the occupancy
ceiling; occupancy is the frame time.

1. **Clip the redundant analytic traces** (cheap, immediate):
   - `traceDrySolidScene` (`lib/webgpu-svo-dry-scene.ts:1954`) runs SVO + full
     terrain march + rigid loop unconditionally and takes the min. Order them
     SVO → terrain → rigid and pass the current best `t` as `tMax` so the
     terrain bracket/secant march (up to ~28 height evals + 4 normal evals) and
     body loop can early-out. Bit-exact by construction.
   - Skip the glass loop when `dry.terrain.y == 0` panes (garden has 0 panes and
     still pays the loop entry per pixel).
   - Compile the cone marcher without the fluid-coverage sample when the scene
     has no solver (`createSvoDryConeMarcherWGSL({fluidCoverage:false})` — the
     flag exists but is hardcoded `true` at `:1453`). Dry scenes pay a 3D
     texture sample per shadow-cone step for a texture that is empty.
   - Strip the 13 diagnostic counters at reduced scales (compile-time, keeps
     scale-1 byte-identical).
2. **Rasterize the analytic geometry instead of ray-marching it.** Terrain is a
   static heightfield — draw it once as a coarse vertex grid into the depth
   buffer in a vanilla raster pass; rigid bodies get proxy hulls. The SVO pass
   then reads scene depth as `tMax` (and skips entirely where raster geometry
   is closer + opaque). This removes the terrain/rigid register state from the
   megakernel — that alone should raise occupancy — and replaces per-pixel
   secant iteration with hardware rasterization. Shading of raster hits reuses
   `dryEvaluateSurfaceMaterial`. This changes the scale-1 shader; renegotiate
   the fingerprint gate (§2.1) or gate it to reduced scales first.
3. **Coarse depth / beam prepass** (nothing like it exists today): trace only
   the top 2–3 octree levels at 1/8 res (wide-fanout pages make this ~free),
   write conservative-min `tEnter` per tile, and start full-res primary rays at
   the 2×2-dilated tile minimum. For a latency-bound traversal, skipping the
   upper-tree descent for every ray is a direct multiplier. The same prepass
   output drives WS-3 tile classification.
4. **Occupancy experiment matrix** (measure, don't assume — xctrace occupancy
   counter is the metric): (a) compact 16 B hierarchy (WS-0e), (b) canonical
   stack 32 → 16 entries with restart fallback on overflow, (c) demote the
   wide-fanout boundary-scan path (64-iteration divergent loop) to canonical
   earlier, (d) a compute-shader wavefront variant of primary visibility
   (traversal kernel → hit buffer → shading kernel) with explicit workgroup
   sizing. (d) is the heavy option; only pursue it if (a)–(c) + item 2 don't
   move occupancy past ~25%.

Expected: primary visibility 22.9 → **6–9 ms** (occupancy 11→25%+ is ~2×;
tMax clipping + terrain rasterization removes several ms of ALU/registers;
beam prepass cuts upper-tree steps). This is the workstream with the widest
error bars — measure at every step.

---

## 5. Workstream 2 — make the cone system pay rent (temporal + memory locality)

The cone system costs ~14.6 ms incremental; the prepass runs at 0.9% occupancy
and is rebuilt from scratch every frame even though TAA and checkerboard
machinery already exist downstream.

1. **O(1) page lookup.** Every page-cache miss in the cone marcher pays a
   24-iteration binary search over the directory texture, 2 `textureLoad`s per
   iteration, and there are *two* independent caches (fine + coarse LOD). Since
   the atlas is only 1,715 pages (garden) / ~7 MB, replace the search with a
   direct-mapped per-level page table (3D `r32uint` texture indexed by
   `pageCoord >> level`, one texel per potential page — texture, so the
   storage-buffer limit doesn't bite). 48 texture loads → 1 per miss.
2. **Temporal prepass reuse.** The half-res visibility plane (AO + 8 light
   visibilities, `rg32uint`) is world-anchored and low-frequency — ideal for
   reprojection. Reproject last frame's prepass with the same identity/depth/
   normal acceptance the TAA already uses, and **recompute only 1/4 of prepass
   pixels per frame** (round-robin quad parity, exactly the checkerboard-shadow
   pattern extended to the whole cone pass), full recompute on identity
   mismatch or motion. Amortized cone cost ÷ ~3 in stable regions; the shadow
   checkerboard (`TEMPORAL_FAILURE_SHADOW_DEFERRED`) proves the acceptance
   machinery works.
3. **Stochastic cone rotation under TAA.** There is currently *no jitter
   anywhere* — 4 AO cones and N light samples are deterministic every frame.
   Rotate 1 AO cone per frame through a blue-noise-offset tangent frame and let
   the accumulator (64-sample cap, variance clamp) integrate the 4-cone result
   over 4 frames. Note this is *not* the rejected "AO off while moving": the
   estimator stays unbiased per-frame-pair and converges on settle rather than
   popping. Same trick for area-light shape samples. Moving frames keep the
   existing 1-cone tier.
4. **Share per-receiver state across cones.** All 8 shadow samples + 4 AO cones
   per receiver rebuild `minimumVoxel`, tangent frames, and start cold page
   caches. Hoist the shared derivations; share one page-cache set across cones
   (they walk overlapping space from the same origin).

Expected: cone system 14.6 → **~4–5 ms** settled amortized (page table ~30–40%
of marcher cost; temporal reuse ÷3 on the remainder; sharing shaves the rest),
converging within 4 frames of any invalidation.

---

## 6. Workstream 3 — near-camera quality promotion (distance-adaptive everything)

Requested pillar: spend the budget where the viewer can see it. Today every
budget is uniform across depth — a receiver 40 m away gets the same 4 AO cones,
8 light samples, and 48-step cones as one at 1 m. Perceptually, distance is the
cheapest place to cut with the least visible degradation.

1. **Per-pixel quality tier from hit distance / projected voxel footprint**,
   computed in the prepass (or from the WS-1.3 beam-prepass tiles):
   - **Tier 0 (near, footprint > ~4 px):** current full budgets — this is where
     quality is *promoted*: keep scale-0.5-equivalent cone density near the
     camera even when the global scale is 0.25 (see item 2), full AO cone
     count, full area-light samples.
   - **Tier 1 (mid):** 2 AO cones, 1 area sample, cone step budget 32,
     `maximumShadedLights` importance-truncated by solid angle.
   - **Tier 2 (far, footprint < ~1 px):** 1 AO cone, punctual-only shadows,
     step budget 16, primary traversal terminates descent when the node's
     projected footprint drops below a pixel and shades from the **node-mip
     mean material/normal** — this fixes the recorded rejection of
     screen-space termination ("coarse AABB proxy lacks representative
     material/normal") by using data the mip atlas already stores.
   - Tier thresholds are `t`-based with hysteresis so the boundary never
     shimmers; tiers are data (`tuningCounts` lanes are already plumbed), so
     scale-1 stays byte-identical by keeping tier-0 ≡ current behavior.
2. **Variable-rate cone prepass.** Instead of one global `coneLightingScale`,
   classify prepass tiles by tier: near tiles evaluated at 0.5 density, far
   tiles at 0.25/0.125 (the marcher already supports all scales; this is a
   dispatch/addressing change, not a new algorithm). This is how 0b's global
   0.25 default becomes visually free: the error concentrates exactly where
   tier 0 overrides it back to 0.5.
3. **Distance-scaled AO radius/aperture** so far cones go coarse-LOD
   immediately (step count ∝ log of range/diameter — wider aperture = fewer
   steps for the same range, and the LOD blend already handles the filtering).

Expected: **1.3–1.6×** on cone + traversal cost in typical framing (most pixels
of an outdoor scene are tier 1–2), with *improved* near-field quality vs the
global-0.25 default. Quality validation: per-tier error histograms from the
existing `errorStats` machinery, plus the standard settle-pop check
(`FLUID_SVO_DRY_FRAME_CAMERA_MOVING` A/B).

---

## 7. Measurement protocol (non-negotiable)

1. **Re-baseline on a quiet machine first.** Today's captures were 24–34%
   uncontended (WindowServer/Chrome/Codex on the GPU). Close Chrome, stop Codex,
   use `run-webgpu-exclusive.ts`'s lock, and re-run the §1 commands. All plan
   multipliers are relative, but the accept gates are absolute.
2. Every change: `benchmark-svo-dry-frame-gpu` A/B (median + p95 + errorStats +
   difference PNGs) at 660×662 garden **and** hose-tank (1.8× the bytes, deeper
   tree — WS-1/2 changes can behave differently there), then one xctrace
   capture per landed workstream to confirm the occupancy story.
3. Track the running product in a ledger table in this doc: `baseline 25.9 →
   WS0 → WS1 → WS2 → WS3`, each row with commit, median, p95, occupancy,
   error mean/p95. The 5× claim is the product of measured row ratios, not the
   sum of estimates.
4. Quality gates per §2.3; temporal changes additionally need a settle-pop
   check (record 120 frames across a camera stop; no visible discontinuity at
   the settle frame) since that is the recorded failure mode that killed the
   naive moving-frame reductions.

## 8. Expected stack-up and landing order

| Stage | Change | Frame (ms, est.) | Cumulative |
|---|---|---:|---:|
| Baseline | split + relight (measured) | 25.9 | 1.0× |
| WS-0 | scale 0.25 + relight default, prewarm, canonical-parametric | ~18.5 | 1.4× |
| WS-1 | tMax clip, terrain raster, beam prepass, register diet | ~10–12 | 2.3× |
| WS-2 | O(1) page table, temporal prepass reuse, stochastic cones | ~7–8 | 3.5× |
| WS-3 | distance tiers + variable-rate prepass | ~5–6 | **4.5–5.2×** |

Risks, ranked: (1) WS-1 occupancy work has the widest error bars — if the
wavefront rewrite is needed, it's the largest engineering item in the plan;
(2) WS-2 temporal prepass reuse must not reintroduce the settle-pop; the
per-frame-pair-unbiased design and the existing variance clamp are the
mitigations; (3) the scale-1 fingerprint gate must be renegotiated once
(WS-1.2) — do it deliberately, with a new pinned fingerprint and a visual
sign-off, not as a side effect.

---

## 9. Implementation ledger (2026-07-30 quiet-GPU rerun)

All rows below use the garden scene at 660×662, split shading,
`full-res-relight`, cone scale 0.25, two encodes per wall-time sample, 4 warmups,
and 16 cycles on the Apple M1 Max. The output gate stayed exact across every
retained arm: scale-1 `0x6246c177`, configured `0x1274ec47`, packed surface
`0x741d3571`, identity/media `0xa37539f9`, and depth `0x827d767e`.

| Stage / experiment | Median ms | p95 ms | Result |
|---|---:|---:|---|
| WS-0 canonical-parametric A | 12.542 | 13.791 | retained production traversal |
| WS-0 canonical-parametric B | 12.552 | 13.373 | repeat control |
| WS-0 hybrid A / B | 22.099 / 23.597 | 25.934 / 25.571 | rejected; much slower, exact output |
| WS-0 compact | 12.600 | 14.851 | measured, not selected; neutral |
| WS-2 direct `r32uint` page table A | **11.938** | 13.652 | retained |
| WS-2 direct `r32uint` page table B | **11.895** | 12.719 | retained; repeat |
| WS-2 direct table, hose-tank | 19.502 | 20.376 | retained correctness/scale check; 8,192 pages |
| WS-1 exact static-primary, settled control | 11.630 | 16.818 | same temporal phase and graph, coherence disabled |
| WS-1 exact static-primary, settled reuse A | **4.119** | 7.842 | retained; exact image and G-buffer hashes |
| WS-1 exact static-primary, settled reuse B | **4.156** | **4.331** | retained repeat; one 8.408 ms warm-tail outlier below p95 |
| WS-1 exact static-primary, hose-tank | 10.245 | 14.433 | retained correctness/scale check; 8,192 pages |

The retained direct table packs each mip level into a Z slab and stores
`atlasSlot + 1` (zero means non-resident). On a page-cache miss the cone shader
does one indexed texture load and derives the 10³ atlas-page origin from the
slot. The old level-ranged Morton directory remains a bounded fallback when a
virtual extent exceeds the device's 3D texture limit or a 64 MiB allocation
cap. This is the renderer-side application of the reference paper's central
locality lesson: page-local sparse uniform-grid access is valuable when the
working set is compact and Morton-organized; repeated global directory walks
are not. The direct table consumes a sampled texture, not an eleventh storage
buffer.

Reduced-rate quality is unchanged by the page-table work: relative luminance
mean 0.009906, p95 0.042071, and guided-upsample fallback 0%. The two retained
runs improve the quiet canonical control by 4.8–5.2%; because the 25.9 ms
handoff baseline was captured with a different timing method under contention,
it is kept as historical context rather than mixed into this wall-time ratio.
Hose-tank exposes the limit of making 0.25 a global rather than adaptive rate:
its post-change mean/p95 are 0.018581/0.076370. The direct table itself remains
exact, but hose does not clear the garden-derived reduced-rate quality bar; it
therefore remains the mandatory acceptance scene for WS-3's near-field 0.5
promotion rather than evidence that the global 0.25 rate is universally free.

Exact static-primary coherence is the change that clears the headline target.
Against an otherwise identical settled temporal-phase control it removes 64.6%
of the frame (11.630 → 4.119 ms); the repeat is 4.156 ms median / 4.331 ms p95.
That is 6.2–6.3× versus the handoff's historical 25.9 ms baseline. The temporal
phase-A configured image (`0x7fd306ae`) and G-buffer fingerprints (packed
`0x9b3a4071`, identity/media `0xa37539f9`, depth `0x827d767e`) are identical
between control and reuse. Production enables this only for static renderer
worlds and paused solvers, with a caller key covering camera, viewport, scene
epoch, rigid transforms, selection, tuning, and environment. Source swaps,
resizes, and authored primitive animation invalidate the renderer-owned cache;
running fluid scenes pass no key and always retrace. Scale-1 also always
retraces because its primary G-buffer legitimately carries alternating
shadow-deferred flags; reduced split shading computes the complete visibility
plane and is parity-invariant.

### Persisted attempts that did not clear the gate

| Attempt | Median / p95 ms | Correctness | Disposition |
|---|---:|---|---|
| SVO→terrain→rigid `tMax` clipping | 12.616 / 14.696 | changed all frame/G-buffer fingerprints | reverted; not bit-exact and not faster |
| zero-pane glass early return | 12.704 / 13.229 | exact | reverted; slower than control |
| reduced-scale diagnostic-counter stripping A / B | 13.136 / 62.702; 12.335 / 14.765 | exact | reverted; unstable and best gain <5% |
| receiver-shared fine/coarse cone page caches A / B | 11.946 / 14.587; 12.050 / 14.608 | exact | reverted; neutral/slower than direct-table-only |
| screen-space temporal prepass, scale 0.25 | 11.935 / 13.123 control; 11.749 / 12.671 reuse | exact under the same temporal phase (`0x7fd306ae`) | reverted; 1.6% median is below the 5% gate |
| screen-space temporal prepass, scale 0.5 | 14.542 / 16.633 control; 14.531 / 17.334 reuse | exact under the same temporal phase (`0xe2bb6312`) | reverted; neutral median and worse p95 |

The production defaults now use cone scale 0.5 (2×2), `full-res-relight`,
canonical-parametric traversal, split shading, and fail-closed exact
static-primary coherence; the performance preset uses scale 0.25. Both 0.25
and 0.5 cone/split pipeline bundles are prewarmed and retained, with single-
flight compilation per scale. The report generator no longer invents or
requires missing historical artifacts: it emits a `missing-evidence` manifest
and names the absent captures, while its split-shading recommendation reflects
the newer 36.5 → 25.9 ms result.

The attempted screen-space temporal reuse copied compact geometry/identity to
non-aliased history textures, accepted only exact identity plus tight
depth/normal matches, recomputed boundaries and one rotating quad parity, and
fell back to full work during camera motion. It compiled and ran on Metal, but
the copies and validation consumed nearly all saved cone work at both rates.
A future temporal attempt needs world-space reprojection or persistent
page-/receiver-space work that avoids copying the compact planes.

The 5× target is demonstrated for exact-key-eligible static and paused frames;
it is deliberately not claimed for running fluid scenes, which fail closed to
fresh primary visibility. World-space temporal prepass reuse, terrain/proxy
rasterization, the beam prepass, and distance-adaptive quality remain the route
to the same target during continuous simulation. Hose-tank also still requires
near-field/adaptive promotion: even global scale 0.5 measured mean/p95
0.010660/0.043760, so simply raising its global rate did not clear the quality
gate.
