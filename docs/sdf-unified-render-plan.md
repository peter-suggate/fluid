> **Superseded by `docs/svo-raster-visibility-handoff.md`** (2026-08-04), which
> verifies this plan against HEAD, corrects the 4096-cliff failure mode, adds
> the Teardown/Claybook research, and records that the incremental SDF
> voxelizer this plan's Phase 2 proposed already exists in production.

# Unifying the SDF set onto the brick-raster architecture — plan for handoff

**Goal:** scenes ~10× the complexity of `hero-garden-hose` (≈5,000 SDF records)
rendering at real-time rates (60 fps ≈ 16.7 ms/frame at ~1600×1240).

**Status quo (measured, 2026-08-04, M1 Max, commit a3adca9 — full evidence in
`docs/svo-render-frame-anatomy.html` + `.data.js`):** the hero frame is 405 ms.
One pass — `Sparse voxel exact live-scene primitive visibility`, the direct
raster over 501 authored SDF records — is 346 ms (85 %). The whole sparse-voxel
octree primary (cull → coverage → resolve → overflow) is **3.0 ms**. Cone
visibility is 43 ms, deferred lighting 11 ms.

---

## 1. Diagnosis

The two raster passes in the frame are the same idea — draw one conservative
box per object, resolve the exact surface in the fragment — built twice, one
with the hidden-surface-removal insight and one without.

- The **brick raster** was redesigned around the tile-based GPU: a coverage-only
  pass appends candidates into a fixed 24-entry per-pixel arena
  (`lib/webgpu-svo-dry-scene.ts:5328`), one full-screen resolve walks each
  pixel's list front-to-back exactly once (`:5344`), and overflow re-runs the
  historical fragment in an isolated pipeline (`:5364`). Capacity is a
  performance parameter, never an image change. Cost: 3.0 ms for 40k leaves.
- The **scene-primitive raster** (`:7314`, fragment at `:2613`) predates that
  insight. Its fragment returns `@builtin(frag_depth)` *and* calls `discard`,
  so the tiler can reject nothing: every covering proxy marches its full field
  even when a nearer proxy already resolved the pixel. A covered pixel sits
  under a median of 7 proxies (p99 33, max 59), 95 % of proxies are sphere-traced,
  and a cluster march step folds over a whole procedural field (8–64 sphere
  distances per lattice octave, + 4 tetrahedral normal taps).

So the user-level conclusion is confirmed: **the SDF work was not built on the
raster speed-up's insights, and the design it needs is already sitting sixty
lines above it in the same file.**

But there is a second, harder fact. The *sound* fix — coverage/resolve with
front-to-back early termination, measured exactly by CPU census over 731
covered pixels — is worth **1.75×** on marches (5.22 vs 9.11 per covered
pixel), bracketed above by the unsound hardware probe's **3.15×**. Applied to
the 346 ms pass that yields roughly 110–200 ms, i.e. a ~170 ms frame ≈ 6 fps —
at **1×** complexity. Per-pixel analytic marching of every authored record can
not reach 16.7 ms at 10× complexity by any constant-factor tuning. Two further
measured dead ends to not revisit: tightening proxies (refuted, −5.2/+3.7/−24.1 ms
against a 30 ms control spread — this pass is per-fragment-cost bound, not
fragment-count bound) and optimising the bonsai's fields first (its 65 % share
of evaluations is mostly work that is thrown away unresolved).

**The path that already has the right asymptotics is in the frame:** the octree
brick path costs 3.0 ms and its per-pixel cost is bounded by the arena walk,
independent of how much scene stands behind the first hit. And the project's
stated architectural destination is exactly that — *one pipeline: very high
resolution voxels sampled from procedural geometry* — with the analytic set as
the authoring representation, not a per-pixel per-frame cost.

There is also a hard scale cliff: `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES = 4096`
(`lib/svo-primitive-candidates.ts`). Above it the candidate build is simply not
called and the set stops drawing. A 10× scene (~5,000 records) is **undrawable
today regardless of speed.**

## 2. Target architecture

One visibility architecture, three tiers, one depth authority:

1. **Voxel content is the default.** SDF records are *sampled into the octree*
   (voxelized, per-voxel material/identity ownership) and rendered by the
   existing brick coverage/resolve path. Authored complexity is paid at
   voxelization time (incremental, on-change, amortized) — **not per frame**.
   Frame cost becomes proportional to pixels and visible surface, which is what
   makes 10× scenes flat.
2. **A bounded near-field analytic band.** Voxel silhouettes are visible close
   up (a 25 mm voxel at 2 m under the hero's 50 mm lens is ~28 px). Primitives
   whose projected voxel error exceeds a screen-space threshold (a small,
   camera-local set by construction) resolve analytically — but through the
   *unified* coverage/resolve raster, never the direct fragment.
3. **Lighting stays on the node-mip pyramid.** Cones already sample the pyramid,
   so lighting is resolution-bound, not scene-bound — voxelized SDF content
   joins the pyramid for free (today the analytic set is invisible to it).

## 3. Phases

### Phase 0 — Acceptance scene + budget gate
- Author a procedural ~5,000-record stress scene (`hero-garden-hose-x10` or a
  tiled variant). It will not draw today (candidate cliff) — that is the point;
  it is the acceptance scene for everything below.
- Wire a paired, interleaved benchmark lane for it (same discipline as the A/B
  in the anatomy doc). Budget: primary ≤ 5 ms, lighting ≤ 8 ms, shading ≤ 3 ms.
- Profiling traps (see memory/anatomy appendix B): use the GPU-timestamp lane,
  not `PROFILE_SECONDS` wall (Dawn `finish()` adds ~11 ms); xctrace needs
  `--launch` + `use_user_defined_labels_in_backend`.

### Phase 1 — Unify the scene-primitive raster onto coverage/resolve
The direct 1.75–3.15× repair, and the structural prerequisite for tier 2.
- Split `Sparse voxel exact live-scene primitive visibility` into:
  - **Coverage:** proxy raster appends `(instance, tEnter)` into a per-pixel
    arena via `atomicAdd`; no depth write, no shading, no meaningful discard.
    Size for p99 33 (e.g. 40 entries) with the existing overflow-fallback
    pattern (isolated pipeline re-running today's fragment) catching the tail.
  - **Resolve:** one full-screen pass per pixel: entry-sort candidates, seed the
    running best with the depth the brick raster already wrote (free, worth
    ~0.5 marches/pixel), walk front-to-back, march each candidate only over
    `[tEnter, best]`, stop when the next candidate begins behind the best hit.
- Fold the rigid-impostor pass into the same arena (or at minimum the same
  resolve), which also removes the blocker that disables
  `stationaryPrimaryReuseEnabled` on the raster-primary path.
- Gate: image parity with today (the exact-intersection contract in
  `tests/webgpu-svo-primitive-exact.test.ts` holds), hero frame SDF pass
  ≤ 200 ms, no overflow-driven image change.

### Phase 2 — Voxelize the set; analytic only in the near-field band
The load-bearing bet for 10×.
- Make the SDF→octree voxelizer incremental: re-voxelize only bricks whose
  overlapping records changed (dirty-tracking keyed on record hash + transform;
  `lib/svo-primitive-motion.ts` already models motion). Budgeted per frame,
  converging over frames like the light cache.
- Per-voxel material/identity ownership so voxelized primitives shade and
  identify like everything else (same requirement the terrain unification has).
- Screen-space-error policy: per record (or cluster), compare projected voxel
  size against a threshold; records failing it stay analytic and go through the
  Phase-1 arena. Everything else stops being drawn analytically at all.
- Densify where the policy demands it (deeper octree locally / finer leaf
  bricks near the camera) under an explicit page budget — never globally.
- Gate: hero frame ≤ 33 ms with band active; frame cost flat (< 1.3×) when the
  authored record count is scaled 10× behind an unchanged camera.

### Phase 3 — Remove the scale cliffs
- Replace the 4,096-leaf candidate cliff with a slope: chunked/streamed
  candidate BVH, or capacity raised with the build made incremental; overflow
  must degrade (more analytic → voxel demotion) rather than stop drawing.
- Audit the other fixed arenas at 10×: proxy instance capacity, node-mip page
  pool (its all-or-nothing withdrawal is a silent 15× regression — add headroom
  *and* a loud tripwire), tetra radiance atlas, per-pixel arenas.
- Gate: the Phase-0 stress scene draws, correctly, at any record count up to 10×.

### Phase 4 — Lighting and reuse at scale
- Cone pass to quarter-rate (`coneLightingScale 0.25`, already in the perf
  preset) + enable `stationaryPrimaryReuseEnabled` on raster-primary (unblocked
  by Phase 1). Prior measurement on garden: 47.6 → 14.7 ms combined.
- Voxelized SDF content now populates the node-mip pyramid, so shadows/AO/GI
  see the whole set through the same resolution-bound machinery.
- Gate: 10× scene ≤ 16.7 ms end-to-end, error stats vs full-rate reference at
  the anatomy doc's thresholds (mean relative luminance ~0.5 %, p95 bounded).

## 4. Frame budget at 10× (target)

| Subsystem | Today (1×) | Target (10×) | How |
|---|---|---|---|
| Octree primary (incl. voxelized SDF) | 3.0 ms | ≤ 5 ms | same path, more resident pages |
| Analytic near-field band | 346 ms (all records) | ≤ 3 ms | Phase 1 arena × Phase 2 band shrink |
| Cone visibility + GI cache | 45 ms | ≤ 6 ms | quarter rate + reuse + pyramid |
| Deferred shading + rest | 11 ms | ≤ 3 ms | resolution-bound already |
| **Frame** | **405 ms** | **≤ 16.7 ms** | |

## 5. Risks
- **Near-field band size.** If the camera puts thousands of records in the
  analytic band (macro shots of the bonsai), Phase 1's arena is the only
  protection. Mitigation: band budget with distance hysteresis; densified
  voxels shrink the band.
- **Voxelization churn on animated records.** Bounded by the incremental
  budget; worst case is convergence latency, not frame spikes.
- **Silhouette/AA quality vs analytic exactness.** The current doc prizes
  byte-identical resolution independence; voxel content trades that for scale.
  The screen-space-error policy is where that trade is governed — make the
  threshold authored, not hard-coded.
- **Memory at 10×.** Densification + pyramid + radiance atlas all scale with
  resident pages; Phase 3's audit must produce numbers before Phase 2 commits
  to a finest-voxel size.

## 6. Explicitly not doing
- Tightening proxy boxes (measured null result).
- Optimising bonsai cluster fields before occlusion is fixed (65 % of its work
  is currently discarded — fix what is asked, not what each ask costs).
- Hardening the bespoke direct-fragment path beyond its role as the overflow
  fallback — it is the path being left.

## 7. Reproduction anchors
- Evidence document: `docs/svo-render-frame-anatomy.html` (all numbers bind
  from `docs/svo-render-frame-anatomy.data.js`; regenerate via
  `tmp/hero-doc-data.mjs` from `artifacts/xctrace-hero-garden-hose-2026-08-04/`).
- Frame + census: `tools/benchmark-svo-dry-frame-gpu.ts` (env-driven; see
  anatomy Appendix B), CPU censuses `tmp/hero-proxy-coverage.ts`,
  `tmp/hero-sdf-census.ts`.
- Key code: brick coverage/resolve/overflow and the scene-primitive raster in
  `lib/webgpu-svo-dry-scene.ts`; record/BVH arena in
  `lib/svo-primitive-candidates.ts`; kinds/ABI in `lib/svo-primitive-kinds.ts`,
  `lib/svo-primitive-abi.ts`.
