# Killing the fixed frame overhead in the SVO renderer — handoff

**Goal:** frame cost tracks visible content. A frame with little on screen must
be cheap: ≤ 8 ms GPU at 800×460 with the scene distant, staying resolution-linear
above that — against a measured **24.6 ms today**. The hero framing must not
regress (~89 ms in the same lane, its own program: `docs/svo-raster-visibility-handoff.md`).

**Context.** The raster-visibility program (W0–W4 of that doc) flattened 10×
record scaling to 2.30× and took hero 415.7 → ~89 ms in the benchmark lane. What
remains — the subject here — is a large overhead that does not scale down when
there is little to draw: few covered pixels still means a slow frame.

All numbers measured 2026-08-04 at commit `3d619aa` (dirty), M1 Max, benchmark
lane `tools/benchmark-svo-dry-frame-gpu.ts`, `raster-primary`, split shading,
cone scale 0.5, GPU-timestamp span (`FLUID_SVO_DRY_FRAME_TIMING=gpu`), which
excludes Dawn CPU. Single-sample point estimates: the lane's noise floor is
±5 % within-arm plus ~9 ms batch drift (`docs/svo-raster-visibility-handoff.md`
§0), so treat deltas under ~10 ms as unresolved.

---

## 1. The measurement — where the floor actually is

Six arms, all the hero-garden-hose scene (501 records) unless noted, 800×460
unless noted:

| Arm | Camera / content | Frame GPU median |
|---|---|---|
| `base-hero-800` | hero framing (scene fills frame) | **89.2 ms** (p95 131.8) |
| `far-hero-400` | same scene 30 m away, 400×230 | 16.1 ms |
| `far-hero-800` | same scene 30 m away | **24.6 ms** |
| `far-hero-1600` | same scene 30 m away, 1600×920 | 35.5 ms |
| `far-x10-800` | 30 m away, **×10 records** (5,039) | **20.0 ms** |
| `sky-hero-800` | camera at y=50 looking at pure sky | **0.85 ms** |
| `base-hero-800-wall` | hero framing, wall timing | 94.9 ms (vs 89.2 GPU) |

Reproduce: probe camera modules `tmp/hero-far-camera-scene.ts`
(`FLUID_HERO_FAR_DISTANCE_M`, `FLUID_HERO_FAR_MULTIPLIER`) and
`tmp/hero-sky-camera-scene.ts`, via

```
FLUID_SVO_DRY_FRAME_SCENE_MODULE=tmp/hero-far-camera-scene.ts \
FLUID_SVO_DRY_FRAME_TRAVERSAL=raster-primary FLUID_SVO_DRY_FRAME_TIMING=gpu \
FLUID_SVO_DRY_FRAME_PHASE_TRACE=1 FLUID_SVO_DRY_FRAME_WIDTH=800 FLUID_SVO_DRY_FRAME_HEIGHT=460 \
node --import tsx tools/run-webgpu-exclusive.ts --import tsx tools/benchmark-svo-dry-frame-gpu.ts
```

### Finding 1 — the empty frame is already essentially free. Pass overhead is NOT the floor.

Pure sky costs **0.85 ms** through the full 16-pass encode. The 18-pass
production structure, its clears, its full-screen draws over sky pixels — all of
it together costs under a millisecond of GPU when nothing is hit. Any plan that
leads with "too many passes / too much per-pass overhead" is attacking a term
that measures at noise level. (The *CPU* side of pass count is real — Finding 5.)

### Finding 2 — the floor is worst-pixel march latency in the primary resolve, and it is O(1) in both pixels and records.

The far-camera frame decomposes (pass windows; untrusted individually per
`[[svo-render-profiling-lanes]]`, but the frame is serial here — overlap 0.993 —
so the aggregate is sound):

| Pass | far 400×230 | far 800×460 | far 1600×920 | far ×10 800×460 |
|---|---|---|---|---|
| primary conservative coverage resolve | 8.3 | 13.6 | 16.1 | **13.6** |
| live-scene primitive resolve | 3.9 | 5.3 | 6.2 | 2.9 |
| deferred dry lighting | 1.2 | 3.8 | 13.8 | 2.9 |
| all compute passes summed | 0.7 | 1.1 | 2.0 | 0.4 |
| **frame span** | **15.0** | **23.5** | **37.1** | **20.0** |

Two facts pin the mechanism:

- **16× the pixels moves the primary resolve only 1.9×** (8.3 → 16.1 ms). A
  fragment pass whose cost were per-pixel would move ~16×.
- **10× the records moves it 0×** (13.63 vs 13.57 ms). The cost is not the
  analytic/exact-upgrade term — that would grow with record count.

So the resolve is **latency-bound, not throughput-bound**: a distant scene
concentrates the entire octree into a small patch of pixels; each of those
pixels walks a deep candidate list and HDDA-marches bricks at the *finest
resident level* regardless of how sub-pixel those voxels are; a few hundred
very long fragments own the frame's critical path while the rest of the GPU
idles. Adding pixels just fills idle lanes (hence 1.9× for 16×). This is the
user-visible symptom exactly: *few pixels to display, frame rate still low*.

**There is no LOD on this path.** `screenSpaceTerminationPixels` is a
canonical-traversal + inline-shading diagnostic only — the constructor rejects
it elsewhere (`lib/webgpu-svo-dry-scene.ts:2131`) — and
`DEFAULT_SVO_RENDER_TUNING.nearFieldBandPixels = 0` ships the analytic band
off. Distant voxels are marched at full depth, then the result lands in a
fraction of a pixel.

### Finding 3 — the pixel-linear floor is ~10–16 ns/px, led by deferred lighting.

Across the far arms the frame fits ≈ **13.5 ms fixed + ~16 ns/px**. Deferred
lighting alone is ~10 ns/px (1.2 / 3.8 / 13.8 ms across 92k/368k/1.47M px) —
two full-res `draw(3)`s with `full-res-relight` reconstruction, paid for every
covered pixel however trivial its lighting. Sky pixels are already cheap (the
sky arm proves the early-out works); *lit-but-simple* pixels are not. At a
2 Mpx presentation target this term alone is ~20 ms before any content detail.

### Finding 4 — the encode contains real fixed work that never scales down (inventoried, mostly sub-ms each on GPU, but they gate the endgame and cost CPU).

From a full audit of `SparseVoxelDrySceneRenderer.encode` at HEAD (all
`lib/webgpu-svo-dry-scene.ts`):

- **Four full-res 48 B/px MRT load/store cycles** — brick resolve, brick
  overflow, SDF resolve, SDF overflow all bind the same 4-attachment G-buffer
  with `store` (`:6042–:6051`, passes at `:6409`, `:6433`, `:6227`, `:6249`).
  ~197 MB stored + ~148 MB loaded per frame at 0.72-scale 1600×1240. The two
  overflow passes usually draw **zero instances** (0-instance `drawIndirect`)
  yet pay full tile load+store, and their two `overflowArgs` compute passes +
  16 B copies run unconditionally (`:6075–:6084`).
- **Near-field band selection runs with the band disabled** (`:6187–:6199`):
  gated on pipeline existence, which is always true (`:5766`), not on
  `nearFieldBandPixels > 0`; `ceil(records/64)` dispatch + 1 KB clear per frame.
- **Glass discovery with zero glass records** (`:8516–:8532`): full-res r32uint
  clear + full-res depth clear for a `draw(6, 0)` on the hero scene.
- **Cone fan-out sized to the light *maximum*, not the scene** (`:8619`,
  `:5088`): `4 + maximumLights(8)` = 12 z-layers of cone-res work (~3.1 M
  invocations at 0.72-scale 1600×1240) for a 1-light scene. Production only —
  the smoke lane runs `coneFanout=false` and never sees it (divergence, W0).
- **Capacity-sized, not count-sized:** brick cull emit+scatter dispatch over
  `brickLeafCapacity` (`:6375`, `:6379`), and the publication-buffer clear at
  `:6371` has no size argument so it wipes `4352 + 32·leafCapacity` bytes —
  32 MB/frame at a 1 M-leaf capacity. Voxel-light population is a fixed 256
  workgroups regardless of queued demand (`:8582`).
- **The count plane is cleared twice per frame** (`:6390`, `:6207`) — 2 ×
  `w·h·4` B, both arms sharing one arena (which itself is never cleared —
  `:5821–:5865` — that design is right; only the double clear is waste).

### Finding 5 — the CPU floor (app path, `lib/webgpu-renderer.ts`), invisible to every GPU lane.

- **`canonicalScene(scene)` every frame** (`:1896` via `publishRenderScene`,
  `lib/model.ts:226–:235`): a full recursive deep-clone of the scene with
  `localeCompare`-sorted keys plus `JSON.stringify`, used purely as a change
  detector. Thousands of allocations + ICU collator calls per frame, fixed,
  content-independent.
- **`advanceLiveSceneAnimation()` on any scene with a swaying prop**
  (`:2037–:2074`; the garden scenery authors `sway: true`): per frame it
  re-poses descriptors, runs a full BVH refit with per-node object allocation
  and *two* full-tree validation walks (`lib/svo-primitive-candidates.ts:228–:320`),
  repacks **every** record, FNV-hashes the packed BVH, uploads the whole
  records+BVH arena (~96 KB at 1×, ~960 KB at 10×,
  `lib/webgpu-svo-dry-scene.ts:7617`), and calls
  `clearPrimaryVisibilityCache()` (`:7626`) — structurally destroying the
  stationary-reuse win every frame.
- **Dawn `encoder.finish()`** on the ~20-pass stream: the known ~11 ms/frame at
  1500² ([[svo-render-profiling-lanes]]); here wall 94.9 vs GPU 89.2. In the
  app it overlaps GPU execution but still burns most of a core; it scales with
  encoded command count, so W3's pass diet shrinks it too.
- **Lane artifacts to not re-trip:** the smoke lane
  (`tools/run-svo-dry-render-smoke.ts:806–:813`) starts its clock *before*
  `encoder.finish()` and fully serializes CPU translate with GPU execution —
  its "frame time" is finish + GPU with zero overlap, which a real app never
  is. And it runs `coneFanout=false` while production passes `true`
  (`lib/webgpu-renderer.ts:1146`) — two production compute passes the lane
  never measures.

### What this sums to at presentation resolution

At 0.72-scale 1600×1240 (~1.47 M px) with the scene small on screen:
~13.5 ms latency floor + ~15 ms pixel-linear + ~2 ms fixed GPU + ~10 ms
`finish()`-class CPU ≈ **the ~30–40 ms "slow with nothing on screen" frame the
user observes**. The three terms are independent and need three different
fixes — that is the workstream structure.

---

## 2. Target architecture

The renderer already has the right global shape (arena visibility, pyramid
lighting). What it lacks is the property Claybook called *perfect LOD*: *work
per pixel is bounded by what the pixel can display*. Concretely:

1. **The primary resolve terminates on projected footprint.** A march stops
   descending (and stops exact-upgrading) once the node/voxel footprint is
   sub-threshold in screen space; the hit resolves from the coarse level. Far
   content then costs O(log) per pixel instead of O(fine-voxel path length),
   and the latency floor collapses to the near field, which is the only place
   detail is visible.
2. **Per-pixel cost is tiered, not uniform.** Empty, simple, and deep pixels
   run different code at different cost — via tile classification — instead of
   one uber-fragment whose register pressure taxes every pixel at the worst
   pixel's occupancy.
3. **Nothing encodes for content that isn't there.** Overflow, band, glass,
   fan-out layers, cull sweeps and clears are sized/gated by live counts, not
   capacities or maxima.
4. **The CPU frame is O(dirty), not O(scene).** Change detection is a revision
   stamp; animation repacks only what moved; reuse survives animation.

## 3. Workstreams

Effort scale: S ≈ days, M ≈ 1–2 weeks. Every gate below is stated against the
probe lanes, which must land first. **Respect the noise floor:** interleaved
repetitions, deltas > ~10 ms or a paired lane, per
[[svo-dry-lane-noise-floor]].

### W0 — Floor lanes + honest attribution  (S · low risk · deps none)

- Promote the probe cameras to real lanes: `test:webgpu:hero-floor-far` and
  `test:webgpu:hero-floor-sky` (move `tmp/hero-far-camera-scene.ts` /
  `tmp/hero-sky-camera-scene.ts` into the scene catalog or bless
  `SCENE_MODULE` in the npm script). Assert the current numbers as ceilings so
  the floor can only ratchet down.
- Fix the smoke-lane divergences: `coneFanout` true to match production; report
  encode-CPU (`finish()`) and GPU span as *separate* numbers, never one sum.
- Per-tier pixel counters in the resolve (count of pixels by candidate depth /
  march iterations, one atomic per tile) — the resolve is a render pass, its
  timestamps are untrusted on this GPU, so attribution must come from counters
  (or from W2's compute resolve, whose timestamps are trusted).
- **Gate:** both floor lanes green in CI with recorded medians; a one-command
  repro for the 3-term decomposition in §1.

### W1 — Screen-space-error termination in the primary resolve  (M · the load-bearing bet · deps W0)

The structural fix for Finding 2. In the brick resolve (and the owner→exact
hybrid inside it):

- During HDDA/descent, compare the current node's projected footprint
  (`projectedSvoNodeFootprintPixels`, `lib/svo-screen-space-termination.ts:57`
  — built for exactly this, currently consumed only by a diagnostic) against an
  authored threshold (~1 px default; separate knob from `nearFieldBandPixels`).
  Below threshold: **stop descending; resolve the hit at this level** — depth
  from the interval midpoint, normal from the occupancy/payload gradient at
  that level, material from the dominant owner/material lane of the node (the
  node-mip pyramid already aggregates occupancy per level; material/normal
  aggregation at coarse levels is the new data work, and can start as
  "descend to the finest resident level but *march it with a
  footprint-proportional step and skip the exact upgrade*", which needs no new
  data and already bounds iteration count).
- Same predicate kills the exact-upgrade: `svoIntersectPrimitiveExact` (the
  48-iteration cluster-lattice march) must never run for a sub-pixel voxel.
  This is the "cheap per-cell rejection" the previous handoff already named as
  worth more than anything left in W3.
- Applies to the SDF-primitive resolve too: a proxy whose *entire record*
  projects sub-pixel should resolve as its bounding-interval hit, not a sphere
  trace.
- **Cost model:** per-pixel march work becomes O(log(depth)) in distance;
  frame floor tracks *near-field* content only.
- **Gate:** `hero-floor-far` ≤ 8 ms at 800×460 and ≤ 1.5× that at 4× pixels
  (kills the flat 13.5 ms term); far ×10 stays flat; hero framing unchanged
  outside the threshold (image parity where footprint > threshold; crown-crop
  depth p99 within the silhouette bound already used by the band work);
  threshold 0 reproduces today's image bit-exactly.

### W2 — Tiered compute resolve  (M · medium risk · deps W0, pairs with W1)

The structural fix for Finding 3's resolve share and Finding 4's biggest rows,
and the permanent fix for render-pass unattributability:

- Replace the two full-screen fragment resolves + two overflow render passes
  with a compute resolve: a classify pass bins 8×8 tiles by arena depth
  {empty, shallow, deep, overflow}; indirect dispatch per tier runs a
  specialized kernel. Empty tiles write G-buffer defaults at memcpy cost;
  the deep kernel is the only one paying full register pressure.
- This deletes 2 of the 4 full-res 48 B/px MRT load/store cycles (overflow
  folds into the deep tier), the 2 `overflowArgs` passes, and the double count
  clear — and makes the primary resolve a *compute* pass, whose timestamps this
  machine reports truthfully.
- Watch item: the fragment resolve currently gets TBDR imageblock locality for
  its 4 MRTs; compute writes storage textures instead. Claybook shipped
  compute-based SDF visibility at 4 ms/720p on 2013 hardware, and our sky arm
  shows the tile hardware isn't where the time is — but keep the fragment path
  behind a flag until parity + a paired win is measured.
- **Gate:** image parity; G-buffer traffic per frame halves (Metal counters or
  derived); `hero-floor-far` and `base-hero` both non-regressing with the
  usual repetitions; per-tier trusted timestamps published in the lanes.

### W3 — The encode diet  (S · low risk · deps none, interleaves anywhere)

Mechanical, from Finding 4's inventory — each row is its own small PR:

| Item | Fix |
|---|---|
| Band pass with band off (`:6187`) | gate on `nearFieldBandPixels > 0` (`nearFieldBandActive` at `:6107` already computes it) |
| Glass pass with 0 records (`:8516`) | gate on `rasterGlassRecordCount > 0` |
| Fan-out 12 layers for 1 light (`:8619`) | size z-layers to `4 + activeLightCount` |
| Cull over capacity (`:6375`, `:6379`) | dispatch over published live-leaf count (the publication header already carries it) |
| Publication clear to buffer end (`:6371`) | clear only the 8-word header + bucket counters |
| Voxel-light populate fixed 256 wg (`:8582`) | indirect off the demand queue count |
| Double count-plane clear (`:6390`, `:6207`) | one clear serves both arms |
| Overflow passes at zero count | subsumed by W2; until then, leave (they're GPU-cheap) |

- **Cost model:** each is O(capacity)→O(live) or dead-code removal; the sum
  also shrinks `finish()` CPU with every pass/command removed.
- **Gate:** all existing suites green (`test:webgpu:svo-dry-render`,
  `hero-garden-hose{,-x10}`, parity tests); sky floor stays ≤ 1 ms; encode-CPU
  (W0's new number) drops measurably at 10×.

### W4 — CPU frame floor  (S–M · low risk · deps W0 for the CPU number)

- **`canonicalScene` → revision stamp.** Replace the per-frame deep-clone
  change detector with a monotonic scene revision (bump on every mutation
  path; the editor already routes mutations through a small surface —
  [[editor-drags-preview-locally]]). Fall back to the string compare only when
  a caller hands over an unversioned scene object.
- **Sway path made incremental:** repack only swayed records (they are known),
  refit only their BVH path to the root, drop the second validation walk and
  the per-frame FNV hash (debug-gate them), upload only the dirty byte ranges.
  Stop calling `clearPrimaryVisibilityCache()` wholesale from sway
  (`:7626`) — invalidate by dirty region or accept one-frame staleness on
  swaying content, so stationary reuse (W5 of the previous handoff) has
  something left to reuse.
- **Gate:** renderer CPU per frame (new W0 number) < 2 ms on hero at 1×, < 4 ms
  at 10×; no per-frame full-arena upload on a still-camera swaying scene.

### W5 — Pixel-linear lighting floor  (M · after W1/W2 land, informed by their lanes)

Deferred lighting's ~10 ns/px is next once the latency floor is gone —
untouched here because at the *floor* it is only ~4 ms at 800×460, but at
2 Mpx it is the largest surviving term:

- Tier the deferred lighting the same way W2 tiers the resolve (sky already
  early-outs; add "directly-lit, no-GI-detail" fast path chosen per tile).
- Size `full-res-relight` reconstruction work to receiver complexity (the
  guided-upsample fallback band already computes a mask).
- Re-measure the cone group at production `coneFanout=true` (the lanes have
  never seen it — Finding 5) before optimizing it.
- **Gate:** deferred + relight ≤ 4 ms at 1.47 M px on `hero-floor-far`;
  luminance error within the anatomy doc's thresholds.

### Dependencies

```
W0 ──► W1 ──► (W1+W2 paired measurement) ──► W5
  └──► W2 ──┘
W3, W4: independent, land anywhere
```

## 4. What NOT to do (measured dead ends, this session)

- **Do not chase GPU pass-count overhead.** Sky = 0.85 ms through the full
  encode. Merging passes for GPU reasons is a null; do it only where it
  deletes memory traffic (W2) or CPU commands (W3).
- **Do not attribute the floor to the analytic/record set.** ×10 records moved
  the far floor 0 ms. The exact-upgrade budget matters at *hero* framing, not
  at the floor.
- **Do not quote render-pass timestamps per pass** — still true, still the
  standing trap ([[svo-render-profiling-lanes]]). The far-frame tables in §1
  are usable only because the frame is serial (overlap ≈ 1.0).
- **Do not conclude from single runs** — ±5 % within-arm, ~9 ms batch drift.
- **Do not "fix" the smoke lane by removing `finish()` from its clock** without
  also reporting it — the CPU number is a real cost the app pays; it just must
  not be *added* to GPU time as if serial.

## 5. Risks

- **LOD popping at the threshold** (W1): a record crossing the footprint
  threshold changes appearance. Mitigations: hysteresis (the band code already
  has the pattern), threshold authored per scene, and the near-field band as
  the quality escape hatch. The gate's "threshold 0 = bit-exact" keeps the
  rollback trivial.
- **Coarse-level shading data** (W1): node-mip stores opacity, not
  material/normal. The interim "coarse step + no exact upgrade" variant needs
  no new data; full coarse resolve needs per-level material/normal aggregation
  in the derived builder — schedule as its own PR, oracle-covered
  (`svo-node-mip-cpu-oracle.ts`).
- **Compute resolve vs TBDR** (W2): losing imageblock locality could cost more
  than tiering wins on some content. The flag + paired measurement is the
  containment; Teardown/Claybook precedent says it lands.
- **Reuse semantics under sway** (W4): one-frame-stale swayed silhouettes are
  visible if the invalidation is too lazy. Author the staleness (frames
  budget), don't let it emerge.
- **Threshold interaction with the ×10 program:** W1's threshold changes the
  effective work at 10× too (in its favor), so re-run the record-scale pair
  after W1 and update `docs/svo-raster-visibility-handoff.md` §0's numbers.

## 6. Reproduction anchors

- Probe modules: `tmp/hero-far-camera-scene.ts`, `tmp/hero-sky-camera-scene.ts`
  (promote in W0).
- Lane: `tools/benchmark-svo-dry-frame-gpu.ts` with `SCENE_MODULE` +
  `TIMING=gpu` + `PHASE_TRACE=1` (command in §1). Per-stage truth when needed:
  xctrace via `--launch` + label env, per [[svo-render-profiling-lanes]].
- Key code: encode graph + resolves `lib/webgpu-svo-dry-scene.ts` (pass sites
  in Finding 4); arena contract `lib/webgpu-svo-brick-raster.ts`; footprint
  primitive `lib/svo-screen-space-termination.ts:57`; tuning defaults
  `lib/svo-render-tuning.ts:130–:170`; app frame loop `lib/webgpu-renderer.ts`
  (`publishRenderScene :1893`, `advanceLiveSceneAnimation :2037`, renderer
  construction `:1136–:1146`).
- Report JSONs from this session's arms: session scratchpad
  (`floor-probe.sh` + `*.json`) — the numbers are all inlined in §1, so the
  doc survives the scratchpad's deletion.
