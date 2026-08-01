# SVO Raster-Assisted Primary Visibility (Teardown-style) — Implementation Handoff

Status: planned, not started. Companion documents:
`docs/SVO_PRIMARY_FAST_PATH_HANDOFF.md` (incremental levers on the existing
traced primary — beam prepass, adaptive rate, register diet) and
`docs/SVO_VOXEL_LIGHT_CACHE_HANDOFF.md` (lighting side). This document is the
strategic rework: replace the full-screen SVO traversal fragment shader with
hardware rasterization of resident bricks + a short in-brick DDA.

## 1. Why — baseline evidence

From `docs/SVO_GARDEN_1500_RENDER_PROFILE.md` (garden-svo-lighting, 1500×1500,
M1 Max, production split pipeline):

- `Sparse voxel primary visibility` = **26.95 ms, 55% of the cones frame,
  83% of the cones-off frame**. Unmoved by every lighting switch.
- **Not bandwidth bound**: 3.7 GB/s read vs ~78 GB/s observed peak, LLC 29%.
- **Register-pressure ceiling**: 22.2% occupancy / 34.6% ALU, identical across
  arms and partitions (`SVO_DAWN_OCCUPANCY_EXPERIMENTS_2026-07-31.md`).
- Cost is pixel-linear (~13–15 ms/Mpixel, no knee from 750² to 2000²).

Root cause: one fragment megakernel (`fragmentMain` / `dryVisibilityMain` →
`traceDrySolidScene`, `lib/webgpu-svo-dry-scene.ts:2426`) fuses four systems —
SVO octree traversal (stack + node-mip descent), terrain secant march, a
12-body rigid loop, and a glass-pane loop. Register allocation is the worst
case of all four; 78% of thread slots sit idle.

Why rasterization attacks this shape directly:

- The rasterizer performs the global "which brick does this pixel hit first"
  search — the part that owns the traversal stack and most registers — in
  fixed-function hardware, with early-Z killing occluded work.
- The fragment shader shrinks to a bounded DDA inside one known 8³ brick:
  tiny register footprint, high occupancy, and work concentrates on visible
  surfaces instead of every pixel marching from the root.
- Cost scales with *visible brick surface*, not pixels × traversal depth.

Shipped proof: Teardown rasterizes each object's OBB (backfaces, so the
camera can sit inside), then runs a mip-assisted DDA through that object's
3D texture in the fragment shader, with periodic depth-copy early-Z and
front-to-back object ordering — 1440p, thousands of destructible volumes, on
OpenGL-3.3-class hardware, no compute, no RT cores. The 2024–25 large-scale
hobby engines (Ethan Gore's Voxy-scale work) independently converged on
"raster primaries, trace secondaries." Fully compatible with dynamic camera
and per-frame deformation — the instance list is just re-emitted from the
sparse topology we already publish every frame.

## 2. Target architecture

A new traversal mode (working name `"raster-primary"`, peer of
`"canonical-parametric"`) that produces the **identical G-buffer + reversed-Z
depth contract** (`lib/svo-gbuffer.ts` — 3 color attachments, 36 B/pixel,
`packSvoGBufferPixel`, oct-encoded normals, motion/flags fields), so the
deferred lighting pass, cone prepass, picking, and water composite are
untouched and every existing image-diff harness doubles as the correctness
oracle.

### 2.1 Instance source

The packed topology already publishes exactly what instancing needs:
`SvoPackedTopologyView` (`lib/webgpu-svo-traversal.ts:38`) — nodes carry
`[mortonLo, mortonHi, level, …, leafIndex, …]`, leaves are 4 u32 words with a
node backlink and `voxelOffset` into the payload arrays. A small compute pass
over the leaves array emits an instance buffer of
`{brickAabbMin, level, leafIndex}` records (world AABB reconstructed from
node Morton + `SvoWorldMapping`). No CPU per-frame work; fluid deformation is
covered because this derives from the same per-frame topology publication the
traced path consumes.

### 2.2 Draw

Instanced cube (36 indices), **front-face culled** (render backfaces) so a
camera inside a brick still shades it — the Teardown trick. Vertex shader
positions the unit cube by instance AABB. Draw via `drawIndexedIndirect` with
a GPU frustum-culling compute pass writing the visible-instance list. No mesh
shaders in WebGPU; instanced indirect draws are the available and sufficient
path.

### 2.3 Fragment: bounded in-brick DDA

Per fragment: reconstruct the pixel ray, clamp entry `t = max(tNear, 0)`
against the instance AABB (computed in brick-local space for precision), then
DDA through the 8³ brick — worst case 22 axis steps, typically far fewer:

- `dryBrickMacroSkip` / macro-HDDA (`lib/webgpu-svo-dry-scene.ts:1204`,
  masks from `lib/svo-brick-occupancy.ts`) skips empty macro-cells.
- On occupied voxel: `materialOwners` lookup; primitive owners resolve exact
  analytic normals via `svoIntersectPrimitiveExact`
  (`lib/webgpu-svo-dry-scene.ts:2163`) — identical hit semantics to
  `traceLeafPayload` (`:2172`), which is the code to lift, not rewrite.
- Hit → pack G-buffer + write `frag_depth` (reversed-Z). Miss → `discard`.

### 2.4 Occlusion: recovering early-Z

Writing `frag_depth` disables hardware early-Z, and WebGPU cannot read the
bound depth attachment in-pass. Three stacked mitigations, cheapest first:

1. **Front-to-back ordering for free from the octree**: bucket instances by
   the 8 octant orders (camera-octant Morton sort in the culling pass);
   within reversed-Z greater-than depth testing, earlier fragments win and
   later occluded fragments fail the depth test *after* shading — ordering
   alone does not cull shading work but maximizes what mitigation 2 can kill.
2. **Depth-feedback culling (the Teardown mechanism)**: copy the depth
   texture every N draws (or once after the nearest octant buckets) into a
   sampled HiZ mip chain; the culling compute and/or vertex shader tests each
   brick's conservative nearest depth against HiZ and drops occluded bricks.
3. **Conservative depth output**: emit per-fragment
   `@builtin(frag_depth)` with a *conservative* declaration if/when Dawn
   exposes depth-clamp semantics; otherwise a two-draw variant (backface
   AABB depth prepass without payload DDA, then `depth_compare: equal`-style
   shading) is the fallback if depth-feedback proves insufficient.

The beam prepass from the fast-path handoff, if built first, doubles as a
free HiZ seed: its 8×8 conservative t-min tile texture is exactly a coarse
occlusion pyramid.

### 2.5 LOD: sub-pixel bricks

At 1500² a distant 8³ brick projects below a pixel; raw instancing of the
full leaf set would drown in vertex/overdraw work. Use the octree cut the
traced path already respects: `lib/svo-screen-space-termination.ts` defines
the projected-size termination criterion — apply the same test in the culling
pass and, for bricks beyond the cut, instance the *parent node* as a single
cube shaded from node-mip mean lanes (`SVO_NODE_MIP_LANES`,
`lib/svo-node-mip-pyramid.ts:15`) instead of descending to payload. This is
the instanced-raster mirror of the traced path's LOD termination, so images
should match by construction.

### 2.6 Non-SVO content

The megakernel's other three systems leave the fragment shader entirely,
which is itself an occupancy win for whatever remains:

- **Glass**: already rasterized (`glassRasterVertex/Fragment`,
  `lib/webgpu-svo-dry-scene.ts:2685/:2690`, pipeline at `:3101`) — the
  in-file template for pipeline layout, bind groups, and depth config.
- **Rigid bodies**: 12 analytic shapes → instanced proxy meshes (or
  per-shape bounding boxes with the existing analytic intersection in the
  fragment), writing the same G-buffer with `motionKind: rigid`.
- **Terrain**: heightfield → either a rastered grid mesh (standard, exact
  normals via `terrainNormalAt` finite differences) or, initially, keep the
  secant march as a full-screen pass that runs *after* brick raster with
  depth test enabled, so it only shades not-yet-covered pixels.

## 3. Phases and gates

Branch discipline per repo policy: no stash/checkout/reset in this worktree;
feature branch from the current tip. Retain all artifacts under
`artifacts/render-raster-primary/` (the 5X handoff lost its experiment
artifacts; `tools/render-svo-optimization-report.ts` still flags
`missing-evidence` — don't repeat that).

### Phase 0 — scoping measurements (½–1 day, no renderer changes)
- Count resident leaf bricks per scene (garden, hose-tank) from the topology
  publication; estimate overdraw = Σ(projected brick area)/screen area for
  representative cameras via a small CPU script over the leaves array.
- Pull DDA step histograms from the existing per-pixel instrumentation
  (`lib/svo-pixel-trace.ts` counters) to bound expected fragment cost.
- **Gate**: median overdraw < ~4× with the LOD cut applied. If the leaf count
  or overdraw is wildly above estimate, LOD (Phase C) moves before perf
  (Phase B); the plan reorders rather than dies.

### Phase A — correctness vertical slice (3–5 days)
- Instance-emission compute pass, frustum-culled indirect instanced draw,
  front-culled cubes, fragment DDA lifted from `traceLeafPayload`, full
  G-buffer + reversed-Z depth write. No ordering/HiZ/LOD yet. New mode wired
  as `"raster-primary"` behind the traversal-mode option, split pipeline
  otherwise unchanged.
- **Gate** (garden, static cameras, `tools/benchmark-svo-dry-frame-gpu.ts`
  image lane): hit/miss mask, depth, owner IDs, and normals match the traced
  primary except at brick-boundary ties; diff PNGs archived. Performance is
  explicitly *not* gated here — expect it to be slow (unordered overdraw).

### Phase B — occlusion and ordering (3–5 days)
- Octant-bucketed front-to-back ordering; depth-feedback HiZ culling
  (periodic depth copy + mip build + cull in compute); macro-mask early-out
  in the fragment.
- Counters: instances submitted/culled/drawn, fragments shaded, DDA steps
  (extend the pixel-trace header).
- **Gate** (1500×1500 garden, quiet-GPU xctrace via
  `tools/profile-svo-render-xctrace.ts`): raster primary beats the traced
  primary's 26.95 ms, and pass occupancy is materially above the traced
  path's 22% — occupancy is the thesis; if it doesn't move, stop and
  re-diagnose before adding features.

### Phase C — LOD cut (2–3 days)
- Projected-size cut in the culling pass; parent-node instancing with
  node-mip shading for beyond-cut bricks; hysteresis on the cut level to
  avoid popping under camera motion.
- **Gate**: 2000×2000 cost growth stays sub-linear in pixels (the traced path
  is linear — this is where raster should pull decisively ahead); no visible
  popping in an orbit capture; image diff vs traced stays within Phase A
  tolerance.

### Phase D — full integration (1 week)
- Rigid proxies + terrain raster (or post-pass), glass ordering audited,
  water composite background handoff verified
  (`setPendingSvoBackground` path), `stationaryPrimaryReuseEnabled`
  interaction decided (raster may be cheap enough to retire reuse; keep the
  flag), hose-tank fluid churn validated (instance buffer regenerates from
  topology publication — confirm no stale-brick frames on topology swaps).
- **Gate**: end-to-end frame time at 1500² beats the current 47.6 ms default
  configuration by ≥ 1.5× with cones on; full test suite green; new unit
  tests for instance emission (AABB reconstruction from Morton), cull
  correctness, and G-buffer parity.

## 4. Risks and pre-planned answers

| Risk | Signal | Answer |
|---|---|---|
| Overdraw at glancing angles / dense brick stacks | Fragments-shaded counter ≫ pixels | HiZ feedback cadence up; octant ordering audit; LOD cut earlier |
| `frag_depth` disables early-Z | Phase B occupancy up but time flat | Two-draw conservative-depth variant (§2.4.3); lean harder on compute-side HiZ culling |
| Cracks/seams at brick borders | 1-pixel dropouts along brick planes in diffs | Shared-plane AABBs from integer Morton math (no per-instance float rounding); entry-t epsilon in brick-local space |
| Sub-pixel brick popping at LOD cut | Shimmer in orbit captures | Hysteresis band + match traced path's termination constant exactly |
| Instance buffer churn under fluid | Stale bricks flash on topology swap | Double-buffer instance list keyed to topology epoch, same discipline as payload atlas swaps |
| Vertex cost at large leaf counts | Vertex ms grows in xctrace | LOD cut + cull earlier in pipeline; octant buckets already bound draw count |
| Raster path diverges from traced semantics over time | Diff drift in CI | Keep traced mode as the reference oracle in perpetuity; add a small parity test to CI on one fixed camera |

## 5. Fallback position

Every phase leaves the traced primary intact behind the traversal-mode
option. If Phase B fails its occupancy gate, the salvage is still real: the
instance/cull machinery becomes a HiZ source and the terrain/rigid/glass
unfusing carries over to the traced kernel — both feed directly into the
companion fast-path handoff. Nothing here strands.

## 6. Sources

1. Teardown: Gustafsson, "Raytracing Voxels in Teardown and Beyond" (talk);
   frame breakdowns at juandiegomontoya.github.io/teardown_breakdown.html and
   zacxalot.github.io/rendering/9-teardown/ — OBB backface raster + per-object
   DDA + periodic depth-copy early-Z + front-to-back ordering.
2. Ethan Gore / large-scale hobby voxel engines (2024–25): raster primaries +
   traced secondaries beats full-screen traced primaries at scale.
3. Aokana (arXiv 2505.02017, 2025): GPU-driven SVDAG open-world voxel
   rendering, up to 4.8× vs prior art — GPU-driven culling patterns.
4. Laine & Karras, "Efficient Sparse Voxel Octrees" (I3D 2010): the traced
   baseline and the projected-size LOD-cut formalism.
5. `docs/SVO_GARDEN_1500_RENDER_PROFILE.md` and
   `SVO_DAWN_OCCUPANCY_EXPERIMENTS_2026-07-31.md`: the measured register
   ceiling this plan exists to break.
