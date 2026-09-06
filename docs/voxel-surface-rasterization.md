# Voxel surface rasterization

The Render panel's **Primary visibility → Rasterized** selection uses cached
opaque voxel boundary triangles. Turn **Smooth surface** off to activate the
mesh. The URL value is `svoPrimary=mesh`; `FLUID_SVO_PRIMARY_TRAVERSAL=mesh`
selects the same production backend in diagnostic environments. Rasterized primary visibility is the default; GI composition is off by
default. Environment refinement requests depth 3 by default, with device-limit
fallback reporting the actual depth. See `docs/svo-depth3-rendering.md` for
validation and startup costs. The historical `raster` value still means brick proxies.

## Geometry and publication

`lib/svo/svo-surface-mesh.ts` compiles into the existing dry-scene shader bundle.
It reads accepted structural nodes, leaf lifecycle and the shared identity
codec, including dense, occupancy and banded payloads. It does not modify the
solver or use authored pre-voxelization geometry as a substitute for voxels.

Each resident voxel brick emits solid-to-empty boundary quads. Within a brick,
matching coplanar identities merge into rectangles. At brick boundaries a
recursive neighbour query splits faces until each patch has uniform adjacent
coverage. This handles coarse/fine transitions and partially empty fine
children. Integer finest-cell lattice coordinates preserve shared positions;
the vertex stage applies the current metre mapping.

The GPU checks topology and scene-geometry revisions before extraction. Camera
motion and lighting changes reuse geometry. A changed publication rebuilds the
whole mesh across frames, extracting at most 128 bricks per frame. Rays remain
active until the complete mesh is published; this implementation does not yet
provide per-chunk incremental remeshing. No CPU payload readback or
per-frame mesh upload is required. Physical source-buffer replacement resets
the cache; current mapping uniforms also support world rescaling.

The quad arena starts at 64 MiB (or the device's smaller limit), with 32 bytes
per quad. Extraction counts the complete requirement even after storage fills.
When that requirement fits the device's storage-binding and buffer limits, an
asynchronous receipt grows the arena to the measured size plus up to 12.5%
headroom. A new bounded build completes before publishing the replacement. Previously
submitted commands retain the old allocation until GPU completion.

Overflow withdraws the entire mesh and uses current-frame rays while sizing and
rebuilding. It never draws a truncated mesh. If the requirement exceeds the
allocation limit, fallback remains active; the panel reports the required quad
count, available capacity, allocated MiB, limit, and build count. A subdivision
stack failure reports an incomplete count and does not trigger allocation.
`FLUID_SVO_DRY_FRAME_SURFACE_MESH_BYTES` can set a smaller hard ceiling in the
Dawn harness. No geometry is dropped to satisfy the budget.

The Frame panel calls this stage **Primary rasterization** and splits its timing
into **Mesh update**, **Planes / ray fallback**, **Mesh culling**, and **Mesh draw**. The subtitle
reflects the sampled active backend, including budget fallback; status receipts
are asynchronous and can lag by roughly 30 rendered frames. Ray mode retains
**Primary traversal**.

## Rendering

Before drawing, a GPU pass rejects back-facing quads and quad bounds outside
the camera frustum. Surviving quad indices are compacted with one global
reservation per workgroup. Culling retains a tolerance band at boundaries and
does not discard subpixel geometry or approximate occlusion. The visibility
index buffer costs 4 bytes per allocated quad (12.5% beyond the quad arena).
It is regenerated for each camera frame without rebuilding cached geometry.

Each quad uses a four-vertex triangle strip instead of six triangle-list
vertices, preserving the original diagonal. The panel reports drawn quads after
culling, separately from the full required quad count. The harness option
`FLUID_SVO_DRY_FRAME_SURFACE_MESH_CULLING=0` disables culling for paired captures.

An indirect triangle draw writes the existing four depth-tested surface planes,
using ordinary triangle reverse-Z depth and no fragment-depth output. Lighting,
water composition, secondary visibility and cone tracing retain their existing
paths. Exact accepted planar boundaries, analytic rigid bodies and glass retain
their separate handling. The historical authored scene-primitive proxy tier is
not drawn over the voxel mesh.

Smooth reconstruction currently remains a ray path: its per-cell tangent-depth
rule falls back to the cell entry depending on the viewing ray, so boundary
quads would not preserve it. Selecting mesh does not silently change the scene's
surface style. A camera inside occupied voxel space also uses rays to preserve
the existing inside-solid behavior.

## Runtime validation

The existing Dawn dry-frame harness accepts:

```bash
FLUID_SVO_DRY_FRAME_SURFACE_MESH=1
FLUID_SVO_DRY_FRAME_SURFACE_STYLE=voxel-flat
FLUID_SVO_DRY_FRAME_TRAVERSAL=raster-primary
FLUID_SVO_DRY_FRAME_SCREEN_SPACE_PIXELS=0
FLUID_SVO_DRY_FRAME_SHADING=split
```

Run it through `tools/run-webgpu-exclusive.ts`. The report includes
`scene.surfaceMeshDiagnostics`. Compare against `canonical-parametric` with
the same scene, camera, surface style and lighting. Use depth, hit/miss and
material comparisons rather than a packed-surface byte hash: producer metadata
intentionally changes, and hardware interpolation can round depth differently.

Full-scale and half-scale shader modules pass Naga validation. The initial
400×240 `garden-svo-lighting` Dawn/Metal smoke drew 84,223 quads, reported one
mesh build across repeated frames, and had no overflow or fallback. This is
evidence of a working cached draw, not a general performance claim.

The captured reference comparison had identical hit/miss masks across all
96,000 pixels, three material differences on common hits, and nine pixels
with view-depth differences above 1 mm. Maximum view-depth difference was
0.4101 m: this is not a pixel-identical replacement for DDA. That capture
preceded the final extension of greedy merging to uniform brick boundaries;
the final shader passed offline validation, but its Dawn rerun was blocked by
another task's GPU lease.

The required Sparse CM12 Dawn gate completed with 13/16 lanes passing:
`mixed-ratio-topology` exited with SIGSEGV, `mini32-performance` measured
51.3147 ms against 40 ms, and `mini64-performance` measured 65.536 ms against
50 ms. No ceiling or lane was changed. The renderer changes introduce no
reported TypeScript errors; the full workspace typecheck still reports errors
in solver, test and harness work. Receipts are retained under
`artifacts/voxel-surface-rasterization-2026-09-06/`.

Ray-work visualizations deliberately select the traced backend. Pixel tracing
under mesh selection is an SVO ray reference, rather than a replay of mesh
triangle rasterization.

## Mesh culling observation (2026-09-06)

The live Chrome `hero-garden-hose-x10`, refinement-2 tab reported 3,645,559
required quads and 423,180 drawn after culling at its existing close garden
camera. Sampled draw time was 1.77–1.80 ms and culling 0.41–0.45 ms. The user's
preceding observation was approximately 9 ms for mesh draw. This is a live-panel
observation, not an isolated paired benchmark or an image-difference proof.
Mesh update remained separately variable (1.81–4.19 ms in those observations).
Both scale-1 and scale-0.5 shader variants pass offline Naga validation.
