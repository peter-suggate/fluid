# Voxel surface rasterization

The Render panel's **Primary visibility → Rasterized** selection uses cached
opaque voxel boundary triangles. Turn **Smooth surface** off to activate the
mesh. The URL value is `svoPrimary=mesh`; `FLUID_SVO_PRIMARY_TRAVERSAL=mesh`
selects the same production backend in diagnostic environments. Rasterized primary visibility is the default; GI composition is off by
default. Environment refinement requests depth 3 by default, with device-limit
fallback reporting the actual depth. See `docs/svo-depth3-rendering.md` for
validation and startup costs. The historical `raster` value still means brick proxies.

## Geometry and publication

`lib/svo/features/primary-visibility/svo-surface-mesh.ts` compiles into the existing dry-scene shader bundle.
`tests/svo-surface-mesh-scheduler-dawn.test.ts` runs its build kernels over a
two-brick synthetic octree: first build, reuse, an incremental re-extraction
from a dirty list, overflow rollback and growth, and a replacement flip.
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
motion and lighting changes reuse geometry. Every brick owns one contiguous
quad range in a bump-allocated arena, recorded in a per-leaf table with the
brick's octree address as its key. A changed publication is answered one of
three ways, decided on the GPU in `surfaceMeshPrepare`:

- **Incremental** (the common edit): the voxelizer's maintenance pass already
  keeps a dirty-brick list for each completed scene revision
  (`SparseVoxelStructuralRenderSource.sceneMaintenance`). When that list
  describes exactly the revision being answered (requested and completed
  revisions agree, it is the next one after the last consumed, and nothing
  overflowed) the mesh re-extracts only the dirty bricks, every brick whose
  slot key changed, and every brick touching a dirty brick's box (its faces
  against the edit may have changed). Each re-extracted brick frees its old
  range in place (the quads keep their slots with a zero extent until a
  compaction) and takes a new range past the cursor. The cached mesh keeps
  drawing throughout.
- **Replacement**: an untrusted dirty list, or a compaction once freed holes
  outnumber the live quads, rebuilds the whole mesh into a second arena the
  host provides on request, while the first keeps drawing; the last batch
  flips the arenas and the host retires the old one.
- **Initial**: the first build, or a build after a structural reset, with
  nothing to draw; the whole scene is traced until it completes.

While any build is pending the dirty boxes of the publication being answered
(up to 63 individually, else their union) mask the frame: the cull withholds
quads lying wholly inside a box and the background pass traces the pixels
whose rays cross one, so the edited region is exact from the first frame and
everything else stays rasterized. A build with no trusted list masks nothing
and, once drawn, keeps drawing.

Extraction runs in two phases per brick, a count and an emit with identical
traversal, so a brick's range is sized before it is written and no
whole-arena overflow can withdraw the mesh. The host paces one batch per
presentation: 512 bricks on a build's first presentation, doubling while it
stays pending, to 2,048 while the mesh is drawn (the frame stays interactive)
or 16,384 while nothing is drawn (every presentation of an initial build also
traces the whole scene at full resolution, and on a large scene that trace,
not the extraction, is most of the wall time: `hero-garden-hose-x10`, 8 ms of
extraction beside 83 ms of fallback per 2,048-brick presentation, ~291
presentations for 595,825 bricks at a fixed batch). A small edit's
re-extraction of its few bricks completes in one presentation.
`surfaceMeshBuildBricks` in `svo-surface-mesh.ts` is the ramp; the GPU worklist
and cursor remain authoritative, and a stale host receipt can neither restart
nor skip a build. The host reads a 224-byte state receipt every presentation
(`interpretSurfaceMeshState`, pure and unit-tested) and acts only on flips,
back-arena requests and overflow growth. No CPU payload readback or
per-frame mesh upload is required. Physical source-buffer replacement resets
the cache; current mapping uniforms also support world rescaling.

The quad arena starts at 64 MiB (or the device's smaller limit), with 32 bytes
per quad. An overflowing batch is rolled back whole to its checkpoint (its
bricks keep their previous ranges) and the GPU pauses; the host grows the
target arena by doubling, copies the retained prefix in queue order, and the
GPU resumes on seeing the larger binding without restarting the build. A
replacement build asks for a back arena sized for the live quads plus a
quarter's headroom (never below 64 MiB); the two arenas coexist until the flip.
If growth would exceed the allocation limit, fallback remains active; the
panel reports the required quad count, available capacity, allocated MiB,
limit, and build count. A subdivision stack failure reports an extraction
fault and does not trigger allocation. `FLUID_SVO_DRY_FRAME_SURFACE_MESH_BYTES`
can set a smaller hard ceiling in the Dawn harness. No geometry is dropped to
satisfy the budget.

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

## Filtered detail (2026-09-08)

The Frame panel's **Primary visibility** band has a **Filtered detail** stage
before **Primary rasterization**. Its master switch retains every setting when
turned off. Filtering defaults off; its saved detail threshold defaults to 1 px.

| Control | Range / default | Effect |
| --- | --- | --- |
| Detail threshold | 0.25–8 reference px; 1 | Slider and numeric input. Larger values permit coarser cells. Reference height is 460 px. |
| Maximum coarsening | Native, 2×, 4×, Full brick; Full brick | Caps geometry independently of smoothing; full brick clamps to the levels the brick actually has. |
| Normal smoothing | On | Uses baked normals independently of geometry coarsening. |
| Smoothing strength | 0–100%; 100% | Blends face and baked normals, then normalizes. |
| Transition stability | 0–30%; 15% | Hysteresis around the threshold. Coarsens below the lower edge and refines above the upper edge. |
| Normal agreement | 0–1; 0.5 | Minimum agreement for a coarse face's averaged normal. Higher values retain more flat faces. |
| Preserve close-up face normals | On | Fades native-level shading back to face normals between one and two thresholds on screen. |
| LOD colours | Off | Cyan: native, green: 2×, amber: 4×, pink: 8×. Other producers are absent. |

The advanced drawer holds transition stability, normal agreement and close-up
preservation. Per-control resets and **Reset filtering settings** restore these
values without enabling the stage. Selected-brick counts report resident surface
bricks per LOD before frustum culling, from asynchronous GPU receipts. Timings
are shared with Primary rasterization and are never added twice to frame totals.

The URL stores the enable flag as `svoMeshFilter`, separately from
`svoMeshLodPixels`. Other controls use `svoMeshNormals`, `svoMeshNormalStrength`,
`svoMeshMaxLevel`, `svoMeshHysteresis`, `svoMeshNormalAgreement` and
`svoMeshCloseNormals`. `svoStage=mesh-lod` selects the inspection view. Old URLs
with a positive valid `svoMeshLodPixels` and no enable flag still enable filtering.
The Dawn tools retain their explicit threshold environment switches.

**Extraction** caches all geometry levels. A coarse cell is solid if any of its
resident voxels is solid, and uses its first solid material. Each exposed coarse
face stores the mean of baked normals pointing into its face hemisphere and an
8-bit agreement value, `length(sum) / count`. Coarse rectangles merge only when
material, packed normal and agreement match; exact rectangles still merge on
material alone. Changing the agreement threshold only changes shading. Geometry,
normal-strength and agreement sliders do not rebuild shaders or meshes.

**Selection** runs once per resident brick in `surfaceMeshSelect`, before quad
culling. An eight-word per-leaf table retains the chosen level and its history.
The projected enclosing-sphere estimate follows the reference-height contract.
Coarsening/refinement uses the hysteresis band; a threshold or cap change bypasses
history, an edited/reallocated brick invalidates it, and a camera inside the
brick sphere forces native geometry. A large camera jump may cross multiple
levels in one frame. Every quad reads the same selected level, including the
no-cull diagnostic variant. The 224-byte state receipt includes LOD counts.

**Shading** samples native-level baked normals per fragment and uses cached
means on coarse faces. It applies the smoothing switch, agreement threshold,
strength and optional close-up fade without moving depth. Geometric normals
remain the rasterized faces for shadow/contact bias. Native geometry plus normal
smoothing is supported through the maximum-coarsening control.

**Inspection** stores selected level + 1 in bits 27–30 of the raster mesh's
split opaque material word. Production mesh shading masks these diagnostic bits
at `drySplitIdentityAt`; material identity, generation, motion and normal fields
remain intact. The normal overlay machinery decodes these bits into the LOD
palette. This does not add a render target or select the traced primary path.

**Two normals** now leave the fragment, which is what stops a baked normal from
also deciding where a ray starts. The quad's face goes into the G-buffer's
geometric slot and the baked normal into the shading slot — the contract has
had both since it was written, and the mesh is the first producer with two
different values to put in them. Shading is unchanged: the f32 geometry plane
still carries the baked normal at full precision, and every closure, the N·L
gate and the environment terms read it. Only the ray origins moved: the shadow
rays and the contact hemisphere bias along the face
(`dryGeometricNormal`), so a silhouette's side face fires outside its own cube
instead of back through it and the black speckle goes with it.

The face travels to the deferred lighting in the free high bits of the opaque
identity plane's metadata word (`DRY_OPAQUE_FACE_VALID`, bits 27..30: a present
bit and a six-axis code), because that plane and the f32 geometry plane are the
only two the lighting entry binds — the packed oct8 plane is not bound to it.
A producer with one normal leaves the bits clear and every reader falls back to
the normal it already had, the same float rather than a requantised one, so no
other pass in the frame moves by a bit. `dryRasterPrimarySurface` is now
`dryRasterPrimaryFacedSurface` called with the surface's own normal for a face.

Still open: global illumination gathers around the shading normal from an
origin biased along it (a whole voxel out, so it does not self-occlude the same
way), and the reduced cone prepass traces its own geometry rather than reading
this plane, so neither consumes the face yet.

Validation so far is offline: scale-1, scale-0.5 and no-cull shader variants
pass Naga, and `tests/svo-surface-mesh-detail.test.ts` covers the tuning
bounds, the URL round trip, and the shader's level machinery.
`tests/svo-gbuffer-normal-split.test.ts` pins the two-normal split — the packed
contract's two slots, the mesh as the only producer passing a face of its own,
the lighting's bias-versus-shade division — and validates the inline, split,
raster and reduced compositions under Naga. The emitted WGSL of every
composition without the mesh was diffed against the previous revision: the only
changes are the new declarations, the `_padding`/`aux` rename, and
substitutions that are the same value when the face bits are clear. No Dawn
capture or in-app measurement of the far camera has been taken.

**Cost (2026-09-09, `test:webgpu:hero-floor-far`, one run per arm, 800x460,
GPU pass timestamps, 64 warmups so the paced build finished in both arms):**
toggle off 1.442 ms median, toggle on 1.769 ms (+0.33 ms, +22.7%), 16-sample
ranges disjoint, well outside the lane's ~±5% single-run noise. The mesh
rasterization window is unchanged (0.131 ms both) and drawn quads fall
64,756 → 53,107; the whole delta sits in the deferred dry lighting window
(0.983 → 1.245 ms). Why lighting grows is unmeasured: candidates are more
pixels passing the N·L gate under smoother normals and so tracing shadow rays,
and the any-solid dilation covering more pixels. The knob for this lane is
`FLUID_SVO_DRY_FRAME_MESH_LOD_PIXELS` with `FLUID_SVO_DRY_FRAME_SURFACE_MESH=1`
and a zero screen-space threshold; the smoke tool's `FLUID_SVO_MESH_LOD_PIXELS`
is inert because that lane never enables the surface mesh. The balanced default
stays off until the lighting delta is attributed.
