# SVO primary visibility on fluid scenes — what landed, and what is left

Date: 2026-08-20. Part 1 is implemented and measured. Parts 2 and 3 are not
started; part 4 is a dead end recorded so nobody walks it twice.

The question asked: `water-box-dam-break` renders slowly through the SVO even
though most of the frame is dark — is there a principled way to cut primary ray
traversal?

---

## Part 0 — Why "dark" is not "cheap"

The intuition that a dark frame should be a cheap one does not hold here. The
scene sits on the spotlit stage, and the stage floor is *voxelized*: it fills
roughly 70% of the viewport, and every one of those pixels is a grazing ray
that marches a long way through the octree before it hits anything. Darkness is
a lighting result, produced after the march. Only the pixels above the horizon
exit free, at the root AABB test, and they are the minority.

So the primary band is expensive in proportion to *screen coverage of voxelized
geometry*, not to how much light comes back. On this scene it measured 46% of
the SVO frame at both resolutions tested.

---

## Part 1 — Bounded reuse of the primary G-buffer (RETIRED)

> **Current status (2026-08-26):** the temporal primary-G-buffer cache described
> below has been removed. Primary visibility is traversed on every frame for
> every scene and tuning profile. The measurements remain as historical context
> only; they are not an available performance lever or an accepted benchmark
> arm. Spatial acceleration must earn its speedup while still performing current-
> frame primary visibility.

### What was wrong

`FluidLabRenderer.draw()` once built a primary-visibility coherence key only
when a stationary-reuse tuning flag was enabled and either no fluid solver was
present or the simulation was paused. That made the exact-reuse cache disappear
for the whole of every running fluid scene. The cache was subsequently removed
rather than broadening that gate.

The solver clause looks like a safety and is not one. **The fluid never enters
this G-buffer.** The dry scene is the background that the raster water pipeline
composites its surface over (`webgpu-water-pipeline.ts:2416`), and the fluid
publication reaches the dry renderer only as the cone-shadow coverage volume,
which the lighting pass resamples every frame regardless. `setFluidCoverage`
does not clear the primary cache, and never did.

What a running solver *does* own in the primary G-buffer is rigid pose, and only
when it owns the poses: with `gpuFluid.rigidRenderBuffer` present the shader
reads poses straight out of the solver's buffer, while the roster the coherence
key is built from is a *readback* of that buffer rather than its source, so the
key can be a frame behind the geometry. That — not the water — is the input the
key is blind to.

### What was evaluated

- A camera/scene coherence key once guarded an exact primary-G-buffer cache.
- The cache and its tuning/UI controls were subsequently removed as a temporal
  shortcut that concealed traversal cost and complicated correctness.
- `tests/svo-primary-reuse-gate.test.ts` now pins that removal: no tuning field
  and no stationary-reuse frame-graph node may return.

### Measured

Dry-frame lane, Dawn/Metal, M1 Max, `water-box-dam-break`:

| arm | 1791×904 | 2488×1256 (app resolution) |
| --- | --- | --- |
| always retrace | 13.63 ms | 25.95 ms |
| static-primary reuse | 7.34 ms | 14.09 ms |

The reused plane is the traced plane **byte for byte** — the scale-1 G-buffer
hashes are identical across the arm: `packedSurface 0x4340f762`,
`identityMedia 0x91647960`, `hardwareDepth 0x65bab534`. This is a skipped
recomputation, not an approximation of one; there is no image to trade away.

In the running app at 2488×1256, with the dam collapsing and the solver
stepping — the case the old gate blocked outright:

| arm | GPU busy | wall |
| --- | --- | --- |
| always retrace | 61.1 / 61.7 / 62.4 / 64.2 ms | 150–166 ms |
| static-primary reuse | 40.66 / 40.67 ms | 116 ms |

About a third of GPU busy time, on a running fluid scene, for a byte-identical
frame.

Note the panel shows no figure against **Primary traversal** in either arm.
That is not a skip indicator — render passes publish no trusted cost on Apple
GPUs at all (see `render-pipeline-accuracy-handoff.md` §A1). Read the top-line
`busy` for this A/B, never the row.

---

## Part 2 — Spatial invalidation for rigid bodies (REJECTED WITH THE CACHE)

The retired cache left one case fully unreused: a scene with solver-owned rigid
bodies while the solver ran. A proposed extension made invalidation *spatial* —
projecting previous/current body bounds to a screen rect and preserving cached
G-buffer data elsewhere. That extension is intentionally not part of the current
design because primary visibility must be current-frame work everywhere.

Four things make this a project rather than a follow-on tweak, and all four were
found by reading the encode path rather than by trying it:

1. **Reverse-Z.** `SVO_GBUFFER_RENDER_TARGET_CONTRACT` is `depthClearValue: 0`,
   `depthCompare: "greater"`. A partial pass has to load depth, so retraced
   fragments would be depth-tested against the *cached* depth and rejected
   wherever the new surface is farther — exactly the case of a body moving away
   and revealing background behind it. It needs a second pipeline variant with
   `depthCompare: "always"`, and the split-variant key already fans out over
   scale × GI × raster-rigid.
2. **Coherence contracts spread temporal state.** Deciding "this frame differs
   only in body pose" required threading cache identity through the renderer and
   benchmark harness; those contracts have now been deleted.
3. **Downstream passes read the whole G-buffer.** Primary seam closure and the
   raster-rigid certificate bridge both `loadOp: "load"` the same attachments
   and write compacted pixels; they need the same scissor or they write refined
   pixels outside the retraced rect from a cone-prepass plane that has moved.
4. **It does not reach the scene that prompted this.** `water-box-dam-break` has
   no rigid bodies, so part 1 already gives it the whole band.

There is also an existing mechanism in this space: past
`SVO_DRY_RASTER_RIGID_BODY_THRESHOLD`, `svoDryRigidPrimaryStrategy` rasterizes
rigid impostors into the primary G-buffer instead of tracing them. Any future
spatial work must compose with that path and still produce current-frame primary
visibility.

`FLUID_SVO_DRY_FRAME_SYNTHETIC_RIGID_MOTION=1` in
`tools/benchmark-svo-dry-frame-gpu.ts` marks one body moving and reports its
first six frames — that is the harness to verify a scissored retrace against,
and the G-buffer hashes make the check exact: a conservative rect must reproduce
the full retrace bit for bit.

---

## Part 3 — The beam prepass (NOT STARTED, and the bigger win)

Reuse is a temporal trick: it pays off only while the camera is settled, and it
returns nothing on the frames a user is actually orbiting. The standard
*spatial* answer for grazing rays over a sparse octree is the coarse
conservative-depth prepass from Laine & Karras 2010 ("Efficient Sparse Voxel
Octrees") — trace one ray per N×N pixel block against the octree, keep the
conservative near distance for the block, then start every fine ray at that
distance instead of at the root AABB.

This codebase has no such pass. The seam it would attach to is
`traceStatic` (`lib/svo/webgpu-svo-dry-scene.ts:5434`): `var minimum = 0.0` at
`:5442` and the cursor begin at `:5447` are where a per-block `tMin` would be
substituted for the root-AABB entry, alongside
`svoTraversalContinuationBegin` (`webgpu-svo-traversal.ts:607`, root AABB with
`ray.tMin` at `:630`).

This is the one that helps the stage floor, because a floor block's beam
terminates almost immediately and every fine ray in that block then skips the
entire march down to it. Unlike part 1 it also helps a moving camera.

---

## Part 4 — Screen-space LOD termination (DEAD END)

`FLUID_SVO_DRY_FRAME_SCREEN_SPACE_PIXELS=3` throws:

```
RangeError: Screen-space termination requires canonical inline or raster-primary split traversal
    at new SparseVoxelDrySceneRenderer (webgpu-svo-dry-scene.ts:6601)
```

The production arm is split canonical-parametric
(`createProductionSparseVoxelDrySceneRenderer`, `webgpu-renderer.ts:717`, with
`screenSpaceTerminationPixels: 0`), so the LOD cut is unreachable there without
new work in the split path. It is also the cut that produces terracing, which is
why it is off by default — see the `terracing-is-face-normals-not-refinement`
note before reaching for it.
