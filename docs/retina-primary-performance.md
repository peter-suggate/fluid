# Retina primary visibility

Measured on 2026-10-25 with an Apple M1 Max, Dawn/Metal and GPU timestamps.
The pool uses its preset overview camera, 25 mm visual cells (refinement 1),
50 mm fluid cells, dual marching cubes, raster primary, and raster AO.
These are warmed dry-render measurements: fluid simulation and water compositing
are excluded. They do not reproduce the screenshot's running simulation state.
The baseline source is `767fc07f`.

## Why the primary band is large

In the supplied capture, primary rasterization accounts for 10.9 of the primary
band's 11.5 ms. Mesh maintenance is 0.07 ms and culling is 0.51 ms. The shared
raster pass draws background/planes, native imported meshes, voxel triangles and
rigid surfaces; its label does not mean all that time is octree traversal.

Resolution changes both fragment work and geometry admission:

- A 3448×1892 framebuffer has four times the pixels of 1724×946.
- Sample-coverage culling retains more small voxel triangles at the larger size.
- Native asset LOD selection uses projected error in pixels, so increasing
  resolution can select finer index streams.
- The four primary colour attachments and depth occupy 52 bytes per pixel:
  approximately 339 MB at the Retina size. The separate HDR output adds another
  8 bytes per pixel, bringing these textures together to about 391 MB.
  This is attachment storage, not a
  measurement of physical memory traffic; tile storage and compression matter.

The final resolution sweep reuses one scene, alternates the sizes twice, and
reruns production culling and LOD selection after each resize. Each measurement
has eight warmups and 64 isolated GPU timestamp samples:

| Viewport | Pixels | Drawn voxel records | Cull median, both visits | Raster median, first / second visit |
| --- | ---: | ---: | ---: | ---: |
| 1724×946 | 1,630,904 | 274,981 | 0.262 ms | 2.687 / 3.408 ms |
| 3448×1892 | 6,523,616 | 493,066 | 0.262 ms | 4.719 / 4.653 ms |

The draw is resolution dependent, but does not scale purely with pixel count.
It also pays for geometry, and desktop load affects absolute timings. The
retained voxel mesh is unchanged at 1,482,200 records.

Draw-removal experiments identify native imported assets as the largest marginal
cost, followed by voxel triangles. In the initial run, omitting native draws
reduced the pass from 4.850 to 2.490 ms; omitting voxel triangles measured
3.408 ms. The ending baseline had drifted to 6.095 ms, so these are diagnostic
attributions, not an additive breakdown or a precise optimization result.

A separate interleaved diagnostic kept native geometry and replaced only its
material/normal sampler with a trivial value. The pass measured 5.112 ms with
the trivial sampler versus 6.095 ms with production sampling. Sampling matters,
but removing it would still leave most of the total pass cost.

## Change and validation

Native asset indices previously retained extraction/simplification order. Each
LOD now receives vertex-cache triangle ordering once during asset preparation.
The encoder's vertex remap is inverted so every level and instance still shares
the original vertex buffer. Faces, winding, vertex positions, index counts,
LOD errors, normals, materials and render resolution are preserved.

The benchmark captures the previous native index buffers, then alternates old
and new buffers in ABBA order on the same device and published scene. Shader,
vertex buffers, draw ranges, attachments and camera stay the same. The final
3448×1892 comparison uses 16 warmups and 256 samples (128 per arm):

| Primary rasterization | Median | p95 |
| --- | ---: | ---: |
| Original order | 4.784 ms | 4.915 ms |
| Vertex-cache order | 4.653 ms | 4.981 ms |

This is a **2.7% median improvement**, with no demonstrated tail improvement.
An earlier, noisier comparison measured 7.733 → 7.340 ms. Neither result implies
that the screenshot's 10.9 ms becomes 4.65 ms. The screenshot also has a separate
13.2 ms lighting-visibility cost that this change does not address.

Overview, close-up and return captures are byte-identical in RGBA16F. The orbit
view differs at one of 6,523,616 pixels by at most 0.0000611 linear intensity;
its displayed RGB remains [75, 90, 62]. All captured depths match exactly.
GPU validation reports no errors.

Tests verify oriented triangle multiplicity, sparse shared vertex addresses,
original/reference buffer replay and resource lifetime. `npm run check:types`
passes. `npm run test:unit` passes with 1,201 passing, 127 skipped and no failures.

Sampler-loop unrolling and duplicate material-vote rejection did not improve
the paired median. Reusing adjacent decoded normals regressed 4.653 → 4.784 ms.
Those experiments were removed.

## Adaptive geometry investigation

The native selector already adapts per instance: it chooses the coarsest stored
index stream whose estimated world-space simplification error projects to at
most **0.25 physical pixels**. It projects using the nearest depth of the asset's
bounding sphere, which is conservative for elongated assets or a camera close to
a large asset. The sphere is also used for frustum culling. This is independent
of the normal/material field sampler; coarser geometry still evaluates the
original source field at each surviving full-resolution fragment.

The voxel mesh has a different policy. `dryMeshLodPixels` scales its reference
pixel threshold with viewport height, but returns zero for dual contouring and
dual marching cubes. In these smooth modes, the existing “Filtered detail”
control cannot lower geometry resolution. Sample-coverage culling only removes
triangles that cannot cover a raster sample; it is not a perceptual LOD.

The first experiment is therefore a native LOD error sweep, keeping the viewport,
materials, source meshes, shadows and voxel geometry fixed. The benchmark-only
`FLUID_SVO_DRY_FRAME_NATIVE_LOD_SWEEP=0.25,0.5,1,2` compiles selector variants and
alternates each candidate with the shipped quarter-pixel selector in ABBA order.
It records native triangle/instance counts, isolated primary GPU time and
display/depth/coverage differences in overview, orbit and close-up views. PNGs
and a `.native-lod.json` report are written alongside the requested output.
The sweep does not change the production selector.

### Full-resolution native LOD sweep (2026-10-25)

The pool sweep used 3448×1892, minimum AO 0, the preset overview, an azimuth
offset of 1.1 radians, and a close camera at one quarter of the preset distance.
Each comparison used eight warmups and 64 ABBA samples (32 per arm). The scene
is static and dry, with no simulated water. These timings isolate the primary
raster pass and exclude culling, lighting and presentation.

| Camera | Error allowance | Native triangles, reference → candidate | Raster ms, reference → candidate | Display RMSE / 255 | Surface pixels differing >8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overview | 0.5 px | 8,712,025 → 6,921,427 | 5.898 → 4.981 | 10.68 | 5.73% |
| Overview | 1 px | 8,712,025 → 6,564,109 | 4.850 → 4.063 | 11.99 | 7.35% |
| Orbit | 0.5 px | 8,812,259 → 6,943,205 | 4.850 → 4.522 | 11.15 | 6.30% |
| Orbit | 1 px | 8,812,259 → 6,564,109 | 4.784 → 4.063 | 13.12 | 8.15% |
| Close | 0.5 px | 8,944,639 → 5,306,922 | 4.260 → 4.194 | 5.20 | 1.72% |
| Close | 1 px | 8,944,639 → 4,569,572 | 4.260 → 3.932 | 7.04 | 2.68% |

Two pixels selected the same native index streams as one pixel in the overview
and orbit views. Close-up it selected 4,314,838 triangles, measured 3.998 ms and
increased RMSE to 8.91. Absolute clocks drifted in the first overview comparison;
ABBA samples are retained in the report, and the rows are independent paired
comparisons rather than a shared fixed baseline. Visible instance counts are
unchanged in every comparison. GPU validation reported no errors.

This is evidence of both geometry and pixel costs: cutting close-up native
triangles by 41% at 0.5 px barely changed raster time, while the overview gained
more. **No looser native error limit is shipped.** The half-pixel result already
shows objectionable speckling on the umbrella. Estimated geometric error does
not bound shading error when simplified geometry samples the original normal
and categorical-material fields. A future LOD policy needs an appearance-error
constraint as well as a silhouette/position constraint. These image errors are
substantially larger than the accepted adaptive AO changes.

Local evidence: `artifacts/primary-adaptive/pool.log`,
`pool.json.native-lod.json` and its overview/orbit/close PNGs. Type checking and
the unit suite passed (1,221 passed, 131 skipped).

Unlike the AO minimum, geometry error should not be normalized so that some
object must always select the coarsest mesh: that could remove a silhouette
even when every visible object needs detail. A screen-space error allowance,
with tighter treatment for thin features and stable transitions as the camera
moves, is the appropriate quality constraint. Smooth voxel geometry would need
compatible simplified meshes and seam handling before it can use that policy.

A larger alternative is to separate full-resolution visibility (depth and
surface identity) from material/normal evaluation, then share those evaluations
inside smooth regions with exact fallbacks at boundaries. That targets fragment
work and attachment traffic rather than triangle count, but requires a new
representation and reconstruction pass; its net benefit has not been measured.

## Follow-up experiment harness

`FLUID_SVO_DRY_FRAME_SURFACE_EXPERIMENTS` accepts a comma-separated list of
benchmark-only variants. The harness shares the production scene, resources,
culling and draw bundles; it alternates each variant with the reference in ABBA
order and captures overview, orbit and close-up images at the requested size.
It records each pair's timestamps, native triangle counts, display-space error,
depth changes and coverage changes in `.surface-study.json`. This is an isolated
primary-pass experiment, not an end-to-end frame-rate claim.

The candidates separate the possible savings:

- `lean-attachments` omits the two diagnostic identity MRTs. It is skipped when
  the renderer must preserve them for overlays or rigid-body picking.
- `homogeneous-material` skips material voting when all eight corners agree.
- `raw-oct-normals` removes corner-normal normalization, retaining normalization
  of the final interpolated normal. This changes interpolation weights.
- `tetra-native` replaces eight-corner trilinear appearance sampling with four
  tetrahedral corners, including categorical material voting.
- `voxel-vertex-normals` removes per-fragment source-normal reconstruction.
- `native-vertex-normals` caches source normals at native mesh vertices;
  `native-adaptive-normals` uses these only where a source cell is no larger than
  1.5 screen pixels, estimated from local-position derivatives. Both retain the
  original per-fragment material sampler and unchanged geometry.
- `native-zero-error` attempts zero-error simplification of the finest index
  stream. `native-appearance-lod` also tests coarser streams constrained by
  sampled normals, colour and locked material-boundary vertices.
- `native-box-depth` uses the oriented box's nearest camera depth instead of
  the enclosing sphere's depth, retaining the quarter-pixel LOD threshold.
- `native-sampling-ceiling` deliberately destroys native appearance to measure
  the maximum benefit available from eliminating that sampler. It is never a
  shipping candidate.

Normal caches and alternate LOD indices are prepared outside the timed loop.
The experimental index-range patch is also outside isolated raster timing;
shipping such a change would use the normal asset metadata publication and
would need a separate full-frame check. The cached normals use the CPU source
sampler, so their image comparison includes its source-normal precision versus
the shader's packed octahedral normals.

### Pool screening results

The 3448×1892 pool screening used eight warmups and 64 ABBA samples per
comparison. Columns are overview / orbit / close-up; positive percentages mean
less isolated raster time. Each number compares its own interleaved reference.

| Variant | Raster time reduction | Display RMSE / 255 |
| --- | ---: | ---: |
| raw-oct-normals | +0.0% / +1.4% / +3.1% | 2.993 / 3.463 / 1.386 |
| tetra-native | +7.9% / +5.5% / +12.3% | 4.791 / 5.149 / 3.411 |
| voxel-vertex-normals | +18.1% / +16.4% / +7.7% | 2.202 / 3.092 / 1.147 |
| native-sampling-ceiling | +15.3% / +15.1% / +27.7% | 73.751 / 72.382 / 57.492 |
| native-vertex-normals | -13.9% / -12.3% / +7.7% | 10.109 / 11.122 / 3.955 |
| native-adaptive-normals | -15.3% / -12.3% / +1.5% | 10.109 / 11.122 / 3.680 |
| native-zero-error | +0.0% / +0.0% / +1.5% | 0.000 / 0.000 / 0.005 |
| native-appearance-lod | -69.4% / -76.7% / -46.2% | 9.988 / 10.168 / 3.115 |
| homogeneous-material | +0.0% / +0.0% / +0.0% | 0.000 / 0.000 / 0.000 |
| native-box-depth | +0.0% / +0.0% / +1.5% | 0.166 / 0.483 / 0.009 |

The native appearance LOD experiment retained 32.15M / 32.15M / 19.44M
triangles, versus 8.71M / 8.81M / 8.94M in the reference: the appearance
constraints overwhelmed its looser geometric error. Cached native normals
added vertex traffic and interpolators while retaining the material sampler;
that regressed the two wider views. Neither is suitable to ship. The tighter
box-depth selector and exact material shortcut showed no meaningful broad
improvement. Zero-error simplification only reduced the close view's count.

Voxel fitted normals were the strongest quality/performance candidate. They
preserved depth and coverage in all views, with 0.53% / 0.75% / 0.18% of surface
pixels differing by more than 8 display levels. The follow-up tests gate source
normal reconstruction by projected cell size, blend through the transition,
and measure complete dry-frame GPU spans as well as isolated primary time.
No diagnostic attachment omission was allowed in the pool: picking requires
those planes even when rigid geometry shares the primary pass.

Evidence: `artifacts/primary-campaign/sweep2.json.surface-study.json` and its
three reference/candidate PNG sets. GPU validation reported no errors.

### Adaptive normals and exact alternatives

The second screening repeats the pool and adds the Blender garden at
3448×1892. It includes 64 interleaved samples of the complete dry-render GPU
span for every comparison. The three adaptive variants stop source-normal
reconstruction below 1, 2 or 4 projected cell pixels and blend back to full
reconstruction at twice that threshold. The footprint is the smaller of the
world-position derivatives, retaining detail at grazing angles.

| Scene / variant | Raster reduction: overview / orbit / close | Complete dry-frame reduction | Largest display RMSE / 255 |
| --- | ---: | ---: | ---: |
| pool / voxel-adaptive-1 | +0.0% / +0.0% / -1.5% | -1.3% / -1.2% / -0.5% | 0.999 |
| pool / voxel-adaptive-2 | +1.4% / +1.4% / -1.5% | +0.0% / +0.0% / +0.0% | 2.432 |
| pool / voxel-adaptive-4 | +4.2% / +2.7% / +0.0% | +1.3% / +0.6% / +0.5% | 3.092 |
| pool / voxel-vertex-normals | +16.9% / +16.4% / +7.7% | +7.6% / +6.8% / +2.4% | 3.092 |
| pool / homogeneous-material | -2.8% / +0.0% / +0.0% | -0.6% / -0.6% / +0.0% | 0.000 |
| hero / voxel-adaptive-1 | -4.3% / -3.8% / +2.0% | -1.2% / +0.0% / +0.0% | 0.000 |
| hero / voxel-adaptive-2 | -1.6% / -2.0% / +0.0% | +0.0% / +0.6% / +0.6% | 0.009 |
| hero / voxel-adaptive-4 | -6.2% / +0.0% / +2.1% | -0.6% / +0.0% / +0.6% | 0.000 |
| hero / voxel-vertex-normals | +1.9% / -2.0% / -1.9% | -0.6% / +0.0% / -0.6% | 0.000 |
| hero / lean-attachments | +9.4% / +3.8% / +7.4% | +1.8% / +1.2% / +1.8% | 0.000 |
| hero / homogeneous-material | +5.7% / +5.8% / +5.7% | +1.8% / +0.6% / +1.8% | 0.000 |

The dynamic adaptive branch mostly loses the unrestricted shader's benefit,
even when its image approaches the same fitted-normal result. This suggests
shader complexity/occupancy or the cost of its runtime decision, rather than
only the number of source samples, remains important. This is an inference,
not a hardware register-counter measurement. The follow-up split test uses
separate near and far pipelines, classifies triangles by projected cell size,
and pays for the extra vertex submissions in both isolated and full-frame
measurements. A further exact test specializes out source types absent from
the scene's maintenance publication.

The garden's attachment and uniform-material variants preserve every captured
output bit. The retained compact-target implementation retains ready full-target
pipelines and switches back for diagnostics, rigid-body picking and seam closure.
It removes 24 B/pixel of attachment footprint from the pass, changing its colour
and depth targets from 52 to 28 B/pixel. Full diagnostic textures remain allocated;
this is not a claim of reduced texture residency or 24 B/pixel less external
traffic, since the previous body-free path already discarded those stores.

The material fast path preserves the original distance and normal arithmetic;
when all eight non-air corners agree, it returns their material without the
repeated weighted election. Mixed materials retain the original vote order and
tie behaviour. It is an exact work reduction, with no visual-quality setting.

### Separate normal pipelines

A follow-up draws near and far voxel triangles through separate pipelines,
using conservative projected cell size to select the list. The first version
submits the original vertex stream twice and clips the other list in each
vertex shader. It keeps the original geometry and blends source normals back
in between one and two times the selected threshold.

| Pool variant | Raster reduction: overview / orbit / close | Complete dry-frame reduction | Display RMSE / 255 |
| --- | ---: | ---: | ---: |
| voxel-split-2 | +5.6% / +0.0% / -9.2% | +3.8% / +1.2% / -2.4% | 2.199 / 2.370 / 0.512 |
| voxel-split-4 | +9.9% / +8.2% / -3.1% | +4.4% / +3.1% / -1.9% | 2.202 / 3.092 / 1.114 |

Depth and coverage remain unchanged. The split improves the wider pool views,
but regresses close-up rendering. Neither split yields a useful complete-frame
garden gain. Specializing away unused normal-source types also produces no
broad improvement (pool complete-frame changes: 0.0% / 0.0% / +0.9%).

The final partition variant moves classification into production culling and
writes near/far index lists with independent indirect counts. Each triangle is
submitted once. It preserves the original visibility output for other consumers
and checks that both lists' vertex counts sum to the original submitted count.
Its complete-frame measurement includes the additional culling work.

Evidence: `artifacts/primary-campaign/split-pool.json.surface-study.json` and
`split-hero.json.surface-study.json`, with their reference/candidate PNGs.

The first partition screening preserves depth, coverage and submitted vertex
counts, but still regresses the complete close-up frame. Its 64-sample run has
substantial desktop/clock variation, so percentages are screening evidence:

| Pool partition | Raster reduction: overview / orbit / close | Complete dry-frame reduction |
| --- | ---: | ---: |
| blended, 2 px | +15.2% / +3.6% / -1.0% | +4.5% / +2.0% / -2.5% |
| blended, 4 px | +17.0% / +19.8% / +1.9% | +11.2% / +5.8% / -3.1% |

The near/far vertex census at 4 px is 141,036 / 1,338,162 for overview,
80,070 / 1,567,848 for orbit, and 545,019 / 173,868 for close-up. The variant
changes shading only; it does not silently drop triangles. Median reductions
computed within each ABBA block also show the wide-view gain and close-view
loss (complete frame: +7.7% / +6.8% / -3.4%).

Evidence: `artifacts/primary-campaign/partition-pool2.json.surface-study.json`.

A separate `voxel-boundary-gradient` test replaces the six central-difference
lookups with the gradient from the first trilinear lookup. Display RMSE is only
0.056 / 0.036 / 0.032, with unchanged depth and coverage, but complete-frame
changes are 0.0% / -0.5% / 0.0%. It is rejected: it trades continuous boundary
normals for no demonstrated frame benefit in this scene. Evidence:
`artifacts/primary-campaign/boundary-gradient-pool.json.surface-study.json`.

The hard partition retains the original full-quality near shader and removes
normal reconstruction entirely from the far shader. It avoids the blended
variant's near-fragment decision. A longer 256-sample pool run gives:

| Hard partition | Raster reduction: overview / orbit / close | Complete dry-frame reduction | Display RMSE / 255 |
| --- | ---: | ---: | ---: |
| 2 px | +12.2% / +4.5% / +0.0% | +5.7% / +2.5% / -0.7% | 1.994 / 1.472 / 0.000 |
| 4 px | +19.8% / +19.1% / +0.9% | +8.9% / +8.3% / +0.0% | 2.202 / 3.092 / 1.052 |

Depth and coverage match in all views. At 4 px, 0.53% / 0.75% / 0.15% of
surface pixels differ by more than eight display levels; differences concentrate
on procedural tree branches. The fixed-camera garden comparison is visually
exact and has only about 1,500 voxel vertices, all in the near list. Its paired
complete-frame changes are small and do not establish a useful gain there.

Evidence: `artifacts/primary-campaign/partition-hard-pool.json.surface-study.json`
and `partition-hard-hero.json.surface-study.json`.

An integration trial reuses the two ends of the existing visible-index half,
leaving the candidate half unchanged. It preserves submitted counts and the
prototype's images without allocating extra buffers. The first trial used
shared shader helpers with constant near/far arguments; it measured only
+6.0% / +4.1% / +0.0% raster and +1.8% / +2.7% / -0.3% complete-frame gains.
The follow-up emits separate shader entry bodies to test whether specialization
recovers the prototype's benefit. This is a compiler-cost hypothesis, not a
measured register-pressure attribution.

Explicit shader entry bodies alone did not recover the prototype's gain:
+2.9% / +7.5% / +1.1% raster and +0.9% / +1.8% / -1.3% complete-frame
reductions. Inspection then found that this integration retained the legacy
box-voxel filter in its far fragment shader. Far-list membership is restricted
to smooth reconstructed triangles, so that path can be omitted statically as
it was in the original prototype. Evidence for the intermediate attempt:
`artifacts/primary-campaign/integrated-static-pool.json.surface-study.json`.

A separate correctness sweep adds six zoom distances, 0.4–0.9 times the preset
distance. Every capture preserves depth, coverage and the reference submitted
voxel count. Display RMSE is 0.896 / 2.063 / 2.176 / 2.154 / 2.129 / 2.175;
0.10–0.53% of surface pixels differ by more than eight display levels. These
are still-image checks, not a temporal anti-popping test. This run uses shared
GPU check slots and its timing samples are deliberately excluded from the
performance evidence. Evidence:
`artifacts/primary-campaign/integrated-zoom-pool.json.surface-study.json`.

### Retained adaptive normal reconstruction

The final integration omits both source-normal reconstruction and legacy
box-voxel filtering from the far shader. Only smooth reconstructed triangles
can enter that list; ordinary voxel boxes retain their original path. Near and
far triangles occupy opposite ends of the existing visibility allocation, so
there is no additional index-buffer allocation or duplicate geometry submission.
The fixed four-physical-pixel threshold uses cell size, viewport height, field
of view and conservative triangle depth. It changes normal reconstruction,
not mesh geometry or render resolution.

This restored the prototype's benefit. The 3448×1892 confirmation uses 256
interleaved samples for each isolated-primary and complete-frame comparison:

| Pool camera | Primary raster ms | Reduction | Complete dry-render ms | Reduction |
| --- | ---: | ---: | ---: | ---: |
| overview | 6.554 → 5.243 | 20.0% | 14.090 → 13.042 | 7.4% |
| orbit | 7.078 → 5.702 | 19.4% | 14.811 → 13.500 | 8.8% |
| close | 6.423 → 6.423 | 0.0% | 20.185 → 20.382 | -1.0% |

Within-block paired medians give complete-frame reductions of +7.8% / +7.5% /
-2.3%. The close view has little work to save and retains a small partitioning
cost; this is not a universal speedup. Depth, coverage, native counts and total
submitted voxel counts match the reference. The normal changes have the same
small visual footprint as the hard-partition prototype. A hard quality boundary
can change small highlights as a triangle crosses it; continuous-motion
perceptual acceptance has not been established by the still-image sweep.

Adaptive normals are enabled by default. The internal
`surfaceMeshAdaptiveNormals: false` option, or
`FLUID_SVO_DRY_FRAME_ADAPTIVE_NORMALS=0` in the benchmark, restores full source
normal reconstruction. The portable non-subgroup path also passes production
inspection, body-count and seam-closure transitions. It keeps the same indirect
counts and attachment/pipeline selection contract. The final GPU check also
asserts conservation across every tested transition and reports no validation
errors.

Evidence: `artifacts/primary-campaign/integrated-pruned-pool.json.surface-study.json`,
its reference/candidate images, and `production-adaptive-final.log`.
Final validation: `npm run check:types` passed; `npm run test:unit` passed
with 1,223 tests passing, 131 skipped and zero failures
(`artifacts/primary-campaign/unit-shipping-final.log`).

The screened local alternatives are now exhausted: less faithful native
sampling, cached vertex normals, looser native LOD, normal blending, duplicate
vertex submissions and boundary-gradient shortcuts either lose quality too
widely, add work, or fail to improve complete frames. Larger changes such as
visibility-buffer shading, geometry preprocessing and on-demand picking are
separate architectural projects, not demonstrated wins from this campaign.

### Retained exact optimizations

Commit `11a8f6c6` retains compact primary targets and the uniform-material
fast path. A longer garden confirmation uses 256 ABBA samples per comparison
(128 per arm), both for the isolated raster pass and for the complete dry-frame
GPU span. It compares the combined changes with the campaign's original
four-target/material-election reference on the same device and scene.

| Camera | Primary raster ms | Reduction | Complete dry-render ms | Reduction |
| --- | ---: | ---: | ---: | ---: |
| overview | 5.374 → 4.653 | 13.4% | 12.780 → 12.386 | 3.1% |
| orbit | 5.308 → 4.719 | 11.1% | 14.615 → 14.090 | 3.6% |
| close | 4.325 → 3.801 | 12.1% | 15.073 → 14.483 | 3.9% |

All captured depths and coverage match. Overview and close-up images are exact;
the orbit PNG differs at one pixel out of 6,523,616 (display RMSE 0.0086/255
across surface pixels). This is not a claim of a 3–4% gain in a full simulated
water frame. The pool has rigid bodies and retains full identity targets for
picking, so it does not receive the attachment saving in its usual state.

Production GPU checks separately toggled inspection, body count and seam
closure in both scenes. They verified the selected attachments, bit-identical
body-free output with/without diagnostic targets, and restoration of the full
render after toggling. GPU validation reported no errors. Type checking and the
unit suite passed: 1,222 tests passed and 131 skipped.

Evidence: `artifacts/primary-campaign/confirm-hero.json.surface-study.json`,
its PNGs, `production-pool.log`, `production-hero.log`, and `unit-final2.log`.
The correctness runs share GPU slots and their wall-time profile is not used
as performance evidence.

## Reproduction

```sh
export WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal
export FLUID_SVO_DRY_FRAME_SCENE=backyard-pool-party
export FLUID_SVO_DRY_FRAME_SCENE_MODULE=tools/preview/pool-party-live.ts
export FLUID_SVO_DRY_FRAME_WIDTH=3448 FLUID_SVO_DRY_FRAME_HEIGHT=1892
export FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT=1
export FLUID_SVO_DRY_FRAME_TRAVERSAL=raster-primary FLUID_SVO_DRY_FRAME_SURFACE_MESH=1
export FLUID_SVO_DRY_FRAME_MESHER=dual-marching-cubes
export FLUID_SVO_DRY_FRAME_SPECIALIZED_LIGHTING=1
export FLUID_SVO_DRY_FRAME_CONE_TRACING=raster-ao
export FLUID_SVO_DRY_FRAME_CONE_FANOUT=0 FLUID_SVO_DRY_FRAME_CONE_SCALE=1
export FLUID_SVO_DRY_FRAME_TIMING=gpu FLUID_SVO_DRY_FRAME_PROFILE_SECONDS=1
export FLUID_SVO_DRY_FRAME_WARMUPS=8 FLUID_SVO_DRY_FRAME_CYCLES=64
export FLUID_SVO_DRY_FRAME_OUT=artifacts/retina-primary/repro.json
export FLUID_SVO_DRY_FRAME_PRIMARY_RESOLUTIONS=1724x946,3448x1892,1724x946,3448x1892
bash tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/benchmark-svo-dry-frame-gpu.ts
```

`FLUID_SVO_DRY_FRAME_PRIMARY_ABLATION=1` omits each draw category in turn.
For paired ordering comparisons, capture the original implementation with
`FLUID_SVO_DRY_FRAME_PRIMARY_INDEX_CAPTURE=<directory>`, then run the reordered
implementation with `FLUID_SVO_DRY_FRAME_PRIMARY_INDEX_REFERENCE=<directory>`.
This comparison requires identical vertex buffers and LOD ranges. The benchmark
rejects missing reference buffers and unequal index buffer sizes.

Local evidence is in `artifacts/retina-primary/`: `final.log`,
`final.json.primary-resolutions.json`, `indices-comparison.json`,
`index-baseline.rgba16f`, `indices.rgba16f` and their camera variants.

For the paired surface experiments, use the common setup above, then:

```sh
unset FLUID_SVO_DRY_FRAME_PRIMARY_RESOLUTIONS
export FLUID_SVO_DRY_FRAME_SURFACE_EXPERIMENTS=voxel-partition-2,voxel-partition-4
export FLUID_SVO_DRY_FRAME_SURFACE_FULL_FRAME=1
bash tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/benchmark-svo-dry-frame-gpu.ts
```

For the retained changes' garden confirmation, select
`FLUID_SVO_DRY_FRAME_SURFACE_EXPERIMENTS=exact-combined`, scene
`hero-garden-hose-x10-blender`, module `tools/preview/hero-blender.ts` and
refinement `0`. The experiment wrapper deliberately restores the campaign's
four-target/material-election reference before comparing candidates, even
though the production renderer now uses the exact optimizations by default.

For production target transitions, unset the surface experiment variable and
set `FLUID_SVO_DRY_FRAME_CHECK_PRIMARY_TARGETS=1`; run through the GPU wrapper's
`check` mode. This validates diagnostics, rigid-body picking and seam-closure
transitions. Timing comparisons must use the wrapper's exclusive `timing` mode.

For the retained adaptive integration, use
`FLUID_SVO_DRY_FRAME_SURFACE_EXPERIMENTS=voxel-integrated-partition` and
`FLUID_SVO_DRY_FRAME_SURFACE_FULL_FRAME=1`. The surface-study wrapper disables
production adaptive normals in its reference and replays the integrated
candidate against it. `FLUID_SVO_DRY_FRAME_SURFACE_ZOOM_SWEEP=0.4,0.5,0.6,0.7,0.8,0.9`
adds the intermediate camera captures. Use `FLUID_SVO_DRY_FRAME_MESH_SUBGROUPS=0`
with the production target check to exercise portable workgroup compaction.
