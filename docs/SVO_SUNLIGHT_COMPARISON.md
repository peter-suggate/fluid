# Whole-scene sunlight comparison

The reduced-rate production worker was still clipping directional shadows to
the simulation container. A receiver outside that box returned zero visibility,
so much of the backdrop went dark even though full-resolution lighting was
correct. This was not a necessary trade-off for performance.

## Changes

- The fan-out worker uses the node-mip world bounds and treats an empty ray
  segment as clear, matching the full-resolution path.
- Its scene layout now binds the existing payload arena so it can use the same
  stored-ring footprint, receiver escape and content-ceiling exit as the main
  cone marcher. Non-dense payloads return an empty backdrop table.
- Sunlight cache addressing follows each backdrop ring's stored voxel level.
  Cache population reconstructs the receiver at that same level.
- Fluid coverage no longer disables the static visibility cache. The shader
  already applies current-frame fluid optical depth *after* the cached solid
  visibility, and continues to correct rigid blockers at full resolution.
- **Cache sunlight** is an opt-in lighting control. Changing it requests a
  matching production pipeline; the default remains off. It costs significant
  memory and adds little to the 2×2 path in this scene.

## Measurements

Apple M1 Max, hero-garden-hose-x10, four detail rings, 800×460, unchanged
48-step cone budget and AO. One world build per comparison; 32 warmup frames
and eight serialized submit-to-fence samples per arm, with arm order reversed
on the second repetition. CPU encoding and compilation are excluded. These
are renderer measurements, not total frame times with a running fluid solver.

| View | Full | 2×2 | Full + cache | 2×2 + cache |
|---|---:|---:|---:|---:|
| Hero | 31.29 / 31.73 | 17.72 / 17.27 | 29.97 / 29.26 | 17.30 / 17.30 |
| Low orbit | 35.08 / 35.92 | 23.74 / 24.52 | 31.89 / 31.82 | 24.05 / 23.84 |
| Wide | 32.98 / 31.38 | 19.85 / 19.32 | 28.98 / 29.84 | 20.61 / 19.29 |

Times are milliseconds. Cache allocation is 207,474,688 bytes (197.9 MiB).
The camera-motion median was 31.83 ms full, 18.07 ms 2×2, 28.83 ms cached full,
and 18.13 ms cached 2×2. Captures between motion samples perturb pacing, so
these are an orbit smoke check rather than sustained animation percentiles.

The corrected 2×2 images have mean display-channel errors of 2.28, 2.22 and
2.59 out of 255 against full-rate lighting; before the fix they were 21.25,
12.04 and 34.15. Geometry depth is identical in the static and camera-motion
comparisons. About 7.5–9.6% of pixels differ by more than 8/255 in at least
one channel: reduced shading is still an approximation around fine details.
Full-rate cached images are closer (mean channel error 0.27–0.48/255), but
the voxel-centre cache is also not pixel-exact.

Raw evidence: `artifacts/backdrop-svo/slice9-sunlight/` (before),
`slice9-cache-fluid/` (cache/fluid changes), and `slice9-final/` (worker fix).
Each contains logs, PNGs, raw linear radiance/depth captures and JSON results.

## Fluid and edits

A deterministic fluid coverage fixture uses the production fill, mip and
lighting-composition paths without advancing a solver. It attaches an empty
volume, fills an elevated slab, then empties it without disabling the cache.
The filled state must change the image; emptying must restore it bit-for-bit.
With the corrected worker, the fluid-attached cache reported 110,560 hits.
The full and cached paths both restored the empty-water image exactly.

Sun rotation and tree removal exercise live light and geometry publication.
The tree comparison must remove the tree once and compare every arm over that
same publication: repeated remove/restore cycles can repack the live voxel
arena, so their images do not isolate a lighting change. Cached visibility is
warmed before removal to exercise invalidation, and the tree is restored after
the comparison.

## Repeat

Keep the browser scene idle and run `npm run test:dawn:sunlight`. It acquires
the repository GPU lease, captures the four arms, exercises movement, sun edits,
tree removal and changing fluid, then checks the linear images. It does not
change any shipping timing ceiling or the Sparse CM12 gate.

The image check requires unchanged depth and mean absolute linear luminance
error no greater than 0.025. This is a broad coverage-regression gate, not a
claim that 2×2 images are identical or that visual inspection is unnecessary.

The final `npm run test:dawn:sunlight` run passed; evidence is in
`artifacts/backdrop-svo/sunlight-check/`. All 33 image comparisons preserved
depth exactly and passed the luminance bound, including the first and settled
frames after removing the tree. Five focused CPU tests and a scoped TypeScript
check over the changed files passed. The whole-repository type check currently
reports unrelated errors in simulation tests/tools being edited concurrently.
The module-boundary check also reports existing violations outside this change.

The next performance target is the cache's memory layout and demand/population
cost. The present atlas allocates space for all node-mip pages, while only
surface entries are useful. Turning it on everywhere is not yet justified by
the measured 2×2 results. Water itself is no longer a reason to exclude it.
