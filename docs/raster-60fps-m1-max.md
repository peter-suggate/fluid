# Raster 60 fps work on M1 Max

Target: hero-garden-hose-x10, refinement 3, raster primary, GI composition off,
1920×1080 physical pixels. A stationary camera or immutable scene is not an
acceptable assumption. The complete frame budget is 16.67 ms; the dry renderer
must leave room for other enabled frame work.

## Quality requirement

Camera activity must never change rendering quality. The old motion-dependent
AO (4→1) and area-light (2→1) sampling switches have been removed from the
main renderer, cone fan-out and pixel-trace diagnostics. Legacy settings preserve
the higher sample count; the UI exposes a single control for each budget.
The lower fixed-sample Performance-profile proposal was reverted. Quarter-rate
reconstruction measurements below are diagnostic experiments, not a shipped
quality reduction or a claim that the requested 60 fps has been achieved.

## Initial attribution

The first attribution used full-rate lighting (`coneLightingScale: 1`), not the
application's configured half-rate/full-resolution-relight combination. These
numbers isolate costs and are not a production baseline comparison.

| Lighting | Stationary frame median | Moving-camera frame median |
| --- | ---: | ---: |
| Shadows and AO | 53.99 ms (50.48 recheck) | 35.21 ms (35.42 recheck) |
| AO disabled | 25.37 ms | 27.01 ms |
| Shadows disabled | 27.94 ms | 14.54 ms |
| Both disabled | 8.77 ms | 9.09 ms |

Each arm used a fully published, non-overflowing 3,812,972-quad mesh. Timings
include CPU encoding and serialized queue completion. A camera-changing
sentinel follows the actual changing poses, so existing moving-quality policy
is represented. The pose path is identical between compared lighting arms.
Pass timestamps are retained, but overlapping Metal render timestamps must not
be summed as independent costs.

## Plan and acceptance

1. Measure the production half-rate/full-resolution-relight configuration.
2. Compare current-frame edge-aware reduced shading at half and quarter rates.
   Keep full-resolution geometry and exact shading fallback when no compatible
   depth/normal/material receiver exists. Inspect captured images before any
   default change. Do not add temporal reuse or assume scene immobility.
3. Address mesh publication separately: current global revision invalidation
   rebuilds every brick. Local edits need dirty-brick and neighbour dependency
   tracking, localized storage replacement, and publication validation. A fast
   completed mesh does not demonstrate fast live edits.
4. Measure moving-camera, moving-body, and live-voxel-edit cases, including
   median, p95 and update latency. Report dry-renderer measurements separately
   from simulation/water and total browser frame time.

`FLUID_STEADY_BENCH=1` extends `tools/probe-svo-depth3-dawn.ts` with
`tools/benchmark-svo-raster-steady.ts` after bounded mesh completion. Use the
repository GPU lease with Fluid Lab browser tabs closed. It retains the 500 ms
submission safety stop. `FLUID_STEADY_WIDTH` and `FLUID_STEADY_HEIGHT` select
physical output size. Captures are raw padded RGBA16F in `/tmp/fluid-steady-*`.

## Current-frame reconstruction comparison

| Configuration | Stationary median / p95 | Moving-camera median / p95 |
| --- | ---: | ---: |
| Production half-rate/full-resolution relight | 25.52 / 28.20 ms | 22.34 / 23.66 ms |
| Half-rate joint bilateral | 19.24 / 21.97 ms | 17.04 / 18.54 ms |
| Quarter-rate joint bilateral | 18.94 / 20.49 ms | 16.44 / 17.69 ms |

The quarter-rate median alone does not establish 60 fps. Full-resolution raster
geometry is preserved in each arm, but reduced radiance visibly softens some
fine lighting detail. The production reconstruction default has not changed.

## Implemented optimization awaiting GPU comparison

The reconstruction path drew every opaque pixel twice: the first shader gathered
compatible reduced radiance and wrote it; the second gathered the same data to
discard those pixels and shade the remaining exact receivers. The main draw now
returns compatible reconstructed radiance directly and shades exact receivers
otherwise. The redundant first draw is omitted. This is current-frame work;
no visibility or lighting is reused across frames. Setting the diagnostic
`singlePassReconstruction: false` retains the former two-draw reference.
The default full-resolution-relight mode is unaffected.

Naga validates both half-rate and quarter-rate shader variants. The GPU run passed with camera and actual rigid-body pose updates. Half-rate
medians remained around 19.4 ms stationary / 16.6 ms moving; quarter-rate
measured 17.9 / 15.1 ms. These old moving-camera timings used the former quality
policy and are not acceptance results under the current requirement. Matched
captures show small differences also present in the unchanged reference arm
(linear RGB RMSE approximately 0.0033–0.0038), so bit-identical output is not
claimed from separate whole-scene builds.

The benchmark now includes actual dynamic rigid-body pose updates, alone and
with camera movement. The pose-update lanes ran without validation errors. Live voxel-edit validation
and full-quality 60 fps acceptance are still pending. They must not be inferred from the moving-camera results above.

## Mesh/progress follow-up

The depth-3 probe with face/layer-parallel extraction and prefix-preserving growth
completed in 1,790 encoded frames, versus 3,575 previously. It published the same
3,812,972 quads, with one build generation, no overflow and no validation errors.
The longest fenced submission was 313.9 ms, within the unchanged safety guard.
The synthetic Dawn scheduler also passed overflow rollback and prefix-growth
checks. Construction is still tied to displayed frames; these results do not
establish fast continuous authored-geometry changes.

Shared owner-defined progress UI now covers startup, background resources and
mesh construction; see `docs/work-progress-ui.md`.

Camera-quality removal was checked on generated full/half/quarter-rate shaders
and the dedicated fan-out worker: none reads `uniforms.viewport.w`, and all pass
Naga. Legacy budget normalization and all preset aliases were checked; 18 CPU
renderer/stage checks and the pixel-trace probe suite passed. The final canonical
Dawn gate could not start within its one-minute wait because other tasks kept
the repository GPU lease occupied. No full gate pass is claimed.
