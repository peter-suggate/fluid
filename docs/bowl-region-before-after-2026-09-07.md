# Stationary bowl: live enforcement edit in Dawn

Executed successfully using Dawn/Metal, without opening the UI. This is a diagnostic experiment, not a passing surface-correctness regression or a physics fix.

## Experiment

`tools/probe-bowl-region-edit-dawn.ts` uses the stationary-bowl scene, 48×32×40 fine cells at 0.05 m, adaptive-mass coarse-first, initially forced to width 4. Gravity, velocity, viscosity, surface tension, diffusion and sharpening are zero/off. After one warmup step, apply the region edit and capture both immediately and after each of four further 1/60 s steps. Topology remains live.

Four sequential arms:

- Control: no edit.
- Additive: add a width-2 right-half region over the existing whole-domain width-4 region.
- Split: replace the whole-domain region with nonoverlapping width-4 left and width-2 right regions, boundary x=0.
- Flat split: same split, but set both initial height-field curvature coefficients to zero, preserving the bowl's other settings and base height.

The right region spans the full tank height and z extent. It isolates one resolution boundary; it is not an exact recreation of the screenshot's box. All three bowl arms had identical before density and publication arrays.

## Measurements

Read density separately from the actual packed sparse presentation scalar. Published heights below are vertical zero crossings of those scalar samples, not rendered pixels. Interior excludes eight fine cells from each horizontal wall.

| Arm | Actual topology after edit | Maximum density change | Maximum column-volume height change | Maximum published-height change | Interior maximum / RMS |
|---|---|---:|---:|---:|---:|
| No edit | all width 4 | 0 | 0 mm | 0 mm | 0 / 0 mm |
| Additive overlap | all width 4 | 0 | 0 mm | 0 mm | 0 / 0 mm |
| Bowl split | left 4, right 2 | 0 | 0 mm | 6.06167 mm | 3.66300 / 1.14667 mm |
| Flat split | left 4, right 2 | 0 | 0 mm | 0.76628 mm | 0.76628 / 0.53476 mm |

The split commits on the first post-edit step: 60 of 120 active bricks change to width 2. Immediately after `applySceneUniforms`, topology and publication remain unchanged. The surface change appears with the committed refinement and persists through the remaining captures. Every capture has zero velocity and no topology/frame fault. Bowl volume remains 4.357438404083254 m³; flat volume remains 4.15199998855577 m³. Density arrays are identical pointwise before/after, not merely equal in total mass; restriction back to original width-4 parent means also gives zero change.

The additive arm is important: overlapping minimum floors preserve width 4 here. A visible region box alone is not evidence that actual resolution changed.

## Interpretation and limits

This isolates a resolution-dependent **surface publication** effect with unchanged represented density. Density transfer, mass loss, fluid motion and time evolution cannot explain the measured before/after difference in this experiment. Lighting, normals and browser rendering are not needed to produce it: the difference already exists in the published scalar and its zero crossings.

The bowl develops alternating height changes tied to the grid in the refined half. The relevant production path constructs compact column scalar nodes and interpolates them using the accepted cell scale (`presentationCoarseColumnPhi`, `presentationInterpolatedVolumePhi`). This is a focused candidate for the next investigation, rather than proof that one individual expression is solely responsible.

The flat control also shifts slightly. Therefore a curvature-only explanation is too strong. This control is **not** `hydrostatic-power-large-offset`; its initial density profile, method settings and live-transfer history can differ from that scene. The user's visibly flat hydrostatic observation remains compatible with these measurements but is not explained by this run.

This does not rule out additional physical errors in the original moving-water scene. A correct fix should make a pure representation change preserve the represented surface without adding grid-frequency structure, and also pass flat, tilted and curved controls. Merely preserving total mass or hiding a seam through lighting is insufficient.

## Reproduce and inspect

Run Dawn exclusively, never concurrently with a browser or another Dawn process:

```sh
node --import tsx tools/probe-bowl-region-edit-dawn.ts
python tools/analyze-bowl-region-edit.py
```

The analysis requires NumPy and Matplotlib. Artifacts are in `artifacts/bowl-region-edit/`: per-arm before/after scenes, density/velocity/pressure/scalar/height/column readbacks, activity/frame/statistics, traces, compiled shader hashes, `summary.json` and `before-after.png`. Production code was not changed for this experiment. Shader hashes record the source compiled in this run; the shared working tree contains other ongoing changes.
