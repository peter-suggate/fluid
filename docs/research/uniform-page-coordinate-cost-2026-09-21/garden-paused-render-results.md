# Garden paused rendering investigation

Compared e49add19 and main 5d4d31f2 with the production cached mesh renderer,
hero-garden-hose, depth 0 (production fluid-scene environment refinement), default
lighting, 4 warmups and 20 measured frames on Apple M1 Max / Dawn Metal. No
simulation advances; wait for complete mesh publication before timing. The CPU
scene factory confirms the scene starts empty. The renderer-only harness excludes
water composition, overlays, browser scheduling and app CPU work.

| Render pixels | Branch median | Main median |
|---|---:|---:|
| 1000 × 620 | 8.375 ms | 8.551 ms |
| 2000 × 1240 | 25.332 ms | 25.299 ms |

Every settled capture passed the GPU producer check: raster mesh pixels present,
zero traced-primary pixels, no validation errors. The default mesh-selection
and rendering implementation are unchanged from main. A primary-work diagnostic
explicitly forces tracing; page visualization does not.

At 1000 × 620 the branch pass trace attributes approximately 7.078 ms to deferred
lighting and 0.066 ms to mesh rasterization. At double dimensions the lighting
interval dominates again, but the recorder flags overlapping render intervals
as untrusted, so their individual durations must not be summed as exclusive
costs. The uninstrumented whole-render medians are the comparison authority.

The app multiplies viewport dimensions by device pixel ratio (capped at 2),
then the selected resolution scale. A 1000 × 620 CSS viewport at DPR 2 and scale
1 therefore corresponds to the slower row. Primary and lighting render every
paused frame; geometry is cached, and water extraction is revision-gated.

No live Fluid tab was available in connected Chrome, so this does not establish
the user's exact viewport/settings or rule out overlay/browser/startup costs.
The evidence does not reproduce a branch-specific raster fallback. It does
identify resolution-sensitive deferred lighting as a substantial paused cost.
Next live check: resolved renderer, actual render dimensions, stage timing,
and page-overlay on/off at the same camera. Optimize the dominant lighting
work or retain unchanged paused presentation only after checking those inputs.

The old run-svo-dry-render-smoke harness interprets resolveSvoPrimaryTraversal's
new mesh result as canonical-parametric. It was not used for these measurements.
The existing depth-3 probe also asserts that pending meshes never show partial
geometry; its startup assertion fails on this scene. That assertion was left
unchanged. The new steady-state probe waits for full publication and verifies
raster provenance before measuring.

Reproduce using the repository exclusive launcher (no concurrent browser/Dawn):

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_PROBE_WIDTH=1000 FLUID_PROBE_HEIGHT=620 FLUID_PROBE_LIGHTING=1 node --import tsx tools/run-webgpu-exclusive.ts --import tsx tools/probe-garden-paused-render-dawn.ts
```

Raw captures: garden-paused-render-measurements.json.
