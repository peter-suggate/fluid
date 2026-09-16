# Half-pool: why 3D loses more resolved motion than 2D

Investigation: 2026-09-16, `coarse-first-pool-impact-half`, current working tree.
No production physics changes were made for this investigation.

## Main finding: velocity transport introduces an extra averaging filter

The 3D path publishes cell-centred velocities reconstructed from face velocities,
then trilinearly samples that cache at each face's departure point:

- `webgpu-sparse-cm12-resident.wgsl.ts`: `publishSparseCM12FaceVelocitySupport`,
  `sampleFaceVelocitySupportAtSpans`, `prepareTransportFaceRow`.
- The source explains that the staggered sampler was retired to reduce cost.

The 2D path (`numerics.rs`, `prepare_faces_impl` / `sample_source`) samples the
staggered face field directly, with limited bicubic interpolation on certified
uniform support and a linear staggered fallback. Both methods use collocated
velocity when tracing characteristics; the significant difference here is the
velocity value sampled **after** tracing.

On a uniform grid, the 3D face-to-cell-to-face round trip smooths velocity even
with zero travel. A divergence-free Fourier vortex with wavelength L and cell
width h retains amplitude cos²(pi h/L), hence kinetic energy cos⁴(pi h/L).
The isolated Dawn probe extracts and executes the production sampler and confirms:

| Wavelength / cell width | Energy retained after one zero-travel round trip |
| --- | ---: |
| 4 | 25.00% |
| 8 | 72.86% |
| 16 | 92.53% |
| 32 | 98.09% |

These are individual Fourier-mode measurements, **not total scene losses per
frame**. Coarsening makes a fixed physical wavelength shorter in cell units and
therefore increases attenuation. Smaller timesteps do not eliminate this filter;
more applications over the same physical time can compound it.

## Causal control in the requested 2D scene

Temporary copies of the Rust workspace changed only characteristic value sampling
inside `prepare_faces_impl`. Production Rust was untouched. Each run used 60
frames at dt=1/30 s, pressure cap 128, relative tolerance 1e-6, and level-set-volume
transport. Adaptivity and sharpening remained enabled.

The metric is mean specific kinetic energy over 1–2 s, integrated by the
trapezoidal rule over every frame. Native compact velocity moments were converted
to physical units using K_specific = K_diagnostic * 0.05² / liquidMeasureFineCells2.

| 2D characteristic sampler | Mean K/m (m²/s²) | Relative to production |
| --- | ---: | ---: |
| Production staggered limited cubic | 0.2722694 | 100% |
| Linear staggered | 0.2678480 | 98.38% |
| Collocated linear (`sample_support`) | 0.2043730 | 75.06% |

This establishes a material effect of the sampling choice in this scene. It is
not an exact attribution of the full 2D–3D gap: nonlinear trajectories and adaptive
grids diverge after changing the sampler, and the 2D collocated implementation is
an analogue of the 3D cache rather than identical code. All three native runs
conserved mass to roughly 1e-8 relative drift or better.

Increasing the native pressure cap from 128 to 256 changed specific kinetic energy
by at most 1.74e-6 m²/s² across the run. Every captured poststep pressure solve in
the 3D sphere run reported convergence with residual below 1e-6. Simple pressure
iteration exhaustion is therefore not supported as the main explanation; this
does not rule out differences in the projection operators themselves.

## The preset also changes the physical experiment

The tank is 3.2 × 2.4 × 3.2 m with a 0.8 m-deep pool and a radius-0.5 m drop.
The 2D drop is a disk; the 3D drop is a sphere.

- 2D drop/pool measure ratio: pi * 0.5² / (3.2 * 0.8) ≈ 0.307.
- 3D drop/pool volume ratio: (4/3) pi * 0.5³ / (3.2² * 0.8) ≈ 0.064.

The sphere has about 4.8 times less relative impact mass, and motion can spread in
the depth direction. A smaller 3D splash is not by itself evidence of dissipation.

An extruded-cylinder control used a 0.4 m-deep tank, rebuilt authored voxel walls,
and symmetry depth boundaries. With cell widths frozen, it completed 60 frames
without GPU validation errors and retained much less late resolved motion than
native 2D. It is not a strict matched-grid comparison: initial cell widths are
frozen while support can still grow, and native 2D continues adapting. Less than
0.1% of accepted liquid volume lay outside the dense diagnostic domain at captured
checkpoints. The fully adaptive extrusion stopped at frame 34 with
`MISSING_COMPILED_TOPOLOGY_FACE` in `validateAndAuthorizeShadowTopology`, generation
35, owner 45. Its valid prefix cannot establish a full two-second adaptive result.

Initial extrusion attempts changed tank depth without rebuilding authored walls.
Those runs allowed substantial liquid outside the measurement window and are
explicitly named `*-INVALID-WALLS.json`. They must not be used to infer energy
loss, sharpening effects, or the effect of freezing adaptivity.

## Secondary algorithm difference: volume departure geometry

`resident-volume.wgsl.ts::gvTranslatedDepartureBox` traces the receiver centre
and translates an axis-aligned box without deforming its eight corners.
`levelset_volume.rs::trace_rk2` traces four corners plus the centre and transports
through deforming polygon overlaps. The 3D approximation misses local deformation
inside the receiver, which can affect interface shape and subsequent pressure and
velocity coupling. This is a source-level difference, not a quantitatively
isolated cause in this investigation. Both volume tracing paths use collocated
velocity; the 2D path here does not use streamfunction tracing.

## Interpretation and next implementation

1. First restore direct staggered face sampling for transported velocity in 3D.
   Start with linear interpolation and retain the current collocated cache for
   RK2 tracing. This targets the dominant demonstrated difference without needing
   a high-order sampler immediately. Verify identity at zero travel on uniform
   support and measure cost on mixed-resolution support.
2. Consider bounded cubic interpolation after measuring the linear staggered
   version. In this scene its incremental 2D benefit was much smaller than
   eliminating the collocated round trip.
3. Investigate deforming 3D departure footprints separately, and resolve the
   adaptive extrusion topology failure before claiming dimensional parity.

Do not infer global energy conservation from these kinetic measurements. In the
3D sphere run the approximate mechanical-energy diagnostic actually increased
(41.52 to 45.51 in unit-density physical units). Cell-centred potential-energy
measurement and geometric/pressure feedback need separate accounting. Stage taps
were unavailable for most GPU-grown topologies because host metadata was missing,
so this investigation does not supply a complete stage-by-stage energy budget.
Sharpening's independent contribution has not been established.

## Reproduction and receipts

Run Dawn probes sequentially, without another Dawn process or browser simulation:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-volume-velocity-roundtrip-dawn.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-half-pool-dissipation-dawn.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-half-pool-dissipation-dawn.ts --extruded=1 --freeze=1 --output=artifacts/level-set-volume/half-pool-dissipation-extruded-frozen.json
```

Native control reproduction: copy `rust/Cargo.toml`, `rust/Cargo.lock`, and
`rust/crates` into a temporary workspace, preserving the working-tree sources.
Build `cargo build --release --example verify_world -p fluid-core` there. Supply
the scene document for `coarse-first-pool-impact-half`, with:

```json
{
  "productionOptions": {"dtS": 0.03333333333333333, "timeStep": "paper"},
  "worldOptions": {"pressureIterations": 128, "pressureRelativeTolerance": 0.000001, "transportExperiment": "level-set-volume"},
  "frames": 60,
  "receiptsOnly": true,
  "observeStageMetrics": false,
  "captureFailure": true
}
```

In each copied variant, replace only the `let characteristic = sample_source(...)`
call inside `prepare_faces_impl`: for the linear control use
`sample_source_linear` with the same arguments; for the collocated control use
`sample_support(graph, fields, departure[0], departure[1], span)[row.axis as usize]`
when `use_extension_phase` is true, otherwise retain the original call. Feed raw
output to `tools/verify-level-set-volume-energy-native.ts --input=... --frames=60
--scene=coarse-first-pool-impact-half --output=...`. Its CLI hardcodes pressure cap
256 in report metadata, so correct that metadata to the actual input cap 128.

Local receipts under `artifacts/level-set-volume/`:

- `velocity-roundtrip-dissipation.json`: analytical versus production GPU sampler.
- `half-pool-dissipation-2d{,-128,-linear-staggered,-collocated}.json` and
  corresponding `-raw.json`: native runs.
- `half-pool-dissipation-3d.json`: completed adaptive sphere run.
- `half-pool-dissipation-extruded.json`: adaptive cylinder, halted frame 34.
- `half-pool-dissipation-extruded-frozen.json`: completed cylinder control.

The probes themselves were executed successfully, subject to the explicit adaptive
extrusion failure above. No large production simulation change was made, so the
full Sparse CM12 regression gate was not rerun for this diagnostic-only work.
