# Sparse-fluid ocean migration

Status: brick-sparse surface and velocity-advection execution landed, compact
pressure-row storage is live, and inspection/debug arenas are lazy. Bulk field
textures and the live pressure owner map remain dense. This document deliberately
distinguishes the measured cutovers from the complete scene-scale sparse-fluid roadmap.

## What changed

The ocean seiche scene is now twice the original width:

| | Previous | Current |
| --- | ---: | ---: |
| Container | 4.8 × 2.4 × 1.6 m | 9.6 × 2.4 × 1.6 m |
| Finest grid | 192 × 96 × 64 | 384 × 96 × 64 |
| 8³ brick lattice | 24 × 12 × 8 | 48 × 12 × 8 |

The coarse level-set path now consumes the GPU-authored resident-brick stream.
Advection, MacCormack correction, seed construction, bounded jump flooding,
redistance finalization, volume reduction/correction, debris culling, and the
level-set commit copy all use the same indirect brick dispatch. No active count
is read back to the CPU.

Velocity predictor, reverse, and MacCormack correction now consume an independent
wet-volume brick stream. The scheduler retains deep liquid, refreshes with the
current `dt` before every CFL substep, scans complete shared faces for downstream
activation, and clears all four retired velocity ping-pongs before reuse. Atlas
sampling and sparse target dispatch have independent controls, so reverse transport
can read the predicted dense transport field while still writing only active cells.

The compact pressure solver now allocates row-indexed pressure, header, and matrix
arenas rather than finest-cell-sized arrays. At 288×96×64 this reduces those arenas
from about 148.5 MiB to 74.1 MiB; the final 384-wide scene has a 406,272-row capacity
and used 213,784 rows in the measured sparse run without overflow.

Residency finalization writes a dedicated 64-cells-per-workgroup, 2D-tiled
indirect dispatch at byte offset 48. Surface kernels map each workgroup and local
lane back to an 8³ resident brick. The dense A/B route remains available through
`FLUID_BRICK_SPARSE_SURFACE=0` and through the Octree fine-settings control.

Other scaling fixes made while widening the scene:

- Raw voxel, brick materialization, and fluid-overlay kernels linearize 2D
  workgroup IDs, avoiding WebGPU's 65,535 workgroups-per-dimension ceiling.
- Sparse volume correction retains its running total and clears only per-frame
  counters; it no longer performs a dense reduction or full 3D texture copy.
- Jump-flood sampling rejects non-resident seeds, preventing stale atlas data
  from crossing a retired brick boundary.
- The smoke gate measures the propagated disturbance in the far half of the
  tank. Tracking only the global tallest crest was incorrect once reflected
  waves kept a taller crest near the source wall.
- Benchmark smoke runs use the default dry environment. The research-station
  background remains the UI preset, but its legacy inspection records add
  hundreds of megabytes unrelated to solver scaling and would contaminate the
  solver A/B.
- Smooth rendering no longer asks the solver for expanded raw voxel/brick records.
  Their producer arenas and renderer instance buffers allocate only when an
  inspection mode is selected; the smoke harness likewise avoids the lazy getter
  unless `FLUID_SPARSE_STATS=1`.
- A settings-write overlap that made pre-activation overwrite
  `includeLiquidInterior` was fixed. Surface residency is narrow again, while the
  independent bulk scheduler deliberately retains deep liquid.

## Before/after result

The final controlled comparison uses the same 384 × 96 × 64 ocean and build,
20 steps (0.1 s), and three interleaved repetitions. “Before” disables brick-sparse surface
and velocity targets; “after” enables both. Sparse transport preparation and
owner-page mirroring remain off in both modes.

| Median GPU stage | Dense dispatch | Brick-sparse | Speedup |
| --- | ---: | ---: | ---: |
| Surface update | 7.143 ms | 2.753 ms | **2.60×** |
| Velocity advection | 17.302 ms | 15.532 ms | **1.11×** |
| Whole step | 43.057 ms | 38.863 ms | **1.11×** |

The surface stage is 61.5% faster, velocity advection is 10.2% faster, and total
GPU time is 9.7% lower. The sparse path lowers the level-set/VOF mismatch fraction
from 0.753 to 0.077. That more faithful interface retains 213,844 active pressure
rows instead of 116,184; median pressure therefore rises from 5.702 to 8.323 ms
and gives back 2.62 ms of the surface/advection win.

Reported solver allocation for the final smooth-path scene is 1,383,284,388 bytes
(1,319.2 MiB), down 23.5% from the previous 384-wide stress trial's 1,724.8 MiB.
This includes the compact pressure arenas and excludes lazy raw-inspection data.

The machine-readable samples and medians are in
`benchmarks/results/sparse-fluid-ocean-surface-2026-07-19.json`.

## Acceptance and re-evaluation

The 9.6 m ocean passes a 400-step/2.0 s Dawn run under the width-calibrated
propagation gate:

- the wave produces a 0.407-cell disturbance in the far half of the tank
  (gate: at least 0.375 cell);
- liquid remains one connected component;
- velocity contains no non-finite values; and
- WebGPU reports no validation errors and pressure capacity does not overflow.

The former 1.5 s gate was too short after widening the physical tank. The final
horizon scales by 9.6/7.2, while the fixed raised-slab volume's amplitude threshold
scales inversely by 7.2/9.6. The corrected sparse-only parameter path was rerun for
the full 400 steps and clears that 0.375-cell bar.

Sparse `buildTransport` preparation was also implemented and measured, but it is
0.42% slower overall on this full-footprint ocean and remains default-off. This is
an intentional re-evaluation: 75–83% of bulk bricks are wet, so an extra indirect
pass cannot amortize itself here. It remains available for localized-water A/Bs.

Post-projection velocity extrapolation now has the same independent cell64 A/B.
Its seed, propagation sweeps, and odd-sweep copyback run over the bulk brick list,
with retired velocity pages explicitly cleared. After correcting the sparse-only
parameter binding, the widened-ocean stage falls from 1.507 ms to 0.786 ms
(1.92×), but the three-run whole-step median rises from 34.931 to 36.110 ms. The
physical telemetry is identical, so `FLUID_BRICK_SPARSE_EXTRAPOLATION=1` remains
opt-in rather than trading a local timing win for a 3.4% total regression.

Sparse column occupancy and conservative flux-scale preparation are implemented
behind `FLUID_BRICK_SPARSE_OCCUPANCY_FLUX=1`. A GPU-only zero-worklist fallback
preserves the exact first-frame dense height texture. The exact Dawn parity test
passes, but the interleaved ocean A/B raises median whole-step time from 36.635 to
37.618 ms (2.7%), so this slice also remains default-off.

## Reproduction

Point `WEBGPU_NODE_MODULE` at the local Dawn `webgpu/index.js`, then run the
same scene with the sparse switch off and on:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_SCENE=ocean-seiche FLUID_METHOD=octree FLUID_TARGET_S=0.1 \
FLUID_CPU_ORACLE=0 FLUID_FIELD_STATS=1 FLUID_BRICK_SPARSE_SURFACE=0 \
FLUID_BRICK_SPARSE_ADVECTION=0 FLUID_BRICK_SPARSE_TRANSPORT=0 \
npm run test:webgpu

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_SCENE=ocean-seiche FLUID_METHOD=octree FLUID_TARGET_S=0.1 \
FLUID_CPU_ORACLE=0 FLUID_FIELD_STATS=1 FLUID_BRICK_SPARSE_SURFACE=1 \
FLUID_BRICK_SPARSE_ADVECTION=1 FLUID_BRICK_SPARSE_TRANSPORT=0 \
npm run test:webgpu
```

Long-horizon propagation gate:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_SCENE=ocean-seiche FLUID_METHOD=octree FLUID_TARGET_S=2 \
FLUID_CHECKPOINT_EVERY_S=0.5 FLUID_CPU_ORACLE=0 FLUID_FIELD_STATS=1 \
FLUID_SPARSE_STATS=0 FLUID_BRICK_SPARSE_SURFACE=1 \
FLUID_BRICK_SPARSE_ADVECTION=1 npm run test:webgpu
```

## Remaining roadmap boundary

This change removes the coarse surface path's O(logical volume) dispatch cost,
but it does **not** yet make the complete simulator scene-scale sparse:

- velocity, occupancy/flux, and transport ping-pongs are dense 3D textures;
- the octree owner map remains dense, although compact pressure iterates no longer are;
- the brick atlas is still a validated mirror rather than the sole authority;
- occupancy, transport preparation, extrapolation, and bulk classification still
  dispatch over the logical box in the production ocean path;
- addressing is a dense brick lattice bounded by the tank, not the planned
  two-level scene-scale brick tree; and
- the calibrated 2.0 s exact-field audit shows that the persistent sparse volume diagnostic
  needs further reconciliation with the fine surface layer before it can replace
  the exact readback as a long-run conservation metric.

Consequently, this is a verified performance milestone—not a claim that the
entire dense-to-sparse roadmap is finished. A packed 32-bit owner-page substrate
now passes CPU and Dawn lifecycle tests and halves full-capacity owner storage,
and its isolated CPU/WGSL lookup ABI decodes leaf-1 through leaf-32 owners across
page seams with a bounded canonical-air fallback. The live owner mirror remains
dense because its GPU-worklist mirror integration triggered a Dawn/Metal native
crash and
was removed rather than shipping a dangerous opt-in. The next safe completion
slice is to resolve that worklist issue, flip owner reads page-by-page, finish
the remaining occupancy/extrapolation cutovers, and finally replace the dense
logical brick lattice with the two-level scene-scale address tree.

The last item now has an isolated substrate in `lib/scene-scale-fluid-bricks.ts`:
a signed, two-level root-to-block-to-brick address space with bounded local
overflow, deterministic missing-air fallback, brick-relative cell coordinates,
and CPU/Dawn lifecycle parity. Its allocation depends only on configured root
blocks and resident brick slots, not world span. It is deliberately not wired
into the solver until the remaining dense consumers can move together.
