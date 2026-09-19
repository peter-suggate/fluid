# Tall-air A/B fixture and tools (Uniform Geometric)

Shared, reusable measurement rig for "what does empty air above the liquid
cost?". Nothing here changes `lib/`. Another agent may use it instead of
building its own fixture.

**Findings: `ab-report.md`.** Raw captures are the JSON beside it
(`ab-1x-4x.json`, `ab-1x-2x-8x.json`, `ab-dense-control.json`,
`cpu-census.json`, `pressure-plan-census.json`).

**The solve window built on top of these findings: `solve-window-report.md`**
(captures `solve-window-direct.json`, `solve-window-indirect.json`,
`solve-window-forced-violation.json`, `solve-window-lag-census.json`). That work
changed `lib/`, and added a `w` arm suffix to the probe (`4w` = the same arm
with `activeRegion: "on"`), a direct/indirect dispatch counter, and the window
fields in the per-frame series.

## The fixture

`tall-air-scene.mts` — `tallAirScene(multiple)` returns one scene where the ONLY
thing that varies is the container height:

- base scene: `createMinimalPowerDamBreak64Scene()` (0.8 m cube, 0.0125 m cells)
- `container.height_m = 0.8 * multiple`, `top: "closed"`
- reservoir pinned in ABSOLUTE metres, `fluid.initialDamBreakDimensions_m =
  { x: 0.4, y: 0.2, z: 0.8 }`, so every arm gets the identical 32x16x64-cell
  column anchored in the same floor corner. (Without this the fill-fraction path
  `damBreakFractions`, `lib/core/initial-fluid.ts:17`, would make the taller
  arm's dam proportionally taller and the arms would not be the same problem.)
- the tank shell is re-authored after the height moves, because
  `solidVoxelShellForScene` bakes voxel indices against the final lattice.

Lattices: `1x` 64x64x64, `2x` 64x128x64, `4x` 64x256x64, `8x` 64x512x64.
Liquid is 35 904 cells in every arm. Measured max |v| over the run is 3.53 m/s
in every arm, so the ballistic ceiling is v^2/2g = 0.64 m < the 0.80 m lid of
even the shortest arm: no arm's liquid can reach its top, so all arms solve the
same physical problem.

## Commands

CPU only, no GPU, no lease needed — lattices, pressure/extension hierarchy
plans, bytes, initial liquid census for 1x/2x/4x/8x:

```sh
node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/cpu-census.mts
```

GPU, under the repository WebGPU lease, one Dawn process at a time. All arms are
built up front and advanced in LOCKSTEP (one frame each per round, rotating
order), because this lane is bimodal:

```sh
FLUID_TALL_MULTIPLES=1,4 FLUID_TALL_FRAMES=70 \
FLUID_TALL_OUT=/tmp/tall-air.json \
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/probe-tall-air-dawn.mts
```

- `FLUID_TALL_MULTIPLES` — comma list of arm specs. A bare number is the shipped
  Uniform Geometric default. A `d` suffix (`4d`) is the SAME height with every
  shipped gate off (`twoLevelVelocity: off`, `twoLevelExtension: dense`,
  `twoLevelAdvection: dense`, `transportWorkMap: dense`, `sharpeningWorkMap:
  off`) — the dense control the method retains. Example: `1,1d,4,4d`.
- `FLUID_TALL_FRAMES` (default 70), `FLUID_TALL_WARMUP` (default 4),
  `FLUID_TALL_OUT` (JSON path; stdout also carries the JSON, stderr the table).

What it reports per arm: per-seam GPU ms from the solver's own hardware
timestamps (spliced into real passes, so an instrumented advance encodes the
same passes as a plain one) with the compute-pass count each seam owns; whole
advance GPU ms; host encode ms; the pressure plan walked on the host (passes by
entry point, by level, by schedule stage; level dimensions; cycle boundaries);
extension hierarchy level and pass counts; live/fine/shell/transport/sharpen
tile counts; cycles encoded and executed; `volumeCellSum`, max speed and
residual per frame; allocated bytes.

Add a new arm by adding a multiple. Memory is ~115 MiB x multiple, so keep the
total under about 1.5 GB per process.

CPU only — the CM11a pressure plan per level and per schedule stage, including
the prefix a lagged budget encodes. `buildPlan()` decides every dispatch extent
from the level dimensions and the fixed schedule alone, so the plan's shape
needs no device; `--validate` checks the mirror against plan censuses the Dawn
probe read out of the live solver, and it currently matches every level, pass
and stage count exactly:

```sh
node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/pressure-plan-census.mts \
  --validate=docs/research/uniform-geometric-tall-air-2026-09-19/ab-1x-4x.json
```

Use this instead of a Dawn run whenever the question is "how many passes /
workgroups, at which level?". Re-run `--validate` after any change to
`buildPlan` or the hierarchy planner, or the mirror silently drifts.

## Caveats that apply to anyone reusing this

- The Dawn timestamp tick is 65.5 us; every seam median is a multiple of
  0.0655 ms and single-tick differences are at the resolution floor.
- Only about half of the advances produce a NEW trace sample (the solver gates
  hardware tracing on `UNIFORM_PHYSICS_TRACE_CADENCE_MS`), so seam medians are
  medians over ~32-42 samples, not per-frame pairs. Wall and CPU encode are
  per-frame and fully paired.
- `info.physicsTrace` persists between samples; dedup by `trace.sampleId`
  (the probe already does).
