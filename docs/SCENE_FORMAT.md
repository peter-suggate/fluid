# Shared Scene and Run Format

The source of truth is versioned canonical JSON. All vectors use right-handed
world coordinates with `+Y` upward. Numbers are IEEE-754 JSON numbers; physical
fields carry units in their names. Unknown fields are rejected in validation
mode and preserved-but-ignored only by an explicit migration tool.

## Canonical scene shape

```json
{
  "schemaVersion": "1.0.0",
  "sceneId": "static-water-001",
  "systems": {"fluid": true},
  "randomSeed": 1,
  "duration_s": 2.0,
  "container": {
    "width_m": 1.0,
    "height_m": 1.0,
    "depth_m": 1.0,
    "fillFraction": 0.5,
    "top": "open",
    "fluidWallMode": "free-slip"
  },
  "voxelDomain": {
    "finestCellSize_m": 0.025,
    "brickSize_cells": 8
  },
  "fluid": {
    "density_kg_m3": 998.2,
    "dynamicViscosity_Pa_s": 0.001002,
    "surfaceTension_N_m": 0.072,
    "gravity_m_s2": {"x": 0.0, "y": -9.80665, "z": 0.0},
    "initialCondition": "tank-fill"
  },
  "nominalResolution": {"length_m": 0.025},
  "rigidBodies": [],
  "numerics": {
    "fixedDt_s": 0.001,
    "maxDt_s": 0.004,
    "pressureRelativeTolerance": 1e-8,
    "pressureMaxIterations": 1000
  }
}
```

`voxelDomain` is required; loaders do not synthesize it from
`nominalResolution`. Its finest spacing is the sole GPU spatial-resolution
authority and its brick edge is shared by the sparse scene and SVO renderer.
Renderer-only scenes may select 4³ or 8³-cell terminal bricks. Fluid-enabled
scenes currently require 8³ bricks because their owner-page lifecycle is
8³-native; the loader fails fast instead of silently degrading a 4³ request.
`nominalResolution` remains only the CPU reference-oracle resolution.
`systems.fluid` defaults to enabled when omitted; setting it to `false`
requests a static render world without fluid authority, coupling, or water
presentation.

## Terrain heightfield (optional)

A scene may carry a `terrain` block describing solid ground inside the
container, following the tall-cell paper's per-column terrain height
H_{i,k}. The block is a compact analytic spec — a base ground level plus up
to eight smooth elliptical features — evaluated identically by every solver,
the renderer and the rigid-body contact solver via `lib/terrain.ts`:

```json
{
  "terrain": {
    "baseHeight_m": 0.38,
    "features": [
      {"kind": "basin", "center_m": {"x": -0.35, "z": -0.12},
       "radius_m": {"x": 0.78, "z": 0.6}, "amount_m": 0.34,
       "rotation_rad": 0.35, "flat": 0.5},
      {"kind": "mound", "center_m": {"x": -1.1, "z": 0.72},
       "radius_m": {"x": 0.75, "z": 0.6}, "amount_m": 0.16}
    ]
  }
}
```

`baseHeight_m` is the ground level above the container floor. Each feature is
an elliptical footprint (`radius_m` semi-axes, optional `rotation_rad`) with a
flat inner plateau (`flat`, fraction of the radius, default 0.45) and a smooth
C1 falloff to zero at the footprint edge. Mounds add height; basins carve it
away, overlapping basins merging through a p-norm smooth union (p = 8) so a
composite pool reads as one organic hollow. A basin cannot be deeper than
`baseHeight_m`; the ground never descends below the container floor.

Solvers treat cells below the local ground height as static solid (zero
velocity, Neumann pressure boundary, closed faces), initial fluid is seeded
only above the ground, and volume accounting excludes the solid region. The
CPU reference oracle does not model terrain; GPU methods are authoritative
for terrain scenes. Rigid bodies collide with the heightfield through a local
tangent-plane contact. Absent `terrain`, behaviour is exactly the historical
flat floor at y = 0.

Each rigid body stores a stable ID, primitive type and dimensions in metres,
mass and density (with exactly one declared authoritative), position, unit
quaternion orientation `[w,x,y,z]`, linear/angular velocities, restitution, and
friction. If mass is authoritative, density is derived from analytic primitive
volume; inconsistent redundant fields are rejected.

Solver-specific numerical settings cannot override container, gravity, fluid
properties, bodies, seed, or initial fluid regions. The legacy `duration_s`
field remains in schema 1.0.0 for compatibility and fixed-duration benchmark
metadata, but interactive simulations run continuously until paused or reset.
Every run records the exact scene content hash before the solver initializes.

## Run record

A run manifest adds:

```text
runSchemaVersion, runId, UTC start time
canonical scene SHA-256
application/core/solver versions and git revision
OS, hardware, GPU, compiler, build flags
CPU/GPU precision and fast-math mode
accepted and rejected dt history with limiting condition
recorded user actions indexed by simulation time
solver convergence histories
metric and snapshot artifact paths plus hashes
final pass/fail contract version
```

The deterministic action log contains commands such as pause, step, drop, and
parameter changes; it never stores wall-clock-dependent UI gestures as physics
inputs. Large checkpoints use an endian-labelled binary format with array type,
shape, byte count, and SHA-256 entries in the JSON manifest.
