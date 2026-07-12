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
  "randomSeed": 1,
  "duration_s": 2.0,
  "container": {
    "size_m": [1.0, 1.0, 1.0],
    "origin_m": [0.0, 0.0, 0.0],
    "top": "open",
    "fluidWallMode": "free-slip"
  },
  "fluid": {
    "density_kg_m3": 998.2,
    "dynamicViscosity_Pa_s": 0.001002,
    "gravity_m_s2": [0.0, -9.80665, 0.0],
    "initialRegions": [
      {"shape": "box", "min_m": [0.0, 0.0, 0.0], "max_m": [1.0, 0.5, 1.0]}
    ]
  },
  "nominalResolution": {"length_m": 0.025},
  "rigidBodies": [],
  "numerics": {
    "clock": {"mode": "fixed", "fixedDt_s": 0.001, "maxDt_s": 0.01},
    "eulerian": {
      "cellSize_m": 0.025,
      "pressureRelativeTolerance": 1e-8,
      "pressureMaxIterations": 1000,
      "advection": "semi-lagrangian-rk2",
      "interface": "volume-of-fluid"
    }
  }
}
```

Each rigid body stores a stable ID, primitive type and dimensions in metres,
mass and density (with exactly one declared authoritative), position, unit
quaternion orientation `[w,x,y,z]`, linear/angular velocities, restitution, and
friction. If mass is authoritative, density is derived from analytic primitive
volume; inconsistent redundant fields are rejected.

Solver-specific numerical settings cannot override container, gravity, fluid
properties, bodies, duration, seed, or initial fluid regions. Every run records
the exact scene content hash before the solver initializes.

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
