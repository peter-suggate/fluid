# Octree solve-tail cutover

The production command graph encodes the paper-backed upper outer-iteration
envelope. GPU residual convergence remains authoritative and zeroes the
indirect tail. No previous-step iteration count, timing result, scene heuristic,
or CPU readback can change a later pressure solve.

## Selected policy

- Encoded outer PCG envelope: 10 iterations, the upper end of the paper's
  reported satisfactory 6–10 iteration range.
- Validation-only hard ceiling: 16 iterations. No command bodies are emitted
  for the difference between the encoded budget and this ceiling.
- Residual acceptance: the established f32 floor of `1e-4`.
- Matching Section 4.3 shell depth: `k = 8` before and after the L1 correction,
  following the paper's stated approximately-eight-sweep convergence setting.
- Scene inputs still produce diagnostic complexity telemetry, but do not
  predict or cap numerical convergence.

## CPU numerical and command sweep

`tests/octree-solve-tail-policy.test.ts` evaluates compact SPD fixtures for a
mini dam break, a quiescent tank, and an elongated river-like domain. The
production setting is `k = 8`, matching Section 4.3's report that approximately
eight boundary iterations were necessary for satisfactory convergence. The
shallower fixture results are retained only as command-cost comparisons; they
do not override the paper-backed production setting.

The exact five-level command-accounting mirror gives:

| Fixture | Outer budget | k=2 | k=4 | k=6 | k=8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quiescent tank | 10 | 607 | 695 | 783 | 871 |
| Minimal 16³ dam break | 10 | 607 | 695 | 783 | 871 |
| UI 24×18×16 dam break | 10 | 607 | 695 | 783 | 871 |
| River-like | 10 | 607 | 695 | 783 | 871 |
| Maximum encoded envelope | 10 | 607 | 695 | 783 | 871 |

Counts include the fixed solver head, one initial preconditioner application,
the merged one-dispatch compact-band operator, the residual and direct-curvature
reductions, and every encoded outer body.
They are command-graph counts, not estimates of GPU-executed work after the
convergence gate.

## Required projection-owner wiring

`WebGPUOctreeProjection` must construct `planOctreeSolveTail` once, after its
finest dimensions and maximum leaf size are known, and retain that immutable
result. The construction call maps directly as follows:

- `encodedOuterIterations` → `WebGPUOctreePipelinedMGPCG.maximumIterations`;
- `boundarySmoothingIterations` →
  `WebGPUOctreeSection43HybridPreconditioner.boundarySmoothingIterations`;
- `relativeTolerance` → the MGPCG relative tolerance;
- `encodedOuterIterations` → `info.pressureIterationBudget`;
- `hardOuterIterationCeiling` → `info.pressureIterationHardBudget`.

The `powerBoundarySmoothingIterations` construction option must then be
deleted. Retaining an override would create a second production policy and
invalidate the command-count guarantee. Work-accounting plans must continue
to use the MGPCG plan's encoded iteration count; the hard ceiling is metadata
only.
