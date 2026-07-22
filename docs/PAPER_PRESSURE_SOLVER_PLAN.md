# Paper-faithful pressure solver plan

This plan replaces the current command-heavy approximation of the pressure
preconditioner with the sparse uniform-grid pyramid described by Aanjaneya et
al. (2017), Sections 4.2-4.3, using the multigrid construction from Setaluri et
al. (2014), Section 5.

## Implementation status (July 2026)

The plan is implemented as an opt-in A/B hierarchy named `paper-pyramid`.
`aggregate-galerkin` remains the rollback default while the new path completes
the broader quality matrix.

- The complete fixed MGPCG schedule is encoded in one WebGPU compute pass. At
  the 128-iteration cap, this reduces compute-pass transitions from 15,357 to
  1; the UI reports dispatches separately from pass transitions.
- The A/B path builds native sparse levels with disjoint active, persistent
  ghost, multigrid-only, and finest-cell masks. Adjacent levels share one
  cell-centred trilinear transfer record set, so prolongation is exactly the
  transpose of restriction. Smoothing excludes persistent ghosts, and the
  hierarchy is forced to a one-degree-of-freedom exact bottom solve.
- A CPU oracle constructs the explicit transfer matrices, verifies `R=P^T`,
  forms Galerkin `RAP`, checks symmetry/SPD, and exercises a complete V-cycle.
- The 0.4 s, 100-step Dawn mini-dam-break A/B run completed on both paths. The
  paper hierarchy encoded 10,730 dispatches versus 18,712 for the rollback
  hierarchy (-42.7%), reduced pressure-solve encoding time from 27.892 ms to
  16.281 ms (-41.6%), and reduced simulation wall time from 11.354 s to
  8.447 s (-25.6%). Both ended at the same maximum speed to 6 decimal places,
  one connected liquid component, no non-finite values, and about 0.108%
  volume drift; their absolute drift differed by about 3.3e-8.
- The paper path also completed the full 2.2 s, 550-step pressure endurance run
  at cap 128. The final solve converged in 9 iterations to a 2.43e-5 relative
  residual, with no pressure rejection or non-finite flag.

The cap reduction is deliberately **not** part of this change: cap 16 still
encountered a 1,444-row generation that did not converge and correctly rolled
back publication. The safe default remains 128. The standard smoke harness also
still reports its pre-existing coarse/fine generation and post-solve divergence
invariants; the no-audit endurance path isolates and passes the pressure solve.

Before promoting `paper-pyramid` to the default, remaining gates are topology-
generation caching (the pyramid currently rebuilds for every pressure solve),
a wider hydrostatic/SPD/scene matrix, and resolution of those upstream smoke
invariants. Coarse stencils are currently rediscretized from captured L1 rows;
unsupported zero-diagonal multigrid-only targets receive fail-closed identity
anchors.

## Why the PCG cap reduction was reverted

The 0.4 s, 100-step Dawn `minimal-power-dam-break` run exposed a rare pressure
generation that did not converge within 16 iterations:

- cap 16 stopped at step 54 (about 0.216 s);
- MGPCG reported `nonConvergence`, `iterations=16`, and 1,440 rows;
- the pressure publication was rejected and the topology transaction rolled
  back;
- cap 128 completed all 100 steps, with sampled solves taking 7, 8, 8, and 9
  iterations at steps 25, 50, 75, and 100.

The default is therefore 128 again. Lower caps remain an Advanced diagnostic,
not a quality or performance preset.

## What already matches the paper

The authoritative outer solver has the correct high-level structure:

1. matrix-free PCG solves the second-order power-diagram operator `L2`;
2. the preconditioner starts from zero;
3. eight damped-Jacobi sweeps of `L2` run in a three-voxel boundary and level-
   transition band;
4. a first-order correction is applied to the residual;
5. eight matching `L2` sweeps complete a symmetric preconditioner;
6. convergence targets a relative residual of `1e-4`.

These properties should be retained.

## Where the current `M1` differs

`WebGPUOctreeFirstOrderVCycle` is a useful SPD prototype, but it is not the
paper's optimized first-order multigrid preconditioner:

| Concern | Current implementation | Paper hierarchy |
| --- | --- | --- |
| Level storage | Hash aggregates derived from compact finest rows | Active, ghost, and multigrid-only cells on native sparse uniform-grid levels |
| Coarse operator | Matrix-free `P^T L1 P`, reapplied from finest rows with atomics | Uniform per-level first-order stencil plus adjacent-level ghost transfers |
| Restriction | Piecewise-constant aggregate map | Copy cells that persist at the next level; distribute finest cells with the standard trilinear stencil |
| Prolongation | Constant parent value | Exact transpose of restriction |
| Smoothing domain | All populated aggregate slots | Only the finest cells represented at each multigrid level |
| Bottom solve | 16 damped-Jacobi iterations | A non-multigrid coarse solver, treated as an exact bottom solve (ICPCG in Setaluri et al.) |
| GPU recording | One compute pass per micro-stage and transfer clears between stages | Streaming level stencils and structured adjacent-level transfers |

For the 24x18x16 fine dam path, one current first-order correction records 94
compute passes. The complete hybrid preconditioner records 113 passes, and one
PCG iteration records 119. A cap of 128 therefore records 15,357 compute-pass
transitions even if the numerical solve converges in eight iterations.

## Replacement architecture

### 1. Build a native sparse multigrid pyramid

Create `WebGPUOctreeSPGridPyramid` alongside the existing implementation.
Rebuild it only when the pressure topology generation changes.

Each level owns compact structure-of-arrays storage for:

- active, ghost, and multigrid-only flags;
- a compact level worklist and indirect dispatch dimensions;
- first-order diagonal and six Cartesian face coefficients;
- right hand side, residual, and pressure ping/pong vectors;
- explicit adjacent-level restriction and prolongation records.

The builder consumes the captured Losasso/ghost-fluid `L1` rows before power
publication replaces the shared row ABI with `L2`. It must not reconstruct
coarse operators through a hash lookup during every preconditioner application.

### 2. Implement the paper's transfer operators exactly

- Cells represented at both adjacent multigrid levels copy their values.
- Finest cells at a level distribute through trilinear restriction.
- Prolongation is stored/applied as the exact transpose of restriction.
- Ghost accumulation and propagation operate only between adjacent sparse-grid
  levels.

This preserves the symmetry required by ordinary PCG. The CPU oracle should
materialize `P`, `R`, and each small Galerkin operator so tests can verify
`R=P^T`, positive energy, and the GPU result on bounded grids.

### 3. Replace the bottom Jacobi approximation

Use an explicit small coarse matrix and a deterministic symmetric bottom
solver. Preferred order:

1. GPU LDLT/Cholesky when the coarse system fits the bounded direct-solve
   capacity;
2. fixed-count IC(0)-PCG from zero for larger coarse systems;
3. fail closed if neither capacity is available.

A fixed linear bottom operation is important: a tolerance-dependent inner
solve would make the preconditioner variable and invalidate ordinary PCG's SPD
assumption.

### 4. Record the solve as a small number of compute passes

WebGPU treats every dispatch as its own ordered usage scope inside a compute
pass. Keep dependent stencil dispatches in one long compute pass and replace
inter-stage `clearBuffer` calls with bounded zero-range compute kernels.
Copies needed to capture `L1` remain outside that pass.

Initial target:

- no more than four compute-pass transitions for a complete pressure solve;
- no bind-group creation inside an iteration;
- no hash-map construction or full-capacity clear inside a preconditioner
  application;
- one dispatch per Jacobi sweep, transfer, or reduction rather than one pass.

This removes the observed pass-transition queue wall even before changing the
outer iteration cap.

### 5. Restore a bounded paper-like iteration schedule only after proof

Keep the safe 128 tail during development. Move to a 16-iteration product cap
only after the replacement hierarchy demonstrates:

- no cap exhaustion or rejected pressure publication in the 2.2 s dam break;
- maximum observed PCG iterations at or below 10 in the paper validation
  scenes, with a small documented allowance for boundary-heavy cases;
- stable convergence across topology generations, not merely at final sampled
  frames.

If rare generations still exceed the paper range, retain the safe tail and
add an explicit retry transaction rather than silently publishing an
under-converged pressure field.

## UI and diagnostics

Expose only controls that have an intuitive numerical meaning:

- **Pressure solver:** `Paper sparse pyramid` / `Current aggregate A/B`;
- **Boundary smoothing:** default 8, Advanced, with symmetry locking pre/post
  values together;
- **Experimental PCG cap:** default 128 until the endurance gates pass;
- live readouts for `executed / cap`, relative residual, pyramid levels,
  coarsest degrees of freedom, dispatch count, and compute-pass transitions;
- a visible `retry/rejected` state when a topology generation cannot converge.

Do not expose restriction, prolongation, or asymmetric smoothing controls:
they are correctness invariants rather than artistic tuning parameters.

## Acceptance gates

### Numerical

- CPU/GPU small-matrix parity for `L1`, every transfer, one V-cycle, and the
  complete hybrid preconditioner;
- randomized symmetry and positive-energy tests;
- hydrostatic two-level and large-offset Dawn scenes;
- 0.4 s and 2.2 s mini dam breaks with no pressure rejection;
- connected liquid component, volume drift, speed/CFL, divergence, dam spread,
  and interface bounds compared against the safe 128-cap baseline.

### Performance

- command-pass transitions reduced from about 15,357 to at most four;
- pressure encode time below 5 ms on the current 24x18x16 Dawn reference;
- timestamped shader work remains proportional to executed iterations;
- no per-step CPU readback or queue fence is introduced.

### Rollout

1. Add the pyramid and CPU oracle behind an A/B solver mode.
2. Prove transfers, symmetry, and the bottom solve.
3. Integrate it into the existing Section 4.3 hybrid.
4. Consolidate the pressure dispatches into a small number of compute passes.
5. Run the complete Dawn matrix and long dam endurance tests.
6. Make the paper pyramid authoritative only after all numerical and
   performance gates pass; retain the current solver as a diagnostic rollback.
