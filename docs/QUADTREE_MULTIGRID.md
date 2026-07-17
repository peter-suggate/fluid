# Quadtree geometric multigrid pressure solve

## Status

An experimental `mg` preconditioner is implemented for the adaptive tall-cell
PCG solve. It is selectable as **Geometric multigrid**; `poly` remains the
default until the GPU-time gate below passes.

The implementation is numerically successful but not yet the fastest runtime
path. On `dam-break-ui`, balanced quality, at 0.1 simulated seconds:

| Preconditioner | Actual PCG iterations | Encoded iterations | Pressure setup | Iteration phase | Projection |
|---|---:|---:|---:|---:|---:|
| polynomial degree 2 | 75 | 86 | 0.59 ms | 12.39 ms | 3.01 ms |
| geometric MG | 13 | 16 | 0.92 ms | 17.89 ms | 2.42 ms |

Both runs reached the same `1e-4` relative-residual target without WebGPU
validation errors. The exact timings vary with the live topology, but the
conclusion is stable: MG removes the conditioning problem (roughly 5.8x fewer
iterations), while its current V-cycle still does too much synchronized work
per iteration. It is therefore an experimental foundation, not a claimed
speedup.

The full 0.2 s dam-break regression also passes: 11 actual / 16 encoded
iterations on the final step, 16.32 ms for the complete timed pressure stage,
no validation errors, finite fields, and represented-volume drift of -0.884%.

## Hierarchy

Pressure DOFs inherit `(x, z, size)` from their dyadic quadtree leaf and `y`
from the tall-cell pressure sample. For multigrid level `l`, a sample maps to
the dyadic ancestor covering the level's horizontal scale. The construction
semicoarsens horizontally first; vertical bins are coarsened only after the
horizontal root is reached. This is intentional: vertical modes are handled
by the block smoother and should not be discarded prematurely.

The measured balanced dam-break hierarchy has six levels and 40--42 DOFs on
the coarsest level.

The transfer is piecewise constant:

```text
(P e_c)_i = (e_c)_parent(i)
(R r)_c   = sum_{parent(i)=c} r_i
R         = P^T
```

Every nonempty aggregate produces exactly one coarse DOF, so `P` has full
column rank and preserves constants.

## Galerkin operator

Topology construction builds only symbolic coarse CSR and two exact maps:

- fine node -> coarse node;
- fine CSR entry -> coarse CSR entry.

After `refreshRows` updates the live free-surface matrix, GPU kernels assemble
each level in order:

```text
A_(l+1) = P_l^T A_l P_l
```

No pressure matrix is read back. Coarse sparsity changes only with topology;
coarse coefficients change every solve. Inactive fine rows contribute the
same identity row used by the fine PCG operator.

The CPU oracle tests the Galerkin energy identity directly:

```text
e_c^T A_c e_c = (P e_c)^T A (P e_c)
```

## Symmetric V-cycle

Each non-coarse level uses one pre- and one post-smoothing step. The smoother
is damped vertical block Jacobi with disjoint two-sample Thomas blocks. Full
height line solves reduced iterations but serialized long columns and were
substantially slower. Scalar Jacobi was fast but left a vertical near-null mode
and eventually stalled PCG; two samples were the smallest stable block in the
dam-break sweep.

For symmetric block solve `S_l`, the level operation is:

```text
z       = S_l r
r_c     = P_l^T (r - A_l z)
e_c     = VCycle(l + 1, r_c)
z       = z + P_l e_c
z       = z + S_l (r - A_l z)
```

The matching pre/post smoother and `R=P^T` make the V-cycle symmetric. With a
stable damping factor, its smoother contribution is `2S - SAS`; the recursive
coarse contribution is also symmetric positive. A dense CPU oracle checks
both `x^T M^-1 y = (M^-1 x)^T y` and positive preconditioner energy.

The coarsest level uses eight fixed, zero-start damped-Jacobi steps in one
workgroup. A fixed iteration count is important: it is a linear symmetric
polynomial in the coarse matrix and is therefore valid inside ordinary PCG.
The earlier one-lane Cholesky prototype was exact but dominated V-cycle time.

## GPU execution

The MG shader has a separate bind-group layout because the projection shader
already consumes the practical storage-binding budget. Per topology it owns:

- one numeric CSR buffer for each coarse level;
- five vector fields per level (`rhs`, correction, defect, and Thomas scratch);
- symbolic transfer/line tables;
- small per-level uniforms.

The V-cycle uses gather restriction. Each coarse row owns its aggregate and
writes one RHS value, avoiding atomic accumulation and a separate clear pass.
All convergence-dependent MG dispatches use the existing indirect-dispatch
gate, so a converged PCG tail executes zero workgroups.

MG also has a tighter iteration-budget predictor. Its first solve encodes 24
iterations instead of deriving hundreds from the hard safety budget; later
solves retain two iterations plus approximately 10% EMA headroom and still use
the existing immediate 2x recovery if a budget is exhausted.

## What remains before making MG the default

The current six-level V-cycle needs 27 hierarchy dispatches per application,
in addition to the PCG reductions. With 13 actual iterations this is still
hundreds of globally decoded commands. The next optimization should preserve
the tested SPD contract while reducing dispatches:

1. Fuse the fine residual/alpha update with the first block-smoothing pass.
2. Fuse correction copy-back with the `r.z` / `r.r` partial reduction.
3. Pack all level vectors into one buffer and evaluate compatible adjacent
   transfer stages in one dispatch.
4. Extend the resident GPU packer to emit parent and inverse-transfer tables.
   MG now uses the GPU face/CSR pack and reads back that compact pack for CPU
   parent-symbolic construction, but unlike polynomial PCG it cannot rebuild
   the hierarchy fully in place yet.
5. Re-run the 0.2 s and long dam-break gates. Promote `mg` only when pressure
   GPU time beats polynomial PCG without increasing topology-stale frames.

Rejected measured variants are retained here to avoid repeating them:

- one-lane exact coarse Cholesky: correct, much too serial;
- full-height vertical lines: strong, much too serial;
- scalar Jacobi blocks: fast, eventually nonconvergent;
- 4:1 hierarchy-level skipping: fewer dispatches, but iteration count rose
  enough to lose overall;
- additive multigrid and alternating unsmoothed levels: SPD, but weaker and
  slower than the full symmetric V-cycle.

## Verification

```bash
node --import tsx --test tests/quadtree-multigrid.test.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js npm run test:quadtree-shaders
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
  FLUID_SCENE=dam-break-ui FLUID_METHOD=quadtree-tall-cell \
  FLUID_TARGET_S=0.2 FLUID_CPU_ORACLE=0 FLUID_PRESSURE_PHASE_TIMINGS=1 \
  FLUID_QUADTREE_PRECONDITIONER=mg \
  node --import tsx tools/run-webgpu-smoke.ts
```
