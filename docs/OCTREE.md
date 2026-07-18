# GPU octree method

The `octree` method is the first pressure-only 3D adaptive implementation. It
keeps dense velocity advection while
using the resident signed-distance field as its single liquid representation.
The adaptive part is the pressure topology and projection.

## Resident representation

Each finest-grid voxel stores an eight-byte owner record:

- the packed `(x, y, z)` origin of its pressure leaf;
- the dyadic leaf size (`1`, `2`, `4`, or `8`).

Only the owner-origin entry carries a pressure value. The dense owner map makes
neighbor lookup constant-time and avoids sparse topology allocation and
topology readback. The pressure solve may compact wet leaf origins strictly as
a GPU execution optimization. It is an intentional MVP tradeoff: pressure
unknowns are adaptive, but topology storage is not yet sparse.

Every simulation step encodes these operations in one GPU command stream:

1. rasterize terrain and rigid primitives into per-cell eight-corner solid
   fractions and a single maximum-coverage owner;
2. tile the domain with the largest in-bounds dyadic roots;
3. refine leaves from their minimum distance to the GPU-resident signed-distance
   surface, producing a finest interface band and progressively coarser deep
   liquid/air cells, while also forcing finest leaves across solid boundaries;
4. split leaves that violate the 2:1 neighbor rule;
5. solve one finite-volume pressure unknown per wet leaf with fixed weighted
   Jacobi iterations, including the rank-six response of dynamic bodies;
6. apply pressure gradients to the dense face-velocity field and extrapolate a
   narrow velocity band for the next advection step;
7. materialize leaf bounds, pressure ownership, mapped leaf pressure, and
   projected divergence into renderer textures.

Coarse/fine pressure coupling traverses a leaf face in finest-grid subfaces. Each
subface contributes `area / centerDistance` to the symmetric finite-volume
stencil, so the same flux is seen from either side of a transition.

There are no topology copies to the host or CPU-dependent convergence decisions
in the advancement path. Rigid-load delivery and `readStats()` use asynchronous
diagnostic/readback buffers after submission; neither controls the encoded
fluid step.

## Surface-detail controls

The fixed `interfaceRefinementBandCells` remains the validated default. Two
advanced controls are available for experiments at a higher finest-grid
resolution, but both default to zero for every scene and quality preset so the
established solver behavior is unchanged. Neither control contains a
scenario-specific path:

- `surfaceDetailStrength` widens the finest pressure support by up to eight
  cells where the resident level set has high discrete curvature or the
  projected velocity varies strongly across a leaf. It runs inside the normal
  full-domain topology rebuild and can only refine more; it never uses the
  previous frame's sparse-residency list to omit physics work.
- `secondaryParticleSurfaceCorrection` optionally folds near-interface
  secondary particles back into the level set. Detached spray stays render
  only. The correction moves a sample by at most `0.2h` per substep, never
  pushes it below `-0.5h`, and does not transfer particle momentum.

Adaptive stepping also enforces the explicit capillary-wave bound
`0.5 sqrt(rho h^3 / (pi sigma))` in addition to the existing velocity CFL
bound. This becomes important as the finest cell width `h` is reduced.

These controls improve pressure support and surface retention around energetic
features, but they do not create resolution beyond the dense velocity/level-set
lattice. Truly sub-lattice ripples still require a finer finest grid or a later
sparse authoritative transport representation.

To inspect the live band, open Scientific view and enable any solver-grid
slice. The default **Structure** field gives finest `1³` pressure cells a
translucent pink tint and pink boundaries, which remain visible when individual
cells are sub-pixel at the overview camera. Select **Cell scale** for the full
hierarchy: pink cells are finest `1³` cells, cyan cells are intermediate dyadic
leaves, and blue cells are the coarsest level supported by the active solver
(the default octree uses `8³`). The palette is normalized to that live maximum,
not to an unavailable fixed tree depth. Both views read the GPU owner texture
directly. Sweeping the X, Y, or Z slice shows the complete liquid- and air-side
surface band for any scene.

## Current scope

- Supported targets include moving free surfaces and immersed rigid bodies.
- Surface geometry: independently advected and narrow-band-redistanced
  GPU-resident level set `phi`. The octree uses monotone RK2 semi-Lagrangian
  transport for `phi`: bounded MacCormack was measurably sharper on smooth
  rotation tests, but generated new zero crossings during the dam-break impact.
  A GPU reduction compares its represented volume
  with the initial level-set reference and applies the corresponding global phi
  shift before sizing/rendering, without consulting VOF or making a CPU
  decision. Mixed-cell contributions are rounded rather than truncated and the
  correction derivative uses the exact smooth-Heaviside support; this prevents
  growing interface area from becoming a false liquid source. Pressure-cell
  liquid classification, velocity transport and extension, force support,
  air-work culling, free-surface projection, and overlays all use `phi` as
  their sole liquid authority. The octree path does not transport conservative
  VOF: it skips the flux limiter, density sharpening, and per-step VOF copies.
  A dormant compatibility texture remains in the shared dense bind layout, but
  has no binding in the octree projection and cannot change a solve decision.
- Pressure solver: row-parallel Chebyshev semi-iteration over the diagonally
  scaled finite-volume operator. Balanced quality's 128-sweep effort maps to
  32 polynomial SpMV passes; high and ultra retain their larger effort budgets.
  The spectral interval `[0.01, 2.2]` covers the modes that the former fixed
  Jacobi budget could materially damp while leaving margin above the
  Gershgorin bound. Dynamic rigid scenes keep this accelerated path by treating
  the uploaded body velocity as prescribed for the current pressure solve and
  applying the newly calculated pressure impulse to the next presentation
  batch. This frame-lagged partitioned split removes the per-iterate global
  `K^T p` dependency; `compact` and `dense` remain exact same-step A/B paths.
- Leaf solve execution (`leafSolver`, default `auto` = `chebyshev`): each solve
  first stream-compacts the wet leaf origins with a deterministic prefix-sum
  scan (per-block reduce, one-workgroup exclusive block scan, rank-and-scatter
  emit) and assembles each row's diagonal, velocity-flux RHS, and merged
  neighbor table once — 2:1 balance bounds a row at 24 entries (6 for finest
  leaves), which bounds the entry pool at six entries per finest cell. The
  Chebyshev passes then run as indirect dispatches over only the compacted rows,
  never touching the owner map or velocity texture. Each row stores its prior
  polynomial correction and recurrence scalar in the row-header padding, so
  there are no reductions, CPU convergence decisions, or single-workgroup
  serialization. The solve is warm-started from the
  persistent pressure buffer instead of a per-step clear (advanced UI toggle
  `pressureWarmStart`, default on; `FLUID_OCTREE_WARM_START=0|1` in the smoke
  harness; the dense ladder always cold-starts). On dam-break-ui (61x46x41),
  the 32-pass Chebyshev path reduced pressure GPU time from 3.28 to 0.72 ms and
  wall time from 4.40 to 2.06 ms/step at 0.2 simulated seconds versus the
  128-dispatch compact Jacobi ladder. Kinetic energy remained within 0.03%, RMS
  divergence within 0.2%, and the 2.2-second impact/rebound gate passed. A
  single-dispatch
  persistent `megakernel` variant (whole loop in one workgroup with a
  workgroupUniformLoad-gated residual early exit) is selectable but measured
  slower (22.7 ms) because transient solves run the full budget serialized on
  one compute unit; it only wins once solves converge in a few iterations.
  `compact` keeps the 128-dispatch compact-row Jacobi ladder and `dense` keeps
  the legacy one-thread-per-finest-cell ladder for A/B
  (`FLUID_OCTREE_LEAF_SOLVER` in the smoke harness).
- Rigid bodies use the adaptive solver's variational volume-of-solid contract.
  Eight corner samples produce a sub-cell solid fraction `s`; each finest
  sub-face then carries the constraint flux `(1-s) u_fluid + s u_solid`, and
  its pressure coefficient is weighted by the open fraction `(1-s)`. The
  default Chebyshev path solves against the body transform and velocity
  uploaded from the previous coupling batch, then returns the current
  equal-and-opposite pressure impulse plus tangential immersed-boundary
  reaction through a pool of asynchronous readback slots. The controller
  accumulates every returned impulse and distributes it over the next fixed
  rigid substeps. This deliberately weakens time coherence by at most the
  bounded two-batch window, but avoids both the 128 in-solve `K^T p` reductions
  and a per-step CPU/GPU fence. The selectable `compact` and `dense` Jacobi
  ladders still include the exact rank-six term `K M^-1 K^T`, with
  `K = grad^T V s L`, and update the body response in the same solve for A/B
  validation. On the 61x46x41 `dam-break-boxes` workload at 0.2 seconds, this
  reduced pressure time from 117.96 to 0.79 ms and wall time from 98.65 to
  3.14 ms/step. Kinetic energy differed by 0.07% and RMS liquid divergence by
  0.11%. A 0.5-second coupled stability envelope, a 14.8%-of-water-volume
  prescribed displacement gate, and a batched two-way dynamic-body smoke test
  all pass. Buoyancy uses the geometric displaced
  volume `sum(alpha_liquid * s * cellVolume)`, including partially covered
  free-surface cells rather than counting body-centre voxels. Surface-volume
  control conserves the complementary open-liquid measure
  `sum(alpha_liquid * (1-s) * cellVolume)`, so increasing immersion raises the
  free surface instead of hiding liquid inside the solid. The resident level
  set also receives the adaptive path's solid-interior `phi-s`
  relaxation so a moving body displaces rather than carries trapped liquid.
- Debug adaptivity control: `0` forces finest pressure cells everywhere, `1`
  enables full distance-graded coarsening, and intermediate values provide a
  controlled quality/performance sweep at fixed finest resolution.
- Renderer topology, pressure ownership, pressure, and divergence slices are
  materialized by a GPU-only pass. The pressure view currently displays the
  solver's velocity-potential variable rather than pressure in pascals.
- Leaf-count telemetry: a conservative startup estimate; exact resident counts
  would require either an asynchronous diagnostics readback or a GPU-drawn UI.

The implementation is therefore a working GPU-first baseline, not yet a claim
of cell-for-cell parity with the quadtree tall-cell solver. The former compacted
Jacobi ladder was dispatch-overhead-bound (~17 us per indirect sweep); the
Chebyshev path removes three quarters of those synchronization points without
collapsing work onto one GPU core. Runtime scheduling queues one presentation
quantum at a time with a second bounded batch allowed in flight, including
frame-lagged rigid scenes (five solves per batch and ten advances maximum at
the default 4 ms clock).
Queue-completion promises, timestamp/statistics maps, and optional diagnostics
readbacks are asynchronous and are never consumed by the step encoder. Rigid
exchange uses reusable staging slots so overlapping maps preserve every
per-step impulse without stalling later submissions. A future geometric
multigrid or communication-avoiding Krylov solve could reduce the remaining
polynomial passes further. An octree-only
shader/bind-layout specialization removing the dormant VOF allocation remains
open.

## Research basis

The discretization direction follows adaptive variational/finite-volume liquid
work such as Ando and Batty's three-dimensional adaptive pressure projection,
while the GPU representation is informed by linear-octree and GPU hierarchy
construction literature. Useful starting points include:

- Ryoichi Ando and Christopher Batty, *A Practical Octree Liquid Simulator with
  Adaptive Surface Resolution* (2020).
- Tero Karras, *Maximizing Parallelism in the Construction of BVHs, Octrees,
  and k-d Trees* (2012).
- Jeroen Bédorf et al., *A sparse octree gravitational N-body code that runs
  entirely on the GPU processor* (2012).

The current dense owner map is simpler than those sparse linear-octree schemes;
it establishes correctness and readback-free stepping before sparse packing is
introduced.

Run the focused stability smoke with `npm run test:webgpu:dam-octree`, or the
2.2-second impact/rebound gate with `npm run test:webgpu:dam-octree-long`.
`npm run test:webgpu:octree-displacement` lowers a box whose volume is more
than 10% of the initial water volume from dry air to full submersion and gates
open-water conservation, free-surface rise, geometric displacement, and the
rigid-load displacement readback on native WebGPU.
`npm run test:webgpu:dam-octree-parity` runs that impact window against the
stable tall-cell level-set configuration and gates checkpoint wet-IoU and
centroid separation, catching coherent-but-wrong trajectories that integrated
volume and component-count tests cannot see.
The dam-break gate measures interface-face growth and enclosed air as well as
volume and dominant-component size, so fragmented topology cannot pass merely
because its integrated volume remains correct.
The late-time volume/coherence regression is
`npm run test:webgpu:dam-octree-soak` (10 simulated seconds).
`npm run test:webgpu:dam-octree-uniform-pressure` forces adaptivity to zero for
the finest-pressure-grid quality control; the matched dense-solver comparison
remains `npm run test:webgpu:dam-octree-compare`.
