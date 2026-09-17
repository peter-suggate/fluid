# 3D staggered transport and air-band correction

2026-09-17. Implementation and validation checkpoint for
[the 2D dissipation port](2d-levelset-dissipation-and-3d-port.md).

The Sparse Geometric sim panel has **Air-band velocity correction**, enabled
by default. It is a runtime control; switching does not reset the scene.
Reset both runs to the same initial state for a numerical comparison. Off
selects the previous cell-interpolated scalar transport. On selects direct
staggered extension/interpolation, air correction, immutable pre-remesh
momentum sampling and the scalar boundary conditions below. This UI comparison therefore measures the combined path;
the frozen probe separately isolates projection with the sampler held fixed.

## Executing path

After the final liquid pressure projection, frontier publication and existing
cell velocity extension, the enabled path:

1. Classifies represented, positive-capacity air cells with extension support
   and `0 < phi <= 2 max(hx, hy, hz)`.
2. Seeds a private face bank from projected liquid-touching faces, closed
   walls, prescribed inflows and one-sided boundary rows. Eight synchronous
   face sweeps extend physical normal components through open cell adjacency.
   Aperture and solid velocity are removed before averaging and reapplied
   once at the receiving face. Seeds are immutable; the cell velocity cache
   does not supply air-face values.
3. Builds connected air components using GPU atomic union/find. A free face
   adjoining represented air outside the active band anchors the component.
   Open one-sided rows on physical domain planes can also anchor it; interior
   sparse allocation edges remain fixed. Liquid-touching faces, closed walls
   and prescribed inflows are never correction degrees of freedom.
4. Assembles `B_freeᵀ W A B_free` from accepted CNX coefficients, static 3D
   dual weights and physical apertures. Fixed face flux enters the RHS.
   Each enclosed component subtracts its volume-weighted mean divergence;
   isolated cells are counted. There is no arbitrary pressure pin.
5. Runs Jacobi-preconditioned f32 CG, capped at **128 iterations**, with
   device-side arithmetic gating after convergence. The squared
   preconditioned residual target is `max(initial * 1e-8, 1e-14)`. A fresh
   operator application supplies the final residual. A 32-iteration pilot
   converged in only 15/60 production frames; it was not retained.
6. Corrects only free air faces in the private bank. Both V and phi
   characteristics sample independent face components. Regular interior
   stencils reduce to staggered trilinear weights; adaptive and boundary
   stencils use physical widths and an affine MLS fit. Unsupported faces
   do not contribute interpolation weight. Mixed-rung stencils include the
   next face plane to retain tangential affine rank at five-term seams;
   uniform interior support is unchanged. Closed-wall normal velocity is
   authoritative at a coincident wall plane, including nonlinear fields.

The liquid pressure field is unchanged by the correction. No correction to
V, phi-volume offset or expanded sharpening radius is introduced here.

Scalar boundary handling runs in the existing phi-advection dispatch. Closed
physical domain walls continue negative interior phi from one finest-cell
interval inward. Corners use one diagonal RK2 trace through the immutable
source field, so axis/write order cannot select a different answer. Positive
interior phi does not erase an existing wall film. Zero capacity and
separating walls exclude continuation. Incoming ambient air at explicit open
domain planes and existing wall release then take precedence over contact.
Aperture-weighted inflow uses the corrected private face bank when available
and is converted to physical speed; prescribed liquid
inflows exclude ambient air. Zero-dt transport leaves contact unchanged.
These rules cover exterior domain planes, not a new moving cut-surface
contact-angle model.

`readAirExtensionReceiptQA()` exposes enabled/ready state, iterations,
active/isolated cells, corrected faces, convergence/breakdown, initial
and compatible divergence, final divergence error, residuals, allocation and
dispatch count. Reads are requested QA operations, never a frame-loop
readback. Stage timing includes **Air-band velocity correction**.
Finite iteration exhaustion may use an improving correction with
`converged: false`; breakdown, nonfinite or worsened residual disables the
private sampler for that frame. This remains observable rather than silently
being labelled converged.

## Scalar-only pilot before the momentum snapshot

Metal, `coarse-first-pool-impact-half`, dt = 1/30 s, independently reset arms,
60 completed frames each. GPU timestamp markers touch actual simulation
buffers to establish resource dependencies. Empty/unrelated marker passes
were rejected as timing evidence.

| Measurement | Off | On, 128-iteration budget |
| --- | ---: | ---: |
| Projection + extension GPU, median frames 2–5 | 2.33 ms | 38.31 ms |
| Conservative transport GPU, median frames 2–5 | 8.03 ms | 18.06 ms |
| Completed-frame wall time, median frames 2–5 | 93.32 ms | 143.41 ms |
| Additional retained storage after enabling | 0 | 93,012,176 bytes (88.7 MiB) |
| Additional encoded dispatches per enabled frame | 0 | 659 |
| Air solves meeting residual target | — | 60/60 |
| Maximum iterations actually used | — | 70 |
| Frame 1 initial / final max air divergence error | — | 3.26888 / 0.000270 |
| Frame 30 initial / final max air divergence error | — | 8.68438 / 0.003479 |
| Frame 60 initial / final max air divergence error | — | 1.02514 / 0.000260 |

The dispatch budget includes eight face sweeps and 128 five-dispatch CG
slots. Converged slots stop their arithmetic on device, but commands remain
encoded. The boundary changes add no full-domain dispatch. The existing
cell extension remains because other consumers still use it.

Long-run timing is confounded by different adaptation: frame 30 accepts
7,367 cells off versus 1,913 on. Across frames 2–60 the wall-time medians are
91.62 and 82.24 ms respectively; **this is not evidence that the extra solve
is cheaper**. The short same-cell-count comparison above exposes its cost.

| Reported signed V–phi residual, fine-cell³ | Off | On |
| --- | ---: | ---: |
| Frame 1 | 37.313 | 32.876 |
| Frame 5 | 48.193 | 48.790 |
| Frame 30 | −662.510 | 185.283 |
| Frame 60 | −2090.844 | −2603.359 |

The final residual is worse, so these trajectories do **not** establish
uniformly improved surface-volume accuracy. Material starts at 69,724
fine-cell³ and ends at 69,716.694 off / 69,720.712 on. The existing residue
retirement reports deleting 7.370 / 3.357 fine-cell³ respectively. This patch
adds no deletion, but these whole-scene runs must not be reported as exact
mass conservation. Long-run transport, remeshing and retirement remain
separate contributors.

Off allocates no correction scratch and encodes no correction dispatches.
First enable grows the transport buffer and copies its cell-velocity prefix
once; later toggles reuse it. The old prefix stays alive until resident
teardown because its copy is in the caller's unsubmitted encoder. Allocation
accounting includes this retained storage. Arrays currently address stable
capacity; cell/row dispatches follow accepted worklists, not a compact air
worklist.

## Combined scalar and momentum pilot

The next 60-frame reset-state run includes the immutable momentum snapshot,
proportional refinement and consistent phi-based thin-feature exposure. It
precedes the final explicit MLS wall-normal constraint (which has focused
GPU coverage); it is a pilot, not a final acceptance claim.

| Measurement | Off | On |
| --- | ---: | ---: |
| Completed-frame median, frames 2–5 | 112.07 ms | 203.21 ms |
| Momentum preparation GPU median, frames 2–5 | 1.41 ms | 13.37 ms |
| Projection, extension and snapshot GPU median, frames 2–5 | 2.43 ms | 53.35 ms |
| Scalar transport GPU median, frames 2–5 | 11.63 ms | 22.91 ms |
| Air convergence | — | 60/60, maximum 73 iterations |
| Frame 60 signed V–phi residual, fine-cell³ | −859.73 | −820.03 |
| Frame 60 material volume, fine-cell³ | 69,718.59 | 69,722.90 |
| Frame 60 maximum V/C | 1.266 | **4.147** |

The stronger momentum field exposes severe conservative-remap crowding.
The research document's additional liquid-capacity balancing was raised
as a separate pass-cost decision; the current three capacity-normalization
rounds do not solve it. Air convergence must not be used to claim the whole
transport is accurate. Total additional retained allocation is 241,239,040
bytes (230.1 MiB), and the enabled path encodes 663 additional dispatches,
plus the snapshot hash clear/copy.

Raw pilot receipts and focused logs are stored under
[`artifacts/level-set-volume/3d-air-extension/`](../../artifacts/level-set-volume/3d-air-extension/)
(ignored artifacts; principal measurements are retained here).

## Frozen projection intervention

`probe-air-extension-frozen-dawn.ts` uses the same manufactured 3D sphere,
face seeds, synchronous extension, fixed liquid faces and MLS sampler in
both arms. Only committing the air correction differs. Integrating 512
surface-normal samples gives:

| X cell widths | Before correction outward flux | After | Reduction in absolute flux |
| --- | ---: | ---: | ---: |
| 1, 1, 1, 1 | −0.000592 | −0.000428 | 27.7% |
| 2, 1, 1 | −0.189810 | +0.032841 | 82.7% |
| 1, 1, 2 | −0.375102 | −0.128282 | 65.8% |

Flux is fine-cell³/s. This is a manufactured frozen-field intervention,
not a capture of a production trajectory. The discrete solve does not make
MLS pointwise divergence-free or guarantee finite-step phi conservation.

## Validation and remaining scope

Focused GPU checks cover affine fields including domain corners, uniform
and mixed widths, actual five-term coarse/fine seam rows, reflections in all
three axes, partial apertures, physical-velocity extension independent of
the cell cache, liquid/prescribed-face preservation, open anchors, enclosed
compatibility and isolated moving-wall cells. Scalar boundary checks cover
six faces, corners, separating walls, source precedence and zero dt/capacity.
Existing adaptive phi-core and wall-release GPU checks pass. Runtime
control/layout and related volume/control unit checks pass. Repository-wide
type checking has existing failures outside these changed files.

The momentum gather now retains an immutable face/geometry/incidence snapshot
before remeshing, with a private old-cell spatial hash. Its samples survive
replacement of the live geometry and connectivity. Injection and switching
off invalidate it; startup and a newly allocated resident use current support
until their first snapshot. Signed sparse-world coordinates are tested.
Snapshot storage is capacity-sized: the half-pool fixture adds 148,226,864
bytes (141.4 MiB), four capture dispatches, a hash clear and a GPU hash copy.
It adds no second air solve. The Sim timing breakdown separates snapshot
capture from air correction.

The audit also removes density from geometric thin-feature exposure and
replaces first-child refinement residual commits with simultaneous proportional
allocation. Sharpening already has simultaneous proposals and shared endpoint
limiters. The three existing conservative-coupling normalization rounds are
unchanged; this is not a port of the CPU's 64-sweep liquid-capacity balancing.
A ten-frame thin extrusion completes in both arms. It preserves material
(3,276.00043 off / 3,276.00046 on fine-cell³) and changes the signed V–phi
residual from 183.852 to −9.701. Maximum fill remains excessive (3.412 off /
2.878 on). Nine air solves meet the target; the tenth true residual exceeds
its target by 0.7% after recursive convergence, which is reported as
`converged: false`. Production frozen replay and moving cut-surface contact
still need further evidence before claiming the entire research acceptance
matrix.

The canonical regression run completed in 418.16 s: **3/17 lanes passed**.
Ten lanes timed out at their unchanged deadlines (including the mini32
performance lane). Mini64 performance halted on missing compiled connectivity;
Long Dam failed its generation-zero publication count (66 versus 106);
Tall Cells Hills halted on a phi-advection support contract; outside-tank
collapse halted on missing compiled connectivity. This is a failing gate,
not production physics sign-off. The run preceded the final corrected-boundary
accessor and mixed-seam interpolation fix; those have focused GPU coverage.
No lane threshold, timeout, performance ceiling or expected front was relaxed.

## Half-pool slab comparison

`coarse-first-pool-impact-half-slab` is available in both the studio and
Advance Lab through their shared catalog. It retains the half-pool's
6.4 × 4.8 m XY section, 0.1 m finest spacing, gravity, zero viscosity/surface
tension and 1/30 s timestep. Depth is 0.8 m (eight fine cells), with closed
free-slip front/back walls. The suspended liquid is a radius-1 m cylinder
centred at (0, 3.65), extended beyond both walls so its caps do not contaminate
the liquid signed distance. Thus the initial section is exactly invariant in
Z; merely clipping the original sphere would not give that comparison.

Use the same preset in Advance Lab's 2D LevelSetVolume mode and in the 3D
Sparse Geometric solver. Compare the centre XY slice at equal simulated time,
and divide 3D material/surface volume by 0.8 m to compare with 2D area. The
initial continuum area is 6.4 × 1.6 + pi = 13.381593 m². Adaptive 3D topology,
pressure and transport still differ from the 2D algorithm, so trajectories
are not expected to be bit-identical. Reset between correction Off/On runs.

Final-head Metal smoke: both reset arms completed 30 frames; all 30 enabled
air solves converged (maximum 72 iterations). At frame 30, signed V–phi
residual is −259.110 off / −107.775 on fine-cell³, and maximum V/C is
3.857 off / 1.576 on. Material volume is 10.703866 / 10.704004 m³. This still
exposes remap overfill; the additional balancing stage is not in this release.
The enabled slab allocates an additional 65,982,688 bytes (62.93 MiB).
Run the scene probe with `--scene=coarse-first-pool-impact-half-slab`.

Reproduce serially under the repository's exclusive WebGPU lease:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal node --import tsx --test --test-concurrency=1 tests/sparse-cm12-air-extension-dawn.test.ts tests/sparse-cm12-levelset-boundaries-dawn.test.ts tests/levelset-volume-core-dawn.test.ts tests/levelset-volume-wall-separation-dawn.test.ts
node --import tsx --test tests/sparse-cm12-air-extension.test.ts tests/adaptive-volume-return-controls.test.ts tests/geometric-volume-compiled-topology.test.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal node --import tsx tools/probe-air-extension-dawn.ts --steps=60 --out=artifacts/level-set-volume/air-extension-3d.json
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal node --import tsx tools/probe-air-extension-frozen-dawn.ts
npm run test:dawn:sparse-cm12
```

The scene probe also accepts `--thin=1` and `--arm=on|off|both`. A previously
resized 1.6 × 1.2 × 1.6 m control failed with
`MISSING_COMPILED_TOPOLOGY_FACE` in the off arm before producing a frame;
the reported timings use the original scene.
