# Quadtree Tall Cells - WebGPU Implementation Contract

This mode implements Narita, Ochiai, Kanai, and Ando, *Quadtree Tall Cells
for Eulerian Liquid Simulation* (SIGGRAPH 2025). The paper and the authors'
July 2025 hindsight note are the specification. Ando and Batty (2020) supplies
the referenced T-junction boundary discretization and the MLS interpolation
(full text extraction: `docs/papers/ando-batty-2020-octree-liquid.txt`);
Irving et al. (2006) supplies the corrected inner tall-cell ghost face.

## Paper-to-code checklist

| Requirement | Implementation |
| --- | --- |
| Recursively evaluate sizing at cell centers, coarse-to-fine | `buildQuadtree`; the split demand is the maximum sizing over the candidate leaf's footprint, because a centre-point sample can never trigger the first split for a sub-leaf feature and the dilation passes only expand refinement that already exists |
| Sizing responds to surface curvature and non-translation velocity variation | `quadtreeSizingFromVelocityAndSurface`, with the vertical maximum flattened to x/z; rigid-body proximity (`initialSizing`) is the only persistent explicit source, so flat regions genuinely coarsen (the deep-water headline case) while curvature spikes always register through the footprint-maximum evaluation |
| No adjacent diameter ratio above two | repeated 2:1 balancing/adaptivity smoothing |
| Quadtree leaves extend from domain bottom to top | one vertical column per x/z leaf |
| Cubes near every interface; tall cells above and below | `populateTallPressureGrid` |
| Splashes/bubbles may create several tall cells in one column | connected runs are segmented independently |
| Cubic pressure at cell centers | `TallPressureSample.kind === "cubic"`; every retained cube is a single-cell segment with its own sample, so cubic bands reduce to the ordinary per-cell stencil |
| Tall pressure at bottommost/topmost replaced cube centers | `tall-bottom` and `tall-top` samples |
| Every pressure sample is horizontally centered in its leaf | corrected hindsight placement in `position.x/z` |
| Horizontal pressure is vertically interpolated on both sides | minimal-face rows in `buildVariationalSystem` |
| Gradient direction is the face normal | pressure difference is divided only by the center distance normal to the face |
| A 2:1 transition uses a 1.5-small-cell center distance | unit test `T-junction gradient...` |
| `[V]` stores the complete dual face-cell volume | center distance times minimal-face area |
| `[A]` stores non-solid area fraction | `openFraction`, populated from rigid-body geometry every rebuild (8-corner sampling per cell; the prescribed inflow channel stays carved out of a filled display nozzle exactly as the legacy kernel carved its inflow cells). The face flux becomes the constraint flux `A u_fluid + (1-A) u_solid` |
| `[F]` uses Ando--Batty's SPD second-order free-surface scale | Eq. (21)/(25), with negative degenerate values clamped to zero and a ghost-fluid ceiling of `maximumFluidScale = 100` (theta >= 0.01): a nearly-emptied surface sample otherwise produces unbounded face gradients that a converged solve injects as kinetic energy |
| Divergence is the exact negative transpose of the gradient | system assembled as `G^T V A F G` and `G^T V A u*` |
| Inner vertical tall-cell ghost volume is present | vertical face rows span consecutive pressure samples |
| Inner ghost velocity is averaged from background vertical faces | `faceVelocity` averages every vertical face over its leaf's full x/z footprint (ghost or single-cell); it is pressure-only |
| Pressure system is SPD and solved with CG | matrix-free CG on WebGPU with sparse uncompensated IC(0) and level-scheduled triangular solves; solution updates stop once the relative residual is reached. The iteration budget has an `O(sqrt(n))` floor: the deep-water column stack (~190k samples from the quarter-depth cubic band) converges superlinearly but only after ~500 iterations with 100x transient residual growth, and a budget it cannot converge within makes the best-iterate guard silently apply a zero correction (reported relative residual exactly 1, whole tank in free fall). MIC(0.97), plain Jacobi, per-column tridiagonal block-Jacobi, and cubic-first/tall-first elimination orderings were all measured worse than plain IC(0) on that system |
| Relative stopping target | `max(scene tolerance, 1e-4)`, so this method follows the paper's `1e-4` target instead of inheriting the CPU-reference default `1e-8`; a dedicated convergence pass writes indirect arguments and suppresses remaining PCG work after convergence |
| Pressure is mapped back before velocity update | sub-faces of multi-sub-face variational faces receive the Ando--Batty MLS pressure-gradient reconstruction (Eq. (33)-(35), epsilon 1e-2) plus the additive shift that makes each face's sub-face corrections average exactly to the solved variational value (Eq. (5)); single-sub-face faces and very large (deep tall) faces keep the constant conservative prolongation. The shift is what prevents the July failure mode from recurring: the mapping cannot invent net face flux |
| Sec. 4.4 monolithic two-way rigid coupling | assembled as `G^T V A F G + K M^-1 K^T` with the rank-6 per-body coupling `K = G^T V (1-A) L` (`c V` is the face area, so `K^T p` is the wetted-surface pressure integral); body impulses are `-rho K^T p`, exposed through the standard rigid-load callback. Bodies without a load consumer stay kinematic (`M^-1 = 0`) |
| Advection uses the saved previous grid/variables | shared semi-Lagrangian/MacCormack cubic field; quadtree is pressure-only as in Algorithm 1 |
| Optical thickness | one quarter of liquid depth, following Irving et al. and Sec. 6 |

Topology rebuilds are pipelined and coalesced. The default cadence is
displacement-driven: it may rebuild after eight accepted substeps once a
speed-plus-gravity bound reaches half a finest cell, and always rebuilds by
32 steps. An explicit `quadtreeRebuildIntervalSteps` still selects a fixed
cadence for comparisons. At a rebuild,
the GPU advects and reconciles the level set, redistances it with a 3D
jump-flood pass, vertically reduces the sizing field, recursively subdivides
the dyadic forest, enforces 2:1 balance, and applies the three smoothing
dilations. One compact mapped staging buffer returns the level set, the
one-word-per-column leaf map, pressure diagnostics, and timestamps. The VOF
field remains GPU-resident and is not read back. The host still constructs the tall-cell face graph, sparse
IC(0), level schedule, MLS rows, and upload images; that remaining symbolic
sparse stage is shown separately in the live profiler and is not described as
GPU work. Further steps may run on the current projection while a rebuild is
in flight, bounded by `quadtreeTopologyLagSteps` (default 3). A completed
rebuild is no longer followed immediately by a catch-up rebuild: accumulated
state is coalesced until the next displacement/cadence boundary.

The pressure solve records all ICCG dispatches in one compute pass. A separate
one-thread pipeline owns the indirect-argument buffer because WebGPU treats
each dispatch as a usage scope and does not permit the same buffer to be both
writable storage and indirect input for that dispatch. This removes thousands
of pass begin/end transitions without weakening the relative-residual gate.
The balanced preset caps at 96 iterations (the live dam-break converges in
about 25); high/ultra retain 160/240, and the large-system `O(sqrt(n))` floor
still overrides these caps where required. The profiler reports the actual
iterations used, so empty indirect-command overhead is visible and tunable.
Uncoupled adaptive scenes may queue up to eight fluid steps behind one renderer
fence; rigid-coupled scenes retain the one-step CPU/GPU impulse handshake.

An exact sparse-topology cache keeps faces, CSR, IC factors, interpolation, and
MLS rows resident when the full tall pressure graph is unchanged; only the
level set and free-surface face weights refresh. Moving dam-break surfaces
usually change the graph and correctly miss this cache. The live profiler
reports reuse rather than presenting every in-flight rebuild as merely
`running...`.

The paper uses EXNBFLIP for additional splash detail, but calls it an
independent enrichment rather than part of the quadtree tall-cell method. This
application retains conservative VOF transport for mass and rendering, while
the pressure geometry is an independently semi-Lagrangian-advected level set
redistanced by a GPU jump-flood distance transform. Because only the
VOF receives sources (inflow) and conservative transport, every rebuild
reconciles the advected level set against it: cells whose wet/dry sign
disagrees with VOF occupancy by more than half a finest cell are reseeded from
the VOF sign, agreeing cells keep their advected sign, and the GPU distance
transform restores a distance-like field. Sub-half-cell interface disagreement is left
alone: stamping the VOF's quantized offsets into phi creates curvature noise
that drives the sizing function to full refinement.
Without this, injected liquid is projection-air and the projection freezes it;
the pre-reconciliation sign-mismatch fraction is exported as a drift
diagnostic and gated by the smoke matrix. No global volume correction is
applied.

## July 2026 UI failure and first violated requirement

The actual balanced dam-break UI first became corrupt during the first
pressure projection. At step 1 (`t = 0.004 s`) the pre-projection kinetic-energy
proxy was `1.456e-4`, the pressure solve reported a small residual, but the old
cubic pressure reconstruction increased the collocated RMS divergence from
`0.309 s^-1` to `1.041 s^-1`. By `t = 0.180 s` the displayed liquid speed was
`1905.8 m/s`; later the topology reached zero liquid DOFs and WebGPU rejected a
four-byte placeholder bound as an eight-byte `array<Entry>`. The host loop kept
advancing, but every command buffer was invalid, which was the apparent freeze.

The initiating violation was Algorithm 1 line 10, "Map Pressure onto Cubical
Cells", together with Equations (1) and (5). A variational face is an
area-averaged velocity unknown. The former projection independently
reconstructed two vertical pressure samples inside each leaf and differentiated
those values on the background grid. That invented unconstrained sub-face
pressure gradients, so the projected cubic face average no longer equalled the
face correction solved by `-G^T V A F G p = -G^T V A u*`.

The correction applies the solved face gradient and free-surface factor to all
of its represented background sub-faces. This is the constant, conservative
prolongation of the variational face unknown: averaging the prolonged values
returns exactly Equation (5)'s face value. The zero-DOF buffers also retain the
WGSL structure stride, preventing the downstream validation failure if a truly
empty pressure domain occurs.

## Paper details not yet satisfied

This implementation must not yet be treated as a bitwise reproduction of the
paper:

- The IC(0) factor follows Bridson's public-domain sparse factorization with
  uncompensated pivots. The paper only describes its own implementation as
  ICCG with minor modifications, so bitwise equivalence is not claimed.
  (Measured on the deep-water system: MIC(0.97), Jacobi, per-column
  tridiagonal block-Jacobi, and elimination reorderings all converge worse
  than plain IC(0).)

Known approximations inside the now-implemented features:

- The parallel 3D jump-flood redistance is not the paper implementation's
  exact fast-sweeping Eikonal solve. It removes the dominant serialized host
  pass while preserving anisotropic physical distances and decisive VOF sign
  reconciliation; nearest-seed Voronoi error remains a documented numerical
  approximation.
- The GPU right-hand side evaluates the open-face fluid flux as
  `A x (average over all represented sub-faces)` of the staged velocity
  texture, whereas the CPU oracle open-weights each sub-face; velocities
  stored inside solids therefore contaminate boundary faces at second order.
- The rank-6 body couplings and `[A]` fields are rebuilt from the previous
  step's body states, the same one-step lag as the topology rebuild.
- The MLS pressure mapping applies to variational faces representing 2..32
  sub-faces (T-junction and small vertical faces); deeper tall faces keep the
  constant conservative prolongation.
- Ando--Batty's MLS for level-set and velocity interpolation is applied where
  this architecture actually interpolates adaptive data: the pressure mapping
  above. The transported level set and the advection traces live on the
  finest-cubic backing grid, so their trilinear interpolation never crosses a
  T-junction by construction (their Sec. 5 note: MLS blends to trilinear on
  regular data).

## Verification

`tests/quadtree-tall-cell-grid.test.ts` is the small deterministic oracle. It
checks recursive subdivision, the ordinary-grid limit, 2:1 transitions,
multiple disconnected tall runs, the 1.5-cell T-junction denominator,
symmetry/positive semidefiniteness, corrected ghost volumes, CG residual
reduction, and the sizing response.

`npm run test:webgpu:dam-break-regression` runs the actual UI dam-break fixture
for at least `0.2 s` with unequal timesteps. Every accepted step records staged
velocity finiteness, peak speed/CFL/energy, projection energy ratio, pressure
and projected variational residuals, exact topology-transfer mass drift,
component count and dominant-component fraction, and front progression. It
also fails on device loss or any WebGPU validation error. The broader smoke
matrix retains settled tank, dam break with boxes, hose inflow, sphere jet, and
deep-water cases. Quadtree runs additionally gate the free-surface scale
against its `maximumFluidScale` ceiling, the pre-reconciliation level-set/VOF
sign-mismatch fraction against 2%, and inflow scenes against a minimum moving
speed (a projection that treats injected liquid as air freezes the field at
numerical zero while volume grows — the failure mode the reconciliation
repairs).

The repaired balanced run reached `0.2 s` in 57 dynamic steps with peak speed
`3.075 m/s`, peak CFL `0.629`, maximum projection energy ratio `1.008`, maximum
relative pressure residual `9.97e-9`, maximum exact transfer drift `1.36e-8`,
and minimum dominant-component fraction `0.9978`. The real browser UI reached
`2.008 s` / 502 fixed steps with matching submitted/completed clocks, visible
liquid, a `1.514 m/s` final speed, residual below `1e-8`, and no new browser or
WebGPU errors.

After the 2026-07-15 remediation (VOF reconciliation, free-surface ceiling,
single-cell cubic segments, footprint-maximum sizing, vertical-face averaging,
iteration floor), the whole six-scenario quadtree matrix passes at `0.2 s`:
dam-break regression peak speed `4.75 m/s`, CFL `0.789`, projection energy
ratio `1.015`, residual `1e-8`, exact drift `1.2e-8`, dominant fraction
`0.9997`; hose and sphere-jet inflow move at `0.78`/`0.47 m/s` (both were
frozen at numerical zero); deep-water holds hydrostatic equilibrium at
`0.7 mm/s` on 917 coarse leaves with residual `9.7e-9` (previously a
construction crash, then a silently stalled solve in free fall).
