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
| Cubes near every interface; tall cells above and below | `populateTallPressureGrid`; the resolved band is computed independently from each connected liquid run's local depth |
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
| Sec. 4.4 monolithic two-way rigid coupling | assembled as `G^T V A F G + K M^-1 K^T` with the rank-6 per-body coupling `K = G^T V (1-A) L` (`c V` is the face area, so `K^T p` is the wetted-surface pressure integral); body impulses are `-rho K^T p`, read back asynchronously every step and exposed through the standard rigid-load callback. Bodies without a load consumer stay kinematic (`M^-1 = 0`) |
| Advection uses the saved previous grid/variables | the pressure level set is semi-Lagrangian-advected every accepted step from a resident GPU texture using the centered value of adjacent MAC faces; narrow-band volume feedback prevents the resident level set from shrinking and collapsing; velocity/VOF remain on the dense finest-cubic backing field |
| Optical thickness | one quarter of each connected liquid column's local depth, following Irving et al. and Sec. 6 |

The pressure level set advances before every pressure solve with the saved-grid
semi-Lagrangian update specified in Section 4.5. A global normal-speed
correction is applied only within 1.5 cells of the interface and is driven by
volume measured from the resident level set itself using the renderer's
four-cell smooth Heaviside. A direct port of the restricted solver's local
two-pass reinitialization produced cadence-boundary volume jumps on this dense
field, so it is not part of the adaptive path. The legacy jump-flood kernels
remain available for explicit experiments but are disabled by default until
their half-cell reset is replaced by a sub-cell-preserving variant.
`refreshFaces` runs on every solve so the free-surface fractions use current phi.

Algorithm 1 evaluates and subdivides the quadtree on every simulation step.
The default cadence is therefore one rebuild per accepted step, and physics
waits for that rebuild instead of advancing on a stale pressure graph. An
explicit `quadtreeRebuildIntervalSteps` remains available only for controlled
comparison runs. At a rebuild, the GPU consumes the resident
level set, vertically reduces sizing, subdivides and smooths the dyadic forest,
then returns the leaf-owner map and one vertical phi profile per unique leaf.
The dense 3D phi and VOF fields remain GPU-resident. Quadtree decode, tall-cell
segmentation, variational assembly, IC factorization and level scheduling, and
MLS packing run in a persistent Web Worker (with the same direct fallback in
non-browser tests). Builder scratch/readback buffers are cached across rebuilds.
No further step runs on the current projection while this work is in flight.

The pressure solve records all ICCG dispatches in one compute pass. A separate
one-thread pipeline owns the indirect-argument buffer because WebGPU treats
each dispatch as a usage scope and does not permit the same buffer to be both
writable storage and indirect input for that dispatch. This removes thousands
of pass begin/end transitions without weakening the relative-residual gate.
The per-iteration denominator/solution work is fused into one ordered kernel,
and residual reduction/best-iterate/direction work into another, reducing the
ICCG command sequence from eight dispatches to five while the sparse matrix
product remains fully parallel. Pressure timestamps are attached to this real
compute pass; empty timestamp-marker passes are not used because Metal may
collapse them and falsely report the solve below timer resolution.
The balanced preset requests at least 96 iterations; high/ultra request
160/240, and the large-system `O(sqrt(n))` cap overrides these minima where
required. The profiler reports the actual
iterations used, so empty indirect-command overhead is visible and tunable.
The advanced preconditioner control exposes a fully parallel diagonal Jacobi
comparison path; uncompensated IC(0) remains the paper-conformant default.
Adaptive scenes retain a one-step CPU/GPU topology handshake so Algorithm 1's
construction cannot be overtaken by another physics step.

An exact sparse-topology cache keeps faces, CSR, IC factors, interpolation, and
MLS rows resident when the full tall pressure graph is unchanged; the resident
level set and free-surface face weights still refresh every step. Moving dam-break surfaces
usually change the graph and correctly miss this cache. The live profiler
reports reuse rather than presenting every in-flight rebuild as merely
`running...`.

The Results section states that the authors incorporated extended narrow-band
FLIP (EXNBFLIP), and the Discussion says it falls back to traditional level-set
methods on coarse grids. This paper does not give enough inline detail to
reproduce that cited method, so this implementation does not claim EXNBFLIP.
Pressure geometry, sizing, and projected liquid/air classification are
level-set-authoritative; the dense VOF field never overrides phi. Volume
feedback is measured and applied directly on that resident level set.

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
- Velocity and VOF advection still use dense finest-grid textures. The paper's
  adaptive-grid advection and memory savings remain a strategic architecture
  change rather than part of this pressure-path remediation.
- EXNBFLIP enrichment and dual-contouring extraction remain deliberately
  omitted presentation features.

Known approximations inside the now-implemented features:

- The resident level set is transported with RK2 + bounded MacCormack and
  redistanced every step by a jump flood measured against projected interface
  points (16.6 fixed point per axis). The narrow band (|phi| < 2.5h) keeps
  the advected phi verbatim, so redistancing never moves the interface; only
  the far field is rebuilt, clamped at 5h. The point-cloud distance carries
  sub-cell tangential ripple in the rebuilt far field.
- Nozzle inflow feeds the resident level set directly (the restricted
  method's phi clamp at inflow cells) and the projection re-imposes the
  prescribed inflow velocity after the pressure gradient; the analytic
  inflow volume still integrates into the volume-controller reference.
- The GPU right-hand side evaluates the open-face fluid flux as
  `A x (average over all represented sub-faces)` of the staged velocity
  texture, whereas the CPU oracle open-weights each sub-face; velocities
  stored inside solids therefore contaminate boundary faces at second order.
- The rank-6 body couplings and `[A]` fields are held between topology rebuilds,
  but displacement by half a finest cell forces a refresh; pressure impulses
  themselves are delivered every accepted step.
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
against its `maximumFluidScale` ceiling and inflow scenes against a minimum
moving speed.

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
