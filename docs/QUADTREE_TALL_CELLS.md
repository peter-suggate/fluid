# Quadtree Tall Cells - WebGPU Implementation Contract

This mode implements Narita, Ochiai, Kanai, and Ando, *Quadtree Tall Cells
for Eulerian Liquid Simulation* (SIGGRAPH 2025). The paper and the authors'
July 2025 hindsight note are the specification. Ando and Batty (2020) supplies
the referenced T-junction boundary discretization; Irving et al. (2006)
supplies the corrected inner tall-cell ghost face.

## Paper-to-code checklist

| Requirement | Implementation |
| --- | --- |
| Recursively evaluate sizing at cell centers, coarse-to-fine | `buildQuadtree` |
| Sizing responds to surface curvature and non-translation velocity variation | `quadtreeSizingFromVelocityAndSurface`, with the vertical maximum flattened to x/z |
| No adjacent diameter ratio above two | repeated 2:1 balancing/adaptivity smoothing |
| Quadtree leaves extend from domain bottom to top | one vertical column per x/z leaf |
| Cubes near every interface; tall cells above and below | `populateTallPressureGrid` |
| Splashes/bubbles may create several tall cells in one column | connected runs are segmented independently |
| Cubic pressure at cell centers | `TallPressureSample.kind === "cubic"` |
| Tall pressure at bottommost/topmost replaced cube centers | `tall-bottom` and `tall-top` samples |
| Every pressure sample is horizontally centered in its leaf | corrected hindsight placement in `position.x/z` |
| Horizontal pressure is vertically interpolated on both sides | minimal-face rows in `buildVariationalSystem` |
| Gradient direction is the face normal | pressure difference is divided only by the center distance normal to the face |
| A 2:1 transition uses a 1.5-small-cell center distance | unit test `T-junction gradient...` |
| `[V]` stores the complete dual face-cell volume | center distance times minimal-face area |
| `[A]` stores non-solid area fraction | represented by `openFraction` in the CPU operator; the current GPU integration does not yet populate it from moving-body geometry |
| `[F]` uses Ando--Batty's SPD second-order free-surface scale | Eq. (21)/(25), with negative degenerate values clamped to zero |
| Divergence is the exact negative transpose of the gradient | system assembled as `G^T V A F G` and `G^T V A u*` |
| Inner vertical tall-cell ghost volume is present | vertical face rows span consecutive pressure samples |
| Inner ghost velocity is averaged from background vertical faces | `faceVelocity` loops over the replaced fine faces; it is pressure-only |
| Pressure system is SPD and solved with CG | matrix-free CG on WebGPU with sparse uncompensated IC(0) and level-scheduled triangular solves; solution updates stop once the relative residual is reached |
| Relative stopping target | scene tolerance; paper reference is `1e-4` |
| Pressure is mapped back before velocity update | each solved variational face correction is conservatively prolonged to every represented cubic sub-face, preserving its area average exactly |
| Advection uses the saved previous grid/variables | shared semi-Lagrangian/MacCormack cubic field; quadtree is pressure-only as in Algorithm 1 |
| Optical thickness | one quarter of liquid depth, following Irving et al. and Sec. 6 |

After each submitted step, the current WebGPU VOF and velocity fields are read
back, the next quadtree/tall population is constructed from that saved state,
and the sparse variational rows are uploaded before another step is admitted.
This explicit synchronization preserves Algorithm 1's per-step ordering. The
advection, PCG iterations, pressure mapping, and velocity update remain WebGPU
compute work. It favors fidelity over throughput; a scan/prefix-sum GPU tree
builder can later remove the transfer without changing the operator.

The paper uses EXNBFLIP for additional splash detail, but calls it an
independent enrichment rather than part of the quadtree tall-cell method. This
application retains conservative VOF transport for mass and rendering, while
the pressure geometry is an independently semi-Lagrangian-advected level set
redistanced by a sub-cell-seeded anisotropic Eikonal sweep. No global volume
correction is applied.

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

This implementation must not yet be treated as a complete reproduction of the
paper:

- Section 4.4's monolithic two-way rigid-body system
  `dt G^T V A F G + J^T M_s^-1 J` is not assembled. The application still
  applies its older immersed-body block and reprojects its result. This is not
  algebraically equivalent to Equation (14).
- The independently transported level set uses the paper's trilinear ordinary
  cell branch on a finest-cubic backing field. The adaptive MLS reconstruction
  from Ando--Batty is not yet used for level-set or velocity traces whose
  support crosses a T-junction.
- The pressure-to-cubic correction is the equation-preserving constant
  prolongation described above, not the authors' MLS pressure reconstruction.
- The IC(0) factor follows Bridson's public-domain sparse factorization with
  uncompensated pivots. The paper only describes its own implementation as
  ICCG with minor modifications, so bitwise equivalence is not claimed.

These remaining departures matter especially to moving-body and high-detail
splash cases. The rigid-body box scenario must not be presented as a complete
Section 4.4 reproduction until Equation (14) is assembled monolithically.

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
deep-water cases.

The repaired balanced run reached `0.2 s` in 57 dynamic steps with peak speed
`3.075 m/s`, peak CFL `0.629`, maximum projection energy ratio `1.008`, maximum
relative pressure residual `9.97e-9`, maximum exact transfer drift `1.36e-8`,
and minimum dominant-component fraction `0.9978`. The real browser UI reached
`2.008 s` / 502 fixed steps with matching submitted/completed clocks, visible
liquid, a `1.514 m/s` final speed, residual below `1e-8`, and no new browser or
WebGPU errors.
