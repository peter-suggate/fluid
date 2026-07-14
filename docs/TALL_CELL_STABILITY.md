# Tall-Cell Stability Audit

## Scope

This audit compares the browser tall-cell implementation with Chentanez and
Müller (2011) and records the no-rigid-body deep tank-fill reproduction used to
investigate the intermittent eruption.

- Paper notes: [`TALL_CELLS_PAPER.md`](TALL_CELLS_PAPER.md)
- Implementation overview: [`TALL_CELL_WEBGPU.md`](TALL_CELL_WEBGPU.md)
- Reproduction scene: **Load deep-water A/B scene**
- Scene: `20 m` high tank, `80%` fill, no rigid bodies, `Delta t = 1/30 s`,
  surface tension disabled, balanced tall-cell grid
- Grid: `61 x 26 x 41` stored, `1021` cubic-equivalent vertical cells,
  maximum tall height `806`

## Reproduced failure

Before the corrections below, the browser reported the following after 35
encoded GPU steps:

| Signal | Baseline |
| --- | ---: |
| Encoded physical time | `1.167 s` |
| Liquid maximum speed | `75.607 m/s` |
| Post-projection maximum divergence | `1.56e3 s^-1` |
| Raw VOF drift | `-0.08%` |
| WebGPU validation errors | `0` |

The UI clock read `1.90 s`, but only 35 paper-sized GPU steps had executed. The
clock discrepancy was itself a bug: render lag was silently discarded by the
GPU solver's `advanceTo` method.

The failure is not caused by rigid bodies and does not begin as a volume-loss
event. Speed and divergence grow by orders of magnitude while VOF volume is
still close to its initial value.

## Paper-to-code comparison

| Area | Paper | Previous implementation | Current status |
| --- | --- | --- | --- |
| Surface field | Advected level set, periodic narrow-band reinitialization | Persistent VOF; pressure `phi` reconstructed each solve | Deliberate departure; retained |
| Velocity extrapolation | Full hierarchical known/unknown solve | `airHalo` fine-grid neighbor passes | Deliberate departure; retained and diagnosed via air-speed maximum |
| Force domain | Euler equations are solved where `phi < 0` | Gravity was added to every active packed sample, including air | Corrected: force integration is limited to liquid samples |
| Remeshing cadence | Every step | Every step | Aligned with Algorithm 1 |
| Remesh constraints | `G_L`, `G_A`, and neighbor delta `D` | Fractional VOF samples broadened the surface range; base zero bypassed `D` | Corrected: sign crossings define surfaces, air wins conflicts, and every base obeys `D` |
| Pressure cycles in examples | One full cycle plus two V-cycles | One full cycle plus one V-cycle | Corrected to the paper's example budget |
| Coarsest solve | Shared-memory Gauss-Seidel to high precision | 24 iterations of weighted Jacobi | Corrected to 256 red-black Gauss-Seidel iterations |
| Pressure convergence evidence | Residual convergence plot | No GPU pressure residual | Added exact finest-level `L-infinity` residual and relative residual |
| Printed pressure gradient | Positive-minus-negative pressure divided by `Delta x` | Implemented literally | Stability departure: two valid samples use their physical `2 Delta x` span; walls remain one-sided |
| Time advancement | One physical time step per algorithm step | Excess requested time was discarded while `lastTime` jumped forward | Corrected: advance by at most `maxDt` and report remaining lag |
| Instability evidence | No prescribed live gates | Volume, speed, post-divergence only | Added stage extrema, locations, residual, CFL, finite-state count, and flags |

## Root causes

### 1. Gravity was integrated in extrapolated air

The extrapolated air band exists to support semi-Lagrangian traces. Applying a
new gravity impulse to those samples creates a falling-air mode that re-enters
the liquid through the collocated interface stencil on the next step.

Restricting the body force to the liquid domain removes this mode. In the first
corrected deep-tank step, extrapolated-air maximum speed changes from one full
gravity impulse (`0.327 m/s`) to zero before the next extrapolation.

### 2. The literal pressure-gradient denominator flips hydrostatic velocity

At an interior collocated sample, `pPlus` and `pMinus` are two cell centers
apart. The previous kernel divided their difference by one cell width. For a
correct linear hydrostatic pressure field, that produces twice the physical
gradient: a downward gravity impulse is reflected into an equally large upward
velocity rather than cancelled.

The paper prints the same one-cell denominator. Direct tests with the corrected
remesher still reach `1.41 m/s` at `0.25 s` and `17.9 m/s` at `0.333 s` in a
settled tank. The samples are physically `2 Delta x` apart, so the printed
formula evaluates twice the centered derivative.

The corrected kernel divides by the actual sample span:

- `2 Delta x` when both positive and negative samples exist; and
- `Delta x` for a one-sided physical wall.

This is retained as an explicit correction to the printed Equation 17. It is
not attributed to a different storage layout: both implementations use
collocated velocity samples.

### 3. The coarse pressure solve was under-converged

The previous shared-memory top solve used 24 weighted-Jacobi iterations even
though the paper specifies Gauss-Seidel to high precision. On the first deep
tank step, the relative finest-level residual was `1.02` and maximum pressure
was only about half the expected hydrostatic value.

With a true red-black Gauss-Seidel top solve, 256 iterations, and the paper's
second V-cycle, the same case reaches a relative residual near `1.3e-2` and a
hydrostatic pressure maximum near `1.57e5 Pa`.

### 4. The GPU clock hid skipped work

When requested time advanced by more than `maxDt`, the solver encoded one
clamped step but assigned `lastTime = requestedTime`. The unencoded remainder
was lost. The corrected clock advances `lastTime` only by the encoded `dt` and
reports `simulationLag_s` until the GPU catches up.

### 5. The VOF limiter admitted speculative replacement volume

Receiver capacity previously included raw outward flux. Some of that outflow
was later reduced by the donor/receiver limiter on the adjacent face, so the
cell received more replacement liquid than it actually released. Clamping the
result to one destroyed the excess. In the 20 m tank this gradually eroded a
few tall-cell averages through the liquid threshold; the resulting remesh then
triggered the eruption.

Receiver capacity is now based only on current empty volume. Every internal
face remains pairwise and bounded, so the closed deep tank has exactly zero
measured volume drift through `4.4 s`.

### 6. The remesher confused fractional density with surface topology

The paper remeshes from zero crossings of reinitialized `phi`. The VOF
translation instead treated every value in `[0.01, 0.99]` as a surface. Small
transport diffusion eventually made most of a column appear to contain
surfaces, creating contradictory `G_L`/`G_A` bounds. A special base-zero path
then bypassed the neighbor bound and allowed a height collapse much larger
than `D`.

The remesher now uses liquid/air sign changes, prefers the air halo when bounds
conflict, and applies `D` to every proposed height.

### 7. The hose source and visible nozzle described different boundaries

The analytic hose outlet originally ended `0.13 m` beyond the rendered nozzle,
which is about 6.6 cells in the balanced grid. Replacing the old painted source
with a conservative one-sided reservoir exposed that gap: the entering liquid
was physically emitted in mid-air and appeared as a detached blob.

The nozzle now terminates at the analytic outlet. Because the renderer's
cylinder primitive is filled rather than hollow, the tall-cell solid mask also
classified the prescribed reservoir cells as solid and suppressed their flux.
Those cells are now treated as the nozzle's open channel: the reservoir
velocity is imposed after rigid coupling and the opening is excluded from the
pressure solid mask. This is a boundary-condition correction, not a density
dilation or stream-shaping force.

At `3.0 s`, native Metal execution admits `4877.8` cell-volumes through the
tall path and `4765.5` through the matched uniform path, a `2.4%` difference.
Maximum speeds are `1.78 m/s` and `1.84 m/s`, respectively, with bounded
volume fractions and no stability flags.

## Corrected measurements

After eight encoded steps (`0.267 s`):

| Signal | Corrected |
| --- | ---: |
| Liquid maximum speed | `0.131 m/s` |
| Extrapolated-air maximum speed | `0.129 m/s` |
| Maximum divergence, pre to post | `16.8 -> 4.78 s^-1` |
| Projection divergence ratio | `0.284` |
| Pressure relative residual | `1.90e-2` |
| Maximum pressure | `1.57e5 Pa` |
| Maximum component CFL | `0.223` |
| Wet samples above CFL 1 | `0` |
| Non-finite values | `0` |
| Raw VOF drift | `0.00%` |

After 131 encoded steps (`4.367 s`, well beyond the original failure at 35
steps):

| Signal | Corrected |
| --- | ---: |
| Liquid maximum speed | `0.123 m/s` |
| Extrapolated-air maximum speed | `0.065 m/s` |
| Maximum divergence, pre to post | `16.9 -> 4.71 s^-1` |
| Projection divergence ratio | `0.280` |
| Pressure relative residual | `1.95e-2` |
| Maximum pressure | `1.57e5 Pa` |
| Maximum component CFL | `0.210` |
| Non-finite values | `0` |
| Raw VOF drift | `0.00%` |

## Live diagnostic contract

Every tall-cell diagnostic readback now includes:

- GPU simulated time and lag relative to the UI request;
- liquid and extrapolated-air maximum speeds;
- maximum divergence before and after projection, their ratio, and locations;
- maximum pressure and its location;
- finest-level maximum pressure residual, relative residual, and location;
- maximum component CFL and the number of wet samples above one;
- volume integral and drift;
- maximum tall height;
- non-finite count across pre-pressure, pressure, and projected fields; and
- explicit stability flags.

Locations use cubic-equivalent `(x, y, z)` coordinates rather than packed `y`
indices, so a failure can be identified as a wall, tall endpoint, free surface,
or regular-band event.

The current flags are:

- `non-finite-state`;
- `pressure-residual` when relative `L-infinity` residual exceeds `0.1`;
- `advective-cfl` when maximum component CFL exceeds `1`;
- `post-projection-divergence` when `maxDivAfter * dt > 0.5`; and
- `projection-amplified-divergence` when the preceding gate is active and
  projection also increases the maximum divergence by more than 5%.

These are detection gates, not claims that the collocated projection should be
idempotent. The dimensionless divergence threshold prevents the paper's
expected small non-idempotence from producing a constant false alarm.

## Verification

- `npm run test:unit`: 74 tests pass, including GPU clock, inflow-boundary,
  remesh, and stability-gate regression tests.
- `npm run build`: passes.
- Native Dawn/Metal WebGPU smoke: the `2.0 s` settled tank and `3.0 s` hose A/B
  runs pass their invariants with no validation errors.
- `npm run lint`: the modified solver paths pass; the repository command still
  reports the pre-existing synchronous state update in
  `components/RecordingPlaybackModal.tsx:39` (and its related hook-dependency
  warning).

The deterministic Node runner can execute the real WGSL kernels when a Dawn
WebGPU module is supplied through `WEBGPU_NODE_MODULE`; see
[`WEBGPU_DIFFERENTIAL_SMOKE.md`](WEBGPU_DIFFERENTIAL_SMOKE.md).
