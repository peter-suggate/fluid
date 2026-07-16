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
| Remesh constraints | `G_L`, `G_A`, and neighbor delta `D` | Fractional VOF samples broadened the surface range; base zero bypassed `D` | Corrected: sign crossings define surfaces, air wins conflicts, and four ping-pong passes enforce `D` on the proposed field |
| Pressure cycles in examples | One full cycle plus two V-cycles | One full cycle plus one V-cycle | Corrected to the paper's example budget |
| Coarsest solve | Shared-memory Gauss-Seidel to high precision | 24 iterations of weighted Jacobi | Corrected to depth-dependent red-black Gauss-Seidel budgets; extreme depth uses 192 initial and 144 correction iterations |
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

## 2026-07-15 dam-break audit: 24-layer devolution reproduced and localized

Reproduction matrix on `dam-break-ui` (61×46×41 cubic grid, dt 0.004 s, 2.0 s,
surface tension pinned to 0 for comparability, per-step stability envelope,
`FLUID_DISABLE_TIMESTAMPS=1`; traces in `tmp/tall-cell-audit/*.jsonl`):

| Config | Backend | Peak projection KE ratio | First CFL>1 | Outcome |
|---|---|---|---|---|
| uniform | uniform | 1.0006 | t=0.27 s (impact, 1.71 peak) | stable, drift 3.5e-7 |
| tall `regularLayers=46` | **uniform (silent fallback)** | — | — | confirms past "wide band" runs never used tall kernels |
| tall `regularLayers=44` (bases pinned 2) | restricted | 1.0008 | t=0.24 s (4.4 peak) | stable; IoU vs uniform ≥0.37 through splash chaos, recovers ≥0.7 |
| tall `regularLayers=24` (deep tall) | restricted | **8.86** | **t=0.084 s** | speeds→1e29, 130k non-finite, all volume deleted by t≈1.8 s |
| tall 24 + semi-Lagrangian | restricted | 3.10 | t=0.092 s | same failure — MacCormack exonerated |

Localization: the projection amplifies kinetic energy from step 1 (ratio >1.1
at t=0.004 s) only when deep tall cells exist. Extremum locations cluster at
the **tall-bottom endpoint** (max pressure 463/500 steps, pressure residual
400/500, divergence-after 196/500). This matches the endpoint-wetness defect:
the packed bottom sample stores the tall-cell column-average VOF, and gravity
(`finishAdvection`), pressure wetness (`buildPressureRhs`), projection
(`project`), and the multigrid φ (`buildFinePhi`) all gate on it as if it were
the paper's point sample (Eq 4/5). Front water flooding a 22-subcell tall
column reads as air (average ≪ 0.5) — no gravity, no pressure constraint — and
a draining column loses hydrostatic support for its entire lower column at
once. `planRemesh` cannot see liquid inside the tall region (crossings are
scanned only over the stored band), so the state persists across steps.

Dawn note: with the `timestamp-query` feature enabled, the tall solver's
step-1 projection pass (and adjacent copies) does not execute under the Dawn
node module — the step-2 pre-projection state equals two uncorrected gravity
impulses. All previous smoke runs had timestamps on. Audit runs disable them
(`FLUID_DISABLE_TIMESTAMPS=1`); root cause not yet identified.

### Fix stack landed 2026-07-15 (measured on the same 24-layer dam break)

1. **Endpoint point-sample wetness (B1)** — gravity, pressure RHS, projection,
   extrapolation seed, multigrid φ, renderer, and readback all derive endpoint
   occupancy from the settled fill `alpha·base` instead of the raw column
   average (`lib/tall-cell-kernels.ts` pointSampleAlpha/settledTallAlpha,
   `lib/tall-cell-multigrid.ts` fineSourceAlpha, `lib/webgpu-water-pipeline.ts`
   fieldCell).
2. **Full-column remesh with no temporal clamp (B2)** — planRemesh treats a
   settled surface inside the tall region as a crossing and bases move as far
   as constraints demand in one step; a representability floor (a base may
   never strand the column's own water above the band ceiling) outranks the
   neighbor bound D at cliffs, with the invariant excusing floor-bound deltas.
3. **Consistent projection pairing (B3)** — one-sided face divergence and
   gradient whose composition is exactly the compact multigrid Laplacian,
   replacing the paper's printed Eq 13/14 + Eq 17 pair (Appendix A.6a).
4. **Conservative remap** — band-copy rescaling when the copy overdraws the
   column, and overflow settling of residuals beyond one full tall cell into
   band capacity.
5. **Volume-correction divergence (mass-conserving paper Sec 3.7)** —
   `min(0.5(ρ'−1), 1)` per 1/30 s for overfull stores.

| Metric (2 s dam-break-ui, 24 layers) | Before | After | Uniform |
|---|---|---|---|
| Outcome | all liquid deleted at t≈1.8 s | runs to completion | runs |
| Non-finite values | 129,924 | 0 | 0 |
| Peak projection KE ratio | 8.86 (compounding) | 1.67 (release transient) | 1.0006 |
| Exact volume drift | −100% | 8×10⁻⁶ | 4×10⁻⁸ |
| Peak liquid speed / CFL | 10³¹ / 10²⁹ | 53 m/s / 10.9 (splash transients) | 8.4 / 1.7 |
| Wet-IoU vs uniform | 0 from t=1.8 s | min 0.42, final 0.54 | — |
| Stored density maximum | 32.5 | 1.00 | 1.00 |

The minimal-tall control (`FLUID_REGULAR_LAYERS=44`, bases pinned at 2)
bottoms out at 0.37 IoU against uniform through the chaotic slosh, so the
deep-tall run now sits inside the achievable envelope of the restricted
scheme. Gates: `npm run test:webgpu:dam-tall-active`.

### Remaining measured gaps (priority order)

1. Splash-phase transients reach CFL ~11 (uniform 1.7; minimal-tall control
   4.4) at wall-impact columns with shallow tall stubs — gated at 16 as a
   regression backstop, not yet at parity.
2. Release-transient projection KE ratio 1.67 while deep tall columns
   dominate (gated at 2.0).
3. ~~Velocity extrapolation~~ — implemented 2026-07-15: paper Sec 3.3.1
   hierarchy (`lib/tall-cell-extrapolation.ts`, `FLUID_HIERARCHY=0` reverts).
   GPU validation pending (see below).
4. ~~Density sharpening~~ — implemented 2026-07-15 on both paths
   (`sharpenCompute`/`sharpenScatter`/`sharpenResolve`, `FLUID_SHARPENING=0`
   reverts); gated relatively in the smoke suite (tall mixed-cell fraction ≤
   2× uniform's). GPU validation pending (see below).
5. Ceiling mist: fractional VOF (>0.001) accumulates at band ceilings during
   splash (~half the columns transiently); sharpening (item 4) is expected to
   shrink this population — measure after GPU validation.
6. Dawn timestamp-query quirk: with timestamps enabled the first step's
   projection pass does not execute (all previous smoke runs were affected);
   audit runs use FLUID_DISABLE_TIMESTAMPS=1. Root cause unidentified.


### 2026-07-15 (later): GPU compute wedge — validation checklist after reboot

### 2026-07-15 (final): post-reboot validation complete — all suites green

After the reboot restored GPU compute, the full battery passes: 125 unit
tests, `test:webgpu:dam-tall-active`, all six scenes across all three methods
(`FLUID_SCENE=all`), `test:webgpu:dam-conservation` (5 s, ≤0.1% drift with
sharpening active), and `test:webgpu:dam-break-regression`. Final 24-layer
dam-break numbers (2 s): exact volume drift 6.3e-5, stored density max 1.00,
mixed-cell fraction 1.02 vs uniform's 1.90 (the sharpened tall interface is
now thinner than uniform's), front reaches the far wall.

Hardening landed during validation:
- Sharpening deposits carry mass in 2^-20 fixed point and tall-bottom scaling
  happens at resolve time, so small contributions cannot round away.
- The advection upper clamp is gone on both paths (it silently destroyed
  sharpening deposits above one); the inflow source alone is bounded by
  remaining cell capacity, restoring the old nozzle behavior. The
  stored-density invariant allows 1.5 on both paths while the correction
  divergence drains temporary excess.
- The uniform path gained the Sec 3.7 correction divergence; the quadtree
  method runs without sharpening (its Narita/Ando tracking is level-set
  driven), with the envelope tolerating its first two async residual samples.
- Inflow scenarios run 0.5 s so the frozen-scene gate measures an established
  jet instead of ambient equilibrium noise (the calmer post-fix equilibrium
  sat below the old 0.01 m/s floor — an improvement, not a freeze); the
  dam-front gate applies only from 1.5 s of simulated time.

The earlier "frozen inflow" reports were the stale gate plus the sub-0.05 s
targets, not code regressions. The Dawn timestamp quirk IS real post-reboot:
with timestamp-query enabled the entire first step's command buffer silently
does not execute under the Dawn node module (step-1 readback shows an
all-zero field; step 2 onward is normal). Suites tolerate the one-step loss;
audits use `FLUID_DISABLE_TIMESTAMPS=1`.

While hardening the fixes, the machine's GPU compute path wedged system-wide
(a trivial Dawn compute dispatch silently no-ops; Chrome's requestAdapter
returns null; copies still work). Every "frozen inflow scene" measurement from
that window is untrustworthy — including the sphere-jet tall 0.0032 m/s and
quadtree hose/sphere freezes recorded above — and must be re-baselined. The
wedge likely followed the repeated NaN/1e31 fault workloads of the blow-up
reproductions. A reboot restores GPU compute.

Post-reboot validation checklist (all with
`WEBGPU_NODE_MODULE=/private/tmp/fluid-webgpu/node_modules/webgpu/index.js`):

1. `node --import tsx tmp/tall-cell-audit/probe-minimal.ts` — must print a
   nonzero compute sum (GPU healthy).
2. `npm run test:webgpu:dam-tall-active` — the calibrated 24-layer gates.
3. `FLUID_SCENE=all node --import tsx tools/run-webgpu-smoke.ts` — full sweep;
   re-baseline sphere-jet/hose-tank before attributing failures to code.
4. A/B the new stages on the dam break: `FLUID_SHARPENING=0` and
   `FLUID_HIERARCHY=0` against defaults (expect: sharpening lowers the
   mixed-cell fraction and ceiling mist; hierarchy changes far-field air
   velocities only, CFL/IoU within envelope).
5. `npm run test:webgpu:dam-conservation` and
   `npm run test:webgpu:dam-break-regression` (non-regression).

### 2026-07-15 (browser + soak): long-horizon hardening

Running the UI dam break past the 2 s gate window surfaced a low-probability
late-slosh blow-up (Chrome lost the GPU device at t≈4.3 s; a Node 5 s run
reproduced non-finite collapse intermittently). A 2×-per-config 5 s matrix
localized it:

| Config (5 s × 2 runs) | Outcome |
|---|---|
| baseline | survives |
| sharpening off | survives |
| **hierarchy off** | **dies both runs (130k non-finite)** |
| dt halved | survives |

- The Sec 3.3.1 extrapolation hierarchy is **required**, not optional: without
  it, far-field air keeps stale velocities forever (the consistent projection
  deliberately passes air–air faces through) and splash-CFL traces sample
  garbage until the field collapses. `FLUID_HIERARCHY=0` is a diagnostic-only
  unstable combination.
- The residual baseline tail risk tracks splash CFL (4–8.5 sustained, spikes
  to 20+; dt/2 survives). Root cause is Appendix C gap 9: our flux-form VOF
  transport is CFL-limited while the papers' fixed 1/30 s step relies on
  unconditionally stable semi-Lagrangian schemes. Mitigation: the tall solver
  now subdivides its step by the last observed component CFL (up to 8
  substeps, `lib/webgpu-eulerian.ts` advanceTo). The paper-true resolution
  remains implementing the mass-conserving paper's conservative
  semi-Lagrangian advection (Appendix B.2) — the one conformance gap still
  open, now explicitly load-bearing rather than cosmetic.
- `npm run test:webgpu:dam-tall-soak` runs the 5 s soak with the stability
  envelope; the 2 s `dam-tall-active` gates remain the deterministic CI bar.

### 2026-07-15: one-tall-cell differential reproducer

`npm run test:webgpu:single-tall-cell` now runs two instances of the same
restricted solver, with identical kernels, multigrid, dimensions, initial
field, and time step. The paper-compliant control has one bottom height-two
tall cell in every column; height two was measured to be exactly equivalent
to the cubic limit. The candidate changes only column `(15,10)` to height
four. That column is fully submerged, its free surface remains in ordinary
cells, and its height delta is two, so it satisfies Sec. 3.1 and Sec. 3.6
Eq. 10. Topology is frozen only for this diagnostic so Algorithm 1 remeshing
cannot add a second variable.

The readback records VOF and reconstructed velocity before and after pressure
projection, split into the probe column, its four face-neighbor columns, and
the far field. On the first `0.004 s` dam-break step:

- advection/pre-projection velocity is byte-identical;
- the all-height-two control projects maximum divergence from `2.0049` to
  `0.00741 s^-1`;
- the one-height-four candidate projects to `0.11685 s^-1`;
- the first velocity difference is at the probe/neighbor stencil, about
  `0.0032 m/s`; and
- by `0.2 s`, the probe's top subcell differs in VOF by `0.8505`.

Twelve pressure cycles reduce the candidate's relative pressure residual from
about `3.7e-3` to `1.2e-5` without reducing the projection discrepancy, which
rules out incomplete multigrid convergence. A one-step height sweep gives
exact equality at height two, then monotonically increasing endpoint error at
heights three through six.

This matches the limitation stated by the tall-cell paper after Sec. 4: the
collocated solver measures divergence only at a tall cell's endpoints and is
unaware of flow through middle faces exposed to neighboring cubic cells. The
paper says this causes volume gain, uses the remeshing `D` constraint only as
mitigation, and explicitly notes that its pressure projection is not
idempotent. A literal A/B of printed Eq. 13--18 confirmed that warning: even
the all-height-two control increased divergence from `2.00` to `3.04 s^-1`,
so that experiment was not retained in production.

The paper's explicit Eq. 5 linear velocity interpolation was also tested in
production. It raised the candidate's first-step post-projection divergence
from `0.11685` to `0.17649 s^-1` and made the existing active-tall regression
fail with a `5.11` peak projection-energy ratio. It was therefore not retained:
the later projection-consistency audit below documents why the compact
endpoint operator currently depends on piecewise velocity reconstruction.
This unresolved paper-conformance conflict is now explicit rather than hidden.
The differential harness is the acceptance path for fixing it: first run the
one-step test, then the `single-tall-cell-soak`, then the active-tall and 5 s
dam-break suites.

### 2026-07-15 (evening): projection-consistency audit — three pumps found

Side-by-side against the uniform reference at matched report cadence, the
tall path's kinetic energy proxy tracked identically through the collapse
(0.35 at t=0.25 s) and then *grew* — 0.87 → 1.47 → 2.19 by t=1 s while the
uniform reference decayed to 0.075 — with `projection-amplified-divergence`
raised and every extremum tagged `tall-top-endpoint`. More pressure V-cycles
made it worse (KE 2.26 at 8 cycles vs 0.97 at 2), the signature of a solve
converging on an operator inconsistent with the applied projection. Four
independent defects, all fixed in `lib/tall-cell-kernels.ts`:

1. **Correction-divergence sign.** The RHS was `ρ(div + c)/dt`; projecting
   with `v -= dt/ρ ∇p` then leaves `div_new = −c`, i.e. *inflow into
   overfull cells* — a positive feedback at every ρ′>1 site (remap residuals
   fire constantly in tall stores). Now `ρ(div − c)/dt` in both the tall and
   legacy-uniform shaders; the conformance test asserts the `−` form.
   (Dormant in the uniform method only because its field never exceeds one.)

2. **Top-endpoint dilution.** With the linear in-cell velocity profile, the
   top endpoint's downward face `v(base−2) = mix(v0,v1,(base−2)/(base−1))`
   moves nearly in lockstep with the endpoint dof, so projection removed only
   `1/(base−1)` of that sample's divergence while the multigrid row assumed
   full-strength coupling — measured post/pre divergence ratio 173/248 = 0.70
   matched the predicted `1−1/(base−1)` at mean base ≈ 4 exactly. Velocity
   now reconstructs *piecewise* inside a tall cell (top world cell = top dof,
   rest = bottom dof; pressure/phi/solid keep the paper's Eq 5 linear
   interpolation), which makes the existing compact stencil the exact
   composition div∘grad at every sample row (verified algebraically for
   base = 2 and base > 2). Remap restriction and the smoke tool's
   reconstruction updated to match.

3. **Band-ceiling ghost kick.** `project` treated the unrepresented cell
   above a column's band ceiling as a liquid–air ghost face (θ clamped at
   0.05 ⇒ up to 20× the solved pressure as an upward kick), while the
   pressure rows carry no term for that face and no volume can cross it —
   an unmodeled ejection at every wet-ceiling column (~1000 mid-slosh).
   The correction is now skipped exactly where the row has no term.

4. **Ghost-fluid θ mismatch.** `project` used the alpha-based
   `interfaceFraction` while the multigrid rows use the phi-based
   `ghostFraction` with floored one-sided distances: a full cell against
   empty air got θ = 0.5 in the projection vs 0.909 in the row — an ~1.8×
   over-applied gradient at every sharp free-surface face, visible as
   post-projection speed maxima exceeding pre-projection ones. `project`
   now reproduces the row's fraction exactly from the alphas.

Also fixed while auditing: `rawVolumeFlux` used a centered face speed where
divergence/projection use the one-sided positive-face convention (transport
now advects with the same face field the projection controls), and the
Sec 3.5 sharpening scatter now returns mass to its own cell when the 2.1-cell
trace never reaches the 0.5 iso-contour — in diffused sub-0.5 fog the
trace-end deposit acted as a clumping operator that nucleated free-floating
droplets (the uniform dam break went from 808 ≥0.5-components at t=2 s to 1).

After the fixes the tall dam break tracks the uniform energy envelope through
the slosh (KE 0.62 at 0.5 s, 0.28–0.49 at 1 s, decaying) with physical front
speeds (max |v| ≈ 3.2 m/s at t=0.25 vs 5.4 before). Remaining known risk:
occasional late-impact CFL spikes (Appendix C gap 9, unchanged) can still
run the flux-form transport hot; the substep gate reacts one readback late.

### 2026-07-15 (night): isolated height sweep and parity boundary

The differential probe was extended with optional Manhattan support rings.
Radii one and two (5 and 13 height-four columns) increased the raw residual
and endpoint-velocity error; making the whole domain height four did so again
despite eliminating every Eq. 10 height transition. This rejects additional
base smoothing as the primary remedy. The discrepancy is intrinsic to the
endpoint-only representation identified in the tall-cell paper's Section 5.

The complete printed operator family was also tested as one unit: Eq. 5 linear
velocity reconstruction plus Eqs. 13--18 centered collocated projection. It
reduced the isolated height-four probe's velocity RMS by 16%, but failed the
active 2 s gate with an `8.41` projection-energy ratio. The current stack does
not include the paper's whole nonsymmetric pressure/advection construction, so
that partial transplant was rejected rather than presented as conformance.

Tall-cell depth is the monotone control that remains within the paper's grid
definition. A height-three probe cuts the height-four projection error almost
exactly in half (`3.72e-4` vs `7.67e-4` RMS; `1.86e-3` vs `3.73e-3` maximum).
In the violent 2 s dam break, capping height at three changes peak projection
energy amplification from `3.66` to `1.001` and remains finite. Making every
dry column height three was still unnecessarily divergent. The paper's
surface constraints do not require that: the near-full-height band covers
those columns, so production mutes dry candidates to the height-two control
and uses height three only below the initial liquid (one-cell jumps, within
Eq. 10). This raises final wet-IoU versus uniform from `0.328` to `0.638`.

The remaining mixed-cell excess was closed without weakening the gate by
using the sharpening paper's permitted cohesion control: `tau=0.45` on the
tall path versus its `0.4` baseline. The update remains the same conservative
Eqs. 14--17 plus local Algorithm 2 scatter. The exact packaged 2 s regression
then passes with early IoU `0.903`, final IoU `0.638`, 48 final raw VOF
components, projection-energy ratio `1.001`, no stability flags, and no
non-finite values.

Production therefore defaults `maximumTallHeight` to 3. The UI and
`FLUID_MAX_TALL_HEIGHT` can raise it for compression experiments. At the
default scene this parity mode stores 97.8% as many cells as uniform, so it is
a correctness boundary rather than a performance result. It is not a claim
that the paper limits tall-cell height; deeper defaults require a coherent
middle-face pressure representation first.

## 2026-07-16 audit: settled-fill reconstruction was the energy pump; parity clamp removed

A single-tall-cell differential probe (`FLUID_SINGLE_TALL_CELL=4`, dam break,
remesh frozen) isolated the dominant instability. The probe column's
reconstructed profile read `1.00 1.00 0.53 0.00 | 0.37 0.27 ...` while the
all-cubic control read `1.00 0.58 0.53 0.50 | 0.38 0.27 ...` with nearly
identical column mass: the settled point-sample reconstruction (fill height
`alpha*base` above the terrain) re-teleported band water to the column floor
on every step, manufacturing a phantom air gap under falling water. The
pressure solve collapsed that gap each step; the collapse re-accelerated the
same mass, and a single height-four column contaminated the far field with
`~7 m/s` velocity differences within one second. The energy budget puts the
whole excess in the advection/forces stage (`+0.482` vs uniform's `+0.107`
KE in `t in [0.25, 0.5) s`); the projection stage never amplified.

Fixes landed (all suites green, 133 unit tests):

- **Constant-density tall reconstruction.** A tall cell is `base` uniform
  cells sharing the stored average (mass-conserving paper semantics on the
  merged cell). `volumeCell`, `pointSampleAlpha`, the advection endpoint
  gates, the multigrid's `fineSourceAlpha`, the transport preparation pass,
  the smoke tool's cubic reconstruction, and the renderer's
  `fieldCell` all share this view (`tallStoreAlpha`). The settled
  interpretation survives only inside `planRemesh` as the draining-pool
  surface estimate, which is a remesh-time prior rather than a per-sample
  reinterpretation. Dam-break KE at `t = 1 s` fell from `0.508` to `0.149`
  against uniform's `0.108`, and the decay slope now tracks uniform.
- **Section 3.7 corrections in density units.** The tall store's excess
  drain and deficit refill were scaled by `base` (subcell units), up to
  `base` times the paper's rate; both now use the paper's density form
  `min(0.5 * |rho - 1|, 1) * 30 /s`. Disabling the refill entirely makes the
  late-slosh event below unrecoverable (KE `87` and residual `8.7` stuck at
  `t = 2 s`), so the refill stays: it damps rather than feeds that event and
  it keeps submerged partial stores away from the `0.5` classification
  cliff.
- **Parity clamp removed.** `maximumTallHeight` defaults to `4096`
  (was `3`), restoring genuine tall cells and grid compression:
  `deep-water` passes its `< 0.5` compression gate again (`0.056` stored
  fraction at 1021 vertical cells) with equilibrium physics clean. The
  clamp was masking the reconstruction defect at the cost of the method's
  entire purpose; with the defect fixed it is a diagnostic control only.

Verification: `settled-tank`, `deep-water`, and `dam-break-boxes` invariants
all pass with deep tall cells; `npm run build` and the full unit suite are
green. Uniform-path behavior is untouched.

**Open item — episodic late-slosh multigrid failure (deep bases only).** In
the 2 s deep dam break the pressure residual creeps from `0.006` to `0.64`
over `t in [1.1, 1.4] s` while remeshing roughens the base field
(heights 2..14, neighbor deltas 4), then the solve diverges for ~0.3 s
(residual up to `8`, local divergence `4500 /s`), the CFL-4 speed rail
contains the burst, and the run fully recovers by `t = 1.9 s`. Eight
V-cycles shrink the burst (KE spike `5.7` vs `52`) but do not remove the
residual spike, so this is a solvability/convergence defect of the
multigrid on remesh-roughened deep base patterns, not under-iteration alone.
It is the reason the historical parity clamp existed. The
`test:webgpu:dam-tall-active` gate documents the gap (projection-energy
ratio `2.7`-`4.5` against its `2.0` limit, plus transient
dry-under-wet columns during the event); the h<=3 configuration passes the
same gate. Reproduction: the gate script with `FLUID_REPORT_EVERY=1`;
watch `pressureRelativeResidual` from `t = 1.1 s`.

## 2026-07-16 (later): store control-volume pressure constraint

The constant-density reconstruction removed the settled-fill pump but the
store still drained under sustained vertical motion: both papers sample
divergence only at the tall endpoints (Sec 9.1 names the resulting volume
error), so the per-row conservative flux integral was unconstrained and the
deep region under the collapsing dam classified dry within a quarter second
(465 dry-under-wet columns at t = 0.224 s; `tools/dump-tall-slice.ts`).

The bottom endpoint's pressure equation now carries its CONTROL VOLUME's
divergence: the stratified row-integral of lateral point divergences over
every shared-dof row plus the exact shared-face vertical closure, divided by
(base-1). Together with the top endpoint's point divergence the two
constraints sum to the store's exact transport boundary flux, so a converged
projection leaves the conservative update mass-neutral. The adjoint pair:
laterally the store is ONE degree of freedom and couples bottom texel to the
neighbouring stores' bottom texels (hydrostatically exact — coupling per-row
linear pressures against a flat own value free-accelerated the settled
tank); vertically it couples to the top endpoint at 1/distance^2. The same
stencil runs in the kernels' Jacobi and all multigrid levels; `project`
applies the matching gradients. At base 2 every formula reduces to the
previous operator. Result: dry-under-wet columns 465 -> 0 through the
collapse, remeshing follows the true surface, and the multigrid residual
stays ~1e-3 through the dam break where it previously diverged episodically
to ~8.

Remeshing additionally keeps tall cells inside the paper's operating
envelope (Sec 5's interior-face error scales with height times interior
velocity): a column whose endpoint dofs exceed 5% of the CFL rail, or whose
submerged store drifts below 0.85 density, descends two cells per remesh
toward the ordinary control height and regrows when calm. An instantaneous
collapse fired tank-wide at startup and the synchronized remap destroyed the
equilibrium scenes; the gradual form does not.

Open items after this pass:
- **y0-row conditioning.** The 1/distance^2 vertical anchor is weak, so the
  V(2)-cycle leaves relative residuals ~0.6-0.8 on quiescent deep tanks
  (micro-currents 0.06-0.2 m/s, no runaway; 10 cycles reach 0.59/0.064).
  `settled-tank` and `deep-water` equilibrium gates fail on the residual
  flag alone. Likely wants y-line relaxation of the endpoint pair or a
  rescaled unknowns basis.
- **Air-velocity igniter.** The episodic blow-up survives (now ~t=2.3-2.5 in
  the 3 s dam break): max AIR speeds bounce 5 -> 18 -> 30 m/s several steps
  before liquid KE ignites, independent of tall height (persists at h<=2),
  refill damps rather than feeds it. Needs a checkpoint-replay bisection of
  the extrapolation/advection air path.
- Dam-break late-slosh KE remains above uniform (~0.2-0.8 vs 0.02-0.1)
  between the fixes and the event.

## 2026-07-16 (third pass): dam-break dome, Eq 5 reconstruction, and the base-1 coarse-level instability

Reproduction (`tmp/tall-cell-audit/probe-collapse.ts`): in the UI dam break
the water column's top rounds into a slow-falling dome while the edges drain
("high internal friction"). At t = 0.05-0.15 s the probe showed the ENTIRE
dam interior — including the floor row — falling uniformly at ~0.88 g
(vy = -9.4 t at every height), pressure at the floor at ~13% of hydrostatic,
and a reconstructed fine-grid divergence of -21 s^-1 at the floor row
cancelled by +0.96 s^-1 across the 21 shared-dof rows above it. The VOF
limiter blocks the implied compression, so mass barely moves while velocity
free-falls: the dome is the visible difference between edge columns that
really drain and interior columns that only pretend to.

### Root cause 1: constant reconstruction + averaged store constraint

The store's control-volume divergence (previous section) averages the closed
floor face against the whole column's lateral flux, and the
piecewise-constant velocity reconstruction (the acknowledged "Eq. 5
departure") makes the interior vertical derivative zero. Together they admit
an exact null mode: uniform free fall balanced by uniform lateral spreading
satisfies the averaged constraint at any amplitude. Gravity feeds that mode
every step; the projection removes ~12% of it.

Fix, following the paper:

- **Eq 5 linear reconstruction.** `validVelocityCell`/`velocityStateCell`
  (and the extrapolation hierarchy and smoke-runner mirrors) interpolate
  velocity linearly between the endpoint dofs. Remap restriction samples the
  old profile at the new endpoint sub-cells.
- **Endpoint point divergences (Eq 13/19).** `divergenceAt` is now the point
  divergence at the sample's own sub-cell for every row; the bottom endpoint
  sees the closed floor face directly (`u_solid`, Eq 14), which pins the
  bottom dof to the wall. With the linear profile, divergence varies
  linearly inside the store, so the two endpoint constraints control every
  interior row up to neighbor-base mismatch.
- **Bottom-row vertical coupling 1/(distance*h).** The Eq 15/16 row at the
  bottom sub-cell (solid below, Eq 5 interpolated pressure above) couples to
  the top endpoint at s/h^2 with s = 1/(base-1) — this is simultaneously the
  paper's collocated row and the exact staggered adjoint of the new bottom
  divergence. The previous 1/distance^2 anchor was one factor of (base-1)
  too weak.
- **Top row keeps the paper's strong coefficients** (band 1/h^2, bottom
  1/(distance*h)). The exact staggered adjoint (band s/h^2, bottom s^2/h^2)
  closes the paper's O(1-s) interface-divergence leak and was verified to do
  so, but it anchors the top-dof layer so weakly at large bases that the
  multigrid diverged outright on the 20 m deep-water scene (relative
  residual 4e5 within three steps). The paper's non-idempotent projection is
  the stable trade; the leak is bounded because the floor row is now exact.

### Root cause 2: degenerate base-1 coarse multigrid levels

The user-visible companion bug: the SETTLED tank at the default 22% fill
blows up at ~1.4 s with no rigid bodies (and the dam break re-ignites at
~1.5 s once its slosh settles to the same shallow state — the previous
section's "air-velocity igniter"). At fill <= 0.3 every column has base 2-3,
and Eq 9 halving (`downsampleBase`) turns that into COARSE base 1, where
both endpoint dofs land on the same world cell (`sampleY` = 0.5 for each):
whole coarse levels carry duplicated unknowns. The coarse correction then
re-injects a floor-row checkerboard that the fine smoother cannot remove —
the residual sticks at relRes ~0.8-1.5 regardless of V-cycle count, each
step leaks ~12% of a gravity impulse into the velocity field, and the tank
detonates after a few hundred steps. Deeper fills only produce base 1 on the
tiny top level, which is why fill 0.35+ was always stable.

Fix: coarse columns own a genuine h >= 2 tall cell whenever the level can
represent one (`if(upper>=2){b=max(b,2);}` in `downsampleBase`), mirroring
the fine layout's own rule. Shallow tanks (fills 0.22-0.30, tension on or
off, dt 4 ms or 8.3 ms) are now motionless for 2+ s with relRes ~7e-3, and
this also retired the previous section's "y0-row conditioning" and
"air-velocity igniter" open items — both were this bug.

### Classification floor for submerged stores

A drained store crossing alpha 0.5 under standing water fell OUT of the
pressure solve, so the Sec 3.7 refill stopped and the column stuck as an air
cushion at alpha ~0.45. `tallStoreAlpha`/`volumeCell`/`fineSourceAlpha` now
apply the paper invariant the code already documented: a store below a WET
band bottom classifies as liquid (occupancy floor 0.5); transport continues
to read the raw conservative average through `transportVolumeCell`.

### Measurements

- Dam break (balanced, dt 4 ms): monotone collapse, no dome; back-wall
  height ~0.45 h0 at t = 0.5 s (Martin-Moyce ~0.45); slosh decays
  KE 145k -> 10k (probe proxy) over 2 s with no re-ignition. The uniform
  method still shows a milder residual mound (its 64 cold-started Jacobi
  sweeps under-converge the hydrostatic mode; untouched by this pass).
- `test:webgpu:dam-tall-active` PASSES: projection KE ratio 1.12 (limit 2),
  peak CFL 3.97 (was 10.9-21.7 on the passing baseline), exact volume drift
  1.6e-5 (limit 1e-3), dry-under-wet columns 0.
- `test:webgpu:single-tall-cell-soak` passes (tall-vs-control RMS ~1e-4).
- Settled tank fill 0.7 (dt 1/120): max speed 1.1 mm/s over 3 s.
- Deep-water 20 m, 1 s: max speed 6 cm/s, relRes 0.076, no flags.

## 2026-07-16 (dam-settling handoff): fixed two-cycle solve was below the remeshed convergence floor

### Durable reproduction

`npm run test:webgpu:dam-settling` now records CPU-side PE, pre/post-projection
KE, projection energy delta, divergence, pressure residual, and exact reconstructed
volume every 50 steps for the same 10 s `tall-cell,uniform` run. The settling
gate rejects a normalized late mechanical-energy slope above `1e-3 /s` and a
late/middle KE-envelope ratio above one. The slope allowance is six times the
same-run uniform proxy floor observed during the reproduction (`1.61e-4 /s`),
but almost nine times below the failing two-cycle tall result (`8.88e-3 /s`).
The release transient's maximum signed-distance occupancy swing remains in the
JSON summary, while the 1% settling-volume assertion uses the final sample;
interface-area growth makes the transient occupancy proxy unsuitable as a
settling assertion. The shorter active-dam regression uses a 2% final
represented-volume limit and a separate 15% catastrophic transient-proxy
backstop.

Before the fix, the tall backend was confirmed active (61 x 46 x 41 cubic
equivalent, 26 stored layers, all 2501 columns tall). Its 10 s trace had:

- late mechanical-energy slope `+6.858e-3 /s` (`+8.884e-3` of initial energy/s),
  versus uniform's `+1.227e-4 /s` proxy floor;
- sampled net projection gain `+5.069e-3` (`+0.00657` of initial energy), while
  uniform projection removed `8.706e-3`;
- late/middle KE envelope ratio `0.578` (so the direct slope, not that secondary
  envelope, is the reproducer).

### Localization and root cause

Stage attribution places the persistent bias in projection: the two-cycle run's
sampled projection budget was positive, whereas uniform's was strictly negative.
The decisive probe was pressure effort on the identical remeshing configuration:

| Refinement V-cycles | final relative residual | sampled net projection delta | normalized late slope |
| ---: | ---: | ---: | ---: |
| 2 | `5.39e-3` | `+5.07e-3` | `+8.88e-3 /s` |
| 4 | `1.77e-3` | `+7.46e-3` | `+4.68e-3 /s` |
| 8 | `3.00e-5` | `+1.53e-3` | `+4.31e-4 /s` |

The trajectory is chaotic enough that the intermediate budget is not monotone,
but only eight cycles cross both the residual and settling-energy floor. In the
late half specifically, eight cycles changed sampled net projection work from
positive to negative (`-1.53e-3`) and mechanical energy from `+3.07e-3` growth
at two cycles to `-1.04e-3` decay. The mechanism is therefore the historical
fixed FMG + two-V-cycle schedule leaving a remesh-conditioned tall pressure
system under-converged; its incomplete correction retains divergence and has a
positive projection-work bias over repeated slosh cycles. This is H2, without
the old pre-264429b signature of convergence toward an inconsistent operator.

A geometry bisection supports the tall/remesh conditioning part of that result:
forcing every tall cell to height two with the same two-cycle budget produced no
positive projection samples, late slope `-4.02e-3 /s`, and late/middle KE ratio
`0.0443`. Residual magnitude alone is not comparable across these geometries;
the failure requires the variable-height remeshed system.

### Fix and negative results

The validated default is now eight refinement V-cycles in the method preset,
direct solver fallback, and multigrid constructor. `FLUID_PRESSURE_CYCLES`
remains available for diagnostic/performance probes. A residual-controlled GPU
schedule would be a future optimization; the correctness default no longer
ships below the measured convergence floor.

Negative results retained from the localization:

- Disabling global level-set volume control did not remove the underlying
  conservation problem: the run gained 163% represented volume by 10 s. The
  controller is necessary, not the late energy source.
- Treating the stronger deep top row as a simple `(base-1)` row scaling and
  scaling its fine RHS likewise was rejected and fully reverted: the 20 m
  deep-water scene lost 92% volume and fragmented. Coarse transfer is not
  invariant under that fine-row rescaling.
- The UI dam's fine bases do not exceed its 24 regular layers, so that deep-row
  branch cannot explain this particular reproduction; the handoff's H1 is not
  the active fine-level mechanism here.

Post-fix valid 10 s run (`tmp/tall-cell-audit/handoff-final-settling.jsonl`):
tall relative residual `8.18e-5`, net projection delta `-1.15e-4`, normalized
late slope `-1.21e-3 /s`, late/middle KE ratio `0.504`, and final sampled exact
volume drift `0.651%`; uniform passed the same assertions. Subsequent 1 s
equilibrium checks also passed (`settled-tank`: drift `0.124%`, speed
`4.09e-4 m/s`; `deep-water`: drift `-0.026%`, speed `0.0134 m/s`, compression
`0.0255`, no flags). Active-dam and long conservation/soak retries then hit an
acknowledged device-state fault (including runs where both independent solvers'
fields stayed at initialization and `encodedSteps` remained zero), so those
runs are recorded but are not physics evidence.

## 2026-07-17: MAC transport and GPU-resident CM12 volume control implemented; live acceptance blocked

The mass-oscillation plan is implemented in the restricted `tall-cell` path:

- Velocity reconstruction and advection now use a component-staggered sampler.
  Every predictor/reverse trace starts at the component's positive MAC face,
  trace velocities reconstruct all three components at their own faces, and
  the MacCormack limiter uses the eight corners of that component's staggered
  interpolation cell. The Eq. 5 endpoint reconstruction is unchanged.
- The CPU/readback-driven normal-speed offset of phi is removed. The substep
  planner captures the completed step's reconstructed volume and wet-interface
  count before the reduction buffer is cleared. On the following step, pressure
  subtracts a globally normalized divergence source with CM12 Sec. 3.7's
  `lambda=0.5` and `eta=1` per-step clamp. A 0.1% deadband is the only
  implementation tolerance. `FLUID_VOLUME_CONTROL=0` still disables the source.
- The settling summary now median-smooths exact drift over the late half,
  reports first-difference sign changes and peak-to-peak drift, and gates at
  at most three changes and 0.5%. `FLUID_REFERENCE_VOLUME_SCALE=1.02` provides
  the prescribed +2% step-response probe.
- `advective-cfl` now means escalation: maximum component CFL above the CFL-4
  speed rail, or at least 32 wet samples above one. CFL telemetry is unchanged.

Archived pre-change traces validate that the new gate is a regression lock:
`handoff-final-settling.jsonl` has 5 late sign changes and 4.645% peak-to-peak
drift for restricted tall cells, versus 0 and 0.00216% for uniform. The
uncontrolled archived run ends at +162.893% drift. Thus the gate fails the old
tall behavior and passes its same-run uniform control.

Static and compilation verification is green: 189 unit tests, standalone
`tsc --noEmit`, targeted ESLint, `git diff --check`, and the production build
pass, and Dawn accepts the current tall-cell pipeline set.

No post-change physics numbers are claimed. The native Metal adapter silently
no-ops even a one-workgroup shader (`probe-minimal.ts` returns sum zero), while
OpenGL/Vulkan adapters are unavailable; the in-app browser independently fails
WebGPU initialization with an invalid external Instance. Runs in that state
leave both tall and uniform fields at initialization and are explicitly invalid.
After the host GPU is recovered, rerun the complete matrix in the mass-
oscillation plan, including the uncontrolled 10 s probe and:

```sh
FLUID_REFERENCE_VOLUME_SCALE=1.02 FLUID_SCENE=settled-tank \
  FLUID_METHOD=tall-cell FLUID_TARGET_S=10 FLUID_ENERGY_EVERY_STEPS=25 \
  FLUID_CPU_ORACLE=0 node --import tsx tools/run-webgpu-smoke.ts
```

Workstream 4 remains intentionally untouched until those live gates pass.

## 2026-07-17 (later): first live run collapsed volume — divergence-control sign was inverted for a level set

The first live run after the MAC-transport + CM12 volume-control migration
collapsed the dam's volume within a fraction of a second. Code audit (no live
GPU needed) localized three defects in the new `planSubsteps` /
`volumeCorrectionDivergence` pair; the implementation matched the plan, and the
first defect was the plan's own prescription.

1. **Inverted feedback sign.** With `rhs = rho*(div - c)/dt` an exact solve
   leaves `div_new = c`. CM12 Sec 3.7's convention (excess => `c > 0`) is
   correct for a *density* field, where continuity dilutes an overfull cell
   under positive divergence. A level set has no continuity coupling: the
   interface rides the flow, so `div_new > 0` moves the 0-contour outward and
   GROWS the enclosed volume. As implemented, both error directions were
   positive feedback, compounding per step until the rate clamp; the release
   transient's occupancy dip picked the collapse direction. The level-set sign
   is `c ∝ (V_ref - V)` (deficit => expansion), matching the FOA03/KLL*07
   interface volume-control family. Fixed in `planSubsteps`.
2. **Eta clamped before a ~20x amplification.** The rate was clamped to
   `±1/dt` globally and then multiplied per cell by
   `reference/interfaceCells` (~20 on the UI dam), so a saturated controller
   commanded ~2500/s of divergence per interface cell versus the uniform
   kernel's 30/s per-cell cap. `governor[7]` now stores the eta-clamped
   PER-CELL rate (`±30/s`) after distribution.
3. **Lambda expressed per frame dt instead of per 1/30 s.** `0.5/0.008 s` is
   ~4x stiffer than CM12's `lambda=0.5` per 1/30 s step (the uniform kernel's
   `min(0.5*excess,1.0)*30.0`). The rate now uses the `*30.0` convention.

With the per-cell 30/s cap, commanded post-projection divergence is at most
`0.24*dt^-1*dt = 0.24` dimensionless — below the 0.5 stability-flag threshold,
so the diagnostics gates need no special-casing. The plan's Sec 6.1 actuator
prescription is rewritten with the sign derivation so it is not re-inherited.
The step-response probe (`FLUID_REFERENCE_VOLUME_SCALE=1.02`, settled tank,
volume must move TOWARD the biased reference) catches an inverted sign in
seconds and must run before any soak. Live acceptance for the whole migration
(uncontrolled 10 s probe, dam-settling matrix) remains outstanding from the
previous entry.
