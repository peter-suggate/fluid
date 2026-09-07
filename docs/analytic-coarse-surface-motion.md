# Tiny coarse-surface motion controls

The scene picker contains five presets on the **Analytic motion** shelf:

- **Coarse surface · translation** (`coarse-surface-translation`): zero gravity,
  initial velocity `(0, -0.4, 0)` m/s, duration 0.5 s.
- **Coarse surface · free fall** (`coarse-surface-free-fall`): initial rest,
  gravity `(0, -9.81, 0)` m/s², duration 0.3 s.
- **Coarse surface · free fall with refinement** (`coarse-surface-free-fall-rerung`):
  the same body; region widths swap from 8/4 to 4/8 at 0.1 s and back at 0.2 s.
- **Coarse surface · standing wave** (`coarse-surface-standing-wave`): a fixed
  B2/B4 seam at x = -0.4 m in a gently sloshing tank.
- **Coarse surface · standing wave, live refinement** (`coarse-surface-standing-wave-live`):
  the same initial wave with no enforced refinement regions.

All use the UI's production adaptive-mass solver and coarse-first selector.
For the first three controls, the 32 × 32 × 8 finest-cell domain has 0.05 m cells. A 0.8 × 0.4 × 0.4 m
rectangle starts with its bottom at 0.8 m and centre at 1 m. It spans the depth
between free-slip walls; there is no viscosity, surface tension, inflow, terrain,
or rigid body. Volume is exactly 0.128 m³. The analytic trajectories end before
floor impact.

Two refinement regions hold the left half at width 8 (B1) and the right half at
width 4 (B2). Initially just two bricks contain liquid, with 1 + 8 accepted wet
cells. Their common horizontal surface moves downward along the x=0 resolution
seam. Air/support cells add to the total simulation workload. The first two are
fixed resolution-pattern controls. The third applies timed region edits through the ordinary live scene-policy path. Accepted swaps were
observed on steps 7 and 13 (the first step starting at each keyframe). In the
first two scenes, editing the regions changes the control; in the third, the
authored keyframes reapply their bounds while the timeline advances.

## Analytic answer and measurements

With upward-positive y, the exact motion is

    centre_y(t) = 1 + v0*t + g*t*t/2
    velocity_y(t) = v0 + g*t

Pressure relative to ambient and lateral velocity should be zero. Shape and
volume should be unchanged. The implementation transports with the old velocity
before applying gravity. Its corresponding discrete trajectory is

    centre_y(n) = 1 + v0*n*dt + g*dt*dt*n*(n-1)/2

The probe reports both continuous and discrete position error, mass error,
mass-weighted vertical velocity, lateral RMS velocity, the difference between
left/right centres, and density L1 error. The density oracle integrates the
translated analytic box over **the actual accepted control volumes**, using the
discrete displacement. Comparing replicated coarse cell averages to sharp
finest-cell samples would incorrectly count representation error as transport
error. L1 is normalized by the initial liquid volume; it can exceed 100%.
Surface rungs are measured on the topmost occupied cells of liquid columns, and
per-rung surface speeds verify that both resolutions move.

## Running

Unload Fluid browser tabs first. The probe acquires the repository WebGPU lease;
never run it concurrently with another Dawn process or Fluid in the browser.

```bash
npm run probe:adaptive-mass:analytic-motion
npm run probe:adaptive-mass:analytic-motion -- --verify
```

The first command is a diagnostic capture, returning all trajectories and budget
violations in `artifacts/analytic-motion/mixed.json`. It still fails for GPU
health, non-finite fields, bad initial state, or missing encoded steps.
`--verify` additionally fails on any analytic budget violation. These are new
investigation budgets, not replacements for the canonical regression gate:

- mass error: 0.01%;
- centre error against the discrete trajectory: 0.025 m;
- vertical velocity error: 0.02 m/s;
- lateral RMS velocity: 0.01 m/s;
- coarse/fine centre difference: 0.025 m;
- control-volume density L1 error: 10%;
- two surface resolutions, each moving after the initial frame.

## Initial observations, 2026-09-07

Both scenes stay healthy and retain B1/B2 surfaces throughout. They already
expose numerical errors, so strict verification is expected to fail on this
working tree. No physics threshold was relaxed to make them pass.

| Final measurement | Translation, 0.5 s | Free fall, 0.3 s |
| --- | ---: | ---: |
| Mass error | 0.00645% | 0.00285% |
| Mean vertical speed, observed | 0.39087 m/s | 2.66582 m/s |
| Mean vertical speed, analytic | 0.40000 m/s | 2.94300 m/s |
| Centre error beyond discrete integration | 6.30 mm | 22.47 mm |
| Left/right centre difference | 5.46 mm | 32.90 mm |
| Control-volume density L1 error | 53.34% | 100.91% |

Translation first exceeds the 10% shape budget at step 13; free fall at step 9,
and its velocity error exceeds 0.02 m/s at step 11. The free-fall continuous
position error is 46.99 mm, of which 24.525 mm is the known integration offset.
These measurements establish small repros; they do not identify the faulty
stage or separate ordinary coarse-grid diffusion from a mixed-seam defect.
A same-solver all-fine control and stage-by-stage receipts are the next diagnostic
steps, outside these first two scene additions.


## Standing-wave controls

The wave tank is 32 × 24 × 8 finest cells, with length L = 1.6 m, mean liquid
height H = 0.6 m, and initial amplitude A = 0.03 m. The initial height is
H + A cos(π(x + L/2)/L), with zero velocity. Cell volumes use the existing
8×8 horizontal quadrature, preserving zero mean perturbation. Cosine extrema
are included in sparse-domain bounds; the same height field is saved in the
scene document and its edits reseed the simulation.

The linear inviscid prediction is A cos(kx) cos(ωt), with k = π/L and
ω² = gk tanh(kH). The period is 1.5744065 s; both scenes run two periods.
A/H = 0.05 and kA ≈ 0.059, so this is a small-amplitude approximation rather
than an exact nonlinear solution. [MIT dispersion relation](https://web.mit.edu/fluids-modules/www/potential_flows/LecturesHTML/lec19bu/node4.html).

The fixed scene uses B2 on the first quarter and B4 elsewhere, so the seam is
away from both the wall and the stationary central node. The live scene uses
curvature tolerance 0.05, starts at B2/B4, and has no refinement regions. Its
initial run had 100 frames with two moving surface rungs, 16 brick promotions,
and 2 demotions. Eventually its surface became entirely B8; that is recorded
as selector/physics behaviour, not hidden by a minimum mixed-coverage claim.
The fixed scene retained mixed moving surface for all 189 advancing frames.
Both stayed healthy through 3.15 s. Neither matches the ideal wave accurately.

```bash
npm run probe:adaptive-mass:standing-wave
npm run probe:adaptive-mass:standing-wave -- --fine --mode=fixed
npm run probe:adaptive-mass:analytic-motion -- --fine
npm run probe:adaptive-mass:analytic-motion -- --rerung
```

Wave receipts report integrated column heights, projection onto the fundamental
cosine mode, analytic amplitude, height RMS error, moving surface rungs and
accepted promotions/demotions. Initial height RMS includes the native
coarse-cell representation error (about 1.2–1.35 mm); it is not zeroed or
presented as transport error. Partial data are saved even if a wave halts.
All-fine controls use the same Sparse CM12 solver with min1/max1 everywhere.

## First diagnosed defect: loss of transport seeds

The all-fine controls preserve mean velocity to approximately 10⁻⁶ m/s.
Their final position errors beyond the explicit integration offset are 0.99 mm
for translation and 0.22 mm for free fall. They still have density L1 errors
of 34.36% and 25.00%, so ordinary transport diffusion is separate from the
coarse velocity failure.

The rerung scene exposes a sharp failure. At step 17 its maximum cell density
is 0.47387: every cell is below the 0.5 liquid isovalue, although nearly all
initial mass remains. On step 18 its mean velocity changes from -2.53568 m/s
to -0.14917 m/s. There is no region swap at that step; the swaps happened on
steps 7 and 13. Velocity-extension initialization seeds only cells with rho >
0.5, so it has no seeds and publishes zero transport velocity.

The new `tests/sparse-cm12-subisovalue-motion-dawn.test.ts` reduces this to a
0.1 m thick moving slab represented by quarter-full width-8 cells. With no
gravity, its first step changes mean velocity from -0.4 m/s to zero. The test
is intentionally red and is not part of the canonical short gate:

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test \
  tests/sparse-cm12-subisovalue-motion-dawn.test.ts
```

An experiment seeded all cells above the transport dry threshold. It made the
new test pass, but worsened the mixed rerung scene: final discrete centre error
increased from 67.1 mm to 108.8 mm and lateral RMS speed rose from zero to
0.023 m/s. The experiment was reverted. Its receipt remains in
`artifacts/analytic-motion/rerung-positive-mass-experiment.json`.

**Next correction:** separate conserved material momentum from velocities
extrapolated for free-surface/pressure support. Trace the dilute-cell velocity
through face preparation and collocation; preserve the force-free momentum
invariant without feeding unreliable dilute velocities back into wet flow.
The lower-threshold experiment is not a shipped physics fix.

## Validation of this ladder addition

- UI production build: passed.
- Focused scene, serialization, initialization and runtime-control tests: 12 passed.
- Both waves: 189 healthy advancing frames; timed free fall: 18 healthy frames,
  with accepted swaps asserted on steps 7 and 13.
- Fully fine wave: also completes two periods, but is not an exact wave oracle.
  At 3.15 s its fundamental amplitude is 30.58 mm versus the linear 30.00 mm,
  while total column-height RMS error is 25.53 mm. Mode amplitude alone hides
  higher-mode/shape errors; both metrics must remain in the comparison.
- New sub-isovalue momentum regression: fails at step 1 on production (mean
  speed becomes zero), passes the rejected lower-threshold experiment.
- Canonical Dawn gate: 5 lanes passed; symmetry failed and five other lanes
  timed out, exhausting the 180 s budget before the last six lanes. No timing
  ceiling or physics threshold was changed. See the saved regression log.
- Repository type check still reports 49 errors, with none in the new scene,
  height-field, keyframe, or ladder-probe code.

Logs and numerical receipts are under `artifacts/analytic-motion/`; source
hashes are recorded in `source-receipt.json`. Other work was present in the
shared checkout, so these results describe that working tree rather than an
isolated clean-HEAD benchmark.

## Surface-seam investigation (2026-09-07 continuation)

The investigation now uses forced-fine min=max=1 as the reference. No UI scene,
analytic threshold, CM12 liquid isovalue, or transport equation was adjusted.
`tools/probe-analytic-topology-stages-dawn.ts` captures native fields after
production stages and verifies captured mass/momentum against published
world diagnostics. It must use `sceneDocument`: calling the raw scene factory
omits the stage's solid boundaries and gives a different physical experiment.

A manufactured affine-pressure patch exposed an error in the mixed free-surface
coefficient. The old unsigned liquid/air distance averages cannot cancel a
pressure gradient tangent to the free surface. A horizontal hydrostatic patch
produced a horizontal gradient of 0.0833333 instead of zero. The correction uses
the signed stencil factor from Ando & Batty 2020, equations 21 and 25, preserving
the existing CM12 theta minimum. Zero-response faces retain their current-flux
contribution to the pressure RHS. The production classification/gradient patch
test covers submerged, tangential-cut, and inclined interfaces. This is an
adaptive interface correction, not a change to CM12 transport or sharpening.
See `docs/papers/ando-batty-2020-octree-liquid.txt`, section 4.5.

The fixed and live standing-wave trajectories were unchanged by that correction;
they remain unsatisfactory. The canonical Dawn gate again produced the existing
D4 symmetry failure and topology/hydrostatic/mini32/mini64 timeouts, exhausting
the 180-second budget. No gate was weakened.

The free-fall native capture showed that cells away from the floor retain the
expected velocity. At 0.3 seconds the offending floor cells have exactly half
the expected velocity (averaging the stationary floor and the moving upper
face). A uniform width-4 control also exhibits the loss, isolating this part
from 2:1 seams. Density diffusion reaches the floor before the analytic body.

The `--surface` option adds independent zero-contour measurements from the
actual published sparse field; raw density L1 remains reported separately.
Final forced-fine translation has 6.43 mm top / 3.31 mm bottom RMS error and
1.82% rendered-volume error. Free fall has 2.11 mm / 4.73 mm RMS error and
1.58% rendered-volume error. This confirms that raw tracking-density diffusion
is not equivalent to visible shape loss.

The added diagnostic `--seam-width=2` keeps the same body, motion and time step,
with fixed width-2 / width-1 regions. It does not edit the UI presets. Final
translation velocity error is 0.00000571 m/s, centroid error 1.85 mm, and rendered
surface RMS about 17.4 mm. Free fall has 0.005575 m/s velocity error, 0.661 mm
centroid error, and about 22.5 mm surface RMS. The original width-8 / width-4
cases still have severe presentation errors, including a detached body appearing
as a floor-connected pool despite its suspended mass centroid.

A trial adding continuous interior-connectivity ratios to the coarse column
reconstruction was rejected and reverted: it improved translation position but
worsened rendered volume and free fall. The detached-presentation regression
records the remaining invalid bracket behavior; it is intentionally red in Dawn
and is not part of the canonical short gate. Do not treat it as repaired.

Browser checks at the analytic endpoints confirm coherent, suspended width-2 /
width-1 bodies at 0.5 s translation and 0.3 s free fall, with visible rounding and
stepping. The original width-8 / width-4 free fall instead renders a floor mound
at 0.3 s, matching the published-field diagnostic rather than the mass centroid.

Two pressure-geometry experiments were rejected and reverted:

- Removing the existing authored-region planar-height override improved the
  one-second wave RMS from 17.80 to 13.72 mm, but at 3.15 s worsened RMS from
  33.69 to 48.18 mm and reversed the fundamental mode (-26.88 versus +30.00 mm).
  Its flatness proof samples finest-coordinate neighbours, which can lie in the
  same coarse cell; this is not a reliable proof of a flat native-cell patch.
- Using integrated column height for each pressure stencil term without the
  authored-region condition also failed the full trajectory: 38.27 mm RMS and
  6.72 mm fundamental amplitude. It is not a shipped correction.

These receipts are saved as `wave-mixed-no-authored-pressure-full.json` and
`wave-mixed-column-geometry-full.json`. Neither experiment changes the conclusion
that pressure-interface geometry needs a stronger local correctness test before
further whole-scene trials.

A floor-apron presentation experiment replaced binary full-cell continuation
(`floorDensity > 1e-6` becomes 1) with the actual floor density in both publication
and its representability proof. Translation's top position improved, but its
rendered-volume deficit grew to 53.67%; free fall lost the zero contour entirely.
The experiment was rejected and reverted. Its receipt is
`mixed-rejected-floor-continuation.json`; `mixed.json` again contains the current
baseline. Removing fictitious floor support alone does not recover the diffuse
coarse body; this needs a complete reconstruction correction, not a cutoff edit.

Final verification after restoring all rejected trials: the production affine
2:1 pressure patch passes in Dawn. Type checking reports the same 49 unrelated
errors, with none in the ladder/probe additions. The previously recorded canonical
Dawn failures still apply; the ladder is not declared passing.

## First-divergence isolation: moving width-2 / width-1 seam

The next experiment holds the physical box, velocity, time step and production
profile fixed, and compares each mixed half with its matching uniform control:
width 2 on the left, width 1 on the right. The original width-8 / width-4 UI
presets remain stress cases. No analytic gate was relaxed.

`probe-analytic-topology-stages-dawn.ts --seam-width=2` now captures the initial
state as well as stage boundaries. It also captures gamma and the diffusion
scratch output; reading only the destination density would incorrectly attribute
a gamma-diffusion change to sharpening. `--label` separates experiment receipts,
and each new run records its scene and simulation-source hashes.
`tools/analyze-analytic-seam-stages.py` matches physical native-cell centres and
widths, comparing each mixed half against the corresponding uniform run.

Observed sequence on the unmodified sharpening stencil:

- Step 1: constant velocity is preserved exactly at every captured stage.
  Transport agrees with analytic cell averages within 1.1e-8. Mixed/control
  density differences are at most 1.5e-8.
- Step 2 through transport: the maximum mixed/control density difference is
  5.96e-8; gamma is exactly 1 and gamma diffusion leaves density unchanged.
- Step 2 sharpening: the maximum density difference becomes 1.2396e-4, with the
  largest differences adjacent to x=0.8 m, the refinement seam.
- Pressure subsequently produces a maximum mixed/control cell-velocity
  difference of 1.6785e-5 m/s. It is downstream of the first scalar discrepancy.

Intermediate destination cell velocities are not necessarily the final velocities
for density newly placed by sharpening; the final collocation must be checked
before interpreting those scratch values as a physical loss of momentum.

A separate manufactured affine-density patch identifies an exact stencil error:
`sharpeningStats` subtracts coarse/fine cell-centre values and divides by their
normal separation, despite the centres also being displaced tangentially. For
rho = 0.3 + 0.1 y, a fine cell sees a false x derivative of 0.03333334 instead of
zero. This failure is independent of the sharpening weight, pressure solve,
transport, or any visual threshold.

The correction samples the existing geometric density interpolant at the
neighbour's normal coordinate and the current cell's tangential coordinates.
This preserves the CM12 upwind formula while correcting its adaptive sample
geometry. Uniform rows retain their existing arithmetic. The manufactured test
injects an analytic interpolation oracle to isolate this geometry; the existing
transport-stencil tests separately check the actual interpolant's affine property.

The moving comparison is not declared passing because of this local correction.
At step 6 the maximum mixed/control density difference decreases from 0.0012660
to 0.0009483, but its integrated absolute difference increases from 1.4639e-5 to
1.5618e-5 m3. Different native resolutions also have different sharpening doses
and trace reaches; equality with piecewise uniform controls is a localization
measurement, not an exact invariant for an arbitrary nonlinear density profile.

Validation of the sharpening geometry correction:

- Four focused Dawn tests pass: 36 affine sharpening configurations (six fields,
  both seam orientations, three axes), affine pressure, reflected transport
  interpolation, and graded 1/2/4/8 transport junctions including clipped geometry.
- The canonical gate again passes five lanes and retains the symmetry failure
  and topology/hydrostatic/mini32/mini64 timeouts, exhausting its 180 s budget.
  Log: `/tmp/fluid-sharpen-seam-regression.log`. No ceiling was changed.
- Type checking still reports 49 errors outside these additions.
- Width-2 / width-1 translation at 0.5 s: centroid error 1.886 mm, velocity error
  0.00000570 m/s, left/right centroid difference 1.369 mm (previously 1.498 mm),
  rendered-volume error 0.455% (previously 0.489%). Top/bottom RMS remains about
  17.4 mm. Browser inspection confirms a coherent suspended body with rounded
  edges at the verified 0.5000 s endpoint.
- Free fall at 0.3 s: centroid error 0.675 mm, velocity error 0.005576 m/s,
  left/right centroid difference 0.796 mm (previously 0.879 mm), rendered-volume
  error 2.182% (previously 2.204%). These are small changes, not resolution of
  the full ladder's known failures.

The corrected width-2 / width-1 free-fall body was also inspected in the browser
at the verified 0.3000 s endpoint: it remains suspended and coherent, with the
same visible edge rounding. The browser was unloaded before releasing the GPU
lease. These visual checks do not cover or certify the original width-8 / width-4
stress cases.
