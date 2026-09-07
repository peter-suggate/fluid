# Coarse-cell grid imprint: density-first investigation

A stationary manufactured bowl confirms a publication defect independently of
fluid evolution. It does **not** establish that publication is the sole cause
of the evolving pool screenshot. A separate gravity-on control also develops
native-density differences between resolutions. No solver or renderer fix was
made in this investigation.

## Evidence and exclusions

![Density and published surface](../artifacts/coarse-surface-grid-imprint/density-vs-surface.png)

Raw coarse density must be interpreted as cell-volume averages. A blocky density
image alone is not evidence of erroneous physics. The independent density check
integrates the analytical bowl over each native column footprint; a separate
64×64 quadrature checks native cell fill.

| Candidate | Controlled result | Scope of exclusion |
| --- | --- | --- |
| Initial density / shape rasterization | Width-4 column height error ≤0.000334 mm; native fill error ≤1.26e-6 against independent finer quadrature | Cannot account for the static millimetre-scale surface error |
| Pressure, gravity, transport | Zero gravity and zero initial velocity; accepted density byte-identical before/after; final velocity exactly zero | Not required to produce the static ripple |
| Gamma diffusion / sharpening | Disabled for static isolation; separate six-step enabled control changes no density | Not the source in these stationary fixtures; not a general exclusion in moving liquid |
| Refinement, restriction, remapping | Actual accepted topology verified unchanged throughout each run | No topology transition is required; the mixed-seam fixture adds a further seam error |
| Rollback / stale accepted state | Frame fault 0; every requested frame committed; published field follows injected bowl | No rejected frame in these captures |
| Page/cache addressing and boundaries | Complete width-4 publication agrees with isolated production volume kernel within 0.00551 mm in the interior ROI | These are not needed for the main periodic pattern; not a universal boundary audit |
| Binary16 quantization | Isolated float32 kernel retains the ripple; binary16 changes crossing by ≤0.00639 mm | Too small to explain the pattern |
| Meshing, shading normals, lighting | Pattern measured in published scalar zero crossings before invoking any of these | Cannot be the origin of this already-present geometry error; additional visual amplification remains possible |
| Evolving physical state | Gravity control reaches 1.74462 mm RMS coarse-vs-restricted-fine native column difference by 0.1 s | **Not ruled out for the original moving scene** |

Presentation fault-record QA returns unavailable (`null`) in this configuration;
that is not recorded as a zero fault receipt. Frame-control and topology receipts
are available and were checked. Full receipts and shader hashes accompany every
arm. Density diagnostics, not cached volume telemetry, supply the mass evidence.

## The proposed A/B scene

A 2.4 × 1.6 × 2.0 m closed tank, finest spacing 0.05 m, with a shallow smooth
bowl-shaped free surface. In finest-cell coordinates:

```
H(x,z) = 17.3 + 0.003 ((x-24)^2 + 0.7 (z-20)^2)
```

The unequal horizontal dimensions avoid inheriting the square-tank symmetry
authority. All arms restrict the same 8×8 area quadrature per finest column.
There is no scene-specific per-resolution rasterization. Gravity, surface
tension, viscosity and initial velocity are zero; diffusion/sharpening are off;
one 1e-8-second step triggers ordinary production publication. Topology is frozen.
The ROI excludes eight finest cells at each horizontal wall.

- **A:** adaptive-mass with whole-domain min1/max1.
- **B:** the same geometry on fixed width 4. This isolates the coarse path.
- Supporting arms: fixed width 4 flat and tilted planes, a frozen width-2/4
  seam, and ordinary coarse-first startup topology (all width 8 here).

The last arm uses the ordinary coarse-first layout selected for the authored
flat tank, then injects the manufactured bowl and freezes it. It is a controlled
publication test, **not** proof that adaptive policy would choose exactly that
layout for an evolved bowl. Fixed coarse arms prevent a future fix from passing
simply by refining everything.

The stationary bowl is deliberately informative at coarse resolution: its cell
averages contain its low-order curvature. The test allows a mean height offset;
it rejects recurring flat panels and sharp bends. Flattening the whole bowl
also fails. A smooth broad approximation can pass without fine/coarse pixel parity.

## Measured static behavior

`curvature` below is the second height difference divided by the finest spacing;
it is dimensionless, with analytical value 0.006 everywhere along x.

| Arm | Height error RMS (mm) | Curvature error RMS |
| --- | ---: | ---: |
| min1/max1 | 0.02088 | 0.0001214 |
| width 4 | 1.05731 | 0.0060059 |
| width 2/4 seam | 0.77592 | 0.0061052 |
| coarse-first startup, frozen width 8 | 4.18599 | 0.0103961 |

Width-4 curvature alternates near 0 and 0.012, instead of remaining at 0.006.
The isolated float32 production kernel gives curvature-error RMS 0.00599999,
so the pattern survives without half-float storage, page lookup or meshing.

[Derivative plots](../artifacts/coarse-surface-grid-imprint/derivatives.png),
[native density checks](../artifacts/coarse-surface-grid-imprint/density-check.png),
[measurements](../artifacts/coarse-surface-grid-imprint/comparison.json), and
[density/kernel receipts](../artifacts/coarse-surface-grid-imprint/density-evidence.json).

The active `presentationInterpolatedVolumePhi` interpolates volume-derived
native nodes trilinearly. That gives continuous values but piecewise slopes.
Also, a curved surface's column **average** is not its height at the column
centre. The resulting height bias depends on cell width; it therefore changes
at a resolution seam. A continuously interpolated scalar alone is not a
sufficient no-grid-imprint contract. The common-height reconstruction module
exists but `SPARSE_CM12_COMMON_HEIGHT_ENABLED` is false in this snapshot.

## Separate physical-state controls

The same bowl is evolved for six 1/60-second steps on frozen width-1 and width-4
grids. Every frame's density, velocity and pressure are saved.

1. Gravity 9.81 m/s², conditioning off: densities still match under restriction
   after step 1. Native density diverges starting at step 2. By step 6 the
   equivalent-column-height difference is 1.74462 mm RMS, maximum 2.79716 mm.
   Final relative pressure residuals are 0.000770 and 0.000186.
2. Gravity zero, conditioning on: density stays byte-identical to its initial
   state for both arms. Tiny velocities remain; relative pressure residuals are
   near one for this almost-zero forcing case, so this is **not** claimed as a
   general pressure-convergence success.

The physical difference is measured after restricting evolved minmax1 density
to the same width-4 volumes, not by comparing a fine point surface to coarse
block averages. It cannot be caused by meshing or lighting. However, divergence
between resolutions alone does not establish a grid artifact or identify
pressure versus transport; it can include ordinary resolution error. The
original pool's physical evolution therefore remains an open candidate.

[Physical density comparison](../artifacts/coarse-surface-grid-imprint/density-evolution.png)
and [per-step measurements](../artifacts/coarse-surface-grid-imprint/density-evolution.json).

Before treating a reconstruction change as the complete fix, use these saved
first-step velocity/pressure and second-step density states to isolate the
restoring-pressure response from liquid flux transport. Retain the static
publication gate as an independent requirement. Existing gravity-wave findings
in `docs/sparse-cm12-gravity-wave-investigation-2026-09-06.md` are leads, not proof
that those historical causes explain this particular screenshot.

## Re-run and acceptance

```
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
  node --import tsx --test --test-concurrency=1 \
  tests/sparse-cm12-surface-grid-imprint-dawn.test.ts

node --import tsx tools/probe-coarse-surface-grid-imprint-dawn.ts \
  --arm=fixed4 --profile=bowl --wait

node --import tsx tools/probe-coarse-surface-grid-imprint-dawn.ts \
  --arm=fixed4 --profile=bowl --gravity=9.81 --dt=0.016666666666666666 \
  --steps=6 --conditioning=off --output=artifacts/coarse-surface-grid-imprint/gravity-fixed4 --wait
```

Repeat the final command with `--arm=fixed1` and a distinct output for the
physical A/B; use `--gravity=0 --conditioning=on` for conditioning controls.
Python analysis tools require NumPy and Matplotlib. They read saved numerical
fields and do not alter them.

The new explicit test is intentionally **red** for curved coarse surfaces:
flat/tilted controls pass; the curved gate fails for fixed4, mixed and coarse-first
startup. The curvature budget is 25% of the known analytic curvature, comfortably
above fine-control quantization noise. It was chosen before the coarse captures.
This is a new proposed contract, not a weakened existing threshold. It is not
added to the canonical short suite while known failing.

Production source changed concurrently, so final matched captures use
`/tmp/fluid-grid-imprint-20260907`; the resident WGSL SHA256 is
`209d3b64ceddde676d45130e3b54f256aa6e8a34f707c38e8385b492aae122e5`.
[Source receipt](../artifacts/coarse-surface-grid-imprint/source-receipt.json)
records the other pinned files. Actual compiled shader hashes are in every
`config.json`. Static tests used Node 25.8.1; dynamic controls used Node 22.22.1,
with a matched runtime within each pair and the same pinned source. All Dawn
processes acquired the repository-wide exclusive GPU lease. Earlier working-tree
captures included a native process crash and fixture-development errors; those
are not the final gate evidence.

[Final Dawn gate log](../artifacts/coarse-surface-grid-imprint/dawn-gate.log).
Only diagnostic tools, this report and the new test were added. No production
fix or large simulation change was made, so the full canonical regression suite
was not rerun for this investigation.

Targeted ESLint passes and both Python analysis files compile. Repository-wide
TypeScript checking still reports existing errors; none name the new diagnostic
files. All ten saved frame receipts pass the final fault/commit-count audit.
