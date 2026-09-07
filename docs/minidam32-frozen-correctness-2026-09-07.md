# Minidam32: frozen topology and min1/max1

The current comparison is `minimal-power-dam-break-32`: A keeps its initial
adaptive topology frozen; B uses a whole-domain min1/max1 region. The completed
four-second experiment makes the entire tank resident at reset and freezes
both arms. Both use balanced coarse-first defaults and 1/30 s steps. The
earlier short captures below predate the complete-air setup.

## Performance checkpoint

The production-default mini32 profile completed 12 hardware-timestamped
samples after three warmups: GPU median 65.6671 ms and frame wall median
73.4953 ms. Face preparation took 22.6099 ms, pressure 12.0586 ms,
sharpening 9.5027 ms, and presentation 1.9005 ms. The 40 ms regression ceiling
was not changed. The user accepted approximately 66 ms for now and redirected
work to correctness; performance optimization is deferred.

## Fixed: a surface step present before simulation

Initial column mass is flat at a waterline of 29.44 finest cells (0.736 m).
Coarse presentation reproduced that height, but fine presentation gave
29.2966 cells. The adaptive surface therefore acquired a 3.585 mm step where
its initial top-surface cells changed width. Uniform min1/max1 had the same
fine reconstruction error everywhere, making the adaptive seam especially
visible.

`presentationPhiAt` gathers seven vertical density samples and passes them to
`presentationResolvedFineColumnPhi`. If *any* sample fell outside the tank,
it returned the point-density scalar instead. At this height, neighbouring
samples straddle that fallback condition: one uses the volume bracket, the
next uses the density scalar. Their zero crossing is consequently displaced.

The function now continues the boundary density sample for an out-of-domain
bracket coordinate. It still rejects internal solid/cut support. This uses the
existing seven-sample stencil, with no additional dispatch or symmetry
averaging. It does not change simulation density or velocity.

Validation:

- The new production-shader test first failed at height 28.5078125 cells.
  It now passes 516 interface-height cases, including an interior control and
  all phases of near-lid surfaces with at least one complete air cell.
- Existing sharp/refined/affine/detached-sheet waterline and native coarse
  volume-surface tests pass.
- The full scene's reset height error falls to 0.000225 mm, including f16
  publication quantization.
- Every density, velocity and pressure value is bit-identical before/after
  the presentation fix at all five captures from steps 0 through 4 in both
  arms. These short captures therefore isolate a presentation-only repair.

![Waterline comparison](../artifacts/minidam32-frozen/analysis/waterline.png)

## Frozen residency prevents the front from continuing

The initial sparse roster contains 60 active bricks. Four dry corner bricks,
`(3,y,3)` for y = 0,1,2,3, are backed but inactive. The current freeze option
freezes membership as well as resolution and disables swept-front activation.

At step 5 (0.1667 s), source cell 2232 at finest coordinate
(23.5, 0.5, 23.5) has nonzero density 2.2102e-5 and velocity approximately
(0.70374, -0.30146, 0.70374) m/s. Its forward characteristic reaches the
inactive diagonal corner and has no recipient. The fail-closed
`EMPTY_DEFICIT_STENCIL` assertion is correct; ignoring the donor would lose
mass. Freezing min1/max1 as an additional control produces the same support
failure. Min1/max1 with normal residency completes 30 steps.

The user approved making the authored tank resident once before freezing.
`initialAtlasResidentForQA` activates all initial atlas bricks during construction;
it preserves the original resolutions, adds no warm-up step, and changes no
physics kernel. In A this adds just 32 coarse air cells: 5,723 becomes 5,755.
All 64 original brick identities, coordinates, spans and resolutions match the
ordinary reset, including the four previously inactive bricks. B has 32,768
width-one cells. The same complete tank coverage removes residency as an A/B
confound.

The new frozen-domain integration regression compares ordinary and complete
reset density and solid openness exactly, verifies every initial cell size,
then advances 12 steps. Liquid enters the formerly missing corner without
any roster/rung change, non-finite value, or simulation fault. It passes.

This opt-in diagnostic setup is separate from the subsequent requested UI
freeze change, which allows new support while retaining accepted cell sizes.

## Four-second fixed-domain result

Both arms complete all 120 steps without a fault. Their full source hashes
match, settings match apart from the authored resolution bound, initial
column integrals match to 1.5e-9 m, and every frame retains its entire reset
roster and cell sizes. The initial centre-of-mass difference is 0.348 mm from
finite-volume projection; it is not an initial volume difference.

| At 4 seconds | A: frozen adaptive | B: frozen min1/max1 |
| --- | ---: | ---: |
| Total mass error | -0.06664% | -0.02485% |
| Centre of mass height | 219.34 mm | 165.86 mm |
| Collocated kinetic energy | 3.174 J | 50.985 J |
| Maximum x/z density error | 0.00820 | 0.50692 |
| Mean x/z density error | 0.0000386 | 0.01108 |
| Maximum x/z velocity error | 0.01449 m/s | 2.72131 m/s |

The A/B integrated column-depth RMS gap is 67.41 mm, with a 154.21 mm maximum.
Thus eliminating the missing-support halt does not resolve the physical
disagreement. Adaptive loses much more wave motion, while B retains larger
oscillations and develops substantial asymmetry. Neither is a certified
physical reference. Kinetic energy here integrates collocated diagnostic
velocity, not the pressure operator's native face energy norm.

Over the whole trajectory, A first exceeds the existing 0.02 m/s maximum
velocity symmetry tolerance at step 106 and the 0.01 density tolerance at
step 118; both mean-error limits remain satisfied. B exceeds those maximum
limits at steps 11 and 16, and its mean velocity/density limits at steps 19
and 22. A's final values returning below the maximum limits do not make its
whole trajectory a symmetry pass.

![Fixed-domain trajectories](../artifacts/minidam32-frozen/fixed-domain-analysis/trajectory.png)

![Published front and mass columns](../artifacts/minidam32-frozen/fixed-domain-analysis/fronts.png)

### Measured pressure-membership amplifier in B

A second 20-step capture records intermediate scalar fields and the actual
pressure membership/row theta buffers. Its complete trace is identical to
the first capture, proving that these diagnostic copies do not perturb it.

Pressure membership remains x/z symmetric through step 14. At step 15, six
cells (three mirrored pairs) receive different memberships. For example,
fine coordinates (30,23,0) and (0,23,30) have densities 0.4999964 and
0.5003051. `pressureCellMembershipFromDensity` classifies them on different
sides of `CM12_LIQUID_ISOVALUE` (0.5), so one pressure unknown and its incident
free-surface boundary differ despite the small density difference.

Maximum velocity asymmetry rises from 0.01810 m/s at step 14 to 0.08196 m/s
at step 15. At the next conservative-transport stage, maximum density
asymmetry becomes 0.14975 before diffusion/sharpening reduce it to 0.06410.
Pressure projection and transport then reinforce the divergence. This is
an identified amplifier, not the origin of all asymmetry: B already exceeds
the velocity tolerance at step 11 while pressure memberships still match.

The native-face audit covers all 95,232 internal fine-grid faces. On step one,
velocity extension, face preparation and gravity are exactly x/z symmetric;
projection introduces a maximum 2.68e-6 m/s difference. By step 11, face
preparation has 0.00594 m/s maximum error and projection raises it to
0.04094 m/s, while pressure memberships still match. This places the first
measured seed in the pressure solve/projection and shows amplification before
the later membership split. It does not justify removing floating-point noise
with a symmetry pass; the relevant target is sensitivity of the physical flux
to those small pressure/interface perturbations.

Do not repair this with mirrored decisions or a density tolerance chosen for
this scene. The next pressure test should perturb an interface across the
membership threshold and check continuity of the projected physical flux,
including the ghost-distance limit and the existing submerged-cell retention
rule. The companion resolution test should reconstruct the same geometric
interface on each rung and compare pressure forces before any advection.

## Remaining fluid and surface mechanisms

After the reset repair, adaptive waterline bands remain at 0.133 s. The
column-mass RMS difference between A and B is 30.30 mm at that point. The
corresponding relative mass errors are only -0.000265% and -0.000656%.
Total mass therefore does not certify the spatial evolution.

Raw coarse-cell plateaux alone are not a failure: the diagnostic publishes
finite-volume means. The relevant comparison is their integrated mass and
the reconstructed geometry against the same physical reference.

There is a concrete remaining pressure-boundary inconsistency.
`classifyPressureRow` uses `(0.5 - density) * cellWidth` as its interface
distance and interpolates it across liquid/air centres. A partial cell's
volume fraction can locate a flat interface, but a full cell's value has
already saturated and no longer gives distance to that interface. For the
same 29.44-cell planar height, this formula places the pressure zero at:

| Cell width | Pressure boundary (finest cells) | Error from actual plane |
| ---: | ---: | ---: |
| 4 | 29.1250 | -0.3150 |
| 2 | 29.6111 | +0.1711 |
| 1 | 29.3929 | -0.0471 |

These are evaluations of the current row formula on the initial horizontal
column, not a claim that this alone explains the full dam trajectory.
`pressurePlanarColumnHeight` already addresses this distinction for a narrow
hydrostatic case, but `pressureHasPartialRefinementRegion` gates its use on an
authored partial region. It is inactive in both arms here. Broadening that
region-dependent correction is not a satisfactory general repair.

The next numerical target is a local, geometry-consistent pressure interface
construction from volume fractions, tested across all supported cell widths,
interface phases, orientations and coarse/fine seams. It must preserve the
compatible pressure operator and handle disconnected/overturning liquid.
The initial velocity and first transported density should then be compared
before attributing later shape changes to sharpening or capacity repair.

![Field and surface comparison](../artifacts/minidam32-frozen/analysis/fields.png)

## Artifacts and reproduction

Captures live under `artifacts/minidam32-frozen/`: `adaptive` and
`adaptive-audit` retain the initial failure; `minmax1-resident` retains the
30-step uniform-width control; `adaptive-after` and `minmax1-after` are the
five matched snapshots after the waterline fix. Every capture records source
hashes and the resolved scene/settings. Initial column integrals match to
1.5e-9 m; initial vertical density profiles differ by finite-volume projection.

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_WEBGPU_BACKEND=metal \
POOL_SYMMETRY_SCENE=minimal-power-dam-break-32 \
POOL_SYMMETRY_FREEZE_TOPOLOGY=1 POOL_SYMMETRY_STEPS=4 \
POOL_SYMMETRY_CAPTURE_PRESENTATION=1 \
POOL_SYMMETRY_OUTPUT=artifacts/minidam32-frozen/adaptive-after \
node --import tsx tools/probe-pool-impact-symmetry-dawn.ts
```

For B, set `POOL_SYMMETRY_MAX_CELL=1`,
`POOL_SYMMETRY_FREEZE_TOPOLOGY=0`, and a separate output directory.
`tools/analyze-minidam32-frozen.py` checks unchanged simulation fields and
produces the comparison figures and numerical summary.

`full-resident-adaptive` and `full-resident-minmax1` contain the full four-second
comparison. `full-resident-minmax1-audit` contains the matching first 20 steps
with stage/membership/theta captures. `tools/analyze-minidam32-fixed-domain.py`
verifies their comparability and produces `fixed-domain-analysis/summary.json`
and both trajectory/front figures.

For the full fixed-domain experiment use `POOL_SYMMETRY_INITIAL_ATLAS_RESIDENT=1`,
`POOL_SYMMETRY_FREEZE_TOPOLOGY=1` and `POOL_SYMMETRY_STEPS=120` in **both** arms,
with maximum cell 0 for A and 1 for B and distinct output directories. Run
arms serially under the WebGPU lease.

## Regression status

The mandatory `npm run test:dawn:sparse-cm12` gate was rerun after the surface
change. It remains red: symmetric expansion fails its density symmetry
assertion; page-budget, hydrostatic, mini32 and performance lanes time out;
the 180-second suite budget prevents the tail from running. Failure-halting,
mixed-ratio topology, clipped transfer, generation storage, and the min8
region surface lane pass. No assertion, time limit or budget was relaxed.
The complete receipt is `artifacts/minidam32-frozen/boundary-waterline-gate.json`.
The targeted three surface tests and 16 source/failure/manifest tests pass.

The later integrated dynamic-freeze/support repair gate is recorded in
`artifacts/minidam32-frozen/dynamic-freeze-final-gate.json` (2026-09-07).
Four-second mini32 correctness now completes and passes in 16.2 seconds;
failure-halting, mixed-ratio topology, clipped transfer, generation storage and
the min8 region surface lane also pass. Symmetric expansion still fails its
horizontal density assertion. Page-budget and hydrostatic lanes time out at
30 seconds. Mini32 performance is 59.7688 ms against the unchanged 40 ms
ceiling; mini64 performance runs out of the remaining suite time, leaving six
tail lanes unrun. The canonical gate remains red. Its full log is
`/tmp/fluid-dynamic-freeze-final-gate.log`.

The four-second A/B captures above predate the later support/freeze and
signed-world presentation changes. They remain mechanism evidence for their
recorded source hashes, rather than a fresh trajectory validation of the
integrated checkout. Focused freeze/growth/live-edit and signed-field checks
are documented in `docs/sparse-cm12-dynamic-topology-freeze-2026-09-07.md`.
