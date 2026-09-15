# Figure 7: falling-front compression and coarse-core loss

## Status

Prepared dynamic pages now carry B1/B2/B4/B8 storage and all 84 valid
face/rung variants. New pages enter at the coarsest backed 2:1 rung. Certified
bulk coarsens immediately, and coarse-first grading/publication uses one atomic
plan so neighbours can coarsen together. The older early-compression correction
below is preserved.

The Figure 7 acceptance probe checks every frame through floor contact (frame
25 at dt=1/30), including four-spacing liquid cells near the centre, mass,
pre-contact shape, and complete liquid/phi coverage within radius 12 of the
moving centre. The full suite remains waived for this investigation.

## Early compression: stale air velocity after projection

`collocateAndDiagnose` publishes the current projected liquid velocity into the
effective transport plane. Air samples still contain the pre-projection
extension. The subsequent extension was gated on `activity[26]`, the projected
receiver-admission count. Frames without admission therefore traced receiver
boxes through a field mixing current liquid velocity and previous air velocity.
Under gravity this is a compressive velocity gradient at the falling front.
Donor normalization conserves mass, so it manifests as over-capacity receivers;
pressure subsequently responds to this artificial excess and deforms the sphere.

Removed the topology-admission gate from post-projection velocity extension.
The existing packet schedule is reused when topology is unchanged. No topology
rebuild, extra search distance, coupling-balancing rounds, or repair is added.
The existing initialization and eight extension sweeps now execute on frames
where the old gate suppressed them. Their cost needs a dedicated stage timing;
the diagnostic tool's frame wall times include mandatory readbacks and are not
an isolated GPU cost measurement.

Dawn at dt=1/30, production Figure 7 settings:

| Frame | Before excess, fine-cell volumes | After excess | Before variance x / y | After variance x / y |
| --- | ---: | ---: | --- | --- |
| 2 | 43.561 | 0 | 80.01 / 80.27 | 80.00 / 80.35 |
| 3 | 117.079 | <0.000001 | 80.04 / 80.35 | 79.96 / 80.48 |
| 4 | 171.199 | <0.000001 | 79.96 / 80.30 | 79.87 / 80.51 |
| 10 | 1034.741 | <0.001 | 83.65 / 74.40 | 79.77 / 81.27 |
| 20 | 4740.258 | 78.564 | 105.69 / 43.61 | 79.3 / 81.7 |

Variance is the density-weighted second central moment in finest-cell units,
including within-cell quadrature. A radius-20 uniform sphere has variance 80
along each axis. After the change the centre follows semi-implicit Euler free
fall to within 0.05 finest cells through frame 20. At frame 4 the wet velocity
range shrinks from [-1.4374, -0.9089] to approximately uniform -1.333333 m/s.

A separate, correctly configured sharpening-off run also has only roundoff
excess through frame 4. The initial attempted off comparison did not apply its
CLI flag and must not be used as evidence. The probe now applies that flag.

The 30-frame diagnostic reaches floor impact without a validation error.
Later excess is still present; this change does not establish exact transport
or resolve all impact behaviour.

## Prepared multi-rung admission

The old dynamic format had only 512 B8 cells per page. It also forced adjacent
host pages to B8 in `enforceGeometricDynamicSeamFloor`, including after the
planner requested coarse bulk. Both ordinary and projected-flow admission now
use the same coarse grading rule. The B8-only enforcement/compilation-request
path is removed.

Each page reserves 585 cells, 2010 rows and 5550 term slots. All geometry and
coefficients are prepared at initialization. The runtime selects a prepared
variant and relocates its cell identities; it does not fit or generate topology.
All compatible 1:1 and 2:1 rung pairs, absent neighbours and six directions are
covered. A mixed CM12 face retains its single coarse-area, five-term row.
Accepted and candidate rung cells do not alias. Missing prepared faces are
terminal, including on newly admitted dry pages.

The dynamic topology image is 28,216 words per reserved page (112,864 bytes).
Cell storage grows by 14.3%; row storage by 16.3%. There is an additional
immutable seam catalogue. These are up-front memory costs. Runtime binding
visits prepared boundary records; it adds no iterative connectivity repair.
A dedicated GPU timing comparison is still needed; diagnostic readback wall
times are not isolated solver timings.

The catalogue currently covers ordinary eight-spacing pages. A required seam
to a macro host is refused before publication; this change does not add a
macro-page topology compiler or silently rebuild the resident.

## Coarse recovery during translation

The accepted-cell submerged-ball receipt already proves the absence of a
surface in bulk. Applying surface settling history delayed recovery until the
sphere had moved off the page. Certified bulk now requests its coarse level
immediately. Coarse-first transactions publish the complete graded plan;
grading can therefore use neighbours' requested rungs. Budgeted legacy
transactions retain their accepted-neighbour constraint.

B2 means two cells across an eight-spacing page: cells four finest spacings
wide. Read `acceptedResolution`, not the original authored `resolution`.

## Late dry-support failure

A moving far-air vertex consumes its signed phase clearance. Once exhausted,
a departure beyond the sparse source domain could neither sample phi nor
extend a usable clearance, and correctly halted. The sparse source plan
provides an independent proof for genuinely unrepresented air. A bounded
check of at most eight adjacent finest cubes returns the radius of an empty
ball (at most half a spacing). Only failed sampling from an already deep-air
vertex with valid velocities can use it. A represented page supplies no such
proof, so missing phi backing still fails closed. No search radius, redistance
rounds, or topology repair is added.

## Apparent missing interior pages in the slice

The grid-overlay reader independently assumed a 512-cell dynamic stride and
rejected every dynamic rung below B8. The fluid and published phi remained
present, but the slice showed holes. The presentation ABI now publishes the
dynamic stride and B1/B2/B4/B8 offsets; the slice consumes those addresses.
Legacy single-rung consumers keep their original layout. The native reader
fixture checks all four rungs, a nonzero page index, and legacy addressing.

## Focused verification

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
node --import tsx tools/probe-figure7-deformation-dawn.ts \
  --steps=25 --verify-coarse-floor \
  --output=artifacts/level-set-volume/figure7-coarse-floor-final.json
```

This checks every frame: four-spacing liquid cells near the centre, conserved
mass, no rejected topology, no activity faults, and no missing/air/low-density
samples in the central radius-12 ball. It checks ballistic motion and variance
through frame 24, requires substantial liquid in the floor row at frame 25,
and validates accepted dynamic transport packet addresses for all rungs.

CPU catalogue tests compare all 84 seam variants and every isolated rung to
the authoritative composite-grid builder, including areas, distances,
coefficients and incidences. Focused Dawn tests cover actual prepared seam
binding, invalid backing, the coarse slice reader and empty-domain clearance.
Full repository typechecking still has unrelated existing errors; changed
production files and the new probes/tests typecheck without errors.

Remaining excess near impact and post-impact dynamics are not claimed solved.
The earlier comparison table describes the separate velocity-extension fix,
not a new performance baseline for multi-rung pages.

## Tiny residual volume cleanup

The cleanup threshold is a maximum density of `1e-4` across a whole accepted
page. Every vertex of its accepted phi cells must have positive deep-air
support. Missing, metric, liquid, non-finite density, or negative-density
support prevents deletion. This deliberately protects resolved interfaces and
thin sheets even when their volume fraction is small.

One workgroup per page first checks densities and reduces volume. Empty or
non-dilute pages skip phi inspection. Eligible pages are cleared exactly to
zero in the accepted destination density and volume planes, and mark their
activity closure dirty. Ordinary retirement still enforces face and frontier
support; clearing a page is not a promise that it can immediately retire.
No topology is generated or repaired by cleanup.

The geometric transport receipt exposes `residueDeletedVolumeFine3` and
`residueClearedPageCount` for the latest frame, plus
`cumulativeResidueDeletedVolumeFine3` and
`cumulativeResidueClearedPageCount` for the resident lifetime. Re-clearing a
previously cleared and subsequently refilled page counts as another event.
Volumes use finest-cell cubed units, with floating-point workgroup reduction
and atomic accumulation. They are diagnostics of deliberate mass loss, not
credits already restored to the fluid. There is no global compensation.

Run the focused GPU fixture with:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
node --import tsx --test tests/sparse-cm12-residue-deletion-dawn.test.ts
```

The 90-frame Dawn capture `figure7-residue-air-departure.json` reaches 3 s
with no simulation halt or GPU validation errors. At frame 35 the liquid
bounds reach the full 128-by-128 floor. At frame 90 the ledger records
12.934781 finest-cell volumes deleted (0.038653% of the initial 33464), and
live volume plus recorded deletion differs from the initial amount by
0.031391 finest-cell volumes (0.000094%). There are 4823 cumulative clearing
events, not 4823 distinct simultaneously resident pages. Two transient
capacity-denial events remain in the growth receipt; required missing material
receivers still fail closed rather than becoming outflow sinks.

Captured checkpoints through frame 25 retain four-spacing core pages and have
zero missing-phi, air-phi, or low-density samples in the central radius-12
ball. A separate every-frame rerun was deferred because another stage-cost
probe held the exclusive GPU lease. The focused cleanup and sparse-air reader
GPU fixtures pass. Full typechecking still reports pre-existing errors;
no errors reference this cleanup, its test, or the Figure 7 probe.

Reclamation also exposed an advection assumption: positive metric vertices
could not use the existing unrepresented-air departure certificate. They now
can, provided both velocity samples are valid and the provider confirms the
departure lies outside every represented source page. The result is deep air,
not an invented metric distance. Represented-but-missing samples still fail
closed, and no sampling radius or topology repair has been added.
