# Level-set liquid separating from a closed wall

## Reproduction

The 2D advance lab uses `water-box-dam-break`, `level-set-volume`, a 1/30 s
step, 256 pressure iterations, and relative tolerance 1e-6. The lab now selects
level set + volume by default. The accepted boundary correction is also implemented in the 3D WebGPU method.

Before this change, the SIMD production World gave the following measurements.
Ceiling volume is integrated over the uppermost one-fine-cell strip; all
volumes below are in fine-cell area units.

| Frame | Total volume | Ceiling-strip volume | Negative ceiling phi vertices |
| --- | ---: | ---: | ---: |
| 30 | 168.000001 | 1.058390 | 3 |
| 45 | 168.000000 | 4.173402 | 2 |
| 60 | 168.000001 | 0.077615 | 2 |
| 90 | 168.000003 | 0.012685 | 1 |

The stage probe at frame 30 found downward velocity in the ceiling cells,
but zero velocity on their closed ceiling faces. Transport reduced the
ceiling-strip volume from about 1.311 to 1.058 in that step. This particular
step was draining, rather than depositing liquid back at the ceiling.

## Two boundary inconsistencies

1. Pressure membership for this method follows the level set, but wall release
   used the conservative density threshold 0.5. Thin phi-liquid contacts lost
   the ability to separate when their density fell below that threshold.
2. Surface advection clamped departure points to the box. A ceiling vertex
   moving downward backtraced above the ceiling, then sampled the old wet
   ceiling value again. Releasing velocity alone could not create an air gap.

The mass-conserving-liquids paper, section 3.7, explicitly uses a pressure
solver with separating solid boundary conditions. The boundary condition must
also be represented by scalar transport; replacing the pressure solver alone
does not fix a clamped scalar departure.

## Shared requirements

- Use the current level set to classify surface contact. Conservative volume
  remains independently conserved; do not erase it to clear a rendered wall.
- Express departure using wall-relative normal velocity, not a special case
  for the upper Y boundary. Liquid can move away from a wall but cannot enter
  the solid. This patch retains the existing acceleration-normal eligibility
  rule; it does not introduce inertial side-wall release under vertical gravity.
- Treat a released wall as a newly exposed liquid-air boundary for scalar
  sampling. The nearby interior and exterior scalar values must describe the
  same boundary, so a translated planar surface retreats by the actual normal
  displacement rather than an amount determined by its old distance scale.
- Preserve resting and tangentially moving surfaces at walls that remain in
  contact, including the hydrostatic floor.
- Check both the authoritative surface and the conservative volume. A dry
  ceiling scalar alone does not prove correct draining, and small ceiling
  volume alone does not prove detachment.
- Check pressure-projected wall velocities. Clamping a velocity after a linear
  solve changes its divergence without updating the reported solve residual;
  it is not equivalent to solving the unilateral boundary problem.

These requirements extend to six box faces in 3D. Embedded or moving solids
need their own wall geometry and relative velocity; an axis-aligned world-wall
implementation should not be treated as proof for those cases.

## Why checking the final pressure velocity matters

During development, enabling departure at all phi-liquid wall contacts exposed
another problem: at frame 30, pressure reversed 14 of 15 initially separating
faces toward the solid. The worst away-normal velocity was approximately
-9.023 fine cells/s after projection. Consequently, a predicate evaluated only
before pressure is insufficient.

The inexpensive correction uses the existing linear pressure solver in a
boundary active-set loop. After projection, bind any released face that points
into the solid. Restore the original preprojection face field, enforce the
newly bound wall velocities, rebuild the pressure operator, and solve again.
Reusing the already projected field would apply an extra pressure correction to
the wrong right-hand side. Every retry binds at least one face, so the number
of initially released faces bounds the retries. Apply this check to both the
primary projection and the projection after a support/topology transfer.

This is a correction to the predicted set of released box faces, not the
paper's full pressure complementarity solver. In particular it does not claim
to solve arbitrary oblique contact, moving embedded solids, or all possible
tensile-pressure configurations. The 3D port retains those limits
and verifies the normal-velocity inequality with its own pressure operator.

## 3D WebGPU implementation

The 3D method keeps the original single pressure-solve schedule. Wall-release
eligibility uses current level-set membership and the existing gravity-normal
rule. The level-set boundary query walks accepted CNX incidence lists and reads
the actual projected MAC velocity relative to the solid, including after a
topology transfer. It only carves an air gap when that velocity points away
from the wall. It does not use an old pressure row ordinal or the collocated
velocity used to trace the interior field.

The first port encoded additional pressure retries. Those retries caused a
large velocity-projection regression and were removed at the user's request,
along with their state, audits, and extra face-publication passes. Unlike the
2D implementation, the final 3D port does not enforce a pressure active set.

For a released face, the scalar continuation at a vertex is

`phi_wall = dt * away_face_speed - inward_distance_to_wall`.

The advection result is the maximum of transported phi and every applicable
wall continuation. The query projects the vertex onto each of the six physical
box planes and uses four tangential epsilon probes to find every accepted cell
incident to a patch edge or coarse/fine seam. It checks the one-term
`ClosedWorld` row, its axis, plane, inward orientation, tangential footprint,
and final wall-relative velocity before accepting the candidate. This keeps
the lookup proportional to six boundary planes instead of scanning the global
row roster. Because discovery happens on the wall plane, the continuation has
unrestricted normal reach: a displacement spanning several boundary cells is
represented without an assumed CFL ceiling.

The same maximum is applied to ordinary trilinear advection, the deep-phase
fast path, and the certified sparse-support fallback. When the wall value wins
inside the four-fine-cell distance band, its support becomes metric so the
redistance pass can use the newly exposed contour as a seed. Values farther
from the contour retain signed deep support. A zero timestep bypasses the wall
query, and bound or non-outgoing faces leave phi unchanged. Conservative
volume transport remains independent and is not modified by this correction.

The analytic WebGPU fixture covers all six directions, face intersections,
zero, half, full, and multi-cell displacements, metric and deep source values,
and inactive walls on an eight-cell adaptive level-set topology. The ordinary
unit suite also checks the injected boundary contract, accepted-CNX lookup,
final-MAC authority, tangential patch search, and support re-tagging.

## 2D result

The final SIMD run retains total volume 168 to better than 1e-6 relative error.
All ceiling vertices are air at frames 30, 60, and 90. The ceiling can become wet
again during a later splash (frame 45); the fix does not permanently mark it dry.

| Frame | Ceiling-strip volume | Negative ceiling phi vertices |
| --- | ---: | ---: |
| 30 | 0.079378 | 0 |
| 45 | 8.647017 | 1 |
| 60 | 0.062600 | 0 |
| 90 | 0.414412 | 0 |

The transported volume field still contains dilute splash residue and is not
identical to the surface's enclosed volume. In particular this change does not
monotonically reduce the ceiling-strip volume at every frame. It fixes the
wall velocity constraint and detachment of the authoritative surface.

Three Node SIMD runs took about 0.73 s per 90 frames after warm-up, versus
0.86 s for the original version on the same machine. Median frame cost was
about 6.5 ms versus 10.6 ms. This compares evolving scenes, including their
different adaptive work; it does not establish a general pressure-solver speedup.
The compact measurements are in
[`ceiling-separation-report.json`](../../artifacts/level-set-volume/ceiling-separation-report.json).

The regression checks 90 frames, conservation, adaptive topology changes,
the complete published ceiling vertex row, and the wall-normal velocity
inequality after both pressure passes. The analytic test starts from deeply
negative wall phi and deliberately mismatches collocated and boundary-face
velocity; it checks exact gap displacement, half-step scaling, and zero-step
identity. Hydrostatic and tangential/side-wall checks pass as well.

## 3D validation

The focused production `water-box-dam-break` test runs at 1/30 s through frame
90 and checks ceiling phi, top-strip volume, topology changes, and conservation.
The three-frame ceiling-contact test checks the first visible air gap. The
analytic GPU test covers all six directions, corner intersections, and gaps
larger than one cell. All three focused GPU tests pass after removing retries. The dam run has zero
wet ceiling vertices at frames 30, 60, and 90; its final relative volume error
is 1.38e-06. Detailed receipts are in
`artifacts/level-set-volume/ceiling-separation-3d-report.json`.

## 3D frame cost after removing pressure retries

Dawn Metal hardware timestamps, timeline instrumentation, sequential runs of
`water-box-dam-break` at 1/30 s; medians over frames 30–90:

| Measurement | Before 3D port | Port with retries | Final, retries removed |
| --- | ---: | ---: | ---: |
| Velocity projection GPU ms | 0.328 | 82.182 | 0.328 |
| Total GPU frame ms | 45.089 | 225.313 | 44.630 |
| CPU encoding ms | 4.719 | 15.280 | 5.039 |
| Compute passes at frame 90 | 90 | 185 | 90 |
| Direct dispatches at frame 90 | 362 | 1391 | 362 |
| Indirect dispatches at frame 90 | 512 | 2137 | 512 |

The final host scheduling matches the pre-port scheduling. The wall-gap query
adds arithmetic and owner lookup inside existing phi advection, with no extra
compute pass. These are evolving-scene timings, not proof that the scalar query
has zero cost. The measurement summary is in
`artifacts/level-set-volume/ceiling-separation-3d-performance.json`.

## Remaining late-time redeposition

This is a boundary correction, not a complete cure for conservative-volume
residue. A native stage probe at frame 90 measures ceiling-strip volume 0.292919
before transport and 0.414413 afterward. None of that volume belongs to a
phi-liquid pressure cell. Its volume-weighted vertical velocity is +4.075
fine cells/s after pressure, then +15.407 after the pretransport velocity
extension. The extrapolated liquid velocity advects the residual density
upward even though the level-set surface is already well below the ceiling.

Thus the user's redeposition suspicion also holds independently of the
pressure wall violation. It is a remaining disagreement between conservative
volume, phi-based velocity support, and extrapolation into air. Resolving it
requires a consistent transport/support or volume-correction policy; clearing
the rendered phi, deleting the residual volume, or clamping a wall face does
not resolve it. After reviewing the 2D result, the user approved porting the boundary
correction to 3D while keeping this remaining transport issue separate.

An attempted extension to inertial side-wall release made this residue worse
and also removed the hillside fixture's mixed-seam coverage. That extension
was removed. The accepted patch preserves the original acceleration-based
eligibility and its side-wall behavior.
