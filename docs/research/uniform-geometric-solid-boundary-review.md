# Uniform geometric solid/boundary review

Date: 2026-09-20. The review below describes the pre-change implementation.
Implementation and Dawn validation are recorded in the follow-up section.

**Update:** Figure 12 exposed a solid/domain geometry and pressure-projection
mismatch introduced by this implementation. See
[the instability investigation and revised proposal](uniform-geometric-figure12-instability.md)
before relying on the validation below.

Target: `uniform-volume` (Uniform Geometric). Comparators: `uniform`
(density-based uniform CM12) and `adaptive-mass` (Sparse CM12). The phrase
“uniform sparse CM12” does not identify a separate registered method, so both
comparators are covered. Sparse CM12's boundary restriction discussed below
is present in both adaptive-mass and adaptive-volume shader implementations.

## Main finding

Uniform geometric explicitly prevents separation at static voxel interfaces.
Figure 8's spherical vessel is an authored voxel shell, so this restriction
applies to its curved interior, including its overhead surfaces. There is also
no embedded-wall incoming-air update for geometric phi. These are strong,
specific explanations for persistent wall contact; their relative contribution
to the observed animation still requires a matched runtime experiment.

This is a normal-velocity/contact issue. It should not be described as proven
excessive tangential friction or viscosity.

## 1. Voxel walls cannot release in uniform geometric

`lib/methods/uniform/webgpu-uniform-reference.wgsl.ts`, `pressureFaceData`
(around line 720), has a geometric-only branch returning zero pressure-face
open fraction whenever either adjacent cell is a static solid voxel.
`projectPressure` (around line 1293) subsequently overwrites a face with zero
open fraction with the solid velocity. For the static vessel that is zero.
Gravity or an advected separating velocity cannot survive this projection.

The distinction between transport aperture and pressure dual-cell fraction
matters: a physically closed solid face need not have a zero variational
dual-cell fraction. The density-based uniform branch of the same function
returns 0.5 at static voxel contact. Its solid-adjacent density continuation
and nonnegative solid pressure bounds provide machinery for separation.
Merely sharing the multigrid implementation does not give geometric mode the
same boundary condition when geometric mode disconnects these faces.

Figure 8 sets `container.shape = "sphere"` in
`lib/core/cm12-paper-scenes.ts:createCm12Figure8`. Scene normalization authors
the shell through `solidVoxelShellForScene`; `lib/core/scene-lattice.ts` uses
`sphericalSolidVoxelShell`. The uniform host packs SolidWorld occupancy into
bits (`webgpu-uniform-reference.ts`, around line 875). This is the voxel path,
not the analytic moving-rigid-body cut-cell path.

## 2. The box-ceiling exception does not cover the sphere

The geometric domain branch of `pressureFaceData` enables a half-open pressure
dual cell only when the inward axis direction follows gravity; an authored
open top is handled separately. With downward gravity, the closed box ceiling
qualifies, vertical side walls do not. At zero gravity, no closed box plane
qualifies. This is an orientation heuristic, not general pressure/velocity
complementarity.

`lib/methods/uniform/uniform-volume.wgsl.ts:uvReleasedWalls` has the same
gravity restriction and only visits the six outer domain planes. It creates
an air gap using the actual inward wall-face speed. An interior voxel ceiling
or spherical shell cannot invoke this update. Passing a flat box-ceiling test
would therefore not establish correct separation in figure 8.

## 3. Surface transport has an independent contact-retention risk

`uvTrace` clamps characteristics to the domain and stops at fully solid cells.
`uvPhi` clamps its sampling coordinates too. This prevents tunnelling, but a
departing liquid surface also needs appropriate incoming-air data where its
backtrace meets a released wall. That exists for selected box planes through
`uvReleasedWalls`, not for embedded voxel walls.

`uvClosedWallPhi` additionally continues negative interior phi onto closed box
wall vertices using `min`. This makes arriving liquid reach a zero-normal-
velocity wall, but is deliberately one-sided wetting continuation. It should
be paired with an actual contact/release decision, rather than a permanent
gravity-based classification.

Consequently, changing pressure alone is insufficient as a complete design.
First establish a separating face velocity, then verify that phi becomes air
at the vacated contact and conserved liquid volume moves away consistently.
The repository's `docs/research/2d-ceiling-contact-and-open-boundary.md` records
an analogous scalar-boundary failure in the Rust 2D implementation; it is
supporting precedent, not runtime proof about this uniform GPU method.

## 4. Sparse CM12 is a useful comparison, but not the paper's full solution

`lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts`,
`rowSeparatingFromClosedWorldState` (around line 1542), can release one-sided
closed-world rows using predicted velocity, gravity, density membership, and
a travel deadband. `rowPressureOpenFraction` and `projectPressureRow` honor
that decision. Unlike geometric mode's unconditional voxel-face closure,
this machinery can permit eligible overhead closed-world rows to separate.

However, it explicitly excludes side-wall/floor release and comments that it
is not a general pressure-based complementarity solve. Less visible sticking
in sparse CM12 does not establish paper conformance. A matched experiment
must also control resolution, timestep, velocity transport, density/phi
surface reconstruction, and pressure convergence.

## 5. Phi/volume disagreement can amplify residual wall films

Geometric pressure membership normally follows vertex phi sampled at cell
centres. Conserved V can remain where phi says air. `pressurePhi` documents
this failure mode: cells can lose pressure rows and their volume can accumulate
against a wall. `uniform-volume-method.ts` currently defaults volume pressure
rows, phi seeding, phi agreement, and volume compaction to off.

This is a secondary candidate, not proof that every wall film is orphaned V.
Measure phi, V, pressure membership and face velocity separately. Turning on
all experimental controls is not a clean boundary fix. Geometric sharpening
also excludes partially open cells, and solid-excess correction is disabled;
these deserve separate cut-cell tests for terrain and moving solids.

## Paper comparison

The primary references are saved locally:

* `docs/papers/massConservingLiquids.txt`, Sections 3.6–3.7: distinguish solid
  volume and transport face aperture, extrapolate corrected density into
  adjacent solid cells, and use CM11a's separating pressure solve. Conservation
  and excess-density correction do not replace separating contact.
* `docs/papers/A_Multigrid_Fluid_Pressure_Solver_Handling_Separat.txt`, Eq. 11
  and Section 3.1: require `p >= 0`, wall-relative outward normal velocity
  `s >= 0`, and `p*s = 0`. Contact may support positive pressure; separation
  has zero pressure. The practical solver bounds pressure at solid nodes and
  transfers constraints through multigrid. The condition is not restricted to
  gravity-facing walls.
* `docs/papers/BBB07_Fast_Variational_Solid_Fluid_Coupling.txt`, Section 4:
  explains grid-scale wall/ceiling films from zero-normal contact, formulates
  the unilateral condition, and discusses failures of simply deciding release
  from pre-projection velocity, particularly for oblique or enclosed domains.
* `docs/papers/ENGF03_PLS_Second_Order_Pressure_Boundary.txt`: subcell
  free-surface pressure placement matters, but ghost-fluid accuracy alone
  does not supply separating solid contact.
* LGF11 and LAF11's saved conservative-advection papers: conserving transported
  mass does not establish momentum preservation or correct pressure contact.
  Splash-energy loss should be measured independently after boundary release.

Separation is permission to peel away, not an elastic restitution rule. A
ceiling impact need not reverse velocity like a bouncing rigid ball. The
relevant tests are impact-driven lateral spreading, subsequent detachment,
and absence of an artificially persistent cell-thick film.

## Recommended implementation and verification order

1. Establish a shared geometric contact contract for domain walls, voxel
   shells, terrain, and analytic solids. Keep transport aperture distinct from
   pressure dual volume. Use CM11a-style constrained pressure with consistent
   solid-side phi continuation, or an explicitly verified equivalent active
   set; do not just open all blocked faces as zero-pressure air.
2. Make incoming-air phi data follow the solved release state at embedded as
   well as domain walls. Preserve contact continuation where liquid arrives,
   and let air win where contact releases, including corners.
3. Add matched small fixtures: flat ceiling versus interior voxel slab;
   side-wall release with no gravity; sphere/oblique-wall peeling; hydrostatic
   pool; and pressure-supported closed contact. Measure penetration,
   separating speed, pressure/contact residual, wall-band V, wall-band phi,
   divergence, mass and kinetic energy.
4. Run figure 8 with matched fine resolution and timestep, comparing geometric,
   density-based uniform CM12, and Sparse CM12. Inspect overhead, vertical and
   lower wall sectors separately; quantify residence time and detached volume.
5. Only then isolate phi/V feedback and velocity-advection dissipation.

Existing uniform geometric tests check arriving wall contact and far-wall
impact, but those checks do not demonstrate embedded-wall detachment. Several
tests named sparse-cm12 import adaptive-volume, so check the imported method
when selecting comparison lanes.

No GPU tests were run for this source-only review. After a substantial boundary
implementation change, run the repository's canonical
`npm run test:dawn:sparse-cm12` gate, exclusively from browsers and other Dawn
processes, as well as targeted uniform geometric boundary regressions.


## Implementation follow-up

Uniform geometric now retains pressure dual-cell weights at static voxel
contacts and all non-symmetry domain walls. Solid pressure rows use one-layer
liquid-phi continuation and nonnegative pressure constraints. The outer solid
halo receives its incident predicted velocity in the pressure RHS; it is not
silently treated as a prescribed zero-pressure opening. Open-top ghost phi
remains atmospheric.

Projection publishes a six-bit contact-release mask in the otherwise unused
MAC texture w component (three positive faces and three low domain faces).
An eligible face must have zero solid pressure and separating travel exceeding
1e-4 cells. The travel threshold filters solver residue; the pressure active
set decides whether supported contact may release. Rigid bookkeeping preserves
this mask.

Phi uses this solved mask for incoming-air data at domain walls and embedded
voxel contacts. Embedded characteristic hits supply air beyond the first cell
of a swept gap. Contact continuation cannot re-wet a dry wall from liquid that
is moving away. Velocity extension uses pressure dual support in geometric
mode, so a blocked mass-transport aperture does not erase a separating MAC
velocity. Conservative volume transport retains its closed-solid aperture.

The focused Dawn command is `npm run test:dawn:uniform-geometric-boundaries`.
It covers box and embedded ceilings, both signs of embedded side-wall release
without gravity, and a 32-cubed version of figure 8. It checks surface release,
projected velocity, mass conservation, finite fields and no volume penetration.
The figure-8 metric follows initially wetted upper-shell vertices, rather than
mistaking newly arriving splash on the opposite wall for failed detachment.


### Validation and remaining limitations

The final focused uniform Dawn run passed all 17 checks (the five boundary
fixtures plus the existing numerical/hydrostatic tests and their parent tests).
See `uniform-geometric-boundary-dawn-2026-09-20.json` for exact measurements and
the complete sparse-gate receipt.

The existing mini32 test's represented-phi-volume cutoff was removed at the
user's request to remove failing existing checks. Its conserved-volume,
finite-field coverage elsewhere in the suite, and final wall-cell capacity
assertion remain. Phi-volume drift is still logged: it is -6.38%
at frame 30 and -6.69% at frame 90, despite conserved V. This is a real remaining
geometry/volume disagreement, not proof of mass loss and not a fully resolved
visual-volume regression. The unchanged implementation passed the old cutoff.
Tightening pressure convergence did not improve it, so the pressure defaults
remain unchanged. No unit tests were added.

The canonical sparse gate ran without concurrent browser or Dawn use and
finished inside its 480-second total budget, but only 4/17 lanes passed. Its
receipt records timeouts, a mini64 timing-ceiling failure, and explicit Long
Dam initial-topology, Tall Cells Hills tank-boundary, and outside-tank compiled
connectivity failures. No sparse lane, assertion, or timing ceiling was changed.

An additional existing adaptive-volume overlay Dawn test, included in the
older `test:dawn:uniform-volume` command, failed at frame 0 with missing
capacity. It is outside the changed uniform solver. Repository-wide TypeScript
checking also reports existing errors in unrelated sparse tests/tools; no
errors were reported in this change's files.

With the three uniform shader files temporarily restored to HEAD, Long Dam
reproduced the exact initial topology failure (792 versus 106 leaves). The
outside-tank lane also failed on that unchanged baseline, but by its 20-second
timeout, so that rerun does not establish the same topology failure. Final
uniform changes were restored after both isolated baseline runs.
