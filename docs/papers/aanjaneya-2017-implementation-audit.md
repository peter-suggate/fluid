# Aanjaneya 2017 implementation and dam-break audit

Date: 2026-07-22

This is the implementation specification and gap record for Aanjaneya et al.
(2017), *Power Diagrams and Sparse Paged Grids for High Resolution Adaptive
Liquids*. The searchable source is
[`aanjaneya-2017-power-liquids.txt`](aanjaneya-2017-power-liquids.txt). The
Ando--Batty 2020 formulation is not an authority for this path.

## Verified mini-dam acceptance

The isolated Dawn `minimal-power-dam-break` acceptance run now reaches exactly
2.000 s in 500 fixed 0.004 s steps. Every accepted step audits one coherent
generation of the pressure topology, power faces, Section 5 velocity band,
global-fine level set, and hybrid-PCG solve. The run admits all 500 generations
without topology rollback, non-finite state, or renderer closure failures. The
named npm smoke command encodes this full 2 s contract.

This verifies consistency and publication, not physical energy parity. The
final maximum speed is about 0.0326 m/s and the metric kinetic-energy proxy is
about 8.3e-6, so the separate tall-cell comparison remains the authority for
the reported excessive damping. Per Section 5, that investigation must focus
on advecting the full old-mesh velocity at each new power-face centroid before
projection onto the new face normal; PCG convergence alone cannot establish
transport fidelity.

## Reproduced symptom

The isolated `minimal-power-dam-break` run reached 0.5 s (125 fixed 0.004 s
steps) with MGPCG relative residual about `5.5e-5` and maximum reported
variational residual about `5.8e-8`, yet the final liquid pressure topology had
become uniform:

- 1,449 liquid leaves, all size 1;
- no size-2 liquid leaves;
- no cross-scale power faces, transition rows, or transition tetrahedra;
- roughly 3,833--4,096 of 4,096 fine pages resident;
- roughly 4,082 of 4,096 coarse cells in the face band.

Thus the old regression could pass after ceasing to exercise the adaptive
pressure path. A small algebraic residual only proved convergence to the
assembled matrix; it did not prove that row classification, signed distance,
and topology represented one physical generation.

After replacing historical acceptance with live final-state checks, the next
isolated run failed at generation 74. Generation 73 was coherent; the current
fine page set then caused the Section 5 graph to include far-air owner cell
4095, which had neither a compact pressure row nor a coarse-phi record. The
bounded march rejected that unrelated row, velocity transport did not commit,
and the target fine generation rolled back while the coarse field retained
generation 73. QA now rejects this mixed-generation state instead of sampling
through it.

## 2017 requirements matrix

| Paper requirement | Source | Current status |
| --- | --- | --- |
| Pressure at octree cell centres; normal velocity on orthogonal power faces | lines 249--307 | Implemented by generalized power faces and compact rows. |
| Finite-volume Eq. (3)/(4), symmetric face coefficient used by assembly and projection | lines 269--301 | GPU operator is algebraically consistent. CPU oracle now includes optional open Dirichlet boundary diagonals and projection. Explicit CSR is an engineering alternative to Section 6.3 storage. |
| Free-surface and solid embedded boundaries may cut adaptive cells near T-junctions | lines 283--301 | Partial. Free-surface coefficients now require a strict current liquid/air crossing. Mandatory unit wall/interface refinement remains a policy gap. |
| Active/ghost SPGrid pyramid with `GhostValuePropagate` and `GhostValueAccumulate` | lines 319--373, 578--613 | Not implemented faithfully. The GPU V-cycle emits one origin ghost per pressure row, lacks the paper's face/edge ghost aliases and explicit propagate/accumulate operators, and drops mapped non-axis off-diagonals from its coarse stencil. |
| Section 4.3 hybrid: 8 L2 Jacobi sweeps near boundaries/transitions, L1 V-cycle, 8 symmetric post-sweeps | lines 379--409 | Broad schedule implemented. Boundary rows are now explicitly marked, including closed/cut solids. The middle L1 V-cycle is not the paper operator and lacks numerical GPU SPD proof. |
| Separate factor-4/factor-8 fine SPGrid narrow band plus coarse octree phi | lines 410--478 | Partial. Two main authorities exist, but a legacy adaptive surface continues as an evolving oracle/seed source. |
| Rebuild fine topology from interface cells, interface blocks plus block one-ring, then FMM activation | lines 430--462 | Recurring external seeds are now restricted to explicitly tagged endpoint support; ordinary cold affine CORE seeds no longer repopulate the whole fine grid. Pre-dilation still allocates the full backtrace/interpolation/redistance width rather than FMM activating it progressively. |
| `m` segmented backtrace steps and local cube/Delaunay scalar interpolation | lines 435--495 | Segmented fine-phi backtrace exists. |
| Advect a full vector at every new power-face centroid, then dot with the new normal | lines 480--495, 583--586 | Missing. Current transport advects regular-axis scalars and transfers/reconstructs power values afterward. |
| On topology change, trace new face locations into the old mesh; avoid separate transfer | lines 578--586 | Missing. Current copy/child-average/parent-injection transfer can change energy. |
| Ordered local Delaunay FMM and face-based closest-interface velocity extrapolation | lines 450--495 | Partial. Production redistance is JFA/CPT; small ordered FMM is diagnostic-only. Whole-domain face propagation was removed and is now bounded by the authored narrow band. |
| Every face/edge ring is exclusively same-or-one-coarser or same-or-one-finer | lines 518--565 | Mixed-ring repair exists. Sparse owner lookup now fails closed instead of synthesizing a coarse owner for missing/stale pages. A required global all-leaf publication audit is still missing. |
| Row-local Delaunay tetrahedra with current-row solid angle below pi/2 | lines 463--479, 542--565 | Implemented for every same/coarser descriptor. The catalog now includes co-spherical sites touching at a Voronoi vertex and searches deterministic non-crossing link triangulations; all six formerly rejected masks are strictly acute without topology refinement. |

## Directly corrected findings

### Generation-coherent free-surface pressure

The power boundary shader cited nonexistent 2017 Section 4.5 and Equation 26,
allowed a liquid row to sample as air, and clamped `theta` to `0.01`. This was a
2020-derived escape hatch that could turn a publication mismatch into a face
coefficient 100 times its geometric scale.

Both fine and coarse boundary paths now require finite `phi_liquid < 0` and
`phi_air > 0`, use the exact zero-crossing fraction, and reject the generation
otherwise. The CPU oracle uses the same strict crossing.

### Fine narrow-band scope

Recurring fine generations now begin with transported interface discovery and
only explicitly tagged power-boundary endpoint seeds. Ordinary affine CORE
seeds remain cold-start data. Face-phi and closest-point propagation use an
interface-band iteration bound rather than `nx + ny + nz` depth. This does not
yet make the graph sparse: pre-dilated fine pages are still mapped as core band
rows, so the current dam break can remain nearly domain-wide.

### Owner topology failure semantics

Paged descriptor lookup now rejects missing, reserved, malformed, in-flight,
out-of-bounds, unpublished, or wrong-generation owner data. It no longer
silently fabricates a canonical coarse owner, which previously could erase a
fine neighbor while producing a superficially valid descriptor. The Section 5
face-band lookup now follows the same fail-closed rule. Terminal completion and
audit skip owners without compact site membership before touching paged owner
storage; support rows induced earlier by a domain-wide core remain a known gap.

### Section 4.3 boundary band

Power-row assembly now publishes explicit boundary incidence for world/free
boundaries and partial/closed solid apertures. MGPCG uses this flag in addition
to a Dirichlet diagonal gap and resolution changes, so solid Neumann rows enter
the symmetric L2 boundary smoother.

### Diagnostics and Dawn acceptance

The compact Eq. (4) diagnostic no longer multiplies aperture twice. A final
power-face metric energy proxy is reported. Historical transition telemetry
may identify when adaptivity disappeared but can no longer replace final live
leaf, face, row, or tetrahedron counts.

### Local Delaunay catalog completeness

The six coarse-neighbor masks previously treated as intrinsically obtuse were
an artifact of catalog construction. Sites whose Voronoi cells met only at a
vertex were omitted, and the remaining link polygon was triangulated as a fan.
The generator now retains every co-spherical site at each Voronoi vertex and
selects a deterministic minimax non-crossing triangulation. All six masks now
have maximum row-local solid angles below `pi/2`, so their special topology
refinement and runtime rejection paths have been removed.

## Remaining high-priority implementation gaps

1. Replace the approximate first-order V-cycle with the actual active/ghost
   SPGrid L1 operator and propagation/accumulation transfers, or construct an
   exact symmetric Galerkin substitute and prove its GPU action linear, SPD,
   and effective on mixed levels and boundaries. The outer eight-sweep
   Jacobi--M1--eight-sweep composition has a CPU algebra test for linearity,
   symmetry, and positivity, but that proof is conditional on a faithful SPD
   M1 and does not validate the current GPU pyramid.
2. Advect velocity as the paper specifies: retain the old generalized mesh,
   backtrace every new power-face centroid, interpolate the old full vector
   with cube/local-Delaunay interpolation, and project onto the new normal.
   Remove the separate topology transfer from the authoritative path.
3. Make ordered FMM/ordered-upwind the production fine redistance method, or
   establish a long-run differential bound proving the parallel replacement
   preserves zero crossing, Eikonal error, and volume.
4. Reduce fine-page allocation to interface blocks plus their one-ring and let
   marching activate only the required valid band. Page count must scale with
   interface area times fixed width, not domain volume. Mapping only compact
   rows was tested but correctly failed staged t=0 because its present support
   closure exhausted capacity and left 690 faces disconnected; that experiment
   was reverted rather than weakening publication gates.
5. Decouple coarse octree sizing from mandatory unit refinement at every wall
   and free-surface crossing. Add authored source/object/boundary refinement
   regions and keep valid coarse pressure cells across embedded boundaries.
6. Remove the independently evolving legacy fine-surface authority after
   bootstrap; rollback should preserve the prior global-fine generation.
7. Add an all-live-leaf publication audit for face and edge 2:1 balance,
   exclusive same/coarser versus same/finer rings, reciprocal current owners,
   and a valid row-local catalog entry.
8. Turn final metric energy into before-projection, after-projection, and
   after-regrid budgets. The current final proxy cannot yet localize energy
   loss between projection, transfer, and advection.

## Required Dawn tranche

Run only through the repository's isolated/staged Dawn launchers.

1. A two-second dam break with a live authored two-level region. At every
   checkpoint require both leaf sizes, transition/oblique faces, transition
   rows/tetrahedra, no grading violations, and coherent generation stamps.
2. Move the interface through pressure-row centres and assert that stale rows
   fail publication; no theta-floor fallback is permitted.
3. Extract the production preconditioner action and test linearity,
   `x^T M y = y^T M x`, positive energy, residual contraction, and PCG
   convergence with free-surface, solid, and 2:1 rows.
4. Manufactured constant, affine, and divergence-free vector fields across
   alternating refine/coarsen steps; measure flux and energy discontinuity.
5. Independently recompute committed generalized-face Eq. (4) residual and
   the face metric energy before/after projection and topology change.
6. Compare production redistance with an ordered FMM oracle for planes and
   spheres over hundreds of steps.

## Verification notes

Focused non-Dawn tests are safe to run without `WEBGPU_NODE_MODULE`. Directly
combining multiple Dawn test files previously ended in a native teardown
`SIGSEGV`; that is invalid orchestration under `docs/SAFE_WEBGPU_BRINGUP.md`
and is not solver evidence.

The full isolated bring-up sequence passed after the catalog, pressure, and
strict-owner corrections. The 125-step dam-break then failed closed at target
fine generation 74 for the domain-wide Section 5 row described above. A later
experimental compact-core filter passed shader compilation but failed the
staged `sparse-t0` publication gate (transition capacity, incomplete support),
so no subsequent Dawn stage was run and the experiment was reverted.
