# Persistent implicit density: implementation and evidence roadmap

Historical foundation: standalone CPU research, 2026-09-08. The subsequent
[production acceptance record](retained-density-production-progress-2026-09-08.md)
records the selected positive diffuse numeric field, production native/surface
adoption and conservative low-order evolution stage. Prescribed motion and
high-order curvature transport remain open roadmap work. This roadmap
implements the priorities in the
[density-field contract](adaptive-mass-density-field-contract.md) and
[architecture proposal](adaptive-mass-interface-architecture-2026-09-08.md).
The contract takes precedence: means are volume integrals, physics refinement
does not change the retained field, and sharp features must not be smoothed
away merely because physics cells become larger. No dual contouring
implementation or interim production surface correction is proposed.

The [first implementation results](implicit-density-ladder-results-2026-09-08.md)
record the completed exact-family algebra ladder and mean-reconstruction tests,
with independent reference overlays. Later rungs remain open.

The [cutover implementation status](retained-density-cutover-status-2026-09-08.md)
records the subsequent native coupling and GPU ownership work, the rejected
Bernstein mean correction, retained affine/curved primitives, and the remaining
production adoption requirements. It includes actual production partition
failures and distinguishes them from passing component tests.

## Current module scope

[`field.ts`](../tools/implicit-density/field.ts) provides physical-coordinate
affine/quadratic density records in local frames, analytic box means, exact
polynomial rebasing, and two-branch min/max evaluation. Its min/max mean
integrator accepts affine branches only. It also integrates a clamped affine
ramp. Clamping is a distinct function, not an automatic positivity guarantee
for every field record. Full quadratic and two-branch affine algebra can
represent useful curvature and a crease; they do not establish an adaptive
global representation, arbitrary corners, thin sheets, or disconnected sets.

[`compiled-stencil.ts`](../tools/implicit-density/compiled-stencil.ts) compiles
geometry-dependent least-squares maps from box means, including second volume
moments, to ten polynomial coefficients. It constrains the home mean and
rejects deficient support. Application gathers donor means once and multiplies
by retained weights, with a generation check and optional float32 arithmetic.
This is an oracle compiler, not an adapter to the production topology ABI.
Its QR diagonal ratio is a support diagnostic, not a rigorous condition-number
or error certificate. Report actual reconstruction residuals and amplification.

Neither module ensures positivity, compatible traces between independently
fitted patches, conservative transport, solid clipping, a persistent sparse
hierarchy, pressure compatibility, or GPU performance. Reconstructing patches
from new physics means after every edit would still violate the desired
persistence contract, even when the local fit reproduces polynomials exactly.
The compiler is useful for initialization, controlled reconstruction experiments,
and eventually explicitly budgeted field updates; it is not permission to
replace retained geometry during a zero-time physics edit.

## Authority and unresolved mathematical choices

Retain one accepted field generation q and derive cell amounts by integrating
q over each cell's open domain. Divide by full cell volume for resident rho;
divide by open volume only for effective density. The free surface is the
boundary of q > 0.5 in open space. For diffuse q its enclosed volume is not
integral q. For a sharp indicator the two agree, but that changes the scalar
model and transport requirements. CM12 permits density above one: silently
clamping it, or calling it a bounded volume fraction, loses its semantics.

Before coupled dynamics, choose and justify either a positive diffuse density
model with explicit excess-density handling, or a geometric indicator model
with conservative transport replacing the corresponding CM12 scalar model.
If a diffuse/excess decomposition is used, define the residual's domain,
sign constraints, mass receipt, transport, and influence on the 0.5 surface.
A separate mass ledger with unconstrained geometry is insufficient.

Local polynomial positivity must hold over its entire retained support, not
only the donor centres. A limiter or clipping operation changes q and its
integrals; record that change and recheck conservation and shape. An affine
clamp's exact mean does not solve positivity for quadratic/min/max patches.

Global assembly needs a declared ownership rule and compatible patch traces.
Select a representation with shared boundary constraints or a common field
basis, or explicitly model the discontinuities as intended material jumps.
Independent home-mean constraints do not enforce either compatibility or all
overlapping-cell integrals. Smooth patch boundaries require scalar and normal
consistency; creases require the intended branch normals and feature location.
Blending is admissible only if its changed field, volume constraints, and
feature errors are measured. Unintended inter-patch jumps are failures.

## Example ladder and acceptance gates

Every rung produces candidate/reference overlays and a machine-readable receipt.
Use an independent evaluator for the reference surface and volume integrals;
do not score a candidate using its own reconstruction or root finder alone.
Initialize from declared data, then remove the authored shape callback from
candidate state. The oracle may retain it. Show all ray intersections along
multiple rotated directions, including lower surfaces and detached components.
Mesh appearance alone cannot pass a rung.

| Rung | Concrete implementation and fixtures | Evidence required to advance |
| --- | --- | --- |
| A: integral algebra | Constant and oblique affine density; a quadratic with nonzero mixed terms; affine min/max crease; clamped planar ramp. Vary box position, aspect ratio, scale and orientation. | Independent quadrature/convergence check, coefficient rebase identity, local child mass sums, full gradient checks and both crease normals. Flag cancellation for nearly axis-aligned or slender cuts; do not hide it with loose tolerances. |
| B: reconstruct from means | Fit the same planes and quadratics from independently integrated uniform and mixed-width donor boxes. Add curved profiles outside the basis, boundaries, missing/degenerate support and noise. | Exact-family reproduction separately from approximation convergence; home and donor integral residuals, rank failures, scalar extrema, normal and curvature error, float64/float32 comparison. Known normals belong only to labelled algebra fixtures. |
| C: globally retained field | Implement sparse support ownership, compatible interfaces and explicit feature records. Start broad tilted plane, shallow sinusoidal wave, sphere/ellipsoid, wedge, three-plane corner, thin double-sided sheet, and two separated bodies in one coarse physics cell. | Independent surface distance, integral and moment errors; all crossings, components/gaps, seam scalar/normal errors and feature normals. Prove that broad smooth regions need not carry an always-finest band. Reject unsupported merges. |
| D: independent partitions | Hold accepted q fixed; integrate uniform/mixed physics grids and partial moving region edits on all axes, edges and corners, including macro and clipped domains. Separately refine/coarsen q's own support. | Physics-only changes preserve canonical q exactly; actual partitions change. Check local amounts, untouched neighbours and 1/10/100 edit cycles. Field merges meet explicit cumulative shape and topology budgets or retain detail. |
| E: prescribed motion | Translation, rigid rotation and reversible divergence-free deformation of the curved, creased, sheet and two-body fixtures. Include high-CFL cases, fixed versus changing physics partitions and deformation beyond representable detail. | Conservative local flux receipts, displacement/phase and shape convergence, no seam locking or phantom connections; positivity/excess receipts and all substep costs. Static remapping does not pass transport. |
| F: coupled vertical slice | Shared geometry for classification, pressure distances, native open-face areas and publication. Flat pool, hydrostatics, free fall, standing wave, ball impact, thin falling sheet and moving solid. | Stable finite evolution and expected physics; one accepted geometry generation. Refine pressure support when disconnected wet pieces/gaps cannot be represented. An arbitrary region width cannot override this requirement. |
| G: sparse GPU and migration | Compile validated state and operators to GPU; measure broad pool, many small interfaces, local edits, moving solids and increasing empty extent. Migrate full production consumers only after the vertical slice passes. | End-to-end time, actual work/storage, churn and failure-transaction receipts; canonical Dawn gate unchanged, then serialized browser inspection. CPU f32 emulation is not WGSL validation. |

Freeze fixtures, seeds, source hashes, units and budgets before captures.
Use the architecture's initial gate-0 targets (mass < 1e-10 unit-cube volume,
plane displacement < 1e-9 unit-cube length) only for their declared exact
float64 scope. For support coarsening, start the documented sweep at
0.02 times reference finest width and 1 degree on smooth normals, measuring
drift from the original accepted field. These are proposed budgets, not
reported successes. Curvature needs its own scale-aware convergence envelope;
normal error alone does not validate subtle curvature. Set that envelope,
quadrature uncertainty, dynamic accuracy budgets and GPU resource limits
before using their captures as acceptance evidence. Keep oracle uncertainty
separate from candidate error, and treat unresolved reference uncertainty as
an inconclusive result.

## Compiled performance strategy

Use the existing accepted native geometry as the adapter input: world origin,
span, rung, valid dimensions, native cell bounds and solid/open geometry.
Do not expand macro cells into a finest-grid donor volume. The production
TEI2 leaf/packet descriptors already carry generation, first cell, native
scale and strides; presentation stages their directory and integrates native
donors. Reuse that authority rather than inventing a second owner resolver.
These are execution precedents, not proof that their current density
reconstruction satisfies this roadmap.

Compile semantic donor lists at topology/support changes. The existing stable
leaf face-neighbour compiler handles sparse dyadic coverage and holes without
materializing a macro face. Its face-only relation is not automatically a
quadratic reconstruction stencil: derive any required edge/corner or wider
support explicitly, validate support rank, and deduplicate native donor IDs.
Keep topology discovery outside steady-state coefficient application.

Separate reusable geometry patterns from instances, following the interned
boundary operator template/instance design. A pattern key must include ordered
relative box geometry, rung/span relations, clipping/open moments, basis and
weighting policy; scalar values do not belong in that key. Normalize equivalent
translated/scaled patterns only with a verified coefficient transform. Retain
native IDs and transforms per instance. Do not assume arbitrary cut geometry
interns effectively: measure unique patterns and worst-case fallback bytes.

The first compiler stores 10 times donor-count float64 weights per packet.
A proposed float32 expanded map alone costs 40 times donor-count bytes plus
4 times donor-count ordinal bytes, before headers, frames, outputs and reverse
dependencies. Compare interned maps with implicit regular-interior arithmetic
and explicit exceptional stencils. Count pattern storage, instance storage,
compile scratch and simultaneous old/new generations. No performance win is
claimed by the current algebra.

Build reverse dependencies from changed donor/material records to affected
coefficient records, physics integration receipts and presentation pages.
Topology changes invalidate maps; density changes apply retained maps where
the selected update model requires reconstruction. Physics-only edits change
coupling integrals without refitting q. Moving solids invalidate clipped
moments and their dependents even if the topology generation is unchanged.
Carry topology, field, boundary and support generations explicitly.
Use final scalar packet masks as an existing source of dirty facts, not as
proof that a new dependency closure is complete. Compare dirty execution to
a full recomputation oracle, including newly wet support and no-op edits.

Benchmark compile versus apply separately and together, at zero, local and
100% churn. Count donor reads, multiply-adds, mask work, lookups, allocations,
launches and cache traffic, plus accuracy at the same retained resolution.
Use broad ocean-like scenes for memory traffic and mini scenes for launch
cost. The compiled-topology handoff records a rejected direct TEI face sampler
that was slower than a derived contiguous cache; compiled metadata alone does
not guarantee lower latency. Keep or add bounded derived caches only with
paired end-to-end evidence and accounted memory. Avoid claims copied from
the handoff's explicitly theoretical speedups.

Publish field/support, coupling and presentation generation references through
reserve/validate/commit/retire. Failed validation, reserve exhaustion and
superseded edits preserve the complete prior accepted generation. A local edit
must have bounded affected support; increasing empty world extent must not
increase fine-volume work. Verify both requirements with receipts.

## Regression cadence and concrete next milestone

The next milestone is A/B algebra and reconstruction evidence plus the C
assembly design, with independently evaluated example captures. It does not
authorize production appearance changes. Record current source and baseline
receipts before the first production vertical slice. During CPU-only research,
run a core-scene check at completion of global-field/zero-time work and again
after prescribed transport, rather than after every algebra edit. Coordinate
those checkpoints with other work; a documentation-only update need not run
the GPU suite and must not claim a new result.

After every large Sparse CM12, topology, terrain/boundary, live-edit or
publication change, and before a production cutover claim, run the full gate:

```sh
npm run test:dawn:sparse-cm12
```

Use `-- --list` to inspect its current matrix and `-- --lane=<id>` only for
diagnosis. The full matrix remains required at the mandated checkpoints.
Do not weaken lanes or raise timing ceilings. Run Dawn serially under the
repository WebGPU lease, with browser simulation unloaded and no other Dawn
process. Perform browser inspection afterwards. Record source revision, dirty
tree state and baseline failures; a core regression pass validates retained
production behaviour, not the new representation's independent surface error.

## Inspected implementation anchors

- `tools/implicit-density/field.ts`: `meanBasis`, `rebase`, `positiveAffineMean`,
  `mean`, `clampedAffineMean`, `splitField`.
- `tools/implicit-density/compiled-stencil.ts`: `compileStencil`, `applyStencil`,
  `packetCost`.
- `lib/methods/adaptive-mass/sparse-cm12-transport-execution-image.ts`:
  `SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF` and packet layout.
- `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts`:
  `presentationLoadLeaf`, `presentationStageTopology`,
  `presentationCompiledOwnerCellAt`, `presentationLeafMass`.
- `lib/methods/adaptive-mass/sparse-cm12-factored-aei-topology.ts`:
  `compileSparseCM12StableLeafFaceNeighbors`, `neighborLeavesByLeaf`.
  `tests/sparse-cm12-stable-leaf-neighbors.test.ts` supplies an independent
  box-overlap oracle and a million-brick macro-face fixture.
- `lib/methods/adaptive-mass/sparse-cm12-interned-boundary-operators.ts`:
  `SparseCM12InternedBoundaryTemplate`, `SparseCM12InternedBoundaryInstance`.
- `lib/methods/adaptive-mass/sparse-cm12-final-scalar-packet-masks.ts`:
  FSM1 generation/topology header and final scalar mask ownership.
- [Compiled topology handoff](sparse-cm12-compiled-topology-handoff.md):
  implementation review and “First follower: face preparation”; distinguish
  measured results from the subsequent proposed architecture and estimates.
- [Canonical regression policy](SPARSE_CM12_DAWN_REGRESSION.md).

Literature scope and primary references are already recorded in the architecture
proposal. This roadmap adds no new literature-derived algorithm claim.
