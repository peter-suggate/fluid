# Level-set + volume in 3D: Sparse Geometric implementation handoff

Date: 2026-09-14. Target: `adaptive-volume`, the application’s Sparse Geometric method.

The implementation checkpoint below supersedes the original prospective plan retained in sections 1–9. Implementation started from `d6e167919316a696a146199f408d03cec15ac904`; the earlier analysis inspected `cc5251396fc5c9cd2d3e9e7e63f57540bd5beb17`. Existing 2D work was preserved.

## Implementation checkpoint — 2026-09-14

The immediate cutover is implemented in `adaptive-volume`. Phi lives on shared adaptive vertices with constrained hanging vertices and two topology-generation slots. There is no full-fine simulation phi field. Pressure, velocity-extension seeds, presentation and surface planning consume this field. Whole-frame RK2 translated-box coupling transports extensive V with three balancing rounds, measured excess, source/outflow accounting and V-only sharpening. The old FCT, low-flux limiter, split transport, resident PLIC/RDF caches and continuation loop have been removed.

The implementation includes authored geometry seeding, refine/coarsen and replacement-buffer phi transfer, bounded closest-point redistancing, signed-gradient frontier extension, wall-tangential characteristic handling, live liquid CSG and admitted continuous-inflow CSG. Oversized topology candidates retain the accepted generation while the host prepares a larger adaptive phi arena. Per-frame phi dispatches use the active vertex count.

Validation completed during implementation:

- Focused CPU numerical/layout/transfer/source tests: 32 passed in the latest combined run; additional stage, source and geometry contract tests passed separately.
- Dawn authored geometry: six analytic cases, 3,072 samples, passed.
- Dawn adaptive phi: mixed-rung plane, hanging constraints, repeated transfer, redistancing, underestimated-distance recovery and coarse H16 metric-band preservation passed, including a final pre-commit rerun.
- Production uniform translation: three frames at Courant 2 with sharpening enabled passed all existing analytic criteria; relative volume-field L1 error stayed below `5.6e-16`.
- Production mini32: four simulated seconds passed the volume/finite-field test; process duration was 30.7 seconds, exceeding the canonical lane's unchanged 25-second timeout.
- Sequential mini32 timing diagnostics on the same backend and Node version measured 113.64 ms median advance for this implementation and 380.63 ms for the preceding implementation in an isolated checkout. These are single-run diagnostics, not a qualified performance claim. Both exceed the unchanged 40 ms ceiling. The new transport stage measured 5.64 ms; face preparation and candidate topology processing remain major costs.
- Scoped TypeScript checks and diff whitespace checks passed. The repository-wide TypeScript command still reports pre-existing errors elsewhere.

**This is an implementation checkpoint, not production sign-off.** The canonical full Dawn matrix has not been run after integration. Earlier focused live-liquid and rigid runs found phi capacity failures; the admission/growth fix passed the subsequent sustained mini32 run, but those live-edit lanes still need rerunning. Run the complete unchanged matrix at the next integration boundary, then diagnose only failed lanes. Do not raise timing ceilings or relax numerical criteria.

Remaining numerical and performance risks are explicit: translated boxes approximate deformed 3D footprints; non-affine H uses bounded quadrature; sharpening skips cut cells, fractional apertures and uncertain face connectivity; disconnected vertex populations can exhaust the current vertex headroom and fail closed. Sustained curved-surface/seam quality, true component conservation, live solids/sources, mini64, and measured adaptive memory/work scaling still need broader GPU validation. The estimator-disagreement receipt is not a bound on H integration error.

Reproduction logs and JSON receipts from this session are under `/tmp/lsv3d-*`; these temporary files are not committed. The new numerical fixtures and diagnostic tools are committed with the implementation. The original milestone table below is historical guidance; its shadow-mode and delayed-cutover suggestions were superseded by the user's immediate-cutover instruction.

## 1. Recommendation and intended result

Overhaul `adaptive-volume` around the current 2D method’s separation of responsibilities:

- **φ owns the free surface.** Transport a shared signed scalar field; derive pressure geometry, surface normals, presentation, and surface adaptivity evidence from that field.
- **V owns liquid amount.** Transport physical volume conservatively, allow measured temporary excess above open capacity, and sharpen volume toward the existing φ surface.
- **Pressure releases excess.** Use the current 2D capped, open-capacity-normalized expansion target, with consistent units and pressure-component compatibility.
- **The simulation remains adaptive.** Keep the sparse cell/face graph, coarse-first planning, GPU execution images, generation leases, and public SparseWorld integration.

**Adaptive-resolution φ storage is a starting requirement and a performance goal.** Store φ on the accepted adaptive brick/cell hierarchy from the first prototype, including coarse free-surface cells. Do not introduce a finest-resolution surface band as an intermediate production architecture. Storage, advection, redistancing, and sharpening must benefit from coarsening; reducing pressure unknowns alone does not satisfy this requirement.

The proposed starting discretization is **adaptive vertex φ with constrained hanging vertices**, using the simulation’s accepted cell rungs. This keeps vertex sampling familiar from 2D while changing its resolution and storage. A shared, continuous world-space reconstruction supplies every consumer. The seam reconstruction and refine/coarsen transfer are numerical work to prove in the first milestone, alongside their cost. Uniform-fine fields are comparison oracles only.

**Execution decision (user update): immediate cutover inside `adaptive-volume`, deleting obsolete production code as it is replaced. Do not maintain a legacy transport arm or a runtime migration switch. Use Sol medium subagents for bounded implementation work. Run focused Dawn tests while iterating, then the canonical regression gate at the completed integration boundary; do not repeatedly run the whole matrix.**

The largest uncertainties are adaptive φ reconstruction/redistancing and detail loss, long-step 3D volume weights, sparse support coverage, and component-safe sharpening. Resolve those before investing in the full UI cutover. Allow **40–70 focused engineer-days** provisionally, including the adaptive-surface prototype, validation and removal of obsolete paths. Re-estimate after M1–M2: interpolation quality and cost are not established, and a failed gate must be resolved within the adaptive-storage requirement.

## 2. Pre-cutover source assessment

### 2.1 The 2D reference to preserve

The implementation has moved beyond several research documents written earlier on the same date. Use current code and newly fingerprinted captures as the behavior authority.

| Current behavior | Source and consequence for 3D |
|---|---|
| φ is stored on a full-domain finest **vertex** lattice, advected with RK2 departure sampling. | [`levelset_volume.rs`](../../rust/crates/fluid-core/src/levelset_volume.rs), `advect_shared_phi`. Preserve the shared-coordinate semantics, but replace the uniform lattice with adaptive samples and explicit seam constraints. This changes spatial approximation; it is not a byte-for-byte surface port. |
| The vertex field is redistanced near the contour; cell φ is subsequently sampled from contour distance. | Same file, `redistance_vertices` and `advance_with_fine_capacity`; [`levelset_redistance.rs`](../../rust/crates/fluid-core/src/levelset_redistance.rs). A 2D segment-distance BVH is not a ready 3D GPU implementation. The vertex and cell-distance samples also need not be identical away from the interface. |
| V uses corner-and-centre RK2 footprints, four-triangle clipping against donor boxes, three capacity-balancing rounds ending with donor normalization. | `trace_rk2`, `raw_weights_from_footprints`, `balance_capacity_marginals`. Donor conservation does not imply exact receiver capacity or a globally non-overlapping departure partition. |
| Invalid footprints fall back to a translated box; uncovered donors get a self edge. | `footprint_triangles`, `raw_weights_from_footprints`. These preserve usable weights/conservation but can introduce diffusion or stationary residue. Add explicit counters in the port. |
| Sharpening moves **only V**, within nearby φ components, toward φ-implied capacity. φ is immutable during this operation. | [`levelset_sharpening.rs`](../../rust/crates/fluid-core/src/levelset_sharpening.rs), `sharpen_volume`. The earlier proposal for moving φ with regional multipliers is not the implemented behavior. |
| Pressure membership/geometry and extension use φ; excess expansion is temporarily added to the physical source. | [`world.rs`](../../rust/crates/fluid-core/src/world.rs), `with_level_set_volume_pressure_source`; [`numerics.rs`](../../rust/crates/fluid-core/src/numerics.rs), `extend_velocity_with_level_set`, `level_set_volume_excess_pressure_source`. Preserve pressure/source lifecycle as well as the formula. |
| Topology transfer explicitly permits excess; resolution planning uses the direct surface and permits vacated support to coarsen. | [`transfer.rs`](../../rust/crates/fluid-core/src/transfer.rs), `transfer_fields_allow_overcapacity`; [`resolution.rs`](../../rust/crates/fluid-core/src/resolution.rs). Excess must survive re-rung and page lifecycle operations. |
| The working tree gives face advection an immutable source bank. | [`numerics.rs`](../../rust/crates/fluid-core/src/numerics.rs), `prepare_faces_for_level_set_volume`; [`energy investigation`](level-set-volume-energy.md). Preserve this property. 3D already has source/destination face banks; audit parity and consumers rather than assuming it has the same bug. |
| Live liquid drops are supported, but the LSV frame rejects rigid bodies and inflow sources. | [`levelset_surface.rs`](../../rust/crates/fluid-core/src/levelset_surface.rs), `union_drop`; [`world.rs`](../../rust/crates/fluid-core/src/world.rs), `advance_frame`. Those 3D features require new integration work, not a mechanical port. |

The latest [energy investigation](level-set-volume-energy.md) reports sustained runs and improved impact motion after immutable face advection, but also two existing LSV regression failures: `zero_velocity_is_identity_on_mixed_adaptive_cells` and `small_translations_cross_fine_coarse_seam_in_both_directions`. These were not rerun for this handoff. Reproduce and resolve or isolate their cause before treating the 2D implementation as an exact transport oracle. Do not encode a known failure into new 3D acceptance thresholds.

The [original lab results](level-set-volume-lab-results.md) and [earlier 3D assessment](level-set-volume-3d-transfer-assessment.md) remain useful historical risk records. Their claims that sharpening or vertex redistancing is absent are superseded. Their old timing numbers and φ-area loss are not measurements of today’s working tree. The current sharpening still cannot correct total φ-enclosed-volume drift: redistributing V with φ held fixed cannot change the volume enclosed by φ. That remains a quantity to measure, not a reason to silently add surface motion.

### 2.2 The 3D method being replaced

[`lib/methods/index.ts`](../../lib/methods/index.ts) selects `adaptive-volume` as the default. Despite inherited `AdaptiveMass`/`CM12` class names, it has its own resident and adapter.

The table below records the pre-cutover implementation that this handoff was
written to replace. Its FCT, microstep, and volume-derived interface references
are historical migration context rather than descriptions of the current path.

| Subsystem | Current authority | Migration implication |
|---|---|---|
| Volume transport | [`resident-volume.wgsl.ts`](../../lib/methods/adaptive-volume/resident-volume.wgsl.ts): geometric FCT with low-flux limiting, capacity constraints, and microsteps. | Replace the transport model; do not retain a hard `V <= C` validator on the new excess-carrying state. |
| Host transport schedule | [`webgpu-sparse-cm12-resident.ts`](../../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts), `conservative-transport` stage. | Repeated interface/flux/limiter/commit dispatches plus continuation readback are a concrete cost-removal opportunity. Direct callers encode 512 packets; continuation mode encodes chunks and reads progress. Encoded packets are not the same as executed microsteps. |
| Interface | Retired resident V/C, PLIC, and RDF reconstruction module. | Replaced by independently transported adaptive φ; the resident module and its cache-specific tests were deleted at cutover. |
| Pressure | [`webgpu-sparse-cm12-resident.wgsl.ts`](../../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts), `pressureCellMembershipFromDensity`, `classifyPressureRow`, PEI hooks. | Membership uses density and submerged/moving/source exceptions; θ uses geometric/height information. Replace membership and geometry together, including cache invalidation. |
| Presentation | Same file, `cm12PresentationExactSample`. | Currently selects cached RDF, column height, fine observations, or coarse interpolation. Replace numerical surface selection with samples of accepted φ. Retain page publication, dirty tracking, bounds, and consumer leases. |
| Generation transfer | [`sparse-cm12-generation-transfer.ts`](../../lib/methods/adaptive-volume/sparse-cm12-generation-transfer.ts), plus in-place shader transfer. | Current device transfer validates capacity bounds and reconstructs from amount/normal. Both transfer routes need an explicit excess-preserving policy. |
| Solids and sources | [`geometric-solid-motion.wgsl.ts`](../../lib/methods/adaptive-volume/geometric-solid-motion.wgsl.ts), [`geometric-source.wgsl.ts`](../../lib/methods/adaptive-volume/geometric-source.wgsl.ts). | Preserve capacity/velocity/source accounting while replacing assumptions tied to FCT microsteps. |
| Public integration | [`adaptive-volume-adapter.ts`](../../lib/sparse-world/internal/adaptive-volume-adapter.ts), [`webgpu-adaptive-mass-solver.ts`](../../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver.ts). | Preserve reset, edit, scene, diagnostic, and generation behavior; update receipts and stage attribution. |

The resident layout currently binds **ten storage buffers**, plus uniforms. Do not assume another storage binding is available. Extend existing arenas with explicit offsets or give isolated LSV passes a smaller dedicated layout. Check actual device limits and per-buffer sizes during planning.

There is 3D level-set infrastructure under [`octree-shared`](../../lib/methods/octree-shared/webgpu-octree-fine-levelset-redistance.ts), including sparse JFA redistancing with local repair and validation. Its fine-lattice topology and JFA strides do not implement adaptive-spacing redistancing. Reuse neutral validation/seed concepts only after checking their assumptions; do not rasterize the adaptive field to that fine lattice to reuse its solver. Shared consumer ABIs in [`fine-levelset-brick-abi.ts`](../../lib/core/fine-levelset-brick-abi.ts) and [`levelset-consumer-abi.ts`](../../lib/core/levelset-consumer-abi.ts) remain useful presentation boundaries. Do not instantiate another solver or import a sibling method to obtain level-set machinery.

## 3. Numerical and storage contracts

### 3.1 Units and authority

Use the resident’s existing finest-lattice units internally, with explicit conversion at physical/public boundaries. Define:

```text
cellMeasure_i = hx_i hy_i hz_i
C_i = openFraction_i cellMeasure_i
V_i = storedDensity_i cellMeasure_i
f_i = V_i / C_i, for C_i > 0
φ < 0 means liquid; φ is a length, never a density proxy
H_i = volume of {φ < 0} intersected with open cell i
```

Storing `V/cellMeasure` in the existing density bank is acceptable; the semantic authority is extensive V. Keep state nonnegative and finite. Excess `max(V-C,0)` remains represented until conservatively moved or explicitly removed by an authorized source/sink. Positive V in zero open capacity requires evacuation or failure; it cannot simply be divided by epsilon.

Compute H using the same surface reconstruction used by pressure/presentation, intersected with the accepted solid representation. Multiplying averaged open fraction by averaged liquid fraction generally does not equal their intersection. The 2D code itself has differently approximated diagnostic and sharpening targets; settle the 3D target once and use it everywhere. Use bounded leaf-local integration of the adaptive reconstruction, with explicit geometric error estimates and finer integration only where necessary. Reuse the voxel solid authority at cut boundaries; do not enumerate every finest liquid voxel inside a coarse leaf. Validate against analytic plane/solid intersections and a small high-resolution oracle, and report integration work/error separately from storage resolution.

### 3.2 Adaptive φ storage, reconstruction, and support

**Storage.** Store f32 values at vertices of the accepted adaptive cells, with one owner for each independent world-coordinate sample. Tie φ spacing initially to the simulation cell widths, including macrobrick span. A brick with `r³` cells has at most `(r+1)³` local corner slots before cross-brick sharing: an 8³ brick has 729 and a 1³ brick has 8. Hanging vertices and ghosts are derived values with explicit dependency records, not additional independently evolved fine detail. Compact allocation and active dispatches must follow the selected rung; a fixed 9³ payload per coarse brick would defeat much of the memory goal.

Keep signed sparse coordinates for floor-only/outside-tank scenes. Encode vertex centring, physical origin/width, ownership, validity and generation in the sampler contract. Existing cell-centred presentation samples may be generated from this field; they must not be reinterpreted as simulation vertices or fed back into it.

**Initial reconstruction candidate.** Use trilinear reconstruction within each leaf, with constraints on hanging vertices so neighboring leaves share the same scalar trace on their common face:

1. Same-resolution cells share corner values by canonical ownership.
2. A fine-side vertex on a coarse face takes the coarse face’s bilinear value; one on a coarse edge takes its linear value. Dependent samples are not advected independently.
3. Compile constraint dependencies and evaluation order at topology acceptance, including edge/corner junctions and every supported rung ratio. Resolve a junction once by geometry, not by whichever cell queries it. Reject conflicting or cyclic constraints.
4. Evaluate `Φ(x)` through one owner lookup and the accepted local polynomial. All surface consumers use this definition, including cell-centre pressure classification. Cache constraints and local sample indices so normal sampling does not run a neighbor search or small least-squares solve every time.

This is a proposed discretization to validate in M1, not a claim that the repository already supplies it. Restricting a bilinear coarse-face polynomial to a fine subface gives the same trace when its corner constraints agree; M1 must establish that the full adaptive hierarchy consistently satisfies those constraints. Plain independently populated trilinear leaves would not do so.

The target is a single-valued **C0** scalar across seams. Normal derivatives can still jump, and coarse traces discard detail that the fine side alone could represent. Measure scalar continuity, zero-crossing position, normal jumps, and shape loss separately. Do not claim that continuity preserves sub-coarse ripples or thin sheets. Treat insufficient detail with selective, error-driven refinement under the accepted quality/region limits, not a universal finest-surface floor. If this candidate fails, evaluate another bounded adaptive reconstruction; a uniform-fine surface is not the fallback architecture.

**Advection.** Trace each independent destination sample through the old accepted velocity field and evaluate the old adaptive `Φ` at its departure. Regenerate dependent samples from constraints after writing the destination bank. Sampling support must remain valid as traces cross rungs, and constraint projection itself can alter transported geometry near a seam. Record its contour/volume effect separately from interpolation and redistancing. The 2D uniform field remains a quality reference, not the adaptive source used in this test.

**Support.** Keep metric samples around the interface and required pressure connections, valid scalar/phase support at departures, and tagged deep-liquid/deep-air continuation outside that region. Allocate additional support at an appropriate adaptive rung. An absent leaf is unknown unless sparse-world evidence proves its phase. A saturated phase value may classify liquid but is not a valid distance for θ or a normal.

Bound reach in physical space using `Ubound Δt`, plus interpolation/redistance neighborhoods measured with the local leaf widths. Traverse adaptive leaves intersecting that envelope; do not convert the envelope into a finest-resolution voxel apron. A few sampled liquid-centre velocities are not automatically a bound on the interpolated extension. Check all RK stages and departure interpolation supports. Pressure connections across coarse cells must either have metric endpoint support or obtain crossings directly from `Φ`. Long-step tracing needs valid velocity along the path and old scalar at departures; these do not require identical storage domains.

**Redistancing.** Implement an adaptive-spacing operator from the start. Seed crossings from the canonical reconstruction on shared physical edges/subfaces, then prototype bounded sparse closest-point propagation and local repair using physical distances and compiled adaptive neighborhoods. Compare against explicit surface-distance queries on small CPU cases. A nearest-seed result is only an approximation unless its surface geometry and search coverage are certified; track distance error, missing support and contour movement. Avoid a per-frame global production triangle BVH.

Do not port uniform-grid JFA strides or uniform-spacing Eikonal updates unchanged. Reapply hanging-node constraints after each independent-value update, and verify that constraint enforcement and redistancing do not fight each other, move a resting plane, or introduce seam oscillation. Freeze appropriate crossing information during repair, measure convergence and bound passes/work. Fine-side propagation can take more graph hops than coarse-side propagation for the same physical distance; include mixed-rung long interfaces in the early GPU cost gate.

**Transfer and quality.** Refinement samples the old reconstruction at new independent vertices; coarsening fits/restricts the old field with an explicit contour/detail error test, then rebuilds constraints. φ is not volume-averaged and is never reconstructed from V. Check affected neighboring leaves too: changing a coarse face changes its fine-side constraints. Repeated refine/coarsen cycles can erode a curved surface even with zero velocity. Measure that error and preserve exact planar fields within numerical tolerance; veto excessive coarsening when allowed by authored limits, and report unresolved detail when a hard coarse limit prevents refinement. Coarse storage cannot recover a feature that was already discarded.

### 3.3 Whole-frame volume transport

The new method should take one material step over the requested frame, without a global material microstep count proportional to maximum Courant. RK trajectory evaluations are a distinct cost and must be counted honestly. If they require bounded local subdivision for accuracy, report that separately; do not call the overall work independent of Courant.

Build nonnegative sparse receiver/donor weights `T_ij`. Begin with the current 2D capacity balancing semantics:

```text
three rounds: scale receiver rows toward C_i, then donor columns toward C_j
finish with donor normalization
V'_i = Σ_j T_ij (V_j / C_j)
donor error_j = Σ_i T_ij - C_j
receiver error_i = Σ_j T_ij - C_i
```

The donor condition conserves mass; receiver residual can remain nonzero and creates excess or dilution. Dry open donors matter to capacity balancing. No fixed number of scaling rounds guarantees both marginals for arbitrary support.

For 3D footprint weights, implement two CPU-reference variants before selecting the GPU kernel:

- A rigid translated box as the inexpensive control and explicitly counted fallback.
- A fixed, face-consistent tetrahedral decomposition of traced receiver geometry, clipped against donor boxes. Trace canonical shared vertices and required interior samples; choose shared-face diagonals deterministically. Check orientation/degeneracy before accumulating **positive** overlap weights.

Select the tetrahedral variant only if it materially improves mismatch/front behavior within the work budget. A warped hexahedron is not a convex box, and individually valid tetrahedra do not certify a global departure partition under nonlinear flow or mixed-resolution tracing. This operator is a conservative normalized coupling, not the earlier exact geometric-remap project. Do not resurrect its unbounded mesh repair/streamfunction machinery.

Search donors through the existing sparse page/leaf directory and spatially bounded overlap lists. Build compact receiver CSR and donor adjacency once for the coupling; reuse them across normalization rounds. Never scan all donors for each receiver. Count overlap tests, accepted edges, maximum row degree, folds, and fallback-carried **volume**, not just cells.

For uncovered donors, initially preserve the 2D self-edge fallback and expose its amount/displacement. Include a forward-traced conservative remainder candidate if it fails the translating-front gate. A mass-conserving stationary tail is still a transport failure. Missing physical support is a planning error; a self edge cannot conceal a dropped source page.

GPU reductions must define their numerical contract. Do not assume portable float atomics or f64. Prefer deterministic gathers over compact adjacency with compensated/pairwise f32 accumulation where useful. Any correction is bounded by measured rounding error; do not turn the 2D global cast-rounding adjustment into a generic global mass repair.

### 3.4 Pressure, velocity, and excess

Publish cell/row pressure membership and cut distances from the same accepted φ generation. Derive θ from physical liquid/air distances along each subface’s pressure connection, retaining the existing operator’s area/dual-distance weights and guarded θ floor. Coarse/fine centre separation is not necessarily one cell width. Test planes crossing faces, edges, and corners, including nearly tangential interfaces.

Use the constrained adaptive reconstruction from M1 as the geometric definition. If rendering samples it trilinearly while H uses tetrahedral clipping and pressure uses a local fitted plane, those are different approximations even when they share vertex values. Measure and bound their crossing/volume discrepancy, or route critical consumers through the same reconstruction. A common buffer alone does not establish geometric agreement. In particular, derivatives and θ must use physical cell widths; equal φ values have the same length units on every rung.

A cell-centre sign test can miss a liquid region within a coarse cell, and coarse vertex samples can miss an entire subcell feature. Use pre-coarsening detail evidence and authored insertion geometry to demand adequate resolution before losing it. For represented partial cells, require adequate pressure support or a documented partial-cell pressure representation. Do not infer that an all-positive coarse sample set proves a previously unresolved drop never existed. Report features that hard resolution limits prevent representing.

Invalidate PEI membership, θ, diagonals, component/nullspace data, and affected preconditioner caches when φ changes the pressure operator, even when topology generation is unchanged. Geometry dirty is not synonymous with topology dirty.

Use the 2D integrated expansion-rate target:

```text
q_i = min(0.5 max(V_i - C_i, 0), C_i) / Δt
0 <= q_i Δt / C_i <= 1
```

Convert it into the resident’s actual RHS convention with the correct sign. It is a per-step adaptation from the 2D implementation, not an unexamined paper transcription. Reapplying the same constraint after a topology projection does not mean injecting it twice; clearing it before that projection would cancel its effect. Keep physical source accounting separate and restore it after temporary pressure assembly.

Closed pressure components cannot realize arbitrary positive net expansion. Define compatibility handling: detect absence of a free surface/open outlet, retain unreleased excess, and report suppressed/incompatible source rather than forcing an inconsistent Poisson solve. Tiny open capacities/apertures can still produce large velocities despite the normalized cap; test actual pressure work and kinetic energy.

Keep immutable old velocity sources through advection and use φ-based seeds for extension. No surface correction should silently rescale velocity, add rebound forces, or reorder projection to match a paper. Evaluate stage-resolved momentum/energy diagnostics before changing ordering.

### 3.5 Volume sharpening

Port the **current V-only operation** first: φ remains immutable, donors/receivers remain within a local metric neighborhood, distinct liquid components do not exchange volume, and receiver room limits deposition. Preserve the ambiguity rule when a coarse cell overlaps multiple components. Report unassigned volume and unresolved eligible residual.

The CPU implementation builds full fine rasters, scans all cells for every donor, and computes component totals with repeated scans. None of those work shapes is suitable for sparse 3D. Label liquid fragments of the adaptive reconstruction, including connectivity through physical subfaces, then resolve boundary equivalences. Use compact band worklists and spatial receiver bins. No finest-grid component raster or H target is allowed in the production path. Distances, neighborhood kernels and thresholds must be expressed in physical units with explicit local-width scaling; do not count coarse and fine samples equally when integrating a physical quantity. GPU deposition must use staged proposals followed by receiver and donor limiting; independent donors cannot each spend the same receiver capacity. Validate order independence, mixed-rung bias and component conservation against a small deterministic CPU oracle.

A full liquid component can span deep coarse bulk outside the adaptive metric band. Resolve connections through a coarse graph of proven liquid regions as well as band fragments; labeling only visible surface patches can incorrectly split one liquid body. Conversely, do not join two drops merely because their air-side extension collars meet. Ambiguous connectivity inside a coarse reconstruction must trigger the chosen detail policy or an ambiguity receipt, rather than an invented component connection.

Track global and per-component `ΣV-H`, band L1 mismatch, and out-of-band residue throughout sustained runs. V-only sharpening cannot repair a shrinking φ surface. If the current behavior’s mismatch becomes unacceptable in 3D, diagnose advection/redistance first. Moving φ to enforce regional volume would be a separate numerical change requiring its own shape/energy acceptance, not an automatic completion of this port.

## 4. Frame and generation lifecycle

The concrete ordering below follows the 2D reference where practical; integrate it with existing resident phases rather than stacking an independent second frame pipeline.

1. Snapshot accepted topology, V, velocity, φ, solid/source state, and all parities. Stage queued edits and support changes with explicit source receipts.
2. Extend/advection-prepare velocity from immutable accepted sources using φ seeds. Apply forces and assemble/project pressure with φ geometry and capped excess source.
3. Bound transport reach using projected velocity. Allocate required sparse receiving/trace support and conservatively transfer simulation fields if topology changes. Rebuild pressure data and reproject if required by the existing transition contract.
4. Seal one source generation for V and φ transport. Build/normalize sparse coupling and gather candidate V over the full Δt; advect independent adaptive φ samples from the immutable old reconstruction using the same physical velocity convention, then evaluate hanging-node constraints.
5. Redistance candidate φ on the adaptive support, enforce and validate its constraints, derive H and component/band worklists, and sharpen candidate V toward that immutable surface. Publish transport, constraint, redistance and sharpening mismatch receipts separately.
6. Compute surface/adaptivity demands from candidate φ plus mass liveness and physics accuracy. Transfer any final re-rung transaction without rebuilding φ from V. Check φ restriction/constraint error on affected leaves and neighbors; preserve excess and velocity/face flux contracts.
7. Validate candidate state, support, source ledger, and generation consistency. Atomically publish the accepted numerical state and renderer generation; advance time only for a completed frame. Retain prior render generations until readers release them.

Planning has to cover newly seeded liquid and moving-solid effects before they are sampled. If a post-projection bound exceeds provisioned support, perform a bounded support transaction or fail the frame; never proceed with invalid samples. Empty/new pages need phase proven from old φ or authored geometry, not reconstructed from diffuse V.

Audit existing failure behavior before promising rollback. First-fault halting alone is not atomic rollback if accepted arrays were already overwritten. New LSV stages should write candidate banks; any reused in-place velocity/source/solid stage must either be staged too or support complete restoration. Fault injection should prove that failed frames do not advance time, consume sources, or publish mixed generations.

Generation transfer needs two coordinated policies:

- **V:** distribute representable base volume using valid φ-derived geometry/open overlap; distribute excess conservatively over open children. Re-restriction sums amounts. Zero-capacity targets receive none; unavailable evacuation capacity defers/faults explicitly.
- **φ:** prolong/restrict the accepted adaptive reconstruction independently of V and rebuild shared-sample constraints. Re-rung now changes the surface approximation: require planar reproduction and bounded curved-surface/detail error rather than asserting exact identity for arbitrary fields. Account for transfer-induced φ-enclosed-volume change separately from conserved V. Sample allocation, retirement, constraints, ghost exchange and renderer publication are part of the same versioned transaction.

## 5. Solids, sources, and presentation are part of completion

**Terrain and closed/open boundaries.** Reuse the accepted voxel/cut-capacity authority. Prevent traces or sharpening transfers through solid barriers. Domain clamping alone does not handle interior terrain. φ continuation used for sampling must be distinguished from visible liquid inside solids. Test contact-line height, tangential flow, tiny apertures, clipped leaves, open-top exits, and the floor-only world.

**Moving rigid bodies.** The current FCT schedule interpolates solid capacity and commits motion across microsteps. Whole-frame V transport needs an equivalent geometric conservation law: capacity changes, wall swept volume, relative fluid/wall flux, newly opened cells, and closing-cell evacuation must agree. Preserve rigid force/impulse exchange and the directed closing-component feasibility checks. Do not retain the old microstep ledger while pretending a one-step transport advances it correctly. This is a major work package and a release blocker for replacing the default.

**Sources and edits.** Live injection updates V and unions authored geometry into φ exactly once, using consistent overlap/solid clipping and velocity initialization. Continuous inflow needs a per-frame emitted volume ledger and a φ seed that follows the same source footprint. Removal/reset must clear φ pages, component labels, and cached pressure/presentation state. Test repeated paused edits, edits after advance, capacity exhaustion, stopped sources, and replay/cancellation without duplicated emission.

**Presentation.** Preserve the public consumer lifetime protocol and use the existing sparse page ABI where it fits. Route distance samples, bounds, normals, and mesh/ray consumers through accepted adaptive φ. Fine display samples may be derived for dirty/visible pages, but are disposable presentation data, never a full-fine simulation authority or hidden redistance/sharpening workspace. Measure their cost separately; prefer rung-aware publication or direct adaptive sampling if fixed-resolution publication erases the intended frame savings. Remove volume-derived column-height/PLIC overrides once φ publication is validated. A smooth image cannot compensate for pressure and rendering observing different zero sets. Quantify packing/resampling error and normal jumps at both storage pages and adaptive seams. Shading-normal smoothing, if used, must not conceal a displaced or discontinuous geometry field in the tests.

**Adaptivity.** Replace density-derived surface/curvature evidence with φ-based evidence in [`coarse-first.wgsl.ts`](../../lib/methods/adaptive-volume/features/adaptivity/coarse-first.wgsl.ts) and its prediction/packing/planning consumers. Keep authored limits, physical cell widths including macrobrick span, hysteresis, and symmetric admission. Coarsen both φ and simulation cells where the error criterion allows; no automatic finest-surface floor. Use pre-restriction contour/normal/detail evidence, since a coarsened field cannot diagnose all detail it already lost. Mass residue can keep a page alive without pinning its samples fine forever. Test coarse φ surfaces over fine interiors deliberately, across stationary and moving seams. Separate fixed-topology tests of representable coarse detail from adaptive tests of whether refinement is selected in time.

## 6. Performance plan and stop conditions

### 6.1 Measure the cost being exchanged

Let `N` be active simulation cells, `F` physical subfaces, `P` stored adaptive φ samples including ghosts, `S` constraint entries, `E` nonzero transport couplings, `L` local sharpening proposals, and `k` executed old transport microsteps. Report independent, constrained and ghost φ samples separately.

```text
old transport ≈ k × (interface fitting + FCT/limiter passes over N,F)
new transport ≈ tracing + overlap construction + fixed balancing passes over E
new surface   ≈ advection/redistance over adaptive samples + constraints over S
                + labeling + sharpening over L + bounded H integration
frame         = transport + surface + pressure + topology + publication + host waits
```

Both removal of k-dependent transport passes and reduction of φ storage/sample work through coarsening are intended gains. The core surface state should scale with represented adaptive samples and bounded seam/support metadata, not represented finest volume. Pressure may get cheaper with coarse surface cells, or more expensive because the changed interface/motion changes membership and iterations. Whole-frame measurements decide; no speedup is claimed here.

| Risk | Required implementation response | Evidence before proceeding |
|---|---|---|
| Hidden finest-resolution work | No full-domain or fine-band φ/owner/component raster; no routine enumeration of a coarse leaf’s finest children. Allocate and dispatch by accepted adaptive rung. | Empty-domain expansion leaves cost approximately fixed; coarsening fixed physical coverage lowers allocated φ bytes and sample work. Audit H, labeling, redistance and renderer resampling too. |
| Band growth at high Courant | Separate source scalar support, target metric support, and velocity trace support; traverse local adaptive reach. | Plot P/N, support-only samples by rung, peak bytes, allocation churn, and invalid samples versus physical travel and local Courant. |
| Constraint/halo cost cancels storage savings | Compact selected-rung payloads; precompile dependency indices; count every bank and leased generation. | Local core vertex slots fall from 729 at r=8 to 125 at r=4, 27 at r=2 and 8 at r=1 before sharing. Measure total bytes including constraints/ghosts; do not claim the ideal 8× cell-count saving for each coarsening step. |
| Seam reconstruction and derivative cost | Eight-corner leaf evaluation with precomputed constraints as the first candidate; bounded alternatives only after profiling. | Samples/second and lookup/constraint cost versus uniform and mixed-rung cases; scalar seam error, normal jumps and coarse-side detail loss. |
| 3D overlap explosion | Compact spatial candidates; fixed decomposition; degree/arena bounds and explicit overflow. | E/N distribution, overlap tests/edge, folds, fallback mass, and stage time on shear/impact scenes. A work cap must fault, not truncate donors. |
| Irregular gathers and redundant lookups | Reuse compiled owner/face information and compact adjacency; cache brick-local stencils. | Time lookup and gather stages separately; count page probes and bytes. Avoid repeated global neighbor discovery per normal or iteration. |
| Sharpening contention/quadratic scans | Local candidate bins, staged conservative proposals, compact component reductions. | L/Nband, label iterations, ambiguous/unassigned mass, receiver overspend count, stage time. |
| Adaptive redistance launch/traffic cost | Adaptive dirty/support lists and physical-distance propagation; bounded repair and constraint updates. | Pass count/hops by rung, seed coverage, max/RMS distance and contour error, stationary-plane stability and no-op cost. |
| Pressure degradation | Invalidate caches correctly; instrument membership, rows, iterations, true residual, excess-source work. | Rest and impact cost separately; no relaxed solve tolerance. |
| Dispatch/readback overhead | GPU indirect worklists; eliminate old continuation readback after FCT removal; reuse scratch with documented lifetimes. | Encoded versus executed dispatches, submissions, CPU encoding, waits, and GPU timestamps. Do not allocate/map a status buffer per numerical stage. |
| Peak generation memory | Budget accepted/candidate φ and simulation banks plus retained consumer generations. | Reset/edit/re-rung stress with peak allocated/live bytes and reclamation receipts. |
| Shader compilation cost | Keep kernels modular and layouts bounded; reuse device compilation cache. | Cold readiness, warm reset, pipeline count, generated shader size, and compilation failure records. |

The 2D `redistance_nanoseconds` receipt currently encloses more than pure redistancing, including subsequent sharpening/mismatch work. Give the 3D stages separate timers rather than copying misleading boundaries.

### 6.2 Benchmark protocol

Freeze scene JSON, source hashes, device/backend, timestep, solver tolerance/budget, adaptivity settings, presentation quality, warmup, and measurement duration. Run comparison arms sequentially on the same machine. Use hardware GPU timestamps for compute cost, plus end-to-end CPU wall time and publication/render cost separately. Report median and p95, startup separately, three alternating repetitions for decisions, and transient peak memory.

Benchmark both complete evolving scenes and frozen-input microbenchmarks. The former includes trajectory-induced topology/pressure changes; the latter distinguishes an implementation cost from changed physics. Include impact and settled windows, not only the first few frames.

The verified [Dawn manifest](../../tools/sparse-cm12-dawn-regression-manifest.ts) has 17 lanes, a 480-second suite budget, and these existing performance limits:

| Lane | Recorded reference median advance | Unchanged ceiling |
|---|---:|---:|
| mini32, B8 / presentation B8 | 24.576 ms | 40 ms |
| mini64, B8 / presentation B8 | 83.5584 ms | 110 ms |

These are recorded manifest references, not freshly measured baselines. The short performance lanes use three warmup and twelve measured frames; add sustained impact/settled measurements for this overhaul.

Proposed additional project gates, to lock after the baseline capture and before tuning:

- Preserve both existing absolute ceilings. Aim for new whole-frame medians no more than 10% above the matched current method on low-motion scenes, and at least 20% faster on scenes where old transport microsteps dominate. These are engineering targets, not promises or reasons to weaken correctness.
- No full-domain growth, no per-donor global scan, no unbounded repair loop. Empty scenes should perform no sample work beyond small control dispatches.
- On identical physical surface coverage with prescribed r=8/4/2/1 regions, demonstrate that coarsening reduces actual allocated φ bytes and advection/redistance sample work. Include a mixed-rung arm to expose seam overhead and report surface error at each rung. Do not pass this gate by keeping φ fine and coarsening only pressure.
- Demonstrate the benefit with φ rendering and sharpening enabled; a shadow φ benchmark alone cannot qualify the replacement.
- If adaptive interpolation/support or overlap/sharpening cost prevents these targets after local optimization, stop before default cutover and report the measured limiting term. Coarsening φ within the locked adaptive quality policy is an intended optimization; changing quality thresholds to hide errors is not. Do not substitute a finest-resolution φ band, smaller timestep, lower presentation quality or relaxed pressure tolerance for satisfying the agreed gates.

## 7. Implementation sequence and deliverables

Each milestone ends in a reviewable implementation, receipts, and a short decision note. Keep behavior switches internal to `adaptive-volume` until cutover. Avoid a broad rename/refactor while changing numerical authority.

| Milestone | Work and primary files | Exit gate |
|---|---|---|
| **M0 — Freeze behavior and contracts** | Capture current 2D/3D baselines; reproduce the two reported LSV failures; inventory live transfer/pressure/presentation routes. Specify adaptive sample ownership/centring, supported rung ratios, seam constraints, quality criteria, receipts and memory/work budgets. Add an internal comparison configuration. | Fingerprinted baseline artifacts, exact scene/settings manifest, resolved oracle limitations, locked thresholds, explicit adaptive-φ requirement. No default behavior change. |
| **M1 — Adaptive 3D numerical oracle** | Extend [`world3d.rs`](../../rust/crates/fluid-core/src/world3d.rs), which currently rejects non-baseline transport experiments. Add separate `levelset_surface3d.rs` / `levelset_volume3d.rs` reference modules or dimension-neutral primitives. Start with adaptive vertex storage, hanging-node constraints, prolongation/restriction, adaptive advection/redistance and H integration; then pressure, box/tetra weights, balancing and V-only sharpening. | First prove planar reproduction, face/edge/corner continuity, normal/detail error and stationary re-rung behavior on mixed rungs. Compare extruded 2D and true 3D motion at stated resolution error; select footprint strategy by fidelity/work. A small uniform-fine field is an independent comparison only, never the adaptive sampler’s hidden source. |
| **M2 — Adaptive GPU φ in shadow mode** | Add compact selected-rung φ storage, sampler, constraint compilation/evaluation, support planning, adaptive transport/redistance and transfer kernels. Wire arenas and generation ownership. Compare against the adaptive M1 oracle while old FCT drives the scene. | Centring/sign/units, mixed-rung seams, support coverage, transfer/redistance error, and measured memory/sample-work reduction with coarsening. Include all constraint/halo overhead. Shadow advection alone is not final numerical validation. |
| **M3 — φ authority cutover in the comparison arm** | Replace pressure membership/θ/extension seeds and publication sampling with the canonical adaptive reconstruction; adapt PEI cache invalidation. Use φ restriction/detail evidence in surface planning. Initially retain old bounded FCT for V to isolate this change. | Mixed-rung hydrostatics, coarse-φ-surface/fine-interior scenes, rendered and pressure zero-set agreement, fixed-generation moving interface, edit/reset publication. No blanket finest-surface floor. |
| **M4 — Whole-frame V and excess** | Replace FCT in the comparison arm with the M1 coupling, sparse adjacency normalization/gather, excess source, and excess-preserving in-place/device generation transfer. Add candidate-bank commits and overflow/fault receipts. | Mass/positivity, small translations, high-Courant fronts, cut-cell stability, transfer identity and zero-capacity handling. Old FCT capacity validators no longer reject legitimate excess. |
| **M5 — Adaptive V sharpening and sustained quality** | Implement adaptive fragment/component connectivity and staged conservative redistribution without fine rasters. Separate stage timings and H integration costs; optimize based on E/P/S/L work counts. | Component conservation, no receiver overspend, no surface change during sharpening, bounded measured mismatch/excess over sustained runs, no rung bias and performance targets with the full method enabled. |
| **M6 — Full production lifecycle** | Integrate rigid swept capacities and coupling, continuous sources, liquid/terrain edits, retirement, macrobrick transfer, outside-world pages, and fault rollback across adapters/solver/runtime. | All canonical lanes plus moving-wall/source/cancellation tests; no reset required for live insertion; no source double spend or mixed generation. |
| **M7 — Default cutover and removal** | Make the new arm the implementation of `adaptive-volume`; remove retired FCT continuation, volume-derived surface overrides, obsolete buffers/pipelines/controls and QA routing. Preserve valid comparison fixtures outside the production path. Update stage labels, method description and receipts. | Full unchanged Dawn gate after removal, focused numerical tests, sustained benchmarks, application visual review, and runtime import/unused-allocation audit. |

M0–M2 are the first decision boundary. Adaptive φ must demonstrate seam/transfer quality and reduced memory/sample work before broad pressure/solid integration. If its constraints, redistancing or geometry repair fail those gates, revise the adaptive discretization rather than switching to finest φ. M4–M5 are the second: a fast surface over poor or expensive volume transport is not sufficient. M6 is mandatory before replacing a default that already supports live rigid bodies.

Suggested new method-local module boundaries are `levelset-volume-layout.ts`, `levelset-volume-constraints.ts`, `levelset-volume-sample.wgsl.ts`, `levelset-volume-transfer.wgsl.ts`, `levelset-volume-transport.wgsl.ts`, `levelset-volume-redistance.wgsl.ts`, `levelset-volume-sharpening.wgsl.ts`, and `levelset-volume-receipt.ts`. These names are proposals; keep orchestration in the existing resident and SparseWorld lifecycle. Extract only reusable pure math/ABIs into shared code, with unchanged sibling-method tests.

## 8. Validation matrix

Validate more than global mass and a screenshot. Define analytic tolerances in physical or finest-cell units, and lock measured nonanalytic tolerances in M0. Float32 GPU results should be compared with a justified accumulation bound; do not copy CPU f64 thresholds blindly.

| Area | Required cases and observations |
|---|---|
| Adaptive reconstruction | Constant/affine fields, oblique planes at face/edge/corner junctions, every supported rung ratio and axis permutation. Check shared scalar traces from both owners, continuity of moving crossings, normal jumps, valid constraint dependencies and sample ownership. Force coarse surface storage in these cases. |
| Adaptive storage/performance | Identical physical coverage at r=8/4/2/1 and mixed rungs, plus larger empty world extents. Report independent/constrained/ghost sample counts, actual bytes by bank, constraint count, queries and redistance work, publication cost and geometry error. No finest-raster helper may supply production values. |
| Identity and transport | Zero velocity/zero Δt; uniform fill; arbitrarily small translations in both directions across each rung pair; diagonal face/edge/corner motion; axis permutations/reflections; Courant 0.1, 1, 5 and 10 on controlled fields. Measure donor mass, row defect, front/centroid displacement, fallback mass and invalid support. |
| True 3D deformation | Oblique slab, translating sphere, rotating ellipsoid, shear/vortex and reversal. Measure geometric volume, shape error, surface area, component count and overlap work. Extruded 2D alone cannot catch warped 3D footprints. |
| Hydrostatic and pressure | Flat/tilted surfaces, different subcell water heights, mixed rungs including a coarse surface over fine bulk, terrain contacts and closed components. Measure maximum velocity, divergence, true pressure residual, membership/θ consistency and waterline drift. |
| Topology and surface | Repeated refine/coarsen at zero velocity; macrobrick split/merge; clipped domain leaves; page retirement/reactivation; coarse cells containing two drops or a thin sheet. Require planar reproduction, measure curved-field zero-set/volume drift and hysteresis, and check constraint changes in neighboring cells. Test detail vetoes before erasure, hard coarse-limit diagnostics, mass/excess transfer and generation identities. |
| Surface quality | φ−V mismatch globally/per component, band/outside-band L1, thin sheets/filaments, holes and merging droplets. Separate advection, constraint projection, redistance and re-rung contributions to shape/volume error. Compare adaptive seam displacement and normal jumps separately from expected resolution-dependent detail loss against uniform-fine reference. A seam metric with zero eligible samples does not pass. |
| Solids and energy | Free-slip tangential flow, approach/rebound, narrow aperture, hillside dam, moving piston and closing/opening cells. Report K and P separately, pressure/source/rigid work, peak speed and force/impulse balance. Allow physical dissipation; disallow unexplained energy injection or a changed timestep as the fix. |
| Editing and failure | Live drop, first rigid insertion, continuous inflow start/stop, terrain change, repeated reset, insufficient page/edge capacity and injected shader faults. Check atomic acceptance, source ledger, first-fault provenance, time, and retained render leases. |
| Sustained scenes | mini32/mini64 through impact and settling; long-dam far wall; Tall Cells hillside; half-pool impact; floor-only collapse. Use at least the existing gate duration and additional 300-frame small-scene runs where practical; inspect mismatch trends rather than just the final value. |

Run the canonical gate after each large simulation, topology, publication, terrain, or edit milestone, and once more after removal:

```bash
npm run test:dawn:sparse-cm12 -- --list
npm run test:dawn:sparse-cm12 -- --lane=hydrostatic-adaptivity
npm run test:dawn:sparse-cm12 -- --lane=mini32-performance
npm run test:dawn:sparse-cm12
```

The first command only lists lanes. Do not run Dawn concurrently with a browser or another Dawn process; retain the repository WebGPU lease and isolated processes. Do not weaken a lane or raise a ceiling. `AGENTS.md` references `docs/SPARSE_CM12_DAWN_REGRESSION.md`, which is absent in this checkout; the runnable suite and manifest exist and were inspected. Use their current matrix and restore/document the missing guidance separately if needed.

Useful existing focused starting points include [`levelset_volume_hydrostatic.rs`](../../rust/crates/fluid-core/tests/levelset_volume_hydrostatic.rs), [`levelset_volume_sharpening.rs`](../../rust/crates/fluid-core/tests/levelset_volume_sharpening.rs), [`levelset_volume_injection.rs`](../../rust/crates/fluid-core/tests/levelset_volume_injection.rs), [`levelset_volume_wall_velocity.rs`](../../rust/crates/fluid-core/tests/levelset_volume_wall_velocity.rs), [`geometric-moving-low-flux-dual.test.ts`](../../tests/geometric-moving-low-flux-dual.test.ts), and the generation/presentation/edit tests named by the Dawn manifest. Add numerical GPU fixtures; text-pattern tests alone cannot validate a new discretization.

## 9. Handoff instructions for the implementer

Start with M0 and M1, with **adaptive-resolution φ from the first implementation**. The first surface deliverable must include coarse/fine reconstruction, refine/coarsen transfer and storage/work scaling; do not defer these behind a full-fine prototype. Preserve the existing energy changes, whether still in the working tree or subsequently committed. Record baseline receipts before changing the default or transport equations. Keep φ immutable under V sharpening; keep V independent under surface publication. Carry excess through every lifecycle route, and derive pressure/extension/presentation from one accepted adaptive φ generation.

Deliver each milestone with source/settings fingerprints, exact reproduction commands, stage costs and work counts, correctness receipts, known unresolved cases, and a pass/fail recommendation for the next milestone. Report new unsupported cases as blockers to cutover, not as silently skipped functionality. Delete the old production path as the replacement lands, per the immediate-cutover decision. Production readiness still requires the existing gate and the additional φ/volume quality tests; deleting a fallback does not establish correctness.

The original analysis was based on source/document inspection and a successful `npm run test:dawn:sparse-cm12 -- --list`. Subsequent implementation and measurements are recorded in the implementation checkpoint above; that checkpoint is the authority for current status.
