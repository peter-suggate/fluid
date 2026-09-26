# Uniform Geometric: manual mixed resolution recovery

## Status

The current architectural direction and minimum implementation order are in
[Uniform mixed architecture: minimum performance bridge](uniform-mixed-performance-bridge.md).
That review supersedes the historical stage-by-stage milestone order below.
The latest user priority is correctness and product/UI enablement first, then
resuming performance work; the single shared frame architecture is unchanged.

The always-live production integration is **not complete**. A complete
ownership-driven frame now advances borrowed native fields in Dawn and remaps
fine → mixed → fine → coarse → fine without allocating fields or rebuilding
pipelines during edits. The production host, rendering and UI still use the
retained native routes. Authored regions do not yet change production Uniform
simulation. The old preparation/switch UI has not been replaced.

The strict manufactured pressure gate and complete-frame performance target
still fail. The complete frame also lacks default orphan cleanup, production
runtime/control handling and ownership-aware publication. Passing stage and
live-remap fixtures does not establish production readiness or acceptance.

## Why the prototype was retired

The previous execution capsule bypassed Uniform runtime settings and replaced
its transport, pressure, surface processing, and publication. The panel then
hid controls that the replacement no longer implemented. Its upwind transport
multiplied fine-cell CFL substeps by receiver-budget sweeps across both tiers.
For 27 substeps and 128 sweeps, that meant 6,912 receiver dispatches alone.
Existing diagnostic runs attributed roughly 200–234 ms to transport versus
about 16 ms to prediction and pressure. These are historical observations, not
a controlled comparison at matched numerical quality.

Removed together: execution capsule, upwind transport, separate PCG pressure,
cell-centred advection, borrowed-field storage, migration kernels, overlapping
surface publication, renderer branches, mixed configuration parameters and
buttons, prototype-only oracles, and tests specific to that substitute method.
There is no hidden fallback to it. Its conservation tests did not establish
Uniform algorithm parity. The old dormant-preparation timing and memory figures
are not acceptance evidence for the replacement.

## Contract

Mixed ownership is the default Uniform Geometric architecture, always live.
There is one frame pipeline for all-fine, all-coarse, and mixed layouts. An
empty region list means fine ownership in that same pipeline. Adding, moving,
or removing a region updates ownership and conservatively remaps the evolved
state at a completed-frame boundary, without a restart or a preparation toggle.
The current runtime described under Status has not yet reached this contract.

Resolution changes work ownership and spacing; it does not select a second
fluid method. Preserve the original timestep, geometric donor normalization,
sharpening, vertex-phi conditioning, momentum transport, velocity extension,
pressure constraints, and applicable control semantics. Maintain one coupled
pressure solve and one authority for each interface flux.

No per-frame full fine expansion, independent coarse/fine advances, duplicated
dense persistent fields, CPU field downloads, or live shader compilation.
There must be no separate fine fast path, prepared coarse solver, or mixed
activation branch. Kernels dispatch over current ownership and actual stencil
work within the same stage sequence. Initialization reserves the bounded scratch
and compiles the pipelines once; region edits reuse them. The old endpoint
routes and their preparation UI must be removed when this pipeline replaces
them, together with code that has no remaining caller.

Acceptance limits remain ≤2% fine timing overhead and ≤3% additional resident
GPU payload. A 50/50 region in minidam64 must improve complete-frame performance
against the pre-change native baseline at equal simulated duration while
preserving physical plausibility. Measure the unified pipeline with and without
regions as well; fewer active cells or isolated stage speedups are insufficient.
Unsupported nondefault numerical options fail closed for every ownership layout,
with an explicit explanation; they must not select a legacy execution path.

## Implemented transport foundation and bounded topology

Manual choices remain 1h/4h. The production tile planner inserts internal 2h transition
cells to enforce strong 2:1 grading across faces, edges, and corners. Forced 4h
regions that prevent this collar conflict with adjacent forced 1h regions and
must be rejected; enforcement must never silently relax an explicit constraint.
Each pressure interface then has at most four fine subfaces per coarse face,
and the transition ratios match the existing hierarchy's 2x restriction steps.
The planner packs the intermediate tier into the tile word with no extra array.

Spatial grading alone does not bound a long backtrace: a 4h receiver may land
beyond its 2h collar in h donors. Do not impose a fine-cell CFL timestep to hide
this. Use a finite family of implicit fragment patterns. Choose a sampling width
that resolves all owners intersected by the departure box, capped at receiver
width. The required ratios are 1, 2, and 4, giving 8, 27, or 125 fragments.
Several fragments can resolve to the same donor. Their normalized contributions
sum to the independent geometric intersection weight; no adjacency list is
necessary. Same-tier interiors keep the original eight-donor specialization.

Reserve the maximum pattern per owner width, not per trace. With one base word,
one fallback weight, and fragment weights, the per-4h-tile edge budget is:

| Owner width | Owners/tile | Bytes/row | Bytes/tile |
| --- | ---: | ---: | ---: |
| h | 64 | 40 | 2,560 |
| 2h | 8 | 116 | 928 |
| 4h | 1 | 508 | 508 |

Thus even the long-trace fallback fits inside the existing finest edge arena.
This bounds edge storage only; runtime owner/work metadata and all other stages
still need a complete lifetime and residency audit. The GPU stage constructs
footprints directly from tile ownership, stores one donor-base coordinate per
row, and resolves donor indices implicitly. Two sampling bits per non-fine owner
fit in one word per tile. It reads only the selected pattern's slots, so an
ordinary coarse interior does not sweep all 125 reserved slots. Compact tier
worklists put 64 h owners, eight 2h tiles, or 64 4h tiles in each 64-lane group.
Dispatch spills into a second workgroup dimension for large scenes. Exact donor
clearing and decoding also visit only active owners, with clearing fused into
decoding between normalization rounds. Like native Uniform, fallback joins the
first row pass and donor division joins the next row reader or final gather;
there are no separate fallback or donor-division passes.

The native unfused row/donor normalization bodies are factored into
`uniform-volume-normalization.wgsl.ts`. The fixed nine-slot native specialization
has exactly the previous generated WGSL apart from whitespace; the fused path
is unchanged. The exact six-limb donor accumulator is also shared, without a
floating-point atomic retry loop. The seam fixture runs the same arithmetic with physical receiver
and donor capacities, three balancing rounds, and a final extensive-mass gather.
It retains self fallback and excess volume without receiver-budget iteration.

Tests compare both generic intersection rows and fixed patterns, on both uniform
endpoints and graded/ungraded seams in all axes. They exercise large departure,
anisotropic spacing, unsampled donors, retained excess, non-negativity, mass
conservation, the 2h collar, forced-region conflicts, and the edge-arena bound.
The variable-row calculation remains solely an independent correctness oracle.

The stage borrows the native r32 volume textures and the existing edge/donor
arena. Its additional resident buffers total **12 bytes per 4h tile + 16 bytes**:
8 bytes for ownership/worklists, 4 for sampling widths, and uniform counts.
The native trace adds one 576-byte field-metadata uniform block; all its field
and scratch resources are borrowed. Its reduced binding layout stays within
the existing ten-storage-buffer device limit.
`uvMixedTrace` appends owner dispatch to the specialized native shader and calls
its unchanged `uvTrace`, including the actual extrapolated MAC sampler. Native
velocity B holds departures after extension; only then may edge/donor work
reuse extension's scratch. Velocity D remains the live face authority.

The standalone displacement fixture now runs this same production stage; its
old shader copy has been deleted. An explicit terminal QA capture integrates
native extension, tracing, restriction and transport on an evolved state. It
consumes that solver because pressure, surface and momentum have not yet been
integrated with mixed ownership. Ordinary simulation does not allocate this
stage or route through it. This is not a complete mixed runtime solver, and
these checks do not establish pressure/surface parity or active-mixed speedup.

## Remaining implementation, in order

1. **Finish transport acceptance.** The native trace, strong grading, borrowed
   fixed storage and compact tier worklists are implemented. Endpoint
   equivalence is covered on an evolved state; profile mixed work at increasing
   sizes next. Run uniform endpoints through the same ownership-driven stages;
   keep long-trace coverage without CSR arrays or silent donor truncation.

2. **Native transport integration.** Encode those rows in the existing stage
   sequence, preserving donor normalization, fallback, dust accounting, gamma
   targets, sharpening, retained excess, and their controls. Share conservative
   accounting across both tiers. Validate fixed planar seams before live edits.

3. **Coupled pressure hierarchy.** Extend the existing constrained multigrid
   hierarchy for mixed ownership. Covered coarse/fine rows must not be counted
   twice. Restriction/prolongation and interface divergence/gradient must agree.
   Preserve fine subface flux degrees of freedom where needed; a coarse face
   consumes their area-weighted sum. Test hydrostatics, adjointness, residual
   acceptance, seams in every direction, and pressure controls. Do not reinstate
   the separate PCG solver as the default method.

4. **Full native stage and presentation parity.** Adapt velocity extension,
   default momentum transport, physical vertex phi, redistance, geometric
   sharpening, and volume/phi conditioning. Use consistent boundary sampling
   and the existing stage trace. Publish a coherent surface rather than hiding
   discrepancies with overlapping fine/coarse meshes. Preserve supported panel
   controls and explicitly reject unverified nondefault options in every layout.

5. **Live transitions and regions.** With the fixed seam proven, add conservative
   restriction/prolongation and ownership changes at completed-frame boundaries.
   Preserve evolved state/time; retain the last valid layout on conflicts.
   Exercise region growth/shrink/movement, queued changes, both uniform limits,
   repeated region edits, and no-allocation/no-compilation instrumentation.

6. **Measured acceptance.** Same initial states, controls, physical duration,
   warmup, and quality thresholds across fine/coarse and 12.5%, 25%, 50% fine
   tile layouts. Record per-stage GPU time, CPU encoding, dispatches, cells
   visited, interface work, and actual resident bytes. Compare dormant fine
   overhead separately from active mixed speed. Reinstate the mixed UI only
   after its actual stages, controls, numerical quality, and budgets pass.

## Verification commands

Run GPU jobs exclusively, with no browser simulation or second Dawn process.

```sh
node --import tsx --test tests/uniform-coarse-controls.test.ts tests/uniform-mixed-layout.test.ts tests/uniform-geometric-seam.test.ts tests/editor-voxel-region.test.ts tests/region-query-both-hosts.test.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test tests/uniform-geometric-seam-dawn.test.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test tests/uniform-mixed-native-transport-dawn.test.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test --test-concurrency=1 tests/uniform-runtime-coarse-dawn.test.ts tests/uniform-native-stages-dawn.test.ts
npm run benchmark:uniform-runtime-coarse
npm run benchmark:uniform-runtime-coarse -- --mixed-transport-preparation
npm run test:dawn:sparse-cm12
npm run check:types
```

## Recovery verification (2026-09-25)

- 34 CPU tests pass, including authoring/URL round trips, strong grading,
  fixed-pattern equivalence, conservation, and edge-memory bounds.
- The fixed-pattern Dawn test passes 60 cases over uniform, direct 4:1, and
  graded 2:1 layouts in all axes. It constructs geometry on the GPU and reuses
  buffers as patterns grow and shrink. No per-edge donor indices are uploaded.
- The three existing global resolution Dawn tests pass: live fine/coarse/fine
  with fixed and lagged pressure schedules, plus nonuniform volume/MAC/phi
  transfers. Actual prepared GPU payload overhead is 1.2824% in that fixture.
- Fine-mode ABBA passes: 20.3715 ms unprepared versus 20.3626 ms prepared
  (ratio 0.99957, below 1.02). Actual resident GPU payload is 54,891,388 versus
  55,595,168 bytes (+1.2821%, below 3%). This measures the retained global coarse
  preparation, not an active mixed solver. Log: `/tmp/fluid-mixed-recovery-abba.log`.
- Generated native volume WGSL, including donor accumulation, is identical to
  HEAD apart from whitespace. No native arithmetic, passes, or buffers were
  added by the shared-kernel factoring.
- Native stage parity initially failed because the test copied unused dense
  MacCormack scratch into native 1x1x1 placeholders. The test now excludes only
  those two known unused fields and asserts their placeholder dimensions and
  semi-Lagrangian mode. It then passes partial-page comparisons but fails long
  dam pressure at frame 10: normalized difference 2.19345e-5 exceeds 1e-5.
  The tolerance has not been changed. This remains unresolved.
- The required Sparse gate passes 5/17 lanes. Failures comprise nine timeouts,
  mini64 median 212.2711 ms above its 110 ms ceiling, and missing compiled
  topology faces in the terrain and outside-drop lanes. Log: `/tmp/fluid-mixed-recovery-sparse.log`.
  These are not labelled baseline failures without a pristine comparison.
- Type checking reports 14 errors in Sparse tests/tools, with none in the
  changed Uniform/region files. Log: `/tmp/fluid-mixed-recovery-types-final.log`.

The full mixed feature is still incomplete. These results authorize further
integration work, not an active-mixed performance or quality claim.

## Native transport integration checks (2026-09-25)

- 34 CPU authoring, topology and transport tests pass.
- The same 60 GPU seam cases now execute the production transport stage, with
  native r32 volume textures, borrowed scratch and compact tier dispatch. They
  also cover reused dirty arenas after compact donor clears were introduced.
- Evolved-state Dawn integration passes native/all-fine, graded h/2h/4h and
  all-coarse ownership. It checks departures from the actual Uniform midpoint
  trace, per-cell agreement with the independent overlap oracle, conserved
  physical volume, all-fine parity, and additional resource bytes. Tiny signed
  roundoff already in native volume with dust flooring disabled is preserved.
  The final fused schedule passes the same checks: maximum per-owner extensive
  volume errors are 2.40e-7 (fine), 1.73e-6 (graded) and 1.14e-5 (coarse), and
  relative mass error remains below 5e-8. Log: `/tmp/fluid-mixed-native-fused.log`.
- The ordinary native transport schedule is shared with the endpoint gate by
  extracting its existing encoder; pass order and arithmetic remain unchanged.
  Native volume WGSL still matches HEAD apart from whitespace. All three
  scratch-layout tests pass.
- The native rectangular-stage regression again passes all partial-page frames
  and fails long-dam pressure at frame 10 with normalized error
  2.193450927734375e-5, exactly the failure observed before this integration.
  Its 1e-5 threshold remains unchanged. A pristine comparison is still needed.
- All three retained global fine/coarse transition tests pass again.
- Combined global-coarse plus mixed-transport preparation uses 55,644,912
  resident payload bytes versus 54,891,388 unprepared (+1.3728%, below 3%).
  The mixed transport and native trace add 49,744 bytes at 64³; no dense field
  is duplicated. The initial ABBA timing run passes but has a slow first
  baseline (29.93 ms versus 20.32 ms for its final baseline), so its apparent
  speedup is not evidence that preparation speeds up fine simulation.
  Log: `/tmp/fluid-mixed-native-abba.log`.
  A repeat ABBA run with stable baselines measures 20.3756 ms unprepared versus
  20.5036 ms prepared (+0.6282%, below 2%). Its resident payload result is
  unchanged. Log: `/tmp/fluid-mixed-native-abba-warm.log`. This verifies dormant
  combined preparation, not active mixed-simulation performance.
- The required Sparse gate passes 5/17 lanes and exhausts its 480-second budget.
  Ten lanes time out; mini64 records 213.975 ms versus its unchanged 110 ms
  ceiling; terrain front progress leaves 18 bricks to the far wall versus the
  accepted maximum of 6. These failures are not labelled pristine-baseline
  failures. Log: `/tmp/fluid-mixed-native-sparse.log`.
- Type checking still reports 14 errors in Sparse tests/tools, none in changed
  Uniform files. Log: `/tmp/fluid-mixed-native-types-final.log`.

Pressure, surface, momentum, live ownership edits and active-mixed performance
are still outstanding. This stage capture cannot resume a partially mixed
simulation and is not wired into the UI.

## Shared face ownership (2026-09-25)

`UniformMixedOwnership` now owns the one topology/worklist buffer pair shared
by transport and face operations. Transport no longer carries a private copy
of its allocation/dispatch implementation. Resource bytes remain unchanged.
`uniform-mixed-faces.wgsl.ts` derives canonical MAC patches from this ownership:
one patch toward an equal/coarser cell, four toward a finer cell. Both incident
cells derive the same positive-MAC anchor and patch width. Negative domain
faces retain the existing boundary-plane convention. No face incidence table,
CSR storage or additional resident face metadata is introduced.

Nine GPU fixtures cover both uniform endpoints and graded seams in all axes.
They verify reciprocal neighbour identities, boundary ownership, patch centres,
complete face areas and cancellation of arbitrary internal physical fluxes on
anisotropic grids. The 60 transport seam cases and evolved native-state checks
also pass after sharing owner lookup. Log: `/tmp/fluid-mixed-faces2.log`.

This is face geometry, not a coupled pressure implementation. Two constraints
must be resolved in that implementation:

- The coarse/fine pressure graph is not bipartite. One coarse cell and two
  adjacent fine neighbours form a triangle, so ordinary red/black in-place
  updates race at the seam. Colouring must cover the actual reconstructed
  operator, including any additional diagonal couplings, rather than merely
  reusing uniform parity.
- A two-point difference between offset coarse/fine cell centres is not a
  consistent face-normal gradient. For a vertical seam in hydrostatic water,
  a tangential centre offset creates a nonzero horizontal pressure correction.
  Reconstruction must preserve affine pressure at the canonical patch centre;
  the divergence, gradient and restriction/prolongation must be designed and
  checked together. Flux cancellation alone does not establish this property.

The mixed pressure, surface, momentum and live-transition work remains open.
The application still must not expose this as a finished mixed simulation.

## Pressure reconstruction and correction experiment (2026-09-25)

The GPU now reconstructs pressure at canonical mixed face centres. On a coarse
side, aligned neighbour samples determine the tangential slope; four finer
neighbours are averaged into an aligned sample. An offset coarser neighbour is
excluded. A domain only one coarse cell thick uses the tangential variation of
incident finer-face quadrants where no aligned sample exists. This replaces the
initial local least-squares fit, which converged substantially more slowly on
an anisotropic seam. No least-squares implementation is retained.

`uniform-mixed-pressure-reconstruction.wgsl.ts` is an operator component, not a
replacement pressure solver. It consumes the shared ownership/face functions
and caller-provided pressure/slope accessors. Slope reconstruction and face
reads execute in separate dispatches. Its Dawn harness owns test-only buffers;
runtime integration must borrow pressure scratch. The ordinary simulation does
not import or allocate this component yet.

The independent CPU oracle constructs faces by intersecting cell boxes. Checks
cover constant/affine pressure, anisotropic spacing, both uniform endpoints,
plane seams in three orientations, a corner region, an interior region, and
thin domains in all three orientations. GPU comparisons include arbitrary
pressure, both incident owners of each face, and exact face coverage. All 30
GPU fixture/field combinations pass, with maximum allowed derivative error
3e-5. Log: `/tmp/fluid-mixed-pressure-reconstruction.log`.

An affine-consistent reconstructed gradient is not the adjoint of unchanged
physical face divergence. Replacing divergence with its transpose would change
what the transported physical flux means. The tests explicitly distinguish
these operators instead of accepting algebraic adjointness as physical proof.

A CPU experiment freezes the tangential correction on the RHS of the symmetric
two-point pressure core. With A0 = -V D G0 and E = G - G0, the update is
A0 p(next) = b + V D E p(previous). D remains physical face divergence. The
small-fixture core inverse is dense Cholesky with a pinned pressure gauge,
strictly a test oracle; no Cholesky or alternative solver is added to runtime.

This experiment is **not an accepted runtime schedule**:

- Full 3D plane/corner/interior fixtures meet 1e-7 divergence and hydrostatic
  velocity error within 16 exact core solves. Exact core solves do not establish
  native multigrid convergence or an acceptable per-frame cost.
- A 16x4x16 domain, one 4h cell high, has the correct hydrostatic fixed point
  but leaves divergence error 0.0269394 after 16 solves. A diagnostic run needs
  64 solves to reach 3.47e-10. Increasing the runtime iteration budget to match
  this is not the proposed fix. A dedicated rejection test preserves this
  counterexample; it must not be counted as a passing convergence fixture.
- The next integration must carry the reconstructed residual/correction through
  the existing multigrid hierarchy and verify convergence under its normal
  controls. A fixed outer loop around complete native solves is rejected on
  cost grounds. Pressure bounds, free surfaces, restriction/prolongation and
  race-free smoothing are still unimplemented for mixed ownership.

The combined CPU authoring/topology/transport/pressure checks pass 28 tests,
including the explicit rejection test. Log: `/tmp/fluid-mixed-pressure-cpu.log`.
Type checking reports the same 14 Sparse test/tool errors and no errors in
these Uniform changes. Log: `/tmp/fluid-mixed-pressure-types-final.log`.
No active-mixed performance claim or end-to-end completion follows from these
operator checks. The feature remains unavailable in the UI pending the full
pressure, surface, momentum, publication and live-transition integration.

## Coupled multigrid components (2026-09-25)

The mixed pressure hierarchy now has a shared production layout builder:
`uniformMixedPressureLevel` raises the minimum owner width to 2h and then 4h,
leaving existing larger owners intact. At 4h, every tile is one uniform owner.
Packing is shared with simulation ownership; enforcement constraints are not
modified. Transfers read two namespaced instances of the existing topology ABI,
so no donor/parent/adjacency table is added.

New GPU components implement:

- The full reconstructed physical pressure operator and frozen seam RHS.
  The RHS evaluates tangential face corrections directly rather than
  subtracting two Laplacians. Rows with no resolution change retain the exact
  original RHS, including its f32 bits for the tested finite values.
- Six tier/parity colours for the direct-neighbour smoother. Reconstruction is
  frozen before the colour passes, so its wider dependencies are not read
  in-place. Each colour dispatch visits only its own tier. Colour separation
  is checked on every direct face in the fixture set.
- Eight-child, volume-consistent residual restriction and bounded eight-tap
  pressure prolongation. Owners unchanged between levels inject directly.
  Prolongation omits out-of-domain samples and renormalizes, matching the native
  boundary convention. Constants are preserved and restricted physical
  residual sums are conserved. Mixed virtual samples resolve through source
  ownership; no affine-exact prolongation claim is made at mixed seams.

The independent CPU cycle oracle now restricts the **full reconstructed
residual** through this hierarchy, rather than wrapping repeated complete core
solves in an outer correction loop. It uses the existing three Full-Cycles,
four V-cycles and six pre/post smoothing sweeps, with an exact test-only bottom
solve. It covers both hydrostatic and arbitrary face velocities in eight
plane/corner/interior/thin fixtures. All residual histories decrease and their
final physical divergence is below 1e-3. Hydrostatic final residual is no more
than 1.05 times the corresponding all-fine CPU oracle (with a 1e-8 roundoff
floor). This comparison is under identical test cycle controls, not parity
with the live GPU solver or its full boundary hierarchy.

In particular, the previously rejected 16x4x16 case now finishes at approximately
9.13e-6 physical divergence under these coupled cycles, compared with 0.0269394
after 16 exact-core outer iterations. The old 1e-7 outer-loop acceptance target
and its explicit rejection test remain unchanged. The coupled-cycle bound and
all-fine comparison are separate tests; this result does not claim the old
fixed outer schedule passes or that the coupled GPU solve has converged yet.

Dawn checks exercise the production full operator, one complete frozen-RHS
colour sweep, restriction and prolongation on both mixed levels of all eight
fixtures and both uniform endpoints. They match the independent CPU oracle
within 1e-5 for pressure operations and 1e-6 for transfers. No CPU-generated
matrices or transfer rows are supplied to the GPU. Test field buffers are
fixture-owned; live integration must use borrowed pressure storage.

Validation:

- 30 CPU tests pass, including the preserved rejection case and coupled-cycle
  checks: `/tmp/fluid-mixed-multigrid-cpu-final.log`.
- Combined production transport, face ownership, evolved native transport,
  reconstruction and pressure-operator Dawn suites pass:
  `/tmp/fluid-mixed-multigrid-dawn.log`.
- After changing RHS evaluation to the direct seam term, reconstruction and
  pressure-operator suites pass again:
  `/tmp/fluid-mixed-multigrid-operators-final.log`. Endpoint coverage is also
  checked in `/tmp/fluid-mixed-multigrid-endpoints.log`.
- Type checking still reports 14 Sparse test/tool errors, with none in Uniform:
  `/tmp/fluid-mixed-multigrid-types.log`.

These are integrated operator and CPU-cycle checks, not a complete GPU mixed
pressure solve. Production cycle binding, pressure minima/free surfaces,
solid/terrain coefficients and accepted-iterate recovery remain to be connected
and verified before a live mixed step can use them. Ordinary fine execution
has no new pressure allocation or dispatch. Active mixed timing, memory use
of the complete solver, and live fine/mixed/coarse transitions are still open.

## Device-only pressure cycles and borrowed stage bindings (2026-09-25)

A complete closed, fully liquid mixed pressure cycle now runs entirely on the
GPU in the integration fixture. There is no CPU solve, field upload, or readback
between cycles. It uses the production mixed reconstruction/operator/transfer
functions and the unchanged native strided coarse solver, including its
projected updates, double-single arithmetic, 1e-4 stopping test and 4096-sweep
cap. The fixture adapts only the coarse kernel's three texture accesses to
buffers, checking each replacement occurs exactly once. It supplies the closed,
fully liquid topology and an inactive lower bound explicitly.

The schedule remains three Full-Cycles, four V-cycles, and six pre/post sweeps.
Sixteen solves cover hydrostatic and arbitrary face velocities on the eight
plane/corner/interior/thin fixtures. Every native coarse solve converges without
exhausting its cap. Final GPU divergence is below the existing coupled-cycle
1e-3 gate in all cases; the worst is 5.2023e-4. The formerly problematic 16x4x16
hydrostatic case finishes at 2.6226e-5. Independent CPU projection of the final
GPU pressure also meets the divergence gate, and hydrostatic parasitic velocity
stays below 1e-3. CPU/GPU pressure gradients agree within 1e-3 without comparing
arbitrary pressure gauges. Reusing the fixture for its second RHS exercises
dirty storage rather than relying on fresh zero-filled fields.

`uniform-mixed-pressure-stage.ts` now provides reusable level and transfer
bindings. They borrow ownership and `GPUBufferBinding` slices and allocate no
buffers or textures. Stage construction and compilation are tested with field
allocation APIs forbidden. Views are trimmed to the required cell counts;
undersized/overlapping fields and nonadjacent transfer levels are rejected.
The integration fixture uses these production stage bindings rather than
retaining its own duplicate shader setup.

The shared-arena test found and fixed an important binding requirement: WebGPU
tracks read-only versus writable storage usage for the entire buffer, even
when bound ranges are disjoint. Therefore field bindings consistently use
writable-storage usage; logically read-only RHS/source fields remain unwritten.
Tests run reconstruction, residual, restriction and additive prolongation on
aligned slices of one deliberately dirty arena, compare results independently,
and verify RHS data, unused destination tails and unrelated bytes are preserved.
No extra field buffer was introduced to bypass the aliasing restriction.

Validation: 20 Dawn checks pass in the combined complete-cycle, operator and
reconstruction run. Log: `/tmp/fluid-mixed-pressure-cycles-final.log`. Type
checking still reports 14 Sparse test/tool errors and none in Uniform. Log:
`/tmp/fluid-mixed-pressure-cycles-types-final.log`.

Scope remains explicit: cycle traversal and coarse fixture adaptation are test
harness code; the reusable stage bindings/operators are production components.
The small fixture reaches a sufficiently small uniform 4h bottom grid. A live
large-scene hierarchy must continue through the existing uniform coarse levels,
not treat the whole 4h grid as one bottom solve. Pressure minima, free surfaces,
solid/terrain coefficients, accepted-iterate recovery and native-host arena
lifetimes still need integration. There is no active-mixed timing claim, UI
activation, or full simulation completion from this closed-box pressure gate.

### Pressure bounds on the coupled hierarchy (2026-09-25)

The mixed pressure operators now support the existing CM11a lower-bound
semantics. Smoothing projects the pressure onto its lower bound. Restriction
uses the maximum child bound, with persistent coarse owners injected unchanged;
it never averages bounds. V-cycles restrict `minimum - pressure`, while full
correction cycles shift the fine minimum by the backed-up pressure, transfer
those bounds, and restore the original minimum for subsequent cycles. Linear
residuals remain the restriction input. The separate convergence measurement
uses the native projected-residual rule and rejects nonfinite pressure/residual
values with a maximal failure diagnostic.

The production bound stage borrows disjoint storage views and allocates no
fields. The unbounded stage needs no minimum binding. Nested-level validation
is shared without constructing a discarded transfer stage. The QA cycle fixture
allocates a second minimum field only at the finest level, where the original
bound must survive a full correction cycle; coarse levels reuse one field.

Eight manufactured bound-active problems pass complete device-only cycles with
the unchanged native coarse solve and cycle budget. Their maximum pressure
errors are below 4e-7 and projected residuals below 1.1e-6. Independent physical
operator checks verify free-row residuals, active-row inequalities and lower
bounds. Shared-arena tests additionally verify all three bound operations,
persistent owners, inactive minima, untouched sources/destination tails, no
field allocation during stage setup, and NaN/Inf failure diagnostics.

Validation: the combined cycle/operator/reconstruction run passes 28 checks
(`/tmp/fluid-mixed-bounds-final.log`). The final cycle rerun including explicit
nonfinite diagnostics passes all 26 checks
(`/tmp/fluid-mixed-bounds-arena.log`). CPU hierarchy, seam and ownership tests
pass all 10 checks (`/tmp/fluid-mixed-bounds-cpu.log`). Type checking continues
to report the same 14 Sparse test/tool errors, with no Uniform errors
(`/tmp/fluid-mixed-bounds-types.log`).

This establishes algebraic pressure bounds, not solid boundary coefficients or
free-surface integration. A specific remaining seam case must be addressed:
a coarse center can lie above a planar water surface while one of its fine
face patches lies below it. Applying center-based liquid/air classification
directly across that offset can introduce a false tangential pressure force.
Pressure and surface sampling at those patches must be consistent and pass
hydrostatic tests before enabling live mixed simulation. This does not authorize
forcing all surface tiles fine, changing manual 4h enforcement, or substituting
a different solver. Large-scene hierarchy continuation, native arena lifetimes,
accepted-iterate recovery, and active mixed performance validation also remain.

### Pressure scaling and free-surface seam direction (2026-09-26)

A new isolated pressure-sweep probe measures fenced encoding/execution after
warmup, with fresh ABCCBA fine/mixed/coarse instances. The mixed layout enforces
h in one eighth of the volume, uses the strong 2h collar, and keeps the rest 4h.
Run `node --import tsx tools/benchmark-uniform-mixed-pressure-dawn.ts --size=128`.
It acquires the repository GPU lease and does not construct the quadratic CPU
face oracle. Both endpoints use the mixed pressure kernels, so this is an
ownership-scaling diagnostic, not a native-fine or full-frame benchmark.

The first 64³ measurement exposed scheduling overhead: mixed sweeps took
0.523 ms against 0.437 ms all-fine despite retaining only 37,871 of 262,144
owners. `encodeSweep` now records reconstruction, frozen-RHS preparation and
six ordered colour dispatches within one compute pass instead of eight.
Dispatch dependencies, colour order, arithmetic and storage are unchanged.
The complete cycle/operator/reconstruction suite still passes all 28 checks
(`/tmp/fluid-mixed-pressure-batched-tests.log`).

After batching, the 64³ probe measured 0.314 ms mixed versus 0.333 ms all-fine
(only about 6% faster). At 128³ it measured 0.499 ms mixed versus 1.848 ms
all-fine, and an independent rerun measured 0.508 versus 1.891 ms: about 3.7x
faster in both runs. The 128³ mixed layout has 296,535 owners versus 2,097,152
all-fine. Logs: `/tmp/fluid-mixed-pressure-batched64.log`,
`/tmp/fluid-mixed-pressure-batched128.log`, and
`/tmp/fluid-mixed-pressure-batched128-repeat.log`. These results support
large-grid scaling but do not establish active mixed simulation performance:
full hierarchy traversal, surface work, projection and native-host scheduling
remain to be measured. Small-grid dispatch cost is still material. The probe's
standalone scratch allocations are not a live resident-memory budget result.

The free-surface algebraic gate now covers 864 mixed liquid/air crossings,
including 612 faces tangential to planar surfaces, across anisotropic layouts
and all three surface orientations. For a liquid/air pair, retain native theta
on the center-to-center segment. Subtract the liquid pressure slope dotted with
the tangential center displacement, divided by normal center separation, from
the usual ghost-fluid normal difference. This removes the spurious sideways
force in the planar tests without inventing a coarse-air pressure unknown or
forcing the surface fine. Aligned faces retain the exact native theta formula,
including the 0.05 minimum. Exact hydrostatic assertions exclude cases where
that existing clamp intentionally moves the effective boundary.

`tests/uniform-mixed-pressure-surface.test.ts` supplies exact affine slopes:
this proves the geometric correction only, not reconstruction or convergence.
Both tests pass (`/tmp/fluid-mixed-pressure-surface.log`). Next, reconstruct
liquid-side slopes with the same surface boundary condition, use the correction
consistently in pressure application/RHS/projection, and verify complete GPU
free-surface cycles with no analytic slope input. Curved/thin surfaces and the
clamped-theta cases remain required integration checks. Type checking still
reports 14 Sparse test/tool errors and none in Uniform
(`/tmp/fluid-mixed-pressure-scaling-types.log`).

### GPU free surfaces and production cycle traversal (2026-09-26)

The pressure level stage now accepts a borrowed phi view, independently of its
optional minimum view. Surface-aware reconstruction supplies Dirichlet ghost
samples from the liquid pressure and the existing CM12 theta rule. A skew
liquid/air face removes the tangential center-to-center pressure derivative;
wet/wet faces retain the prior canonical-patch reconstruction. Both incident
owners evaluate the same positive-axis gradient. Air pressure storage is not
used as a liquid slope donor. Smoothing, residuals and measurement respect the
liquid classification; pressure minima keep their native projection semantics.
The closed-liquid specialization remains available without a phi binding.

Complete-cycle testing exposed a real hierarchy defect in the initial fixture:
sampling analytic phi at coarse centers could erase a thin free surface,
leaving an incompatible closed coarse system. Production phi restriction now
uses native CM11a sign-aware child reduction, with unchanged owners injected.
The native indexing is `paperDestination=M-destinationIndex`; its
`paperDestination >= M-C` rule applies to the first C=2 coarsenings. The host
supplies that policy explicitly. Tests provide only finest-owner phi, and the
GPU builds the remaining levels. No analytic pressure or slope input enters
the solve.

`uniform-mixed-pressure-cycles.ts` now owns the reusable production traversal;
the duplicate traversal was removed from the Dawn fixture. It borrows all
pressure, RHS, phi, minimum, slope and backup views and creates no buffers or
textures. A shader backup supports disjoint views within one arena. The h/2h/4h
hierarchy is checked, and the 4h callback must continue the native uniform
hierarchy for large scenes. Acceptance/recovery remains the native host's job.
Cycle setup is tested with field-allocation APIs forbidden.

The surface gate covers 160 complete solves: planar hydrostatic and irregular
velocity inputs in each orientation/sign, hydrostatic cases with zero pressure
minima, and curved liquid/air configurations. It checks native coarse stopping,
liquid divergence and the physical projected velocity using independently
constructed face geometry. Both uniform endpoints and mixed anisotropic/thin
layouts also pass the direct GPU gradient check. The physical tolerance remains
1e-3. An initial absolute-pressure assertion was replaced with the existing
velocity/divergence gate: native coarse tolerance can leave a 0.00272 pressure
error over a long wavelength while the actual projected velocity and divergence
meet that gate. Pressure error remains reported for hydrostatic diagnostics.
No cycle count, native coarse tolerance or physical acceptance ceiling changed.

Validation:
- Combined cycles, surface gradients, original operators and reconstruction:
  190 checks pass (`/tmp/fluid-mixed-surface-final.log`).
- Final shared-arena rerun, including both phi reduction policies, backup,
  optional surface/minimum views and nonfinite diagnostics: 26 checks pass
  (`/tmp/fluid-mixed-surface-arena-final.log`).
- CPU algebraic surface, hierarchy, seam and layout gates: 12 checks pass
  (`/tmp/fluid-mixed-surface-final-cpu.log`).
- Type checking reports the same 14 Sparse test/tool errors and no Uniform
  errors (`/tmp/fluid-mixed-surface-verified-types.log`).

The scaling probe accepts `--surface`. Its 128³ planar free-surface sweep
measured 0.529 ms mixed versus 1.203 ms all-fine (2.27x), using the same mixed
kernels at both endpoints (`/tmp/fluid-mixed-surface-pressure128.log`). This
remains an isolated sweep measurement, not native-fine or complete-frame parity.

**UI completion remains open.** The normal simulation host still has only a
terminal mixed transport capture; it does not advance complete mixed frames.
Remaining work includes native mixed vertex-phi advection/redistance and surface
conditioning, face momentum/extension, pressure field binding and continuation
through the native uniform hierarchy, accepted-iterate recovery, coherent
publication and bidirectional live field handoff. These must preserve the
existing fine controls and avoid full fine expansion per frame. Only then can
the runtime control be enabled and active-frame timing/resident-memory budgets
be measured. The current UI remains accurately unavailable.

### Mixed MAC tracing and resolution handoff (2026-09-26)

Transport tracing now reads canonical mixed MAC faces, including the native
negative boundary planes. The RK2 characteristic is shared with native Uniform.
Fine velocity is restricted once from the native extrapolation halo into the
borrowed velocity A field. Restriction averages physical face patches and packs
all components sharing an RGBA anchor in one write, avoiding races where coarse
and fine face patches intersect. It does not materialize inactive fine faces.
Velocity C remains lazily allocated for native MacCormack; mixed preparation
does not turn that dummy field into a full allocation or borrow velocity D,
which carries native face-open data.

The mixed MAC sampler uses bounded h/2h/4h interpolation with no incidence table.
Simply choosing an interpolation grid by the containing owner produced jumps
of 0.46–1.07 in arbitrary-field seam probes. Local distance weights blend the
interpolants over narrow bands around finer tiles. This blends sampling only;
ownership, physical fluxes and transported cells remain the mixed Uniform grid.
Uniform endpoints retain native MAC interpolation. Constant/affine, arbitrary
field, continuity and NaN-poisoned inactive-face tests pass.

The reverse cell transfer copies intensive cell volume to fine children and
interpolates velocity only along each face normal, matching the native global
resolution handoff. Tangential patch values remain constant so restriction
recovers each canonical face's integrated flux. Dense fine work is confined to
this resolution-switch operation, not mixed steps.

Vertex phi has a shared authority and sampler as well. The coarsest incident
cell owns a vertex, with compact owner index breaking ties between equal cells.
Restriction copies only coincident authoritative vertices. Hanging vertices
are evaluated from that cell's corners, with the two possible coarsening levels
statically expanded. There are no duplicate seam values, hanging-node buffers
or updates of inactive fine vertices. Prolongation reconstructs the fine vertex
lattice only on the return transition; phi stays in physical units. Tests cover
all uniform/mixed/thin fixtures, affine and curved fields, physical boundaries,
seam continuity and NaN-poisoned inactive vertices. Stage construction is tested
with buffer/texture allocation forbidden.

The terminal native capture exercises these transfers on water evolved for
three native frames. It returns compact velocity, volume and phi, and
`captureMixedProlongationForQA()` returns reconstructed fine cell/vertex fields.
Fine-endpoint transport agrees with native; mixed transport conserves mass;
the return transfer preserves cell mass, canonical face fluxes and authoritative
phi. This is a state-transfer check, **not a complete mixed simulation step**:
phi is transferred but not yet advected/conditioned by the mixed frame. The host
still refuses to resume after the terminal capture.

Validation:
- MAC sampling, restriction and native transport/cell return: 10 checks pass
  (`/tmp/fluid-mixed-mac-handoff-final.log`).
- Vertex sampling/transfer plus evolved native cell/vertex return: both suites
  pass (`/tmp/fluid-mixed-native-handoff-final.log`).
- Final dormant ABBA benchmark: 19.757 ms baseline versus 19.556 ms prepared,
  ratio 0.98986, within the unchanged 2% ceiling. Actual resident GPU field
  payload is 54,891,388 versus 55,644,912 bytes (+1.37275%), within 3%. This
  includes all new handoff stages and prepared global coarse resources
  (`/tmp/fluid-mixed-handoff-dormant-final.log`). These timings establish no
  measured dormant regression; they are not an active mixed speedup result.
- Type checking reports 14 existing Sparse test/tool errors and no Uniform
  errors (`/tmp/fluid-mixed-handoff-final-types.log`).

Remaining UI blockers are full mixed face momentum/extension and forces,
vertex advection/redistance and geometric surface conditioning, native pressure
binding/acceptance/recovery and uniform-hierarchy continuation, and coherent
publication/live lifecycle. The existing fine controls must remain functional.
Active full-frame speedup has not been measured, and mixed UI activation remains
disabled until these pieces form a validated complete frame.

### Momentum, forces and geometric conditioning integration (in progress)

The next frame components are implemented as borrowed-field stages:

- Canonical face momentum uses the native long-characteristic RK2 rule and
  supports SL, MacCormack prediction/reversal/bounded correction, liquid-only
  donor filtering and the native negative/positive wall carry conventions.
  The original native characteristic now shares its source generator.
- A pure sampler payload carries extended velocity, supported physical
  velocity and support weight together. A 4h sampling cache restricts these
  values once instead of reconstructing coarse samples inside every trace.
  It is a sampling cache, not an independently advanced coarse simulation.
  Its three borrowed RGBA halo fields have extent D/4+2; host field assignment
  and the resulting complete prepared-memory measurement remain to be done.
- Canonical face forces apply gravity, molecular viscosity and balanced
  capillarity. Gravity retains the smoothed occupancy, airborne and compressed
  volume eligibility rules. Interface normals come from the shared vertex
  authority; capillary differences use the pressure seam reconstruction.
- Mixed geometric fill uses the same native planar-exact/eight-probe source
  generator. Sharpening follows prepare/propose/limit/commit, including dose,
  compaction, orphan reception/gathering and dust preservation near the surface.
  Cell budgets and face fluxes are physical mass in fine-cell units. Subface
  area shares split offers before the reciprocal budget limiter. Its transient
  data fits inside 36N bytes of the existing 40N transport-edge slice.

Preparation exposed an important implementation cost: nested sampler expansion
made the initial momentum shader variants exceed the 240-second test envelope.
Tier and operation variants were consolidated into one compact traversal and
one uniform-selected operation. The 4h sampling cache removed the remaining
expensive coarse reconstruction. All seven momentum fixtures now complete in
62.25 seconds together; the timeout and numerical tolerances were not raised.
This is test/preparation timing, not an active mixed-frame speed measurement.

Validation so far:
- Momentum, forces, face restriction and scalar sampling passed in
  `/tmp/fluid-mixed-cached-momentum.log`. That combined run also found reserved
  WGSL identifier errors in the new geometry/sharpening shaders, since fixed.
- The corrected geometry/sharpening and evolved native transport/handoff suites
  all pass (`/tmp/fluid-mixed-sharpen-final.log`). The geometry test compares
  oblique planar fills with an independent box-volume CDF; sharpening checks
  physical mass conservation, bounds and effective control changes over eight
  sweeps at both endpoints and across mixed seams.

**Still not live/UI-ready.** These stages have not yet been connected into the
full advancing host. Besides vertex evolution and velocity extension, pressure
needs its native separating-wall halo rows, their half-dual-face coefficients,
minimum-pressure constraints, restriction/prolongation, accepted-iterate
recovery and the native uniform hierarchy below 4h. The existing isolated mixed
pressure checks cover closed/interior and free-surface operators; they do not
prove native wall-release behavior. Publication also needs mixed ownership
sampling rather than per-frame expansion of inactive fine cells. These are
required before UI activation, followed by complete-frame and dormant budgets.

### Separating walls and pressure/velocity integration (in progress)

The mixed pressure rows now include the native one-cell wall continuation.
Closed wall faces use half-dual coefficients and constrained halo pressures;
the open ceiling retains its air/ghost-fluid treatment. Halo indices are
computed from the six domain planes, without a connectivity table. The halo
unknown attached to an owner is updated on its opposite colour, so smoothing
still has ordered, race-free dependencies.

Residuals, backups, pressure bounds and corrections now include these rows.
Restriction clamps the normal child coordinate and averages the tangential
children. Bound restriction uses the native maximum, optionally after pressure
subtraction. Prolongation preserves the native rule of ignoring exterior taps
and renormalizing the remaining interior weights. Existing owners inject
unchanged across mixed pressure levels.

The pressure/velocity stage builds the RHS from canonical face fluxes with the
same wall fractions and physical cell volumes as the matrix. Projection uses
the shared reconstructed pressure gradient, preserves airborne momentum, and
publishes the solved wall-release bits with the MAC field. It consumes the
caller’s authoritative pressure phi and geometric volume-correction divergence;
those inputs must still be connected to the live frame.

Validation:
- Closed/open wall rows at both endpoints and a mixed layout meet the original
  2e-4 pressure/residual tolerance. Local smoothing alone needed 8,000 sweeps
  for this manufactured contact problem; this is an operator test, **not** the
  intended live pressure schedule. No tolerance was relaxed.
- Halo restriction, minimum transfer and prolongation pass independent CPU
  references on both endpoints and five mixed layouts, across both transitions.
- These checks plus the existing free-surface cycle suite pass 163 cases in
  `/tmp/fluid-mixed-wall-transfers.log`.
- Pressure RHS and projection checks pass on seven layouts, including an affine
  pressure field, half-dual boundary fluxes and explicit release flags
  (`/tmp/fluid-mixed-pressure-velocity.log`).

A native hierarchy continuation entry point and a borrowed-field 4h bridge
are implemented and under validation. They reuse the existing native V-cycle
below 4h rather than treating a large 4h lattice as a coarsest solve. The bridge
only traverses D/4+2, never the inactive fine cells. The native fine plan remains
unchanged when this optional entry point is unused.

**UI activation is still pending.** Remaining integration includes pressure
phase/volume-correction authority, acceptance/recovery, velocity extension,
vertex evolution, a complete host step, topology-aware rendering, and the
complete-frame performance and prepared-memory gates. The new isolated tests
do not establish those results.

#### Native continuation validation correction and current blocker

The first continuation fixture inherited the host's startup `dt=0`. That made
its coarse stopping residual identically zero. Its apparent convergence was
therefore not valid evidence for the lower native solve. The fixture now uses
nonzero, unit `dt/rho` and strict native coarse accuracy. It first checks that
one bridged V-cycle exactly matches the same native operation and leaves finest
native scratch untouched. That transfer comparison passes.

The complete h/2h/native-4h Full-cycle/V-cycle sequence does **not** yet pass the
constrained-wall manufactured problem: 757 owners give maximum pressure error
0.36076343 and projected pressure residual 13.3396616, against unchanged targets
2e-4 and 2e-3. Reproduction:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test \
  tests/uniform-mixed-pressure-continuation-dawn.test.ts
```

Latest failing evidence: `/tmp/fluid-mixed-complete-pressure-4.log`. Repeated
unchecked native inner V-cycles also oscillate on this problem. Do not describe
an inner correction as a complete accepted solve. The live integration still
needs the native acceptance/rollback/recovery policy, and the boundary
correction behavior needs diagnosis against that policy before UI activation.
The failing test is retained as the integration gate; its assertions have not
been relaxed.

Separately, the new pressure-authority stage passes endpoint and mixed checks
for all three V-based pressure classification modes, airborne support, and
physical-mass-weighted global excess/deficit balance at both ordinary and tiny
timesteps. The native relaxation formula is now shared with the mixed stage.
The authority, sharpening and evolved native transport suites pass together
in `/tmp/fluid-mixed-authority.log`. This does not establish a complete frame.

GPU finite-iterate acceptance, rollback (including halo rows), and indirect
recovery dispatch are now implemented as borrowed-field stages. The targeted
safety test passes for improving, worsening, nonfinite-sentinel and
post-convergence candidates. Recovery launches only after rejection and before
convergence; the native 8-by-8 recovery ceiling is unchanged. Pressure ownership
and level stages accept optional indirect launches for that recovery path.

This does not resolve the integration gate: the 757-owner wall case improves
monotonically under the seven scheduled cycles, so rejection-based recovery
correctly does not trigger. Its error and residual remain 0.36076343 and
13.3396616. Latest combined run `/tmp/fluid-mixed-pressure-current.log`: safety
and halo transfers pass; complete pressure continuation fails. Keep the strict
assertions and diagnose the wall correction/convergence behavior rather than
raising the sweep ceiling or enabling the UI over this failure.

Latest typecheck reports the same 14 Sparse-related errors and no Uniform
errors (`/tmp/fluid-mixed-latest-types.log`). `git diff --check` passes. No shared
Sparse simulation or rendering path has been edited in this continuation.

### Default-only integration scope and Full-Cycle continuation correction

Per the latest user direction, first live mixed support targets the default
numerical options. MacCormack, airborne momentum, liquid-only momentum,
V-based pressure rows, compaction, phi seeding/agreement, preserved/sparse
redistancing, non-relay orphan handling/rendering and isolated-body correction
fail closed. The host checks before preparation and again before consuming a
prepared state, so changing options after preparation cannot bypass the check.
Fine mode retains its controls. The three option-contract tests pass, including
validation against the actual Geometric defaults and rejection before any GPU
access or consumed-state mutation. This is wired into the existing QA entry
points; the live UI entry point still needs to use the same contract.

Cubic phi advection and ghost-phi draining are **on** in the actual defaults.
They remain required frame stages; the old parameter-file comment claiming
otherwise has been corrected. Default-only does not mean silently dropping
those operations. Existing isolated optional-stage coverage is not a promise
that those options are supported by the live mixed frame.

A concrete pressure traversal bug is now corrected: the descending mixed
Full-Cycle previously switched to a native V-cycle at 4h. It must continue
restricting correction RHS and minima to the native coarsest level, solve
there, then prolong and run the nested V-cycles back up to 4h. The continuation
now prepares separate Full-/V-cycle plans and the mixed traversal explicitly
selects the correct one. Both reuse the existing native kernels and fields;
the normal fine-grid plan is unchanged.

With unchanged 3 Full-Cycles, 4 V-cycles and six pre/post sweeps, the strict
757-owner wall fixture improved from pressure error **0.36076343** / residual
**13.3396616** to **0.00045645237** / **0.0183900185**. It still fails the retained
**2e-4 / 2e-3** thresholds (`/tmp/fluid-mixed-pressure-full-continuation.log`).
Per-cycle diagnostics show monotonic improvement, so rejection-triggered
recovery correctly stays idle. An all-coarse diagnostic also misses this
strict budgeted gate (error 0.000481844, residual 0.00350475); the remaining
convergence issue is not exclusive to mixed seams. Temporary diagnostic edits
were removed. No extra sweeps or relaxed assertions were retained.

The full default frame, live switching, ownership-aware presentation and final
whole-frame performance/memory measurements remain outstanding. Mixed is not
yet ready to enable in the UI.

Verification for this continuation: 164 Dawn checks passed across evolved native
transport, acceptance/rollback, halo transfers and surface cycles
(`/tmp/fluid-mixed-default-scope-regression.log`), plus three option-contract
unit tests. Both native V and Full continuation bridges match exactly and leave
the finest scratch prefix unchanged; the complete wall gate still fails at the
numbers above (`/tmp/fluid-mixed-pressure-default-scope-final.log`). Typecheck
continues to report 14 existing Sparse-related errors and no Uniform errors;
`git diff --check` passes. These Uniform-only edits do not touch Sparse or shared
presentation/topology code.

### Complete-frame profiling and remap/surface corrections (2026-09-26)

The earlier `/tmp/fluid-frame-profile64.log` did not measure a remap defect.
Its profiler allocated 8,192 timestamp queries, exceeding the WebGPU maximum
of 4,096; invalid query bindings invalidated the remap submissions. The test
now stays within that limit, bounds-checks query use, and checks GPU errors
before interpreting remapped mass. Profiling is separately opt-in with
`FLUID_MIXED_FRAME_PROFILE=1`.

The comparison also previously reseeded volume/phi while retaining evolved
velocity. It now creates a fresh host after measuring native frames, warms
both arms for four frames, measures eight subsequent frames, uses the resolved
sharpening controls, and reports canonical positive-face speeds rather than
stale inactive texels. Cleanup is explicitly disabled in both arms: default
orphan cleanup is still missing, so this remains a diagnostic, not acceptance.
`FLUID_MIXED_FRAME_LAYOUT=fine` runs the same unified frame without regions.

Corrections made in this continuation:

- Live remapping transfers wall-release metadata. Restriction marks a coarse
  wall patch released only if its entire source footprint was released;
  refinement inherits that classification. A new repeated-transition test
  checks nonuniform-volume conservation, all wall orientations, partial contact,
  and forbidden field allocation/shader compilation during edits.
- Redistancing retires drained positive plateaus only after proving no nearby
  nonpositive surface sample exists. The test covers both endpoints and mixed
  ownership, disabled draining, and a nearby contour on the outer wall.
  A per-tile evidence pass borrows a small prefix of native scratch before
  transport. All-positive tiles skip repeated corner searches; clipped tiles
  containing surface evidence retain the detailed check. No persistent field
  is added. The first uncached version was prohibitively expensive.
- Canonical vertex and unit-face sampling avoid unnecessary full owner/face
  construction. Tests retain poisoned inactive storage and seam continuity.
- The frame compiles momentum with its supported default numerical options.
  The same specialization runs every ownership layout; it is not a separate
  endpoint route. Direct GPU comparisons against the general momentum kernel
  agree within 1e-6 on canonical faces. Optional-stage tests still exercise
  the general kernel.
- Two Uniform test type errors were fixed. Typecheck now reports the same
  14 Sparse test/tool errors, with no Uniform errors.

Latest measurements (four warmup frames, then eight measured frames, 64³):

| Diagnostic | Native ms/frame | Unified ms/frame |
| --- | ---: | ---: |
| 50% manual region, timestamps enabled | 30.778 | 199.618 |
| 50% manual region, timestamps disabled | 34.033 | 192.174 |
| No regions, timestamps disabled | 31.584 | 140.313 |

The mixed layout has 118,784 owners. These separate runs are not an ABBA
acceptance benchmark, and both cleanup controls are zero. The region therefore
fails both the native speedup target and comparison with the unified fine
layout. At evolved frame 5, the cached profile attributes about 43.2 ms to
momentum, 18.0 ms to redistancing, 12.8 ms to forces and 8.0 ms to phi advection.
The prior uncached retirement diagnostic averaged 434.2 ms; the evidence cache
addresses that new cost but does not solve the original performance blocker.
Logs: `/tmp/fluid-frame-profile-evidence-cache.log`,
`/tmp/fluid-frame-evidence-unprofiled.log`, and
`/tmp/fluid-frame-evidence-fine-unprofiled.log`.

Verification:

- Ten CPU ownership/options/memory/hierarchy checks pass:
  `/tmp/fluid-mixed-current-cpu.log`.
- Momentum, forces, vertex transfer and the complete live-edit frame passed:
  `/tmp/fluid-mixed-frame-regression-current.log`.
- The final targeted run passes momentum/default-specialization parity,
  guarded live remapping and cached plateau retirement. The retained strict
  pressure continuation test still fails at error **0.00045645237**, residual
  **0.0183900185**, against **2e-4 / 2e-3**. Native Full-/V-cycle bridge parity
  still passes. No cycle budget or threshold changed.
  Log: `/tmp/fluid-mixed-final-dawn.log`.
- The profiled/unprofiled complete 64³ runs pass their finite-field and mass
  assertions. These are weaker than the outstanding hydrostatic, seam,
  symmetry and native-endpoint physical acceptance requirements.
- Typecheck: `/tmp/fluid-mixed-current-types-final.log`.

Production host scheduling, default dust/orphan accounting, runtime options,
scene edits, ownership-aware water/overlay publication, legacy-route/UI removal,
strict pressure convergence and accepted performance remain unfinished. This
continuation does not enable the production path over those failures. No Sparse
or shared presentation implementation was changed; no new Sparse gate result
is claimed.


### Correctness resumption: default cleanup

Default regular/orphan cleanup and physical discarded-mass receipts are now
implemented in the standalone frame, using borrowed volume fields and immutable
neighbourhood reads. Fine/coarse/mixed cleanup oracle checks and the evolved
five-frame remap fixture pass with default cleanup enabled. Native fixed-point
pressure parity passes; the strict fixed-budget convergence gate remains
unchanged and failing. Production host, publication and UI work remains open.
See [the current execution plan](uniform-mixed-performance-bridge.md) for the
current priority, limits and verification logs.
