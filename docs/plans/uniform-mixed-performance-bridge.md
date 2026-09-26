# Uniform mixed architecture: minimum performance bridge

## 2026-09-26: Water box investigation and fail-closed contract

The latest product requirement supersedes earlier recovery discussion below:
production uses one mixed GPU algorithm, with no CPU substitute, pressure
restoration, or recovery sweeps. Unsupported execution fails before projection.
The convergence checkpoint now only reads residuals, latches failure and reports
convergence; it cannot bind or change pressure. Its accepted-pressure copy and
indirect recovery dispatch have been removed. The GPU regression verifies that
failure remains latched even if a later checkpoint has a smaller residual.

The supplied Water box URL uses a 24×16×16 lattice, a coarse region at
66.6667–100% X / 0–50% Y / 0–50% Z, and pressure tolerance 0.001. Preserve this
tolerance when comparing. Diagnostic per-pass timestamps measured pressure at
8–36 ms while whole advances reached 104–203 ms; surface and momentum account
for much of the real slowdown. The previous cross-submission timestamp chain
could include host receipt waits. Each submission now closes and resolves its
own hardware timestamp segment before its receipt wait. Missing or invalid
hardware timing is reported as unavailable, never replaced with CPU time.

Native 4h continuation topology and coefficients are now initialized once per
pressure solve and reused across cycles. Reuse matches a fresh native traversal
bit-for-bit in both V and Full bridge checks. Pressure seam preparation uses
immutable ownership lists; regular/seam smoothing specializes the same operator.
The exact Water box 12-step diagnostic retains the same accepted residuals.
These changes do not establish the required 98% throughput.

The exact browser URL was exercised through 3.2333 s of ordinary playback,
with no console errors. Warmed UI timing was 40.7 ms for multigrid and 135.3 ms
per whole advance. Its stale native recovery/pass-count descriptions were
replaced with coupled-solve state and the actual accepted residual. GPU stage
labels no longer describe receipt waits when using the segmented mixed trace.

Additional checks: the complete frame and momentum matrices pass after removing
recovery. A wall-probe exclusion uses the already published frame maximum speed;
surface tests verify bit-identical released-wall continuation. Pressure transfer
reconstruction was investigated and reverted: the strong 2h collar already
preserves interior affine corrections, so extra reconstruction was unnecessary.
The new affine invariant test records that fact. Dynamic bounds for deeply
nested general vertex-sampling loops are being measured; this is not an accepted
performance gain. The acceptance benchmark now checks the original 3% resident
memory ceiling explicitly as well as throughput.

Unresolved: two hydrostatic cycle fixtures exceed the unchanged 0.001 residual
limit (0.00138235 and 0.00436735). The strict manufactured wall continuation case
also remains above its original limits. Updating reconstruction on both Jacobi
substeps did not resolve convergence and was reverted. No tolerance or cycle
budget was loosened. Production UI and performance acceptance remain unfinished.


Architecture review, 2026-09-26. This is the execution plan for closing the gap;
it supersedes the stage-by-stage milestone order in
[the implementation history](uniform-manual-mixed-resolution.md). It does not
change the acceptance limits or authorize a second solver.

## Current priority: correctness and product enablement

User direction, 2026-09-26: pause performance optimization briefly and finish
correctness plus product/UI enablement before returning to the 98% gate.
The architectural decision below remains unchanged.

1. Resolve the coupled pressure regression with the existing cycle budget and
   tolerances. Establish evolved-state correctness, including default cleanup,
   conservation accounting, separating walls and live ownership remapping.
2. Connect the same mixed frame to the normal host lifecycle and ownership-aware
   presentation. Apply region edits at completed frame boundaries; preserve
   evolved state, time and controls. Replace preparation/coarse-switch controls
   with working manual region editing and truthful resolution status.
3. Exercise draw, move, resize and remove in the product, including paused edits
   and edits while a frame is pending. No regions must use fine ownership in
   the same pipeline. Run the repository Dawn regression gate after integration.
4. Resume performance work with the one-coarse-air-tile case, default controls,
   matched publication/completion boundaries and >=98% pre-mixed throughput.

The UI must not claim regions affect simulation until the host and publication
consume the same accepted ownership generation.

### Resumed correctness work, 2026-09-26

The standalone frame now includes native regular dust flooring followed by
orphan cleanup, once after transport and before surface-volume correction.
Each pass reads an immutable volume field. Surface evidence and neighbouring
liquid/cluster evidence protect deposits; mixed neighbourhoods integrate
canonical owner fill over physical overlap, including fine evidence beside
coarse owners. This stage assumes the existing empty-solid mixed domain;
embedded-solid open fractions remain required for production integration.
No simulation fields are allocated. The existing accounting/receipt buffers
increase by 64 bytes in total, and frame receipts report owner counts and
physical discarded mass in native quantized units.

`tests/uniform-mixed-cleanup-dawn.test.ts` passes fine/coarse/mixed checks against
an independent box-overlap oracle, including disabled cleanup, negative dust,
surface protection, dilute orphans, dense evidence, clusters, threshold equality
and physical mass-counter quantization. The evolved five-frame remap fixture
now uses default cleanup and checks mass against the discarded-mass receipt.
It passes, including the strict-failure projection-withheld check.
Log: `/tmp/fluid-cleanup-frame-final.log`. Type checking reports no Uniform
errors; it still fails on unrelated Sparse/SVO files in the shared working tree.

A new native fixed-point assertion verifies that the manufactured constant
pressure plus separating-wall halo is unchanged by a native V traversal within
2e-6. It passes. Explicit strict coarse accuracy also leaves the existing
757-owner convergence failure unchanged (0.0004564524 / 0.0183900); no cycle
budget or acceptance threshold was changed. Operator fixed-point parity does
not establish convergence in the fixed budget, or production baseline parity.

Production host/publication/UI integration and the strict pressure gate remain
unfinished. Performance optimization remains paused. The prior Sparse regression
failure is still unresolved; this Uniform-only cleanup change does not alter
Sparse, presentation, terrain or live-edit infrastructure.

### Implementation checkpoint

Implemented in the standalone unified frame:

- Frozen geometric neighborhood masks, reused after unchanged ownership, and an
  ownership hold spanning asynchronous pressure receipts.
- One start-of-frame sampling/extension support census, with the native two-tile
  fine sampling reach and one additional extension tile. This is not yet the
  full transport/donor/pressure work selection plan.
- Direct regular vertex/face addressing, canonical sampling across seams, native
  prediction-face eligibility, and reuse of current cell geometry for forces.
- Sharpening face admission cached once across eight sweeps in borrowed scratch.
- Native V-first adaptive correction scheduling, strict failure before projection,
  and refusal to resume a partially advanced failed frame without reset.

The single-air-tile diagnostic asserts exactly one 4h air tile and the required
26-tile 2h collar, with fine ownership elsewhere. The latest profiled measurement
before the final extension/force adjustments was 73.324 ms mixed versus 24.904 ms
native (~34% throughput), with 821,028 additional bytes / 54,612,876 native bytes
(~1.5%). Cleanup is disabled in both diagnostic arms and mixed publication is not
integrated, so these are not production acceptance measurements.

Pressure checkpoint diagnostics show no rejected cycle in the strict wall
fixture. Its 3 Full + 4 V cycles finish at pressure error 0.0004564524 and residual
0.0183900, failing the unchanged 0.0002 / 0.002 limits. All-coarse ownership also
fails (0.000481844 / 0.00350475). Simply changing to 4 V + 3 Full does not fix it
(0.000689745 / 0.0255903). This narrows the investigation beyond seam geometry;
it does not justify extra cycles or relaxed assertions.

### Correctness investigation and repository gate

The native-only diagnostic outer traversal also misses the manufactured
constant-pressure target: after 3 Full cycles its maximum pressure error is
0.007066; the following unaccepted V traversals increase it to 0.014626.
This diagnostic builds outer corrections on the CPU and calls the real native
4h continuation; it is not the production native acceptance/recovery driver.
Validate that baseline/fixture pairing before interpreting the strict mixed
failure as a seam defect. The strict assertions remain unchanged.

`npm run test:dawn:sparse-cm12` was run serially and **failed**. Its final report,
not its progress labels, is authoritative. Several lanes timed out; mini64
reported 228.7206 ms against its existing 110 ms ceiling. Tall Cells Hills
reported 18 remaining bricks to the far wall against maximum 6. The suite
exhausted 480,000 ms before outside-tank symmetric collapse. These failures are
not attributed to the Uniform changes without an isolated comparison. No
limits were changed. The report is in
`/tmp/fluid-frame-plan-sparse-regression.log`.

Targeted frame, remap, momentum, sharpening, extension, force-cache parity,
stencil, frame-census, pressure-operator and reconstruction checks passed.
The projection-withheld test confirms velocity remains unchanged on failure
and the partially advanced frame cannot resume. Type checking still reports
14 errors in existing Sparse test/tool files, none in the Uniform changes.

## Decision

**Establish one mixed ownership/topology and execution plan at the frame
boundary, then run the ordinary Uniform approach against that plan.**

Fine, 2h and 4h are work items in the same pipeline. No regions means all fine
ownership. There is no layout-dependent choice between frame implementations.
Regular stencils and interface stencils are local work classes used together in
a mixed frame, not separate fine and coarse solvers.

The current implementation got the ownership representation and many seam
operators working, but composed them into an execution path that omits much of
native Uniform's work selection and reuse. Reducing owner count cannot pay for
repeatedly executing expensive general stencils over the remaining domain.

## What made native Uniform fast

Reviewed both HEAD (`d1ee4c66`, before the uncommitted mixed work) and the current
native implementation. The optimizations below are not speculative additions.

| Native mechanism | Relevant source | What the mixed path must retain |
| --- | --- | --- |
| A support census and separate fine-sampling, extension-shell, transport and donor work sets | `webgpu-uniform-reference.ts`: `encodeSupportTopology`; `uniform-volume.wgsl.ts`: two-level seed/dilation and `uvTransportSkip` | Ownership says where values live; support says which values need work. They are separate decisions. |
| Far-air and individual-face prediction exclusion | `webgpu-uniform-reference.wgsl.ts`: `semiLagrangianAdvection`, `uvPredictionLive` | Do not trace momentum that projection will provably discard. Preserve wall, source, solid and liquid dependencies. |
| Fine sampling near supported liquid and a 4h extension table elsewhere | `webgpu-uniform-reference.ts`: two-level controls and extension bindings | This was already a sampling optimization. It did not change simulation ownership. Manual resolution must coexist with it. |
| Cached cell/face geometry through eight sharpening sweeps | `uniform-volume.wgsl.ts`: `uvCacheSharpenCells`, `uvCacheSharpenFaces`; host `encodeGeometricVolume` | Refresh budgets as V changes; reuse geometry while phi is unchanged. Preserve zero-flux skips and the existing planar scratch layout where applicable. |
| Bounded surface work and retirement of obsolete support | `uniform-volume.wgsl.ts`: `uvPhiFarAir`, `uvNoNearbySurface`, `uvOrphanDust`; surface-volume support windows | Dust accounting and phi retirement help make work genuinely disappear. They are part of the default workload, not optional benchmark cosmetics. |
| Specialized regular indexing, cached coefficients, shared scratch and batched passes | native shader specialization, pressure multigrid and `uniform-scratch-arena.ts` | Preserve cheap regular stencils and lifetime reuse. General topology reconstruction must not become the price of every texture tap. |
| Adaptive accepted pressure continuation | `uniform-pressure-continuation.ts`, host adaptive pressure loop | Preserve cycle choice and coarse accuracy; the latest mixed contract removes rollback/recovery and fails closed. Native already reads current-frame receipts; eliminating every receipt is not the missing architecture. |

Historical evidence supports these mechanisms, without predicting a mixed
speedup: the sharpening report records proposal time falling 3.47 → 1.34 ms;
the extension report records 7.67 → 4.78 ms; the retirement report records a
large contraction of phi-only work. These concern different scenes/stages and
must not be added together as a minidam64 savings estimate.

References:
[sharpening](../benchmarks/uniform-sharpening-performance-2026-09-24.md),
[extension](../benchmarks/uniform-velocity-extension-performance-2026-09-24.md),
[retirement](../benchmarks/uniform-splash-retirement-2026-09-24.md).

## Necessary differences, and avoidable overhead

Mixed resolution genuinely requires:

- h/4h manual bounds and an internal strong h/2h/4h collar;
- conservative remapping when ownership changes, before advancing the frame;
- one canonical face-patch flux, with consistent physical area/volume factors;
- authoritative vertices and reconstruction of hanging samples;
- mixed donor footprints for characteristics crossing ownership boundaries;
- coupled pressure restriction/prolongation and interface terms in the existing
  hierarchy, continuing through native levels below 4h;
- presentation that interprets canonical mixed fields rather than inactive fine
  texels.

It does **not** require tracing all owners, resolving neighboring tile widths
repeatedly inside every sample, recomputing fixed sharpening geometry eight
times, expanding fields to fine resolution, or maintaining a second frame
scheduler. A long characteristic still needs query-dependent sampling; knowing
its receiving tile alone cannot prove its entire path is regular.

Specific avoidable costs in the current implementation:

1. `UniformMixedOwnership` has resolution worklists, not the native stage
   support worklists. Momentum, forces, surface operations and sharpening mostly
   use `dispatchAll`. Redistancing has a value-band predicate, but still launches
   all owners. Compact ownership is not compact active work.
2. `uniform-mixed-velocity-sampling.wgsl.ts` can examine 27 neighboring tiles to
   choose blending weights, then perform nested tier interpolation and face
   lookup. `uniform-mixed-vertex-sampling.wgsl.ts` similarly resolves authority
   within sampling. These operations multiply through RK traces, cubic phi,
   Newton iterations and viscosity stencils.
3. `uniform-mixed-sharpening.ts` resamples face phi inside every proposal sweep;
   native caches that fixed geometry. Forces also reconstruct center phi even
   though the frame already owns a center-phi field for the corresponding state.
4. Mixed pressure reconstructs slopes and freezes RHS across owners each sweep,
   although the correction is identically zero for regular equal-width
   neighbors. Existing tier dispatch already omits absent tiers; this is not a
   claim that all-fine launches all six color kernels. Interface work should be
   limited by its actual dependencies while retaining one operator and solve.
5. Frame orchestration has drifted from the native policy. The mixed loop tries
   Full-Cycles before V-cycles. Native adaptive plans are V-first and can jump
   to Full-Cycles and tighten coarse accuracy on a stall. More seriously, the
   mixed frame projects even if its final receipt says unconverged, whereas the
   native adaptive host withholds projection and reports failure. Reuse the
   native policy rather than independently reconstructing it.

The latest diagnostic gives native ~31.6 ms versus unified all-fine ~140.3 ms,
and native ~34.0 ms versus mixed ~192.2 ms. The large all-fine regression proves
that seam work alone is not the explanation. These are separate short runs,
not a controlled decomposition. Cleanup was disabled, which also disables
native transport-tile scheduling; native production performance is therefore
not represented by that control. Native includes publication work that the
standalone mixed test lacks. Those runs locate a problem, not its exact cost
or an acceptance baseline.

## One frame plan

At the boundary, apply pending scene changes, validate region constraints and
remap evolved fields if ownership changed. Establish a single generation of:

- owner ranges by width, canonical face/vertex addressing, boundary relations
  and the h/2h/4h pressure hierarchy;
- regular-stencil versus interface-stencil classifications, including the
  required neighboring support for interpolation and reconstruction;
- conservative work envelopes for extension, phi, transport receivers/donors,
  prediction, sharpening and pressure, with compact dispatch information;
- the scratch layout and lifetimes needed by those stage consumers.

Unchanged ownership geometry can be reused from the preceding frame. The frame
still selects one coherent snapshot; “once per frame” does not mean rebuilding
unchanged metadata or compiling shaders each frame.

Build work envelopes from V, phi, boundaries, sources and movement, with closure
under the actual stencils and trace reach. Width alone is not a support test.
Keep global mass accounting global even when expensive surface work is bounded.
A newly active region and both ping-pong fields must have defined state; stale
scratch or inactive texels cannot become donors by accident.

The snapshot does not freeze evolving physics. Advection changes phi and V;
pressure setup consequently updates liquid masks, coefficients and RHS on the
fixed ownership graph. Pressure iterations change pressure and reconstructed
slopes. Stage-local masks may narrow the conservative envelope, but cannot
silently discover needed work outside it. Either prove closure or conservatively
include that work in the frame plan. No mid-frame resolution replanning.

Cache topology/addressing at its geometric lifetime, derived geometry at the
lifetime of its input fields, and numerical values only until those fields
change. In particular, do not reuse start-of-frame phi geometry after advection,
or pressure slopes after an iteration modifies pressure.

Use small tile metadata, existing work storage and borrowed transient scratch.
Avoid a dense per-cell/per-face adjacency catalogue or another full 2h/fine
sampling field. Any proposed cache needs an explicit simultaneous-lifetime byte
budget under the existing 3% resident-payload ceiling.

## Minimum implementation work

### 1. Put ownership and work selection behind one shared frame-plan contract

Extend the existing ownership/work machinery rather than introducing another
solver capsule. Separate resolution membership from stage support. Resolve
regular/interface addressing once; construct the conservative dependency work
sets once for the frame and give all stages the same generation.

First consumers: momentum/forces and phi, because these dominate the current
profile. Retain native far-air/face eligibility and sampling-support semantics.
Do not start by shaving arithmetic from their all-owner general kernels.

**Required evidence:** empty, full, newly activated and long-trace cases agree
with exhaustive evaluation; poisoned inactive values are never read. No field
allocation or pipeline compilation on region edits. Lists and scratch fit the
live memory budget, including worst-case permitted layouts.

### 2. Make that plan drive the existing numerical approach

Keep native stage order, controls, transport normalization, sharpening sweeps,
extension policy and pressure acceptance. Adapt their addressing and stencils:
regular work uses direct width-strided access; interface work uses the existing
canonical-patch and hanging-vertex rules. Both are jobs in the same stage.
A sample crossing a seam uses the seam rule regardless of its receiver's class.

Reuse existing geometry caches and borrow scratch for bounded interface work.
Retain the mixed transport/remap/operator tests and useful arithmetic helpers;
consolidate duplicated stage bodies and scheduling as native equivalents become
ownership-aware. The current complete-frame class is integration scaffolding,
not a second permanent production scheduler.

Restore default dust/orphan accounting and normal runtime controls. Route mixed
pressure through native adaptive policy, including withholding an unconverged
projection. Diagnose the strict manufactured failure against matched native
operators and transfer boundaries; do not change thresholds, add cycle budget,
or launch a separate pressure-solver redesign as a shortcut.

**Required evidence:** same-input regular-stage parity first, then coupled seam
and strict pressure gates. All-fine uses this very pipeline and must meet the
unchanged ≤2% overhead target. Mixed must reduce actual expensive work and then
complete-frame time; owner count alone is insufficient.

### 3. Finish the production boundary, once

Integrate the plan into `WebGPUUniformReferenceSolver`'s existing advance,
pending-frame, runtime-update, scene-edit, diagnostic and destruction lifecycle.
Use its normal geometry/source handling with the same canonical mixed authority.
Do not silently drop terrain, body, inflow or simulation-control semantics.
Unverified experimental numerical options may explicitly fail closed.

Publish completed fields together with their topology generation. Water and
grid-overlay consumers sample that ownership through the level-set consumer ABI;
they must not see partially remapped fields or require per-frame dense expansion.
Remove prepared fine/coarse switching, activation controls and obsolete routes
when the unified path replaces them. Keep ordinary simulation controls.

**Required evidence:** live add/move/remove regions, paused edits, both uniform
limits, physical hydrostatics/dam fronts/symmetry/seams, coherent rendering and
no restart/recompile. Run the Sparse gate when the shared presentation/live-edit
integration reaches its required scope.

## Measurement needed to judge the bridge

Use a stable pre-change native reference from the existing code/benchmark, not
the standalone mixed frame as its own baseline. Preserve the user's working tree.
Compare native, unified all-fine and unified 50/50 minidam64 with fresh matching
initial fields, default controls, timestep policy and simulated duration.
Record where the coarse region overlaps supported liquid, not just its volume.

Use repeated interleaved runs; separate ordinary wall timing from pass profiling.
Include the same completion/publication boundary, and report rendering separately.
Count active owners/faces, regular versus interface work, sampling fallback work,
passes, CPU encoding, receipt waits and actual live GPU payload bytes. Start with
same-input stage comparisons to separate implementation cost from trajectory
changes, then require evolved complete-frame physical/performance acceptance.

The minimum first code change is therefore **the shared frame plan and its work
selection consumers**, followed by native stencil/policy reuse. Additional
standalone kernels, blanket caches, altered sweep counts, a fine-only execution
route, or more isolated performance claims do not bridge the architectural gap.

### Production integration in progress, 2026-09-26

The production `uniform-volume` factory now always constructs the host with
mixed ownership. Region edits update the same borrowed fields at completed
frame boundaries. A paused topology fence invalidates the retained water mesh
and updates resolution status. The preparation/activation controls are removed
from parameter normalization and replaced by live resolution counts.

Water, combined phi/volume, density, velocity, pressure, divergence, wall release
and sampling-support overlays consume canonical ownership. Pressure scratch is
cleared on ownership changes so a paused overlay cannot read old owner indices.
Diagnostics reduce physical mass over canonical owners without dense expansion.
Shared native source WGSL supplies drop quadrature and inlet volume/velocity;
projection restores the prescribed inlet velocity. The host test checks paused
and pending edits, stable publication bindings, time continuity, drop mass and
canonical diagnostic mass. Continuing-inlet/tap-off checks are added and pending.

A cached pressure bind-group bug is fixed: explicit reserved buffer capacities
must not shrink to the first edited layout's active count. The regression starts
mixed, then returns to all-fine ownership. The host lifecycle/drop/diagnostics
check and 27 pressure-cycle checks passed. Shader validation passes for the
ownership-aware water and grid overlays. Width-specialized surface and momentum
jobs now use the existing tier worklists in the same pipeline; no endpoint
solver selection is introduced. The momentum regression passed all seven
layouts. Their performance impact is not yet measured.

The strict manufactured pressure fixture still failed at 0.0004564524 pressure
error and 0.0183900 residual on the integration checkpoint. A subsequent test
sets native coarse accuracy explicitly to strict (outer tolerance zero alone
leaves the relative inner gate enabled). Native D4 directional summation and
regular six-neighbor coefficients are being restored in the mixed operator.
Results are pending; no cycle count or acceptance threshold has been relaxed.

The required Sparse gate was rerun serially after presentation integration:
`/tmp/fluid-mixed-production-sparse-gate.log`. It failed the 480-second suite
budget. Mini64 measured 206.1107 ms against 110 ms; multiple lanes timed out,
and Tall Cells Hills failed its front-progress assertion. Five lanes passed.
These resemble the prior gate's failures, but causality is not isolated from
concurrent Sparse/SVO work in the shared checkout. No lane or ceiling changed.
Type checks have no Uniform errors but still fail in Sparse/SVO test/tool files.

Remaining release blockers include embedded-solid/terrain/body coupling,
strict pressure convergence, live browser interaction verification, and the
98%/50:50 performance gates. Do not describe this as production complete.
`tools/benchmark-uniform-mixed-production-dawn.ts` provides default-cleanup,
ABBA-ordered, accepted-completion timing for fine, one-air-tile and half-domain
ownership; normal runs assert the original targets, `--diagnostic` reports
failures without claiming acceptance. Rendering time is explicitly excluded.
