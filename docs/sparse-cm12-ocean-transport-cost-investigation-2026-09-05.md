# Ocean-seiche transport cost investigation — 2026-09-05

## Live replacement investigation

A production browser run subsequently confirmed three background resident
publications by 38.7667 simulated seconds. No stale candidates were reported.
Preparation nevertheless took 96.3 seconds for the first replacement and 54.2
seconds for a later one. The maximum measured simulation-worker realization slice
was 402.7 ms. This is not a realtime result, even though physics continued during
most of preparation. The paused pressure residual was 4.09e-4.

A separate CPU-only reconstruction of an ordinary ocean replacement recorded
335 resource operations in 4.91 seconds, including 51 shader modules and 189
compute-pipeline compilations. It allocated 380.5 MB of buffers and recorded
213.3 MB of uploads (decimal MB). Its 399,664-cell / 1,301,071-row capacity includes
512 reserved B8 frontier pages; it is a construction profile, not the accepted-cell
count of the browser run. Structured cloning took another 86 ms in Node.

The remaining browser wall time cannot be attributed to CPU packing alone.
Replay awaits all 189 pipeline requests. The compilation manager's direct API
assigns each a fresh request key rather than caching equivalent pipelines, and
many shader sources embed topology-dependent offsets. The next measurement must
attribute the maximum synchronous slice to its specific resource operation and
measure final handover separately. Both are now exposed in diagnostics.

A CPU comparison using 64 synthesized wet B4-to-B8 requests near the raised wave,
followed by ordinary physical 2:1 closure, found only **2 of 51 identical shader
modules and 2 of 189 identical pipeline descriptors**. Entry points and structural
binding layouts all matched; embedded generation data changed the other 187 shader
sources. Counts rose from 399,664 to 432,515 cells with the same 2,560 leaves.
This was a representative planning experiment, not a GPU-read policy receipt.
An exact pipeline cache alone would therefore save about 1% of requests. The next
structural step is a stable shader interface with generation-specific offsets/counts
in data buffers, coupled to reusable resident storage. Do not add a large cache and
claim that recurring pipeline compilation has been solved.

Changes made during this investigation:

- Release temporary recipe shader/encoder handles after their final construction
  use, retaining handles owned by the hydrated resident. Previously replay kept
  every module alive until the entire replacement finished.
- Protect the multi-read source snapshot with a short, revocable topology lease.
  Release it before expensive preparation. Urgent frontier growth and backed
  promotions revoke it; a revoked or mixed snapshot is rejected. The final stale
  guard still validates ownership before transferring the latest GPU field banks.
- Show the publication count even while another update prepares, plus stale count,
  CPU slice attribution, preparation duration, and maximum handover duration.

The new changes require the subsequent GPU regression and instrumented browser
measurement; the earlier three-publication observation does not validate them.

The reported 36.6 ms transport band is consistent with three concrete execution
costs: repeated world-directory resolution inside every face characteristic,
capacity-sized velocity-extension dispatch, and a surface/air band that contains
nearly all represented cells despite strong compression of deep water. The UI's
“dirty rows only” description does not describe the current face kernels.

This investigation used source inspection, existing timestamp receipts, and a
CPU-only reconstruction of the ordinary initial atlas. No GPU workload or browser
was started while the user's UI was running. The screenshot is a single sample,
not a controlled before/after benchmark; it cannot establish the time spent in
each substage or the number of supported rows that actually traced.

## Ordinary scene census

Reconstructed with `createOceanSeicheScene`,
`adaptiveMassPresentationDimensionsForScene`, and
`initializeSparseBrickAtlasFromScene` using B8, one surface ring, and the ordinary
one-ring initial surface coarsening bias. This exactly reproduces the screenshot's
2,560 resident leaves. The CPU grid contains 137,520 cells and 416,335 gradient
rows. `sparseCM12InitialActiveBrickKeys` selects all 2,560 leaves.

| Physical cell width | Leaves | Represented cells | Wet cells | Represented wet volume in finest cells |
|---|---:|---:|---:|---:|
| 1h | 70 | 35,840 | 10,240 | 10,240 |
| 2h | 1,530 | 97,920 | 25,600 | 204,800 |
| 4h | 400 | 3,200 | 3,200 | 204,800 |
| 8h | 400 | 400 | 400 | 204,800 |
| 16h | 140 | 140 | 140 | 573,440 |
| 32h | 20 | 20 | 20 | 655,360 |

The 16h/32h interior represents 66.3% of liquid volume using only 160 cells.
However, 97.3% of all represented cells are still in the 1h/2h surface and air
band. Of 137,520 represented cells, 97,920 are initially dry. Dry support cannot
simply be discarded: velocity extension must give a moving front a velocity
before liquid reaches it. These numbers explain why making already-compressed
deep water still coarser alone will not remove most transport work.

Construction entry point:
`lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts:593`.

## Face preparation: the largest target

`webgpu-sparse-cm12-resident.ts:6215` dispatches support publication across the
resident brick domain, followed by the BFA1 interior, seam, sparse-air, and dynamic
row programs. In `webgpu-sparse-cm12-resident.wgsl.ts:2009`, support publication
visits each accepted leaf's cells; there is no dirty predicate.

`sparse-cm12-brick-tile-face-address-program.wgsl.ts:30` calls
`prepareTransportFaceRow` whenever a row is valid and accepted. Its interior and
seam callers also contain no dirty-mask predicate. The dynamic suffix at
`webgpu-sparse-cm12-resident.wgsl.ts:3134` similarly visits accepted dynamic rows.
The cheap early exit in `prepareTransportFaceRow` is lack of extended velocity
support, not unchanged velocity. A quiet submerged row still traces.

The expensive part is the lookup shape:

1. `prepareTransportFaceRow` (`webgpu-sparse-cm12-resident.wgsl.ts:3089`)
   resolves two support points beside each row.
2. `traceFaceDeparture` (`:2082`) samples the initial velocity and midpoint
   velocity for a single RK2 substep. The caller samples the departure again.
3. Each trilinear sample (`:2049`) calls `faceVelocitySupportAt` eight times.
4. Every support query (`:1989`) calls `compactOwnerCellAt` (`:1905`), which
   resolves a WDR owner and reconstructs its accepted cell address before loading
   four cached state values.
5. `cm12WorldOwnerAt` (`sparse-cm12-world-directory.ts:301`) probes each dyadic
   span level until a hash entry matches; `cm12WorldLookupExact` (`:246`) performs
   atomic hash/coordinate/state checks and collision probing.

That is **26 owner resolutions per supported row for one substep**. Each extra
RK2 substep adds 16 more. If all initial rows had support, one frame would issue
10,824,710 owner resolutions just in face preparation. This is an upper-work
illustration, not a measured supported-row count. Boundary segment clipping adds
separate work.

The face support cache is physically cell-sized, which is necessary for vast
spaces, but it no longer gives cheap direct sample addressing. This tradeoff is
already visible in historical measurements: the rejected direct TEI face sampler
in `docs/sparse-cm12-masked-full-transform-plan.md:803` increased complete face
preparation from 3.0802 to 10.0925 ms median. This is a different implementation
and an older topology; it supports the lookup-cost diagnosis, not a claim that
the present 14.8 ms has been precisely attributed.

The next useful experiment is a generation-stamped, resident-bounded sampling
address cache or packet-local owner fast path with the WDR lookup retained for
seams and misses. Preserve the current interpolation lattice and corner order.
Do not restore an authored-finest-volume allocation: that would undermine the
large-space requirement. Also do not add a naked dirty skip: its receipt must
cover velocity, timestep, topology, body/boundary forces, and all characteristic
donors. Prior measured experiments that merely fused seam kernels or kept row
centres live regressed; see
`docs/sparse-cm12-performance-analysis-2026-09-01.md:647`.

The current UI text should be corrected immediately. It currently promises
“a stable submerged brick costs nothing here” at
`sparse-cm12-stages.ts:341`, which is false for this execution path.

## Velocity extension: a bounded existing optimization

`webgpu-sparse-cm12-resident.ts:6167` chooses direct dispatch in production.
For B8 the profile reserves eight 4³ packets per physical leaf irrespective of
accepted rung (`sparse-cm12-velocity-extension.ts:27`). The ordinary 2,560 authored
leaves therefore launch at least 20,480 workgroups per sweep; reserved world-growth
pages increase this. The current 512-page reservation makes that 24,576 groups.
Only **3,050 packets** are occupied in the reconstructed initial atlas: one for
each R1/R2/R4 leaf and eight for each of the 70 R8 leaves. There are eight sweeps
plus initialization. The complete-packet shortcut (`sparse-cm12-velocity-extension.wgsl.ts:255`)
avoids neighbor arithmetic in full wet packets, but follows dispatch and
topology/mask prologue work.

The existing `velocityExtensionPacketCompactionForQA` arm is directly relevant.
`compileSparseCM12AcceptedVelocityExtensionPackets`
(`sparse-cm12-transport-packet-authority.wgsl.ts:77`) traverses accepted leaves,
expands their actual rung, and appends only valid stable packets. It then runs the
same numerical initialization and eight sweep kernels through an indirect list.
Current ordinary Ocean would reduce the dispatch domain roughly eightfold; this
does **not** imply an eightfold stage-time improvement because useful neighbor
arithmetic and compiler/list overhead remain.

Existing matched 16-frame measurements are recorded in
`docs/sparse-cm12-min8-domain-work-analysis.md:593`:

| Scene/profile | Direct VEX median | Compact VEX median | Whole-frame result |
|---|---:|---:|---|
| Ocean with full-domain min8 region | 2.2938 ms | 1.5073 ms | 37.3555 → 36.8968 ms |
| Fine-biased mini64 | 3.1457 ms | 3.2768 ms | No established win |

Receipts are `artifacts/ocean-seiche-min8-vex-control-current-20260902.json`,
`artifacts/ocean-seiche-min8-vex-compaction-20260902.json`,
`artifacts/mini64-vex-control-20260902.json`, and
`artifacts/mini64-vex-compaction-20260902.json`.

The arm was retained as QA because the fine case did not benefit and the ocean
whole-frame win was modest. No numerical rejection was found in that report.
Its production form should cache the accepted packet image on topology changes
and choose direct versus compact dispatch from accepted occupancy, not scene ID.
The current list shares storage with the later transport compiler, which resets
and overwrites it every frame (`webgpu-sparse-cm12-resident.ts:6192`); persistent
reuse requires a separate list or a topology-owned accepted packet image.

Before enabling it generally, close these code-level gaps:

- An empty compact list dispatches no sweep workgroup, so the existing frame
  receipt writer, which runs only at `dispatchOrdinal == 0`, cannot publish the
  empty frame. Provide an explicit empty-domain completion path.
- The compact VEX decoder uses only `wid.x`
  (`sparse-cm12-velocity-extension.wgsl.ts:53`). Counts beyond 65,535 groups need a
  two-dimensional indirect dispatch/ordinal mapping. B16 and vast spaces can
  reach this even when current B8 budgets do not.
- Direct initialization clears masks for absent packets; compact initialization
  omits them. Prove that retired/rerung packet masks and cell-depth values cannot
  be consumed through stale addresses, or explicitly invalidate lifecycle deltas.
  This is a coverage requirement, not a demonstrated failing simulation.
- Atomic append changes workgroup order. Jacobi sweeps preserve dependency depth,
  but compare raw field/receipt results through rerung, activation, signed growth,
  and moving-solid cases; do not rely only on source-contract tests.
- The current stage-cost CLI switches to a separate constructor and changes other
  production optimization selections when `--vex-packet-compaction=1`
  (`tools/probe-sparse-cm12-stage-cost.ts:539`). A fresh A/B must isolate only this
  selection before drawing current production timing conclusions.

## Conservative transport

The stage contains receipt clearing, trace, deficit scatter, and conservative
gather (`webgpu-sparse-cm12-resident.ts:6240`). Trace performs RK2 interpolation
and caches an eight-corner departure stencil; scatter can perform a second
forward trace for deficient donors and adds density, gamma, and three momentum
components atomically. See `webgpu-sparse-cm12-resident.wgsl.ts:3335` and `:3378`.

The existing hybrid coarse-cell packing addresses low lane utilization for
R1/R2/R4 packets. That does not erase the 133,760 cells concentrated in the
ordinary fine surface/air band. Establish trace/scatter/gather times and dirty
packet/lane counts before selecting the next change. The screenshot's single
9.96 ms sum cannot distinguish sampling, atomics, or overbroad selected work.

## Measurement order after the browser releases the GPU

1. Capture the ordinary B8/P8, paper-timestep scene with existing substage seams:
   support publication versus row preparation; VEX initialization versus sweeps
   versus packet authority; trace versus scatter versus gather. Keep ordinary
   scene settings and record accepted cell/row/packet counts at each sample.
2. Run an isolated compact-VEX A/B with identical other production options,
   including empty-domain and lifecycle correctness checks. The old min8 receipt
   is supporting evidence, not current production acceptance.
3. A/B an owner-address cache for face preparation with unchanged numerical
   sampling, checking both frame time and mass/front/symmetry receipts.
4. Run the canonical Dawn regression gate serially with all browser GPU work
   stopped. Do not adjust correctness assertions or timing ceilings.

The multi-second topology pause is separate from these per-step GPU costs.
Removing it requires bounded asynchronous preparation and a bounded GPU
publication transaction; accelerating these transport kernels alone will not
make whole-resident reconstruction realtime.

## Implementation follow-up

A bounded production VEX schedule is now implemented, pending GPU validation:

- A dedicated VEX activity-tail list and generation/slot receipt cache accepted
  stable packet IDs; the later transport compiler cannot overwrite them.
- Three short schedule dispatches check the topology stamp, rebuild only when it
  changes, and seal arguments. A topology-changing frame executes the full direct
  domain to preserve retired-mask invalidation. Subsequent frames use the compact
  list below 75% accepted packet occupancy and direct dispatch otherwise.
- A separate 12-byte indirect argument record supports two-dimensional execution.
  Empty domains run a sentinel group that publishes the usual completion receipt.
- The VEX QA header now reports accepted packet count, selected mode, rebuild,
  schedule generation, and scheduled workgroups. Existing numerical kernels and
  eight recurrence depths remain unchanged.
- Face-stage UI text now says “supported rows” and describes the actual work.

CPU schedule/layout, packet-prologue, and stage-partition tests pass (24 tests).
No new TypeScript errors were found in these changes; repository-wide typechecking
still reports unrelated errors. No GPU timing or numerical acceptance is claimed
for the new schedule until the browser can release the GPU for isolated Dawn runs.

## CPU preparation recipe profile

A CPU-only ordinary-Ocean recording on 2026-09-05 used the initial B8/P8 atlas,
its ordinary SolidWorld, and 512 reserved world-growth pages. Atlas construction
took 257 ms, `recordPreparedGeneration` took 4,914 ms, and an explicit
`structuredClone` of the resulting recipe took 86 ms. The actual worker uses
transferable buffers, so the clone measurement is not its exact handoff cost.

The recipe has 335 operations: 34 buffer creations, 31 uploads, 51 shader-module
creations, and **189 asynchronous compute-pipeline compilations**, plus layouts,
bindings and SolidWorld initialization commands. Total shader text is 3.37 MB;
the largest module is 166 KB and the largest simulation module is 96 KB.

Buffer allocation totals 380.5 MB, dominated by topology/worklists (118.8 MB),
resident state (101.0 MB), the transport execution image (56.8 MB), and candidate
fields (45.9 MB). Upload source data totals 213.3 MB. Replay splits uploads into
bounded chunks, but individual buffer creations and shader-module driver calls
remain indivisible. The 399,664-cell/1,301,071-row resource capacities include
262,144 cells and 884,736 rows reserved for those 512 future pages, above the
137,520 cells and 416,335 rows in the initial authored grid.

The main task subsequently observed successful publications with zero stale
retries, taking 54–96 seconds to prepare while physics continued. Therefore
continual stale rejection is a potential failure mode, not the measured cause
of those long preparations. The roughly five-second CPU recording does not
explain the full elapsed time: realization and driver compilation account for
the remaining work/contention. The largest observed synchronous slice needs its
new operation label before attributing it to buffer creation, shader-module
creation, or upload.

The compilation manager's direct-pipeline path uses a new `direct:${sequence}`
job key on every call (`lib/core/gpu-compilation-manager.ts:458`), so recipe
replay asks it to compile all 189 pipelines again. Many WGSL constants depend on
arena offsets and topology capacities. Stable arena layouts and shared pipelines
are the route to removing that recurring cost; merely moving construction to a
worker cannot remove it.

Source capture now uses a short topology lease across its separate readbacks and
releases it before background construction. It does not hold optional adaptivity
for the 54–96-second preparation interval. Reachable frontier growth, activation,
and executable promotions revoke the GPU lease immediately; the capture checks
the lease and epoch after its last read, and final publication still rejects a
changed topology. Publication duration and the largest realization operation
are exposed independently for diagnosis.

An exact CPU comparison tested whether a small physical refinement could reuse
those pipelines. It requested promotion of 64 wet B4 leaves near the raised-wave
end of ordinary Ocean, using the ordinary request budget and the real CPU 2:1
closure. These requests were synthesized; they were not read from a GPU frame.
The target retained 2,560 leaves while cell capacity rose from 399,664 to 432,515
and row capacity from 1,301,071 to 1,398,328. Recording took 5.02 and 5.57 seconds.

Only **2 of 51 shader-module sources** and **2 of 189 pipeline descriptors** were
identical. Every entry-point name and structural binding layout matched; changed
shader text caused the other misses. Comparison used source SHA-256, entry point,
specialization constants, and recursively resolved layout descriptors, excluding
labels and symbolic resource IDs. The reusable pipelines were
`clearSparseCM12TransportReceipts` and `finalizeIncrementalActivityMasks`.
Consequently a pipeline cache alone would barely help this representative change;
generation-independent shader addressing must precede meaningful reuse.
