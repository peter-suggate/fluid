# Sparse CM12 masked full-transform migration plan

Status: proposed, revised 2026-08-23.

Scope: every Sparse CM12 frame operation except the iterative pressure solve. The
pressure-system build is in scope; MGPCG iteration is not. Phase 1 is one vertical
slice through velocity extension. This document is an implementation and measurement
plan, not an authorization to preserve a second production path.

Related analysis:

- `docs/sparse-cm12-paper-data-transformation-map.md`
- `docs/sparse-cm12-compiled-topology-handoff.md`
- `docs/sparse-cm12-non-pressure-temporal-coherence-handoff.md`

## 1. Decision

Treat a frame as a small number of data transformations over one topology image that
is immutable for the frame:

```text
derived = predictor(T[current], state[current], inputs)
system  = assemblePressure(T[current], derived)
pressure = solve(system)
result  = correctAndClassify(T[current], derived, pressure)
T[next] = compileTopologyDelta(T[current], result.topologyIntent)
```

The transforms may still contain necessary numerical sweeps. What they must not do is
construct a new persistent control plane around each sweep.

The common execution shape is a **masked full transform**:

1. Dispatch over every packet in the accepted compiled sparse domain.
2. Read one 64-bit dynamic mask for the packet.
3. Exit uniformly if it is empty.
4. Resolve selected lanes through the immutable TEI/IBO/ITR operator image.
5. Write fields and, where needed, the next packet mask.

"Full" means the accepted sparse topology, not the dense finest lattice. A consumer
does not compact a mask merely to avoid launching empty packet workgroups. A producer
owns and completely overwrites its output mask plane; command ordering is the mask's
lifecycle.

We explicitly accept that shared mask production and maintenance may initially make
the total frame slower. Phase 1 succeeds if the migrated stage becomes materially
faster, the physics is preserved, the old stage-specific authority is deleted, and
the resulting mask/operator contract is credible for later stages. We will optimize
or amortize shared mask maintenance only after more than one stage consumes it.

This plan has two independent performance axes:

1. **Control plane (T1/T5):** enumerate the compiled domain directly and represent
   dynamic classification as ballots instead of exact per-consumer catalogues.
2. **Data plane (T2/T3):** make local neighbours arithmetic/one-level and arbitrary
   point ownership shallow enough that the physics-heavy stages stop paying dependent
   topology latency inside every sample.

Deleting control planes alone addresses only part of the non-pressure cost. The
shared topology image is successful only if it also improves the data access inside
transport, face advection and sharpening.

## 2. Hypothesis

The current pipeline often spends more on proving which work should run than on
visiting the compiled sparse domain. The proof is particularly expensive when the
selected population is broad:

- cell/row generation stamps;
- claim and append atomics;
- root, frontier, blast and touched-item lists;
- begin/seal/finalize state machines;
- storage-to-indirect buffer copies and pass breaks;
- repeated global incidence and row-term walks;
- consumer-specific copies of the same immutable topology.

A masked full transform exchanges that control traffic for coherent packet scans. Its
cost is approximately:

```text
packetCount * cheapMaskRead + selectedLaneCount * physics
```

rather than:

```text
classify + claim + append + deduplicate + seal + copy + indirectDispatch + physics
```

This should win when selection is broad, when empty packets can exit before expensive
work, or when the stage-specific catalogue itself dominates. It should also simplify
correctness: one topology-slot generation validates all structural addresses, and a
fully overwritten mask cannot contain stale lanes from an earlier frame.

## 3. Target frame data model

### 3.1 Immutable per topology slot

Candidate transfer compiles these once and they remain unchanged for an accepted
frame:

| Data | Purpose |
|---|---|
| Cell packet directory | packet/lane to stable cell, brick and rung |
| Row packet directory | packet/lane to stable row and row geometry |
| Point-owner directory | one-level fine point to accepted cell for trace stencils |
| Cell-port/local operator | implicit uniform interiors plus compact wall/seam records |
| IBO boundary references | exact cross-leaf and 2:1 seam connectivity |
| Optional packet-mask recipes | cheap brick/tile-local ballot shifts; no exact item catalogue |
| Cell-neighbour value operator | canonical ordered neighbour IDs and extrapolation weights |
| Brick directory | stable brick neighbours and packet spans |
| Pressure hierarchy structure | static restriction/prolongation topology |
| Dispatch arguments | fixed packet/brick/hierarchy dispatch sizes |

The image may be double-buffered, but the selected slot does not change during a
frame. Generation is checked at the slot boundary, not on each cell or row visit.

The two hot access services have explicit contracts:

- **T2 neighbour access:** uniform interiors use dyadic/index arithmetic with no
  dependent topology loads. Only wall, clipped and 2:1 seam packets read compact
  compiled records. Numerical consumers preserve the established row/term order.
- **T3 point ownership:** one widened brick descriptor lookup, with a small
  descriptor neighbourhood staged per workgroup only when it is reused enough to
  pay for the storage. A sampler deeper than approximately one dependent global
  lookup has already lost to arithmetic in the measured trace kernels. The failed
  20 KB halo is not a design to revisit.

### 3.2 Mutable per frame

| Data | Lifetime |
|---|---|
| Physical field banks | persistent across frames |
| Dynamic cell/row/brick masks | produced and consumed within the frame |
| Scatter receipts and reductions | one transformation or reduction |
| Dynamic pressure membership and coefficients | pressure assembly through projection |
| Topology intent and delta | final classification through candidate commit |

The intended eventual mask vocabulary is small and physical:

```text
cells: WET, INTERFACE, EXTENDED, PRESSURE, CHANGED, BULK
rows:  TRANSPORT, GAMMA, PRESSURE, PROJECTION
bricks: INTERFACE, ACTIVITY, TOPOLOGY_CHANGE
```

These are data products. They are not separately generated catalogues with their own
item generations.

### 3.3 Compaction escape hatch

Compaction is allowed only when a measured stage has an extremely sparse mask and a
large per-selected-lane cost. If needed, it should be one generic transient service:

```text
packet mask -> compact packet list
```

It must not grow into another consumer-specific authority. Candidate topology
transfer is expected to remain compact because its work is expensive and its changed
leaf population can genuinely be small.

## 4. Migration principles

1. **Preserve the maps, delete their control planes.** Numerical sweeps, conservative
   receipts and physical predicates remain unless a separately authorized numerical
   change replaces them.
2. **One structural compiler.** Candidate transfer is the only path that creates
   cell, row, boundary or neighbour topology.
3. **One owner per mask plane.** Direct output masks are completely overwritten.
   Many-to-one propagation clears a packet plane once, then ORs compiled shifted
   masks into it.
4. **No hidden cost transfer.** Mask production, mask propagation and numerical work
   receive separate timing seams during migration.
5. **No permanent dual production architecture.** Existing QA/oracle paths may be
   used for comparison, but the accepted runtime path is replaced and its private
   catalogue is deleted.
6. **Topology changes only at the frame boundary.** Frame `n` physics never reads a
   provisional `T[n+1]` address.
7. **Full scans are acceptable evidence.** We do not reject a simpler stage because
   it invokes more lanes; we measure its complete GPU cost.
8. **Selection may be a superset; numerical access may not.** A broader work ballot
   is value-safe when the lane repeats the same map. Neighbour IDs, weights, pressure
   membership and conservative transfers remain exact.
9. **Occupancy is part of the access contract.** Report workgroup memory, register
   pressure and dependent lookup depth with timing; do not call a lookup "compiled"
   while retaining the same hot-path latency.

## 5. Roadmap

| Phase | Vertical concern | Intended deletion | Capability proven for later work |
|---|---|---|---|
| **1** | Velocity extension as eight tile-major full transforms | VEX cell root/frontier/blast catalogues, per-cell generations, tail planning and pressure-CSR recurrence | Broad ballots, implicit interiors, exact seam value gather, direct static dispatch |
| 2 | Unified one-level sampler, proven first against face preparation | Legacy deep point-owner chains; dense face raster only if the new sampler beats it | T3 shared by transport, faces, sharpening, tracers, D4 and presentation |
| 3 | Projection as a masked row transform plus full cell collocation | FPA private leaves/frontier/repair transaction | Row masks over ITR, shared cell-port gather, removal of dynamic row catalogues |
| 4 | Pressure assembly as cell/row maps | PCM/PTR structural rediscovery; retain dynamic membership and solver-local numeric cache where justified | Full dynamic classification over shared structural identities |
| 5 | Scalar conditioning on shared cell-port maps | Remaining sharpening/solid-excess incidence walks and cell-finalizer catalogues | Reusable local stencil maps and masked trace/scatter transforms |
| 6 | Activity, presentation and adaptivity as terminal transforms | Activity dirty catalogues, capacity-sized presentation planning and duplicate candidate images | Packet-to-brick reductions, accepted-leaf publication through the unified sampler and one structural compiler |
| 7 | Consolidation | Obsolete begin/seal/finalize protocols, indirect snapshots and duplicate topology storage | One frame execution image and one structural mutation boundary |

Later phases are contingent on Phase 1. A Phase 1 failure should refine or reject the
masked-full-transform cost model before pressure or projection is rewritten.

## 6. Why Phase 1 is velocity extension

Phase 1 migrates the `transport-velocity-extension` stage's VEX execution. It also
deletes the cross-stage VEX planning and root-production machinery that exists solely
to feed that execution.

It is the best vertical test because:

- one short provisional ocean-seiche capture measured **5.7016 ms** for VEX
  initialization, eight recurrence sweeps and commit, inside a **6.2259 ms** stage;
- the same provisional capture measured **1.4418 ms** for final-scalar one-ring VEX
  root compilation;
- presentation publication carries roughly thirty VEX planning launches and nine
  indirect copies before next-frame execution;
- provisional captures indicate that the ocean VEX root/blast population is broad,
  but the accepted-versus-allocated domain mismatch must be reconciled by the Phase 1
  rebaseline before this becomes a quantitative receipt;
- VEX is pure on a fixed topology: seed velocity plus eight deterministic neighbour
  sweeps produce extended velocity;
- current VXI1 contains potentially reusable packet-mask recipes, but its exact
  source/frontier lifecycle is not a Phase 1 requirement; the ordered value-gather
  half is the missing compiled-topology service Phase 1 must add;
- the result is consumed immediately by both face and scalar transport, making it a
  real physics slice rather than an isolated bookkeeping experiment.

Other possible first slices are weaker:

| Candidate | Why not Phase 1 |
|---|---|
| Projection diagnostics | Easy full transform, but too small and proves no dynamic mask propagation |
| Activity measurement | Tests full brick scans, but not cell/row operator reuse or repeated sweeps |
| Gamma diffusion | Already close to the target through DCA/ITR and has a smaller prize |
| Velocity projection | Strong Phase 3 candidate, but the shared T2/T3 access contracts should be proven first |
| Pressure topology | Largest simplification opportunity, but too broad to distinguish a failed execution model from a pressure-specific integration error |

## 7. Phase 1 target transformation

### 7.1 Input and output

```text
input:
  accepted topology slot
  projected/collocated seed velocity
  current density-derived liquid seed
  moving-solid velocity seed, when present

output:
  extended transport velocity for every lane reached within eight sweeps
  validity/depth facts needed by diagnostics
```

The topology image supplies stable packet cells, cheap brick/tile ballot connectivity
and an exact same-level/2:1 neighbour value operator. The latter preserves the
established directed-edge accumulation order and extrapolation weights. Current VXI
recipes alone are not sufficient: they transform packet masks but do not provide the
weighted velocity gather. They are reused only if they reduce to the simple direct
ballot operation without retaining VXP source/frontier catalogues. No pressure edge
CSR is part of the VEX runtime contract.

### 7.2 Numerical chronology

The replacement retains the current eight-sweep limit and stable neighbour
accumulation order:

```text
initialize:
  seed lane     -> source velocity, valid
  non-seed lane -> zero, invalid

for depth 1..8:
  dispatch every accepted cell packet directly
  read source validity ballots from this packet and its compiled neighbour packets
  if no lane in the packet can receive a value: exit uniformly
  valid lane: preserve the source value
  invalid lane with valid neighbours:
      accumulate compiled neighbour values/weights in canonical order
      normalize with the compiled extrapolation weights
      publish destination value and validity bit
  otherwise:
      remain invalid
  swap source/destination banks and mask planes

commit:
  publish the final extended-velocity plane consumed by face preparation and
  conservative transport
```

The first implementation should prefer this Jacobi-shaped full transform over a
frontier optimization. It is the cleanest comparison against the established full
eight-sweep physics and avoids carrying `visited`, root, frontier and blast
catalogues. Work selection is allowed to be a conservative tile/brick superset. An
empty packet still exits before staging value records.

The simplest acceptable support test is a 64-bit tile ballot combined through the
brick's local neighbour directory. It does not need to identify the exact receiving
cell before the numerical lane runs. If all accepted packets are cheaper than
maintaining even that support ballot, the static accepted mask is also a valid Phase
1 arm: the experiment measures the complete cost rather than assuming exact
selection is valuable.

If preserving already-valid lanes by copying dominates, a later optimization may
split persistent seeds and newly reached lanes inside the same packet transform. That
is not part of the Phase 1 hypothesis.

### 7.3 Mask planes

Phase 1 requires only two ping-pong validity masks, each two words per cell packet,
plus at most one conservative packet/brick support ballot if it demonstrably avoids
enough value work. The initialization pass owns and overwrites mask A. Every
recurrence pass owns and overwrites its destination mask. No clear, generation stamp
or touched-packet list is needed for direct one-packet-to-one-packet output.

The recurrence uses the compiled neighbour/weight view. It is a destination-owned
value gather, not a fanout append, so each destination packet remains single-owner
and writes its output validity ballot directly. Existing VXI/IBO mask recipes are an
implementation candidate only if they beat direct tile dilation without requiring
exact source lists, frontier lists or per-packet generations.

For uniform brick interiors the value operator is implicit arithmetic. Only wall,
clipped and 2:1 seam packets receive compiled neighbour records. Both forms must
produce the current directed-edge order exactly; Phase 1 must not trade the pressure
CSR for another full-domain duplicate CSR with a different order.

Moving bodies contribute seed lanes during initialization. Topology changes do not
need roots: the accepted slot already describes the complete current domain, and the
full transformation recomputes the extension from current seeds every frame.

### 7.4 Static scheduling

Store the VEX packet dispatch triplet in each compiled topology slot. Phase 1 uses:

```text
initialize                    direct cell-packet dispatch
advance depth 1..8            eight direct cell-packet dispatches
commit                        direct cell-packet dispatch
```

There are no storage-to-indirect copies between depths and no presentation-stage VEX
plan for the next frame.

### 7.5 Required deletion

Phase 1 is incomplete until the runtime dependencies below are removed:

- per-cell VEX root stamps, root causes and root list;
- frontier A/B item lists, counts, generations and indirect arguments;
- blast stamps, depths and blast list;
- root sealing, frontier prepare/seal and blast finalization pipelines;
- presentation-stage `encodeVelocityExtensionPlan` and its nine indirect copies;
- `compileSparseCM12VexRootMasks` and final-scalar incidence-based VEX root compile;
- VEX root publication from projection, topology transfer, injection and moving
  solids where its only purpose was incremental recomputation;
- the pressure-edge CSR dependency from VEX recurrence;
- the accepted VEX-cache generation/owner machinery that exists solely to reuse clean
  cells across frames.

Any retained VEX diagnostic must observe the new mask/velocity output rather than
keeping an old catalogue alive.

### 7.6 Expected code areas

The implementation is expected to touch, delete from or retire code in:

- `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts`
- `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts`
- `lib/methods/adaptive-mass/sparse-cm12-velocity-extension.ts`
- `lib/methods/adaptive-mass/sparse-cm12-velocity-extension.wgsl.ts`
- `lib/methods/adaptive-mass/sparse-cm12-vex-packet-frontier.ts`
- `lib/methods/adaptive-mass/sparse-cm12-vex-packet-frontier.wgsl.ts`
- `lib/methods/adaptive-mass/sparse-cm12-vex-ibo1-image.ts`
- `lib/methods/adaptive-mass/sparse-cm12-vex-ibo1-image.wgsl.ts`
- `lib/methods/adaptive-mass/sparse-cm12-transport-producer-masks.wgsl.ts`
- candidate topology/VEX effect publication and obsolete source-contract assertions.

This is a deletion-led integration. Do not leave the prior production VEX selectable
behind a runtime flag. VXI files survive only if their recipe format is useful without
the VXP exact-frontier lifecycle; otherwise Phase 1 retires them with that tower.

### 7.7 Implementation sequence

1. **Freeze the baseline.** Re-capture at least 24 samples for VEX execution, the
   complete stage, final-scalar root compilation and presentation VEX planning. Record
   the exact revision/configuration plus accepted cells, allocated/template cells,
   leaves, packets, roots, blast population, maximum depth and physical receipts. Do
   not use the current 451,040-versus-537,220 population mismatch as a baseline until
   those domains are reconciled.
2. **Compile the missing value operator.** Extend the accepted topology image with
   the implicit-interior/compiled-seam neighbour and weight view. Validate its order
   and values against the current pressure-edge traversal before changing execution.
3. **Install static packet scheduling and mask planes.** Add the topology-slot direct
   dispatch and two validity-mask planes. Start with a direct tile/brick support
   ballot, not an exact frontier compiler. Attribute initialization separately.
4. **Replace the numerical execution in place.** Run initialization, eight direct
   recurrence sweeps and commit from the accepted slot. No accepted runtime may
   choose between old and new execution.
5. **Delete the temporal producer chain.** Remove root publication from the other
   stages, the final-scalar and transport root compilers, presentation-tail planning,
   root/frontier/blast storage and indirect copies.
6. **Audit dependencies.** Prove by source search that VEX no longer imports the
   pressure edge CSR or any retired generation/list symbol.
7. **Run the behavior and performance gates.** Report mask density and mask cost next
   to stage timing, then record the Phase 1 conclusion in this document.

## 8. Timing attribution and success gates

Phase 1 must make cost movement visible. Add temporary or permanent timing seams for:

```text
shared mask initialization
VEX recurrence sweeps 1..8
VEX commit
remaining transport-velocity-extension work
```

The old root compiler and tail planner must remain attributable in the baseline so
their deletion is not mistaken for a recurrence improvement.

### 8.1 Mandatory physics gates

Use the established behavior lanes; do not add unit-test scope for this migration.

| Lane | Gate |
|---|---|
| Symmetric expansion B8 repeat 2 | Exact authoritative digest |
| Symmetric expansion B16 repeat 2 | Exact authoritative digest |
| mini64 multi-frame | Same final topology generation, FSM populations, accepted cell/row populations and clean diagnostics; exact state receipts where already available |
| ocean-seiche multi-frame | Same physical/FSM/topology behavior and no validation faults; exact receipts where already available |
| Moving-solid/injection smoke | Current-frame seeds reach the same eight-sweep support without a prior-frame root queue |
| Structural validation | TypeScript, Dawn entry-point validation and source deletion audit pass |

A bit change is not automatically acceptable because the new operator is simpler. A
non-exact result pauses the phase until it is explained as either an ordering defect
or an explicitly authorized numerical change.

### 8.2 Mandatory performance gates

Measure at least 24 hardware-timestamped samples per arm, preferably in three or more
interleaved captures on an idle machine. Report median and p95 for mini64 and ocean.

| Metric | Required result |
|---|---|
| VEX recurrence + commit, ocean | At least **20% faster** than the controlled rebaseline; **35%** is the goal (5.7016 ms is provisional only) |
| Complete `transport-velocity-extension` stage, ocean | At least **20% faster** than the controlled rebaseline after including VEX-local mask initialization; no cost may be hidden in an adjacent stage (6.2259 ms is provisional only) |
| mini64 VEX stage | No material regression beyond timestamp quantization; direct scheduling should help the launch-sensitive lane |
| VEX tail planning/root compilation | Removed, not merely moved |
| Runtime VEX dispatch shape | At most initialization + eight recurrence sweeps + commit, all direct; zero per-depth indirect copies |
| Memory | Net deletion of cell-sized root/frontier/blast bookkeeping; packet masks reported separately |
| Overall frame | Reported, but may regress in Phase 1 because shared-mask infrastructure has only one consumer |

The 20% threshold is a falsification threshold, not the eventual target. A smaller
improvement does not justify replacing a mature path with a new shared abstraction.

### 8.3 Scalability gates

Phase 1 also has to show that the design is reusable:

1. The mask ABI is expressed in TEI packet/lane space, not VEX-specific cell IDs.
2. Cross-leaf mask connectivity is brick/tile local; any reused IBO/VXI recipe proves
   that it is cheaper than direct ballot dilation and has no exact-frontier lifecycle.
3. Uniform neighbour values are arithmetic; only seam/wall packets read compiled
   records, in canonical numerical order.
4. The topology compiler emits the operator once; the frame does not rebuild it.
5. Empty-packet, active-lane and neighbour-load counts are published so later stages
   can build a cost model from actual mask density.
6. Workgroup memory, register pressure and dependent load depth are reported.
7. No VEX-specific generation protocol is required to make the masks safe.
8. The same dispatch helper can host a later masked cell or row transform without
   allocating another catalogue.

## 9. Reading the result

| Phase 1 result | Interpretation | Next action |
|---|---|---|
| Stage improves, physics exact, mask cost modest | Hypothesis supported | Proceed to the unified-sampler Phase 2 |
| Stage improves, overall frame regresses because mask infrastructure is isolated | Acceptable Phase 1 outcome | Retain the simpler stage; assess mask amortization when projection becomes the next mask consumer |
| Stage improves only because planning cost moved elsewhere | Failed attribution | Refile all moved cost and repeat; do not proceed |
| Stage does not improve, but compiled seam/value loads dominate | T2 operator problem, not necessarily mask-model failure | Improve the implicit-interior/seam split once, then repeat Phase 1 |
| Stage does not improve because empty/full packet scanning dominates | Masked full-transform hypothesis weakened | Measure packet compaction as the one generic escape hatch |
| Physics differs | Correctness failure | Stop and resolve ordering/seed/seam semantics |
| Implementation retains both authority systems | Architectural failure | Finish deletion before measuring scalability |

## 10. Expected leverage after Phase 1

If Phase 1 succeeds, the same primitives apply directly to the remaining pipeline:

| Reusable Phase 1 primitive | Next consumers |
|---|---|
| Direct topology-slot packet dispatch | cell finalizers, body forces, diagnostics, activity, presentation |
| Completely overwritten mask planes | pressure membership, scalar classification, projection selection |
| Brick/tile ballot transformation | changed one-rings, sharpening neighbourhoods, activity closure |
| Implicit-interior/compiled-seam gather | sharpening statistics, pressure RHS, collocation, activity census |
| ITR row resolution | gamma, pressure rows, projection rows, face preparation |
| Explicit mask timing | decide full scan versus generic compaction per transform |

The immediate follower is the unified T3 sampler. It must first challenge the current
face-velocity raster, because that raster is the measured winner against deeper sparse
owner lookups. The new sampler stages only a small brick-descriptor neighbourhood and
must report occupancy as well as time. The dense raster is deleted only if the
complete face-preparation stage improves. Transport and sharpening then reuse the
same sampler rather than retaining TEI/legacy variants.

Velocity projection follows the sampler: pressure assembly authors the pressure-row
mask, projection consumes it directly, and FPA can disappear. That phase validates
the row-domain half of the model after Phase 1 validates the cell-domain half.

Presentation is not primarily a dirty-mask target. Its eventual transform dispatches
at accepted leaf count rather than capacity, lets already-cheap dry pages run, and
uses the unified sampler to attack the wet pages' owner/hash lookup cost. No plan
should claim a win from skipping dry pages whose publication is already effectively
free.

Gather-form gamma or sharpening is not part of this roadmap's preservation phases.
It changes pair evaluation and rounding order and therefore requires a separate,
explicitly authorized numerical migration after the shared access model is proven.

## 11. Completion definition

Phase 1 is complete when:

- velocity extension is a direct, topology-slot-sized sequence of eight tile-major
  full transforms with ballot validity;
- the output is physically exact on the established regression lanes;
- its measured stage time clears the mandatory gate;
- old root/frontier/blast planning and storage are deleted from the runtime;
- VEX no longer reads the pressure edge CSR;
- uniform VEX neighbours are arithmetic and only seam/wall packets use compiled
  value records;
- no exact VXP/VXI frontier lifecycle survives merely to optimize selection;
- shared mask overhead and densities are separately reported;
- the code and receipts demonstrate a reusable packet/operator contract for the next
  stage; and
- the plan is updated with measured conclusions before Phase 2 begins.
