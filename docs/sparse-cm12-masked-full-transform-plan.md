# Sparse CM12 masked full-transform migration plan

Status: Phase 1 runtime migration implemented and accepted, 2026-08-23. B8/P8 is
the target profile; B16/P16 results are secondary evidence. Performance thresholds
are diagnostic signals for packing/access work, not release vetoes.

Scope: every Sparse CM12 frame operation except the iterative pressure solve. The
pressure-system build is in scope; MGPCG iteration is not. Phase 1 is one vertical
slice through velocity extension. This document now records the implementation and
measurement receipt as well as the remaining roadmap. It is not an authorization to
preserve a second production path.

The iterative pressure solve is now an explicit follow-on rather than part of the
closed Phase 1 gate. Its experiments are registered in §14 so failed layouts remain
useful evidence instead of surviving as runtime alternatives.

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

It was the best vertical test because:

- the controlled 24-frame ocean-seiche B16/P16 baseline measured **7.0779 ms**
  for VEX initialization, eight recurrence sweeps and commit, inside a
  **7.4711 ms** stage;
- the same baseline measured **2.5559 ms** for final-scalar one-ring VEX root
  compilation;
- presentation publication carried 29 VEX planning launches and ten indirect
  copies before next-frame execution;
- the controlled baseline and replacement use the same 488,580 accepted cells,
  1,430,079 accepted rows and terminal topology generation;
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
  moving-solid contribution already collocated into wet seed velocity, when present

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
accumulation order. The implementation avoids copying already-valid values between
two full cell banks: the effective-velocity plane is authoritative and accepted
depth makes the in-place publication Jacobi-safe.

```text
initialize:
  seed lane     -> effective velocity, depth 0, valid in mask A
  non-seed lane -> zero effective velocity, invalid depth/mask

for depth 1..8:
  dispatch the topology-slot packet grid directly
  stage the source validity ballot once per packet
  if no lane in the packet can receive a value: exit uniformly
  valid lane: retain its already-published effective velocity without copying
  invalid lane with valid neighbours:
      accumulate compiled neighbour values/weights in canonical order
      normalize with the compiled extrapolation weights
      publish effective velocity, then accepted depth and destination validity bit
  otherwise:
      remain invalid
  swap mask planes

commit:
  fused into successful lane publication; there is no separate commit dispatch
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

Moving bodies affect the collocated wet seed velocity; matching the established
physics does not create independent dry solid seed lanes. Topology changes do not
need roots: the accepted slot already describes the complete current domain, and the
full transformation recomputes the extension from current seeds every frame.

### 7.4 Static scheduling

The host derives a bounded direct grid from the topology profile. Mask storage keeps
TEI's stable 64-packet stride per leaf, while direct work enumerates only the packet
slots the profile can address: one for B4, eight for B8 and 64 for B16. The shader
maps compact dispatch ordinal to `leaf * 64 + localPacket`. WebGPU guarantees at
least 65,535 workgroups per dimension, so large direct domains are encoded as
`min(dispatchPacketCount, 65535) × ceil(dispatchPacketCount / 65535)`. Phase 1 uses:

```text
initialize                    direct cell-packet dispatch
advance depth 1..8            eight direct cell-packet dispatches
commit                        fused into successful sweep publication
```

There are no storage-to-indirect copies between depths and no presentation-stage VEX
plan for the next frame.

### 7.5 Required deletion

Phase 1 is incomplete until the runtime dependencies below are removed:

- per-cell VEX root stamps, root causes and root list;
- frontier A/B item lists, counts, generations and indirect arguments;
- blast stamps, depths and blast list;
- root sealing, frontier prepare/seal and blast finalization pipelines;
- presentation-stage `encodeVelocityExtensionPlan` and its ten indirect copies;
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

The implementation makes cost movement visible through these timing seams:

```text
velocity-extension-mask-initialization
velocity-extension-sweeps (eight sweeps plus fused value publication)
frame-control-authority
transport-packet-authority
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

### 8.2 Performance diagnostic targets

Measure at least 24 hardware-timestamped samples per arm, preferably in three or more
interleaved captures on an idle machine. Report median and p95 for mini64 and ocean.

| Metric | Diagnostic target |
|---|---|
| VEX recurrence + commit, ocean | **20% faster** than the controlled rebaseline; **35%** is the optimization goal |
| Complete `transport-velocity-extension` stage, ocean | **20% faster** after including VEX-local mask initialization; no cost may be hidden in an adjacent stage |
| mini64 VEX stage | No material regression beyond timestamp quantization; direct scheduling should help the launch-sensitive lane |
| VEX tail planning/root compilation | Removed, not merely moved |
| Runtime VEX dispatch shape | At most initialization + eight recurrence sweeps + commit, all direct; zero per-depth indirect copies |
| Memory | Net deletion of cell-sized root/frontier/blast bookkeeping; packet masks reported separately |
| Overall frame | Reported, but may regress in Phase 1 because shared-mask infrastructure has only one consumer |

The 20% threshold is a diagnostic/falsification target, not a release veto. Falling
short requires identifying whether launch packing, empty work, reduction structure
or data access explains the gap. Acceptance then weighs that evidence together with
physics, deletion, complete-stage and overall-frame results; it must not add a new
control plane merely to cross a percentage boundary.

### 8.3 Scalability gates

Phase 1 also has to show that the design is reusable:

1. The mask ABI is expressed in TEI packet/lane space, not VEX-specific cell IDs.
2. Cross-leaf mask connectivity is brick/tile local; any reused IBO/VXI recipe proves
   that it is cheaper than direct ballot dilation and has no exact-frontier lifecycle.
3. Uniform neighbour values are arithmetic; only seam/wall packets read compiled
   records, in canonical numerical order.
4. The topology compiler emits the operator once; the frame does not rebuild it.
5. Empty-packet and active-lane density are derived from the shared mask; neighbour
   loads are measured by bounded diagnostics/profiling rather than production atomics.
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
- its measured stage time is explained against the diagnostic target and complete-frame cost;
- old root/frontier/blast planning and storage are deleted from the runtime;
- VEX no longer reads the pressure edge CSR;
- uniform VEX neighbours are arithmetic and only seam/wall packets use compiled
  value records;
- no exact VXP/VXI frontier lifecycle survives merely to optimize selection;
- shared mask overhead and densities are separately reported;
- the code and receipts demonstrate a reusable packet/operator contract for the next
  stage; and
- the plan is updated with measured conclusions before Phase 2 begins.

## 12. Phase 1 implementation receipt

### 12.1 Runtime architecture completed

The accepted runtime now has one VEX2 initialization dispatch and eight direct
packet sweeps. Sweep publication is the commit: a newly reached lane writes the
effective-velocity plane and then its accepted depth, so later work in the same
dispatch cannot consume a same-depth value. A lane already selected by the staged
input packet mask performs no value-bank copy. The output mask is completely
overwritten by its owning packet workgroup.

The packet ID is TEI packet/lane space, not a VEX cell catalogue. Strict leaf
interiors use arithmetic neighbours in canonical `x−, x+, y−, y+, z−, z+` order.
Boundary and 2:1 cases retain the accepted incidence/term order; common two-term
rows resolve the opposite term directly and variable rows retain their ordered
loop. The extrapolation weight is the established absolute neighbour-term
coefficient. VEX no longer reads the pressure edge CSR.

The following runtime tower was removed rather than retained behind a flag:

- root, frontier A/B and blast stamps, lists, causes, depths and generations;
- the 29-dispatch/ten-copy presentation-tail VEX planner;
- VEX root publication from transport, final-scalar, projection, topology,
  injection and moving-solid paths;
- VEX delta authority, packet frontier, VXI consumer/image and reverse-dependency
  modules, their dedicated tools and their obsolete tests;
- the persistent accepted-velocity cache and accepted owner/reuse metadata; and
- VDA participation in topology commit, leaving TFX/PTR as the topology-effects
  authority.

The dedicated effective-transport-velocity plane is now authoritative for face
preparation, conservative transport and tracers. The tracer pass binds that plane
explicitly. Transport characteristic support remains separately owned by transport;
it was not deleted with VEX state.

The final depth-eight ballot is exposed as the physical `EXTENDED` mask through
stable TEI packet/lane accessors. Face-support publication is its first downstream
consumer: it reads extension membership from the mask while still fully overwriting
owner, span, liquid and velocity data for every dense support cell. Transport dirty
masks and pressure membership remain distinct physical products. No buffer, binding,
copy or generation protocol was added for sharing `EXTENDED`.

The last VAB naming/composition wrapper was deleted. Resident composition now owns
the VEX layout, initialization and the one-plus-eight pipeline descriptors directly.
Final mask density is derived only during QA readback; production execution has no
counter atomics or reduction pass. Dead always-zero executed-cell and neighbour-load
receipt fields were removed rather than retained as misleading telemetry.

The standalone moving-solid/injection lane exposed one stale construction guard:
the resident had retained complete rigid-coupling resources and encode ordering but
unconditionally rejected rigid scenes and instantiated no coupling object. The guard
was removed and the existing coupling is now constructed only when rigid resources
are present; the no-rigid path remains allocation-free.

### 12.2 Behavior and structural receipts

No unit tests were added or run for this migration. The evidence is standalone
Dawn behavior, exact long-run field receipts and structural compilation:

| Receipt | Result |
|---|---|
| Symmetric B8, 500 steps, old versus new | Exact combined SHA-256 `9289e7fee59b90f7c0efb618d5735a2c7ca0a6c642dd634c00fd655c7275933c` |
| Symmetric B16, 500 steps, old versus new | Exact combined SHA-256 `b11645b995c3c860663f7554726b4130d43243dbc8cb9d6f61f96535ce50cdc9` |
| Mini64 temporal short smoke | Five steps passed with clean diagnostics after the topology-effects seal cutover |
| Ocean B8 production-sized trajectory | 24 steps passed; no validation/gate errors, 64 pressure iterations per step, FCA generation 2→25, final positive-mass drift −0.0091% |
| Moving solid + liquid injection | Standalone three-frame Dawn lane passed; body-only, first post-injection and subsequent recomputations had zero validation/VEX faults and exact mask/depth/value agreement |
| TypeScript | `npm run check:types` passed |
| Production build | `npm run build` passed |
| WGSL/Dawn | 681 compute entry points across three B/P variants and four stage-lens programs passed; focused B8/P8 VEX entry points also passed |
| Managed pipeline audit | Passed |
| Stage timing contract | Passed |
| Source deletion audit | No old VEX planner/root/frontier/blast or pressure-CSR recurrence dependency remains in runtime code |
| Shared `EXTENDED` consumer | Face-support membership reads the final TEI packet/lane ballot; no duplicate mask plane or scheduling authority |

The ocean trajectory also exposed and closed a production-only scheduling defect.
At the UI-default B8 topology, TEI capacity is 163,840 packets. Dispatching that
count in X exceeded WebGPU's 65,535 workgroup-per-dimension limit, rejecting the
whole frame before FCA commit, PCM classification, pressure or presentation could
run. VEX direct scheduling now uses a bounded 2D grid and shader-side linear packet
IDs. The 24-step Dawn trajectory above passed after the fix, and the UI scene was
confirmed advancing in production.

### 12.3 Controlled performance receipt (primary B8/P8)

The primary matched arms use ocean-seiche B8/P8, eight warmups and 24
hardware-timestamped frames on the same Apple M1 Max Metal/Dawn environment.
Configuration and adaptive representation are identical: 451,040 accepted cells,
1,319,052 accepted rows, 245,170 pressure cells, 731,756 pressure rows, 1,780
resident bricks and topology generation 1.

| Metric | VEX1 baseline | Final VEX2 | Change |
|---|---:|---:|---:|
| VEX local work, median / p95 | 5.1118 / 6.4881 ms | 3.8666 / 4.7186 ms | **−24.36% / −27.27%** |
| Mask initialization, median / p95 | included above | 0.1966 / 0.2621 ms | separately attributed |
| Eight sweeps + fused publication, median / p95 | included above | 3.6700 / 4.5220 ms | separately attributed |
| Complete `transport-velocity-extension`, median / p95 | 5.6361 / 6.8157 ms | 4.4564 / 5.4395 ms | **−20.93% / −20.19%** |
| Face preparation, median / p95 | 2.6214 / 4.0632 ms | 3.0802 / 3.8666 ms | +17.50% / −4.84% |
| Non-pressure total, median / p95 | 43.8436 / 46.8582 ms | 38.1420 / 40.3702 ms | −13.00% / −13.85% |
| GPU advance, median | 70.9100 ms | 67.1089 ms | −5.36% |
| Wall frame, median / p95 | 78.4592 / 80.8381 ms | 74.1486 / 77.1739 ms | −5.49% / −4.53% |

The threshold did its intended diagnostic job. The first B8 VEX2 arm dispatched all
64 reserved mask slots per leaf and measured 10.0270 / 13.5660 ms locally and
10.8134 / 14.0247 ms for the stage. B8 can address only eight packets per leaf.
Mapping a compact direct ordinal to the stable `leaf * 64 + local` mask address,
then returning uniformly for structurally empty packets, removed that packing waste.

The shared `EXTENDED` face consumer remains a median cost signal. A 512-byte
workgroup mask cache was measured and made face preparation worse
(3.6045 / 4.6531 ms), so it was deleted. The final code uses the simpler one-word
mask read; it improves p95 versus VEX1 while leaving a 0.4588 ms median opportunity
for the later shared-sampler/data-packing phase.

The exact 32-frame B8 QA census reports 439,600 valid cells over 20,480 directly
dispatched packets: 7,360 packets are nonempty and 13,120 are empty. Stable mask
storage remains 163,840 packet slots, but only the profile-valid 12.5% is dispatched.
Selected-lane density is 33.54% and nonempty-packet density is 35.94%; VEX and
validation faults are zero.

### 12.4 Controlled performance receipt (secondary B16/P16)

The matched arms use ocean-seiche B16/P16, eight warmups and 24 hardware-timestamped
frames on the same Apple M1 Max Metal/Dawn environment. Both arms finish with
488,580 accepted cells, 1,430,079 accepted rows, 505 resident bricks, topology
generation 1 and zero validation errors. Independent runs ended one dynamic pressure
cell and four active pressure rows apart (272,015/811,382 versus 272,014/811,378),
so the receipt claims matched topology and represented work rather than byte-identical
dynamic scheduling.

| Metric | VEX1 baseline | VEX2 | Change |
|---|---:|---:|---:|
| VEX local work, median / p95 | 6.6847 / 9.1750 ms | 4.3909 / 6.5536 ms | **−34.31% / −28.57%** |
| Mask initialization, median / p95 | included above | 0.2621 / 0.3277 ms | separately attributed |
| Eight sweeps + fused publication, median / p95 | included above | 4.1288 / 6.2259 ms | separately attributed |
| Complete `transport-velocity-extension`, median / p95 | 7.1434 / 9.5683 ms | 4.7186 / 6.8157 ms | **−33.94% / −28.77%** |
| Presentation publication, median / p95 | 2.9491 / 3.6045 ms | 0.5243 / 0.9830 ms | −82.22% / −72.73% |
| Final-scalar VEX root compilation, median / p95 | 2.5559 / 2.7525 ms | removed | −100% |
| Non-pressure total, median / p95 | 48.5622 / 51.9045 ms | 40.8289 / 43.9747 ms | −15.92% / −15.28% |
| GPU advance, median | 75.8252 ms | 70.8444 ms | −6.57% |
| Wall frame, median / p95 | 82.7419 / 86.5287 ms | 77.1899 / 81.9671 ms | −6.71% / −5.27% |
| Allocated bytes | 633,026,356 | 601,407,176 | −31,619,180 bytes (−5.00%) |

The VEX-local and complete-stage gates clear the required 20% at both median and
p95; local median work is within one percentage point of the 35% goal. These are
raw 24-sample stage/chunk distributions, not p95 values inferred from medians. The
shared-mask face consumer is included: complete face preparation measured
5.4395 / 6.4881 ms median/p95 versus 5.7016 / 7.2745 ms in the baseline.

The final ocean QA mask census reports 466,886 valid cells over 38,400 launched
packets: 7,500 packets are nonempty and 30,900 are empty. That is 19.00% selected
lane slots and 19.53% nonempty packets. The receipt has topology generation 1,
source-frame generation 32 and zero VEX faults. Density is computed from
the final packet plane during QA readback and contributes no production GPU work.

### 12.5 Phase conclusion

The primary B8/P8 and secondary B16/P16 results support the hypothesis: exact physics
was retained, the target stage and complete frame improved, and more than six
thousand net lines of obsolete VEX authority, tools and tests were removed. B8's
initial failure identified a real packing defect rather than forcing restoration of
the deleted control plane. Compact profile dispatch fixed it with less work and no
new authority. Phase 1 is accepted; the remaining items below are follow-on evidence
and later-phase work.

## 13. Next steps

1. **Close profiler evidence.** Capture compiler/profiler evidence for registers,
   occupancy, workgroup memory and dependent-load depth. The matched ocean and
   mini64 raw stage/chunk distributions and timestamp quantization are complete.
2. **Finish bounded observability without rebuilding authority.** Add neighbour-load
   evidence only through a bounded profiler/diagnostic mechanism; do not restore
   runtime counters. Presentation stage 5 remains the enforced FPL contract; the
   stale stage-0 VEX-consumer comment has been retired with the planner.
3. **Continue the compiled-topology rollout.** The tracer sampler and projection-row
   cuts below are complete. Apply the same deletion-led test to DCA gamma scatter,
   sharpening and other consumers whose access order can be represented exactly by
   the existing packet/row operators. Retain the dense face support plane until a
   replacement demonstrates equivalent locality. Each accepted cut must delete its
   private catalogue and measure the complete producer-plus-consumer stages.

## 14. Pressure extension experiment register

The standalone B8/P8 pressure-solve design, exceptional-face cost model, and next
cutover sequence now live in
[`sparse-cm12-b8-pressure-execution-image.md`](sparse-cm12-b8-pressure-execution-image.md).

Pressure work follows the same rules as Phase 1: one experiment at a time, one
production arm, reuse immutable topology and existing packet/tile descriptions, and
remove a rejected implementation after recording its evidence. Symmetric expansion
is the numerical gate. Ocean-seiche and mini-dam 64 are the performance lanes.

| Experiment | Symmetric expansion | Ocean pressure solve, median / p95 | Mini64 pressure solve, median / p95 | Verdict |
|---|---|---:|---:|---|
| Implicit-interior / compiled-seam SpMV | Exact B8 and B16 raw-bit receipts | 26.4765 / 26.7387 → 25.3624 / 25.5590 ms (**−4.21% / −4.41%**) | 13.5660 / 14.5490 → 12.9106 / 13.9592 ms (**−4.83% / −4.05%**) | **Accepted** |
| Global PAB reorder, literal 4³ tiles | Passed before the layout refinement | 27.1974 / 28.0494 → 30.0155 / 30.9330 ms (**+10.36% / +10.28%**) | 13.4349 / 14.6801 → 14.4179 / 15.5976 ms (**+7.32% / +6.25%**) | Rejected |
| Global PAB reorder, subgroup-major tiles (16×2×2 at B16, 8×4×2 at B8) | Residual 4.0445e-6 exceeded the 4.0e-6 gate; symmetry and divergence checks otherwise passed | 27.1974 / 28.0494 → 27.3940 / 28.4426 ms (**+0.72% / +1.40%**) | 13.4349 / 14.6801 → 13.4349 / 14.7456 ms (**0.00% / +0.45%**) | Rejected |
| Compact pressure worklist, neighbour plane at base zero | Exact work and residual receipts across both repeated B8 arms | 27.6234 / 28.6065 → 27.5251 / 28.2133 ms (**−0.36% / −1.37%**) | 12.5501 / 14.4835 → 13.1400 / 15.2371 ms (**+4.70% / +5.20%**) | Diagnostic failure: smaller allocation, worse mini64 locality |
| Split 4³ tile SpMV + canonical reduction | Exact B8 raw-bit field and combined SHA-256 receipt | 27.5251 / 28.2133 → 30.4087 / 31.3262 ms (**+10.48% / +11.03%**) | 13.1400 / 15.2371 → 13.9592 / 15.0733 ms (**+6.24% / −1.08%**) | Rejected: saved gathers did not repay the second cell pass |

The two tile arms isolate the failure. A literal 4³ lane layout turns each 32-lane
half-wave into eight four-cell runs spread across z planes, damaging coalescing.
Making x contiguous across a subgroup recovers almost all of that loss. What remains
is not a useful speedup: globally reordering the PAB changes every pressure-cell
kernel and the reduction order, while no operator data is staged or reused. The
near-threshold residual change is therefore treated as a real numerical regression,
not relaxed away. Both rejected implementations were removed; production retains
canonical PAB order.

The compact-worklist arm removed 42.32 MB (7.45%) from ocean and 30.19 MB (8.97%)
from mini64 by deleting retired pressure payloads and moving the dense neighbour
plane to offset zero. It preserved ocean's 245,170 pressure cells, 731,756 rows and
64 iterations, and mini64's 52,925 cells, 157,223 rows and 64 iterations exactly.
The opposite timing response is evidence that footprint reduction alone did not
improve the hot gather layout: changing the neighbour plane's base address likely
changed cache-set colouring while leaving the solve's dependent gathers unchanged.
Keep the deletion evidence, but do not call this a locality win; the next image
experiment must remove dependent authority reads or create actual operator reuse.

The split-tile arm retained canonical PAB reduction order and was raw-bit exact:
spatial 4³ workgroups staged 64 centre values, wrote `Az`, and a separate canonical
pass formed the unchanged per-64-rank reduction tree. Mini64 exposed the structural
loss directly: 52,925 pressure cells require 827 canonical workgroups but occupy
1,033 nonempty tiles, so the recurring operator phase launched 1,860 workgroups
where the fused path launched 827. Explicit staging reduced a full tile's neighbour
vector loads from at most 384 to 160, but duplicated vector reads, a workgroup
barrier, partial-tile lanes, authority lookup and the extra dispatch outweighed that
reuse. Ocean's larger regression confirms that its hardware cache already captures
enough stencil reuse. The arm was removed. Further tile work must keep canonical
reduction fused with the operator, or delete a larger independent cost such as the
under-filled brick/hierarchy schedule.

The next tile experiment must be operator-local. It keeps canonical PAB and reduction
order, schedules only SpMV through subgroup-coalesced tiles, stages a tile centre and
halo in workgroup memory, and uses the compiled seam for non-interior terms. Before
cutover, capture compiler/profiler evidence for register pressure, achieved occupancy,
workgroup-memory use and dependent global-load depth. This is a gating measurement:
tile staging is valuable only if reuse outweighs its registers, barriers and halo
traffic on both performance lanes.

## 15. Compiled-topology rollout register

Phase 2 applies the Phase-1 rule beyond VEX: reuse compiled topology only where its
access shape matches the consumer, measure the complete B8/P8 stage, and remove a
rejected arm rather than retaining parallel production paths.

| Cut | B8/P8 evidence | Verdict |
|---|---|---|
| Shared TEI tracer sampler | `seedTracers` and `advanceTracers` stage the existing TEI directory once per 64-marker workgroup; the legacy `ownerCellAt` velocity sampler and duplicate RK2 trace were deleted. Tracers are disabled in the ocean performance lane, so this is a structural/code-reduction receipt rather than a timing claim. | **Accepted** |
| Direct TEI face sampler | Replacing the dense support plane with staged TEI owner resolution increased complete face preparation from 3.0802 / 3.8666 ms to 10.0925 / 12.1242 ms median/p95. | **Rejected and removed** |
| Face-stage attribution | The restored dense arm measures 0.9830 / 1.1141 ms for support clear/publication and 3.1457 / 4.0632 ms for dirty oriented-row preparation. | **Retained diagnostic seam** |
| Full pressure-row projection | A direct full-PCM-row prototype was stopped before simulation: clean rows retain already-projected face values, so replaying every row would subtract an unchanged pressure jump twice. | **Rejected before timing** |
| Exact compiled dirty-row transform | Six row families over 12 mask words per compact packet preserve the positive-owner x/y/z rows and the otherwise-unaddressable positive-side, negative-only sparse-air rows. Dirty preparation, pressure-bit changes, first-frame bootstrap and value-changing force/solid/boundary roots OR into the same direct mask. | **Accepted** |
| Direct masked projection | The shared mask drives `projectPressureRow` directly and mirrors both face banks. B8/P8 projection fell from 3.9322 / 4.6531 ms to 2.8180 / 2.8836 ms median/p95. Symmetric exact, moving-solid/injection and ocean Dawn lanes pass. | **Accepted; FPA deleted** |
| Compact gamma row masks | TPA now owns six `u32` words per compact packet. ITR compiles TPM's sealed surface ballot into the three positive-owner row families, and the same immutable mask is replayed for snapshot and refinement. One-sided sparse-air rows are exact gamma no-ops; their `ownerTerm == 0xf` sentinel is rejected before term decode. | **Accepted; DCA deleted** |
| TPA gather-list gamma replay | Replacing direct compact replay with the existing gather-family list added list/atomic indirection. Gamma measured 2.2938 / 4.3909 ms versus 2.2282 / 3.9322 ms for direct replay, and the complete frame also worsened. | **Rejected and removed** |

The face result is a data-layout finding, not a reason to abandon compiled topology.
TEI is effective when a workgroup amortizes its directory across packet consumers;
it is not competitive with eight independent dense corner reads inside every RK2
face sample. The accepted face cut therefore targets compiled row ownership while
retaining the dense velocity plane.

The first three-axis row-mask design was rejected during proof: a positive-side
sparse-air row may contain only one negative coefficient, so it has no positive
ITR owner and collides with the canonical incoming row if forced into that address.
The accepted ABI gives those rows three distinct mask families beside the ordinary
positive-owner x/y/z families. The mask is completely overwritten from dirty brick
ownership, while persistent accepted-pressure bits are sentinel-initialized and
bit-compared to publish first-frame and later pressure changes. `forceFaces` compares
the before/after value and roots only changed rows. This is the exact union that full
pressure membership alone could not provide.

On the matched 24-sample B8/P8 ocean capture, dirty face-row preparation plus mask
compilation rose from 3.1457 / 4.0632 ms to 4.3909 / 5.1773 ms, while projection fell
from 3.9322 / 4.6531 ms to 2.8180 / 2.8836 ms. The combined pair is nearly flat at
median and improves p95; complete GPU advance fell from 67.5021 to 62.3903 ms and
wall-frame median/p95 fell from 75.4679 / 78.9570 to 68.8264 / 71.4229 ms. This is
not attributed solely to projection because the tree contains other concurrent
compiled-pressure work; the directly attributable result is the producer/consumer
pair and its deletion.

FPA's 37,500,160-byte arena was replaced by a 2,986,880-byte DFRM allocation, a net
34,513,280-byte reduction. Its private layout, stamps, frontiers, rank tree, repair
transaction, 11 pipelines, four indirect-copy/pass seams, indirect buffer, QA/lens
plumbing, exact-producer adapter, dedicated checkers and obsolete test were deleted.
The replacement modules are 206 lines; the cut removes more than 1,500 net lines
across the authority island and resident integration. Executable-source audits have
no FPA imports or symbols. No unit tests were added or run.

Validation receipts for the final cut are: unchanged B8/P8 500-step combined SHA-256
`9289e7fee59b90f7c0efb618d5735a2c7ca0a6c642dd634c00fd655c7275933c`;
three-step moving-solid/injection Dawn pass with exact EXTENDED mask/depth/value
agreement; 24-sample ocean pass with zero validation errors and exact timing closure;
TypeScript, managed-pipeline, stage-timing and three-profile Dawn WGSL checks all
pass.

DCA's gamma row-scatter deletion is also complete. Its two source lists, row/cell stamps, six
row and two cell planes, touched lists, header, indirect words, 12-byte indirect
buffer, seven resident pipelines and copy-induced pass seam were deleted. Its VEX
and cell halves were already production no-ops and were removed rather than kept as
compatibility hooks. TPM's dead density ballot, counter and two stable density planes
were deleted with it. Executable source contains no DCA or dynamic-closure symbols.

The replacement adds a 491,520-byte gamma mask at B8/P8 and removes 10,486,272 bytes
of DCA plus dead TPM storage, for a net 9,994,752-byte reduction. The first direct
24-sample arm measured gamma diffusion at 2.2282 / 3.9322 ms versus 2.0316 / 2.4248 ms
immediately before the cut, and producer-mask compilation at 0.5243 / 0.9175 ms
versus 0.2621 / 0.5243 ms. This is accepted as a deletion-led result, not presented
as a stage speedup. The rejected gather-list replay confirms that the remaining cost
is packing/load shape: selecting fewer packets through an atomic list was slower than
the uniform direct packet plane.

Production has no layout-free shader specialization or no-op DFRM hook. Resident
composition requires ITR, TPA/TPM and DFRM, and the product build plus three-profile
Dawn checker compile that required path. The B8 500-step exact hash remains unchanged;
the three-step moving-solid/injection lane and production ocean trajectory pass with
zero validation faults. A same-frame FSM-driven theta or pressure-membership refresh
now calls `dfrm1MarkRow` directly, closing the bit-identical-pressure projection gap.
The next deletion candidate is the remaining private catalogue in sharpening or
another stage whose exact access order is already representable by the shared packet
and row operators.
