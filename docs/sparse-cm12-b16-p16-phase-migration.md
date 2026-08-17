# Sparse CM12 B16/P16 rollback-free migration

## Contract

This sequence converts the current resident into the separated phase arenas
described by `sparse-cm12-phase-arenas.ts` and the immutable topology described
by `sparse-cm12-hot-topology.ts`. It is deliberately a sequence of small,
one-way cuts. No cut introduces a runtime choice between an old and new
algorithm.

Every cut obeys all of these rules:

1. B16/P16 is the only implicit physical ABI. A different brick or page size
   must fail construction.
2. Persistent source bytes are copied by GPU commands. Publication predicates,
   generation checks, mismatch counts, indirect counts, and fault latches are
   GPU-authored. A CPU readback is an offline receipt and never schedule input.
3. A cut is published only after the destination byte comparator reports zero
   mismatches for every copied region. Physical stages additionally compare
   density, gamma, cell velocity, face velocity, pressure, liquid, theta,
   accepted topology, and conservation receipt words.
4. There is no runtime fallback to the old path. Before publication, the new
   generation is invisible. After publication, the old implementation is not
   selectable. A fault zeros the affected indirect packet, latches its first
   bad brick/tile/row, and prevents that generation from sealing.
5. Failure blast radius is the smallest owner that can certify the data: a
   region during byte migration, a row/cell packet during HTP access, a brick
   and its ancestors during pressure repair, a prepared brick plus incident
   rows during topology transfer, or one P16 page during presentation.
6. A frame with an unsealed required generation does not advance or publish
   partially updated physical state. This is fail-stop, not a whole-domain
   recovery solve and not a rollback pass.

“Bit exact” means zero unequal `u32` words, not a tolerance. The physical hash
receipt is a second, compact audit: four independently seeded 32-bit streams
over `(semantic field id, canonical element id, word)`, plus word count,
generation, and first mismatch. Hash equality without a zero mismatch count is
not an acceptance result.

## Stable cut boundaries

Only these command-order boundaries may publish a migration:

- **F0 — frame boundary:** after presentation and accepted-indirect snapshot,
  before the next parameter generation is consumed;
- **P0 — pressure cache boundary:** after membership, theta, effective edges,
  aggregate edges/diagonals, and hierarchy coefficients are complete, before
  `initializePCG`;
- **T0 — topology candidate boundary:** after the pressure/projection consumers
  are finished, before `transferCandidateCells` writes candidate channels;
- **C0 — topology commit boundary:** after every candidate conservation and
  face receipt, immediately before the accepted generation flip;
- **R0 — presentation boundary:** after topology commit/retirement and before
  any P16 page publication.

There is no legal cut inside a velocity extrapolation sweep, transport scatter
transaction, gamma scatter/finalize pair, sharpening scatter/finalize pair,
Krylov iteration, D4 preserve/commit pair, or topology receipt transaction.

## Concrete migration slices

| Slice | Boundary | Change and exact byte rule | GPU publication and local fail-closed rule |
| --- | --- | --- | --- |
| M00 | F0 | Install canonical word-comparison and physical-hash receipts. No simulation bytes change. | Receipt ABI publishes only when every declared field range is covered once. Missing coverage faults that field generation. |
| M01 | F0 | Upload dormant HTP1. Its embedded LOD1, cells, rows, terms, incidence CSR, directed edges, and requirements must equal the authoritative templates word for word under the HTP1 equivalence map. | HTP1 remains unread by physics. Header failure or one malformed record invalidates only that record receipt; HTP1 cannot publish until all records validate. |
| M02 | F0 | Allocate all phase buffers without consumers. GPU-copy every legacy `ResidentStateLayout` field to its migration-map destination, including padding where the old ABI exposes padding. Copy candidate, conditioning, presentation, and journal capacities without reinterpretation. | Each region owns a copied-word count, first mismatch, and complete bit. No complete bit means all new indirect counts remain zero. No host retry path. |
| M03 | F0 | Cut PhysicsState reads/writes to the new buffer. Density A/B, gamma A/B, cell velocity A/B, face A/B, and pressure retain their old byte offsets. Rigid fields use the checked migration offsets. | Compare old/new physical fields for one shadow-authored frame before the one-way binding publication. A field mismatch faults its brick packets and prevents frame seal. |
| M04 | F0 | Cut RHS, diagonal, residual, preconditioned vector, direction, applied image, divergence, sharpening delta, symmetry gamma, conditioning, partials, and scalars to ScalarScratch. Copy bytes first; do not use “scratch is dead” as permission to skip the copy at the cut. | Region receipts seal independently. The phase descriptor exposes ScalarScratch only to transport, pressure, projection, and diagnostic phases that declare it. |
| M05 | P0 | Copy the former `candidateState[0..pressureScratchBytes)` into PressureCache at `pressureScratchBase`. Copy liquid and theta from their old state ranges. Relative offsets of effective edges, brick aggregates, hierarchy levels, and density cache remain identical. | Publish one coefficient generation per repaired brick and ancestor chain. A mismatch zeros that brick’s solve packet and leaves the new pressure generation unsealed; it never launches a complete rebuild. |
| M06 | T0 | Copy the complete former candidate allocation into TopologyCandidate, then cut all six B16 cell channels and six B16 face channels. Channel addressing remains exactly `6*B^3 + 6*B^2` floats per candidate slot. | A candidate brick owns its transfer and receipt generation. Failure rejects that candidate descriptor and incident-row publication; accepted physical bytes are never overwritten then repaired. |
| M07 | F0/R0 | Move tracers and presentation storage, then journal and readback storage. The 16-byte presentation uniform remains a separate physical buffer. Copy every metadata, worklist, sample, tracer, journal, and initialized work word. | One page/journal region seals independently. Diagnostics are terminal consumers and cannot author worklists or policy. MAP_READ storage is physically separate from writable storage. |
| M08a | F0 | Replace logical-owner hash probes with embedded LOD1 accessors. Query results are compared for every logical key and representative finest coordinate; sampling arithmetic and corner order do not change. | A malformed owner fails only that query/packet and prevents its coverage generation. There is no hash-probe fallback. |
| M08b | F0 | Replace literal cell geometry and brick/rung access with HTP1. Preserve the exact 32-byte source geometry bits and canonical cell ids. | Cell identity/metadata faults reject the cell packet and its containing brick seal. |
| M08c | F0 | Replace row and term access with tagged HTP1 packets. Common two-term rows remain in source term order; mixed rows retain every variable term in source order. | Row identity, range, requirement, or term fault rejects the row packet and incident coefficient generation. |
| M08d | F0/P0 | Replace incidence and pressure/extrapolation CSR access. Preserve per-cell row order and directed-edge neighbor order exactly. | A bad CSR range rejects only its cell packet and connected row receipts. No reconstructed or global edge path is allowed. |
| M09 | F0 | Publish deterministic B16 tile-owned FramePlanCurrent/Next beside the still-complete accepted dispatches. Then cut one packet family at a time: face, mass, gamma, surface, pressure coefficient, and presentation. | A tile packet seals only after donor/receiver/incidence/characteristic closure is covered. Overflow or stale provenance zeros that packet; it does not select the full accepted list. |
| M10 | P0 | Make liquid, theta, effective edges, brick aggregates, and hierarchy coefficients persistent. First run must rewrite and compare the complete pressure cache. Later frames may reuse only equal `(topology, scalar, solid, coefficient)` generations. | Dirty brick and ancestor masks are bounded owners. Any generation mix suppresses that brick/ancestor solve packet and leaves pressure publication unsealed. No full MGPCG repair path exists. |
| M11 | Existing mathematical boundaries | Apply only the provable traversal coalescings listed below, one at a time. Keep invocation mapping, source loads, f32 operation sequence, integer quantization, and final stores unchanged. | Each fused packet carries both original coverage bits. Missing either bit rejects the packet and frame seal. |
| M12 | Gamma image boundaries | Replace row-owned gamma atomics with cell-owned incidence gathers only after a checker proves identical signed integer receipts, identical overflow semantics, and identical source snapshot. Preserve the two image boundaries. | A receipt-count or sum mismatch rejects the affected cell/tile generation. There is no row-scatter fallback. Full physical byte/hash proof through two seconds remains mandatory. |
| M13 | C0 | Cut candidate preparation to prepared-brick and incident-row packets. Grading closure, synthesis, transfers, row reconstruction, and receipts stay device-owned and locally generated. | Only fully receipted brick descriptors flip. Faulted candidates retain the already accepted descriptor by never publishing a replacement; this is transaction rejection, not rollback work. |
| M14 | R0 | Cut P16 publication to dirty page packets. A clean page is reused only under equal accepted-topology and destination-density generations plus complete dependency coverage. | A page fault suppresses that page generation. It cannot trigger all-page publication. Compare every emitted and reused sample word. |
| M15 | F0 | Move parity, frame generation, CFL revalidation, and phase indirect publication to GPU-owned state. The host encodes one fixed schedule and external controls only. | An unsealed plan publishes zero indirect counts. CPU readback remains diagnostic and can neither retry nor broaden work. |

Slices M08a–M08d are deliberately separate. A single “switch topology” change
would make a mismatch impossible to attribute and would enlarge a one-record
failure into an arena-wide fallback temptation.

## Coalescing classification

All coalescings still pass the physical byte/hash gate. “Provable” means the
ordering proof does not depend on scene behaviour. “Hash proof” means a
closure, duplicate, floating reduction, or sparse reuse claim remains to be
proved empirically across the required two-second gates. “Forbidden” means a
global data dependency exists; hashing a few successful scenes cannot remove
it.

### Provably order-independent

| Coalescing | Proof obligations |
| --- | --- |
| `prepareTransportFaces` + `forceFaces` | One invocation owns one row and both write the same face in a fixed sequence. No intervening gamma, surface, or symmetry kernel reads or writes face velocity. Preserve the prepared f32 value before adding `open*dt*acceleration`. |
| `bakeEffectivePressureEdges` + `preparePressure` | One invocation owns one pressure cell. The first writes its directed-edge range; the second writes RHS/diagonal and reads theta, face velocity, and row topology, not effective-edge output. Preserve each loop’s source order. |
| `collocateAndDiagnose` + the first `measureDivergenceDiagnostics` reduction | The collocation invocation owns the cell divergence it immediately contributes. The diagnostic reduction is `max(abs(divergence))`; preserve accepted-cell/workgroup mapping and mixed-row predicate. The final singleton reduction remains separate. |
| Activity occupied-bit production + presentation wet classification | One brick workgroup already computes fixed-point density mass and occupied state. Publishing that exact bit removes a second cell traversal; presentation only consumes it after the activity barrier. |
| Producer-local dirty-mask OR + producer output | OR is idempotent and targets a disjoint scheduling word/generation from the physical output. The producer owns the tile bit and must not append to an unordered list. |
| Copying all phase indirect triplets in one snapshot | The source words are sealed, disjoint, and read-only during the copy. Packing multiple copy ranges changes no dispatch count or order. |

### Require full physical hash proof

| Candidate coalescing/change | Why algebra alone is insufficient |
| --- | --- |
| Final velocity-validity/CFL classification in the last extrapolation sweep | Correctness depends on the exact final ping-pong bank, dirty closure, and equal treatment of invalid/dry cells. |
| Gamma row scatter to cell incidence gather | Integer addition is order-independent, but duplicate incidence, receipt recomputation, conversion/rounding, and signed overflow must be shown identical. |
| Any accepted-list traversal replaced by dirty tile packets | Completeness depends on donor, receiver, characteristic, incidence, and scatter-target closure rather than local arithmetic. |
| Persistent pressure cache reuse or incremental membership repair | A clean certificate must cover liquid, theta, solid scaling, effective edges, aggregates, and every hierarchy ancestor at one generation. |
| Brick aggregate or hierarchy traversal fusion/reordering | Workgroup floating reductions and edge accumulation order affect pressure and Krylov bits. |
| Pressure row classification combined with coefficient repair | Theta requires complete row classification and membership closure; a local-looking fusion can consume a partially published neighbor. |
| Activity measurement combined with resolution planning | Planning reads a complete brick census and neighbour grading state. Producer-local scores may be ready while global closure is not. |
| Candidate synthesis/transfer restricted to prepared bricks | Completeness depends on grading, incident-row closure, conservation, and exterior face receipts during large topology changes. |
| Dirty P16 page reuse | Physics is unchanged, but exact presentation depends on every density/topology/wetness source and macro-page coverage. |

### Forbidden across a single dispatch boundary

- consecutive velocity-extension sweeps;
- `traceGammaAndBeta` → `scatterDensityDeficit` →
  `gatherConservativeDensity`;
- gamma scatter → gamma finalize, and snapshot finalize → refinement scatter;
- sharpening prepare → scatter → finalize, including solid-excess scatter;
- D4 preserve → commit;
- pressure cell classification → row classification → theta publication;
- `projectFaces` → `collocateAndDiagnose`;
- any Krylov update, reduction, guarded residual, restart, or recovery boundary;
- activity reduction → neighbour grading closure;
- candidate transfer/reconstruction → receipt reduction → generation flip;
- topology commit/retirement → presentation generation seal.

These pairs have cross-invocation reads or transaction-wide publication. They
may share a compute pass when WebGPU ordering permits, but they may not become
one dispatch without an equivalent global/workgroup synchronization design.

## Acceptance matrix

Every slice records zero word mismatches and equal physical hashes at every
checkpoint for:

- ocean seiche, B16/P16, sparse CM12, including the combined non-pressure p95;
- mini dam break 64 through at least two simulated seconds, including dam-front
  position, mass/gamma/momentum ledgers, pressure membership/operator hashes,
  and topology commits;
- weakened symmetric expansion through at least two seconds, including density
  and velocity D4 hashes and topology evolution.

M10–M14 additionally require frames with many simultaneous topology changes.
The expected result is more independent brick packets, not a different global
path. No slice is accepted from a one-step or stable-ocean result alone.

## Host authority elimination map

Construction may still choose immutable capacities and pipeline capabilities.
It may not make an evolving per-frame work/no-work decision. Every evolving
authority below lives in a GPU state word with a generation, and every
conditional family is encoded every frame through an indirect triplet. The
disabled value is `x=0`; omission of the dispatch on the host is not allowed.

| Current host value or branch | Classification | Final authority and fixed schedule |
| --- | --- | --- |
| `parity` and `this.parity ^= 1` | Evolving per-frame authority | A GPU frame-generation word owns the low bank bit. Source/destination offsets, renderer bank selection, and diagnostics read the same published word. A seal kernel advances it only after all physical packet coverage; the host never flips or selects a bank. |
| `horizontalD4Authority` and the two D4 `if` branches | Evolving physical authority | A GPU D4 receipt contains density generation, gamma generation, topology generation, active-body generation, authority bit, and fault. Density and velocity D4 families are always encoded with GPU-authored indirect triplets. `x=0` when authority is absent, stale, faulted, or an active body breaks symmetry. Injection/topology/solid producers invalidate the word on device. |
| Accepted/shadow cell and row counts, pressure member counts, dirty tile/page counts, prepared candidate counts | Evolving topology/frame authority | FramePlanCurrent/Next, PressureCache, and TopologyCandidate own stable segmented counts and indirect triplets. Every consumer uses indirect dispatch. Empty, incomplete, or faulted generations publish `x=0`; the host does not inspect a count or choose a full-capacity fallback. |
| `packed.brickCount`, template cell/row capacity, HTP1 word count, hierarchy level/group capacity | Static construction-time configuration | These size buffers, compile constants, and fixed maximum-domain dispatches only. They never claim how much evolving work is live. Four B16 grading rounds may be encoded unconditionally because `log2(16)` is immutable; their live brick packets are still GPU indirect. |
| `hostTemplateVariants`, page-pool capacity, rigid-coupling capability, presence and geometry of a static boundary | Static construction-time configuration | These select an ABI/pipeline and allocate capacity once. They may not be retested to omit a per-frame phase. A capability with no current work still has a fixed encoded indirect family whose GPU count is zero. |
| `bodyCount > 0`, `this.boundary || bodyCount > 0`, and `bodyCount === 0` around solid/D4/reaction work | Evolving per-frame authority over a static capability | External body descriptors and controls may be uploaded, but voxelization publishes the authoritative active-body count and generation. Voxelization, solid-excess, D4, projection coupling, and reaction families are always encoded. Their GPU predicates author complementary indirect counts (`solid/reaction > 0`, `D4 = 0` when bodies are active). Static boundary capability is one input to the same predicate. |
| `tracersEnabled`, `tracerSeedPending`, and zero tracer count | Evolving presentation authority | UI enable is an external control word. GPU presentation state derives seed/advance indirect triplets and consumes the published physics generation. Disabled or zero-capacity views produce `x=0`; host code does not add or remove tracer dispatches. |
| `journalArmed` and host-selected snapshot iterations | Evolving diagnostic authority | Arm state is an external diagnostic control, while record/snapshot cursors and the log-spaced schedule are GPU-derived from the fixed iteration ceiling. Journal record and snapshot families remain encoded and publish `x=0` while disarmed. They cannot affect solve dispatch or convergence. |

The CPU may still upload `dt`, acceleration, body descriptors, UI visibility,
and diagnostic controls. Those are inputs, not scheduling authority. It also
encodes the fixed pipeline schedule and construction-time maximums. A GPU
predicate translates inputs plus published generations into indirect work; a
CPU `if` must never do so.
