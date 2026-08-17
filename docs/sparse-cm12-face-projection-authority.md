# FPA1 exact face/preparation and projection authority

FPA1 is the standalone B16/P16 work authority for the current
`prepareTransportFaces` and `projectFaces` equations. It changes invocation
enumeration, not either equation. Stable ids are HTP1 row ids and both work
sets are deterministic rank-select views over 256-row leaves.

## Exact certificate

A clean row is reusable only while its producer-owned local dependency
generation is unchanged. That generation advances on an exact input-bit
change, not on a tolerance, hash, zero-velocity predicate, or expected ocean
behavior. Preparation dependencies cover the complete prior characteristic
read closure: topology owner/rung, extended cell-velocity words and validity,
source density phase words, source face word, dt/CFL policy bits, embedded
boundary and moving-solid words. Projection dependencies cover the prepared
authority word, PCM membership, PCF theta/effective coefficients, pressure
endpoint words, force and solid-open words.

This admits stable nonzero solenoidal motion: if every authoritative bit read
by the prior invocation is unchanged, the unchanged HEAD invocation would
take the same branch/loop order and produce the same f32 word. A producer
generation/provenance gap faults the affected FPA generation and publishes
zero indirect work; it never launches a global runtime traversal.

FPA1 stores the exact pre-pressure face word separately. A projection-only row
restores this word before executing the unchanged projection body. After the
last operation, a projected row mirrors its final word into both face banks.
A prepared row outside PCM mirrors its prepared word into both banks. Thus a
clean row needs no copy or invocation on the next parity.

## Local closure and work construction

- Producer events call `fpaMarkPreparationRow` or `fpaMarkProjectionRow` with
  the exact current dependency generation.
- `fpaMarkTopologyCellBlast` walks the HTP1 cell incidence and one row/cell
  ring. It marks directly incident rows at depth zero and neighboring incidence
  rows at depth one. PCM gates projection membership.
- Candidate rows use generation-stamped tokens. The first event for a
  256-row leaf performs one atomic append; repeated row events only OR cause
  bits and maximize closure depth.
- The union of the previous and current non-empty leaves is locally repaired.
  A 32-way count tree supplies canonical stable-id rank select. The active-leaf
  append order is irrelevant because numerical invocation order comes only
  from the tree.
- Execution receipts are written per unique row, then verified/reduced once
  per active leaf. Missing producer or execution coverage faults and zeros the
  indirect triplets.
- Bootstrap and the construction-only QA oracle use a GPU-authored indirect
  triplet. Production compiles `qaFullOracle=false`; no runtime word can select
  the oracle or a full fallback.

## Resident integration map (serial ownership only)

Do not integrate while another lane owns resident source.

1. Build HTP1 first and require its cell/row counts to match the resident
   template capacities. Append FPA1 after the final FPL/FCA1/SAW1 control
   allocation in the shared scheduling arena. `baseWords` is the aligned
   previous `totalWords`; grow the initializer through `fpa.totalWords` and call
   `initializeSparseCM12FaceProjectionAuthorityWords` on that complete shared
   array (the layout uses absolute offsets). The convenience creator is for a
   standalone arena and must not be copied over a live prefix.
   FPA1 does not overlap PCM, ACT1, FPL1, FCA1 or SAW1.
2. Expose the existing scheduling storage binding as `array<atomic<u32>>` to
   `createSparseCM12FaceProjectionAuthorityWGSL`. Supply HTP1 accessors and PCM
   row membership/generation. FCA1 supplies frame generation, parity, body and
   boundary generation; SAW1/velocity authority supplies exact density and
   extended-velocity dependency events.
3. Reserve six copy-isolated indirect snapshots: preparation and projection
   bootstrap/oracle, preparation and projection repair, and preparation and
   projection work. Host encoding is fixed copies plus indirect dispatches.
   Counts and phase never reach the CPU.
4. After velocity extension and before face preparation: begin only the
   preparation epoch, seed its prior active leaves, emit velocity/density/
   topology events, run its zero-count bootstrap indirect, repair/finalize its
   rank tree, execute, verify and accept it. Do not open projection here: its
   PCM and pressure dependencies do not exist yet.
5. Replace only the invocation helper in `prepareTransportFaces` with
   `fpaPreparationRowInvocation`. Preserve its body byte-for-byte. Store the
   resulting destination word through `fpaStorePreparedAuthority`; if the row
   is outside current PCM, mirror it to the source bank; then call
   `fpaPreparationComplete`.
6. After PCM/PCF and the pressure solve have accepted their current
   generations, begin the projection epoch. Seed its prior leaves, seed rows
   from the accepted preparation work set, emit pressure/coefficient/force
   events, then repair/finalize. Before the unchanged `projectFaces` arithmetic,
   restore destination face
   from `fpaPreparedAuthorityBits(row)`. Replace only its invocation helper
   with `fpaProjectionRowInvocation`. After subtraction, mirror the final word
   to both banks and call `fpaProjectionComplete`.
7. Verify and accept projection before collocation. FPL1 stage 0 consumes FPA
   preparation receipts; FCA1 accepts face parity only after both independently
   sequenced FPA stages seal for the same frame generation.
8. Construction-only QA creates a second solver with `qaFullOracle=true`.
   Compare every prepared-authority word, both final face banks, cell velocity,
   pressure and divergence after each step. The production solver never
   switches modes after construction.

The resident touchpoints are therefore limited to the arena/initializer and
indirect-buffer plumbing in `WebGPUSparseCM12Resident.create`, WGSL generator
arguments/hooks, the fixed face-authority schedule around `face-preparation`
and `velocity-projection`, and the two unchanged numerical entry points.

## PCF1 placement against FPL/FCA1/SAW1

PCF1 belongs to the non-aliased `cm12.pressure-cache` phase arena, not the
FPL/FCA1/SAW1 scheduling arena:

- `theta` maps exactly to `PressureCache.theta`.
- effective directed-edge words map exactly to
  `PressureCache.effectiveEdgeWeights` and retain HTP1 edge order.
- PCM cell/row bitsets and rank trees occupy the beginning of
  `PressureCache.membership`.
- Call `createSparseCM12PersistentPressureCacheLayout` with
  `controlOffsetWords` equal to the aligned PCM end relative to the membership
  region. PCF1 then places its header, persistent diagonal, dirty tokens,
  dirty-leaf stamps and lists after PCM without aliasing.
- FPL stage 4 and FCA1 pressure-family indirects consume PCF1 accepted
  generation. SAW1 only contributes density/phase transition events; it does
  not own coefficient storage.
- PCM transitions mark incident HTP1 rows and their directed edges. Topology
  or solid changes mark only affected leaf bricks, aggregate edges and
  hierarchy ancestors. The existing full `bakeEffectivePressureEdges`, brick
  aggregate and hierarchy bakes remain available only in the separately
  constructed QA oracle, never as a production fallback.

Serial integration touches the phase-arena plan/allocation, pressure bind-group
buffer selection, PCM-to-PCF event calls, pressure coefficient entry-point
invocation, and FPL/FCA1 receipt wiring. It does not alter the HEAD coefficient,
diagonal or RHS expression order.

## Performance target

The current ocean medians are about 14.09 ms for face/receiver preparation and
8.52 ms for projection. A topology-quiescent frame should issue no numerical
workgroups for certified rows; only the small begin/finalize and previously
active leaf repair remain. The target is below 1 ms combined at B16/P16.
Dam-front work remains proportional to the changed characteristic/interface,
PCM and topology closures rather than to 1.57 million accepted rows.
