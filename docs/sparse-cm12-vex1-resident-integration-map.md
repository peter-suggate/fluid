# VEX1 resident integration map after FCA1/FPL1

## Frozen baseline

This map is intentionally standalone. It does not edit the resident while the
FCA/dataflow cutover owns that source. The accepted baseline to preserve is:

- FPL1/FPP1 presentation bytes match the immutable publisher oracle for five
  dam64 steps;
- FPP1 scheduled, executed, and published page counts agree, with zero omitted
  pages and faults;
- the Grid structure selector reads live FPL1 records and renders missing or
  invalid generations as unknown/magenta;
- the 60-step local/full pressure oracle is physically byte-exact; and
- ocean B16/P16 presentation publication measures 0.2621 ms median.

VEX1 must not change FPP metadata, the FPL current/next flip, pressure
membership, or any scalar/face bank arithmetic.

## Stable placement and helper dependencies

FCA1 is appended to `topologyArena`. It does not shift either VEX1 tail.

The activity order remains:

```text
dirty scheduler -> ACT1 -> PCM -> FPL1 -> FPP1 -> VEX1
```

Create the VEX1 activity layout with
`activityTailWords = framePlanPresentationLayout.totalWords`. Extend the
initial activity allocation to `velocityExtensionLayout.totalWords`, then copy
`createSparseCM12VelocityExtensionInitialWords` at
`velocityExtensionLayout.headerBaseWords`.

The persistent vector cache is an aligned `vec4f[cellCapacity]` after the
complete current state layout, including optional pressure-journal storage.
`createSparseCM12VelocityExtensionResidentLayouts` is the single composition
helper for these two placements. No pressure/candidate scratch may alias it.

The current immutable edge contract is unchanged:

```text
pressureTemplateWord(15)
pressureEdgeRows()
pressureNeighborOffset()
pressureExtrapolationWeightOffset()
rowAccepted(row)
```

HTP1 may later replace the implementation behind those accessors. VEX1 must
not learn a second edge order: directed edges are accumulated in the exact
existing CSR order.

FCA helpers used by the integration are:

```text
cm12FCAcceptedGeneration()       // source fields read by this extension
cm12FCCandidateGeneration()      // output/root generation producers author
cm12FCSourceScalarParity()
cm12FCDestinationScalarParity()
cm12FCSourceFaceParity()
cm12FCDestinationFaceParity()
```

At runtime FCA is sealed before velocity extension. FPL Current is the
previously accepted plan at that point and must carry the same accepted frame
generation as FCA's source generation.

## Temporal lifecycle correction

The first standalone draft cleared `rootCount` in
`beginVelocityExtensionCandidate`. That loses exactly the roots produced by
projection, topology commit, injection, and solids after the preceding
extension. The corrected lifecycle is:

1. FCA seals generation `g+1`; source fields and FPL Current are generation
   `g`.
2. VEX1 begin validates and seals the root queue already collected for `g`.
3. The compact eight-edge recurrence executes and accepts cache generation
   `g`.
4. VEX1 immediately reopens its now-dead root queue for FCA candidate `g+1`.
5. Projection, classification, topology, injection, and moving-solid
   producers append roots for `g+1`; no later kernel clears them.
6. After FCA commits, those roots become the next frame's source-generation
   work.

Construction is the only full bootstrap. Encode it in the immutable initial
publication path, not in every runtime frame. The construction authority may
use `accepted+1` as the next root generation before FCA first enters its
sealed phase. A runtime generation discontinuity faults; it never dispatches
the legacy full-domain extension.

Injection must be an FCA transaction. Mutating accepted density while leaving
the FCA generation unchanged would make any temporal cache unsound and must
fail closed rather than receive a VEX-specific exception.

## Conflict-free resident patch map

Apply only after the FCA owner releases the shared resident files.

### `webgpu-sparse-cm12-resident.ts`

1. At the `ResidentStateLayout`/`residentStateLayout` anchors, append
   `velocityExtensionAcceptedVelocity` and grow `floatCount` through
   `createSparseCM12VelocityExtensionStateLayout`. Preserve every existing
   field offset byte-for-byte.
2. Immediately after `framePlanPresentationLayout`, create the composite VEX1
   layout. Grow `initialActivity`; initialize VEX1 after FPP1. Do not reorder
   FCA, PCM, FPL, or FPP.
3. Add one 12-byte `INDIRECT|COPY_DST` VEX snapshot buffer. Reuse it serially
   for root, alternating frontier, and blast triplets; counts remain GPU-owned.
4. Append VEX generator arguments after all existing arguments. Do not reorder
   the FCA/pressure/FPL signature.
5. Register begin/bootstrap/root seal, seed, frontier prepare/expand/seal,
   blast finalize, candidate initialize, eight recurrence specializations,
   commit, and finalization pipelines. The eight recurrence depths and A/B
   parity are construction-time pipeline constants, never frame uniforms.
6. Add every VEX buffer to destruction/accounting. Expose its 32-word header
   only as optional diagnostic provenance; scheduling never reads it on CPU.

### `webgpu-sparse-cm12-resident.wgsl.ts`

1. Emit VEX1 after FCA, pressure-edge, dirty, and FPL helpers exist. Configure:

   ```text
   arenaName = activity
   stateName = state
   sourceFrameGeneration = cm12FCAcceptedGeneration()
   nextFrameGeneration = cm12FCCandidateGeneration()
   frameAuthorityReady = FCA phase is sealed
   topologyGeneration = accepted topology epoch
   provenanceHookPrefix = cm12Resident
   ```

2. The provenance bridge maps a template cell minimum to the stable B16 4^3
   tile. It must:
   - record projected/class/topology/solid roots as FPL stage 0 direct;
   - record positive graph depth as stage 0 closure with that exact depth;
   - validate every blast cell against FPL Current stage 0 before mutation;
   - return a local descriptor signature from
     `VelocityExtensionOwner(cell)`, not merely brick/resolution; and
   - latch both FPL local unknown and FCA frame fault on missing coverage.
3. A retired cell calls `cm12ExtensionInvalidateRetiredCell`; every still-live
   incident endpoint is an ordinary topology root. Never attempt to append an
   inactive retired cell to the recurrence.
4. Transport and face sampling call `cm12ExtensionTransportVelocity`. Do not
   copy clean cache values into the ordinary velocity banks.

### Encode order and indirect seams

The runtime order is fixed and contains no CPU count or parity branch:

```text
FCA begin/publish/seal
VEX begin
seal roots -> copy root triplet -> seed roots
seal seed frontier
for depth 1..8:
  prepare fixed depth
  copy prior frontier triplet
  expand indirect
  seal output frontier
finalize blast -> copy blast triplet
initialize candidates indirect
advance depth 1..8 indirect, exact legacy bank parity
commit dirty air cache indirect
finalize VEX candidate / reopen next root collection
FCA/FPL coverage seam
face preparation indirect
```

Construction additionally authors the one-time full bootstrap before sealing
roots. A VEX/FPL validation fault zeroes downstream FCA indirect work after a
storage/copy seam. Partially written scratch is never accepted or sampled.

## Grid-selector truthfulness audit

FPL tiles are the scheduling authority. CMD1 remains backwards-compatible
provenance, not an independently reconstructed answer.

| Grid logical stage | Direct evidence | Closure evidence | Executed receipt owner | Required clean/unknown behavior |
| --- | --- | --- | --- | --- |
| Face preparation (stage 0) | VEX projected/class/topology/solid roots; face characteristic, boundary, and CFL roots | exact VEX graph depth plus face-incidence/support dilation | terminal face-stage receipt after VEX and face packets both finish | valid pre-commit active tiles outside the mask are skipped; any missing VEX/face witness is unknown, never reused |
| Mass transport (stage 1) | SAW density/phase/injection roots | characteristic support, deficit/scatter dependency closure | after trace, deficit scatter, and conservative gather all finish | clean tile is skipped only when all three subpasses certify it |
| Gamma transport (stage 2) | SAW gamma/density/class roots | row-incidence and diffusion closure | after transport and every required gamma-diffusion image | a coalesced mass/gamma packet retains both logical masks and one truthful physical packet epoch |
| Surface conditioning (stage 3) | phase/density/gamma and boundary roots | prepare/scatter/finalize incidence closure | terminal surface finalize | disappearing maxima force the local tile rebuild; stale summary is unknown |
| Pressure coefficients (stage 4) | PCF membership/theta/solid/topology roots | incident rows, directed edges, aggregate and ancestor closure | after coefficient/diagonal/RHS publication, before solve consumes it | clean coefficients are skipped only with matching PCM/PCF/topology generations |
| Presentation (stage 5) | density/phase/topology/activation/retirement/boundary roots | presentation dependency closure | FPP candidate-copy commit | missing page or generation is magenta; omitted page is locally invalid, never silently stale |

For every valid tile and stage exactly one of executed or skipped must be set.
Direct and closure are mutually exclusive. Producer and consumer generations
must equal the accepted FCA/FPL generation. Inactive retained bricks are not
counted as reused. Stages 0-4 use the topology generation they actually
traversed; stage 5 uses post-commit topology.

The temporary unconditional stages-0..4 presentation-era receipt must be
deleted only when all five real consumers above author their own receipts in
their actual packet epochs. Until then, an uncut stage remains explicitly
global/direct with generation-mismatch provenance; it must never appear green
or reused.

Pressure solve iteration work is not stage 4 coefficient dirtiness. Its active
cell/row/hierarchy census remains a separate pressure receipt; do not paint it
as skipped merely because coefficients were reused.

## Acceptance after handoff

No threshold changes are allowed. Required evidence is:

- standalone TypeScript/lint and integrated Dawn validation;
- initial/bootstrap and injection generation receipts;
- five-step dam and weakened-symmetry smoke;
- two-second authoritative density, gamma/internal transport, velocity,
  pressure, and divergence hashes against the frozen baseline;
- zero uncovered VEX/FPL/CMD/FCA writes and first-mismatch diagnostics;
- ocean B16/P16 VEX roots, blast depth/cell counts, executed/reused counts,
  and hardware stage time; and
- a Grid capture showing direct, closure, skipped, unknown, cause, depth,
  generation, and pass-packing modes for every stage.
