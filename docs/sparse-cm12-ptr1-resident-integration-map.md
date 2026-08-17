# PTR1 bounded pressure-topology repair integration map

PTR1 replaces the three domain-sized singleton loops in the resident pressure-topology
stage. It is a binding-free WGSL module backed by a versioned atomic arena. Runtime work
is proportional to the topology journal and its exact cell/incidence closure; invalid or
incomplete provenance publishes zero indirect counts and faults closed.

## ABI and arena placement

- Layout: `createSparseCM12PressureTopologyRepairLayout({ brickCapacity,
  rowCapacity, baseWords, brickFineResolution: 16, presentationPageResolution: 16 })`.
- Source: `createSparseCM12PressureTopologyRepairWGSL({ layout, arenaName, prefix:
  "ptr", workgroupSize: 64 })`.
- Append PTR1 after the final PSA1 region in the existing pressure authority arena. This
  preserves the existing PCM, PCF/PCA1, and PSA1 offsets. Before that combined arena is
  introduced, appending to the mutable `pressureWorklists` tail is ABI-safe.
- No new storage binding is required. The selected arena declaration must be
  `array<atomic<u32>>`; immutable pressure topology remains in its existing binding.
- Allocate one 60-byte `INDIRECT | COPY_DST` snapshot. Copy five triplets into it:
  brick seed, brick repair/work, row seed, row repair/work, and commit may instead use a
  72-byte snapshot when both seed triplets are retained. Offsets come only from
  `sparseCM12PressureTopologyRepair*IndirectByteOffset`; host arithmetic is forbidden.
- Initial accepted brick states are construction data. Upload each physical brick's
  `{active bit | accepted resolution}` to `brickAcceptedStateBaseWords`. Runtime never
  asks the CPU whether PTR1 is bootstrapped.

## Required resident hooks

The binding-free source requires generation hooks for frame/topology, PCM cell/row, and
PCF; journal count/record hooks; template cell ranges; cell incidence; current pressure
membership and row/theta classification; and the existing PCM/PCF candidate/event APIs.
The exact hook names are exercised by
`tools/check-sparse-cm12-pressure-topology-repair.ts`.

The topology commit producer must emit one compact record per changed physical brick:
`vec4u(brick, oldState, newState, causeMask)`. The record is generation-stamped and
transactional. Identical duplicate producers merge their causes; conflicting old/new
provenance faults. A publishing token prevents a duplicate from observing a partially
written record.

## Resident touchpoints

1. In `WebGPUSparseCM12Resident.create`, append and initialize PTR1 after the combined
   pressure authority layout. Extend only the existing arena allocation and binding.
2. Pass the layout into `createSparseCM12ResidentWGSL`; append the generated binding-free
   helpers after PCM and PCF helpers so their APIs are in scope.
3. Add PTR1 entry points to the resident pipeline list. Their names are the exported
   compute names in the generated source; reduction entry points are fixed by the
   construction-time tree depth.
4. In `validateAndCommitShadowTopology`, emit the old/new physical brick record into the
   existing topology journal. Do not run pressure classification there.
5. In `stage("pressure-topology")`, remove these singleton dispatches:
   `classifyPressureTopologyCells`, `classifyPressureTopologyRows`, and
   `finalizePressureTopologyRepair`.
6. Preserve temporal cell/row PCM producers. PTR1 contributes only topology blast roots;
   PCM candidate writes coalesce by stable ID before canonical repair.
7. Replace transient global coefficient/aggregate/hierarchy bakes with PCF/PCA1 repair,
   then run PSA1. PTR1 emits membership, topology-cell and theta events directly into
   PCF before PCF finalization.

## Fixed host encoding and ordering

1. Begin PCM cell/row and PCF candidates; begin PTR1.
2. Import the prior topology commit journal indirectly. Seed prior active brick/row leaves
   through PTR1-authored indirect counts.
3. Finalize brick frontier; copy the brick-repair triplet; repair dirty 256-brick leaves.
4. Run each brick count-tree level in ascending order. Every level dispatches over the
   same dirty-leaf indirect count and touches only ancestors of those leaves.
5. Compact brick-plan finalizer reads only header/root counts. Copy brick-work triplet.
6. Execute one 64-lane workgroup per changed B16 brick. Lanes stride old cells, cross a
   uniform barrier, then stride new cells. Incidence rows are generation-deduplicated.
7. Finalize/accept PCM cells, then finalize the PTR1 cell epoch. Copy row-repair triplet.
8. Repair row leaves and their local ancestors; compact row-plan finalizer; copy row-work.
9. Recompute theta in canonical stable row rank order; finalize/accept PCM rows.
10. Repair and accept PCF fine edges, brick aggregates, hierarchy edges and diagonals.
11. Validate topology/PCM/PCF generations, copy commit triplet, commit only changed brick
    states, and run the WG1 compact finalizer. It reads counters and roots only.

Passes 3/4 and 8 may share a compute pass when indirect copies occur before that pass.
Cell execution cannot share a pass with PCM cell finalization. Row execution cannot share
a pass with PCM/PCF finalization. No topology-commit journal record becomes accepted until
the final PTR1 transaction commits.

## Fail-closed receipts

The 64-word header publishes phase, candidate/accepted/frame/topology generations,
PCM/PCF generations, expected/covered producer receipts, changed brick/row counts,
cell/row execution counts, cause mask, fault, first family/ID, and every indirect triplet.
Allocator, range, provenance, closure, execution, or generation gaps zero all PTR1
indirects. There is no dirty-ratio threshold or global runtime fallback.

## Exact construction fixture and checks

`node --import tsx tools/check-sparse-cm12-pressure-topology-repair.ts` runs an exact
two-brick fixture: brick 1 changes B16 to B8 and brick 6 is created at B16. It verifies
stable brick order, lane-range closure, duplicate incidence-row elimination, PCF event
sets, conflicting-provenance rejection, absence of singleton capacity walks, and Naga
validation of the complete binding-free source.
