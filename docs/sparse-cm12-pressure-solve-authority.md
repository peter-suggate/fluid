# PSA1 sparse pressure-execution authority

PSA1 removes empty physical bricks and empty pressure-hierarchy nodes from the
pressure topology, RHS/preconditioner, and encoded solve work without changing
the pressure equations or the order of any retained invocation. PCM1 remains
the stable-ID authority for liquid cells and active rows. PCF1 remains the
stable coefficient/diagonal authority. PSA1 is the derived execution domain:

```text
PCM1 liquid cells / active rows
             | exact membership transitions
             v
PCF1 edge + diagonal repair ---- accepted PCF generation
             |                              |
             +--------------+---------------+
                            v
                PSA1 wet bricks + live hierarchy nodes
                            |
                            v
        canonical rank-select pressure topology / RHS / solve work
```

There is one production algorithm. Missing provenance, an incomplete producer
receipt, an invalid stable ID, or a generation mismatch publishes zero indirect
counts. There is no dirty-ratio threshold and no complete runtime bake/solve
fallback.

## Stable domains and layout

The standalone ABI is implemented by
`sparse-cm12-pressure-solve-authority.ts` and has magic `PSA1`, version 1.
It is fixed to B16/P16.

- Wet-brick IDs are immutable physical HTP1 brick IDs.
- Hierarchy-node IDs are `levelOffset[level] + group`, with levels concatenated
  in the existing pressure hierarchy order.
- Both domains have persistent bitsets, generation-stamped 256-entity dirty
  leaves, and 32-way count trees.
- Rank select visits stable IDs in increasing order. Dirty-leaf append order is
  therefore never observable by numerical kernels.
- One repaired leaf performs eight word writes and one signed count delta at
  each count-tree ancestor. There is no global bitset or tree scan.
- The two compact solve-tail banks contain four indirect triplets each: PCM
  cells, wet bricks, hierarchy nodes, and scalar/reduction work.

The arena is placed in the non-aliased `cm12.pressure-cache` membership tail:

1. PCM1 begins at `PressureCache.membership`.
2. PCF1 begins at the first aligned word after PCM1.
3. PSA1 `baseWords` is `pcfLayout.controlEndWords`.
4. `pressureMembershipWords` must cover PSA1 `totalWords` through the end of
   the membership region.

PSA1 does not use FPL/FCA/SAW transient arenas and does not alias pressure
vectors or scalar scratch. Its A/B indirect triplets are copied to small
`INDIRECT | COPY_DST` buffers at pass seams; `cm12.pressure-cache` remains
`STORAGE | COPY_SRC | COPY_DST`.

## Exact local closure

PCM calls `psaMarkPCMCellTransition` only for actual membership transitions.
That marks the containing 256-brick leaf once. Topology publication calls
`psaMarkTopologyBrickBlast` for every created, retired, or identity-changed
physical brick. Coefficient-only events stay in PCF1 unless they also change
wet membership.

Brick repair evaluates exact wetness from PCM membership for only the dirty
brick leaves. For every brick bit that changes, repair marks the brick's parent
at every existing hierarchy level. Node repair evaluates exact activity from
the repaired wet-brick bitset and the immutable HTP1 child list. This is the
complete local ancestor closure because the current hierarchy maps every
physical brick directly to one group at each level. A topology change never
requests a complete hierarchy walk.

The receipt records the frame, topology, PCM, and PCF generations, expected and
covered producer counts, cause union, dirty/repaired leaf counts, accepted
counts, and first fault family/stable ID. Acceptance requires all four
generations still to match. Faulting zeros brick, node, bootstrap, and both
tail-bank indirect families.

## Preserving HEAD arithmetic and order

Existing numerical kernel bodies are unchanged. Only their entity lookup and
dispatch source change:

| Existing family | PSA1 invocation | Dispatch x | Arithmetic/order contract |
| --- | --- | --- | --- |
| brick aggregate diagonal/RHS/refinement | `psaWetBrickInvocation(wid.x)` | wet-brick work indirect | same 64-lane loops and reductions for the returned stable brick |
| hierarchy diagonal/RHS/refinement | `psaActiveHierarchyNodeAddress(wid.x)` | active-node work indirect | same level/group body; rank is only translated to the canonical stable node |
| hierarchy-to-brick combine | `psaWetBrickInvocation(gid.x)` with one invocation per workgroup or a 64-thread rank wrapper | wet-brick work indirect | existing level loop remains in increasing level order |
| fine operator / CG vectors | existing PCM stable cell rank | existing PCM indirect, or tail cell bank after convergence gate | unchanged |
| active pressure rows / projection | existing PCM stable row rank | existing PCM indirect | unchanged and outside PSA1 |

The construction-only `qaFullOracle` specialization returns raw stable IDs and
publishes full brick/node counts. It uses the same numerical kernels and is not
a runtime policy bit. The CPU oracle compares wet/node membership and optional
per-entity u32 result words, reporting the first stable mismatch.

## PCF1 + PSA1 topology/RHS cutover

The pressure epoch is one transaction:

1. PCM accepts current liquid-cell and active-row generations.
2. PCF receives the same PCM/topology/solid events, locally repairs HTP1 edges
   and cell diagonals, and accepts a coefficient generation.
3. PSA receives the PCM transition/topology brick receipts, repairs wet bricks,
   then repairs only their hierarchy ancestors. It accepts only the exact PCM
   and PCF generations from steps 1–2.
4. Pressure topology uses PCF dirty-cell/edge work and its aggregate-owner /
   ancestor extension, masked by PSA wet brick/node authority. PCF1 now maps
   the existing persistent brick aggregate edge/diagonal and hierarchy
   edge/diagonal regions, and publishes four stable-rank local repair families.
   PSA does not pretend that an activity bit repairs a coefficient.
5. `preparePressure` and fine PCG work retain PCM stable-cell rank/order.
6. Brick aggregate RHS and all brick refinement use PSA wet-brick rank.
7. Hierarchy restriction/refinement use PSA active-node rank. PCF owns local
   coefficient/aggregate/ancestor repair; PSA only owns whether a node executes.

This coalesces membership, coefficients, aggregate topology, and RHS work under
one accepted tuple `(topologyGen, pcmGen, pcfGen, psaGen)`. An unsealed tuple
does not launch pressure work; it does not switch to a global bake.

## Converged encoded tail

Two A/B banks are provided because a gate-writing reduction cannot safely
rewrite the indirect triplet currently being consumed. The fixed host command
stream alternates banks after every scalar gate publication:

1. a direct one-workgroup publisher reads the existing exact
   `pipelinedPressureActive` authority and writes the next bank;
2. eligible arithmetic kernels dispatch indirectly from that copied bank;
3. the existing gate-writing reduction executes in its original position;
4. the next direct publisher writes the opposite bank.

Skipping is legal only for kernels whose entire externally visible effect is
already guarded by that exact boolean. It is not legal for journal records,
snapshot publication, final true-residual measurement, gate publishers, fault
receipts, or any unconditional clear. Those remain encoded exactly as today.
Thus a zero tail dispatch deletes hardware work that HEAD defines as a no-op;
it does not move or approximate an arithmetic operation.

## Resident integration touchpoints after ownership handoff

Rebase by symbol, not by current line number, because frame-dataflow owns the
shared resident files during standalone development.

In `webgpu-sparse-cm12-resident.ts`:

- extend the phase-arena `pressureMembershipWords` calculation after PCM1 and
  PCF1 with `createSparseCM12PressureSolveAuthorityLayout`;
- initialize/upload PSA1 in the same `cm12.pressure-cache` buffer;
- add copy-isolated brick-repair, node-repair, brick-work, node-work, and two
  four-family tail indirect buffers;
- append the generated source to the pressure WGSL generator arguments;
- add PSA lifecycle pipeline names beside PCM/PCF pressure-topology pipelines;
- inside `stage("pressure-topology")`, encode begin, fixed producer hooks,
  bootstrap indirect, brick frontier/repair, node repair/finalize, and copy the
  accepted work triplets;
- replace fixed `packed.brickCount` dispatches in pressure topology/RHS/solve
  with the PSA wet-brick indirect, and fixed
  `this.pressureHierarchyGroupCount` dispatches with the active-node indirect;
- retain PCM indirect for fine pressure cells/rows;
- in the encoded pressure loop, insert A/B tail publishers only at existing
  convergence-gate seams and leave journal/final residual dispatches direct.

In `webgpu-sparse-cm12-resident.wgsl.ts`:

- emit PSA1 after HTP1, PCM1, and PCF1 helpers;
- provide exact hooks named in the PSA1 generator preamble;
- replace direct `wid.x` brick ownership in `bakeBrickAggregateDiagonal`,
  `restrictBrickAggregateResidual`, all `refineBrickAggregate*`, and hierarchy
  combine with `psaWetBrickInvocation`;
- replace `pressureHierarchyGroupAddress(wid.x)` in hierarchy diagonal,
  restriction, and refinement with `psaActiveHierarchyNodeAddress`;
- keep the inner loops, accumulation expressions, workgroup reductions, and
  write order byte-for-byte unchanged;
- route PCF/PCM/topology producer events to PSA1; do not add a census shader.

The integrated WGSL checker must generate production and construction-oracle
variants for all six B/P combinations, while PSA itself rejects non-B16/P16
layouts.

### PCF1 coefficient transaction now available to integration

PCF1's fine repair marks a brick aggregate owner whenever a cell diagonal or
internal fine edge changes, and marks the canonical aggregate edge owning each
changed cross-brick fine edge. Aggregate repair preserves the original packed
contribution order and the original 64-lane brick reduction. Changed aggregate
edges map locally to either one hierarchy edge or one internal hierarchy-node
diagonal at every level; changed brick diagonals map to their one parent at
every level. Hierarchy repairs preserve the packed contribution/child/internal
edge order and original workgroup reduction.

The fixed schedule is:

1. `pcfBegin`, followed by the four previous-active-leaf seed indirects;
2. ordinary PCF event collection and fine-leaf repair;
3. `finalizePersistentPressureFineCache`;
4. aggregate brick/edge workset repair, plan, and numerical repair;
5. hierarchy node/edge workset repair, plan, and numerical repair;
6. `finalizePersistentPressureCache` accepts the single generation.

Each family exposes seed, repair, and work byte offsets through
`sparseCM12PersistentPressureCacheAggregateIndirectByteOffset`. Production
rank-select is canonical stable ID; construction-only `qaFullOracle` publishes
full capacities and returns raw stable IDs. Any fault zeros all fine and
aggregate indirect families. The global `bakeBrickAggregateEdges`,
`bakeBrickAggregateDiagonal`, `bakePressureHierarchyEdges`, and
`bakePressureHierarchyDiagonal` dispatches can be removed only after this full
transaction replaces them.

## Expected savings

This slice is aimed at the original 50%-air pressure-domain waste:

- quiescent pressure topology should lose the global brick and hierarchy bakes;
  only PCF dirty leaves and PSA changed ancestors remain;
- every RHS/preconditioner application visits wet bricks and active hierarchy
  nodes rather than the physical atlas/hierarchy capacities;
- after convergence, eligible encoded arithmetic dispatches become zero-sized,
  while journals and final residual authority still run.

The acceptance target is pressure topology below 1 ms on topology-quiescent
ocean B16/P16, and brick/hierarchy solve work proportional to the reported
wet/active counts. This does not promise to remove fine-cell MGPCG work in deep
liquid; PCM correctly retains those liquid cells. Measurements must report
full and accepted brick/node counts and the encoded-versus-executed tail count.

## Standalone validation

`tools/check-sparse-cm12-pressure-solve-authority.ts` validates layout bounds,
indirect offsets, exact CPU membership/arithmetic comparison, first-mismatch
reporting, generated producer APIs, stable rank helpers, and Naga acceptance.
No resident source or GPU runtime is involved.
