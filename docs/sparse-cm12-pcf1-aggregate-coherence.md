# PCF1 persistent aggregate and hierarchy coefficients

PCF1 now carries one pressure-coefficient transaction from changed HTP1
cells/rows through persistent fine edges and diagonals, brick aggregates, and
every affected hierarchy ancestor. It removes the four recurring global bakes
without adding a global runtime recovery path.

## Persistent values and work authority

The numerical values use the existing non-aliased `cm12.pressure-cache`
regions:

- `effectiveEdgeWeights` and PCF fine diagonal;
- `brickAggregateEdgeWeights` and `brickAggregateDiagonal`;
- `hierarchyN.edgeWeights` and `hierarchyN.diagonal`.

The membership tail contains the original PCF header/fine dirty leaves followed
by a `PCA1` extension. PCA1 owns four ephemeral work authorities: aggregate
brick, aggregate edge, flattened hierarchy node, and flattened hierarchy edge.
Each has generation tokens, 256-ID bitset leaves, generation-stamped dirty leaf
queues, a 32-way count tree, an active-leaf list, and GPU-authored seed/repair/
work indirect triplets. Rank select is stable-ID order; queue append order never
selects numerical order.

Previous active leaves are seeded into the next dirty set so old work bits are
locally cleared. A quiescent frame therefore repairs only those prior leaves,
not the domain. Construction-only `qaFullOracle` publishes physical capacities
and returns raw stable IDs.

## Exact closure

Fine repair records an aggregate brick when a cell diagonal or internal fine
edge changes. A changed cross-brick directed edge also records its immutable
aggregate-edge owner. Aggregate repair uses the original packed fine-edge
contribution order and the original 64-lane brick reduction.

For every changed aggregate edge and every hierarchy level, immutable topology
identifies exactly one of:

- a hierarchy edge containing that aggregate edge; or
- the hierarchy node for which that edge is internal.

A changed brick diagonal records its one parent node at every level. Hierarchy
edge repair retains packed aggregate contribution order. Hierarchy diagonal
repair retains the original child-then-internal-edge lane order and 64-lane
reduction. No scalar substitute is used for a workgroup reduction.

The immutable aggregate topology must prove that every brick appears exactly
once in each level's children and every aggregate edge appears exactly once as
internal or cross-group at every level. CPU construction rejects gaps,
duplicates, and parent mismatches.

Mutable brick resolution/occupancy is represented by PCM/topology events over
that immutable physical graph. If the packed aggregate graph generation itself
changes after bootstrap, PCF1 fails closed: repairing old and new reverse maps
would require explicit dual provenance, which this ABI intentionally does not
guess or replace with a full bake.

## Transaction and failure behavior

The GPU sequence is fixed:

1. begin PCF and snapshot topology, PCM, and aggregate-topology generations;
2. seed four previous-active-leaf lists;
3. collect complete fine events, repair fine leaves;
4. repair/plan aggregate worksets, then repair aggregate edges and bricks;
5. repair/plan hierarchy worksets, then repair hierarchy edges and nodes;
6. validate execution counts and unchanged generations; accept once.

Producer coverage remains the original exact PCF expected/covered event count.
Derived owner and ancestor work is internal to that covered transaction. A
header, capacity, topology, count-tree, execution, or generation fault zeros
all fine and aggregate indirect families and leaves the candidate unaccepted.
There is no dirty-ratio switch or global bake fallback.

## Resident integration API

Generate helpers with `createSparseCM12PersistentPressureCacheWGSL`; it now
includes the aggregate source. Provide the immutable mapping hooks listed in
the generated preamble. Obtain copy offsets with
`sparseCM12PersistentPressureCacheAggregateIndirectByteOffset(layout, family,
kind)` for `kind` `seed`, `repair`, or `work`.

Replace, in order:

- global `bakeBrickAggregateEdges` with
  `repairPersistentPressureAggregateEdges`;
- global `bakeBrickAggregateDiagonal` with
  `repairPersistentPressureBrickDiagonals`;
- global `bakePressureHierarchyEdges` with
  `repairPersistentPressureHierarchyEdges`;
- global `bakePressureHierarchyDiagonal` with
  `repairPersistentPressureHierarchyDiagonals`.

The inner arithmetic and packed topology order in those replacements is the
existing HEAD arithmetic. PSA1 subsequently uses accepted wet-brick/active-node
rank for recurring RHS and solve work; PCF1 alone owns coefficient validity.

## QA contract

`bakeSparseCM12PressureAggregateAuthorityQA` emulates f32 additions, lane
assignment, reduction order, and `max(diagonal, 1e-12)` exactly. The checker
performs local-vs-full comparison for fine edges, fine diagonals, RHS, aggregate
edges, brick diagonals, hierarchy edges, and hierarchy diagonals; it also
removes an aggregate owner deliberately and requires the first mismatch to be
reported. Naga compiles both production and construction-oracle variants.
