# SRR1 producer-authored scalar result authority (B16/P16)

## Decision

SAW mass promotion must use exact producer-authored result receipts, not a
geometric dirty radius or a constant-state classifier.  The ocean census found
104,798 exact-support cells, but the required mass operator closure connected
those islands to non-constant/interface cells and left zero clean tiles.  More
dilation tuning cannot establish exact temporal equivalence.

SRR1 instead asks one exact question for each tile: do **both physical scalar
banks already contain the bit-identical HEAD mass result for every covered
word**, under the currently certified dependency and topology generations?  A
tile can become clean only when the physical mass producer answers yes after a
full word comparison. SRR1 never reads, writes, repairs, or canonicalizes a
physics buffer.

## Frame sequence

1. FCA begins candidate generation `G`; construction generation 1 dispatches
   every tile exactly once. Later frames retain accepted work/clean trees.
2. Scalar, velocity, topology, gamma, surface, and solid producers append only
   tiles whose authored result or dependency receipt changed. A generation CAS
   deduplicates tile leaves. Appending immediately makes that tile work.
3. Seal snapshots the persistent work-tree root and writes the canonical
   rank-selected work list. Fixed indirect dispatch consumes this list. There is
   no host count branch.
4. The existing mass producer executes each work tile. In the same invocation,
   after producing HEAD output, it compares all declared words in physical bank
   0 and bank 1 with the HEAD output and writes mismatch counts into the
   candidate-receipt bank. This is a fused epilogue, not another field traversal.
5. Commit first proves that every scheduled tile authored exactly one candidate
   receipt. It then promotes candidate receipts in work-list order, changing only
   exact tiles to clean in the persistent count trees. Finally FCA may publish
   the candidate scalar bank/parity.
6. If any receipt is missing or duplicated, candidate publication is rejected,
   fixed indirect X becomes zero, and the previous accepted generation remains
   authoritative. Candidate receipts never overwrite accepted receipts, so no
   rollback exists. The persistent affected work set retries next frame; there
   is no global full-path fallback.

## Arena and access map

All records are immutable in size and B16/P16-only. Exact offsets are emitted by
`sparseCM12ScalarResultByteMap()` for the chosen tile capacity.

| Region | Record | Ordering | Hot access |
| --- | ---: | --- | --- |
| control | 128-byte header in a 256-byte region | singleton | one invocation at begin/seal/commit |
| accepted tile receipts | 64 bytes/tile | logical tile id | candidate invalidation, overlay, commit |
| candidate receipts | 64 bytes/tile | logical tile id | contiguous producer epilogue, commit |
| candidate leaves | 16 bytes/leaf | generation-stamped append evidence | changed producers, candidate repair |
| work list | 4 bytes/tile | canonical work rank | mass producer, commit |
| work count tree | 4 bytes/node | binary heap over tile id | delta repair, rank-select |
| clean count tree | 4 bytes/node | binary heap over tile id | delta repair, diagnostics |

The two 64-byte tile records align one record to a cache line. Heavy mass work
reads only the packed work list and existing physics/topology bindings; receipt
construction is an epilogue with one contiguous candidate-record write per tile.

## Atomics and ordering

- Candidate append publishes through a per-tile generation lock, clears the
  prior epoch's cause, and allocates one leaf only for the winning producer.
  Same-generation duplicates only OR diagnostic causes.
- A clean/work transition uses one leaf exchange and one `atomicAdd`/`atomicSub`
  along each binary-tree parent path. Heavy mass arithmetic has no SRR1 atomics.
- Work dispatch and promotion use a sealed canonical list. Producer invocation
  order is irrelevant; commit order is deterministic.
- The candidate receipt generation provides the producer-completion proof.
  An atomic executed count is only a fast aggregate; commit also checks every
  scheduled tile's generation, so a rogue duplicate cannot hide a missing tile.
- Required barriers are phase boundaries only: producer invalidations → seal,
  seal/work-list → mass, final gamma/surface scalar publication → receipt,
  receipt → commit, commit → FCA publication.

## Physical correctness invariants

- Canonical checks are `u32` word comparisons. `+0` and `-0`, NaN payloads, and
  all subnormal payloads remain distinct. A clean packet is exact dry or flooded
  capacity in both physical rho banks, exact one in both gamma banks, and full
  accepted/open six-sided same-phase support.
- `coveredWords == expectedWords`; a coarse-rung stable tile with no uniquely
  owned cell is an explicit exact `0/0` empty packet, while partial edge tiles
  carry their own exact expected count.
- Bank receipts name physical bank 0 and bank 1. Scalar parity changes roles but
  cannot invalidate an already exact dual-bank result.
- Dependency current generation must equal its producer-certified generation.
- Topology is local per tile. A topology commit appends old and new owner spans,
  incidence receivers, and activation/retirement identities; unrelated tiles
  retain accepted receipts.
- A candidate leaf is always work until a physical producer executes. Speculative
  topology can never recertify an old receipt clean.
- Immutable full QA remains construction-only. Runtime faults reject publication
  and retry the persistent local work set; they never select the QA path.

## Dirty visualization contract

Expose SRR1 beside Grid Structure using the existing dirty-visualization catalog.
The overlay reads GPU buffers only and offers:

- persistent work versus exact-clean tree leaves;
- current-generation candidate leaves colored by producer cause;
- receipt state: missing execution, partial coverage, bank-0 mismatch, bank-1
  mismatch, dependency mismatch, or exact;
- accepted/candidate generation and topology stamp;
- scheduled versus executed work counts and first rejected tile.

Cause display must match the current-generation leaf or tile stamp. A new stamp
clears prior provenance before publishing its generation.

## Resident cutover slices

Each slice retains the current full mass invocation as an immutable paired oracle
until its byte gate passes. No slice changes physics arithmetic.

1. In a private integration epoch, allocate SRR1, bootstrap all work, and prove
   the comparator observational while the physical path remains unchanged.
2. Before handing the production resident back, connect all producer events and
   switch trace/scatter/gather to the sealed HTP1 tile-cell packets in one cut.
   Delete the legacy global classifier/count/scatter planner, accepted-domain
   mass dispatch tokens, velocity/classifier scan, closure dilations, and global
   next-candidate publication together; no merged runtime intermediate retains
   any of them.
3. Preserve a separately constructed immutable full-oracle solver that dispatches
   the full identical kernel. Require
   density, velocity, pressure, and divergence hashes at dam5 and canonical60.
4. Measure ocean B16/P16 planner, mass, clean/work/candidate counts, and combined
   non-pressure p95. Optimize only producer fusion/tree delta traffic; never move
   a full scalar scan into the planner.
5. Replicate the same receipt transaction for gamma, then surface. Each stage has
   its own result/dependency generations and work tree; shared producer epilogues
   may coalesce candidate appends, but stage publication remains independently
   receipted and hash-gated.

Standalone implementation and contracts:

- `lib/methods/adaptive-mass/sparse-cm12-scalar-result-receipts.ts`
- `lib/methods/adaptive-mass/sparse-cm12-scalar-result-receipts.wgsl.ts`
- `tools/check-sparse-cm12-scalar-result-receipts.ts`
- `tools/check-sparse-cm12-scalar-result-receipts-wgsl.ts`
