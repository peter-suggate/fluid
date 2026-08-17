# Sparse CM12 serial production integration manifest

The executable source of truth is
`sparse-cm12-production-integration-manifest.ts`. It defines the one-way
B16/P16 cutover from the current FCA1/FPL1/FPP1/SAW1 option-B resident to
FPA1, PCF/PCA1, and PSA1. The host always encodes the same command graph.
Counts, generations, acceptance, and fault-zero decisions are GPU authored.
There is no dirty-ratio branch, CPU readback decision, or runtime full-domain
fallback.

## Physical allocation map

All offsets below are absolute `u32` words in the named allocation and are
64-word aligned where an ABI begins.

### `cm12.activity`

The current prefix is deliberately not repacked during this cutover:

| Region | Exact range |
| --- | --- |
| old PCM1 | `[activityPCM.baseWords, activityPCM.totalWords)`; becomes an inert tombstone |
| FPL1 | `[fpl.baseWords, fpl.totalWords)` |
| FPP1 | `[fpp.baseWords, fpp.totalWords)` |
| SCA1 candidate | `[sca.candidateBaseWords, sca.candidateBaseWords + sca.candidateWords)` |
| SCA1 accepted mass | `[sca.acceptedMassBaseWords, sca.totalWords)` |
| FPA1 | `[align64(sca.totalWords), fpa.totalWords)` |

The tombstone preserves every existing FPL/FPP/SCA offset while PCM moves to
its permanent pressure owner. No shader may read the tombstone after the
one-way pressure-cache PCM generation publishes. FPA1's base must equal
`align64(sca.totalWords)` exactly; allowing a caller-supplied gap would make an
accidental second authority harder to detect.

### `cm12.topology-arena` and `cm12.scalar-authority`

FCA1 remains at `[fca.baseWords, fca.totalWords)` after the immutable template,
mutable topology worklists, and page pool. FPA/PCF/PSA do not extend this
arena.

SAW1 remains a dedicated planner allocation. Its control is
`[0, saw.totalWords)`. The copied SCA1 candidate immediately follows at
`[saw.totalWords, saw.totalWords + sca.candidateWords)`. SAW1 has no binding to
physical state; its existing activity copies remain explicit command-encoder
seams.

### `cm12.pressure-cache`

`PressureCache.membership` is the only production pressure-authority tail:

```text
membership.offset / 4
  -> PCM1 baseWords
  -> PCM1 totalWords
  -> PCF1 headerBaseWords == align64(PCM1 totalWords)
  -> PCA1 families within PCF1 control
  -> PCF1 controlEndWords
  -> PSA1 baseWords == align64(PCF1 controlEndWords)
  -> PSA1 totalWords <= membership end
```

PCM is not copied as a raw byte range because its header contains absolute
word offsets. On the first new-buffer generation the ordinary GPU bootstrap
classifies the current cell/row domain into the new layout, builds the same
stable count trees, compares membership/rank against the old activity PCM,
and publishes only on exact equality. Subsequent frames run the same encoded
path with bootstrap `x=0` and local repair. The old bytes then remain an unread
tombstone; they are not a fallback.

PCF maps numerical values to the phase-arena regions `theta`,
`effectiveEdgeWeights`, `brickAggregateEdgeWeights`,
`brickAggregateDiagonal`, `hierarchyN.edgeWeights`, and
`hierarchyN.diagonal`. Only its control, dirty leaves, count trees, and PCA
work authority occupy membership. PSA follows PCF and owns wet-brick and
active-hierarchy bitsets/count trees plus the two tail-indirect banks.

## Phase binding ABI

This extends the phase-arena binding map without aliasing owners:

| Binding | Buffer | Use |
| --- | --- | --- |
| 0 | `cm12.htp1` | immutable read-only topology |
| 1 | `cm12.physics-state` | persistent physical fields |
| 2 | `cm12.scalar-scratch` | phase scratch and Krylov vectors |
| 3 | `cm12.pressure-cache` | PCM/PCF/PCA/PSA and pressure values |
| 8 | `cm12.activity` | FPL/FPP/SCA/FPA scheduling state |
| 9 | `cm12.topology-arena` | FCA and mutable topology control |

Transport binds 0/1/2/8/9. Pressure preparation and solve bind 0/1/2/3/8/9.
Projection binds the pressure cache read-only and activity/topology control
writable. The dedicated SAW planner retains its own two-binding group. This is
intentional: adding FPA to SAW or PCF to activity would recreate the lifetime
alias the phase arenas remove.

## Copy-isolated indirect seams

Storage-authored indirect triplets are never consumed from the storage
allocation. A planner pass ends, the sealed source words are copied, then a
new compute pass consumes a buffer created with `INDIRECT | COPY_DST`.

Existing option-B snapshots remain:

- FPL1: one 72-byte range (`6 * vec3<u32>`);
- FPP1: one 12-byte range;
- FCA1: one 168-byte range (`14 * vec3<u32>`).

New snapshots are packed as follows:

| Buffer | Destination byte offsets | Meaning |
| --- | --- | --- |
| `cm12.fpa1-indirect` | 0, 12, 24, 36, 48, 60 | preparation bootstrap/repair/work, projection bootstrap/repair/work |
| `cm12.pcf1-indirect` | 0 | fine PCF repair |
| same | 12..144 in 12-byte steps | brick, aggregate-edge, hierarchy-node, hierarchy-edge; seed/repair/work for each |
| `cm12.psa1-indirect` | 0, 12, 24, 36, 48 | bootstrap, brick repair/work, node repair/work |
| `cm12.psa1-tail-0-indirect` | 0, 12, 24, 36 | cell, wet brick, hierarchy node, scalar tail A |
| `cm12.psa1-tail-1-indirect` | 0, 12, 24, 36 | the same families for tail B |

FPA previous-active-leaf seeding is a fixed metadata dispatch over its leaf
capacity because FPA1 does not expose a seed triplet. Every invocation checks
the GPU previous-active count; it never launches numerical row work. Bootstrap
still uses its copy-isolated triplet and is zero after initialization.

All sealed, disjoint triplets at the same boundary may be copied in one encoder
copy section. A copy may not cross a planner or acceptance dispatch. In
particular, PCA requires distinct seams for previous-leaf seed, aggregate
workset repair, aggregate numerical work, hierarchy workset repair, and
hierarchy numerical work. PSA requires the brick repair to complete before the
node repair count is copied.

## Serial schedule and safe pass coalescing

The manifest records a dependency DAG rather than relying on host source
order. Its pressure transaction is:

1. accept current PCM cell/row membership and theta;
2. begin PCF/PCA and seed previous active leaves;
3. collect PCM/topology/solid events, repair fine leaves, and seal fine work;
4. repair aggregate worksets, then execute aggregate edges before brick
   diagonals using their original kernels;
5. derive hierarchy worksets, then execute hierarchy edges before hierarchy
   diagonals and accept PCF once;
6. begin PSA, repair wet-brick leaves, mark all hierarchy parents, repair node
   leaves, validate the `(frame, topology, PCM, PCF)` tuple, and accept;
7. copy wet-brick/node work triplets and run the unchanged stable-rank pressure
   bodies;
8. inside the encoded solve loop, publish tail A, copy/consume it, run the
   unchanged gate reduction, then publish/copy/consume tail B. Journal records,
   snapshots, final true residuals, gate publishers, and unconditional clears
   never use a tail indirect.

Preparation is sequenced after velocity validity and before pressure. Projection
opens only after preparation, PCM, PCF, and the pressure solve have accepted.
It restores the separate prepared face word, executes the unchanged projection
body, mirrors the exact result to both banks, verifies execution coverage, and
accepts before collocation.

The following are safe within one compute pass while retaining distinct
dispatches: direct begin/mark/finalize dispatches with no intervening copy;
producer-local dirty marking in the invocation that owns the physical output;
and stable-rank translation around an otherwise unchanged numerical body.
PCF effective-edge repair and `preparePressure` may share one cell invocation
only in their existing expression order, as already proven by the phase
migration contract.

The following may not become one dispatch: FPA preparation/projection; PCM
cell and row/theta publication; PCF fine and aggregate numerical repair; PCA
aggregate and hierarchy repair; PSA wet-brick and node repair; any Krylov
update/reduction/gate boundary; a tail publisher and its gate-writing
reduction; projection and collocation; or topology commit and presentation
publication. A compute-pass boundary may also be forced by an indirect copy;
removing it is invalid even if a backend appears to make storage writes visible.

## Fail-closed receipts

The executable authority table requires every authority to declare a nonempty
producer set, bounded closure, acceptance receipt, and complete list of
indirect families zeroed on fault. It covers:

- PCM pressure-cache one-way cutover;
- FPA preparation and projection separately;
- PCF fine edges/diagonals;
- PCA aggregate and hierarchy coefficients;
- PSA execution domains;
- PSA converged tail publication.

Generation, capacity, provenance, or execution coverage failure leaves the
candidate unaccepted and publishes zero for every dependent indirect. It does
not select a full traversal. A missing aggregate topology reverse map is a
construction error; an aggregate graph generation change after bootstrap is a
local transaction fault until dual old/new provenance is implemented.

## Construction-only QA

Production constructs FPA/PCF/PSA with `qaFullOracle=false`. The manifest
rejects any production layout with that flag enabled. QA constructs a separate
solver whose immutable pipeline specialization is `true`; no runtime word,
URL state, ratio, readback, or fault can enable it.

The paired oracle compares physical fields plus direct pressure authority:
density, gamma, cell velocity, both face banks, prepared face words, pressure,
divergence, PCM membership/theta, PCF fine/aggregate/hierarchy coefficients and
diagonals, RHS, and PSA wet-brick/node membership. Topology generation,
allocator layout, dirty worksets, and queue order remain diagnostic only.

## Serial gates

No later slice starts until the prior slice passes its static, dam, and ocean
gate without threshold changes.

| Slice | Dam gate | Ocean gate |
| --- | --- | --- |
| FPA preparation | two independent five-step hashes, then paired 60-step/2 s exactness; zero coverage faults | 24 measured frames, strict face-preparation improvement, no full-row numerical dispatch |
| FPA projection | paired 60-step physical and pressure-authority exactness | combined face preparation + projection p95 below 1 ms |
| PCF fine | edge/theta/diagonal/RHS bytes equal the full construction oracle through 2 s | quiescent repair proportional to dirty leaves |
| PCA aggregate/hierarchy | all aggregate/hierarchy coefficient and diagonal bytes exact through topology bursts | pressure-topology p95 below 1 ms and zero global aggregate/hierarchy bakes |
| PSA domains | wet-brick/node sets and physical bytes exact through 2 s | dispatched brick/node work equals accepted counts; no empty hierarchy work |
| PSA tails | encoded arithmetic/order and physical bytes exact; journals and final residual remain present | encoded/executed tail receipts present and total non-pressure p95 below 10 ms |

Each slice first runs TypeScript, focused lint, its CPU/Naga contract, the
integrated six-variant Dawn checker, and the manifest checker in resident mode.
The five-step paired run precedes the 60-step run.

## Static enforcement

Run:

```sh
node --import tsx tools/check-sparse-cm12-production-integration-manifest.ts
```

This constructs a complete synthetic option-B layout and rejects overlaps,
misordered PCM/PCF/PSA placement, a noncontiguous FPA tail, missing producer /
closure / receipt / fault-zero declarations, uncopied indirect consumers,
unsealed copies, destination aliasing, dependency cycles, production oracle
flags, or omitted per-slice gates.

During serial resident integration run the stricter form:

```sh
node --import tsx tools/check-sparse-cm12-production-integration-manifest.ts --resident
```

That additionally requires every named producer, closure, receipt/finalizer,
tail publisher, layout constructor, and indirect allocation token in the
resident host/WGSL sources. It is expected to fail before the cutover lands;
that failure is the integration checklist, not a reason to weaken the checker.

