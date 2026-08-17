# Sparse CM12 A4D2 production activity cutover

## Acceptance boundary

`A4D2` replaces ACT1's per-cell exact-bit atomic closure and unordered atomic
brick append. It does not change density, gamma, velocity, pressure, topology,
or the current activity score/history formulas. The old `measureBrickActivity`
may be constructed as a QA oracle during cutover, but it is never a runtime
fallback and its output is never selected after an A4D2 fault.

The ocean B16/P16 gate is **median and p95 below 0.5 ms** for the complete
activity seam: trigger classification, deterministic compaction, dirty-tile
rebuild, brick re-reduction, and local census commit. Static tools do not claim
that performance result; it must be measured after resident integration.

## Resident arena map

Append `SparseCM12ProductionActivityLayout` after the current FPP1 tail in the
binding-12 activity arena. Do not insert it between dirty scheduler, ACT1, PCM,
FPL1, or FPP1 while the serial cutover is being compared. After A4D2 is
accepted, ACT1 storage can be removed in a separate mechanical change.

The arena is 256-byte aligned and consists of:

1. a 64-word fail-closed header and indirect packets;
2. one 16-byte producer-owned trigger per physical-brick 4^3 tile;
3. one persistent 16-word exact summary per tile;
4. one 16-word aggregate/receipt per physical brick;
5. one-word brick dirty flags and deterministic prefix/list arrays;
6. at most 256 compaction block sums/prefixes;
7. 260 signed census-delta words per dirty-list block and 256 accepted score
   bins.

B4, B8, and B16 use 1, 8, and 64 valid tile lanes respectively. All variants
launch a 64-lane brick workgroup; invalid lanes reduce zero. The B16/P16 path
therefore has exactly one lane per 4^3 tile and no tile list or per-tile global
atomic journal.

## Producer contract

FCA/FPL first seals one strictly increasing activity-candidate brick packet:
the deterministic union of bricks named by its stage/dependency packets. Its
generation, count, list, and indirect triplets are GPU-owned. Add
`buildProductionActivityTriggers`, one workgroup per brick in that packet.
Each lane owns and overwrites its one four-word trigger record for the FCA
candidate generation. It gathers already sealed producer metadata; it does
not scan all bricks, accepted cells, or rows. The four words are:

| Word | Meaning |
|---:|---|
| 0 | FCA candidate generation |
| 1 | origin/inherited cause mask |
| 2 | direct FPL stages 0..5, closure FPL stages 0..5, four-bit closure depth, high-bit exact decision certificate |
| 3 | accepted physical-brick topology signature |

The gather sources at production cutover are:

| Activity dependency | Sealed source | Local expansion |
|---|---|---|
| density, gamma, surface, sibling detail | SAW stage tile records | SAW's exact dependency closure |
| projected/collocated velocity and swept support | face/VEX packet tile masks | owner tile plus accepted incidence ring |
| face velocity at an interface | face packet row-to-owner masks | incident owner tiles |
| solid openness/boundary | FCA/face authority receipts | incident owner tiles |
| activation, retirement, rung/ownership change | FCA topology-change brick list | all valid tiles of only changed physical bricks |

No mass, gamma, surface, face, or pressure kernel calls an A4D2 atomic helper.
Missing generation, missing required source receipt, or a topology signature
change without a full valid-tile mask is an uncovered-write fault. Bootstrap is
an explicit FCA generation in which active bricks publish all valid tile bits;
it is not a fallback. A large topology edit makes many *changed bricks* locally
full, but never selects a domain cell scan.

Every current-generation trigger, including a clean one, must carry all A4D2
certificate bits: exact fixed moments; exact metric score classes; feature
threshold predicates; reason predicates; support/swept masks; local topology;
and activity-policy generation. This is what replaces the exact-bit velocity
predicate. In particular, a uniformly full or empty tile under changing
nonzero solenoidal velocity is reusable when its *activity decision
projection* is unchanged. Raw velocity bits are not part of that projection;
velocity-floor class, nonzero-deformation reason, score bin, feature-hot/detail
thresholds, and swept support are. Crossing any one of those boundaries marks
the local tile. A dt or activity-policy change that alters the partition also
marks it through the policy-generation certificate.

Those projection words are reduced by the producer's existing brick/tile
workgroup and written once by the tile lane. Heavy per-cell consumers never OR
into A4D2. If an existing producer cannot yet publish all certificate bits,
the tile is unknown and the candidate frame faults; it is not conservatively
sent through a global measurement.

## Exact tile evaluator

The resident hook `cm12ActivityRebuildExactTile(brick,tile)` is the body of the
current `measureBrickActivity` owner loop, moved without algebraic rewrites.
One tile lane enumerates its 64 finest positions and evaluates a resolved cell
only when that position equals `cellMinimum(cell)`. Thus a macro cell
contributes once, in the tile containing its finest-coordinate minimum.

The persistent summary stores the accepted decomposable quantities:

- i32 density and normalized x/y/z moment sums using `ACTIVITY_FIXED` and the
  current per-owner rounding points;
- f32 maxima for deformation, predicted motion, sibling detail, and velocity
  travel;
- ORs of activity flags, support, and swept-support masks;
- exact accepted-owner contribution count, cause/depth, and a validity check.

Do not pre-sum, reassociate, or approximate the per-owner f32 expressions.
Only their maxima are reduced. Integer addition, maximum, and bitwise OR are
decomposition-exact, so the brick aggregate is independent of the tile
partition. A clean tile may retain a prior continuous maximum only after its
producer proves the complete decision projection above unchanged; monotonic
maximum plus identical score/threshold classes then produces identical public
activity words. Any later density/topology/policy root rebuilds the full exact
summary before use. The brick hook `cm12ActivityPublishExactBrick` must retain the
existing score, reason, temporal moments, hot/quiet history, receiver support,
and velocity-floor code byte-for-byte. Its QA gate compares all existing
activity record words, the 256-bin histogram, and scene census against the old
kernel before ACT1 is removed.

At construction, seed the A4D2 brick score/reason/history flags from the
existing accepted activity records. The explicit bootstrap packet rebuilds
all active tile summaries, while the brick publication hook reads the existing
accepted mean/moment words for the first temporal comparison. Unchanged-brick
hot/quiet aging moves into `planBrickResolution`, which already visits those
bricks; this preserves the current topology-epoch semantics without adding an
activity scan.

## GPU schedule and exact edit anchors

Keep physics scheduling untouched. At the current ACT1 activity seam in
`webgpu-sparse-cm12-resident.ts` (pipeline names near the existing
`beginIncrementalActivity`/`measureBrickActivity` group and dispatches near the
activity block), replace only that seam with:

1. `beginProductionActivity` (1);
2. producer-owned trigger gather, indirectly over the sorted candidate packet;
3. `classifyProductionActivityBricks`, indirectly over that same packet;
4. `scanProductionActivityBrickBlocks`, indirectly over candidate blocks;
5. `scanProductionActivityBlockSums` (1);
6. `scatterProductionActivityBricks`, indirectly over candidate blocks;
7. copy the GPU-authored rebuild and census triplets to dedicated indirect
   buffers, preserving the project's storage/indirect usage seam;
8. `rebuildProductionActivityBricks` indirectly over sorted dirty bricks;
9. `reduceProductionActivityCensusBlocks` indirectly;
10. `commitProductionActivityCensus` (1), then `acceptProductionActivity` (1).

The classifier and prefix kernels visit only the local candidate packet.
Fine samples and incidence are visited solely in set tile lanes. There is no
CPU page count, parity, list construction, readback, all-brick/all-cell scan,
or global recovery dispatch. Bootstrap is a distinct FCA packet containing
all initially active bricks; it is not selected as a recovery mode.

The WGSL generator call should be appended after the existing FPL/FPP helpers
and receive `productionActivityLayout`; do not reorder FCA, PCM, pressure, SAW,
or FPL generator arguments. The resident hooks named by the standalone
generator are the only semantic integration surface.

Current read-only anchor map (symbol is authoritative if line numbers move):

| File / current line | Serial cutover edit |
|---|---|
| `webgpu-sparse-cm12-resident.ts:55` ACT1 imports | add A4D2 layout/initializer/generator imports; retain ACT1 import for QA construction mode |
| `webgpu-sparse-cm12-resident.ts:2178` `createSparseCM12IncrementalActivityLayout` | append A4D2 after the final current FPP/activity tail; do not shift FCA topology-arena offsets |
| `webgpu-sparse-cm12-resident.ts:2511` shader-generator arguments | append `productionActivityLayout` after current activity/FPL/FPP arguments without reordering pressure/FCA arguments |
| `webgpu-sparse-cm12-resident.ts:2575` activity pipeline names | add the ten A4D2 entry points; keep `measureBrickActivity` only when QA construction mode is enabled |
| `webgpu-sparse-cm12-resident.ts:3109` ACT1 schedule | replace only begin/mark/finalize/measure/census with the ten-step GPU schedule above |
| `webgpu-sparse-cm12-resident.ts:3180` post-topology mark | remove ACT1 post-topology atomics; FCA next candidate packet owns activation/retirement roots |
| `webgpu-sparse-cm12-resident.wgsl.ts:24` ACT1 generator import | add standalone A4D2 generator import |
| `webgpu-sparse-cm12-resident.wgsl.ts:340` generator signature and `:378` helper composition | append the optional A4D2 layout and generated source; preserve existing argument order |
| `webgpu-sparse-cm12-resident.wgsl.ts:3228` `measureBrickActivity` | extract owner-loop expressions into `cm12ActivityRebuildExactTile` and score/history publication into `cm12ActivityPublishExactBrick`; do not rewrite arithmetic |
| `webgpu-sparse-cm12-resident.wgsl.ts:4510` FPL population | supply `cm12ActivityPublishFramePlanRoot` against FPL Next; root publication must not set executed/skipped |

Required binding-free WGSL hook ABI:

| Hook | Result / authority |
|---|---|
| `cm12ActivityCandidateGeneration()` | FCA candidate generation |
| `cm12ActivityCandidateBrickCount()` | GPU-sealed local packet count |
| `cm12ActivityCandidateListGeneration()` | generation of that packet |
| `cm12ActivityCandidateBrickInvocation(i)` | strictly increasing physical brick id |
| `cm12ActivityTopologyGeneration()` | accepted global topology epoch for the header receipt |
| `cm12ActivityFramePlanGeneration()` | FPL Next generation |
| `cm12ActivityBrickTopologySignature(brick)` | local accepted active/rung/span/owner signature |
| `cm12ActivityBrickTopologyChanged(brick)` | FCA local change predicate |
| `cm12ActivityBuildTileTrigger(brick,tile)` | `vec4u(generation,cause,packed certificate,local topology signature)` |
| `cm12ActivityRebuildExactTile(brick,tile)` | exact `A4D2TileSummary` |
| `cm12ActivityExpectedTileContributionCount(brick,tile)` | O(1) count from accepted rung/tile alignment |
| `cm12ActivityExpectedTileCheck(brick,tile)` | O(1) local topology/owner validity word |
| `cm12ActivityPublishFramePlanRoot(...)` | true only after exact FPL Next tile-root publication |
| `cm12ActivityPublishExactBrick(...)` | existing score/reason/history/output publication; returns A4D2 brick words |

## FPL truthfulness audit

A4D2 publishes dependency roots into **FPL Next**. It never claims that a heavy
stage executed. The stage owner remains responsible for executed/skipped and
consumer-generation receipts:

| FPL stage | Root/direct/closure authority | Execution authority |
|---:|---|---|
| 0 Face preparation | FCA + VEX/face tile packet | face packet seal |
| 1 Mass transport | SAW mass dependencies | SAW mass seal |
| 2 Gamma transport | SAW gamma dependencies | SAW gamma seal |
| 3 Surface conditioning | SAW surface dependencies | SAW surface seal |
| 4 Pressure coefficients | HTP/FPA/PCF coefficient dependencies | PCF coefficient seal |
| 5 Presentation | density/phase/topology page causes | FPP commit receipt |

Until a row in that table is live, its valid accepted tiles must be published
truthfully as direct+executed by the existing global stage, not silently
skipped. A4D2 root receipts cannot be used to infer execution. Generation,
topology signature, packet epoch, stage mask, direct/closure depth, and tile
coverage must all agree before FPL candidate acceptance; otherwise the Grid
selector remains magenta through the existing local fault path.

## Fail-closed gates

Reject A4D2 candidate acceptance on invalid/stale trigger generation, missing
FPL receipt, prefix/list disagreement, stale clean tile after topology change,
contribution mismatch, compaction capacity, census underflow, histogram/active
count mismatch, or producer/consumer/FCA/FPL generation disagreement. Report
the first brick/tile/code and leave the previous accepted activity generation
visible. Never dispatch ACT1 or a global activity measurement in response.

Static validation commands:

```sh
node --import tsx tools/report-sparse-cm12-production-activity-contract.ts
node --import tsx tools/check-sparse-cm12-production-activity-wgsl.ts
```

Resident acceptance then requires the existing dam-front and weakened
symmetric-expansion regressions, an exact ACT1 shadow comparison, and the ocean
B16/P16 timing gate. No threshold widening is permitted.
