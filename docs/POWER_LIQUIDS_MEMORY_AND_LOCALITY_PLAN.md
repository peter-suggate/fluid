# Power Liquids — memory, locality and per-item cost (2026-08-03)

The third axis of the perf program. Bets 1–3 attacked **launch shape** and the
result is now settled: `docs/POWER_LIQUIDS_WORK_AND_DATA_20260803.md` §4.3
records three independent confirmations that deleting dispatches, copies and
bytes moves neither the wall nor the pass count. Its learning #1 — "launch count
is not the wall; the program has been optimizing a proxy" — is the premise of
this doc.

This axis asks a different question: not *how many* launches, but **what each
launched thread has to touch to do its work.** Three sub-questions, which turn
out to have one root cause:

1. GPU memory footprint and the data structures that set it.
2. Shader code quality inside the kernels that dominate the frame.
3. Scattered access / pointer chasing, and whether a layered lookup structure
   would fix it.

## TL;DR

**The solver's *iteration* is live-count-shaped. Its *addressing* is
domain-shaped and deliberately scrambled.** Bet 1 succeeded at the loop bounds
and never reached the memory system. Every live-count-shaped loop dereferences a
hash slot into an arena sized by the container, at an index computed by a
full-avalanche mixer whose design goal is to destroy the spatial correlation the
cache needs. So the work is `O(live)` while the memory system behaves as
`O(domain)`, and the frame is latency-bound on dependent random gathers issued
from a single workgroup that has no occupancy to hide them.

That one sentence explains all three sub-questions, and it is consistent with
the already-recorded diagnosis that the 84.9 ms solve is **latency, not
bandwidth** (`WORK_AND_DATA` §4.1: one sweep moves ~295 KB, ~0.12 ms at M1 Max
bandwidth, against 84.9 ms measured — a ~700× gap that bandwidth cannot explain
and dependent DRAM round-trips can).

Four measured facts, all new:

| | |
|---|---|
| Resident GPU memory, `power-droplet-64`, **100 cells of water** | **229.5 MiB** |
| Resident GPU memory, `power-droplet-256`, **the same 100 cells** | **895.9 MiB** |
| SPGrid level-0 arena occupancy, `large-power-dam-break` | **1,289 / 262,144 = 0.49%** |
| Live solve working set vs. allocated solve state, same lane | **333 KB vs ~31 MiB** |

The droplet family holds the fluid fixed and sweeps the container, so **every
byte of the 229 → 896 MiB growth is container, not fluid.** Commit `65b2427`
removed the domain tax from the *launch shapes*; it is still fully present in
the *footprint*.

## 0. The instrument (new, landed)

There was no GPU memory census in the tree. `GPUCommandAudit` looks like one —
it has `bufferAllocationsByLabel` — but `webgpu-smoke-executor.ts` calls
`commandAudit.reset()` immediately before the measured window, deliberately, so
that audit reports only what a *recurring advance* allocates. The resident set is
allocated during construction, entirely inside the window that reset discards.
**Nothing in the tree had ever reported resident GPU memory.**

`GPUResidentMemoryCensus` (`tools/webgpu-smoke-gpu-audits.ts`) is cumulative from
device creation, never reset, tracks `destroy()` so a grow-and-replace capacity
path is scored on what it holds, and attributes every buffer to the `lib/` module
that asked for it by capturing a stack per `createBuffer`. Enable with
`FLUID_GPU_MEMORY_CENSUS=1`; it emits `phase: "gpu-memory"` at
`when: "after-construction"` and `when: "after-run"` (a difference between them
is a solver growing arenas while stepping).

`benchmark-power-dam.ts` forwards the record unconditionally. That harness
consumes the child's stdout and discards what it does not re-print — the trap its
own comment documents ("a verdict that only exists inside a pipe is not
evidence"), and which silently ate the first two runs of this census.

A CPU-only analytic model of the SPGrid plan predicted `power-droplet-64` at
87.23 MiB against a measured 87.74 and `power-droplet-256` at 176.86 against
177.36 — **0.6% and 0.3%**. The model can therefore size any lane without a GPU
run.

## 1. Where the 896 MiB is (`power-droplet-256`, 100 live cells)

```
resident 895.89 MiB across 307 buffers
   259.91  lib/webgpu-octree-fine-levelset-topology.ts
   177.36  lib/webgpu-octree-spgrid-vcycle.ts
   139.19  lib/webgpu-octree-fine-levelset-bricks.ts
   129.88  lib/webgpu-octree-owner-pages.ts
    65.14  lib/webgpu-octree.ts
    ... 20 more modules, 124.41 MiB combined
```

Ranked by buffer, with the sizing expression that produces it:

| MiB | buffer | shape |
|---:|---|---|
| 128.75 | `Simulation octree owner pages` | `plan.allocatedBytes`, domain-derived |
| 128.50 | `fine-levelset sparse topology scan` (2×) | `max(sparseCandidateCapacity, logicalBrickCount)` |
| 128.50 | `fine-levelset direct identity mask …` (2×) | `max(maximumResidentBricks, logicalBrickCount)` |
| 128.88 | `… worklist and direct page directory generation 0/1` | per-generation, domain-shaped |
| **64.00** | **`Structured ceiling separation mask`** | **`nx*ny*nz*4` — one u32 per finest cell** |
| 123.00 | `SPGrid native sparse topology` + `immutable candidate topology` | `topologyBytes`, ×2 |
| 53.86 | `SPGrid six-face stencils and vectors` + candidate copy | `26 * totalLevelSlots * 4`, ×2 |

Growth from `power-droplet-64` to `power-droplet-256` — domain ×64, **fluid
constant** — names the domain-shaped terms exactly:

| module | 64³ | 256³ | factor |
|---|---:|---:|---:|
| `owner-pages` | 2.03 | 129.88 | **64×** (exactly the domain) |
| `webgpu-octree` (the ceiling mask) | 1.62 | 65.14 | **40×** |
| `fine-levelset-topology` | 6.92 | 259.91 | **37.6×** |
| `fine-levelset-bricks` | 13.19 | 139.19 | 10.6× |
| `spgrid-vcycle` | 87.74 | 177.36 | 2.0× |

`fine-levelset-topology.ts:1024` already states the occupancy in its own comment:
*"The logical fine-brick lattice is a uniform occupancy grid: 16.7M keys at a
256-cubed container against ~565 live bricks (**0.003%**)."* That comment
introduces a per-256-key block-skip so the recurring **passes** can avoid
streaming the mask. The **allocation** was never compacted — the 67 MB mask is
still resident, twice.

## 2. The locality mechanism, and why it is the wall

### 2.1 The slot index is a full-avalanche hash

`webgpu-octree-spgrid-vcycle.ts:3699`:

```wgsl
fn insertionHash(key:u32,l:u32)->u32{
  var h=key*0x9e3779b1u; h=(h^(h>>16u))*0x7feb352du;
  return (h^(h>>15u))&(levelCapacity(l)-1u);
}
```

That is a splitmix-class mixer. Its *purpose* is to make adjacent keys land at
maximally uncorrelated slots — which is the right property for open-address
probing and the exactly wrong property for everything downstream that reads it.

### 2.2 The smoother's loop is compact; its addressing is not

`webgpu-octree-persistent-mgpcg.wgsl.ts:956`:

```wgsl
for(var i=lane;i<n;i+=LANES){ let slot=workSlot(l,i); ... applied(l,slot,source) ... }
```

The iteration is over `n = count(l)` — the **live** count. Bet 1 is honoured. But
`workSlot` returns the **hash slot**, and every access downstream is addressed by
it:

- `at(c,l,s) = c*totalLevelSlots() + levelBase(l) + s` (`:594`) — channel-major.
- Adjacent lanes in a 32-wide SIMD group get uncorrelated `slot` values, so a
  wavefront that could have touched **one** 128-byte cache line touches **32**.
- `applied()` then does, per row: 18 strided coefficient loads, 18 `topology[]`
  neighbour-slot loads, and **18 dependent gathers** `loadf(source,l,other)` at
  an arbitrary `other`. The second load cannot issue until the first returns —
  a true pointer chase, ~58 memory operations per row per sweep.
- All of it inside `@workgroup_size(256)`, **one workgroup**: 1 of 32 M1 Max
  cores, ~3% of the machine, with no other resident warps to hide the stalls.

At `power-droplet-256`, `totalLevelSlots = 271,506`, so consecutive channels of
one row are **1.09 MB apart** and one row's 18 coefficients span **19.5 MB** of
address space.

### 2.3 The prize, sized

`large-power-dam-break`, from `artifacts/scene-size-overhead/large-before/traced.log`:

| level | capacity | occupied | occupancy |
|---:|---:|---:|---:|
| 0 | 262,144 | 1,289 | **0.49%** |
| 1 | 32,768 | 393 | 1.20% |
| 2 | 4,096 | 134 | 3.27% |
| 3 | 512 | 51 | 9.96% |
| 4 | 64 | 22 | 34.4% |
| 5 | 8 | 4 | 50.0% |

Total live: **1,893 slots**. The live solve working set is
`1,893 × 26 ch × 4 B = 197 KB` of state plus `1,893 × 18 × 4 B = 136 KB` of
neighbour indices — **333 KB**, against ~31 MiB of allocated state.

**333 KB is small enough to stay resident in on-chip cache**, against ~31 MiB
that cannot. A compacted, coherently-ordered solve would turn 18 dependent DRAM
round-trips per row into 18 cache hits — an order-of-magnitude latency
difference, not a measured one. That is the mechanism by which a 700×
bandwidth/latency gap can exist at all, and the reason it is addressable.

Second data point, and it matters for scope: `mini` runs at **18.25%** level-0
occupancy and `symmetric-expansion` at 6.48%. Occupancy sets the *footprint*
prize and scales with domain; the avalanche hash destroys *coalescing*
regardless of occupancy. **They are two independent defects and the second one
is present on every lane, including the small ones.** This is a candidate
explanation for the four-cell matrix being anti-correlated with scene size.

## 3. Experiments, ranked

Cost/benefit order, not narrative order. Every one gates on the standing
`symmetric-expansion` oracle: bitwise D4 must still first diverge at **step 68**
(volume/velocity/pressure/rhs) and **69** (diagonal/topology). Score by parsing
the `"phase":"diagnostic-evaluation"` line, never the exit code — that lane
always exits rc=1.

### E5 — bank the two landed shader fixes — **RUN, 2026-08-03: −43.1%**

**Result first.** `benchmark-power-dam-ab.ts --lane=droplet-256 --steps=60
--repeats=3`, interleaved, A/A sampled every round:

| arm | ms/advance | Δ vs control | own spread | verdict |
|---|---:|---:|---:|---|
| control (6 runs) | 151.63 | — | 0.93 | — |
| `FLUID_SPGRID_PARALLEL_LEVEL_COMMIT` | **89.55** | **−62.08 (−40.9%)** | 0.18 | **FASTER** |
| `FLUID_OCTREE_MGPCG_STAGED_SMOOTHER` | 149.05 | −2.57 (−1.7%) | 1.13 | inconclusive |
| both | **86.28** | **−65.34 (−43.1%)** | 0.60 | **FASTER** |

Two readings that matter:

1. **The parallel level commit is worth −40.9% on the largest lane** — against
   −16.1% previously measured on mini. `WORK_AND_DATA` §4.1 predicted exactly
   this ("score both flags on `--lane=large`, where the serialization should be
   worse"): `commitCandidateLevels` at `@workgroup_size(1)` is ~41% of the
   droplet-256 frame. This is axis 2 — unoptimized shader code — in its purest
   form, and it was already written and sitting default-OFF.
2. **The smoother's honest estimate is the paired one.** `smoother` vs `control`
   is −1.7% and inconclusive; but `both` vs `parallel` — two arms measured in the
   same rounds, spreads 0.60 and 0.18 — is **−3.27 ms, −3.7%**, which reproduces
   mini's −3.6% almost exactly. Its win is real and simply masked while the level
   commit dominates the frame.

**Noise caveat, stated because the harness flags it.** The reported A/A floor is
**28.83 ms**, set by a single round-2 outlier (`control-aa` 180.28 against
`control` 152.00). Rounds 1 and 3 agreed to 0.15 and 0.53 ms. The most likely
cause is background load from concurrent work on this machine during the run.
The verdicts survive the pessimistic floor anyway (−62.08 is 2.15× it), and each
arm's own 3-round spread is under 1.2 ms — but the floor is not clean, and a
re-run on an idle machine is worth one more round before this is quoted as final.

**The D4 gate is green on this tree with both flags on.** `symmetric-expansion`,
250 checkpoints, `FLUID_SPGRID_PARALLEL_LEVEL_COMMIT=1
FLUID_OCTREE_MGPCG_STAGED_SMOOTHER=1`:

| hook | first divergence | contract |
|---|---:|---:|
| `volume`, `velocity`, `pressure`, `rhs` | **68** | 68 |
| `diagonal`, `topology` | **69** | 69 |
| `wall-contact` | all four walls at 68, spread **0** | PASS |
| `checkpoint-count` | 250 | PASS |
| `validation-clean` | 0 errors | PASS |

The window is exactly the standing contract. The run exits rc=1 and that is the
expected outcome, not a regression — the gate criterion is the step the hooks
reach, never the exit code.

**Both flags are therefore ready to flip to default-ON**, which is the protocol
`WORK_AND_DATA` §7.3 asks for ("flip defaults for whatever passes with the D4
window still ≥68"). One line each; not done here, because changing a production
default is a separate decision from running the experiment.

### E5 — the original framing (kept for the record)

`WORK_AND_DATA` §5 landed `FLUID_SPGRID_PARALLEL_LEVEL_COMMIT` (**−41.0 ms,
−16.1%**) and `FLUID_OCTREE_MGPCG_STAGED_SMOOTHER` (**−9.1 ms, −3.6%**),
together **−47.6 ms, −18.7%**, both default-OFF, both bit-identical, D4 window
verified unchanged. The staged smoother is squarely axis 2: it replaces the
dynamically-indexed `var terms:array<f32,18>` that Metal spills to scratch with
named scalars, and stages the solve-invariant header into workgroup memory.

**Only round 1 of the interleaved A/B completed** — the lane went red mid-run.
Control and control-aa agreed to 0.29 ms, so the effects are two orders of
magnitude outside within-round noise, but they are single samples.

Work: run the 3-round interleaved median, then flip the defaults. No new code.
This is the largest measured, implemented, unbanked win in the tree.

### E6 — the grading closure: one lane, 32,768 serial atomics — **CLOSED, −14.7%**

**This displaces E0 as the top target.** After E5 the frame reorganized
completely and the old ranking is void. Re-profiled at HEAD, droplet-256, 80
advances, label isolation (ranking honest, absolute ~19% high):

| pass | ms/advance |
|---|---:|
| **Octree resident grading closure** | **27.53** |
| Octree persistent MGPCG (whole solve, one workgroup) | 14.88 |
| Advect fine phi rare | 3.72 |
| March Section 5 sparse changed frontier (×2) | 7.09 |
| `SPGrid V-cycle - publish validated exact level deltas` | 2.47 *(was ~96)* |

`FLUID_GRADING_ROUND_PROBE=1` splits grading by round: **r00 10.20, r01 8.69**,
r02 1.53, r03–r09 ~0.81 each. So 18.9 ms is real split work and ~5.7 ms is a
per-round floor.

**The mechanism, and the in-tree A/B that proves it.** `splitLeaf` materializes
a leaf's whole size³ owner partition — for size 32 (the authored leaf size on
these lanes) that is 64 pages × 512 cells = **32,768 dependent atomic
read-modify-writes in a single lane**, while the other 63 lanes of its
`@workgroup_size(4,4,4)` workgroup idle. `balanceTopologyAt` can raise up to 37
such splits per lane per parity, over 8 parities.

The tree already contains the other strategy: `balanceCoarseBlock` fans the same
size³ partition across its 256 lanes. **The profiler A/Bs them on the same lane
and the same work: the cooperative coarse dispatch costs 0.148 ms/advance
against 18.9 ms for the serial candidate rounds.**

Fanning out is observationally free — `claimLeafSplit` already elects exactly one
materializer per split, and the write is `atomicMin` of a value depending only
on (origin, size, cell), so it is commutative and idempotent and no lane can
observe another's order.

**Landed so far:** `materializeSplitStrided(origin, size, lane, lanes)`, the
shared primitive. It divides over **pages**, keeping the per-page nested triple
loop byte-for-byte serial. `splitLeaf` calls it with `lanes = 1` and measures
87.18 vs 86.97 ms/advance — cost-neutral, as required.

**Two measured negative results, recorded because they shaped the design:**

1. **A workgroup-local split queue drained behind a barrier does not work.**
   v1 had each lane claim into `var<workgroup>` storage, then the workgroup
   drain it cooperatively. droplet-64 passed; **droplet-256 produced zero
   advances in 600 s** where it otherwise runs 60 advances in ~90 s. The cause is
   structural, not a bug to chase: on queue overflow a lane runs the full
   32,768-cell serial walk *while 63 siblings block at `workgroupBarrier()`*,
   and the barrier also couples every lane to the slowest one — whereas the
   serial code lets lanes retire independently. A workgroup-local queue cannot
   bound a producer that raises up to 37 splits per lane per parity.
2. **Flattening the inner triple loop to stride cells costs +6.05 ms.** The same
   refactor with `flat % span.x` indexing measured **93.02 vs 86.97** at
   `lanes = 1`. Three integer div/mods on each of 32,768 cells is a real cost;
   this loop is ALU-sensitive as well as latency-sensitive. Stride pages, never
   cells.

**Next: v2 — claim into a device worklist, materialize by indirect dispatch.**
One workgroup per claimed split, `@workgroup_size(256)`, i.e. exactly the
`balanceCoarseBlock` shape, with the producer and consumer in *separate*
dispatches so nothing blocks on a barrier. Three constraints shape it, all
verified:

- **No new storage binding is available.** `OCTREE_PROJECTION_CORE_BUFFER_LAYOUT`
  carries 9 storage bindings and `tests/webgpu-octree-projection-binding-budget.test.ts`
  asserts `storageCount + 1 <= 10`, reserving one slot for activity
  instrumentation. The worklist must live in an existing arena — `compaction`
  has a planned base-offset map (`planOctreeCompactionAllocation`) and is the
  natural host.
- **`compaction` is `array<u32>`, not atomic**, so a lock-free append counter
  cannot live there. `owners` is `array<atomic<u32>>` with a 16-word header;
  the counter can live in a reserved header word while the payload lives in
  `compaction`.
- **`compaction` has no `INDIRECT` usage**, so the published dispatch args need
  one `copyBufferToBuffer` into an existing indirect buffer per round — the
  pattern already used at `webgpu-octree.ts:3534-3536`. Ten rounds means ten
  extra copies and pass closures; §4.3 of `WORK_AND_DATA` says that is not the
  wall, but it must be measured rather than assumed.

Predicted: 18.9 → ~1–3 ms, i.e. ~16 ms off an 87 ms frame. Gate as always on the
`symmetric-expansion` D4 window, which this cannot move — the write set is
unchanged and its order is unobservable.

### E6 closure — **LANDED, 2026-08-03: −12.58 ms, −14.7%** (`aa78e90`)

**v2 was not needed and its premise was wrong.** The 18.9 ms was never a lane
count. Two defects inside the inner body carried all of it, and both are
removals rather than additions — no worklist, no indirect dispatch, no extra
binding, no extra pass.

| arm (droplet-256, `benchmark-power-dam-ab.ts --steps=60 --repeats=3`, interleaved) | ms/advance | vs landed | own spread |
|---|---:|---:|---:|
| **landed** (all three below) | **73.08** | — | 0.20 |
| `control-aa` | 73.05 | +0.00 | 0.18 |
| `FLUID_OCTREE_GRADING_PAGE_FILL=0` — the original per-cell walk | 85.67 | **+12.58 (+17.2%)** | 0.25 |
| `FLUID_OCTREE_GRADING_MEMBERSHIP_LOAD=1` | 80.98 | +7.90 | 1.03 |
| `FLUID_OCTREE_GRADING_SPLIT_HELPERS=0` | 73.43 | +0.35 | 0.08 |

**A/A noise floor 0.27 ms.** The headline delta is 47× it and the sign is
identical in all three rounds (85.78/85.67/85.53 against 73.17/73.12/72.97).

Per-round attribution, `FLUID_GRADING_ROUND_PROBE=1` with label isolation:

| round | before | after |
|---|---:|---:|
| r00 | 10.20 | **2.31** |
| r01 | 8.69 | **2.10** |
| r02 | 1.53 | 0.95 |
| r03–r09 | ~0.81 each | ~0.77 each (untouched) |

**Split materialization: 18.9 ms → ~3.0 ms.** The pass is now ~10.8 ms, of
which **7.7 ms is the ten-round probe floor** — a different defect, see below.

#### 1. The owner word is page-invariant and was recomputed 512 times (−4.69 ms)

A child of size ≥ 8 is 8-aligned and so is the owner page, so a page lies wholly
inside **one** child; and `encodePagedOwner` keys the origin delta off the
*cell's brick origin*, which is page-invariant too. **Every one of a page's 512
cells therefore receives the identical 32-bit word.** The inner body rebuilt it
per cell through three runtime integer divisions by `child` plus a
`firstTrailingBit`. `splitPageAt` resolves it once per page; the fill collapses
to a contiguous 512-word loop with no address arithmetic at all.

This is the same ALU sensitivity the recorded +6.05 ms negative result measured
from the other direction — that experiment *added* three div/mods per cell and
paid 6 ms; this one *removes* about six and gains 4.7. The loop was never
bandwidth-bound; it was arithmetic issued on a dependent chain.

#### 2. The per-cell membership load can never observe a set bit (−7.55 ms)

`storeOwnerRequired` preserves `OWNER_WORD_TOPOLOGY` by loading the current word
and OR-ing the bit into the `atomicMin` candidate. That load is a **second
device round trip on a dependent chain** — half the traffic of the entire
materialization — and inside the topology candidate view it is provably a load
of zero:

- membership is a *leaf* property, published only by `markAcceptedOwner` during
  frontier emission, which runs **after** grading in the same encode;
- `commitOwnerPageCandidate` rewrites the whole inactive payload bank with
  `word &= ~OWNER_WORD_TOPOLOGY` (its own comment: *"Membership is a leaf
  property, not a resident-page property"*) in
  `Prepare inactive owner-page generation`, the pass **immediately before** the
  topology dispatches;
- every page reachable through the candidate directory is in that candidate key
  set by construction, so there is no page the split can address whose bit
  survives.

`splitOwnerWord` therefore returns the word unchanged when
`topologyCandidateView == 1u`, and keeps the loading form verbatim everywhere
else. The cross-module premise is asserted by a test rather than left in a
comment (`tests/octree-balance-elision.test.ts`), because if the candidate
clear were ever removed the materializer would silently *coarsen* marked leaf
origins: `OWNER_WORD_TOPOLOGY` sits at bit 21, **above** the exponent at bits
18–20, so a set bit inverts the finer-wins order of the `atomicMin`.

#### 3. Fan-out is real but no longer where the time is (−0.35 ms)

`materializeSplitPages` lets a losing asker take one page of the split it asked
for, claimed by `atomicMin` on the page's first cell — the write the fill would
perform anyway, so a helper that loses retires having spent exactly the three
device ops it already spent. No barrier, no shared memory, no queue: this is the
property the workgroup-local queue could not have. **Measured −0.9 ms while the
membership load was still present, and −0.35 ms after it went.** Kept, at 1.3×
the A/A floor and the same sign in all three rounds, but it is a rounding error
next to the two removals. **Depth was not the constraint; per-iteration cost
was.**

#### Negative results worth recording

1. **"Gather over allocated owner pages" is domain-shaped, not live-shaped, and
   would have violated Bet 1.** The plan's own framing assumed the page
   directory is sparse. It is not: the owner map is a **total partition of the
   domain** — every cell must return an owner from `ownerAt` — so
   `planOctreeOwnerPages` sizes `capacity == logicalBrickCount` and essentially
   all 32,768 pages at 256³ are resident. One workgroup per allocated page is
   16.7M cells per sweep, ten sweeps per advance. The sparsity that makes the
   *iteration* live-shaped elsewhere does not exist in this structure.
2. **Inverting the paper repairs into a self-check on the coarse leaf is more
   expensive than the scatter it replaces.** `repairPaperRatioNeighbors`
   inverts exactly — "L splits iff it has an 18-neighbour smaller than
   size(L)/2" is literally `ownerAtIsTooFine`, which `balanceCoarseBlock`
   already evaluates over 6 faces with 256 cooperating lanes. But
   `repairPaperMixedNeighbors` is a **2-hop** predicate (L splits iff some
   adjacent anchor has both a finer and a coarser neighbour), and evaluating it
   from L means walking L's 6·32² = 6,144 boundary cells × 18 owner lookups ≈
   110k dependent gathers — three times the 32,768 writes it would replace.
3. **The ablation route to sizing this is closed.** Filling one cell per page
   instead of 512 does produce a timing lower bound, but the resulting topology
   trips `air-support-failure`, which `--quality-invalid-probe` does not allow,
   so the run never reaches its report. Attribution has to come from correct
   arms, which is what the three overrides above are for.
4. **Memoizing the owner-page directory lookup does not pay** — implemented,
   measured, reverted (`nomemo` −0.41 ms, −0.6%, INCONCLUSIVE against a 1.23 ms
   A/A floor, but faster in *all three* rounds — 72.98/72.97/73.00 against
   73.27/74.28/73.50 — and the grading probe floor was consistently ~5% worse,
   0.808–0.839 against 0.766–0.784 ms/round). The premise was sound in both
   halves: the directory really is dispatch-invariant (`ownerPageMap` already
   caches the arena header on that reasoning, and every atomic store in the
   module targets the rejection latch or a payload word), and lookups really do
   arrive in page-local bursts (one balance invocation walks eight parities of a
   2³ box, always one page; `neighborTooFine` sweeps a face y-fastest and
   changes page once in eight). **The error was treating that load as a
   dependent DRAM round trip.** The directory entry is re-read by every lane of
   every workgroup, so it is the hottest line in the arena and was already a
   cache hit; the memo removes a hit and pays two extra thread-local registers
   plus a compare on *every* lookup. The dependent-load argument that carries
   §2.2 does not transfer to a line that is already resident. The comment is
   left on `ownerPageEncoded` so the next reader does not re-derive it.

#### What is left in this pass: the probe floor, 7.7 ms

r03–r09 cost ~0.77 ms each and did not move, because they raise no splits at
all — they are pure probing. Ten rounds × 0.77 ms is now **the majority of the
grading pass**. Two follow-ups, in cost order:

- **Short-circuit the fixpoint.** A device-side "splits raised since the last
  round" counter (a reserved `owners` header word incremented by
  `claimLeafSplit`) plus a per-round indirect argument publish would collapse
  rounds 3–9 to zero workgroups once the closure has converged. Predicted −5 ms;
  needs one extra tiny dispatch and one `copyBufferToBuffer` per round, and the
  early-out must be conservative or it moves the topology.
- ~~Memoize the owner-page directory lookup.~~ **Tried and reverted; see
  negative result 4.** What remains of that idea is the *second* load: the
  payload word, which is genuinely scattered. Collapsing it needs the owner map
  to stop being 512× redundant — a page inside a leaf of size ≥ 8 stores one
  word 512 times — i.e. a per-page uniform-owner summary consulted before the
  payload. That is the E3b-class structural change for this arena and it would
  also take a size-32 split from 32,768 writes to 64.

#### Two owner-side leads, parked but not lost

Both found while ranking `symmetric-expansion`; neither is being chased yet.

- **The owner-page directory is a full permutation on every lane**, so its load
  is pure indirection. `planOctreeOwnerPages` clamps `capacity <=
  logicalBrickCount` and every lane hits equality — symmetric-expansion 32/32,
  mini 8/8, droplet-256 32,768/32,768, ocean 4,800/4,800 (verified on CPU). Every
  `ownerAt()` therefore spends a **second dependent device load** indexing a
  fully populated table. Assigning pages as the IDENTITY when
  `capacity == logicalBrickCount` lets `ownerPageEncoded` compute the page
  instead of loading it, halving the dependent-load count shader-wide, and the
  directory would still be *written* with identity values so every existing
  reader — the CPU lookup, the read-only WGSL ABI, the diagnostics — is
  unaffected. Open question before it can land: `requireOwnerPageEncoded` has a
  missing-page rejection path, so full residency has to be established rather
  than assumed.
- **`refineCoarseBlock` and `balanceCoarseBlock` run their size-cubed predicates
  on lane 0 alone**, between barriers, with 127 (or 255) lanes idle — the same
  sync smell as the serial split materializer that E6 already cashed for
  −12.58 ms. It is a likely follow-up to the per-page uniform-owner summary,
  which touches the same two functions.

### E7 — the symmetric-expansion frame, measured — **the grading axis is spent on this lane**

`symmetric-expansion` is the lane that matters, and profiling it inverts the
whole ranking this document was built on. Baseline, interleaved A/B, 3 rounds:
**268.59 ms/advance**, A/A noise floor **3.65 ms**.

Label-isolated pass profile (274.03 ms frame; ranking honest, absolute ~2% high):

| pass | ms/advance | share |
|---|---:|---:|
| Octree persistent MGPCG — whole solve in one workgroup | **61.33** | 22% |
| Advect structured families | **40.34** | 15% |
| March Section 5 sparse changed frontier (x2) | **43.56** | 16% |
| Advect fine phi rare + common | **29.01** | 11% |
| Transfer accepted velocity to changed topology faces | 11.80 | 4% |
| Scatter recurring fine-band seed halos | 5.13 | 2% |
| **Octree resident grading closure (r00)** | **0.437** | **0.2%** |

The top six are **70% of the frame**. No other grading round reaches the
reporting threshold at all.

**Grading is 0.437 ms here against 27.53 ms on droplet-256** — 0.2% of the frame
instead of 32%. The reason is in this repo's own commit `db4f487`: droplet-256
holds ~100 live pressure rows by construction, while symmetric-expansion holds
**2,124**. droplet-256 is a container sweep with almost no fluid; the
fluid-heavy lane spends its time per ROW, not on topology. E6's −12.58 ms was
real and remains banked, but **the grading axis cannot move this lane**, and
neither can its two named successors:

- the **fixpoint short-circuit** (E6 follow-up) targets rounds r03-r09;
- the **per-page uniform-owner summary** targets split materialization writes.

Both are inside a pass worth 0.437 ms. Ranked by ceiling, they are worth less
than the A/A noise floor of 3.65 ms on this lane.

#### The fixpoint short-circuit does not boot, and is not worth repairing here

Measured as instructed. `FLUID_OCTREE_GRADING_FIXPOINT=1` failed in **all three
rounds**, verdict **NO DATA**, every round exiting on

    Initial sparse authority cold-topology published no liquid-row frontier

So the polarity inversion recorded in `2d501ea` (zero means keep grading, so a
zero-filled arena runs the full closure) was **necessary but not sufficient** —
something else in the flag, most likely the 16-to-24-word owner arena control
block that the flag also enables, breaks the analytic cold bootstrap. It stays
default-off and unrepaired: the ceiling above says repairing it buys at most a
fraction of the 0.437 ms it gates.

**The lesson for this document.** Every ranking in sections 1-3 was taken on
droplet-256, whose fluid is pinned at ~100 cells. Those numbers rank the
*container*, not the *solver*. Re-derive any target on a fluid-heavy lane before
spending on it. The four terms that dominate droplet-256's footprint are
0.26 MiB each here; the pass that dominates droplet-256's frame is 0.2% here.

### E2a — publish the A2 apply's stencil columns — **LANDED, 2026-08-03: −1.4%**

**The mechanism, first.** The SPGrid V-cycle smoother has a memoized neighbour
table: `applied()` reads `topology[columnBase + k*span]`, eighteen precomputed
columns per slot. **The Section 6.3 A2 apply has none.** `applyRow` re-derives
each of its eighteen spokes from scratch through `opPageSlot` — a page-neighbour
probe, a page-record decode, three brick-directory words, a ranked-slot
indirection, then the neighbour's flags and owner. Ten loads on a five-deep
dependent chain, per spoke, per call.

And it is called *constantly*. `applyBandRows` runs `2*sweeps - 1 = 15` times per
Section 4.3 correction, and there is one correction per CG iteration plus one for
the initial preconditioner — well over a hundred repeats of the identical
resolution per solve.

**It is identical every time, and that is provable from the bindings.**
`topology`, `coefficients`, `geometry` and `metrics` are all `read` bindings.
The only `state` channels this kernel ever writes are `S_RHS`, `S_A` and `S_B` —
the single writer is `storef`, and grepping its call sites gives exactly those
three. `S_KEY`, `S_FLAGS` and `S_OWNER`, the three the chase reads, are never
written by any phase. So the resolution is a pure function of the dispatch's
immutable inputs and hoists to a new phase P1b.

`FLUID_OCTREE_MGPCG_STENCIL_COLUMNS` (default **ON**) publishes 36 channel-major
columns per row — eighteen resolved owner rows for the full apply, eighteen for
the reduced class-0 apply — into an arena region appended above the two staged
input channels, so `channelByteOffset` and the external staging ABI do not move.
Per spoke the apply then does three loads on a two-deep chain instead of ten on
a five-deep one, and the surviving column read is **channel-major**, so a
wavefront of adjacent band items reads adjacent words instead of eighteen
uncorrelated chases. **Addressing only** — same eighteen terms, same order, same
`canonical18Sum` fold, same `reportAt` stage ordinals.

| arm | ms/advance | Δ vs control | own spread | verdict |
|---|---:|---:|---:|---|
| control (6 runs) | 82.16 | — | 0.25 | — |
| `FLUID_OCTREE_MGPCG_STENCIL_COLUMNS=1` | **80.98** | **−1.17 (−1.4%)** | 0.20 | **FASTER** |

`--lane=droplet-256 --steps=60 --repeats=3`, interleaved, A/A every round.
**A/A noise floor 0.30 ms**, so the delta is 3.9× the floor — and unlike E5's
run, this floor is clean (control 82.13 vs control-aa 82.18 in the same round).

**D4 gate green**, exactly the standing contract:

| hook | first divergence | contract |
|---|---:|---:|
| `volume`, `velocity`, `pressure`, `rhs` | **68** | 68 |
| `diagonal`, `topology` | **69** | 69 |
| `wall-contact` | all four walls at 68, spread **0** | PASS |
| `checkpoint-count` | 250 | PASS |
| `validation-clean` | 0 errors | PASS |

**Footprint, stated because this axis is also a memory program.** The columns
cost `36 * rowCapacity` words: 576 KiB at the droplet lanes' authored 4,096-row
reserve, 1.1 MiB at `large`'s 8,192, 9 MiB at the 65,536-row live ceiling and
21 MiB at the widened ocean's container-derived 148,600, taking that lane's
whole persistent arena to 32 MB — against 177 MiB for `spgrid-vcycle` alone and
inside Dawn's 128 MiB storage-binding default. It is a real addition and it is
not free; it is two orders of magnitude below the dense lattices E3 targets.

The one documented deviation: a resolution report now fires once at build time
rather than once per apply. The *set* of raised (flag, stage) pairs is unchanged
— P1b resolves exactly the rows P2 would, off the same accepted worksets — and
every one latches `control[0]`, so the solve fails closed at P1b instead of P2.
Which claimant wins `control[6]` can differ, but that word is already decided by
a 256-lane `atomicCompareExchangeWeak` race and is not a deterministic output.
The declined variant emits a **byte-identical** module (asserted in
`tests/webgpu-octree-persistent-mgpcg.test.ts` across seven option
combinations), and all twelve option combinations are naga-validated offline.

#### The more useful half of this result: −1.17 ms is a *bound*

The pre-registered prediction was ~4 ms, from a model that put the Section 4.3
band sweep at ~60–70% of the 14.88 ms slice and the eighteen-spoke chase at ~70%
of `applyRow`. The measurement says **the main-loop chase is ~9% of the
persistent solve**, so at least one of those two factors is badly wrong.

Recorded by mechanism, because it re-ranks everything below:

- **The main-loop page/brick/rank chase is not the wall.** Removing six of ten
  loads per spoke, across every call site, on the phase the code comments name
  as the hot one, moves 1.4% of the frame. Any further work that targets *that*
  chase — E2's row-major operator blocks, for instance — is now bounded above by
  roughly what it just bought.
- **What is left, ranked by remaining surface.** (a) `finerAdjoint`: eight
  children by eighteen directions, each with its own `opPageSlot`, two state
  loads, a `coefficientForDirection` pair and an arena gather — **~1,650 loads
  for a ghost-owning row against the 54 the cached main loop now costs**, and
  entirely uncached. (b) The V-cycle: `restrictLevel` runs a full 64-element
  bitonic sort (21 stages × 64 compare-exchanges on a dynamically indexed
  `array<f32,64>`, which Metal backs with thread-local scratch) to sort at most
  a handful of real terms. (c) The reductions: `reduceMerged` walks one 128-row
  virtual group at a time with a seven-level tree and ~10 barriers per group,
  ~2,300 barriers per solve.
- **The falsifier that fired.** The plan's own §2.2 attributes the slice to
  `sparseSmoothPhase`. That attribution now has to compete with a measurement:
  the A2 apply's addressing, which §2.2 does not mention at all, is worth 1.4%
  on its own, and the smoother's `applied()` was *already* memoized. The
  remaining candidates are not the ones §2.2 names.

### E2c — the restriction sum's sorting network — **−3.8%, gate owed**

**The purest unoptimized-shader-code hit in the tree.** `sorted64PrefixSum` runs
the FULL 64-wide bitonic network unconditionally — 21 stages × 64
compare-exchanges = **1,344 iterations**, each two dynamic loads and up to two
dynamic stores against an `array<f32,64>` that Metal backs with thread-local
scratch because the index is not a constant — to sort the handful of terms one
coarse slot actually receives in `restrictLevel`. Restricting it to
`m = next_pow2(n)` is **28× less work at n ≤ 8**.

| arm | ms/advance | Δ vs control | own spread | verdict |
|---|---:|---:|---:|---|
| control (6 runs) | 72.97 | — | 0.13 | — |
| `FLUID_OCTREE_MGPCG_RESTRICTED_PREFIX_SORT=1` | **70.17** | **−2.80 (−3.8%)** | 0.13 | **FASTER** |

`--lane=droplet-256 --steps=60 --repeats=3`, interleaved. **A/A floor 0.15 ms**
(control 72.95 vs control-aa 72.97) — the delta is **18.7× the floor**, the
cleanest margin this axis has produced. Twice the stencil-column win, from a
loop bound.

*The control moved 82.16 → 72.97 between the two experiments because the grading
agent landed `aa78e90` in the interim. Both A/Bs are interleaved within a single
source state, so both deltas stand; only the absolute baselines differ.*

**Bit-identity, proven offline rather than argued.** A differential harness
transcribed both WGSL bodies and compared the returned f32 **bits** over 23,400
cases: nine generators (uniform, wide-exponent, heavy ties, signed zeros,
all-equal, ascending, descending, near-sentinel) at every n in [0, 64]. **Zero
mismatches.** The argument the harness confirms: a bitonic network is a sorting
network, so both widths leave the n real values ascending in `[0, n)`; equal f32
differ in bits only as signed zero, and no permutation of signed zeros can
change a sum that starts at `+0.0`.

**And it removes a latent HEAD bug.** Outside the precondition — a NaN term —
HEAD's 64-wide network sorts the NaN past position n and pulls a padding
sentinel INTO the summed prefix, returning a **finite** `3.4028e38` that
`restrictLevel`'s `!finite(sum)` guard **accepts** and stores into S_RHS at the
coarse level. The restricted network returns non-finite, so the guard reports
OVERFLOW stage 87 and skips the slot. Same input; one launders it, one fails
closed. Found by the harness, not by reading.

**Status: default OFF, D4 gate owed.** The tree's protocol is that the A/B and
the symmetry gate flip a default, not review. The A/B is done and conclusive;
the gate could not be run because `lib/webgpu-octree.ts` is mid-edit in another
agent's worktree and its "GPU-resident octree projection" shader module does not
currently compile — the control run of `symmetric-expansion`, **without this
flag**, fails identically. This is the §5 failure mode again, in a different
file. Flipping the default is one line once the lane is runnable.

### The measurement that re-ranks everything: 100 unknowns

Two instruments landed with E2c, and between them they void most of §2 and §3.

**The band census, finally readable.** `readPersistentBandCensus()` had **no
caller anywhere** in `lib/` or `tools/`, so `FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS=census`
published four GPU words that nothing read — the same trap as the harness
swallowing child stdout, sprung and unnoticed. Wired through the smoke executor
(loudly: it throws if the mode was requested and no census came back) and
forwarded by `benchmark-power-dam.ts`. `power-hybrid-census` was silently
discarded by the same reader and is now forwarded too.

On `droplet-256`:

| | |
|---|---:|
| `liveRows` | **100** |
| `bandRows` | **100** — the band is every row |
| class-0 band rows | 18 (18.0%) |
| class-0 band rows at level > 0 | **0** |
| regular / power / identity rows | 18 / 82 / 0 |

**The whole pressure system is 100 unknowns.** `LANES` is 256. So every
row-strided loop — `for(var r=lane;r<liveRows;r+=LANES)` — runs **exactly one
iteration on 100 of 256 lanes**, and 156 lanes idle in every phase. The
persistent solve is not throughput-bound in any phase; it is a critical path.

**The idempotent phase-repeat probe.** The solve is one dispatch, so pass-label
isolation cannot split it and every ranking of its internals has been a model.
`FLUID_PERSISTENT_MGPCG_PHASE_REPEAT=band:3` re-runs a phase N extra times;
`applyBandRows` and `applyAllRows` are pure gathers whose items write only their
own row of an output channel that is never their input, so a repeat recomputes
the identical value — **value-neutral by idempotence**, iteration count and
convergence unmoved, only the wall changes. The bound is read from a uniform
word, never emitted as a literal, so the Metal backend cannot fold the repeats
away and silently measure nothing.

| probe | Δ for 3 extra passes | implied total cost |
|---|---:|---:|
| `applyBandRows` | +1.70 ms | **0.57 ms/advance** |
| `applyAllRows` | +0.70 ms | **0.23 ms/advance** |

**The entire Section 6.3 A2 apply surface costs 0.80 ms/advance** — about 6% of
the persistent solve. Consequences, all of them negative results by measurement:

- **E2b (`finerAdjoint` caching) is CANCELLED.** Its whole home is 0.80 ms, so
  the maximum available win is under 0.8 ms — for 4.7 MB and real complexity.
  The `coarseRegularBandRows = 0` reading kills it independently: every class-0
  band row is at level 0, where `finerAdjoint` returns `0.0` with zero loads.
- **E2d (dual-group reductions) is CANCELLED.** With 100 rows,
  `wPartials = ceil(100/128) = 1`. `reduceMerged` folds **one** virtual group,
  not the ~11 the specification assumed, so the barrier count is ~210 and not
  ~2,300 and folding two groups at once saves nothing.
- **`FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS=route` is not worth gating.** It can
  reach 18% of 100 band rows, none of them coarse — exactly the "a lane whose
  rows are all at level 0 should expect much less" caveat its own author wrote.
- **E2a's mechanism was right and its share estimate was wrong.** Before E2a the
  apply surface was ~2.0 ms; it is now 0.80. That is the 10-loads-to-3
  prediction landing almost exactly — on a phase that was never 60% of the
  slice.

**Where the remaining ~11 ms is: the V-cycle.** By elimination, and confirmed
constructively — E2c took 2.80 ms out of `restrictLevel` alone. `smoothLevel`,
`restrictLevel` and `prolongLevel` are the only large phases left unmeasured.

**The probe extends to them, and that is the next move.** `sparseSmoothPhase`
reads `source` and writes `destination` with `smoothLevel` alternating
`S_A→S_B`, `S_B→S_A` — the same read/write disjointness that makes the band
probe value-neutral, so a repeated phase recomputes the identical values. One
more probe arm buys the V-cycle's internal split before anyone writes code for
it. Do that before designing anything: this axis has now produced two models
that were wrong by 3.4× and by an order of magnitude, and one probe that settled
each question in a single run.

#### E2b/E2d — the cancelled successors, kept for the record

Ranked by remaining surface, each with its equality argument done, so the next
session can implement rather than re-derive. All three are inside the persistent
MGPCG slice.

**E2b — extend the column cache to `finerAdjoint`.** Same invariance proof,
eight times the surface. `finerAdjoint` resolves, per row, eight child ghosts
and then eighteen directions inside each owned one — `opPageSlot`, two `state`
probes, a `coefficientForDirection` pair (`metrics[other]` then
`coefficients[base+1+channel]`) and an arena gather per direction. For a
ghost-owning row that is **~1,650 loads**, against the 54 the cached main loop
now costs. Cache the pair `(coefficient, ownerRow)` per (child, direction): 288
words per row.

*The footprint is the design problem, and it is why this is not a copy of E2a.*
288 words/row is 4.7 MB at the droplet lanes' 4,096-row reserve but **75 MB at
the 65,536-row ceiling**, which is a real regression against Dawn's 128 MiB
storage-binding limit. Only rows at level > 0 that actually own fine ghosts need
an entry — the class-1/3 transition rows, a minority — so the payload must be
indirected: one `adjointIndex[row]` word plus a dense payload per ghost-owning
row, with a capacity guard that falls back to the chase (never fails the solve)
on overflow. Build it in the same P1b pass.

**E2c — landed and measured above; this specification is superseded.**

**E2d — the reductions run 128 lanes wide and barrier per 128 rows.**
`reduceMerged`/`reduceCurvature` walk **one** 128-row virtual group at a time;
each group costs a 7-level `reductionTree` (8 `workgroupBarrier`s) plus a
`storageBarrier`, and lanes 128–255 idle throughout. At ~1,300 rows that is ~11
groups × ~10 barriers × ~21 reduction calls per solve ≈ **2,300 barriers**, on
half the workgroup.

*The fold order is preserved if each group keeps its own tree.* `merged` is a
128-entry workgroup array and the association order inside a group is what
`CompensatedF32` needs; nothing requires the groups to be processed serially.
A second 128-entry `merged` array (4 KB more workgroup storage, well inside the
16 KiB floor) lets lanes 128–255 fold group `vg+1` while lanes 0–127 fold `vg`,
halving the barrier count and doubling the active width, with every group's
internal fold untouched. `storePartial` writes distinct `vg` slots, so the
stores do not collide.

### E0 — sort the worklist (cheap, and most of the prize)

**Re-scoped by E2a.** The 41-of-59 coalescing claim below is **overstated at the
measured occupancies**. Sorting the worklist makes slot addresses *monotonic*,
not *contiguous*: at level-0 occupancy of 0.49–1.5%, consecutive occupied slots
are 70–200 words apart, so 32 sorted lanes still touch 32 distinct cache lines.
What sorting buys is a compact *span* (a few KB per wavefront instead of a
1 MB level), i.e. DRAM-page and L2 locality, not coalescing. Contiguity needs
E1's dense index space. Keep the ordering — slot sort before Morton — but do not
expect E0 to deliver 69% of the traffic.


The worklist is filled in **insertion order**: `claimCount` increments as keys
are claimed (`spgrid-vcycle.ts:4038`), so `workSlot(l,i)` walks avalanche-hashed
slots in claim order. Sort that array ascending by slot once per topology epoch —
not per iteration — with the radix sort that already exists and is Dawn-validated
(`lib/webgpu-radix-sort-u32.ts`). The same slots are visited by the same lanes
doing the same arithmetic; only *which lane gets which slot* changes.

**This is not merely a probe — it addresses most of the traffic.** Counting the
memory operations in `sparseSmoothPhase` + `applied` per row per sweep:

| ops | address | coalesces after sort? |
|---:|---|---|
| 4 | `slot` (source, flags, diag, rhs) | yes |
| 18 | `slot` (`loadf(S_XP+k,l,slot)` coefficients) | yes |
| 18 | `slot` (`topology[columnBase+k*span]` neighbour ids) | yes |
| 1 | `slot` (store to destination) | yes |
| ≤18 | **`other`** (`loadf(source,l,other)`) | **no — the true gather** |

**41 of ~59 accesses (69%) are `slot`-addressed** and become contiguous the
moment the worklist is sorted. Only the ≤18 dependent gathers need E1.

- **Falsifier:** if the solve pass does not fall at fixed dispatch count, launch
  shape and occupancy, the scatter is not the cost and E1/E2 are cancelled.
- **Numerical status, precisely:** `sparseSmoothPhase` is Chebyshev ping-pong —
  it reads `source` and writes `destination`, and `smoothLevel` alternates
  `S_A→S_B`, `S_B→S_A` — so **for the smoother, reordering lanes is provably
  bit-identical.** It is *not* bit-identical for the dot products:
  `reductionTree` folds fixed per-lane partials, so permuting which slot lands on
  which lane permutes the `CompensatedF32` summation. The D4 window is the gate.
  Sorting by slot is a *hash-order* sort, which is arbitrary but deterministic;
  a Morton sort would be better for E1 but is **not** D4-equivariant, so it
  carries a real risk of moving the window. Do the slot sort first.

### E3a — the ceiling separation mask: 64 MiB for ~1,200 live entries

`webgpu-octree.ts:2930` allocates `nx*ny*nz*4` — **one u32 per finest cell**,
64.00 MiB at 256³. Both writer and readers reach it the same way:

- writer `markSeparationRow` (`structured-dynamics.ts:2498`):
  `cell = rowGeometry[rbase()+row].x` → `separationMask[cell]`
- readers `separationFresh` / `separationMarked` (`structured-boundary.ts:721`,
  `:738`): `cell = geometry[rbase()+row].x` → `separationMask[cell]`

So it is **only ever addressed at cells belonging to live rows** — ~1,200 of
16.7M entries at 256³, **0.007% utilization**. `structured-boundary.ts:735`
already names the dimension-sizing as a nuisance: *"the mask is dimension-sized
and the prepare kernel is one lane."*

It is **not** simply deletable, and it is not a single bit. The word packs 6 face
bits plus a 26-bit epoch stamp driving a 2-epoch renewal hysteresis
(`age = (epoch - (previous>>6)) & 0x3ffffff`, `age <= 2`), and that hysteresis is
load-bearing — `SVO`-adjacent free-fall drop oracles measure the top liquid layer
saturating near 0.9 m/s under a 2.8 m/s parabola without it.

It is also **correctly cell-keyed rather than row-keyed**: rows are renumbered
between the writing epoch and the reading epoch, cells are not. That is why the
obvious "index it by row" fix is wrong.

The right change is the one this axis is about: **a cell-keyed sparse map sized
to the row capacity** — an open-addressed table of `2 × rowCapacity` (key, word)
pairs. At droplet's 4,096 rows that is 8,192 slots × 8 B = **64 KB against 64
MiB, ~1,000×**, with the same stability guarantee and the same bit layout.

### E1 — compact, coherent solve addressing (the structural bet)

Keep the avalanche hash where it belongs — key→row lookup during setup — and give
the *iterative* solve a dense index space `[0, count(l))` in a spatially coherent
order, published once per topology epoch with its inverse.

- State goes from `26 × totalLevelSlots` (~31 MiB) to `26 × sum(count(l))`
  (~197 KB) — **L2-resident**.
- Lane-to-address becomes contiguous: full coalescing.
- The 18 neighbour gathers become gathers into a cache-resident array; with a
  Morton-ordered compaction most of the 18 land in the same few lines.
- Only run **after** E0 has confirmed the mechanism and **after** E5 has banked
  its win, so the A/B is not contaminated.
- Trap, explicitly: this is adjacent to the "restore the deleted LDS smoother"
  mistake (`WORK_AND_DATA` §8) — that failed because it changed the *iteration*
  (different summation tree, stale cross-page neighbours), not just the layout.
  **E1 must change addressing only.** `canonical18Sum`'s fold order is
  load-bearing and must be preserved term-for-term.

### E2 — row-major operator blocks

The 19 coefficients and 18 neighbour slots of one row are channel-major, so the
persistent kernel touches 37 separate streams to assemble one row. Pack them into
one contiguous per-row block.

Caveat that makes this a *second* cut, not a first: channel-major is the
*correct* layout for the wide data-parallel V-cycle setup kernels, where adjacent
lanes read the same channel of adjacent slots. Only the single-workgroup
persistent solve wants row-major. Decide per consumer, and consider publishing
both — E3 frees enough memory to make a duplicate cheap.

### E3b — the remaining dense lattice arrays

`fine-levelset-topology` 259.91 MiB, `fine-levelset-bricks` 139.19 MiB,
`owner-pages` 129.88 MiB, at 0.003% live occupancy. The block-skip that already
exists for the passes proves the sparsity is known and exploited at *dispatch*
level; the allocations were simply never followed. Predicted: 896 MiB → well
under 100 MiB at 256³.

Beyond footprint, this feeds back into E1: a smaller total resident set leaves
more of the live working set cache-resident.

### E4 — occupancy of the solve itself

Even with perfect locality, `@workgroup_size(256)` × 1 workgroup is ~3% of the
machine. The plan's item 0 asks for the row-count threshold above which a
parallel solve should be selected; the honest form of that question after E1 is a
**multi-workgroup persistent solve** with a grid-wide barrier. Sequence it last:
its value depends on E1 having removed the latency that currently makes extra
lanes useless.

## 4. What this does not claim

- **No wall has been measured for any experiment in §3.** E5's numbers are prior
  single-sample measurements; everything else is a prediction with a stated
  falsifier. Given this program's own history — three "structural prizes" that
  measured zero — treat every number in §3 as unbanked until a 3-round
  interleaved median exists, and report the nulls as loudly as the wins.
- The memory census measures **resident allocation**, not working set or cache
  residency. It sizes the structures; it does not prove they are read.
- The 0.49% occupancy figure is from a `large-before` artifact captured at the
  container-derived default row capacity (33,536), i.e. an `environment:
  CONTAMINATED` configuration. The authored capacity is 8,192, which would read
  ~1.5% — still two orders of magnitude of headroom, and the coalescing defect is
  capacity-independent either way.

## 5. Blocked

GPU lanes are currently unrunnable: `lib/webgpu-water-pipeline.ts:921` has a
syntax error from an edit that landed at 17:38 during this session (unrelated
render/caustics work). `npx tsc --noEmit` does not catch it; the esbuild
transform does, which is why `node --import tsx -e "import('./lib/…')"` belongs
in the loop before any measured run — `WORK_AND_DATA` §6 learning 6, hit again.

The two memory censuses in §1 completed before that edit and are valid.

**2026-08-03, the same class again, different file.** `symmetric-expansion`
cannot run: `lib/webgpu-octree.ts` is dirty with ~120 uncommitted lines in
another agent's worktree and its "GPU-resident octree projection" shader module
fails to compile. The control run of the gate, with no experimental flag set,
fails identically — which is how you tell an upstream break from your own. E2c's
D4 gate is owed on this.

## 5b. OWED: the fluid-heavy re-measurement, and why it could not run

**The most important number this session produced is `liveRows = 100`, and it
undercuts the session's own headline.** The droplet family pins the fluid at
~100 cells and sweeps the container — that is its design, and it makes
`droplet-256` an instrument for *container overhead*. A 100-unknown pressure
system on a 256-lane workgroup is not evidence about how any of these wins
behave when there is actually water in the scene. Until the same wins are
scored on a fluid-heavy lane, **none of them may be quoted as a general
speedup**; each is, so far, a measured win on an overhead-dominated lane.

That applies to all three of the session's landed perf commits:
`aa78e90` (grading page fill), `9675040` (stencil columns) and `833d8b0` (E2c).

**Ready to run, one command, the moment the tree is green:**

```
node --import tsx tools/benchmark-power-dam-ab.ts --lane=fill-800 --steps=60 --repeats=3 \
  --arm=chased:FLUID_OCTREE_MGPCG_STENCIL_COLUMNS=0 \
  --arm=narrow:FLUID_OCTREE_MGPCG_RESTRICTED_PREFIX_SORT=1 \
  --arm=gradefill0:FLUID_OCTREE_GRADING_PAGE_FILL=0 \
  --arm=smoothx3:FLUID_PERSISTENT_MGPCG_PHASE_REPEAT=smooth:3
```

`fill-800` fixes the 256-cubed container and sweeps the fluid, so it is the
droplet family's exact dual. Two arms are *negative* (they disable a landed
default), so a win that transfers reads as SLOWER there; `smoothx3` returns the
V-cycle smoother's share by the same idempotence the band probe used. Pair it
with `FLUID_POWER_HYBRID_CENSUS=1 FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS=census`
on a short run first to get that lane's `liveRows` — the single number that says
whether the lane is throughput-shaped at all.

**Why it did not run: every GPU lane is down, and it is not this work.** At
19:40 `fill-800` failed at t=0 with "Initial sparse authority cold-topology
published no liquid-row frontier". Two controls attribute it:

| control | flags | result |
|---|---|---|
| `droplet-256` — the lane that produced this session's A/Bs green at 19:20 | **none** | t=0 cold-topology failure |
| `fill-800` | `FLUID_OCTREE_MGPCG_STENCIL_COLUMNS=0` | identical failure |

With that variable at `0` the persistent kernel emits a module byte-identical to
pre-session HEAD and the arena reverts to its original size, so the persistent
solve is excluded by construction. `lib/webgpu-octree.ts` — the grading agent's
file — carries ~135 uncommitted lines and was last written two minutes before
these runs; earlier the same tree failed with `[Invalid ShaderModule
"GPU-resident octree projection"]`. **Do not attempt to fix or work around it.**

This also blocks E2c's D4 gate. Both are one command each once that agent
commits a green tree.

## 6. Traps this axis has now paid for twice

- **`node --import tsx -e "import('./tools/webgpu-smoke-executor.ts')"` RUNS A
  SMOKE.** That module self-executes on import: it constructs a device, takes
  the exclusive GPU lock and starts advancing a scene. Type-checking it that way
  costs ~90 s of somebody else's GPU time and leaves a lock behind. Use
  `npx tsc --noEmit`, or import a `lib/` module, never that one.
- **A census with no consumer is not a measurement.** `readPersistentBandCensus`
  had no caller in `lib/` or `tools/` at all, and `power-hybrid-census` was
  printed by the child and discarded by `benchmark-power-dam.ts`. Both are wired
  now, both loudly. Before trusting any `FLUID_*_CENSUS`-style knob, grep for a
  consumer.
- **A model of this kernel is worth one run of a probe.** Two models in this
  section were wrong by 3.4x and by an order of magnitude. The idempotent
  phase-repeat probe settled each question in a single interleaved A/B.
- **Lane choice is a claim about what you measured.** The droplet family pins
  the fluid and sweeps the container, so a droplet win can be pure container
  overhead. Any win quoted as a general speedup needs a fluid-heavy lane
  (`fill-800`, `large`) beside it.
