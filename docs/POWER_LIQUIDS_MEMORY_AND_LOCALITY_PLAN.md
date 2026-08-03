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

### E0 — sort the worklist (cheap, and most of the prize)

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
