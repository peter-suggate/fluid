# DCGrid evaluation against the power-octree implementation

Evaluates Raateland et al. 2022, *DCGrid: An Adaptive Grid Structure for
Memory-Constrained Fluid Simulation on the GPU* (I3D / PACMCGIT 5(1) art. 3),
against the current WebGPU power-octree solver.

Paper: `docs/papers/raateland-2022-dcgrid.pdf` / `.txt` (downloaded 2026-07-26
from `graphics.tudelft.nl/~klaus/papers/DCGrid.pdf`).
Reference implementation: `github.com/wouterraateland/dcgrid` (CUDA).

Date: 2026-07-26. Every claim about the current tree is anchored on a symbol
name; re-grep before acting.

---

## Bottom line

DCGrid contains **five separable mechanisms**. Four are already present here in
equal or better form, or are inadmissible for a free-surface liquid. **One is
genuinely missing and is worth taking: precomputed cell-level apron indices.**

The catch is that its value here rests on a claim the paper does not actually
establish, and that this tree currently cannot measure — see
[Measurement is blocked](#measurement-is-blocked).

| DCGrid mechanism | Present here? | Verdict |
| --- | --- | --- |
| Hierarchy of sparse uniform grids, factor-2, nested containment | Yes — `rebuildCandidateLevelSetFor` inserts `MG_ONLY` parent aliases at every coarser level | No change |
| GPU hash table for random access | No — direct brick-mask + popcount rank (`directoryLookup`) | **Reject.** Ours is strictly better; DCGrid's hash is our known bottleneck |
| **Precomputed apron cell indices** | **Only at block granularity** (`pageNeighbour`, 27-page; `FINE_LEVELSET_HALO_COUNT = 27`) | **Adopt at cell granularity** — the one real gap |
| Memory-constrained topology optimization | No — refinement is an unbounded physics predicate, fails closed on overflow | **Partially adopt.** Framing does not transfer, the memory bound does |
| Priority-score k-selection re-arrangement | No | **Reject as primary.** Applicable only to discretionary refinement above the mandatory band |

---

## Part 1 — How much of DCGrid's evidence survives scrutiny

The paper's headline is "DCGrid executes stencil and streaming operations
substantially faster than SPGrid and GVDB." That statement needs decomposing,
because the two comparisons are not of equal quality.

### The SPGrid comparison is a platform comparison, not a structure comparison

- **Fig. 11** (stencil + streaming kernels, the plot that carries the
  data-structure claim): caption states "SPGrid operations are performed on an
  Intel i7-3820 (4 cores)". DCGrid runs on a GTX 1070. This is a 4-core CPU
  against a 1920-core GPU.
- **Table 1** (the 449× end-to-end number): the SPGrid row is footnoted "Data
  for the SPGrid result is sourced from [Setaluri et al. 2014]. Experiments run
  on an Intel Xeon E5-2670." Different machine, different year, no rendering
  time, and DCGrid used 82% of the cells.

The authors caveat this themselves: *"Since DCGrid and SPGrid run on different
systems... the performance difference might be less significant in other
scenarios."* That is an understatement. **Nothing in the paper establishes that
DCGrid's addressing scheme beats SPGrid's addressing scheme on the same
hardware.** Setaluri's SPGrid is CPU-only by construction (it relies on Haswell
page-table behaviour), so the comparison the paper needed could not be run.

For this project the SPGrid comparison is therefore worth **zero**. We are
already a GPU SPGrid derivative; the delta the paper measured is one we already
banked.

### The GVDB comparison is the real evidence, and it is narrow

Fig. 13 / Table 1: GVDB and DCGrid, same GPU, same 2563 domain, same 97 MB
budget, same solver configuration (cell-centred semi-Lagrangian, vorticity
confinement, 10 Jacobi iterations).

| Stage | GVDB | DCGrid | Ratio |
| --- | ---: | ---: | ---: |
| Advect | 15 ms | 5.2 ms | 2.9× |
| Project | 102 ms | 9.1 ms | **11.2×** |
| Topology | 9.3 ms | 3.5 ms | 2.7× |
| Total | 142 ms | 29 ms | 4.8× |

The authors attribute the projection gap to a specific cause: *"we attribute a
large part of the performance difference... to the projection operation, where
GVDB has to synchronize its apron cells after each iteration."*

That is the load-bearing sentence in the entire paper, and it is a claim about
**apron values**, not apron indices: GVDB copies neighbour *data* into an apron
every Jacobi iteration; DCGrid stores neighbour *indices* once per topology
change and reads through them. The 11.2× is the cost of re-materialising
adjacency inside the solver loop.

That is directly transferable, because we do a version of the same thing.

### The honest cost side, which the abstract omits

Fig. 10 and Table 1 rows 6a/6b, same 2563 domain, 8 channels, fully dense:

| | Uniform | DCGrid dense | Ratio |
| --- | ---: | ---: | ---: |
| Memory | 537 MB | 1177 MB | **2.19×** |
| Time/step | 76 ms | 130 ms | **1.71×** |

Break-even against a uniform grid is at **~50% active cells** (8 channels) and
**~60%** (17 channels). The paper says the apron indices are "the main
contributor to the memory overhead". So DCGrid's structure costs roughly 2×
memory and 1.7× time, repaid only by sparsity. That is a fair trade for smoke,
where the priority field is diffuse. It also means **the apron is not free** —
any adoption here must budget for it.

---

## Part 2 — What the current implementation actually does

Establishing this precisely matters, because three of DCGrid's five mechanisms
turn out to be already present.

### Addressing: brick occupancy bitmask + popcount rank

`directoryLookup` / `find` (`lib/webgpu-octree-spgrid-vcycle.ts:1640-1648`)
resolves `(level, q) → slot` by:

1. `brickRecord(l,q)` → a 4-word record per 43 brick (generation, low mask, high
   mask, rank base);
2. generation check;
3. 64-bit occupancy mask test;
4. `countOneBits` rank within the brick;
5. `rankedSlots[base + rank]` indirection;
6. key verification against `state[KEY]`.

That is ~5–6 loads, three of them a dependent chain. It is **deterministic and
probe-free** — strictly better than DCGrid's open-address hash with tombstones
and periodic refill.

`pageSlot` (`:1433-1444`, and the V-cycle copy at `:1653-1666`) is the
page-relative variant: it takes the 27-entry immutable page adjacency record
(`pageNeighbour`) to skip the page directory, then still performs steps 3–6 per
cell.

### Block-level aprons already exist

- MG pages carry a 27-entry physical page adjacency record
  (`PAGE_RECORD_WORDS = 28`, `pageNeighbour`).
- Fine level-set bricks carry a 27-entry halo (`FINE_LEVELSET_HALO_COUNT = 27`,
  `lib/octree-fine-levelset-bricks.ts:13`), carried across republication
  (`lib/webgpu-octree-fine-levelset-topology.ts:1683`).
- The SVO renderer already uses apron-padded 103 physical pages
  (`lib/webgpu-svo-node-mip-pyramid.ts:156`).

**So the apron pattern is understood in this codebase — it is applied at block
granularity, never at cell granularity.** That is exactly the gap.

### Nested containment already holds

`rebuildCandidateLevelSetFor` (`:1813-1824`) inserts, for every row of native
level `n`, a slot at every level `l ≥ n`: `ACTIVE` at `n`, `MG_ONLY` above. That
is precisely DCGrid's restriction rule (1) — every cell has a parent at the next
coarser level. An apron walk is therefore well-defined here without any new
invariant.

### The topology invariant here is *stricter* than DCGrid's

DCGrid imposes **no grading condition at all**. Its rules are only (1)
parent-existence and (2) coarsest-level density. Its apron construction walks
"increasingly coarser mipmap levels" until it finds a cell, and then — this is
the key sentence — *"we now treat all cell indices in aprons as if they
originate from the same mipmap level."*

So a DCGrid level-0 cell may sit next to a level-3 cell and take its value at
level-0 spacing. That is a first-order approximation with an unbounded stencil
error, accepted because smoke tolerates it.

We cannot do that in the pressure operator. Our entire catalog — 1,608 cases,
19 channels, `catalog.coefficientData` — exists precisely because the topology
guarantees an exclusive same-or-one-finer / same-or-one-coarser 1-ring.

**But this cuts in our favour**, and it is the most useful observation in this
document: because our invariant is stricter, an apron here resolves to a cell
that is *genuinely* a valid stencil neighbour. **The apron stops being an
approximation and becomes pure memoization** — the identical value the current
code computes, cached. It can be gated bit-exactly against the current path,
with no numerical review at all.

### Refinement is a hard predicate, not a priority

`pressureRefinementEvidence` (`lib/webgpu-octree.ts:4985-4998`) returns a
**boolean**: refine if the fine summary crosses the interface, or is within
one displacement/support ring of it, or intersects inflow protection.

There is no score, no budget, and no selection. Capacity is provisioned ahead of
time (`maximumSparseSlots = nextPowerOfTwo(rowCapacity * 16)`, `:691`) and
overflow **fails closed** — `candidateReport` rejects the epoch and the previous
topology stays live.

---

## Part 3 — The one mechanism worth taking

### Where adjacency is re-derived inside the solver loop

This is the GVDB failure mode, in our code.

**`applyRow`** (`:1472-1482`) — the second-order operator, run once per PCG
iteration per row:

```
per row:      geometry[row], metrics[row], 19 coefficients   (coalesced, ~100 B)
per channel:  pageSlot(...)          ~6 loads, dependent chain
              state[FLAGS]           1
              state[OWNER]           1
              inputVector[owner]     1
              -----------------------------
              ~9 scattered loads × up to 18 channels
```

Addressing costs roughly **8× the coefficient traffic in load count**, and far
more in latency, because `pageSlot`'s six loads are serially dependent.

**`finerAdjoint`** (`:1460-1471`) is worse. Per row it walks 8 children × 18
candidate directions, and each surviving pair calls `coefficientForDirection`
(`:1454-1455`), which **linearly scans all 18 channels applying a sign/permute
transform** to find one coefficient. Worst case ≈ 144 `pageSlot` calls plus
144 × 18 = 2,592 coefficient comparisons **per row per apply**.

**`applied`** (`:1991-1994`), called from `restrictAndGhostAccumulate`
(`:2075`), uses the *full* `find()` — generation check included — for all 18
channels, per fine slot, per restriction, per V-cycle.

**`smoothPage`** (`:2037-2051`) is the closest thing to right: it stages a page
plus a one-cell halo into workgroup memory and runs 2–4 Chebyshev sweeps
locally. But it re-resolves the staging every invocation —
`HALO_ELEMENTS = 600` `pageSlot` calls to smooth `PAGE_ELEMENTS = 256` cells
(`:1524-1526`). With 4 levels that is 6 smoother dispatches per V-cycle, one
V-cycle per outer iteration.

And `pageAppliedA/B` (`:2004-2021`) re-loads `state[at(KEY,l,slot)]` from global
memory **inside the 18-channel loop** — the same key, 18 times per cell.

### Why the Section 6.3 bandwidth argument does not currently apply

`lib/octree-section63-operator.ts` asserts `storedNeighbourIndices: 0`, and
`estimateSection63Bandwidth` (`:239-252`) justifies the layout:

```
resolvedGatherBytes  = rows × (4 + n × 12)      // 220 B/row at n = 18
section63StreamBytes = rows × (19 × 4 + 4)      //  80 B/row
→ 2.75× byte reduction
```

**That model charges zero bytes for the neighbour address.** That is true in
Setaluri's paper, where the operator runs over a dense page and a neighbour is a
compile-time offset into staged memory. It is **not** true in `applyRow`, which
is dispatched over compact *row* worksets and must call `pageSlot` — six loads —
to convert a direction into a slot.

So the "addresses are derived, not stored" property, which the implementer
handoff records as landed, is real at the level of *encoding* (`section63Direction`
+ a 6-bit transform code replace 36 stored indices) but **the derivation is not
free at runtime**. The bandwidth model describes an implementation that does not
exist yet. This is the sharpest finding here and it is independent of DCGrid.

There are two ways to make the model true:

- **(a) Make `applyRow` page-resident**, like `smoothPage`. Then addresses
  genuinely become arithmetic. The obstacle is that band worksets are sparse
  scattered rows, so page-resident dispatch schedules mostly-empty pages.
- **(b) Adopt DCGrid's apron**: resolve the index once per topology epoch and
  store it.

Section 6.3 was chosen to avoid (b). Its avoidance is only real under (a).
DCGrid's contribution is precisely that it *chose* (b) and measured it fast.
That is a real and non-obvious datapoint — with the caveat from Part 1 that the
measurement proving it never actually ran against a GPU SPGrid.

### The concrete proposal: a case-indexed sparse apron

Not DCGrid's dense apron. A hybrid that exploits what we have and DCGrid does
not — the dense case ID.

1. **Regular interior rows (case 0) store nothing.** Their six neighbours are
   ±1 offsets; if the row is page-staged they are shared-memory arithmetic.
   DCGrid pays for these; we should not.
2. **Non-regular rows store only their live channels.** The case ID already
   determines which of the 18 are non-zero. Average live channels is well below
   18 — the count is exactly what `catalog.coefficientData` already encodes.
3. **Entries are page-relative, not global.** The invariant guarantees the
   neighbour lies in the 3×3×3 page neighbourhood: 5 bits of page ordinal +
   10 bits of halo index (600 < 1024) = 16 bits. Two per `u32`.
4. **`finerAdjoint`'s 8 × 18 walk collapses into a stored ghost incidence list**
   — it is a fixed function of the topology, recomputed every apply today.
5. **`coefficientForDirection`'s linear scan becomes a stored channel index.**

Expected effect per non-regular channel: 6 dependent scattered loads → 1
coalesced load plus 1 scattered value fetch. Storage: ~2–8 bytes/row for typical
case mixes, against 104 bytes/slot of existing state (26 channels,
`STATE_CHANNELS = 26`) — under 10% overhead, versus DCGrid's own ~2× because it
stores a dense apron for every block.

The same idea applied to `smoothPage`'s halo: store 344 shell indices per page
(the 600-element halo minus the 256 interior), = 1,376 B per page = 5.4 B/cell,
about 5% on top of the MG state. Deletes 344 `pageSlot` calls per smoother
invocation.

**Every one of these is bit-exact against the current path.** The gate is a
differential test, not a numerical review.

---

## Part 4 — Memory-bounded adaptation: half-transferable

DCGrid's second contribution is topology adaptation as constrained optimization:
global block limit `B_max`, per-level limits `B_max,l`, priority score `p(c)`
(vorticity magnitude), and Algorithm 1's swap — pair the lowest-priority active
block at level `l` with the highest-priority active subblock at level `l+1` and
exchange them, so **block count is exactly conserved and allocation never
grows**. Move limit `m_l` is predicted from the previous step's violation count
via `max(0.8 m_l, 1.5 v_l)`.

**The framing does not transfer.** DCGrid's premise is that any distribution of
cells is physically valid and merely less accurate. For a free-surface liquid it
is not: coarsening the interface band does not degrade the surface, it destroys
it. `pressureRefinementEvidence` is a correctness requirement, not a preference.

**The memory bound does transfer, and its absence is a real defect.** Today
refinement demand is unbounded and capacity is a fixed provision; exceeding it
rejects the topology candidate and retains the previous epoch. The surface keeps
moving while topology does not follow. `docs/POWER_LIQUIDS_PERF_HANDOFF.md`
documents exactly that failure shape ("post-stall regime… step cost explodes
30–100× nondeterministically with identical dispatch counts"), attributed there
to a correctness bug — but capacity rejection produces the same signature.

The usable synthesis is a **two-tier budget**:

- **Tier 1 — mandatory.** Interface band, inflow, solid support. Never subject
  to a budget. If tier 1 alone exceeds capacity, that is a hard failure and must
  stay a hard failure.
- **Tier 2 — discretionary.** Everything above the mandatory band: extra
  refinement rings, vorticity-driven or curvature-driven detail. **This is where
  DCGrid's algorithm applies verbatim** — score it, budget it, swap it.

Tier 2 does not exist yet, so this is a feature proposal, not an optimization.
Its value is that it converts "stall unpredictably at capacity" into "degrade
predictably", which is worth having independent of speed.

Two implementation notes: DCGrid performs its per-level sort **on the CPU**
every timestep, which violates this project's no-hot-path-readback rule and
would need a GPU partial selection. And DCGrid's hash requires periodic refill
because deletion leaves unusable tombstones — a problem our rank-based directory
does not have and should not acquire.

---

## Part 5 — What DCGrid does not solve, and one thing it does

### Does not help

- **Handoff item 2 — the single-threaded MG candidate rebuild.**
  `buildCandidateLevelSetsAndGhosts` and `buildCandidateLevelDeltas` are
  `@workgroup_size(1)` at `[1,1,1]`, with `rebuildCandidateGhostsFor` doing
  `levels × rows × 18 channels × cLookup`, where `cLookup` (`:1807-1808`) is a
  256-probe linear hash. DCGrid's answer to lookup **is** a linear-probe hash —
  it would make this worse, not better. An apron *adds* work to this stage
  unless the rebuild is parallelised and delta-driven first.
- **The five cutover-suite failures**, including the 12-vs-10 storage-buffer
  limit contract. Unrelated.
- **Section 4.3 shell depth `k = 2`** and the invented even-`k` parity rule.
  Unrelated.

### Does help, and is worth noting separately

DCGrid states the apron refresh rule precisely: *"we calculate the apron cell
indices for each newly inserted block and refresh the apron cell indices that
pointed to refined or coarsened blocks."*

That is a **delta rule with an explicit dependency closure** — dirty = inserted
∪ pointed-at-a-changed-block. Handoff item 3 records that our topology-change
predicate is hard-wired `true`, making the `if(!topologyChanged)` comparison
branch unreachable, and asks for "a cheaper *sufficient* dirty predicate". This
is that predicate, stated by a paper that had to solve the same problem.
It is worth taking **whether or not the apron itself is adopted.**

---

## Measurement is blocked

`docs/OCTREE_M1_MAX_IMPLEMENTER_HANDOFF.md` item 1 states the GPU lock was free
and "nothing blocks capture — it simply had not been run."

**That is no longer true.** Verified 2026-07-26 with the lock free and no
concurrent session:

```
npm run profile:power-dam-mini   → exit 1
npm run benchmark:power-dam-ui   → exit 1

Error: Initial sparse authority cold-topology published no liquid-row frontier
  at WebGPUUniformEulerianSolver.publishInitialSparseScenePhase
     (lib/webgpu-uniform-eulerian.ts:724)

mini: frontier=[0,1248,0,0,1248,0,0,1,1,1958,0,...]
ui:   frontier=[0,1352,0,0,1352,0,0,1,1,1798,0,...]
```

Both lanes fail at **cold bootstrap**, before any stepping. The octree method
does not initialize in the current working tree. `frontier[3] == 0` fails the
guard in both; the mini lane additionally reports a zero selected count.

Consequently **no performance claim in this document is measured**, including
mine. Every ratio above is derived from load counts read out of the shader
source. That is enough to rank hypotheses; it is not enough to authorize a
structural rewrite.

---

## Recommendation

**Do not start apron work now.** The order that respects the dependencies:

1. **Fix the cold-topology bootstrap.** Nothing below can be evaluated, and
   handoff item 1 is blocked on it, not merely unstarted.
2. **Capture the four baselines** (handoff item 1) — now genuinely unblocked
   once (1) lands.
3. **Fix the dirty predicate** (handoff item 3), using DCGrid's inserted-∪-
   pointed-at closure rule. Cheap, and it is the prerequisite for any per-epoch
   precomputed table: you cannot add work to a publisher that runs every step.
4. **Parallelise the candidate rebuild** (handoff item 2). Same reason,
   and DCGrid's hash is evidence *against* the current `cLookup` design.
5. **Then, and only then, the apron**, cheapest-first:
   - **5a.** Stored halo indices for `smoothPage`. Smallest blast radius, fully
     contained in one function, bit-exact. This is the decisive experiment: it
     either shows that removing 344 `pageSlot` calls per page-smooth moves the
     wall clock, or it does not, and the answer settles 5b and 5c without
     building them.
   - **5b.** Stored ghost incidence for `finerAdjoint`, plus a stored channel
     index replacing `coefficientForDirection`'s linear scan. Largest arithmetic
     win, no layout change.
   - **5c.** Case-indexed sparse apron for `applyRow`. Largest change; requires
     the Section 6.3 bandwidth model to be re-derived with a non-zero address
     cost first, because the current model would score it as a regression.
6. **Two-tier refinement budget** — a feature, not an optimization. Schedule
   against reliability goals, not performance goals.

The single most valuable thing to carry forward from the paper is not a data
structure. It is the GVDB result: **an 11.2× projection gap caused entirely by
re-materialising adjacency inside the solver loop.** We do that, in four places,
and we have no measurement of what it costs.
