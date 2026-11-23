# GPU shape M0e: dependent-load chains in the heavy brick kernels

Lane A, 2026-11-23. This is offline analysis of the WGSL the dam census compiled (kcache hashes in
[m0b-census.md](m0b-census.md) and census/dam/census-caps.tsv; phiSolid comes from seep). It ends with one short GPU
validation run.

The question comes from lane B's counters ([m0c-footprint.md](m0c-footprint.md)):
- The heavy kernels idle the memory system: buffer-read limiter ≤ 8%, 2–22 GB/s, occupancy 6–24%.
- So each lane is probably waiting on its own chain of dependent loads.

M0e asks which chains those are, how long they are, and which ones to shorten.

## Summary

- **The per-link unit is the brick directory walk.**
  - Every lookup is `nbPageWord` (dir) → `nbChild` (dir) → data. A slot hit is 2 dependent directory loads plus 1 data load.
  - Ownership (`foCellOwnerH`, `foTileOwnerH`) is `nbPageWord` → `nbTileH` (CMASK), which is 2 more.
  - Every lane re-walks these for every tap. Nothing caches a slot across taps, and no table holds a brick's neighbours.
- **Address dependence alone gives short chains: 5–50 links per lane.**
  - The time per resident wave implies about 50–800 links.
  - The gap is control dependence. Each lookup helper branches on its own result (slot hit, owner bit, early return), and an in-order SIMT lane cannot issue the next tap's loads until that branch resolves.
  - The GPU run below measured that cost: per-tap guards were 1.9× slower, an early-exit loop 3.0× and a divergent owner select 3.6×, all against the same taps without branches.
- **Chain length explains the time for lAdvect, phiSolid, pMove, nbOperator, pSeed and lDistance.**
  - Their measured µs per wave falls inside [data bound, control bound] × per-link latency.
  - It does **not** explain uExtend, uTransfer, uAdvect or pDeposit. For these four, even the fully serialised chain is shorter than the measured time per wave. Their time is in workgroup-memory staging, barriers and ALU work at 1–2 groups per core.
  - censusDecide and lcRetire run only 3–5 waves. Long serial per-lane loops over C tiles at low lane counts dominate them.
- **Top 3 levers. These are estimates, not measured; the assumptions are under each lever below.**
  1. Look up each distinct brick's slot once, then load the stencil branch-free: levers (c) and (d). Estimate: −1.0 to −2.0 ms on the dam step.
  2. A per-brick neighbour table written once per step, holding the 27 neighbour slots and owner bits: lever (b). Estimate: −1.0 to −2.0 ms on the dam step. It overlaps lever 1 by about half.
  3. Hoisting the brick-uniform prologue and ring loads to one per simdgroup: lever (a). Estimate: −0.2 to −0.4 ms.

## Method

`tools/experiments/gpu-shape/chains.mts` is an abstract interpreter over the WGSL that inlines every call (WGSL has no recursion).

**What it models**
- Every value carries the time at which it is ready, counted in device-load latencies. ALU work is free.
- A device load is either:
  - an index into a storage binding, or
  - an atomic whose result is used.
- Workgroup memory counts as free.

**Two bounds per kernel**
- **data**: address dependence only. This is a lane that speculates every branch.
- **ctrl**: data, plus the rule that no load issues past an unresolved branch whose arms load or exit. Loop conditions gate the next iteration too.

**The common path**
- Conditions fold where a term is:
  - a bool constant;
  - a parameter bound to a constant at the call site;
  - a slot hit on a loaded value (`s != NB_NONE`, `s < NB_SEAM`);
  - an in-range guard;
  - an owner predicate in `--assume`. The default assumes the h-owned interior of a tiered F region.
- Rare paths are free via `--cold-fn`: nbFail, nbxTraceFatal, and the released-wall walk nbrReleasedWalls/MayChange.
- The surface trace runs one RK3 step (3 iterations). Other unknown trips run once, and strided `j = lid; j < N; j += S` loops run ⌈N/S⌉ times.
- Both arms of an unfolded branch run, and the slower arm sets the clock. This gives the slowest lane, which sets the workgroup's time.
- Calls are memoised shift-invariantly, so a 1500-line kernel takes 1–60 s.

**Limits**
- Where a kernel has several unfolded, mutually exclusive expensive arms, the ctrl bound stacks the worst arm everywhere. No real lane may take that path. This is the case for nbOperator, censusDecide, lcRetire and pSeed, so their ctrl bound is a loose upper bound.

**Measured time per wave** = t × (32 cores × resident groups per core) / G.
- t, G and L come from the census.
- Resident groups come from lane B's occupancy × 3072 lanes per core / L. The 3072 is an assumption; lAdvect's 8.4% gives 1.0 group, which matches cap 384 / L 256. phiSolid and pDeposit have no counter row, so their residency is ⌊cap/L⌋.

**Per-link latency** comes from the M0(a) ladder (m0a-tables.md):
- 121 ns at 16 KB, 141 ns at 128 KB, 270–290 ns at 1–32 MB, 456 ns at 256 MB.
- The hot directory is small: page table 2 KB, child table 32 KB, CMASK 8 KB on the dam, plus FREC key words.
- *Inference:* directory links cost about 0.12–0.14 µs and data links (pools of 10+ MB) about 0.28–0.45 µs.
- Implied links below use a blended range of 0.15–0.30 µs per link.

## Link costs of the shared helpers (lAdvect's module, dam tiering; links = dependent device-load latencies)

| helper | loads | data | ctrl | chain (common path) |
|---|---|---|---|---|
| `fgSlot` | 2 | 2 | 2 | nbPageWord → nbChild |
| `lPhi0` (FC_PHIU) / `fgPhi` (FG_PHI_RESOLVED) | 3 | 3 | 3 | page → child → fcell / phi. The slot-hit branch gates the caller's next tap. |
| `foCellOwnerH` / `fgTileHDir` | 2 | 2 | 2 | nbPageWord → nbTileH (CMASK) |
| `foVertexCoarse` (`fgVertexOwned`) | 28 | 2 | 4 | a slot test, then up to 8 owner tests with early return |
| `foSFaceU` (an owned face) | 7 | 3 | 5 | owner(c) ‖ owner(c−e), then slot, then vel |
| `fgVelAxis` (8 faces) | 192* | 3 | 51* | 8 × foSFaceU. Each face's owner branch gates the next. |
| `fgPhiAt` (8 corners) | 24 | 3 | 17 | 8 × fgPhi, serialised by each corner's slot-hit return |
| `nbfCubicLimited` (4³ taps) | 192 | 3 | 129 | 64 × lPhi0. 64 slot walks touch at most 8 distinct bricks. |
| `nbfSurfaceTrace` (one RK3 step) | 123 | 18 | 27 | 3 velocity samples, each on the previous stage's position |

\* As a standalone entry, `base` is unknown there, so all of fgFace's arms run. In the kernels `base` is constant and folds.

**Redundancy.** For one interface vertex, lAdvect does about 120 slot walks (nbChild) and 237 page-word loads. It reads at most 8 distinct bricks; the trace moves the stencil by ≤ 2 cells. So the same lookup repeats about 15–30× per lane, and every repeat sits on the control path.

## GPU validation: does a per-tap guard serialise independent lookups? (one run, measured)

`tools/experiments/gpu-shape/branchy.mts`; results in `results/m0e-branchy.jsonl`; run through `fluid-gpu-run.sh timing`, rc 0, 0 gpuErrors.

**Setup**
- Each lane runs serial rounds. A round is 16 independent taps of 3 dependent links each: dir 68 KB → dir → data 16 MB.
- Per-round time is the slope between 4 and 12 rounds, so the dispatch floor drops out.

| variant (L256 × G32, one group per core) | µs per round | × free |
|---|---|---|
| free: taps branch-free | 12.3 | 1.0 |
| guard: `if (child != NONE) {load} else {load}`, the miss arm never taken (lPhi0 / fgPhiWord) | 23.1 | 1.9 |
| exit: a loop with `if (page == NONE) {break}` (foVertexRead4) | 37.1 | 3.0 |
| owner: `if (bit) {load a} else {load b}`, 50/50 divergent (the owner selects) | 44.6 | 3.6 |
| serial: each tap's address takes the previous tap's value (48 links, the calibration) | 60.8 | 4.9 |

- The ordering is free < guard < exit < owner < serial.
- *Inference:* a guard per lookup halves the memory-level parallelism. An early exit loses two thirds of it. A divergent owner select, which runs both arms, is close to fully serial.
- The ctrl bound is therefore the closer model for code written as these kernels are. The data bound is what a branch-free rewrite could reach.
- The L32 × G1 rows (free 9.1, guard 4.0, exit 17.1, owner 4.1, serial 27.7 µs per round) are not self-consistent: free is slower than guard, and guard's R4 time exceeds its R12 time per round. This run had no clock probe, and these dispatches are tiny (30–300 µs), so DVFS dominates them. I do not use them.

## Per kernel (dam unless noted; µs/wave measured, links from chains.mts)

| kernel | t µs | G × L | groups/core | µs/wave | implied links (0.30–0.15 µs) | data | ctrl | explains? | dominant chain on the common path |
|---|---|---|---|---|---|---|---|---|---|
| lAdvect | 1955 | 2089 × 256 | 1.0 | 30.2 | 100–200 | 29 | 583 | **yes** | per vertex: lPhi0 c → 10-tap gradient → foot → trace: 3 velocity samples, each 8–24 face lookups → departure cubic, 64 × lPhi0. Repeated for 2 vertices per lane (loop line 1465: +290 per vertex). |
| phiSolid (seep) | 910 | 1265 × 256 | 1 (cap) | 23.0 | 77–153 | 50 | 1311 | **yes** | lAdvected plus psCubicBuried (250 loads) and psClosed (240 loads) on top of the lAdvect chain, 2 vertices per lane |
| pMove | 1216 | 2404 × 128 | 3.1 | 50.5 | 168–337 | 33 | 261 | **yes** | per particle: fgPhiAt (8 corners) → pSeparation (8 cells) → pSupported (8) → surface shift (k = 1..6 serial). Per-lane particle loop, trip unknown, run once. |
| nbOperator | 1167 | 727 × 512 | 1.0 | 50.3 | 168–335 | 17 | (stacked) | **yes** (likely) | 6 faces × poBandFace → poRootPhiG (5-iteration root find, each step a fresh corner gather: 850 ctrl links per call) |
| pSeed | 286 + 915 | 2089 × 256 | 1.55 | 28.5 (2 passes) | 95–190 | 17 | (stacked) | **yes** (likely) | pSeedCellMask: a serial j-loop × psSeam's 27-neighbour walk (22 per neighbour); fgCornersOf |
| lDistance | 923 | 2089 × 256 | 2.9 | 41.0 | 137–273 | 11 | 191 | partly | 2 vertices × 8-corner fgCornersOf + per-candidate loop. ALU limiter 57% (lane B), so ALU shares the bound. |
| uTransfer | 1594 | 2197 × 256 | 1.1 | 26.5 | 88–177 | 6 | 65 | **no** | staging (window, ring, 729-cell depth, 125 tiles) + 2 cells × 3 axes deposit. 220 of 354 loads are lane-uniform nbRing reads. |
| uExtend | 1631 | 2089 × 512 | 1.3 | 33.1 | 110–221 | 7 | 28 | **no** | 3 axes × (seed → wgmem sweep xSweepAt → write) with 5 barriers per axis. Device chains ≤ 28 links; the time is the wgmem sweep and barriers. |
| uAdvect | 1207 | 2197 × 256 | 2.0 | 34.8 | 116–232 | 11 | 55 | **no** | staged (wgmem 11 KB) trace; 70 of 275 loads lane-uniform |
| pDeposit | 692 | 2089 × 256 | 3 (cap) | 31.8 | 106–212 | 5 | 22 | **no** | per-particle scatter into wgmem (27.5 KB) atomics; device chain 5 per particle |
| censusDecide | 1108 | 880 × 64 | 2.9 | 118 | 393–787 | 14 | (stacked) | yes, serial | cdTile: ncSurfaceQuiet / ncCurveShift (3 × ~560 ctrl links) over the C tile's ncTileNormal 8-corner gathers. Only 9.4 waves. |
| lcRetire (paddleboats occupancy) | 702 | 864 × 64 | 5.9 | 152 | 507–1013 | 1 | (stacked) | yes, serial | lcRetirePage: lcBoundMetric 7 × 345 and lcBuildDistance 5 × 277 serial loops of 8-corner lcSample gathers |

**How to read "explains?"**
- **yes**: implied links fall inside [data, ctrl]. The GPU test says real code sits near ctrl/2–ctrl/3 when guards dominate.
- **no**: the fully serialised device chain (ctrl × 0.30 µs) is shorter than µs/wave:
  - uTransfer: 65 × 0.30 = 19.5 < 26.5
  - uExtend: 8.4 < 33.1
  - uAdvect: 16.5 < 34.8
  - pDeposit: 6.6 < 31.8
- For the "no" kernels, shortening chains cannot remove most of their time. *Inference:* their time goes to staging loops, barriers, workgroup-memory work and ALU at 1–2 resident groups (brick-register-bound note).
- censusDecide and lcRetire are serial in a different sense. 55k lanes each, ≤ 10 waves, and per-lane loops of hundreds of links. The GPU is under-filled, and their lever is more lanes per C page, not shorter chains.

**Brick-uniform prologue (lever a).**
- Every F kernel starts with `nbListCount` → `nbListSlot` → `nbFRec(KEY)`, which is 3 links before any lane-dependent load.
- uTransfer, lAdvect and uAdvect also read 39–216 lane-uniform `nbRing` words.
- All of this repeats in every lane: the same address per workgroup, but on every lane's issue path.

## Levers, ranked (all savings are estimates)

**1. Look up the distinct bricks once, then gather branch-free: levers (c) and (d).**
- A stencil (64-tap cubic, 8 corners, 8 faces, 27 neighbours) touches at most 8 bricks.
- Compute the ≤ 8 slots and owner words first. They are independent, so they issue together; that is 2 links.
- Then do the taps as branch-free loads off those slots, with a select for the miss.
- This removes the 15–30× redundant walks and the per-tap guard serialisation (1.9–3.6× in the GPU test).
- Targets: nbfCubicLimited, lCorners0/fgCornersOf, fgVelAxis, poRootPhiG's corner gathers, psSeam, lcSample.
- Estimate: lAdvect 1.96 → about 1.0 ms. phiSolid takes a similar share on seep. pMove, lDistance, nbOperator and pSeed save −0.1 to −0.3 each.
- Total −1.0 to −2.0 ms dam, *assuming* time scales with ctrl links and the cubic/corner loops are about half of each chain.

**2. A per-brick neighbour table, written once per step: lever (b).**
- For each F slot, store its 27 neighbour slots and the owner bits of the tiles around it (`nbTileH`) in one record of about 32 words.
- Every `fgSlot` then becomes 1 link (record[nIdx]) instead of 2, and `foCellOwnerH` becomes a bit test on a word in the same record instead of 2 links.
- The record is lane-uniform per workgroup, so it stays hot in L1.
- This shortens every remaining lookup from 3 to 2 links and owner tests from 2 to 1. It also helps the "no" kernels' staging loops a little.
- Estimate: −1.0 to −2.0 ms dam across the chain-bound kernels (6.5 ms of chain-bound time × one quarter to one third of links). About half of it overlaps lever 1. Directory links are the cheap ones (L1-resident, *inference*), so the low end is more likely.

**3. Hoist brick-uniform chains to one lane per simdgroup and broadcast with subgroup ops: lever (a).**
- This covers the prologue (3 links) and the uniform ring words.
- Do not use workgroup memory for this: it has a measured +45% tax on heavy kernels.
- Estimate: 0.5–1 µs per wave, or 1–3% of each kernel. That is −0.2 to −0.4 ms over these twelve.
- It is cheap to do and composes with levers 1 and 2.

**Not a lever here.**
- Software prefetch, lever (c) in its narrow form, is subsumed by lever 1. The loads to issue early are exactly the distinct-brick slot walks.
- For uExtend, uTransfer, uAdvect and pDeposit, chain work does not reach their time. Their limits are residency, barriers and staging cost.

## Caveats

- **The common path is a model.**
  - It assumes the dam's h-owned interior: owner predicates true, slots hit, released walls cold, a one-step trace.
  - Lanes at tier boundaries (the 4h interpolant paths, foVertexPhi) and on released walls run longer chains.
  - Particle loops (pMove, pDeposit, pSeed) count one particle per lane.
- **The per-link latencies are unloaded single-chain figures.** Under the kernels' 2–22 GB/s load, queueing adds little (*inference*, from lane B's ≤ 8% buffer-read limiter).
- **The GPU validation is one run with no clock probe.** Its ratios at L256 × G32 are consistent, but they are a single sample. The L32 rows were discarded as clock-dominated.

## Reproduce

```
node --import tsx tools/experiments/gpu-shape/chains.mts ~/code/me/fluid-scratch/gpu-shape/kcache/<hash>.wgsl <entry> \
  --cold-fn=nbFail,nbxTraceFatal,nbrReleasedWalls,nbrReleasedMayChange[,vcTransferPage] --trip=1304:3,1427:3 [--data]
```

- A helper name as `<entry>` gives that helper's own cost.
- `--all-arms` disables the common-path folding. `--assume=` replaces the owner assumptions.
- The GPU validation:

```
tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/branchy.mts <out.jsonl>
```
