# GPU-shape M3: the FC passes' C pages, critical path and reshape (lane B2, 2026-11-24)

**Verdict: NO-GO at the 0.5 ms line.** The C pages of uTransfer and uAdvect *are* the critical-path tail of their FC
dispatches: launched last, a few long page workgroups run alone for 0.27–0.48 ms per kernel. Running them first hides
uTransfer's tail exactly, but the best exact reshape nets only:

| | dam | paddleboats |
|---|---|---|
| best net per step | **−0.57 ms** (uTransfer first2 −0.50, uAdvect app4 −0.08) | **−0.30 ms** (uTransfer first2 −0.27, uAdvect app2 −0.04) |

Paddleboats misses the line. uAdvect does not respond to reordering, and splitting cannot get below the cost of one
lane's face.

The uTransfer reorder is a sub-threshold option for the coordinator: an exact ~10-line text change worth 0.25–0.50 ms.
The larger lead is the **C-list passes**: 35 placements that sum to 4.9 ms (dam) and 8.0 ms (paddleboats), some of them
single-group tails.

Plan: docs/plans/gpu-shape-2026-11-23.md (M3). Tool: tools/experiments/gpu-shape/cpages.mts. Data:
docs/plans/gpu-shape/m3/<scene>/cpages.json (the run's HEAD and `git status`, every sample), selected per-workgroup
clock traces (clock-*.tsv), caps.txt. Run logs are in ~/code/me/fluid-scratch/gpu-shape/m3-*.log (gitignored).

## Runs

- Every GPU job ran through `tools/gpu/fluid-gpu-run.sh timing` at HEAD c1d8c24e. lib/ is unchanged since 734dd9e2, M2's runs.
- All runs returned rc 0 with no GPU errors.
- Frozen step 100, census settings, the same harness as M2.
- dam: crit, shape and clock. paddleboats: crit, shape, clock (plus the C-list placements) and step.
- In-situ samples: min of 7, interleaved, base in the same rounds.
- Clock probe after every round:
  - dam: 76.3–76.7 µs.
  - paddleboats: 76.2–78.7 µs, except one 491 µs shape round and two 408–410 µs step rounds (low clock). The minima come from top-clock rounds.
- Other sessions' uncommitted files, as in m1b-staged.md, do not touch the brick kernels.

## FC and C-list placements in the step (measured: groups read from the args words in situ)

FC placements are F_ALL's bricks then C pages, one workgroup each:

| placement | dispatch | lanes | dam groups (nF + nC) | paddleboats groups |
|---|---|---|---|---|
| resMark (p1) | FC_G1 | 64 | 1972 + 99 | 1678 + 274 |
| nrReach (p1) | FC_G1 | 256 | 1941 + 98 | 1679 + 274 |
| vGauge | FC_G1 | 64 | 1941 + 98 | 1679 + 274 |
| uAdvect | FCA_G1 (C_ACT) | 256 | 1941 + 98 | 1679 + 274 |
| uTransfer | FCA_G1 (C_ACT) | 256 | 1941 + 98 | 1679 + 274 |

C_ALL equals C_ACT on both scenes (no STILL pages at step 100).

C-list placements: 35 per step, on C_ALL or C_ACT. G1 is a workgroup per page; X8 is eight per page; pressure runs
512 lanes. Their full times are in §4.

## 1. Critical path: full / F-only / C-only (µs, in situ, min of 7)

- fonly: the C groups return at entry.
- conly: the F groups return at entry (lane B's `0-cpages`).

| kernel | dam full | fonly | conly | full − fonly | paddle full | fonly | conly | full − fonly |
|---|---|---|---|---|---|---|---|---|
| uTransfer | 1778 | 1296 | 588 | **482** | 1708 | 1307 | 476 | **402** |
| uAdvect | 1151 | 855 | 395 | **296** | 1263 | 991 | 420 | **272** |
| vGauge | 249 | 242 | 55 | 7 | 231 | 211 | 68 | 20 |
| nrReach | 113 | 93 | 45 | 20 | 150 | 129 | 52 | 22 |
| resMark | 88 | 71 | 50 | 17 | 85 | 61 | 50 | 24 |

- **uTransfer and uAdvect: the C work is a tail.** full ≈ F-only + 0.27–0.48 ms, with only ~0.1 ms of the C-only time
  overlapped.
- vGauge, nrReach and resMark: hidden or nearly so (7–24 µs). Stop there for those.

### Per-workgroup clock (measured)

Method: one extra workgroup (group 0, launched first) increments a device atomic. It stops when every payload group has
finished, and is bounded so it cannot hang. Each group records the tick count and an entry/exit ticket at its start and
end; ticks are calibrated against the pass timestamps.

The clock costs +0.1 ms on uTransfer but +0.5 ms on uAdvect, so absolute clocked times are indicative only. The shape
of the dispatch is what these traces show.

uTransfer:

| | dam | paddleboats |
|---|---|---|
| groups in flight while F runs | 58–63 | ~60 |
| F groups end at | 1373 µs | 1256 µs |
| C pages start / end at | 1180 / 1827 µs | 1144 / 1747 µs |
| C tail after F ends | 454 µs | 491 µs |
| C duration p50 / p90 / max | 77 / 151 / **616** µs | 41 / 120 / 359 µs |

- On dam, groups in flight in the tail fall to 9, 5, 2, 1: two pages dominate, at 616 and 475 µs (the next is 304).
- Slot-time (sum of group durations) is small for C. On dam it is 8.0 ms·slot, against 73 ms·slot for F.

uAdvect:

| | dam | paddleboats |
|---|---|---|
| C tail | 631 µs (clocked) | 484 µs |
| C max | 448 µs | 247 µs |

Diagnosis (measured shape, inferred cause): the FC passes hold ~60 groups (about 2 per core). The C pages are the last
group indices, so they launch once F is nearly drained. The longest single page then runs at a few groups in flight,
and that page is the critical path. The C path's total work is not the problem.

## 2. Shape of the C path (from source)

**Counts.**
- G_C = C_ACT count: 98 on dam, 274 on paddleboats, against nF 1941 / 1679.
- One 256-lane workgroup per page, sharing the F path's workgroup size.
- A page is 512 tiles (8³ tiles of 4h, 32³ h cells).
- Each lane loops serially over **2 tiles × 3 faces = 6 face items**. That is the same item count as F's 2 cells × 3
  faces, but each C item is much heavier, and the C path stages no owner window (fgWinOn stays false), so every lookup
  walks the directory.

**uTransfer C face** (vcTransferPage → vcTransferFace). Links are counted as in m0e-chains.md:

| step | lookups | links |
|---|---|---|
| vcFace | 2 owner walks (page word → CMASK), short-circuit | 4, serial |
| face velocity | vel load | 1 |
| vcFar | 3 × fgReach1 (page → child → FC_TILE), short-circuit | 9, serial |
| vcTransferFace (near faces only) | fgDepth (reach + 8 depth vertices) | — |
| | **4×4×4 = 64 cells × fgRun** (page, child, BIN_COUNT, BIN_START, FC_RUN: ~5 links), then that cell's particles (1 load each, control-serial) | 64 × (~5 + particles) |
| | 2 × vcHeat (8 tiles × 3 links) | — |
| vOcc4(O), vOcc4(T) and vCoarseMass | 8 C-vertex loads each | 2 each |

- chains.mts control bound on the page body: 873 load latencies with every loop run once, and **3915** with 4-trip
  cell loops, 4 particles a cell and all six faces near.
- A far face costs ~20 links; a near face ~400–1000. The cost sits in the few near-seam faces, which matches the
  measured skew: two pages carry the tail.

**uAdvect C** (vcAdvectPage):
- Classify: vLive4 of T and of T − e_a, 8 C corners plus the centre each.
- Then the compacted trace list, ≤ 1536/256 = 6 traces a lane, each an nbfMomentumTrace at width 4.
- chains.mts gives 25 latencies, but it runs each trace's loops once, so this understates the traces.
- Measured (app8: ≤ 1 trace a lane): sub-page groups of 60–112 µs, so **one 4h trace costs ~0.1 ms** (inferred).

## 3. Reshape prototypes (no lib/ edits; text variants of the schedule's kernels)

The variants:
- **first**: the C groups go before the F bricks (k = 1).
- **appK / firstK**: each page is split over k = 2, 4, 8 workgroups (sub-blocks of 512/k tiles), with the C groups
  appended (appK) or first (firstK).
- uTransfer k ≥ 4: a lane per (tile, axis) item; the tile's gravity-kick bits are joined in workgroup memory (atomicOr)
  and its flags word is written after a barrier.
- uAdvect: sub-block classify; each workgroup keeps its own trace list.
- Dispatch: nF + k·nC from an untimed one-lane args kernel.

**Exactness: bit-identical.** Every variant of both kernels on both scenes differs in 0 words, compared over every
read-write binding (uTransfer: set, vel, phi, fcell, fdep; uAdvect: set, vel). Each check starts from one in-situ
snapshot and includes a base-vs-base control.

**Thread caps** (Metal, size attribute stripped; caps.txt):

| kernel | base | variants |
|---|---|---|
| uTransfer | 512 | app2 576, first2 576, others 512 |
| uAdvect | 768 | app* 768, first* **704** |

**Timing** (µs, in situ, min of 7; Δ against base in the same rounds):

| variant | dam uTransfer | dam uAdvect | paddle uTransfer | paddle uAdvect |
|---|---|---|---|---|
| base | 1731 | 1066 | 1701 | 1309 |
| first | −373 | +71 | −251 | +241 |
| app2 | −189 | −3 | −193 | **−36** |
| app4 | −152 | **−78** | +27 | +43 |
| app8 | −211 | −58 | +76 | +184 |
| first2 | **−495** | +80 | **−265** | +224 |
| first4 | −315 | +84 | −18 | +346 |
| first8 | −293 | +167 | +37 | +578 |

**uTransfer first2 hides most of the tail.** On dam it reaches F-only (1236 µs against fonly 1296). On paddleboats it
stays 0.13 ms above (1436 against 1307), because 274 pages also take slots from F. F's span is ~1.3 ms.

The clocked sub-block maxima show why splitting alone fails:
- dam uTransfer's C maximum goes 616 → 546 → 346 → 244 µs for k = 1, 2, 4, 8. That is not 1/k.
- app8, at one face item a lane, still has a 230 µs group: **one near face's 64-cell gather is ~0.2 ms**.
- Appended, any split leaves that lane as the tail. Only ordering it first hides it.

uAdvect does not pay to reorder:
- With its C pages first, they hold ~52 of the ~60 slots for the first 0.1–0.3 ms. F starts late (158 µs on
  paddleboats) and ends later, so the pass is slot-bound (inference). F's slot-time / 60 slots ≈ 1.0 ms, and C's
  slot-time adds to it about as much as the tail removes.
- Finer splits add groups and slot-time (paddleboats first8 +578).
- Its floor is one 4h trace (~0.1 ms) a lane, which no exact split divides.

**Net per step, best per scene:**

| | uTransfer | uAdvect | total |
|---|---|---|---|
| dam | first2 −0.50 | app4 −0.08 | **−0.57 ms** |
| paddleboats | first2 −0.27 | app2 −0.04 | **−0.30 ms** |

- One consistent choice, uTransfer first2 + uAdvect app2, gives −0.50 / −0.30 ms.
- Whole step on paddleboats, first2 + app4: −0.27 ms by min, −0.22 ms by median. Two of the 7 rounds ran at the low
  clock, so the wall does not resolve this; the in-situ figures are the gate.
- **Below the 0.5 ms line on paddleboats: no landing design.**

The one cheap exact win: uTransfer's C pages first, k = 1 (−0.25 / −0.37 ms) or k = 2 (−0.27 / −0.50 ms). It needs
only the stagedHead group order plus, for k = 2, the tile loop's sub-block. It is below the line; it is the
coordinator's call whether to land it on its own.

## 4. The larger lead: C-list passes (measured)

Sum of the 35 C-list placements: **4.89 ms dam, 7.96 ms paddleboats**. Full times (µs), dam / paddleboats:

| placement | dam | paddleboats |
|---|---|---|
| censusDecide (X8, 64 lanes) | 979 | 1750 |
| lcAdvect | 631 | 1133 |
| lcRetire | 682 | 840 |
| uExt4Seed | 395 | 720 |
| censusCoarse | 279 | 724 |
| pressureRowsC | 323 | 465 |
| uExt4 | 269 | 306 |
| certError / certCoarse | 102 / 140 | 250 / 230 |
| pressureProjectC | 133 | 245 |

Clocked on paddleboats:

| placement | span | group durations | reading |
|---|---|---|---|
| lcRetire | 944 µs | one group of **926 µs** | a single-group critical path |
| uExt4Seed | 715 µs | max 444, p50 48 µs | tail |
| pressureRowsC | 413 µs | max 196, p50 5 µs | tail |
| lcAdvect | — | p50 113, max 287 µs | — |
| censusDecide | — | 2192 groups, uniformly long (p50 237, p90 257 µs) | throughput at 64 lanes |

These are the "few long workgroups" pattern that the FC C pages showed, at larger total cost. The next lane could
start with lcRetire's longest page, uExt4Seed and pressureRowsC (attribution first), and with censusDecide's per-group
serial loop.

## 5. uExtend (diagnosis only, measured shape)

- One 512-lane group per F brick: 1941 groups (dam), 1679 (paddleboats).
- Groups in flight hold **26–32 all pass long**. With 32 cores that is **one group per core**.
- Durations are even: p50 20 / 38 µs, p90 50 / 48, max 73 / 63. The pass ends evenly, with in flight still ~30 until
  the last bins.
- So uExtend is **not a tail of a few long workgroups**. It is bound by residency of one group per core.
- Why one per core:
  - The 768-thread cap admits only one 512-lane group per core; two would need 1024.
  - Its ~23 KB of workgroup memory, wV/wD 13.8 KB plus wS4 6 KB, would bound it too if the per-core limit is 32 KB
    (inference).
- Each of its ~17 barriers (5 in the prologue plus 4 a sweep × 3 axes) therefore idles the core, because no other group
  is resident to switch to. The `nobar` cut's −0.22 / −0.29 ms (m1b-staged.md) measures that.
- A reshape would need two groups per core: ≤ 384 lanes, or a lower register cap plus < 16 KB of workgroup memory. Not
  prototyped.
