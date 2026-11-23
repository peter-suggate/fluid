# GPU shape M0d (lane B): slot order go/no-go

Date: 2026-11-23. Tool: `tools/experiments/gpu-shape/slot-order.mts`, run through `tools/gpu/fluid-gpu-run.sh timing`.
Data: `docs/plans/gpu-shape/m0d/<scene>/` (slot-order.json, counters.tsv).

## Result: no-go

The remap is complete. Renumbering live slots into a dense range (compact), or into Morton order (morton), does what it should to locality: morton cuts the TLB miss rate of the heavy kernels by 1.5–4×. It does not move their time. Every heavy kernel lands within ±4% of identity, with both signs, which is the noise of this measurement. Neither variant reaches the ~10% go line on either scene. Inference: the heavy kernels hide their TLB misses (the MMU limiter stays at 1–5% in every variant), so slot order is not a lever on step time.

| scene | wall min, identity → compact / morton | heavy-12, identity → compact / morton |
|---|---|---|
| dam (run 2) | 31.09 → 30.89 (−0.6%) / 30.77 (−1.0%) ms | 14.59 → 14.56 (−0.2%) / 14.43 (−1.1%) ms |
| dam (run 1, another frozen state) | 30.07 → 29.74 (−1.1%) / 29.54 (−1.7%) ms | 14.32 → 14.34 (+0.2%) / 14.15 (−1.2%) ms |
| paddleboats | 33.86 → 33.57 (−0.8%) / 34.01 (+0.4%) ms | 17.01 → 16.95 (−0.4%) / 17.01 (0.0%) ms |


## Method

1. Warm the brick engine up 100 steps at a fixed dt of 1/60 (the census's warm-up; paddleboats with the moored motor bodies), then `save()` it.
2. Build four snapshots of that frozen state on the CPU and upload each with `write()`:
   - **identity**: unchanged. It is the noise baseline.
   - **compact**: live F and C slots renumbered 0..live−1 in their current relative order. Free slots follow in their old order.
   - **morton**: live F slots renumbered in the Morton order of their brick key, and live C slots in the Morton order of their page key.
   - **scatter**: a seeded random bijection of all F and all C slots. It is the hardest case for the proof and is never timed.

   A slot is live when its record key (word 29) is not NONE. For compact and morton, the free stacks are rebuilt as `FREE[i] = cap−1−i` with `HIGH = live` and TOP unchanged, so the next pop takes slot `live`. For scatter, the old free stack is carried through π. K and B slots are not renumbered: B is unused (LIVE_B = 0 is asserted), and K is reached only through F record word 30.
3. **Proof.** From each snapshot, run the engine's own recorded step (its pipelines, bind groups and ring, as census.mts) for 1, 2 and 4 steps. Then read observables keyed by brick or page key, never by slot:
   - the live counts and the FAIL word;
   - per F brick and per C page: the φ sum, the liquid-vertex count (φ < 0) and the U0–2 sums;
   - per bank: the particle count and the float64 position sums.

   Identity is re-run from the same snapshot to give the run-to-run noise floor.
4. **Timing.** census.mts's two measures, with the three variants interleaved over R = 5 rounds and the minimum taken per measure:
   - wall: one timestamped pass per part;
   - replay: each placement alone in a timestamped pass.

   Before every measurement the variant's GPU shadow is restored.
5. **Counters.** The child cycles the three variants' replay passes, labelled `m0d.<variant>|<entry>`, while Instruments' Metal GPU Counters is attached for 4 s. The trace is reduced with `tools/xctrace-hardware-limiters.ts` and aggregated per variant and entry, weighted by time.
6. **Locality.** For each snapshot:
   - over F_ALL: the slot spread of each brick's 27-neighbourhood (max − min slot), and the distinct 16 KB pages that the neighbourhood's channel-0 pool rows touch;
   - the same for C_ALL pages, using their page-window neighbours.

## Remap inventory

Every buffer save() returns is either remapped or kept. A buffer in neither list throws. Each remapped region's computed extent is checked against the buffer, and each buffer's word count is checked against its layout (`fluidLayout`, `mgLayout`, `CENSUS.STRIDE`, `brickStaticNeedWords`, `BrickLayout`). "Positional" means the slot's words move to its new slot. "Values" means words that hold a slot number are mapped through π.

Before remapping, these checks are asserted on the frozen state:
- LIVE_F and LIVE_C equal the counts of keyed records;
- TOP + live = cap, and the free stack is exactly the complement of the live slots;
- F_ALL and C_ALL equal the keyed slots;
- HIGH covers every live slot;
- LIVE_B = 0.

Any slot-valued word that is ≥ cap and not NONE is counted as an anomaly. There were 0 on both scenes.

| buffer | region | slot kind, words/slot | positional | values remapped |
|---|---|---|---|---|
| vel, phi, pres | F channels, then C channels (channel-major, 520/slot) | F, C | yes | — |
| cut | channels and BMASK | K | kept | — |
| dir | page window | lattice | — | C slot (bit 31 clear); BULK/NONE kept |
| dir | child | C, 64 | yes | F slots |
| dir | frec | F, 32 | yes | words 0–26 by kind (F slot, SEAM\|C, BULK kept); word 27 parent C; 28–31 kept (mask, key, K slot, class) |
| dir | crec | C, 32 | yes | words 0–26 asserted NONE |
| dir | cmask | C, 16 | yes | — |
| dir | closed, inexact, dtl | lattice planes | kept | — |
| set | header | — | — | TOP/HIGH F,C rebuilt; LIVE and counts kept |
| set | ring, cand (keys) | — | kept | — |
| set | freeF, freeC | stack | rebuilt (see method) | — |
| set | freeB, freeK | stack | kept | — |
| set | markF, boundF (6), binStart, binCount, binCursor, pLiquid (16), tileOcc, trig, psupport (8), act | F | yes | bins hold particle record indices: kept |
| set | markC, boundC, want (16), airGate (16), reach0/1 (512), reachStep, wallTile (384), candMask (8) | C | yes | — |
| set | lists 0–18 (all capacity) | — | — | F or C slots by listKind; B_ALL kept |
| set | tlist (2/entry) | — | — | word 0 `(cslot<<9)\|tile` |
| census | CENSUS.STRIDE = 20280 per page | C | yes | keys only |
| staticNeed | 4/C, 4 global, 32/C | C | yes | — |
| fcell | CNT, RUN, NEAR, NEAR2, KEY, HOT, DEPTH (512), MASK (16), TILE (96), SLOT, BOUND, LIQ (16), UNIT (2187), PHIU (512), HMASK (8), RES F words | F | yes | NEAR is a packed cell and SLOT a key + 1: kept |
| fcell | CACHE4 (1536), DIL4 (512), RES C words | C | yes | — |
| fdep | 3072 | F | yes | — |
| fext4 | 9216 | C | yes | — |
| fpart, fmoved, pEnergy, parts, xfer | per particle or per tlist entry | — | kept | — |
| pmgF | W0, W1 (520), L1 (10 ch × 64), RKU (1560), FC (3640), DRY | F | yes | — |
| pmgF | STG (81) | F | yes | 0–26 F slots, 27–53 C slots (f32, −1 none) |
| pmgF | HBL, HPL, ZLL, RLL | lists | — | F slots (f32) |
| pmgC | L2 (15 ch × 512), L3/L4/L5 (10 ch × 64/8/1), PDELTA (8), PH (2 parity × 512), SEAMT (512), SEAML (3072), SEAMN | C | yes | — |
| pmgC | SEAMF (3072) | C | yes | F slots (f32) |
| pmgC | QSTG (27) | C | yes | C slots (f32) |
| pmgTop | IDX5 | C | yes | values compact indices: kept |
| pmgTop | SLOT5 | compact index | — | C slots |
| pmgTop | RB / SB | list | — | F slots / `(slot<<2)\|k` |
| pmgTop | RBF, ZK, ZS, FS | F | yes | — |
| pmgTop | ZC | F | yes | parent C slot |
| pmgTop | PHW (2 parity × 2) | C | yes | — |
| pmgTop | PAR5..R5, level 6/7, INV, counts | compact index | kept | — |
| pmgRed | per-workgroup reductions, rewritten each step | — | kept | — |
| pmgB | body exchange | K / body | kept | — |
| args, solid, solidEdit, src, boot | no slots | — | kept | — |
| bodies, bodyCuts, bodyExchange (external) | per body | — | kept (restored with the shadow) | — |

## Proof that the remap is complete

Every permuted snapshot, run one recorded step, agrees with identity by key at the level of float-reordering noise. This holds on both scenes and for every variant, including the random scatter. In detail, after step 1:
- the same F and C key sets and the same live counts, with FAIL = 0;
- the per-brick φ sums are **bit-equal**, and so are the liquid-vertex counts;
- the particle counts and float64 position sums of both banks are **bit-equal**;
- the largest per-brick U-sum difference is 2.7–6.2e-5 on dam, against 4.1–5.6e-5 between identity runs. On paddleboats it is 2.0e-6–1.1e-4, against 1.05e-4 between identity runs.

One step reads every structure the previous step left (records, lists, bins, histories, certificates, staged neighbour words, the free stacks through any allocation it makes). A wrong or missed slot reference would show up there as a wrong key, a lost brick or a moved φ. None did. No slot-valued word anywhere was out of range (0 anomalies).

After 2 and 4 steps, the permuted runs stay inside identity's own run-to-run spread. The engine is not deterministic from a fixed state: particle ranks come from atomics, and identity re-run from the same snapshot already differs by tens of particles at step 2 and by per-brick φ sums up to ~380 at step 4. Occasionally a few fully liquid deep bricks (neg = 512) are kept or released one step apart. That happened in compact and scatter runs, and equally in an identity re-run: dam run 2, identity3, step 4, bricks (12,1,0) and (12,1,1). It is a threshold decision of the census flipping under that noise, not a remap gap.

| scene, step 1 | F keys | φ sums | liquid vertices | particles (both banks) | max per-brick ΔU sum, permuted (identity re-run) |
|---|---|---|---|---|---|
| dam run 1 | 2053 equal | bit-equal | 335954 equal | 298208 / 296135, bit-equal positions | 2.7e-5 – 4.7e-5 (4.1e-5) |
| dam run 2 | 2072 equal | bit-equal | 332437 equal | 309687 / 306826, bit-equal positions | 5.1e-5 – 6.2e-5 (5.5e-5) |
| paddleboats | 1689 equal | bit-equal | 413188 equal | 163121 / 163432, bit-equal positions | 2.0e-6 – 1.1e-4 (1.05e-4) |

The equivalence-only runs (logs in `~/code/me/fluid-scratch/gpu-shape/m0d-eq*-dam.log`) gave the same result on two further frozen dam states.

## The permutation did what it should

The figures below are over F_ALL bricks (C_ALL pages for the C rows). "Spread" is max − min slot over a brick's 27-neighbourhood. "Pages" counts the distinct 16 KB pages that the neighbourhood's channel-0 rows of 2080 B touch.

dam (run 2: F live 2113 of 4913, C live 105 of 125):

| | identity | compact | morton | scatter |
|---|---|---|---|---|
| HIGH_F | 3604 | 2113 | 2113 | 4913 |
| HIGH_C | 125 | 105 | 105 | 125 |
| F slot spread p50 / p90 | 3214 / 3485 | 1864 / 2049 | 294 / 1192 | 4505 / 4798 |
| F 16 KB pages per neighbourhood, mean / p90 | 19.8 / 28 | 19.1 / 26 | 7.9 / 11 | 22.9 / 30 |
| C slot spread p50 / p90 | 89 / 120 | 81 / 102 | 58 / 65 | 112 / 124 |
| C 16 KB pages per neighbourhood, mean / p90 | 7.6 / 10 | 7.4 / 10 | 5.7 / 8 | 11.6 / 14 |

paddleboats (F live 1688 of 10240, C live 274 of 384; C was already dense):

| | identity | compact | morton | scatter |
|---|---|---|---|---|
| HIGH_F | 7122 | 1688 | 1688 | 10236 |
| HIGH_C | 274 | 274 | 274 | 384 |
| F slot spread p50 / p90 | 2910 / 6144 | 703 / 1415 | 160 / 445 | 9431 / 9971 |
| F 16 KB pages per neighbourhood, mean / p90 | 11.8 / 16 | 11.8 / 16 | 8.0 / 12 | 23.8 / 31 |
| C slot spread p50 / p90 | 184 / 242 | 184 / 242 | 52 / 94 | 346 / 371 |
| C 16 KB pages per neighbourhood, mean / p90 | 8.5 / 12 | 8.5 / 12 | 5.5 / 8 | 16.1 / 20 |

- Compact shrinks the occupied range: HIGH falls from 3604 to 2113 on dam and from 7122 to 1688 on paddleboats. It barely changes pages per neighbourhood. A neighbourhood's 27 rows are scattered across a range of ~2000 slots, which is far more than 16 KB / 2080 B ≈ 8 slots, and compaction keeps that relative scatter.
- Morton is what packs neighbourhoods. It cuts F pages per neighbourhood by about 2.5× on dam and 1.5× on paddleboats.

## Step and kernel times

Each figure is the minimum over R = 5 interleaved rounds. Each placement runs alone in a timestamped pass, summed per entry. The heavy-12 set is identity's top 12 entries.

dam (run 2):

| | identity | compact | morton |
|---|---|---|---|
| wall min (ms) | 31.09 | 30.89 (−0.6%) | 30.77 (−1.0%) |
| wall median (ms) | 31.48 | 31.05 | 30.96 |
| sum of placements (ms) | 31.48 | 31.24 | 31.11 |
| heavy-12 (ms) | 14.59 | 14.56 (−0.2%) | 14.43 (−1.1%) |
| lAdvect | 1.957 | 1.932 (−1.3%) | 1.926 (−1.6%) |
| uTransfer | 1.646 | 1.664 (+1.1%) | 1.662 (+1.0%) |
| uExtend | 1.592 | 1.567 (−1.6%) | 1.545 (−3.0%) |
| pSeparate | 1.291 | 1.301 (+0.8%) | 1.275 (−1.3%) |
| pMove | 1.214 | 1.222 (+0.7%) | 1.221 (+0.5%) |
| nbOperator | 1.139 | 1.125 (−1.3%) | 1.095 (−3.9%) |
| pSeed | 1.124 | 1.093 (−2.7%) | 1.113 (−0.9%) |
| uAdvect | 1.097 | 1.129 (+2.9%) | 1.103 (+0.6%) |
| censusDecide | 0.941 | 0.970 (+3.0%) | 0.957 (+1.7%) |
| censusFine | 0.928 | 0.919 (−1.0%) | 0.920 (−0.8%) |
| lDistance | 0.916 | 0.907 (−1.0%) | 0.891 (−2.8%) |
| pmgUpdate | 0.741 | 0.734 (−0.9%) | 0.726 (−2.1%) |

paddleboats:

| | identity | compact | morton |
|---|---|---|---|
| wall min (ms) | 33.86 | 33.57 (−0.8%) | 34.01 (+0.4%) |
| wall median (ms) | 33.90 | 33.76 | 34.18 |
| sum of placements (ms) | 34.13 | 33.98 | 34.55 |
| heavy-12 (ms) | 17.01 | 16.95 (−0.4%) | 17.01 (0.0%) |
| nbOperator | 2.437 | 2.430 (−0.3%) | 2.470 (+1.3%) |
| lAdvect | 2.425 | 2.356 (−2.8%) | 2.409 (−0.7%) |
| uExtend | 1.912 | 1.899 (−0.7%) | 1.889 (−1.2%) |
| censusDecide | 1.719 | 1.717 (−0.1%) | 1.725 (+0.4%) |
| uTransfer | 1.697 | 1.686 (−0.6%) | 1.714 (+1.0%) |
| uAdvect | 1.330 | 1.274 (−4.2%) | 1.286 (−3.2%) |
| lcAdvect | 1.153 | 1.206 (+4.6%) | 1.131 (−1.9%) |
| censusFine | 0.978 | 0.977 (−0.1%) | 0.979 (+0.1%) |
| lcRetire | 0.977 | 0.981 (+0.4%) | 1.053 (+7.7%) |
| bodyStamp | 0.869 | 0.902 (+3.7%) | 0.857 (−1.5%) |
| project | 0.792 | 0.792 (0.0%) | 0.787 (−0.6%) |
| lDistance | 0.722 | 0.729 (+0.9%) | 0.714 (−1.2%) |

## Counters

Metal GPU Counters, aggregated over the replay loop and weighted by time per entry. Each pass in the loop has its own timestamp writes, which keeps it in its own Metal encoder. Without them, Dawn merges the passes and only one interval comes back. Values are listed identity / compact / morton. Full tables are in `m0d/<scene>/counters.tsv`. The loop's `us` column counts a different number of cycles per variant within the 4 s window, so use the replay table above for time.

dam:

| kernel | TLB miss % | MMU limiter % | buffer-read limiter % |
|---|---|---|---|
| nbOperator | 38.8 / 31.7 / 20.0 | 2.2 / 2.2 / 2.1 | 0.5 / 0.6 / 0.5 |
| lAdvect | 26.3 / 20.8 / 8.0 | 1.2 / 1.2 / 1.0 | 4.3 / 4.3 / 4.3 |
| uTransfer | 20.4 / 14.6 / 6.0 | 4.5 / 4.4 / 4.1 | 2.6 / 2.6 / 2.6 |
| uExtend | 33.4 / 23.9 / 13.4 | 2.0 / 2.0 / 1.5 | 1.3 / 1.3 / 1.3 |
| uAdvect | 20.2 / 15.5 / 5.1 | 1.7 / 1.7 / 1.4 | 8.1 / 8.1 / 7.8 |

paddleboats:

| kernel | TLB miss % | MMU limiter % | buffer-read limiter % |
|---|---|---|---|
| nbOperator | 26.5 / 19.0 / 17.9 | 2.3 / 2.3 / 2.3 | 0.7 / 0.7 / 0.7 |
| lAdvect | 11.6 / 6.8 / 3.7 | 1.1 / 1.1 / 1.1 | 4.8 / 5.0 / 4.9 |
| uTransfer | 11.5 / 6.3 / 4.1 | 5.1 / 5.0 / 5.0 | 3.6 / 3.6 / 3.6 |
| uExtend | 18.9 / 10.2 / 6.3 | 2.5 / 2.5 / 2.4 | 1.9 / 1.9 / 1.9 |
| uAdvect | 8.8 / 4.0 / 2.9 | 1.9 / 1.8 / 1.9 | 7.2 / 7.0 / 7.1 |

The miss rate follows the slot order exactly as the locality predicts: compact lowers it somewhat, and morton by 1.5–4× (nbOperator on dam: 38.8 → 20.0%). The limiters that would show a TLB-bound kernel do not move. Neither does occupancy, nor the ALU or LLC limiters (in the tsv). Inference: M0c's high TLB miss % reflects the access pattern and not a bottleneck, because the misses are overlapped. This matches M0c's finding that these kernels run far below the random-chase rates.

## Caveats

- Both measures are offline, on frozen states after 100 steps. They test the layout a slot-order policy would produce, not the cost of keeping that order up. A live compaction or Morton allocator would add its own work and invalidations.
- R = 5 minima over interleaved rounds. Per-kernel deltas of ±3–4% occur in both directions, including identity-to-identity across runs (dam run 1 vs run 2), and should be read as noise. The two dam runs started from different frozen states: the warm-up is itself not deterministic.
- K (cut) slots were not reordered. On paddleboats 280 K slots are live and reached through F record word 30. The cut kernels (bodyStamp) are 0.87 ms, so the bound on that remaining gain is small.
- The counter loop ran under Instruments with labels on, which makes passes slower than the replay table. The tables use only its ratios and percentages.

## Files

- `tools/experiments/gpu-shape/slot-order.mts`: the tool (parent and child in one file). Env: STEPS, R, CTR_SECS, EQ_ONLY, PROOF_STEPS, ID_REPS.
- `docs/plans/gpu-shape/m0d/dam/{slot-order.json, slot-order-run1.json, counters.tsv}`, `docs/plans/gpu-shape/m0d/paddleboats/{slot-order.json, counters.tsv}`.
- The raw traces, ndjson tables and logs are in `~/code/me/fluid-scratch/gpu-shape/m0d/` and `~/code/me/fluid-scratch/gpu-shape/m0d-*.log`.

Runs: all through `tools/gpu/fluid-gpu-run.sh timing`, rc 0, 0 GPU errors. Peak GPU memory was 4.2 GB on dam and 7.7 GB on paddleboats.

`git status --short` at write time (HEAD 797ab2d4):

```
 M lib/core/webgpu-water-pipeline.ts
 M tools/experiments/lane-n/dump-msl.mts
 M tools/experiments/lane-n/features.mts
?? docs/plans/gpu-shape/m0d-slot-order.md
?? docs/plans/gpu-shape/m0d/
?? lib/core/water-caustic-receiver.wgsl.ts
?? probe-inflow.mts
?? tests/water-caustic-receiver.test.ts
?? tools/experiments/gpu-shape/m1/
?? tools/experiments/gpu-shape/slot-order.mts
```

Only `slot-order.mts`, this doc and `m0d/` are lane B's. No lib/ file was changed.

