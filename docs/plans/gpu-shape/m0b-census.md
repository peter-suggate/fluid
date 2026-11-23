# GPU-shape M0(b): frame fill census (2026-11-23)

Plan: docs/plans/gpu-shape-2026-11-23.md. Tool: tools/experiments/gpu-shape/census.mts (+ msl.mts, psoinfo.swift).
Data: docs/plans/gpu-shape/census/<scene>/ (census-caps.tsv: every placement; families-<engine>.tsv; summary.md;
census.json with the run's git HEAD and `git status --short`).

**Verdict (measured, crude slab = 32 × thread cap): the hypothesis does not hold for the parent engine.** Sub-slab
placements are many (54–81 live plus 40–105 gated of ~290 a step) but they are short: they hold **2–10% of the parent
step** (2–19% if the slab were twice as large). 63–96% of every parent step is in placements at ≥ 4 slabs, the
register/instruction regime the plan says is never packed. The fine (detail) engine is the exception: **24% sub-slab
(+5% gated), 39% at a 2× slab**, which is what ride-along already measured. Lane A's slab model may move these lines;
the 2× and 4× columns show how far.

## Method

- Scene with `engine: "brick"`, fixed dt 1/60 (`numerics.fixedDt_s = maxDt_s = 1/60`), presentation off, **100 warm-up
  steps** on every scene (paddleboats with the moored roster, tools/moored-motor-bodies.ts, as the gate bench). Teaser:
  `fluid.automaticDetail = { factor: 2 }` (default budget 128), both engines.
- Freeze: fence, copy every engine buffer and external (body) buffer into GPU shadow buffers.
- **Wall**: restore, write the next step's ring words exactly as `submitStep` (`set` at ring + 256·(step % 4),
  `engine.ringWords(step)`), and encode the engine's own recorded dispatches (`engine.recorded[slot]`) as it does: one
  compute pass for part 1, one for part 2, timestamped at pass begin/end. Min of 3. Excludes the rigid integrator's
  command buffer (paddleboats), the detail allocator, presentation and CPU gaps between submits.
- **Placements**: restore, ring words, then the same recorded dispatches in schedule order, **each alone in a compute
  pass** with timestamps; before each indirect pass its 3 args words are copied out (so G is the real workgroup count
  on the real state). Every placement runs once on its true input state; time = min over 3 replays. Not the
  back-to-back repeat the brief suggested: repeating a state-mutating kernel in place corrupts the state that later
  placements read (args, counters), and a replay of each prefix per placement was unnecessary once the in-situ sum
  matched the wall (below). Indirect args were identical across the 3 replays in every engine (0 unstable).
- **Fine engine**: restore the parent, run its step N+1, then the allocator's 6 dispatches (each timed), then shadow
  the fine engine and run the same wall + replay (the order of BrickNBFlipSolver.advanceTo).
- **Caps** (compile only, `--caps`): each distinct kernel text (sha1 of entry + WGSL, cached in
  ~/code/me/fluid-scratch/gpu-shape/kcache) through Dawn `dump_shaders` → MSL; L = Tint's
  `max_total_threads_per_threadgroup`; that attribute stripped and the MSL compiled by Metal (psoinfo):
  `maxTotalThreadsPerThreadgroup` = the register-bound thread **cap**. **wgmem**: Tint passes workgroup memory as a
  `[[threadgroup(n)]]` argument, so psoinfo's static length is 0; wgmem is the byte size (WGSL layout rules) of the
  `var<workgroup>` declarations the generated MSL still uses.
- **Crude slab** = 32 cores × cap; fill32 = G·L / slab. *Sub-slab* = live (G > 0) and fill32 < 1. Gated = G = 0
  (converged PCG slots, empty lists): counted separately.
- **Dependency facts** per placement (TSV `reads`, `writes`, `rwUnwritten`): the storage bindings it binds, and which
  of its read_write bindings its text assigns to or takes `&` of (a text scan). Caveat for M1: almost every kernel
  writes `set` (the FAIL latch and counters), so binding-level sets alone give few antichains; M1 needs region-level
  facts (which words of `set`/`args`/`pmgF`).
- Runs: 5 timing runs via tools/gpu/fluid-gpu-run.sh timing, all rc 0, no GPU errors, peak GPU memory 3.0–7.9 GB.
  HEAD 7d2424f4 (seep, dam) and e8287a76 (letters, paddleboats, teaser; a cloth-only commit landed between).
  `git status --short` at the runs (outside this lane's files): ` M lib/core/webgpu-water-pipeline.ts`,
  ` M lib/cloth/stage-timing.ts` (seep, dam only), ` M tools/experiments/lane-n/{dump-msl,features}.mts`,
  `?? lib/core/water-caustic-receiver.wgsl.ts`, `?? probe-inflow.mts`, `?? tests/water-caustic-receiver.test.ts`.
  The levelset/fluid-grid WGSL files were clean by then, so the runs compiled HEAD's kernels.

Reproduce: `~/code/me/fluid-scratch/gpu-shape/run-all.sh` (each scene:
`SCENE=<key> STEPS=100 R=3 tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/census.mts`,
then `... census.mts --caps <OUT>/<key>`).

## Check: the placements add up to the step

Σ of the isolated in-situ placement times is within −4%…+6% of the wall on every engine (table). So dispatches in the
engine's one pass do not overlap: the step *is* the sum of its placements, each as long as it is alone. The
pass-a-placement replay span is 9–10 ms longer (≈ 33 µs a pass boundary, encoder overhead outside any dispatch;
irrelevant to the engine, which uses one pass a part). An empty 1-workgroup kernel alone in a timestamped pass reads
3.2–3.4 µs (median 3.4–4.0); a gated (0-workgroup) placement in situ costs ≈ 6 µs.

## Per-scene summary

| engine | wall ms | Σ ms | span ms | placements | sub-slab live (n / ms / % Σ) | at 2× slab | gated (n / ms) | < 25 µs (n / % Σ) |
|---|---|---|---|---|---|---|---|---|
| seep parent | 19.33 | 20.05 | 28.9 | 290 | 80 / 1.94 ms / **10%** | 19% | 40 / 0.24 ms | 125 / 8% |
| dam parent | 31.58 | 31.98 | 41.2 | 283 | 81 / 1.82 ms / **6%** | 13% | 40 / 0.23 ms | 121 / 5% |
| letters parent | 43.95 | 44.34 | 54.1 | 283 | 54 / 1.12 ms / **3%** | 4% | 80 / 0.48 ms | 139 / 3% |
| paddleboats parent | 33.57 | 34.29 | 44.6 | 307 | 61 / 1.02 ms / **3%** | 4% | 105 / 0.65 ms | 174 / 4% |
| teaser parent | 73.98 | 71.33 | 82.7 | 290 | 54 / 1.24 ms / **2%** | 2% | 60 / 0.35 ms | 109 / 1% |
| teaser fine | 9.39 | 9.92 | 18.8 | 264 | 114 / 2.35 ms / **24%** | 39% | 81 / 0.48 ms | 209 / 24% |

wall = engine encoding (2 passes); Σ = sum of placement times; span = the pass-a-placement replay; "at 2× slab" =
share of Σ in live placements with lanes < 2 × 32 × cap; < 25 µs = placements of any fill under 25 µs.

Share of Σ by total lanes (G·L) of the placement:

| engine | ≤1k | 1–4k | 4–16k | 16–64k | 64–256k | >256k |
|---|---|---|---|---|---|---|
| seep parent | 4% | 1% | 4% | 13% | 41% | 37% |
| dam parent | 3% | 0% | 1% | 21% | 7% | 67% |
| letters parent | 2% | 0% | 1% | 4% | 20% | 73% |
| paddleboats parent | 3% | 0% | 1% | 2% | 35% | 59% |
| teaser parent | 1% | 0% | 0% | 1% | 20% | 77% |
| teaser fine | 11% | 4% | 11% | 43% | 30% | 0% |

Share of Σ by fill32 (placements):

| engine | gated | < 0.25 | 0.25–1 | 1–4 | ≥ 4 |
|---|---|---|---|---|---|
| seep parent | 1% (40) | 5% (49) | 4% (31) | 26% (117) | 63% (53) |
| dam parent | 1% (40) | 3% (50) | 3% (31) | 19% (63) | 75% (99) |
| letters parent | 1% (80) | 1% (30) | 1% (24) | 12% (61) | 85% (88) |
| paddleboats parent | 2% (105) | 1% (30) | 2% (31) | 6% (24) | 89% (117) |
| teaser parent | 0% (60) | 1% (32) | 1% (22) | 2% (26) | 96% (150) |
| teaser fine | 5% (81) | 12% (61) | 12% (53) | 52% (66) | 19% (3) |

Detail allocator (teaser, 6 dispatches after the parent step): 0.63 ms, not in the fine Σ above.

## Top families by time (all placements of a label summed)

G is the live workgroup range; cap the Metal thread cap; fill32 the largest over the family's placements.

**seep parent** (Σ 20.05 ms)

| family | placements (live) | G | L | cap | wgmem B | max fill32 | ms | µs each |
|---|---|---|---|---|---|---|---|---|
| uTransfer | 1 (1) | 1448 | 256 | 512 | 12844 | 22.6 | 0.990 | 990 |
| phiSolid | 1 (1) | 1265 | 256 | 384 | 0 | 26.4 | 0.910 | 910 |
| lcAdvect | 1 (1) | 1464 | 64 | 512 | 3388 | 5.7 | 0.709 | 709 |
| uExtend | 1 (1) | 1265 | 512 | 768 | 23204 | 26.4 | 0.669 | 670 |
| censusDecide | 1 (1) | 1896 | 64 | 576 | 52 | 6.6 | 0.664 | 664 |
| lcRetire | 1 (1) | 1464 | 64 | 704 | 0 | 4.2 | 0.660 | 660 |
| lAdvect | 1 (1) | 1265 | 256 | 384 | 0 | 26.4 | 0.628 | 628 |
| uAdvect | 1 (1) | 1448 | 256 | 768 | 11272 | 15.1 | 0.588 | 588 |
| operator | 1 (1) | 246 | 512 | 512 | 16076 | 7.7 | 0.543 | 543 |
| lResolve | 3 (3) | 1265 | 256 | 1024 | 1260 | 9.9 | 0.516 | 172 |

**dam parent** (Σ 31.98 ms)

| family | placements (live) | G | L | cap | wgmem B | max fill32 | ms | µs each |
|---|---|---|---|---|---|---|---|---|
| lAdvect | 1 (1) | 2089 | 256 | 384 | 0 | 43.5 | 1.955 | 1955 |
| uExtend | 1 (1) | 2089 | 512 | 768 | 23204 | 43.5 | 1.631 | 1631 |
| uTransfer | 1 (1) | 2197 | 256 | 512 | 12844 | 34.3 | 1.594 | 1594 |
| pMove | 1 (1) | 2404 | 128 | 512 | 0 | 18.8 | 1.216 | 1216 |
| uAdvect | 1 (1) | 2197 | 256 | 768 | 11272 | 22.9 | 1.207 | 1207 |
| pSeed | 2 (2) | 2089 | 256 | 576 | 3312 | 29.0 | 1.201 | 601 |
| operator | 1 (1) | 727 | 512 | 576 | 16076 | 20.2 | 1.167 | 1167 |
| pSeparate | 1 (1) | 2404 | 128 | 1024 | 0 | 9.4 | 1.134 | 1134 |
| censusDecide | 1 (1) | 880 | 64 | 576 | 52 | 3.1 | 1.108 | 1108 |
| censusFine | 1 (1) | 16664 | 64 | 640 | 2712 | 52.1 | 0.943 | 943 |

**letters parent** (Σ 44.34 ms)

| family | placements (live) | G | L | cap | wgmem B | max fill32 | ms | µs each |
|---|---|---|---|---|---|---|---|---|
| uTransfer | 1 (1) | 3330 | 256 | 512 | 12844 | 52.0 | 3.030 | 3030 |
| pSeed | 2 (2) | 3166 | 256 | 576 | 3312 | 44.0 | 2.318 | 1159 |
| uExtend | 1 (1) | 3166 | 512 | 768 | 23204 | 66.0 | 2.223 | 2223 |
| pMove | 1 (1) | 4750 | 128 | 512 | 0 | 37.1 | 2.154 | 2154 |
| censusDecide | 1 (1) | 1336 | 64 | 576 | 52 | 4.6 | 1.988 | 1988 |
| lAdvect | 1 (1) | 3166 | 256 | 384 | 0 | 66.0 | 1.969 | 1969 |
| pSeparate | 1 (1) | 4750 | 128 | 1024 | 0 | 18.6 | 1.842 | 1842 |
| lDistance | 1 (1) | 3166 | 256 | 832 | 10228 | 30.4 | 1.625 | 1625 |
| operator | 1 (1) | 966 | 512 | 576 | 16076 | 26.8 | 1.622 | 1622 |
| lcRetire | 1 (1) | 1312 | 64 | 704 | 0 | 3.7 | 1.359 | 1359 |

**paddleboats parent** (Σ 34.29 ms)

| family | placements (live) | G | L | cap | wgmem B | max fill32 | ms | µs each |
|---|---|---|---|---|---|---|---|---|
| operator | 1 (1) | 1472 | 512 | 512 | 16076 | 46.0 | 2.461 | 2461 |
| lAdvect | 1 (1) | 1695 | 256 | 384 | 0 | 35.3 | 2.396 | 2396 |
| uExtend | 1 (1) | 1695 | 512 | 768 | 23204 | 35.3 | 1.905 | 1905 |
| censusDecide | 1 (1) | 2192 | 64 | 512 | 52 | 8.6 | 1.748 | 1748 |
| uTransfer | 1 (1) | 1969 | 256 | 512 | 12844 | 30.8 | 1.726 | 1726 |
| uAdvect | 1 (1) | 1969 | 256 | 768 | 11272 | 20.5 | 1.305 | 1305 |
| lcAdvect | 1 (1) | 2192 | 64 | 512 | 3388 | 8.6 | 1.137 | 1137 |
| lcRetire | 1 (1) | 2192 | 64 | 704 | 0 | 6.2 | 0.999 | 999 |
| censusFine | 1 (1) | 13576 | 64 | 640 | 2712 | 42.4 | 0.974 | 974 |
| bodyStamp | 2 (1) | 280 | 512 | 768 | 16 | 5.8 | 0.857 | 428 |

**teaser parent** (Σ 71.33 ms)

| family | placements (live) | G | L | cap | wgmem B | max fill32 | ms | µs each |
|---|---|---|---|---|---|---|---|---|
| uTransfer | 1 (1) | 6096 | 256 | 512 | 12844 | 95.2 | 4.675 | 4675 |
| uExtend | 1 (1) | 5763 | 512 | 768 | 23204 | 120.1 | 3.846 | 3846 |
| operator | 1 (1) | 1596 | 512 | 512 | 16076 | 49.9 | 3.818 | 3818 |
| pSeed | 2 (2) | 5763 | 256 | 576 | 3312 | 80.0 | 3.192 | 1596 |
| pSeparate | 1 (1) | 5983 | 128 | 1024 | 0 | 23.4 | 2.673 | 2673 |
| censusDecide | 1 (1) | 2656 | 64 | 576 | 52 | 9.2 | 2.487 | 2487 |
| phiSolid | 1 (1) | 5763 | 256 | 384 | 0 | 120.1 | 2.456 | 2456 |
| pMove | 1 (1) | 5983 | 128 | 512 | 0 | 46.7 | 2.393 | 2393 |
| lDistance | 1 (1) | 5763 | 256 | 832 | 10228 | 55.4 | 2.351 | 2351 |
| lAdvect | 1 (1) | 5763 | 256 | 384 | 0 | 120.1 | 2.226 | 2226 |

**teaser fine** (Σ 9.92 ms)

| family | placements (live) | G | L | cap | wgmem B | max fill32 | ms | µs each |
|---|---|---|---|---|---|---|---|---|
| operator | 1 (1) | 241 | 512 | 512 | 16076 | 7.5 | 1.307 | 1307 |
| uAdvect | 1 (1) | 241 | 256 | 704 | 11272 | 2.7 | 0.677 | 677 |
| lAdvect | 1 (1) | 241 | 256 | 512 | 0 | 3.8 | 0.649 | 649 |
| pSeed | 1 (1) | 241 | 256 | 640 | 3312 | 3.0 | 0.392 | 392 |
| uTransfer | 1 (1) | 241 | 256 | 640 | 12844 | 3.0 | 0.380 | 380 |
| censusFine | 1 (1) | 2168 | 64 | 640 | 2712 | 6.8 | 0.353 | 353 |
| pMove | 1 (1) | 517 | 128 | 576 | 0 | 3.6 | 0.290 | 290 |
| uExtend | 1 (1) | 241 | 512 | 896 | 17220 | 4.3 | 0.268 | 268 |
| pmgTopCycle | 8 (4) | 1 | 1024 | 1024 | 15884 | 0.0 | 0.234 | 29 |
| lClear | 1 (1) | 241 | 256 | 1024 | 6904 | 1.9 | 0.183 | 183 |

## The sub-slab families (summed over the six engines)

| family | sub-slab ms (6 engines) | G | L | cap | wgmem B | where |
|---|---|---|---|---|---|---|
| pmgTopCycle | 1.75 | 1 | 1024 | 1024 | 12 / 13264 / 15884 | every engine, 27–50 µs × live PCG slots |
| lDilate | 0.94 | 31–721 | 64 | 1024 | 0 | 6 a step, 19–43 µs |
| pmgDown3 | 0.61 | 16–333 | 64 | 1024 | 404 | 12–18 µs × live slots |
| pmgUp3 | 0.51 | 16–333 | 64 | 1024 | 372 | 11–16 µs × live slots |
| pmgCoarsenTop | 0.50 | 1 | 1024 | 896 | 26664 | once a step, 71–87 µs |
| pmgCheck | 0.43 | 1 | 64 | 1024 | 4 | 8 a step, 8–9 µs |
| pmgUp1 | 0.32 | 72–1167 | 64 | 1024 | 584 | seep, dam, fine |
| solidStamp | 0.28 | 18–159 | 64 | 1024 | 28 | seep, teaser (static solids) |
| pmgDown1 | 0.26 | 72–1167 | 64 | 1024 | 584 | seep, dam, fine |
| pSeparate | 0.17 | 206 (seep) | 128 | 1024 | 0 | seep only (few moved records) |
| setLists, setArgs, setMarkPages, censusCubes, nrSpread, regionSpread, pressureRowsZero, regionStill | 0.08–0.16 each | 1–333 | 64–256 | 1024 | 0–1732 | every engine |
| censusDecide, censusImportance (fine only) | 0.16, 0.09 | 160 | 64 | 576 / 896 | 52 / 24 | fine |

The whole MG coarse chain of one PCG slot (pmgDown1..3, pmgTopCycle, pmgUp3..1, pmgCheck) is the dominant sub-slab
work: ≈ 100–130 µs a live slot on every engine, 3–6 live slots a step.

## Packing candidates vs solo

**Candidates** (sub-slab, cap 1024 so light registers, little or no workgroup memory): lDilate (0 B), the MG chain's
pmgDown1/Down3/Up3/Up1 (372–584 B), pmgCheck (4 B), setLists, setArgs, setMarkPages, regionSpread, regionStill,
pressureRowsZero, nrSpread (0 B), solidStamp (28 B), censusCubes (1.7 KB). Together ≈ 0.5–1.0 ms a parent step and
≈ 0.6 ms on the fine engine; the parent+fine pairs of the same families are the natural pack (tier and engine sources,
plan §2–3). Same-step antichains inside the MG chain do not exist (each level reads the previous), so for the chain the
lever is the plan's M4 additive MG, not packing.

**Single-workgroup chains with workgroup memory** (pack only within their class, or remove algorithmically):
pmgTopCycle (1 × 1024, 13–16 KB on dam/teaser, 12 B on the others: its text is layout-specialised), pmgCoarsenTop
(1 × 1024, 26.7 KB, cap 896), pmgBodyFold (paddleboats, 24 × 512, 16 KB). pmgTopCycle is the single largest sub-slab
family (0.2–0.4 ms an engine).

**Solo** (many slabs at low caps; most of every step): uTransfer (cap 512, 12.8 KB), lAdvect and phiSolid (384),
uExtend (768, 23 KB), uAdvect (704–768, 11 KB), operator (512–576, 16 KB), pSeed (576), pMove (512), censusDecide
(parent: 512–576, fill 3–9), lcRetire (704), lcAdvect (512), lDistance (832, 10 KB), pDeposit (832, 27.5 KB),
censusFine (640), pSeparate (parent, 1024 but fill 9–23). These hold 63–96% of the parent steps at fill ≥ 4: their
lever is R_k (registers, and — inference — workgroup memory: 12.8–27.5 KB per workgroup allows only 1–2 resident
workgroups a core at 32 KB, which caps residency at the same 512 lanes as the register cap) and per-element work.

## What this says about the plan (go/no-go input; lane A's slab replaces the crude one)

- Measured: on every parent step, an upper bound on the packing prize (all sub-slab time hidden for free) is 2–10% at
  the crude slab and 2–19% at twice that slab, under the plan's ~25% stop line. Seep is the most favourable parent
  (10% / 19% / 36% at 1×/2×/4×); teaser parent the least (2% at any multiple).
- Measured: the fill 1–4 band (last-wave tail regime) is 26% (seep), 19% (dam), 12% (letters), 6% (paddleboats),
  2% (teaser parent), 52% (teaser fine) of Σ. Inference: the persistent-grid lever (plan regime 2) has more room
  than packing on seep and dam; it needs lane A's per-class slab to price.
- Measured: the fine engine is the one place above the line (24% sub-slab + 5% gated, 39% at 2×, 76% at 4×), in
  agreement with ride-along's 53% saving on latency-bound fine families. Engine packing (parent ∥ fine) is the
  source worth M1; same-engine antichains on the parent are not.
- Gated placements (converged PCG slots, empty lists) cost 0.23–0.65 ms a step (≈ 6 µs each in situ; paddleboats 105
  of them). Not a packing question, but a measured, cheap target (inference: fewer encoded slots or GPU-side skip of
  whole slot ranges).

## Tool problems and caveats

- Dawn's MSL dump through a pipe lost the tail of large modules (33 of 116 kernels on the first try); the child now
  writes to a file and retries once. psoinfo's static threadgroup length is 0 for Tint output (see wgmem above).
- The cap is a per-threadgroup register limit without the size attribute; it ignores workgroup memory and is not the
  resident threads per core. 32 × cap is therefore crude in both directions; the 0.5×–4× sensitivity is in each
  summary.md.
- In-situ single-shot times include cold caches as in the real step; the min of 3 replays is used. Back-to-back
  repeated timing (warm caches) was not used.
- The wall excludes the rigid integrator (paddleboats) and the allocator (teaser), both outside the engine schedule.
