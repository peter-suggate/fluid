# GPU-shape M6: the whole-step residency audit (lane B2, 2026-11-24)

**Result.** Almost every pass over 50 µs on dam and paddleboats already runs as many groups as its registers or the
thread limit allow. One pass was held back by workgroup memory: **pDeposit**. It landed as two exact commits:
- 57d75b95: two rounds of face axes, 26.4 KB down to 17.6 KB of windows.
- 07a16bfa: round 1 reads its support bit from the record, so no workgroup memory is added.

Peak groups in flight went from 63 to 95 (three groups per core, not two). Against its parent the pass is 4–14% faster:

| scene | parent → landed (stepab pd2 + pd3) |
|---|---|
| seep | 0.220 → 0.202 ms |
| dam | 0.686 → 0.660 ms |
| letters | 1.203 → 1.067 ms |
| paddleboats | 0.493 → 0.433 ms |

The other deficits in the ranking are not workgroup memory, and packing does not reach them:
- uAdvect is bound by registers that psoinfo under-reports.
- projectDry is bound by launch overhead and its tail.
- censusDecide and regionDeep are bound by their tails.

**Cumulative GPU-shape effect** (timing round c12d19cd → 07a16bfa, order-balanced):

| scene | change |
|---|---|
| seep | −4.0% |
| dam | −5.4% |
| letters | −5.7% |
| paddleboats | −4.3% |

Drift is within spread except seep, 0.09 pp lower. One intermittent paddleboats failure is not diagnosed. Both are
under "Timing round" below.

Plan: docs/plans/gpu-shape-2026-11-23.md (M6).

Tools:
- tools/experiments/gpu-shape/m4.mts, with `AUDIT=<us>` and the clock in flight.
- tools/experiments/gpu-shape/m6.mts, with `caps` and `table`.
- tools/experiments/gpu-shape/m6-wglimit.mts.

Data: docs/plans/gpu-shape/m6/. The per-workgroup clock TSVs (19 MB per scene) stay in
~/code/me/fluid-scratch/gpu-shape/m6/a1-{dam,paddleboats}/.

## Method

- **Placements.** The audit ran at 0f97d8af on dam and paddleboats, using the census settings and the frozen state at
  step 100, as m4.mts uses. Every indirect placement is timed (`MODE=time,clock AUDIT=50`). Each one over 50 µs is
  then clocked: m4's per-workgroup clock helper records each group's start and end ticks.
  - **Peak in flight** is an event sweep over those intervals.
  - **avg in flight** is Σ duration / span.
  - **fill** is avg / peak.
  - **tail** is the time from 95% of groups ended to the last group ended, as a share of the span.
  - The ticker group 0 is not a record.
- **Clock fix.** The ticker limit is now `CLK_MAX_TICKS = 1 << 16`. A group queued behind the ticker on its core can
  deadlock it. That is what hung paddleboats' `operator` for 8 s (rc 77, kept as audit-paddleboats-INVALID-rc77.txt).
  `operator` is reported from the 95%-ended window of a ticker-capped run.
- **Census inputs.** `m6.mts caps` takes these per placement from the lowered module:
  - **cap** is psoinfo's maxTotalThreadsPerThreadgroup for the MSL. It is a register predictor: the module is
    compiled with no size attribute.
  - **wgmem** is the bytes of the workgroup variables the MSL uses, counted as in census.mts.
  - **L** is the workgroup size.
- **Bounds**, per GPU (32 cores):
  - cap-implied = 32 × floor(min(cap, 3072) / L).
  - wgmem-implied = 32 × floor(65536 / wgmem).
- **Classes** (m6.mts):

  | class | rule |
  |---|---|
  | few | G < 0.8 × cap-implied: too few groups to fill the GPU |
  | wgmem | wgmem-implied < cap-implied, and peak ≥ 0.85 × wgmem-implied |
  | registers | peak ≥ 0.85 × cap-implied, with cap < 1024 |
  | full | peak ≥ 0.85 × cap-implied, with cap = 1024 (the API maximum, so the bound is a floor) |
  | launch | otherwise, with p50 group duration < 2 µs |
  | other | anything else |
  | +tail | added when tail > 35% |

- **Ranking.** deficit = max(0, 1 − peak / cap-implied), and the score is µs × deficit.

## The per-core limits, measured (m6-wglimit.mts, m6/wglimit.txt)

A light kernel touches W bytes of workgroup memory, then spins for about 100 µs over 4096 groups, counting the groups
alive at once.

| W (bytes) | groups per core |
|---|---|
| 6144 | 10 |
| 8192 | 8 |
| 10240 | 6 |
| 12288 | 5 |
| 16384 | 4 |
| 20480 | 3 |
| 24576 | 2 |
| 32768 | 2 |

- **Workgroup memory:** groups per core = floor(65536 / W) exactly, for L = 64, 256 and 512, so the per-core limit is
  64 KB.
- **Threads:** the limit is 3072 per core (12 × 256, 6 × 512). At L = 64 it is about 48 groups.

psoinfo's cap is only a coarse predictor of the register limit. uAdvect (cap 768, L 256) runs 2 groups per core, not 3.
lDistance (cap 832) runs 3.

## The ranking (families; m6/table.md has every placement)

Every family with a score of 2 or more is listed.

| scene | pass | µs | G | L | cap | wgmem B | cap-impl. | wgmem-impl. | peak | fill | tail % | class | deficit | score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paddleboats | uAdvect | 1300 | 1964 | 256 | 768 | 11272 | 96 | 160 | 64 | 0.82 | 21 | other | 0.33 | 433 |
| dam | uAdvect | 1080 | 2165 | 256 | 768 | 11272 | 96 | 160 | 64 | 0.69 | 36 | other+tail | 0.33 | 360 |
| dam | **pDeposit** | 692 | 2063 | 256 | 832 | 27532 | 96 | 64 | 63 | 0.75 | 18 | **wgmem** | 0.34 | 238 |
| paddleboats | **pDeposit** | 493 | 1690 | 256 | 832 | 27532 | 96 | 64 | 63 | 0.66 | 26 | **wgmem** | 0.34 | 169 |
| dam | projectDry | 194 | 1119 | 64 | 512 | 10292 | 256 | 192 | 96 | 0.10 | 83 | launch+tail | 0.63 | 122 |
| dam | censusDecide | 967 | 807 | 64 | 576 | 52 | 288 | 40320 | 252 | 0.32 | 44 | registers+tail | 0.13 | 121 |
| paddleboats | operator (ticker capped) | 2343 | 1471 | 512 | 512 | 16076 | 32 | 128 | 31 | 0.94 | - | registers | 0.03 | 73 |
| paddleboats | projectDry | 114 | 1541 | 64 | 512 | 10292 | 256 | 192 | 95 | 0.15 | 55 | launch+tail | 0.63 | 72 |
| paddleboats | censusCoarse | 624 | 2191 | 64 | 704 | 4532 | 352 | 448 | 319 | 0.71 | 24 | registers | 0.09 | 58 |
| dam | censusFine | 907 | 16359 | 64 | 640 | 2712 | 320 | 768 | 303 | 0.80 | 13 | registers | 0.05 | 48 |
| dam | regionDeep | 86 | 807 | 64 | 896 | 316 | 448 | 6624 | 270 | 0.28 | 52 | other+tail | 0.40 | 34 |
| dam | pmgCoarsenC | 57 | 101 | 512 | 1024 | 18204 | 64 | 96 | 32 | 0.70 | 6 | other | 0.50 | 29 |
| dam | censusCoarse | 251 | 807 | 64 | 704 | 4532 | 352 | 448 | 319 | 0.51 | 24 | registers | 0.09 | 24 |
| dam | lcClear | 82 | 101 | 256 | 1024 | 7376 | 128 | 256 | 102 | 0.64 | 16 | few | 0.20 | 17 |
| dam | pSeed (2) | 1058 | 2063 | 256 | 512 | 3312 | 64 | 608 | 63 | 0.62 | 35 | registers(+tail) | 0.02 | 17 |
| paddleboats | censusImportance | 172 | 2191 | 64 | 896 | 24 | 448 | 87360 | 415 | 0.51 | 42 | registers+tail | 0.07 | 13 |
| dam | uCache4 | 58 | 101 | 256 | 1024 | 972 | 128 | 2144 | 102 | 0.43 | 7 | few | 0.20 | 12 |
| dam | pmgUpdate (6) | 729 | 552 | 512 | 1024 | 6792 | 64 | 288 | 63 | 0.68 | 27 | full | 0.02 | 11 |
| paddleboats | lcAdvect | 698 | 547 | 256 | 512 | 5416 | 64 | 384 | 63 | 0.78 | 18 | registers | 0.02 | 11 |
| paddleboats | pSeed (2) | 597 | 1690 | 256 | 512 | 3312 | 64 | 608 | 63 | 0.72 | 18 | registers | 0.02 | 9 |
| paddleboats | pmgUpdate (3) | 569 | 1015 | 512 | 1024 | 6792 | 64 | 288 | 63 | 0.75 | 21 | full | 0.02 | 9 |
| dam | lcAdvect | 488 | 203 | 256 | 512 | 5416 | 64 | 384 | 63 | 0.55 | 28 | registers | 0.02 | 8 |
| paddleboats | censusDecide | 1760 | 2191 | 64 | 512 | 52 | 256 | 40320 | 255 | 0.79 | 21 | registers | 0.00 | 7 |
| dam | censusImportance | 90 | 807 | 64 | 896 | 24 | 448 | 87360 | 415 | 0.22 | 46 | registers+tail | 0.07 | 7 |
| paddleboats | pmgSetupF | 330 | 1471 | 512 | 1024 | 25068 | 64 | 64 | 63 | 0.78 | 17 | full | 0.02 | 5 |
| dam | pmgSetupF | 194 | 721 | 512 | 1024 | 25068 | 64 | 64 | 63 | 0.81 | 12 | full | 0.02 | 3 |
| paddleboats | uExt4Seed | 437 | 1095 | 128 | 640 | 3084 | 160 | 672 | 159 | 0.59 | 37 | registers+tail | 0.01 | 3 |

(Rows of 2 or less are omitted: pressureProjectC, lResolveD, setTransferFaces, bodyCouple and a long tail of others.)

**Class totals over both scenes** (µs of the audited families):

| class | µs |
|---|---|
| registers | 33 548 |
| full | 17 330 |
| other | 2 523 |
| wgmem | 1 185 (pDeposit only) |
| launch | 308 (projectDry) |
| few | 140 |

21 of the families also carry +tail.

**What limits the high-ranked ones:**
- **uAdvect (other).**
  - It runs 2 groups per core. Its 11 KB of workgroup memory would allow 5, and psoinfo's 768-thread cap suggests 3.
  - The binding limit is registers that psoinfo under-reports (inference).
  - Packing workgroup memory cannot help it.
- **pDeposit (wgmem).** It runs 2 groups per core: 27.5 KB, where 3 would need 21.8 KB or less. This is the one pass
  Step 2 targets.
- **projectDry (launch+tail).**
  - Its p50 group duration is 0.8 µs, and most groups exit early.
  - Its fill is 0.10–0.15 and its tail is 55–83%.
  - Its cost is the launch of mostly empty groups plus a few long ones. That is not residency.
- **censusDecide and regionDeep (+tail).** One placement's late groups run on after the rest have ended. This is the
  M4 diagnosis (throughput-bound on paddleboats, tail on dam).
- **pmgCoarsenC and lcClear (other/few).** 101 groups on dam cannot fill 32 cores.
- **lSplat** is not in the ranking (full+tail, deficit 0). It runs 5 groups per core:
  - That is above its cap-implied 4. psoinfo's 1024 is the API maximum, so for cap-1024 passes the bound is a floor;
    setTransferCells also exceeds it, 542 against 512.
  - It is below the 6 its 10.6 KB of workgroup memory allows, which fits allocation overhead above the raw bytes
    (inference, not probed).
  - Its tail is 36–43%. Packing wKey and wHot (2 × 1331 words) is a candidate, not measured.

## Step 2: fixes (m4.mts variants; stepab gate)

| family | variant | wgmem | peak | result (min of rounds) | verdict |
|---|---|---|---|---|---|
| pDeposit | axis3: three groups per brick, one axis each | 9 932 B | 95 | +22%/+8% (dam/pb): staging and record reads ×3 | no |
| pDeposit | axis2: two groups per brick, axes 0–1 and 2 | 18 732 B | 95 | +5%/+4% | no |
| pDeposit | seq2: one group, two rounds, recompute support | 18 732 B | 95 | m4 −7%/−7%; stepab −3.2% dam, +7.0% seep | weak |
| pDeposit | seq2 + support bits in 512 B of workgroup memory (57d75b95) | 19 244 B | **63** | stepab −6.7/−8.4/−10.5/−12.0% | landed |
| pDeposit | seq2 + support from the record's mode bit (07a16bfa) | 18 732 B | 95 | vs 57d75b95: −7.5/+1.2 (A/A +1.5)/−2.1/−3.8% | landed |
| pmgSetupF | planes: wK, wKU, w1K and w1KU sized to the part | 25 068 → ~20 500 B | 64 | −2%/−6% min, −2/−4% median; registers bind | not landed |
| pmgCoarsenC | planes: w2K, w3K and w4K sized to the part | — | 32/63 | +3%/+5% | no |

The parent pDeposit is 27 532 B: 26 400 B of windows and 1 132 B of other variables. In the stepab column, the
four figures are seep / dam / letters / paddleboats.

- **seq2 is the shape that wins.**
  - A pass bound by workgroup memory does not want more groups per brick. Each extra group repeats the brick's
    staging (the window, the vertex ring, wLiquid) and its record reads.
  - It wants the same group to need less memory at once: two rounds over the brick's records through one window
    two axes wide.
  - Round 1 repeats only the records' scatter. With 07a16bfa it also skips the 8-tap support test, because round 0
    already wrote the verdict into before.w's mode, and the same invocation rereads that record.
  - The sums are integer atomics, so the rounds change no word.
- **57d75b95's 512-byte support array was a mistake.** It pushed the group back to 2 per core (peak 63; m6/clock-landed-57d75b95-dam.txt). Three groups of 19 244 B
  would fit raw in 64 KB, so there is overhead per group or per variable beyond the raw bytes (inference). 07a16bfa removes
  the array, and peak is back to 95 (m6/clock-landed-modebit-dam.txt).
- **Gates.**
  - Both commits are exact under stepab on seep, dam, letters and paddleboats; the changed placements are identical.
  - Both passed `npm run check:types` and `test:unit`.
- **pmgSetupF planes.**
  - It is exact and cuts workgroup memory by 4.6 KB. Residency does not move (63 → 64), because the 3072-thread and
    register limits bind first at L 512.
  - Its 2–6% is within what the m4 rounds can resolve, so it is not landed. It is a stepab candidate if a later
    batch wants it.

## Gate tiers used

Both tiers are from docs/plans/gpu-shape-2026-11-23.md, lines 146–149.

- **Bit-identical, the default.** Both pDeposit commits pass here, so the ulp tier was not needed.
- **ulp tier (`TOL=ulp4`).** For reuse or reorder only:
  - integer, flag and slot words stay exact;
  - f32 words are within 4 ulp;
  - counts and seed decisions are identical;
  - drift is within spread;
  - no scene is slower.

## Timing round, c12d19cd → HEAD (07a16bfa)

The whole GPU-shape batch so far is in this round: 26 commits from cbb5f5b5 to 07a16bfa, covering lanes A2 and B2,
M1 to M6.

**Trees.**
- Base: an export of c12d19cd.
- Candidate: an export of 07a16bfa.
- For paddleboats only, the base also carries 67c09e96's tools/nbflip-observe-brick.ts
  (m6/round/base-observer-67c09e96.patch). c12d19cd's observer rejects paddleboats' open-top box at step 0
  (rc 1, "the box 512x32x192 is not the engine's box 512x64x192"), and that file is host-only and outside lib/.

The runs used `tools/nbflip-timing-round.sh` in balanced order. Every run was rc 0, and metal.log is empty.

| scene | fenced median ms | pipelined ms/step | order-balanced | drift at the end (base → cand, fenced / pipelined) |
|---|---|---|---|---|
| seep (1200 steps) | 26.04 → 24.73 | 24.47 → 23.73 | **−4.0%** | −0.55 / −0.55% → −0.65 / −0.64% |
| dam (165) | 33.88 → 31.81 | 35.37 → 33.73 | **−5.4%** | −1.09 / −1.14% → −1.12 / −1.13% |
| letters (240) | 54.10 → 50.45 | 44.29 → 42.24 | **−5.7%** | +0.64 / +0.45% → +0.40 / +0.53% |
| paddleboats (300) | 37.09 → 35.56 | 35.23 → 33.64 | **−4.3%** | +0.61 / +0.67% → +0.65 / +0.59% |

- **Drift.** dam, letters and paddleboats stay within the spread of their own pair. Seep's drift is 0.09 pp lower
  than its base, where its base pair spans 0.006 pp.
  - The whole step is not replay-identical: pSeed's claimed slots and pMove's ranks are order-dependent.
  - So a small change in the 20 s seep run is expected even from exact kernels (inference). The 0.09 pp has not
    been attributed to a commit.
- **Failure at step 297.** The first paddleboats rerun (m6/round/round-paddleboats-FAILED-step297.txt) failed one
  candidate fenced run at step 297 with "Step 297 refused with no frame pending" (rc 1).
  - That is a failMixedFrame, and the bench does not print its message.
  - The same tree ran 300 steps in both other rounds (round 1 and the rerun above).
  - Its cause is not diagnosed: intermittent, one in three candidate fenced runs.
