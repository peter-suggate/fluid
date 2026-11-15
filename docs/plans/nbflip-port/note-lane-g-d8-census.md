# Note to lane G: D8's particle count, receipt or census (lane F, 2026-11-16)

**Asked of the referee:** should the dam's D8 row (particles, window mean 30–90) count the references' live
records, the old engine's census, in place of their receipt count? Lane F does not apply the correction. D8
stays gated on the receipt row, as your 2026-11-16 note rules, until you rule on this.

## What each side counts

- **The references' `particles`** (ref-2761350a/dam-ref-{a..e}): the old engine's receipt count at the fence.
  It leaves out the spill tail and the frame's seeds.
- **The old engine's census** (`readParticleMassCensus`, with `narrowBandRetireCounting` on): every live record at
  the same fence.
- **Ours** (the brick engine): `PCOUNT[bank]`, which is pScan's filed records plus pSeed's appended seeds.
  `--pstat` finds 0 dead records at every step, and there is no spill reserve or tail. It counts every live
  record, so it compares like for like with the census.

## The 90-step evidence

The old engine: one check-slot run, read at the gate's sample steps (d8-old.mts). Ours: the gated tiered run,
round-s6/dam-300ti.jsonl, unchanged. Files are in ~/code/me/fluid-scratch/lane-f/d8old/: d8-old-90.jsonl,
-summary.json and run90.log.

| step | refs mean (receipt) | old receipt | old census | old census / receipt | ours | ours / refs (receipt, as gated) | ours / old census |
|---|---|---|---|---|---|---|---|
| 16 | 351,574 | 350,224 | 388,580 | 1.110 | 367,319 | 1.045 | 0.945 |
| 30 | 794,055 | 805,468 | 848,452 | 1.053 | 884,787 | 1.114 | 1.043 |
| 45 | 952,090 | 979,559 | 1,007,699 | 1.029 | 1,027,665 | 1.079 | 1.020 |
| 60 | 585,115 | 583,585 | 615,839 | 1.055 | 664,302 | 1.135 | 1.079 |
| 75 | 414,090 | 432,914 | 454,727 | 1.050 | 505,880 | 1.222 | 1.112 |
| 90 | 408,960 | 422,060 | 436,372 | 1.034 | 493,162 | 1.206 | 1.130 |
| **30–90 mean** | 630,862 | | 672,618 | | 715,159 | **1.134** | **1.063** |

- **The census runs 3–11% above the receipt at every sample.** Measured against live records, the receipt row
  overstates our excess by that much.
- **Against the census, D8 is 1.063 over the window,** which is 0.63 of the 10% tolerance. Another method
  scales each reference by this run's census/receipt factor at that step; it gives 1.087.
- **Caveats.**
  - Both figures rest on one old-engine run.
  - That run's receipt is 2.2% above the five references' mean over 30–90.
- **What the census does not explain.**
  - Steps 75–90 still read 1.11–1.13 against the census.
  - From step 60 we keep more records. The 30-step accounting agrees: we retire less by deep retirement, by cut
    and merge, and by splits.
  - Step 9 is a seed burst: ours seeds 53.7k there. The old engine's per-step seed count for comparison is still
    to be measured.
  - These are seeding and retirement differences, which a counting correction would not cover.

## Earlier 30-step accounting (the same comparison, denser in time)

| step | ours / refs (receipt) | ours / old census, records | ours / old census, mass |
|---|---|---|---|
| 5 | 1.035 | 0.916 | 0.923 |
| 7 | 1.220 | 0.976 | 0.991 |
| 9 | 1.232 | 1.120 | 1.167 |
| 10 | 1.250 | 1.118 | 1.146 |
| 13 | 1.129 | 1.051 | 1.089 |
| 16 (D8c) | 1.045 | 0.953 | 0.977 |
| 20 | – | 1.045 | 1.068 |
| 25 | – | 1.060 | 1.090 |
| 30 | 1.114 | 1.068 | 1.086 |

Files: d8old/{d8-old-30,d8-new-30}.jsonl.

## Options for the ruling

1. **Keep the receipt.** D8 compares ours (every live record) against the receipt. It then gates a measurement
   difference of 3–11% as well as the physics.
2. **Count the census.** Re-record the references' `particles` from the census (five check runs of the old
   engine), or record the census beside the receipt. D8 then compares like for like.
3. **Scale.** Correct the receipt by a measured census/receipt factor. This is weaker: the factor varies by step
   (1.03–1.11) and comes from a single run.

The details are in lane F's progress log (progress-lane-f.md, "D8 over the gated window: the old engine's
census, 90 steps" and the entry before it).
