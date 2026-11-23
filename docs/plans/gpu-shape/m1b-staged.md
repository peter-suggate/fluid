# GPU-shape M1b: phase attribution of the staged kernels (2026-11-23/24)

Plan: docs/plans/gpu-shape-2026-11-23.md. Tool: tools/experiments/gpu-shape/staged.mts (lane B, finished by lane B2).
Data: docs/plans/gpu-shape/m1b/<scene>/ (staged.json with the run's HEAD and `git status`, counters.tsv, caps.json) and
the run logs m1b/*.log; bulky traces stay in ~/code/me/fluid-scratch/gpu-shape/m1b/.

## Method

Frozen step 100 (brick engine, fixed dt 1/60, presentation off; paddleboats with the moored roster), as the census.
For each of uExtend, uTransfer, uAdvect, pDeposit: text variants that end the entry after phase k (a sink XOR of the
phase's workgroup words written behind an impossible-value test, then `return`). Each sample restores the frozen
state, writes the ring words, replays the step's placements before the kernel untimed, then the variant alone in a
timestamped pass; R = 7 interleaved samples, min. Phase cost = t(k) − t(k−1). `nobar` (uExtend) removes the a-sweep's
barriers (racy, timing only). Counters: Metal GPU Counters over a 4 s loop of every variant; caps: Metal's thread cap
(size attribute stripped).

Runs: dam at HEAD cab164fb (lane B, 2026-11-23, no `0-cpages` cut), paddleboats at 734dd9e2 (lane B2, all cuts),
dam `0-cpages` re-run for uTransfer/uAdvect at 734dd9e2. All rc 0, no GPU errors. Outside this lane's files the tree
held the other sessions' uncommitted edits (webgpu-water-pipeline.ts, lane-n tools, water-caustic-receiver) and lane
A2's m1/ tools; none touches the brick kernels.

## Correction to the earlier reading: the "window" phase was mostly C pages

uTransfer and uAdvect are FC passes: workgroups past F_ALL run a C page (vcTransferPage, the C advect) and return
before any F brick stages its window. Lane B's first cut ("1-window") therefore held the C pages, the F prologue and the
window together. The `0-cpages` cut (C pages done, F groups return right after `nbFOrigin`) separates them:

| ms (min of 7) | dam 0-cpages | dam window | paddleboats 0-cpages | paddleboats window |
|---|---|---|---|---|
| uTransfer | **0.695** | 0.017 | **0.526** | 0.021 |
| uAdvect | **0.413** | 0.011 | **0.428** | 0.012 |

So the owner-window staging is 0.01–0.02 ms in these kernels; the 0.71 / 0.42 ms is the C-page work (plus a ~12 µs
F prologue, the floor every staged kernel shows; M2's generic cuts). This is what killed M2 (m2-window-table.md).

## Phases, both scenes (ms, min of 7; phase = t(k) − t(k−1))

uTransfer (cap 512; occupancy 6–13%):

| phase | dam | paddleboats |
|---|---|---|
| 0-cpages (C pages + F prologue) | 0.695* | 0.526 |
| 1-window | 0.017* | 0.021 |
| 2-ring (φ ring 10³) | 0.248 | 0.143 |
| 3-ringmark | 0.217 | 0.052 |
| 4-stage | 0.237 | 0.354 |
| 5-liqstore | 0.087 | 0.137 |
| 6-half (first r of 2) | 0.192 | 0.289 |
| full (second half) | 0.173 | 0.308 |
| total | 1.867 | 1.831 |

uAdvect (cap 768):

| phase | dam | paddleboats |
|---|---|---|
| 0-cpages | 0.413* | 0.428 |
| 1-window | 0.011* | 0.012 |
| 2-ringmark | 0.009 | 0.035 |
| 3-ring | 0.075 | 0.087 |
| 4-live4 | 0.027 | 0.088 |
| 5-classify | 0.199 | 0.234 |
| full (traces) | 0.403 | 0.448 |
| total | 1.133 | 1.333 |

uExtend (cap 768; 512 lanes):

| phase | dam | paddleboats |
|---|---|---|
| 1-ring (ring-brick exit) | 0.080 | 0.071 |
| 2-window | 0.019 | 0.028 |
| 3-stage | 0.078 | 0.088 |
| 4-reach | 0.293 | 0.370 |
| 5–8 a0 seed/sweep/back/finish | 0.155 / 0.081 / 0.039 / 0.089 | 0.229 / 0.066 / 0.050 / 0.145 |
| 9-a1finish (a = 1 whole) | 0.359 | 0.476 |
| full (a = 2) | 0.355 | 0.447 |
| total | 1.548 | 1.969 |
| nobar (sweep barriers removed) | 1.328 (−0.22) | 1.682 (−0.29) |

pDeposit (cap 832):

| phase | dam | paddleboats |
|---|---|---|
| 1-window (incl. prologue) | 0.087 | 0.073 |
| 2-ring | 0.039 | 0.029 |
| 3-liquid | 0.044 | 0.030 |
| 4-zero | 0.015 | 0.019 |
| 5-scatter | 0.421 | 0.264 |
| full (finish) | 0.064 | 0.072 |
| total | 0.670 | 0.488 |

\* dam rows marked * are the 734dd9e2 re-run (m1b/dam-cpages); the other dam rows are lane B's cab164fb run, whose
"1-window" (0.712 / 0.421) is 0-cpages + window. Kernel totals of the two dam runs agree within 3% (uTransfer 1.87 /
1.92, uAdvect 1.13 / 1.08).

## Counters (measured, time-weighted means over each variant's passes)

Occupancy 6–21%, ALU limiter ≤ 42%, buffer-read limiter ≤ 12.5%, LLC limiter ≤ 18.5%, MMU ≤ 9% in every phase of
both scenes (counters.tsv). uTransfer's C-page cut alone runs at 6–7% occupancy and 10–12% ALU: latency-bound.
uExtend runs at 23–30% occupancy (512 lanes); removing its barriers buys 0.22–0.29 ms. Caps per cut: the window/ring
cuts compile at 576–1024 threads, the full kernels at 512 (uTransfer), 768 (uAdvect, uExtend), 832 (pDeposit).

## What the phases say (inference)

- The owner window is not where the staged kernels spend time (≤ 0.03 ms each). The big blocks are: uTransfer's and
  uAdvect's **C pages (0.41–0.70 ms each, 0.94–1.11 ms a step together)**, uExtend's reach + a-sweeps (~1.2–1.5 ms)
  and barriers (0.22–0.29 ms), pDeposit's scatter (0.26–0.42 ms), and the φ rings (uTransfer 0.14–0.25 ms).
- The C-page share is the strongest new lead: it runs at the lowest occupancy of any phase and is a separate body in
  the FC pass (its own workgroups), so its shape (lanes per page, staging) can be changed without touching F's.
