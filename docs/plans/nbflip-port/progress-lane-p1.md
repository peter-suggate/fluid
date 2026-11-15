# NB-FLIP port, lane P1: pressure (progress)

Brief: docs/plans/nbflip-port-2026-11-15.md, lane P1 (items 13, 3, 10, 8, 5 of lane A's §6).
Scratch: `~/code/me/fluid-scratch/lane-p1/` (probe `pc.mts`, `run.sh`, `sum.py`, snapshot trees
`mktree.sh`). Every probe runs from a snapshot tree (HEAD plus this lane's files), never from the
shared checkout: another lane's in-progress WGSL there fails pipeline creation.

## 2026-11-15 STARTED (lane P1)

HEAD f72b3ca6. Probe: lane A's `pressure-classes.mts`, extended with the tail's row counts,
CPU-side cap counts from the residual history, the GPU cap record and the Krylov words.
Windows: seep 600-640, dam 30-90, letters 100-240, paddleboats 300-340, check slot.

## 2026-11-15 MILESTONE P1-1: the pressure cap as a reported state (item 13)

What it does:
- The band verdict (pVerdict) counts every normal solve that ran the encoded cycle cap and
  ended above the target (final measure past `UNIFORM_PRESSURE_ARENA_RETURNS_RESIDUAL`). These
  go into frame status words 10–13 (`UNIFORM_MIXED_STATUS.pressureCapped`, `pressureShort`,
  `pressureCappedWorst`, `pressureCappedFrame`).
  - The words are sticky, like the failure record.
  - A solve above target that encoded fewer cycles than the cap counts as short.
  - The kick's re-solves count nothing: a second pipeline of the same module with `pCount`
    off, under one explicit bind group layout.
- The record is a reported state, not a failure. `UNIFORM_PRESSURE_CAP_FATAL` (false) turns it
  into `pressureUnconverged` once every gate scene converges.
- The receipt decodes it (`uniformPressureCapRecord`). The solver info carries it as
  `uniformPressureCappedFrames`, `ShortFrames`, `WorstCappedResidual` and `LastCappedFrame`.
  The two benches print `pressureCap` in their summary.
- The schedule gains `tailSweeps` (default 32, the old TAIL_SWEEPS) for the go/no-go below.
- With diagnostics on, pTail counts the tail's live rows (32h, 16h, resident 8h, BULK16).

Equivalence (exact by construction: no iterate is touched):
- NB-FLIP is not reproducible run to run on HEAD. Two HEAD dam runs differ from frame 3, by
  up to 15% in a frame's final residual.
- The candidate's per-frame cycles and residuals lie inside that A/A spread (dam 1–45,
  paddleboats 1–40).
- The GPU count equals the CPU count from the residual history over the same frames: dam
  25 / 25, paddleboats 0 / 0.

Gates (lane G, check slot, f72b3ca6 + this patch): **PASS on all four scenes**. The dam
completed 300 steps. Report: `~/code/me/fluid-scratch/lane-p1/gates/cand13/gates-cand13.md`.

Cap record, full gate runs (the baseline this lane works down):

| scene | steps | capped | short | worst capped residual (1/s) | last capped frame |
|---|---|---|---|---|---|
| seep | 1200 | 461 (38%) | 0 | 6.59 | 1195 |
| dam | 300 | 85 (28%) | 0 | 13.19 | 288 |
| letters | 240 | 23 (10%) | 2 | 3.49 | 221 |
| paddleboats | 300 | 24 (8%) | 0 | 2.46 | 294 |

Probe windows (CPU count from the history, f72b3ca6 lib):

| scene | window | capped | mean cycles | median final residual |
|---|---|---|---|---|
| seep | 600–640 | 18 / 41 | 3.56 | 0.040 |
| dam | 30–90 | 32 / 61 | 3.98 | 0.051 |
| letters | 100–240 | 19 / 141 | 3.82 | 0.029 |
| paddleboats | 300–340 | 1 / 41 | 2.76 | 0.021 |

## 2026-11-15 FINDING: item 3 (exact coarsest, then one tail cycle) is a no-go in the arena

Live tail rows (maximum over each window):

| scene | 32h | 16h | 8h | BULK16 items |
|---|---|---|---|---|
| seep | 122 | 843 | 1855 | 70 |
| dam | 48 | 283 | 1824 | – |
| letters | 91 | 494 | 3183 | – |
| paddleboats | 100 | 775 | 4657 | – |

So a dense exact 32h solve would fit. The proxy: 256 or 1024 red-black sweeps of pTail, from
zero, on at most 122 rows, which is near exact. The results:

| config | seep capped / 41 | dam / 61 | letters / 141 | paddleboats / 41 |
|---|---|---|---|---|
| default (3 tail cycles, 32 sweeps) | 18 | 32 | 19 | 1 |
| 1 tail cycle | 18 | 55 | 55 | 0 |
| 1 tail cycle, 256 sweeps | 17 | 46 | 58 | 0 |
| 1 tail cycle, 1024 sweeps | 18 | 38 | 43 | 10 |
| 9 tail cycles | 18 | 24 | 16 | 4 |
| 9 tail cycles, 256 sweeps | 21 | 31 | 13 | 0 |

- **One tail cycle loses** on the dam and letters, even with an exact 32h.
- **Nine tail cycles do not gain:** their counts are inside the window-to-window noise (the
  same config run twice: ±2 capped).
- **So the tail's three cycles resolve 8h and 16h modes, not 32h ones.** Their page-local
  sweeps run against a frozen ring. An exact 32h solve does not replace them.
- An exact 8h solve (up to 4657 rows) does not fit a dense inverse: 87 MB, and a read per
  cycle.
- Item 3 is not landed.

## 2026-11-15 FINDING: the band solve has a residual floor near 1e-2 1/s

Deep runs (target 5e-4 1/s through an env patch, `bandCycles` 8). Mean log10 of the max-norm
measure per cycle:

| scene | GCR(1) on V-cycles (the default) | V-cycles only (`krylov` off) |
|---|---|---|
| dam | 2.37, 1.02, 0.18, −0.55, −1.38, −1.99, −2.13, −2.08, −2.02 | 2.05, 0.84, 0.16, −0.22, −0.60, −0.97 (cycles 1–6) |
| letters | 1.87, 0.67, −0.22, −1.13, −1.93, −2.70, −2.91, −2.78, −2.78 | 1.69, 0.43, −0.13, −0.56, −0.97, −1.36 |

- The Krylov step's 2-norm (KRYLOV_L2) stops falling after cycle 5: dam 7.49 → 4.92, then
  4.86, 4.85, 4.85.
- The V-cycle alone contracts by only about 0.43 a cycle from cycle 3 on.
- The stagnating rows are band free-surface rows (class 7). At cycle 6 on the dam they sit at
  −2.14; the other band classes are at −2.65 to −2.99 and the native rows below −7.
- No active-set events on the dam or letters (no hits, joins or restarts).
- An operator check (in the deep tree's pFinal) found every band–band coupling symmetric: 0 of
  10.1M faces differ by more than 1e-5. The 5.5% one-sided faces all face an inactive row,
  which is fixed data.
- So the 0.05 target sits 1–1.5 decades above a floor. The late cycles' slowdown is the floor
  plus a weak V-cycle (about 0.43 a cycle), not the tail.
- **For the restructure:** the seep-style preconditioner must fix the V-cycle's rate at the
  band's free surface and around the seam. The seep's staggered Schwarz addressed exactly this
  (thin films at brick faces). The source of the floor is still to be attributed: the wall
  `max(x + lift, -base)` clamp is active on about 25% of wall faces, and the seam and BULK16
  couplings are unchecked.

## 2026-11-15 P1-1 timing (interleaved A/B, timing slot, f72b3ca6 against f72b3ca6 + item 13)

| scene | fenced ms | pipelined ms/step |
|---|---|---|
| seep | 34.76 → 34.20 (−1.6%) | 32.23 → 32.05 (−0.5%) |
| dam | 46.52 → 45.56 (−2.1%) | 49.81 → 50.71 (+1.8%) |
| letters | 82.40 → 83.25 (+1.0%) | 70.97 → 69.52 (−2.0%) |
| paddleboats | 48.88 → 47.38 (−3.1%) | 43.71 → 45.74 (+4.7%) |

Mixed signs, inside the noise. That is expected: the change adds no dispatch and touches no iterate. Metal log
clean. Report: `~/code/me/fluid-scratch/lane-p1/timing/item13/timing.md`.

## 2026-11-15 FINDING: the Krylov recursive residual drifts; replacing it fixes the 2-norm, not the max-norm floor

These are deep runs: target 5e-4 1/s, `bandCycles` 8, dam frames 40–50, letters frames 120–130, check slot,
Metal clean.
- **The deep tree's diagnostic.** It compared the recurrence `KR = r − αq` with the true residual at `KX` at the
  final measure. Mean log10 max|KR| is −0.13 (dam) and −1.24 (letters). The true residual is −2.24 and −2.49.
  So the GCR(1) dots see a residual one to two decades larger than the real one, and α and β come from a
  corrupted r.
- **Residual replacement.** In experiment tree `trees/rr`, kApplyT and kApplyN recompute the true residual at
  the step's start, every cycle (van der Vorst & Ye 2000, SIAM J. Sci. Comput. 22(3)). They use kStart's
  formula, including the wall `max(x + lift, −base)` term. A neighbour's start value is `x − z`, because a
  neighbour outside the live tile list has a stale `P_KX`.
  - A first version read the neighbours' `P_KX` directly and lost badly: capped dam 38/61 and letters 41/141.
  - The fix makes it a valid A/B.

| run | dam: log10 max-norm, cycles 1–9 | dam KRYLOV_L2, cycles 1–8 | letters: log10 max-norm, cycles 1–9 | letters KRYLOV_L2 |
|---|---|---|---|---|
| today (recursive KR) | 2.15 … −2.17, −2.32, −2.29, −2.24 | 7.34 → 4.84, 4.81, 4.80, 4.80 | 1.74 … −2.41, −2.49, −2.52, −2.49 | 6.81 → 3.80, 3.72, 3.68 |
| replaced (true r) | 2.15 … −2.14, −2.40, −2.44, −2.45 | 7.37 → 3.62, 3.07, 2.87 | 1.81 … −2.74, −2.80, −2.76, −2.81 | 6.83 → 2.80, 2.38, 2.29 |

- **The 2-norm plateau was the drift.** With the true r, the 2-norm keeps falling by about 0.2–0.5 a cycle.
- **The max-norm floor barely moves:** about −0.2 to −0.3 log10.
- **At the default schedule** (target 0.05, 4 cycles, frames 30–90 and 100–240), the capped counts are inside
  the noise: dam 27/61 against 29/61, letters 19/141 against 15/141.
- **Decision: not landed.** It costs an extra neighbour read per row per cycle and gains no cap.
- **The floor is a few rows** (band free-surface rows, class 7) that the 2-norm minimiser does not weight.
  Driving it down is the smoother's job: the staggered Schwarz of M0(d). GCR on the max-norm would not do it.
- **For M1's solver:** use the true residual (or a periodic replacement) in the PCG dots. With a nonlinear wall
  term and the active set, the recurrence is not trustworthy past about 5 cycles.

## 2026-11-15 Lane P (pressure and bodies) from here

- The structural design was approved at 2f29b9d1, and the brick ABI was frozen at 7bc7514e.
- This lane is now lane P:
  - **M0(d):** the staggered Schwarz smoother, the exact coarsest solve and the check before the V-cycle, on
    today's arena, gated on all four scenes.
  - **M1:** the same kernels on the brick pool.
  - **M5:** bodies.
- Header lanes 128–191, lists 24–29 and args from 512 are this lane's. `PRESSURE_CAP` is STATE bit 0;
  `FAIL_PIVOT` is a failure.
- **For M0(d), the evidence so far:**
  - the arena's exact 32h coarsest gains nothing (item 3 finding above);
  - the convergence floor is band free-surface rows;
  - the V-cycle contracts by only about 0.43 a cycle.
  So the smoother comes first.
- **The candidate smoother.** pRelax's tiles freeze their halo at the last sweep, so a mode across a 4³ tile
  face relaxes only through the coarse levels. A second pass over tiles shifted by (2, 0, 2) breaks that. That
  is the seep's staggered partition, at the arena's 4³ tile instead of its 8³ brick. Post-smoothing applies it
  in reverse order (Smith, Bjørstad & Gropp 1996, ch. 1–2).
- **Its gate:** the Float64 symmetry oracle and the thin-film test (the seep-smoother.test.ts model), then
  capped counts on the probe windows.
