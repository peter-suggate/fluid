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

## 2026-11-15 M0(d) go/no-go: staggered Schwarz on the arena — convergence go, step time no-go, not landed

- **Candidate:** pSchwarz, one 512-lane workgroup per 8³ block: the brick (U), the block shifted by (4, 0, 4)
  (S), then U again, 3 red-black sweeps each in a 10³ staged tile, halo frozen at the pass entry, colours
  reversed in post-smoothing (Smith, Bjørstad & Gropp 1996, ch. 1–2). Default `schwarz: true, schwarzSweeps: 3`.
  Float64 oracle: tests/uniform-pressure-schwarz.test.ts (symmetric, beats the brick partition on the film).
- **Convergence (probe windows, default schedule):**

  | scene | capped steps base → cand | median r base → cand | deep floor (log10) | mean cycles |
  |---|---|---|---|---|
  | dam | 29/61, 27/61 → 1/61 | 0.047 → 0.020 | −2.24 → −2.42 | 4.00 → 3.95 |
  | letters | 15/141, 19/141 → 0/141 | 0.030 → 0.0155 | −2.49 → −2.88 | 3.82 → 3.62 |

- **Four-scene gates** (98b96d2c + patch, ~/code/me/fluid-scratch/lane-p1/gates/xcand11): Overall PASS
  (seep, dam, letters, paddleboats); seep G7 drift −0.0050 against −0.0051.
- **Interleaved timing A/B** (timing/schwarz11, every run rc 0, no Metal aborts 14:28–14:45):

  | scene | fenced median ms base → cand | pipelined ms/step base → cand |
  |---|---|---|
  | seep | 35.76 → 38.59 (+7.9%) | 32.57 → 35.26 (+8.3%) |
  | dam | 45.55 → 50.57 (+11.0%) | 51.69 → 56.51 (+9.3%) |
  | letters | 88.75 → 102.16 (+15.1%) | 70.99 → 83.84 (+18.1%) |
  | paddleboats | 52.51 → 51.97 (−1.0%) | 45.68 → 47.18 (+3.3%) |

- **Decision: not landed.** The cap is nearly gone, but cycles a step barely fall (the schedule stops at the
  target, not the floor), so the 3× smoother cost is paid in full: +5 to +13 ms on three scenes, far outside the
  2.5% noise. The checkout is reverted; the patch and its test are kept at
  ~/code/me/fluid-scratch/lane-p1/schwarz.patch and uniform-pressure-schwarz.test.ts.
- **What it settles for the pool MGPCG:** the staggered partition is what removes the band floor, so the pool
  solver keeps it at level 0. Its price is paid back there only if it cuts PCG iterations, which the pool
  gates must show. A cheaper arena variant (1 sweep a pass, or U·S only) was not tried, to keep the time box.

## 2026-11-15 M1: pool MGPCG at the slot — oracles pass

- lib/methods/narrow-band/pressure-mgpcg.ts, `poolMGPCG(L, {slots 8, tolerance 1e-3 1/s, fineSweeps 3,
  coarseSweeps 2})`, a `PressureSolver` for brick-abi.md §6.1:
  - 8 levels fixed at build: L0 (pres), L1 (pmgF), L2–L5 (pmgC, a C slot's tiles to the page), L6/L7
    compact (pmgTop, ≤ 512 / ≤ 64 rows, FAIL.UNSUPPORTED over), Galerkin P^T A P / 1.6 (Braess 1995);
  - L7 inverted once a step by Gauss-Jordan without pivoting (FAIL.PIVOT on a pivot ≤ 0);
  - L0 staggered Schwarz (U, S(4,0,4), U; Smith, Bjørstad & Gropp 1996), coarse block red-black GS;
  - PCG (Saad Alg. 9.1), the check before every V-cycle, the true residual every iteration;
  - setup 4 dispatches, 16 a slot, finish 1; header 128–139, gates at args 512/515/518.
- C records' neighbour words are not maintained in M1 (all NONE), so neighbour pages come from the
  directory with a key check (`pPageSlot`). That was the one bug the trace oracle found.
- Oracles (lib/methods/narrow-band/pressure-mgpcg.test.ts, GPU):
  - one V-cycle against `PressureModel({levels: 8}).vcycle`: 8.1e-7 (corner block, 31 640 rows, top
    rows 4/1/1) and 7.8e-7 (wide sheet, 168 000 rows, top rows 35/12/4);
  - hydrostatic box: 5 iterations, |X − X64|/|X| 1.0e-5, analytic 1.8e-5, air X = 0;
  - corner block converged: 6 iterations to 1e-3 h², against Float64 MGPCG 1.9e-5, air X = 0.
- Not yet registered in brick-pressure-solver.ts: per the composite-system ruling an all-h solve is
  bring-up only, so it registers with the C rows and seams.

## 2026-11-15 Census of the old engine's active set and seam heads (measure before porting)

- Instrumented the old arena, compiled in only when `UniformPressureArena.diagnostics` is true. It has
  72 census words after the per-class block (`UNIFORM_PRESSURE_CENSUS`) at these sites:
  - the band row build (pRows) and the native row build (pNative);
  - the MG active-set update for band rows (pRestrict0) and native rows (residual);
  - the Krylov start (kStart).

  The Krylov layer's joins, hits and restarts are the arena's own header words. The probe is
  tools/probe-pressure-active-set.ts. Runs were seep 300, dam 200, letters 220 and paddleboats 300
  steps (bodies driven), all rc 0 with no Metal aborts. The raw output is in
  ~/code/me/fluid-scratch/lane-p1/census/*.json.
- Values are means per step (MG and Krylov words are summed over a step's cycles), with the largest
  step in brackets.

  | state | seep | dam | letters | paddleboats |
  |---|---|---|---|---|
  | band rows | 1.7k | 103k | 153k | 62k |
  | K_WALL faces (separating wall) | 456 | 5.2k | 4.5k | 1.5k |
  | K_WALL binds (halo clamped at total 0), MG | 127 (429) | 1743 (11.3k) | 1694 (5.6k) | 4.6 (47) |
  | K_WALL binds at Krylov start | 44 | 437 | 425 | 1.2 |
  | K_SEAM faces / with nonzero bSeamHead | 460 / 427 | 23.3k / 23.3k | 38.6k / 38.6k | 19.5k / 19.5k |
  | largest \|bSeamHead\| (mean of step max) | 361 | 351 | 1296 | 595 |
  | bounded band rows (bInside) | 300 | 0 | 0 | 4457 |
  | band rows at bound (closed clamp), MG | 59 | 0 | 0 | 462 (2274) |
  | band rows frozen (no P_CORRECT), cycles 0/1/2/3 | 19/21/14/4 | 0 | 0 | 111/120/121/110 |
  | P_CORRECT changes, cycles 0/1/2/3 | 19/5.1/1.7/0.09 | 0 | 0 | 111/15/2.2/0.57 |
  | cycle-0 frozen set vs last step's end: flips | 3.2 | 0 | 0 | 82 |
  | bInside flips since the last step | 0 | 0 | 0 | 587 |
  | K_CLOSED / K_NEUMANN / K_OPEN faces | 1487 / 0 / 0 | 0 / 307 / 0 | 0 / 118 / 0 | 6372 / 2.7 / 0 |
  | native rows | 34k | 9.8k | 17k | 36k |
  | N_WALL / N_HELD faces | 512 / 3968 | 47 / 2198 | 64 / 3635 | 265 / 7901 |
  | N_WALL binds, MG / Krylov start / init | 0 | 41 (418) / 10 / 8.7 | 0 | 0 |
  | N_CLOSED rows / at bound, MG | 373 / 0.84 | 0 | 0 | 0 |
  | N_CORRECT changes, cycles 0/1/2 | 0.22/0.06/0.07 | 0 | 0 | 0 |
  | wall/held/closed state changes since the last step | 0 | 9.1 | 2.5 | 0.26 |
  | N_SEAM / N_BULKN faces | 30 / 4096 | 1464 / 16 | 2411 / 0 | 1223 / 0 |
  | Krylov joins / hits / restarts | 2.0 / 0.05 / 0.017 | 0 | 0 | 1.1 / 2.0 / 0.57 |
  | final complementarity (at bound, asking to leave) | 0.37 (29) | 0 | 0 | 0.03 (8) |

- **What binds, and therefore gets ported:**
  - *The separating wall (K_WALL, N_WALL).* It binds on every scene: thousands of face-cycles a
    step on dam and letters, where no row is bounded. A face into a non-resident page (the box
    exterior; a SOLID or BULK16 page) is a one-cell halo at max(x + lift, −base). It carries the
    wall's flux until the halo's total pressure would go negative, and then the liquid separates.
    This is the unilateral solid condition (Batty, Bertails & Bridson 2007; Chentanez & Müller
    2012). M1's brick operator closes box faces as Neumann, so this mechanism is missing there today.
  - *Bounded rows (bInside, N_CLOSED) and the P_CORRECT/N_CORRECT active set.*
    - Seep: 300 band rows and 373 native rows.
    - Paddleboats: the hulls, with 4.5k band rows. A hundred of them are frozen per cycle, and the
      set moves with the bodies (587 inside flips a step, 82 flips against the last step's end).
    - The per-cycle changes fall by about 10x a cycle (paddleboats 111 → 15 → 2.2 → 0.57). So the
      outer loop around the MGPCG has to re-decide the set, at least every cycle.
    - The Krylov layer's feasible step (Polyak 1969; O'Leary 1980) hits and joins a few rows a step.
      Its final complementarity is met on all but 0.37 and 0.03 rows a step.
  - *N_HELD.* Thousands of faces on every scene, as flux-only wall slots: an RHS term with no
    diagonal entry.
  - *The seam head.* It is nonzero on 92–100% of seam subfaces, up to |head| 4.8k (letters), so it is
    in the system everywhere a seam exists.
- **What never bound on any gate scene:**
  - K_OPEN, a band face into a non-resident AIR page. It is a ghost-fluid formula, so I'll port it
    rather than cut it, at no cost.
  - Band K_WALL masks changing between steps (0 everywhere).
  - Krylov crosses and rows under their bound at the final measure (0 everywhere).

  Nothing is proposed for cutting. Everything else binds on at least one gate scene.

## 2026-11-15 — row formulas ported, at parity with the old arena's resolved text

- `lib/methods/narrow-band/pressure-rows.wgsl.ts` holds the composite system's row formulas as pure
  WGSL with no addressing, ported verbatim from the old arena: pRows (the band face of every kind,
  the RHS, the wall lift and halo), bSeamHead, pNative (the root wall slot N_WALL/N_HELD, the native
  link and ghost, the seam RHS, the first residual with its closed clamp) and pCorrectionFree (the
  active set). The operator kernel will supply the addressing.
- `tools/nbflip-pressure-parity-extract.ts` extracts the old functions from lane F's dam capture
  (frozen-dam-120, commit a22d512d) into `parity/uniform-dam-pressure.wgsl`.
  `pressure-rows.test.ts` runs old and new side by side on the GPU, using lane F's harness:
  - The band face, over 8192 random rows reaching every kind and solid branch: bit for bit.
  - The native wall, ghost, link, seam, first residual, lift and active set, over 8192 rows: bit for bit.
  - The seam head, over 2048 rows: 180 differ, by at most 1.29 ulp of the sum of the term
    magnitudes. The head is the sum of two tangential terms, rise/count·offset, and the random bases
    (±1000) let them cancel to a head far smaller than either, so the relative error on the head
    reaches 128 ulp. Metal fuses the multiply-add or not depending on the code around it (lane F's
    finding), so the test bounds each head at 4 ulp of its terms' scale (Higham 2002, §2.6).
- Next: the composite operator kernel in pressure-operator.wgsl.ts on E's tier ABI (6f319a55:
  stored wall faces through nbFacePlane; setTransfer to reach a real 4h state), then frozen-state
  row parity on the dam.
