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

## 2026-11-15 — the operator's RHS carries the volume control

- At lane F's request (relayed by E), the M1 operator's RHS reads the volume-control rate, which
  F's gauge writes to header word 64 (FH.RATE). The RHS is now `−h(Σ A US − rate·h·capacity)`,
  with capacity 1 and no solids. This is pRows' `control = authority.z·open`, with the old
  authority.z being the same rate (uniform-mixed-surface-volume.ts; Kim et al. 2007's
  proportional control). The native rows get the same term (buildRhs:
  `−ρ(div − z·open)/dt`) when they land. A rate of 0 changes nothing. brick-engine and
  pressure-mgpcg pass on the GPU.

## 2026-11-15 — the composite rows on the tiers, at frozen-state parity with a Float64 oracle

- `lib/methods/narrow-band/pressure-composite.wgsl.ts` holds `rowsF` (F_ALL: the h rows of h-owned
  tiles) and `rowsC` (C_ALL: the native 4h rows). They put pressure-rows.wgsl.ts's formulas on E's tier
  storage through the owner readers (brick-owner.wgsl.ts). They write X (the start), RHS, DIAG,
  PPHI, K0..2, and KU0..2 (the seam to a native row above, on F only).
- Units: the slot contract's X = p·dt/ρ, so each old row is multiplied by h²·dt/ρ. The couplings
  are:
  - an h face: V/θ;
  - a seam: V/(2.5θ'), with θ' = 1 on a liquid band cell;
  - native–native: 4V;
  - a native ghost: 4V/θ (the marched umFaceSurfaceTheta);
  - a native wall: 4·0.5;
  - a band wall: V = 0.5;
  - K_OPEN: 1/θ(own, 0.5).
- The rows are written in total variables. The start enters only through the seam heads (an RHS
  shift that keeps the matrix symmetric: the band sees X_n + head, the native sees X_b − head) and
  through the walls' initial state (attached when start + lift ≥ 0; N_HELD always attached).
- Starts:
  - A warm h cell (FLAGS bit 7) starts from X.
  - A cold h cell prolongs the tiles' starts with bCoarse and bCoarseCarry, ported verbatim.
  - A tile's start is its C cell's X, which holds the tile's last pressure. The solve's finish
    must write it: the native solution, or an h tile's mean over its liquid rows (083's band-tile
    mean).
  - A first version averaged the warm h rows' X on the fly. That raced with rowsF's own X writes
    (a warm but dry h cell is zeroed while a neighbour workgroup averages it), and the RHS was off
    by 2e-6. Reading C X removes the race, because rowsF writes only F X, and rowsC, which runs
    after rowsF, writes C X only on dry native rows.
- `pressure-composite.test.ts` (GPU) uses a 32³ mixed-tier frozen state: a 4h block from the floor
  through a rippled surface beside h tiles, a 4h block on the x = 0 wall, and 70% warm h cells.
  Every written channel matches a Float64 oracle of the same rules within 2e-4 relative. Coverage:
  9046 F rows, 35 C rows, 701/52 seams (F/C) with 701 heads, 1999 attached and 146 separated walls,
  13 N_HELD, 9 native ghosts, 132 native links, 4571 cold starts and 745 surface carries.
  `H.FAIL` stays 0.
- Open deviations, measured later on the dam:
  - Non-resident ghosts use PHI_AIR in place of the old RO_FAR (0.1 m).
  - The h centre φ is the 8-vertex mean (bCentrePhi), not 036's umPressurePhiCell.
  - The owner readers and the cold-start prolongation run on the fly; their cost is unmeasured.
- Still to do:
  - The finish kernels: set and clear WARM, and write the tile's C X.
  - Wall-state bits 8–10, for the active-set outer loop.
  - The composite MGPCG.

## 2026-11-15 — the outer loop's shape, from the active-set census; verbatim centre φ and RO_FAR

- **The census** (probe and words committed in e932d604; the table is in that entry above). Its
  diagnostics compile out when `UniformPressureArena.diagnostics` is off. The table covers all four
  gate scenes, from short runs: per-state counts, changes since the last step, and the
  P_CORRECT/N_CORRECT changes per MG cycle. It fixes the outer loop's shape:
  - **N_CORRECT rebuilds never change anything** on dam, letters or paddleboats. On seep they
    change 0.22, 0.06 and 0.07 native rows a step on cycles 0, 1 and 2, all N_CLOSED rows, so all
    solid. On the dam a rebuild never fires.
  - **P_CORRECT changes** occur only where rows are bounded (bInside), on seep and paddleboats. On
    paddleboats they run 111, 15, 2.2 and 0.57 a step over cycles 0 to 3: the set is still moving
    at cycle 3, so a fixed 1–2 outer passes would not converge it. The old engine re-decides the set
    every MG cycle, and the Krylov layer's feasible step (Polyak 1969; O'Leary 1980) joins and hits
    a few rows a step after that. Both arise only with solids (cut bricks, which latch UNSUPPORTED
    now).
  - **On the dam and letters, the whole active set is the separating wall.** K_WALL binds 1743 and
    1694 times a step (MG), and N_WALL binds 41 on the dam. The wall/held/closed state changes 9.1
    and 2.5 times a step. So the M2 solver's set is the walls' halo, max(x + lift, −base), applied
    each MG cycle as the old one does, with the attached state re-measured at the Krylov start.
    No separate outer loop of whole MGPCG solves is needed for M2.
  - The outer loop for bounded rows (P_CORRECT/N_CORRECT, Chentanez & Müller 2012) follows the old
    per-cycle rule when solids land. The census shows the set does not settle in fewer passes.
- **The h centre φ is umPressurePhiCell, verbatim.** With no solids it is the old centerPhi texel:
  gcEvaluate's corner mean, each of the 8 vertices times 1/8, then d4Sum8's pairing
  (uniform-mixed-surface-geometry.ts). poCentre now uses that order, as does the 4h root's corner
  mean (poMean8). The closed-cell branch needs the solid open fractions, and a cut brick latches
  UNSUPPORTED.
- **RO_FAR, verbatim:** a tile no page stores (off the lattice, or in a non-resident page) has root
  φ 16 cells, not PHI_AIR's 6.
- **C slot entry X = 0:** E confirms setInit zeroes every pres channel of an entering C slot.
  pressure-composite.test.ts now asserts it on every C slot after create. The test passes as
  before.
- **Absent vertex reads (with E).**
  - The old engine's reads of a vertex in an absent page return the field's empty value,
    fluidLatticeFarPhi(scene). E's readers now return NB_PHI_FAR, the lattice's largest extent in
    h, which is the same value. Stored air stays PHI_AIR = 6. The composite oracle reads FAR there.
  - A page's upper-face vertices belong to the next page. Residency (setMarkPages: at least
    7 cells of reach) keeps such pages resident near liquid, so a liquid row never reads them. The
    rows now assert it: a liquid row whose own interior corner vertices fall in an absent page
    latches FAIL.TIER.
- **The composite MGPCG's Float64 reference** is `pressure-composite-model.ts`.
  - The h rows are smoothed at L0 and L1. The native rows are carried through L0 and L1 and
    smoothed with the h tiles' aggregates from L2 up.
  - The coarse operators are W⁻¹PᵀAPW⁻¹, with W = √OC on aggregated rows and 1 on carried ones.
    Aggregates keep the pool MGPCG's OC over-correction (Braess 1995); native rows take none. The
    operator stays SPD.
  - `pressure-composite-model.test.ts` uses a synthetic mixed system: 10233 h rows and 48 native
    rows, with seams, walls and ghost faces. The operator is symmetric, the V-cycle is SPD, and
    PCG reaches 1e-9 in 8 iterations.
  - The GPU solver follows this design. It is saved in
    ~/code/me/fluid-scratch/lane-p1/composite-mgpcg-design.txt.

## 2026-11-15 — the composite MGPCG on the GPU (pressure-mgpcg.ts)

- **poolMGPCG now solves the composite rows: h rows in pres F, native 4h rows in pres C.** The
  all-h solve is unchanged: the pool MGPCG tests give the same V-cycle errors, iteration counts and
  X as before.
  - **Native neighbours.** A cell of a 4h-owned tile inside an F brick has DIAG 0. Its F slot is
    never read for a value. The brick head stages each neighbour brick's native tiles (8 bits) and
    its page's C slot, and pNatAt reads pres C. This also covers the stale W0 of DIAG = 0 cells.
  - **The h side.** h rows' apply, pRow and the L0 tile staging read a native neighbour's value:
    X (+ αD), Z, 0 before the coarse correction, and its L2 EPOST after it. The upper coupling is
    the row's own KU.
  - **The hierarchy (the composite model).**
    - An F1 or F2 link to a native row is its children's sum over √OC. The upper one is in the
      new L1 and L2 KU channels.
    - pmgCoarsenC (now a lane a tile) writes the native L2 rows unscaled. A native row's link to an
      F2 tile below is that tile's L2 KU. pmgCoarsenC also clears h tiles without a liquid F brick,
      and aggregates L2 → L3 → L4 → L5 (moved from setupF).
    - pmgDown1C (a page) computes a native row's L2 right side, r + Σ_m ch(m)(e0(m) + e1(m/2)/√OC),
      from the seams gathered from the h side. It then pre-smooths L2 over all 64 tile blocks of
      the page. Page kernels skip pages with L5 D = 0. The native kernels skip pages with no
      native rows (a count in L5 KU0).
  - **Native PCG vectors.** These run in page kernels beside the brick kernels: the setup residual
    (in pmgCoarsenC), z = L2 EPOST (in pmgUp2, with r·z), pmgDirectionC, pmgUpdateC and
    pmgFinishC. Their dots share pmgRed. RMAX counts a native residual over 64.
- **The GPU against Float64** (`pressure-composite-mgpcg.test.ts`). The test uses the composite
  rows' frozen mixed-tier state, now shared as `parity/composite-fixture.ts`: 9081 rows, 35 of them
  native, with all-native and mixed F bricks.

  | check | h rows | native rows |
  | --- | --- | --- |
  | r0 vs b − A x0 | 5.8e-8 | 5.8e-8 |
  | one V-cycle, Z vs model.vcycle | 4.9e-7 | 5.9e-8 |
  | converged X vs Float64 PCG | 5.4e-7 | 2.8e-7 |

  The converged run took 8 iterations, with r0 3.9 → 4.4e-7 (tolerance 9.8e-7).
  - The first run had a sign error in the native L2 right side: r − A e has +ch, because A's
    off-diagonals are −ch. The V-cycle was 10–17% off, and PCG still converged (it is a valid SPD
    preconditioner). The fixed version matches.
- **Next:** the finish (WARM, the h tiles' C X mean, dry X = 0), project with seam faces and the
  Neumann hand-back, the wall outer loop, and registration.

## 2026-11-15 — operator and project in place, the native rows and faces, the solver registered

**In place** (brick-abi.md §6: operator and project are P's, extended in place):
- `operator` (nbOperator, F_ALL) is now the composite h rows, rowsF. It also writes B, the row's start, which is a
  pres channel E added for this (37893df6). On every cell that is not a row (dry cells, cells of 4h-owned tiles,
  and the skirt), X is 0: the slot contract's zero rows.
- `project` (F_LIQ, which today equals F_ALL) is the composite h-face projection, umProject's bFace in the slot
  units:
  - U = US − sign·(c/V)·(ghost − X)/h, evaluated from the liquid side, with the ghost set by the face kind:
    - band: the neighbour's X;
    - seam: the native X plus the head. The head comes from bSeamHead on the stored starts B and PPHI, so it
      is the head the row's RHS carried.
    - OPEN and air: 0;
    - separating wall: X + lift while attached in the state the row took (start + lift ≥ 0), otherwise 0.
  - Neumann keeps US.
  - From a dry h cell, only a seam to a liquid native row moves: the native row's zero ghost.
  - It raises BOUND_F and writes FLAGS.WARM (bit 7) on every cell: set on h-owned liquid rows, cleared
    elsewhere, so a refined tile's rows start cold.
- The native rows (rowsC, C_ALL) open the registered solver's setup, because the engine places one operator.
- `projectC` (C_ALL) closes its finish:
  - it writes every 4h face stored at a page's tiles: native–native gives the 4h gradient (c/(16hV) = 1/(4h));
    native–air or a non-resident page gives the marched θ ghost; a native wall separates in the rows' state,
    or N_HELD; a face with no liquid side gets U = US;
  - it writes each h tile's C X as the mean X of its liquid rows (083), which is next step's tile start.
- Registered: `brickPressureSolver = compositeMGPCG()` (brick-pressure-solver.ts), which is poolMGPCG with those
  two placements.

**Consistency on the composite fixture** (pressure-composite-mgpcg.test.ts, new test):
- On every liquid row, −h Σ sign·V·U + h²·rate equals the row's Float64 residual b − A X: 9,046 h rows to
  1.45e-7 and 35 native rows to 3.74e-7, relative to h·max|U|. A native row uses 16× that, with its seam faces'
  16 h patches.
- WARM is exact, and the h tiles' C X mean is within 2e-8.
- E's engine tests (hydrostatic with the bring-up CG, the zero rows, tiers, transfer, owner, census) and the pool
  MGPCG tests pass on the composite operator: 25/25 plus 27/28 earlier. The 28th was zero X on the skirt, now
  fixed.

**First dam runs** (all-h, which is what F's stages support today; f 4913; scratch tree tree-reg):
- Drift, KE, front and wet floor match F's dam-300c to 3 digits through step 60. At step 16: drift −0.0576 vs
  −0.0570, KE 282.4 vs 281.3 J, wet floor 15,943 vs 15,917.
- Profile (per-dispatch passes, steps 13–60):
  - pressureRowsC is 448 ms and operator 85 ms.
  - The MGPCG kernels sum to 48.5 ms at 7.8 iterations, about 6.0 ms an iteration over about 690k rows: 8.7 ns
    per row per iteration under the per-dispatch profile.
- rowsC's cost comes from 296 phantom native rows. In the all-h engine, 4h-owned tiles of non-resident bricks
  read C φ, which is still bootPhi's column after the bricks leave. Reported to E with a proposal: all-h CMASK
  all-ones.
- operator's cost comes from owner-resolved vertex reads. A cold row's prolongation alone reads 125 vertices a
  tile, and the seam heads prolong their neighbours. Next: stage the brick's halo vertices and its 4×6×4 tiles'
  root φ and starts in workgroup memory, which changes no values.

**Open:**
- The wall state's outer re-decision (Chentanez & Müller 2012), as the census sized it.
- The t = 0 seam-head Picard iterations.
- The dam gate on the composite rows at 300 steps, and ns per row per iteration against seep-m1max (about 0.55 ms a
  cycle over about 620 liquid bricks, so roughly 1.7 ns per row per iteration).

## 2026-11-15 — staged rows and project, the dam gate on the composite solver, ns per row per iteration

**Staging** (pressure-composite.wgsl.ts; values unchanged, so the composite tests print the same numbers to the last digit):
- rowsF and projectF stage their brick's halo in workgroup memory once per brick:
  - vertices o−1..o+9 (1,331), computed by nbVertexPhi;
  - centres o−1..o+8 (1,000), from poMean8 of the staged vertices;
  - the tile owner bits of tiles 2B−1..2B+2 (64).
- Readers go through poVertex, poCentre and poOwnerH, which fall back to the direct reader outside the window.
- If a brick has a cold row (liquid without WARM), rowsF also stages the root φ and start of the 4×6×4 tiles that
  poStart and the carry read (poRootPhi and poTileStart, one lane a tile).
- The absent-page guard becomes tiered-only. All-h, an absent page is h air, the same as an absent brick
  (brick-abi.md §2.5 Residency, E's 27852ac1), so the old guard caught only the page half of that case.
  `pressureOperatorKernels(L, { tiered })` takes the engine's flag; the default keeps the guard. All-h, it latched
  TIER intermittently: 2 of 4 runs, at steps 127 and 179. The 179 run's vertex (40, 64, 31) sits on the page-y-2
  boundary under the spray. Residency there is E's and F's question (reported). F's dam-300c ran the same states
  without a guard.

**Dam profile** (HEAD 27852ac1 with E's all-h CMASK, so natives 0 and wet4 0; 60 steps, per-dispatch passes, steps 13–60):

| kernel | before | after |
|---|---|---|
| pressureRowsC | 448 ms | 0.04 ms (no 4h tiles all-h) |
| operator (rowsF) | 85–110 ms | 9.1 ms |
| project (projectF) | 8.0 ms | 2.7 ms |
| profiled step median | 200 ms | 110 ms |

- About 21k cold rows a step (3% of about 687k), measured by a set difference of DIAG > 0 across steps 20–23.
  Their prolongation was the operator's cost.

**Dam gate** (`dam-p-300c`, fluid-scratch/lane-p1/gates; 300 steps on the check slot, rc 0, no Metal aborts; judged by
tools/nbflip-gates.ts against the five ref-2761350a references):
- C0, D1, D2, D3, D4, D4c, D5c, D5ce, D5k, D5p, D5e, D7t, D7d and D8 pass.
- D6 (wet floor at 8 and 12) fails at 1.12 and D8c (particles at 16) at 1.16, the same rows and margins as
  F's dam-300c (1.10, 1.19). F attributes D6 to the extension's reach and D8c to the seeding, so neither is the
  pressure.
- The fenced check-slot median is 75.0 ms (references 46–68 ms, informational).

**ns per row per iteration** (the profile, MGPCG kernels only, 7.83 iterations a step): 5.08 ms an iteration.
- Per liquid row (about 687k): 7.4 ns, against seep-m1max's about 1.7 ns (0.55 ms a cycle over 620 liquid bricks,
  about 317k cells).
- Per brick cell (4,685 bricks × 512): 2.1 ns, against seep-m1max's 1.7 ns.
- So the gap is mostly occupancy: all-h, the dam's bricks are 29% liquid rows. Tiers (4h bulk, the band at h)
  close it, or a row-compacted level 0.
- The heaviest per-iteration kernels are pmgPostS and pmgPreS (0.78 ms each) and pmgUpdate (0.75 ms).

**Open:**
- The wall state's outer re-decision (Chentanez & Müller 2012).
- The t = 0 seam-head Picard iterations.
- The engine passing `{ tiered }` to pressureOperatorKernels (E's brick-engine.ts).
- MGPCG level-0 occupancy.

## 2026-11-15 — the active-set table, with the wall's state at the solve's end against the build

**Probe:** tools/probe-pressure-active-set.ts on the old engine, all four gate scenes, 120 steps each (paddleboats
with its bodies driven), on the check slot. All runs rc 0 with no Metal aborts. Raw output is in
fluid-scratch/lane-p1/census2/*.json. Three census words are new, all diagnostics-only:
- 37: band K_WALL faces whose state at the solve's end (x + lift ≥ −base, pFinal) differs from the build's
  (lift ≥ −base, the iterate at its start). The build's state is what the composite rows decide today
  (start + lift ≥ 0).
- 38: the same for the native N_WALL faces (pResidual2's last measure).
- 39: band K_WALL faces separated at the build.

Means a step, the largest step in brackets.

| state | seep | dam | letters | paddleboats |
|---|---|---|---|---|
| band rows | 1.5k | 113k | 103k | 50k |
| band K_WALL faces | 188 (369) | 5.9k (14.1k) | 1.9k (5.7k) | 910 (2.0k) |
| band K_WALL separated at the build | 3.2 (12) | 459 (1743) | 47 (334) | 0.41 (6) |
| **band K_WALL state at the end ≠ the build's** | **23 (53)** | **441 (1791)** | **51 (412)** | **0.79 (8)** |
| band K_WALL binds, summed over MG cycles | 63 (172) | 2.0k (8.4k) | 310 (2.1k) | 3.4 (36) |
| band K_WALL binds at the Krylov start | 22 (53) | 514 (2.2k) | 78 (521) | 0.91 (9) |
| band wall masks changed since the last step | 0 | 0 | 0 | 0 |
| native rows | 34k | 9.5k | 18k | 36k |
| native N_WALL faces | 512 | 67 (158) | 111 (192) | 303 (352) |
| native N_WALL lift below start (init) | 0 | 12 (74) | 0 | 0 |
| **native N_WALL state at the end ≠ the build's** | **0** | **9.6 (61)** | **0** | **0** |
| native N_WALL binds, MG | 0 | 52 (313) | 0 | 0 |
| native wall/held/closed changes since the last step | 0 | 13 (55) | 2.4 (20) | 0.08 (3) |
| bounded band rows (bInside) / flips since the last step | 294 / 0 | 0 | 0 | 4.5k / 584 |
| P_CORRECT changes, cycles 0/1/2/3 | 18.5/5.2/1.7/0.11 | 0 | 0 | 118/15.5/2.0/0.49 |
| N_CLOSED rows | 373 | 0 | 0 | 0 |
| N_CORRECT changes, cycles 0/1/2 | 0.24/0.06/0.07 | 0 | 0 | 0 |
| Krylov joins / hits / restarts | 1.9 / 0.08 / 0.03 | 0 | 0 | 1.1 / 1.6 / 0.49 |
| MG cycles | 2.5 | 4.0 | 3.0 | 2.9 |

**What it decides:**
- **Item 2 (the wall re-decision).** The dam does re-decide its walls after the first build: 441 band faces a step
  (7.5% of its wall faces) and 9.6 native faces end in a different state from the build's. So does every other
  scene: seep 23, letters 51, paddleboats 0.8. The old engine re-decides with max(x + lift, −base) at every
  residual and smoother evaluation. The composite rows fix the build's state for the whole solve. So the outer
  loop is needed on every scene, the dam gate included.
- **N_CORRECT** (bounded native rows) changes anything only on seep (0.24, 0.06 and 0.07 a step on cycles 0–2).
  Those are N_CLOSED rows, all solids, which latch UNSUPPORTED today. P_CORRECT needs bodies (seep, paddleboats).
  Neither binds on the dam or letters.
- **Item 3 (the seam-head Picard at t = 0):** the seam heads are nonzero on 92–100% of seam subfaces (the earlier
  table). The Picard runs once at t = 0 only, so this table doesn't change its case.

## 2026-11-15 — item 2: the separating walls' outer loop around the MGPCG; rowsC 543 → 0.044 ms on the tiered dam

**The loop** (pressure-mgpcg.ts `walls`, on in compositeMGPCG). It follows Chentanez & Müller 2012 (multigrid with separating
solid boundaries), and its decision rule is the primal-dual active set of Hintermüller, Ito & Kunisch 2002. That is a
semismooth Newton step on the wall ghost max(x + lift, 0).
- **State.** FLAGS bit 8+a sits at the face's ABI storage: the cell or tile itself on plane 0, the skirt cell or tile N/4
  on plane N. rowsF and rowsC write the build's state (start + lift ≥ 0, as before). The single writer of each word is
  the face's liquid row.
- **When it decides.** `pmgWallGate` is one workgroup after updateC. It opens WF and WC (args 521, 524) when the slot is
  open and either max|r| ≤ tol or the slot is the last. A slot that doesn't decide launches nothing.
- **What it decides.** `pmgWallsF` (edge bricks) and `pmgWallsC` (edge pages, N_HELD excepted) re-decide every lattice wall
  face of a liquid row at the iterate x + αp. A face that changes state moves c (F 0.5, C 2) between the RHS (c·lift) and
  the diagonal. The row's residual becomes r + ΔRHS − ΔDIAG·(x + αp); the coupling is diagonal-only, so no other row
  moves. A changed brick's first V-cycle pass (W0) is redone from the new r and rows. The new max goes to RMAXW, so every
  walls workgroup decides on the update's max.
- **Restart.** pmgCheck takes max(RMAX, RMAXW). On a change it restarts PCG (RZ = 0, so the next β is 0) and keeps α,
  because the deferred x += αp is the iterate the walls decided at. Header words: FLIPS 140, WALLS 141 (changes this
  step), DECIDES 142, RMAXW 143.
- **Where the old rules differ.** The old engine evaluates max(x + lift, −base) at every residual and smoothing of its
  V-cycles, freezes the state at the Krylov start, and projects with the final x. Here a solve's system stays fixed until
  it converges (a Krylov method on a fixed SPD matrix), and the state changes only between converged solves, or once at
  the last slot. So a capped solve still ends on its final iterate's state, as the old projection does.
- **Preconditioner.** The coarse levels keep the setup's aggregates of the build's diagonal. The V-cycle stays SPD: a
  symmetric smoother on the current L0 rows plus any SPD coarse correction. So it remains a valid preconditioner.
- **The projection uses the final state:** projectF (poProjectFace) and projectC read the bit, not B + lift.

**Lane E's release bit (agreed with E).** After projectF and projectC, bits 8–10 at the same storage are the old umRelease:
- not held (PPHI < −width, with width 1 for F and 4 for C);
- the projection's wall ghost ≤ 0 (attached: X + lift; separated or dry owner: 0), which is the old "wall-slot pressure ≤ 0";
- and −sign·U·dt > 1e-4·width·h.

projectF writes bits 8–10 of every cell of its bricks, and projectC writes them for every tile of its pages, with 0 off
the walls. Between the rows and project the bits are P's solver state, which E doesn't read. The §2.4 row of brick-abi.md
(E's file) should say both meanings.

**rowsC (lane F's report: 543 ms a step, 4 ms a page on the tiered dam, tripping the macOS watchdog).** Each seam
subface's head called poStart for its tangential h neighbours: a cold row prolongs 8 tiles' root φ (an h tile's held fit
is 125 vertices) with the carry. That is 16 subfaces × 4 neighbours a seam face. rowsC now reads what rowsF stored: the
centre φ from PPHI and the heads through poHeadB (starts from B). These are the same values, and the rows oracle test
passes unchanged. On the tiered dam (13 steps, --profile=11) rowsC takes **0.044 ms** and the profiled step 57 ms.

**Tests** (GPU, pressure-composite-mgpcg.test.ts):
- On the composite fixture's cold, frozen state the loop takes 6 decisions and 1458 changes, 553 h and 1 native net of
  the build, in 32 iterations.
- The converged X is the Float64 PCG's on the final rows (h 4.3e-7, native 2.5e-7).
- Complementarity at the end, converged (40 slots) and capped (3 slots): wall state ≠ (X + lift ≥ 0) on 0 of 2141 h faces
  and 0 of 4 native faces.
- Projected divergence = residual (h 2.5e-7, native 6.5e-7).
- The release bits match the CPU umRelease on all 4865 wall faces (1697 released).
- The rows oracle, pool MGPCG and unit tests pass (1830 / 0). Types are clean apart from the 3 lane-n errors.

**Dam (all-h, 60 steps, trees at 74712f19, check slot, rc 0, no aborts):**

| | median step | p95 | drift @16 | KE @16 | wet floor @16 | wall changes / step (15, 16, 30, 45, 60) |
|---|---|---|---|---|---|---|
| base (no loop) | 105.8 ms | 109.9 | −0.0579 | 282.0 J | 15989 | — |
| walls | 109.4 ms | 115.3 | −0.0576 | 282.3 J | 15969 | 28, 184, 2583, 2901, 403 |

The tiered profile prices the loop at 0.18 ms a step (gate 0.066, wallsF 0.087, wallsC 0.022). The all-h median
difference is mostly run-to-run noise.

**The dam caps every step: 8 of 8 slots at tol 1e-3, before and after this change.** So the loop decides once a step, at
the last slot, and never re-solves after a change. The dam's walls are now consistent with its final x, as the old
engine's are. Restarted solves need slots the dam doesn't have, so the convergence rate is the real gap; that is item 4's
question.

**Dam gate (all-h, 300 steps, check slot, rc 0; fluid-scratch/lane-p1/gates/dam-w-300.md), against dam-p-300c:**
- It FAILs on the same two rows. D6 (wet floor cells, 10162 against 10765 at step 8) is unchanged at margin 1.12. D8c
  (particles at 16) goes 1.16 → 1.26.
- Every other row passes, e.g. D4c drift at the impact 0.84 → 0.76, D4 0.35 → 0.62, D5p 0.45 → 0.79.
- Step median 69.5 ms, p95 108 (75.0 / 112 at the older HEAD).
- The walls' state isn't what holds D6. The wet-floor front is lane E's released-walls port, which now reads the release
  bits above.

## 2026-11-16 — the solve's iteration count: the Jacobi guess and the walls decided at it

**No-gos from the rows work (logged, dropped).**
- rowsF (about 1400 µs over HB, about 850 bricks) does not respond to ILP restructuring. A skip of empty lanes, a
  hoisted face loop and a restructured stage were each flat or slower, and every added code path costs 2–5%. Its
  attribution: stage and writes about 550 µs, face loop about 500 (velocity loads 171), heads 60, starts 50,
  facePhi 57, page words 41.
  - The two-pass stage took 5 vertex rounds against 3. With the enumeration fixed it was still no win.
- Folding SetupF into rowsF: no-go. Only Apply (44–57 µs) and RKU (22 µs) would go away. The reductions, the
  aggregation and the writes would move into rowsF with the fold.
- WallsF/WallsC's two lane-0 pRedSum calls moved to pRedSums: no gain.

**Attribution.** I dumped the composite rows (h and native: DIAG, K, KU, RHS, X, start B) at steps 16–18, 20 and 22
of the tiered dam. Then I solved them in the Float64 model (pressure-composite-model.ts, OC overridable in a scratch
copy) with the walls fixed at the build:
- **Start.** The warm start's max|r| is 1e5–3e5 1/s, on surface rows whose ghost-fluid diagonal grew (r ≈ D x0).
  After one Jacobi step it is 1.2e3–1.5e3. The model's PCG took 8.0 iterations from x0 over the six states, 6.5 from
  x0 + D^-1 r0 on every row, and 6.2 with the step on the h rows only. From 0 it took 7.
  - What is left is real. 43% of the h rows are new each step: refined (34k) or newly wet. The warm rows' X changes by
    25–40% a step during the impact (rms dX 0.015 on X 0.05, and successive increments correlate at 0.9).
  - Linear extrapolation, 2X_n − X_{n−1} on rows warm in both steps, cuts the start's rms error 0.0152 → 0.0102.
    It is worth about 0.3–0.5 iterations in the model. Parked: rowsF's warm start would need the last step's start,
    which rowsF overwrites, and its halo reads X.
- **Residual history.** There is no tail stall. Rows over tolerance after iterations 1–6: 139k, 132k, 112k, 48k, 4.4k,
  16. Brick-face rows are 58% of them, the same as their share of all rows. The worst rows are interior rows at the
  impact against the far walls.
  - The max-norm falls about 5x an iteration. The seep's own model falls 3–10x (seep-techniques.md §2: white-noise
    5/7/9 its to 1e-2/1e-3/1e-4). So the dam's count is set by the start's 4–7 decades, not by a weak cycle.
  - The seep's 3.3 cycles are on the seep scene, where the drop from start to tolerance is a few decades.
- **Smoother and OC (model, from the Jacobi start, its over the six states).**

  | setting | iterations |
  |---|---|
  | OC 2 | 6.5 |
  | OC 2.2 | 6.7 |
  | OC 2.5 | 7.0 |
  | OC 1.8 | about 6.7 |
  | OC 1.6 | about 7.3 |
  | OC 1.4 | about 9 |
  | fine sweeps 4 | 6.0 |
  | fine sweeps 6 | 6.0 |
  | coarse sweeps 4 | 6.3 |
  | native sweeps 4 | 6.3 |
  | BS | 6.7 |
  | 2 or 3 Jacobi sweeps | 6.3–6.5 |

  - The seep's OC 1.6 is worse here.
  - Fine sweeps 4 is −0.5 iterations for one more sweep in each of the six L0 passes a slot: not taken.
- **Walls.** With the guess, the default policy decides within 1000x tolerance. The guess's first slot is often above
  50 1/s, so the decision moves to slot 2 and its restart costs about 1.3 iterations: 7.50 its, against 6.17 with the
  build's state alone (24-step dam, steps 13–24, two runs each).

**Change (pressure-mgpcg.ts).**
- **pmgGuess.** Before setupF, every h row's start takes one Jacobi step, Z = x0 + D^-1 (b − A x0) (Saad 2003 §4.1, the
  seep's pcgGuess). Native rows keep x0. setupF forms r from Z and writes X = Z. The guess builds the stage cache,
  which setupF then reads.
- **pmgWallsG.** The walls' decision is taken again at the guess, before the setup forms r and the aggregates, so it
  restarts nothing.
- **Re-decision at the first slot.** The one re-decision is now at the first slot's iterate (MG_WALL_LOOSE Infinity).
  Before the guess, 1000x tolerance was in practice the first slot's iterate.
- **Options.** `guess` and `guessWalls` default on. guessWalls runs only with the walls and wallDecisions > 0, so
  wallDecisions 0 keeps the build's state.
- **The physics is the same.** Operator, criterion and wall rule are unchanged.

**Iterations and walls** (tiered dam, 24 steps, steps 13–24 mean; "inconsistent" is the share of h wall faces whose
state differs from X + lift ≥ 0 at the end, summed over steps 13–24):

| policy | its | capped | inconsistent |
|---|---|---|---|
| no guess (HEAD) | 8.00 | 3 | 1.26% |
| guess, decide within 1000x | 7.50 / 7.50 | 0 | 1.13 / 1.48% |
| guess, decide at the first slot | 7.08 / 7.08 | 0 | 1.46 / 1.40% |
| guess, decide at the guess and within 1000x | 7.33 | 0 | 0.85% |
| **guess, decide at the guess and at the first slot (default)** | **7.08** | **0** | **0.80%** |
| guess, decide at the guess only | 6.08 | 0 | 8.3% |
| guess, the build's state only | 6.17 | 0 | 12.2% |

- **The 60-step dam** (samples at 15, 30, 45 and 60):
  - Iterations: 8, 7, 6, 8 → 8, 7, 5, 6.
  - Capped steps: 3 → 0.
  - Drift: −0.0268/−0.0669/−0.0296/−0.0598 against −0.0268/−0.0659/−0.0316/−0.0591.
  - KE at 60: 69.6 → 68.6 J.

**Timing** (one hold, tiered dam, rowsWindow = the fenced median's steps 18–23, per-area mean):

| | HEAD (hd) | guess |
|---|---|---|
| iterations, steps 18–23 | 8, 8, 8, 8, 8, 8 | 8, 8, 7, 7, 7, 7 |
| pressure solve | 9.07 ms | 8.59 ms (Guess 0.087 and WallsG 0.022 included) |
| setup and projection | 3.99 ms | 3.93 ms (SetupF 339 → 273 µs: the stage moved to Guess) |
| fenced median, steps 18–23 (n 6) | 75.5 ms | 74.4 ms |

- **What an iteration is worth.** A saved iteration is worth about 0.88 ms, not the open slot's 1.15. A gated, closed
  slot still costs about 0.28 ms: its roughly 25 empty indirect dispatches.
  - On this window an iteration saved is worth less than cutting the closed slots' cost. In calm steps of 4–5
    iterations, 3–4 closed slots cost about 1 ms.

**Tests.**
- The composite r0 test now builds the guess in the model: |R − r0|/|r0| 5.2e-6.
- The pool V-cycle tests run with `guess: false`, so their r0 is the RHS.
- The live-lists test is bit for bit: the guess has setupF's live early-out.
- Converged X against the Float64 PCG: h 7.7e-6, native 8.3e-6.
- Walls default policy: 349 of 2141 h faces inconsistent, against the build's 998.
- GPU pressure suite: 21/0. Unit: 1842/0. Types are clean apart from the 3 lane-n errors.

## 2026-11-16 — the guess's dam-300 gate; the slots' launches measured; walls only in the deciding slots

**dam-300 on 9b39ae68** (check slot, tiered composite, poolF 4913, lane F's brick-dam-l.mts with D8 counted as the
receipt; judged on ref-2761350a; `fluid-scratch/lane-p1/gate300/`). rc 0, no 00000206 aborts. Against lane F's g1
run (259e2e59 + lUnion/lcCouple, the same runner, `lane-f/logs/rc/gates-g1`):

| | C0 | D1 | D2 | D3 | D4 | D4c | D5c | D5ce | D5k | D5p | D5e | D6 | D7t | D7d | D8c | D8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g1 | 0.00 | 0.80 | 0.10 | 0.38 | 0.05 | 0.00 | 0.17 | 0.10 | 0.19 | 0.03 | 0.01 | 0.57 | 0.35 | 0.12 | 1.68 F | 0.83 |
| guess | 0.00 | 0.80 | 0.10 | 0.49 | 0.04 | 0.00 | 0.17 | 0.10 | 0.22 | 0.01 | 0.00 | 0.58 | 0.35 | 0.13 | 1.59 F | 0.77 |

- No row moves by more than 0.1 (D3 +0.10: 30-90 window 43.91 against 44.40, ref 45.30, tol 2.85).
- D8 window means: 6.792e5 / 4.062e5 / 3.750e5 against g1's 6.834e5 / 4.072e5 / 3.761e5 (ref 6.309e5 / 3.838e5 / 3.484e5).
- D8c fails as before, the pre-existing low count: 2.957e5 against g1 2.926e5 (ref 3.516e5, tol 3.52e4).
- Drift at 300: -1.03% (g1 -0.99%).

**What a closed slot costs, unprofiled.** The 0.28 ms a closed slot above came from the profiled floor run, which
puts every dispatch in its own timestamped pass. Unprofiled (one timing hold, tiered dam, 36 steps, fenced median of
steps 18-35 less the samples, n 18; `runs5/T1-*`):

| | slots 8 | slots 16 |
|---|---|---|
| HEAD 9b39ae68 | 74.59, 76.17 | 75.71 |
| walls in the deciding slots only | 75.51, 74.53 | 76.81 |

- Iterations are 6-8 a step (1 closed slot at 8, 9 at 16), so slots 16 adds 8 closed slots a step.
- +8 closed slots: about +1.1 ms over the four pairs (-0.5 to +1.8 run to run), so a closed slot is about 0.13 ms
  (0.07-0.2), about 5.5 µs a launch of its 24. That is half the profiled figure. On the dam window it is about
  0.13 ms a step in all.

**Walls in the deciding slots only (this commit).** With wallLoose Infinity a slot k decides exactly when
k < wallDecisions, so slots 1-7 now encode no wallGate, wallsF and wallsC: 24 launches a slot to 21, 189 a solve
to 168 at 8 slots. Bitwise by construction (those slots' WF and WC gates were 0). Unmeasurable in the hold above:
slots 8, 75.0 against 75.4 mean of the two medians.

**The native passes' order (Float64 model, six dumped states, the Jacobi start; `cmodel.ts` NAT).** Merging a
native pass into an h launch needs the two halves independent:

| native passes at L0 and L1 | mean its |
|---|---|
| multiplicative, h then native in pre (the GPU's) | 6.17 |
| additive at both levels (-4 launches a slot) | 6.67 |
| additive at L0 only, or L1 only (-2) | 6.67, 6.67 |
| native first in pre at L0 (-1: nativePre0 into pmgUpdate's C half) | 6.67 |
| none | 8.67 |

- Each costs about 0.5 iterations (about 0.9 ms a step) for at most about 0.1 ms of launches a slot: no-go.

**What is left.** The slot is a chain of about 19 global dependencies: L0 S, U, the native L0 pass, D0, the native L1
pass, D1, D1C, D2, D3, top, U3, U2, the native L1 post, U1, the native L0 post, U, S, Z, then direction, update and
check. Without changing the cycle, the remaining fusions are page kernels that recompute a neighbour's values:
- D1 into D1C (a page computes its bricks' L1 residuals);
- U3 into U2 (a page also post-smooths the 24 face-adjacent L3 blocks it reads).
- At about 5.5 µs a launch plus the tail, these are worth about 0.1-0.2 ms a step together.

**Ruled (coordinator, 2026-11-16).** The D1 -> D1C and U3 -> U2 fusions above (about 0.1-0.2 ms a step together) are
below the large-wins bar: logged as available, not taken.

## 2026-11-16 — pmgTopCycle off the single core's loads; the slot's kernels priced in bytes

**Attribution rig.** The per-dispatch profile cannot price the one-workgroup top: its timestamps tick at about 10.9 µs
and the same unchanged kernel reads 11-23 µs across runs. Because the top is idempotent (it rebuilds B5 from L4 and
rewrites only the L4/L5 post values), a scratch build dispatches it 21x in every slot. In an unprofiled timing hold
(tiered dam, 36 steps, fenced median of steps 18-35 less the samples, n 18), the step then carries 20 x (open slots,
about 6.9 a step) extra calls, so (step with repeats - step without) / 138 is the top's unprofiled cost per call
(`runs5/T7`, `T8`, `T10`; scratch tree `tf`, switches TV):

| top, unprofiled, µs a call | |
|---|---|
| HEAD 45356bb8 | about 74 (86.5 against 76.2 ms) |
| L4 residual folded into pmgDown3 | about 63 |
| + the L4 post cached a lane a cell | about 43-53 (three sessions) |
| 20 empty 1024-lane dispatches instead of the repeats | about 15-20 a dispatch |
| the 16 L5/L6 passes doubled | +18, so about 1.1 µs a barriered pass |
| workgroup 512 instead of 1024 | no change (83.6 against 82.3) |
| the six link terms written out (constant indices) | no change (82.0 against 81.5): not kept |

- Why: the top is one core's load stream. Before this change, its L4 residual was 94 lanes, each looping over 8 cells
  and about 20 scattered device loads a cell (about 15 000 loads on one GPU core). Splitting that loop 8 lanes a page
  inside the same workgroup made it slower (+20 µs profiled): the loads, not the lanes, were the limit. The gather has
  to leave the single workgroup.
- Fold: pmgDown3, a workgroup per page, now also writes each page's parts of its L4 residual into L4's spare channel
  (CH4R = channel 7; L4 has no KU). Part 0 is the page's own sum: its rows' residuals with the couplings inside the
  page, in cell order. Parts 1 + 2a + s are, for each face, the couplings across it times the page's own E there. The
  top's L5 RHS is then its page's part 0 plus the face neighbours' facing parts: 13 words a page, from 94 workgroups'
  work.
  - The terms are the same; only the f32 summation order changes, and that order is independent of slot.
  - A neighbour with no rows (IDX5 NONE) adds nothing, since its parts are not written that step.
- Cached L4 post (8 C <= 1024 lanes and words; the dam's C 125): each lane holds its cell's couplings and frozen halo
  (EPRE + the L5 value of the page across) in registers, with the page's E in wv's L6 words (dead by then). The four
  passes then read no device memory. Same terms, same order (bitwise by construction); the device loop stays for
  larger pools.
- What is left in the top's 43-53 µs is about 15-20 for the launch and the 1024-lane start, about 18 for the 16 L5/L6
  barriered passes, and about 10-15 for the links, L7 and the transfers.
  - It is a chain of about 30 dependent stages over at most 125 rows.
  - Spreading it across workgroups needs a launch at each stage (about 5.5 µs each), more than a stage costs inside one
    workgroup. So the rest of the top stays one workgroup.
  - The remaining lever is fewer stages (structure: coarse sweeps or levels), not more workgroups.

**Gates.**
- The V-cycle against the Float64 model is unchanged to the printed digits:
  - composite BS / BSB: |Z - Z64|/|Z| 4.41e-7 / 5.08e-7;
  - pool corner block / wide sheet: 9.32e-7 / 7.61e-7.
- Converged X:
  - composite: h 7.69e-6, native 8.26e-6;
  - pool corner: 2.13e-6.
- GPU pressure suite: 21 pass, 0 fail.
- brick-wide-dam "residency and lists follow the fluid" is bimodal on HEAD itself: step-7 liveF is 1575 or 1604 per
  run, and it failed 3 of 4 runs of HEAD's tree m2 (`runs5/T6`). It is flaky on the tolerance, not caused by this change.
- Whole step, unprofiled (the same hold, alternating, n 18 each):

  | | runs (ms) | mean (ms) |
  |---|---|---|
  | HEAD | 75.19, 75.97, 76.65 | 75.94 |
  | this commit | 74.85, 75.94, 76.15 | 75.65 |

  That is -0.3 ms, inside the noise. The rig's -20 to -30 µs a call at about 7 calls is -0.15 to -0.2 ms a step.

**pmgUpdate, pmgDirection and the L0 passes, in bytes.**
- Bytes are the unique words a call touches:
  - F channels × 563 row bricks × 520 words × 4 B (1.17 MB a channel);
  - C channels × 39 pages with native rows (0.08 MB a channel).
- Times are the profiled F-guess10 table's µs a call (dam steps 13-24).
- The F channel lists:

  | kernel | reads | writes |
  |---|---|---|
  | Update | RHS, X, D, DIAG, K0-2, RKU0-2 | R, W0 |
  | Direction | DIAG, D, X, Z, Q, K0-2, RKU0-2 | X, D, Q |
  | an L0 pass | its source plane, R, DIAG, K0-2, RKU0-2 | its target |
  | Down0 | as an L0 pass, plus a little L1 | L1 B, EPRE |

| kernel | F + C channels | MB a call | µs a call | GB/s | of 400 | workgroups | ns a workgroup |
|---|---|---|---|---|---|---|---|
| pmgUpdate | 12 + 9 | 14.8 | 133.3 | 111 | 28% | 657 | 203 |
| pmgDirection | 14 + 10 | 17.2 | 108.0 | 159 | 40% | 657 | 164 |
| pmgPostS | 10 | 11.7 | 79.7 | 147 | 37% | 869 | 92 |
| pmgPreS | 10 | 11.7 | 78.2 | 150 | 37% | 869 | 90 |
| pmgDown0 | 9.75 | 11.4 | 74.5 | 153 | 38% | 563 | 132 |
| pmgPostZ | 10 | 11.7 | 67.0 | 175 | 44% | 563 | 119 |
| pmgPostU | 10.1 | 11.9 | 55.9 | 212 | 53% | 563 | 99 |
| pmgPreU | 10 | 11.7 | 50.6 | 231 | 58% | 563 | 90 |

- None of these is bandwidth-bound:
  - The solver's working set (about 15 F channels, pmgF, the C pages: about 20 MB) fits in the M1 Max's 48 MB
    system-level cache, so a call does not even go to DRAM.
  - PreS and PreU move the same unique bytes but take 78 against 51 µs, in proportion to their workgroup counts (869
    against 563). Both cost 90 ns a workgroup. The cost follows workgroups, not bytes: these passes are latency- and
    occupancy-bound, each workgroup a serial chain.
- Measured pieces of that chain (profiled, a section duplicated):
  - In Update, the 6 barriered pSmooth passes are about +16 µs and pRedSums (each workgroup's 128 serial adds) about
    +12 µs.
  - In Direction, pRedSums is about +6 µs.
- So the lever is staging (shorter per-workgroup chains: a two-level pRedSums, fewer barriered passes, or more resident
  workgroups), not fewer words. Interleaving x, r, d and q, or merging vector reads, would save bytes that are not the
  limit.

## 2026-11-16 — occupancy of the per-row kernels; the CG dots summed in a fixed order

**The capture.**
- xctrace was attached to the tiered dam (HEAD 4fdb5642, 4913-brick pool) using Metal Application, GPU, and Metal GPU
  Counters.
- Steps 16-24 were captured, 64 calls of each slot kernel, 64 of 64 intervals clean of other processes' GPU work.
- The scratch tool's `--split` puts each dispatch in its own compute pass, labelled with its kernel and step and fenced
  by a 4-byte blit (Dawn otherwise coalesces consecutive passes into one Metal encoder).
- The split does not distort the step. GPU busy per step, steps 17-23, unsplit capture: 57.5, 57.6, 60.8, 64.1, 65.4,
  67.8, 70.3 ms. The same steps split: 55.9, 57.6, 62.2, 64.5, 67.9, 68.5, 67.7 ms.
- Occupancy is the Compute Occupancy counter, the mean over the four partitions.
- "Plateau" is the p90 over a kernel's 10 µs samples: its steady state, not counting ramp and tail.
- Resident workgroups a core = occupancy × 3072 threads / 512.
- Thread caps and threadgroup memory come from the Dawn-generated MSL:
  - `dump_shaders`, compiled with MTLDevice.makeComputePipelineState: maxTotalThreadsPerThreadgroup.
  - The dynamic threadgroup struct's size: threadgroup memory.

| kernel | µs a call | occupancy mean / plateau | resident WG a core (plateau) | ALU util p50 | ALU limiter p50 / p90 | TG memory | thread cap |
|---|---|---|---|---|---|---|---|
| pmgUpdate | 138.0 | 29.5% / 35.5% | 2.1 | 51% | 58 / 62% | 7.1 KB | 1024 |
| pmgDirection | 106.6 | 34.2% / 41.6% | 2.5 | 37% | 41 / 46% | 5.5 KB | 1024 |
| pmgPostS | 78.0 | 45.5% / 60.3% | 3.6 | 65% | 80 / 86% | 4.3 KB | 1024 |
| pmgDown0 | 77.0 | 43.7% / 60.3% | 3.6 | 38% | 43 / 51% | 2.6 KB | 1024 |
| pmgPreS | 76.0 | 45.1% / 60.3% | 3.6 | 62% | 77 / 87% | 4.3 KB | 1024 |
| pmgPostZ | 63.7 | 43.3% / 60.2% | 3.6 | 51% | 63 / 80% | 6.7 KB | 1024 |
| pmgPostU | 55.8 | 40.6% / 60.2% | 3.6 | 48% | 60 / 84% | 4.3 KB | 1024 |
| pmgPreU | 49.6 | 43.9% / 68.9% | 4.1 | 41% | 53 / 91% | 4.3 KB | 1024 |

- Every other limiter is low over these kernels:
  - buffer read 4-13%, buffer write ≤ 1.6%;
  - threadgroup load 2-14%, threadgroup store ≤ 1.6%;
  - last-level cache 13-18%, MMU 7-16%.
- The kernels are not idle silicon waiting on one chain. At their plateau the L0 passes issue ALU work 77-87% of the
  time, Update 58-62%.
- Residency is not threadgroup memory: 2.6-7.1 KB a workgroup would let 4-12 fit a core. Nor is it the thread cap, which
  is 1024 everywhere, so registers allow a full-size threadgroup.
- The plateaus (2.1, 2.5 and 3.6 workgroups a core) are the per-core register file's limit. Neither Metal nor WebGPU
  reports the register count, and no lever in WGSL moves it directly.

**What that leaves of the levers.**
- *Fewer barriered smoothing passes.* The 6 red-black passes are the smoother (P_FS2, 3 sweeps), not the structure.
  - Inside a pass, half of each SIMD group's lanes sat masked off (the other colour).
  - Experiment "comp": lanes 0..255 take the colour-0 cells and 256..511 the colour-1 cells (pLaneCell). Each pass's
    colour is then uniform over a SIMD group, and the other colour's groups skip it instead of issuing it masked. This
    covers the L0 brick passes and Update.
  - Hold (below): fx → comp +2.4 and +0.8 ms; fx + PreS 21× → comp + PreS 21× −1.7 and −1.1 ms. No measurable
    gain: **no-go**, not kept.
- *Less threadgroup memory or fewer registers.* Threadgroup memory is not the limit. Registers are not reachable from
  here without the compiler's numbers.
- *A two-level, fixed-order reduction* (lane E's run-to-run spread, the coordinator's ruling): landed. It is the
  measured win.
- An anomaly, recorded for whoever prices these next:
  - PreS dispatched 21× a slot (it is idempotent) costs +1.6 ms a step over 20 × about 6.9 open slots: about 11 µs a
    repeat, against 76 µs for the first call in its slot.
  - The repeats read the same 11.7 MB the first call does. The in-situ cost is therefore the first touch of the data
    after the slot's other kernels, not the kernel's own work.
  - This is consistent with the latency-bound picture above, but a byte count of a single kernel will not show it.

**The fixed-order dots.**
- Before: a workgroup's pSum512 partial was CAS-added into one of 64 float bins (g & 63). Arrival order changed the last
  bits of alpha and beta from step 0. Every consumer workgroup's head then folded the 128 bins serially (pRedSums), and
  the check cleared them.
- Now each producer workgroup writes its partial to its own word of pmgRed (8 + 2(F + C) words):
  - R_ZF + g for pmgPostZ's row brick g;
  - R_ZC + g for pmgNativePost0's page g (0 for a page without native rows);
  - R_Q + g for pmgDirection's workgroup g (0 for a page that returns).
- pFixed folds the words in index order:
  - lane j takes j, j + 512, …;
  - then pSum512 folds the lanes;
  - it runs once in the C | F entry, in uniform control flow.
- Every consumer workgroup runs the same fold over the same words, so every reader sees one value. The first row
  brick's workgroup stores it: R_RZ in Direction, R_PQ in Update.
- The walls kernels and the check read those two words. Nothing is cleared, since every slot rewrites all of [0, n).
- No float atomic touches a CG scalar any more. Slot claims and the max-norm stay atomic: the max is exact in any order.
- A bug the suite caught on the way. The first version stored the fold from workgroup 0, which is a page; a page without
  native rows returns before the fold, so R_RZ and R_PQ were never written. The pool MGPCG tests (all h) then ran with
  alpha 0. Fix: pFixedWriter, the first row brick's workgroup.

**Gates (fx = this commit's pressure-mgpcg.ts on 4fdb5642).**
- GPU pressure suite 21/0.
- Float64 model V-cycle unchanged: |Z − Z64|/|Z| h 4.41e-7 native 1.54e-7 (L0 BS), h 5.08e-7 native 1.43e-7 (BSB).
- Converged X: h 7.75e-6 native 8.30e-6, 8 its. HEAD was 7.69e-6 / 8.26e-6, 8 its; it varies run to run.
- Pool MGPCG corner: 5 its, 2.14e-6. Static box: 4 its.
- Aborts:
  - Two earlier suite runs logged Metal command-buffer aborts while other lanes' GPU jobs were resident: invalid, rerun.
  - The final run had none, and there were no 00000206 entries.
- Unprofiled hold: tiered dam, 36 steps, fenced median of steps 18-35 (n 18), alternating.

  | run | tm (HEAD) | fx |
  |---|---|---|
  | medians, ms | 75.93, 76.01, 78.60, 76.86, 76.72 | 74.25, 74.98, 75.15, 76.67, 75.43 |
  | mean of medians, ms | 76.83 | 75.30 |
  | its, steps 18-35 | 123-125 | 119-127 |

  - Adjacent pairs: −1.68, −1.03, −3.45, −0.19, −1.29 ms, every one in fx's favour. Mean **−1.5 ms a step**.
  - Iterations are unchanged: the same counts within the runs' own spread.
- The fixed order saves time against the CAS bins rather than costing it. It replaces:
  - the CAS loops (about 14 workgroups a bin contending at each producer's tail);
  - every consumer workgroup's 128-load, 128-add serial head;
  - the check's clear.
- Run to run (the tiered dam twice on each tree, 10 steps, sampled every step, one pair each):
  - Neither tree is bitwise reproducible. The step-0 sample's kinetic energy (before the first step, after the boot)
    already differs, so other order-dependent adds remain upstream of the solve.
  - The spread is smaller with fx:

    | | tm | fx |
    |---|---|---|
    | Δke, step 0 | 8.1e-9 J (1.7e-8 relative) | 5.9e-10 J (1.3e-9 relative) |
    | Δ total litres, step 1 | 1.1e-6 | 2.1e-7 |
    | Δ total litres, step 2 | 5.6e-6 | 3.5e-7 |

  - Iterations are identical across the pairs (steps 1-9: 4 4 4 4 5 4 5 6 6).
  - The D gates' ±0.1-0.2 spread therefore narrows but does not vanish.

## 2026-11-16 — a block-local L0 smoother with Down0 fused: no-go in the Float64 model

**The premise.** The L0 smoother is already block-local in threadgroup memory. Each L0 pass (pmgPreS, pmgPreU,
pmgPostU, pmgPostS, pmgPostZ) is one launch, with one workgroup per brick or shifted block. A launch loads the 10³ tile
(the block plus a one-cell halo) once and runs the three red-black sweeps, six `workgroupBarrier` half-sweeps, in `tp`.
It then writes once. The six barriered passes are inside the workgroup, not global.
- The order is already the seep's S2m staggered Schwarz, as the palindrome BSB.
- The first brick pass is already fused into pmgUpdate (and into pmgWalls' restart), from zero.

**What is left of the item:**
- removing partitions: BS saves pmgPreU and pmgPostU; plain bricks saves all four S/U passes;
- fusing Down0 into the last pre pass.

The fusion needs the native L0 pass (nativePre0) out of the middle. It now sits between the last h pass and Down0.

**The model** (`cmodel.ts` with the L0 pass order overridable, the six dumped states, the Jacobi start, mean PCG
iterations to tolerance):

| L0 passes, pre (post is the reverse) | its per state | mean |
|---|---|---|
| BSB (the GPU's; baseline) | 7 6 6 6 6 6 | **6.17** |
| BS, staggered, one pair | 7 7 6 7 7 7 | 6.83 |
| SB | | 6.83 |
| BX (shift 4,4,4) | | 6.83 |
| BS, 4 sweeps | | 6.83 |
| BS, native additive at L0 (Down0 fusable) | | 6.83 |
| B, plain Schwarz (bricks only) | 8 8 8 8 8 8 | 8.00 |
| B, 4 or 6 sweeps | 8 8 8 8 8 8 | 8.00, 8.00 |
| B, native additive at L0 (Down0 fusable) | 8 9 8 8 8 8 | 8.17 |

- The gate was a mean of at most 6.17 + 0.5 = 6.67 iterations. Every variant fails it.
  - Staggered BS is +0.67.
  - Plain Schwarz is +1.83, and more sweeps do not move it. This is the seep's brick-face finding: block smoothers
    aligned with the bricks, with a frozen halo, leave the cross-face error.
- In time, at about 0.88 ms an iteration and about 6.9 open slots:
  - BS saves about 106 µs a slot (PreU 50, PostU 56 in situ), about 0.7 ms a step. It costs about 0.6 ms: net
    about 0.1 ms at best.
  - Plain B saves about 260 µs a slot (about 1.8 ms). It costs about 1.6 ms.
  - Both are below the 1 ms bar even before the gate.
- Not taken. Nothing changed on the GPU.

## 2026-11-16 — a matrix-free interior for the L0 rows: measured, no-go

**The idea.** On the dam, most L0 rows are interior liquid rows. Their DIAG is 6 and every coupling is 1. A flag word
plus full coefficients only for the other rows would shrink what each per-row kernel reloads.

**Interior share** (the six dumped states, steps 16–18, 20 and 22, not 18–23).
- "Interior" means DIAG 6, all six couplings 1, every upper neighbour an h row, and no native coupling.
  DIAG = 6 coincides exactly with this test on the dam.

  | | per dump |
  |---|---|
  | interior rows | 62.5–65.2% |
  | boundary rows | 35–37.5% (81–85 a brick) |
  | rows a brick | 222–241 of 512 lanes |
  | all-interior bricks | none (0 of 525–715) |
  | 128-byte lines (32 lanes) with no boundary row, skippable by flags | 37–41% |
  | coefficient words in a compact layout (packed flag bits + 7 a boundary row) | 16–17% of now |

**First-touch words per launch, per brick (512 lanes).** C is coefficients (DIAG, K×3, RKU×3, and Down0's L1
coefficients). V is vectors, reads plus writes.

| kernel | C | V | C share |
|---|---|---|---|
| PreS, PreU, PostU, PostS, PostZ (each) | 3584 | 1536 (R, src, dst) | 70% |
| Update | 3584 | 2560 (RHS, X, D; R, W0) | 58% |
| Down0 | 2496 | 1152 | 68% |
| Direction | 2048 | 3584 (D, X, Z, Q; X, D, Q) | 36% |
| a slot | 26048 | 14976 | 63% |

- Per row a slot (about 233 rows a brick): about 447 B of coefficients and about 257 B of vectors.

**Timing.** Profiled tiered dam, 24 steps, per-kernel means over steps 13–23 at about 7.33 slots.
- **The scratch hack mf.** DIAG is read as the flag. A row with DIAG 6 takes constant couplings, a boundary row loads
  them, and an air lane loads nothing. It is in pRow (the five fine kernels and Update's brick pass) and in both forms
  of fineApply (Update, Down0, Direction).
  - It is exact on the dam: the GPU pressure suite gives 21/0, with converged X h 7.67e-6 and native 8.30e-6.
  - Three alternating pairs, µs a call:

    | | Update | Direction | Down0 | PreS | PreU | PostU | PostS | PostZ | L0 kernels, ms a step |
    |---|---|---|---|---|---|---|---|---|---|
    | base | 141.0 | 117.2 | 73.2 | 72.7 | 51.4 | 51.9 | 77.0 | 62.1 | 4.74 |
    | mf | 144.5 | 117.2 | 73.5 | 71.0 | 50.4 | 56.6 | 74.0 | 63.1 | 4.77 |

  - No gain. The hack skips about a third of the coefficient lines, and the boundary lanes' coefficient loads now
    wait on DIAG.
- **The bound mfuS.** PreS only, every row's couplings constant, so only DIAG of 7 is loaded. This removes about 60% of
  PreS's unique words. The numerics are wrong, but iterations were the same within 1 at two steps.
  - Two pairs: PreS 73.7, 72.2 → 60.3, 61.8 µs a call, about −12 µs (−16%).
  - Removing 60% of the bytes buys 16% of the time. The first-touch cost (about 73 µs against about 11 repeated) is
    mostly not coefficient bytes.
- **Projection.**
  - If every L0 kernel's coefficients were free (the bound, about −12 µs for each fine kernel and similar for
    Update, Down0 and Direction): at most about 80 µs a slot, about 0.6 ms a step.
  - A compact layout keeps 17% of the coefficient words and adds the flag/index work, so at most about 0.5 ms.
  - Per row: at most about 0.55 ns a row a slot, about 4 ns a row a step, at 145k h rows.
- Below the 1 ms bar: not taken. Nothing changed in the tree. The hack is in the scratch file mf/mf-hack.patch.

**Open, parked (coordinator, 2026-11-16).** The pressure solve is parked. Three structural levers were each priced under
1 ms a step:
- the block-local L0 smoother;
- the matrix-free interior;
- the dot reductions, already landed.

The iteration count is set by the dam's start distance. One question is left open for later: what holds the
remaining about 61 µs of PreS's first-touch cost, with no coefficient loads? It is not coefficient bytes (mfuS above).
The candidates are the 1000-entry tile load with its slot and native lookups, the staged neighbour table, R, the
source plane and the write. Each would be removed one at a time in the profiled rig.

## 2026-11-16 — "transfer and seeding" and "move" attributed per kernel (measurement only)

**Reconciliation.** The two areas are counted over different kernel sets in E's two profiles.
- E's earlier rule (areas.py) is `{pDeposit, pSeed}`, 3.4–3.6 ms. Its move is `{uCache, pMove, pScan, pScatter,
  pTiles, pDelta, pJoin, pArgs}`, and uTransfer sat in velocity.
- E's r1/areas.mts (the 9.15 / 9.32 ms tables) moved uTransfer (from velocity) and pDelta, pJoin and pTiles (from
  move) into transfer and seeding. uCache went to velocity, and pLiquid joined move.
- On E's r12-head profile, the 9.32 ms is uTransfer 4.77 + pSeed 1.92 + pDeposit 1.58 + pDelta 0.59 + pJoin 0.33 +
  pTiles 0.12. The old pair is 3.51 ms, so nothing grew; the kernels were relabelled.
- On HEAD 2348147a, uTransfer is 3.23 ms (F 2.77 + C 0.46, split below), against 4.77 in E's r12-head run.

**Method.**
- Profiled tiered dam on HEAD 2348147a (scratch tree `trees/tr`), 24 steps, `--every=100 --dense=16 --profile=13`.
- The early window is steps 18–23: 4 runs, mean (min–max). The late window is a 106-step run, `--every=2`, profiled
  steps 94–105.
- Scratch flag `splitT`: uTransfer's F bricks and C pages as two dispatches. They are the two independent halves of
  one dispatch, so the split is the same computation.
- Records are pMove's lanes (dispatch × 128): about 419k early and 450k late. F_ALL bricks are pSeed's groups: about
  3514 early and 4066 late.
- Every kernel here is 100 µs or more a call, except pArgs, whose 11–44 µs is the timestamp tick (about 1 µs of work).

| kernel | what | µs/call, 18–23 | ns/record | µs/brick | µs/call, 94–105 | ns/record |
|---|---|---|---|---|---|---|
| pMove | G2P, Heun trace, pair separation, surface shift, merge, split, rank | 4345 (4238–4511) | 10.4 | 1.24 | 4779 | 10.6 |
| uTransfer, F half | P2G resolve, FLIP keep, gravity, per F brick (staged window, 125 tiles, 9³ depth) | 2772 (2720–2829) | 6.6 | 0.79 | 2922 | 6.5 |
| pSeed | reseed: a count pass and a write pass over 512 cells a brick | 1925 (1901–1988) | 4.6 | 0.55 | 1557 | 3.5 |
| pDeposit | P2G scatter into 3300×2 workgroup atomics, per brick | 1273 (1201–1322) | 3.0 | 0.36 | 1180 | 2.6 |
| pDelta | FLIP delta per record | 584 | 1.4 | 0.17 | 546 | 1.2 |
| uTransfer, C half | 4h faces: gathers the 4³ cells' records a face | 456 | 1.1 | 0.13 | 344 | 0.8 |
| pJoin | census join and support bits per record | 320 | 0.8 | 0.09 | 262 | 0.6 |
| pLiquid | liquid bits per brick | 281 | 0.7 | 0.08 | 431 | 1.0 |
| pScan | the rebin's per-brick scan | 229 | 0.6 | 0.07 | 246 | 0.6 |
| pScatter | the rebin's filing | 172 | 0.4 | 0.05 | 164 | 0.4 |
| pTiles | tile heats | 112 | 0.3 | 0.03 | 104 | 0.2 |
| sum | | 12.5 ms | | | 12.5 ms | |

E's transfer set is 7.44 ms here and E's move set 5.04 ms.

**Removal pricing.** Same rig, one run each against the four-run baseline. These are timing probes; the numerics
change where noted.

| probe | kernel | µs/call | Δ |
|---|---|---|---|
| pMove without the pair separation (`noSep`) | pMove | 2316 | about −2.07 ms (−47%, about 4.9 ns/record) |
| pMove without the surface shift (`noShift`) | pMove | 3878 | about −0.47 ms |
| pSeed's count pass run twice (`seedDup`, exact) | pSeed | 3058 | +1.13 ms: the count pass is about 1.1 of 1.93 ms |
| uTransfer F without its near-surface theta block (`noNear`) | uTransfer F | 2796 | about 0 |
| uTransfer F with empty-bin bricks on a copy-only fast path (`uEmpty`, approximate: it skips neighbours' deposits and gravity), 2 pairs | uTransfer F | 2032, 2097 against 2753, 2785 | about −0.70 ms |

**Levers (proposals for F; I made no edits to their files).**
1. **The pair separation in pMove: up to about 2.1 ms a step (about 4.9 ns a record).**
   - Each coupled record reads every record of the 2^3 cells about floor(q − 0.5), 48 B a record, for
     position, liveness and mass.
   - F has shown that the separation plus surface shift is bitwise the same computed in a pre-move pass (`pBand`,
     progress-lane-f.md M0(c)). That allows a per-brick pass that stages the brick-plus-halo records' 16 B
     (xyz plus mass) in threadgroup memory, or at least a 16 B side array written at filing.
   - Expected about 1 ms. This needs a go/no-go probe: the pressure work showed fewer bytes buys less than
     proportional time, and E's whole-move staging lost.
2. **uTransfer's F half on bricks with nothing to transfer: about 0.7 ms a step by the probe.**
   - Each of about 3500 F_ALL bricks stages a window, a vertex ring, 125 tiles and 729 depth vertices, even
     where no record deposits.
   - An exact fast path needs a per-brick flag meaning no deposit weight on any face and no liquid on either
     side of the brick's faces. pDeposit and FC_LIQ could write it. The path then copies US = U = UP and
     zeroes fdep.
3. **Runner-up: pSeed.** Keep the count pass's accepted sites as a per-lane mask, so the write pass only samples
   velocity. That is at most the write pass's re-tests, about 0.5 ms.
- With E's R1+R2(b) at −30 to −40% records, the per-record lever scales with records: about 1.3–1.4 ms of
  separation bound left. The per-brick lever scales with F_ALL bricks.
