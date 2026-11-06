# Lane P progress: pressure work follows the residual

Tank: `--pure --px=16 --py=8 --pz=8 --cell=0.01 --steps=60 --every=60 --pipelined`.
Per-frame record: `tools/probe-tank-pressure-trace.ts` (unpipelined; `--diagnostics`
splits each measure by row class and prints the f32 round-off scale of the start).

## Baseline (main cd6b085f, before coldSurface "still")

- Tank 16x8x8: median 18.4 ms; 1 cycle a step at rest; accepted 0.040; drift -4.3e-7.
- With the kernel timer on (steps 40-60) the step falls to 8.6 ms: it is
  then the CPU's encode alone, so the pipelined 18.4 ms is GPU-bound.
- Pressure kernels (means, ms a step): pTail 1.37, project 1.05, pUp4 0.46,
  pDown2 0.42, bulkSelect 0.40, buildRhs 0.35, pBulkRows 0.33, pDownB 0.30,
  pUp3 0.26, pResidual2 0.23, pDown3 0.22, pNativeRelax 0.22, pUpB 0.20,
  pNative 0.18, pRelax 0.13, pDown4 0.13, kStart 0.13, warm copy 0.12.

## Step 1: why rest runs a cycle

- Natives start at round-off (initial 3e-7 against an f32 scale of 2.9e-7). BULK16
  interior pages hold 1.5-2.9e-6, the sign alternating frame to frame. It is not
  a discretisation imbalance.
- The frame map: with the pressure warm-started and the unsolved divergence left
  in the velocity, init(n+1) = 2 r(n) - r(n-1). A cycle with residual ratio rho
  gives frame roots of l^2 = 2 rho l - rho, unstable for rho < -1/3.
- The first V-cycle overshoots the BULK16 rows: after it, about 5000 of 7168 page
  rows flip sign at 0.47-0.67 of their start (trace rho1). The stop at nFloor
  (1e-6, 4-5x the f32 scale) holds the mode there: 1 cycle every frame.
- The cause is the page tail's block-Jacobi iteration: rows flip at 1.5x with
  tailCycles 1, 0.6x with 3 (the default) and 0.4x with 5. tailScale 1 (from 0.75)
  still flips at 0.6-0.7. A frame's second cycle (a GCR step) cuts 20-50x.

## Re-baseline (HEAD 90a14ef1 code, clean `git archive` export)

The shared checkout's other WIP broke pipelines mid-run (a mixture binding), so
every arm below ran from an export of HEAD plus only this lane's patch, one
run each (`--steps=60` tank/drop, seep 300, letters 120).

| arm | tank 16x8x8 ms (late half) | drop 16x8x4 ms / drift | seep ms / drift | letters ms / drift |
| --- | --- | --- | --- | --- |
| HEAD (ctrl) | 7.52 (7.41), drift -3.7e-7, 1 cycle/frame | 22.58 / -6.2e-5 | 32.45 / +2.96e-4 | 55.02 / -9.05e-3 |
| page tail rows held in registers | 7.57 (7.50) | | | |
| + pGate (converged solve launches nothing) | 7.21 (7.24) | 23.53 / -6.9e-5 | 32.46 / +2.61e-4 | 55.65 / -9.15e-3 |
| + Krylov step from cycle 0 | 6.89 (4.69), drift -1.2e-7 | 23.64 / -5.2e-5 | 33.46 / +3.23e-4 | |
| Krylov step from cycle 0 alone | 7.58 (5.25) | | | |
| tailScale 0.5 | 8.02 (7.92), drift 0 | 22.96 / -7.1e-5 | 32.56 / +3.16e-4 | 54.16 / -8.91e-3 |

- A resting frame's cost is its cycle: 0-cycle frames run 4.0-4.4 ms, 1-cycle
  frames 7.2-8.0 ms (per-step times). One V-cycle is ~3.5 ms of the 7.5 ms idle.
- Held tail rows: no step-time change (the tank's pool lists 1152 rows over
  1024 lanes, so the second row a lane is reloaded 64 times; not the cost).
  Dropped.
- pGate (each cycle's launches through an indirect buffer that a one-thread
  kernel zeroes once a measure meets the target): -0.31 ms on the tank, +0.95
  ms on the drop, likely Dawn's validation dispatch ahead of each indirect
  dispatch (no toggle skips it), over the drop's 2-4 cycles of them. Dropped; the kernels' own
  bConverged exits stay the gate.
- Krylov step from the first cycle (GCR(1) started on the warm start, so the
  first V-cycle's correction is a minimal-residual step): the rest tank skips
  the cycle in ~60% of frames (trace: initial 4.5-9.9e-7 under nFloor 1e-6)
  but every cycled frame pays the extra step (+1 ms), and seep runs one more
  cycle (33.46 ms). Dropped: it helps one scene and hurts two.
- tailScale 0.5: the BULK16 flips fall to 0.35-0.45 of their start and the
  unpipelined trace skips 8 of 12 rest frames, but the pipelined tank still
  runs 1 cycle in most frames (8.02 ms). Dropped.

## Open

- The rest mode is the first V-cycle's BULK16 overshoot (rho < -1/3 under the
  frame map). Its fix belongs in the page tail's block-Jacobi pass (X/Y
  channels between pDown3 and pUp3), so the cycle contracts the 16h page
  rows without a sign flip. Reaching 0 cycles at rest is worth ~3.5 ms on the
  tank. Not built.
- Any further launch gating must avoid indirect dispatch (Dawn's validation
  dispatch per call) or show a net gain on the drop.

## Round 2: the 16h up leg red-black over pages

- The frame map is not an extrapolation: the warm start is the presented
  pressure, the projection leaves r(n) in the velocity and gravity adds the
  same G each frame, so init(n+1) = r(n) + G - A p(n) and G - A p(n) =
  r(n) - r(n-1). Any warm-started inexact projection does this; the fix is
  a cycle that does not flip sign.
- The flip was pUp4 (the 16h post-smoothing): each page swept its rows against
  its neighbours' rows from before their own sweep (block Jacobi), with half
  of a 16h row's couplings leaving its page. Landed: the last of the three
  tail cycles runs pUp4 as two colour passes by page parity (colour 1 reads
  its face and corner neighbours' result: block Gauss-Seidel). The first two
  stay one Jacobi pass. Two Jacobi passes square the flip, so they keep a
  nonnegative spectrum, and the last pass's Gauss-Seidel damps them. One
  extra dependent pass per V-cycle.
- Trace (16x8x8 rest, unpipelined, last-only): the largest first-cycle sign
  flip falls from 0.47-0.67 (about 5000 rows) to at most 0.16 (50 rows, one
  frame); 5 of 12 frames run 0 cycles; drift 0. A damped block Jacobi (weight
  0.5, one pass) left flips at 0.95 and the pipelined tank at 7.69 ms:
  dropped.
- tests/pressure-tail-red-black.test.ts (8x4x4 rest tank, diagnostics): the
  largest flip stays under 1/3. With the one-pass Jacobi it fails at 0.54.

Arms on ca39ca77 (one run each, paired, same export):

| arm | tank 16x8x8 | drop=2 | drop=8 / drift | seep / drift | letters / drift |
| --- | --- | --- | --- | --- | --- |
| Jacobi | 8.07 (late 8.03; 3 of 50 frames at 0 cycles) | 17.61 | 24.44 / -6.9e-5 | 32.45 / +2.96e-4 | 54.45 (late 77.84) / -8.86e-3 |
| red-black, every tail cycle | 5.80 (late 5.04; 30 of 50) | 18.42 | 23.62 / -6.1e-5 | 32.07 / +2.34e-4 | 58.00 (late 83.43) / -9.32e-3 |
| red-black, every cycle, half launches | 6.44 (late 5.61) | 18.41 | 23.36 / -5.5e-5 | 31.48 / +4.22e-4 | 55.36 (late 79.17) / -9.04e-3 |
| **red-black, last tail cycle (landed)** | **5.67** (late 5.67) | **17.27** | **21.54** / -5.1e-5 | **31.57** / +4.1e-4 | 57.25 (late 76.47) / -9.01e-3 |

On HEAD a44ed3aa (clean export, paired): tank 9.57 -> 7.10, drop=2 20.11 ->
19.91. HEAD's own tank is slower than ca39ca77's 8.07 (the cloth commit's
arena hunks or the shared GPU); the landed arm still removes about 2.5 ms.

- A 0-cycle tank frame runs 4.9-5.3 ms, a cycled one 8.0-8.9.
- Every-cycle red-black costs letters 12 dependent passes a frame: under the
  kernel timer pUp4 rose 0.29 -> 0.65 ms. Half launches (two pages a
  workgroup) were no cheaper (0.74). Last-only adds 4 passes. Letters' wall
  median is noisy (control and Jacobi, same code, 55.02 / 54.45), and its late
  half improves.
- Drift is within 3% everywhere: seep +4.1e-4, letters -0.90% (Jacobi -0.89%).
- Open: pDown3's per-page 16h pre-step is still block Jacobi; the remaining
  cycled rest frames come from native interior rows near the 1e-6 floor;
  the solve is not bit-reproducible run to run, so only paired runs compare.

## Round 3

### Letters after 1058cc0b: noise, plus pUp4's 0.13 ms

One 120-step letters run each under the kernel timer (from step 40),
a44ed3aa (Jacobi) against a44ed3aa + 1058cc0b (clean exports):
- pUp4 0.277 -> 0.408 ms (+0.13: the four extra colour passes a frame).
  Every other kernel moves by 0.06 or less; total kernel time +0.65.
- Cycles: frames 30-120 have the same pattern, and the last 60 frames run
  233 cycles in both arms. The red-black arm has 12 more, all in frames
  1-15 (falling letters, 20 ms frames), as alternating 1/0 frames. The
  unpipelined trace (frames 1-4, diagnostics) runs 0 cycles in both, with
  the same residuals. So the alternation is the pipelined run's, not
  reproducible frame by frame.
- So the +2.8 ms wall median was mostly noise. The median sits between the
  cheap early frames and the 76 ms late ones, and moves with any early
  change. The late halves agree (76.47 against 77.84 untimed).

### Plan step 3 (pressure work over the residual's support): knockout, not built

At drop=2, the larger tank runs more cycles, but not much more per cycle.
Unpipelined trace, frames 40-60: 27 cycles on 8x8x4, 34 on 16x8x4. Both sit
near the 0.05 target after cycle 1. The native rows' initial residual is
4e-4..2e-3 everywhere (25-100x under the target); only the band near the
drop is above it.

Knockout (scratch, env-gated, a44ed3aa + 1058cc0b, 16x8x4 drop=2):
- Every per-cycle kernel walks compacted lists, built by one
  single-workgroup pass a frame: level 2 pages and BULK16 slots with page
  x < 8 (the 8x8x4 tank's half). Covered: pDownB, pDown2..4, pTail, pUp4,
  pUp3, pBulkLift, pUpB, pNativeRelax, pResidual2 per cycle, and the Krylov
  apply/bound/step.
- Fixed grids, no indirect launch.
- The kick, kStart, pDownB@0's bake and the final measure keep the full
  lists. Inactive pages hold their pressure: their level 2 correction and
  tail X/Y are zeroed, so they are Dirichlet data for the active half.
- The first try kept stale state on the frozen half (the kick's
  corrections) and diverged; zeroing fixed that.

| 16x8x4 drop=2, one run each | median ms | under the kernel timer: median / arena solve kernels | cycles 40-60 (trace) | drift at 60 |
| --- | --- | --- | --- | --- |
| all pages (KO 999) | 20.62 | 20.36 / 6.00 | 34 | 1.9e-7 |
| far half frozen (KO 8) | 21.14 | 20.87 / 5.04 | 24 | 9.0e-5 |
| 8x8x4 (reference) | | 15.64 / 3.80 | 27 | |

- The knockout removed 0.96 ms of arena kernel time (pUp4 0.69 -> 0.44,
  pDown2 0.44 -> 0.27, pUp3 0.35 -> 0.23, pTail 0.78 -> 0.69: most at the
  8x8x4 values) and 10 cycles of 34. The step did not get faster. So at
  this size the arena's per-cycle kernels are not what sets the
  pipelined frame time; the kernel timer's sum over-attributes the tank
  gap to them.
- It is also not correct as a rule. The drop raises the level across the
  tank, so a frozen region's residual grows each frame (2.9e-6 -> 3e-3 in
  60), and drift rose 90x. A per-row max under a fraction of the target is
  the wrong certificate for a large still region. It would have to bound
  the region's summed divergence and wake the region, so it would wake
  often.
- Not built: below the 0.3 ms bar on wall time, and it loses volume.
- Open: the 8x8x4 -> 16x8x4 wall gap needs a wall-time attribution
  (xctrace/gpu-frame-profile: CPU encode, the critical path), not
  per-dispatch timer sums. The root projection (one 1.28 ms dispatch at
  16x8x4, +0.55 over 8x8x4) is the one pressure kernel big enough to
  matter. Its COLD faces compute the full row only for the freeze's
  correction accumulate (Lane S's certificate), so trimming it is a change
  to that protocol.

## Round 4: host side (late-submit stall, CPU encode)

8x8x4 drop=2, pipelined, HEAD 0401fd1d exports. Two GPU runs: one System
Trace capture (timing) and one call-count run (check). The rest is
re-reduction of existing traces and runs, plus one `--cpu-prof` run
(check).

### 1. Late-submit stall: none outside the x6/d2 recording

- **System Trace capture.**
  - Setup: System Trace template plus Metal Application and GPU. The
    recording attaches at step 4 and lasts 6 s, which covers steps 4-60.
    Pass labels were on, as in x6.
  - The GPU was busy for 748.6 of the 813 ms span (92%).
  - Main thread over that span: Running 783.5 ms, Blocked 14.4,
    Interrupted 6.1, Preempted 4.9, Runnable 3.7.
  - GPU idle gaps over 1 ms: 3 in 56 advances.
    - 36 ms at the first advance after the attach pause, with the main
      thread Running (JIT, then the first submit).
    - 3.5 ms just after it.
    - 2.5 ms at the final drain.
  - None is a blocked main thread. The longest Blocked interval inside any
    gap was 0.67 ms.
  - The probe's own steps in that run: median 13.6 ms, late-half p95
    14.6.
- **Untraced runs of the same command do not stall.** The step-time
  quantiles exclude steps 1-2 and the report step:

  | run | probe p95_ms | steady median | steady p95 | steady max |
  | --- | --- | --- | --- | --- |
  | f1-d2 (untraced) | 31.8 | 16.0 | 17.3 | 17.5 |
  | fk-d2 (untraced) | 23.0 | 14.6 | 17.7 | 20.7 |
  | k5-base (untraced) | 30.2 | 15.7 | 18.5 | 21.7 |
  | t8-d2 (untraced) | 24.4 | 13.6 | 14.7 | 17.3 |
  | x6-d2g (traced, counters) | 24.7 | 16.4 | 17.5 | 18.1 |
  | x6-d2 (traced, Metal System Trace) | 47.8 | 15.8 | 39.6 | 47.8 |

  - Only x6-d2's recording has the 2-33 ms gaps.
  - In that trace, the main thread drops out of the samples in the middle
    of Metal encoding. Stacks either side of the holes show
    `__kdebug_trace64 < IOGPUDeviceTraceObjectLabel < -[IOGPUMetalCommandEncoder setLabel:]`,
    `IOGPUMetalCommandBufferStorageAllocResourceAtIndex` and
    `MakeDebugName`. That is the recording's own overhead, not a wait
    in the solver.
- **The "untraced p95 of about 30 ms" is the probe's quantile, not a stall.**
  - p95 of 60 steps is the third-largest step.
  - The three largest steps are step 1 (compile, 0.2-0.6 s), step 2
    (25-170 ms), and step 60. Step 60 is the report step, which drains
    the receipt ring (about 30 ms).
  - The probe now also reports `steadyMedian_ms`, `steadyP95_ms` and
    `steadyMax_ms`. These exclude steps 1-2 and, when pipelined, the
    report steps.
- **Same check elsewhere.**
  - Seep 300: steady p95 41.0 against a median of 38.0.
  - Rest: p95 8.5-10 against a median of 3.4-4.0. That is warm-up in
    steps 3-7 (6-13 ms), not gaps.
  - Letters 120: p95 104.6 against a median of 63.4. Its frames grow
    steadily as the letters spread (16 -> 109 ms), with no spikes.
- **Nothing to fix.**
  - A blocked mapAsync or onSubmittedWorkDone would show as Blocked.
    Instead the main thread waits on the receipt ring by spinning in the
    paced Dawn poll (Atomics.load, Running).
  - Untraced, steady p95 is already within 1.5-3 ms of the median.

### 2. CPU encode: no per-frame creation; the remainder is per dispatch

- **Per-advance WebGPU calls** were counted by a device proxy in steps
  30-39 (scratch only).
  - Creation calls: none. That covers createBindGroup, createBuffer,
    createTexture, createComputePipeline, createShaderModule and
    createBindGroupLayout.
  - 4 createCommandEncoder, 4 submit.
  - 443 setPipeline, 386 dispatchWorkgroups + 63 indirect, 304
    setBindGroup (including the shared pass's slot resets).
  - 97 logical passes, which run inside a few native shared passes.
  - 53 clearBuffer + 52 copyBufferToBuffer, mostly turned into compute
    launches. 33 writeBuffer.
- **Tint is not on the per-frame path.**
  - No main-thread sample in x6/d2's steady window (230.35-231.38 s)
    has a tint frame.
  - The "dawn/tint" bucket in profile-probe-xctrace is every stack inside
    dawn.node.
- **Main-thread time per advance** (x6/d2 samples, steady window):

  | family | ms/advance |
  | --- | --- |
  | Dawn poll spin (waiting) | 2.30 |
  | Dawn -> Metal encode at submit (CommandBuffer::Fill/Encode, AGX) | 1.41 |
  | JS encode logic | 1.20 |
  | NAPI/interop wrappers (Unwrap, InstanceOf, Wrappers::For) | 1.06 |
  | Dawn sync-scope usage tracking (AddDispatchSyncScope) | 0.78 |
  | Dawn validation | 0.17 |
  | V8 compile/IC | 0.14 |
  | Dawn command recording | 0.12 |

  - About 4.9 ms of the main thread's work is encoding. That is about 11
    µs per dispatch, setPipeline and setBindGroup included.
  - The `--cpu-prof` run agrees on the split. Profiled, queue.submit
    takes 2.3 ms/advance, the shared pass's dispatchWorkgroups 2.2 ms,
    and pressure-arena dispatch 0.77 ms. No pure-JS function takes
    more than 0.35 ms self.
- **No host-side change clears the 0.3 ms bar now.**
  - The wall time is GPU busy time (15.3 of 15.8 ms), and encode overlaps
    it under pipelining.
  - The per-dispatch remainder sits in Dawn and dawn.node internals (NAPI
    unwrap, sync-scope tracking, Metal encode). Only fewer dispatches and
    bindings move it.
  - So when Lane S's block-local launches cut the GPU side, the CPU
    bound falls with the dispatch count, at about 11 µs per removed
    dispatch.

## Round 5: halo fusion outside pressure — measured, not built

All runs are one each, 60 steps, `--every=60 --pipelined`, from clean
`git archive HEAD` (d4769214) exports. The metric is steadyMedian_ms.

### Extension hierarchy: knockouts

Env-gated scratch knockout in the hierarchy encode, 8x8x4 drop=2:

| variant | launches removed | steady median ms | delta |
| --- | --- | --- | --- |
| base | 0 | 14.18 | |
| fuse: restrict 16h/32h and prolong 16h/8h skipped (what halo fusion of the block-local restrict and the 1-cell-halo prolong would remove) | 4 | 14.07 | -0.10 |
| all: the whole hierarchy (restrict x3, sweep x3, prolong x3) skipped | 9 | 13.41 | -0.77 |

- "all" is not a clean bound. With the levels left unwritten, the physics
  changes downstream (drift flips sign).
- The fusable part is about 0.10 ms. That is the same as the run-to-run
  noise and below the 0.3 ms bar.
- The three 32h sweeps are cross-block: each reads the 26 neighbour blocks'
  previous sweep. Fusing them needs a 3-block halo, which means recomputing
  the restrict over 7^3 blocks. Not pursued.

### Slab sort: 24 launches to 3, built, then reverted

slabSortDense merged each dense cell (over 256 records) inside the one
workgroup that finds its start, with workgroup barriers between widths. It
replaced the 22 slabSortMerge launches.

- The GPU regression passed: records, links and bins were bit-identical to
  the per-width reference path, with dense cells present.
- Types were clean and unit tests passed, apart from the known foreign
  failure.

| scene | HEAD | slabSortDense | delta |
| --- | --- | --- | --- |
| drop=2 | 14.18 | 14.07 | -0.11 |
| rest | 2.88 | 2.90 | +0.01 |
| drop=8 | 17.47 (p95 18.49) | 18.29 (p95 20.87) | +0.82 |
| seep 300 | 30.40 | 29.24 | -1.16 (live particles 105878 vs 98168: the trajectories diverged, not comparable) |
| letters 120 | 51.10 | 50.70 | -0.40 |

- At drop=2 no cell is dense. All 22 merges early-exit, so the 0.11 ms they
  cost is pure launch cost: **about 5 us per chained, empty launch**.
- At drop=8, dense cells exist, and one workgroup serialises log2(count)
  global-memory merge rounds per cell. Each round is a binary search over
  full records. The old per-width launches spread every round over 1024
  workgroups. slabSort's comment already says why it leaves dense cells to
  dispatch-wide rounds ("one brick cannot serialize them").
- Splitting a dense cell into aligned tiles merged in parallel would still
  need log2(capacity / tile) global rounds. That is about 11 launches, or
  0.05 ms. The patch is kept in scratch and is not committed.

### Consequence for the remaining families

At about 5 us per chained launch, the planned fusions save roughly:

| family | launches | saving |
| --- | --- | --- |
| advect | 15 to 4 | ~0.05 ms |
| redistance | 14 to 5 | ~0.05 ms |
| momentum and P2G | 15 to 5 | ~0.05 ms |
| the whole 475-dispatch frame | | ~2.4 ms of launch cost in total |

Each family is below the 0.3 ms bar and inside the single-run noise. The
step's time is in the work of a few stages, not in the launch count, so
none of them were built.

## Round 6: per-stage critical paths outside pressure

Durations come from the in-process kernel timer (`--kernels=40`), which
isolates every dispatch in its own timestamped pass, so it is the
in-process form of a per-label isolated trace. Means are used because the
timestamps are quantised at about 65.6 us. "Net" takes off 5 us per
dispatch, the isolation cost.

Source runs: HEAD d4769214, 8x8x4 drop=2, 60 steps, kernels from step 40.
At step 60 that frame held 5266 particles, 76 h tiles and tiers
[384,124,4]. Dispatch counts come from a scratch build of the timer that
counts labels.

### Top 20 non-pressure stages

Sum 10.71 ms (9.23 ms net).

| stage | ms | dispatches | net ms |
| --- | --- | --- | --- |
| advectParticlesDeferred (move) | 0.406 | 1 | 0.401 |
| project | 0.390 | 1 | 0.385 |
| pair_sweepSeams | 0.300 | 2 | 0.290 |
| classifyCoarse | 0.275 | 1 | 0.270 |
| depthCoarse | 0.234 | 1 | 0.229 |
| geometryRegional | 0.222 | 2 | 0.212 |
| phi resolve | 0.212 | 4 | 0.192 |
| metric (global surface volume) | 0.197 | 1 | 0.192 |
| advectDeferred (fast) | 0.172 | 1 | 0.167 |
| remapFaces | 0.156 | 2 | 0.146 |
| advectDeferred (cubic) | 0.137 | 1 | 0.132 |
| buildRhs | 0.131 | 1 | 0.126 |
| localSpeed | 0.131 | 1 | 0.126 |
| mixedSweepBlocks | 0.140 | 3 | 0.125 |
| surfaceHeld | 0.131 | 2 | 0.121 |
| wetRegions | 0.122 | 1 | 0.117 |
| arenaDistance | 0.119 | 1 | 0.114 |
| surface volume dilate | 0.131 | 4 | 0.111 |
| transferSeams | 0.115 | 1 | 0.110 |
| mixedProlongUnknownVelocity | 0.125 | 3 | 0.110 |

### Pressure arena stages for Lane S

Every arena stage, ranked by net duration. All times and counts come
from one run, so cycle counts match: tree q at drop=2, with 12235
particles, 128 h tiles and tiers [355,151,6], 2 cycles, all band. The sum
is 5.23 ms (4.41 ms net). HEAD's lighter 5266-particle frame sums to
4.00 ms. The coarse levels are a small part of the total.

| stage | ms | dispatches | net ms |
| --- | --- | --- | --- |
| pTail | 0.777 | 10 | 0.727 |
| pUp4 | 0.484 | 13.3 | 0.417 |
| pRelax | 0.418 | 20 | 0.318 |
| pDown3 | 0.331 | 10 | 0.281 |
| pDown2 | 0.268 | 3.3 | 0.252 |
| pNativeRelax | 0.296 | 13.3 | 0.230 |
| kApplyNB | 0.200 | 2.3 | 0.188 |
| pDownB | 0.197 | 3.3 | 0.180 |
| pUp3 | 0.228 | 10 | 0.178 |
| pUpB | 0.159 | 3.3 | 0.142 |
| pBulkRows | 0.147 | 1 | 0.142 |
| pResidual2 | 0.150 | 4.3 | 0.128 |
| pNative | 0.131 | 1 | 0.126 |
| pDownA | 0.119 | 3.3 | 0.102 |
| pRows | 0.100 | 1 | 0.095 |
| pDown4 | 0.140 | 10 | 0.090 |
| pRebase | 0.106 | 4.3 | 0.084 |
| kReduce | 0.075 | 2.3 | 0.063 |
| pProjectNative | 0.066 | 1 | 0.061 |
| pProject | 0.059 | 1 | 0.054 |
| pNativeOperator, kStart, kApplyT, pRestrict0 | | | 0.05 each |
| the other 24 labels | | | under 0.05 each |

### Explanations of the top stages

Each entry gives the launch, the per-lane serial loop and its dependent
loads in the WGSL, and the knockout that bounds the loop.

- **advectParticlesDeferred** (the move).
  - Launch: a slab job of NB_MOVE_CHUNK records, plus 64-entry flat jobs.
  - Per-lane loop: with 256-record chunks each lane walked 4 records in
    turn, each a Heun 3-stage field trace of tens of dependent loads.
  - **Fixed:** 64-record jobs. The move fell 0.41 -> 0.25-0.27 ms at
    drop=2 and 0.95 -> 0.68 ms at drop=8. Committed d77c509b.
- **project**.
  - Launch: residentAll, a lane per owner, 1024-group grid. A 4h seam
    owner's lane walks up to 16 patches x 3 axes.
  - Knockout: forcing seam owners to one patch only went 0.390 -> 0.365.
  - A spread version (scan of patch counts, items over the group's lanes,
    one evaluate site) measured 0.61 ms. **Not kept.**
  - The time is the owner count times the evaluate cost, not a seam lane.
- **pair_sweepSeams**.
  - Launch: a group per seam h tile, about 76 groups, one wave.
  - Per-lane loop: each lane runs 3 serial ueExtend calls of 6 neighbour
    answers each.
  - Knockout: one axis instead of three only went 0.300 -> 0.259.
    **Not the critical path.**
- **classifyCoarse**.
  - Launch: 384 4h tiles over 6 groups of 64 tiles. Lanes share the
    queued depth samples of their job's crossing tiles.
  - Smaller jobs (8 to 64 tiles, a group per 8) went 0.275 -> 0.293.
    **No gain; reverted.**
- **depthCoarse**.
  - Per-lane loop: a lane per build entry runs a 5-iteration gradient
    trace, then 7 vertex samples of the bound metric.
  - Knockout: no trace iterations went 0.234 -> 0.150. Bound 0.08 ms,
    so not built.
- **metric**.
  - Per-lane loop: a lane per owner. A measured 4h owner walks 8 corners,
    each with authority, 8 umOwnerAt and a 6-load slope.
  - Knockout: 4h owners take one corner, 0.197 -> 0.066. Bound 0.13 ms.
  - A corner queue shared over the group (bit-identical by construction)
    measured 0.46 ms. **Not kept:** the atomics, shared memory and the
    indirect corner loop cost more than the walk.
- **geometryRegional**.
  - Launch: 4h jobs of 64 tiles, about 2 groups.
  - Per lane: a 4h lane runs gcEvaluate (plane, same-sign test or tetra
    fill). The h tiles' cut terms are already spread a lane per cell.
  - No serial list left to spread.
- **phi resolve**.
  - Launch: a 125-lane group per seam tile, 4 launches. Each lane resolves
    one vertex from the staged 27-vertex lattice.
  - No serial loop.
- **mixedSweepBlocks**.
  - Launch: one workgroup has work for about 64 blocks.
  - Per-lane loop: a 27-neighbour serial loop.
  - It is 0.125 ms over 3 launches. Not built.
- **wetRegions**.
  - Launch: strides every capacity page (UM_PAGES) over a grid of
    min(1024, pages) groups, with 2 to 3 barriers per region, absent
    regions included.
  - Its cost scales with capacity, not with the live regions. A candidate
    for a resident-page list, not yet measured.

### Result and what it says

- Only the move chunk was a true lane-serial critical path.
- The other loops bound at 0.03-0.13 ms each, all under the 0.3 ms bar.
- Spreading two of them across the group made each slower.
- These stages behave as throughput-bound at their register-limited
  occupancy: owners times per-owner cost. Single long lanes are not what
  bounds them.
- drop=2 gains 0.10-0.16 ms in total, short of the 1.5 ms target. The
  move fix pays more where particles are dense: seep -1.05 ms, letters
  -3.3 ms.
- Not verifiable bit for bit: drop=2 and the trough dam are not
  deterministic run to run even at HEAD (HEAD against HEAD: 826 against
  823 h tiles at step 5).
- tests/uniform-nbflip-symmetry fails at clean HEAD at step 6 (reflected
  surface error 0.039 m), before the repeatability check is reached.
