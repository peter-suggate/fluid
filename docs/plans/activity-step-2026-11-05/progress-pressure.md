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
