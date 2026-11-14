# Seep rewrite: technique catalogue for the NB-FLIP port (lane S, 2026-11-15)

This is the Phase 0 deliverable of `docs/plans/nbflip-port-2026-11-15.md`. It covers every technique lane S landed
or measured from S2c to S2q. The source for each entry is the matching section of
`docs/plans/seep-rewrite/progress-lane-s.md`, which also links the compare, audit and ledger artifacts.

**How the numbers were taken**
- **Hardware:** M1 Max, Dawn (node `webgpu` module, Metal backend, validation on).
- **Gate:** `tools/gpu/fluid-gpu-run.sh check`.
- **Speed claims:** interleaved A/B rounds of fenced medians (each step is advanceTo + awaitFrameCompletion). Unless
  stated otherwise, these are on the seep at 600, 1100 or 1200 steps.
- **Kernel times:**
  - per-dispatch timestamps (`profileStep`, coarse);
  - kbench on a frozen step-1100 state (below);
  - xctrace counters for occupancy and limiters.

**Equivalence classes used below**
- **bitwise:** every value anything later reads is bit-identical on a fixed state. Values that nothing reads may
  change, and the argument for "nothing reads it" is stated.
- **ulp-bounded:** differs by at most the stated ulps in the stated fraction of words.
- **iterate-changing:** the pressure iterate or the active set changes. Such a change meets the same residual
  tolerance and is then judged by the trajectory gates.

**Fenced step time over the rewrite (check slot, 1200 steps unless noted)**

| milestone | fenced ms |
|---|---|
| S2b | 17.2 |
| S2c | 13.3 |
| S2d | 12.2 |
| S2e | 11.43 |
| S2f | 10.93 |
| S2g | 10.65 |
| S2i | about 9.8 (-0.42) |
| S2j | about 9.05 (600 steps) |
| S2k | 7.78 (600 steps) |
| S2l | about 7.65 |
| S2m | 7.33 (600 steps) |
| S2n | 7.715 (1100 steps) |
| S2o | 7.38 |
| S2p | 7.03 |
| S2q | 6.79 |

Pipelined is 6.22 ms a step at S2q, one run that is not an A/B.

---

## 1. Dawn dispatch and submission economics (S2e, S2f, S2q): the biggest transferable item

NB-FLIP issues about 300 dispatches a step (paddleboats: 296 kernels), so everything here applies at full scale.

### 1.1 Findings

1. **Every encoded dispatch costs about 3.5 µs of GPU time, whatever its form.**
   - **Measured in situ:** an empty PCG cycle slot of 14 dispatches costs about 50 µs (see 2). With the CPU share
     taken out, that is about 3.2–3.5 µs a dispatch.
   - **Direct and indirect cost the same for tiny dispatches.** Launching the 64-thread pcgCheck and the 1024-thread
     coarseSolve directly with one workgroup, each returning at once when converged, measured 6.84 / 6.84 / 6.87 ms.
     Indirect with zero workgroups, plus Dawn's validation, measured 6.86 / 6.87 / 6.82 ms (`ab-chkd-1200.txt`).
   - **So a tiny real dispatch costs what Dawn's validation dispatch costs.** Only `dispatchWorkgroups(0)` is free, and
     only the CPU can choose it, which would need a readback (forbidden).
2. **Dawn validates every `dispatchWorkgroupsIndirect` with an injected one-workgroup compute dispatch.** That
   dispatch copies and validates the arguments against maxComputeWorkgroupsPerDimension, and Dawn then restores your
   pipeline and bind group.
   - **Diagnostic:** the `enable-dawn-features=skip_validation` toggle, used only in the lane S smoke runner, never in
     the product. At 600 steps:

     | MAX_CYCLES | validation on | skip_validation |
     |---|---|---|
     | 9 | 6.57 | 6.15 / 6.17 |
     | 15 | about 6.89 | 6.14 / 6.19 |

   - **What that shows:** with validation off, empty cycle slots become free, and the whole step is 0.4 ms faster.
   - **The toggle is not a fix.** Browsers validate, and the device belongs to the harness.
3. **An empty PCG cycle slot costs about 50 µs** (13 dispatches at S2p, 14 at S2q).

   | MAX_CYCLES (S2p build, 1200 steps, 3 interleaved rounds) | fenced ms |
   |---|---|
   | 9 | 7.02 / 6.98 / 7.02 |
   | 12 | 7.15 / 7.15 / 7.11 |
   | 15 | 7.30 / 7.24 / 7.34 |

   - **The tail:** with a mean of 3.3 cycles and a cap of 9, about 5.7 empty slots a step cost about 0.28 ms.
   - **Earlier estimates were higher:** about 6 µs (S2e) and 6.4 µs (S2j) per empty indirect dispatch. These were
     MAX_CYCLES deltas that included the CPU submit share and a different dispatch count per slot.
4. **Command translation happens at `queue.submit`, and fenced GPU work waits for all of it.**
   - **Measured** with a timer around `queue.submit` in a scratch tree: 0.152 ms with MAX_CYCLES 9 and 0.228 ms with
     15. That is about 13 µs a cycle slot, or about 0.8 µs a dispatch.
   - **With skip_validation:** 0.092 / 0.131 ms, so Dawn also spends CPU encoding its validation dispatches.
   - **Separately, JS-side encoding** (setPipeline, setBindGroup and dispatch for about 200 dispatches) cost 0.70 ms of
     CPU a step at S2f, before the GPU could start.
5. **Microbench** (`dispatch-bench.ts`: 200 dispatches alternating 13 pipelines, each binding one read_write storage
   buffer, 512-thread workgroups). Use it for ratios only; the absolute values are above the in-situ figures.

   | µs a dispatch | validation on | skip_validation |
   |---|---|---|
   | indirect, 0 workgroups | 12.78 | 7.21 |
   | indirect, 1 workgroup | 16.71 | 11.14 |
   | direct, 1 workgroup that returns | 4.59 | 4.59 |
   | direct, 64 workgroups that read a gate word and return | 5.24 | 5.24 |

### 1.2 Levers landed

| technique | gain (condition) | class | generality |
|---|---|---|---|
| **Encode-ahead (S2f).** A step's commands depend only on its parity, since the records are fixed and every data-dependent count is an indirect argument. advanceTo submits a command buffer built during the previous step, then encodes the next while the GPU runs. It rebuilds if another path advanced the step and the parity no longer matches. | 11.49 / 11.49 -> 10.91 / 10.94 ms, fenced (-0.55) | bitwise (no GPU change) | Holds wherever the step's command list is fixed and data-dependent sizes are indirect. Check NB-FLIP for CPU-sized dispatches, which would break it. |
| **Two-part submit (S2q).** The step goes out as two command buffers, the first 4 dispatches (classify, clear, count, scan) and the rest, submitted one after the other. The GPU starts the prefix while Dawn translates the rest. | Unsplit 6.57 / 6.57; split at 4: 6.46 / 6.41 / 6.37 / 6.43; at 2: 6.43 / 6.46; at 16: 6.50 / 6.46; at 4 and 24: 6.45 / 6.48; at 60: 6.57 / 6.57 (600 steps). Fenced only; pipelined already overlaps. | bitwise | General. The first part must be short to translate but long enough on the GPU to cover translating the rest. A 60-dispatch prefix gave nothing. |
| **Convergence test before the V-cycle (S2q).** A one-workgroup `pcgCheck` runs right after the update X_k and gates the whole of iteration k. This is standard PCG order: test the residual, then apply the preconditioner (Saad 2003, Alg. 9.1). Before, the test sat in coarseSolve after iteration k's down sweep (5 real passes), so the converged iteration still ran that sweep. | About -0.09 ms (`ab-chk-1200.txt`); the whole S2q batch was -0.30 ms (7.10 / 7.08 / 7.10 -> 6.76 / 6.80 / 6.80) | bitwise by construction (same math and record layout), but not checked on a frozen state; cycles and gates were unchanged | General for any gated multigrid-PCG with a fixed slot count. |
| **Gated slots (S2e).** Every later iteration's dispatches read their workgroup counts from records the previous test writes; on convergence all later gates close at once. coarseSolve k was indirect from alternating buffers cgate0/1, because a dispatch may not read its indirect arguments from a buffer it binds writable (S2q replaced this with direct launches that read C_CONV). | About 50 µs per idle coarseSolve, about -0.25 ms a step at the time | bitwise | General. Note the rule on indirect arguments in a writable binding. |
| **Fewer dispatches a cycle.** D2 fused into D3 (S2g, -1 a cycle). Level-3 block Jacobi instead of two colours (S2g, -2 a cycle each way, see 2). | 10.92 -> 10.69 and 9.94 -> 9.76 ms | iterate-changing | Per-cycle savings multiply by the cap, not by the mean. |

### 1.3 Ruled out
- **Lowering MAX_CYCLES.** The coordinator ruled it stays at 9: the peak of 7 is one realisation of a
  nondeterministic run, and a capped step leaves pressure unconverged.
- **Zeroing later slots' arguments, or a cycle budget computed on the GPU.** Neither removes a slot's cost, because
  that cost is its encoded dispatch count.
- **Converting indirect launches to direct ones.** No gain for tiny dispatches (above).
- **Folding the check into pcgUpdate's last workgroup.** It needs device-scope ordering between C_RES's atomicMax and a
  done counter, which WGSL's relaxed atomics and workgroup-scope storageBarrier do not give; that would be a hack.
  The same holds for folding coarseSolve into mgRestrict3's last workgroup.

**Port note.** Count NB-FLIP's encoded dispatches per step, and its indirect ones in particular, before optimising
kernels. At about 3.5 µs each, 300 dispatches are about 1 ms of floor. Lane A's attribution should report empty
indirect dispatches per step.

---

## 2. Pressure solve (aggregation multigrid-preconditioned CG over a liquid-brick list)

| technique | gain (condition) | class | what made it pay; generality |
|---|---|---|---|
| **Staggered L0 smoother (S2m).** Multiplicative Schwarz over two partitions: bricks, then blocks shifted by (4, 0, 4) cells (Smith, Bjørstad & Gropp 1996, *Domain Decomposition*). Post-smoothing runs the exact reverse: the shifted pass with its colours reversed, then the brick pass reversed. Shifted blocks are stamped by the operator pass into a 65×32×9 flag table and listed. 3 sweeps in the shifted passes. | Mean cycles 5.17 -> 3.32, max 12 -> 7. 7.655 -> 7.335 ms at 600 steps (MAX_CYCLES 12 -> 9 included; the shift alone at MAX 12 was 7.46). | iterate-changing | The late cycles were spent on 1–2-cell films at brick faces (a probe logged every row with r > TOL at k >= 3: x, z = 0 or 7 mod 8). Brick-aligned block smoothers with a frozen halo converge slowest across brick faces. This holds for any brick-sparse solver with thin films (dam break, letters). A 3D shift took the same cycles over twice the blocks, so xz only. 2 shifted sweeps: no gain; 1: slower (3.96 cycles). |
| **Symmetry oracle.** `lib/methods/seep/seep-smoother.test.ts`, a CPU Float64 model on a 32×16×32 film-plus-pool mask: z^T M^-1 y = y^T M^-1 z to 1e-12, and positive. A mutated post order fails it. | | test | Port it with any smoother change, since a non-symmetric preconditioner silently breaks CG. |
| **Exact coarsest solve (S2i).** The 32×32 L6 operator is inverted once per step in coarsenTop by Gauss-Jordan (SPD, no pivoting; FAIL_PIVOT latches on a non-positive pivot). coarseSolve applies one mat-vec in place of 41 red-black sweeps (Trottenberg et al. 2001; Golub & Van Loan 2013 §4.2). | With the Jacobi guess: cycles 6.29 -> 5.48, -0.42 ms at 600 steps | iterate-changing | The seep's coarsest level is fixed by the scene. NB-FLIP needs a general coarsest of bounded size (an aggregate count cap, or the top of its own hierarchy). |
| **Jacobi warm-start guess (S2i).** One Jacobi sweep from last step's pressure before PCG (Saad 2003 §4.1). Rows whose ghost-fluid diagonal changed otherwise start with a residual of d·x0. | Part of the -0.42 above | iterate-changing | General (any warm-started free-surface solve). |
| **Level-3 block Jacobi (S2g).** Additive Schwarz over level-4 blocks (Smith et al. 1996), 8 blocks a 64-thread workgroup. Pre-smoothing from zero has a zero halo, so blocks are independent. Post-smoothing reads each neighbour's pre-smoothed plus prolonged value, so the cycle stays symmetric. Replaced two-colour multiplicative. | Cycles 6.26 -> 6.35, 2 fewer dispatches each way a cycle: 10.92 -> 10.69 ms | iterate-changing | Pays when the dispatch floor costs more than the extra cycle fraction. Two colours took 0.15 fewer cycles for 2 more dispatches each way: a net loss. |
| **Restriction fused into pre-smoothing (S2g).** D2 (level-2 residual to level 3) is formed per level-4 block in the workgroup that pre-smooths that block. | -1 dispatch a cycle. Same 300-step state: 6.62 vs 6.67 cycles, 9.94 -> 9.76 ms. | iterate-changing (sum order) | General. |
| **Preconditioner settings (S2e, S2j).** Aggregation over-correction A_c = R A P / OC (Braess 1995) at OC 1.8 -> 1.6: 7.6 -> 6.7 cycles (1.4, 1.5 and 2.1 were also measured). Fine red-black sweeps 2 -> 3: 6.8 -> 6.3 cycles, net -0.15 ms. Level-4 and level-5 sweeps 4+4 -> 2+2: 5.50 -> 5.39 cycles. | as stated | iterate-changing | Settings are tuned to the seep; re-tune per scene set, with cross-scene gates. |
| **Two-level fold reductions (S2d).** pcgUpdate, pcgPrecond and pcgDirection: 32 threads fold 16 each, then thread 0 folds the 32. That is 2 barriers on a conflict-free stride, instead of a 512-thread tree. PCG dot sums load their 64 partial slots in parallel (S2c). | Part of S2c / S2d (13.3 -> 12.2 ms with the other S2d items) | ulp-bounded for dots (reassociation); bitwise for min / max | General. |
| **Liquid-brick list (S2a).** PCG and MG run over liquid bricks only (about 620 of about 2600 active), published to indirect arguments. | Part of S2a | bitwise (rows outside are not unknowns) | Check NB-FLIP's own list. |

**Measured and rejected (CPU Float64 model `mgmodel.mts`, 64×32×32 and 128×64×32 masks with a free surface).**
Iterations to 1e-2 / 1e-3 / 1e-4; "-" means the target was not reached in 40 iterations.

| preconditioner | white-noise rhs | smooth rhs | surface rhs |
|---|---|---|---|
| V-cycle (ours) | 5 / 7 / 9 | 7 / 8 / 10 | 1 / 2 / 3 |
| BPX additive, OC 1.6 | - / - / - | - / - / - | 8 / 15 / 27 |
| BPX additive, OC 1.0 | 33 / 40 / - | 40 / - / - | 5 / 8 / 20 |
| brick-local V + additive coarse | 29 / 40 / - | 40 / - / - | 3 / 7 / 14 |
| zero-halo S3 + V4 hybrid | - / - / - | - / - / - | 4 / 12 / 23 |

Additive forms need 30–40 or more iterations against 7–10, so cutting a cycle from 11 passes to 5 cannot pay.

**Other rejections**
- More fine sweeps (8 instead of 4): -0.6 cycles, about -0.1 ms.
- Fine passes 6 / 10 / 16: 6.56 / 6.29 / 6.15 cycles; the fine level does not limit convergence.
- Capping the Dirichlet part of the coarse diagonal: no change.
- THETA_MIN 0.05: -1 cycle, but it changes the discretisation (fixed at 1e-3).
- Subgroup-shuffle coarsest sweeps: +0.2 ms, unexplained.
- Fewer coarsest sweeps: more cycles.
- BULK through the surface row (seep-only): cycles 5.2 -> 11.8.
- Chronopoulos–Gear / pipelined CG to merge precond and direction: z on the halo is not available inside the
  dispatch.

**coarseSolve's cost is mostly serialisation.**
- **The probe:** running the V-cycle body twice costs only +0.11 ms fenced, so the body is about 21 µs a call. The
  rest of the profile's about 85 µs a call is a one-workgroup dispatch between two full-GPU ones.
- **kbench split:**

  | part | µs |
  |---|---|
  | whole call | 83–103 |
  | empty 1024-thread dispatch | 17.5 |
  | preamble | about 5 |
  | each level-4 sweep (offS work, not barriers) | about 8 |
  | level-5 sweeps | about 15 |
  | top solve | about 10 |

- **Spills are not the main cost:** moving level-4 couplings to workgroup memory took the register limit from 704 to
  896 threads, and the gain was within noise (1-ulp fma differences).
- **Single-workgroup work is dear.** Restricting level-3 blocks inside coarseSolve to drop mgRestrict3 cost +0.1 ms
  (S2f).

---

## 3. Kernel shape: occupancy, register pressure, lane balance

| technique | gain (condition) | class | what made it pay; generality |
|---|---|---|---|
| **Split register-heavy traces into their own kernel (S2p).** velocity pass 1 (PHIC, fill, transfers, far-skip) writes each cell's need mask; `seep.velocityTrace` runs the backtraces as 8 slabs of 64 cells per brick. Each slab compacts its traced faces into a workgroup list, so full SIMD groups run the RK2 trace. There is no velocity tile. | velocity + trace 812 -> 618 µs at step 1100 (kbench, 7 rounds). S2p overall 7.39 / 7.34 / 7.32 -> 7.03 ms. | bitwise (US identical on a frozen state) | The merged kernel's register limit was 832 threads, plus 28.7 KB of workgroup memory: one 512-thread brick a core. The trace alone sits at 704–832 whatever the form (24 trilinear loads in flight), so isolating it lets light bricks stop holding a core. Holds wherever one kernel mixes cheap per-cell work with a heavy sampler (NB-FLIP advection, extension). |
| **Spread one-thread-per-item tails over 16 lanes (S2p).** velocityBulk lists its interface coarse faces; 16 threads trace each face's 16 fine faces, instead of one thread doing all 16 serially. | velocityBulk 264 -> 157 µs | bitwise | General. |
| **Pack sparse bricks into one workgroup (S2c, S2l).** Level-2 MG and coarsenBrick run 8 bricks in a 64-thread workgroup, instead of 8 threads of a workgroup per brick. | Part of S2c; coarsenBrick about -0.1 ms with S2l | bitwise | General for coarse levels. |
| **Stage the inputs a workgroup reuses.** One 10³ vertex tile for velocity's level-set reads (S2c). A coarse-velocity tile in velocityBulk: 0.92 -> 0.26 ms (S2c). The shadow kernel's 4³ coarse faces per component: 192 cface calls instead of 12k, 0.25 -> 0.02 ms (S2e). A sign tile for redistInit's 26-neighbour test: 0.52 -> 0.13 ms (S2c). | as stated | bitwise (same interpolation order) | General. The limit is workgroup memory: about 29 KB leaves one workgroup per core (see rejections). |
| **Stage a tile only when enough work uses it (S2d).** Velocity staged its tile only for bricks with 96 or more traced faces; sparse bricks read global memory. | Part of S2d | bitwise | Superseded by S2p's slabs. |
| **Contiguous particle positions (S2d).** Positions in one block, velocities at +CAP, so separate, count and scatter stream half the bytes. | Part of S2d | bitwise | General. |

**Method for register pressure**
- Dump Dawn's MSL (dump_shaders, disable_symbol_renaming), strip `max_total_threads_per_threadgroup`, and compile
  with Metal. The pipeline's maxTotalThreadsPerThreadgroup is then the register limit (`mstat.sh`, `mtlstat`).
- Measured limits:

  | kernel | threads |
  |---|---|
  | velocity (merged) | 832 |
  | velocity pass 1 | 1024 |
  | velocityTrace | 768 |
  | velocityBulk | 512 |
  | coarseSolve | 704 |

- The MSL also shows that Dawn zero-initialises workgroup memory and inserts loop counters (tint_loop_idx).

**Rejected (numbers from the S2p and S2l sections)**
- **Trace variants, velocity + trace µs:**

  | variant | µs |
  |---|---|
  | merged HEAD | 812–852 |
  | split, per-brick 512 threads, with tile | 692–705 |
  | split, per-brick 512 threads, no tile | 795 |
  | compacted per brick | 645–655 |
  | compacted, 256 threads, no tile | 633–649 |
  | compacted, sorted by axis | 677–686 |
  | slabs of 32 / 64 / 128 threads | 655 / 618–636 / 636–638 (64 kept) |

  Sampling components in a non-unrolled loop did not lower the register limit.
- velocityBulk folded into velocity as leading workgroups: no gain.
- Velocity tile halo 2 -> 1: within noise.
- Issuing velocity's 11 tile loads up front: -9 of 812 µs.

---

## 4. Level-set maintenance

| technique | gain (condition) | class | what made it pay; generality |
|---|---|---|---|
| **Skip fixed-point tiles (S2a, S2o).** Redistance and extension skip tiles with nothing to do. Redistance also skips all -PHI_MAX tiles, a fixed point since eik(6, 6, 6) = 6.58 > 6 and nearSign keeps solid-interior signs, as all +PHI_MAX tiles already were. Extension skips sweeps when the tile has no unknown face. | Part of S2o | bitwise | Pays in a large deep reservoir. In NB-FLIP the band is narrower, so check the fraction of skippable tiles first. |
| **Packed sign-code tile for the cut / frozen test (S2o).** Codes 0 (not open), 1 (open, phi >= 0) and 2 (open, phi < 0), read at constant offsets, replace a 27-neighbour test with k/3 and k/9 index arithmetic (ALU-bound). | redistSweep.2 275 -> 223 µs, sweep.1 173 -> 140, init 100 -> 92 | bitwise | General. |
| **Mass fit: compact true cut cells before the tetrahedral fill (S2o).** Neighbour chamber flags and the 9³ particle counts are staged with the busy test. The mismatch is computed only for cells with a band corner (\|phi\| < 1.5). A cell with no negative corner fills 0, one with all corners negative fills 1, and cut cells are compacted so whole SIMD groups run the fill. A tetrahedron of zeros counts as empty in `tetra`, so a cell whose top corner is 0 takes the full path. | 148 -> 109 µs a pass, 4 passes a step | ulp-bounded: at most 1 ulp (1.2e-7) in 0.17% of words, because the shortcut gives exactly F - 0 or F - 1 where the full path rounded | Seep's mass-conserving fit (van der Pijl et al. 2005) does not exist in NB-FLIP; the compaction pattern transfers to any per-cell fill. |
| **One barrier per extension sweep for all three components (S2d).** UNK and SHUT sentinels in the tile. | Part of S2d | bitwise | General. |

**Rejected**
- extend: staging the 27 neighbour DEEP flags, 238 -> 255 µs.
- redistInit into redistSweep.1: needs one more phi buffer for the read/write race, for about 0.1 ms.
- Multi-iteration massFit in one tile: either the frozen halo over-corrects at brick boundaries, or an exact 2-halo
  tile changes the column carry.

---

## 5. Particle transfer and separation

| technique | gain (condition) | class | generality |
|---|---|---|---|
| **P2G transposed lanes (S2n).** Lane l of SIMD group l/32 takes every eighth particle. In sorted order, a cell's particles share a stencil, so the lanes of one SIMD group would otherwise hit the same shared-memory word at once. Fixed-point integer sums are order-independent. | p2g 0.33 -> 0.23 ms at 1100 (310k particles) | bitwise | General for sorted particle splats with shared-memory atomics. |
| **Flag stores instead of contended atomicAdd (S2n).** B_PCOUNT is read only as nonzero, so G2P loads it and stores 1 if needed, instead of every particle doing atomicAdd on one word per brick. Fast bits are ORed only when missing. | g2p 0.41 -> 0.36 ms | bitwise (only "nonzero" is read) | Check what NB-FLIP actually reads from its per-brick counters. |
| **Skip a redundant atomicOr (S2n).** count skips the octant atomicOr when its atomicAdd already returned that bit. | count 0.08 -> 0.07 ms | bitwise | General. |
| **Separation restructure (S2l).** The four x-rows' (start, end) are loaded up front, a row's two contiguous cells are merged, and two neighbours are processed per iteration in the same accumulation order. | With the coarsenBrick packing, -0.1 to -0.3 ms | bitwise | General. |

S2n's batch was -0.065 ms fenced over 1100 steps (7.78 -> 7.715), and the gain grows with particle count.

**Attribution (probe kernels, step 1100, ms)**

| kernel | total | breakdown |
|---|---|---|
| separate | 0.52 | candidate loop 0.41 |
| g2p | 0.39 | no brick atomics 0.34 |
| p2g | 0.31 | shared-atomic splat 0.25 |

**Separate: rejected variants (ms, against 0.52–0.56)**

| variant | ms |
|---|---|
| shared-memory tiles (30 KB, serial phases at about one workgroup a core) | 1.72 |
| one flattened loop over the eight ranges | 0.66 |
| eight cells in lock step | 0.64–0.89 |
| the 3³ cells around the own cell | 0.82 |
| workgroup 64 / 128 | 0.59 / 0.54 |
| inverseSqrt form | 0.54 |
| x-only loads, strided | 0.39 vs 0.41 |
| x-only loads, packed | 0.54 |
| staging the neighbour box (29 KB) | 0.72 vs 0.52 |

The candidate loop costs about 100 lane-cycles a candidate; it is latency at limited occupancy (17%, from the
counters). Tiled per-brick G2P (9³ faces in workgroup memory) was slower, because crowded bricks unbalance it.

---

## 6. Velocity and advection: skip what nothing reads

| technique | gain (condition) | class | what made it pay; generality |
|---|---|---|---|
| **No backtrace for chamber air faces without particle weight (S2c).** The projection reads faces of liquid cells only. G2P's FLIP difference reads the trilinear stencil around a particle, which lies inside its own quadratic B-spline P2G support (1.5h), so every face it reads carries weight. Standard FLIP keeps no grid-advected velocity where particles carry the liquid (Zhu & Bridson 2005). | velocity 2.9 -> 1.4 ms (the jet's air band was half the kernel) | bitwise in everything read | Holds in NB-FLIP wherever particles carry the liquid. State the "read by" argument per kernel. |
| **Far-air skip (S2a).** Faces whose two cells lie wholly beyond FAR_AIR = 4 cells into air get Us = 0 with no trace. Projection, FLIP and particle support read faces next to liquid; advection and extension read U, which projection zeroes there. | Part of S2a | bitwise in everything read | General. |
| **advectPhi skips vertices the particle surface overwrites (S2c).** | 0.72 -> 0.39 ms | bitwise in what is read | Applies where a particle-built surface overwrites grid phi. |
| **Per-axis CFL activation band (S2j).** Chamber bricks activate two bricks along an axis only from a brick whose particles moved more than FAST_CELLS = 6 cells a step along it, and one brick otherwise. 6 cells keeps a leaving particle and its 1.5h support in the next brick (Bridson 2015 §6.3; Ferstl 2016 §3). | Active chamber bricks 1451 -> about 906. -0.75 ms at 600 steps (9.81–9.83 -> 9.04–9.13). | iterate-changing (active set) | Holds for moving fluid with a per-axis speed bound. Particles whose stencil touched an inactive brick were unchanged: 41.4k vs 40.7k summed over 60 samples. |

**Not landed: the reservoir air-face skip (-24 µs, S2o).**
- **What it skips:** projection reads Us only if a face is open and either cell's PHIC is negative, or the face borders
  a BULK cell. An exact skip of faces between two non-negative-PHIC cells must use the PHIC that projection will
  read.
- **Why the corners are not enough:** a neighbour cell in a DEEP or inactive brick has no PHIC written this step (DEEP
  bricks skip the level set; inactive ones keep an older value). So 12 non-negative corners do not prove that
  projection skips the face.
- **The landable form:** compute the neighbour's PHIC from the tile in the same sum order, skip only when the
  neighbour brick is active and not DEEP, and add a fixed-state bitwise test.

---

## 7. Physics fix with general relevance: emission velocity (S2h)
- **The defect:** emission gave a new particle a trilinear sample on the outlet plane. Closed wall faces next to the 2×2
  throat (u = 0) diluted u_x by 0.875² = 0.766 (2.59 -> 1.98 m/s), and v and w were fed back from the falling jet.
- **The fix:**
  - a particle crossing an outlet face takes the face's own MAC normal velocity (Harlow & Welch 1965);
  - it takes the donor cell's tangential velocity (Courant, Isaacson & Rees 1952).
- **Effect:**
  - impact x 354 -> 375.8 (NB-FLIP 379.8);
  - slice RMS against NB-FLIP about 9 -> 1.4 cells.
- **Generality:** this applies wherever NB-FLIP seeds particles from grid velocity next to closed faces, which is
  iterate-changing physics. Check that NB-FLIP's sampleBound and seeding do not do the same thing.

---

## 8. Seep-only: do not port
- Dense brick-major storage of the whole 512×256×64 box (about 1.05 GB), which violates memory scaling.
- The composite h/4h BULK reservoir (S2a; Losasso, Gibou & Fedkiw 2004). Its rows are baked: S2k raised BULK to row
  184 for -1.0 ms, and a smaller 2×2×2-brick throat block gave -0.25 ms. BULK through the surface row broke the
  preconditioner (cycles 5.2 -> 11.8).
- Particles confined to one chamber, and the outlet emission source.
- The scene-fixed 32×32 coarsest level (needs a general equivalent; see 2).
- Preconditioner constants tuned to the seep (OC, sweep counts).

---

## 9. Tools for a port (all under `~/code/me/fluid-scratch/lane-s/`)

All of them load an implementation tree through `SEEP_IMPL` or a kernels path, so they never write the checkout. Each
needs its solver-specific hooks retargeted at NB-FLIP.

| tool | what it does | port use |
|---|---|---|
| `tools/kbench.ts` + `tools/kb.sh` | Warms the solver to a step (default 1100) and runs the step up to the dispatch before a target. Then it times the target (same cfg and launch) and variant kernels on that frozen state: 30 back-to-back dispatches a timestamped pass, 7 interleaved rounds, medians. Variants: `tree#export`, multi-kernel `a+b`, y-split indirect `name@yK`. `--check=buf:off:words` compares outputs bitwise, after poisoning the slice with 0x7fc0dead before each run (`--nopoison` turns that off). | The per-optimisation fixed-state equivalence check the plan requires, and the most reliable per-kernel cost figure. It needs a hook to stop NB-FLIP's step at a named dispatch. |
| `tools/seep-xctrace.ts` + worker | Hardware counters per kernel at a late step, following the gpu-frame-profile skill: attach, quiet-GPU wait (< 15% for 5 s), each dispatch its own labelled pass, and intervals joined by encoder-id. Split passes serialise the step, so read occupancy and limiters, not times. | Lane A's top-10 occupancy and limiter table. |
| `tools/mstat.sh`, `mtlstat`, `dump-msl.ts`, `dump-wgsl.ts` | WGSL -> Dawn MSL -> the uncapped Metal maxThreads (the register limit). | Before splitting or reshaping a heavy kernel. |
| `tools/dispatch-bench.ts` | Dispatch-cost microbench (ind0 / ind1 / dir1 / dirG, `--toggles=`). | Pricing dispatch-count changes; ratios only. |
| `tools/seep-solver-smoke.ts` | Fenced median, the CPU advanceTo time (`enc`), the submit time when the tree records it (`sub`), `--hist` (ctrl 200..215 probe words), `--profile`/`--profile-steps`, `--ledger`, and `SMOKE_TOGGLES` (diagnostic device toggles). | Template for an NB-FLIP smoke runner. |
| `abe.sh` (`abe.sh steps rounds "name\|tree\|ENV"...`), `ab2.sh` (HEAD vs working tree from a scratch copy), `snap.sh` (snapshot a tree, lib/core symlinked) | Interleaved fenced A/B: cycles, cap hits, u, q, fenced, enc, sub. | Every speed claim. Always interleave, since the check slot drifts by about 0.1 ms between sessions. |
| `ledger.sh steps every` | The staged volume ledger (chamber phi against delivered after each stage). | Seep-specific; the pattern (per-stage conservation readout) transfers to the dam break. |

**Process pitfalls we hit**
- **WGSL uniformity:** an early `return` on a non-uniform value before a barrier fails to compile. Fold it into the
  work predicate, or use `workgroupUniformLoad`.
- **Indirect arguments:** a dispatch may not read its indirect arguments from a buffer it binds writable. Use paired
  buffers or direct launches.
- **Bitwise checks:** a check without poisoning passed falsely on leftover values.
- **GPU contention:** other lanes' runs disturbed several A/B rounds (S2l), so check that the GPU is quiet before
  counters and repeat suspect rounds.
- **Workgroup memory:** about 29–30 KB per workgroup leaves one workgroup per core. Three staging ideas lost to it
  (separate, tiled G2P, separate tiles).

---

## 10. Open items left by lane S
- **Capped-step failure bit: landed for the seep; open for NB-FLIP (port lane P1).**
  - **Before:** the pressure cap was counted in diagnostics (`capHits`, C_CAP_HITS) but was not a failure, so the
    bench and `assertSimulationHealthy` passed a capped step silently, with an unconverged pressure.
  - **Seep now:** `FAIL.PRESSURE_CAP = 16` is latched by pcgCheck when it stops at MAX_CYCLES above tolerance, and the
    failure text carries the capped-step count. The unit test is `lib/methods/seep/seep-diagnostics.test.ts`.
  - **Checked on the GPU:** with tolerance 1e-12 the solver latches at step 1 ("Seep solver latched PRESSURE_CAP
    (..., capped steps 1)"), which is the halt condition seep-bench throws on. A normal 1200-step check run latches
    nothing; left tank 363.468 L against lane H's 363.472 L for 2be18980.
  - **For P1:** NB-FLIP's pressure solve should treat its iteration cap the same way: a latched failure bit, not only a
    counter. Not ported by lane S.
- **mgRestrict3 folded into coarseSolve:** prototype in `v/r3` (8 threads a block, 128 blocks a round, same sum
  order). It is untested on the GPU, and S2f's similar attempt cost +0.1 ms (see 2).
- **coarseSolve fixed cost:** the 17.5 µs empty 1024-thread dispatch, possibly a smaller workgroup. Not started.
- **Reservoir air-face skip:** see 6.
