# NB-FLIP particle-kernel scaling — 2026-10-27

Question: does any particle-touching kernel's per-step cost grow faster than linearly in particle count, so that frame time falls off steeply as the count grows?

Answer: no. Every particle kernel measured is linear in live particle count, from 16k to 1.14M particles, across three scenes. The only cost that grows faster than count is **sample separation** (`nbSeparation`, inside the move kernel). Its per-particle cost goes up with local density, not with total count, and it is the largest particle cost at high counts. The move kernel is also capped at 12.5% occupancy by its register use (Follow-up 2), which made separation cost about twice as much inside the move as on its own. One change was kept: separation now runs as its own pre-pass ahead of the move (Follow-up 3), which saves about 9 ms a step (about 12%) at 1.1M particles and is neutral on the seep. The canonical within-cell order turned out to matter for the physics, so it cannot be dropped to save time.

Logs and run directories: `artifacts/particle-scaling/` (run with `run.sh`, reduced with `kern.py` and `fit.py`).

## Instrumentation added (kept)

`tools/probe-tier-budget.ts`:

- `--kernels=a-b,c-d` times each kernel over the given step windows with `tools/gpu-kernel-timer.ts`, and prints median ms per label.
- `--cells` prints a census at each sample step: the histogram of cell populations (1–8, 9–20, 21–64, 65–256, >256), the largest cell, the number of records in cells over 256, Σk²/N, and the largest brick and the 8 most-populated bricks.

These numbers come from the timer, so they show where time goes rather than absolute cost. The timer splits each pass per dispatch and has a quantum of about 65 µs. All timing ran in `fluid-gpu-run.sh timing` mode with `volumeControlSeconds:0`.

## Scaling table (baseline)

µs per 1k particles comes from a least-squares fit over each run's windows. The low-half and high-half slopes agree to within noise for every kernel; none rises at the high end.

| kernel | seep 16k–210k | letters 62k–618k | letters uniform+band4 94k–1.14M | at 1.14M (ms) |
|---|---|---|---|---|
| move (`advectParticlesDeferred`) | 12.8 | 11.7 | 12.4 | 14.2 |
| deposit (`transferDeposit`) | ~1.0 | 2.5 | 1.8 | ~2.1 |
| `slabSort` (+canonical merge) | 1.33 | 1.68 | 1.52 | 1.9 |
| `slabSortMerge` (all widths) | 0.13 | — | small | <0.2 |
| `slabSortFinish` | — | — | 0.33 | 0.4 |
| seed | 0.9 | — | — | — |
| splat (plan + occupied) | ≤0.2 | 0.9 | 0.79 | 0.9 |
| recordDelta | — | — | ~0.8 | 0.9 |
| timer total | — | — | 44.5 (intercept 23 ms) | 74.5 |

Step time: seep median 36.3 ms (p95 45.2), letters 50.3 ms, letters uniform 50.6 ms (p95 87.5).

The seep physics in the baseline matched the reference: drift −0.0013, throat Cd 0.813.

Cell census (largest values seen over each run):

| scene | max cell | cells >256 | max brick (records) | Σk²/N |
|---|---|---|---|---|
| seep | 184 | 0 | 2882 | 6.7–8.2 |
| letters | 77 | 0 | 3586 | 6.2–8.2 |
| letters uniform+band4 | 111 | 0 | 5237 | 9.0–9.5 |

Dense cells over 256 never occur in normal running, so the dispatch-wide `slabSortMerge` rounds are almost always no-ops.

## Findings

1. **Sample separation is the largest particle cost, and it depends on density.** With `sampleSeparation:off` as a diagnostic (`diag-lettersu-nosep`), move fell from 14.2 ms to 6.0–6.2 ms at 1.1–1.2M particles. That puts separation at about 7 µs/1k, or about 8 ms at 1.1M. For each sample it loops over every record in the 2×2×2 neighbouring cells. Its cost per particle therefore grows with Σk²/N, the local crowding, rather than with total count. The loop reads full 48-byte `particles[j]` records when it needs only position and mass, and it evaluates `pow(m, 1/3)` for every pair. This code belongs to the seep agent, so I left it alone. Recommended changes, in order of value:
   - read neighbour position and mass from the compact `links` stream (16 B, already in cell order) instead of the particle records;
   - use an 8-entry cube-root table, since mass is an integer from 1 to 8;
   - stop early once enough neighbours are found, or cap the neighbour loop at a fixed per-cell sample. This would change behaviour, so it needs the physics gates.
2. **Crowded-cell merge** (`nbMergeRun` / `nbMergeAbsorbed`) costs O(k²/cap) per crowded cell. At the observed maximum cell size of about 180 it stays within move's linear fit. It only becomes a problem if cells pile up (see 3).
3. **Canonical within-cell order is needed for correct physics, not only for repeatability.** As a diagnostic I turned the canonical merge off, leaving atomic-arrival order (`diag-lettersu-noorder`, `-noorder2`). `slabSort` fell from 1.9 to 0.98 ms, so the canonical merge costs about 0.9 ms at 1.1M. But the dynamics changed:
   - the largest cell grew to 828, with 17 cells over 256;
   - Σk²/N rose to 13–18;
   - move rose to 17–23 ms, 35–60% slower.

   The probable mechanism: the crowded-cell merge picks the run's lowest ranks as gatherers. With arrival order, full-mass gatherers (mass 8) stay at the head of the run and refuse gathers, so crowded cells never drain. The NB-FLIP symmetry test (`tests/uniform-nbflip-symmetry.test.ts`, bit-identical phi over 60 steps) also depends on this order. Removing it is rejected.
4. **Binning and sorting are close to the hardware limit.** Without ordering, `slabSort` takes 0.98 ms, against an estimated bandwidth floor of about 0.65 ms for 1.1M records. Running one workgroup per brick slab therefore isn't causing a large load imbalance at these brick sizes (largest brick 5237 records). A device-wide radix or counting sort would save at most about 0.3 ms out of a 50–75 ms step.
5. **The fixed cost dominates at moderate counts.** At 1.14M particles the timer total is about 74.5 ms, made up of a 23 ms intercept (grid, pressure and arena kernels) plus 44.5 µs/1k for particles. The particle share is linear, so any falloff the user sees comes from density (more particles per cell means more separation pairs, and pile-ups mean more merge work). Total count alone doesn't explain it.

## Changes considered and not kept

- **Dropping canonical order (atomic order only).** Rejected because physics changed (finding 3).
- **Hybrid canonical order.** The idea: an all-pairs rank using compact position-bit keys for cells of 64 or fewer, with merge rounds only for larger cells. Upper bound on the saving: 0.5–0.9 ms at 1.1M, about 1% of the step, which is well inside the 1.66× repeat spread. It can't pass the gate and adds complexity, so I didn't implement it.
- **Device-wide radix sort replacing the per-brick slab sort.** The upper bound is about 0.3 ms (finding 4), so it's not worth the rewrite.

The temporary `NB_SORT_DIAG` switch in `uniform-particle-slab.wgsl.ts` has been reverted, and that file again matches its state before this campaign.

## Gates

- `npm run check:types`: pass.
- Focused unit tests: 22 passed and 4 skipped (GPU-only) under plain Node. These cover `tests/uniform-particle-*.test.ts`, `gpu-kernel-timer`, `particle-mass-readback` and `uniform-narrow-band-seeding-params`.
- GPU slab and canonical tests in check mode: 2 of 2 passed.
- I made no physics or solver change, so no timing A/B was required.

## Remaining items with superlinear cost or outsized constants

1. `nbSeparation`: per-particle cost grows with neighbour population. This is the top recommendation, for the seep agent.
2. Crowded-cell merge, O(k²/cap) per cell: fine at the current maximum of about 180, bad if cells pile up.
3. Canonical merge in `slabSort`: about 0.9 ms at 1.1M, linear, needed for correct physics.
4. Running one workgroup per brick: no measurable imbalance at the current maximum of about 5k records per brick. Re-check if bricks exceed about 20k.

## Follow-up 1: where the rest of the slope goes

Two regressors were fitted to every one of the 240 kernel-timer labels: live particle count and fine tiles (`uniformMixedFineTiles`). The data were 12 windows of 9 steps on letters uniform+band4, covering 0–1.14M particles and 420–8100 fine tiles (`artifacts/particle-scaling/kern12-lettersu.log`, reduced by `fit2.py`). Particle count and fine tiles are strongly correlated (r = 0.92), which is why a fit on particles alone loads the grid's cost onto particles.

| fit | intercept | per 1k particles | per fine tile |
|---|---|---|---|
| GPU timer total, particles only | 26.5 ms | 43.5 µs | — |
| GPU timer total, two regressors | 13.8 ms | 13.7 µs | 5.9 µs |
| Untimed step wall time, two regressors (`--walls`, 480 steps) | 14.1 ms | 15.1 µs | 6.2 µs |

- The move kernel is almost all of the true per-particle term (14.9 µs/1k, separation included).
- The other ~27 µs/1k from the single-regressor fit is grid work on the fine band. That band's area tracks the particle count, but its cost is per tile (about 5.9 µs per fine tile per step, roughly 92 ns per fine cell).
- The untimed wall-time fit matches the GPU fit, so readbacks and CPU work add no slope. They sit in the ~14 ms intercept.

Top 10 labels by per-tile slope (µs per fine tile):

| label | µs/tile |
|---|---|
| arenaDistance | 0.34 |
| resolve | 0.28 |
| pRelax | 0.26 |
| arenaPublishCanonical | 0.24 |
| remapFaces | 0.20 |
| classify | 0.20 |
| pProjectNative | 0.19 |
| coupleFine | 0.18 |
| pRows | 0.16 |
| geometryRegional | 0.15 |

The cost is spread over about 200 kernels, none above 0.34 µs/tile. By group: pressure-arena solve 0.73, surface canonical metric/publication 0.65, surface advect 0.34, transfer 0.32, census 0.30, phi resolve 0.30. No single grid kernel dominates, so cutting the slope means cutting the fine band's per-tile work broadly, or covering fewer fine tiles for the same water.

## Follow-up 2: why move sits far above its traffic floor

**Counters** (xctrace, dispatches split, letters scene: `artifacts/particle-scaling/xctrace-letters-split`, reduced by `xred.py` and `xocc.py`). One 4.28 ms `advectParticlesDeferred` dispatch was captured:

- Limiters were all low: ALU 12.7%, LLC 12.9%, buffer read 3.1%, MMU 3.9%, read bandwidth 9.8 GB/s.
- Compute occupancy by tenths of the dispatch: 11.7, 12.5, 12.5, 12.5, 10.7, 2.8, 0.9, 0.4, 0.2, 0.1.

So the move is neither traffic-bound nor ALU-bound. It is latency-bound under a hard occupancy cap of 12.5%, followed by a straggler tail lasting about half the dispatch.

**The cap is registers.** Dawn's MSL for the move was dumped (`dump_shaders`) and compiled offline with Metal, with the WGSL size-64 cap removed (`move-nomax.metal`, scratchpad `mslstats`). Its `maxTotalThreadsPerThreadgroup` is 384, and 384 / 3072 threads per core is exactly 12.5%. Leaf functions were then stubbed in the MSL (`stub.py`, `loo.sh`):

| variant | max threads |
|---|---|
| all of `nbAdvect` stubbed | 1024 |
| all leaves stubbed, body kept | 512 |
| only `umSampleVelocity` kept | 384 |
| only `particleDepthIn` kept | 384 |
| only `nbFinish` (solid walk) kept | 384 |
| only `nbRigidContact` kept | 384 |
| only `gridSupported` or `nbParticleBlocked` kept | 448 |
| only `nbSeparation`, `nbMergeRun`, `nbSparseAge` or `nbHullUncut` kept | 512 |

**Consequences:**
- Separation does not set the cap.
- Splitting the trace from the finish cannot lift the trace above 384, because the trace needs the sampler.
- A solid-free twin pipeline would fold the walk away, but e4041e9a removed twin pipelines on purpose, because they compile different arithmetic. Rejected.

**Separation's cost under the cap.** A temporary timing-only pass ran `nbSeparation` alone over every live non-ballistic record (env `NB_SEP_PROBE`, since removed). It took 4.7 ms at 950k particles and 7.5 ms at 1.08M. In those same windows the move took 14.6 ms and 24.6 ms with separation inside it, against about 5 µs/1k without it. Separation's latency-bound neighbour loop therefore costs about twice as much inside the 12.5%-occupancy kernel as it does on its own. That made a separate separation pass the top lever. It was built as the pre-pass described below.

**Chunk size, reverted.** `NB_MOVE_CHUNK` was changed from 256 to 64 to shorten the straggler tail, which is one 64-lane workgroup running a 256-record job as four barrier-separated batches. Results of the 3+3 interleaved A/B (timing mode):

| scene | metric | 256 | 64 | difference vs spread |
|---|---|---|---|---|
| letters uniform+band4, 300 steps | step median (ms) | 72.3 / 70.4 / 73.3 | 70.9 / 70.1 / 69.8 | Δ1.7 vs spread 2.9 |
| letters uniform+band4, 300 steps | p95 (ms) | 84.3 / 94.3 / 87.8 | 80.5 / 82.8 / 81.9 | — |
| seep, 1200 steps | step median (ms) | 34.9 / 34.3 / 34.7 | 34.7 / 32.9 / 33.4 | Δ0.9 vs spread 1.8 |

Neither scene clears 1.66× spread, so the change was reverted. The physics stayed within spread:
- letters drift: −0.110 to −0.117 at 256, −0.109 to −0.111 at 64;
- seep throat Cd 0.813–0.816 in both arms;
- seep right volume 0.568–0.635 at 256, 0.491–0.551 at 64, within 1.66× spread.

Job storage would not have limited it. Jobs are recomputed per index from each entry's first-job word, and the fixed 1024-group grid strides over all of them, so there is no list to overflow.

## Follow-up 3: separation as a pre-pass (kept)

**What changed** (`uniform-narrow-band-flip.ts`). Under `sampleSeparation: "pairs"`, a new launch named `separationPrepass` runs first in the advect batch, ahead of `advectParticlesDeferred` and `advectWalk`.

- **Which records.** It visits the move's own records: the slab entries from state[22] and each entry's first-job word, plus the flat spill lane.
- **Job size.** It uses quarter-jobs of 64 records, so the dense bricks of a slab job spread across four workgroups. With whole 256-record jobs the pass took 8.8 ms at 1.04M particles, because of stragglers.
- **What it computes.** For each live record whose mode is not ballistic, it calls `nbSeparation` (the seep agent's body, unchanged) on the intact current bank, at the record's own position and mass.
- **Where the result goes.** The result is three words at `NB_SEPARATION + 3i`, appended to the `links` buffer after the arena windows. That costs 12 B × capacity, which is +0.09 GiB on the seep (3.40 GiB total), counted in construction memory.
- **How the move uses it.** In `nbAdvect`, `end += sepd` stays where it was: before the finite check, `nbTraceEndpoint`, the reach test, `nbHullInclude` and `nbFinish`. The walk and the blocked/cut-cell tests therefore see the separation as motion. `sepd` comes from the pre-pass words while `sepShared` holds, meaning the sample's q and mass are still the record's own. A crowded-cell merge gatherer, or a record a rigid contact moved, still calls `nbSeparation` inline. The inputs are therefore identical by construction.
- **Probes removed.** The env A/B switches `NB_SEP_PROBE` and `NB_SEP_PREPASS` are both gone.

**Race check.** The in-move `nbSeparation` reads `particles[j]`. The only move path that writes `particles[]` is `nbListWalk`. It is unreachable because `routed` needs `nbFastSample`, which has been `const false` since e4041e9a. `nbListSlow` writes only `links`, and appends and spills go to `source`. So there is no race today. It would become a latent one if the fast or walk paths came back, and the pre-pass removes that hazard by construction.

**Kernel timer** (letters-uniform with band depth 4, one run each; logs `sepprepass-k`, `sepprepass4-k`, `sepprobe-k`, `chunk256-k`):

| particles | in-move separation (move) | pre-pass + move |
|---|---|---|
| ~0.95–0.99M | 14.6 ms (950k), 30.7 ms (993k) | 5.6 + 5.6 = 11.3 ms (978k) |
| ~1.08–1.14M | 24.6 ms (1.08M), 17.3 ms (1.14M) | 6.4 + 6.6 = 13.0 ms (1.11M) |

**3+3 interleaved timing A/B** (timing mode; `ab-sep.sh`, `ab-sep.out`, `ab-{lu,seep}-sep{off,on}-r{1,2,3}`; same `nbSeparation` body hash in all 12 runs):

| scene | metric | off (in-move) | on (pre-pass) | Δ vs 1.66× spread |
|---|---|---|---|---|
| letters-uniform, 300 steps | step median | 73.5 / 67.5 / 73.0 | 67.2 / 67.9 / 64.5 | 5.8 vs 10.0 (not clear) |
| letters-uniform, 300 steps | p95 | 91.5 / 94.0 / 87.5 | 76.9 / 78.6 / 76.0 | 14.6 vs 10.8 (**clear**) |
| letters-uniform, steps 150–300 | mean wall | 80.8 / 79.6 / 78.9 | 70.2 / 71.8 / 70.1 | 9.1 vs 3.2 (**clear**) |
| seep, 1200 steps | step median | 34.4 / 34.4 / 33.3 | 33.9 / 32.8 / 32.8 | 1.6 vs 1.8 (not clear, no loss) |
| seep, steps 600–1200 | mean wall | 37.4 / 35.8 / 35.5 | 37.9 / 34.1 / 34.5 | within spread |

The whole-run letters median is diluted by the early low-count steps. Where the count is high, the gain is about 9 ms a step, roughly 12%. The seep, at about 210k particles, gains about 1–1.5 ms, inside its spread.

**Physics and gates:**
- **Seep:** throat Cd 0.813–0.816 in both arms. Right volume 0.621 / 0.545 / 0.518 (off) against 0.396 / 0.615 / 0.511 (on); the means, 0.561 and 0.507, are within 1.66× spread. Drift −0.0027 to −0.0049 in both arms.
- **Letters drift:** −0.108 to −0.116 in both arms.
- **Closed rest** (`probe-reservoir-bulk --closed`, step 120): identical digits in both arms (`rest-sep{0,1}.log`).
- **Symmetry** (`uniform-nbflip-symmetry`): fails identically in both arms, with "run 0, step 6: reflected surface error 0.0002026297152042389 m". That predates this change. A clean tree of HEAD af0e5dea fails the same way. A commit bisect (`artifacts/particle-scaling/sym-bisect/`) traces the step-6 failure to 049221ca's 4h corner curve term in `uniform-mixed-dynamic.ts`. The term itself is mirror-symmetric, and the stage probe shows the layout stays symmetric. Beneath that sits an older step-21 failure from 24d05d06 (the clipped-wall surface carry). Not fixed; see the open item below.
- **Long-slide residency** (3 runs per arm): "empty bottom-left page must retire" passed 2/3 with the pre-pass and 1/3 without, the same flake in both arms (`gates-sep.log`, `gates-sep-off.log`).
- **Checks:** `npm run check:types` is clean; focused CPU tests 24 passed, 4 GPU tests skipped; focused GPU tests 18/18; `npm run test:unit` passes 1253 with 0 failures.

**Remaining levers:**
1. **Register cap.** The move is still capped at 384 threads (12.5% occupancy) by the sampler, `particleDepthIn`, `nbFinish` and `nbRigidContact`. Slimming their live state is the next move-side lever.
2. **Fine-band per-tile cost.** About 5.9 µs per fine tile, spread over about 200 grid kernels; see the top 10 in Follow-up 1.
3. **Separation body cost.** About 6 µs/1k in the pre-pass. The body belongs to the seep agent.

## Open item: NB-FLIP z-mirror symmetry (not fixed)

`tests/uniform-nbflip-symmetry.test.ts` fails on main, with or without this campaign's changes: "run 0, step 6: reflected surface error 2.026e-4 m". The scene is `sparse-cm12-ladder-symmetric-2d`: a 32×16×16 tank with an 8×8×16 dam and `depthBoundary: "symmetry"`.

**Bisect** (`artifacts/particle-scaling/sym-bisect/`; `sym-commit.sh COMMIT`, `sym-variant.sh NAME COMMIT FILE SED`, `sym-bisect-loop.sh`):
- 853470a0 is good.
- 24d05d06 (clipped-wall surface carry) is the first bad commit, failing at step 21 (3.97e-3 m).
- 049221ca (the 4h corner curve term in `uniform-mixed-dynamic.ts`) moves the failure to step 5/6 (2.03e-4 m).
- Knock-outs at 049221ca: removing the 4h `coarseCurve` term restores the step-21 signature. Removing the h curve, the moving-shape gate or the z Laplacian, making the Laplacian commutative, or excluding the z=0/16 corners all leave it unchanged.

The term itself is mirror-symmetric. It changes which tiles go to h, and the bug sits downstream of that.

**Stage dump** (`tools/probe-nbflip-symmetry.ts --capture-extended=true`, at 049221ca; scratchpad tree `sym-probe/`):
- **Layout.** It stays mirror-symmetric throughout. h tiles first appear at step 4 (x tiles 0–3, y tiles 0–2, every z).
- **The face (7,4,0) reading was an artefact.** The "root-pressure" capture of face (7,4,0) showed 0 against −0.357 at its mirror. That face lies in a projected-band h tile, which `toSimulationFine` skips (`umSkipProjectedBand`), so neither value is written at that stage. The capture shows stale p2g and scratch contents in a tile-local pattern. The root transfer is not the source.
- **Band pressure.** Velocity asymmetry at step 4 is 7.2e-5 and spread over the whole domain. Strict band exit with 8 cycles cuts it to 1.4e-5, but the step-5 phi asymmetry stays at 7.2e-4. So the band-solve residual is not the driver either.
- **Velocity extension into air.** This is where the asymmetry appears: it is 3.3e-7 at step 4 and 0.137 at step 5. Example: u at (0..3, 10..11, z=0) = 0.078, against 0.215 at the mirror z=15, with zero band input at both. It is concentrated at z=0/1 against z=15/14 in h tiles near the z walls. Phi advection then carries it into the surface, with the largest error at (8,12,0).
- **Ruled out.** `--wall-contact=off` and `--exact-surface=true` do not change the step-5 phi asymmetry. Neither does bounding `ueRequestInside` to the box walls, which tested the out-of-box z=16..31 cells being "present" in the 32-cell page while z=−1 is not (extended velocity was still 0.137).

**Next suspect.** The extension's neighbour and width rules in `umNeighbor` (`uniform-mixed-extension.ts`): the low-side-only `max(below, wall)` clamp and the `lowWidth`/`highWidth` treatment at h tiles next to the z walls. To reproduce, run `node --import tsx tools/probe-nbflip-symmetry.ts --steps=6 --runs=1 --capture-extended=true` and compare `5-extra-extended-velocity` near z=0 against z=15.
