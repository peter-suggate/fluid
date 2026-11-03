# Pressure stage progress (paddleboats small-scene step)

## 2026-11-03 — P2 projection by owned root, launch shapes from band counts, correction-mode rebase

### Changes
- **P2 `pProjectNative`.** The kernel now runs one job per band tile:
  - 42 lanes run the 6 faces × 7 ownership tests into workgroup atomics;
  - lane 0 lists the owned roots;
  - each root's 96 face-weight and value reads are spread across the workgroup. Before, one lane walked the stencil serially.
  
  The weighted sum keeps the original sequential order, so the output is bit-identical. The old kernel stays as `pProjectNativeRef`, built only when `UniformPressureArena.reference` is set.
- **P5 launch shapes.** `observeWork` clamps the lagged brick launch to `min(bricks, tiles)` and the level-1 launch to `min(level1, bricks)`. pDownA and pUpA had launched 1,024×512 lanes for about 140 bricks. The kernels grid-stride the GPU counts, so any width is correct.
- **P3 (part).** In correction mode, `pRebase` skips the seam-head and coupling `w`/`d` loads and the residual measure. Their outputs are unused there, so the result is unchanged.

### Verification
- `tools/pressure-arena-equivalence.ts --scene=paddleboats --bodies --frames=20,45,70`:
  - poisons the output texels, then runs the reference kernel and the rewrite on the same state;
  - result: 0 differing words of 167,984 / 203,984 / 246,992 written;
  - it uses error scopes and counts words written, so an invalid command buffer cannot pass.
- `npm run check:types` is clean. `npm run test:unit`: 1806 tests, 0 failures.

### Kernel timer (paddleboats, probe-reservoir-bulk, 60 kernel frames, ms per step)

| kernel | before | after |
|---|---|---|
| pProjectNative | 1.114 | 0.205 |
| pDownA | 0.476 | 0.279 |
| pUpA | 0.262 | 0.173 |
| pRebase | 0.677 | 0.531 |
| pressure-family total (includes the in-flight authority work) | 14.56 | 12.75 |

### Gate (one run each, snapshot of HEAD 294ed5cc plus these files)

| scene | drift | step late median ms | residual | cycles |
|---|---|---|---|---|
| seep 300 inviscid pipelined | +0.000370 | 31.6 | 0.0154 | 3 |
| letters 120 | −0.01120 | 104.1 | 0.0440 | 4 |
| paddleboats 90 | −0.00669 | 52.8 (timing run; the kernel table above is the A/B) | 0.0434 | 4 |

The bodies gate measured seep +0.000315 at 31.3 ms and letters −0.0113 at 104.4 ms, so these runs are within run-to-run noise.

### Measured and dropped (each worth under 0.3 ms on paddleboats)
- Folding Krylov `kReduce`/`kBound` into the apply kernels.
- A per-solve seam-head plane. Only the correction-mode skip above landed.
- Fusing the 16h and 32h tail levels. Their cost is dependent latency, not launches.
- Zero-workgroup launches for converged cycles.
- `tailCycles=1` cut the tail from about 1.5 to 0.48 ms, but the step-90 residual rose from 0.033 to 0.153. Rejected: a weaker tail is not a convergence-neutral change.

## 2026-11-03: P7 body head as a list, and P6 dropped (sub-implementer)

Landed in 31d3607b. The simulation authority's bodies-only head (`encodeBodies`) used to scan each held region as eight 8-tile items, repeating the 27-neighbour reach scan eight times and running each item's h tiles in series. Now:

- `bodyRegions` takes one region per workgroup, one lane per tile.
- It surfaces reached 4h owners in place and appends reached h tiles to one list.
- `bodyFine` runs exactly the listed tiles through an indirect launch. The args are copied between two passes, and the launch has zero workgroups when nothing is reached.
- Root topology, phi and balance are untouched. Checked by a GPU probe on paddleboats at frames 20, 40, 60 and 80: the owner set is identical to the old scan (missed 0, extra 0), and every written word equals `encodeAll`.
- Regression: `tests/uniform-authority-body-list.test.ts`, a randomized mirror of the old and new selection.

Paddleboats kernel ms (90 steps, kernels=60) against `pr/base.json`:

| Kernel | Before (ms) | After (ms) |
|---|---|---|
| surfaceRegions | 0.531 | — |
| bodyRegions | — | 0.099 |
| bodyFine | — | 0.063 |
| bodyArgs | — | 0.006 |
| Total | 0.531 | 0.168 |

Gate. The tree carries other lanes' uncommitted edits. check:types is clean and test:unit is 1540 pass / 0 fail.

| Run | Drift | lateHalfMedian (ms) |
|---|---|---|
| seep 300 | −3.5e-5 | 31.7 |
| letters 120 | −0.0111 | 93.6 |
| paddleboats 90 | — | 47.8 (median 44.6) |

Not done, with measurements (stub A/B, paddleboats):

- **The rest of the P7 list idea.** It is not worth it on this scene. With held = 0 (every quiet region borders a non-quiet one), changed ≈ 877 of 1536 regions and nearChanged ≈ 1535.
  - surfaceHeld's scan overhead is only about 0.06 ms of its 0.55 ms: every owner genuinely reruns.
  - wetRegions (0.40 ms) and cutRegions (0.27 ms) do genuine work for every near-changed region.
  - A change list cannot shrink any of these three. The real lever is the relayout-head surfaceHeld pass, which is mostly redundant. Removing it needs an "unchanged revisions" held criterion: a correctness question, so it is left to the owner.
- **P6 is dropped because it cannot be bit-identical.**
  - project, buildRhs and toPressure write every resident 4h owner, and skipping air owners leaves `velocityScratch` holding forced velocities where project wrote zeros.
  - The air-owner share of project is about 0.18 ms (0.78 → 0.60 when stubbed). buildRhs, toPressureFine and toPressureCoarse are 0.26, 0.18 and 0.09 ms, mostly liquid work.
  - roKeys (0.10 ms) is per key, not per liquid owner.
- **Memory accounting.** The simulation authority's `allocatedBytes` (stamps and the new list, about 0.1 MB) is not summed in `uniform-mixed-frame.ts:473`, which counts only `split.authority`. This gap predates the change.

## 2026-11-03: loose ends, and P9 diagnosed (cycles), no fix landed

**pTailV (the fused page tail)** is removed and was never committed. Fused-tail timings were 1.06, 0.99 and 0.71 ms against the base tail's 1.46 ms, none under the 0.6 ms bar. Letters' residual was also worse: geo-mean final 0.053 against 0.042, worst 1.73 against 0.47.

**Authority body list (31d3607b).** Independently checked:

- `pr/exactA.mts` on paddleboats, new path only, at frames 20, 45 and 70: written 130986, 155052 and 165825 words, with diff 0 and valueVsAll 0 (every written word equals `encodeAll`).
- Back-to-back kernel A/B, 90 steps, kernels=60: authority total 1.888 → 1.524 ms (−0.36 ms). surfaceRegions 0.539 is replaced by bodyRegions 0.082, bodyFine 0.070, bodyReset 0.006 and bodyArgs 0.002.

**P9 diagnosis.** The new tool `tools/probe-pressure-classes.ts` reads the arena's per-class header diagnostics (`UniformPressureArena.diagnostics`, off and empty by default). Paddleboats, frames 31–90:

- Baseline: 3.07–3.13 cycles, geo-mean final 0.016–0.027 (it varies with the tree).
- The residual falls about 10× per V-cycle: log10 max 1.2 → 0.15 → −0.8 → −1.7.
- **The worst rows are band rows with no surface, seam or wall faces.** In 40–60 of 60 frames the largest residual is a deep-interior band row, one whose six faces are all unit-coefficient liquid links (no θ, seam or wall):

  | Cycle | Deep interior: log10 max / rows past target | Free surface (θ < 1 or ghost face): log10 max / rows |
  |---|---|---|
  | 0 | 1.21 / 14.5k | 1.10 / 4.3k |
  | 1 | 0.16 / 6.6k | 0.00 / 0.9k |
  | 2 | −0.80 / 1.1k | −0.95 / 100 |
  | 3 | −1.58 / 27 | −1.74 / 1 |

  Seam rows sit roughly 0.5 log10 below the deep-interior rows and walls about 1 below. Native rows sit 2–3 below.
- **Ruled out: smoothing.** Contraction is the same about 10× per cycle at sweeps 3, 4 and 5. sweeps 4 gives 2.85 cycles and sweeps 5 gives 2.70, but their geo-mean finals are worse (0.0194 and 0.0234 against 0.018), so they fail the equal-or-better rule and were not timed. coarsestSweeps 12 changes nothing (3.08), so the level-2 boxes are solved.
- **The factor moves with coarseScale:**

  | coarseScale | Cycles | Geo-mean final | Worst |
  |---|---|---|---|
  | 0.4 | 2.88 | 0.0147 | 0.054 |
  | 0.5 | 3.07 | 0.018 | — |
  | 0.6 | 3.43 | 0.027 | 0.44 |

- **Ruled out: the scale itself.** 0.5 is the consistent value. Sum restriction R needs 8× the 2h FV operator, which is 2/h² per link, against 4/h² for the 4 aggregated fine links. 0.5 is also the 7-point lumped Galerkin of R A P with the trilinear P.
- **Cause: the 7-point lumping.** The true Galerkin operator is a tensor 27-point stencil, (−½, 1, −½)/h² along an axis times M = (¼, 3/2, ¼) across each transverse axis. Lumping M to its row sum makes the coarse operator up to 4× too stiff on transverse-oscillating coarse modes. Those modes are under-corrected, so a smaller scale helps.
- Because of that, 0.4 is a fitted number, not a derivation, and is not landed.

**Fix needed: a Galerkin (27-point) coarse operator at levels 1 and 2.** Stored couplings, or the analytic tensor form for all-unit aggregates. It touches:

- the level-1/2 bake;
- the box smoother (27-point reads, colouring);
- the coarse residual and restriction;
- the Krylov apply;
- the level-2 composite with the 7-point native 4h rows, which then need a consistent 27-point form.

It is a hierarchy redesign, not started here. The expected gain, from the coarseScale response, is about 0.2–0.4 cycles (0.4–0.8 ms). Reaching 2 cycles needs about 0.03 per cycle, so it may also need a better initial guess.

## P10: Galerkin and box-residual prototypes, 2026-11-03 (nothing landed)

**Goal.** Cut V-cycles from 3 to 2 on paddleboats and from 4 on letters, with the final residual equal or better.

**CPU prototype.** CSR multigrid with the same hierarchy and transfers: sum R, trilinear renormalised P, flexible GCR(1), exact coarsest solve. Pool and hull scenes. Per-cycle contraction, geo-mean:

| Coarse operator | Per-cycle contraction | Decades per cycle | Verdict |
|---|---|---|---|
| Lumped 7-point (current) | ~15× | 1.18 | — |
| Galerkin R·A·P, 27-point | 22–26× | 1.38 | — |
| Full P^T·A·P, ~125-point | 42–45× | 1.64 | — |
| Energy-minimising step length | no gain | — | rejected |
| W-cycle, lumped | ~27× | — | 2× the coarse work, rejected |
| RAP on level 1, lumped below ("face7") | 6.5× | — | worse, rejected |

**Emulating the GPU's overlapping 12^3 box smoothing at levels 1 and 2.**

| Variant | Contraction |
|---|---|
| Lumped with boxes | 10× (matches the GPU's ~10× per cycle) |
| Boxes off at both levels | 16× |
| Boxes off at level 2 only | 11.3× |
| Level-1 residual taken from neighbours' kept cores instead of the box view ("l1global") | 13–14× |
| RAP with boxes and l1global | ~15× |

What this shows:
- The box at level 1 is what holds the GPU's contraction down. The defect is that its residual reads the box's own view of the neighbour cells, which no up leg ever sees.
- With boxes in place, RAP is not worth a hierarchy redesign (15× against 13–14×).
- A more exact tail changes nothing.

**GPU probes on a clean export.** `probe-pressure-classes`, frames 31–90:

| Variant | Paddleboats cycles | Geo final | Worst | Letters cycles | Geo final | Worst |
|---|---|---|---|---|---|---|
| Base | 3.217 | 0.0248 | 0.187 | 3.817 | 0.0368 | 0.101 |
| tailCycles 5 | 3.35 | 0.037 | 1.00 | | | |
| tailCycles 9 | 3.233 | 0.034 | 1.58 | | | |
| tailScale 1 | 3.15 | 0.0278 | 0.226 | | | |
| tailScale 1, tailCycles 9 | 3.167 | 0.0276 | 0.131 | | | |
| Scale 0.4 at level 1 | 2.983 | 0.0242 | 0.405 | | | |
| Scale 0.4 at level 2 | 3.15 | 0.0202 | 0.190 | | | |
| Scale 0.4 at level 3 | 3.133 | 0.0255 | 0.315 | | | |
| x1: level-1 residual from neighbour cores (pDownR) | 3.067 | 0.0202 | 0.070 | 3.783 | 0.0388 | 1.60 |

- **x1 (pDownR)** costs one extra dispatch, about 0.04 ms per cycle.
  - On paddleboats it saves 0.15 cycles, roughly 0.15 ms net. That is under the 0.3 ms bar.
  - It fails the equal-or-better rule on letters (worst 0.10 → 1.6).
  - Reverted.
- **The scale changes** are fitted constants, so they are not landable.
- **Reaching 2 cycles** needs about 1.25 decades per cycle where the GPU now gets about 0.95. Only the full P^T·A·P coarse operator reached that on the CPU, and that would be a hierarchy redesign. The CPU also suggests it is mostly lost again under box smoothing unless the level-1 smoother becomes global.
- **Prototypes:** `scratchpad/pl-proto/{mg.ts,gpu.ts}`. They are not checked in.

## P11: pricing a global smoother with PtAP at levels 1–3, 2026-11-03 (no-go, not built)

**CPU prototype with GPU-realistic coarse smoothers.** These are colour-parallel, so each colour updates in parallel as it would on the GPU. GCR, frames past the first, geo-mean per-cycle contraction:

| Coarse operator and smoother | Pool | Hull |
|---|---|---|
| Lumped, 2-colour or 8-colour | 10–13× | — |
| Lumped, 8-colour inside 8^3 bricks with Jacobi across bricks ("brick8": one dispatch a sweep) | ~9× | ~7× |
| PtAP (70 nnz at level 1, 90 at level 2), 2-colour red-black | diverges (same-colour couplings) | — |
| PtAP, parameter-free L1-Jacobi | ~15× | — |
| PtAP, 8-colour | ~38× | ~30× |
| PtAP, 27-colour | ~41× | ~33× |
| PtAP, brick8 | ~33× (1.52 decades) | ~27× (1.43) |
| PtAP at levels 1–2 only, lumped below, brick8 | ~28× | ~25× |
| PtAP at level 1 only, brick8 | ~13× | ~11× |

PtAP is needed through level 3 (8h, inside the page tail), with a multicolour smoother.

**What 2 cycles requires.** The history must reach the target (log10 about −1.3) after 2 cycles:

| Scene | Start (log10) | Decades per cycle needed | Contraction needed |
|---|---|---|---|
| Paddleboats | 1.24 | 1.27 | ≥19× |
| Letters | 1.66 | 1.48 | ≥30× |

On the CPU, PtAP covers paddleboats with little margin and letters only at the pool scene's best.

**The CPU model overpredicts GPU gains.** The model matches the GPU baseline (~10×). But:
- it predicted x1 at +12% log-contraction, and the GPU gave +4%;
- it predicted scale 0.4 to be catastrophic, and the GPU improved.

The real hierarchy (native 4h rows at level 2, seams, page tail) is not in the model.

**Cost:**
- **Bake.** θ moves every step across the whole band, and the band is the surface region. So nearly every level-1 row's PtAP support changes every step, and a per-layout cache does not hold.
  - The per-step bake gathers about 64 fine rows × about 56 AP entries per coarse row: about 72M FMAs at level 1, plus levels 2 and 3. Today's 7-point pRows alone is 0.35–0.6 ms.
  - Estimated bake: 0.4–0.8 ms per step.
- **Per cycle.** Level 1 and level 2 become global brick8 sweeps: about 9 dispatches each, at 20–40 µs latency each. They read about 10× the coefficients of today's two box kernels. Level 3 PtAP sits in the tail.
  - Estimated extra: +0.5–0.9 ms per cycle, so 2.4–2.8 ms against today's 1.9.

**Projection for paddleboats** (today 3.2 × 1.9 = 6.1 ms):

| Cycles | Step cost | Net |
|---|---|---|
| Exactly 2 | 5.2–6.4 ms | −0.9 to +0.3 ms |
| 2.5 (if half the CPU gain transfers, as x1 did) | 6.4–7.8 ms | worse than today |

**Verdict:** under the 1 ms bar even in the best case. Not built.

The structural fact: once θ is in the band, a Galerkin hierarchy pays its operator every step, and level-1 global sweeps cost dispatches. The per-cycle work has to come down, not the cycle count.

## P12: cheaper cycles, 2026-11-03 (pDownR landed; tailCycles 1 held; launch merges priced, not built)

**Where the solve's time goes (paddleboats).** GPUPassProfile gives the solve pass 5.8–6.0 ms of GPU time at 3.1–3.2 cycles, over 171 dispatches. All passes come to about 41 ms of a ~45 ms step, so the step is GPU-bound. The kernel timer gives each dispatch its own pass, which inflates short kernels, so its numbers are for attribution only.

| Part, per step | Kernel timer ms |
|---|---|
| Level 0 smoothing (pRelax) | 0.93 |
| Native rows' smoothing (pNativeRelax; nativeSweeps is 1 in composite mode) | 0.58 |
| Tail ladder (pDown3, pDown4, pTail, pUp4, pUp3; 3 tail cycles, 15 dispatches a cycle) | 1.51 |
| Levels 1–2 transfers (pDownA/B, pDown2, pUpA/B, pRestrict0, pResidual2, pProlong0, pNativeOperator) | 1.58 |
| Krylov GCR(1) (kStart, kApplyT, kApplyNB, kReduce, kBound, kStep) | 0.91 |
| pRebase (initial and correction) | 0.43 |

The pressure authority's `encode` (1.30 ms, 16 dispatches) is work-bound: surfaceHeld 0.47, wetRegions 0.41, cutRegions 0.24 ms. It is not dispatch-bound.

**pDownR (landed).** Level 2's RHS now sums the residual of level 1's cores, each read against its neighbours' kept cores (P10's x1), and no longer uses the box's own view. That costs one extra level-1 dispatch per cycle.
- In P10 I rejected x1 on letters' worst final. Worst final is not a usable statistic: across repeat runs on HEAD it swung from 0.12 to 2.7. The geometric mean is stable to about 5%.
- `probe-pressure-classes`, frames 31–90:

  | Run | Paddleboats cycles | Paddleboats geo final | Letters cycles | Letters geo final |
  |---|---|---|---|---|
  | HEAD (three runs) | 3.217 | 0.0267–0.0294 | 3.77–3.80 | 0.039–0.042 |
  | pDownR | 3.133 | 0.0187 | 3.80 | 0.0340 |

- The regression test `tests/pressure-residual-history.test.ts` runs 90 frames of paddleboats (moored, motors driven) and letters through `createUniformSolver`, and covers frames 31–90:

  | Tree | Paddleboats cycles | Paddleboats geo | Letters cycles | Letters geo |
  |---|---|---|---|---|
  | HEAD | 3.15 | 0.0286 | 4.00 | 0.1035 |
  | pDownR | 3.05 | 0.0219 | 4.00 | 0.0840 |

  Its bounds sit between the two: paddleboats at most 3.1 cycles and 0.025, letters at most 0.093. The box view fails the test and pDownR passes it.
- Whole-pass profile, solve pass:

  | Tree | Run | ms | Cycles |
  |---|---|---|---|
  | HEAD | A3c | 5.820 | 3.133 |
  | HEAD | E0 | 5.840 | 3.133 |
  | HEAD | D0 | 6.022 | 3.200 |
  | pDownR | X3 | 5.829 | 3.100 |
  | pDownR | X3b | 5.808 | 3.083 |

  175 dispatches against 171. The time is neutral: the extra dispatch is paid for by the cycles saved. The win is accuracy at no cost.

**Gate for pDownR** (one run each, clean export of df4934c5 plus pDownR and the test):

| Scene | Drift | Step median / late median ms | Final residual | Cycles |
|---|---|---|---|---|
| seep 300, inviscid, pipelined | +0.00011 | 30.0 / 31.1 | 0.0101 | 3 |
| letters 120 | −0.01116 | 90.6 / 99.3 | 0.0446 | 4 |
| paddleboats 90 | −0.00669 | 45.4 / 49.2 | 0.0372 | 3 |

Drift matches the earlier gates: seep +0.0003, letters −0.0112, paddleboats −0.0067. `npm run check:types` and `npm run test:unit` pass (1585 pass, 0 fail). The residual-history test passes on the export: paddleboats 3.05 cycles, geo 0.0220; letters geo 0.0816.

**tailCycles 1 (held, not landed; the decision is the user's).**
- Solve pass: 5.01 ms against 5.91 ms with 3 tail cycles, a saving of 0.9 ms of GPU time per step on paddleboats.
- Convergence, frames 31–90:

  | Tree | Paddleboats cycles | Paddleboats geo | Letters geo |
  |---|---|---|---|
  | HEAD, tail cycles 3 | 3.217 | ~0.028 | 0.039–0.042 |
  | HEAD, tail cycles 1 | 3.217 | 0.0279 | 0.052–0.055 |
  | pDownR, tail cycles 3 | 3.133 | 0.0187 | 0.0340 |
  | pDownR, tail cycles 1 | 3.150 | 0.0231 | 0.0479 |

- Paddleboats holds its cycles. Letters, which runs to the 4-cycle cap, ends about 25–40% looser in geometric mean, so it fails equal-or-better on letters.
- Interleaved step A/B (2 pairs, tail cycles 3 against 1): medians 43.6/50.1 against 50.2/48.9 ms. That is ±3 ms of noise, so it cannot resolve 0.9 ms.
- tailCycles must stay odd: the arena throws at 2, because level 2's up leg reads the odd cycles' channel.

**Krylov off** (`krylov:false`, pDownR tree): paddleboats 3.43 cycles, geo 0.0355; letters geo 0.208. Off, letters ends 6× looser in geometric mean and paddleboats needs 0.3 more cycles. The kernel timer puts GCR(1) at 0.91 ms, but that is inflated and its real cost was not separated here. It stays on accuracy alone.

**Exact launch merges (priced, not built).**
- Independent pairs, by data flow:
  - pResidual2 ‖ pRestrict0. Both read the post-smoothed iterate and write disjoint words. pRestrict0 flips only P_CORRECT, and pResidual2 reads only face kinds.
  - pRebaseC ‖ pNativeOperator. Both read both correction sets and write PA_D/PA_K and PB_K/PB_D respectively.
  - kApplyT ‖ kApplyNB. These are separate dot segments.
  - Everything else in a cycle is a dependent chain.
- Each merge needs a workgroup-size reconciliation:
  - 64-lane tiles inside 512-lane workgroups, as eight virtual 64-lane groups, keeping each tile's reduction tree.
  - For the Krylov pair, the tile stride and partial index must equal today's `tileGroups`. That value is per-frame, so it has to come in by buffer.
  - pNativeOperator is in the `native` bind layout, but pRebaseC is in `solver`.
- The estimate of ~0.4 ms assumed about 25 µs per removed launch. I measured instead:
  - **Empty launches.** Ten empty tile dispatches added per cycle (40 a step) cost 0.28 and 0.37 ms of solve-pass time. After the cycle-count difference that is about **4–7 µs per launch**, so the removed launches are worth about 11 × 5 µs ≈ 0.05 ms.
  - **Ceiling from overlap.** Dispatching pRestrict0, pNativeOperator and kApplyT twice each cost +0.36 to +0.55 ms (D1/D1b 6.385/6.376 ms against D0/E0 6.022/5.840 ms). So the three shorter kernels' serial time is about 0.45 ms. Merging can save at most that, and only if the shorter kernel of each pair overlaps fully.
  - With up to 4096 tile workgroups in several waves, the overlap is mostly the head and tail of a wave.
  - **Projection:** 0.05 ms + a fraction of 0.45 ms, about 0.15–0.25 ms. That is under the 0.3 ms bar. Not built.

**Dispatches and GPU time per pass, whole step** (paddleboats, HEAD df4934c5, GPUPassProfile mean over frames 31–90, run A3c, 3.13 cycles):

| Pass | Dispatches per step | GPU ms per step |
|---|---|---|
| Uniform pressure arena solve | 171 | 5.820 |
| Narrow-band FLIP advectParticles | 3 | 3.607 |
| Uniform mixed extension | 40 | 3.377 |
| Uniform mixed surface advect | 9 | 1.930 |
| Uniform dynamic ownership census | 11 | 1.923 |
| Narrow-band FLIP surface canonical metric and publication | 3 | 1.680 |
| Uniform mixed surface advect deferred | 6 | 1.493 |
| NB-FLIP surface pressure authority | 16 | 1.300 |
| Narrow-band FLIP surface reconstruction | 7 | 1.293 |
| Narrow-band FLIP transfer | 7 | 1.257 |
| Uniform mixed geometric fill | 11 | 1.168 |
| Uniform mixed all-4h solid record | 5 | 1.145 |
| Uniform mixed remap | 9 | 0.997 |
| Uniform pressure arena marks and rows | 8 | 0.980 |
| Uniform mixed momentum | 4 | 0.956 |
| Uniform mixed pressure project | 1 | 0.850 |
| Uniform local speed certificate | 8 | 0.772 |
| Uniform mixed body forces | 4 | 0.755 |
| Uniform mixed phi resolve | 5 | 0.675 |
| Uniform pressure arena projection | 5 | 0.609 |
| Narrow-band FLIP surface arena search | 7 | 0.607 |
| Narrow-band FLIP particle sort | 25 | 0.519 |
| Uniform mixed hanging fine taps | 3 | 0.451 |
| Narrow-band FLIP regional air update | 2 | 0.442 |
| Uniform mixed momentum tiers | 4 | 0.437 |
| Narrow-band FLIP seed | 3 | 0.427 |
| Uniform mixed layout build | 11 | 0.374 |
| Uniform shared frame plan | 8 | 0.345 |
| Narrow-band FLIP coarse retirement | 3 | 0.304 |
| Uniform mixed 4h sampling cache | 1 | 0.292 |
| Uniform mixed pressure rhs | 1 | 0.252 |
| Uniform dynamic ownership residency | 2 | 0.245 |
| Uniform mixed transfer to pressure | 2 | 0.223 |
| Uniform bulk tier selection | 2 | 0.220 |
| Narrow-band FLIP snapshot | 1 | 0.199 |
| BULK16 reclaim quarantine | 1 | 0.189 |
| Uniform mixed extension regions | 4 | 0.188 |
| Uniform mixed rigid coupling | 1 | 0.177 |
| Narrow-band FLIP surface canonical view | 1 | 0.151 |
| Uniform mixed transfer to simulation | 2 | 0.135 |
| Uniform mixed extension retain | 2 | 0.118 |
| Narrow-band FLIP surface canonical far metric | 1 | 0.117 |
| Uniform page pool commit | 8 | 0.116 |
| Uniform region work heat | 6 | 0.115 |
| Uniform live block copy | 7 | 0.115 |
| Uniform pressure root owners | 1 | 0.104 |
| Narrow-band FLIP activity | 2 | 0.081 |
| Uniform detail ring enter | 3 | 0.081 |
| Uniform mixed remap publish | 2 | 0.079 |
| Uniform block neighbours | 4 | 0.078 |
| GPU resident rigid-body integration and contacts | 1 | 0.063 |
| Uniform mixed global surface volume | 4 | 0.059 |
| Narrow-band FLIP particle slabs | 3 | 0.059 |
| Fluid particle display energy | 1 | 0.059 |
| Uniform region dynamics classify | 2 | 0.054 |
| unlabelled | 1 | 0.054 |
| Uniform mixed rigid body tiles | 1 | 0.052 |
| Uniform region dynamics publish | 3 | 0.047 |
| Uniform BULK16 page transfer | 8 | 0.046 |
| Narrow-band FLIP crossing search | 2 | 0.045 |
| Uniform mixed solid record of claimed blocks | 4 | 0.043 |
| Narrow-band FLIP particle placement | 3 | 0.039 |
| Uniform cold freeze | 2 | 0.038 |
| Uniform shared regional frame plan | 2 | 0.038 |
| Uniform mixed phi twins | 2 | 0.031 |
| Uniform detail h slot init | 3 | 0.028 |
| Uniform mixed remap adopt | 1 | 0.025 |
| Uniform mixed pressure bulk faces | 1 | 0.024 |
| Uniform16h remap producer revisions | 1 | 0.019 |
| Uniform page initialise | 4 | 0.017 |
| Uniform detail h slot claim | 2 | 0.015 |
| Uniform pressure arena verdict | 1 | 0.014 |
| BULK16 publish live slots | 1 | 0.012 |
| Uniform pressure arena unmark | 1 | 0.010 |
| Narrow-band FLIP page words | 1 | 0.008 |
| Uniform mixed solid simulation widths | 1 | 0.005 |
| Uniform dynamic ownership page words | 1 | 0.004 |
| Uniform 16h work evidence page claims | 1 | 0.002 |
| Uniform page tile words | 1 | 0.001 |
| Uniform detail h slot release | 1 | 0.001 |
| **All 80 passes** | **513** | **40.68** |
