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
