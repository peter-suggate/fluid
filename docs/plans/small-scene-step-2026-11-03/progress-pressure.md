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
