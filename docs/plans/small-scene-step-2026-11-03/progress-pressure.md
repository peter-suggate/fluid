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
