# Small-scene step: status (2026-11-03)

## Milestone 1, 9edb739d against plan commit 98ce7bae

Paddleboats `--bodies`, 300 steps, `fluid-gpu-run.sh timing`, clean `git archive` exports, one run each:

| | 98ce7bae | 9edb739d |
|---|---|---|
| median step ms | 60.9 | 53.8 |
| late-half median ms | 63.2 | 55.7 |
| h tiles (step 300) | 4975 | 3494 |
| pressure band tiles | 2771 | 1753 |
| live particles | 353k | 320k |
| drift | −1.24% | −1.10% |

## What landed

| commit | change | kernel ms |
|---|---|---|
| 294ed5cc | couple over each tile's candidate bodies, workgroup reduction | couple 2.13 → 0.63 |
| af94e6fa | per-tile body masks, bodyMask lane per tile, certified dirty tiles | bodyMask 0.40 → 0.07, pRows 1.51 → 1.03, coupleFine 1.18 → 0.90, buildRegions 1.45 → 1.22 |
| dfd7ac1d | pProjectNative per band tile; launches sized from band counts | pProjectNative 1.11 → 0.21, pDownA 0.48 → 0.28, pUpA 0.26 → 0.17, pRebase 0.68 → 0.53 |
| 31d3607b | body head list for the surface pressure authority | 1.89 → 1.52 |
| 56b738d7 | solid promotion ring only where the solid supports it | h tiles 4176 → 3130 (90 steps) |

## Measured and rejected (see progress-*.md)

- Particle move staged in workgroup memory (M1): slower. The lookups were already cache-resident.
- Far-tile certificate in advect (levelset P1): 0.17 ms net. Existing exits already skip most far vertices.
- Arena workgroup-memory halving and fusion: slower.
- Fused coarse pressure tail: over budget, and letters converge worse.
- Census atomic reductions and classify replay: no gain, and the replay premise fails.
- The extension sweeps already share one geometry pass and already run only over held blocks.
- Layout churn is real need (90 of 90 frames have tiles entering). Hysteresis cannot remove it.

Lesson: the plan's dependent-chain estimates overstated what staging could remove. Removing items (tiles, rows, launches) is what paid.

## Open

- Seep shows an intermittent "closure over cap 32" failure at frames 3–4. It failed in 1 of 5 runs from af94e6fa and in 1 of 1 from 9c2fc849, against 0 of 5 from 98ce7bae. The bodies lane is investigating.
- Wake latch: join tiles are 2128 of 3083 h tiles, all from particle refine. The census lane owns it.
- Static/body solid record split, about 2.8 ms of remaining body cost. The bodies lane owns it.
- Galerkin coarse pressure operator: an expected 0.4–0.8 ms. Parked as a redesign.
- Relayout cost scaled to changed tiles (head re-extension). Unowned.
