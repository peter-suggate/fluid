# Particle half of momentum transport: progress

## 2026-11-03: M1 (brick-staged move) measured and rejected

The plan's M1 (item 9) assumed the move is bound by re-resolving per-tile chains: the sampler, the bin runs and the nodal depth. I first attributed the move's time with knockouts, then built two staged variants. Neither was faster, so nothing from M1 lands.

Runs used `fluid-gpu-run.sh timing`, paddleboats `--bodies --steps=90 --every=90 --kernels=60`, normalised by `rows[-1].particles.live`. Baseline: advectParticlesDeferred 3.579 ms, 13.89 ns per particle; separationPrepass 0.858 ms, 3.33 ns per particle.

### Knockouts (move ns per particle; noise about ±0.8)

Each knockout skips one part of the move. They are not physics; they are only for attribution.

| knockout | move ns/particle | Δ |
|---|---|---|
| none (baseline) | 13.89 | 0 |
| skip RK stages 1–2 field sampling (`umSampleVelocity`) | 14.16 | ~0 |
| `gridSupported` always true | 12.86 | −1.0 |
| no `nbRigidContact` | 12.67 | −1.2 |
| no `nbSurfaceShift` | 12.50 | −1.4 |
| no `nbFinish` solid walk | 12.46 | −1.4 |
| no merge check (destination run, centre depth, `nbMergeRun` gather) | 10.87 | −3.0 |
| no pair separation (prepass and the move's shift) | 8.82 | −5.1, plus the 3.33 of the prepass |

Skipping two of the three velocity samples saves nothing, so the sampler chain is not the critical path. The cost sits in separation and merge. The merge knockout removes the gather over the cell's other records as well as the run lookup.

### Staged variants (all reads proven equal)

A GPU regression enabled a verify mode in the move. The move recomputed every staged read through its device lookup (`nbRun`, `nbBinCell`'s run, `particleDepthIn`, compared bit for bit) and counted mismatches. This ran on `nbflip-figure-8-letters` over 40 steps, with h, 4h and 16h regions all live: 0 mismatches, both variants.

| variant | move ns/particle | prepass ns/particle | move + prepass |
|---|---|---|---|
| baseline (64 lanes) | 13.89 | 3.33 | 17.22 |
| A: 128 lanes; stage the 10³ cell runs (16-bit packed) and 10³ vertex depths a slab job | 15.52 | 3.34 | 18.86 |
| A, separation folded into the move (no prepass) | 18.61 | — | 18.61 |
| B: 64 lanes; stage the brick's own 512 cell runs (a tile a round, coalesced) and 9³ depths by one vertex run, one barrier | 15.31 (about 14.7 after the run's +4% on every other kernel) | 3.49 | — |

Staging costs more than it saves. Paddleboats has 1555 bricks at about 163 records each, so a job's staging (about 750–6000 loads behind a barrier) replaces lookups that already hit cache. At 10% occupancy the barrier's latency is not hidden. Folding the prepass into the move is neutral (18.61 vs 18.86).

The patch and its regression are kept outside the tree. They are not committed because they bring no gain.

### What would move the particle kernels

- **Separation (about 2.2 ms: the prepass plus the move's shift)** is the largest particle cost. `nbSurfaceShift` takes 7 depth probes per surface sample; the pair loop reads 48-byte records for 16 bytes of position.
  - A lever is a depth gradient from one cell's 8 vertices. It changes arithmetic, so it needs drift gates rather than bit identity.
  - Another is reading the compact position stream for pairs, if its w word can carry the live/heat test.
- **Merge (about 0.8 ms)** is the run gather in `nbMergeRun`, not its lookup.
- **Transfer.** `transferSeams` 0.50 ms and `transferDeposit` 0.44 ms. With lookups cache-resident, staging per brick would recover less than 0.3 ms each; dropped under the 0.3 ms rule.
- **`slabSortMerge`.** 0.14 ms of empty launches; below 0.3 ms, dropped.
- **M4 (gravity into the momentum store).** This is worth about 1 ms (forces 0.46 + 0.30 + 0.23), but it is more than a fold. Forces turns `advected` into `output`, and also owns:
  - the low-face `boundary` words;
  - gravity's occupancy and coarse-mass predicate;
  - COLD retain/keep;
  - frozen and far-air faces;
  - inflow.

  Moving them into `momentumStep` changes the frame's texture flow. Not started.

## 2026-11-03: the separation and merge levers, measured and dropped; M2/M3(a) premises checked

Runs from different times are not comparable: other agents' uncommitted edits change the flow, and live counts reached 238k–258k on the same move code. Every comparison below is therefore a back-to-back pair on one tree (paddleboats `--bodies`, 90 steps, ns per particle).

| change (all exact unless noted) | before | after | ms |
|---|---|---|---|
| prepass pair reads from the compact position stream (bin order, 16 B) instead of the 48 B record; inexact form (no w test) | separationPrepass 3.58 | 2.93 | −0.14 |
| merge: skip the centre depth when the run is within the cap's lower bound (`nbCellCapFloor`), plus surface shift reusing G2P's depth (`velocity.w`) when the rigid push-out left q unchanged | advect 14.40 | 14.26 | −0.035 |

Both rows are under 0.3 ms, so both are dropped. The pair loop already reads its neighbours contiguously in sorted order. The merge gather (`nbMergeRun`/`nbMergeAbsorbed`) already walks the cell's contiguous sorted run, and only in cells over the cap (20, ungraded).

The two exact patches remove one or two depth evaluations per particle and buy nothing, which confirms that the depth and run chains are cache-resident. The knockout deltas were the work itself (pair arithmetic, merges), not lookup latency.

**M2 already exists.** The tail's `pair_*` extension (`uniform-mixed-extension-pair.wgsl.ts`) carries both payloads as one vec2: "all distance decisions and ordered reductions stay scalar and execute once". The second, non-pair sweep is the head's `encode()`, which re-extends after relayout. It is regional when the layout holds, but the paddleboats layout changes every frame (census churn).

Per-step costs: pair sweeps 1.15 ms, head sweeps 0.68, hierarchy 1.14 (prolong 0.64, block sweeps 0.30, restrict 0.20), publish 0.86.

**M3(a)'s launch-floor premise does not hold.**
- Every hierarchy launch strides only the held-block list (`heJobs` scales with `heHeldBlocks()`, not with capacity).
- The dirty and old-payload gates already zero the jobs.
- The 5.7 × 52–98 µs per run is per-cell work over about 192 held blocks.
- The skippable part is the level-0 prolong over blocks fully known at 4h. The coarser levels feed neighbour blocks' prolongs, so they would need a dilated list. Known cells already exit after 3 loads, so this bounds the gain at about 0.1–0.2 ms, under 0.3.

The remaining lever on the extension is the head re-extension that churn forces: the census agent's area.
