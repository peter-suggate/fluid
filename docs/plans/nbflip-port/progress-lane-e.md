# NB-FLIP brick engine, lane E: engine core (progress)

**Brief.** Lane E owns the brick engine's core in `lib/methods/narrow-band/` (design
`docs/plans/nbflip-structural-2026-11-15.md` §5):
- storage, the directory and the `set*` kernels;
- lists, the schedule and record(), the param ring and submission;
- the host, contract and presentation adapters;
- `solidStamp` and the scene-apply bridge;
- the method-id switch;
- the brick ABI doc.

Milestones: M0(e) the ABI, M0(b) peak counts, M1 the skeleton, then M2 tier transfer, M3 solids
and M6. Scratch: `~/code/me/fluid-scratch/lane-e/`.

## 2026-11-15 STARTED

Base a610cde4. Read the design (all of it, with the §7.1 rulings), the brick arena
(`uniform-brick-arena.ts`), the seep's solver host, layout and WGSL (`lib/methods/seep/`), the
BULK16 page record, NB-FLIP's particle record and the window directory header.

## 2026-11-15 M0(e): the brick ABI, frozen

- `docs/plans/nbflip-port/brick-abi.md`: geometry and staggering (a cell owns its lower faces and
  corner; skirt bricks hold the upper vertex planes), the pools and channels at stride 520, the
  T2o directory (a toroidal page window, then 64 child words a C slot), the 32-word F and C
  records, neighbour words and the seam rule, the `set` buffer (control header, param ring, free
  stacks, marks, candidates, bounds, lists, bins), the indirect arguments and fold rule, the
  particle banks and bins, bindings and the recorded schedule, the brick-set update, and the
  failure bits.
- `lib/methods/narrow-band/brick-abi.ts`: the same as code. `BrickLayout` computes every offset
  for a scene's capacities, and `brickAbiWGSL()` gives a module the constants and accessors.
- `brick-abi.test.ts`: 6 CPU tests of the invariants.
- The prelude compiles on the GPU (check slot, rc 0): a probe kernel using every accessor builds
  its pipeline in 193 ms. The adapter reports 10 storage buffers a stage and a 4 GiB binding.

## 2026-11-15 M0(b): per-scene peaks and authored pools

Measured on the old engine (check slot, one run a scene, rc 0, `tools/gpu/fluid-gpu-run.sh`). The
probe is `~/code/me/fluid-scratch/lane-e/peaks.mts`, and its logs are `peaks-<scene>.jsonl`. Each
step it counts:
- the h tiles the brick reach rule would hold (the F ring);
- the tiles holding liquid (the core);
- the 4h pages over the ring;
- the BULK16 pages;
- the particles.

| scene | steps | F ring (step) | F core | liquid ring | C pages | BULK16 | particles (step) |
|---|---|---|---|---|---|---|---|
| high-resolution-dam-break | 300 | 2562 (60) | 1146 | 4022 | 122 | 3 | 997,037 (40) |
| letters | 240 | 6037 (230, still rising) | 2180 | 6991 | 207 | 0 | 977,923 |
| paddleboats | 300 | 6794 (0) | 3480 | 7399 | 238 | 0 | 316,155 (300) |
| reservoir-seep | 1200 | 1451 (1200, slowly rising) | 541 | 3951 | 169 | 72 | 219,671 |

Authored `scene.fluid.brickPools` (`lib/core/model.ts`, validated):
- f and c are 1.5x the peak ring and page counts, rounded up. The seep gets more, because it is
  still rising at step 1200.
- b is the BULK16 peak plus slack.
- particles is 1.25–1.5x the old engine's peak bank.
- k (cut bricks) stays 0 until `solidStamp` lands (M3), where it is measured.

| scene | f | c | b | k | particles |
|---|---|---|---|---|---|
| high-resolution-dam-break | 3968 | 192 | 16 | 0 | 1,250,000 |
| letters | 9088 | 320 | 16 | 0 | 1,500,000 |
| paddleboats | 10240 | 384 | 16 | 0 | 480,000 |
| reservoir-seep | 2560 | 256 | 128 | 0 | 330,000 |

Open items:
- Letters was still rising at step 240, so it is re-measured over the full take before M2 gates
  it.
- The frozen reach formula (two rings at rest) would have needed about 2.6x these F counts. M1's
  cell-precise reach box (brick-abi.md §5.4, amended) is what the table counts.
- During the letters run the runner reported 3 Metal aborts in other processes. `log show` showed
  none from our runs.
