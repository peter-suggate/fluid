# Lane F: progress (items 12, 13, 15 support)

2026-11-10. Work on main, uncommitted. Plan: `docs/plans/paddleboats-60fps-2026-11-10.md`, phase 4.

## Item 13: pressure launches

### The launch list as a pure function

`lib/methods/uniform/uniform-pressure-arena-launches.ts` derives the arena's launches for one step,
in order, from the schedule, the configuration and the host's widths: prepare, solve (with the kick's
re-solves), verdict and project. Each record holds its pass, entry, workgroups, stage family, cycle and
tail cycle. `tests/uniform-pressure-arena-launches.test.ts` builds the real `UniformPressureArena`
prototype over mocked fields and records every `dispatchWorkgroups`, `dispatchWorkgroupsIndirect` and
buffered launch on a recording pass and encoder. It then asserts the recorded list equals the plan entry
for entry, including the workgroups, across 2,880 configurations:
- cycles 1, 2, 4 and wideCycles 0, 1, 3, Infinity;
- sweeps 2, 3 and tail cycles 1–3;
- native sweeps 0, 1 and Krylov on and off;
- BULK16 absent, live 0 or live 40;
- the super-block tail on and off;
- solid on and off, and an empty band.

The kick (23 re-solves) is checked too.

### Paddleboats (512 x 32 x 192 h cells), level by level

Lattice: 16 x 1 x 6 pages. The pool holds 288 blocks and 192 are held at rest, because claimed air pages
sit above the open top. BULK16 capacity is 72; live is 0 at rest. The super-block tail is off: 288 + 72
rows is under 4096. The frame's schedule is cycles 4, sweeps 3, tail cycles 3, native sweeps 1
(composite), Krylov on.

| level | cell | grid on this box | rows | launches a V-cycle |
|---|---|---|---|---|
| 0 | h | 512 x 32 x 192 cells; band tiles only (about 1.8k) | band tiles | 3 pre-sweeps (+1 settle in cycle 0), 3 post, pResidual2, pRestrict0, pRebaseC, pProlong0 |
| 1 | 2h | band bricks | level-1 bricks | pDownA, pDownR, pUpA |
| 2 | 4h | 128 x 8 x 48 (8³ a page) | the held pages (192), plus BULK16 slots | pNativeOperator, pDownB, pDown2, pBulkLift, pUpB, 2+2 native |
| 3–5 | 8h / 16h / 32h | 64x4x24 / 32x2x12 / 16x1x6 per page | held pages + live slots | 3 tail cycles of pDown3, pDown4, pTail (32 sweeps, one workgroup), pUp4 (x2 on the last), pUp3 |
| Krylov | – | – | – | kStart (cycle 0); then kApplyT, kApplyNB, kReduce, kBound, kStep |

Degenerate levels: none. Every level has real rows on this scene. The 32h level is a 16 x 6 slab one row
tall, but a real 2-D grid. No tail cycle is structurally empty.

What does no work is only known at runtime:
- the converged remainder of the 4 encoded cycles (2–3 run), 43 launches a cycle;
- the BULK16 loops while live = 0.

The first needs either indirect gating or a CPU readback. pGate was measured and rejected for exactly
that reason (`activity-step-2026-11-05/progress-pressure.md`: +0.95 ms on the drop from Dawn's indirect
validation). It is left as it is.

Launches sized by capacity rather than live counts:
- `kApplyNB` = pages + min(BULK16 capacity, 1024). The kernel splits its grid at K_SLOTS (`gN=groups.x-K_SLOTS`), so its width is structural. Not changed.
- `kStart`, `kBound` and `kStep` rows = max(tiles/8, pages, slots). Pages dominate here.
- The fixed live-list grids: `pUnmark` and `ubaUpdate` (UNIFORM_LIVE_KEY_GROUPS), `ubaLists` (UNIFORM_LIVE_LANE_GROUPS) and `pClear2` (UNIFORM_LIVE_GROUP_GROUPS).
- The super-tail capacity6/7 grids. The super-block tail is off on paddleboats.

### Accounting, a step (paddleboats, 4 cycles encoded)

| pass | before: launches | before: GPU dispatches (indirect = 2) | after |
|---|---|---|---|
| unmark | 1 | 1 | 1 |
| marks and rows | 8 | 8 | 8 |
| solve | 179 (107 direct incl. pLaunch, 72 indirect) | 251 | 178 direct |
| verdict | 1 | 1 | 1 |
| projection | 5 | 5 | 5 |
| **arena** | **194** | **266** | **193** |
| step (probe `boats-moored`) | 491 (417 + 74 indirect) | 565 | 490 (488 + 2) = 492 GPU |

The solve's cycles: cycle 0 has 40 launches and cycles 1–3 have 43 each. Before the change, 17 a cycle
were indirect: pResidual2, pDown2, 3 x (pDown3, pDown4, pUp4, pUp3) with 4 pUp4 in all, pBulkLift and pUpB.
Four more came once a solve: pBulkRows, pResidual2F, pBulkProject and pBulkCommit. The fixture test
asserts these counts against the probe: 72 indirect, 107 direct = 178 − 72 + 1.

### The change: every BULK16 count launch direct

Before, pLaunch, a one-thread launch, wrote 6 indirect arguments a solve from the level-2 page count and
the BULK16 live count. 72 launches read them, and Dawn runs a hidden validation dispatch ahead of each
(`validatedParams`, confirmed in `node_modules/webgpu`'s dawn.node).

Now:
- pLaunch, its buffer (96 B), its pipeline and the `{indirect}` launch kind are gone.
- `uniformPressureArenaCountWidths(pages, bulk)` gives:
  - level2 = pages
  - either = max(pages, bulk)
  - items = min(pages + bulk, 1024)
  - live = max(1, bulk)
  - live8 = max(1, ceil(bulk / 8))
  - items64 = max(1, ceil((pages + bulk) / 64))
- `bulk` is the new `observeBulk` budget. It uses `uniformBufferedWork` on the latest receipt's BULK16 live count, with the relayout reserve as pages and the capacity as the ceiling, and starts at the capacity. `uniform-mixed-frame.ts` calls it beside `observeWork`.
- The arithmetic is unchanged. Every kernel on these launches strides its own GPU count by `groups.x` (`for(e=group.x;e<n;e+=groups.x)`, or 8 a group for pBulkCommit) and reduces with atomicMax only, so a width sizes parallelism and nothing else. The WGSL is untouched.

Removed: 1 launch and 72 validation dispatches a step (266 → 193 GPU dispatches in the arena, 565 → 492
in the step). The kick's 24 solves lose 73 each.

Risk (unmeasured): a burst of new BULK16 slots runs serially for the receipt lag, about 2 frames, at the
decayed width. Growth is immediate on evidence. Before, the indirect count was exact each frame.

## Item 12: static and body solid record split

Finding: the split is already in place. In `uniform-mixed-solid.wgsl.ts`:
- `encodeCoarse` (line ~1144) does a full build only when `full` is set: at the first build and after `invalidate()`, which a live edit calls (with an epoch bump).
- When bodies move it runs `invalidate(true)`. `bodyMask` writes per-tile dirty bits from the swept-body certificate (`umBodyTileSwept`). Then `buildRegions` and `cutsRegions` rebuild only the masked regions' tiles, and `surfaceClearBodies` runs.
- The static per-tile flags (`UM_STATIC_NO_VOXEL`, `UM_STATIC_NO_TERRAIN`) are built with the full build.
- Every consumer (pRows via `umPressureFaceVRecorded` and `umSolidApertureAt`, the couple, remap) reads the combined record.

The 2.8 ms in the plan comes from 11-03: buildRegions 1.2, couple 0.63, pRows body share 1.0. Since then:
- couple fell to 0.19 ms (5bf2d836);
- buildRegions fell to about 0.9 ms;
- pRows fell to 0.72–0.86.

A wider buildRegions grid was measured at −0.19 ms. A compact dirty-tile list would have to beat that,
and it adds a compaction launch. I did not build it: it changes WGSL and needs a timing A/B, which this
lane may not run.

Regression added, `tests/uniform-solid-record-split.test.ts` (GPU). It uses three free bodies (two
spheres and a box) with seeded random poses, velocities and spins, and a local voxel fill at step 4,
over 6 steps:
- every step, the record built incrementally answers every cell's openness and three face apertures bit for bit as the direct geometry does (no record);
- at steps 3 and 6 a forced full build answers bit for bit as the incremental record did;
- every tile's cut flag is a cut cell;
- the filled cell is closed.

Existing coverage of the other AGENTS.md items:
- static flags under edit, undo, redo, save and load, and edits staying local (a distant tile keeps its flags): `uniform-solid-static-flags`;
- one-body pose publication: `uniform-body-query-reuse`;
- tile body masks: `uniform-body-tile-mask`.

## Item 15 support: `tools/dispatch-diff.ts`

`npx tsx tools/dispatch-diff.ts before.json after.json` takes `--dispatches` probe JSONs. It prints
direct, indirect and GPU launches (direct + 2 x indirect), before, after and the delta, by stage family
and by changed label. `tests/dispatch-diff.test.ts` runs it on `tests/fixtures/paddleboats-moored-dispatches.json`,
the moored paddleboats step 300 from before this change.

## Verified, and how

- `npm run check:types`: clean for these files. Lane R's in-progress `webgpu-rigid-body.ts` and `rigid-motor-gpu.test.ts` errors are not mine.
- `npm run test:unit`: 1695 pass, 2 fail, both outside this lane: `paddleboat-contacts.test.ts` imports the deleted `box3d/motor-contacts`, and `reservoir-particle-cost` checks a shader string in a narrow-band file.
- The launch plan against the encoder: the mock test above (no GPU).

### GPU check runs (3 of 3 used, `tools/gpu/fluid-gpu-run.sh check`)

1. Working tree: `uniform-solid-record-split`, `pressure-super-tail-bulk16` and `uniform-bulk-slope-rest`.
   - `uniform-bulk-slope-rest` passed (BULK16 rest under the direct widths).
   - The split test's first version failed by design. It checked cut flags against the pose the frame had just published, while the record still held the pose it was built from. It was rewritten to build bodies-only to the current pose first, as `uniform-body-query-reuse` does. Its openness and aperture parity held at step 1.
   - The super-tail test failed: "w1: the groups hold resident pages too (9 against 9)".
2. Invalid. Both arms (a working-tree snapshot without my change, and the working tree) failed in about 1 s with an empty GPUPipelineError. This follows from peer in-progress edits (Lane R's `rigid-assembly-contact.ts` changed between runs) and is not attributable to this lane.
3. Clean snapshots in `~/code/me/fluid-scratch/`: `lane-f-base` = HEAD 112163d3, and `lane-f-cand` = HEAD plus this lane's arena, launches module, frame line and split test.
   - **The super-tail test fails identically on base and candidate** (9 against 9). It is pre-existing and not caused by the direct widths.
   - The split test was SIGKILLed at 240 s. The kernel's memorystatus killed a node process at 172 GB compressed. The likely cause is a failing `assert.deepEqual` on a whole buffer building its diff, but the run cannot show which comparison failed. **Item 12's regression is therefore unverified and may expose a real bodies-only/full/direct mismatch.** The test now fails at the first differing word with a short message. Next run:
     `WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js tools/gpu/fluid-gpu-run.sh check node --import tsx --test tests/uniform-solid-record-split.test.ts`

## GPU gates still needed (timing, not run by this lane)

Interleaved A/B, base = HEAD without these files and candidate = with them, one run each:

```
tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/probe-reservoir-bulk.ts --scene=paddleboats --bodies --pipelined --steps=300 --every=60 --dispatches --out=/abs/boats-<arm>.json
tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/probe-reservoir-bulk.ts --steps=300 --every=60 --pipelined --out=/abs/seep-<arm>.json
tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/probe-reservoir-bulk.ts --scene=nbflip-figure-8-letters --steps=120 --every=120 --dt=0.016666666666666666 --pipelined --out=/abs/letters-<arm>.json
npx tsx tools/dispatch-diff.ts /abs/boats-base.json /abs/boats-candidate.json
```

Expected:
- the arena solve shows 178 direct and 0 indirect;
- step time drops by about 72 x the validation-dispatch cost. Earlier work measured about 5 µs per chained empty launch, so about 0.3 ms.
- seep and letters drift should not change, since the arithmetic is unchanged; atomic slot order still varies.

## Follow-up (coordinator, 2026-11-10)

### Job 1: `uniform-solid-record-split` was a test-ordering bug, not a record bug

The run-4 failure (-1 against 1 over 95,232 words) came from the probe ordering. `invalidate()` bumps the aperture epoch with
`queue.writeBuffer`, and a queue write executes before any command buffer submitted after it. So the probes in the
same submission read the new epoch against records stamped with the old one, and `umPressureFaceVRecorded` returned
its stale sentinel, -1. The test now encodes bodies, invalidates and encodes coarse, then probes cached and cut in a
separate submission at steps 3 and 6. The comparison is unchanged: word by word, all five buffers, first difference
reported. **GPU check (follow-up run 1 of 2, `lane-f-cand`): pass, 94 s.**

### Job 2a: markTiles certifies box bodies with the boxes' separating-axis gap

`lib/methods/uniform/uniform-body-tile-bound.ts` (`umBodyBoxTileGap`), used by `umBodyMarkTile` in
`uniform-mixed-bodies.ts` for shape code box only.

- **Old lower bound.** The shape's 1-Lipschitz distance at the tile centre, less the half diagonal length(2h). It is
  exact only along the diagonal. Beside a face it understates the gap by (sqrt 3 - 1) x 2h = 1.46 h.
- **New lower bound.** max(old, gap). The gap is the largest separating-axis value over:
  - the body's 3 axes;
  - the world's 3 axes;
  - their 9 cross products, normalized and skipped below length 1e-3;
  - the direction from the body box's closest point to the tile centre.
- **Why it is provably conservative.**
  - For convex A and B and any unit n, every point pair is at least |n.c| - sum e_k|n.u_k| - sum a_j|n.w_j| apart
    along n. That is the centres' separation less the two projected half widths, so dist(A, B) >= gap(n) for every n.
  - A body cut only removes solid, so the uncut box's gap still bounds a cut box below.
- **Unchanged:**
  - the margin (|v| + |w| r) x horizon + cell;
  - the ball prefilter, which is sound under spin because the body stays inside its ball;
  - inverted shapes, which the ball still marks.
- **CPU mirror test, `tests/uniform-body-tile-bound.test.ts` (3 tests, 18 s):**
  - (i) Hull and blade boxes at random poses: the gap never exceeds the distance sampled densely over the tile
    (h/2 spacing). Beside a face it is exact (3h, where the old bound gives 1.54 h).
  - (ii) Random poses with |v| up to 2 m/s and |w| up to 15 rad/s: no unmarked tile comes within a cell of the swept
    body, sampled at 9 times over the horizon. 2,542 tiles were reached and none was missed. The new marks are a
    subset of the old (4,499 down to 4,252).
  - (iii) Paddleboats at t = 0.2 s, with the blades posed by `rigidMotorPose` at 1/3 throttle and their omega from
    the ramp: **2,004 marked tiles down to 1,512 (-25 %)**, the hulls alone 1,272 down to 900. Liveness is ignored,
    and the grid origin is (-w/2, 0, -d/2).
- Naga: the rigid-bodies module (065) validates. In the live tree, 4 other modules fail naga: mixed momentum and
  three narrow-band FLIP modules (`nbBoundaryCubicHit`). All are peers' uncommitted edits.
- GPU check (the mark kernel, `lane-f-cand` plus these files): see below.

### Job 2b: left as it is, and why

A closed tile (solidRecord.x == 0) becomes h in `solidPromoteTile` (uniform-mixed-dynamic.ts) by one of two routes.

1. **`wallSupport`.** The tile is the closed padding at the nearest open tile, and that tile is decided-fine or
   static-fine. Coarse padding would impose 4h hanging vertices on the h liquid wall it shares. A closed tile with
   no open tile among its 26 neighbours is already not promoted (wallDistance stays 4).
2. **The support stencil.** The body's swept shape reaches the tile and an active neighbour exists.
   - The stated failure can occur for a closed tile here. A coarse dry tile owns vertices it shares with the cut
     tile and can interpolate liquid across a thin wall.
   - solidRecord holds the current pose only. A tile closed now can open within the step as the body moves.

A certificate that the tile and its ring stay closed would need the body's interior depth at the tile to be
>= length(3h) + (|v| + |w| r) x horizon. The census has no body roster or horizon to compute that depth, so this
route was not changed and the ring rule was not touched.

### GPU check for the mark kernel (follow-up run 2 of 2)

The run used `lane-f-cand` plus the bound and the markTiles change, with `uniform-solid-promotion-stencil`,
`uniform-pool-launch-width`, `paddleboats-scene` and `paddleboat-contacts`: 9 tests, 7 pass, 1 skip, 1 fail.

- **Passed:**
  - the thin moving plate (promotion stencil): dry side dry, uncoupled water coarsens;
  - all paddleboat scene and contact tests.
- **Failed: `uniform-pool-launch-width`** on `ubaRecord` ("3 blocks against 131", 322 against 98 groups). markTiles,
  the first watched entry, matched across the two pool sizes.
  - ubaRecord's width is groupsOf(32 (2 workBricks + workLevel2), 64), from `observeWork`. Neither this lane's arena
    diff nor the frame line touches it: `observeBulk` sets only `workBulk`.
  - So the failure is most likely pre-existing at 112163d3, but it is **unverified**: no base run was left in the
    budget. Next run:
    `WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js tools/gpu/fluid-gpu-run.sh check node --import tsx --test tests/uniform-pool-launch-width.test.ts`
    in `~/code/me/fluid-scratch/lane-f-base`.
