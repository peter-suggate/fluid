# Census and item-count progress (plan items 20, 21, 23; census.md 5–9)

Probe: `tools/probe-census-reasons.ts` (via `tools/gpu/fluid-gpu-run.sh check`) reads, after a paddleboats run
with its motors driven, the census reason of every h tile and splits it by source:
- solid: active contact, cut neighbour, wet or dry uncut neighbour;
- join and the travel reasons: by Chebyshev distance to a tile holding particles;
- NB-FLIP activity: triggered criteria per tile, and the tile heat of every particle tile.

`--churn=K` also records, over the last K steps, which frames change the layout, why tiles enter and leave,
and how many flip back within 1, 2, 3–4 or 5–10 steps.

## 2026-11-03: solid promotion stencil sized to the solid's support (plan item 20)

**Measurement.** At step 300 on paddleboats (HEAD before this change), 1,703 of 4,360 h tiles were `solid`:
- 293 active contacts (cut, and wet or in the band);
- ~1,410 ring tiles promoted because one of their 26 neighbours was active.

Most ring tiles were wet floor tiles under the hulls, deep water beside them (tile rows y1–y4), and hull interiors.
No wall reaches them. The ring rule promoted every neighbour of an active tile, including neighbours no solid
reaches. The record's body flag, a bounding-ball test at the build pose (0.76 m for a hull), also marked every
tile within a hull's half diagonal as body-cut.

**Change** (uniform-mixed-dynamic.ts `solidPromoteTile`, uniform-solid-refinement.ts):
1. **The ring promotes only supported tiles.** A neighbour of an active contact is promoted only if the solid
   support reaches it (`umSolidTile`): a static cut cell within one cell of the tile (the host coupled plane), or
   a body's swept shape within its travel plus a cell (markTiles' key bit).
   - An unsupported neighbour holds no wall this step, and neither does any vertex it shares.
   - So it meets the cut tile as any 4h tile meets an h tile in open liquid; no coupling term reads it.
   - A wall within one cell of the shared boundary makes the neighbour coupled, so it is still promoted.
   - The closed-padding `wallSupport` rule is unchanged.
2. **Body contact for the exact-geometry test** is now the record's ball flag AND the swept-shape key bit
   (`umCensusBodyContact`).
   - Measured alone, this change is worth little: 4,081 h tiles instead of ~4,160.
   - It is kept because the ball flag is wrong about which tiles a body cuts.

**Regression:** `tests/uniform-solid-promotion-stencil.test.ts`.
- Setup: a two-cell-thick plate spans the tank, holds liquid on its −x side and moves +x at 0.1 m/s for 90 steps.
- Every 10 steps the test checks that:
  - the dry side holds no liquid (< 1e-4 cell volumes);
  - the liquid is still there;
  - a deep, uncoupled tile one tile and more from the plate runs 4h on at least 6 of 9 reads.
- HEAD fails it (4h on 3 of 9 reads); this change passes (8 of 9). Both keep the dry side dry.

Other regressions:
- `tests/uniform-voxel-containment.test.ts`: the thin partition and the settled tank pass.
- dam-break leaks 0.41 cell volumes at frame 30 with this change and 0.43 without it, so that failure predates this change.
- `tests/uniform-body-contact-refinement.test.ts`: the submerged sphere passes.
- The WGSL fixture test fails to compile with or without this change; it uses the default `umSolidBodyContact`
  path, which is textually unchanged.

**Counts:** paddleboats, step 300, probe, one run each.

| | HEAD | this change |
|---|---|---|
| h tiles | 4,160 | 2,978–3,083 |
| solid | 1,703 | 734 (298 active) |
| pressure band tiles | 2,620 | 1,653–1,689 |
| particles | 380k | 358k–366k |

**Gate.** One run each, on a snapshot of HEAD (af94e6fa) and HEAD plus this change. Drift is at the last step;
times are medians (lateHalfMedian).

| | HEAD | this change |
|---|---|---|
| paddleboats 90: step ms | 49.09 (52.74) | **46.27 (50.02)** |
| paddleboats 90: h tiles | 4,176 | 3,130 |
| paddleboats 90: band tiles | — | 1,613 |
| paddleboats 90: particles | — | 260k |
| paddleboats 90: drift | −0.006698 | −0.006625 |
| letters 120: drift | −0.011074 (rerun −0.010843) | −0.011159 |
| letters 120: step ms | 86.07 | 86.74 |
| seep 300: drift at 150 / 300 | fails at frame 4 (closure cap 38 > 32) | −0.000108 / −0.000277 |
| seep 300: step ms | — | 30.21 (31.48) |

- The letters drift difference is within run-to-run variance: HEAD's two runs differ by 0.023 pp, and this
  change is 0.0085 pp from the first.
- HEAD's seep failure is a residency closure cap in someone else's change: the same cap fails at frame 3 on the
  live tree. It does not reproduce on HEAD plus this change.
- `npm run check:types` is clean. `npm run test:unit`: 1,540 pass, 0 fail.

## 2026-11-03: layout churn (plan item 23): measured; the layout changes from need, not oscillation

`--churn=91`, paddleboats, steps 210–300. **Every one of 90 frames changes the layout, both before and after this change.**

| per frame | HEAD | item 20 | item 20 + `detailHoldSteps=8` |
|---|---|---|---|
| frames changing layout | 90 / 90 | 90 / 90 | 90 / 90 |
| tiles entering h | 101 | 82 | 93 |
| solid tiles entering h | 79 | 54–58 | 54 |

- The changes are dominated by `solid` tiles following 16 paddles.
- The paddles' tips move 2.3 h a step, and each wheel's next paddle re-enters a vacated tile about 8 steps later.
- One-step flips, where a tile re-enters the step after it leaves, are only about 13 tiles a frame
  (9 of them solid); a hold cannot remove the other ~70.
- `join` (particles drifting with the flow) and travel/closure (waves) add about 25 tiles a frame.
- An 8-step hold leaves every frame changing and adds tiles.

A frame with no layout change is therefore not reachable on this scene by hysteresis. The 2–3 ms of relayout
(remap, planeSums, the head's full re-extension) must instead scale with the changed tiles. Remap already visits
only tiles within one of a change. The re-extension is the lever: it runs regionally only when the layout holds.
This belongs to the extension's owner.

## 2026-11-03: join tiles and the wake (plan item 21): measured

Paddleboats at step 300, with item 20:
- 2,128 join tiles;
- 1,490 hold particles, and 540 lie one tile from one;
- the join reason splits no further by source on this scene: every join bit comes from NB-FLIP refine.
  Bodies add none, and the host join plane is empty.

Of the 2,207 tiles holding particles:

| tile heat | tiles | of which triggered this step |
|---|---|---|
| 0 | 1,088 | |
| 0–1 | 88 | |
| 1–2 | 299 | |
| 2 | 732 | 732 |

- Shape triggers 778 tiles, 535 by shape alone. Impact triggers 215, thin 29.
- Half the particle tiles are cold: their samples carry heat out of the triggered region and retire only when it
  expires (the fade).
- So join is set by two things: the shape trigger, and the samples' heat lifetime.
- The shape trigger reading particle roughness is the latch. It needs a noise-robust shape measure (the error at
  scales 4h loses, not the 0.2 h sampling roughness) or a smoother particle surface (APIC, the particles lane).
  Both are physics changes that need the letters volume regression first; not attempted here.

## 2026-11-03: census kernels after item 20; the per-workgroup atomic reductions were measured and dropped

The paddleboats census at HEAD 56b738d7 (`--bodies --steps=90 --kernels=60`, kernel means, ms) costs 4.4 ms over about 40 kernels. The largest:

| kernel | ms |
|---|---|
| classify | 0.91 |
| localSpeed | 0.65 |
| classifyCoarse | 0.65 |
| decide | 0.32 |
| pageMark | 0.24 |
| frame plan seed | 0.22 |
| solidPromote | 0.15 |

The global surface volume kernels (`metric` 1.05 and `dilate` 0.59) are outside the census and belong to the level-set lane.

One experiment measured the census.md item 7 atomics, together with the same fix for classify and classifyCoarse:
- classify and classifyCoarse flushed the group census (reach, extremes) once per workgroup instead of once per tile;
- localSpeed reduced `regionVelocity` in workgroup memory, with one flush per workgroup for the workgroup's region.

Both are maxima, so the values are unchanged. Result: classify 0.909 → 0.922, localSpeed 0.651 → 0.668, classifyCoarse 0.649 → 0.662 ms.
The contended atomics are not what these kernels wait on, so both changes are reverted.

The census.md premise for item 5 (classify replays an unchanged class) also does not hold here. A tile's class carries its face-velocity bounds, which change every step wherever liquid moves. Deep COLD regions already replay.

What remains is latency per tile in classify, classifyCoarse and localSpeed. The h tiles they visit fell 25% with item 20, but the kernel times above did not fall with them. The next lever there is the vertex-centric classifyCoarse (census.md item 6).
