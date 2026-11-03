# Phase 1 progress: bodies resolved once per step

## 2026-11-03 — couple over the bodies that reach each tile, reduced in workgroup memory (plan item 3)

`couple` (uniform-mixed-bodies.ts, now the exported `uniformMixedCoupleWGSL`) ran a lane per h cell
over all 18 bodies and added every word of every pair to the global exchange with atomics. Now:

- jobs are the h tiles only (the 4h owners never took a body term);
- lanes 0..count-1 test each body's bounding sphere against the tile's cell-centre box with the
  shared 1.01·|h| near margin, giving a workgroup-uniform body set (the old per-cell test is kept);
- a cell whose centre lies more than 0.505·|h| from the surface (the shape distances are 1-Lipschitz
  and signed as `umInsideBody`, and every face and corner sample lies within 0.5·|h| of the centre)
  skips its 32 shape samples: its face fractions cancel and its fraction is 0 or 1;
- contributions sum in workgroup atomics, zero terms add nothing, and one flush per tile adds
  each non-zero word (and sets the pressure-coupled flag) in the exchange.

Exchange words are integer fixed point, so the result is bit-identical to the old all-body kernel:
`tests/uniform-mixed-couple-reference.test.ts` compares all 288 words over six randomized trials
(1–24 bodies, every shape, overlap, an inverted box, scaled shapes, strided workgroups).

| paddleboats `--bodies --steps=90 --kernels=60` | before | after |
|---|---|---|
| couple (mean ms) | 2.133 | 0.630 |
| step lateHalfMedian (ms) | 69.8 | 67.5 (other lanes' uncommitted edits in tree) |
| drift | −0.00672 | −0.00668 |

Gate (one run each): check:types clean, test:unit 1538 pass / 0 fail;
seep 300 inviscid pipelined: drift +0.000315, lateHalfMedian 31.3 ms;
letters 120: drift −0.0113, lateHalfMedian 104.4 ms (no bodies; unaffected by this change);
paddleboats 90: drift −0.00668, lateHalfMedian 67.5 ms.

## 2026-11-03 — per-tile candidate body masks; bodyMask a lane per (region, tile); SDF-certified dirty tiles (plan items 1, 4, part of 2)

- **bodyMask** is now a workgroup per 16h region: lanes below the capacity take a body each
  (region reaches by `atomicMin` of the monotone `umBodyReach`, moved and near sets by `atomicOr`),
  then a lane per tile writes the tile's dirty bit and its **candidate mask**: every slot whose
  bounding sphere comes within 3 hmax (+1% of a cell) of the tile's cells, inverted slots always.
  Four keys a record row (exact f32, ≤ 24 bits), a stamp row per block (its page, live) and a frame
  row (frame origin, body count) after the 8 per-tile rows.
- `umBodyTileMask(t)` serves the mask only while `umSolidBodyPoseCurrent` and the stamps match,
  else the whole roster. `umCellInsideSolid`, `umSolidTileClear`, `umSolidAtWorldCell`,
  `umBodyPoseNearTile` (margin ≤ 3) and `umSolidFaceVelocity` visit only those bits; each keeps its
  own near test, so results are identical (same candidates, same roster order).
- **Dirty tiles** (`umBodyTileSwept`): a moved slot present at both poses with the same touched
  flag no longer dirties a tile whose centre lies, at both poses and with one sign, farther than
  the reach to every sample (|2h + 2hmax|, 1-Lipschitz shape distances), provided outside the
  sphere reaches all or none of the tile's face centres at both poses (umPressureFaceV's body-near
  form differs from the body-free one beside terrain).

Regressions: `tests/uniform-body-tile-mask.test.ts` (six moving bodies of every shape beside a
terrain mound, three steps, before and after a bodies-only rebuild): every masked query equals its
all-roster oracle bit for bit, every body reaching a cell's centre or corners within the mask reach
is in its mask, the mask is the whole roster after pose publication, and the rebuilt record equals
the direct geometry. `uniform-body-query-reuse` passes (its lifetime test's body buffers were sized
for a 12-body capacity; fixed to the 24-body one).

| paddleboats `--bodies --kernels=60`, mean ms | before (294ed5cc) | after |
|---|---|---|
| bodyMask | 0.397 | 0.068 |
| pRows (body queries) | 1.505 | 1.027 |
| buildRegions | 1.454 | 1.216 |
| coupleFine (umSolidTileClear) | 1.180 | 0.903 |

Gate: check:types clean; test:unit 1540/0; seep 300: drift −0.0027, lateHalfMedian 48.8 ms; letters
120: drift −0.0112, 93.4 ms; paddleboats 90: drift −0.00658, lateHalfMedian 61.8 ms (was 67.5).
The tree carries other lanes' uncommitted pressure/surface/census edits; the seep change (31.3 →
48.8 ms, drift +0.0003 → −0.0027) is not from this change: seep has no bodies, and its bodyMask is
0.034 ms with no other solid-record kernel above 0.06 ms.

## 2026-11-03: per-cell dirty certificate measured and dropped; what remains

When the tile-level SDF certificate fails, a cell-by-cell form could still prove a tile unchanged:
each cell centre beyond 1.51 hmax of both surfaces on one side, and each face centre reached or
missed by both poses' spheres. The extended tile-mask and query-reuse regressions passed (4/4).
It cut paddleboats dirty tiles only from about 1930 to 1800 per step (80–88 of 2320 regions dirty).
That is 7% of a buildRegions of about 1.2 ms, so at most about 0.08 ms, below the 0.3 ms bar. It was
reverted unmeasured, and the timing run was cancelled.

What remains is a body-free static record, so the body pass rebuilds only the cut rows of these tiles:

- buildRegions (about 1.2 ms) is per-tile build latency over about 1800 genuinely touched tiles.
  The census found a wider grid bought only 0.19 ms.
- couple's remaining 0.63 ms is umBodyAmbient's six all-body probes per cut cell.
- pRows' body share (about 1.0 ms) still runs the face-fraction geometry per cut face.
