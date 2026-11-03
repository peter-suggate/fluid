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

## 2026-11-03: seep's intermittent closure overrun was a stale slot base (0cbf053a)

af94e6fa inserted the candidate-mask rows ahead of the solid record's 77-row slots. Two readers
had re-derived the slot base by hand: the census refinement (uniform-solid-refinement.ts) and the
wall cache (uniform-mixed-momentum-cache.ts). After the insertion they read other tiles' slots,
wherever the racy claims put them, so seep's coarsening changed from run to run. Its step-1
readback volume is a deterministic detector: 688.2k is correct; the bad builds gave 835k–884k, and
the "closure over the cap of 32" failure followed at frame 3–4.

- The fix: one definition of the slot row, `uniformSolidSlotRowWGSL`, used by the record and by
  both readers.
- The guard: tests/uniform-solid-slot-layout.test.ts fails on any hand-written `77u*` slot base
  outside the solid file, and checks the layout arithmetic.

Reproduction on clean exports, 4–8-step check runs:

| Build | Bad volume (835–884k) | Closure failure |
|---|---|---|
| af94e6fa, and HEAD before the fix | 2 of 2 | not reached in 4 steps (coordinator: 2 of 6 runs) |
| After the fix | 0 of 3 (688.2k each) | 0 of 3, and none in the seep 300 gate |

## 2026-11-03: couple drops its ambient-velocity probes (5bf2d836)

Couple wrote words 7–9 and 11 of the body accumulator, the ambient velocity and its weight. The
integrator reads them only for the drag of a body that no cell couples by pressure. Any body couple
would weigh them for has some cell near it, and that cell sets word 10, which zeroes the drag. So
the six probes per displacing cell were dead work. The couple-reference GPU test was updated and
passes.

- couple: 0.62 → 0.19 ms on paddleboats.
- Gate on a clean export:
  - seep 300: drift +0.00025, 31.8 ms, no closure error;
  - letters 120: drift −0.0112, 99.1 ms;
  - paddleboats 90: drift −0.0066, 48.7 ms.

## 2026-11-03: exact clearance certificates and a static shortcut for the body queries

Knockouts on paddleboats, which has no terrain, put the cost of the face apertures at about
0.82 ms of buildRegions' 1.26 ms. The body sampling was only part of that: each sample's static
voxel and terrain probe costs about four memory reads. Two exact changes:

- **Clearance certificates.** These apply in umFaceOpen, umCellBodyFraction and umSolidFaceVelocity.
  - Each candidate body takes one signed distance at the face or cell centre. Shape distances are
    1-Lipschitz, and a 1% slack covers rounding.
  - A body farther away than the sample radius holds no sample, so it is dropped.
  - A body deeper than that radius, with its bounding sphere around every sample, holds all of
    them. The aperture is then 0, the coverage is 1, or the first such body owns every
    non-static sample of the face velocity.
- **Static shortcut.** Every aperture or dual-cell sample of a face lies in one of the face's two
  cells. If neither cell has a voxel or a boundary record and there is no terrain, the samples skip
  the static probes.

Regressions:
- tests/uniform-body-tile-mask.test.ts now compares umFaceOpen and umCellBodyFraction bit for bit
  with sample-by-sample all-roster oracles (umSolidFaceVelocity already was). It requires both
  certificates to fire, and checks that no sample of a face with two static-free cells is a voxel.
- query-reuse, containment and couple-reference also pass.

| paddleboats `--bodies --kernels=60`, mean ms | before | certificates | + static shortcut |
|---|---|---|---|
| buildRegions | 1.260 | 1.114 | 0.901 |
| pRows | 1.013 | 0.998 | 0.941 |

Together that is −0.43 ms of paddleboats GPU time.

Gate on a clean export: check:types clean; test:unit 1542/0; the body GPU regressions 10/0.

| Scene | Drift | lateHalfMedian |
|---|---|---|
| seep 300 | +0.00047 | 31.3 ms, no closure error |
| letters 120 | −0.0112 | 106.4 ms |
| paddleboats 90 | −0.0068 | 47.8 ms (was 48.7) |

Letters has no bodies, so none of these queries runs there. Its 99 → 106 ms change is noise from
the shared GPU.

What remains of the body cost is mostly buildRegions at 0.90 ms. A knockout that also skipped the
face-level voxel and record reads of body-near faces reached 0.78 ms. Recovering that last
0.13 ms would need the tile's per-cell static flags shared through workgroup memory, which is
below the 0.3 ms bar, so it is not done.
