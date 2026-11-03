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
