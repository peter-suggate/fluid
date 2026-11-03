# Level set on paddleboats: progress

## 2026-11-03: P1, P3(b), P4, P5, P2 and P8 measured or ruled out; nothing clears 0.3 ms

All runs used `fluid-gpu-run.sh timing node --import tsx tools/probe-reservoir-bulk.ts --scene=paddleboats --bodies --steps=90 --every=90 --kernels=60`. Kernel means are in ms per step. Other agents' uncommitted work was in the tree, so step totals are not attributable. Kernel means of the advect and arena launches are mine alone.

**Lattice.** The probe runs paddleboats at its own lattice, h = 25 mm. The output file's name, `paddleboats-5mm-default`, is only a label. At step 90 the global maximum speed is 2.33 m/s. That is 1.55 cells per step, so the frame plan's global bound already allows a 3-cell box everywhere. A per-region bound adds nothing on this scene.

### P1: narrow-band transport certificate. Built and measured; it nets 0.17 ms, so it is not landed.

The patch is `patches/levelset-p1-far-certificate.patch`. It contains the surface changes and a GPU regression, `tests/uniform-surface-far-certificate.test.ts`.

**What it does.** An h vertex keeps its input bits when two conditions hold:
- Every cell of its departure box, of radius ceil(travel)+1 with a minimum of 2, has eight corners of one strict sign at 2h or more.
- The cell is open.

A 4h vertex keeps its input bits when every tile of [t-2, t+1] is "clean". Clean means corners at 8h or more (the 4h cubic's threshold), built by a separable spread.

Kernels that build the certificate's tables before advect:
- farFine: the good-cell masks of h tiles.
- farClean: which 4h tiles are clean.
- farSpread ×3: the separable spread of clean tiles.
- farSpeed: the speed bound of each 16h region.

**Why it is safe.**
- The vertex's sign equals the traced sign.
- The arena rewrites every vertex within reach from the crossings, so the post-redistance field there is identical.
- Beyond reach, the field carries sign plus ≥ 5h, as before.

**Census of the geometric potential at step 90, ignoring speed** (a scratch census of the read-back phi):

| vertices | count | share that can certify |
|---|---|---|
| h, box radius 2 | 63.7k of 110k | 58% |
| h, box radius 3 | 51.5k of 110k | 47% |
| 4h, clean | 17.9k of 47.4k | 38% |

**Kernel A/B**, comparing the base (before any change) with P1 using the local speed bound:

| kernel | base | P1 |
|---|---|---|
| advectFine | 0.979 | 0.975 |
| advectTrace + advectFinish | 0.761 | 0.547 |
| advectDeferred (twinned) | 1.484 | 1.406 |
| advectOwners | 0.095 | 0.040 |
| advect (merged coarse) | 0.201 | 0.258 |
| farFine + farClean + farSpread + farSpeed | 0 | 0.170 |
| **advect stage total** | **4.303** | **4.133** |

The gross saving is about 0.34 ms. The certificate's tables cost 0.17 ms.

**Why the gross saving is small.** Most far vertices already leave advect through cheap exits:
- `umKeptAir`
- `umEarlyFarAdvected`
- `umCheapAdvected`
- the COLD store

The certificate only removes the listed traces (advectTrace and advectFinish) and part of the coarse queue.

**What would make it land.** Taking the masks from the frame plan's `localSpeed`, which already loads all 125 vertices of every h tile and the eight corners of every 4h owner, would remove most of the 0.17 ms of tables. Even then the estimate is about 0.25–0.3 ms, and that file belongs to the census agent.

### P3(b): halved arena workgroup memory. Measured slower; reverted.

The change:
- Pack the nearest cells as u16 pairs, `array<u32,500>`.
- Use a 768-entry queue, with an in-lane solve on overflow.

arenaDistance went from 0.953 to 1.068 ms. Fusing the arena kernels, P3(a), was already tried: `arenaDistanceCanonical`, the fused metric and publication, is the dead path the comment at `arenaPublishCanonical` explains. One workgroup per core held the metric's windows and queues.

### P8: cell-granular or shorter reach. Not pursued.

The reach is already a parameter, `surfaceRedistanceReach`. Reach 1 failed the letters A/B in 39b6f25b: drift moved past the within-arm spread, and particles rose by 6–8%. It also saved only about 0.27 ms at 10k fine tiles. Reach 2 would save less than that.

### P4 and P5: resolve and geometry fill on change only. Ruled out by the layout.

The seam tiles are where h tiles meet 4h tiles. On paddleboats that is one to three tiles from the surface, inside the redistance reach. The arena rewrites their 4h lattice vertices every step the surface moves, so a changed-tile list covers almost every seam tile in the post-redistance resolve. Without P1, advect and couple also change far values every step.

The resolve runs about 4 launches per step, at about 0.24 ms each in xctrace. An exact skip would need a snapshot of the 27 lattice values per seam tile, and I estimate it saves under 0.2 ms.

geometryRegional is already driven by the frame plan's regions and producer revisions.

### P2: coupleFine, surfaceClear and splat over particle tiles. Ruled out earlier.

The earlier measurement (`e980affd`) found that coupleFine's time is the gather and finish, not the copy, so skipping tiles with no particles cannot help. The bodies agent owns coupleFine's body loop.
