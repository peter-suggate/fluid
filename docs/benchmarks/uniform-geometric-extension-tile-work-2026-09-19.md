# Uniform Geometric: 4h work map for the velocity-extension front

**Status: measured, NOT shipped — the implementation was reverted.** The whole
change (five source files, the Dawn test and the benchmark tool) is archived as
a single re-appliable patch at
`docs/research/patches/uniform-geometric-extension-tile-work-2026-09-19.patch`.
Nothing in `lib/` carries it; there is no `extensionWorkMap` parameter.

**Verdict against the bar.** The map is provably bit-exact and measurably
faster, but it saves **2.1%** of the mini64 frame and **0.9%** of the
large-power frame, with the whole-frame wall gain inside frame noise on both
scenes. Only the GPU stage timestamps are decisive. The bar was a significant
whole-frame win (~3%+, against the sharpening map's 4–7%), so it was not
accepted. The measurement, the corrected exactness bound and the diagnosis of
where the extension's time actually goes are the deliverables.

## Result

The map skipped the Sec. 3.3 narrow-band FIM front (seed / update / resolve) in
4×4×4 tiles that provably cannot reach the accurate band, and left the hierarchy
restrict/prolong and the transport-shell pack untouched.

Dawn/Metal, production method values, geometric volume, balancing off. Four
fresh runs per scene in dense/tiled/tiled/dense order; 30 frames each, first
three excluded; run medians then averaged. Frame wall time fences the queue.
Stage spans are hardware timestamps on observable marker passes behind copy
barriers, identical in both arms; the tiled front span includes classification.

| Scene | Dense front | Tiled front | Dense extension | Tiled extension | Dense frame | Tiled frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mini64 dam | 5.767 ms | 4.555 ms | 6.095 ms | 4.981 ms | 55.205 ms | 54.037 ms |
| large-power dam | 2.359 ms | 2.032 ms | 2.589 ms | 2.228 ms | 37.435 ms | 37.083 ms |

The front is 21.0% faster on mini64 and 13.9% faster on large-power. The whole
extension stage saves 1.114 ms and 0.360 ms = 2.02% and 0.96% of the respective
dense frames; measured whole-frame savings agree at 1.168 ms (2.1%) and 0.351 ms
(0.9%).

The stage saving is real and repeatable — the two repeats of each arm produced
identical stage medians (tiled extension 4.981/4.981 and 2.228/2.228 ms) and the
front quartiles barely overlap. The whole-frame wall gain is not: wall IQR is
≈2 ms on mini64 and ≈2.6 ms on large-power, against savings of 1.2 ms and
0.35 ms. Only mini64 survives as a ranked median, and only just (dense run
medians 54.761–55.649 ms versus tiled 53.738–54.336 ms). The adjacent JSON holds
every per-frame sample.

## Why it falls short of its ceiling — the extension front is dispatch-bound

Ceiling = the dense front's frame share times the far-tile fraction:

| Scene | Front share of frame | Far tiles | Ceiling | Achieved | Fraction of ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| mini64 dam | 10.4% | 43.4% (1778/4096) | 4.53% | 2.02% | 45% |
| large-power dam | 6.3% | 79.2% (1014/1280) | 4.99% | 0.96% | 19% |

**The front is not body-work bound at these sizes.** Removing 79% of the
per-cell work on large-power removed only 14% of the front, implying roughly
**83% of that front's time is fixed cost**: ~16 update + prepare pairs per step,
each an indirect dispatch (repo memory: indirect ≈15–25 µs against direct ≈3–6
µs, and a zero-workgroup indirect is not free), plus the two rgba32float stores
every far lane must still make to keep the ping-pong textures exact. On mini64,
3.2× the cells with liquid filling more of them, body work is about half the
front and the map recovers 45% of its ceiling. **The lever for this stage is FIM
pass count and direct rather than indirect dispatch, not spatial work.**

That fixed cost is per-step, not per-cell, while the body work the map removes
scales with the lattice; so the map's share of the front should grow on much
larger domains, and a far-tile fraction that stays high there would matter more
than it does here. That is an expectation from the structure of the numbers
above, not a result — it was never measured, and the two scenes here are the
only evidence.

## Mechanism and correctness (as built, for anyone re-applying the patch)

One u32 record per 4³ fine workgroup, in a dedicated `8 + 2N`-word buffer on a
new binding (binding 12), built per `encode` because the authority pass rewrites
the density texture before each call:

1. `classifyExtensionTiles` — one 4×4×4 workgroup per tile, workgroup-atomic
   any-liquid reduce over exactly the shader's own liquid predicate on exactly
   the texture `density()` reads. The comparison is negated, so a non-finite
   density is conservatively liquid.
2. `dilateExtensionTiles` — 27-neighbourhood OR into the second plane.
3. `countExtensionTiles` — near-tile tally into the header, for diagnostics.

A far-tile thread in seed/update/resolve writes the inert constant
(primary `(0,0,0,known=0)`, secondary `(INF,INF,INF,active=0)`) to both outputs
and returns before any texture load. It writes rather than skips: the ping-pong
textures still hold the previous frame's state where the tile was near. Every
texture the hierarchy and pack stages read is therefore bit-identical to dense
by construction, with no neighbour-read gating and no change to those kernels.

### The exactness bound, corrected — keep this

Distances are written only when `updatedDistance <= accurateBandDistance()` =
2·max(h), and a cell only activates when its Godunov distance is within that
band. The obvious argument — that a first-order Godunov hop raises the distance
by at least one lattice step of min(h) — **is wrong**: the three-axis solve
admits a hop as small as **min(h)/√3**. The band therefore admits
`floor(2·√3·max/min)` hops, not `floor(2·max/min)`. For cubic cells that is 3
hops, and a source face sits one cell from its liquid cell, so the reach is
**4 cells — exactly one tile of dilation, with zero margin**, not the 3 cells
the naive bound gives.

Two further traps:

- The `max(b·b − a·c, 0)` clamped-discriminant branch of `godunovDistance` has
  no bound on its increment. It is unreachable while **2·(max/min)⁴ ≤ 3**. Both
  inequalities were checked at construction, and anything failing either kept
  the dense schedule. All three benchmark scenes are exactly cubic.
- In the resolve pass's bind group, binding 1 is `distancesB` and `baseDims()`
  is the padded (n+2)³ extent. A classifier cannot reuse `density()`/`baseDims()`
  there, and padded lanes must take the dense path — the real dims have to
  travel in the map header.

The map was built only for the geometric-volume solver and only when the
extrapolator's AABB `activeDispatch` path was absent. The tile-work shader
source was emitted conditionally, so the non-geometric uniform method's module
text stayed byte-identical to HEAD (verified: 22570 chars); dense and tiled
pipelines then came from one module differing only by the
`UV_EXTENSION_TILE_WORK` pipeline-overridable constant. Classification used
three non-atomic passes rather than storage atomics in the hot kernels (repo
memory: `atomicLoad` is an FP barrier).

Validation, all green before the revert:

- `tests/uniform-volume-dawn.test.ts` plus the archived
  `tests/uniform-volume-extension-tile-work-dawn.test.ts`: 12 tests pass under
  Dawn. No ceiling raised, no test loosened.
- Synthetic cases match dense bit-for-bit with deliberately poisoned ping-pong
  and resolved textures: centred blob, blob against the boundary with
  non-multiple-of-4 dims (18×16×16), all air, all liquid, a single cell on a
  tile corner, a tile-corner pair whose bands cross diagonal neighbour tiles,
  and an isovalue-exact plus NaN cell. Non-vacuity asserted on the dense arm.
- Real-frame replay: identical surface/velocity fields pushed into a dense and a
  tiled extrapolator at the authority seam, comparing resolved values, resolved
  distances, the packed transport shell interior and the four convergence
  diagnostics words, bit-for-bit on every one of 30 mini64 and 30 large-power
  frames. Live on/off/on toggling moved no bit.

## Reproduction

The source is reverted, so reproducing this requires re-applying the patch
first:

```sh
git apply docs/research/patches/uniform-geometric-extension-tile-work-2026-09-19.patch
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test --test-concurrency=1 tests/uniform-volume-dawn.test.ts tests/uniform-volume-extension-tile-work-dawn.test.ts
node --import tsx tools/benchmark-uniform-geometric-extension-tile-work-dawn.ts
```

The benchmark asserts neither scene's tiled whole-frame median exceeds its dense
control, and writes the adjacent JSON with every per-frame sample before that
assertion. `--quick` runs a six-frame dense/tiled pilot; its three measured
samples per arm are far inside the noise floor and decide nothing.

Had it shipped, the product surface would have been an `extensionWorkMap` select
(fine tier, runtime update) mirroring `sharpeningWorkMap`, info fields
`uniformExtensionWorkMap` / `uniformExtensionTilesNear` /
`uniformExtensionTilesTotal`, and a `4h work map · <pct>% tiles` chip plus a
`Near tiles` readout on the velocity-extension stage of the SIM pipeline panel.
