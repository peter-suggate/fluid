# Lane E progress: ubaStage, B4 (projection to tail), census and T

Design: `docs/plans/brick-resident-step-2026-11-08.md`. Lane T owns B1-B3.

## ubaStage interface (for Lane T)

`lib/methods/uniform/uniform-brick-stage.ts`, `ubaStageWGSL(arena, K, options)`.
It works on top of the arena table (`arena.tableWGSL`) and pool (`arena.poolWGSL`).
There is one call for each fused kernel K, and any number of channels.

```ts
ubaStageWGSL(arena,"K",{
 level:0|2,                 // 0: h bricks (live slots); 2: 4h pages (stored blocks)
 workgroup:N,               // >= 32 lanes; record staging uses lanes 0..27
 resolve?:"fn",             // fn(word:u32, neighbourOrigin:vec3i)->u32: once a neighbour, the base a core fn needs
 fineOwned?:"fn",           // level 2: fn(c:vec3i)->bool; default: the 4h cell's tile is live in the level 0 table
 channels:[{
  name, layout:"cell"|"face"|"vertex", axis?:0|1|2 (face), halo:0..8,
  core:"fn" | {pool, component?, bank?},   // fn(word, base, l:vec3u)->f32, l = local owned index 0..7
  seam:"prolong" (level 0) | "restrict" (level 2) | {fn},  // {fn}: fn(p:vec3i, word:u32)->f32
  coarse?:"fn",   // prolong: fn(c:vec3i)->vec2f (value, known>0) at 4h owned index c
  fine?:"fn", weight?:"fn",  // restrict: fn(p:vec3i)->f32 at h owned index p
  outside?:"fn",  // fn(p:vec3i)->f32 where no page is stored (SOLID/AIR); default 0
 }]})
```

Generated WGSL:
- `ubaStage_K(slot, lane)` stages the record, then every channel.
- `ubaRecord_K(slot, lane)` stages the record only. `ubaStage_K_<ch>(lane)` stages one window, so a pass can stage channels one after another as it uses them.
- Inside a window, `ubaGet_K_<ch>(c)`, `ubaPut_K_<ch>(c, v)` and `ubaAt_K_<ch>(c)` read and write it. `c` is relative to the brick's first cell, `-halo .. 8+halo`.
- The record accessors are `ubaNeighbour_K(n)`, `ubaMask_K(n)`, `ubaBase_K(n)` and `ubaFirst_K()`, where `n = x+3y+9z` over `(-1..1)³`.
- Every entry contains barriers, so call it from uniform control flow. Each window is its own workgroup array. Their sum plus 128 words must fit in `UBA_STAGE_WORDS` (8192 words, 32 KiB).
- Owned indices follow the arena's owned-high layouts. A cell is p. A face is the +axis face of cell p. A vertex is lattice vertex p+1.

The seam rules are cited in the file header:

| level | layout | seam rule |
|---|---|---|
| 0 (prolong) | cell | 4h value plus the measured central, one-sided or zero slope (d7fa24db; exact for linear fields) |
| 0 (prolong) | face | linear between the 4h owner's two faces on the axis, constant across it. The h subfaces on a 4h plane carry the 4h face exactly, so the V-weighted sum is the coarse flux. |
| 0 (prolong) | vertex | trilinear in the known 4h corners (the hanging vertex rule) |
| 2 (restrict) | cell | V-weighted mean of the 64 h cells |
| 2 (restrict) | face | V-weighted mean of the 4x4 subfaces on the face plane (the reflux) |
| 2 (restrict) | vertex | the coincident h vertex |

Regression: `tests/uniform-brick-stage.test.ts`.
- The fixture is 4x2x2 pages with an AIR page, and half-live bricks whose odd id leaves only the low-z tiles live.
- Every staged value is checked against a CPU twin of its case: core through the record, core through a resolver, prolong, restrict, or outside.

## Baseline: HEAD 334745cb, clean export

| scene | step median (p95) | drift | notes |
|---|---|---|---|
| seep 300 | 31.5 ms (34.0) | +3.2e-4 | right tank 0.00145 m³ at step 300 |
| letters 120 | 57.7 ms (88.9) | -0.0092 | 995k particles |
| rest 16x8x4 (60, pipelined) | 4.11 ms (15.9) | 0 | steady median 3.80 |
| drop=2 16x8x4 (60, pipelined) | 16.50 ms (26.5) | -6e-8 | steady median 16.42 |

## Decisions

- **ubaStage is a module of its own, beside `UniformBrickArena.stageWGSL`.** That older helper (used only by the pressure arena's level 2 octant passes) stages seams through a caller fn. It is pressure-side, so it is left alone. ubaStage adds the h↔4h rules, a level 2 restrict path and a per-kernel budget check.
- **B4 extension plan:** a fused 4h pass, then an h pass. Together they replace the paired retain/prepare/finish sweeps and the hierarchy restrict, sweep and prolong.
  - The 4h pass restricts the h faces and sweeps at 4h.
  - The h pass stages with halo 2 (prolonged 4h neighbours), sweeps in threadgroup memory, and publishes the canonical scratch.
  - Cited methods: Zhao fast sweeping (2005); Adalsteinsson & Sethian (1999) for the extension velocity; Bridson (2015), §4 for extrapolation.

## Milestones

1. ubaStage. Generator, `baseExpression` made public, GPU regression.
   - No frame path changes, so the scene gates are the baseline rows above.
   - `npm run check:types` passes. `npm run test:unit` reports 1 failure, `reservoir-particle-cost.test.ts`, "merge cap protects neither the outer layer nor mass-saturated deep runs".
   - That test passes at clean HEAD. It fails only with the peer's uncommitted NB-FLIP merge hunks in `uniform-narrow-band-flip.ts`, so it is not this lane's.
