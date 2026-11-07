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
| rest 8x8x4 (60, pipelined) | 2.94 ms (11.6) | 0 | |
| drop=2 8x8x4 (60, pipelined) | 13.58 ms (18.9) | | |

Non-pressure excess for 8x8x4 drop=2 against rest. Source: Lane T's base run at 334745cb, 32 steps, `--kernels=29`, `lane-t/excess.py`.

| scope | excess |
|---|---|
| all non-pressure | **8.87 ms** (drop 15.60, rest 3.53; pressure 5.63 / 2.43) |
| extension, hierarchy, snapshot, recordDelta (B4) | 1.14 ms |
| plan tail, certificate, heat, dynamics publish (B4) | 0.80 ms |
| census, layout, remap, residency, fill, hanging, 4h cache, tiers, transfer (T) | 2.35 ms |

The B4 and T rows are regex groups of stage labels, so they are approximate.

## Decisions

- **ubaStage is a module of its own, beside `UniformBrickArena.stageWGSL`.** That older helper (used only by the pressure arena's level 2 octant passes) stages seams through a caller fn. It is pressure-side, so it is left alone. ubaStage adds the h↔4h rules, a level 2 restrict path and a per-kernel budget check.
- **B4 extension plan:** a fused 4h pass, then an h pass. Together they replace the paired retain/prepare/finish sweeps and the hierarchy restrict, sweep and prolong.
  - The 4h pass restricts the h faces and sweeps at 4h.
  - The h pass stages with halo 2 (prolonged 4h neighbours), sweeps in threadgroup memory, and publishes the canonical scratch.
  - Cited methods: Zhao fast sweeping (2005); Adalsteinsson & Sethian (1999) for the extension velocity; Bridson (2015), §4 for extrapolation.

## ubaExtend (B4's extension core)

`lib/methods/uniform/uniform-brick-extension.ts`, `ubaExtendWGSL(arena, K, {level, workgroup, sweeps, spacing, reach, components, write})`.
- **Per component:** it stages the value window and the seed distance window with halo = sweeps. It runs `sweeps` Jacobi sweeps of today's `ueCombine` in threadgroup memory: the Godunov upwind root, the band cut-off at `reach` cells, and the upwind-weighted extension value. It hands each core face's (value, distance) to `write`.
- **Domain:** a seed distance of 0 marks a source, 1e20 an unreached face, and −1 (`UBA_BLOCKED`) a face outside the domain. A blocked face is never swept and is never an upwind neighbour. It is the default outside the stored pages. Seam faces are swept like any other face.
- **Exactness:** sweep s computes the window inset by s+1. The core therefore equals that many global Jacobi sweeps, which is today's slot-array result on uniform-width regions, with no seam jobs.
- **Memory:** 4 windows of (8+2·sweeps)³ words, reused across components. At sweeps=2 that is 6912 words plus the record.
- **Channel sharing:** `ubaStage` gained `window` (a channel reuses an earlier channel's array) so a kernel can stage one field after another.
- **Regression:** the third test in `tests/uniform-brick-stage.test.ts` compares each brick's core against two global Jacobi sweeps over the live faces, on anisotropic spacing.

## B4 integration plan (not landed)

**What blocks it:** no production level-0 arena exists outside pressure.
- The frame has no slot space for its h bricks. `UniformFineBrickStore` (the design's one store) is built only by its test, and pressure's arena is pressure's.
- B1 (Lane T) needs the same frame store, for its inboxes. It should be built once and shared: whichever lane wires it first owns it, and the other adopts it.

**Wiring the extension, once the frame store exists:**
1. **Level 2 pass** (pages, 4h spacing).
   - Seeds: `umSource`, the closed and low walls (the negative buffer).
   - Restrict seams wherever the 4h cell is h-owned.
   - It writes the 4h extended state at the base-block texels, the canonical format today's consumers read.
2. **Level 0 pass** (h bricks).
   - The core fn reads `f.velocity` through the resolver: the brick's texel origin is umFineSlot(block)<<5 plus 8 times its local index.
   - Prolong seams take the level 2 result (value and distance).
   - It writes `velocityScratch` and `negativeScratch`.
3. **Far faces** (past the 4h band): the design's lagged fixed point (§5 risk 1). They keep last frame's extended value.
   - This replaces the hierarchy pyramid. Its restrict, sweep and prolong (`webgpu-uniform-velocity-extrapolation*`) cost about 0.37 ms of round 12's excess.
4. **Paired FLIP delta:**
   - The old field runs the same kernel on `f.departure` into an old-extended texture.
   - `recordDelta` (`uniform-narrow-band-flip.ts`) reads the two textures instead of the companion slot arrays.
   - That file carries a peer's hunks, so the change lands as a patch through safe-commit.
5. **COLD regions:** a brick whose 16h region is COLD exits at once (the heat tiers). The held, regional and replay machinery has no counterpart. A fused pass is cheap enough to recompute.
6. **Deleted in the same commit:**
   - `uniform-mixed-extension.ts`, `-pair.wgsl.ts`, `-slots.wgsl.ts` and `uniform-compiled-extension.wgsl.ts`;
   - the hierarchy, where it has no other user;
   - the frame's retainPair/preparePair/finishPair/finishSnapshot branches and the reusableExtension/bodyExtension bookkeeping.

## Milestones

1. ubaStage. Generator, `baseExpression` made public, GPU regression.
   - No frame path changes, so the scene gates are the baseline rows above.
   - `npm run check:types` passes. `npm run test:unit` reports 1 failure, `reservoir-particle-cost.test.ts`, "merge cap protects neither the outer layer nor mass-saturated deep runs".
   - That test passes at clean HEAD. It fails only with the peer's uncommitted NB-FLIP merge hunks in `uniform-narrow-band-flip.ts`, so it is not this lane's.
   - Commit f1da3003.
2. ubaExtend, the fused extension core, with its regression. The stage window sharing landed with it.
   - Nothing in the frame uses it yet, so the scene gates are the baseline rows.
   - Commit d78a265f.

## Status, stopped at the shared frame store

- ubaExtend is wired into nothing, so the excess is still 8.87 ms. B4 and T both need a production level-0 brick slot space for the frame's h tiles. None exists outside pressure.
- B1 (Lane T, the design's step 0 go/no-go) needs the same store for its inboxes.
- Two lanes building two stores would be a dual path. Which lane builds the shared frame store (`UniformFineBrickStore` on the frame's ownership) is therefore a coordination decision. Its owner should settle it before B4 is wired.
- Once it exists, the B4 integration plan above lands as one commit. The T piece follows: the census as a reduction, with ubaUpdate run only on frames whose marks changed.

