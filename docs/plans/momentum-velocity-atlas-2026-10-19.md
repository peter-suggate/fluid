# NB-FLIP grid momentum through the texture unit (phase 1), 2026-10-19

Scene: `nbflip-figure-7-pour`, frames 94–100, dt 1/60 s. The baseline run is
`vg1` at 59.02 ms per frame, on HEAD 9ed83e12.

## What changed

**One RK2 step of the whole dt, with no substep limit.**
- The 1.5-cell substep loop is deleted from both momentum characteristics:
  - `uniformVelocityDepartureWGSL`, the general sampler;
  - `umUnitTrace`, the unit sampler.
- A face's departure is now a single midpoint step, `p − dt·u(p − ½dt·u(p))`.
  It is clamped exactly as before (`umClampMomentum`: walls and SOLID pages).
- The tiers pass's affine admission still assumes a single step. That stays a
  valid, conservative check.

**The unit sampler is the only fine sampler.**
- `uniformCachedMomentumSamplingWGSL(sampler)` samples three things:
  - the unit halo (35³ texels per block). This already holds exactly the unit
    interpolant's taps, including the low-wall faces of SOLID cells;
  - the 4h sampling cache;
  - the support-band blend weight.
- A sample with fine weight always has a stored halo home: a support-bit tile
  is stored, and the halo holds its low-wall taps. So no face escapes, and
  `momentumDeferred` together with its list buffer is deleted.
- The general sampler in the same module already read this halo for such
  samples, so deleting the escape changes no value.

**Kernels**

| Before | After |
|---|---|
| `momentumUnitStep`: workgroup per general h tile, substepped, with escapes | `momentumFine`: workgroup per general h tile, one face per lane, single step |
| `momentumDeferred`: claimed list of escaped cells | (deleted) |
| `momentumStep` merged (seam and regular 4h jobs), general sampler | the same dispatch, evaluating `umUnitMomentum`: the unit sampler |
| `momentumRegularStep` | unchanged (`umRegularFine`, all-h stencils) |

**Sampler param**

The NB-FLIP method param `momentumSampler` has `update:"solver"`, so changing it
restarts the solver.

| Value | What it does |
|---|---|
| `texture32` (default) | `textureSampleLevel` from one call site, through a linear clamp sampler. It reads the rgba32float unit halo and the rgba32float 4h cache, bound a second time as filterable textures (`unitHaloFiltered`, `coarseFiltered`). Needs `float32-filterable`. Without it, the frame selects `texture16`. |
| `texture16` | The halo fill (`fillUnitHalo`) also writes an rgba16float copy of the halo (`UniformMixedHangingTaps.halfHaloOf`), and the texture unit filters that copy. The 4h interpolant stays in software. |
| `software` | Today's eight loads and f32 blend: exact, apart from the step change. |

Devices:
- Production (`fluidExecutionDeviceFeatures`) requests `float32-filterable`
  when the adapter has it.
- The test device helper (`withUniformDevice`) does the same.

The texture coordinate is `(q + home + 0.5) / extent`.
- The filter's lower tap is therefore the software path's `floor(q)`, and its
  weights are `fract(q)`, rounded to the filter's subtexel precision.
- Seams have exactly today's semantics, because the texture reads the same
  halo texels and the same 4h cache texels the software sampler reads.
- The phase-0 bench's maximum absolute difference of 5.7 came from the bench,
  not the product. The bench sampled a trilinear 4h texture where the product
  uses the cache's taps. The product carries no such difference.

## Deviations from the phase-0 design, and why

- **No separate 6³ apron-brick atlas.**
  - The unit halo is that atlas already: an h-resolution, apron-padded block
    per page, holding exactly today's hanging taps and solid rules.
  - It is produced every frame anyway, for the other consumers: surface advect,
    the NB-FLIP snapshot/update, and particles/APIC.
  - A second atlas would duplicate the build and add a copy of the field. So
    phase 1 filters the halo, and the 4h cache, in place.
  - The 4h cache, `unitVelocityTaps`, `markUnitHalo` and `fillUnitHalo` all
    stay, because each has consumers besides momentum.
- **No tile-context extension yet.** Phase 1 needs nothing new per tile. The
  classification each sample still makes (`umSampleTile`, the blend weight,
  the halo home) is the next lever; see Next.
- **No lane-per-owner `momentumCoarse`.** The merged seam/regular 4h dispatch
  is kept, and it now evaluates through the unit sampler. That cut it from
  0.590 to 0.328 ms without a new list.

## Stepping schemes

The phase-0 bench compared the schemes on a synthetic scene: 690,637 live
faces, mean |u| 1.9 cells/frame, accelerating to 15 cells/frame. The reference
is RK2 with 0.5-cell substeps, sampled through hw32.

| Scheme | ms | Mean relative error | p99 | Max | Vertical bias |
|---|---|---|---|---|---|
| Old: 1.5-cell substeps | 0.895 | 5.2e-4 | 0.008 | 4.18 | 9.2e-4 |
| **(a) single RK2 midpoint on the h halo** | **0.262** | **1.6e-3** | **0.055** | 5.18 | 1.8e-3 |
| (b) single RK2, 4h-cache midpoint | 0.220 | 8.2e-3 | 0.185 | 6.32 | −6.6e-3 |
| (c) single forward Euler | 0.144 | 7.6e-3 | 0.25 | 10.3 | +0.016 |

- No scheme produced a non-finite value.
- The maxima of about 4–5 sit on the synthetic field's velocity jumps.
- **(a) is the default:**
  - It stays bounded at 15 cells/frame and does not blow up.
  - Its bias is small (1.8e-3 cells/frame).
  - It is 3.4× faster than substepping.
- (b) gives 5× (a)'s error for a 16% saving.
- (c) carries a visible vertical bias of +0.016 cells/frame and the worst
  maximum error.
- M6, skipping faces that the transfer replaces, stays off.

## Measured: pour, frames 94–100

Each run uses one profiled, split probe: `P/pr.sh`.
- `vg1` is the baseline: HEAD, software sampler, substeps.
- `mom1` is the new default (texture32). `mom1sw` and `mom1h` select
  `software` and `texture16`.
- Particle counts differ between runs (818k / 878k / 737k / 766k), so the
  frame totals are noisy. Per-pass momentum times are comparable.

| Pass (ms) | vg1 | mom1 (texture32) | mom1sw (software) | mom1h (texture16) |
|---|---|---|---|---|
| momentumUnitStep / momentumFine | 2.575 | 0.871 | 1.245 | 1.142 |
| momentumDeferred | 0.103 | – | – | – |
| momentumStep (merged 4h) | 0.590 | 0.328 | 0.449 | 0.459 |
| momentumRegularStep | 0.028 | 0.009 | 0.009 | 0.019 |
| **Momentum total** | **3.296** | **1.208** | 1.703 | 1.620 |
| fillUnitHalo | 0.187 | 0.197 | 0.187 | 0.253 |
| Frame GPU | 59.02 | 57.11 | 53.66 | 55.53 |

texture16 is slower than texture32 in the product, for two reasons:
- its 4h interpolant stays in software;
- the halo fill pays about 0.06 ms for the half copy.

It remains only as the fallback for devices without `float32-filterable`.

## Speed of light

The bound for one RK2 step, read through hw32:
- 0.69M live faces × 7 component samples (3 + 3 + 1) = 4.8M filtered samples;
- at 13.5 G samples/s (rgba32float, measured), that is 0.36 ms;
- the 4h owners add about 0.02 ms, and dispatches about 0.01 ms;
- so the bound is about 0.39 ms.

| | Measured | Bound | % of bound |
|---|---|---|---|
| `momentumFine` | 0.871 ms | 0.36 ms | 41% |
| Momentum stage | 1.208 ms | 0.39 ms | 32% |
| Old stage | 3.296 ms | 0.39 ms | 12% |

The bench's lean single-step kernel (M7 shape, hw32) runs at 0.26 ms. The
product kernel's remaining 0.6 ms is per-face and per-sample work outside the
filter:
- `umSampleTile` with its 8-cell hold box, the blend-mask loop, the halo home
  and the coarse home, three times per face;
- the retained, cull, freeze and drop-face tests per face;
- in texture16 and software, the eight-load 4h interpolant.

## Reconciliation with the particle atlas

The particle research proposes one velocity atlas of apron bricks for both
momentum and particle G2P:
- 10³ faces per 8³ brick per component;
- holding (u, u*): FIN writes u*, and u is built after extension;
- 1.95× overhead, against 3.4× for the per-tile 6³ bricks of the phase-0
  design.

Phase 1 keeps the fine atlas's layout behind a small interface, so that
switch touches no momentum kernel:
- `umUnitHaloHome` resolves a sample's home once;
- `umAtlasTexel(q)` maps a tap coordinate to that atlas's texel;
- `UniformMixedHangingTaps` builds the atlas (`fillUnitHalo`, plus
  `halfHaloOf` for the rgba16float copy).

Both interpolants and the trace read the atlas only through `umAtlasTexel`.
A brick-apron layout replaces those three pieces and the bindings
(`unitHaloFiltered`), and momentum then reads the u channel.

## Next

- Move per-sample classification into the per-tile context
  (`uniform-tile-context.ts`), as the phase-0 design laid out:
  - a sample's tile key, blend weight and homes resolved once per tile window,
    instead of three times per face;
  - the per-face tests (retained, frozen, drop) hoisted to the tile.
  
  This is the gap between 0.87 and 0.26 ms.
- Remap, geometric fill and forces, following the phase-0 stage design.
