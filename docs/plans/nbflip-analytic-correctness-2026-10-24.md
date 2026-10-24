# NB-FLIP analytic correctness suite (2026-10-24)

Short, isolated scenes whose motion has an exact or classic answer, used to
find and fix general correctness defects in narrow-band FLIP and its
COLD/WARM/HOT region tiers. Every fix is general (no per-scene rules) and adds
no work pass.

## The suite

`lib/core/nbflip-analytic-scenes.ts`, listed in the app on the validation shelf
"NB-FLIP analytic". One closed free-slip box of 64 x 64 x 32 cells at 1 cm,
dt = 1/60 s, liquid spanning the depth (two-dimensional flow), so one Dawn
device builds the pipelines once.

| case | frames | answer |
|---|---|---|
| rest | 60 | u = 0, level surface at 0.24 m, every region COLD |
| rest-step | 60 | as rest, over a submerged step and around a pier through the surface |
| free-fall | 16 | rigid translation, y = y0 - g t^2 / 2, grid velocity g (t + dt/2), no deformation |
| standing-wave | 60 | fundamental seiche, ends follow A cos(wt), w^2 = g k tanh(kH), middle level |
| dam-break | 18 | surge front against Martin & Moyce (1952), n^2 = 2 |
| mound | 120 | total energy never rises; the surface returns to rest |

`tools/probe-nbflip-analytic.ts` runs any subset (`--case`, `--frames`,
`--set=key:value`) and scores each case. Every case: phi and fill-field volume
drift <= 1%, late tier changes per live region-frame (0 for rest cases, <= 0.05
otherwise), reversals (a tier changing back to the one held 2-3 frames
earlier: flicker) <= 1%. The whole suite takes about 40 s.

Readback notes the probe encodes:

- `readMixedTexture(device, solver.velocityTexture)` without an address is the
  logical lattice; passing the live address reads zeros.
- Velocity faces live at each owner's +face anchor (origin + w - 1 on the
  axis); a cell's -face is its neighbour's +anchor.
- The grid velocity after a step is the leapfrog half-step value.
- Surfaces are measured above solids: buried padding phi is not a surface.

## Gaps (not covered)

- Tilted gravity: the forces read `gravity_m_s2.y` only
  (`webgpu-uniform-reference.ts`, the force params), so a tilted free fall is
  silently vertical. The suite uses vertical gravity.
- Initial velocity: NB-FLIP does not consume `initialVelocity_m_s`, so uniform
  translation and rotation tests need that plumbing first.

## Defects found and fixed

### Round 1

1. **Start-up heat seed** (`68ea9af6`, uniform-region-heat.ts). Missing tile
   certificates on the first frame OR'd into the measured detail trigger and set
   heat 2, so a pool at rest was HOT for 15 frames and WARM for 15 more. Unknown
   evidence is now the existing one-frame held HOT, with no heat (Sato: heat
   comes only from a measured high-pass or curvature). Rest goes COLD at frame 4
   instead of 31.
2. **Thaw/refreeze relaxation oscillation** (`68ea9af6`). A COLD region passed
   its freeze budget, refreshed for one frame plus the hold, refroze and passed
   it again: WWWCCWWWCC in a smooth standing wave. A refreshed region now
   refreezes only if its current omission rate stays within budget over the
   heat's 0.5 s horizon. Detail heat also grows with its excess (Sato Eq. 5,
   A_u = 5), so a marginal crossing is HOT for a few frames, not 0.25 s.
   Standing wave: late changes 0.156 -> 0.017, reversals 0.014 -> 0.001.
3. **Retirement measured phi through solids** (`67627152`,
   uniform-narrow-band-redistance.wgsl.ts). Once heat decays, coarse-corner
   retirement re-measures metric distance at 4h corners. A corner whose eight h
   cells are all solid, but which a cut 4h owner reads, took the distance
   through the solid to the contact line (-2h, 2h inside a pier) instead of the
   owner's plane continued into the solid. The bent surface made the band read
   the free-surface face at the pier as a wall, keeping the full gravity step:
   0.17-0.29 m/s of spurious flow in a still pool. Fully closed corners now
   keep their plane. rest-step max |u| 0.29 -> 6e-5 m/s, no particles seeded.
4. **Trilinear phi resample at 4h** (`0f226810`, uniform-mixed-frame.ts).
   `phiCubicAdvection: off` (the default) mapped to surface flag 0, turning the
   cubic off at 4h vertices too. No particles carry a 4h surface, and a
   trilinear resample erodes a moving cornered or curved zero set 16 times
   faster there than at h: a particle-free falling block lost 22% in 14
   frames. Off now selects the existing variant 2 (cubic at 4h, trilinear at
   h). The block holds its volume within ~3% until it lands.

### Round 2

5. **Free-fall "internal flow" was the probe** (`571b39d3`). With volume
   control off every liquid face stays exactly uniform through step 15. The
   projection zeroes faces between two air owners, so liquid in a partly
   filled owner whose centre is air read half or no speed. The probe now reads
   the extended field (what the next step transports) and takes extent from
   phi. The remaining small non-uniformity is volume control injecting a
   uniform divergence to close the corner-rounding gap, as designed.
6. **Translation made regions HOT** (`853470a0`, uniform-region-heat.ts).
   Error-mode detail took max(tile defect, half the step's travel): any body
   moving over 2 cells per step went HOT with 0.5 s of heat. HOT buys a
   uniform field nothing (WARM momentum is already the full trace above 1.5
   cells per step, and HOT changes neither h resolution nor seeding). Detail is
   now the tile defect alone, Sato's translation-insensitive high-pass (Eq. 3).
7. **Frozen regions re-charged solver noise** (`853470a0`). A frozen region's
   projection re-sees the accepted solve's leftover divergence every frame and
   the freeze budget charged it every frame, thawing a still pool beside a
   pier after ~18 frames. Omission is now charged above the band's stopping
   residual r (per-frame correction above r h, per-step displacement above
   r dt). Risk: a steady drift slower than ~5e-4 m/s per step (a slow drain)
   is never charged.
8. **Clipped wall vertices kept stale far phi** (`24d05d06`,
   uniform-mixed-surface.ts `umWallContact`). In "clip" wall tracing a
   wall-plane vertex departed by its own trace, which has no normal travel, so
   its continuation sampled air and was discarded; the floor vertex under a
   landing 4h tile kept its t = 0 value and placed the block's bottom 2.5 cells
   high (14% lost in one step). The vertex now departs on each contact axis by
   the inside point's normal speed and takes the lesser value. Free-fall drift
   0.144 -> 0.069. Particles now first seed at frame 13 rather than 16.

Diagnosed, not fixed in round 2:

- The Godunov extension copies the nearest source value, so the first air
  layer is not divergence-free: a partly filled owner at the front is squeezed
  by up to 13% of its volume per step (the largest common loss in dam-break,
  mound and post-impact free fall). Round 3.
- Census coarsening and coarse retirement round curved and cornered surfaces
  at 4h (mound 0.5% at step 1; free-fall 3% at step 1 and 0.03-0.08 L/frame
  later). Round 3.
- Semi-Lagrangian advection across the impact's velocity jump (departures 4.4
  cells, grad u dt ~ 1) is not conservative; fixing it needs substeps, which
  are ruled out.
- Residual divergence in full cells after a 3-cycle band solve at impact
  (~7% per step in floor h cells); tightening it adds cycles.

### Round 3

9. **The census ignored one-signed curvature error** (`9af2baa2`,
   uniform-mixed-dynamic.ts `umSurfaceDisplacement`). Coarsening tested only
   the largest crossing displacement against the 4h trilinear. A curved
   surface's trilinear error is one-signed, (L^2/12) laplacian(phi), so a convex
   body moved inward uniformly under the tolerance; by Gauss-Bonnet a closed
   section lost ~8.4 h^2 whatever its radius. A tile now also needs
   4 |mean signed displacement| <= tol. Mound retirement losses 27-54 ->
   0.4-8 cells per frame for ~13% more h tiles; flat surfaces coarsen as before.
10. **The probe's phi gauge** (`b26e0e02`, tools/probe-nbflip-volume.ts) read
    full 4h owners through one orientation-biased 6-tet split (an aligned 16^3
    block read 7.68 of 8.192 L); they now sum their 64 h subcells.

Implemented, measured, and held back:

- **Divergence-free first air layer** (Foster & Fedkiw 2001 per-axis closure in
  the extension combine and at publication; patch in
  `artifacts/solenoidal-extension-2026-10-24.patch`, ~85 lines). First-layer
  divergence on dam-break falls ~10x and every exact case is unchanged, but
  dam-break phi drift worsens 2.4% -> 3.3%: its old first layer *expanded* at
  frames 9-11 and hid a semi-Lagrangian loss at the collapsing column top.
  Energy rise: dam-break 0.122 -> 0.102, mound 0.205 -> 0.220 (volume control
  re-adding more). It adds slot reads to the extension sweep and publish
  kernels, unmeasured for frame time. Needs a decision.

Remaining losses, located but not fixed:

- **4h resampling of cornered surfaces** (dam-break frames 3-9, free-fall in
  flight; uniform-mixed-surface.ts `umCatmull`, ~606-671). A CPU twin
  reproduces the GPU loss (-3.5% by frame 10 for the falling block): inherent
  to resampling a cornered SDF at 4h. Fix needs a volume-preserving 4h
  resample, or keeping the moving surface at h.
- **Transit coarsening** (uniform-mixed-dynamic.ts `surfaceAllowed` ~1293).
  With detailSurfaceDistance 0 every non-crossing tile coarsens, including the
  predicted band, so a moving h surface is regenerated from the 4h
  interpolant. detailSurfaceDistance 1 cuts free-fall loss to ~2% but costs
  ~250 h tiles (from ~5). A work-versus-accuracy call.
- **Coarse retirement's metric store raises cornered air-side corners**
  (uniform-narrow-band-redistance.wgsl.ts ~161; free-fall floor contact,
  -114 cells). A min(old, metric) store would reinstate the particle-union
  magnitudes the metric store exists to remove.
- **Impact**: semi-Lagrangian advection across the velocity jump (departures
  4.4 cells) and the 3-cycle band solve's residual divergence in floor cells.
  Both need substeps or more cycles, which are ruled out.
- **Mound impact frames 7-10** (-0.4 to -0.5 L per frame): partly filled h
  owners classed liquid with large divergence; remap/relayout or the band
  solve.
- **Dam-break front** 12.8% behind Martin & Moyce (limit 12%), likely the same
  front volume loss.

## Results

Full suite, HEAD b26e0e02, default parameters, 1/60 s:

| case | pass | failing checks |
|---|---|---|
| rest | yes | |
| rest-step | yes | |
| standing-wave | yes | |
| free-fall | no | phi drift 6.7%, extent 1.04 cells, late tier changes 0.117 |
| dam-break | no | phi drift 2.5%, front 12.8%, energy rise 12% |
| mound | no | phi drift 4.8%, energy rise 20%, final KE 25% of peak |

At session start every case failed: rest was HOT for 15 frames and WARM for
15 more; rest-step developed 0.29 m/s of spurious flow and seeded ~4900
particles; the falling block lost 22% in flight and 14% more at impact; the
standing wave flickered (reversals 1.4%).

### Region tiers

- Still pools go COLD at frame 4 and stay COLD (rest and rest-step, 120
  frames), with no changes or reversals.
- The standing wave's liquid is uniformly WARM (late changes 0.017, reversals
  0.001); its air regions are COLD.
- Free fall is WARM while it translates and HOT only on measured detail;
  0.117 late changes against ~0.0625 that entry and exit alone require, the
  remainder at the leading and trailing faces and particle seeding from frame
  13.
- Dam-break and mound are mostly HOT from particle heat, as their splashing
  warrants; reversals stay under 0.6%.

## Open decisions

1. Land the divergence-free extension patch (principled; exposes rather than
   adds the dam-break advection loss; needs a frame-time check)?
2. Spend ~250 h tiles to keep moving surfaces at h (detailSurfaceDistance 1),
   or develop a volume-preserving 4h resample?
3. Plumb `initialVelocity_m_s` and tilted gravity into NB-FLIP so the suite
   can add uniform translation, rotation and a tilted fall.

Also failing, independent of this work (they fail identically before it):
`tests/uniform-hydrostatic-offset.test.ts` "a detached liquid source preserves
the pool and its solid boundary while refining", and
`tests/uniform-region-closure-walk.test.ts`.
