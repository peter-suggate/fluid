# Minidam64 small enforced region: performance diagnosis

## Reproduction

User scene: `minimal-power-dam-break-64`, default Uniform Geometric settings,
`regions=87.5_0_0_100_12.5_12.5_4_4`. This produces 4,069 h tiles,
19 transition tiles at 2h and eight enforced tiles at 4h. The all-fine control
uses the same algorithm and settings with no regions.

The live UI showed 86.77 ms/advance around simulation time 2.27 s: extension
6.82 ms, vertex level set 17.4 ms, momentum plus forces 35.3 ms, and multigrid
cycles 18.1 ms. A browser restart had resolved the earlier browser/Dawn timing
discrepancy; that is not an explanation for this region regression.

The regression reproduces in standalone Dawn with the browser unloaded.
Both comparison arms ran from the same bundled source snapshot, protecting the
comparison from concurrent edits. Snapshot and raw captures are in
`/tmp/fluid-small-region-repro/` (temporary, not checked in). `source.sha256`
identifies `probe.mjs`, and `inputs.json` lists its source inputs.

```sh
node --import tsx tools/probe-uniform-mixed-pressure-dawn.ts \
  --arm=fine --scene=minimal-power-dam-break-64 --tolerance=5 --steps=72
node --import tsx tools/probe-uniform-mixed-pressure-dawn.ts \
  --arm=mixed --region=87.5_0_0_100_12.5_12.5_4_4 \
  --scene=minimal-power-dam-break-64 --tolerance=5 --steps=72
```

For attribution only, add `--split-dispatches`. This separates momentum,
forces, surface advection and pressure-sweep dispatches by entry point and
specialization constants. It adds pass/encoder boundaries, so its timings are
for identifying expensive work, not replacing the unmodified-pass baseline.

## Measured costs

Median GPU pass time over matched frames 49–72 (1.63–2.40 simulated seconds):

| Work | All fine (ms) | Small region (ms) |
| --- | ---: | ---: |
| Momentum advection | 0.393 | 21.266 |
| Body forces | 0.131 | 14.877 |
| Pressure sweeps | 2.032 | 14.516 |
| Surface advection | 1.311 | 12.091 |
| Surface cell tracing | 0.328 | 2.884 |
| Surface redistance | 1.343 | 2.556 |
| All measured passes | 17.531 | 84.345 |

The total is the median of each frame's summed GPU pass times, not a sum of
individual medians. It excludes CPU gaps and rendering. The momentum UI node
combines advection and forces; it must not be compared to advection alone.
A second region run from the same solver snapshot measured 83.919 ms over
the same window, confirming the regression. Both arms execute two pressure
cycles in this matched window. More expensive
cycles, rather than more cycles, explain the pressure regression.

## Where the work grows

1. **The transition sampler is the largest momentum bottleneck.** In the
   dispatch-separated early-frame capture (frames 9–24), the 2h momentum
   dispatch alone has a median 28.279 ms; general h is 9.765 ms, 4h is 1.704 ms,
   and certified h remains 0.262 ms. There are only 19 transition tiles, or
   152 transition owners. General face dispatch enumerates canonical patches,
   evaluates multiple velocity components per invocation, and invokes the
   generated h/2h/4h interpolation and restriction functions. The optimized
   certified-h path instead traces one component per lane with direct taps.
   This isolates the expensive path; it does not prove register spilling or
   a specific compiler failure.

2. **The shared characteristic certificate spreads general work well beyond
   the region.** `UniformMixedFramePlan.certify` uses one domain-wide maximum
   face speed and radius `ceil(speed * dt / (4*h) * 1.00001) + 2`. Every fine
   tile within that radius of a non-fine tile enters the general worklist.
   At frame 72 there are 485 general h tiles. During frames 9–24 that count
   reaches all 4,069 h tiles. With no region, all 4,096 tiles remain certified.
   This affects momentum and surface advection even where local sampling
   would remain entirely fine. Tightening this bound would require a proven
   conservative local reach calculation, not relaxing the certificate.

3. **Viscosity loses its direct-site sampling path at interfaces.** Certified
   h forces load exact MAC sites directly. The general `umLaplacian` obtains
   the center and six offsets through mixed interpolation for each face.
   This reuses the expensive sampler machinery on transition/interface
   work. The scene has dynamic viscosity 0.001002 Pa·s and zero surface
   tension, so curvature evaluation is disabled. The measured body-force
   increase is 14.7 ms; attributing every millisecond specifically to the
   Laplacian would still require term-level isolation.

4. **Pressure repeats the interface work inside every sweep.** Each sweep
   reconstructs interface slopes, freezes the corrected RHS, then performs
   two Jacobi updates through regular and interface tier dispatches. The
   dispatch-separated capture attributes most added work to 2h/4h interface
   smoothing, reconstruction and RHS correction. A tiny region therefore
   activates costly work repeatedly through both coupled levels, while the
   all-fine case skips it. Reconstruction depends on the current pressure and
   cannot simply be reused across sweeps. Immutable face geometry is a
   separate candidate for precomputation.

## Next work, in priority order

- Profile and simplify 2h/interface MAC sampling and force stencil evaluation;
  those are the dominant costs of this exact region.
- Reduce unnecessarily general h work with a conservative, frame-frozen reach
  certificate, keeping the same characteristic and interpolation rules.
- Reduce repeated pressure interface geometry evaluation and tiny tier
  dispatch overhead without changing reconstruction frequency, Jacobi order,
  tolerance, or accepted-cycle policy.

The initial diagnosis changed instrumentation only. The implementation below
is additional to the prior momentum, extension and coarse-visit improvements.

## Implementation: parallel face evaluation

Momentum and interface forces now launch one workgroup per topology tile, with
one lane per native anchor and vector component. Only canonical faces evaluate
the numerical expression. A 192-entry workgroup array packs the results into
one RGBA store per anchor after a barrier. Negative boundary faces retain their
original owner and index. This replaces the serial patch/component traversal;
it does not change interpolation, force terms, ownership, or the solver.

The 19 transition tiles previously used just three workgroups of serial owner
work. They now use 19 groups with independent face evaluations. Certified fine
momentum and regular fine forces retain their existing kernels. The shared
ownership API exposes a tile-group launch option without changing its packed
worklist ABI or other stages' launch defaults. No persistent GPU allocation was
added; the face kernel uses 1,536 bytes of workgroup scratch.

Frozen before/after bundles and captures are under
`/tmp/fluid-region-face-dispatch/`. The first face-only comparison over frames
49–72 measured momentum 21.594 → 6.652 ms, forces 14.877 → 3.998 ms, and total
GPU pass time 85.426 → 58.720 ms. All 72 frames had exactly matching accepted
pressure residuals, cycle counts, and regular/general work counts.

The force and momentum tests compare every canonical face and negative
boundary against the previous serial GPU traversal. Momentum retains all seven
layout cases, now registered separately with the same per-case watchdog because
the extra reference pipelines made the single combined test exceed its old
four-minute watchdog. No performance limit or numerical assertion was relaxed.

## Retained surface change and final comparison

Redistancing also distributes canonical vertices across a tile workgroup;
certified fine vertices keep the existing direct kernel. Vertex authority and
the Newton calculation are unchanged. The same scheduling experiment for
surface advection was removed: it passed isolated vertex tests but introduced
small full-scene residual differences from frame 9 and eventually changed
accepted cycle counts. Its numerical difference is unresolved and its faster
timings are not claimed as a retained improvement.

The final 72-frame small-region capture (`final.log`) matches the original
(`before.log`) accepted residuals and cycle counts exactly on every frame.
Median GPU times over frames 49–72:

| Work | Before (ms) | Retained changes (ms) |
| --- | ---: | ---: |
| Momentum advection | 21.594 | 6.521 |
| Body forces | 14.877 | 3.965 |
| Surface redistance | 2.621 | 2.359 |
| Surface advection | 11.993 | 11.895 |
| Pressure sweeps | 14.385 | 14.549 |
| All measured passes | 85.426 | 58.491 |

This is 31.5% less GPU time, or 1.46x throughput for the measured passes.
Both versions report 55,892,780 allocated GPU bytes. Pressure and advection
were not optimized in the retained change; their timing differences are noise.
The near-native performance target is not met. General mixed sampling,
surface advection, and pressure interface sweeps remain the next bottlenecks.

Validation:

- Momentum (all seven layouts) and forces passed exact comparisons against
  the previous serial traversal; surface tests cover canonical/inactive
  vertices, curved-interface rebuilds, walls and retirement.
- The live frame test passed all-fine/mixed/all-coarse ownership edits and
  allocation assertions. The final retained surface variant passed its test.
- All-fine matched frames 49–72: 17.727 → 17.695 ms, identical residuals,
  cycle counts and worklists. No measured all-fine regression.
- `git diff --check` passed. Repository-wide TypeScript checking still reports
  unrelated Sparse/SVO test/tool errors; none refer to these changed files.

No CPU path, runtime repair, tolerance change, new persistent field, or
additional solver algorithm was introduced. Concurrent static-solid changes
in the same Uniform files were preserved.

## Paused state and work amplification

Implementation is paused at the verified face-parallel momentum/force and
vertex-parallel redistance changes above. A subsequent direct-address velocity
restriction experiment was removed: completed stage comparisons passed, but
its full-scene residual history changed and that difference remains unresolved.
Its frozen candidate is `restriction.mjs` in the temporary capture directory;
its timings are not retained production results. Its remaining experimental
momentum test was stopped. No optimization process remains running from this
work. Other tasks may still hold the shared GPU lease.

For the reported 64³ scene, the physical owner counts are:

| Layout | h tiles | 2h tiles | 4h tiles | Active cells |
| --- | ---: | ---: | ---: | ---: |
| All fine | 4096 | 0 | 0 | 262144 |
| Reported corner region | 4069 | 19 | 8 | 260576 |

A tile spans 4×4×4 finest cells, and holds 64 h owners, eight 2h owners, or
one 4h owner. A single interior tile forced to 4h normally introduces 26
transition tiles; the reported region has eight forced tiles and 19 transition
tiles because it touches three domain walls. The reported region reduces the
active cell count by just 0.598%. The full native 64³ cell/velocity textures,
65³ vertex textures, negative-boundary planes and scratch capacities remain
allocated. Pressure arrays use compact owner indices inside borrowed storage.
The existing 18³ 4h sampling caches remain present in both layouts.

There are three distinct affected neighborhoods:

- Physical grading: 19 tiles change from 64 h cells to eight 2h cells.
- Fixed interface work: the finest level has 37 h, 19 2h and seven 4h
  interface tiles. Its regular tiles are 4032 h and one 4h tile.
- Characteristic support: a global speed bound gives momentum/surface a
  larger general-work list. This was 485 h tiles at frame 72 and reached all
  4069 h tiles earlier. These tiles remain fine; their sampler changes.

The frame freezes ownership and shared worklists. That is not a precomputed
address/weight program for every face and query. Fixed face geometry is still
rediscovered within pressure sweeps; moving characteristic queries resolve
ownership and interpolate/restrict at their actual sample locations.

Work by stage:

- Momentum: certified h characteristics use direct MAC loads. General work
  resolves canonical sites, blends h/2h/4h interpolants, and restricts finer
  patches where needed. Every characteristic evaluates vector velocity
  repeatedly. An equal-level face is one patch; a face against a finer
  neighbor is four patches under 2:1 grading. The retained change evaluates
  components independently and packs one texture store per anchor.
- Forces: regular h viscosity uses seven exact MAC sites. Interface viscosity
  samples those sites through the general interpolator. This uses the fixed
  interface list, not the larger characteristic list. Gravity is simple;
  surface tension is zero in this scene. Nonzero tension would also require
  mixed occupancy gradients and curvature.
- Surface: only canonical vertices are advanced; the coarsest incident owner
  owns a vertex. Hanging values are reconstructed when sampled. RK2 velocity
  sampling, up to 64 cubic phi taps, wall continuation and drain checks can
  therefore invoke additional owner resolution. Redistancing repeats gradient
  sampling/Newton searches, with physical bands proportional to owner width.
- Extension: the same seed, front sweeps, hierarchy fill and publication remain.
  Interface work handles split MAC patches and differing spacings; regular h
  retains direct stencils. The small-region capture did not identify this as
  the dominant added cost.
- Conservative transport: fine/equal-grain translated boxes have eight donor
  candidates plus the identity slot. A 2h box sampling h can have 27 candidates;
  a 4h box sampling h can have 125. Donors resolve to canonical owners and are
  normalized by physical capacity. The three balancing rounds already exist
  in all-fine mode; nonempty extra tiers add dispatches, not new rounds.
- Geometry, cleanup, global volume correction and eight sharpening sweeps
  retain their stage order. Values/fluxes are weighted by owner volume and
  canonical face area. Extra tiers and interface handling add work.
- Projection: regular two-point gradients become reconstructed mixed-face
  gradients at interfaces, with physical face area/distance and boundary rows.
  Canonical publication does not expand a separate dense simulated fine grid.

Pressure is the largest repeated scheduling multiplier. No extra multigrid
levels are introduced: level 0 becomes h/2h/4h, level 1 becomes 2h/4h, and the
4h continuation is uniform as before. Every mixed sweep reconstructs slopes,
freezes interface RHS corrections, then performs the same two Jacobi updates.

| Coupled level | All-fine dispatches per sweep | Reported region |
| --- | ---: | ---: |
| Finest | 2 | 16 |
| Next | 2 | 12 |

Finest: three reconstruction + three RHS-freeze + twice (two regular-tier +
three interface-tier) dispatches. Next: two reconstruction + two RHS-freeze +
twice (two regular-tier + two interface-tier) dispatches. With six pre- and six
post-sweeps and two V-cycles, those two levels grow from 96 to 672 dispatches.
This excludes transfers, residual checks and the unchanged coarser solver.
It combines repeated geometry/arithmetic with small, underfilled launches;
the dispatch count alone does not quantify how much time is launch overhead.

Necessary mixed work includes grading, canonical patch/vertex ownership,
conservative restriction/interpolation, and pressure interface coupling.
The broad global-speed sampling certificate, repeated fixed geometry lookup,
serial patch evaluation and the number of tiny tier launches are implementation
costs rather than requirements of mixed resolution. The CPU still only
orchestrates and reads convergence receipts; it does not solve a fallback.

## Follow-up diagnosis (2026-09-26): attribution, certificate, bitwise identity

Frozen bundles, logs and field dumps are in `/tmp/fluid-region-next/`
(temporary). `current.mjs` is byte-identical to the paused bundle and its
72-frame split capture reproduces `final.log` residuals exactly.

**Per-dispatch attribution** (split capture, median over frames 49–72;
split passes are quantized to ~65 µs, so treat small rows as rough):

| Dispatch | ms/frame |
| --- | ---: |
| Surface advect, 2h (19 transition tiles, 3 workgroups) | 6.95 |
| Surface advect, 4h | 2.36 |
| Surface advect, general h / certified h | 1.31 / 1.11 |
| Momentum 2h face kernel / general h | 3.47 / 2.62 |
| Pressure sweep, seam 4h / seam 2h Jacobi | 3.34 / 2.49 |
| Pressure sweep, regular 4h Jacobi (one live lane) | 1.44 |
| Pressure sweep, seam freeze/reconstruct (2h+4h) | ~5.2 |
| Pressure sweep, regular h Jacobi | 1.31 |

Surface advection is dominated by serial transition owners, not by the broad
certificate. Pressure is dominated by latency-bound one-to-three-workgroup
seam launches (~20 µs each over 48 sweeps) that recompute fixed face/phi
geometry every update.

**The certificate spikes are real flow.** A velocity readback shows liquid
faces at 9–11 m/s (max 18.8 m/s, including air) during frames 7–13, at the
z≈60 far wall, ~13 tiles from the corner region. General-h counts follow
(3+r)³−27 exactly for radius r. In those frames general-h momentum costs
15–29 ms. A local certificate would remove most of it. Steady frames have
radius 4–5, set by the flow near the region itself.

**General and certified fine paths are not bitwise equal.** Forcing every
all-fine tile onto the general list (`let list=2u`) diverges residuals from
frame 2. After frame 1, exactly one phi vertex differs, by one ulp. Velocity
and volume are bitwise equal. A standalone Dawn/Metal kernel shows that
`let t=a*b; let s=t+c;` compiles to a fused multiply-add. The same arithmetic
through an array in a loop was not fused. Dawn exposes no fast-math or
contraction toggle. Contraction therefore depends on kernel shape (unrolling
and inlining), not on the WGSL arithmetic. This is the probable cause of the
unresolved residual changes in the earlier tile-parallel advect and
direct-address restriction experiments. Consequences:

- The current mixed result already depends on the global speed bound, because
  that bound decides which fine tiles use which compiled path.
- Any narrower certificate, tier fusion or restructured kernel can change
  ulps even when the arithmetic is identical. Whole-scene bitwise residual
  histories cannot be the acceptance criterion for these restructurings.
  Otherwise every such change is blocked.
- A bitwise gate is still meaningful for changes that keep each kernel's code
  shape intact. Examples are integer-only lookup memoization and cutting
  empty launches.

In the all-fine forced-general run, general-h kernels cost momentum 1.8 ms,
advect 1.8 ms and redistance 1.9 ms for 4,096 tiles. Certified costs are
0.39, 1.31 and 1.31 ms. The high per-tile cost in the mixed run therefore
comes from tiles whose samples reach the blend band next to the region.

## Implementation (2026-09-26): tile-parallel advect, fused pressure, hanging taps

Acceptance criterion (Peter): frame time. Results need not be bitwise equal.
Each change keeps the arithmetic, so differences stay at rounding level.
Every capture below uses the matched command
`--arm=mixed --region=87.5_0_0_100_12.5_12.5_4_4 --scene=minimal-power-dam-break-64 --tolerance=5 --steps=72`.
Figures are median GPU pass sums over frames 49–72, with baseline and
candidate run back to back. All-fine is 17.69 ms.

| step | pass sum (ms) | total cycles |
| --- | --- | --- |
| checkpoint 88865ce4 | 58.20 | 146 |
| v1: tile-parallel surface vertices + fused pressure tiers | 43.32 / 44.63 (two runs) | 146 |
| v2: + per-frame hanging fine-tap cache | 35.45 / 35.26 (two runs) | 145 |
| v3: + cached viscosity sampling (4h cache + hanging taps refilled for forces) | 32.51 | 147 |

**v1, surface.** `advect` and `redistance` launch one 125-lane workgroup per
tile, with one lane per canonical vertex. This replaces a serial loop over
each transition owner's corners. Certified tiles keep the owner kernels
(`advectOwners`, `redistanceOwners`).

**v1, pressure.** `reconstructFused`, `freezeRhsFused` and
`smoothJacobiFused` run one workgroup per tile for the seam tiles of all
tiers, plus every regular tier with at most
`UNIFORM_MIXED_FUSED_REGULAR_TILES` (64) tiles (`umFusedOwner`). Launches
drop from 16 to 6 per sweep on the finest level and from 12 to 6 on the
next.

**v2, hanging taps.** A fine MAC tap whose cell lies in a 2h tile resolves
through nested `umSampleVelocity2` and restriction/`umSampleVelocity4`.
Characteristics revisit the same taps many times.
`UniformMixedHangingTaps` (uniform-mixed-momentum-cache.ts) evaluates
`umVelocityTap1` once per frame for every (fine cell, axis) of every 2h tile,
right after the 4h cache. The unchanged sampler writes these values into
`ownership.hangingGroup`: T tile→slot words, then 192 values per transition
tile, growing on live edits. The frame's surface and momentum stages compile
with `hangingGroup` and load those values. They share one fill because both
sample velocityScratch, negativeScratch and 4h cache 0, and nothing writes
those between the cache and forces. The fill costs about 0.1 ms.

Stage deltas, v1→v2: momentum 7.08→1.97, traceCells 3.15→1.05, advect
4.59→2.82. Frames 1–6 are bitwise equal, even though the cache already serves
19 transition tiles. Divergence starts at 1e-5 relative in frame 7, and total
cycles are 145 against 146. The cache lives in its own bind group, not in the
ownership group. Adding it to the ownership group put the two-ownership
pressure stages over the default of 8 storage buffers per stage.

Remaining gaps (mixed v2 vs all-fine, ms): pressure sweep 7.0 vs ~2.5; body
forces 3.93 vs <0.2; advect 2.82 vs 1.31; redistance 2.42 vs 1.31; momentum
1.97 vs 0.39; traceCells 1.05 vs 0.33.

**v3, forces.** A split capture showed that viscosity was all of the
mixed-forces cost (3.93 ms). Removing the curvature term changed nothing;
removing viscosity took forces to about 0.1 ms. Its seven-point Laplacian
sampled `velocity` + `negativeDeparture` through the uncached sampler, which
restricts 4h patches per tap and nests hanging fine taps. Momentum is the last
reader of the 4h cache textures and the hanging buffer. The frame therefore
refills both from the forces field after momentum (about 0.1 ms each), and
`UniformMixedForces(..., cachedSampling=true)` samples through them. Forces
fell from 3.93 to 0.98 ms. Residuals are bitwise equal through frame 13; from
frame 14 they differ at 7e-6 relative.


**Rejected after v3 (clean captures, browser idle).**
- *v4, fold freeze into reconstruct.* +0.46 ms. The larger fused kernel lost
  more than the saved launch.
- *No cap on fused regular tiers* (fold every regular tier into the fused
  launch). +0.46 ms. A large regular tier is cheaper as a flat 64-lane launch.
- *v5, cooperative seam lanes* (split each interface row's six faces across
  lanes, with a workgroup reduction). Null within noise. The patch-loop
  dependency chain is not the bottleneck.

**Marginal-cost probes.** Each probe duplicates one idempotent launch per sweep
and divides by the launch count, because the 65.5 µs timestamp quantum makes
per-launch split numbers only statistical. Results per launch: fused smooth
~23 µs, reconstruct ~21 µs, regular fine smooth ~23 µs. A width-1 row costs ~10
µs, width-2 and width-4 ~23 µs. Width-2 modes: trivial body ~4.6, core terms
only ~13, full interface path ~20 µs. The seam cost is therefore the generic
interface code path: face enumeration, boundary terms and correction lookups
evaluated per row per sweep. It is not occupancy or launch count.

Ranked mixed-specific costs from the v5 split capture, in ms:
smoothJacobiFused 2.29 (96 launches), fine regular smooth 2.23,
reconstructFused 1.05, freezeRhsFused 0.98, general-h momentum 0.98,
advect w2/w1g/w4 0.66/0.59/0.46, momentum w2 0.59, redistance w4/w1 0.52/0.46,
retirementEvidence 0.59, per-tier reconstruct/measure/residual outside sweeps
0.59/0.59/0.39. Next levers, in order:
1. Per-frame seam pressure records: neighbour indices and weights for each
   interface row, precomputed once per frame and read by every sweep.
2. Fuse the per-tier launches outside the sweeps.
3. Coarse redistance: its `umNoNearbySurface` tile scan is serial.
4. A local speed certificate to shrink the general-h set (median 702 tiles).

**v6, seam pressure records.** Every step of an interface sweep is linear in
pressure: reconstruction (including ghost samples and the thin-domain
tangential fallback), the frozen seam correction, and the core terms (patch
weights, ghost theta, wall coefficients). The coefficients are fixed by
ownership, phi and static solids for the whole solve.
`uniform-mixed-pressure-records.wgsl.ts` writes them once per level per frame
(`buildRecords`, from `encodeSurfaceRestriction`, and lazily for a new
ownership generation). The sweep then runs gathers: `reconstructRecords`,
`freezeRecords`, the regular launch, then `smoothRecords` twice. These cover
the fused job set (seam tiles plus small regular tiers).

Records live on the ownership (`recordGroup`), because stages never allocate.
They use one chunk per job: 64 words of header, then rows of 16+10E words, with
E=6 at h and 24 otherwise. That is ~23 KB per job, so a compact layout is
needed before large seam counts.

Result: sweep 7.14→3.74 ms, frame 32.05→29.43 ms. Frames 1–5 are bitwise equal
to the patch-walking operator; later frames differ by ulp-level chaotic drift,
and the cycle count is 144 against 147. The record sweep kernels cost ~10 µs
per launch (reconstruct 0.46, freeze 0.39, smooth 0.92 ms per frame).

**v7, merged tile launch.** The general-h (certificate list 2), 2h and 4h
tiers of surface advect, redistance, traceCells and momentum were three
dependent, latency-bound launches with a handful of workgroups each. They are
now one indirect launch (`umTileJobOwner`/`umMergedTileJob`). The frame plan's
`publishWork` writes the third indirect args (general + 2h + 4h tile jobs) at
support header words 12..14, and `certifiedDispatch` grows to 48 bytes.

Under `umMergedTiles`, `umOwner` maps 64 slots per job, so flat owner kernels
(`*Owners`, `traceCells`, the serial test oracles) run unchanged in the merged
launch. The tile face kernel keeps its per-tier text, with a runtime width only
under the merged override. Without that, forces moved by 1 ulp against its
serial oracle.

Result: frame 29.43→27.66 ms, bitwise equal to v6 over 72 frames. Advect
2.75→2.29, traceCells 1.05→0.66, momentum 2.03→1.70 ms.

**v8, local speed certificate.** `certify` used the global maximum face
speed. `UniformMixedFramePlan.encodeCertificate`, which runs after extension,
now bounds each fine tile by the extended-field speed in the box of the global
reach around it:
- `localSpeed`: per-tile maximum over canonical faces and the negative planes.
- `spread0..2`: a separable box maximum.

It stays rigorous, because a characteristic from tile t never leaves that
box. Storage is `ownership.speeds` (2T words). Result: the general-h median
went 485→455 tiles and the frame 27.85→27.39 ms. The gain is small because the
far-wall region is in the dam front's path.

**v9, fused forces.** Forces made up to six launches: regular per tier, plus
per-tier seam launches, each ~0.2 ms of serial latency. The seam and
small-regular tiers now share one launch (`umFusedJobs`), with the regular fine
kernel unchanged. Forces 0.92→0.39 ms, frame 26.54 ms, bitwise equal.

Peter (2026-09-26): dispatch optimisation is usually not worth it; remove the
*work* differences between mixed and the all-fine baseline. From here the
comparison is stage by stage against all-fine (`cmpstage.mjs`). Trajectories
now diverge chaotically, so redistance (1.3–5.8 ms/step with splash events)
is excluded from the frame comparison.

**Probes: where general sampling spends its time.** In a timing-only probe,
the general sampler takes the fine path everywhere: momentum 1.64→0.52, advect
2.23→1.44, traceCells 0.66→0.33 ms. Making every h/2h tap a single load while
keeping weights and blending gives 0.59/1.70/0.39. So the cost is resolving
*taps* into coarse data (site lookup, restriction loops, nested 4h sampling),
not the blend.

A fine-trilinear sampling image of the blended interpolant would be cheaper
still. It is not the same method: the sampler Dawn test pins native 4h
trilinear in all-4h layouts and seam continuity below 1e-3. Staggered 4h kinks
straddle fine cells, so it would break both.

**v10, velocity tap cache.** The hanging fine-tap cache is now a memo of
`umVelocityTap1` and `umVelocityTap2`. It covers every 2h tile and every seam
tile of each tier, including the native negative boundary plane taps. Record:
192 + 24 + 48 + 12 = 276 words. Buffer: tile→slot, slot→tile, records. One
276-lane workgroup per slot fills it with the unchanged sampler. A tap whose
tile has no slot is evaluated in place, so values are those of the defining
code. Momentum 1.64→0.85, advect 2.23→1.84 ms; frame excluding redistance
24.77→23.72 ms; fill cost unchanged at 0.2 ms.

**v11, record residual and measure.** Outside the sweeps, reconstruct,
residual and measure ran the generic patch-walking operator over every owner
of every tier: 0.52/0.39/0.59 ms against all-fine's 0.13/0.07/0.13.
- Regular tiles now use the regular operator.
- Seam rows and small regular tiers use their records:
  `reconstructRecords`, `freezeRecords`, then `residualRecords` or
  `measureRecords`, with rhs − Ap = frozen + Σw(p_j − p) − (diag − Σw)p.
- Projection reads slopes only on the coarser side of a width change, which is
  always a record row, so it uses the record slopes.

Result: reconstruct 0.52→0 (folded into the record passes), measure 0.59→0.20,
residual 0.39→0.20 ms. Frame excluding redistance 23.72→22.0–22.2 ms; cycle
counts unchanged.

The first v11 run went non-finite in the frame Dawn test after a layout
change. `umPressureGhostCorrection` read slope(liquid) on every liquid/air
face and multiplied it by a tangential offset that is exactly zero for equal
widths. The old generic reconstruct wrote every owner's slope. Now only seam
rows are written, and the slopes range is shared scratch, so NaN × 0 poisoned
the projection. Equal-width faces now return 0 without reading a slope. That
is the same value for any finite slope, and the operator no longer depends on
slopes it does not need.

State after v11 (dam64, region 87.5_0_0_100_12.5_12.5):
- Frame: median 24.1 ms over steps 49–72, 22.2 ms excluding redistance.
- All-fine baseline: 17.8 ms, or 16.45 excluding redistance.
- Mixed Dawn list: 214/214 (v10); frame and pressure files 196/196 after the
  guard.

Remaining work differences against all-fine, in ms:
- Sweep seam passes: +1.3. Reconstruct and freeze per sweep; composing them
  into seam rows would drop both.
- Advect: +0.46.
- Momentum: +0.46.
- Sharpening: +0.5.
- Extension: +0.5.
- traceCells: +0.3.
- Hanging fill: +0.2.
