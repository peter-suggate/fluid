# Adaptive narrow band surface and tile retirement

The narrow-band method can release both particles and h tiles on surfaces
that the 4h Eulerian level set can represent. The implementation adds an
activity transition to Dynamic detail, with the existing fixed particle band
available as a reference. Full and Requested detail retain their previous
particle behavior. Uniform Geometric is unaffected.

## Relationship to Sato

[Sato et al., 2018](../papers/narrow-band-flip-2026/sato-2018-exnbflip/paper.pdf),
sections 3.2.5 and 3.3, provides the useful separation: heat identifies where
particles are needed, particle-carried heat supplies temporal persistence,
and a rasterized transition blends the particle and Eulerian surfaces.
Their velocity and geometry heat comes from Gaussian residuals; this version
reuses the repository's existing resolution/error and motion census.

That adaptation matters because the desired saving extends beyond particle
work. Here, particles request swept h support. Removing particles after a
controlled handoff also allows remapping to 4h, reducing work in the mixed
grid stages. Shape error and thin-feature requirements remain independent
reasons to retain fine tiles. A stationary feature is not necessarily safe
to coarsen.

## Heat and transition

The app profile uses the displacement shape metric with a 0.5h tolerance,
plus thin-feature, strain, impact and approach criteria. An enabled triggered
criterion admitted by the budget sets tile heat to 2. Active sources also heat their neighborhood.
Two GPU passes update local heat and spread half-strength heat to the 26
neighboring tiles, providing a one-tile overlap collar without a read/write
race. There is no additional step-count hold in the tile census.

At the default retirement time of 0.5 seconds, heat cools at 4 units per
simulated second. A particle takes the maximum of cooled heat and target heat.
Velocity and particle-addition influence hold for half the retirement interval,
then fade. Surface-erasure influence fades during the first half, retaining a
seeded support collar until retirement. Ballistic particles retain their heat.
Zero-heat particles retire, and zero-target regions stop reseeding.

The expanded heat raster still controls velocity transfer and adding liquid.
A separate occupied-tile raster controls erasure through `clamp(heat-1,0,1)`.
Air without particles is excluded from this interpolation's denominator; cold
liquid remains included. This avoids eroding an unseeded pool while preserving
particle geometry around thin sheets and droplets. Both heat fields travel
with the particles. The refinement criteria and budget ranking are unchanged.

```
phiNB = min(phiEulerian + h, phiParticles)
theta = phiNB > phiEulerian ? erasureHeat : expandedHeat
phi   = (1 - theta) * phiEulerian + theta * phiNB
```

An empty particle gather keeps the Eulerian surface. In particular, cold
4h regions retain their raw level set rather than using a particle-band
distance guard for a search that was skipped. Velocity transfer similarly
hands back to the advected grid velocity, using particle coverage during
the fade. Fully hot transfer recovers the existing fixed-band operator.

## Ordering and reactivation

The existing particles request swept fine support before the head census.
The census then updates heat before layout construction and particle
advection. Cooling particles continue requesting support even if every
user-selectable refinement criterion has been disabled. Once particles
retire, those requests cease and otherwise-unneeded h tiles can disappear.

New activity can bootstrap the outer surface layer after fine ownership is
available. The overlap can remain 4h. Source insertion retains priority and
can activate a previously particle-free simulation. Switching from Dynamic
to Full or Requested refills the cold surface band; returning to Dynamic
resumes cooling.

## Candidate budget and interactive controls

Adaptive surface is enabled in the app by default, with a 50% candidate
budget and 0.5 second retirement time. The low-level fixed-band reference
remains available. The Simulation detail toolbar exposes **Adaptive**;
**Tune** contains **Adaptive budget** and **Retirement time**, alongside
the existing criterion toggles and thresholds. The same settings appear
in the detailed panel and the simulation pipeline. Budget and retirement
changes apply continuously while dragging; toggling Adaptive rebuilds the solver.
Tune measures its dropdown against the current viewport pane, clamps both
axes and scrolls long content. Its sliders use the full available width,
with the value above the track. Browser checks at 480×640 and 480×400
confirmed the menu remained within the pane. The initial browser check observed clock advancement, but did not prove
solver retention; the renderer-lifetime correction below closes that gap.
[Verified narrow-window controls](../verification/narrow-band-tune-viewport.png).

The percentage denominator is the number of automatic tiles that request
fine detail in the current census. At 100%, every requesting tile is
admitted. At 50%, the highest-scoring `floor(requesting / 2)` tiles are
admitted. At 0%, no automatic candidates are admitted. This is not a
percentage of the domain. Sources, explicit fine regions, required contact,
closure and cooling particle support can add fine tiles beyond that quota.
The live fine count therefore need not immediately match the candidate
percentage, especially while existing particles are cooling.

Ranking uses the maximum existing normalized score over enabled shape,
thin, strain, rotation, impact and approach criteria. No extra splash
heuristic or weighting is introduced. The existing byte quantization and
saturation are retained; equal scores are resolved by tile order. A
same-census histogram cutoff and prefix of tied tiles enforce the exact
integer quota, even when every score saturates. Rejected candidates lose
their hold and stop renewing heat, allowing particles and h support to
retire together.

Below 100%, the classifier runs four additional GPU stages. The budget
arena adds `259 + tiles + ceil(tiles / 64)` words only for adaptive NB.
At 100%, all four selection stages are skipped. Retirement time is tunable
from 0.05 to 2 seconds; erasure influence fades before particle support retires.

## Scope and cost

The adaptive arena costs six additional words per possible 4h tile and two
small census passes per step. Particle capacity and grid storage remain
capacity allocations; fewer live particles and fine tiles reduce executed
work, but do not proportionally reduce reserved GPU memory. The disabled
reference compiles out activity helpers and does not allocate the extra
arena.

The implementation is EXNB-inspired, not a reproduction of Sato's Gaussian
heat or PDE propagation. Tile heat is deliberately conservative and can
keep a collar active beside a small feature. Benefits depend on how much
surface can settle into the coarse representation. Highly active surfaces
can retain nearly the whole band and pay transition overhead.

Refinement can only respond to information still represented by the current
fields. It cannot reconstruct a feature already erased by 4h advection. Thin
and approach criteria protect existing detail and anticipate contact; the
shape tolerance is an error criterion on the represented surface, not a
guarantee against all future subgrid feature loss.

This does not fix the existing narrow-band mass-loss, surface-noise or
Figure 2 growth failures documented in the
[volume investigation](narrow-band-volume-growth-2026-10-08.md). No global
mass correction, extra pressure cycles, smaller timestep or relaxed
accuracy threshold is introduced.

## Reproduction

`tools/benchmark-narrow-band.ts` explicitly selects the fixed profile unless
`--adaptive` is supplied, independently of the interactive app defaults.
Both profiles use the same scene, timestep and pressure configuration.
`--activity-census` adds a final read-only breakdown of refinement triggers.

```sh
node --import tsx tools/benchmark-narrow-band.ts activity-fixed --scene=nbflip-figure-8-letters --dt=0.041666666666666664 --steps=48
node --import tsx tools/benchmark-narrow-band.ts activity-adaptive --scene=nbflip-figure-8-letters --dt=0.041666666666666664 --steps=48 --adaptive --adaptive-budget=100 --activity-census
npm run test:dawn -- narrow-band-activity narrow-band-surface
```

The new GPU tests cover a hydrostatic cold pool, repeated activation and
retirement, Full-mode restoration, a stationary thin sheet, source insertion
above a cold pool, and a translating thin component. The manufactured fine
MAC transfer check covers full, partial and zero heat against an independent
quadratic-weight reference at the existing error tolerance.

The focused activity run disables optional solid-contact refinement in its
flat-wall pool control. It measured zero particles and zero fine tiles for
that hydrostatic pool after one simulated second, with maximum surface drift
0.0000673h and volume drift 0.000359%. Both hot/cold cycles retired from
32,768 particles to zero. The detached 2h sheet retained 2,304 particles
and 40 fine tiles. Over six translation intervals, the fixed and adaptive
sheet centroids moved 1.266220h and 1.266221h, respectively, against the
1.28h prescribed displacement; their represented volume ratios were
0.963047 and 0.963048. These are short controlled checks, not a claim that
the existing long-run narrow-band volume errors have been resolved.

## Initial transition validation (before the budget follow-up)

`npm run check:types` passes. `npm run test:unit` passes 912 tests with
113 GPU skips. The complete serial `npm run test:dawn` run passes 78 of
81 files, including all five new activity tests, all six surface/transfer
tests, and the maintained Uniform Geometric tests.

The three failing files are the same files that failed before these changes:
`uniform-narrow-band-contract-dawn.test.ts` (hydrostatic drift),
`uniform-narrow-band-settling-dawn.test.ts` (surface roughness and translating
sheet volume), and `uniform-narrow-band-volume-dawn.test.ts` (Figure 2 volume).
The adaptive app profile still breaches the Figure 2 10% volume bound near
0.9 seconds. This is not a clean repository gate or a volume-accuracy fix.
No failing assertion or timing ceiling was loosened.

Evidence: [full Dawn output](../verification/narrow-band-activity-dawn.log),
[baseline failures](../verification/narrow-band-activity-dawn-before.log),
[focused activity output](../verification/narrow-band-activity-focused.log),
[type check](../verification/narrow-band-activity-types.log), and
[CPU tests](../verification/narrow-band-activity-unit.log).

## Unbudgeted transition performance results

These runs predate the candidate budget and correspond to 100% admission,
not the new 50% app default. Four runs used an Apple M1 Max with Dawn/Metal, a 1/24 s timestep,
and 48 steps (2 simulated seconds). They ran serially after the GPU suite
finished, with no other Dawn test process active. All four record source
hash `741f4ac74d4d912f747ec163f18321d97fd6ac2f28d6824e36d76352f35e75d3`
and no GPU errors. These are headless step timings, not rendered application
frame rates.

| Scene and window | Fixed mean step | Adaptive mean step | Fixed h tiles | Adaptive h tiles | Fixed particles | Adaptive particles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Letters, frames 7 to 12 | 76.33 ms | 35.53 ms | 5,301 | 1,194 | 1,227,765 | 193,368 |
| Letters, frames 43 to 48 | 98.11 ms | 101.74 ms | 6,297 | 6,483 | 1,618,169 | 1,612,152 |
| Pouring, frames 43 to 48 | 70.71 ms | 75.80 ms | 8,343 | 8,335 | 1,029,550 | 1,014,556 |

Times are mean wall-clock step durations over each six-frame window;
counts are the final receipt in that window. The benchmark enables GPU
pass timestamps only for the final six steps, consistently in both modes.
Compare the two modes within each window. Absolute timings vary across
runs and the early and late windows have different profiling overhead.

Before impact, the large calm Letters pool permits 77% fewer fine tiles
and 84% fewer particles, giving a 2.15x step speedup. During the splash,
activity covers nearly the whole surface again; the conservative handoff
retains slightly more fine tiles and costs about 4% more time. Pouring has
almost no fine-tile saving and costs about 7% more time. This demonstrates
the benefit of retiring grid work as well as particles, but does not show
a universal speedup or establish whole-animation averages.

Final measured volume drift was -2.82% versus -3.32% for fixed versus
adaptive Letters, and -29.24% versus -28.51% for pouring. The substantial
pouring error remains a limitation of both configurations. Performance
results must not be read as a validation of long-run volume accuracy.

Data: [fixed Letters](../verification/narrow-band-activity-serial-letters-fixed.json),
[adaptive Letters](../verification/narrow-band-activity-serial-letters-adaptive.json),
[fixed pouring](../verification/narrow-band-activity-serial-pour-fixed.json), and
[adaptive pouring](../verification/narrow-band-activity-serial-pour-adaptive.json).
Earlier timing runs that overlapped another GPU test job are excluded from
this comparison.

## Budget follow-up results

The exact GPU selector passes mixed-score, disabled-criterion and saturated
tie cases at 0, 1, 25, 33, 50, 99 and 100 percent, including repeated
selection and padded two-dimensional dispatch. The seven lifecycle tests
pass, including candidate-based counts, live 0-to-100 reactivation and
faster particle/fine-tile release with shorter retirement time. Type
checking and all 912 CPU tests pass (116 GPU skips).

Browser verification confirmed Adaptive on, budget 50%, retirement 0.5 s,
and live budget changes reflected in both the URL and pipeline readout.
The screenshot is [saved here](../verification/narrow-band-adaptive-controls.png).

Serial Dawn/Metal runs compared the same adaptive criteria at 100%, 50%
and 25% admission, with the default 0.5 s retirement time. All use 48
steps at 1/24 s. Times average frames 43–48; counts are at frame 48.
Source hash: `a2786347e745b91fe9e1127db3691bbe108d078a58dccddd38d2279e85ba5664`.

| Scene | Admission | Mean step | Fine tiles | Particles | Volume drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Letters | 100% | 60.31 ms | 6,527 | 1,604,311 | −3.23% |
| Letters | 50% | 50.37 ms | 6,374 | 1,512,162 | −3.92% |
| Letters | 25% | 46.55 ms | 5,856 | 1,238,071 | −4.31% |
| Pouring | 100% | 71.22 ms | 8,358 | 1,022,856 | −28.82% |
| Pouring | 50% | 70.54 ms | 7,791 | 777,104 | −32.50% |

At 50%, Letters is about 16% faster in this short window; pouring gains
only about 1% despite fewer particles. At 25%, pouring stops at frame 24
with the unchanged 128-cell particle-trajectory safety limit. Lower
budgets are therefore a quality/performance tradeoff and are not robust
on every scene. The quota is exact over requesting seeds; support and
cooling explain why fine-tile counts do not fall proportionally. Neither
these timings nor the budget resolve the existing volume-accuracy issues.

Evidence files use the prefixes `narrow-band-budget-letters-` and
`narrow-band-budget-pour-` in `docs/verification`, with the percentage
as suffix; the failed 25% pouring run is retained as `25-failed.json`.

The final complete serial Dawn run passes **79 of 82 files**, including all
seven activity tests, the exact-budget selector, six surface/transfer tests
and every maintained Uniform Geometric lane. The same three baseline
files fail: contract (hydrostatic drift 0.00755354 m), settling (roughness
0.0768646h and thin-sheet volume loss), and volume (Figure 2 reaches
−13.885% for the reference and −16.466% for the adaptive app profile near
0.9 seconds). The repository gate remains unclean for those accuracy
failures; no numerical bound or timing ceiling was changed.

The runner was paused between files for the narrow-window UI check, then
resumed with that browser closed. No GPU test ran alongside that check.
Type checking and all 912 CPU tests (116 GPU skips) passed again after
the live-drag and menu-placement changes.

Final evidence: [Dawn](../verification/narrow-band-budget-dawn.log),
[types](../verification/narrow-band-budget-types.log),
[CPU tests](../verification/narrow-band-budget-unit.log),
[focused lifecycle](../verification/narrow-band-budget-activity.log), and
[selector](../verification/narrow-band-budget-selector.log).

## Renderer lifetime correction

The first UI check missed a real rebuild: declaring a parameter as runtime
in the method schema was sufficient for the controller, but the renderer
uses a separate `runtimeParamKeys` list. NB inherited Uniform Geometric's
list, which omitted budget, retirement time and fine-grid padding. Every
edit changed the renderer's construction key despite the live controller
classification. Clock advancement alone did not detect this.

NB now derives that list from its own complete parameter schema. The
renderer regression test fails before the correction by entering
`beginGPUFluidInitialization`, then passes after it: repeated runtime edits
retain the exact attached solver and adopt the new values. Allocation-mode
changes still produce a different construction key. Browser drags from
50% to 82% to 29% during playback no longer show an initialization overlay.
Evidence: [verified browser](../verification/narrow-band-live-budget-fixed.png).

Type checking and 913 CPU tests pass after this correction. The change is
method lifetime metadata; GPU simulation code is unchanged from the
79/82 full Dawn run above.
