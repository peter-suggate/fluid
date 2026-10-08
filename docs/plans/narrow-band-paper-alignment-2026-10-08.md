# NB-FLIP surface authority and large timesteps

## Current surface contract: level set with particle detail and optical spray

The following implementation supersedes the particle-enclosure contract in the
historical notes below. It is an EXNB-inspired adaptation, not a verbatim Sato
implementation: the existing quadratic 95% FLIP transfer and 4h velocity-sample
band remain. APIC and activity-driven retirement of calm velocity samples are
not part of this change.

- Advect the level set and particles as before. Before reconstructing anything,
  classify escaped sparse samples against this independently advected interface
  and the existing liquid support. Ballistic samples re-enter only once they
  are at least 0.25h inside supported liquid. The entry/exit distinction avoids
  using a particle's own reconstructed sphere as evidence for re-entry.
- Store particle surface activity separately from velocity (`position.w` is
  1 + activity; 3 identifies spray). A local high-pass velocity signal and a
  geometric second-difference signal raise activity; it decays at 2 per second.
  Uniform translation alone does not activate particle geometry. Metadata is
  published in a separate dispatch to avoid a read/write race in neighbor bins.
- Near the interface, the sphere envelope is only a candidate correction.
  Clamp its difference from the Eulerian field to 0.5h, and weight it by
  `0.5 * activity * (1 - exp(-8 dt))`. Do not erode calm/uncovered liquid, and
  never use a remote particle cluster to overwrite distant Eulerian air.
- Calm, nearly planar patches use a bounded symmetric six-tap relaxation of
  phi. It preserves affine planes, skips domain contacts and strong curvature,
  and removes small pre-existing wrinkles even when advection is stationary.
  This is a local filter, not a target-volume shift or an implementation of
  Goldade's pressure-based error correction. Its volume error is tested.
- Redistance also follows the local phi gradient across cell boundaries.
  Nearest crossing-cell *centres* can select a farther interface beside a tiny
  ripple: the old method reported about sqrt(2)h at a vertex about h below the
  surface. The local solve removes that noise source; the crossing-cell solve
  remains a fallback, and vertex signs remain unchanged.
- Spray does not participate in surface gathers, particle-to-grid transfer,
  liquid coverage or swept fine-grid requests. It retains gravity, collision
  and PIC re-entry. A GPU pack publishes only spray to the existing ellipsoid
  optical-interface renderer, so droplets share water shading and depth.
  The diagnostic particle overlay remains available independently.

There is no artificial lifetime deletion and no second spray simulation. These
are velocity/geometry samples, not fixed-mass parcels: neither `h^3/8` per
sample nor the optical radius defines a conserved liquid budget. Occupancy
continues to measure the level-set liquid; spray is separately reported rather
than silently included in liquid volume. Exact conservative liquid/spray mass
exchange remains outside this method's contract.

The old tests requiring every isolated sample, or a manually teleported sample
patch, to create liquid are replaced with the opposite authority checks. Tests
also cover a disordered but stationary pool, decay of pre-existing wrinkles,
gravity-driven settling, spray refinement exclusion, optical front/back
intersections, and re-entry. Existing maintained Uniform Geometric gates and
numerical thresholds are unchanged.

Measured new regressions: a 0.2h tangential disorder of surface samples leaves
an initially flat, stationary level set unchanged. A pre-existing 0.015h ripple
falls from 0.0075h RMS to approximately 0.000033h after one simulated second.
After three seconds under gravity, the disturbed pool has about 0.00005h RMS
roughness and 0.00077 m/s maximum component speed. Its liquid volume loss is
about 0.435%, compared with 0.438% in an undisturbed pool at the same duration;
the disturbance changes volume by less than 0.003%. This is not exact long-run
volume conservation. The existing 20-step hydrostatic absolute bounds remain.

Historical implementation and measurements follow; they do not describe the
new surface or spray behaviour unless explicitly stated.


The experimental `uniform-narrow-band-flip` method follows Ferstl et al.
Sections 3.1–3.3 for surface authority and sampling. `uniform-volume` retains
its conservative transport, surface-volume constraint and pressure recovery.

## Step contract

1. Advect persistent particles and the interior level set with matching RK4
   characteristics of the accepted extended velocity. Trajectory subdivisions
   target half an h cell of travel; the global pressure timestep is unchanged.
   Both paths report a sticky error above 256 subdivisions rather than hanging
   or silently truncating the trajectory. Particle collision walks remain.
2. Reconstruct the particle surface and union it with the interior eroded by h
   in sampled regions. Uncovered, explicitly coarse regions retain Eulerian
   surface tracking. Only the zero-crossing neighbourhood needs erosion;
   repeatedly adding h to unreinitialized deep distances would hollow the bulk.
3. Reinitialize distance and measure cell occupancy from that surface. There is
   no independent conservative volume transport, dust cleanup, surface-volume
   shift, ghost-phi drain or volume-recovery pressure source in this method.
   The occupancy field remains available to existing pressure, refinement and
   presentation consumers, but it is derived geometry, not a mass target.
4. Use particle velocity at faces within depth r=2h, and the semi-Lagrangian
   velocity deeper inside. The switch is sharp. With quadratic transfer, keep
   particles to R=4h, with a 2h overlap collar. This is a practical overlap
   assumption, not the former worst-case diagonal kernel envelope (R=5h). The optional coarse-face transfer also samples
   at the h particle scale. Forces and existing 4h global / fine-band pressure
   follow, with matched pre/post liquid-face extrapolation for FLIP increments.
5. Update particle velocities, then retire deep or excess inner samples and
   seed underfilled inner-band cells from the projected velocity. The outer h
   is protected; no spacing relaxation moves it. Newly admitted refinement
   regions bootstrap their surface samples once. Retired region coverage cannot
   continue eroding a surface after its particles have been removed.
6. Only samples within the outer 2h request swept fine support. The inner
   particle collar does not keep simulation tiles fine by itself. Residency
   prediction uses accepted grid motion, matching supported-particle advection,
   rather than the potentially different FLIP residual velocity. End-of-step
   membership and pre-erosion bulk distances are also guarded by actual
   zero-crossing cells: a bounded local distance search rejects stale shallow
   interior phi. The guard measures distance to nearby crossing-cell boxes with
   cell-scale slack; near the surface, redistanced phi sets the 4h threshold.
   This prevents old surface samples from spreading mandatory refinement into
   the bulk, and prevents their retirement from hollowing old coverage.
   Explicit Full/Requested layouts retain their existing semantics.

The particle surface is the union of one sphere of radius 0.875h per sample.
The paper requires a zero contour that encloses all particles. On an h vertex
grid a sample at a cell centre is √3/2·h ≈ 0.866h from every vertex, so no
smaller radius makes the interpolated field liquid at every sample; the
remainder is a 1% margin. The former weighted-centroid field (radius 0.505h,
capped at `nearest − 0.75h`, united with 0.433h spheres) was calibrated to a
flat eight-samples-per-cell lattice and left sparse samples outside: a CPU
replica enclosed 10% of isolated samples and 58% of a 0.5h sheet at two
samples per cell.

Seeding follows the same convention: samples sit one reconstruction radius
below the surface they define. Bootstrap candidates shallower than that move
down the distance gradient, stopping at the medial axis of features thinner
than two radii. The target depth, √(0.875² − 0.125) ≈ 0.800h, accounts for the
quarter-cell tangential offsets of the lattice from a vertex, so an axis-aligned
plane at an integer height is reproduced exactly; other axis offsets sit up to
0.06h low and oblique planes up to 0.075h high. Displaced samples are stored
without counting toward the cell they land in, so seeding stays independent of
dispatch order. Steady-state sampling still protects the outer h.

Enclosure has a floor: any sampled feature is at least 1.75h thick in the
field, and one isolated sample claims a 0.875h sphere rather than its nominal
h³/8. Thin sheets and spray therefore read as more liquid than their sample
count carries. This is the paper's trade, made on a field that is also the
pressure and render surface here.

## Diagnostics

`solver.narrowBandFlipInfo` includes:

- `beforeMaxOutside`: maximum positive particle distance to the advected grid
  surface, in h cells, before union.
- `afterMaxOutside`: the same measurement after reconstruction/redistancing.
- `outsideSurface`: number farther than 0.5h outside that surface.
- `unsupported`: number without resolved liquid velocity support.
- `deepInterior`: number deeper than R before end-of-step retirement.

These describe the existing particles before reseeding. No volume shift follows
reconstruction, so the post-reconstruction distance is also the final surface
comparison. Diagnostic maxima are observations, not a proof of timestep accuracy.
In particular, the paper's sufficient gap allowance of h must be checked for
highly deforming flows; RK tracing alone does not make arbitrary timesteps exact.

Subgrid unresolved samples retain the existing gravity fallback and PIC re-entry.
They are counted explicitly as unsupported. This is an extension to the paper,
not evidence that their surrounding liquid is resolved. The optional all-4h
experiment remains resolution-limited: its h-sized droplets cannot be faithfully
represented by 4h surface vertices. Fine coverage is the accuracy reference.

## Verification contract

The GPU tests check that volume transport, cleanup and target-volume correction
are never invoked in NB-FLIP, occupancy equals reconstructed geometry, and the
pressure recovery source is zero even with direct runtime overrides. They also
cover resting planes, live refinement, detached sheets, unsupported droplet
re-entry, free-fall acceleration, particle budgets, transfer kernels and extension
reuse. A manufactured translating drop uses 100 ms pressure steps (CFL_h=2.56)
and compares its endpoint with 50 ms steps, without requiring pressure subcycling.
A manufactured stale-distance case checks deep-particle retirement and return
from Full to a bounded dynamic fine band. The gravity-driven hydrostatic check
permits 0.05h surface drift and 0.2% volume drift over 20 steps; observed drift is about 0.025h / 0.16%. Tightening
pressure residual tolerance did not remove that reconstruction/sampling error.
The zero-gravity lattice equilibrium still retains its strict 1e-5 bounds.

The old experimental mass assertions measured a separate conservative V even
when it disagreed with the displayed surface. They are replaced by exact
occupancy/geometry agreement and a 2% short-trajectory volume-error bound for the
fine-band dam. Resting-pool and static refinement checks retain their strict
bounds. All-4h surface-volume drift is reported rather than hidden by conserving
an unrelated V field. No maintained Uniform Geometric threshold is changed.

## Figure 9 residency and performance

The 900-step, 60 Hz Figure 9 regression finishes at 15 s with 1,025 of 6,500
fully wet tiles fine (15.8%), 465,786 particles and no reseeding budget overflow.
The preceding membership-only guard still hollowed old covered regions and hit
1,048,576 particles. Applying the geometric guard before erosion as well as at
resampling removed that feedback. This regression checks coarse bulk retention;
it is not a claim of exact long-run mass conservation.

A six-frame timestamp window at frames 43–48, using 17 ms global steps, measured
39.1 ms mean wall time and 35.1 ms GPU work. The largest relevant groups were:

| Work | GPU ms/step |
| --- | ---: |
| Particle and level-set advection | 9.0 |
| Particle surface reconstruction | 4.2 |
| Geometric band guards, both rebuilds | 4.2 |
| Particle-to-grid transfer | 2.3 |
| Unused conservative-volume cell traces | 2.4 |

NB-FLIP now omits the last item entirely; momentum writes that scratch field
before its next reader. A subsequent short run measured 38.6 ms mean wall time,
so the measured end-to-end gain was modest, rather than the entire isolated
pass duration. These short windows include timestamp quantization, scheduling
variation and nondeterministic particle accumulation order.

The guard search now also uses the finite 4h support requirement: three
separable 11-tap axis scans replace three 27-tap jump-flood scans. That reduces
candidate reads from 81 to 33 per cell and computes the nearest crossing-cell
centre exactly within the search box. A brute-force GPU regression checks the
bounded minimization. The timing table above predates this simplification.

The next experiments, not implemented here, should be:

1. Adopt the paper's trilinear transfer and R=3h/r=2h instead of quadratic
   transfer and R=4h. A thick planar band then has 25% less particle volume;
   thin structures remain fully sampled. This deliberately uses the paper's
   empirical overlap choice rather than a worst-case diagonal kernel envelope.
2. Keep matching RK4 traces but allow travel of about 2h per subdivision in
   smooth velocity fields, with tighter subdivisions when their curvature/error
   estimate requires it. The current half-cell rule spends many velocity samples
   on nearly straight trajectories. Collision walks and the pressure timestep
   remain separate. A varying-velocity large-CFL regression is needed before
   adopting this change; constant translation alone cannot validate it.
3. Restrict exact geometric guards to the active particle/fine band plus its
   dependency and swept-motion halo. Keep reliable geometric membership at the
   surface; preserve the liquid sign in the distant coarse bulk. Every halo must
   cover the chosen timestep and all propagation passes, including new sources.

Pressure tolerances and outer-surface particle retention should remain intact.


## Audit of inherited geometric machinery

The NB method needs one evolving liquid description: the particle surface near
its boundary, united with the advected interior level set. Occupancy is a
measurement of that surface. It must not independently claim pressure rows,
retain liquid in air, or move the surface to restore a mass target.

Implemented in this pass:

- Do not construct conservative transport, dust cleanup or surface-volume
  correction for NB. Grid ownership is no longer obtained by constructing a
  transport solver. Their pipelines, worklists and scratch requirements are
  absent from NB preparation and capacity reservations.
- At both simulation and pressure ownership, use the surface-only authority.
  It writes pressure phi, liquid donor phase and zero correction in one owner
  pass. It neither lets volume claim a row nor retains detached-mass phase.
  No excess/deficit/stranded-mass reductions or recovery passes are compiled.
  Previously setting correction dt to zero disabled the source but still ran
  these passes and still allowed mass-based classification.
- Keep closed-solid continuation and the cut-tile surface-plane fit: these
  define the pressure boundary across grid resolutions, not a mass target.
- A regression corrupts stored occupancy to ten cell capacities and enables
  the old correction controls. NB pressure phi, phase and zero correction
  must remain unchanged, including the all-coarse mode.

Remaining work, in dependency order:

| Inherited stage | Why it remains suspect | Required NB replacement |
| --- | --- | --- |
| Conservative refinement remap | Redistributes a stored volume budget into children, then surface geometry later replaces it | Remap phi and velocity; regenerate occupancy before any occupancy consumer |
| Solid displacement of stored volume | Moves a mass field which NB subsequently overwrites | Keep particle collision and moving-solid pressure coupling; remove mass deposits once their remaining readers are eliminated |
| Volume-supported wall surface repair | Embedded contact uses occupancy as evidence for retained liquid; several wall probes and deferred traces predate particle authority | Surface/particle contact rule, verified with wall detachment and moving-solid tests |
| Frame support census | Stored volume can activate support independently of phi | Surface and swept-characteristic support, covering sources, edits and large timesteps |
| Level-set redistance plus two geometric band measurements | Limited inherited redistance can leave shallow values deep inside, requiring a separate guard against false particle membership | One distance representation that reliably distinguishes the surface band from signed bulk |
| Grid momentum inside the particle-owned band | Semi-Lagrangian face results are subsequently overwritten by particle coupling | Advect the interior and only the overlap/fallback faces actually needed |

Removing these stages blindly would leave stale occupancy or break boundary and
adaptive-grid dependencies. The next useful simplification is a surface-only
remap/census/contact chain, followed by sharing one band-distance computation.
Pressure tolerance, collision checks and the global timestep do not need to be
relaxed to achieve those savings.


## Measured result of this pass

On the same 48-step Figure 9 benchmark (17 ms simulation steps; means over the
last six frames), the combined changes measured:

| Metric | Guarded baseline | Current NB path |
| --- | ---: | ---: |
| Wall time per frame | 39.13 ms | 33.11 ms |
| GPU work per frame | 35.09 ms | 29.19 ms |
| Final particles | 561,520 | 428,733 |
| Reported allocation | 386,286,160 bytes | 384,041,616 bytes |
| Surface-derived volume drift at 0.816 s | -1.446% | -1.459% |

This is a combined result for dropping unused volume traces, bounded separable
band-distance scans, R=4h particle retention and surface-only pressure authority.
It does not isolate the pressure change's contribution. The main frame's three
authority invocations use three owner dispatches instead of twelve; solid cut
fits and absent-page publication remain. Short timing windows are noisy.
Evidence: `docs/verification/narrow-band-surface-authority.json`.

The 900-step / 15-second run ends with 2,381 fine tiles, 381,672 particles and
no clipped reseeding. Of 6,516 fully wet tiles, 965 remain fine (14.81%); hence
85.19% of the fully wet interior stays coarse. The final maximum positive
particle/surface separation is 0.233h, with seven unsupported particles.

**Unresolved accuracy issue:** the same long run reports +29.67% volume drift.
The residency test only proves that refinement does not consume the bulk; it
is not a long-run surface-volume accuracy test. The short dam and resting-pool
bounds pass, but they do not cover this accumulation. Investigate the surface
advection/reconstruction/redistance and remap chain next, measuring which stage
changes geometric volume. Do not mask it by restoring an independent mass
field or target-volume correction. The source of that growth is not yet isolated.

Validation: `npm run check:types` passed; `npm run test:unit` passed (898 tests,
92 skipped); `npm run test:dawn` passed all 71 files. After the final pressure
change, the focused contract, FLIP and residency run passed all eight tests,
including the corrupted-volume authority check and large-timestep translation.
No maintained Uniform tolerance or timing ceiling was changed.


## Surface-distance default

NB now defaults to Surface distance 0: elective refinement is restricted to
actual crossing tiles. The particle refinement predictor independently joins
current/predicted outer-band particle tiles, padded by 2h; these joins bypass
the distance filter. Mandatory solid promotion also retains priority. Thus
zero is not strictly crossing-only ownership and does not remove the existing
motion-dependent support for large timesteps. Margin and hold can affect the
ordinary selection but cannot override this crossing filter by themselves.

The earlier performance and 15-second results above used Surface distance 2.
They are not measurements of the new zero default. This default change is
configuration-tested; GPU comparison remains outstanding.


## Surface-only remap, contact and shared distance

The second simplification pass removes the first five dependencies in the
remaining-work table above from the normal NB frame:

- Refinement/coarsening remaps phi and velocity without gathering a donor mass
  budget, integrating child fills or redistributing volume. Occupancy is rebuilt
  from surface geometry immediately after either host or GPU relayout, before
  phase, extension or pressure can read it. Uniform Geometric keeps its
  conservative remap.
- NB no longer creates the solid-displacement mass-deposit stage. Particle
  collision, walked surface traces, solid cut geometry and moving-solid pressure
  coupling remain. Surface advection omits the domain liquid-continuation probes
  and embedded contact repair backed by volume. Released-wall air and buried
  vertex handling remain boundary conditions.
- The frame support plan and dynamic census use phi for NB support and
  classification; independently stored volume cannot activate owners. A tiny
  crossing tolerance (one millionth of h) includes both tiles when the surface
  lies on a tile boundary, so Surface distance 0 can restore wet-side particles.
- The reconstructed surface's bounded nearest-crossing-cell search feeds both
  h-vertex redistancing and end-step particle membership. Fine distances come
  from a local trilinear solve with edge-crossing fallback. Vertex signs are
  preserved; coarse reconstructed samples are copied unchanged, preserving
  hanging-vertex signs as well. This keeps the cached crossing-cell set valid
  through redistance and pressure. Deep values without a nearby crossing cannot
  be mistaken for particle-band distances.
- The advected surface still needs its own crossing search before particle
  reconstruction: it is a different surface epoch. The legacy redistance's
  separate evidence, preparation and search passes are no longer dispatched or
  compiled for NB, and its coarse travel counters are no longer accumulated.
  Standalone membership updates still rebuild their cache when needed.

The global pressure timestep and half-h trajectory subdivision are unchanged.
No volume target, pressure recovery source or mass-conservation correction was
introduced. Fine redistancing preserves the crossing topology, not the exact
interpolated zero position; its surface-volume error remains measured rather
than corrected.

New GPU regressions corrupt the remap's incoming volume across repeated h/4h
changes, compare stationary wall advection under empty and overfilled volume,
and verify the shared crossing set across a mixed-resolution seam. Existing
hydrostatic, large-timestep translation, live policy switching and stale
interior-particle tests retain their tolerances.

The detached-sheet regression also exposed a coverage lag: a newly occupied
fine tile could bootstrap outer-layer particles even though incoming particles
already defined its surface. The survivor pass now records that current support
before reseeding, alongside the existing spatial-bin writes. Seeding only reads
this frozen mask, avoiding an order-dependent check against its own new samples.
The regression still requires the exact original detached particle count.

### Measurements for the surface-only pass

Both short runs below use Surface distance 0, 48 steps of 17 ms, and means over
frames 43–48. The baseline is before this pass's remap/contact/distance changes.

| Metric | Before | After |
| --- | ---: | ---: |
| Wall time per frame | 33.58 ms | 32.15 ms |
| GPU work per frame | 29.57 ms | 27.86 ms |
| GPU dispatches per frame | 386 | 378 |
| Final particles | 432,843 | 442,291 |
| Fine tiles | 2,313 | 2,313 |
| Measured occupancy-volume drift at 0.816 s | -1.301% | -1.469% |

The total improvement is modest: approximately 4.3% wall time and 5.8% GPU
work in this window. It includes changed trajectories and particle counts;
it is not an isolated kernel-speed claim. Evidence:
`docs/verification/narrow-band-zero-before-hangover-removal.json` and
`docs/verification/narrow-band-surface-only-final.json` (source hashes included).

The final 900-step benchmark uses 17 ms steps and ends at 15.3 seconds with
1,508 / 16,384 tiles fine (9.2%), 420,325 particles, zero clipped reseeds in
the final step, 31 unsupported particles and maximum positive particle/surface
separation 0.871h. Its final six frames average 22.82 ms. During the splash it
reaches the 1,048,576-particle budget; optional reseeding is temporarily deferred
(up to 102,812 samples in a step). Existing samples retain budget priority.
Evidence: `docs/verification/narrow-band-surface-only-final-long.json`.

**Volume accuracy remains unresolved:** the final long benchmark's measured
occupancy grows by 10.35%. The older +29.67% result used a different timestep
and Surface distance 2, so these are not a controlled before/after comparison.
The long-run simplification does not establish volume accuracy. The separate
`representedVolumeDrift` diagnostic uses a smeared centre-phi estimate and is
not the geometric occupancy integral; it is unsuitable for measuring the
volume of this intentionally non-distance coarse phi field.

The surface-only cleanup passes the complete repository gate: types, 898 unit tests (95 GPU skips), and all 72 Dawn files. No Uniform tolerances or timing ceilings changed.


### Independent particle retention and grid coverage

The particle band remains R=4h, with particle velocity used inside r=2h.
Particle admission, retention, coverage and transfer no longer require an h
owner. Switching to Requested with no fine regions leaves the particle band
alive on the 4h grid. The existing all-4h option now selects the separate
particle render publication; it no longer controls particle lifetime.

`fineGridPadding` is a live h-unit control, independent of particle width and
of the existing Surface distance control (which is measured in whole tiles).
The current default is 1h; the 0h splash stress run failed the existing pressure-cycle acceptance check, so zero padding has not been established as a safe default. Prediction uses nearest surface-crossing
cells and sweeps them through the full macro timestep. Escaped samples request
their reconstructed sphere extent. Optional padding expands this geometry;
the inner particle collar never expands grid ownership by itself. First-frame
particles are bootstrapped from the head's extended velocity before the first
census, so the initial large step also gets predicted coverage. Half-h RK4
trajectory subdivision and the requested global pressure timestep are unchanged.

The shared nearest-crossing search now reuses its expired first bank for an
h-spaced nodal metric field. Particle membership and P2G switching sample that
field on either owner width; coarse advected phi is not assumed to be a signed
distance. Fine redistance copies those same metric samples. Coarse phi retains
its reconstructed values and hanging-vertex signs. Reconstruction uses the
crossing-cell bulk guard before the metric field is rebuilt.

The concurrent reconstruction change is integrated: nearest-particle spheres
replace weighted-centroid gathers. Their radius is 0.875h, slightly above half
a cell diagonal, to enclose particles after nodal interpolation. Bootstrap
shell samples are fitted to that representation. For the quarter-cell lattice,
the two tangential offsets give a planar seed depth sqrt(0.875²−2·0.25²)h.
Keeping those tangential locations avoids jitter-induced rest-state ripples;
this is a sampling calibration, not a volume target or correction.

New GPU coverage tests change padding live through 0, 1, 2, 0h at rest, with
32,768 particles throughout. At 0h, 24,576 particles live on coarse owners;
only the 64 surface-crossing tiles are fine. The 2h setting uses 128 fine tiles.
A moving drop covers 2.56h per 0.1s pressure step at all three settings, retaining
the existing velocity, centroid and surface-separation bounds. The integrated
contract, coverage, live-policy and reconstruction suites pass (17 GPU tests),
including the strict resting pool, hydrostatic pool and detached-sheet checks.


### Figure 2 volume investigation

The 128 × 128 × 8 Figure 2 drop reproduces severe geometric volume gain:
+467.81% after 300 steps of 17 ms (5.1 s). This is measured occupancy from the
surface, not the separate centre-phi volume estimate. The stage probe measures
represented tetrahedral volume immediately before advection, after advection,
after particle reconstruction, and after redistance. It is read-only and does
not feed back into simulation.

With the sphere-union reconstruction, the first 180 steps accumulate +35,615
h³ in reconstruction, -23,583 h³ in advection, -6,340 h³ in redistance and only
-42 h³ across refinement remaps. The dominant gain is in reconstruction, not
conservative refinement remapping. Controlled experiments:

| Figure 2 variant | Final geometric drift | Qualification |
| --- | ---: | --- |
| Original reseeding and sphere union | +467.81% | 5.1 s |
| Disable reseeding after initialization | -2.60% | Loses 49.38% earlier; unsuitable fix |
| Separate initialization; reseed below 2h | +132.92% | Reduces gain but does not solve it |
| Same, full fine grid | +118.70% | Gain also occurs without resolution seams |
| Seed only wholly empty interior cells | +109.89% | Still unacceptable |
| Stationary drop, zero gravity | -0.030% | 3.06 s; motion-dependent failure |

Artifacts are `docs/verification/narrow-band-figure-2-*.json`, with arguments
and source hashes. The probe adds GPU work; its frame timings are not clean
performance measurements. The full-grid and no-reseed controls are diagnostic,
not proposed shipping policies.

The paper's Section 3.3 resamples only between -R and -h, explicitly to avoid
changing the surface. It has no global target-volume correction. The separate
spray/activity work changes surface authority from the paper's unconditional
union to a bounded correction of the advected level set. Its combined volume
accuracy must be measured independently; these sphere-union results do not
establish the accuracy of that newer reconstruction.
