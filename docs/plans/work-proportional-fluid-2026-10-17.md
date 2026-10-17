# Fluid work proportional to numerical change

The next substantial saving should come from retaining valid numerical results and taking cheap early exits before expensive sampling, reconstruction, geometry evaluation, and iteration. Keep the existing launch structure. A stored h tile, a wet wall, or a particle support region establishes where a result must be valid; it does not establish that the result needs recomputing this frame.

Use different criteria for different operators. Surface displacement, velocity variation along a characteristic, distance-field defect, changed pressure coefficients, pressure residual, and particle occupancy answer different questions. Kinetic energy alone cannot safely govern this pipeline. Retain the global 4h pressure solve and use its current result to validate fine corrections everywhere, including previously quiet regions.

The investigation below establishes the baseline. Implementation has now begun with the first milestone below. There is no scene-name or hydrostatic-mode predicate.

## Current work shape

The implementation has one 16h control lattice and retains 32h storage pages. It uses local operator error bounds, not an energy threshold that switches a region between frozen and fully active. The global pressure response remains current.

- **Surface:** current signed velocity intervals and current phi models select an exact endpoint exit, an affine update, bounded Euler/trilinear transport, or the existing full operator. The controller precomputes the 64 possible rectangular support summaries once per region; this is a lookup table on the same 16h lattice, not another spatial level. The latest revision also reduces directional native-edge slopes during the existing phi scan. These bound actual piecewise trilinear transport across cell boundaries without penalizing a calm curved field for its whole-region affine-fit error. A destination-cell bound remains available when regional outliers make even those slopes pessimistic. Existing support validity replaces repeated canonical-authority and page-projection checks where it already proves their result.
- **Geometry:** current-corner plane integration replaces the detailed fill calculation within an absolute owner-fill budget of 0.005. Exact retained-input snapshots stay enabled: measured total geometry cost was lower with their scan than with the metadata-only ablation.
- **Normals and curvature:** geometry supplies candidate regions. Forces compare current normals against fixed retained anchors and reuse curvature around unchanged anchors. A component tolerance of 0.01 is admitted only when its local capillary displacement bound is at most `1e-4*hmin` for the current timestep/material. Topology, body support and page changes require exact refresh. Fixed anchors ensure gradual normal drift eventually triggers work.

The latest solid localization reuses the same 16h scan and support table. One lane per existing tile scan reduces its current canonical-open solid mask into the region; each support summary carries validity and an all-open bit without another dispatch, scan, or table allocation. Whole-affine surface transport reads that bit before fine solid probes, allowing distant regions to use the cheap operator despite a globally present body. The support includes the full smoothing and velocity footprint (15h for near-band 4h owners). This certifies only the surface operator’s canonical open-cell predicates: partially cut cells can still be open, so this is not a general collision-clear or uncut-geometry certificate. Domain-wall, source and released-wall guards remain.

Full reconstruction also queries the current support once, after its existing far-band exit. A clear region skips six coarse quarter-step solid predicates, the foot-open predicate and nine smoothing-open predicates; phi loads, gradient, clamping and smoothing arithmetic remain unchanged. Cubic sampling queries the actual destination cell’s support and can skip up to 64 buried-tap predicates. These exact exits remain useful when the approximate transport tiers reject a curved or moving field. Missing/tagged pages and support beyond the 27-region lookup window explicitly fall back before region-index arithmetic. The traced cell walk remains unchanged: its short-walk cost may be lower than a coarse certificate lookup, so the prepared segment-certificate experiment is not integrated.

The whole-affine tier also uses the midpoint of the current regional velocity enclosure instead of sampling point velocity. Its residual bound can shrink with actual displacement, while retaining smoothing, velocity variation and a nonshrinking floating-point allowance. It takes the tighter of the old absolute residual bound and the new displacement-scaled bound; the local error budget remains 0.02h and maximum travel remains half a base cell. These changes are not yet accepted on performance grounds; their total cost must be measured.

At unequal-width faces, an exactly zero cached curvature sum now exits before pressure-gradient reconstruction. This carries derived-geometry savings into force evaluation without changing nonzero capillary contributions. Equal-width faces retain their cheaper original evaluation order.

These choices retain cheap updates for moving, smooth water as well as exactly unchanged outputs. Their limits are local: surface transport admits up to `0.02h` phi error per operator evaluation, and neither it nor the normal certificate bounds the accumulated difference between two coupled simulations. Benchmarks therefore include interface fields, common-world-point velocity, body response and intermediate activity, not just acceptance counts. Normal storage and the optional input-snapshot cache are bounded; overflow conservatively computes full results, so arbitrary-ocean scaling beyond those capacities is not yet established.

The measured historical revisions below have **not** achieved the 5× target. Failed approximation attempts, classification scans and retained-state traffic all count against savings. The current revision is being measured with those costs included before extending evidence to other expensive stages.

## Execution plan

Target: reduce the measured settled-pool simulation from about 42 ms to about 8 ms through less numerical work. Keep the global 4h pressure response and fixed launches; never use last-frame speed alone to suppress pressure or wake-up. Each milestone needs its own physical regression and before/after measurement. These are cumulative architectural steps, not additive speedup promises.

Acceptance is wall-clock scaling with numerical need in the same domain. A larger number of cheap-path hits is useful diagnostic evidence but is insufficient by itself. Compare the authored pool at rest with a localized impact in that same pool, count final outputs by work tier and distance from the impact, and include control scans, retained-state traffic and fallback attempts in frame timing. The intended cost is a fixed global/coarse floor plus inexpensive updates across calm regions and expensive work concentrated around changing features. Amortize shared evidence across consumers; do not hide its initial overhead or promise linear scaling from operation counts alone.

| Milestone | Concrete change | Validation required | Status |
|---|---|---|---|
| 1. Current pressure residual | Measure the prolonged 4h predictor before ordinary fine smoothing; explicit measurement validity; strict band tolerance; unchanged projection | Exact-zero acceptance, large residual in any workgroup prevents acceptance, next-frame wake, settled pool, curved drop, rigid impact and buoyancy; measure band cost | Implemented; eight targeted GPU tests passed |
| 2. Local body dependencies | Use old/new swept body support to invalidate only intersecting fluid dependencies; fast exits before body sampling | Distant moving body, body entering water, rotation and surface crossing; compare local geometry/flux to full rebuild | Local support implemented for surface/geometry evidence; other consumers pending |
| 3. Persistent derived geometry | Separate cached coefficients/aggregates and surface geometry from transient RHS and aliased scratch; invalidate on source/topology/geometry changes | Exact retained/full comparisons, page recycle, remap, changing boundaries | Surface fill/center geometry implemented; pressure coefficients and other derived data pending |
| 4. Consumer prerequisites | Carry current GPU particle occupancy to snapshot/cache/search producers so empty readers skip preparation; preserve seeding and grid publication | Empty-to-active transition, source activation, impact and disappearing particles | Pending |
| 5. Bounded surface and momentum work | Retain or use inexpensive local predictors where operator error permits; accumulate skipped error/displacement and reopen dependencies | Slow waves over many frames, uniform translation, thin features, local and total volume, velocity and surface error | Current affine/Euler/trilinear surface tiers and current-corner fill implemented; net performance still regresses. Endpoint-proof and retained-normal cost reductions in progress; momentum pending |
| 6. Spatial fine correction | Reopen fine regions against current coarse pressure and changed neighboring corrections; evaluate tall-cell candidates across all stages | Distant response to localized impact, conservation and wake latency; cross-stage candidate counts | Pending |

The first milestone is deliberately narrow: it establishes trustworthy residual exits. It does not remove row assembly, geometry, force, extension, or surface costs, and cannot deliver 5× by itself. Do not begin a compressed-column representation until counts show a worthwhile reduction across those stages.

### Spatial work control: one fixed 16h level

Following the design discussion, use a single fixed coarse work-control lattice rather than a recursive work hierarchy. Start with 16h regions, independently of the existing 32h storage pages and the pressure solver's numerical multigrid levels. Each region covers 4×4×4 existing 4h tiles; eight regions fit in a storage page. The existing local residency machinery already uses this partition, two-word tile masks, and stage-specific directional dependency footprints (`uniform-local-residency.ts`, `uniform-mixed-dynamic.ts`). Residency proves that inputs exist; it does not prove that cached results remain numerically valid. Add separate validity/error evidence rather than interpreting residency masks as permission to skip computation.

Use flat region-indexed records for stage validity, accumulated error, and invalidation reasons. Fixed launches read the appropriate region/tile decision before expensive gathers. Reuse existing producers' reductions where possible; do not add a full fine-field scan merely to decide whether to avoid a full fine-field update. A region can retain valid data, use a bounded cheap update, or require detailed work. Existing tile masks can preserve localized exceptions without introducing another decision hierarchy. Current pressure residuals, sources, body motion, page generations, and complete operator dependency footprints must reopen work. Global coarse pressure remains active.

The next prototype should connect these records through surface evolution and retained derived geometry, with per-region counts of expensive paths and their invalidation reasons. Compare 16h decisions with the conservative union of eight regions (32h decisions) using the same frame evidence. Measure both classification cost and the extra detailed work caused by grouping; this comparison measures control granularity, not allocation savings. Validate localized impact and smooth translating waves as well as settled water.

Evaluate 16h physical paging separately. It can reduce allocation rounded around thin or irregular occupied regions, but cannot reduce the interior payload of a fully occupied volume. For the same fully occupied volume, eight 16h pages replace one 32h page: page metadata grows eightfold and total page-face area doubles. Under the current vertex block layout, fine vertex storage changes from 33³ to 8×17³ (about 9.4% more), and 4h vertex storage from 9³ to 8×5³ (about 37.2% more). Allocated pressure-root face slots double from 384 to 768 per equivalent volume. These are layout calculations, not measured runtime penalties; sparse boundary savings may outweigh them.

A physical paging experiment must also preserve the current world reach (the 64-page directory spans half the distance at 16h) and the 32h pressure level (currently one unknown per 32h page). Address keys, vertex skirts, neighbor tables, initialization, pool capacities, and consumers must migrate together. Do not change only `UNIFORM_PAGE_POOL_EDGE`. Before that migration, count requested 16h subregions versus held 32h pages and estimate total field, metadata, and face-slot bytes under both layouts. The first work-control prototype retains 32h allocation; a physical page-size change remains an independent measured decision.

### Historical surface and derived-geometry prototype (superseded)

The following describes the superseded first prototype; its foot cache and surface epochs have since been removed. The first 16h implementation retains 32h paging and the existing surface/geometry launches. It adds three numerical choices: retain exactly valid derived geometry, update a surface foot or fill using a local model within its error budget, or run the existing detailed calculation. It does not classify scenes as hydrostatic, suppress current pressure, freeze surface advection, or change dispatch counts in response to motion.

`UniformSurfaceWorkControl` records independent surface and geometry evidence. Geometry compares resolved phi inputs and ownership/address metadata exactly. Surface permits a maximum 0.001h change from its retained reference; below-threshold samples do not replace that reference, so many small changes eventually trigger refresh. Surface epochs include neighboring 16h regions because the foot/smoothing stencil reaches 8h. Current and previous body support, static edits, changed geometry and page claims invalidate the corresponding evidence. A separate selection pass and a workgroup-uniform owner broadcast avoid races and invalid barrier control flow.

The surface predictor retains a well-conditioned stencil normal and smoothing correction, then recomputes the foot from current center phi. Its checks bound the pretrace foot approximation to 0.02 lattice widths and the change from the retained foot to 0.05 widths, and protect smoothing gate transitions. It still traces current velocity, samples current phi, applies wall handling and sources, and publishes every required output. Solid-aware launches may use it only with locally clear stencil evidence; global presence of a body does not exclude distant clear stencils. These checks do **not** establish a formal bound on the final advected field or on errors accumulated by the entire simulation.

Derived geometry has a second approximation independent of temporal reuse: fit an affine plane to the current eight corners and bound its fill error from the corner residual and dominant gradient. It uses one plane-box fill evaluation only when the estimated absolute fill error is at most 1e-4. Otherwise it retains the production quarter-cell/tetrahedral calculation. Cut coarse owners retain solid-weighted integration. The simulation and pressure geometry are refreshed together, and the transient volume mirror still publishes current geometry even on a cache hit.

Evidence snapshots have a fixed memory budget: each bank holds up to 512 selected 16h regions, with all-region epochs and body evidence. The surface model holds up to 8,192 compact fine-owner slots plus coarse vertex slots. Cache ownership changes force refresh, and capacity misses take the full calculation. This is a bounded prototype, not a claim that temporal reuse already scales without a capacity limit to an arbitrary ocean. The current-corner fill approximation continues to work outside the temporal cache.

Two costs remain explicit: input scans currently generate the evidence instead of piggybacking on existing writers, and the surface predictor removes stencil reconstruction rather than the subsequent characteristic and cubic sampling. Both costs must be included in end-to-end measurements. Extending the savings to those samplers, normals/curvature, extension geometry and pressure coefficients remains necessary for the 5× goal.

### First surface/geometry prototype: measured result

The first prototype is numerically sound on the tested cases but **does not reduce frame time**. Retaining a foot stencil removes too little of the surface operator, and explicit evidence scans consume the geometry saving. This is an unsuccessful performance milestone, not completion of the approximately 8 ms target. Continue with a broader bounded approximation that removes characteristic and donor sampling as well as stencil reconstruction.

Measurements used the actual pre-feature source snapshot, including the earlier pressure residual change and the user's staged smoothing fix; `FORCE_FULL` on the new implementation was not the baseline. Baseline source hash: `c1a21d40df9c6c86e631f942e0f82a65df4f1520e3f19cd8ac5e0835e787adf9`; candidate: `4f9a1b72e20952a11a3a72ddcd3d0af22cc41b362d53d0e7456d787d3d28c279`. Runs used the shared GPU lock, the production balanced defaults and authored body, and excluded rendering. Per-pass timestamp profiling was enabled for both variants. Instrumented work counts came from separate runs.

| Scenario | Actual baseline | Prototype | Interpretation |
|---|---:|---:|---|
| Authored settled pool, frames 61–120 | 40.869 ms mean; later recheck 41.045 ms | 42.311 ms mean | 3.1–3.5% slower; recheck median 41.151 → 42.270 ms |
| Localized sphere impact, frames 1–30 | 29.044 ms mean, 27.162 ms median | 29.637 ms mean, 27.919 ms median | 2.0% slower by mean, 2.8% by median; first-frame cold work is included in both |

The settled recheck followed all candidate captures to check timing drift after long Metal compilation. Its close agreement with the original baseline supports a real regression rather than a small speedup obscured by timing history. These are serial measurements, not a statistical confidence interval.

For settled frames, geometric fill fell from 1.185 ms in the recheck to 0.667 ms. However, geometry evidence cost 0.658 ms and surface evidence 0.228 ms. Deferred surface advection increased from 2.709 to 3.313 ms; ordinary surface advection was 1.933 versus 1.951 ms. The localized impact had 0.308 ms of combined controller cost and deferred surface advection increased from 1.488 to 1.772 ms. Net stage costs, including classification, govern acceptance.

Separate steady diagnostics (frames 61–72) recorded 2,697 predicted and 346,815 full foot evaluations per surface encode: only **0.772% predicted**. These count calls, including possible deferred attempts, rather than unique vertices or equally expensive operations. Many full calls can take the existing far-band return. During impact, all 3,681,227 foot evaluations across 30 frames used the full path. The final settled encode recorded 100 dirty surface regions of 384 stored regions and 64 dirty geometry regions, with 64 cache misses. Geometry counters reset on every encode; these are final-encode statistics and do not measure aggregate work saved across all geometry passes in a frame.

Quality comparisons against the actual baseline:

- Settled resolved phi (13,282,017 floats) and velocity (52,428,800 floats) were bit-identical. Expanded density changed at 80,896 samples by at most 5.96046e-8, RMS 3.31525e-9. Interface-band phi and liquid velocity errors were zero. Maximum speed remained 5.02773e-6 m/s and reported volume drift was zero.
- Impact resolved phi, density, velocity, negative boundary velocity, geometry target and center phi were all bit-identical. Pressure reaction was 494.054 N s, deepest body center 1.317813 m, crown 0.400 m above the pool, and final particle count 104,184 in both runs. Maximum body momentum/pressure-impulse discrepancy was 1.3263e-5 N s. No nonfinite values were found.
- The legacy `gridPressureTexture` capture is a one-value placeholder in this mixed solver; it is not evidence of mixed pressure-field agreement. Current pressure residuals and the physical momentum comparison provide separate checks.
- Production GPU regressions also exercise moving inputs, accumulated subthreshold drift and refresh, controller cache collisions/halos/page claims, surface smoothing, geometry error bounds, rest, curved drop, waterline and buoyancy. These tests establish the tested behavior, not a formal global accumulated-error guarantee.

The pool's surface predictor uses 55,015,120 bytes, plus 34,699,680 bytes for the controller. Total surface allocation is 57,745,240 bytes, compared with 2,714,688 bytes before the feature. The bounded capacities are 8,192 fine-owner slots and 512 evidence regions. The impact scene uses 41,484,880 predictor bytes and 9,736,736 controller bytes. These costs also remain part of the acceptance decision.

Raw captures and comparisons are under `artifacts/work-adaptive-investigation/work-control-*.json`, including `work-control-before-settled-recheck.json`, `work-control-settled-comparison.json`, `work-control-settled-recheck-comparison.json`, and `work-control-impact-comparison.json`. `tools/compare-pool-work.ts` reports whole-field and interface/liquid-restricted maximum/RMS errors. Final repository type checking passed. The next experiment must demonstrate meaningful reductions in expensive transport work and net frame time while retaining current-source/body/topology wake-up and explicit accumulated error budgets.

### Current experiment: three costs for surface transport

The next implementation keeps the single 16h control lattice and 32h paging. The existing extended-velocity certificate now publishes signed, unquantized velocity intervals per region from the face values it already reads. This is current velocity evidence, not a still/moving classification. Its halo intervals include an outward floating-point allowance; exact zero remains exact zero. The surface evidence scan also fits current affine phi models and bounds their residuals over the resolved vertices it already loads. Current models cover every stored region, including regions outside the bounded temporal cache.

Every transported vertex can use one of three costs:

1. **Current affine update:** use the midpoint of the current certified velocity enclosure and update current phi using the regional gradient, without a point velocity sample. The local error estimate includes the affine residual, velocity variation, omitted smoothing and floating-point error. This replaces foot reconstruction, the characteristic trace and cubic interpolation together.
2. **Current correction with cheaper transport:** compute the full current foot and smoothing, then use one velocity sample and trilinear Euler transport. Its error estimate shrinks with travel even when the field is not almost planar. This avoids forcing calm curved water directly to the full operator.
3. **Full treatment:** retain the existing characteristic and cubic interpolation wherever the bounds, coverage or solid/wall checks reject the cheaper choices.

Both approximate choices currently require a local phi-error estimate no larger than `0.02h` and travel no larger than half a base cell. They publish current output every step. These are local comparisons with the full operator, not a proof that the entire coupled simulation stays within `0.02h` after many frames. Long-run field, volume and rigid-body comparisons remain necessary.

The middle tier's cross-cell estimate uses canonical vertices that are corners of every possible destination cell. Relative to the same affine model anchored at that vertex, the cubic residual changes by at most `6 rho * travel_1`, and the trilinear residual by at most `2 rho * travel_1`. The production cubic clamp preserves the bound because the trilinear result lies inside the inner-corner interval. The full current smoothing is retained so it cancels in this comparison. A separate position-rounding allowance covers the characteristic's coordinate arithmetic. Required support must avoid projection and embedded-wall effects; wall release and sources retain their existing handling.

Early diagnostics exposed a spatial bound problem rather than a need for a larger motion threshold: 150 of 384 pool regions had local phi residual below `1e-6 m`, but unrelated solid/air markers inflated broad halo residuals to 10–100 m in 336 regions. A blanket 27-neighbor test admitted almost no cheap work. The current implementation publishes a coverage mask and, when the broad residual fails, tightens it using only the neighbors intersecting the actual sampling footprint. It still includes every marker that the operator could read. This preserves the single control level; it does not introduce smaller storage pages or recursive adaptivity.

Current-corner geometry now permits absolute owner-fill error up to `0.005`, using the same residual/slab bound and exact fallback. This is a local fill bound, not a claim of 0.5% total mass error. Cut coarse owners keep solid-weighted integration. The production-target test checks 9,820 configurations, including gentle curvature, against the reported bound.

The diagnostic probe supports twenty surface counter words. Word0 now counts early far-band endpoint acceptance and word1 counts actual-cell far-band acceptance; older captures used both words for the removed foot cache. The remaining words describe affine and middle-tier attempts, their maximum accepted phi-error bounds in meters, far-band middle acceptance, and detailed rejection reasons. Affine rejection can enter the middle tier, so rejected attempts cannot be added or treated as equally expensive full traces. Separate per-region final-store counters classify identity, affine, middle, full trilinear and full cubic publication. They exclude discarded deferred attempts and provide a more useful cost denominator. The maximum local error bound does not establish a global accumulated bound. Counter atomics and readbacks are enabled only by `WORK_COUNTERS=1`; performance runs keep them off. Per-frame `readStats()` is deliberately avoided because it requests another volume measurement and changes diagnostic work cadence.

`tools/compare-pool-activity.ts` compares timed and diagnostic captures of the authored settled pool and `SCENARIO=pool-impact`, which preserves the same container, geometry, timestep and paging while starting the ball immediately above the water. It reports final-store tier counts and dirty normal/geometry region counts by horizontal distance from the splash center. World coordinates decode the current page-window frame as well as packed block coordinates. Timings include controller work; diagnostic atomics remain off. Report the impact time with its cold first frame excluded alongside the complete 30-frame window, so startup cost is not mistaken for activity-dependent work.

Before accepting this broader approximation, compare the measured characteristic/cubic stage cost and net frame time with the actual baseline, inspect hit/rejection distributions on settled and localized-impact scenes, and compare field errors near the interface and in liquid as well as across whole arrays. Moving-plane and accumulated-drift GPU fixtures must show nonzero updates, budget refresh and fallback; a frozen settled surface is not sufficient evidence. Source/body/topology wake-up and the current global pressure response remain required.

`FORCE_FULL=1` is a transport/geometry work-control diagnostic: it does not disable the retained normal anchor and is not a whole-pipeline full reference. Actual pre-feature source captures remain the reference.

The next normal revision evaluates candidate normals in the force normal pass instead of repeatedly publishing them from geometry, and retains an anchor under a `0.01` local capillary-error criterion. That criterion bounds the local pre-projection capillary step; it does not bound the accumulated trajectory, long-time equilibrium or projected pointwise velocity. Fixed-anchor hysteresis can damp tiny capillary modes. Topology/body changes refresh exactly. The existing curved-drop regression uses surface tension `0.072 N/m`, but its ten gravity-driven frames check fall speed and particle detail rather than a global capillary shape or volume bound. Longer capillary-mode validation remains necessary before claiming those guarantees.

### Three-tier revision: measured result (22a3b526)

The broader approximation substantially changes the work mix, but **still regresses total frame time**. These captures use source `22a3b526c6820da3ab5e448710aa61ff543540d95e071833f6b6bfad34c46d29`, the actual pre-feature snapshot, unchanged 32h paging and timestep, and the authored body. Timed runs disable diagnostic atomics and include controller overhead.

| Same authored pool | Actual baseline | Three-tier revision | Change |
|---|---:|---:|---:|
| Settled frames 61–120, mean ms | 41.045 | 42.291 | +3.04% |
| Impact frames 2–30, mean ms | 49.363 | 54.695 | +10.80% |
| Impact all 30 frames, mean ms | 61.801 | 68.078 | +10.16% |

The first impact frame includes startup work. The original smaller impact benchmark also regressed: 29.044 to 30.587 ms across all 30 frames (+5.31%). It retains its separate physical checks and is not a same-domain activity comparison.

Final-store diagnostics show that numerical work responds to activity. In settled frames, full transport publishes only 665 values per frame; same-pool impact averages 206,496 full publications. The following counts average separate diagnostic windows (settled 61–72 and impact 1–30). Cheap means affine plus middle-tier transport; identity stores are excluded. Region distance is horizontal distance from the impact center.

| Distance | Stored regions | Settled cheap / full stores | Impact cheap / full stores | Settled / impact dirty normal regions |
|---|---:|---:|---:|---:|
| 0–1 m | 24 | 13,088 / 0 | 2,304 / 29,994 | 21.25 / 24 |
| 1–2 m | 44 | 44,892 / 0 | 4,477 / 42,394 | 41.25 / 44 |
| 2–4 m | 220 | 171,364 / 160 | 37,424 / 128,214 | 191.75 / 220 |
| 4–8 m | 96 | 7,259 / 505 | 750 / 5,894 | 70.75 / 96 |

GPU pass costs explain why broad cheap admission is insufficient. Settled forces decrease from 3.933 to 3.422 ms, and geometry from 1.185 to 0.714 ms, but deferred surface rises from 2.709 to 3.234 ms; geometry and surface controllers add 0.891 and 0.544 ms. During same-pool impact, geometry rises from 1.479 to 3.810 ms and deferred surface from 3.447 to 5.145 ms. Repeated mixed-grid normal publication and late cheap-path certification are the next measured targets. This revision does not meet the wall-clock success criterion.

Settled near-interface phi (baseline `abs(phi) <= 0.1 m`) differs by at most `3.80e-7 m`, RMS `7.36e-8 m`; density maximum difference is `2.68e-6`. Same-pool impact near-interface phi differs by at most `0.11390 m`, RMS `0.003059 m`; smaller impact differs by at most `0.12047 m`, RMS `0.006212 m`. Whole-field phi maxima are 0.03723 m settled, 0.62096 m same-pool impact and 4.4091 m smaller impact; distant markers and changed ownership contribute to these maxima. These are measured accumulated differences, not local-budget guarantees. Raw velocity texture differences are recorded in comparison artifacts, but mixed-grid ownership can differ, so their maxima require authoritative sampling before interpretation as physical velocity errors. No captured field contains nonfinite values.

Same-pool baseline/candidate body impulse is 494.844 / 494.486 N s, minimum ball height 1.343062 / 1.343376 m, crest 0.5 / 0.5 m above the waterline, and particle count 106,961 / 104,195. Smaller-impact candidate impulse is 493.921 N s, depth 1.317628 m, crest +0.4 m, particle count 104,972; its physical assertions pass. Momentum-identity errors remain below `1.8e-5`; reported volume drift is zero. Close aggregate body metrics do not establish field agreement.

The removed temporal-foot cache now uses zero bytes. The pool's controller uses 17,642,912 bytes, surface module 3,023,248 bytes and bounded normal/geometry allocation 14,342,292 bytes. The latter previously scaled with the full fine capacity. GPU fixtures validate coarse and mixed fine normal equivalence, including capacity overflow and growth.

Artifacts under `artifacts/work-adaptive-investigation/`: `transport-{settled,pool-impact,impact}-comparison.json`, `transport-activity-comparison.json`, timed `transport-after-*.json`, baseline `work-control-before-settled-recheck.json` and `transport-before-pool-impact.json`. Their field and regional sidecars preserve the underlying evidence.

The probe accepts `IMPACT_SPEED` (positive downward speed, default 8 m/s), `IMPACT_X` and `IMPACT_Z` (default zero). These are recorded in configuration and preserve pool geometry and ball size. `FROM` already filters both wall-clock and per-pass summaries. Next-revision runs at 0.1, 1 and 8 m/s will test whether work scales through intermediate activity rather than only between unchanged endpoints and full transport.

### Endpoint-proof / force-normal pilot (f0d4bdf5)

The next pilot moves normal evaluation into forces and adds early endpoint-equivalence proofs. Source `f0d4bdf506c0fd4cc3ef3ff7505219f253217b5b71b3405c9d634bb1f0d08833` passes types, 859 unit tests (35 skipped), and production surface/normal GPU fixtures. Settled frames 2–30 average 42.143 ms; this short early window is not the matched 61–120 acceptance window. The settled deferred pass falls to 1.406 ms and geometry to 0.368 ms, but surface owner work rises to 2.425 ms. A diagnostic frame has zero dirty normal regions and 248,358 early endpoint acceptances.

Same-pool impact still regresses: candidate frames 2–30 average 55.750 ms versus the fresh baseline 49.467 ms (+12.70%). Geometry improves to 1.776 ms, but deferred surface reaches 7.161 ms, owner surface 2.521 ms and forces 4.441 ms. Expensive failed admission attempts now outweigh the saved geometry work. This pilot is rejected as a performance result; no final 120-frame run is justified yet. Artifacts use the separate `transport-local-*` prefix.

A common-world-point velocity capture now samples production mixed MAC interpolation at 7,200 fixed 4h centres throughout the authored liquid box and 38,400 h centres in the impact surface neighbourhood. Sampling applies the current page-window frame before lookup. The original pilot fields omitted this frame conversion and produced invalid all-zero authoritative-velocity segments; their timings and other fields remain valid. Corrected captures carry the `-qa` suffix. Their common-point velocity vector difference has maximum `1.06969 m/s`, RMS `0.022007 m/s` across 45,600 points; the maximum occurs near the ball at `(-0.125, 1.525, 0.075) m`. The coarse liquid-box subset has maximum `0.39818 m/s`, RMS `0.006666 m/s`; the fine impact-surface subset has maximum `1.06969 m/s`, RMS `0.023807 m/s`. Corrected impact interface phi differs by maximum `0.14077 m`, RMS `0.003849 m`. The unmasked maximum lies inside the sphere. Restricting velocity to trilinearly sampled `phi < 0` in both runs and outside both spheres gives maximum `0.73540 m/s`, RMS `0.013409 m/s` at 41,769 points. Requiring another h of sphere clearance gives maximum `0.30187 m/s`, RMS `0.011092 m/s` at 41,417 points; the maximum is at `(-0.875, 1.775, -0.825) m`, away from the body. Interface phi outside both spheres still differs by maximum `0.09969 m`, RMS `0.003057 m`. These older captures recorded final body Y but omitted X/Z; mask metadata explicitly uses configured X/Z zero and actual final Y. Future captures record full final XYZ. These sizable local differences remain unresolved despite similar body impulse and crest. The comparator checks identical world positions and reports authoritative vector maximum/RMS errors; raw physical-storage velocity metrics remain explicitly labeled and cannot substitute for this comparison.

### Home-gate / actual-cell pilot and geometry ablation (3cc295ab)

Home-only necessary rejection tests and an actual-cell far-band fallback reduce failed surface certification. Four profiled 30-frame captures use frozen source `3cc295ab6b72ab16d87d99766736d7323d0800082226789a916bec2259df4dd6`. Settled uses frames 2–30; impact warm time excludes frame 1. No large field capture or final 120-frame acceptance run is justified by this pilot.

| Geometry snapshots | Settled mean ms | Impact warm mean ms | Settled geometry + controller ms | Impact geometry + controller ms |
|---|---:|---:|---:|---:|
| Off: metadata-only evidence | 42.649 | 56.151 | 1.801 + 0.409 | 2.508 + 0.435 |
| On: exact phi snapshots | 42.165 | 55.348 | 0.339 + 0.888 | 1.625 + 0.950 |

Snapshot retention saves more geometry evaluation than its scan costs in both scenarios, so the metadata-only ablation is unsuccessful here. Both variants produce the same recorded impact impulse (494.440 N s), body depth (1.342893 m), crest (+0.5 m) and particle count (104,718). This is aggregate agreement, not a full field-equality test.

The impact GPU stage averages in this ablation cover all 30 frames; its warm wall-clock column excludes frame 1. With snapshots on, settled owner/deferred surface costs are 2.443 / 1.293 ms; impact costs are 2.394 / 6.724 ms. The latter deferred pass improves from 7.161 ms in the previous pilot, but total frame time still misses the pre-feature baseline by a substantial margin during impact. Force work remains 3.869 / 4.443 ms settled/impact. Artifacts are `transport-gates-{off,on}-{settled,pool-impact}.json`; counters run separately. Further changes must remove measured operator or certification cost rather than increase nominal acceptance counts. Separate impact diagnostics average 24,156 early endpoint and 14,539 actual-cell far-band acceptances per frame. The final impact encode still rejects 89,583 middle attempts for coverage, 1,884 for solid/projection, 21,729 for speed and 220,930 for general error. Final-store full transport averages 30,319 / 39,870 / 103,585 / 5,671 values in the 0–1 / 1–2 / 2–4 / 4–8 m rings. Actual-cell admission helps the shoulders, but detailed impact work remains broad. The five-frame settled diagnostic reaches zero dirty normal regions by its last frame; its startup-inclusive mean is not a steady-state estimate.

### Shared support-table / trace-seed pilot (127cae18)

The next revision replaces repeated local-neighbor reductions with 64 shared support records per 16h region and reuses the cheap tier's current-point velocity as a full-trace seed. Frozen source `127cae18e4165ebe96becf991b3f8eb51b1c5331d12abb4929121dc874f57eeb` passes producer footprint/poison/stale-record tests and seven production surface/helper GPU tests. Geometry snapshots remain enabled after the preceding ablation.

Settled frames 2–30 improve from 42.165 to 41.734 ms (median 41.482 ms). Same-pool impact frames 2–30 average 56.175 ms versus 55.348 ms previously, but contain one 87.67 ms frame among otherwise 49–57 ms frames; its median improves from 55.601 to 54.836 ms. Keep the outlier in the mean and treat the net impact result as uncertain rather than asserting a speedup. No final 120-frame run was started.

For the all-30-frame GPU stage window, impact deferred surface decreases from 6.724 to 5.844 ms and owner surface from 2.394 to 2.333 ms. Surface evidence rises from 0.557 to 0.616 ms; geometry rises from 1.625 to 1.859 ms. Surface/controller/geometry allocations are 10,920,336 / 25,632,544 / 14,373,140 bytes. Sharing the support records reduces repeated computation but adds storage and publication work; the remaining failed-path and force costs need direct subgroup measurement.

The final calm diagnostic has 233,829 early endpoint acceptances and zero dirty normal regions. The final impact encode has 4,908 actual-cell acceptances, 104,112 coverage rejections, 918 solid/projection rejections, 23,647 speed rejections and 213,145 general-error rejections. These are attempts; per-region final stores remain the cost denominator. Artifacts use `transport-support-*`. Body impulse is 494.820 N s, minimum height 1.343087 m, crest +0.5 m and particle count 105,789; no new whole-field quality comparison is claimed for this timing pilot.

### Native-edge / guarded-path pilot (7eeb4d27)

The latest pilot adds native-edge slope summaries without changing the `0.02h` local surface budget, removes independently proved redundant guard work, and takes the cached zero-curvature force exit earlier. Source `7eeb4d279a091daf6a768995369067c034fcdf7e5d4f518ddc2c4b5fd8e48ec9` passes the final ten-test surface and two-test producer GPU suites, type checking and 860 unit tests (47 skipped). Heavy CPU checks finished before measured frames.

Every timing below uses exactly frames 2–30, including outliers. The support-pilot stage values have been reaggregated from per-frame timestamps to that same window; they therefore differ from its all-30-stage figures above.

| Same-window pilot | Settled mean / median ms | Impact mean / median ms |
|---|---:|---:|
| Support-table predecessor | 41.734 / 41.482 | 56.175 / 54.711 |
| Native-edge revision | 42.573 / 42.176 | 53.852 / 53.796 |

The native-edge impact improves against its predecessor, while settled water regresses. Its actual pre-feature impact baseline is 49.232 ms for frames 2–30, so the new revision remains slower. The stored baseline settled profile starts at frame 61; its 41.045 ms mean must not be treated as a matched acceptance comparison against this early 30-frame pilot. A final 120-frame acceptance run has not been justified.

Sum of surface owner/deferred, geometry fill, both controllers and page-claim GPU passes is 5.733 ms settled / 10.829 ms impact, versus predecessor 5.580 / 11.338 ms. Including the entire force pass, which contains normal/curvature work, gives 9.482 / 15.202 ms versus 9.424 / 15.751 ms. These sums cover named pass execution, not every intervening copy/idle or CPU cost; frame wall time includes those costs. The exact-window impact baseline totals are 6.764 ms without forces and 11.033 ms including forces. Native impact owner/deferred surface costs are 2.305 / 5.412 ms, geometry 1.360 ms, controllers/page claims 1.751 ms and forces 4.373 ms.

Separate counters leave substantial full work: final settled stores are 3,531 affine, 243,713 middle, 40,153 full linear and 87 full cubic; final impact stores are 35,535 middle, 197,135 full linear and 28,833 full cubic. The controller/surface allocations rise to 28,640,224 / 13,881,744 bytes. Timing and count artifacts use `transport-native-*`; no new field-quality acceptance is claimed.

Because total performance remains unsatisfactory, per-dispatch split diagnostics precede more full captures. They distinguish normal cache, curvature cache, fine/coarse/fused force paths and solid/fast/merged surface variants. Pass splitting adds boundaries and timestamp overhead; its values identify expensive operators rather than predict production wall time. At settled frame 5, split times identify fine solid-twinned forces (2.032 ms), fused forces (0.852 ms), normal cache (0.590 ms), merged owner surface (1.769 ms) and merged solid-twinned deferred surface (1.180 ms). At impact frame 30, merged solid-twinned deferred surface dominates at 5.898 ms; fine forces cost 2.359 ms, merged owner surface 2.032 ms, fused forces 0.983 ms, normal cache 0.721 ms and curvature cache 0.262 ms. Solid-free variants are mostly empty. Timestamp values are quantized near 0.065536 ms, and pass splitting changes boundaries, so these are operator-location diagnostics. Both captures finished on frozen source before further edits.

The speed-curve runner is prepared for 0.1, 1 and 8 m/s in the same pool and exact 2–30 window, with separate timed/counter runs and optional actual baseline captures; the curve has not yet run.

### Matched final surface/geometry capture (0a00bd8f)

The final capture of this phase combines displacement-scaled affine transport, regional midpoint velocity, retained geometry/normals and exact solid-predicate elision in the full foot/cubic fallback. Candidate source is `0a00bd8f7ac0a5b1130c2bf13f318a6b50452073b71bfb5a0ac1fc03b94602b2`; the actual pre-feature snapshot is `c1a21d40…`. Both retain the authored ball, 32h paging, 16h work regions, h=0.05 m and dt=1/60 s. Four serial shared-lock runs use identical current probe code, diagnostic atomics off and geometry snapshots on. Required CPU checks finished before timing. The candidate's final seven focused GPU checks passed, including bitwise same-state strong-motion comparisons for exact predicate elision.

| Exact matched window | Actual baseline wall mean | Candidate wall mean | Change |
|---|---:|---:|---:|
| Settled, frames 61–120 | 40.840 ms | 41.201 ms | +0.88% |
| Settled, frames 2–30 | 41.459 ms | 42.127 ms | +1.61% |
| Same-pool impact, frames 2–30 | 49.524 ms | 54.638 ms | +10.33% |

Settled deferred surface improves from 2.689 to 1.315 ms, but owner surface grows from 1.922 to 2.328 ms and controllers/page claims add 1.707 ms. Geometry fill drops from 1.177 to 0.360 ms. In the impact, surface owner/deferred grow from 1.959/3.516 to 2.414/5.792 ms. Surface, geometry and controller pass sum grows from 6.766 to 11.302 ms; including the full force pass gives 11.048 to 15.688 ms. These include shared controller costs. Frame wall time also includes costs outside those named GPU passes. This revision does not meet the requested performance result; counts and intermediate-speed runs are paused for review rather than treated as acceptance.

Quality uses final current fields, not stale budget telemetry. After timing, the existing read-only phi-volume measurement updates the diagnostic budget without advancing dt. At settled frame120 both volumes are 51.200000 m³. At impact frame30 they are 51.243742 and 51.240102 m³, a candidate difference of −0.003641 m³ (−0.00710%). Both actual final sphere positions are recorded. Authoritative MAC velocity is sampled at the same 45,600 worldpoints; raw storage comparisons remain separately labeled because ownership may differ.

| Final field comparison | Maximum difference | RMS difference |
|---|---:|---:|
| Settled interface phi, baseline abs(phi)≤0.1 m | 1.81e−6 m | 3.42e−7 m |
| Settled joint-liquid authoritative velocity | 1.34e−5 m/s | 2.44e−6 m/s |
| Impact interface phi, unmasked | 0.11796 m | 0.002946 m |
| Impact interface phi outside both final spheres | 0.08694 m | 0.002546 m |
| Impact authoritative velocity, unmasked | 1.05997 m/s | 0.025221 m/s |
| Impact joint-liquid velocity outside both spheres | 0.73064 m/s | 0.011464 m/s |
| Same joint-liquid mask with one-h sphere clearance | 0.26610 m/s | 0.008864 m/s |

Joint liquid requires trilinear resolved phi<0 in both fields. Sphere exclusion uses the union of both measured final centers and the authored 0.24 m radius. All captured fields are finite. Close impact impulse (494.844→494.732 N s), equal 0.5 m crest and similar particle counts (106,961→105,310) do not establish global field agreement. The local 0.02h transport and 0.01 normal-anchor criteria do not bound accumulated trajectory or projected pointwise velocity. The substantial impact differences remain an unresolved quality limitation.

Artifacts use `transport-exact-{before,after}-{settled,pool-impact}` with JSON, field binary and metadata sidecars; `transport-exact-{settled,pool-impact}-comparison.json` record full masks and norms. `transport-exact-settled-early-timing.json` uses exactly frames2–30 and intentionally omits frame120 fields. The single-float pressure placeholder is not a pressure-field comparison. Current counts and the speed curve have not been captured on this revision.

### Local-solid implementation sprint timing (31924d72)

Two quick integrated candidate profiles use frozen source `31924d72e0265ee56742993b306fc9de1196dacad8f31820bb725c97f5afb1bb`, compared with the existing actual pre-feature captures over exactly frames2–30. Both retain the authored body and geometry snapshots. No fields, diagnostic counters, speed curve, repeated runs or new baseline captures were collected.

| Same2–30 window | Actual baseline mean / median | Candidate mean / median |
|---|---:|---:|
| Settled | 41.459 / 41.117 ms | 40.997 / 40.743 ms |
| Same-pool impact | 49.524 / 49.385 ms | 54.544 / 53.946 ms |

Settled mean improves1.11%; impact mean regresses10.14%. Settled forces improve3.914→3.557 ms and pressure-band list/rows2.583→2.298 ms. Impact list/rows improve2.886→2.547 ms, while forces are4.282→4.314 ms and surface owner/deferred1.959/3.516→2.438/5.971 ms. Surface/fill/controllers total5.790→5.758 ms settled and6.766→11.494 ms impact; including forces9.704→9.315 and11.048→15.808 ms respectively. These exact operation exits improve selected costs but do not establish the required overall work scaling. This timing-only capture makes no new field-quality or current-volume claim; ordinary budget volume telemetry may be stale. Artifacts are `solid-sprint-{settled,pool-impact}.json` and matching `-comparison.json`.

### Milestone 1 implementation

The frame now supplies `min(requested pressure tolerance, 1e-4)` in divergence units to the band. The fallback for an unset/nonpositive requested tolerance is 1e-4. The band's initial measurement runs after coarse-pressure rebasing and required shut-row settling, before ordinary fine smoothing. Every iterative kernel still launches; its existing live-count gate exits when a completed measurement meets the bound. Fine projection and pressure publication still run.

An explicit validity bitmap distinguishes a measured zero from cleared storage. The initial residual has a separate word; cycle histories keep their previous offsets. Reductions only consult earlier completed measurements, never the global maximum currently being assembled by other workgroups. Header clearing resets all evidence every frame. Nonfinite residuals reduce to infinity, including at complementarity bounds, so they cannot authorize an early exit.

The first regression manufactures zero, a large residual in the latter half of the band, and a nonfinite residual, then returns to zero. It uses the production solver and shaders; only those manufactured RHS perturbations suppress projection to isolate the convergence mechanism. The physical pool, falling-drop, impact, and buoyancy regressions retain normal projection. This is a band-wide gate: one genuinely active region still wakes the whole band. Spatial stopping remains milestone 6.

Validation of the implementation: `npm run check:types` passed; `npm run test:unit` passed with 859 passing and 25 GPU-dependent tests skipped. The explicit Metal run below passed all eight tests, with no skips. The authored pool stayed particle-free for 120 frames, with a final peak component of 3.92e-6 m/s. The impact test recorded 494.05 N s of pressure impulse, a minimum ball-center height of 1.318 m, and a crown 0.400 m above the pool. Stationary waterline, freely floating, and larger floating-body cases also passed their existing physical bounds. These checks validate this first pressure change; the broader transport/cache acceptance matrix below still applies to future milestones.

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test --test-concurrency=1 tests/uniform-pressure-band-convergence.test.ts tests/uniform-pool-rest.test.ts tests/uniform-rigid-impact.test.ts tests/uniform-rigid-rest.test.ts
```

The GPU log is retained at `artifacts/work-adaptive-investigation/pressure-band-gpu-tests.log`. Cold Metal compilation accounts for most of the roughly eight-minute run, rather than simulation time.

### Milestone 1 measurements

Fresh native Metal measurements of the authored scene, still frames 61–120 at balanced defaults and 50% particle budget:

| Implementation | Mean frame | Band solve | Completed band V-cycles at frame 120 | Final maximum speed |
|---|---:|---:|---:|---:|
| Original implementation, rerun from an isolated source snapshot | 43.16 ms | 2.87 ms | 4 | 2.74e-5 m/s |
| New implementation, stopping disabled | 43.49 ms | 3.16 ms | 4 | 2.74e-5 m/s |
| New implementation, 1e-4 stopping bound | 41.42 ms | 1.23 ms | 0 | 5.03e-6 m/s |

The original-source recheck reproduced hash `7df676ef18ced0beb4c37c3ac0af752cd92f89afde392baf20f40346fc8feef5`. Both new runs have hash `86556d1689b4cfd8c337f53e16d2e92cf6ad549d3ef9af76dfd4a338a4b09f22`; they encoded identical launch schedules, and every frame matched ownership counts, particle counts, band size, remap jobs, and transfer pairs. All three reported zero volume drift. The new full-work control includes the added initial scan and validity handling, so its 2.07 ms difference from stopping is not the net improvement over the original code. The original-to-new reduction in this sample is **1.74 ms per frame, about 4%**, including **1.65 ms less band solve work**. These are sequential sample means, not a statistical confidence interval.

Zero completed V-cycles does not mean zero fine smoothing: at frame 120 the initial residual was 1.0613e-4 s^-1, above the bound, so ordinary pre-smoothing ran before restriction measured 1.2365e-5 s^-1 and the remaining work exited. This first change does not eliminate band initialization, row assembly, or projection. The 5× total-frame target remains open.

An additional unprofiled 600-frame/ten-second rest run with the ball removed remained particle-free on every frame, ended at 1.64e-5 m/s maximum speed, and reported zero volume drift. Removing the ball here isolates prolonged resting stability; it is not a performance claim for the authored scene.

Raw results are in `artifacts/work-adaptive-investigation/pool-original-recheck.json`, `pool-band-full-work.json`, `pool-band-early-exits.json`, and `pool-band-long-rest.json` in the same directory.

## Original investigation measurements

The production NB-FLIP solver was run on native Metal with the authored backyard scene, balanced defaults, h = 0.05 m, dt = 1/60 s, and its actual rigid ball. Construction and rendering are excluded. The steady profiles measure frames 61–120. They use the default 50% adaptive particle budget; the supplied screenshot shows 10%, so these are a reproduction of the workload problem, not an exact replay of that UI configuration.

All original investigation runs in this section have solver-source hash `7df676ef18ced0beb4c37c3ac0af752cd92f89afde392baf20f40346fc8feef5`. Pass timestamps add overhead; the per-kernel split profile is diagnostic only. Neither pass splitting nor changing dispatch counts is a proposed optimization.

| Experiment | Profiled frame mean | Band solve pass | Extension passes | Final maximum speed | Reported volume drift |
|---|---:|---:|---:|---:|---:|
| Authored scene | 42.53 ms | 2.80 ms | 3.77 ms | 2.74e-5 m/s | 0 |
| Probe-only band tolerance 1e-4 s^-1 | 40.68 ms | 1.01 ms | 3.78 ms | 6.49e-6 m/s | 0 |
| Ball removed, otherwise same scene | 35.20 ms | 2.81 ms | 1.25 ms | 2.74e-5 m/s | 0 |

All three runs retain zero particles. The baseline and ball-removed runs also match final band residual histories and reported layout counts. These aggregate comparisons do not establish complete field equivalence. The tolerance experiment covers two seconds before impact; it does not validate the change for splashes or long-term rest.

Separate steady runs without pass profiling measure **41.54 ms/frame** for the authored scene and **39.91 ms/frame** with the probe-only band tolerance, a 1.63 ms reduction. Their files are `pool-plain-steady.json` and `pool-band-plain-steady.json`. These are single serial runs, not a statistical performance gate.

The baseline has 5,556 h tiles, 19,020 coarse tiles, 4,704 band tiles, and zero remap jobs/particle transfer pairs. The band contains 301,056 h row slots, including non-liquid rows. Particle inactivity has not removed this numerical workload.

The following are sums of pass durations per frame, not the UI's phase boundaries. A label may occur more than once.

| Baseline work | Mean GPU pass time | Immediate explanation |
|---|---:|---|
| Surface advection and deferred advection | 4.62 ms | Characteristics and surface sampling still run on owned vertices |
| Body forces | 3.91 ms | Capillarity/viscosity/bed and gravity work over the represented field |
| Velocity extension | 3.77 ms | Head extension plus post-pressure extension work |
| Band solve | 2.80 ms | Four cycles despite tiny residuals |
| Band list and rows | 2.64 ms | Membership, coefficients, forced faces, and aggregates rebuilt |
| Momentum | 1.52 ms | Grid advection survives zero particle count |
| Surface pressure authority | 1.49 ms | Repeated phase/pressure geometry evaluation |
| Crossing searches | 1.30 ms | Search data rebuilt at several surface stages |
| Pressure RHS | 1.19 ms | Current flux calculation, including body-aware terms |
| Geometric fill | 1.18 ms | Derived quantities rebuilt from phi |
| Redistance | 0.87 ms | Distance construction over the existing interface neighborhood |
| Surface reconstruction | 0.80 ms | Grid publication survives absent particle contributions |

The split profile locates the heavy work inside `advectDeferred`, `bandPrep`, `forcesFine`, `advect`, `bandSweep`, `momentumUnitStep`, `buildRhs`, `geometry`, and `depthSeeds`. The issue is substantially the numerical payload, not just launching kernels.

Measured artifacts are in `artifacts/work-adaptive-investigation/`: `pool-passes-steady.json`, `pool-band-tolerance.json`, `pool-no-body.json`, and `pool-kernels.json`. Early 24-frame profiles are retained but should not be used for speedup comparisons: they include startup drift and overlap with CPU checks. The steady comparisons above were run serially after those checks.

## Concrete causes found in the investigation

### Band convergence was effectively disabled (fixed in milestone 1)

Before milestone 1, `UniformMixedFrame` initialized band parameter word 8, `solve.x`, to zero and `write()` never updated it. `bMet()` also treated a zero residual word as unmeasured. Consequently neither a positive residual nor an exactly zero residual could satisfy that stopping rule.

At baseline frame 120 the recorded residuals, in s^-1, are:

```
after first pre-smoothing: 8.61e-6
after next pre-smoothing:  5.39e-7
                          1.34e-7
                          8.60e-8
after final cycle:        6.97e-8
```

The original probe changed only the host band's tolerance word to 1e-4. Existing shader gates then stopped after the first pre-smoothing checkpoint, reducing the band solve pass from 2.80 to 1.01 ms. The receipt reported zero completed cycles, but this did **not** mean zero band work: assembly, initialization, rebasing, shut-row settling, and first pre-smoothing had already executed.

Milestone 1 adds the initial residual check before ordinary smoothing, separate measured/valid bits so zero can mean converged, and a strict stopping policy. The root's loose default of 5 s^-1 is not a suitable band threshold. The rest tests and initialization comments document why apparently small pressure errors can grow through surface feedback.

Sources: [parameter initialization](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-mixed-frame.ts:439), [parameter writes](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-mixed-frame.ts:739), [band convergence](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-pressure-band.ts:350), [solve order](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-pressure-band.ts:1317).

### A distant body changes the fluid execution path

The frame head invalidates extension whenever bodies are supplied. The relayout path then treats solids as moved, while the tail's extension reuse is restricted to frames without bodies. Band RHS construction additionally enables body flux work wherever a liquid row exists and the global body count is nonzero.

The solid record itself already has a useful local predicate: `umBodyTileDirty()` considers old/new body neighborhoods. The proposal is to carry that locality through the dependent stages. The ball experiment removes 7.32 ms from the profiled mean even though the ball has not reached the water. That is a measured opportunity scale, not a guarantee that one predicate will recover all of it: deleting the ball also removes body-dependent branches and integration work.

For each fluid consumer, test the swept body support against that consumer's complete dependency region. A disjoint region retains its solid geometry and extension validity and early-exits body flux evaluation. Include old and new poses, rotational sweep, sampling halos, impending wetting, and newly claimed pages. Body integration still runs normally.

Sources: [frame invalidation](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-mixed-frame.ts:852), [relayout dependency](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-mixed-frame.ts:1335), [local solid dirtiness](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-mixed-solid.wgsl.ts:277), [band body flux](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-pressure-band.ts:663).

### Particle activity is separate from Eulerian work

The adaptive heat already follows shape, strain, impact, sources, and retirement. It controls particle influence. Zero heat/count does not establish that phi, velocity, pressure coefficients, or extension are unchanged.

`reconstruct()` invalidates its band data, clears the metric bank, searches crossings, and publishes phi. `retire()` searches again; `redistance()` searches and rebuilds the metric. `snapshot()` and `update()` construct sampling data before their particle kernels discover there are no samples. Several banks are deliberately shared scratch, so their earlier contents cannot simply be retained by skipping writes.

Mandatory solid promotion also keeps h tiles around wet solid neighborhoods. Every wet h tile becomes a band tile. Removing this representation solely because the flow is slow would invalidate the current boundary contract. Keeping its cached geometry while avoiding repeated work is the less invasive first step.

Sources: [surface and particle sequencing](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-narrow-band-flip.ts:1248), [mandatory solid promotion](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-mixed-dynamic.ts:1242), [band membership](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-pressure-band.ts:589).

## Proposed execution contract

Keep fixed/buffered launches and put a cheap decision at each region's first expensive operation. There are three useful outcomes:

1. **Retain:** required inputs and parameters are unchanged and the output still exists. Leave it valid, or perform the minimal publication copy when input/output banks differ.
2. **Predict:** a cheaper local update has a bounded error within this operator's budget. Publish its error bound with the result.
3. **Recompute:** perform the existing full work, publishing new validity and error information.

A single global “inactive” bit is insufficient. Keep per-tile versions or validity for surface values, crossing geometry, velocity, solid geometry, ownership/page generation, particle occupancy, and derived operators. Separate geometry changes from field-value changes: an unchanged pressure matrix still needs a current RHS.

Produce change evidence while a stage already has its old/new values in registers. Fuse speed ranges, gradient bounds, residual maxima, or changed flags into existing producer work where possible. Propagate dependency information with cheap tile-level operations. Recomputing all fine geometry merely to decide whether to skip that geometry defeats the purpose.

For a launch that uses workgroup barriers, the exit must be uniform across participating lanes. Read one regional decision before staging expensive neighborhoods; for a workgroup striding multiple tiles, decide separately for each tile rather than returning from the entire workgroup at its first quiet tile. Preserve the existing full-work fallback when a bound is unavailable, nonfinite, or invalidated.

This gives the intended cost shape:

```
global coarse pressure work
+ cheap tests over represented tiles
+ numerical work in regions whose operator error requires it
+ minimal publication of retained results
```

There is no dispatch compaction requirement. A small live fine patch inside a large calm domain should add fine numerical cost proportional to that patch and its true dependencies.

### Advance a cheap local model and resolve only its missing detail

Exact cache reuse is the first step, but tiny numerical changes can invalidate it everywhere. The broader mechanism should retain a local low-order model plus a bound on what that model misses. The existing 4h field provides one predictor; planar phi and constant/affine velocity provide useful local models inside h regions that must retain fine solid geometry.

For example, a plane `phi(x) = n dot x + c` transported by constant velocity U remains a plane with `c_new = c - dt * n dot U`. Its surface can move quickly while advection requires a few arithmetic operations per output vertex, its normal stays unchanged, and its distance metric remains valid when n is normalized. No stationary-scene detection is involved. With spatially varying flow, bound the deviation from that model over the full departure footprint and take the existing full sampling path when the bound is too large.

Similarly, retain a measure of fine velocity detail relative to the prolonged coarse/affine predictor. A large smooth translation can have almost no such detail. A small localized vortex can require substantial detail. Static fine cut-cell coefficients can remain authoritative in both cases; a cheap velocity model does not authorize coarsening the wall geometry.

This connects the stages: the surface producer publishes its model, geometric change, and approximation bound; distance, curvature, geometry, and pressure preparation can cheaply update or retain their own results from that evidence. Merely adding independent low-speed thresholds to every kernel would miss this shared saving. Tall cells are a more structural, vertically affine version of this idea, with additional storage and flux changes.

### Output lifetime is part of validity

Current band rows share `stageScratch` with extension and are overwritten after the solve. Curvature uses `phase` plus scratch normals; phase is subsequently republished. Surface nearest-distance banks also have multiple owners within a frame. An early return that leaves such a destination untouched can publish a different stage's data.

The first implementation must separate persistent reusable data from temporary payloads. For band pressure, retain coefficients/kinds/apertures and keep forced velocities, RHS, and iterates transient. At 4,704 band tiles, the current complete solid row payload is about 20.1 MiB. Retaining only diagonal, kinds, six coefficients, and 240 face apertures per tile would be about 13.5 MiB at that live count, before capacity reserve, indexing, and alignment. This is a memory-for-work tradeoff that needs explicit budgeting.

Version persistent state by page generation as well as tile key: a recycled key must never inherit the previous occupant's “clean” flag. Layout changes must transfer or invalidate errors and cached values. Shared canonical vertices/faces and low boundary planes must remain valid even when their neighboring tiles choose different work levels.

Source: [shared scratch and band rows](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-mixed-frame.ts:502).

### Receipt-sized retained caches (follow-on)

The current 512 geometry regions and 8,192 fine normal tiles are bounded cache capacities, not numerical activity criteria. A calm ocean larger than either cache can still fall back to full work. Grow retained payloads from observed residency rather than eagerly allocating them at the page pool or fine-owner reservation ceiling; retain the same 16h control records and conservative miss paths. This is not implemented.

The required counts already return with frame receipts: `workDomainInfo.fineTiles` comes from the ownership work receipt, and `pagePoolInfo.residentPages` comes from the pool status. Geometry currently selects all stored regions, so its complete working-set target is eight times resident pages, not just liquid/dependency regions. Add that existing pool count to the fixed-lag `LaggedPlan` record and use frame N−2 consistently, as current work planning does; do not let whichever asynchronous receipt finishes last change allocation policy. Start with a modest cache, grow geometrically with the existing census/edit reserve, and shrink only after sustained low occupancy. A delayed or underestimated count must only cause a cache miss, never omit current computation.

Use `fineReservation`/`capacityBuffers` to check device binding limits, the configured byte budget, and old-plus-new allocations during replacement. The between-frame `reserveFine` hook already calls `geometry.reserveNormals`, rebinds geometry and force groups, and invalidates geometry outputs. Fine normal lookup already derives its retained prefix from buffer length; replace the fixed 8,192 clamp with a separately budgeted cache reservation driven by the live fine count. Preserve the existing fine-prefix/coarse-tail copy and topology refresh. Optional cache growth should be refused before mandatory simulation refinement: keep the smaller cache and use the original force scratch on overflow.

Geometry needs an additional separation: move snapshots/owner tags into a capacity-sized payload binding with a runtime capacity header, leaving region dirty/topology/model records fixed. Otherwise the current compile-time `WC_CACHE` and offsets require pipeline reconstruction on growth. Preserve GPU region→slot indirection and page-generation/coordinate checks. A later stable free-slot map can avoid moving many cache owners when a newly resident physical block changes compact rank; release and allocate in separate passes so each slot has one owner. No hash collision may silently imply validity.

The honest memory bound remains linear in the cached working set. Current exact geometry snapshots cost 33,792 bytes per retained region: 4,096 regions alone cost 132 MiB, before normals, metadata and transient replacement. Fine normals cost 1 KiB per retained fine tile, plus the current coarse tail at 16 bytes per physical tile key. Growth must stop at the memory budget, with miss counters reported alongside timings. Larger calm worlds therefore still need compact coarse snapshot records, producer-side bounded evidence, or a compressed representation; receipt-driven growth alone does not establish ocean-scale constant work or a fivefold speedup.

Sources: [existing receipt counts](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-mixed-frame.ts:237), [fixed-lag planning](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-mixed-frame.ts:829), [capacity admission and reserve hook](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-mixed-frame.ts:606).

## Criteria for the expensive stages

### Surface advection and surface maintenance

Use **surface motion and representation error**, rather than particle heat or total speed alone.

A conservative displacement bound is `d <= dt * U`, where U bounds velocity throughout the complete characteristic support. With a bound L_phi on the level-set gradient, holding phi incurs at most `L_phi * d` in the frozen-field model. Maintain accumulated error; a sub-threshold movement repeated forever cannot be discarded forever.

Normal interface motion, approximately `dt * abs(u dot grad(phi)) / |grad(phi)|`, can be far smaller than total travel. It is useful for a cheap planar/affine predictor, but a single point's normal velocity is not a certificate: curvature, velocity variation, crossing topology, solid contact, and the entire characteristic footprint matter. Fast uniform tangential flow should be inexpensive where the represented surface supports that conclusion.

The code already exits an RK trace when the first sampled velocity is exactly zero. It still pays the initial sample and the surrounding surface pipeline. Another scalar zero test is not the next-level solution. A tile-level validity/predictor decision must occur before general sampling and then propagate through reconstruction, crossings, metric, geometry, and pressure coefficients.

For approximate holds, choose an explicit model: retain the cumulative transport/characteristic information needed to catch up, or use an inexpensive consistent predictor each step and bound its accumulated error. Merely waking a tile later and advancing it with the current velocity over a larger dt is incorrect for time-varying flow.

Reconstruction can early-exit particle distance gathering when no live particle support reaches the tile. It must still publish the advected phi correctly, preserve existing particle effects until their retirement is complete, and process sources. The guard belongs before preparation done exclusively for particle consumers.

Sources: [existing stationary trace](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-mixed-surface.ts:224), [particle reconstruction](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-narrow-band-surface.wgsl.ts:75).

### Crossing search and redistance

Crossing topology and metric accuracy are separate. Reuse crossing masks when their defining samples are unchanged. When crossing positions move without a sign change, the old nearest candidate may still be useful, but the metric and candidate validity need an error check; sign bits alone do not certify them.

Use a bound on signed-distance defect, such as variation of `|grad(phi)|` from one over the represented stencil, together with zero-set movement and the error requirements of pressure/particle membership. Sampled gradients alone need validation or a conservative cell bound. A translated plane can remain an exact distance field indefinitely even at substantial speed.

When the metric is already valid, `depthGeometry`, `buildDistance`, and distance publication should return before their geometric searches. For active queries, terminate Newton/gradient tracing on a geometric residual target instead of always taking the full six/five iteration envelope. These solve local geometry, so their stopping criterion should be positional/distance error rather than fluid energy. A deferred follow-on could derive gradient lower and Hessian upper bounds from the eight current corners already cached by `nbSurfaceDistance`, then certify projected-Newton residual and movement before skipping remaining steps. Keep the exact edge fallback and require a metric-error bound (proposed `0.01h`) plus the existing `1e-4` residual condition. This is not implemented. Recent redistance/crossing-search costs are roughly 0.86–1.14 / 1.3–1.43 ms, so it targets a sub-millisecond saving rather than the full performance goal; physical long-time validation still applies.

The current implementation preserves interface-adjacent vertices during fine redistance. Retain that invariant. Coarse retirement can move a hanging zero, so it must publish a surface change before downstream consumers decide to reuse their data. Invalidating only the changed tile is insufficient: bounded nearest searches, shared vertices, and interpolation taps require the corresponding neighborhood.

Source: [distance construction and iteration limits](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-narrow-band-redistance.wgsl.ts:68).

### Momentum advection and forces

For momentum, estimate the **change from resampling**, not the amount of momentum present. A frozen velocity field with Lipschitz bound L_u and travel d has sampling change bounded by roughly `L_u * d`. A large constant velocity therefore need not cost more than a small constant velocity. Moving boundaries, sources, nonuniform donors, and changes in validity must invalidate the shortcut. A zero divergence test is insufficient: divergence-free vortices still advect momentum.

Copy a face's known valid value when the resampling result is certified unchanged. For small variation, use a local affine predictor and compare its error bound with the velocity budget; otherwise run the existing characteristic. Account for cumulative approximation error and the actual MAC/extension sampling semantics.

Forces require separate treatment:

- Gravity remains a cheap addition every step, followed by pressure balance. Low current speed does not justify omitting gravity or free fall.
- Persist normals/curvature while their phi, solid, and ownership inputs are unchanged. A flat patch still pays expensive geometry in the current implementation; reuse the actual discrete value rather than assuming analytic curvature zero near seams or walls.
- Bound a viscosity contribution with `dt * nu * |Laplacian(u)|`. Only evaluate its full stencil where that contribution exceeds the allocated velocity error. At this scene's h and dt, `nu * dt / h^2` is about 6.7e-6; this is a strong candidate for a cheap bound, not permission to disable viscosity globally.
- Gate bed and body terms by their local geometric influence. Zero velocity does not rule out a capillary or bed force.

The surface tension setting selects the general per-face path in `forcesFine`, and curvature preparation is currently encoded whenever capillarity is enabled. This explains why “momentum transport” needs investigation beyond particle transfer alone.

Sources: [momentum characteristics](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-velocity-departure.wgsl.ts:3), [force evaluation](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-mixed-forces.ts:246), [fine force path](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-mixed-forces.ts:289).

### Extension and sampling caches

Separate the extension's **geometry** from its **velocity payload**. Phase, canonical face connectivity, donor masks, and source reach can remain unchanged while velocity values change. Retain those geometric results and update only the payload required by readers.

If the donors and their geometry are unchanged, early-exit rebuilding their extension. If donor values change within a bounded range, propagate a bound to consumers; convex weighted extension can bound payload changes when weights and donor selection are fixed. A changed donor mask or nearest-source choice invalidates that argument.

The hierarchy creates dependencies beyond the nearest tile. Dirty evidence must propagate up restriction and back through affected interpolation support; a one-tile wake halo is not enough. More selectively, certify the actual reader's reach so distant extension values need not be updated merely because they have storage.

The old/new FLIP extension pair must use compatible operators. Otherwise extension changes become artificial particle impulses. With no current particles reading an old snapshot, its expensive snapshot preparation can take a GPU-count-based early exit. This must use the current frame's count after bootstrap/motion, not a delayed CPU count. Current projected velocity still needs valid publication for surface transport and new seeding.

Sources: [extension representation](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-mixed-extension.ts:80), [paired extension and G2P](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-mixed-frame.ts:1022).

### Pressure system construction

Separate `A`, the operator geometry, from `b`, the velocity/forcing-dependent RHS. Cache the former until wetness, interface theta, apertures, wall topology, layout, or relevant parameters change. Early-exit matrix/aggregate construction on clean tiles and clean hierarchy regions. Recompute current fluxes/RHS where needed and retain global acceptance.

Band `prep` currently combines coefficients, kinds, forced face values, divergence, and initialization. Split these responsibilities logically so the existing launch can bypass expensive geometric evaluation while still refreshing current payloads. Likewise, root `encodeSetup()` rebuilds its geometry pyramid, coefficients, and cycle list before every initial residual check.

Small phi motion can be particularly significant near small interface theta or narrow apertures. A surface-displacement tolerance cannot automatically authorize matrix reuse. Use exact dependency versions first; approximate reuse later needs a bound on coefficient change times the candidate pressure, and an authoritative residual check against the current operator.

Source: [band preparation](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-pressure-band.ts:655), [root setup](/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-mixed-pressure-cycles.ts:1118).

### Fine pressure correction

Use the current 4h result as the global predictor and solve fine detail only where the h operator requires a correction. The natural activity signal is the **fine residual of that predictor**, including current body flux and free-surface conditions. Previous-frame speed cannot provide this certificate: a distant impact communicates through incompressible pressure immediately.

The current band already solves a correction relative to its prolonged coarse start. Extend that design: retain a valid fine correction between frames, rebase it against the new coarse start, and evaluate the current residual before doing more work. Where the residual meets the accepted bound, every subsequent smoothing kernel early-exits. A spatial version must reopen a tile when neighboring pressure corrections change its residual; freezing individually converged tiles without that dependency is unsound.

An accepted pressure residual bounds divergence under the discrete projection; it does not by itself bound every velocity or surface error. Keep the existing global acceptance and evaluate pressure-gradient/velocity effects in validation. Apply the coarse/base pressure contribution and boundary fluxes even when the additional fine correction is zero. Simply skipping h projection would leave the gravity impulse or an inconsistent coarse/fine face field.

For a further general reduction, a previous-pressure predictor can algebraically combine forces with `-grad(p_reference)/rho` and solve for a pressure increment. This avoids using the large gravity impulse as an activity metric. It requires consistent discrete operators, boundary values, and translated pressure bounds; it is a numerical formulation change and should follow the simpler cache and residual gates.

## Tall cells across the pipeline

The paper's compressed representation is used by extension, level-set work, velocity advection, remeshing, and pressure. Its values interpolate vertically between the bottom and top of each tall cell, with a regular slab above. It also reduces redistance cadence. Thus the paper offers more than a pressure-only optimization. Its collocated discretization has documented volume-gain and projection-consistency limitations; its exact operator should not replace the current MAC formulation unquestioned. [Chentanez and Müller 2011](https://matthias-research.github.io/pages/publications/tallCells.pdf), sections 3.1–3.7 and 5; local text: `docs/papers/tallCells.txt`.

### The pool already has substantial bulk compression

Ignoring boundaries, the 8 by 4 by 1.6 m liquid occupies 160 by 80 by 32 h cells. Its all-4h interior is only 40 by 20 by 8 cells: 6,400 values per scalar field.

A naive h-horizontal tall grid has 12,800 columns. Just two endpoint values per column require 25,600 values, already four times that all-4h interior count, before adding a regular slab. This is a representation-size calculation, not a measured runtime comparison. It rules out assuming that “tall” automatically beats the existing bulk grid.

At 4h horizontal spacing there are only 800 columns. Replacing eight vertical values by two endpoints plus two retained regular liquid layers would halve that idealized liquid count. The actual gain depends on boundary support, air layers, body geometry, and which h work remains. It cannot eliminate unchanged surface processing by itself.

### Conditions for a worthwhile extension

Use tall cells as an additional representation for vertically smooth regions, selected by error of representing velocity, pressure, and solid geometry. A second-derivative bound gives a starting interpolation criterion of approximately `height^2 * max|d2q/dy2| / 8`. Acceleration and boundary flux must also fit the representation. Recheck or refine when the current global pressure result, bodies, or sources invalidate it. Low energy is neither necessary nor sufficient: a smooth moving column may compress well, while a low-energy thin shear layer may not.

| Stage | Required saving from tall representation | What would erase that saving |
|---|---|---|
| Momentum and forces | Advect/evaluate a small number of column coefficients or endpoints; preserve conservative face flux | Expanding the column into h/4h samples every frame |
| Extension | Restrict/prolong compressed payloads and fill only the surface reader slab | Filling the old full-height cache anyway |
| Surface and redistance | Maintain the interface slab and derive deep phi without repeated vertical reconstruction | Running existing full tile searches and publications after decompression |
| Pressure setup and solve | Build operators on persistent column geometry with consistent coarse/fine flux | Saving root iterations while h wall/band geometry still rebuilds everywhere |
| Rigid coupling | Expand locally around genuinely intersecting or approaching bodies | Promoting whole columns for a distant body |
| Rendering and diagnostics | Sample the compressed field directly or materialize only requested surface data | Mandatory dense expansion during every simulation step |

A staggered/conservative extension would need new neighbor/face integration rules and sampling semantics. Current ownership, detail storage, face addressing, geometry, and remap assume h or 4h cubic owners. A tall cell has independent vertical extent and potentially varying lateral overlap, so it cannot be introduced merely by assigning a larger owner width.

Use a candidate counter first: how many column values and regular slab cells would each stage actually visit, including solids and all h work? Proceed only if this predicts substantial savings in transport, extension, geometry, and surface maintenance together. For this shallow pool, cross-stage reuse and error-sized work have the stronger immediate case. Tall compression becomes more attractive for deep water and should use those same validity/error mechanisms.

## Implementation order and acceptance criteria

1. **Make the existing band residual gate usable — implemented.** Zero residual has an explicit valid state, the policy uses a strict positive bound, and an initial measurement precedes ordinary smoothing. Moving-surface/body checks passed; measured results are above.
2. **Propagate local body influence.** Early-exit body terms and preserve dependent fluid state outside swept support. The ball comparison identifies a 7.3 ms profiled frame opportunity across several stages; do not add that number to other stage savings because they overlap.
3. **Persist geometry and distinguish changed values from changed shape.** Start with exact reuse of pressure coefficients/aggregates, crossing geometry, normals/curvature, and extension geometry. Correct the scratch lifetimes first. This is the enabling work for broad early exits.
4. **Gate work by its actual consumers.** Zero current particle readers should avoid old FLIP snapshot preparation and particle-only geometry. Empty occupancy already exits particle kernels; move that decision ahead of their expensive prerequisites while preserving grid publication and seeding.
5. **Add bounded inexpensive predictors.** Surface and momentum choose retain/predict/full work using operator error and accumulated budgets. Geometry/matrix reuse must consume the resulting error bounds, not silently treat approximate output as exact.
6. **Add spatial residual stopping.** Keep a global coarse response and authoritative current residuals; avoid unnecessary fine smoothing with dependency-aware early exits. Evaluate compressed columns after these mechanisms can apply across every expensive stage.

Measure actual full-path entries, early exits, geometric query iterations, sampled donors, rebuilt rows, and retained output bytes. Use fixed launches for both variants so reduced numerical work is the explanation for any improvement. Gather detailed counters in diagnostic runs, not the main timing runs.

Validation must include a long rest take, a very slow wave that accumulates sub-threshold motion, a uniformly translating surface, a low-amplitude thin feature, localized impact waking a distant quiet region through pressure, source activation, moving/rotating submerged bodies, and page recycle/relayout. Compare maximum and RMS velocity, divergence, surface displacement, local and total volume, body impulse, and wake latency against the full computation. Short rest stability alone is not acceptance.

For exact reuse, require the same outputs or an explained floating-point reduction difference. For approximate predictors, define positional, velocity, and divergence budgets explicitly and test accumulation; do not infer a formal global error guarantee from one local estimate. If the certification work costs nearly as much as the skipped kernel on a fully active scene, move its evidence into an existing producer or reject the approach.

## Reproduction and checks

Use [the investigation probe](/Users/petersuggate/code/me/fluid/tools/investigate-pool-work.ts) from the repository root:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js PROFILE=1 BAND_TOLERANCE=0 STEPS=120 FROM=61 OUT=/tmp/pool-full-work.json node --import tsx tools/investigate-pool-work.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js PROFILE=1 BAND_TOLERANCE=0.0001 STEPS=120 FROM=61 OUT=/tmp/pool-band.json node --import tsx tools/investigate-pool-work.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js PROFILE=1 NO_BODY=1 STEPS=120 FROM=61 OUT=/tmp/pool-no-body.json node --import tsx tools/investigate-pool-work.ts
```

Omit `PROFILE=1` for plain frame timing. `PROFILE=1 SPLIT=1 NB_SPLIT_ALL=1` identifies individual expensive kernels and deliberately changes pass boundaries; do not use it to estimate speedup. The tolerance override exists only in the diagnostic process. After milestone 1, the default uses residual stopping. `BAND_TOLERANCE=0` is the full-work control: identical encoded launches, including the new initial measurement, with early convergence disabled. Specify both controls explicitly for a matched comparison; both perform the same extra diagnostic parameter upload.

Repository checks after milestone 1: `npm run check:types` passed; `npm run test:unit` passed with 859 passing, 25 skipped, zero failing tests. The explicit targeted GPU suite passed all eight tests, separately from the unit-suite skips. Production-solver profiles and the ten-second rest probe provide the additional evidence described above.


## Local-solid implementation sprint

The 16h control layer now carries two distinct solid certificates. Canonical-open support is sufficient for the surface operator's existing open-cell predicates, including bounded single-step tracing. Pressure and forces require the stronger uncut certificate: all 64 constituent 4h tiles have zero cut flags (each includes its one-h-cell ring), and no current rigid-body bounding sphere intersects the 16h box expanded by 3h. The existing solid `surfaceClear` pass produces this stronger bit in the previously unused `.w` flag component. Storage paging stays 32h; no new buffer or dispatch is introduced.

The uncut bit is explicitly named `umSolidRegionClearAtBuild`. It is valid at the frame head after solid rebuilding, and must not be consumed after rigid integration changes body state. Domain-wall validity remains a separate face condition. Page claims/window moves refresh region reductions in the existing incremental flag pass.

Implemented consumers:

- Surface: existing fast fine-owner and merged deferred pipelines admit locally certified open regions even with solids elsewhere. Current velocity bounds limit travel to half an h cell, including floating-point allowance; the complete foot, cubic and incident-owner footprint must be covered. Cache misses and uncertain regions retain the general path.
- Pressure preparation: a single region proof bypasses inside/closed predicates, capacity and body-flux evaluation, six face-volume evaluations and neighbor closure tests. Non-clear rows use the exact current head-of-frame cached capacity; zero body-flux weights avoid body-velocity queries. Domain-wall faces retain their original handling.
- Fine forces: one region proof is shared by the three face evaluations, bypassing bed reconstruction and redundant capillary closure queries. Viscosity, pressure reconstruction, sources and domain handling remain intact.

These are operation-level exits, independent of whether a scene is hydrostatic. This sprint does not establish the requested 3x or 5x speedup; timing and validation are recorded separately. Per user direction, validation is restricted to the required repository checks, focused same-input GPU comparisons and two short candidate timing captures against the existing matched baseline window.


## Regional validity and persistent consumers

The next implementation moves evidence to field producers. Surface advection, particle reconstruction, air clearing and redistance advance a region revision only when an actual canonical phi value changes. A shared face/edge/corner advances every incident 16h closure. Revisions saturate rather than wrap into a valid old cache key. Remaps, page claims and external uploads have separate explicit invalidation paths.

The control layer now checks compact revisions and topology before fine gathers. One cooperative 64-lane group per 16h region checks identity, ownership and current solid masks; an unchanged model keeps its fit, residual and slope bounds. Current velocity support summaries still refresh. Geometry can likewise bypass its fine snapshot scan when producer revisions and metadata agree.

Clean normal/curvature stages consume their persistent records directly. They no longer materialize retained values into scratch which is subsequently overwritten. Capacity overflow uses the original scratch path.

Crossing masks now have their own persistent payload in the existing narrow-band buffer tail. A regional decision precedes depthSeeds' 125 vertex gathers and crossing classification. It compares current phi closure revision, sampling-owner topology, world identity and current/previous solid validity. Particle activation can request an already valid mask; it does not itself change that mask. Cached masks are restored into the transient search bank before downstream readers. Nearest-crossing and metric banks still rebuild; their wider dependencies and scratch lifetimes are not covered by this first reuse slice. Extension likewise still needs its own payload/dependency contract.

This changes the ownership of evidence and results, not the physical update rule. New reuse is exact. The existing bounded surface approximations remain local operator estimates, with no new claim of a global trajectory-error bound. The early retain/bounded-update/full policy must ultimately propagate through the remaining consumers; this slice establishes the producer/consumer validity contract and removes repeated fine scans/materialization where exact retention is available. No wall-clock improvement is claimed until a subsequent integrated measurement.

Validation for this slice: `npm run check:types` passed; `npm run test:unit` passed with 860 passing, 53 skipped and zero failures. The production control-layer GPU fixture passed. The isolated production crossing-cache kernel fixture passed, including exact mask restoration with zero vertex samples on reuse (125 on the initial scan), invalidation and nonfinite rejection. It was rerun successfully after the final cache-ablation control change.

A two-frame settled-pool integration smoke completed without shader or frame-order errors (`/tmp/fluid-regional-retention-smoke.json`, source hash `554af50983db311bfcccf9c689ce8edefa108ba67653ec6844f2b35586e3b393`). Its second frame took 45.23 ms; this cold, single-frame observation is not a speedup benchmark. The final diagnostic toggle/header change was covered by types and the isolated kernel fixture, not another full solver run. This slice has not yet received a matched settled/splash timing comparison or long-run physical validation.


## Nearest, metric and extension propagation

The next slice carries retained evidence into the three separable nearest-cell stages. Current crossing masks, world origins and neighboring page identities determine their integer results; tiny phi changes alone do not invalidate them. Persistent X and XY rows are read directly by dirty downstream search consumers, so clean X/Y jobs return without scratch copies. Only final nearest keys are restored for existing consumers. Fixed control passes compare regional metadata, and free physical pages exit before mask and halo gathers. One cooperative 64-lane group shares the nearest/metric halo decision for each 16h region. Binary solid-open masks are compared explicitly: unchanged partial solid regions can reuse these operators, whose only solid predicate is whether a cell is open above 1e-5. Exact metric reuse reads the prepared regional flag rather than repeating 27 region lookups per tile.

The current crossing producer also fits planes from its existing 125 samples. A regional aggregation combines those compact tile planes and residuals into a 16h plane plus a 64-bit coverage mask. No additional fine phi scan establishes this proof. Reconstruction-only searches omit plane preparation; the first metric consumer gathers a missing plane once, including when its crossing mask was retained. An evaluated but unusable plane has a separate state to prevent futile rescans. Regional fitting prefers a nonzero-gradient crossing tile and excludes far-clamped inconsistent tiles from its certified coverage. A cheap signed-distance update requires a positional residual bound of at most 0.02h and coverage of the entire nearest-distance ball; incomplete coverage, solids, ambiguous signs and poorly conditioned slopes fall back. The crossing-cell fallback separately uses its existing eight corners to admit a plane whose normal correction segment stays inside the cell, avoiding its edge enumeration and Newton iterations. This segment condition admits flat surfaces whose projected feet lie on cell edges.

Interior fine redistance vertices can retain their current value when the current-field SDF defect is bounded by 0.02h. The bound includes existing distortion, so accumulated departures from signed distance reopen work. Canonical tile-edge ownership and far-reach clamping retain their original path. These bounds concern distance to the represented trilinear zero set; they are not a strict 0.02h difference from the original approximate Newton algorithm, nor a global coupled trajectory bound.

Extension now retains the completed post-pressure result at every non-kick frame tail, including frames with bodies. Coupling changes body state without changing canonical fluid velocity. Extension reads no body buffer, and its phase uses the still-current head-built solid masks/capacities and liquid center phi. The next head still invalidates and rebuilds after updating body geometry. This removes a duplicated numerical extension; it does not introduce a scene test. Seed/sweep kernels also reject air payload reads and unchanged source work before unnecessary face gathers. A general regional approximate extension hierarchy is still outstanding: its far-source dependencies cannot be certified by local phi revisions alone.

Persistent nearest/metric payloads currently cost about 239 MB for the pool's 3,856 reserved 16h regions. The implementation falls back to the original computation when the combined storage binding cannot fit. This allocation is a material limitation, even though only live regions perform numerical work. The diagnostic reports capacity and bytes so a silent fallback cannot be mistaken for a measured cache result.

A first candidate showed why control cost matters: settled mean frame time was 40.75 -> 40.57 ms over frames 2–30. Extension fell 3.79 -> 2.54 ms, but crossing search rose 1.57 -> 2.38 ms and redistance rose 0.87 -> 1.08 ms. That result prompted the free-page exits, shared halo checks and broader regional plane described above. Revised timings are recorded below when available.


### Measured propagation result

Matched production-profile windows, frames 2–30, 32h paging and authored pool geometry:

| Scenario / stage | Before | After | Change |
| --- | ---: | ---: | ---: |
| Settled frame mean | 40.749 ms | 39.503 ms | -3.06% |
| Localized splash frame mean | 54.386 ms | 53.398 ms | -1.82% |
| Settled extension | 3.785 ms | 2.547 ms | -1.238 ms |
| Settled crossing search | 1.573 ms | 0.890 ms | -0.683 ms |
| Settled redistance | 0.866 ms | 1.069 ms | +0.203 ms |

Baseline source hash: `f65668829e7b60206dc2ce4207b1575fdb96876268c1e8933e3d4839f56ad625`. Final candidate: `504b13dc83cb4e5564300b9316b78ae9c8c38c3294e6e8231db494c7fa2a647a`. Captures and comparison are in `artifacts/work-adaptive-investigation/propagation-{before,after}-{settled,pool-impact}.json` and `propagation-comparison.json`. Each is a single short capture, not repeated statistical trials.

The final settled snapshot reports 384 nearest-reusable regions, 120 exact-metric regions and 102 regional plane certificates. Splash reports 384, zero and 94 respectively. These are final control snapshots, not counts of work avoided over the timing window. No region has complete plane coverage: current cheap updates use certified partial coverage. Nearest classification responds to discrete crossing changes while the current-plane path can handle continuous field changes. This is not yet a measured work-versus-energy scaling curve.

The splash's integrated vertical impulse changed from 494.732 to 494.457 N s (-0.055%); deepest body height differed by 0.000253 m; both momentum consistency checks remained below 1.46e-5 N s. These are short-run checks, not a long-run fidelity result. Current full-field volume and velocity differences were not captured; scheduled volume telemetry may be stale.

Validation: `npm run check:types` passed; `npm run test:unit` passed with 860 passing, 54 skipped, zero failed. Three targeted GPU tests passed: existing redistance behavior, crossing-cache dependency invalidation, and retained nearest results versus the full search including a changed crossing mask. Both production scenario runs completed.

The 3x–5x goal remains unmet. Redistance still regresses by roughly 0.2 ms: per-vertex predictor coverage checks and retained metric traffic do not yet pay for themselves. The next cut should publish cheaper workgroup-level eligibility and preserve metric outputs with less copying/storage; a full-region validity claim must not replace the current partial-coverage proof. General bounded regional velocity extension also remains outstanding; the completed extension change removes redundant tail work without weakening its current numerical treatment. The persistent cache's 238.6 MB allocation should be reduced or mapped onto active regions before treating this design as scalable in memory.


## Stream 2: local body dependencies

A moving body no longer invalidates the frame head globally. Before this change, any frame with bodies called `invalidate(true)`, rebuilt the whole all-4h solid record, marked solid widths stale and dropped all derived geometry. Removing the airborne ball cut about 7.3 ms of profiled frame time before impact. The head now rebuilds only what the bodies' old-to-new swept support reaches. Rigid integration and coupling are unchanged: the tail still integrates the bodies and coupling still sees the total pressure impulse.

What was built:

- A swept-body region mask, one word per 16h work-control region (64 tiles; the region index is the existing `8*(t>>9)+((t>>2)&1)+2*((t>>5)&1)+4*((t>>8)&1)`). A fixed launch with one lane per region writes it at every `encodeCoarse`, ahead of the record build. The swept distance is the distance from the region's box to a capsule along each moved body's built-to-current centre segment, with radius equal to the larger of the two bounding spheres. The sphere covers rotation and the capsule covers translation. The pose distance uses every present body's current sphere. A body counts as moved when its presence differs, or its position, dimensions or orientation differ bitwise from the mirror taken at the last record build (`builtBodies`). A resting body therefore leaves its tiles authoritative.
- Record rebuilds are gated by the mask. The cut and build passes test each tile with `umBodyTileDirty`: the region's swept reach must be at most 2, and the tile box must lie within 2h of a moved body at its old or new pose. The build exits workgroup-uniformly (lane-0 decision, `workgroupUniformLoad`) for clean tiles. `surfaceClear` exits for regions with swept reach above 8. Static solids are still built once, and an edit still rebuilds in full.
- Widths no longer go stale for body rebuilds. The record build writes the tile's simulation width flag from the current topology, so the listed widths stay valid. The frame head no longer sets `solidWidthsStale` on body frames.
- Geometry evidence is local. The geometry work-control finish marks a region dirty, with topology, only when a mask is pending and the region's swept reach is at most 3. The previous per-region body word and the 8h current-pose loop are deleted. After the geometry control pass the mask's pending word is cleared (`consumeBodyEvidence`), so the gather launch later in the frame does not redo body regions.
- Solid helpers are body-local. `umCellOpen` reads the record for a stored tile unless a body touched the tile at build or a body's sphere is within half a cell diagonal. `umFaceOpen` and `umPressureFaceV` take the record or terrain boundary-box path unless a body is near the face. The old eight-sample loop in `umPressureFaceV` is deleted.
- Frame head (`uniform-mixed-frame.ts`): `if(bodies){encodeBodies; invalidate(true); geometryCurrent=false; reusableExtension=false}`. `invalidate(true)` now means "rebuild the moved bodies' tiles", not the whole record. The extension token is left for stream 3 to replace.

Mask layout (`UniformMixedSolid.bodyRegions` returns `{buffer, regions, headerWord}`):

- Words `[0, regions)`: one word per region.
  - Bits 0–7: swept reach, `floor(d/hmax)` saturating at 255. A reach of at most m means the region lies within m·hmax of the swept support.
  - Bits 8–15: pose reach, defined the same way.
  - Bit 16 (`UNIFORM_BODY_REGION_UNKNOWN`): the region's block is not live. Its reaches read 0, so a newly claimed or recycled page key is dirty and never inherits validity.
- Header (`UNIFORM_BODY_REGION_HEADER_WORDS=4`), starting at word `regions`, in this order: the current body count, the body count at the last build, the moved-body bit mask, and an any-moved flag.
- The record holds a mirror at `umBodyRegionBase()=count+22*tiles`, one vec4 per region: (swept reach, pose reach, unknown, 0). A pending vec4 follows it: (1, any moved, body count, 0). Consumers that already bind the record therefore need no new binding.
- WGSL accessors on the record:
  - `umBodySweptReach(region)` and `umBodyPoseReach(region)`
  - `umBodySweptNearTile(tile, margin)` and `umBodyPoseNearTile(tile, margin)`, which return true for an unstored key
  - `umBodyEvidencePending()` and `umBodyRegionOfTile(tile)`
- `uniformBodyRegionsWGSL(name)` reads the raw buffer.
- Stream 1 uses `umBodyPoseNearTile(tile, 3+stencil)` for body flux and `umBodySweptNearTile(tile, stencil+2)` for retained coefficients.

Memory, allocated at construction: the mask buffer is `4*(regions+4)` bytes and the record grows by `16*(regions+1)` bytes. For the pool's 3,856 regions (482 blocks) that is 15,440 B plus 61,712 B, about 77 KB. The record buffer also gained `COPY_DST` so it can clear the pending word.

Smoke (dt 1/60 s, 5 steps from frame 2, profiled, one run each, not an A/B): both scenarios completed without shader, validation or frame errors.

- Settled: mean 42.06 ms. Solid record 0.57 ms, geometry work control 0.82 ms.
- Pool-impact: mean 50.84 ms. Solid record 0.93 ms, geometry work control 0.84 ms. Body y went from 2.06 to 1.65 m and the impulse reached 157.9 N s by frame 5. Momentum error stayed below 1.5e-5 N s.
- Gates: `npm run check:types` is clean. `npm run test:unit` gives 860 passing, 54 skipped and zero failed. No tests were deleted.

Limitations:

- No matched A/B was run, so the saving against the 7.3 ms global invalidation is not yet measured.
- Body frames no longer run the full controlled geometry pass (see the addendum below).
- Other body-global gates are outside this stream:
  - The extension (`reusableExtension`, `phaseKeyed=!bodies`) belongs to stream 3.
  - Band flux and the pressure-velocity gates belong to stream 1.
  - Narrow-band FLIP, particle contact and the mixed-bodies passes were not touched.
  - "Uniform mixed body forces" is the fluid force stage (gravity, viscosity, capillarity), not rigid-body work. Its cost was the same with and without the ball (see the addendum).
- The tail's coupling moves bodies after this frame's mask was written. That motion appears in the next head's mask, which rebuilds against `builtBodies`, so no motion is lost.
- Static solid edits are not covered by the mask. `editSolids` still rebuilds the record in full, and consumers invalidate on the host.
- Face apertures in body scenes now come from the record or terrain path away from bodies, not from the old eight samples. Fluid away from a body therefore sees a slightly different, consistent aperture.
- The capsule distance is a 24-step ternary search, a conservative lower bound rather than an exact distance.

### Stream 2 addendum: body-frame record and geometry cost

Two follow-ups remove most of what a body frame still paid in these files.

Record build. The ball in the settled scene rolls slowly along its board, about 0.5 mm per frame, so it counts as moved every frame. A probe of the mask over frames 2–4 found no live region within 8h of its swept support. All 384 live regions had a swept reach above 8; the other 3,472 regions were free blocks (UNKNOWN). The 0.52 ms record cost was therefore dispatch overhead, not rebuilding:

- The per-tile build launched one workgroup per tile key, about 247k of them. The first replacement, a fixed 8192-workgroup grid, still made each workgroup do about 30 serial read-and-barrier steps.
- UNKNOWN regions read as reach 0, so they entered the per-tile path in every builder launch.

Body frames now use two fixed region launches:

- `cutsRegions`: up to 1024 workgroups stride the regions, with one lane per tile.
- `buildRegions`: `64*ceil(regions/64)` workgroups. Workgroup w owns tile `w % 64` of 64 regions. Its lanes read those 64 region words in one parallel round into a workgroup bit mask, then build only the tiles of dirty regions. A workgroup with no dirty region exits after that one read. One region's tiles are spread over 64 workgroups.

Builder launches treat UNKNOWN as not dirty, since a free block has no tile to build. `surfaceClear` exits for UNKNOWN regions too. The per-tile body variants are deleted, and full builds still use the per-tile launches.

Geometry. A frame whose geometry was current until bodies moved no longer runs the controlled select/scan/finish pass. `UniformMixedSurfaceGeometry.encode(..., {bodies:true})` instead:

- runs `compactBodies`, a fixed region grid that lists every live tile of each region with swept reach at most 3 through `gcList`, the same marking `compactChanged` uses: region dirty, topology, normals and snapshot stale;
- runs `geometryChanged` on that list;
- mirrors only the listed owners (`mirrorListed`);
- takes the mask's evidence.

The frame head records `geometryBodies` and `geometryBodiesOnly`, and `encodeGeometry` chooses the body launch when it is given no options. This launch requires that no adoption happened since the last encode, and it fails loudly otherwise. The relayout head then uses the generation's changed list after the remap whenever an encode has already taken the body regions, instead of a full pass. The tail gather recomputes the listed tiles' pressure outputs, because their snapshots were made stale.

Rigid-body passes. `uniform-mixed-bodies.ts` and the particle contact were already local or cheap:

- Rigid coupling cost 0.03–0.13 ms.
- Integration and contacts cost 0.07 ms.
- Body tile marking cost at most 0.02 ms.
- Stream 3 gates the particle rigid contact per tile with `umBodyPoseNearTile`.

I changed none of these.

Smoke (dt 1/60 s, 5 steps from frame 2, profiled, one run each). Other streams' uncommitted work was in the tree, so whole-frame means are not comparable between runs. The pass deltas below are measured against a NO_BODY settled run on a near-identical tree:

| Settled pass | Before addendum (body / no body) | After (body / no body) |
| --- | ---: | ---: |
| Solid record | 0.573 / 0.049 ms | 0.213 / 0.066 ms |
| 16h geometry work control | 0.819 / 0.262 ms | 0.344 / 0.328 ms |
| Geometric fill | 0.344 / 0.213 ms | 0.393 / 0.246 ms |

Pool-impact completed without errors. Body y was 1.6487 m and impulseY 157.916 N s at frame 5, the same as before the addendum to about 1e-6. Momentum error was at most 1.46e-5 N s. `npm run check:types` is clean, and `npm run test:unit` gives 860 passing, 54 skipped and zero failed. No tests were deleted. No A/B was run.

Remaining body-frame costs in settled come from passes owned by other streams:

- extension, about +1.3 to +2.7 ms (stream 3)
- pressure rhs and band rows, about +0.3 to +0.4 ms each (stream 1)
- frame plan and surface pressure authority, about +0.2 to +0.5 ms each

Also, about 0.15 ms of fixed launch cost remains in the record pass and in the geometric fill.

### Stream 2 addendum: frame plan and surface pressure authority

This follow-up covers the frame plan and the surface pressure authority. The authority lives in `uniform-mixed-pressure-authority.ts`, which stream 1 handed to stream 2.

Body-frame overhead:

- **Frame plan.** The frame heads no longer encode the plan when the surface is regional (`if(!regional)this.plan.encode(...)`). That removes the 0.26 ms launch the body head paid in settled. The plan launches that remain also run without a body.
- **Phase authority on body-only frames.** A body-only frame is one where geometry was current until bodies moved, and the phase key is otherwise unchanged (`phaseBodiesOnly`). On such a frame the head runs `encodeBodies` instead of the full authority:
  - `surfaceRegions` is a fixed launch of up to 1024 workgroups striding the regions.
  - Its 27 lanes mark the neighbour regions whose swept reach is at most `UNIFORM_BODY_GEOMETRY_REACH`. The decision is workgroup-uniform.
  - A 4h tile reached by a marked neighbour is recomputed in its lane.
  - Reached h tiles go into a workgroup bit list and are recomputed one owner per lane after a `workgroupUniformLoad`.
  - The old surface body is now `umSurfaceOwner`, so each entry has one call site.
  - A first try that strided tile keys (4096 workgroups) cost about 0.33 ms and was dropped.

Settled cost of the split authority (the root/coarse-cut instance, about 1.49 ms in the original baseline):

- The authority is now regional. `stamp` uses one thread per region and writes 13 words:
  - the region's live phi epoch from `surfaceWorkControl.revisions` (0 when the region is not live);
  - its window-relative page coordinate;
  - the eight block words of its 32h neighbourhood.
- A region is changed when any of these hold:
  - a word differs from the last pass;
  - the epoch is unknown (0 or at least 0xfffffffe);
  - the body mask's swept reach is at most 3;
  - the pass is forced. Forcing happens on the first pass, on `invalidateExtension`, on a work-control generation bump, and when more than one body head ran since the last pass.
- Page claims bump the neighbouring epochs, and the page coordinate and block words are compared exactly. A recycled page key therefore never inherits validity.
- `cutRegions` refits only the changed regions' tiles. Its gate is one stamp word, so the exit is workgroup-uniform.
- `wetRegions` recomputes the wet topology of any live tile whose reads (tiles t-1..t+2 on each axis) reach a changed neighbour region. It uses a 27-bit neighbour mask with a uniform exit.
- Cut fits now persist in a retained buffer. They no longer go to shared frozen scratch.
- An axis-parallel `wetRegions` was tried: 192 lanes, one face each. It gave no measurable gain (0.655 vs 0.59–0.65 ms), so it was reverted.

Memory, allocated at construction:

- retained cut fits: `scratchBytes`, `8*(1+2*(groups+chunks)+owners)` B;
- stamps: `4*13*regions` B, which is 200,512 B for the pool's 3,856 regions.

Both buffers have `COPY_SRC` for diagnostics.

Smoke (dt 1/60 s, 5 steps from frame 2, profiled, one run each, not an A/B; other streams' work was in the tree; the GPU timer quantum is about 0.066 ms):

| Pass | Before | After |
| --- | ---: | ---: |
| Settled body-head frame plan | 0.26 ms | not launched |
| Settled body-head phase authority | 0.20–0.39 ms | 0.00–0.07 ms |
| Settled split authority (coarse cut + wet) | 0.85–0.98 ms | 0.52–0.66 ms |
| Pool-impact body-head phase authority | 0.13–0.20 ms | 0.20–0.26 ms |
| Pool-impact split authority | 0.59–0.72 ms | 0.59–0.79 ms |

Pool-impact ran clean: body y 1.64866 m, impulseY 157.926 N s, momentum error at most 1.56e-5 N s. `npm run check:types` is clean, and `npm run test:unit` gives 860 passing, 54 skipped and zero failed. No tests were deleted.

Limitations:

- **Settled saving is capped.** All 384 live regions lie in the surface band, and about 60 of them change their phi epoch each frame. With the dilation to neighbours, wet still covers a large share of the band. Most of what remains is `umWetFace`, with 75 loads per face, in changed and adjacent regions.
- **Pool-impact phase authority is about break-even.** Many h tiles near the ball are processed serially within one region's workgroup.
- **The frame plan's settled cost is not regional.** Its support words live in the shared `ownership.support` buffer, which the plan clears and other passes reuse. Its speed maximum also needs every owner. Retaining per-region results would need a private support buffer and a per-region speed reduction, which was not done.

### Stream 2 addendum: region work heat and its surface consumers

Peter's direction: exact-change revisions no longer decide whether non-pressure work runs. A per-16h-region work heat does, and as many regions as possible stay at 16h with no fine work.

**The descriptor** is `lib/methods/uniform/uniform-region-heat.ts`, the `UniformRegionHeat` class. It is EXNB-style: dimensionless, set to 2 by a trigger, cooling at 2/0.5 s, and spread to the 26 neighbours at half weight. There is no speed threshold.

One-frame triggers make the region and its neighbours HOT for this frame only. They add no heat:

- a host invalidation (the work-control generation) or the first frame;
- region identity (block word or window coordinate);
- a page claim or remap (the work-control `publish` and `initialize` hooks call `rhPromote` on the region and its 26 neighbours);
- unknown evidence (a saturated speed or an infinite dp).

On the frame after a one-frame trigger, the pressure change is ignored, because its baseline was unknown. The first solve's dp is the whole hydrostatic field.

Heat triggers:

- body swept reach of at most `UNIFORM_BODY_GEOMETRY_REACH`, or an unknown body mask;
- a source within (8√3+4)h;
- the accumulated displacement exceeding 0.02 h. The accumulation is Σ dt·|u|max/h from the previous frame's region velocity extrema, plus dt²·|dp|/(ρh²) from stream 1's dynamics word 6. It resets whenever the region runs non-COLD.

The NB particle activity heat (the maximum over the region's tiles) raises the heat directly.

A region's tier is set from level = max(own, 0.5·neighbours), where a region's level is its heat, or 2 on a one-frame trigger:

- HOT if level ≥ 1;
- WARM if 0 < level < 1;
- COLD if level is 0.

The tiers are computed once in the frame head, right after the coarse bodies, by two fixed launches: `heat` (a region per workgroup, at most 1,024 workgroups, with free regions exiting) and `spread` (a thread per region).

The accessors come from `uniformRegionHeatWGSL(group, binding, regions)`:

- `umRegionTier`;
- `umRegionTierAtTile`;
- `umVertexTier`, the lowest tier among the stored regions holding a lattice vertex, so a COLD closure is never written;
- `UM_TIER_*`.

`uniformRegionTierWGSL(regions, load)` gives the same accessors over a copy for stages at their storage-binding limit.

Memory, allocated upfront: tiers take 4·(regions+4) B and state takes 32·regions B, 138,864 B in total for the 3,856-region pool. The surface claims and the work-control surface buffer each gain one word per region for the tier copy, and `curvatureDirty` gains 32 KB of normal tags.

**Consumers in stream 2's stages:**

- **Surface transport** (`umAdvectStore`, `advectWalls`): `umHoldCold` stores the input phi into the scratch when a vertex is COLD. The bits are identical, so no revision moves, and no trace, cubic or deferred listing runs. The tiers reach this module through a per-frame copy into the surface claims buffer, because the stage is at its 10-storage-buffer limit. WARM runs the existing cheap tiers, the same as HOT.
- **Work control:**
  - the surface model of a COLD region stands, and its model epoch is not advanced;
  - the geometry scan skips a COLD region unless its metadata or world changed, and its geometry epoch is not advanced. The geometric fill and normal candidates follow through the controller's dirty words.
- **Split surface authority** (regional mode):
  - the stamp of a COLD region keeps its stamped revision and ignores bodies, while page words still decide;
  - stamp word 12 now carries bit 0 (changed) and bit 1 (held). Held means the region and all 26 neighbours are COLD and nothing changed;
  - `surface` keeps the held owners' phi and phase (the root's owner index is the tile key, so it is stable) and rewrites only their correction, which follows this frame's gauge.
  - The tiers are copied beside the revisions before the pass.
- **Retained h normals, a rank-shift fix:** h owners' retained normals are indexed by owner rank. A relayout that adds or removes an h tile shifts every later tile's slot, but only the changed neighbourhoods were marked.
  - A new geometry pass, `normalTags`, tags each of the 8,192 retained slots with its tile (key + 1).
  - On a mismatch it takes over the slot and marks that region's normals as candidate- and topology-changed, so the force stage rewrites them.
  - Stream 1 confirmed that REUSE never skips a region whose topology is set.

The revision checks remain the exact test inside non-COLD regions. Nothing became dead.

**Tiers** (`heat-probe`, dt 1/60 s, distance by the body mask's pose reach):

- Settled: frame 1 has all 1,680 live-or-claimed regions HOT, from identity and pending triggers. From frame 2 on, all 384 regions are COLD. The largest held displacement is 8e-5 h at frame 12.
- Pool-impact (ball at 8 m/s) never goes mostly COLD within 45 frames. The impact's wave keeps about 200 regions re-triggering on displacement each frame, and remaps promote 50–200 more.
  - Every region with pose reach under 32h (0–1.6 m) is HOT throughout.
  - Of the 32–95h band, 55–122 of 286 regions are WARM over frames 17–29, and the first COLD regions appear around frame 33 (8 at frame 33, 50 at frame 37).

**Stage costs** (investigate tool, one run per arm, other streams' code in the tree). Before is a probe-only preload that forces every region HOT each frame:

| Stage (mean ms) | Settled before | Settled after | Pool-impact before | Pool-impact after |
| --- | ---: | ---: | ---: | ---: |
| Frame | 34.7 | 24.7–25.1 | 93.5* | 52.0 |
| Surface advect | 2.56 | 0.48 | 2.87 | 2.84 |
| Surface advect, deferred | 1.43 | 0.07–0.09 | 13.6* | 6.77 |
| Surface pressure authority | 1.08 | 0.57–0.58 | 2.29 | 1.47 |
| NB crossing search (stream 3) | 0.93 | 0.58–0.62 | 1.88 | 1.91 |
| NB surface reconstruction (stream 3) | 0.98 | 0.24–0.25 | 1.31 | 1.27 |
| Surface work control | 0.57 | 0.44–0.52 | 0.84 | 0.88 |
| Geometric fill | 0.30 | 0.21–0.23 | 0.81 | 0.77 |
| Geometry work control | 0.28 | 0.22 | 0.41 | 0.36 |
| Region heat | 0.04 | 0.01–0.03 | 0.04 | 0.00 |

Settled was measured over frames 34–40 and pool-impact over frames 24–30. \*The pool-impact before run was taken after other streams had edited the tree (its first two attempts failed on another stream's mid-edit shader), so its deferred-advect and frame figures are not a clean comparison. Both pool-impact runs end with the body at y ≈ 1.369 m and volume drift 0.0066–0.0067.

**Limitations:**

- A COLD region does not catch up the displacement it held when it reheats. The error is bounded by the 0.02 h budget per COLD stretch.
- The settled transport still launches over every vertex, loading and storing each COLD vertex: `advectFine` costs about 0.3 ms with everything COLD. Skipping COLD regions per workgroup would need the scratch phi to stay equal to phi without a store.
- The held authority still visits every owner to rewrite its correction, about 0.58 ms in settled. The work-control surface pass still costs about 0.44 ms when everything is COLD.
- Pool-impact gains little, because the impact keeps the pool HOT for the whole window.

### Stream 2 addendum: pool-impact grading and COLD residues

Peter's direction: give the splash full treatment and the rest of the pool progressively less. Displacement past the budget should refresh a region WARM, not HOT. HOT is kept for detail the cheap tiers cannot certify.

**Heat grading** (`uniform-region-heat.ts`):

- **Displacement over the 0.02 h budget** is now a one-frame WARM trigger (bit 64, level 0.75). It adds no heat and resets the accumulation.
- **A new HOT heat trigger, the velocity span** (bit 128), is the transport's own certificate:
  - the span condition is ½·dt·Σₐ(maxₐ − minₐ)/h > `UNIFORM_SURFACE_TRANSPORT_ERROR_H` (0.02 h), over the region's velocity extrema;
  - the travel condition is dt·|u|max/h > ½.
  - The span uses the interval width, not the speed.
- The other heat triggers are unchanged: body reach, sources, and NB particle activity. So are the one-frame HOT triggers: invalidation, identity, claims, and unknown evidence.
- **The spread falls off with distance:**
  - HOT needs the region's own level ≥ 1;
  - neighbours contribute 0.5 × their level, so they can only make a region WARM;
  - a region beside a non-COLD region is at least WARM, so the tiers fall off one ring per level instead of saturating a 3×3×3 block.
- **Promotion hooks:** pending now carries a tier, and `promoteWGSL` gives `rhPromote` (HOT) and `rhPromoteWarm` (WARM), both by `atomicMax`.
  - Page claims still promote HOT.
  - The remap `publish` now promotes only the remapped region, and only to WARM. Before, it promoted the region and 26 neighbours to HOT.
  - The remap's changed-tile list is already the 3×3×3 tile dilation of each width change, so the extra 27-region promotion was a second dilation.
  - A width change re-resolves the region's phi through the WARM paths; WARM momentum is the self-validating affine tier.
- The probe found no identity-preserving remap that loses heat. The heat state is keyed by region slot and keeps its level across a publish. A one-frame identity trigger fires only when the block word or window coordinate actually changes.

**Tiers by pose-reach ring** (`heat-probe`, pool-impact, dt 1/60 s; counts are COLD/WARM/HOT):

| Frame | Ring | Before | After |
| --- | --- | --- | --- |
| 5 | 0h / 1–7h / 8–31h / 32–95h | 0/0/4, 0/0/8, 0/0/86, 0/0/286 | 0/0/4, 0/0/8, 0/84/2, 204/82/0 |
| 15 | same | all 384 HOT | 0/0/4, 0/0/10, 0/37/47, 88/198/0 |
| 25 | same | 0/0/4, 0/0/10, 0/0/84, 0/106/180 | 0/0/4, 0/0/10, 0/27/57, 101/171/14 |
| 35 | same | 0/0/4, 0/0/10, 0/0/86, 8/59/249 | 0/0/4, 0/0/10, 0/26/60, 50/207/59 |
| 45 | same | 0/0/4, 0/0/8, 0/0/92, 1/39/272 | 0/0/4, 0/0/8, 0/18/74, 31/205/92 |
| | **Totals at 5 / 25 / 45** | 0/0/384, 0/106/278, 1/39/376 | 204/166/14, 101/198/85, 31/223/178 |

- Per frame after the change: 55–213 displacement triggers (WARM), 12–140 span triggers (HOT), 0–37 WARM remap promotions, and 0–42 HOT claims.
- Before, 54–208 displacement triggers and 60–204 remap promotions each made a 27-region HOT block.
- HOT grows over the window (14 → 85 → 178) as the impact wave spreads span triggers outward.
- Settled is unchanged: frame 1 is HOT/claimed, and all 384 regions are COLD from frame 2.

**COLD residues:**

- **Surface work control** (0.44–0.52 → 0.22 ms in settled):
  - `selectSurface` now holds a COLD region that has a valid model, support coverage, and an unchanged window: one lane reads four words and the workgroup exits. Its first non-COLD frame compares the metadata stored before it went COLD, so any change refits.
  - `summaries` skips held regions and keeps their support. Only a region's own transport reads its support, and its vertices are held.
  - The surface model scan is a new kernel, `surfaceScan`. It launches a workgroup per eighth of a region, taking 8 tiles in turn, instead of a workgroup per tile capacity (246k launches, mostly empty). A reused region's 8 workgroups exit at once.
  - The geometry `scan` lost its dead surface branches.
- **Split authority:**
  - The stamp no longer marks every absent region as changed each frame; an absent region's words change on the frame it goes.
  - The `absent` kernel skips held pages, and it is now a strided fixed grid of 1,024 workgroups (it was a workgroup per page).
  - The pass measured 0.58 → 0.56 ms. Most of this pass is not the split authority: stream 1 timed three `surface` dispatches of about 0.13 ms each. Two are the head authority (`this.authority`: non-regional, every simulation owner, the phase build, and the capillary re-encode), which has no held path.
- **Lazy gauge correction is not done.** The correction is `params.z·capacity` on liquid owners, and `params.z` is the surface-volume controller's proportional rate, which changes every frame, so an equality test never fires.
  - The per-region form would let the two readers compute it themselves: `uniform-mixed-pressure-velocity.ts` RHS line 256 and `uniform-pressure-band.ts` prep line 777, both stream 1's files. They would read the rate and the owner's capacity and phase.
  - The authority would then stop writing the correction, and held owners would cost nothing.
- **Transport deep-COLD exit is not done.** Stream 3 found that NB's reconstruct-group crossing search reads phiScratch at COLD vertices:
  - `depthTiles` corner signs;
  - the `depthSeeds` 125-vertex gather, whose committed mask is then retained unvalidated while the region stays COLD.
  - So the scratch must still equal phi at COLD vertices, and `umHoldCold` keeps its store (`advectFine` ≈ 0.27 ms in settled).
  - Stream 3 proposed the fix: reconstruct-group reads at COLD vertices go through `outputPhi` (= f.phi there, never written for COLD). Once that lands, the transport can exit uniformly for regions whose 27-region neighbourhood is COLD.

**Stage costs** (investigate tool, one run per arm, other streams' work in the tree, so this is not an A/B):

| Stage (mean ms) | Settled before | Settled after | Pool-impact before | Pool-impact after |
| --- | ---: | ---: | ---: | ---: |
| Frame | 25.07 | 21.83 | 52.00 | 48.23 |
| Surface advect | 0.48 | 0.49 | 2.84 | 2.42 |
| Surface advect, deferred | 0.09 | 0.07 | 6.77 | 5.56 |
| Surface pressure authority | 0.58 | 0.56 | 1.47 | 1.47 |
| NB crossing search (stream 3) | 0.58–0.62 | n/a | 1.91 | 1.60 |
| NB surface reconstruction (stream 3) | 0.24 | 0.23 | 1.27 | 1.15 |
| Surface work control | 0.52 | 0.23 | 0.88 | 0.63 |
| Geometry work control | 0.22 | 0.22 | 0.36 | 0.34 |
| Momentum (stream 3) | 0.78 | 0.32 | 1.69 | 1.58 |
| Region heat | 0.01 | 0.04 | 0.00 | 0.04 |

Settled is frames 34–40, and pool-impact frames 24–30. The before columns are the previous addendum's after runs. The settled momentum drop is stream 3's work.

**Limitations:**

- Span triggers dominate HOT late in the impact. The certificate is the transport's 0.02 h error budget, which is tight for a travelling wave. A looser HOT threshold would need the cheap tiers to certify more.
- WARM regions still run the existing cheap transport tiers, the same as HOT. The tier split pays off in the stages that grade WARM separately (momentum's affine tier) and in COLD holds.

### Stream 2 addendum: head authority held mode, lazy correction, transport deep-COLD exit

**Deep-COLD flags** (`uniform-region-heat.ts`). The tiers buffer has two more sections after the counts:

- `UNIFORM_REGION_HEAT_DEEP_WORD(R)`: 1 where the region is live and it and every stored neighbour of its 26 are COLD.
- `UNIFORM_REGION_HEAT_EPOCH_WORD(R)`: a per-region epoch. Each deep pass that finds the region non-COLD increments it.
- The `deep` kernel runs after the spread in `encode`. It runs again from `encodeDeep` after the relayout's claim and remap promotions (one line in `uniform-mixed-frame.ts`).
- `uniformRegionTierWGSL` takes an optional deep load and then defines `umRegionDeepCold`.

**Head authority held mode** (`uniform-mixed-pressure-authority.ts`). The head authority (`this.authority`: surface-only, non-coarse, non-resident) now takes the surface work control's revisions, the same as the split authority.

- **Stamps.** Each region gets 16 stamp words: words 0–11 are the inputs, then a changed/held word, the epoch at the last pass, COLD at the last pass, and quiet.
- **Quiet** = COLD now, inputs unchanged, COLD at the last pass, and epoch unchanged. So a region that went non-COLD between two passes cannot hold on stale words.
- **Held** = the region and all its stored neighbours are quiet (`stampHeld`). The split authority's held test uses the same stricter quiet.
- **`surfaceHeld` kernel.** It strides regions over the fixed region grid, a workgroup each. A region that is held and whose tiles' owner bases are unchanged returns. Each tile's base + 1 is stored after the stamp words, because the head's phi buffer is indexed by owner rank, so a rank shift must rewrite.
- **`encodeAll`.** This is the old full owner launch. It is now used only after forces on a kick (a kick skips the head authority).
- **Forces no longer overwrite phase** (stream 1). Curvature lives in `normals[owner].w`. The post-forces re-encode is now kick-only, and nothing has to be restored in held regions.

**Lazy gauge correction.**

- Stream 1's RHS (`uniform-mixed-pressure-velocity.ts`) and band prep (`uniform-pressure-band.ts`) compute `params.authority.z × capacity` for a liquid owner themselves, bit-identical to the texel.
- `umSurfaceOwner` no longer stores the correction, so a held owner does nothing.
- Both NB-FLIP authorities are surface-only, so that store was the frame's last correction writer. The non-surface-only `resolve` path is still in the file, but nothing constructs it.
- The `correction` texture stays bound. The reference aliases it to `gammaA`, so removing it frees no memory.

**Transport deep-COLD exit** (`uniform-mixed-surface.ts`). The surface claims buffer gains a deep word per region (23 words per region, was 22), copied from the heat tiers beside the tier copy.

- In a deep-COLD region, `umHoldCold` no longer stores non-aligned h vertices.
- 4h lattice vertices (4-aligned) are still stored, because the reconstruct group's 4h corner paths read them from the scratch.
- `advectFine` reads the deep flag once per tile job (workgroup-uniform) and evaluates only the aligned lanes there.
- This relies on stream 3's `nbColdCurrentPhi`. The reconstruct-group crossing search reads COLD h vertices from `outputPhi` (`f.phi`).
- The scratch phi resolve reads only 4h lattice vertices. A non-COLD reader's stencil is at least a region away from any deep-COLD region.
- So a skipped scratch texel, which holds momentum's between-frame data, is never read.

**Exactness** (scratch probe, final fields compared word for word: authoritative velocity, phi, density, velocity, negative velocity, geometry target and centre phi):

- Deep exit on vs off (the deep copy suppressed): 0 differing words, settled 8 frames and pool-impact 6 frames.
- Head held mode vs every-owner `encodeAll`: 0 differing words over the same windows.

**Stage costs** (investigate tool, one run per arm, other streams' work in the tree):

| Stage (mean ms) | Settled before | Settled after | Pool-impact before | Pool-impact after |
| --- | ---: | ---: | ---: | ---: |
| Frame | 21.83 | 22.51 | 48.23 | 49.57 |
| Surface pressure authority | 0.56 | 0.24 | 1.47 | 1.52 |
| Surface advect | 0.49 | 0.47 | 2.42 | 2.15 |
| `advectFine` (split run) | 0.27 | 0.22 | n/a | n/a |
| Surface advect, deferred | 0.07 | 0.11 | 5.56 | 5.79 |
| Surface work control | 0.23 | 0.23 | 0.63 | 0.57 |
| Region heat | 0.04 | 0.05 | 0.04 | 0.05 |

Settled is frames 34–40, and pool-impact frames 24–30. The before columns are the previous addendum's after runs. The frame totals include other streams' concurrent changes. Several runs taken while other sessions' GPU profiles were running read 2–5× high and were discarded.

**Limitations:**

- **advectFine in settled.** Every settled region is deep-COLD from frame 2, yet `advectFine` still costs about 0.22 ms. That cost is the claim loop over the plan's fine list: a claim, two barriers and a deep-flag load per h tile job. Removing it needs the fine list to leave out deep-COLD tiles in the plan, or a region-granular job list. Both are outside this stream's files.
- **Pool-impact authority.** Few regions there are held, and the stamp passes cost about what they save.

## Stream 3: momentum, extension and lifetimes

### What was built

Particle lifetime gate. `UniformNarrowBandFlip.liveGate` is a 16-byte uniform buffer. After the particle count is committed, the frame copies this frame's GPU particle count into it with `copyBufferToBuffer(state → liveGate)`, with no CPU readback and no delayed count. Every particle consumer reads it up to `update` and exits immediately when the count is zero:

- the NB-FLIP transfer,
- the paired extension's companion seed, retain and snapshot entries (`ueParticlesLive`),
- the cache and stage-tap builders.

The grid publication and seeding still run. The extension's own pipelines bind an `alwaysLive` gate, a 16-byte buffer that is always 1.

Particle advection body gate. Per particle, the body-global gate in the NB-FLIP advect is now `bodyNear = umBodyCount()>0 && umBodyPoseNearTile(tile, 2)`. A particle away from every body's posed support skips the body sampling and contact branch.

Momentum tiers (`uniform-mixed-momentum-tiers.ts`, new). A `decide` kernel runs ahead of momentum in momentum's pass, with one lane per 16h region. It reads the plan's `regionVelocity` bounds over the 3×3×3 regions around the region, then retains the region when all of these hold:

- the velocity span w is within the remaining budget, where the budget is `τ × hmin` of accumulated skipped error and τ is the h band's pressure residual target (see the addendum below);
- the trace stays inside the neighbourhood: `dt·max|u| ≤ 8 hmin`;
- no body swept within 24h (stream 2's mask);
- no inflow or drop source is near;
- the record's page key matches, so a recycled key starts in full.

A retained region's faces keep their extended value (`umRetainedMomentum`) instead of tracing. Its error accumulates and resets on the next full region. A constant velocity has zero span, so its cost does not grow with speed until the travel bound is reached. The addendum below adds an affine tier between retain and full.

Persistent extension slots. The slots live from the tail to the next head. (A first version gave the extension its own slot arrays. The addendum below moves them back into the `stageScratch` prefix.) With the slots kept across frames, a body-frame head can rebuild only the regions a body touched:

- The frame no longer throws away the reusable extension on a body frame. `bodyExtension` remembers that only bodies changed, and both heads (`advance` and `encodeRelayoutHead`) call `extension.encode(..., regional)`.
- Each swept encode first runs a marks pass (`Uniform mixed extension regions`):
  - `stampRegions` compares each tile's key and support bit with the stamp from the last swept encode. A region is changed when a stamp differs, when the encode is in full mode, or when stream 2's body mask gives a swept reach within `UNIFORM_EXTENSION_BODY_REACH = UNIFORM_BODY_GEOMETRY_REACH + 1`.
  - `classifyRegions` marks a region dirty when a changed region is within Chebyshev distance 1, and halo within distance 2. It writes the state into both slot arrays after the slots, plus one global "any dirty" word.
  - A full mode is used whenever the retained slots are missing, the sweep parity differs, or new blocks were initialised (`invalidateRetained`). The settled scene runs `extensionSweeps = 1`, so parity matters.
- How each stage treats the regions:
  - Seed and sweep work only in dirty regions.
  - Halo regions copy their retained final slots into the other array, so the sweeps read consistent neighbours.
  - Restriction (`restrictBand`) skips clean tiles. They keep their root.down texels, because the last writer was the tail's current restriction of the same retained slots.
  - When no region is dirty, every publish entry exits (the any-dirty word, read with `workgroupUniformLoad` where barriers exist), leaving the tail's publication of `velocityScratch` in place. This is the same reuse argument the non-body reuse path already relies on.
  - Only the main restriction and publication use the `ueRegionalRetain` override. The companion, snapshot and replay variants are unchanged.
- The hierarchy (not a stream 3 file) still runs globally.

Exactness. A scratch twin (`probe-twin.mts`) ran a full encode straight after each regional head in the same frame and compared the final slots and the published `velocityScratch` texels bit for bit:

| Scene | Frames | Differing slots | Differing texels |
| --- | --- | ---: | ---: |
| pool-impact | 2–6 | 0 | 0 |
| settled | 2–6 | 0 | 0 |

The comparisons covered about 4.3M live slots and 16.3M texels. On pool-impact, 72–80 of 384 live regions were dirty and 88–96 were halo. On settled, all 384 were clean.

### Memory

Everything is allocated up front.

| Item | Size |
| --- | --- |
| Extension slots | `2 × (8 × extensionSlots(address, fineTiles) + regionBytes)`, where `regionBytes = ceil(8(pages+1)/256) × 256`. This is about 2 × 494 MB at pool capacity. It is the `stageScratch` prefix again (see the addendum), so it adds nothing. |
| Region stamp | `4 × (2·tiles + 2·regions)` bytes |
| Mark mode uniforms | 48 B |
| Always-live gate | 16 B |
| Momentum tier record | 16 B per 16h region |
| Momentum affine models | 144 B per 16h region (36 words) |
| Momentum tier tolerance | 16 B |
| NB-FLIP live gate | 16 B |

### Smoke

Settings: dt 1/60 s, 30 steps from frame 2, profiled, one run per arm, under the GPU lock.

The machine was running about 20–30% slow during these runs. Stream 1 saw the same thing, and untouched passes such as the pressure V-cycle doubled. Other streams' uncommitted work was also in the tree, so whole-frame means are not comparable with the baselines:

- settled: 49.1 ms against base 39.3;
- pool-impact: 68.1 ms against base 53.8.

The extension pass results are better evidence:

| Pass | Base | This tree |
| --- | ---: | ---: |
| Settled head extension | 1.270 ms | 0.753 ms + 0.086 ms marks |
| Settled tail extension (marks, preparePair, snapshot, finish) | 0.757 / 0.581 / 0.506 ms | 0.036 / 1.087 / 0.497 / 0.628 ms |
| Settled momentum | 1.521 ms | 1.085 ms |
| Settled NB-FLIP transfer | 0.655 ms | 0.221 ms |
| Settled NB-FLIP snapshot | 0.470 ms | 0.077 ms |

On settled the tail extension passes are noise-level, with no stream 3 change in them beyond the marks. The settled head's remaining 0.75 ms is the global hierarchy plus fixed empty launches.

On pool-impact the head extension costs 1.96 ms regional against 1.92 ms with regional forced off in the same tree, which is break-even. The dirty and halo regions around the impact hold most of the fine tiles, so skipping clean 4h regions saves little there.

`npm run check:types` is clean, and `npm run test:unit` gives 860 passing, 54 skipped and zero failed. No tests were deleted. No A/B was run.

### Limitations

- The extension hierarchy, the plan and the phase authority still run globally on body frames. `phaseKeyed = !bodies` sits in stream 1's pressure authority and is unchanged.
- The tail's paired extension (preparePair, snapshot, finishPair) is still full every frame. Retaining slots across pressure would leave stale fluid-face source velocities in the publication, and the companion payload is transient.
- Halo regions keep retained finals at a dirty boundary. Their dirty neighbours therefore read converged values where a full encode reads the previous sweep. The twin showed no difference in either scene, but this is not proven in general.
- The quiet publish skip and the clean-tile restriction both assume that nothing writes `velocityScratch` or root.down between the tail and the next head, which is the existing reuse contract.
- The stale extension tools are deleted (see the addendum).

### Stream 3 addendum: memory, tools, tail extension, affine tier

#### Memory (measured on device)

`probe-memory.mts` reads `frame.allocatedBytes`, the stage scratch, the slots and the frame's capacity list at pool capacity (fineTiles = poolTiles = 245760; 246784 tiles; 3856 16h regions).

| | Persistent slots (first version) | Slots lent from `stageScratch` |
| --- | ---: | ---: |
| Frame allocation | 6,064,739,448 B | 4,576,182,080 B |
| `stageScratch` | 1,101,004,800 B | 1,101,004,800 B |
| Extension slots | 988,695,552 B (own buffer) | 0 (the `stageScratch` prefix) |
| Extension own buffers | 988,633,600 B | 2,005,184 B |

The first version added about 0.99 GB. The slots are now the `stageScratch` prefix again:
- `UniformMixedExtension.useSlots(buffer)` takes the frame's stage scratch in `bindStage`.
- `stageBytesAt` sizes the scratch to at least `slotBufferBytesAt(fineTiles)`.

What is left over base:
- the extension stamp and gates, 2.0 MB;
- the momentum record, models and tolerance, under 1 MB.

The frame allocation's other differences from base belong to the other streams.

The slots are dead while other stage consumers use the prefix. Those consumers are surface, forces normals and band rows, and they run between the head extension and the tail. The frame therefore calls `extension.invalidateRetained()` right after the head extension. The tail's `preparePair` re-seeds the slots, and the next head runs regional from them. That is the only window the body-frame regional head needs.

After lending, the twin (`probe-twin.mts`, pool-impact frames 2–6) still showed 0 differing slots and 0 differing texels.

#### Deleted tools

These files are deleted:
- `tools/probe-paired-extension-full.ts`
- `tools/probe-extension-direct-snapshot.ts`
- `tools/probe-extension-retained-seed.ts`
- `tools/extension-coarse-sweep.ts` and `tools/extension-retained-seed.ts`. These two were helpers only the deleted probes imported.

Nothing else imports any of them. Only docs mention them.

#### Tail paired extension: not done

A payload-only regional tail gated on a per-region velocity-change record was not built, for three reasons:

1. **Lifetime.** The tail's slots would have to outlive the stage consumers that reuse the `stageScratch` prefix between head and tail. That means separate slot arrays, the +0.99 GB the memory item removed.
2. **Indexing.** The slots are indexed by the reservation, not by live tiles. Sizing them to the live fine-tile count needs an allocator remap.
3. **No velocity record.** Stream 1's dynamics record (10 words a region) has dp and residual but no velocity-change word. Without one, a payload-only rebuild cannot prove a region's source velocities unchanged.

Measured tail cost being left on the table (30 steps from frame 2, mean per frame):

| Tail pass | settled | pool-impact |
| --- | ---: | ---: |
| preparePair (`Uniform mixed extension` #2) | 0.753 ms | 0.823 ms |
| slot snapshot | 0.346 ms | 0.565 ms |
| finishPair (`Uniform mixed extension` #3) | 0.511 ms | 0.522 ms |
| marks (regions #2) | 0.036 ms | 0.027 ms |
| Total | 1.65 ms | 1.94 ms |

#### Affine momentum tier

> Superseded by the heat-driven tiers below. The τ × hmin budget, `setTolerance` and per-region error accumulation are gone. The fit, residual and bound described here still apply to WARM regions.

`UNIFORM_MOMENTUM_TIER` is now full, retain, affine and pending. Pending exists only inside the tier pass. The tiers run in their own pass, `Uniform mixed momentum tiers`, ahead of `Uniform mixed momentum`, in four steps:

1. **`decide`**, one lane a region. Applies the retain test. A region the budget alone refused, with every forced-full condition false, becomes pending. Its 27-region speed and its budget left are stored in its model record.
2. **`momentumAffineFit`** (momentum module), one 256-lane workgroup per (pending region, part).
   - Parts 0–2 fit one velocity component each. That component's faces form a regular n³ grid (n = 16/width, spacing width). About the grid's own centre the normal equations are diagonal: c = mean v and g_j = Σ v x_j / Σ x_j². There is no solve and nothing can be singular.
   - The parts also write the own-face RMS residual from Σv², a lower bound on the max residual.
   - Part 3 checks eligibility: all 27 regions stored, every tile at the region's width, so the sampler holds no departure.
3. **`momentumAffineResidual`** (momentum module), one workgroup per (candidate region, part), with 16 parts. It measures r = max |u − M| over every face the trace can read. Those are the region's own faces plus a halo of 3 cells of its width, all inside the 27 regions: a one-step departure moves at most 1.5 h, and the sampler reads faces within one cell of it. A candidate is pending, eligible, has RMS ≤ budget left, κ ≤ 1 and dt(speed + RMS) ≤ 1.5 hmin. The affine decision applies the same test, so a skipped region's stale residual is never read.
4. **`affine`**, one lane a region. A pending region becomes affine if `acc + e ≤ budget`, otherwise full.

Both fit kernels gather a workgroup's items lane-parallel (item = group + groups × lane) and exit at once when none are pending. Consecutive regions land in different workgroups.

An affine face runs the momentum's own one midpoint step on M (`umAffineMomentum`) and returns M at the departure. It uses no sampler.

**Budget, derived from velocity error.** One face off by e changes its cells' divergence by e/h. The projection already accepts the h band's residual target τ (1/s; 1e-4 at pool settings) as undetected divergence, so τ × hmin (5e-6 m/s at h = 0.05 m) is the velocity error a skipped face may add. Each region accumulates its tiers' per-step error bounds against that budget and resets on the next full step. `setTolerance` writes τ when the frame's key changes.

The per-step bounds:
- **Retain:** w, the 27-region span of `regionVelocity` plus 64 ulp of speed. Every sampler value there is a convex combination of those faces.
- **Affine:** e = r(1 + κ(1+κ)e^{2κ}) + 64 ulp (speed + r), with κ = dt max_a Σ_b |G_ab| / hmin ≤ 1. The bound requires dt(speed + r) ≤ 1.5 hmin, so both traces take one substep. The sampler reproduces affine fields, so a step perturbed by r per sample moves the departure at most dt r (1+κ) e^{2κ} / h.

**Measured** (`probe-affine.mts`: the same frame's momentum run again with the tolerance zeroed, which forces full, then the departure textures compared):

| | settled frames 2–12 | pool-impact frames 2–12 |
| --- | --- | --- |
| Regions full / retain / affine (of 384) | 0–357 / 27–384 / 0–4 | 62–384 / 0–322 / 0 |
| Fitted (eligible pending) | 0–6 | 2 |
| r of fitted regions | 2.3e-6 to 6.8e-6 m/s | 1.6e-3 to 0.19 m/s |
| Affine accepted | 4 regions in one frame | none |
| Affine max face difference against full | 1.5e-7 m/s (step bound 4.6e-6) | none |

At this budget the affine tier rarely qualifies. Settled velocities are about 1e-6 m/s, so a region's nonlinearity already uses up most of the 5e-6 budget. In pool-impact the residuals of the non-body regions are 1e-3 to 0.2 m/s.

Cost (mean per frame, 30 steps from frame 2; KO = fit, residual and affine dispatches removed in the same tree):

| | settled | pool-impact |
| --- | ---: | ---: |
| Tiers pass, KO | 0.027 ms | 0.029 ms |
| Momentum, KO | 1.496 ms | 1.602 ms |
| Tiers pass | 0.240 ms | 0.282 ms |
| Momentum | 1.184 ms | 1.600 ms |
| Sum, KO / with affine | 1.523 / 1.424 ms | 1.632 / 1.882 ms |

The first versions were latency-bound:
- One workgroup per region with a 4×4 Gauss-Jordan cost 2.3 ms in momentum.
- The 256-lane diagonal fit cost +0.6 ms.
- An unfiltered parallel residual cost +0.84 ms on pool-impact.

The RMS prefilter brings the cost to +0.25 ms on pool-impact and −0.1 ms on settled.

### Stream 3 addendum: heat-driven non-pressure work

The narrow-band producers and consumers, momentum and the regional extension head now read the per-16h-region work heat (`UniformRegionHeat.tiers`: COLD, WARM, HOT). The heat is the only gate. The revision-keyed and body-region checks it supersedes are deleted.

#### Momentum

`UniformMixedMomentumTiers.decide` maps the heat one to one:
- **COLD** retains every face.
- **WARM** runs the affine fit and residual. It goes affine when dt·e ≤ `UNIFORM_REGION_HEAT_BUDGET_H` (0.02) h, the displacement a COLD region may hold back in one step, and full otherwise.
- **HOT** runs full.

The fit, the residual and the bound e are those of the affine section above. Only the budget and the gate changed.

#### Narrow band: COLD means no fine work and no writes

`UniformNarrowBandFlip.attachHeat(tiers)` binds the heat. `syncTiers` copies the tiers into the reuse `bins` (new block `tiers`, one word a region) at the start of `measureBand` and `clearAir`. It throws if no heat is attached.

WGSL predicates (`uniform-narrow-band-reuse.wgsl.ts`):
- `nbRegionCold(region)` is true when the region's tier is COLD.
- `nbTileCold(tile)` is true when the tile is stored and its page is COLD. The reuse region of a tile is its page.
- `nbVertexCold(p)` is true when any stored tile holding vertex p (up to 8 tiles at tile faces) lies in a COLD region. With this lowest-tier rule, a COLD region's closure is never written, even by a hotter neighbour.

Gates:
- **Producers.** `nbStoreReconstructed` and `nbStoreInPlace` return at once for a COLD vertex. Reconstruction, redistance and air clearing therefore write no phi in COLD regions and bump no revision.
- **Prepare reuse.** A live, COLD, cached region marks itself clean (current revision 0, allowed 1) and exits before any fine work. The flag is folded into lane 0's workgroup value, so the exit is workgroup-uniform ahead of the barriers.
- **Crossing search.** `surfaceCells` skips a COLD tile's interior cells (local 1–13 on every axis). `coupleFine` and `couple` exit for COLD owners.
- **Redistance.** `depthGeometry` keeps a tile whose metric can be reused: it stores the geometry capacity and takes a uniform exit. `redistanceFine`, `depthCoarse` and `redistanceCoarse` skip COLD tiles.
- **Air.** `airSeeds` keeps a COLD tile's previous seed, or marks it wet if it holds surface tiles. `airClearFine` and `airClearCoarse` skip COLD tiles.
- **Metric.** `depthReuseSelect` no longer dirties a COLD region's metric because a neighbour key is unstored. A claim or remap promotes the region and its 26 neighbours to HOT (stream 2), so a COLD region's neighbourhood cannot change under it.

Stream 2's surface advect holds COLD vertices (`umHoldCold`), so the input phi of a COLD region is also unchanged.

#### Extension head

- The mark pass binds the heat (binding 4) and stamps each region's tier (`MR_TIER`).
- A region is re-extended when its topology or support stamp changed, the mode is full, or its tier or stamped tier is not COLD.
- The block-word key check and the body-region reach are deleted, along with `UNIFORM_EXTENSION_BODY_REACH`, the `bodyRegions` constructor argument and its WGSL include. The heat already triggers on bodies, sources, identity and unknown evidence.
- `UniformMixedFrame` passes `same` (not `regional`) to the head, so a COLD frame reuses the previous tail's final slots.

#### Extension tail: still not retained

The slots live in the `stageScratch` prefix, and forces and pressure overwrite it between the head and the tail. A COLD-only retained store would need the full slot set in the worst case, when every region is COLD. That is the same +0.99 GB as before, so the tail runs in full. Memory stays at base.

#### Measured

The settled scenario was run with SPLIT=1, profiled frames 36–44, all 384 regions COLD from frame 32. The figures are pass means per profiled frame.

"Before" (h0) is HEAD 0a932ced plus the other streams' work in progress at the start. "After" (h2) also includes stream 2's surface advect hold, so the total GPU time is not stream 3 alone.

| Pass | Before | After |
| --- | ---: | ---: |
| Total GPU | 27.11 ms | 21.20 ms |
| Narrow-band redistance | 1.092 | 0.262 |
| Narrow-band crossing search | 0.954 | 0.670 |
| Narrow-band surface reconstruction | 0.881 | 0.269 |
| Narrow-band retire empty air | 0.735 | 0.102 |
| Narrow-band coarse retirement | 0.284 | 0.058 |
| Uniform mixed momentum | 0.830 | 0.794 |
| Uniform mixed extension (head + tail) | 1.959 | 1.937 |
| Surface advect + deferred (stream 2) | 3.707 | 0.553 |

Selected dispatches, before → after (ms):
- buildDistance: 0.670 → 0.146
- coupleFine: 0.575 → 0.080
- couple: 0.146 → 0.022
- redistanceFine: 0.269 → 0.029
- depthSeeds: 0.291 → 0.087
- depthGeometry: 0.124 → 0.058
- depthTiles: 0.306 → 0.255
- depthNearestZ: 0.146 → 0.117

Exact-metric regions went from 120 to 384.

The extension head went from 0.670 to 0.641 ms. 35 of 36 heads took the regional path, but the cost is the hierarchy dispatches, which run at fixed size. The tail went from 1.879 to 1.871 ms.

**COLD writes.** `probe-cold-phi` diffs phi texels between consecutive frame ends. It found 0 changed texels on frames 32–42 (all COLD) and 4.6k–5.2k on the WARM frames 29–31.

**Pool-impact** (frames 2–30, mostly HOT, so close to unchanged):

| Pass | Earlier (pre-heat) | After |
| --- | ---: | ---: |
| GPU | | 43.57 ms |
| Extension | 2.404 | 2.725 |
| Crossing | 1.797 | 1.828 |
| Momentum | 1.600 | 1.659 |
| Redistance | 1.340 | 1.390 |
| Reconstruction | 1.092 | 1.184 |
| Retire air | | 0.655 |

Single runs vary by 10–20%. maxMomentumError is 1.555e-5, the same as before, with 105,052 particles.

**Memory.** The narrow-band bins grow by 384 words (1.5 KB): regional reuse is 238,624,716 B, against 238,609,292 B before. The extension stamp is the same size. Nothing else changed.

**Gates.**
- `npm run check:types` is clean. Every changed WGSL module compiles on Dawn.
- `npm run test:unit`: 909 tests, 851 pass, 54 skipped, 4 fail. The failures are gpu-startup-mesh-memory, svo-gbuffer-normal-split, svo-surface-mesh-build-pacing and svo-surface-mesh-detail. All are `ReferenceError: Cannot access 'SVO_MATERIAL_FUNCTION_IDS' before initialization` in `lib/svo/features/materials/svo-procedural-material.ts`, from another session's SVO work.
- No tests were added or deleted.

### Stream 3 addendum: COLD exits in extension, momentum and crossing search

**The tail cannot leave COLD output untouched.** The request was to skip the tail extension's seed, sweeps and restriction in COLD regions and leave their published values in place. That does not work, because the published field is `velocityScratch` and three stages overwrite it every frame in every region, COLD included:

- forces write u* there;
- the projection writes its output there;
- the band copy writes there.

After the projection, a COLD region's air faces hold u* built from the last extension plus g·dt, not extended values. Skipping their publication would let gravity build up in the air faces. It would also feed a wrong speed to the heat's displacement measure (`plan.encodeSpeed`), and to advection and G2P.

Keeping the old extension would need either a persistent extended copy, a field-size texture (about 250 MB or more at this capacity), or forces and projection that leave COLD air faces alone (stream 1's files). So the current payload's tail stays full. What I built instead, by area:

**Tail: old FLIP payload, regional (`uniform-mixed-extension.ts`).**
- In a full encode, the mark pass now gives a region state 3 when it lies within one region (Chebyshev) of a hot region. A hot region is non-COLD, or one whose topology word or support bit moved.
- Only state-3 regions publish the old payload:
  - The snapshot restriction (`ueOldPayload`) restricts only state-3 tiles. Other tiles keep the last restriction's texels.
  - `snapshotFine` skips its job with a workgroup-uniform exit. `snapshotCoarse` skips per lane.
- The mark pass writes two work words: any dirty region, and any state-3 region.
  - The snapshot's hierarchy runs under gate 2, so it exits when no region is state 3.
  - The current payload's finish then replays the old selections (gate 2) or prolongs instead (gate 3, the complement).
  - Where a region saved no tag, the publication replay already falls back to the current payload's own far value.
- G2P (`uniform-narrow-band-flip.ts`) takes no FLIP delta at a COLD region's samples: `flip = velocity`, still blended toward the current PIC value. The snapshot does not write `before` there. Both read the same tier copy (`syncTiers` before snapshot and update).
  - So no COLD sample reads the old payload.
  - A non-COLD sample's taps lie within one cell of its region, which state 3 covers with a 16h margin.
  - Old/new operator compatibility therefore holds wherever the delta is taken: the old and current payloads share selections in state-3 regions. Elsewhere, the current payload prolongs on its own and no delta is taken.
- After the tail, nothing reads the slots except the next head's regional rebuild. That rebuild reads the final slots and region words, not the companion payload.

**Head hierarchy (`webgpu-uniform-velocity-extrapolation*.ts`).**
- Every hierarchy kernel (restrict, block sweep, prolong, replay) now starts with an immediate, workgroup-uniform exit. Lane 0 reads a work word (`heWork`, binding 8), and `workgroupUniformLoad` shares the result.
- Pipelines are compiled per gate at attach, using the override `heGate`:
  - 0: always run;
  - 1: skip when no region is dirty;
  - 2: skip when no region is state 3;
  - 3: skip when one is.
- A regional head with no dirty region runs no hierarchy.
- The previous head mark (any non-COLD region was "changed") caused the pool-impact head regression of 1.14 → 1.45 ms. It is replaced by a body test: a region changes when a tile word or support bit moved, or when it is not COLD and a moved body's swept reach (`UniformMixedSolid.bodyRegions`) is within `UNIFORM_BODY_GEOMETRY_REACH + 1` h of it.
  - A regional head runs only after bodies moved: the tail extended the same velocity, and G2P moves only particles.
  - COLD regions beside a body are maintained at 16h only, as agreed.

**Momentum in COLD (`uniform-mixed-momentum*.ts`, `uniform-mixed-face-dispatch.wgsl.ts`).**
- A new tier, `hold`, applies to a region that was COLD at the last decide too.
- The first COLD frame still retains: it writes its extended values into the departure.
- After that, the region holds and writes nothing:
  - `momentumUnitStep` and `momentumRegularStep` take a workgroup-uniform exit;
  - `momentumStep` drops a held owner's faces before evaluation, through a new optional `skipOwner` argument, which forces does not pass.
- Only momentum and the kick copy write the departure. A claim or recycle heats the region first, so a region never holds values it did not write.
- In a COLD region, the departure is the extended velocity of the frame it went COLD. P2G falls back to it only at particle-less faces.

**Crossing search (`uniform-narrow-band-reuse.wgsl.ts`, `-nearest.wgsl.ts`).**
- 3,472 of the 3,856 pool regions are free in the settled scene. Every search (three a frame) cleared three words per lane for each of them.
- `NB_REUSE_ALLOWED = 2` now marks "free, words already cleared". Those words stay clear until the region is live again, because only listed, live tiles commit.
- The two readers now test `== 1`.
- `depthNearestZ`'s COLD copy from the nearest cache is unchanged. Removing it would need every `NB_DEPTH_B` reader to read the cache and flags instead.

**Settled** (frames 36–44, split timing, against `h2-split`; GPU 21.20 → 19.03 ms, which includes the other streams' concurrent edits):

| Pass | Before | After |
| --- | ---: | ---: |
| Momentum | 0.794 | 0.320 |
| ↳ momentumUnitStep | 0.510 | 0.124 |
| ↳ momentumStep | 0.218 | 0.138 |
| Extension slot snapshot | 0.415 | 0.189 |
| Extension head | 0.641 | 0.517 |
| Extension tail (retain, pair, finish) | 1.871 | 1.755 |
| Crossing search | 0.670 | 0.619 |
| ↳ depthTiles (×3) | 0.255 | 0.160 |
| ↳ depthNearestZ (×3) | 0.117 | 0.124 |

The settled scene has no state-3 region. The snapshot's hierarchy exits, and the current payload prolongs.

**Pool-impact** (frames 2–30, not split, against s3l from before the heat work and h2 from the heat work):

| Pass | s3l | h2 | Now |
| --- | ---: | ---: | ---: |
| GPU | 42.89 | 43.57 | 42.10 ms |
| Extension head | 1.143 | 1.451 | 1.157 |
| Extension tail | 2.050 | 2.059 | 2.158 |
| Momentum | 1.600 | 1.659 | 1.605 |
| Crossing | 1.797 | 1.828 | 1.625 |

- The head regression is gone.
- The tail's +0.1 ms is within the 10–20% run-to-run swing.
- Two of the four impact runs were inflated two- to seven-fold by another GPU client (Chrome); those runs are discarded.
- The simulation result is the same across repeats: 105,259 particles and 74 unsupported, against 105,052 and 57 before. That difference comes from the COLD G2P and momentum rules.
- maxMomentumError is 1.76e-5 (1.56e-5 before).

**Memory.**
- The extension stamp loses its per-region tier word: −4 B a region, −15.4 KB here.
- New buffers:
  - the work words, 16 B;
  - a no-body mask, 16 B;
  - the hierarchy's always-run gate, 16 B.
- Nothing else changed. Regional reuse is still 238,624,716 B.

**Gates.**
- `npm run check:types` is clean. Every changed WGSL module compiles on Dawn.
- `npm run test:unit`: 926 tests, 873 pass, 53 skipped, 0 fail.
- No tests were added or deleted.
- One non-split pool-impact run ended with a validation error in another session's work: `Uniform mixed phi resolve held tiles` lacks `COPY_DST` for its `clearBuffer`.

### Stream 3 addendum: held extension tail, COLD always retains, current-phi crossing (2026-10-18)

This replaces three parts of the previous addendum:
- "The tail cannot leave COLD output untouched": it now does, through held values.
- Momentum's `hold` tier: removed.
- The `skipOwner` argument in `uniform-mixed-face-dispatch.wgsl.ts`: reverted, so that file is back to HEAD.

#### Held extension tail (`uniform-mixed-extension*.ts`)

**Held buffer.** A persistent buffer, `held`, keeps one f32 per extension slot (binding 18, main layout only). Every publication from swept slots stores the value it published there, through `umPublished`, `uePublishFine` and `publishList`.

**Held regions.** A region is held when it is clean or halo and its slots are not final. A held region publishes:
- a closed or low face from physical, as before;
- a face with support where `umSource` holds from physical, which is the projection's output;
- every other face from `held[slot]`.

The tail republishes every canonical texel, so this covers every writer between tails: forces, the projection, the band copy and NB-FLIP's departure seed. No writer needs to know about COLD.

**Region words.** A region word gains 8 when the region is held. A held halo region (9) seeds from its own slots instead of copying the halo. A per-region `MR_VALID` word in the stamp records two bits: bit 0, the held values are current; bit 1, the slots are final.

**Mark modes.** The tail's paired encode uses a third mark mode, held (mode 2), once any swept marks exist (`heldReady`). In mode 2:
- A region changes when it moved, is unpublished, is not COLD, or has a body near it. Only changed regions seed, sweep and restrict.
- A region within one region (Chebyshev) of a hot region takes state 3, the old payload.
- The quiet word is forced to 1, so the publication always runs.

**Hierarchy gate.** The current payload's finish runs the hierarchy under gate 1 after a held tail. The head's regional rebuild then finds the slots invalid (valid bit 1 cleared) and publishes from `held`.

**Paired layout.** The paired layout cannot take binding 18: `maxStorageBuffersPerShaderStage` is 10 on this device, and paired already uses 9. The paired source therefore stubs the held accessors. Replayed publication stores its replayed value.

**Verified** with a scratch probe that diffs `velocityScratch` between consecutive frame ends, settled frames 35–40. A texel counts as air when its cell and its +x/+y/+z cells are air; block-edge texels are excluded.

| Store | Air texels | Air changed | Other texels | Other changed |
| --- | ---: | ---: | ---: | ---: |
| h | 14,387,224 | 0 | 389,112 | ~204,000 (liquid) |
| Base | 660,016 | 0 | 20,496 | ~15,400 |

#### Momentum: COLD always retains (`uniform-mixed-momentum*.ts`)

**Hold removed.** The tiers are now full, retain, affine and pending. COLD retains every frame, so the departure takes this frame's extended value, which the last projection corrected. All three hold paths are gone: the `umMomentumHeldUniform` exits in `momentumUnitStep` and `momentumRegularStep`, and the `skipOwner` filter in `momentumStep`. Settled bandResidual and rootResidual over frames 1–14 match the all-HOT pattern, for example:

| Frame | bandResidual | rootResidual |
| ---: | ---: | ---: |
| 3 | 3.09e-5 | 6.13e-5 |
| 9 | 2.02e-5 | 7.04e-6 |
| 13 | 1.32e-5 | 1.70e-5 |

They no longer sit at 9.58e-5 / 3.67e-5.

**Cheaper retain.** A retained owner now resolves its face first, in `umRetainedFace`:
- A low face reads its table, because its anchor's page is not stored.
- A drop gives zero below an open face.
- Every other face takes the extended value.

This skips the cull's four loads and the closed-wall test. Two arguments make it exact:
- At a closed positive face, extended equals physical, because the extension publishes closed walls from physical.
- A culled face has both cells dead. It is no pressure row, and the projection zeroes it.

The residuals, particles and unsupported counts are bit-identical to the unreordered build in two runs: settled frames 1–14, and pool-impact frames 1–8.

#### Crossing search at COLD vertices (`uniform-narrow-band-membership.wgsl.ts`, `uniform-narrow-band-flip.ts`)

The reconstruct group's crossing search compiles with the override `nbColdCurrentPhi`. This covers the `@noplanes` pipelines, `depthTiles` and the `depthSeeds` gather, but not `depthReuseSnapshot`.

With the override, `nbCrossingVertex` reads an h vertex of a COLD tile (`nbTileCold` of the sampling cell's tile key) from `outputPhi`, which is `f.phi` there.

4h corners still read the scratch phi. The detail rewrite allows only h loads on `outputPhi`, which is a read_write storage texture. So stream 2 must keep storing scratch base corners in COLD regions.

#### Measured

**Settled, all-COLD frames** (frames 36–44, split, ms):

| Pass | Hold + full tail (h3b) | Held tail + retain (h4) | Held tail + reordered retain (h6) |
| --- | ---: | ---: | ---: |
| GPU | 19.03 | 19.14 | 19.24 |
| Extension tail (retain, pair, snapshot, finish) | 1.755 | 1.107 | 1.136 |
| Extension head | 0.517 | 0.510 | 0.524 |
| Momentum, total | 0.335 | 0.816 | 0.744 |
| ↳ momentumUnitStep | 0.124 | 0.481 | 0.386 |
| ↳ momentumStep | 0.138 | 0.233 | 0.255 |
| Crossing depthTiles | 0.160 | 0.138 | — |
| Crossing depthSeeds | 0.138 | 0.109 | — |

**Extension tail.** The tail falls by 0.65 ms. The biggest drops:

| Tail pass | Before | After |
| --- | ---: | ---: |
| Pair sweeps | 0.51 | 0.14 |
| Seam seed | 0.14 | 0.04 |
| Fine seed | 0.10 | 0.04 |

Two passes did not fall: `publishSeams` (0.20) and prolongation (0.14).

**Momentum.** Retain costs about 0.4 ms over hold in an all-COLD frame. This is the price of feeding the projection's correction back. The pre-tier momentum was 0.794 ms.

The reorder takes `momentumUnitStep` from 0.459–0.524 ms per frame to 0.393 ms. The timer quantum is 0.066 ms.

The remaining retained cost is the dependent chain from page table to texture, for the extended load and the departure store. A cheaper exact form would store only the faces that the projection can change. In a region held in both the last tail and this head, those are the `support && umSource` faces, because the rest publish constant held values. That needs two things not built here:
- the extension's held words visible to momentum, at both stamps;
- proof that the source set did not change, which depends on stream 2's transport exit.

**Pool-impact HOT** (frames 1–8, held tail against held tail knocked out by a temporary switch, not landed):
- Residuals, particles and unsupported counts are identical.
- Tail: 2.14 ms held against 2.06 ms knocked out, one run each.

A later 30-frame impact run showed every pass 20–70× slower in frames 2–6. That run did not reproduce and is discarded.

#### Memory

| Item | Size |
| --- | ---: |
| One face field (`velocityScratch`: h texture 128×128×992 rgba32float 260,046,848 B, plus base 32×32×992 rgba32float 16,252,928 B) | 276,299,776 B |
| `held` buffer (`heldBytesAt`: 4 × `extensionSlots`, rounded to 256 B) | 247,158,528 B |
| Stamp growth (`MR_VALID`) | +4 B per region |

`held` is allocated with the slots, recreated only when its size changes, and never grown mid-run. It is 89% of one face field, so it stays under the one-face-field limit. `allocatedBytes` includes it.

#### Gates

- `npm run check:types` is clean. Every changed WGSL module compiles on Dawn.
- `npm run test:unit`: 925 tests, 880 pass, 45 skipped, 0 fail.
- No tests were added or deleted.

## Stream 1: incremental pressure

### What was built

Per-region dynamics record (`uniform-mixed-region-dynamics.ts`, new). One record of 10 words for each 16h region (`umPageOfTile(tile)`, 3856 regions at pool capacity), allocated once and never replaced:

| Word | Content |
| ---: | --- |
| 0 | force tier this frame (`UNIFORM_FORCE_TIER`: retain, bounded, full) |
| 1, 9 | own geometry key: block word and every tile's width bits |
| 2, 8 | neighbourhood key over 27 regions, plus the force parameters and a host generation |
| 3 | viscous velocity dropped since the last full evaluation (m/s) |
| 4 | frame stamp of the last full evaluation |
| 5 | 27-region speed bound used by the budget |
| 6 | max \|Δp\| (Pa) of the accepted 4h root pressure over the region's 64 tiles; +Inf when unknown |
| 7 | largest band predictor row residual (1/s) over the region's liquid band rows; 0 with no band row, +Inf for a free region |

Readers bind it with `uniformMixedRegionDynamicsWGSL(group, binding)`, which provides `umRegionPressureChange` and `umRegionPredictorResidual`. The forces read the tiers through `uniformRegionTierWGSL`, a separate 2-word-per-region mirror. That keeps the forces' storage-buffer count unchanged.

Two passes do the work:

- `encodeClassify` runs before the forces with one lane per region:
  - it hashes the keys;
  - it marks a region full when the neighbourhood key changes or when a body's swept reach (stream 2's `bodyRegions` bits 0–7) is within `UNIFORM_FORCE_BODY_MARGIN = 18` h;
  - otherwise it charges the viscous budget.
- `encodePublish` runs after the band solve. It writes word 6 from a per-tile pressure history (8 B per tile), using a fixed 4096-group grid stride. Word 7 is written by the band's initial measure with a workgroup atomic max per tile. Classify clears it.

Force tiers are viscosity only. A retain region drops the viscous term while the bound `dt·ν·4·vmax·Σ1/h²` accumulated since its last full evaluation stays within `UNIFORM_FORCE_RETAIN_BUDGET = 1e-6` m/s. vmax comes from the plan's `regionVelocity` bounds over the 27 regions, and the bound is independent of geometry. Bounded and full regions are evaluated exactly. Gravity, the bed and capillary terms are evaluated on every tier. The forces gate both viscosity sites (the cleared Laplacian and the `forcesFine` vector path) with `umViscosityExact(owner)`. `invalidate()` bumps the generation on solid edits, layout updates and coarse vertex phi publishes, which forces a full tier everywhere.

Body flux gating:

- In `umCutFlux` (pressure velocity) the moving-wall face velocity is evaluated only where `umBodyPoseNearTile(tile, UNIFORM_BODY_FLUX_REACH = 3)` holds.
- The band prep's `bodies` flag has the same gate.

This is exact. `umSolidFaceVelocity` is zero beyond 2·hmax of a body's bounding sphere, and a face lies within 1h of its region box.

### Design item A: incremental pressure

No new code was needed. The root solve already warm-starts from the previous accepted pressure, so it is already an incremental solve against p_ref, and quiet frames close their slots. On settled, word 6 shows the root pressure bitwise unchanged from frame 2 onwards (max |Δp| = 0; frame 1 was 13626 Pa).

The remaining lever is that closed-slot V-cycle and full-cycle launches still cost about 1.2 ms each when the root pressure is unchanged. Exiting those launches early on `max |Δp| = 0` plus a converged residual is the next step. It was not done here.

### Abandoned: per-face retained forces

The first design retained the whole force increment F per face and validated each retained face against the occupancies and curvature it was evaluated with.

Clean-state settled forces timings:

| Variant | Forces |
| --- | ---: |
| Tiered | 3.69 ms |
| Forced full | 4.30 ms |
| Base | 3.59 ms |
| All-retained upper bound | 2.26 ms |
| Without viscosity | 2.90 ms |

The per-face store and validation cost about 0.7 ms, which ate the saving. Revisions in the key made this worse: about 55 regions per frame changed their own key, which pushed about 280 regions to FULL through the 27-region neighbourhood. Revisions were therefore dropped from the key, and the tier was cut back to viscosity only. The saved variant is in the session scratchpad (`forces-retained-F.ts`).

The record originally also published a per-owner velocity change and speed (words du and speed). Nothing consumes them: stream 3's momentum tiers read `regionVelocity` directly. They needed 32 B per owner slot, which is 502,005,760 B at pool capacity, plus a full owner pass every frame. They were removed.

### Deferred

- **Band A cache separated from the transient RHS.** The band consists of the surface tiles whose phi changes. Its slots are reassigned every frame, and its rows live in lent stage scratch. A persistent operator cache needs a stable tile→cache map and about 3 KB per band tile. Not built.
- **Retained band fine correction.** The band runs 0 cycles on settled, so there was nothing to retain there.

### Memory

| Item | Size |
| --- | ---: |
| Record (10 words × 3856 regions) | 154,240 B |
| Tier mirror | 30,848 B |
| Params | 16 B |
| Root pressure history (8 B × 246,784 tiles) | 1,974,272 B |
| **Total, measured `allocatedBytes`** | **2,159,376 B** |

### Smoke and measurements

Settled, dt 1/60 s, under the GPU lock:

- Frame 1 is all full.
- From frame 2, all 384 stored regions retain viscosity, with budgets up to 4.8e-10 m/s after 4 frames.
- Word 7 is filled on 192 regions, at 1e-4 to 1.9e-4 1/s.
- Band residual histories match the base.

The six-arm settled A/B ran with the machine in its slow state, with pass sums of 47–50 ms against about 33 ms normally. Arms were base, viscosity tiers, and viscosity forced full via `invalidate()` every frame, two runs each. They used the earlier 12-word record with the velocity publish, which was removed afterwards.

| Stage | Base | Tiers | Forced full |
| --- | --- | --- | --- |
| Forces | 4.37 / 5.03 ms | 4.01 / 3.97 ms | 5.87 / 4.98 ms |
| Pressure rhs (body flux gating) | 1.31 / 1.16 ms | 0.72 / 0.70 ms | — |
| Band list and rows | 3.02 / 2.73 ms | 2.53 / 2.13 ms | — |
| Frame mean | 55.9 / 58.25 ms | 57.04 / 62.44 ms | 57.02 / 56.92 ms |

Stage times favour the change. The frame means are noise-dominated and do not show a whole-frame gain. The coordinator's A/B is the measurement to trust.

`npm run check:types` is clean and `npm run test:unit` passes. No tests were deleted.

### Stream 1 addendum: closed root launches and settled forces

All numbers come from settled runs at dt 1/60 s, under the GPU lock, as medians of frames 3–8. The machine was in its slow state for these runs: about 34 ms wall clock with per-pass profiling, against about 28 ms of summed passes. Profiled timestamps are quantised at 0.0655 ms. Per-dispatch numbers (`SPLIT=1`) inflate fixed costs, so the whole-pass numbers are the ones to compare.

#### Closed root slots

The requested gate is already in place:

- The schedule's gate pass before each slot decides from the checkpoint of the latest measure. That is the residual of the current system, not the previous frame's |Δp|, so an impact anywhere opens the slots in the same frame.
- Every root kernel (smoothTiles, descend, ascend, coarse, measure, restrict) tests `umSlotClosed()` first, and the test is workgroup-uniform.
- On settled, the lagged plan gives `expected = 0`, so all three floor slots (V, V, Full) already launch narrow at `ROOT_CAPACITY_THREADS`.

What remains is fixed per-dispatch cost:

- With 0 cycles executed, the closed V-cycle passes cost 1.245 ms and the closed full-cycle pass costs 0.918 ms, about 2.2 ms per frame.
- Launching every root dispatch as a single workgroup (64 threads) left V at 1.245 ms and moved Full only to 0.786 ms.
- So the cost is about 15 µs for each of the roughly 137 encoded dispatches. It is not threads, and no earlier exit inside the kernels can remove it.

The levers left are fewer launches per slot (for example a tile-local multi-sweep smoother replacing the 18 smoothTiles launches of each V) or fewer floor slots (`UNIFORM_MIXED_SCHEDULE_FLOOR`, which impact frames rely on). Neither was done.

The band's capacity-cycle narrowing from the earlier session stands: cycles past the window's largest executed count launch narrow. Band solve measured 1.38 / 1.31 ms narrow against 1.57 / 2.16 ms wide.

#### Settled forces

Revision churn feeds the normal pass:

- With every phi revision producer knocked out, cacheNormals fell from 0.524 to 0 ms and cacheCurvature from 0.131 to 0.066 ms.
- The producers are surface advect (about 50 regions per frame), NB-FLIP reconstruction (about 45), and redistance (4–19). Each one rewrites ulp-level phi changes into the same surface regions.
- Stream 2 now owns stopping that churn. The forces already key normal and curvature retention on those revisions (`normalDirty`), so the 0.6 ms goes away when it lands, with no change here.

forcesFine profile (per dispatch, settled):

| Variant | forcesFine |
| --- | ---: |
| General per-face path (before) | 1.507 ms |
| Vector path, per-face capillary | 1.18 ms |
| No capillary code | 0.197 ms |
| Equal-width capillary only | 0.459 ms |
| Coarse slope forced to zero | 0.26 ms (seam owners); fused `forces` 0.79 → 0.33 ms |

The cost is execution, not register pressure. Every unequal face rebuilt its coarse side's pressure-occupancy slope (`umReconstructPressureSlope`, six faces of up to 16 finer neighbours), so each seam 4h owner's slope was rebuilt up to 96 times.

What was built:

- `forceSlopes` reconstructs each seam 4h owner's slope once. It runs one lane per seam 4h tile, after cacheCurvature, into the forces' borrowed normal scratch. That scratch is dead after the curvature pass under the stage scratch contract.
- `umPressureSlope` reads the slope from that scratch when the curvature is cached.
- The fields are bit-identical to the previous code after 6 impact frames: 0 of 3,535,202 words differ.

Tried and dropped:

- An interior/seam split of forcesFine. The interior pipeline got a light equal-width capillary, and the seam pipeline ran over the seam h list. Before the slope cache it cost 0.52 + 0.72 ms against 1.18 ms unsplit. After the slope cache it cost 0.52 + 0.26 ms against 0.59 ms unsplit.
- The in-tile quiet-face skip (`umFineOccupancy`) is exact but measured no gain. It is kept as the capillary loop's shortcut.

Forces dispatches, before → after:

| Dispatch | Before | After |
| --- | ---: | ---: |
| cacheNormals | 0.524 | 0.524 |
| cacheCurvature | 0.131 | 0.131 |
| forceSlopes | — | 0.131 |
| forcesFine (two twins) | 1.25–1.31 | 0.59–0.66 |
| forcesRegularCoarse | 0.13–0.20 | 0.13 |
| forces (fused seam tiles) | 0.786 | 0.328 |
| **Sum** | **≈ 2.9 ms** | **≈ 2.0 ms** |

The whole-pass forces median after the change is 1.77 ms. Root residuals match the previous code in every arm.

Memory: `forceSlopes` adds no bytes. It reuses the normal scratch (16 B × owners, which `bind` already checks), and it adds one pipeline compiled at attach. The band's cycle window is 16 B on the host. `npm run check:types` is clean and `npm run test:unit` passes (860 pass, 54 skipped). No tests were deleted.

### Stream 1 addendum: forces follow the region work tier

The forces, normals and curvature now read stream 2's per-region work tier (`uniform-region-heat.ts`: COLD, WARM or HOT for each 16h region, constant for the frame).

**How the tier reaches the forces.** Classify copies `umRegionTier(region)` into the forces' mirror and sets each region's mode in the same lane (`umRdForce`). There is no separate forces-side pass. The inputs are:

- the heat tier;
- the dynamics tier (FULL when the 27-region key changed: neighbour widths, force parameters, host generation, or a body within the margin);
- the geometry's topology plane (read only when capillarity is on).

| Region | Mode | Fine and 4h faces | Retained normals and curvature |
| --- | --- | --- | --- |
| COLD, unchanged, primed | reuse | Kept drive plus fresh gravity and inflow; no viscosity, bed, capillary or normal gathers | Skipped |
| COLD, unchanged, not primed | store | Evaluated in full, viscosity exact; drive kept | WARM rule |
| COLD but changed, WARM, HOT | evaluate | Exactly as before the tier: the viscosity budget (`UNIFORM_FORCE_TIER.retain` drops it within 1e-6 m/s) | Exactly as before the tier: revision candidates against the fixed anchors; curvature from the dirty plane |

The first frame after reuse evaluates every retained normal and curvature in the region exactly (`normalsExact`), because revisions and neighbour changes were not tracked during reuse. COLD reuse replaces the revision check entirely. Everywhere else it stays.

**The kept drive.** For each owner's +faces it is viscosity + bed + capillarity as f32 velocity increments. A storing lane derives it as the evaluated value less the advected value and gravity. It is not summed in parallel, so evaluating lanes do no extra arithmetic. Gravity is a per-face add that needs only the two occupancies, so it is always applied fresh, and so is inflow. A dt change alters the force-parameter key, which ends reuse. The drive is kept in three places:

- `forcesFine`: one lane per h owner, 3 components.
- `forcesRegularCoarse`: one lane per 4h owner, 3 components, decided once before the face loop.
- The fused seam launch: each seam 4h quad lane keeps its own (patch, axis) by seam-list position. A regular 4h owner riding the launch keeps by owner when its face is one patch.

**Entries are tagged.** h owner indices are rank based and move on a relayout, so each entry carries a hash of the owner's tile, its width and the region's keeping stamp.

- A mismatch in a reusing region evaluates that owner without keeping it, and unprimes the region. The region keeps again with a new stamp the next frame.
- So no lane ever writes an entry that another lane may be reading.
- An owner beyond the slots (h index ≥ 64·8192, or seam position ≥ 8192) always evaluates.

**Topology plane.** Classify reads it only when capillarity is on, and the forces clear it only then, as before. Every geometry encode, including stream 2's rank-shift `normalTags`, runs before classify in the same frame. Without capillarity the bits accumulate. A surface tension change keys the neighbourhood, which gives a FULL dynamics tier.

**Not reduced in COLD.** `forceSlopes` still runs for every seam 4h tile. Its scratch slope is read by any owner that evaluates (a tag mismatch, an owner beyond the slots), and the stage scratch holds stale data otherwise.

#### Measurements

Settled scene, dt 1/60 s, under the GPU lock. The machine was in its slow state (about 34 ms wall clock).

| Arm | Forces pass median |
| --- | ---: |
| Before (previous addendum) | 1.769 ms |
| Tier forced WARM | 1.769 ms |
| Tier forced HOT (also the real heat for frames 2–8, startup) | 2.228 ms (first cut; fixed below) |
| Tier forced COLD, owner retention only | 0.852 ms |
| Tier forced COLD, with seam retention | 0.786 ms |
| Real heat, frames 33–40 (the scene goes COLD from frame 31) | 0.72–0.79 ms per frame (one frame 1.05) |

HOT costs about 0.46 ms more than before. The extra time is the exact normal pass (cacheNormals 0.66, cacheCurvature 0.26 ms per dispatch), which the revision filter used to skip.

COLD per dispatch (`SPLIT=1`):

| Dispatch | ms |
| --- | ---: |
| cacheNormals | 0.07 |
| cacheCurvature | 0.07 |
| forceSlopes | 0.13 |
| forcesFine | 0.13–0.20 |
| forcesRegularCoarse | 0.07–0.13 |
| fused seam | 0.33 (was 0.39 before seam retention) |

#### Accuracy

Final fields after 6 settled frames:

- COLD against HOT: velocity max |Δ| 3.3e-7 m/s, with max |u| 4.3e-6.
- WARM against HOT: 4.6e-7 m/s, from the viscous budget.
- phi, density and geometry: bit-identical in every arm.
- Root and band residuals: unchanged.

The forced COLD arm reuses the drive kept in frame 2 for frames 3–6 of a startup transient. Under the real heat, reuse begins only once a region has been quiet for about half a second.

#### Memory

The dynamics' tier mirror became the forces' store, bound read-write at the forces' existing binding 16, so no binding was added. With `forceFineSlots = 64·min(poolTiles, 8192)` = 524,288 on the settled scene:

| Item | Bytes |
| --- | ---: |
| Mirror (6 words × 3856 regions) | 92,544 |
| Owner entries (16 B × (524,288 h + 246,784 tile keys)) | 12,337,152 |
| Seam entries (49 words × 8192) | 1,605,632 |
| **Force store** | **14,035,328** |
| Region dynamics total (with the record, params and pressure history) | 16,163,856 (was 2,159,376) |

No pipeline was added. The mode decision lives in the existing classify kernel.

#### Gates

- `npm run check:types` is clean.
- `npm run test:unit`: 851 pass, 54 skipped, 4 fail. The 4 failures are `gpu-startup-mesh-memory`, `svo-gbuffer-normal-split`, `svo-surface-mesh-build-pacing` and `svo-surface-mesh-detail`. All four fail with `Cannot access 'SVO_MATERIAL_FUNCTION_IDS' before initialization` in `lib/svo/features/materials/svo-procedural-material.ts`, from another session's in-progress SVO and material edits, and are not in this stream's files.
- No tests were added or deleted.

#### HOT fix (2026-10-18)

The first cut made HOT "exact": bitwise normal tests, viscosity always on, and curvature always recomputed. That cost 2.23 ms against 1.77 ms before. It also did not match the pre-tier fields, because before was itself approximate: anchor tolerance ε = 0.01 and the 1e-6 m/s viscosity budget.

Attribution on one tree:

| Arm | Forces |
| --- | ---: |
| Before | 1.835 ms |
| Exact HOT | 2.10 ms (velocity Δ 5.4e-7 m/s against before) |
| Budget plus bitwise normals | 2.03 ms (Δ 2e-8) |
| Budget plus anchors | 1.90 ms, bit-identical |

**What changed:**

- Every non-reusing region (HOT, WARM, changed COLD) now runs the pre-tier revision-filtered normal and curvature retention and the viscosity budget, unchanged. The one exception is the exact pass on the first frame after reuse.
- `forceRegions` folded into classify.
- The evaluation order restored.
- The kept drive derived from the evaluated value.
- The fused launch looks up retention only for +faces.

**Result:** settled, dt 1/60 s, forces pass plus classify, frames 3–8, two runs per arm:

| Arm | Medians | Minimums |
| --- | --- | --- |
| Before | 1.245, 1.311 ms | 1.180 ms |
| HOT (forced) | 1.311, 1.311 ms | 1.245 ms |
| COLD (forced) | 0.786–0.852 ms per frame | — |

- HOT is within one 0.0655 ms timestamp quantum of before and well under 1.77 ms. The machine is faster today than for the 1.77 ms runs.
- HOT final fields are bit-identical to before: 0 differing words in every field.
- COLD against before: velocity max |Δ| 5.4e-7 m/s at max |u| 5.3e-6. Everything else is bit-identical.
- Gates: `npm run check:types` reports only 2 errors, both in another session's untracked `lib/svo/features/construction/svo-buried-solid.test.ts`. `npm run test:unit` reports 872 pass, 0 fail, 54 skipped. No tests were added or deleted.

### Stream 1 addendum: unowned passes in an all-COLD frame (2026-10-18)

I profiled one settled, all-COLD frame per dispatch: SPLIT=1 with NB_SPLIT_ALL=1, dt 1/60 s. The figures are per-dispatch medians over frames 36–44, summed per label per frame. The timestamp quantum is 0.0655 ms. I then gave the unowned stages COLD exits that are uniform per workgroup.

#### What changed

- **Frame plan seed (`uniform-mixed-frame-plan.ts`).**
  - Seed no longer computes the speed maximum. A per-tile `held` word (bit 31 marks it held) keeps a COLD tile's support flags: retaining encodes are a pipeline-override variant (`planRetain`), so a COLD tile with the bit set skips its flag scan. Any non-COLD or non-retaining seed rewrites the word.
  - The frame speed comes from `localSpeed` instead. It reduces each workgroup's maximum in shared memory and makes one global `atomicMax`. `localSpeed` must stay current, because region heat reads `regionVelocity`.
  - The separate tail speed seed (`SPEED_PASS`) is gone. `encodeTail` re-encodes the plan only when it is stale.
  - The constructor drops the velocity and negative bindings.
- **Region key (`uniform-mixed-region-dynamics.ts`).** `regionKey` now runs one 64-lane workgroup per region instead of a serial walk. It builds the same hash and exits early for regions with no tiles.
- **Phi resolve (`uniform-mixed-phi-resolve.ts`).**
  - Held variants of the full and gated launches skip a 4h tile whose 27-tile neighbourhood is entirely COLD and whose held marker is set.
  - A non-held launch of the held field clears the markers.
  - The scratch resolve stays unheld by design.

#### Per pass, all-COLD frame (ms)

| Pass | Before | After |
| --- | ---: | ---: |
| Frame plan seed (head + mid) | 0.393 | 0.066 |
| Frame speed seed (tail) | 0.066 | removed |
| regionKey | 0.328 | 0.066 |
| Phi resolve, total | 0.524 | 0.328 |
| Phi resolve, per dispatch | 0 / 0.131 / 0.131 / 0 / 0.131 / 0.131 | 0 / 0.131 / 0.066 / 0 / 0.066 / 0.066 |
| Census, plan, certificate, resolve and regionKey combined | 2.359 | 1.507 |

- The frame's pass sum went from 21.89 to 19.33 ms. Other streams changed the tree between the two runs, so only the per-pass rows above are attributable to this work.
- Health counts were identical between the runs:
  - bandResidual moved from 1.85e-5 to 9.6e-5;
  - rootResidual moved from 6.7e-5 to 7.8e-5.
- No A/B was run, so the residual change is not attributed.

#### Left unchanged, with the reason

- **Census `classify` / `classifyCoarse`.** Their velocity bounds feed decide, residency and the cube reach, so they can't be held exactly. Their depth samples read outside the region's closure. A knock-out of the error and depth samples saved one quantum each: classify 0.262 → 0.197 ms, classifyCoarse 0.197 → 0.131 ms.
- **Census `pageSeed` and `pageRequest`.** pageSeed costs 0.066–0.131 ms and pageRequest 0.066 ms.
- **`toPressureFine` (0.328 ms) and `toSimulationFine` (0.066–0.131 ms).** These carry this frame's forced and projected velocity. That velocity changes everywhere and sits on the pressure path.
- **Passes in stream 2 and stream 3 files.** I sent their measurements to the owners instead of editing:
  - Stream 2:
    - surface work control: scan 0.328, summaries 0.131;
    - geometry work control: scan 0.131;
    - advectFine 0.262;
    - NB-FLIP surface authority: 3 × 0.131;
    - buildRegions 0.131.
  - Stream 3:
    - momentumUnitStep 0.524, momentumStep 0.262;
    - extension: publishSeams 2 × 0.197, pair_sweepSeams 0.197 / 0.131, pair_seedSeams 0.131, pair_sweep 0.131;
    - NB-FLIP: prepare, prefixTile and markParticleTiles 0.197 each, surfaceCells 0.131, buildDistance 0.131;
    - update and hanging taps: 0.131 each.

#### The COLD air-texel gate was not landed

The proposal: forces, the projection and the band copy would leave COLD extension-owned air texels of `velocityScratch` untouched, so that the tail extension could skip them. Those mid-frame values are not dead.

- **toPressure reads them.** `tFineToPressure` takes a plain 16-patch mean of each 4h face's h patches, read from the forced field. That mean is the coarse root's flux for `umPressureLiquid` owners.
  - Take a surface 4h owner whose centre is liquid but whose top h row is fully air. Its +y flux is built from exactly the texels the gate would skip.
  - Holding them replaces u* there with last tail's extrapolation: in a settled pool, a g·dt jump within one 4h face.
  - The result is a spurious divergence at every surface 4h owner every frame.
- **toSimulation reads the projection's output.** The projection writes `velocityScratch` in all-4h pressure ownership at canonical texels, and `toSimulation` reads those texels back.
- **Band project reads the band copy.** Band project reads the band copy's seam texels.
- **A phase-only predicate can't prove the skip exact.** That needs the pressure layout's liquid and detached flags.
- **Forces see a different phase.** With surface tension, `cacheCurvature` overwrites `f.phase` before the face pass, and frame.ts restores it after forces. The projection and the band see the tail's phase.

I proposed an exact alternative to stream 3, entirely on its side: keep a retained copy of the published air texels and restore it at a COLD-held tail. My stages stay untouched, and the cost is one face field of memory.

#### Incident: missing COPY_DST

The phi resolve's held buffer was briefly created without COPY_DST, which its clear needs. That raised a WebGPU validation error in the encoder "Uniform resolve remapped phi" during a pool-impact run. Adding COPY_DST fixed it. A settled smoke and a 6-frame pool-impact smoke then ran clean.

#### Memory

- Frame plan held flags: 987,136 B.
- Phi resolve held markers: 987,136 B.
- Dynamics and forces are unchanged.

#### Deleted

- `tests/uniform-region-velocity.test.ts`: it pinned the old frame-plan constructor.
- `tools/probe-frame-plan-speed.ts`: a probe for the removed speed seed.
- `tools/probe-nbflip-speed.ts` now calls `encodeTail`.

#### Settled residual rise: momentum's COLD hold (2026-10-18)

The settled residual rise comes entirely from momentum's COLD **hold** tier. The fix belongs to stream 3.

| Arm | bandResidual | rootResidual |
| --- | --- | --- |
| Pre-tier run | ≤ 6.6e-5 after frame 4, oscillating; root cycles every other frame | down to ~1e-6 on alternate frames |
| Current tree | 9.58e-5 from frame 2 onward; no band or root cycle ever runs | 3.67e-5, creeping to 7.8e-5 by frame 44 |

The 9.58e-5 band residual sits just under the 1e-4 band gate.

**Method.** Each arm was a probe-only build of the current tree with one heat consumer's tier reads forced HOT (not part of the code). Each arm ran the settled scene once, at dt 1/60 s, for 14 frames.

| Arm | Result |
| --- | --- |
| All consumers HOT | Matches the pre-tier run |
| Momentum HOT only | Bit-identical to the pre-tier run |
| Forces HOT only | Still pinned |
| NB-FLIP HOT only | Still pinned |
| Surface transport HOT only | Still pinned |
| Momentum's COLD tier always *retain* (hold removed), nothing else changed | Bit-identical to the pre-tier run |

**Cause.** `decide` moves a region that was COLD at the last decide from retain to hold. In hold, momentum skips the departure store, so the departure keeps the extended velocity of the first COLD frame. Retain would copy the region's current extended velocity, which the projection corrects every frame. Under hold, that correction never feeds back, so every frame re-solves frame 2's pressure problem and keeps its residual. The root residual creeps with the volume gauge.

**Fix.** The cheapest exact fix is to have COLD always retain. This costs momentum's retain store in COLD regions. A hold would need exact evidence that the region's extended values are unchanged. I sent the measurement to stream 3, which owns `uniform-mixed-momentum-tiers.ts`. My own files needed no change.

#### Volume correction and capillary curvature leave the authority's fields (2026-10-18)

Both changes serve stream 2's authority hold: a held owner must cost nothing, so no reader may need a field the authority rewrites every frame, and no pass may overwrite one.

**Volume correction.** The authorities stored `params.z × capacity` per liquid owner every frame only because the rate `params.z` changes every frame. The two readers now form the product themselves from `params.authority` (bound as a 16 B uniform; word 2 is the rate):

- The RHS (`uniform-mixed-pressure-velocity.ts`) uses its liquid predicate `umPressureLiquid` and the authority's own capacity (coarse topology.x; else the cell or tile open fraction when solids are enabled; else 1).
- The band prep (`uniform-pressure-band.ts`) uses `bWet` and the h cell's open fraction.

Neither binds the correction texture any more, and stream 2 drops its stores. A bodies-only head used to leave last frame's rate outside body regions until the post-forces re-encode; the readers now always use the current rate. Stream 2 accepted that.

**Capillary curvature.** `cacheCurvature` used to store an unretained owner's curvature in `phase`, which the head authority then had to restore after the forces. That covered h owners past the retained prefix, or every owner without retained normals. The curvature now lives in the borrowed normal scratch's `w` (`normals[owner.index].w`, beside the normal in xyz), and `forceSlopes` preserves it. The forces write no field. Stream 2 has made the post-forces re-encode kick-only.

**Exactness.** Each arm pair was base vs after, one run each, dt 1/60 s:

| Run | Result |
| --- | --- |
| Settled, 8 frames | Identical |
| Pool-impact, 6 frames | Identical |
| Curvature, forced overflow (retained prefix 0), settled 6 and pool-impact 5 frames | Identical |
| Liveness check: zeroing the overflow curvature | Changes velocity by up to 0.39 m/s, so the overflow path is exercised |

"Identical" means every band and root residual and cycle count matched exactly, and every final field had zero differing words.

**Memory.** Unchanged. The scratch is already a vec4f per owner, and the correction texture stays allocated: the reference aliases it to gammaA, so removing it frees nothing.

**Deleted.** Two tests pinned the removed `curvature` force field: `tests/uniform-retained-normals.test.ts` and `tests/uniform-zero-curvature-seams.test.ts`.

## Settled melt: stream C (2026-10-18)

Stream C owned pressure and the per-dispatch floor. In a settled frame the pressure band now launches nothing, and the host has a stale, non-blocking view of the region heat. Any stage can use that view to leave out launches that would have no work. Two NB-FLIP method params (`update: "solver"`, tier fine) control the new behaviour:

- `coldRegions` (Melt A's key): "frozen16h" (default) or "fine".
- `coldFrameSkip` (this stream): "on" (default) or "off". With "off", every launch is encoded and B's heat hold is disabled.

With "off" for both, the solver behaves as it did before this work. That path is kept as a working option, not a fallback.

### What was built

**Host cold-frame evidence: `uniform-cold-frame.ts` (new).** The frame owns `readonly cold: UniformColdFrame`.

- `cold.ceiling` is the highest region tier any live region may hold this advance. `cold.quiet` means the ceiling is COLD.
- At the end of each frame, the receipt copies five region-heat header words: the COLD, WARM, HOT and free counts, and the mid-frame PROMOTED count. It also records the frame's live NB-FLIP particle count.
- Frame N reads frame N-2's receipt. The receipt ring already guarantees that receipt has been checked before N encodes, so there is no blocking readback. This is the same deterministic lag as the pressure plan.
- The ceiling is HOT, and every launch is encoded, in any of these cases:
  - there is no evidence yet;
  - the evidence predates the last host trigger;
  - the evidence frame promoted a region mid-frame;
  - the evidence frame had live particles.
- Host triggers are:
  - extension invalidation, a layout update, a relayout attach, a fine-capacity reservation;
  - a kick, a change of parameters, a host edit, added source volume, a census first frame;
  - a page drop or injection;
  - any body whose pose changed between advances (`rosterPosesMoved`, every body in the roster).
- With `coldFrameSkip` "off", the ceiling is always HOT.

**Band skip: `uniform-pressure-band.ts`.**

- The band receives the cold frame (`fields.coldFrame`). When the freeze is present and the frame is quiet, the band's `empty` gate leaves out:
  - the whole prepare pass: list, spatial order, prep and bakes;
  - the whole solve: init through measure, copy and project.
- The header clear and the slot-map clear still run. They are the band's entire result in an all-COLD frame, because under the freeze every COLD region is freezing or frozen and lists no band tile.
- A quiet frame is about 107 fewer dispatches.

**Empty-band receipt.** The final measure now reports cycle 0 and DONE when the band holds no tile. Before, it reported every encoded cycle, which widened the next frames' capacity launches for no work.

**Freeze integration of root and band** (agreed with A). This was done earlier in this stream and is summarised here:

- The root rhs and project read A's region texture (binding 14) and write corrections to an r32float storage texture (binding 15). Textures were used because both kernels were already at the adapter's limit of 10 storage buffers.
- The band list leaves out frozen tiles. A band face that borders a frozen cell is Neumann at its frozen F plus the region's T.

**Allocation and planning.**

- The receipt heat copy is now a plain loop; it used to allocate a `subarray` every frame.
- `mixedBodyTiles` clears its marks sparsely, so `syncMixedLayout` no longer sweeps the whole plane every frame. It accounted for 7.5 ms of self time in an earlier settled profile window and does not appear in the new one.
- The pose array grows only when the roster grows.
- What remains of the encode cost is mostly Dawn's native cost per dispatch (58% of steady-state main-thread time), not JS planning.

### Measurements

Pool scene at 1/60 s, frames 6 to 20, one run per arm. GPU utilisation was 3-4% before the timed arms.

| Settled | coldRegions fine, skip off (today) | frozen16h, skip on (default) |
| --- | --- | --- |
| Median frame (wall) | 19.3 ms | 14.3 ms |
| Dispatches per frame | 579 | 474 |
| Encode (mean, CPU) | 3.7 ms | 3.4 ms |
| Pressure passes, profiled | 8.05 ms | 2.86 ms |
| Band list and rows / solve / projection, profiled | 2.06 / 2.39 / 0.39 ms | 0 / 0 / 0 |
| maxSpeed | 2.56e-6 m/s | 2.57e-6 m/s |

The profiled pressure sum includes timestamp overhead. The remaining pressure work is:

| Pass | Time |
| --- | --- |
| Root V-cycle slots | 0.75 ms |
| Full-cycle slots | 0.47 ms |
| rhs | 0.68 ms |
| setup | 0.26 ms |
| project | 0.21 ms |

With frozen16h and skip off: 581 dispatches and about 17.5 ms. The GPU was not idle at that run's start, so treat its timing as indicative only.

Pool-impact, 30 frames, frozen16h with skip on against fine with skip off:

| Check | frozen16h, skip on | fine, skip off |
| --- | --- | --- |
| Impulse | 493.78 N s | 494.51 N s |
| Deepest point | 1.3434 m | 1.3434 m |
| Max momentum error | 1.56e-5 | 1.46e-5 |

- Impact frames are never quiet: the ceiling stayed HOT every frame, so the skip encoded everything.
- The impulse difference (-0.15%) comes from the freeze.
- The baseline arm's median frame was 45.4 ms. In the skip-on arm, frames 3 to 11 ran at 35 to 50 ms, matching the baseline arm. From frame 12, an unlocked `tools/benchmark-svo-dry-frame-gpu.ts` from another session shared the GPU, so those frames carry no timing.

**Residuals.**

- With the freeze, the root's entry residual rises by about 6.5e-6 per frame as the frozen terms accumulate. A single gated root cycle resets it every 4 to 5 frames.
- The baseline arm runs one root cycle about every second frame.
- The band holds no tile in a settled frozen frame, so its receipt residual is 0. In the baseline it stays between 1e-5 and 7e-5.

### Lag and contract

- A region that heats with no host trigger at frame M is fully encoded from frame M+2. In frames M and M+1, a skipping stage may leave it out.
- B's hold (`holdRises`, on only when `coldFrameSkip` is on) keeps a GPU-raised rise visible in the evidence for two more frames, so every rise reaches the host.
- This is 2 frames, not the "no later than the next frame" that COMMON.md states. A lag of 1 would need a blocking readback, because the app keeps two frames in flight.

### Not done, and why

- **Root slots.** These were not cut in quiet frames. Root non-convergence is fatal (fail fast), and the slots are gated on the GPU. Encoding fewer than the floor's three slots on host evidence that may be 2 frames stale could turn a heating frame into a fatal frame. `UNIFORM_MIXED_SCHEDULE_FLOOR` records a past failure of exactly that kind (dam 128^3). Closed slots still cost about 1.2 ms.
- **Root rhs, setup and project restricted to non-COLD rows.** Not done:
  - The root is a single global all-4h solve, and frozen regions' rhs changes every frame through T.
  - Setup writes into the shared pressure arena, which other stages rewrite every frame, so it cannot be retained either.
- **Seam compatibility risk.** On a band face bordering a frozen cell, the Neumann flux is F + T(new). It is not that face's own root projection, so where T varies across a region the band rows carry a small compatibility error.
- **Other stages' skips.** No other stage consumes `cold.quiet` yet. A gates its field stages itself (momentum, forces, transfers, extension publishes, remap publish), on `cold.quiet && freeze`.
- **Tests.** `npm run check:types` and `npm run test:unit` pass (935 pass, 51 skipped). No tests were deleted.

## Settled melt: stream B (2026-10-18)

Stream B owned whole-domain bookkeeping (census, work control, plan, certificate, surface volume, region heat), the particle machinery, and NB-FLIP geometry. In a settled pool the frame now reaches an *idle* state from frame 7. In that state no census runs and no NB-FLIP stage launches, and a frame encodes 215 dispatches where it used to encode 460. All of the remaining dispatches are pressure work (169), sensors, or work whose input changes every head.

### Settings

All of these are NB-FLIP method params with `update: "solver"`. In each case the fast setting is the default, and "off" (or "full") gives the behaviour from before this stream exactly.

- **`coldCensus`**: "replay" (default) or "full".
  - With "replay", a deep-COLD region keeps its census decision and replays its classification records; it does not decide again.
  - With "full", every region decides every census.
- **The tier hold** (UNIFORM_REGION_HEAT_HOLD_FRAMES = 2) follows `coldFrameSkip`. C passes `holdRises: coldFrameSkip !== "off"`.
- **The T-displacement trigger** follows `coldRegions`. This is the frozen region's T term added to the heat's displacement budget (`rhColdSpeed`).
- **Idle** has no key of its own. It needs both the freeze (`coldRegions` "frozen16h") and quiet frames (`coldFrameSkip` "on"), so if either key is off, no frame is ever idle.

The list-strided particle changes are not result-altering and have no toggle.

### What was built

**NB-FLIP idle: `uniform-narrow-band-flip.ts`.**

- `beginAdvance(cold)` sets the flag: `idle = cold.quiet && quietLast && freeze present && started`.
  - Quiet means that the evidence from frame N-2 has a live-particle count of 0.
  - A quiet frame seeds nothing; it runs the `resampleKeep` batch, which reuses existing pipelines and needs no new compile.
  - So an idle frame holds no live sample.
- In an idle frame these stages encode nothing: move, refine, reconstruct, transfer, snapshot, redistance and update. retire returns false. clearAir only clears the air-changed word.

**The freeze in the transfer: `uniform-narrow-band-flip.ts` and `uniform-narrow-band-surface.wgsl.ts`.**

- `attachFreeze(freeze)` adds binding 18, A's region texture.
- transferOwners and transferProperties leave frozen cells alone. transferFace leaves frozen faces at their original value and adds the thaw term (umThawFaceTerm) to the face mean.

**The census skip in the relayout head: `uniform-mixed-frame.ts`.**

- An idle frame drops its relayout.
- The drop comes after the residency and censused lines. So `censused` stays set, residency is not reset, and the last census's generation, residency certificate and page claims stand.
- The first frame off idle runs a normal census.

**Host census inputs: `webgpu-uniform-reference.ts`.** Every host-known event that can need a census breaks idle on its own frame through `cold.trigger()`:

- C's triggers: a page drop or injection ahead, and roster poses that moved;
- added source volume;
- solid edits and option or detail changes, through `invalidateExtension`;
- `setRelayout` and `updateLayout`;
- `mixedCensusMoved`, which covers these census inputs:
  - the stale-mixed count and the scene revision;
  - dt, compared with Math.fround because advanceTo's dt alternates by one ulp;
  - the body count;
  - padding, percent, fade and sparse;
  - a pending edit join;
  - a dynamic detail policy that is not current.

**Idle guards for B's per-frame stages: `uniform-mixed-frame.ts`.** These guards run only in idle frames. They come in two kinds.

The first kind leave out work whose inputs are unchanged since the last frame, so the outputs already hold what a run would write:

- `plan.encode` in encodeParticleMomentum. It reads phi, velocity, ownership and tiers, and no solid.
- `plan.encodeCertificate`. Its inputs are the extension (not run under extensionIdle), the support, ownership and dt.
- `surfaceVolume.beginStep`. The gauge's rate is a function of phi, the budget and dt. The copy of the rate into the authority parameters still runs, because the host write zeroes that word every frame.

The second kind leave out work whose result nothing reads:

- the surface advect, its 16h surface work control, and the scratch phi resolve. Only reconstruct and redistance read the scratch, and neither runs.
- the three resolves of phi after reconstruct, clearAir and redistance. Phi is unchanged in an idle frame.

**Census replay: `uniform-mixed-dynamic.ts` and `uniform-narrow-band-budget.wgsl.ts`.** Deep-COLD regions keep their census decision (CC_DECIDED) and replay their classification. Non-COLD regions, and regions that neighbour a change, decide again.

**Region heat: `uniform-region-heat.ts`.**

- The 2-frame hold of GPU-raised tiers (state word 8).
- The freeze binding (binding 8), and the T displacement (`rhColdSpeed`).
- `encodeDeep` runs only in the relayout head.

**Particle lists: `uniform-narrow-band-flip.ts` and `uniform-narrow-band-order.ts`.** Particle launches stride the occupied-tile and live-particle lists rather than the whole bin range.

### Measurements

All runs: pool, 1/60 s, frames 6 to 20, one run per arm.

**Dispatches in a settled frame.**

| State | Dispatches |
| --- | --- |
| Start of this stream | 474 |
| Every launch encoded (NB idle forced off) | 460 |
| Idle, NB stages only | 382 |
| Idle, plus the census skip | 248 |
| Idle, plus the frame-level guards above | 215 |

The 215 dispatches of an idle frame break down as follows:

- **Pressure: 169.** Full and V-cycle slots, setup, measure, checkpoints, gates, rhs, project, root owners and to-pressure.
- **Kept on purpose: 46.**
  - geometric fill and its geometry work control: 13;
  - NB surface pressure authority: 10;
  - body forces: 8 (A's);
  - solid record: 4;
  - heat: 3;
  - region dynamics classify and publish: 2 + 2;
  - freeze: 2;
  - rigid coupling and integration: 2.

**Wall time.** The GPU read 40 to 99% busy from another session for the whole of this stream, so no clean timing exists. One pair of runs, made back to back under the same load, is indicative only:

| Arm | Median frame | Encode (mean, CPU) |
| --- | --- | --- |
| Idle on | 7.3 ms | 2.1 ms |
| Idle forced off | 16.8 ms | 4.0 ms |

Pool-impact is never quiet, so idle never engages there. Its encode path and dispatch count (614.5 per frame) are unchanged by this stream. Its wall times under the same load are not comparable with the 49 ms base.

**Bit identity.**

- **20 settled frames, idle on against idle forced off.** These all match:
  - the dumped fields: phi, velocity, pressure, density, geometry target, centre phi and the authoritative velocity samples;
  - the presented pressure and phi;
  - the certificate words, region velocities, surface-volume budget and authority parameters.

  These differ, as expected:
  - The census bookkeeping of the 3472 free regions. Forced off, their epochs advance through encodeDeep twice a frame.
  - The NB reuse caches. They are validated by revision.
  - The certificate's tile lists. Their order differs from run to run because of the atomic appends; as multisets they are equal.
- **Resumption.** A host trigger at frame 18, run to frame 24: the fields are bit-identical. Idle resumes at frame 21.
- **coldCensus "full" against "replay".** The fields are bit-identical both with idle on and with idle forced off, the second being the case where the census runs every frame.

**Physics.**

| Check | Result |
| --- | --- |
| Pool-impact impulse | 493.776 N s |
| Pool-impact deepest point | 1.34343 m |
| Settled maxSpeed | 2.57e-6 m/s |

These are unchanged by every B change. The impulse is -0.15% against the fine base of 494.5 N s; that difference is A's freeze, as C recorded.

### Latency from a GPU-raised trigger (coldFrameSkip on)

**The worst case is 2 frames.** A region the GPU heats at frame M, with no host trigger, gets full treatment from frame M+2.

1. At frame M the heat raises the region's tier on the GPU. Stages gated on the GPU treat the region in frame M itself. Stages the host left out in frame M, because its evidence from frame M-2 was quiet, do not.
2. Frame M's receipt carries the raised tier counts. With UNIFORM_MIXED_RECEIPT_RING = 2, frame M+2 is the first frame whose evidence is frame M. That frame sees a ceiling above COLD and encodes everything. Idle also ends there, because idle needs quiet at both N and N-1.
3. The hold keeps a GPU-raised tier for 2 more frames, through M+2. So when the host's full encode arrives at M+2, the region is still above COLD on the GPU, and its region-gated kernels do its work.

Without the hold, a rise that lasted only frame M would be reported to the host at M+2, but the region would already be COLD on the GPU again by then. It would get no host-skipped treatment at all. A hold of 2 frames is exactly the length of the receipt ring, so it is the shortest hold that covers the gap. A lag of 1 frame would need a blocking readback, because the app keeps two frames in flight.

**Lags specific to idle.**

- **A body integrated on the GPU** that starts moving with no host trigger is caught by the heat's body bit. The census and the NB stages resume at most 2 frames later. Derived geometry is not affected by this lag, because it is never skipped (see below).
- **Host events** break idle on their own frame through `cold.trigger()`, so they have no lag.
- **C's open risk**, a census remap in the middle of a frame under stale cold evidence, cannot happen in an idle frame, because no census runs there.

### Not done, and why

- **Geometric fill and geometry work control (13 dispatches).** Each head's swept-body mask runs from the record's last build to the current pose, and the record rebuilds every frame that has bodies. The geometry work control is the mask's only consumer (`consumeBodyEvidence`). If an idle frame left geometry out, a body that moved in that frame would lose its swept evidence for good, and the derived geometry in those regions would stay stale. Skipping it safely needs evidence that accumulates over skipped frames.
- **NB surface pressure authority (10).** This file is not B's. Its cut pass reads the solid, which is rebuilt every frame that has bodies.
- **Solid record, heat, dynamics and freeze.** These are the sensors that end idle, so they must run.
- **Timing.** There are no clean milliseconds, because of GPU contention; see Measurements.
- **Gates.**
  - `npm run test:unit` passes: 940 tests pass, 51 are skipped, 0 fail. No tests were deleted.
  - `npm run check:types` currently fails only in another session's files, which are in progress: `uniform-spatial-performance.ts` (`band` is private), and earlier `webgpu-uniform-reference.ts` at `mixedFrameTrace` (`origin_m`). B's files type-check.

## Settled melt: stream A (2026-10-18)

COLD regions are kept at 16h with no fine/4h field work. A COLD region's fine velocity is frozen, and its pressure response is carried as one 16h velocity term per region. This replaces per-texel retention copies. The NB-FLIP method parameter `coldRegions` selects it:
- `"frozen16h"` is the default;
- `"fine"` builds no freeze and is today's path exactly.

It is a `select` with `update:"solver"`, wired through `narrowBandColdRegions` to `UniformMixedFrameCapacity.coldRegions`. C's band exclusion and B's transfer freeze key on the same freeze instance.

### What was built

**`uniform-cold-freeze.ts` (new).** `UniformColdFreeze` holds three pieces of state:
- **Canonical region record:** 4 f32 per region, T.xyz (m/s) and state.
- **`regions` snapshot:** a sampled rgba32float 2D texture (256 wide) that every stage reads. Texel `REGIONS` holds dt·g. A texture rather than a buffer, because C's root rhs/project is at the 10 storage-buffer limit.
- **`corrections`:** an r32float 2D texture with one texel per 4h face, at `3*tile+axis`.

Region states:

| State | Meaning |
| ---: | --- |
| 0 | live |
| 1 | freezing: the first COLD frame; T restarts at 0 |
| 2 | frozen |
| 3 | thawing: the first non-COLD frame; T is folded in once by `umThawFaceTerm`, then reset at the next head |

Two kernels:
- **`encodeHead`** runs once a frame on the final tiers: after `heat.encode`, or after `heat.encodeDeep` in a relayout head, so it follows its promotions.
- **`encodeAccumulate`** runs after the root projection and before toSimulation. One workgroup per region. Each frozen region adds the mean of its tiles' corrections to T. It then resets each correction it read to `NO_FACE`. The texture is also filled with `NO_FACE` at initialize, so a face the projection never evaluates never counts.

WGSL accessors: `umFrozenRegion`, `umFrozenCell`, `umFrozenFace`, `umColdTermOf`, `umColdFaceTerm`, `umThawFaceTerm`, `umColdForced` (raw + T + dt·g on an occupied axis-1 face, the forces' gravity rule) and `umColdCorrect`. With no freeze they compile to constants.

**A frozen face** is any face incident to a frozen cell. Its value F stays in `f.velocity`, `f.negative` and the forced field. Freezing happens after projection, never before.

**Momentum (`uniform-mixed-momentum.ts`, plus a `skipOwner` hook in `uniform-mixed-face-dispatch.wgsl.ts`).**
- Frozen owners are neither evaluated nor stored, and the fused dispatch skips them.
- A warm owner's frozen face returns the physical F.
- A thawing face adds T.

**Forces (`uniform-mixed-forces.ts`).**
- Frozen owners return early in `forcesFine`, `forcesRegularCoarse` and the fused dispatch.
- A warm owner's face against a frozen cell keeps the advected (frozen) value.

**Ownership transfers (`uniform-mixed-remap.ts`, `UniformMixedOwnershipTransfer`).**
- toPressure skips frozen (state 2) tiles. A freezing tile still restricts once, so the h tiles' pressure-only base anchors hold F.
- toSimulation skips freezing and frozen tiles. A warm tile's split plane or +face against a frozen cell keeps velocityScratch's frozen value.

**Extension (`uniform-mixed-extension.ts`).** The region word carries a frozen bit (16), set from tier COLD in `classifyRegions` when the instance was built with `coldFreeze`. The fine, coarse and list publishers write nothing in a frozen region, so velocityScratch is invariant there.

**Frame (`uniform-mixed-frame.ts`, small hunks).**
- Construction, initialize, head, accumulate and destroy.
- `coldFreeze` is passed to the heat. The heat is B's: it thaws a region whose T exceeds the WARM budget.
- `freeze` is passed to the band fields, the root projection and the NB-FLIP transfer.
- Host skips that use C's `UniformColdFrame`:
  - `frozenQuiet = freeze && cold.quiet` leaves out momentum, the 4h sampling cache, the hanging taps and toSimulation. With every region freezing or frozen, their kernels write nothing.
  - `extensionIdle = frozenQuiet && narrowBandFlip.idle` (B's idle) also leaves out every extension encode (census/head, retainPair, preparePair/finishSnapshot/finishPair) and the forces. Phi is held, so no capillary cache is dirty. A skipped tail leaves the extension unretained, so the next swept encode marks every region.
  - toPressure is kept. A region that flaps COLD→WARM→COLD inside the 2-frame evidence lag is in the freezing state under a quiet ceiling, and still needs its F restricted.

### Measurements (1/60 s, settled and pool-impact, frames 6–20)

Final arms (code as landed). Each arm waited for a quiet window (GPU utilisation under 10% for about 10 s), and the four ran back to back under the GPU lock. Utilisation was 0% at the start of every arm and rose to about 70% by its end, as another app's unlocked SVO benchmark resumed. The last frames of each arm are therefore contended; treat the numbers as indicative.

| Scene | Arm | Wall mean | GPU pass sum |
| --- | --- | ---: | ---: |
| settled | `coldRegions:"fine"` | 22.1 ms | 15.5 ms |
| settled | `"frozen16h"` | 6.0 ms (median 5.4) | 3.9 ms |
| pool-impact | `"fine"` | 48.5 ms | 42.9 ms |
| pool-impact | `"frozen16h"` | 51.4 ms | 42.9 ms |

- **Settled:** the stream-A stages (momentum, hanging, 4h sampling, body forces, extension, transfer to simulation, phi resolve, surface advect, remap) fall from 4.77 ms to 0.26 ms of GPU time. The freeze itself costs 0.013 ms. What remains is mostly the root pressure (V-cycle 0.72, rhs 0.67, Full-cycle 0.48, setup 0.26, project 0.21 ms).
- **Pool-impact:** GPU time does not change. The impact frames have WARM/HOT regions, so the quiet skips never fire there, and the per-owner frozen skips cost about what they save. The 3 ms wall difference is contention: its GPU sums are equal.
- **Earlier, more contended runs:** settled fine 25.3 / GPU 19.5 ms vs frozen 9.2 / GPU 5.0 ms, under a steady 55–60% load.

**Dispatches per settled frame:**

| Configuration | Dispatches |
| --- | ---: |
| Before (C's band skip only) | 474 |
| With A's frozenQuiet gates | 382 |
| Sustained idle frames, with B's dt fix and B's idle guards | 215 |
| Final, with the forces gate under idle | 207 (first quiet frame 457) |

The remaining dispatches are the root pressure (V/Full cycles 137, setup 14, measure, gates, rhs, project), the authority and geometry fill (20), the dynamics classify (the forces themselves are skipped), the solid record, heat, the freeze (2) and toPressure (2).

**Why quiet kept breaking.** The quiet evidence broke every 2–3 frames because `mixedCensusMoved` compared dt exactly, and dt alternated by one ulp. B now compares `Math.fround(dt)`.

### Physics

- **Pool-impact, 30 frames:** impulse 493.78 N s frozen vs 494.95 N s fine; deepest 1.3434 m in both.
- **Pool-impact, 20 frames from 6:** impulse 423.72 vs 423.66 N s; deepest 1.34343 vs 1.34338 m.
- **Settled maxSpeed:** 2.57e-6 m/s frozen vs 2.50e-6 m/s fine.
- **Settled, all skips on vs `coldFrameSkip:"off"` (both with the freeze):**
  - velocity differs by at most 1.5e-6 m/s;
  - phi by at most 3e-8;
  - geometry target and centre phi are identical.

### Not done, and why

- **Samplers do not compose T.** Momentum traces, extension seeds and G2P read F without T inside a frozen region. This is bounded by B's thaw trigger: T past the WARM budget thaws the region.
- **No in-kernel frozen skip for the hanging taps, halo or 4h cache in mixed frames.** The same `UniformMixedHangingTaps` also fills NB-FLIP's snapshot taps from a different source, so a skip needs a separate pipeline variant. Settled frames skip these launches whole. In pool-impact the launches are about 0.4 ms in total.
- **Phi resolve and surface advect.** They already hold deep-COLD vertices. In idle frames B now leaves them out entirely.
- **C's section is out of date.** Its note that no other stage reads `cold.quiet` predates these gates.
- **Timing.** The final arms each started from an idle GPU but ended contended. A fully clean before/after is left to the coordinator's integrated A/B.
