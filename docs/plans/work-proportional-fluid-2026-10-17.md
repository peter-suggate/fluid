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
