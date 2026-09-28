# h level set over 4h bulk

The requested architecture keeps moving bulk at 4h and treats the surface at h. Do not widen `coarseningReach` or the bulk ownership band to recover accuracy. The user explicitly wants the moving front bulk at 4h; setting `fastTravel=0` is not a user requirement and currently disables the fast-front coarsening exception. The live path still couples level-set authority to bulk ownership: mixed sampling reconstructs non-corner values from 4h corners, and relayout/resolve can erase h detail.

## Implemented component

`UniformMixedSurface` has an optional final constructor flag `fineSurface` (default false). It advects and rebuilds all h level-set vertices independently of the bulk owner width. Trilinear and cubic phi interpolation use h samples, and surface-retirement evidence examines the h field. Momentum sampling and bulk cell tracing retain the mixed ownership. This mode is not enabled in the frame.

`uniform-fine-surface-overlay-dawn.test.ts` passes with an entirely 4h bulk grid: a corrugated interface moves four h cells in one step while preserving 405 checked interior h samples. The corrugation is invisible at the 4h corner samples. The test also checks all 4913 vertices of planar h redistance, finite advection output, and zero fine bulk tiles. The original mixed-surface regression suite passes unchanged. Evidence: `docs/benchmarks/uniform-fine-surface-overlay-2026-09-28.json`.

## Remap and volume correction components

`UniformMixedRemap` now has an optional final `fineSurface` flag. It leaves the persistent h phi texture untouched during both remap and publication; bulk child fill reads the h field. The test cycles through eight fine/coarse/mixed ownership changes, checks all 729 phi samples bit-for-bit (including negative zero), and conserves volume within 1e-4 cell volumes.

`UniformMixedSurfaceVolume` has the same optional final flag. Surface integration, band indexing, gradients and correction operate at h while the desired volume comes from the actual mixed bulk owners. It does not refine bulk ownership or allocate extra fields. Its full h dispatch is a correctness baseline, not a demonstrated timing win. The test compares every output sample with existing all-fine surface correction over an entirely 4h bulk grid, for both correction signs and both in-place/separate-output paths. Conservative volume is unchanged.

The five related component tests pass (`/tmp/fluid-front-regression/fine-surface-components.log`). The separate original remap test fails its no-allocation guard in `UniformMixedOwnership.reserveHanging`; this also fails with the staged pre-change remap source (`/tmp/fluid-front-regression/remap-baseline.log`). Do not report that gate as passing. Typecheck still has the existing unrelated Sparse/tool errors, with none in the new surface components.

## Integration work

Select the optional remap and volume-correction modes alongside the surface mode, and stop reconstructing the h field from 4h corners. Ensure dynamic census and pressure surface classification inspect the h field. Merely selecting `fineSurface` in the frame is insufficient: subsequent old resolve/remap/correction stages can discard or inconsistently modify the newly retained values.

The first integration gate is the original long-dam front test, including actual toe-strip volume at frames 10 and 20. Then run comparable one-tile and dynamic timing windows, followed by the existing 120-frame ABBA gate. A passing analytic advection test does not establish scene correctness or the 2% target. Optimize h dispatch around actual surface work only after validating field preservation; never turn fast-moving bulk into a wider fine-cell band.

The pressure-subspace prototype is a separate optional experiment. Its isolated pressure matrix passes, but its live-scene experiment fails front accuracy and worsens one-tile timing. Do not enable it as a prerequisite for this level-set integration.

## Front ownership diagnosis and first scene integration

The screenshot's h strip is explained by live policy, not the pressure-only 2h collar: `coarseningSurfaceTolerance` was changed from 0.5 to 0 and `coarseningFastTravel` from 4 to 0 in this checkout. `umFinishTile` therefore marks every sign-changing tile required, and `decide` retains it plus departure-box coverage even when reach/hysteresis are zero. The structure overlay reads simulation owner widths. This conflicts with the requested 4h moving front bulk. Do not treat the h bulk strip as the desired final architecture.

Restoring the former speed threshold is only an approximation under the current coupled ownership: it also coarsens level-set authority and pressure rows. The intended fix needs separate bulk and h surface/pressure decisions, conservative flux transfer between the two, and once-per-frame pressure access construction. Wall impact/lift needs h pressure resolution without necessarily promoting the entire bulk tile.

`tools/experiment-uniform-fine-surface-scene.mjs` builds an isolated front-test bundle with h advection, remap, volume correction, direct h phi readers and disabled mixed-corner resolve. It preserves original pressure/momentum and does not change dynamic defaults. It is NOT enabled in production. Fine and fixed-corner fronts pass (90/169); dynamic fronts fail (87/164) with toe masses 0.000149/48.359 instead of required 10–40/200–800. Thus independent h phi alone is insufficient. Next integration must decouple pressure surface ownership and bulk decisions, rather than declare this prototype complete or widen the bulk band.

## Independent pressure-band builder

`UniformMixedBuilderLevel.band` now optionally overrides the default bulk band with a distinct GPU buffer/word offset. The existing single-band path is unchanged. This permits regular 4h bulk with a separately moving h pressure band and its pressure hierarchy. `adoptBuiltLayout` now compares the actual bulk/pressure tile words; absence of a 2h collar no longer incorrectly implies matching layouts.

`tests/uniform-independent-pressure-band-dawn.test.ts` validates five pressure-band generations against CPU layouts, including all-fine pressure over all-coarse bulk (no 2h collar). Bulk stays entirely 4h throughout, and pressure band storage uses a separate buffer and nonzero offset. This is builder support, not live pressure census integration. The original byte-for-byte GPU layout-builder test also passes. Next: derive the independent pressure band from the current h phi, construct/adopt pressure ownership at the correct point in the frame, and validate projection/flux transfer and front progression before changing production defaults.


## Independent pressure census and scene experiment

`UniformPressureSurfaceBand` scans all 125 h vertices of each tile and rebuilds a crossing bit mask. It detects detail invisible at 4h corners, treats exact zero as surface, and does not change bulk ownership. The census/builder tests pass for moving sub-4h features, planar crossings, empty/full fields, and independent pressure masks. Typecheck retains the 14 existing unrelated errors.

The isolated scene tool now accepts `--pressure-band`: it rebuilds pressure ownership after current-frame phi/momentum and before RHS, using the h surface census plus existing fine bulk tiles. For this experiment it restores fastTravel=4 and surfaceTolerance=0.5, with reach unchanged at zero. It adds a host readback and is a correctness experiment, not a performance implementation. It leaves production untouched.

Result: fine and fixed-corner fronts remain 90/169, but dynamic fronts are 73/151 and both reference toe strips contain zero volume. This is rejected. Evidence: `docs/benchmarks/uniform-independent-pressure-surface-2026-09-28.json`. Inspect the projected velocity transfer and subsequent bulk extension next: current projection writes h pressure velocities into scratch, restricts them to bulk faces, and census extension overwrites scratch. Independent h surface/pressure ownership alone does not establish correct momentum/transport coupling. Preserve the original acceptance bounds.


## Velocity restriction diagnosis

The scene tool supports `--keep-bulk-policy` to leave fastTravel/surfaceTolerance at the live zero defaults. Independent pressure then gives exactly the same dynamic front/toe results as h phi alone (87/164), whereas restoring fast coarsening gives 73/151.

`--velocity-restriction` reads projected h velocities and transferred bulk velocities before census extension. It compares all 16 h positive-face patches only where the source pressure tile is h, the bulk tile is 4h, and the adjacent bulk tile is also 4h (or a domain wall). Mean flux is preserved to roundoff; pointwise differences reach 3.4881 m/s. One advancing patch is 3.05237 m/s in projected h velocity but 0.190773 m/s in the 4h area mean. This is evidence of surface motion being diluted by full-face averaging, not a failure of conservative flux summation.

Next implementation: retain projected h surface velocities through pressure-to-bulk transfer and census extension, and sample them for h surface advection while keeping conservative bulk flux at 4h. Keep the overlay's validity tied to the pressure generation and refresh/extend it around current surface samples. Do not simply renormalize bulk flux, which would change mass transport. This is a supported hypothesis, not yet a verified scene fix. Evidence: `docs/benchmarks/uniform-pressure-velocity-restriction-2026-09-28.json`.

## h surface velocity sampler

`UniformMixedSurface` has an optional final `velocityOverlay` flag and matching `surfaceVelocityOverlay` texture binding. The h MAC overlay stores xyz velocities plus three component-validity bits in w. `uniform-surface-velocity-overlay.wgsl.ts` samples it independently of bulk ownership, falling back to the existing bulk sample for invalid taps/components and negative boundary planes. Bulk cell characteristic tracing explicitly retains the bulk sampler; the overlay only changes surface motion/contact sampling.

The h-surface Dawn test now runs both paths over entirely 4h bulk: ordinary bulk velocity 4 m/s moves the corrugated surface four h cells; overlay x velocity 2 m/s moves it two h cells while invalid y/z components contain NaN and safely fall back. Clearing validity restores the four-cell bulk motion. Bulk cell departure stays unchanged in both cases. Original mixed-surface tests also pass. Log: `/tmp/fluid-front-regression/surface-velocity-overlay-final.log`.

This is the consumer, not live frame integration. Still required: populate and retain projected h velocities with appropriate extension/validity, protect the overlay from bulk restriction and scratch reuse, and rerun the independent-pressure scene experiment. Do not claim front or timing recovery from this analytic test.


## Retained projected velocity and rejected first integration

`UniformPressureSurfaceVelocity` captures h-addressed projected velocities into an independent texture, with per-component validity. The optional bulk ownership restricts capture to h pressure tiles over coarse bulk. Every capture clears obsolete validity, so bulk relayout cannot reinterpret the spatial samples. The capture test verifies five ownership combinations, unchanged source values, and retention after source scratch overwrite.

The scene tool's `--surface-velocity` captures after projection and before restriction, and binds the retained texture to surface advection. Raw capture is **not sufficient**: dynamic fronts are 53/86, with no liquid in either reference toe strip. Fine and fixed-corner fronts remain 90/169. This is disabled in production. Next: wet-face validity and appropriate surface velocity extension rather than treating all projected h air faces as valid. Do not relax the front gate.

An exact-fallback bug was found and fixed: summing eight copies of the fallback with interpolation weights introduced rounding changes even with an empty overlay. The sampler now accumulates corrections relative to fallback and returns it unchanged when correction is zero. The complete empty-overlay surface output is tested exactly against the original. Both component tests pass; typecheck retains the 14 existing unrelated errors. Evidence: `docs/benchmarks/uniform-retained-surface-velocity-2026-09-28.json`.


## Wet-face validity, extension, and user review pause

Capture optionally accepts the pressure-phi binding and marks a component valid only when its positive face has at least one liquid pressure row. `UniformSurfaceVelocityExtension` extends invalid components from six neighboring valid h samples using separate texture parities; valid seeds are unchanged, bulk fields are untouched, and h bulk tiles are excluded. The component test checks wet/dry validity, preservation, and two Jacobi sweeps against an independent CPU reference.

The scene tool supports `--wet-velocity` and `--extend-velocity` (four h extension sweeps). Wet capture gives dynamic fronts 76/142; extension gives 76/149. Both have zero volume in the reference toe strips and fail the unchanged acceptance gate. Fine and fixed-corner fronts remain 90/169. These modes are NOT enabled in the browser UI. Evidence: `docs/benchmarks/uniform-wet-surface-velocity-extension-2026-09-28.json`.

User requested finishing the current changes and then pausing for UI feedback. The component and scene checks are complete, all our GPU test processes have ended, and the goal is paused. Do not resume autonomous work until the user resumes. The browser still runs the existing live configuration, including the h front bulk policy previously diagnosed; the new surface/pressure/velocity architecture remains isolated and has not met correctness or timing targets.


## UI enablement requested by user

The user explicitly requested finishing neighboring-air h velocity extension and enabling it in the UI. The live `UniformMixedFrame` now selects independent h phi, per-frame h pressure census/build, wet projected-velocity retention and four Jacobi extension sweeps before the next surface advection. Bulk characteristic tracing stays on bulk velocity. Phi consumers and presentation explicitly read the independent h field; coarse-corner resolve was removed from this path. Resource accounting includes the new buffers/textures. Fast bulk travel and surface tolerance defaults are restored to 4 and 0.5, with reach unchanged at zero.

Unlike the earlier isolated experiments, this path is now active in the browser. Chrome's existing long-dam tab completed initialization and advanced to 0.4000 s; it showed Play simulation (paused), 0 fine bulk tiles and 9216 coarse bulk tiles. The user can now review it. CPU grid-overlay tests pass, and typecheck has only the same 14 unrelated errors. No Dawn runs were launched alongside the open browser; the wet/extension components had already passed their GPU reference checks before enablement.

Known limits remain: the last isolated dynamic front test was 76/149 with empty reference toe strips, so correctness and the 2% timing target are NOT met. This enablement is for the user's requested UI review. Keep the broad goal paused pending feedback rather than continuing autonomous experiments.
