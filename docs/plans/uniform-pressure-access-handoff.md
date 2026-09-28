# Handoff: Uniform Geometric pressure access records

**Integration update:** regular-only bulk momentum and extension were found to delay the long-dam front and have been disabled in the live frame. Keep the restored h surface motion intact. `tests/uniform-long-dam-front-dawn.test.ts` now passes for all-fine and the reported coarse corner at both 0.333 and 0.667 seconds; include it when validating pressure changes. See `../benchmarks/uniform-long-dam-front-regression-2026-09-27.md` for the controlled comparison. The original bulk/surface architecture and 2% target remain active, but experimental timings that used the regressed front are not acceptance evidence.

**Dynamic policy update:** the primary workstream now defaults `coarseningSurfaceTolerance` and `coarseningFastTravel` to zero, retaining h at the moving surface. The previous 0.5/4 defaults could make this dam entirely coarse and cannot establish acceptance for the requested architecture. Use the same explicit 0/0 policy when comparing pressure candidates from older checkouts. The whole-frame benchmark also checks front progression against all-fine at frames 10 and 20; conservation alone is insufficient.

**User constraint:** do not widen the fine band; fast-moving fluid bulk should stay at 4h. The default `coarseningReach` remains **0**. A temporary change to 1 was reverted. Use surface tolerance 0 / fast surface exception 0 / reach 0 for subsequent dynamic diagnostics. Fix transfer/access behavior within that architecture rather than increasing fine bulk coverage.

The following dynamic results used the now-removed all-owner graph. **Historical baseline failure:** at frame 20, reach 0 reaches index 179 versus the control's 171. Wider-band diagnostics did not fix the liquid profile: reach 1 reaches index 171 but has only 43.65 h³ in the advancing strip, below the unchanged lower bound of 200 (all-fine 694.86; fixed coarse corner 467.64); reach 2 gives front 169 and strip mass 20.74. The primary workstream is investigating this; distinguish it from a new pressure-candidate regression. Do not loosen the test or pursue wider bands.

## Task and architecture

Implement and validate a fast pressure access path whose geometry, neighbor addresses, and coefficients are built once per frame and reused throughout the pressure solve. This is an independent workstream within the larger bulk/surface refactor.

The user wants interwoven or overlaid regular 4h bulk fields with h surface pressure and level-set treatment. Dynamic ownership changes often. Avoid maintaining a complex general-purpose topology for every stage; seams should be confined to pressure where necessary.

The overall acceptance target is candidate frame time no more than 1.02 times comparable all-fine frame time, for both the one-coarse-tile control and dynamic long-dam scene. Preserve numerical correctness. Pressure improvements alone need not achieve the whole-frame target; report their contribution honestly. Do not loosen residual tolerances, increase cycle budgets, weaken refinement/boundary rules, disable dynamic changes, or slow all-fine to obtain a passing ratio.

## Ownership boundaries

Own these files and their pressure-specific tests:

- `lib/methods/uniform/uniform-mixed-pressure-stage.ts`
- `lib/methods/uniform/uniform-mixed-pressure-records.wgsl.ts`
- Related pressure operator, reconstruction, and cycle modules as necessary.
- `tests/helpers/uniform-mixed-pressure.ts`
- `tests/uniform-mixed-pressure-surface-cycle-dawn.test.ts`

Coordinate changes to ownership buffer methods in `uniform-mixed-ownership.ts`: pressure records/worklists live there, but the file is shared infrastructure. Keep field layout and public pressure-stage interfaces stable where possible.

The originating workstream owns bulk momentum, extension, surface transport/geometry, dynamic classification/remap, `uniform-mixed-frame.ts`, `webgpu-uniform-reference.ts`, allocation telemetry, and the whole-frame benchmark. Describe required integration changes instead of editing those concurrently. Leave the grid-overlay work intact.

## Workspace warning

The shared checkout has substantial staged and unstaged work. Many pressure changes predate this handoff. Inspect both `git diff` and `git diff --cached`; do not reset files to HEAD or discard the staged implementation. The experimental all-owner graph has been selectively removed. Preserve unrelated changes and the ownership-managed pressure worklist fix described below.

No independent pressure worker had been started when this document was written. This document does not create a new chat or reserve the GPU.

## Existing pressure design

- Simulation ownership is h/4h. Pressure uses a separate graded root from `uniformMixedPressureLayout`, with pressure-only 2h transitions; level 1 has minimum width 2, level 2 is uniform width 4.
- Keep pressure-only transitions for now. A direct 4:1 pressure experiment failed correctness and was reverted. Do not treat removing the 2h layer as a proven optimization.
- `encodeSurfaceRestriction` builds pressure records once per frame at levels 0 and 1 (`buildRecords`, `linkRecords`).
- Existing interface records use a chunk of `64 + 5632` words per job, with six entries per h row and 24 per 2h/4h row. Scratch ownership includes slopes/RHS/field buffers.
- Sweeps perform two Jacobi updates while freezing interface corrections from the old iterate across both halves. Preserve this update order, boundary/ghost-fluid coefficients, reconstruction, restriction, residual acceptance, and withheld projection on failure.
- Staged work includes liquid-culling dispatches. Preserve its behavior unless a measured replacement is better.

## Removed all-owner graph experiment

The primary workstream has removed its unaccepted all-owner graph from the live checkout, preserving staged pressure work and the ownership-managed worklists. The removed experiment did the following:

1. Prepend 16 u32 words per packed owner to the record buffer. `umGraphRows = counts.x*64 + counts.y*8 + counts.z`; `umRecordBase = umGraphRows*16` shifts seam chunks.
2. Run `buildGraph` after `buildRecords`/`linkRecords` once per frame. Each row contains owner index, diagonal, seam pointer (or regular sentinel), boundary flags, six neighbor addresses, and six coefficients. Neighbor high bits encode air/open handling.
3. Use `smoothGraph` for every owner in each Jacobi half. Regular rows gather cached values; seam rows delegate to existing interface smoothing.
4. Allocate an additional 64 bytes per owner. Old regular/liquid pipelines remain compiled even where bypassed.

This experiment was **not accepted**. It slowed the all-fine control and added 64 bytes per owner. Historical dynamic timings also regressed, though those runs used now-rejected velocity paths and are not acceptance evidence. The live code again uses regular/liquid tile dispatches plus per-frame interface records; `buildGraph`, `smoothGraph`, and the record prefix are absent. Compact seam-only records with regular arithmetic addressing may be preferable to a full graph for all fine cells. The removed patch is preserved locally at `/tmp/fluid-front-regression/removed-pressure-graph.patch`.

The useful ownership fix, separate from the graph experiment: `pressureWorklists()` in `uniform-mixed-ownership.ts` lazily allocates/owns the list and indirect dispatch buffers. `initializeLiquid` borrows those buffers instead of allocating pressure traversal fields through the stage device. Preserve allocation accounting and destruction. This fixes an existing helper proxy that rejects stage-owned traversal allocations.

## Correctness state and first investigation

The original eight surface-pressure layouts (160 subtests) pass with the restored pre-graph pressure path. Added tests include a fine island and a coarse island, each centered in a 12³ domain with anisotropic spacing `[1, 2, 0.5]`; tests currently grade them through `uniformMixedPressureLayout`.

The latest run has 195 passing tests and six failing counts (five subtests plus their parent): coarse-island axis-1 hydrostatic/parasitic-velocity cases for both signs, and a curved sign-negative random case with final residual about 0.00105458 against 0.001 after the existing six-cycle budget. The new fine-island cases pass. The restored pre-graph run reproduces all five failures, including the same curved-case residual. They are not introduced by the all-owner graph. Raw run: `/tmp/fluid-front-regression/restored-pressure-surface-tests.log` (195 passes, five failing subtests plus parent).

The pre-graph comparison is complete. Investigate the coarse-island failures without relaxing fixture thresholds.

Also verify ownership changes invalidate/rebuild accesses, hydrostatic balance, anisotropic spacing, both interface orientations, free surfaces, solids and domain boundaries. Report graph build cost and allocated bytes as well as sweep time.

## Scenes and measurements

The user's long-dam scene is `sparse-cm12-long-dam-break`, run with the **Uniform Geometric** method despite the scene name. Lattice is 192×96×32, h = 0.0125 m, tank 2.4×1.2×0.4 m, closed/free-slip boundaries, 30 Hz, four seconds. Use the default pressure tolerance of 5 for the scene comparison.

The reported fixed region is `91.6667_0_75_100_8.3333_100_4_4` (min XYZ, max XYZ percentages, min/max cell width). It contains 16 coarse owners replacing 1024 fine cells; it is not literally one coarse tile. The separate one-tile control is `minimal-power-dam-break-64` with `87.5_87.5_87.5_93.75_93.75_93.75_4_4`.

Original timestamp-instrumented window 49–72 medians:

| Case | All fine wall | Mixed wall | All fine GPU sum | Mixed GPU sum |
| --- | ---: | ---: | ---: | ---: |
| Mini64 single air tile | 18.748 ms | 23.723 ms | 11.633 ms | 14.877 ms |
| Long dam fixed region | 21.069 ms | 27.473 ms | 13.107 ms | 18.579 ms |

Long-dam V-cycle time increased from 2.785 to 4.653 ms; matching two-cycle frames still showed approximately 2.818 versus 4.686 ms. Repeated mixed accesses matter independently of convergence count.

Before the new graph, regular 4h bulk momentum plus graded pressure completed the dynamic scene but did not meet the target. Window 73–120 wall median was 29.600 ms versus 30.230 ms for the earlier dynamic implementation. These are instrumented diagnostic runs, not final acceptance measurements.

## Commands and GPU isolation

Run from `/Users/petersuggate/code/me/fluid`. Coordinate GPU windows with the originating worker. Never run Dawn concurrently with a browser simulation or another Dawn process. Inspect `/tmp/fluid-webgpu-exclusive.lock/owner.json`; do not delete another process's lease. The user previously closed the browser simulation, but verify isolation before measuring.

Pressure correctness:

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test --test-concurrency=1 tests/uniform-mixed-pressure-surface-cycle-dawn.test.ts
```

Diagnostic dynamic run:

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-uniform-mixed-pressure-dawn.ts --scene=sparse-cm12-long-dam-break --arm=dynamic --steps=120 --tolerance=5
```

Fixed-region diagnostic: use the same command with `--arm=mixed --region=91.6667_0_75_100_8.3333_100_4_4`; all-fine uses `--arm=fine`. The live probe now throws on simulation errors; older frozen bundles can still stop early and exit zero. Verify all 120 rows and absence of error fields.

The originating worker owns final acceptance via `tools/benchmark-uniform-bulk-surface-dawn.ts`: uninstrumented ABBA runs, two samples per arm, 120 frames, windows 9–24/25–48/49–72/73–120, maximum ratio 1.02. It includes accepted completion, classification and remap; renderer is excluded. It has not yet passed. Independent pressure timing is diagnostic, not a replacement for whole-frame acceptance or numerical tests.

If changes extend into Sparse CM12 simulation/topology/presentation/terrain/live editing, repository instructions require `npm run test:dawn:sparse-cm12`; do not weaken lanes or raise ceilings.

## Evidence on this machine

- `docs/benchmarks/uniform-long-dam-small-region-2026-09-27.md` and companion JSON: original investigation.
- `/tmp/fluid-seam-investigation/probe.mjs`: frozen original probe; original fine/mixed logs and `summarize.mjs` are alongside it.
- `/tmp/fluid-bulk-surface/before.mjs`, `before-dynamic.log`: earlier dynamic baseline.
- `/tmp/fluid-bulk-surface/bulk-graded.mjs`, `bulk-graded-dynamic.log`: bulk momentum with original graded pressure.
- `/tmp/fluid-bulk-surface/graph.mjs`, `graph-dynamic.log`, `graph-correctness.log`: current unaccepted graph experiment; bundle predates latest allocation telemetry fixes.
- `direct-pressure*` and `bulk-momentum-dynamic.log` in that directory: rejected direct-4:1 experiments. The latter failed pressure acceptance at frame 34.

Temporary bundles/logs are local evidence, not durable repository artifacts. Existing TypeScript checking reported 14 unrelated Sparse test/tool errors before the graph change; rerun and distinguish new errors.

## Deliverable

Return an implemented pressure path with unchanged public integration contracts where possible; a clear explanation of what is built once per frame and what each iteration reads; correctness results including the new coarse-island failures; isolated build/sweep timings, memory cost, and repeated scene timings. Explicitly identify unresolved failures and any integration changes needed by the bulk/surface worker. Do not claim the overall 2% target from isolated pressure improvements.

## Latest original-reference comparison

The restored original pressure path matches the frozen original solver: all-fine fronts 90/169 and coarse-corner fronts 90/168 at frames 10/20. The test retains ±2h tolerance and unchanged strip-volume limits, with frame-20 reference corrected from the unaccepted graph's 171 to the original 169. Dynamic currently fails frame-10 strip volume (8.981445 < 10). See the front report's original-reference section. No wider band is enabled.

## Current pressure contribution to one-tile cost

Restored-path mini64 pass profiling, frames 9–32, gives median all-fine wall/GPU 19.693/12.812 ms and one-tile 23.966/15.991 ms. Both arms have twenty two-cycle and four three-cycle frames. Median V-cycle cost rises 2.818 → 3.375 ms (+0.557 ms). Other increases include rowsDivide +0.393 ms, momentum +0.295 ms, surface advect +0.197 ms, redistance +0.164 ms; the penalty is not confined to pressure. Separate uninstrumented ABBA windows show ratios 1.228, 1.195, 1.282, 1.171, so the 2% target remains unmet. Durable raw captures: `docs/benchmarks/uniform-original-pressure-one-tile-{2026-09-27,profile-2026-09-27}.json`.

An independent pressure-geometry experiment resolved phi once in pressure ownership and then used the resolved geometry sampler. It preserved geometry bit-for-bit across seam/island fixtures and fixed-scene front measurements, but did not demonstrate whole-frame improvement (ABBA ratios 1.257, 1.251, 1.269, 1.189). Its live integration was removed; the equivalence check remains in the vertex-transfer test.

The complete restored-pressure dynamic front diagnostic now reaches indices 88/174 at frames 10/20, with advancing-strip volumes 8.981445/575.377191. It fails the frame-10 strip lower bound and the frame-20 front bound. These replace the earlier graph-era dynamic reference; no fine-band widening is enabled.

## Independent transport work landed

Primary now enables parallel coarse-row normalization; all seam fixtures match scalar output bit-for-bit and pass the independent conservation oracle. The original pressure files remain untouched. Mini64 whole-frame ratios still range 1.166–1.247; dynamic diagnostic ratios 1.350–1.498 with the original pressure path, and dynamic still fails its front check. A directional dispatch-certificate experiment is under scene validation, without changing the fine band. See the bulk/surface progress report for exact scope and evidence.

Directional dispatch certification passed the fixed-scene front bounds; its one-tile ABBA ratios remain 1.232/1.163/1.214/1.184. Primary keeps it and parallel coarse row normalization enabled. A general h-vertex addressing shortcut showed no timing benefit and was removed. No fine-band widening is enabled. The combined geometric-seam, frame-plan and complete-frame gate passes; dynamic front and new pressure island failures remain unresolved.

Pressure diagnostic update (28 September): the five island subtest failures persist. The four hydrostatic/bounded maxima occur on h/h faces, with speeds 0.00120258/0.00116491, pressure error about 0.005 and residual about 0.00065; they look like remaining long-wave convergence error rather than a large direct seam kick. The curved case ends at 0.00105458. Detailed assertions now report the offending face, widths, phi, pressure error and residual. Tightening the long-dam pressure tolerance from 5 to 0.5 does not reconcile dynamic and fine (frame-20 fronts 167 versus 190); default remains unchanged. See the progress report and `uniform-pressure-tolerance-diagnostic-2026-09-28.json`.

Integration update: primary now fuses regular h and seam-record pressure smoothing launches, including the repeated liquid-only sweeps. The existing liquid tile list is copied once per solve into a record-buffer prefix (16 + tile-count words); all row addresses include that prefix. Small fine tiers already held in records use the previous path. Operator/reconstruction arithmetic and the once-per-solve record build remain unchanged. One-tile diagnostic V-cycle median improves 3.375 → 3.080 ms with identical recorded numerical results. Tests preserve the known surface-island and dynamic-front failures. If editing record addresses, account for `UM_REC_PREFIX`; see the latest progress report.

Integrated fused-sweep one-tile ABBA ratios: 1.217/1.142/1.198/1.172, with conservation passing. The 1.02 target remains unmet. Non-surface pressure cycle checks pass; no Sparse simulation code changed.

### New isolated failing system: adoption before pressure

Primary tested moving dynamic census/adoption between forces and pressure to avoid invalidating a second velocity extension. This variant is **not integrated**. At long-dam frame 6, the first pressure cycle raises residual 61.811619 → 6040.591309 and projection is withheld. A transfer/checkpoint integration error has not yet been ruled out; do not assume this proves an operator defect. The captured equation/state is in `docs/benchmarks/uniform-midstep-pressure-failure-2026-09-28.json.gz`: lattice, simulation tile words, parameters and all three pressure levels' tile words/RHS/phi/minimum/pressure with boundary storage. Counts: 42,312 / 14,984 / 9,216 owners. Ignore inactive halo slots; they can contain stale scratch values. The snapshot is after rejection. Reproduce with `/tmp/fluid-front-regression/build-midstep-adapt-capture.mjs` then its generated bundle, dynamic long dam, 8 steps, tolerance 5, through the normal GPU lease. This gives an independent next case for testing pressure robustness and transfer consistency.

### Independent pressure replay (28 September)

`tools/replay-uniform-pressure-capture-dawn.ts` restores the captured simulation layout, asserts exact pressure tile words/counts at all three levels, restores RHS/phi/minimum, zeros pressure scratch, and runs the production pressure hierarchy without advancing fluid or performing mid-step adoption. On Dawn Metal it gives residuals **61.811619 → 6.830243 → 0.361684**: convergence in two V-cycles at unchanged tolerance 5. Thus the captured equation does not reproduce the mid-step experiment's blowup in isolation. The integration-state mismatch still needs isolation; do not treat the snapshot as proof of intrinsic pressure-operator divergence.

Two separately compiled mid-step controls—clearing pressure/slopes/frozen/residual scratch before RHS, and rebuilding ownership metadata on the CPU after adoption—both preserve the original frame-6 rejection (6040.591309). Neither is integrated. The live fine band and 4h bulk policy remain unchanged. Replay results are in `docs/benchmarks/uniform-pressure-capture-replay-2026-09-28.json`. Typecheck reports the same 14 unrelated Sparse errors, none in the replay tool. The dynamic-front and 2% performance acceptance goals remain unmet.

### Pressure classifier growth defect isolated and fixed

The pre-cycle capture has exactly the same root RHS, phi and minimum as the post-failure capture and again converges independently (61.811619 → 6.830243 → 0.361684). The discrepancy was a cached liquid-list bind group in `UniformMixedPressureLevelStage.bind`: unlike the sweep field bindings, it truncated phi to the owner count active on first binding. Subsequent refinement could move wet owners outside the bound range, omitting them from repeated liquid sweeps. Binding the validated reserved phi view fixes this without allocation, topology changes or band widening.

The new GPU regression `tests/uniform-pressure-liquid-layout-growth-dawn.test.ts` first binds while coarse, refines to h with a wet owner beyond the old range, coarsens and refines again. Before the fix it reports no wet tile instead of one; after the fix it passes. Together with the non-surface cycle suite, all 27 checks pass. Typecheck retains 14 unrelated errors and none in the changed files.

This fixes the scheduling experiment's pressure blowup, but does **not** validate that schedule: it now completes 20 frames with fronts 86/191 at frames 10/20 and toe-strip volumes 158.463617/651.845598. It still fails the original accuracy limits and remains disabled. Only the phi binding fix is integrated. The 2% whole-frame objective remains open.

Live long-dam verification after the capacity fix preserves passing all-fine/fixed-corner checks and the exact existing dynamic failures: frame-10 strip mass 8.9814445388424 and frame-20 front 174 versus 169 ± 2h. Raw result: `/tmp/fluid-front-regression/liquid-capacity-front.log`.

### Direct h/4h pressure experiment and refreshed profile

Fresh post-restriction mini64 profiles: all-fine wall/GPU 19.959/12.812 ms; one-tile 23.347/15.237 ms (frames 9–32). Largest median stage penalties include momentum +0.262 ms, V-cycles +0.197 ms, surface-volume correction +0.164 ms, and several +0.131 ms increases in surface, geometry, authority, projection and extension. Momentum culling is already enabled in the live constructor; this is not an available new optimization.

An isolated pressure variant uses the simulation h/4h layout directly, removes the 2h collar and transfers, and allows 96 entries per 4h pressure record (6/24/96 for h/2h/4h; existing row capacity suffices). It is **not integrated**. The one-tile median wall/GPU is 23.387/15.696 ms: V-cycles rise 3.080 → 3.768 ms with unchanged cycle counts, offsetting geometry/authority/presentation/transfer savings. Dynamic median is 29.869/19.431 ms but fronts 87/166 and strip masses approximately 0/15.048 fail the original bounds.

The complete independent surface-cycle matrix reports 193 passing checks and seven failed subtests (eight failures including parent). Besides the five existing island failures, direct fixture 8 has two slow-converging random cases: axis 0 negative ends at residual 0.002113; axis 1 negative at 0.216169 after six cycles. Increasing record capacity alone does not make direct h/4h pressure acceptable. Raw profiles/snapshots/errors: `uniform-direct-pressure-experiment-2026-09-28.json`; local test log `/tmp/fluid-front-regression/direct-pressure-surface-test.log`. No tolerance, band or live pressure-layout change.

### Mixed-level prolongation consistency diagnostic

Added `tools/probe-uniform-pressure-prolongation-dawn.ts`, a GPU affine-field diagnostic independent of the pressure operator. For a fine island and a coarse island, h→mixed-2h/4h prolongation preserves a constant exactly but makes a **0.25 coordinate-unit error** on each of x/y/z linear fields (64 and 448 interior h cells checked). Example: expected 4.5 at [4,5,5], got 4.25. The sampler weights a 4h center value as though it lived at the requested 2h sample position.

An isolated coarse-gradient reconstruction evaluates that displaced sample at its actual position and makes every affine check exact. It is **not integrated**: in the direct h/4h pressure surface matrix, plain reconstruction has 191 passes and nine failed subtests (plus parent), adding hydrostatic/bounded failures on fixture 8. A second variant uses the existing ghost-fluid theta for air neighbours and also has nine failed subtests. Its worst random fixture remains at residual 0.213870; affine consistency alone does not resolve the direct-interface convergence problem. A binding-alias error in the first ghost-conditioned harness was repaired before collecting these results; invalid zero-output runs are excluded.

The live solver is unchanged by these reconstruction experiments. Typecheck has the same 14 unrelated errors, none in the new diagnostic tool. Durable evidence: `uniform-pressure-prolongation-affine-2026-09-28.json`.

### Direct-interface correction direction and error location

The failing direct fine-island case (fixture 8, vertical axis 1, negative sign, random velocity) reduces residual monotonically over seven cycles: 1.5473 → 1.0449 → 0.7101 → 0.4663 → 0.3516 → 0.2717 → 0.2162. A post-hoc unweighted L2 line minimization along each whole-cycle correction gives scales 2.27–5.02, all above one. Underrelaxation is therefore not supported by this case; even the final optimally rescaled correction leaves infinity residual 0.1031 against 0.001. This analysis does not implement or validate overrelaxation.

The largest final errors are in h cells around the fine-island perimeter (for example physical center [4.5,15,3.75], phi −2.5), with opposite signs on the two x faces. This focuses the next investigation on seam-local smoothing and representation of the coarse correction, rather than a wider surface band or deep bulk work. Live code remains unchanged. Only this one selected subtest was executed; bypassed test bodies are not passing evidence. Durable vectors, locations and method: `docs/benchmarks/uniform-direct-pressure-correction-direction-2026-09-28.json`.

### Seam smoother coefficient experiments

Three isolated direct h/4h variants ran the complete 200-subtest surface matrix with unchanged cycle counts and limits. Refreshing the linked seam correction on both Jacobi halves reduces the worst fine-island random residual from 0.216169 to 0.141345. Also accounting for the composed correction's self-pressure coefficient in the Jacobi diagonal lowers it to 0.0485982. Combining that exact diagonal with the affine-exact prolongation gives 0.0616015, so the transfer reconstruction does not improve this combination. Each variant passes 194 subtests and fails six (plus the parent): the worst fine-island random case and the five existing coarse-island cases. The other previously failing fine-island random case now passes.

None is integrated: the remaining residual is still far above 0.001, and the more expensive gather needs a once-per-frame coefficient build before any timing claim. These results support investigating a composed linear seam operator/block correction, not widening the fine band, changing limits, or adding unexplained global cycles. Full errors and experiment builder/shader source are saved in `docs/benchmarks/uniform-direct-pressure-seam-smoother-2026-09-28.json`.

### Cached seam diagonal prototype

`tools/experiment-uniform-pressure-seam-smoother.mjs` now builds reproducible isolated `fresh`, `diagonal`, and `cached` variants of the full surface-cycle gate. No live source file is rewritten. The cached version computes the self-pressure coefficient in `linkRecords`, using completed base slope records, and stores it for every Jacobi update. It adds no dispatch and 256 bytes per record tile (row storage 5632 → 5696 words). Full readback SHA-256 values match the uncached exact-diagonal variant for all 200 solves, including every cycle's residual vector, final pressure/slopes and coarse state. Both still pass 194 subtests and fail six plus parent. Evidence: `docs/benchmarks/uniform-cached-seam-diagonal-2026-09-28.json`. The prototype remains disabled pending a correction that satisfies accuracy and scene timing; this result establishes only coefficient-build equivalence.

### Residual-subspace prototype passes the direct-interface surface matrix

The cached-diagonal fine-island failure is not missing a useful correction direction: an affine combination of its seven existing iterates predicts infinity residual 0.00001847 versus 0.04859824 before combination. Reapplying that combined pressure through the GPU operator measures 0.00001395 and passes the independent divergence check. Six iterates already predict 0.00012679. These are post-hoc combinations, not extra multigrid cycles.

The reproducible correctness prototype is `tools/experiment-uniform-pressure-subspace.mjs` (build with `cached`, then run its printed bundle with the repository Dawn module). It captures pressure and signed residual histories, computes a twice-reorthogonalized QR solution on the CPU, skips dependent columns, and combines pressures with coefficients summing to one. For bounded cases it captures signed residual separately from the projected measure, clamps the pressure candidate to its minimum, and accepts only when the GPU projected measure improves. Unconstrained candidates also require GPU residual improvement. On the full unchanged surface matrix this passes **all 200 subtests plus parent**. It resolves the direct fine-island random case and the existing coarse-island hydrostatic/curved failures without widening ownership or changing tolerances/cycle counts.

This is **not production integration or a performance result**. It introduces CPU readback/QR, pressure-history copies and additional residual evaluation. The bounded fixtures are hydrostatic, so arbitrary active-set behavior still needs separate coverage. The next implementation step is GPU-resident history/reductions with an explicit memory and dispatch budget, followed by constrained-cycle tests and actual scene progression/timing. Keep the original live path until those checks justify integration. Evidence and per-case GPU acceptance values: `docs/benchmarks/uniform-pressure-subspace-prototype-2026-09-28.json`. Dynamic-front accuracy and the 1.02 frame-time limit remain unmet.

### GPU-resident history and QR implementation

`lib/methods/uniform/uniform-pressure-subspace.ts` now implements pressure/residual history capture, twice-reorthogonalized float32 QR, rank-deficient-column handling, bounded candidate application, and owned resource cleanup. It is **not wired into the live frame**. `tools/experiment-uniform-pressure-subspace-gpu.mjs` reproduces the full surface test with the cached diagonal and this GPU module. All 200 surface subtests plus parent pass. The candidate fitting/application need two dispatches; seven histories require 80 bytes per owner plus 28 bytes. The coefficient kernel currently uses one workgroup, so scene-scale cost is unverified and may require hierarchical QR. Bounds/acceptance remain caller responsibilities; the diagnostic harness still reads acceptance on the CPU.

The original non-surface gate has two hydrostatic CPU-reference comparison failures (23 passes, two subtests plus parent fail). Diagnostic inspection shows GPU spurious velocities 0.000144/0.000324 versus finite-cycle CPU-reference velocities 0.001945/0.007435. A separate run advances only the independent CPU oracle by 32 additional V-cycles, leaving GPU budget and every assertion threshold unchanged: all 25 subtests plus parent then pass. CPU-reference velocities fall to about 4.2e-7, confirming the GPU correction moves toward the stationary solution. Original test source remains unchanged; do not describe its finite-budget reference gate as passing. All eight manufactured constrained cases pass in both runs.

Typecheck retains 14 unrelated Sparse errors and none in the new module. Evidence: `docs/benchmarks/uniform-gpu-pressure-subspace-2026-09-28.json`. Next: measure the QR/application cost at actual owner counts, implement GPU acceptance and persistent capacity-aware resources, then validate the live direct h/4h candidate against dam fronts and whole-frame timings. No band or live path change has been made.

### Hierarchical QR scaling

The single-workgroup prototype is too costly for scene integration: synthetic median fit/application GPU times are 4.063 ms at 42,312 owners, 30.474 ms at 262,144, and 71.631 ms at 589,824. `UniformPressureSubspace` now has an optional hierarchical implementation in `uniform-pressure-subspace-qr.wgsl.ts`: independent 256-row QR blocks, recursive QR of their stacked triangular factors, backsolve, then pressure combination. This avoids squaring conditioning through normal equations. It preserves all 200 surface-pressure subtests plus parent.

The same synthetic benchmark reports hierarchical medians 1.114 / 2.425 / 2.490 ms at those counts, with materially varying late samples at the largest count. Memory at 262,144 owners falls from 20,971,548 to 15,081,500 bytes. These are only fit/application measurements; history capture, residual evaluation, acceptance and the rest of the frame are excluded. Neither strategy is selected by the live frame. Reproduce scaling with `tools/benchmark-uniform-pressure-subspace-dawn.ts --hierarchical`; the surface experiment builder accepts `--hierarchical` after its variant/output/test-entry arguments. Raw samples and scope: `docs/benchmarks/uniform-pressure-subspace-scaling-2026-09-28.json`.

The next integration requirements remain GPU acceptance, capacity-aware history lifetime across relayouts, constrained checks for the hierarchical version, then scene front/timing validation. The two-percent target remains open.

### GPU-only candidate acceptance

Added `UniformPressureSubspaceAcceptance`: checkpoint the original infinity norm, reduce the candidate norm and finite-pressure flag on GPU, then retain or restore the final captured pressure history on GPU. Signed residuals use absolute values; projected measures may be supplied directly. The rejection path restores pressure bit-for-bit. Derived slopes/residuals must be recomputed after commit. The receipt is for diagnostics; it does not require host-side decision-making.

`tests/uniform-pressure-subspace-acceptance-dawn.test.ts` passes eight cases plus parent, including worsening/nonfinite values, a nonfinite pressure with finite residual, signed residuals, repeated resets, and a partial-workgroup tail. The full hierarchical surface prototype now uses this GPU-only acceptance path and still passes all 200 subtests plus parent. Readback occurs only for final assertions. Source: `uniform-pressure-subspace-acceptance.ts`; evidence: `docs/benchmarks/uniform-pressure-subspace-gpu-acceptance-2026-09-28.json`.

This remains a fixed dense-equation prototype. The live solver's boundary storage includes inactive slots; do not feed those directly to this reduction. Integration still requires active equation/boundary handling and a history lifetime tied to each equation/layout. No whole-frame or dynamic-front result is claimed.

### Active boundary storage and ownership changes

`UniformPressureActiveRows` builds storage membership from current GPU ownership in one pass after clearing the mask. It marks physical rows and their incident canonical boundary equations using the live boundary-index ABI. `UniformPressureSubspace` now optionally captures masked histories (inactive slots become zero) and applies pressure only to active rows. GPU acceptance accepts the same mask, so stale/nonfinite inactive storage cannot influence the norm or restoration.

The new `uniform-pressure-active-rows-dawn.test.ts` passes through mixed h/4h → all 4h → all h → mixed h/4h using one 896-slot reservation. It compares GPU membership against independent cell-face enumeration, fills inactive pressure/residual slots with NaNs, verifies finite/accurate candidate application on active rows while inactive slots remain untouched, and verifies that a NaN on an active boundary row rejects and exactly restores pressure. Buffer capacity is checked before each mask build. Evidence: `docs/benchmarks/uniform-pressure-active-rows-2026-09-28.json`.

This addresses stale storage correctness without building persistent complex connectivity. It does not yet integrate live arena offsets, variable cycle-history length, or resource lifetime with the frame. Fitting currently traverses the full reservation (zero inactive rows), so compact/runtime-sized fitting remains a performance consideration. Histories must be captured afresh whenever the equation/layout changes. No scene or whole-frame acceptance claim.

### Shared arena binding support

Subspace capture/application and acceptance now accept either `GPUBuffer` or explicit `GPUBufferBinding` views. `uniform-pressure-subspace-view.ts` validates the reserved count against the supplied range and returns an exact binding range; copy-based capture uses source offsets. Read-only numerical inputs that share an arena with writable pressure use compatible `read_write` WGSL storage declarations (without writing those inputs), because WebGPU forbids mixed read-only/writable usage of one buffer within a pass even with disjoint views.

The new `uniform-pressure-subspace-arena-dawn.test.ts` uses one guarded arena for pressure, residual and minimum. It checks masked hierarchical capture, minimum-clamped candidate pressure, rejected-pressure restoration at the correct offset, unchanged inactive values/minimum/residual, guard words, and rejection of an undersized view. Together with active-boundary and acceptance tests, all 11 tests including parents pass. Evidence: `docs/benchmarks/uniform-pressure-subspace-arena-2026-09-28.json`.

Live integration still needs variable cycle-history length: the scene often accepts after two cycles, while the present fitting prototype assumes seven populated histories. Do not add seven cycles merely to use acceleration. The frame also needs persistent resources and a fresh history lifetime per equation/layout. No live selection or scene timing claim.

### Variable-length history and in-sequence correction

Subspace fitting/application now accepts a populated-history count from two through the reservation size. Immutable aligned uniform records select the prefix; unused columns are zeroed algebraically, and application never reads unpopulated histories. There are no per-candidate parameter writes or unused-history copies. Acceptance can restore the selected last history slot instead of assuming the reservation's final slot.

`uniform-pressure-subspace-prefix-dawn.test.ts` passes for both QR strategies with two through seven histories, NaN-poisoned unused slots, shrinking/reused prefixes, multiple selections in one command encoder, and exact restoration of the selected last iterate. The related acceptance/boundary/arena gate retains all 11 passing checks. Typecheck has the same 14 unrelated errors and none in subspace code.

The full direct h/4h surface prototype also passes all 200 subtests plus parent when correction/acceptance is applied within the original seven-cycle sequence after each cycle from the second onward. Use `--hierarchical --each-cycle` with `tools/experiment-uniform-pressure-subspace-gpu.mjs`. No cycles or tolerances were changed. This is still the fixture's full-cycle-first schedule; the live frame uses V-cycles first and may stop much earlier. Evidence: `docs/benchmarks/uniform-pressure-subspace-prefix-2026-09-28.json`.

For live integration, consider capturing the initial zero-pressure equation as history 0, so the first completed cycle provides two iterates. The implementation supports eight reserved histories, enough for the initial equation plus seven cycles, but this initial-history variant has not yet been validated. Preserve fresh history per RHS/phi/layout and original acceptance thresholds. Scene fronts and whole-frame timing remain open.

### Initial-history validation and isolated scene integration

The surface experiment now supports `--initial-history --v-first` together with `--hierarchical --each-cycle`: capture the zero-pressure equation first, then fit after each completed V-cycle/Full-cycle using the populated prefix. All 200 surface subtests plus parent pass with unchanged thresholds and cycle count.

`tools/experiment-uniform-pressure-subspace-scene.mjs` builds a reproducible isolated scene variant using direct h/4h pressure, cached exact seam diagonal, GPU active-row masks, arena views, eight reserved histories, initial-history fitting, and GPU acceptance. It allocates once and recaptures the equation each frame; acceleration is enabled only when coarse owners exist. The live frame source remains unchanged.

The scene result is **not acceptable**. Dynamic long dam at frames 10/20 gives fronts **86/170** and toe-strip volumes **0/136.095876**, versus required fronts 90/169 ±2 and strips 10–40 / 200–800. Frames 9–20 median wall/GPU are 32.427/21.430 ms. The one-tile mini64 control over frames 9–32 gives **27.599/18.776 ms** wall/GPU, with 21 two-cycle and three three-cycle frames. Earlier current-path diagnostics were 23.347/15.237 ms; all-fine 19.959/12.812 ms. Those comparisons are diagnostic, not a simultaneous ABBA gate, but the candidate clearly does not establish the 1.02 target.

Thus better isolated pressure convergence alone does not recover the dynamic front, and unconditional fitting cost is not recovered on the one-tile scene. Do not integrate this bundle as a fix. Evidence, snapshots and pass timings: `docs/benchmarks/uniform-live-subspace-experiment-2026-09-28.json`. Remaining work must address actual scene surface/bulk behavior and avoid imposing this cost on normally converging one-tile solves.
