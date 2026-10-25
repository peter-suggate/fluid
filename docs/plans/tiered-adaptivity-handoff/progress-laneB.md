# Lane B progress

Written by the lane B agent; read by Peter and the coordinator. The format and rules are in `../tiered-adaptivity-handoff-2026-10-25.md` §6. Newest entry first.

## Status
- **Tip:** 2b6f32be (codex/tiers-lane-b)
- **Doing now:** c628c773 storage tranche ready; independent pressure-cleanup 58c62618 gate bundle queued; B1 actual-frame RHS proof queued.
- **Next:** Complete the same-frame RHS comparison; then replace native work under an explicit compact hierarchy contract.
- **Waiting on:** One focused GPU rerun and the exact58 default bundle queued. Reusing completed full unit evidence; B0 target and B1 hydrostatic speed threshold remain unmet.

## Log

### 2026-10-25 · PROGRESS · independent pressure tranche and actual RHS proof
- Package **58c62618** independently of B1: exact snapshots c628c773 → 58c62618 now run one full default registered bundle, `b0-pressure-default`, volume control off and construction locked. Existing two-pool memory/smoke evidence (958.3 KiB/block), required full-unit evidence, and c628 default/compact bundles are reused. No new full suite or duplicate snapshot.
- Actual-frame diagnostic b2c8a46f compiled successfully; types have only the known two errors. Its initial enable guard incorrectly required optional `pRhs`, which is off: the active legacy root `buildRhs` already precedes selection and solve. **2b6f32be** removes that requirement while retaining strict admission and pre-frame enable; only closed35 is rerun. Failure log is preserved separately as `bulk-actual-rhs-enable-failure.log`.
- Coordinator audit clarified the next B1 contract: current PB correction coefficients restore native64 stiffness even inside members, and pUpB performs 4h sweeps before averaging. Compact PN row equivalence does not prove the current V-cycle equivalent. A parallel CPU restriction/prolongation reference will make the replacement explicit before active list removal.

### 2026-10-25 · MILESTONE · strict flux admission a8428ee0
- Shared GPU certificate test passed 1/1; types have only the known two TS2307 errors. Closed35 retained **552 members** with an exact capacity sum **35,328**, matching 64 per member; no unsupported neighbours, unchanged shadow comparison and zero drift.
- Max speed remains **0.0208358 m/s**, so hydrostatic acceptance still fails as previously recorded. Default selection is unchanged; strict admission is explicitly enabled before frame 1. Current actual-frame RHS diagnostic is a separate uncommitted change, queued for types and one closed35 comparison.
- Durable evidence: `types-strict.log`, `bulk-flux-certificate.log`, `bulk-strict-closed.{log,json}`. Lane C's manual timing hold was cleared immediately after the tiny direct fixture completed; later B checks use normal shared queueing.

### 2026-10-25 · COORDINATION · actual-frame boundary RHS comparison
- Edit `uniform-pressure-arena.ts`, the bulk record tail, and `probe-reservoir-bulk.ts` for an explicit diagnostic enabled before frame 1. Both pipelines/storage are created upfront; default dispatch stays unchanged. Gather boundary flux before projection and retain the full incident-face contribution budget, then compare raw RHS and held-wall lift with native rows from the same solve.
- Strict preparation is committed as a8428ee0. Types have only the known two TS2307 errors and its shared GPU certificate test passed 1/1. Closed35 is running against that immutable loaded source; no full suite was rerun.

### 2026-10-25 · MILESTONE · independent boundary RHS fixture cab45335
- The repaired direct boundary-RHS GPU fixture passed **1/1** with no validation errors. It covers anchor, mean, weighted, cut, moving-body and COLD wall fluxes, including zero/near-zero aperture with nonzero listed flux. All outputs must be finite; tolerances use an independently computed CPU contribution scale, and the GPU scale is separately checked.
- The producer remains unhooked: this establishes boundary gather semantics, not active 16h dispatch or interior cancellation. Strict admission/capacity retention is in its own focused batch, followed by an actual-frame forced-flux comparison before any list removal.
- Exact final log: `artifacts/lane-b-2026-10-25/bulk-rhs-gpu-test.log` in the lane worktree. The tiny fixture released its slot and lane C was notified to resume costs. Reused full-unit evidence; no new full suite.

### 2026-10-25 · COORDINATION · opt-in strict flux selection preparation
- Next edits: `uniform-bulk-tier.ts`, bulk record constants, one bulk-binding call in `uniform-mixed-frame.ts`, and diagnostic activation in `probe-reservoir-bulk.ts`. Allocate a capacity-sum plane upfront and precompile a strict-selection variant; default selection remains unchanged. The diagnostic must enable it before frame 1.
- Strict selection will conservatively reject present-body support within the flux reach using the existing current body-region mask. Without that row-specific moving-wall term, both incident native rows use the same interior source face, cut mask/aperture and COLD addition; exact retained capacity sum handles the volume source. This certificate is separate from the boundary-gather fixture and must match in both A/B arms before active integration.
- Historical W4 notes already show the intended closed-rest gate's unmet speed threshold in both arms (0.0255 initially, about 0.005 after 420 frames). The current step-35 transient is consistent with that history; no new broad baseline run is needed now.

### 2026-10-25 · PROGRESS · B1 shadow proof and failed rest acceptance
- Synthetic GPU row/partition test passed: whole/mixed pages, reversed held list, seam heads, held/bounded walls and unchanged native inputs. Final types have only the two known TS2307 errors. CPU reference fixture strengthening is included as 57f76ec1 (source 9174f782; coordinator's five focused tests and types passed).
- Closed seep at 35 frames exercises **552 members**, **66 whole pages + 50 native/mixed pages = 116 held**, no unsupported neighbours and exact diagonal agreement. RHS absolute error is 503.997 and output-relative error 1.956, but error divided by absolute contributions is **5.028e-7**; the row is cancellation dominated. Durable logs: `bulk-shadow-test.log`, `bulk-shadow-closed.{log,json}` and `types-shadow-final.log`.
- **Hydrostatic acceptance is not met:** drift is zero, but max speed is **0.0208358 m/s**, above 1e-5. Shadow diagnostics execute only after the completed frame and active native dispatch remains unchanged. This is a row-comparison result, not B1 completion.
- The direct RHS GPU fixture's first run failed with zero output: its texture omitted the explicit 3D dimension. The fixture is repaired and now asserts validation errors before arithmetic comparisons; a focused rerun is queued. No full suite rerun.

### 2026-10-25 · MILESTONE · pressure-cut allocation audit 58c62618
- Two-pool audit confirms **958.3 KiB/block**: 226 blocks use 779.439 MiB, 400 use 942.281 MiB. This is another 85.309 KiB/block below c628c773, about 39% below the original 1,570 KiB/block, still above B0's target. Both three-step smokes have no fatal/errors, band peak 241 and metric peak 865. Durable `pressure-smoke-{226,400}.log` and `perblock-pressure.txt` are in the lane artifact directory.
- This allocation/smoke result does not extend the c628 readiness claim to 58c62618. Shadow equivalence and direct boundary-RHS GPU tests remain queued as one focused batch; no full suite rerun.
- The direct RHS integration has a new explicit proof obligation: near-unit membership alone does not prove common interior flux when moving-body cut terms use row-specific capacities. Active integration must exclude that support or prove/retain its interior correction. The unhooked synthetic fixture checks boundary gather, not membership cancellation.

### 2026-10-25 · READY TO MERGE · bounded B0 tranche c628c773
- **Merge candidate: c628c773**, on codex/tiers-lane-b. Compact NB scratch, surface-volume scales and low-wall extension slots reduce measured marginal storage from about 1,570 to **1,043.6 KiB/block**. B0's well-under-0.5-MiB target remains unmet; this is a partial storage tranche.
- Both full registered bundles passed against c8b25c39, volume control off: default splash 70 × 2 / seep 60 × 3 / letters 60 × 3, worst ratios **0.310 / 0.227 / 0.301**; identical-arm `start:"4h", bricks:512` bundle worst ratios **0.158 / 0.589 / 0.380**. Default and compact addressing paths are covered.
- Required unit evidence: **1,224 passed, 134 skipped, 0 failed** in `unit-scale.log`; final types have only the two permitted TS2307 symlink errors. The strengthened missing-page/low-wall GPU address test passed separately, as did NB and surface-volume slot coverage. The full suite began before c628's final test-file addition; the final runtime modules and that added GPU test were checked as recorded below.
- Durable evidence: `/Users/petersuggate/.codex/worktrees/tiers-lane-b/fluid/artifacts/lane-b-2026-10-25/`, including `bundle-final-default.log`, both `b0-final-{default,4h}-{splash,seep,letters}.log` bundles, `perblock-wall.txt` and focused/unit/type logs. The added map scan's paired cost is still unmeasured; this claim is storage/correctness only.
- **Excluded from this readiness claim:** newer pressure cleanup 58c62618 and B1 shadow/direct-RHS preparation. They have separate pending GPU evidence. No active 16h scheduling/storage saving or B1/B2/B3 completion is claimed. Lane B has not merged anything.

### 2026-10-25 · PROGRESS · direct RHS preparation
- Incorporated the coordinator's independent boundary-flux CPU reference as 9bc89717 (source 59dbf244, five focused tests passed). It compares 64 native rows with 96 boundary subfaces, including COLD terms, volume sources and held/bounded wall handling. This is reference evidence, not an active solver change.
- Drafted an unhooked lean GPU producer and a focused fixture for anchor/mean/weighted/cut/moving-wall paths. The coordinator caught a missing near-zero wall-aperture guard; it is fixed with zero/near-zero aperture cases carrying nonzero flux. It will be committed after its queued focused GPU check.
- Per Peter's steering, reuse the completed full unit evidence and run focused affected tests plus relevant types/GPU checks for incremental B1 work. No new full unit suite is queued by lane B.

### 2026-10-25 · PROGRESS · pressure cleanup 58c62618
- Frozen 226-block audit confirms 779.439 MiB live versus 798.267 MiB at c628c773: 18.828 MiB saved, exactly 85.309 KiB/block. The three-step seep smoke has no fatal/errors; fine tiles 54 → 63 → 84, band peak 241/16,384 and metric peak 865/16,384. Volume drift is 3.730e-6. The second pool size remains queued, so the final marginal table is pending.
- Default c628c773 splash gate passed (70 × 2, worst ratio 0.310); default seep and letters remain queued. The full compact-tier bundle already passed. Later pressure cleanup and shadow diagnostics still need their own GPU evidence.
- Confirmed another future dead-allocation candidate: the 14.1 KiB/block compatibility scratch has no shader access beyond its declaration. Keeping the current validation snapshots fixed; no extra storage edit yet.

### 2026-10-25 · MILESTONE · combined compact-tier candidate c628c773
- Full compact-tier bundle passed: splash 70 × 2, seep 60 × 3 (worst ratio 0.589, bandTiles), letters 60 × 3 (0.380, particles). All use volume control off and identical `start:"4h", bricks:512` in both arms. Logs: `/private/tmp/fluid-gates/gates/b0-final-4h-{splash,seep,letters}.log`; bundle verdict: `/private/tmp/fluid-laneB/bundle-final-4h.log`.
- This validates the three combined storage cuts measured at 1,043.6 KiB/block. The default bundle is now queued for this exact immutable candidate. Broad READY remains pending that bundle; the newer pressure cleanup and B1 diagnostics are separate commits with pending GPU evidence.

### 2026-10-25 · PROGRESS · tip 6c5336dc
- Shadow row diagnostics now also report error divided by the sum of absolute RHS/link contributions. This complements absolute/output-relative error when the true row nearly cancels. The synthetic GPU assertion uses a floating-point contribution budget. No active solver arithmetic or dispatch changed.
- Last full unit pass is at a2b1858b; final type/GPU diagnostic checks are queued for 6c5336dc. c628c773 compact-tier letters has acquired the GPU after the external campaigns; default gates remain next in the bundle sequence.

### 2026-10-25 · PROGRESS · tip a2b1858b
- Required full unit suite completed after the pressure cleanup and shadow implementation: **1,224 passed, 135 skipped, 0 failed** (`/private/tmp/fluid-laneB/unit-pressure.log`). The newly added shadow GPU test is a separate queued check. Full suite wall time includes intentional pauses during exclusive GPU timing.
- Combined storage candidate c628c773 passed compact-tier splash and seep (seep worst ratio 0.589, bandTiles); letters and default gates remain queued. Frozen58 pressure-cut marginal audit and B1 synthetic/closed-scene diagnostics remain pending. No broad READY claim yet.

### 2026-10-25 · MILESTONE · compact-tier bins candidate 87b3ed35
- All three registered scenes passed with identical `start:"4h"` tiers and volume control disabled: splash 70 × 2 (worst ratio 0.473), seep 60 × 3 (0.764), letters 60 × 3 with identical bricks=512 (0.261). Letters' earlier bricks=256 run failed baseline capacity and is retained separately; the successful rerun keeps compact storage active.
- Full candidate unit check: 1,221 passed, 133 skipped, 0 failed; types have only the two permitted TS2307 errors. This is a scoped compact-tier PASS, not broad READY TO MERGE: the default/dense fallback still needs its bundle. The newer combined candidate c628c773 is currently completing compact-tier gates, followed by default gates.
- Gate logs: `/private/tmp/fluid-gates/gates/b0-band-slots-4h-splash.log`, `b0-band-slots-4h-seep.log`, `b0-band-slots-4h-letters-bricks512.log`. Full unit log: `/private/tmp/fluid-laneB/unit-final.log`. B0's overall memory target remains unmet.

### 2026-10-25 · PROGRESS · tip a2b1858b
- `58c62618` removes the unused legacy pressure hierarchy and four dead root planes (expected 85.3 KiB/block saved). The coordinator reviewed the live root span; no pressure arithmetic changed. Frozen snapshot audits are queued. B0 remains above its target even if this expected saving is confirmed.
- `a2b1858b` adds explicit `readBulkShadow()` diagnostics: exact held-page partition into whole-member/native lists, 96-boundary-link member row assembly and comparison with the 64-row native sum. Native start pressures are read from preserved PN_B, allowing the check after a completed frame. The normal frame does not dispatch this work. The root RHS is still aggregated from 64 native cells; active 16h scheduling and compact storage remain unfinished.
- A synthetic GPU test covers whole/mixed pages, reversed held-list order, native/member seam slopes and mixed held/bounded wall slots; it verifies the diagnostic leaves native rows, membership and seam heads unchanged. This test and a 35-frame closed seep shadow probe are queued after the pressure-cut audit. No B1 equivalence pass is claimed yet.
- Type check has only the permitted two TS2307 errors. Full unit suite is paused during another user's exclusive GPU timing; its resume-and-wait now holds a check slot, preventing overlap with later timing jobs. Current GPU gates still validate frozen c628c773, not the newer commits.

### 2026-10-25 · COORDINATION · tip 58c62618
- Dead pressure allocations are removed in a separate commit. Checks/audits are queued against the frozen candidate; the CPU suite is paused during another user's exclusive GPU timing job.
- Next B1 preparation edits `uniform-pressure-arena.ts`, bulk record layout constants, a new shadow WGSL helper and the bulk diagnostic probe. An explicit diagnostic dispatch will partition held pages into whole-member versus native/mixed lists and compare boundary-assembled member pressure rows against existing row sums. Active solve dispatch remains unchanged until this proves equivalent. Root RHS aggregation remains native initially; no completed 16h scheduling or storage milestone is claimed.

### 2026-10-25 · COORDINATION · tip c628c773
- Coordinator found an unused legacy pressure hierarchy and four unused root planes. Next edit touches `uniform-scratch-arena.ts`, `uniform-mixed-pressure-memory.ts`, `uniform-mixed-frame.ts`, and the mixed-frame construction in `webgpu-uniform-reference.ts`. Live root pressure, RHS, minimum and authority workspace remain; no kernel arithmetic changes. Expected saving is about 85.3 KiB/block.
- Combined two-pool audit is 1,043.6 KiB/block (226 blocks: 798.267 MiB; 400 blocks: 975.605 MiB), about 33.5% below the original estimate but still above the B0 target. Final full unit suite: 1,224 passed, 134 skipped, 0 failed; types have only the two permitted TS2307 errors. Missing-page low-wall GPU address test passed.
- Corrected bins-only letters gate passed all 60 frames × 3 repeats (largest ratio 0.261, occupancy); prior splash and seep gates passed. Final c628c773 compact/default bundles remain in progress against immutable snapshots. No claim of completed B0 or GPU 16h scheduling.
- After the dead allocation removal, B1 will first add a shadow compact-row assembler and page-list partition, preserving active pressure dispatch until equivalence is demonstrated.

### 2026-10-25 · PROGRESS · tip c628c773
- Three cuts are committed: compact NB cell/vertex scratch; compact surface-volume scales; h low-wall faces reuse fine seam slots. Runtime is fixed for validation. The first wall-address GPU test passed; the strengthened test adds a second physical block beyond an absent page, including its low wall at world x=64.
- Combined 226-block smoke passed: 798.267 MiB live, stage scratch 17.0 MiB and held extension 4.2 MiB. Before the last two cuts those were 819.891, 36.0 and 6.9 MiB. Step counters and volume drift match the earlier three-step smoke; no fatal/errors. The 400-block combined audit is queued.
- Bins-only splash and seep repeat gates passed (worst ratios 0.473 and 0.764). Compact-tier letters stopped on **baseline fine-budget exhaustion**: 2,056 required h tiles versus 2,048 available at bricks=256. This is not a completed A/B metric gate. Preserve the failed log; rerun both arms identically with bricks=512 (4,096 h tiles), which keeps compact storage active on this fixture.
- The bins letters rerun and final candidate's compact-tier/default bundles are queued under construction locks, all with volume control off. No READY TO MERGE yet. Candidate snapshot uses an APFS clone plus a Git-verified patch to avoid duplicating 1 GiB of static assets.
- Full unit suite was paused for A/C paired timings and has now resumed against the final uniform-module tree; final type check is running. B1 CPU reference is included, but GPU 16h implementation and the B0 target remain unfinished.

### 2026-10-25 · COORDINATION · tip ef34593e
- Next B0 cut edits `uniform-mixed-extension-slots.wgsl.ts` and `uniform-mixed-extension.ts`: reuse fine tiles' existing 48 negative-seam slots for h low-wall faces, eliminating the dense h wall-plane reservation. Coarse low-wall faces keep their current slots. All sweeps and arithmetic remain the same.
- Measured bins candidate: 1,171.1 KiB/block total versus about 1,570 before; bins 97.7 versus 497 KiB/block. Pool 226: 819.891 MiB, pool 400: 1,018.895 MiB. Normalized texture-family table: `/private/tmp/fluid-laneB/perblock.txt`.
- Compact-tier registered splash passed (70 steps × 2 repeats, worst ratio 0.473). Seep and letters remain queued. Surface-volume GPU slot test passed; its 3-step seep smoke passed with unchanged observed counters/drift and 811.509 MiB live, stage scratch 27.6 MiB instead of 36.0. Extension slots are now the largest consumer of that shared scratch.

### 2026-10-25 · PROGRESS · tip 30ace2e5
- Added compact surface-volume corner scales: 729 canonical words per block plus 125 words per bounded fine-neighbour slot. Each slot is determined by a fine-owner index and octant; no dynamic allocation or new buffer binding. The seam resolver omits only coarse interior outputs that no owner reads; its reconstruction lattice still reads every canonical corner.
- GPU coverage test spans all 71,874 physical vertex keys of two blocks, including block skirts and slot recycling. It and a short full-solver smoke follow the pending two-pool audit. The eight-neighbour tile-map scan adds work to clearBand; paired costs still need measurement.
- Bins candidate `87b3ed35` passed the full required unit suite: 1,221 passed, 133 skipped, 0 failed (`/private/tmp/fluid-laneB/unit-final.log`). Surface-volume type check has only the two permitted symlink TS2307 errors (`/private/tmp/fluid-laneB/types-scale-final.log`). New combined unit suite is paused for A/C timing and will resume afterward.
- Incorporated coordinator's compact 16h CPU row reference (`4e73e33e`, cherry-picked as `4f1ceacb`): direct 96-subface rows match the constrained 64-cell reference, including symmetry, hydrostatic rest and interface flux. Coordinator validation: focused 8/8; full 1,224 passed, 131 skipped, 0 failed. This is B1 preparation only; no GPU 16h work or storage saving is claimed.
- The B0 target below 0.5 MiB/block remains unmet; no READY TO MERGE claim until the registered gates finish.

### 2026-10-25 · COORDINATION · tip 87b3ed35
- Next B0 cut edits `uniform-mixed-surface-volume.ts` and only its constructor call in `uniform-mixed-frame.ts`: replace the dense 33³ corner-scale scratch plane with canonical 9³ coarse corners and bounded fine-neighbour slots. No scalar volume arithmetic changes.
- The first compact-bin runtime smoke passed three seep steps: 819.891 MiB live; band slots 241/16384 peak, metric slots 865/16384 peak, no fatal, volume drift 3.39077e-6. This is an early smoke, not mature-particle or registered-gate evidence. Log: `/private/tmp/fluid-laneB/metric-smoke-226.log`.
- Candidate `87b3ed35` includes the construction-lock gate harness. Two-pool allocation audit and full `start:"4h"`, volume-control-off repeat bundle are queued safely; final full unit suite is running with concurrency 2. No READY TO MERGE claim yet.

### 2026-10-25 · PROGRESS · tip 9ae05441 + vertex-compaction work in progress
- DEPTH_A, surfaceHot and the packed XY scratch now have compact addressing as well. Canonical 4h corners retain 729 words per block; noncanonical physical vertices use 125-word band slots. Physical block skirts retain distinct identities.
- Metric maps rebuild before reconstruction and after arena search, after the preceding field expires. Domain includes fine tiles, distance tiles and particle support, with the positive-neighbour collar needed by cell-below vertex ownership. Missing planned storage is fatal 13; exhausted metric slots are fatal 12. GPU smoke will test this closure before landing.
- Focused GPU checks: 4/4 pass, including all 71,874 physical vertices of two blocks (no aliases), crossing-slot recycling/overflow/canary, nearest-search arithmetic and fused lists. Log: `/tmp/laneB-nearest-gpu.log`.
- Full unit suite after first cut: 1,221 passed, 131 skipped, 0 failed (`/tmp/laneB-unit.log`). Vertex-stage type check has only the two permitted symlink TS2307 errors (`/tmp/laneB-types-metric.log`).
- The old dense letters `--marks` and `--band-readers` modes now fail explicitly; their host mirrors require a separate physical-key migration. Surface-splat diagnostics use the actual bin layout.

### 2026-10-25 · PROGRESS · tip 5c109346
- Implemented a tile-to-band-slot map for nearest-crossing bank B. Fixed upfront capacity is eight times the h-tile budget, floor 4096, bounded by all tile keys. Full-domain capacity keeps identity addressing without allocating a map.
- Physical nearest-cell values and nearest-search arithmetic are unchanged. Overflow is fatal 11 with the attempted tile count; receipt word 25 records high-water usage. Probe output reports bandCap/bandPeak.
- Added gate `--tiers=JSON` so both arms can exercise the compact `start:"4h"` path; the default bundle alone would not test it.
- Focused GPU remap/overflow/canary test passed (`/tmp/laneB-slots-gpu.log`). Type check found only the two permitted `app/app/*-lab` TS2307 errors. Full unit run is in progress.
- Expected marginal saving is 126 KiB/block; this is a first cut, not the full B0 acceptance target. DEPTH_A and surfaceHot still use physical vertex keys and must preserve the canonical block skirts and reconstruction/metric lifetimes.

### 2026-10-25 · COORDINATION · tip c8b25c39
- Worktree: `/Users/petersuggate/.codex/worktrees/tiers-lane-b/fluid`.
- Editing `uniform-narrow-band-flip.ts` constructor bin layout, WGSL constants and receipt fatal reporting; `uniform-particle-brick.wgsl.ts` nearest-crossing address only. Also membership, nearest, redistance and surface-brick-arena WGSL addressing. No particle transport, seeding or physics changes.
- First storage cut: map each search tile to its compact DEPTH_B slot when depthLists builds the existing list, with a fixed upfront budget and fatal overflow. Preserve physical nearest-cell keys and all nearest-search arithmetic.
- DEPTH_A and surface-hot vertex storage have different lifetimes and canonical block-edge addressing; they follow after the first cut is gated.

### 2026-10-25 · HANDOFF
- The lane was handed over from the Claude lane B agent. The previous owner's note is `laneB-*.md` in this directory.
