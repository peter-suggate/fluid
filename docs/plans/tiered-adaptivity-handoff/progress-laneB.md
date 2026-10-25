# Lane B progress

Written by the lane B agent; read by Peter and the coordinator. The format and rules are in `../tiered-adaptivity-handoff-2026-10-25.md` §6. Newest entry first.

## Status
- **Tip:** 6c5336dc (codex/tiers-lane-b)
- **Doing now:** B0 pressure cleanup committed; B1 shadow row/partition diagnostics committed, validation queued.
- **Next:** Validate final B0 allocations/gates and B1 shadow equivalence before changing active dispatch.
- **Waiting on:** GPU queue. Required unit suite passed (1,224 / 135 skipped / 0 failed); B0 target remains unmet.

## Log

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
