# Lane B progress

Written by the lane B agent; read by Peter and the coordinator. The format and rules are in `../tiered-adaptivity-handoff-2026-10-25.md` §6. Newest entry first.

## Status
- **Tip:** 87b3ed35 (codex/tiers-lane-b)
- **Doing now:** B0 cell/vertex scratch smoke passed; reducing surface-volume scratch next.
- **Next:** GPU correctness bundle and two-pool allocation audit; compact surface-volume corner scales.
- **Waiting on:** Whole-process GPU lock for two-pool audit and compact-tier repeat gates.

## Log

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
