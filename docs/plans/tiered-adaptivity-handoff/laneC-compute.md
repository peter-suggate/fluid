# Lane C handoff: compute follows tiers (2026-10-25)

Previous owner: Claude lane C (W2, absorbing W3 stage launches, W9 scale ladder, W11 host construction).
Branch `worktree-agent-a29128b184e85ad73`. Everything is merged to main, the last commit at 406183a8 (main 9785c8fe). There is no work in progress.

## What landed
- **Live lists in the frame head.**
  - The shared helper is `lib/methods/uniform/uniform-live-lists.ts`.
  - Region and key launches, census and frame-plan keys, layout builder keys, sentinel blocks, pageMark, tile context, detail ring, solid region clear, 16h revisions, regional plan, remap/geometry changed lists, builder classify/scatter.
- **Census classify/classifyCoarse** are sized from the ownership's buffered work. The momentum cache runs over held blocks, the extension stamps over held pages, and resident-page launches and arena page grids are sized by receipt evidence.
- **Native seam return** (`pProjectNative`, `uniform-pressure-arena.ts`) is launched over the band's faces, not every held 4h tile: 1.64 → 0.20 ms on the 4×-wide seep.
- **Extension `retainPair`/`finishSnapshot`** grids are sized by live work (`uniform-mixed-extension.ts`).
- **One call site per heavy WGSL helper** in surface/pressure (umSlideWalk, umOpenPointCell, umStoredCellIsOpen, umUnitBoxOpen, …): seep cold compile 333 s → 199 s.
- **Tools:**
  - `tools/profile-solver-construction.ts --dump=<dir>` writes WGSL and pipelines.json.
  - `tools/probe-pipeline-compile.ts` times cold compiles per pipeline.
  - `tools/probe-scale-ladder.ts --timestamps=<step>` gives per-phase GPU ms.
  - `tools/benchmark-region-tier-paired.ts` runs paired A/B timing.
  - `tools/region-tier-command-profile.ts`.

## Measured
- **Milestone C (frame time follows live work, not pool size):** met.

  | Scene | Pool | Memory | Median step |
  |---|---|---|---|
  | Seep 5 mm | 226 | 3.5 GiB | 22.4 ms |
  | Seep 5 mm | 452 | 7.0 GiB | 22.8 ms |
  | Seep 5 mm | 904 | 14 GiB | 20.6 ms |
  | Splash | 20 | | 15.7 ms GPU |
  | Splash | 400 | | 16.0 ms GPU |

- **4×-wide seep** (5 mm, `--width=2 --depth=2`, pool 834) on the current defaults: about 44 ms a step.
  - 65% of held blocks (341 of 528) are submerged, all-COLD liquid with no h tiles.
  - The four 4h pressure rows cost about 11.6 µs per held block, about 4 ms a frame over that bulk.
  - That work is NOT skippable by an exact proof: the system is elliptic, and T changes every frame. Its fix is the 16h tier (lane B).
- **Construction:**
  - Warm (private Metal cache): 6.7–7.9 s. The target was under 15 s.
  - Cold: 199 s summed sequentially; about 110 s wall at 3 in flight.
  - Below 30 s cold is not reachable by call sites alone. 357 of 401 pipelines take under 1 s each, 90 s together (per-pipeline Metal cost).
  - The user has not asked for more here.

## Closed with a verdict (don't redo)
- The `regionVelocity` whole-buffer clear (~160 KB, ~1 µs) stays. Making it live would need clears of regions that leave.
- Extension retention over frozen faces is worth at most ~0.3 ms of the 0.72 ms hierarchy. Deferred.
- Velocity advection growth from base to wide seep follows the surface (h tiles ×1.77, particles ×1.72), not the bulk. Bulk is already skipped via umActiveCoarse.
- classifyCoarse sits at ~1.5 ms flat. Four exact or measurement-only explanations were tested; none accounts for it:
  - shared-sample spread across the workgroup;
  - grid width;
  - the impact scan;
  - face-part bounds.

  The shared-sample kernel is kept for reference only.
- seedScoring "every"'s 2.38 ms on letters is state-driven (more particles and h tiles), not sampling cost.

## Open, for the next compute owner
1. **What remains of coldSurface "warm" cost** once lane A's still-surface criterion is the default.
   - Before it, wide seep warm against budget was +1.7 ms: grid surface advection +1.18, momentum +0.52, classifyCoarse +0.39, velocity advection +0.26.
2. **Cost of the 16h tier** once lane B lands member pages leaving the 4h lists. Re-run the wide-seep ladder and the base/wide per-phase split.
3. **Rest-scene cost** after the still-surface flip. It was 9.4 → 3.1 ms in lane A's test.
4. **`tools/benchmark-nbflip-letters.ts` fails on main** at line 578 ("complete exactly one requested clock step"), with both the default dt and `--dt=0.0166667`. Fix it or retire it.

## Measurement pitfalls
- `probe-scale-ladder` turns layout views on every 5th step. That advances the census policy generation and invalidates deep-COLD replay records. Use `--every=1000` for timing.
- Per-phase timestamps vary up to 2.4× between runs of the same code. Trust only within-run alternation or paired runs (`benchmark-region-tier-paired.ts`).
- Other sessions (e.g. `tools/benchmark-svo-dry-frame-gpu.ts`, the renderer work) may run outside the GPU lock. Check `ps` before believing a timing.
- Warm construction timing needs `FLUID_METAL_SHADER_CACHE=<private dir>`. Two processes sharing one cache dir miss it.
- On 32 GiB the pool-1900 splash rung pages (~29.6 GiB). Don't use rungs that don't fit.
