# Near-settled pool performance: handoff (2026-10-18)

The goal is NB-FLIP pool work that scales with the resolution the physics needs, not with "is anything moving". Peter's directions this round, in order:

- "we want work to scale with the resolution needed to represent realistic physics"; don't lean on whole-frame idle.
- "be bold! all new settings should be enabled in UI by default. Implement!"
- "it's ok to have physics changes. particularly if those options are behind toggles in the UI".
- Faster cadence: "being less pedantic. landing larger chunks of code change before GPU tests". Compare against recorded baselines. Run unit tests once at the end.

Design history and per-phase measurements are in the "Settled melt" and "Near-settled" addenda of [work-proportional-fluid-2026-10-17.md](work-proportional-fluid-2026-10-17.md).

## Where it stands

All runs are Dawn headless, quiet GPU, 1/60 s.

| Scene (probe `SCENARIO`) | e5b7dc1f (before melt) | 7a09d94a | ef484cc6 (now) |
| --- | ---: | ---: | ---: |
| Settled pool (`settled`, all-COLD frames) | 23.9 ms | 5.3 ms | ~4.5 ms median |
| Gentle drift, frames 100–150 (`drift`) | 51.8 ms | 54.8 ms | ~20 ms |
| Gentle drift, frames 6–30 | | 38.3 ms | ~19.6 ms |
| Pool impact (`pool-impact`, frames 6–30) | 47.4 ms | 47.6 ms | ~26.2 ms |

Physics references, which must hold:

- **Pool impact:** impulse ~494–495 N s at frame 30; deepest 1.343 m; momentum error ~1.6e-5; volume drift 0.
- **Settled:** maxSpeed ~2.6e-6.
- **Drift:**
  - the ball floats at y 1.742–1.762 and keeps drifting (vx ~0.04 at frame 150);
  - ripple rms is ~2–3.6 mm;
  - the drip ring is ~43–49 mm max;
  - volume drift is −0.073%.
- **Pool party ball (`settled` run past frame 246):** the ball rolls off the board and enters the water without a pressure halt. It ran to frame 310.

## Commits

| Commit | What |
| --- | --- |
| 4c2765c9 | Settled melt. COLD regions freeze at 16h under a coarse correction term (`UniformColdFreeze`). The host cold-frame skip uses two-frame-stale heat words (`UniformColdFrame`). Census replay. NB idle. Params `coldRegions`, `coldFrameSkip`, `coldCensus`. |
| 7a09d94a | Drift scene `createBackyardPoolDriftScene` ("Backyard pool · gentle drift"). Probe `SCENARIO=drift` reports `encode_ms`, `tiers` and `dispatches`. |
| a6bcb5b8 | Phases 1–4b. Error-based region tiers; WARM at 4h, including WARM cut tiles staying 4h owners; seeding on need. Redistance coarse-metric hole fix. Lane-spreading of latency-bound kernels (buildRhs cut faces, air seeds/clear, depthCoarse, wetRegions, root smoothTiles at 256 lanes, cutsBodies). |
| 46df4546 | `UniformComputeBlit`: frame copies and clears run as batched compute launches, because each compute↔copy switch cost ~40 µs idle on Dawn Metal. Also stops the COLD/WARM relayout flicker: a COLD region keeps its widths under warmResolution "4h". |
| 001bbcf1 | Halt fix (below). |
| ef484cc6 | `bandExit` param. |

## Method params

All are in `lib/methods/uniform/uniform-narrow-band-method.ts` as `SelectParamSpec`s with `update:"solver"`. The near-settled ones resolve through `uniformNearSettled()` in `uniform-near-settled.ts`. Defaults are the new behaviour; the other option is a maintained path reproducing the earlier results.

| Key | Default / other | Effect |
| --- | --- | --- |
| `coldRegions` | `frozen16h` / `fine` | COLD fine velocity and phi freeze; pressure acts there at 16h. |
| `coldFrameSkip` | `on` / `off` | The host skips launches the lagged heat proves empty. |
| `coldCensus` | `replay` / `full` | Deep-COLD regions replay census records. |
| `regionTiering` | `error` / `motion` | Velocity span or travel only holds a region WARM. HOT needs 4× the budget or 2 cells of travel. Only HOT warms its neighbours (a one-region halo). |
| `warmResolution` | `4h` / `h` | WARM tiles leave the h band and take P(U) from the 4h solve. Deep WARM vertices hold phi. WARM cut tiles don't promote beside liquid. |
| `particleSeeding` | `need` / `activity` | Seed on thin, strain, rotation, impact, approach, and shape only within 3 tiles of a body. |
| `bandExit` | `returns` / `strict` | The band stops after a full V-cycle that cut the worst residual by less than 8×, once it is ≤ 0.05 1/s. Encodes one cycle past the last 4 frames' maximum. |

## The halt fixed in 001bbcf1

Peter's app halted when the pool party ball entered the water: "pressure did not converge … layout generation 16".

The cause was in 4c2765c9. The cold-frame skip judged a frame quiet from heat two frames old and skipped forces, extension, momentum and the transfer to simulation. Meanwhile the freeze head thawed the 17 regions that had just left COLD, so they ran live with no gravity and no extension. The root rhs got a g·dt shock on 144 surface rows, the initial residual jumped from 1e-4 to 3.27, and the error cascaded until the halt.

The fix: in a quiet frame, every region holds frozen (`encodeHead(..., quiet)` and `params.y` in `uniform-cold-freeze.ts`).

Lesson: **any stage-skip keyed on lagged evidence must also hold every consumer that the skip starves.** Probe windows must cover the event, so run the ball past frame 246.

Peter's screenshot said frame 41 but the headless run halts at frame 246. The cause matches, but this hasn't been confirmed in the app yet.

## What's left, in suggested order

1. **Seeding triggers versus band noise.** With `bandExit` "returns", late drift frames carry ~30% more particles (14.1k → 18.3k), which eats the band's 0.33 ms GPU saving. This was seen in one trajectory only. Check two runs per arm; if it's systematic, make the strain and rotation triggers (`narrowBandActivityTriggerWGSL` in `uniform-narrow-band-activity.wgsl.ts`) robust to residuals of ~0.05 1/s.
2. **Kernel fusion for HOT regions.** The ball's ~10 HOT regions cost ~10 ms over ~30 kernels, each 0.3–1 ms and latency-bound: V smoothTiles 0.95, bandSweep 0.78, advectDeferred 0.63, remapFaces, buildDistance, cacheNormals, geometry, surfaceHeld. The profile is flat, so a fusion gains ~0.2–0.5 ms each. An earlier single-workgroup fused band sweep was slower beyond ~8 tiles.
3. **The ~582-dispatch floor** on any frame with a HOT region, about 11.6 ms GPU plus ~3.2 ms encode at ~5–8 µs each. Tier-presence encoding doesn't help drift, because it always has HOT regions. What's needed is structural fusion of stage chains, or encoding stages only for the tiers they act on.
4. **Root pressure closed slots:** ~0.5 ms and ~100 dispatches, kept at a floor of 3 for safety.
5. **Remap churn** from transient demotions, ~0.6–0.85 ms in drift. Consider demotion hysteresis, which needs a param if it changes results.

## How to measure

- **Probe:** `WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js SCENARIO=settled|pool-impact|drift STEPS=… FROM=… PROFILE=0|1 OUT=<file> node --import tsx tools/investigate-pool-work.ts`.
  - `SPLIT=1 NB_SPLIT_ALL=1` gives per-dispatch timing.
  - Windows used:
    - drift late `STEPS=150 FROM=100`;
    - early `STEPS=30 FROM=6`;
    - pool-impact `STEPS=30 FROM=6`, and up to 60 for smoke;
    - the pool party ball needs `settled` to ≥ 260 frames to cover water entry.
- **GPU lock:** every Dawn process, including `npm run test:unit`, runs under `lockf /private/tmp/claude-501/fluid-gpu.lock …`. Never run two Dawn processes. Gate timing on `ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*'` < 10%. Discard contended runs, which are ±0.5 ms at best.
- **Baselines:** compare the new arm only, against the recorded numbers above and in the plan addenda. Don't rebuild base trees.
- **Gates:** `npm run check:types` (run it inside a pipeline with `pipefail`, or check its own exit code), one combined Dawn smoke (settled, pool-impact 60, drift 150), one A/B, and `npm run test:unit` once at the end.
- **Scratch tooling.** It lives in the session scratchpad and may be gone:
  - `scratchpad/drift/probe.mts` has RIPPLE metrics and a DIAGBAND band-convergence log;
  - `scratchpad/fix/head/tools/investigate-pool-work.ts` has `STATUS=1` (status record each frame), `FEATURES=` and `INJECT_FRAME`/`INJECT=x,y,z,r`.
  Re-create these hooks in a scratch copy of the probe if needed. Don't commit them.

## Working rules that bit this round

- **Shared worktree.** Other sessions keep uncommitted hunks in `uniform-mixed-frame.ts` (performance getters, `surfaceFlags[2]`), `uniform-narrow-band-flip.ts` (`performanceTileCounts`), `webgpu-uniform-reference.ts` (spatialPerformance), `uniform-pressure-band.ts` (an `empty` getter keyed on the cold-frame ceiling, with a `UNIFORM_REGION_TIER` import), and several other `lib/methods/uniform` files.
- **Commit through a temp index.** Commit only your own hunks: `git read-tree HEAD`, `git apply --cached` on hunk-filtered patches, `write-tree`, type-check the archived tree, `commit-tree -p HEAD`, `update-ref`, then align the real index with `update-index --index-info`. Never stash, checkout or reset. Re-check HEAD right before `commit-tree`, since peers commit too.
- **No compile during simulation.** Pipeline variants are picked at attach.
- **Dispatch rules.** No indirect dispatch, no blocking readbacks, zero per-frame JS allocation.
- **Fail fast.** No CFL or step limits, and fixes add no expensive work.
