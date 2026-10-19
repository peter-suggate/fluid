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


## Implementation measurements (2026-10-18)

**The requested ~20 ms → ~10 ms gentle-drift target is not achieved.** Timings
below are provisional: the GPU remained shared, and Peter explicitly asked to
continue and report provisional numbers. Paired arms use the same frozen source,
alternate execution order each frame, and compile before the measured window.
They include encoding and GPU completion; diagnostic reads are outside timing.
A source hash verifies that the copied implementation stayed fixed during a run.
Other sessions' work in the shared checkout is not attributed to this change.

The retained implementation adds `bandSmoothing` (`fused` by default, `separate`
as the UI comparison, `update:"solver"`). A small band's 2h descent/ascent visits
keep their coefficients in registers and corrections in workgroup memory. Each
visit uses one launch, saving five launches per V-cycle with the current schedule.
Larger observed bands keep the parallel colour launches. Unexpected growth past
the lagged count uses a complete, synchronized storage fallback in the one group;
no row or iteration is dropped. This adds no GPU buffers or production readbacks.

`tools/benchmark-near-settled.ts` provides repeatable paired runs and records every
frame's body state, particle count and pressure residuals, plus final volume drift
and census trigger counts. `tools/investigate-pool-work.ts` accepts `METHOD_VALUES`
as JSON and includes those overrides in its output.

| Experiment, drift frames 100–150 | Paired median saving | Disposition |
| --- | ---: | --- |
| 2h smoothing fusion | 0.036 ms | Retained; exact trajectory, negligible total-frame gain |
| h + 2h smoothing fusion | −0.042 ms | h fusion removed |
| h + 2h fusion and reserve 1024 → 256 | 0.039 ms | Compact reserve removed; no convincing gain |
| Detail hold 0 → 8 frames | 0.001 ms | Default unchanged; identical trajectory |
| Wider surface authority + deferred advection | −0.821 ms | Removed |
| Wider surface authority alone | −0.651 ms | Removed |

The band-exit study ran twice per arm. Strict versus returns produced exactly the
same respective trajectories in both repeats. Mean late-frame particle counts
were 16,686.6 versus 20,255.2 (+21.4%); paired median savings were 0.204 and 0.172 ms.
Final speed maxima were 0.436 versus 0.778 m/s. The final admitted census criteria
were shape/thin for strict and shape for returns, with zero strain/rotation in
both. This final snapshot does not establish the earlier cause of the extra
particles, so strain/rotation thresholds were not changed on that evidence.

The diagnostic split profile still shows distributed costs: root V smoothing
0.910 ms, deferred advection 0.611 ms, remap faces 0.577 ms, root Full-cycle smoothing
0.501 ms, surface authority 0.372 ms, root RHS 0.336 ms and distance construction
0.299 ms. That profile includes the subsequently removed h fusion and smaller
reserve, and its 29.4 ms wall median includes diagnostic pass-splitting overhead.
It must not be compared directly with unprofiled frame timings.

Final retained-implementation source hash:
`a6c06f556762da86c7f92b0deac6f8485e16bd3c79315babcb322c3755e86174`.
Two final drift repeats measured separate/fused medians of **25.341/25.062 ms**
and **21.112/21.018 ms**. Their paired median savings were **−0.005 ms** and
**0.164 ms**; contention and outliers prevent claiming a reliable speedup.
Every recorded body position/velocity, particle count, band count/cycle/residual
and root residual matched between the two arms in both repeats. At frame 150,
y was 1.759018 m, vx 0.039613 m/s and reported volume drift −0.073033%.

Validation:

- `npm run check:types`: passed.
- `npm run test:unit`: 969 passed, 58 skipped, zero failures.
- Dawn `tests/uniform-pressure-band-middle.test.ts`: passed bit-for-bit for
  descent/ascent, non-power-of-two diagonals, inactive rows, missing neighbours,
  and unexpectedly oversized lists (1, 8, 9 and 130 slots with 64 lanes).
- Gentle drift: 150 frames, two final repeats per arm.
- Pool impact: both arms completed 60 frames with zero reported volume drift.
  A separate/separate control also diverged at frame 6: this impact trajectory
  is already nondeterministic. The fused arm matched the second separate control
  at every recorded frame. Deepest y ranged from 1.350211 to 1.350252 m across
  these four trajectories, versus the older handoff's 1.343 m. The earlier smoke
  measured 493.08 N s at frame 30 and momentum error 1.56e-5 N s. The old depth
  reference is therefore not reproduced by the frozen shared checkout either.
- Pool-party water entry: the retained implementation reached frame 310 without
  a pressure halt; reported volume drift was zero. The quiet frames 61–120 had
  a provisional median of 7.04 ms. Quiet max speed and drift ripple/drip-ring
  field metrics were not remeasured separately in this pass.

Raw reports are retained locally under `artifacts/near-settled/2026-10-18/`
(ignored artifacts). The new benchmark defaults to the separate/fused comparison:

```sh
lockf /private/tmp/claude-501/fluid-gpu.lock env \
  WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
  REPEATS=2 STEPS=150 FROM=100 OUT=/tmp/near-settled-paired.json \
  node --import tsx tools/benchmark-near-settled.ts
```

The 10 ms target, broader HOT-stage fusion, root closed-slot cost and the cause
of the extra seeding remain open. Root safety floors and seeding thresholds are
unchanged; the rejected experiments were removed rather than enabled by default.

## Structural fusion follow-through (2026-10-18)

The follow-up explicitly targets HOT-stage fusion and the dispatch floor. The
10 ms target remains the acceptance goal; the intermediate measurements below
are **not** a claim that it has been reached. GPU timings remain provisional.

Retained implementation:

- `rootReserve: fused / parallel`: an unexpected safety slot executes its entire
  V- or Full-cycle in one workgroup. Storage barriers order its global Jacobi
  iterations. Expected work keeps the parallel traversal; the safety floor,
  projected residual checks and fail-fast publication gate are unchanged.
- `rootTiles: page / octant`: a whole root page shares one radius-three halo
  across its eight octants. This keeps all nine smoothing iterations and their
  original global boundaries. The page kernel uses the device's supported coarse
  workgroup width and at most 64 direct workgroups, striding the current GPU list.
- `rootCoarse: fused / tiled`: the 16h and 32h levels share a dispatch. A current
  compact resident-block list avoids visiting free capacity. Visits fitting two
  rows per lane cache coefficients in registers and iterates in workgroup memory;
  larger current sets use a complete storage traversal.
- `surfaceSplat: fused / separate`: occupied-cell selection and surface scatter
  share a launch, using the particle sort's current occupied-tile list. This
  removes the full-lattice cell scan and temporary list append.

The new choices default to the first option and restart the solver when changed.
The existing root job buffer gains a compact block list and inverse map
(`2 * blocks + 1` words). There are no new production readbacks, indirect launches,
per-frame JavaScript arrays, or shader compilation during simulation.

Intermediate experiments:

| Experiment | Observation | Decision |
| --- | --- | --- |
| Local block-Jacobi root visits | Initialization rejected a non-improving cycle: 1.557 → 2.012 residual | Removed |
| Three global iterations per visit | Initialization rejected 1.557 → 1.981 | Rejected in scratch copy |
| Every root traversal in one workgroup | Paired median **4.113 ms slower**; 20.868 → 25.043 ms medians | Only safety slots use this path |
| Fused safety slots alone | 0.402 ms paired median saving; identical recorded drift states | Retained |
| Safety slots, deep levels and occupied scatter | 0.543 ms paired median saving | Continued with wider page fusion |
| Whole-page visits, 256 lanes, plus fused air stencil | 0.553 ms paired median saving | Widened the page workgroup |
| Whole-page visits, 1024 lanes / 64 groups, plus fused air stencil | 0.777 ms paired median saving; 20.572 → 19.932 ms medians | Retained wider page path; removed air stencil |

A frozen frame-150 stage comparison alternated 24 runs of each path. Occupied
surface scatter measured approximately **0.240 → 0.035 ms** mean GPU time. The
four-stage air dilation/gate measured **0.060 → 0.096 ms** when fused: its exact
halo algorithm passed a missing-page test but regressed, so it was removed.
These small stage timestamps are quantized and contended; use them as directional
measurements, not precise independent speed guarantees.

The deep-root experiment that fused 8h as well recorded 415 dispatches and
17.115 ms summed GPU pass time in an instrumented early-drift profile. Its broader
single-workgroup level was not selected. This is diagnostic evidence that fewer
launches alone do not halve the frame: the instrumented timings and that variant
must not be presented as the final implementation's unprofiled frame time.

GPU correctness fixtures compare page smoothing with the original octant
traversal on an explicitly initialized bounded pressure problem, including wall
rows and absent pages. The page V- and Full-cycle results were bit-identical.
The safety traversal differed by at most 4.66e-9 in pressure and 1.61e-6 in the
projected residual in that fixture; closed slots left mixed fields untouched.
The occupied scatter fixture is exact for cold cells, long particle runs and
launch widths of 1, 3 and 1024 workgroups. Earlier replay attempts used pressure
scratch after its frame lifetime; the final fixture initializes its own problem.

The 256-lane page version completed drift 150, impact 60 and settled 310 without
pressure failure. Impact reached y=1.350906 m, with maximum momentum error
1.33e-5 N s and zero reported volume drift. The pool-party body entered the water
and continued through frame 310. These checks precede the final wider page and
compact shared-memory changes; final verification is recorded below.

Final small-pool capture before switching to the splash regression used source
`2a0628b92c369611e9ad5def495ef5f2114093940858ca4edd7a64a5ae71648b`.
Two paired drift repeats measured old/new medians **20.784/19.886 ms** and
**20.742/19.924 ms**, with paired median savings **0.934/0.814 ms**. The respective
trajectories repeated exactly between repeats. The candidate ended at body
`y=1.756260`, `vx=0.034094`, band residual `0.046857`, root residual `0.000210`,
and reported volume drift `−0.073033%`. This does **not** meet the 10 ms target.

The retained version passed its three focused GPU fixtures (page/reserve,
small-band 2h, and occupied scatter), impact 60, and pool-party water entry
through frame 310. Impact deepest y was 1.350784 m and maximum momentum error
1.33e-5 N s. Both smokes reported zero volume drift; that telemetry is the
last scheduled phi measurement and may be stale, not an independent final mass
measurement. Type checking passed. The end-of-work unit suite is still pending.

Further scratch-only experiments were rejected or left unpromoted: shared-memory
8h red-black smoothing did not improve total time; local coarse retirement
expanded the final drift band from 108 to 352 tiles and cost about 5.9 ms; a
single-dispatch h-band reserve cycle kept the drift trajectory but its two
heavily contended comparisons disagreed (+1.10/−2.45 ms saved), and its live-cycle
fixture did not yet produce a valid comparison. None of those three experiments
is in the production checkout.

Peter then redirected the work to **`nbflip-figure-8-letters` at 2.133 s, under
50 ms without shortcuts**. A first 60 Hz capture around frame 128 found that
the small-pool fusion defaults regress Letters by about 1.4 ms, chiefly occupied
surface scatter. That path is being reworked to select a tile's occupied cells
in parallel. The larger historical regression is being compared against the
pre-paging revision `854c7193` and pre-retention code; provisional results and
source snapshots live under `/tmp/letters-regression` and `/tmp/fluid-letters-*`.
The small-pool measurements above must not be generalized to splash-heavy work.

### Letters regression investigation (in progress)

Use the authored `nbflip-figure-8-letters` scene with `timeStep: sixtieth` and
`particleTransfer: flip`. Frame 128 is 2.133333 s. The authored profile otherwise
selects scene time (24 Hz), so an unspecified time step is not this comparison.
These are alternating, fenced simulation frames on one device, with the renderer
excluded. The GPU remains contended; all timings are provisional. Historical
arms evolve independently and are not physics-equivalence tests.

| Comparison, frames 120–132 | Baseline mean | Candidate mean | Paired median saving |
| --- | ---: | ---: | ---: |
| Pre-paging `854c7193` → frozen current | 44.84 ms | 73.88 ms | −29.40 ms |
| `30f892b9^` → first paging commit `30f892b9` | 43.90 ms | 68.78 ms | −24.95 ms |
| Before `0a932ced` → frozen current | 57.64 ms | 73.64 ms | −15.66 ms |
| `0a932ced^` → `0a932ced` alone | 59.38 ms | 68.33 ms | −8.92 ms |
| Old fused scatter → parallel occupied-cell mask | 73.43 ms | 72.97 ms | +0.67 ms |
| Eight closure probes → incident subsets, synchronized inputs | 75.85 ms | 75.43 ms | +0.44 ms |

The first paging commit adds approximately 24.90 ms of measured GPU work,
headed by surface advection +6.75 ms, redistance +2.78 ms, reconstruction
+2.19 ms, transfer +1.64 ms, and crossing search +1.62 ms. It has about 5%
more fine tiles and 4% more particles at frame 128; those work-count changes
are much smaller than the timing regression. Subsequent optimizations recover
part of that initial cost. The historical deltas are from separate paired
runs and should not be added as though all trajectories were identical.

The October 17 retention commit adds approximately 8.74 ms of measured GPU
work: surface advection +1.51, crossing search +1.44, surface control +1.41,
redistance +1.31, deferred surface advection +0.58, and smaller increases.
Against pre-paging, the biggest increases are redistance +3.46 ms, crossing
search +3.39 ms, and surface reconstruction +2.83 ms. Current frame 128 has
about 7,884 fine tiles versus 8,807 before paging, but more particles
(1.46 million versus 1.25 million). Fine tile count alone does not explain
the regression; more detailed crossing/metric work counts are being added.

The occupied scatter now selects a tile's nonempty eligible cells in parallel
and loops only its two occupancy masks. It retains every particle and the
same distance atomics. Closure tier checks and producer revision writes now
enumerate only incident 16h regions, including all stored regions touching a
boundary vertex. Neither change reduces resolution or solver iterations.

Scratch-only experiments not promoted: direct integer crossing samples,
shared fine-tile address runs, packed crossing-reach rows, cached cell-plane
fits, and splitting metric construction into nearest and trace launches.
They gave no repeatable whole-frame benefit, or regressed. Native buffer blits
saved roughly 0.6 ms paired median here, but have a known small-scene dispatch
penalty and are not substituted globally. Disabling narrow-band reuse saved
only about 1.6 ms mean; it does not explain the full regression. A HOT-only
surface-model bypass saved about 1.9 ms, but is also not promoted: a necessary
velocity-bound admission check is being tested instead.

Results are `/tmp/letters-regression/letters-*.json`, with resolved settings,
source hashes, stage timings, and final diagnostics. The pre-paging archive
needed initialization-only compatibility shims for missing imported helpers;
its result records that fact. No frame kernels were changed in that baseline.
`npm run check:types` and `npm run test:unit` passed before the closure change
(971 passed, 60 GPU-dependent skips). Focused GPU boundary coverage and final
combined verification are pending. **The under-50-ms target is not met.**

#### Follow-up, 2026-10-19

The combined retained scatter/closure fixes were also measured without GPU
stage timestamps, frames 120–144: frozen current **75.38 ms**, retained
**73.42 ms** mean; paired median saving **1.95 ms**. These are still provisional
simulation-only measurements, not a claim that the user's >100 ms observation
has been reproduced or that the 50 ms target is reached. Both arms had identical
fine/particle/band counts through frame 132. The final search counts were about
16,400 crossing candidates, 24,400 nearest-search tiles, and **34,400 metric
tiles**. Fine tile count alone considerably understates the surface workload.

The focused GPU closure test and occupied-scatter test both passed. The closure
test checks every vertex of a 65³ lattice against a geometric CPU oracle over
three tier epochs, including missing regions and shared boundaries. Type
checking also passed after these changes.

Additional scratch experiments: batching producer revision writes within the
existing workgroup saved no time (75.51 → 75.78 ms); sharing the fine advection
foot stencil regressed (75.77 → 76.34 ms). Neither is retained. The conservative
transport-fit admission proof saved about 0.50 ms GPU / 0.75 ms paired median,
but remains unpromoted pending dedicated transition coverage.

A fixed-residency **diagnostic only** pins all 192 authored pages and substitutes
the verified initial page/block mapping in the candidate's shader source. Its
purpose is to measure the upper bound available from removing lookup work;
this mapping is never used by the production solver. The first independent-
trajectory run measured 71.79 → 66.61 ms, but the candidate also had more fine
tiles and particles. A synchronized-state repeat is needed to isolate the
lookup cost. This diagnostic required a larger pin-list capacity in its scratch
snapshot; the production pool and residency policy are unchanged.

In progress: an exact region-local nearest-crossing kernel combines X, Y and Z
searches using a five-cell halo and shared intermediate minima. It preserves
the bounded search and tie order, and has a separate exhaustive GPU/CPU fixture.
A second experiment retains a distance trace's eight field corners while it
remains in the same cell. Both remain in scratch snapshots until measured and
validated. No resolution, time step, iteration count or tolerance was reduced.

#### Compact shared address plans (implementation in progress, October 19)

The shared plan has 27 neighbor words and 27 packed canonical vertex-home
words: 216 bytes per staged page, plus a small workgroup header. A home word
packs a 26-bit physical block and three two-bit displacements. The same home
record serves the h and 4h vertex lattices. The first stored incident page,
including the original ascending tie order, remains authoritative.

The persistent home words follow the existing wall-face list in each topology
mirror (108 added bytes per physical block). `bnRows` builds them in the same
GPU launch that rebuilds neighbor rows after a page commit. Consumers stage
both arrays cooperatively, then use shared neighbor reads and direct interior
vertex addresses. Queries outside the staged neighborhood fall back to the
existing directory. The loader validates the block/page relationship, including
stale records during window transitions. No field values, ownership decisions,
solver iterations or tolerances are cached or changed.

Initial consumers: crossing seeds, crossing geometry, distance construction,
occupied surface splats, fine reconstruction, transfer properties, fine surface
advection and momentum. The shared face traversal also stages plans for forces,
pressure projection and extension publication. Nonparticipating entry points
compile out the plan path. Broad integration and end-to-end performance are
still under validation; this is not a claim of meeting the 50 ms target.

The separate fixed-geometry specialization ablations found little benefit from
pool-capacity specialization alone (75.81 → 75.60 ms), but retaining immutable
world/frame overrides saved 6.71 ms (75.34 → 68.63 ms, 7.11 ms GPU). Both together
saved 7.44 ms (75.75 → 68.32 ms, 7.71 ms GPU). These remain scratch experiments;
restoring these overrides trades cross-scene pipeline reuse for frame speed.
The page-local crossing dilation experiment did not improve total GPU time and
is not retained.

#### HOT consumer audit and regression correction (October 19, continued)

The acceptance target remains **simulation frame 128 (2.133333 s) below 50 ms**
for `nbflip-figure-8-letters`, fixed 1/60 s, FLIP, unchanged physics. The
mixed-pool 10 ms target is secondary while this regression is unresolved.
All timings below are provisional: competing GPU activity remains present.
Synchronized comparisons copy all live GPU resources before each measured
frame, alternate the source arm and execution order, and include plan rebuild.
Independent historical trajectories cannot be added as exact causal deltas.

The generic shared address plan is **integrated but disabled by default**.
The stage ablation measured 73.77 → 78.54 ms wall time, +4.71 ms summed GPU
passes. Rebuild itself was only about 0.07 ms (previous rows about 0.035 ms).
Making the table fit shared memory did not make its consumers cheaper: the
consumer still executes the sampling/ownership algorithm, plus plan readiness,
range and fallback decisions and cooperative loading. Register pressure or
spills have not been established by counters and must not be stated as fact.
`UNIFORM_ADDRESS_PLAN_CANDIDATES` preserves the complete integration;
`--baseline-plans` / `--candidate-plans` explicitly enable all or selected
consumers for further controlled experiments. The backing layout and rebuild
stay identical when the reader path is disabled.

A diagnostic dispatch profiler (`--kernels=true`) isolated the largest plan
regressions. Its added pass boundaries change overhead, so its whole-frame
numbers are not acceptance measurements:

| Kernel | Directory (ms) | Shared plan (ms) |
| --- | ---: | ---: |
| buildDistance | 4.555 | 5.833 |
| advectFine | 3.015 | 3.965 |
| momentumUnitStep | 2.195 | 2.916 |
| coupleFine | 1.507 | 2.032 |
| transferProperties | 0.524 | 0.885 |
| momentumStep | 1.212 | 1.507 |
| project | 0.721 | 0.885 |

These consumers were optimized individually before deciding whether to retain
this generic cache. Scratch experiments and GPU-sum deltas against their own
paired baselines (negative is faster; not additive):

| Experiment | GPU delta | Decision / reason |
| --- | ---: | --- |
| Batch four tile claims and reuse page plan | +3.09 ms | Reject; worse load distribution / throughput |
| Whole-tile direct particle gather | +4.16 ms | Reject; loses coherent particle-stream access |
| Two coherent cell teams with shared splat minima | ~0 ms | No whole-frame gain |
| Direct 4³ cubic address run | −0.045 ms | No useful whole-frame gain |
| Subgroup crossing/error atomic reductions | +0.35 ms | Reject |
| Page-only cache without canonical homes | +5.41 ms vs directory | Reject |
| Expand homes to final fine/coarse texel offsets (~1.1 KiB shared) | +0.71 ms | Reject |
| Reduce claimed/counted grids to 512 groups | +2.81 ms | Reject; fewer launched groups do not imply lower cost |
| Merge plan publication into existing stage barriers | −0.47 ms | Does not recover generic plan regression |
| Private home-page fast branches | +2.39 ms | Reject |
| Cache cell plane bounds and twelve edge fractions | +0.74 ms | Reject; extra geometry traffic outweighs saved arithmetic |
| One lane computes all three momentum components | +0.25 ms | Reject |
| Ordinary metric-bank loads/stores, isolated redistance module | +0.045 ms | Reject; redistance stayed 4.401 ms in both arms |

The ordinary-access experiment pruned the kernel call graph and rejected any
remaining bin read-modify-write operation before replacing atomic loads and
stores. It remains scratch-only; there is no generic shader-pruning layer or
unproven non-atomic writer conversion in production.

Retained consumer cleanup: reconstruction enumerates 64 positive vertices on
interior page tiles, using the compiled positive-incidence mask. Page-boundary
tiles retain the previous 125-vertex authority traversal. A native GPU fixture
with permuted physical blocks, AIR holes and mixed h/4h tiles caught a corner
case missed by face-only wall detection; the conservative boundary traversal
passes the full fixture. The same eight-entry mask removes the seven-neighbor
loop from fine surface advection without changing its selection rule. Earlier
ownership-prototype timings must not be treated as measurements of this final
boundary-safe version.

The larger retained correction is explicit fixed-world specialization in hot
sampling modules: NB-FLIP, surface transport, momentum, forces, extension,
remap and phi resolve. Only ownerships already promising immutable world
geometry supply these constants; movable frames and mutable worlds retain
runtime addressing. Capacity remains runtime. Unselected modules still share
pipelines across scene bounds. Hot modules now trade some cross-scene pipeline
reuse for cheaper repeated sampling; compilation remains construction-time.

`letters-hot-specialized` compares this change alone after disabling plan
readers and applying the same vertex cleanup in both arms. Frames 120–132:
**70.179 → 65.556 ms GPU**, saving **4.623 ms**; wall means 83.530 → 76.603 ms,
paired median saving 3.742 ms. This run had substantial wall-clock outliers.
At frame 128 specifically, wall 79.466 → 77.638 ms and GPU 72.810 → 71.827 ms;
rebuild was 0.065536 ms in both arms. Do not substitute the window mean for
that exact-frame measurement. Maximum speed, front, volume-cell sum and root
pressure residual matched; represented-volume sum differed by 0.007324 out of
1.439 million and band residual by 1.49e-7 (floating-point differences).

Validation at this point: type checking passes; unit suite **974 pass, 68
GPU-dependent skips**; native address-plan, fine-listing and sparse
reconstruction ownership fixtures pass. The address fixture compares slots,
load texels and store texels after window movement, including stale records.
The <50 ms target is still unmet. Combined unprofiled timing and a hardware
counter capture at the requested physical state are the next evidence, not a
reason to declare the regression fixed.

Combined acceptance check, `letters-hot-cleanup-combined` (no timestamp
instrumentation): frames 120–144, synchronized state, **80.081 → 70.481 ms**
mean, paired median saving **9.525 ms**. At frame 128 / 2.133333 s:
**82.397 → 72.651 ms**, saving **9.746 ms**. Both arms had 7,884 fine tiles,
1,455,681 particles, 169 resident pages and 4,780 band tiles. Rebuild is inside
the wall measurement. Final sampled speed/front/volume sums matched; root and
band residuals differed by about 1.5e-5, consistent with different floating
point evaluation/scheduling. The equations and tolerances are unchanged.
This recovers the new shared-plan regression plus part of the generic
sampling cost; it does not recover the complete historical paging regression.

To measure hardware limits instead of assuming atomics or shared-memory size
are dominant, the Instruments worker now supports a pre-capture warmup and
explicit transfer/time step. Capture invocation:

```sh
lockf /private/tmp/claude-501/fluid-gpu.lock node --import tsx tools/profile-mini-dam-xctrace.ts \
  --scene=nbflip-figure-8-letters --warmup-steps=127 \
  --time-step=0.016666666666666666 --transfer=flip --steps=180 \
  --frames=1 --first-frame --counter-seconds=3 --counter-gate-warmup-ms=0 \
  --no-baseline --gpu-report-only --out=/tmp/letters-counters-oct19
```

Here “first frame” is the first **captured** advance (physical frame 128),
not bootstrap. The manifest records warmup/settings. Encoder-isolated timings
are diagnostic, not shipping performance. GPU counters are device-wide and
must only be attributed where the report establishes exclusive coverage.

The first counter capture was unusable for stage attribution: Dawn implemented
the diagnostic tiny clear as compute, allowing neighboring stages to share a
Metal encoder. The profiling helper now uses an explicit copy between two
scratch buffers. The second capture (`/tmp/letters-counters-oct19-blit`) has
separate major stages, but the **strict complete-report check still rejected
it**: 26 smaller buckets include real `Uniform compute copy` work. These
buckets are not reported as individual-stage costs. A separately named
`diagnostic-composite-summary.json` retains the incomplete-attribution status.
The frame parser now recognizes the current NB-FLIP solid-record head;
previously its semantic start list covered older solver graphs only.

Usable, individually attributed samples from that diagnostic capture:

| Stage | Occupancy | ALU utilization | Read bandwidth | Samples |
| --- | ---: | ---: | ---: | ---: |
| Surface advection | 11.8% | 19.7% | 5.98 GB/s | 15 |
| Redistance | 18.8% | 35.2% | 8.07 GB/s | 4 |
| Particle transfer | 20.0% | 46.6% | 57.52 GB/s | 9 |
| Surface reconstruction | 13.3% | 24.1% | 5.09 GB/s | 7 |

These are a small sample from an instrumented late frame, not acceptance
numbers or proof of register spilling. They motivate testing occupancy and
latency limits before adding more shared-memory caches. The complete capture
had about 94.7% exclusive interval coverage; individual samples exclude
observed foreign GPU overlap.

The narrowed shared-plan experiment (`letters-hot-bounded-plan`) stages only
crossing/geometry/redistance/reconstruction/transfer-property consumers. It
uses private header values, one staging barrier, and a compile-time bound
removing the neighboring-page range fallback in these bounded stencils. It
still regressed **65.592 → 67.089 ms GPU**, +1.497 ms. It remains scratch-only.
This rejects the hypothesis that loader barriers and generic range checks
alone explain the failed shared-cache approach. The benchmark now disables
Dawn timestamp quantization when profiling; older tables retain their original
65.536-microsecond quantized samples and are not silently recomputed.

The paired 128-lane tile experiment did not produce a material frame gain:
0.438 ms GPU / 0.131 ms wall saved, with identical physics diagnostics.
Surface advection saved 0.214 ms, redistance only 0.048 ms. It remains
scratch-only; it does not explain the large remaining regression.


### Continued regression audit: final-address row plans

The acceptance criterion remains **<50 ms at frame 128 (2.133333 s)**, with
unchanged physics and all rebuilds included. It has not been met. Additional
controlled experiments against the retained specialized baseline:

| Experiment | GPU saving (ms; negative regresses) | Decision |
| --- | ---: | --- |
| Two tiles / 128 lanes | +0.438 | scratch only; marginal |
| Flatten particle runs into one team stream | -0.572 | reject |
| Private resident-page shortcut in eight HOT entries | -3.515 | reject |
| 80-byte final-row address plan + deferred final key conversion | +0.375 | retain |
| HOT home short-circuit + unique incident regions for phi resolve | -0.009 | reject; neutral |
| Subgroup tile maxima + immutable bounds in surface work control | -0.357 | reject |

Each row is its own synchronized paired experiment over frames 120–132;
these deltas must not be summed. Results are provisional under the user's
accepted concurrent GPU load.

The retained nearest-row plan is stage-specific: five final buffer bases for
each of four tile jobs, **20 u32 / 80 bytes of shared memory**, with one
barrier. Each base includes presence/reach checks and the current-versus-
retained bank decision. Every Y/Z search then adds its local row offset to
that base. This reduces up to 176 repeated address resolutions per tile to
five. Z selection retains coordinates and resolves physical cell keys once
after choosing winners, preserving strict comparisons and tie order.

`letters-hot-nearest-address`: crossing search **3.448 → 3.169 ms**;
whole GPU **66.169 → 65.794 ms**, wall **72.012 → 71.900 ms**. Final physics
diagnostics matched. A native GPU fixture compares every output word with
the prior Y/Z kernels across permuted tile keys, absent tiles, current and
retained rows, equal-distance ties and a partial four-tile workgroup.
The implementation adds no dispatch, persistent allocation, or readback;
plan construction is included in the consumer kernel timing.

This distinguishes a final-address plan from the unsuccessful generic
page cache. The generic variants retain per-query bounds/fallback logic and
added execution cost even with private headers or no shared staging. The
row plan removes the lookup and bank-selection computation itself. Its
local saving still leaves most of the frame unchanged; it is not evidence
that the broader <50 ms objective has been reached.

The surface-certification fusion is now retained. Each job stages all eight
tile lattices in **4,000 bytes of shared memory**, then takes residual/slope
maxima across the whole job. It removes per-tile resets and reductions:
roughly forty tile-loop barriers become one lattice barrier plus the
existing job-boundary barrier. The represented samples, nonfinite guards,
rounding allowance and derivative formula are unchanged. Maxima associate
exactly; no sum is reordered.

`letters-hot-surface-fused` measured surface work control **2.823 → 2.494 ms**.
The whole GPU saved 0.190 ms; wall means were contaminated by a 19 ms
candidate outlier (paired wall median saved 0.430 ms). The subsequent subgroup
variant measured the same stage saving, **2.821 → 2.496 ms**, despite reducing
atomic operations further. Only the plain fused version is retained: the
evidence favors removing repeated tile setup/barriers, not adding subgroup
machinery. Its native GPU coverage/page-claim and mixed-resolution slope-bound
tests passed. Existing concurrent regional-geometry edits were preserved.

The following conclusions remain bounded by those measurements:

- The generic page cache added work rather than removing the expensive
  consumer lookup path. Replacing only its storage location or headers did
  not fix that; the private-home experiment also regressed by 3.515 ms GPU.
- A final-address plan can eliminate repeated address and bank decisions,
  as the nearest-row result demonstrates, but that kernel is only part of
  the frame. It is not a 20 ms explanation by itself.
- Atomics and barriers are not interchangeable performance diagnoses.
  The subgroup-only scan regressed, while fusing repeated scans helped.
  Packing more lanes into a workgroup was also only marginal.
- Historical paging and retention regressions remain larger than the
  recovered savings. Low sampled occupancy/bandwidth does not identify
  register spills or prove a single hardware bottleneck. The <50 ms target
  remains open; no resolution, band extent, iteration or tolerance was
  reduced to manufacture a timing result.


Fresh synchronized acceptance comparison of the two newly retained changes,
with the concurrent regional-geometry changes present in **both** snapshots:
`letters-hot-current-pair`, profiling disabled, frames 120–144, state copied
before each measured frame. All rebuild and staging work is included.

| Measurement | Before new row-plan/scan fusion | After | Saving |
| --- | ---: | ---: | ---: |
| Mean frame | 71.540 ms | 70.951 ms | 0.589 ms |
| Frame 128 / 2.133333 s | 74.143 ms | 73.646 ms | 0.497 ms |

Paired median saving: **0.825 ms**. Final speed, volume,
front and pressure diagnostics matched; no solver errors. Frame 128 had
8,032 fine tiles, 1,499,718 particles and 4,886 band tiles in both arms.
This workload differs from the earlier combined-cleanup comparison, so do
not subtract its 73.646 ms from that earlier 82.397 ms as a new speedup.
The exact-frame target is still missed by **23.646 ms**.

Validation after integration: `npm run check:types` passed;
`npm run test:unit` passed **976**, skipped **69**, failed **0**. Native GPU
nearest-plan and surface-evidence tests are also run explicitly (the default
unit invocation skips GPU tests).


Further diagnostics after the fresh pair (not production changes):

- `letters-hot-static-spacing`: baking the unchanged f32 cell spacing into
  NB-FLIP kernels gave no convincing total saving (+0.056 ms GPU mean,
  −0.553 ms paired GPU median). Concurrent load was substantial and unrelated
  stages moved by several tenths of a millisecond. This does not establish
  a useful optimization; the runtime spacing path remains.
- `letters-hot-no-metric-retention`: remove temporal metric payload reads,
  writes and its completion barrier, while retaining current-field plane
  math and recomputing the same metric. Redistance **4.954 → 4.993 ms**;
  physics diagnostics matched. The apparent 0.951 ms aggregate GPU saving
  came from unrelated stages moving in both directions, including a 1.309 ms
  pressure-band regression; it must not be attributed to metric retention.
  No production cache behavior changed. Metric payload traffic alone is not
  the large missing saving under this workload.

### Hardware investigation, October 19

The machine is an M1 Max, 32 GPU cores, 32 GB unified memory. Stop treating
low occupancy as proof of a paging-induced register problem. In the historical
pre-paging capture, surface advection already averages about 14% occupancy;
the older current capture averages about 13%. The original five-counter export
was insufficient for a hardware diagnosis. The investigation now retains all
31 counters and captures the actual compiled Metal pipelines at physical
frame 128. No production shader changes were made during this diagnostic step.

Reproducible tooling:

- `tools/profile-narrow-band-worker.ts` can load a frozen historical source
  root while keeping the capture harness fixed. Optional frame and pipeline
  labels identify the physical frame and entry point. Target-frame work
  counts are logged without adding a GPU readback.
- `tools/profile-mini-dam-xctrace.ts --source-root=... --frame-labels
  --counter-reduction=1` exports raw tables for historical counter comparison.
  Frame-labelled mode deliberately bypasses the ordinary recurring-stage
  report: its semantic frame anchors do not match the new label prefixes.
- `tools/xctrace-hardware-limiters.ts --artifact=... --pid=... --frame=128`
  joins counters only to unambiguous stage interiors. It excludes observed
  other-process overlap, overlapping different stages, mixed encoder labels,
  and one hardware sample period at each boundary. `--through-frame=132`
  supports a wider window when the single frame has little exclusive coverage.
  Reports preserve per-ring sample counts and coverage; ring means are averaged,
  never summed into a claimed physical bandwidth.
- `tools/metal-capture-bridge.mm` is an optional Node addon for capturing one
  frame with `MTLCaptureManager`. Its build command is in the source. Use
  `MTL_CAPTURE_ENABLED=1`, `FLUID_METAL_CAPTURE_BRIDGE`,
  `FLUID_METAL_CAPTURE_FRAME=128`, and `FLUID_METAL_CAPTURE_OUTPUT` with the
  profiling worker. Capture overhead invalidates its wall time as a benchmark.
- `python3 tools/read-metal-capture-stats.py CAPTURE.gputrace --out DIR`
  extracts Apple compiler statistics, generated MSL, and actual spill/reload
  remarks. It joins captured pipeline object IDs to dispatched pipelines,
  rather than reporting every compiled but unused variant. The reader accepts
  only the installed Xcode's MTSP 0x400 / cidx v0 diagnostic formats; preserve
  the original capture as the authority.

The current frame-128 capture (`/tmp/fluid-metal-hardware/current-frame128.gputrace`,
source `/tmp/fluid-hot-current-after`) has statistics for all 326 direct
dispatches. Selected actual dispatched pipelines:

| Pipeline | Static instructions | Temporary registers | Compiler spilled bytes |
| --- | ---: | ---: | ---: |
| Surface `advectOwners` | 144,471 | 124 | 352 |
| Surface `advectFine` | 36,828 | 124 | 192 |
| Surface coarse merged `advect` | 64,408 | 124 | 80 |
| Redistance `buildDistance`, h/free | 9,686 | 124 | 64 |
| Redistance `buildDistance`, h | 11,906 | 124 | 64 |

These are static compiler figures, not executed instruction counts, measured
spill traffic, or an attribution of milliseconds. The compiler remarks contain
actual spill/reload operations, so spills are established independently of the
occupancy counter. The historical comparison below separates initial paging
costs from subsequent growth. The stats' zero *static* threadgroup memory does not
mean no shared memory: Dawn also binds dynamic threadgroup memory, which the
reader records separately.

Counter artifacts: `/tmp/letters-hardware-before` is pre-paging `854c7193`,
PID 60039, with exact frame labels. Its frames 124–132 report has 53% exclusive
coverage for redistance, compared with only about 10% for exact frame 128;
the exact-frame redistance counters alone are not representative. The older
current trace `/tmp/letters-counters-oct19-blit` has all 31 counters recovered
in `hardware-counters.ndjson`, but its first fully labelled frame is not proven
to be frame 128. Do not relabel it as an exact-frame comparison.

Interpretation constraints: TLB miss percentage is not MMU stall time; low
reported bandwidth alone does not identify the limiter; existing swap usage
is not evidence of swap activity during a frame. Instrumented stage durations
include changed encoder scheduling and contention, and are not the <50 ms
acceptance result. Apple's explanations of register allocation, spilling, and
occupancy are in [Metal Compute on MacBook Pro](https://developer.apple.com/videos/play/tech-talks/10580/);
counter interpretation is in [Optimize Metal apps and games with GPU counters](https://developer.apple.com/videos/play/wwdc2020/10603/).

The first-paging counter export is complete at `/tmp/letters-hardware-after`
(PID 81204, `30f892b9`, 10,550,292 counter rows). At frame 128 it reports 9,054
fine tiles, 1,280,752 particles and 5,867 band tiles. Both historical snapshots
have only initialization compatibility shims, documented in their
`benchmark-snapshot-note.json`; no historical shader or stepping code is patched.
Toolchain: Xcode 27.1 (27A9275), Node v22.22.1, `webgpu` 0.6.0, Metal backend.

Surface reconstruction has useful exclusive coverage in both exact-frame
captures. Percentages below are equally weighted hardware-ring means over
unambiguous interiors, not whole-frame GPU averages:

| Frame 128 reconstruction | Pre-paging | First paging |
| --- | ---: | ---: |
| Exclusive stage coverage | 98.4% | 83.9% |
| Compute occupancy | 25.52% | 19.07% |
| ALU utilization | 47.29% | 30.66% |
| F32 utilization | 9.37% | 3.89% |
| GPU last-level-cache limiter | 18.73% | 11.64% |
| MMU limiter | 12.85% | 5.57% |

This supports investigating lower execution efficiency rather than asserting
new memory-system saturation in this stage. It does not by itself prove a
register-capacity threshold or quantify how many milliseconds spills cost.
First-paging surface advection has only 4.5% exclusive coverage at frame 128
(3.8% across 124–132); its uncontended subset must not be presented as a
representative whole-stage hardware comparison. Redistance has limited
exact-frame coverage in both captures; the wider-window comparison remains
diagnostic, not a replacement for an isolated benchmark.

The pre-paging native capture is complete at
`/tmp/fluid-metal-hardware/before-frame128.gputrace`, with compiler statistics
joined to all 388 dispatches, including four indirect dispatches. Its frame 128 has 8,780 fine tiles,
1,298,721 particles and 5,715 band tiles. Independently evolved native captures
are compiler evidence, not a new synchronized physics/timing comparison.

| Kernel / static compiler statistic | Pre-paging | First paging | Current |
| --- | ---: | ---: | ---: |
| Reconstruction `coupleFine`: instructions | 1,786 | 3,135 | 19,868 free / 21,524 general |
| Reconstruction `coupleFine`: INT32 instructions | 521 | 1,059 | 9,046 free / 9,257 general |
| Reconstruction `coupleFine`: temporary registers | 56 | 84 | 116 free / 124 general |
| Reconstruction `coupleFine`: reported spilled bytes | 0 | 32 | 64 |
| Redistance `buildDistance`: instructions | 5,589 | 11,409 | 9,686 free / 11,906 general |
| Redistance `buildDistance`: INT32 instructions | 1,410 | 4,369 | 4,282 free / 4,704 general |
| Redistance `buildDistance`: temporary registers | 80 | 104 | 124 |
| Redistance `buildDistance`: reported spilled bytes | 0 | 240 | 64 |
| Surface `advectOwners`: instructions | 45,032 | 113,200 | 144,471 |
| Surface `advectOwners`: temporary registers | 92 | 120 | 124 |
| Surface `advectFine`: instructions | 1,310,795 | 1,942,999 | 36,828 |
| Surface `advectFine`: reported spilled bytes | 2,928 | 3,056 | 192 |

This falsifies a blanket explanation that today's regression comes from larger
surface-advection binaries or universally new spills. Modern advection has
split/specialized paths and is not a one-to-one kernel replacement. Other HOT
stages have clearly grown: reconstruction now has roughly 17 times as many
static INT32 instructions and twice the temporary registers. Its free variant's
FP32 count is 387 versus the old kernel's 407; the static expansion is chiefly
address/control work, not an equivalent increase in floating-point surface math.
The first-paging native capture separates the initial paging cost from
subsequent changes: most of today's reconstruction instruction growth happened
**after** the first paging commit. Do not attribute all of it to `30f892b9`.

Keep reported spill allocation separate from the cause of that allocation.
Current reconstruction reports 64 spilled bytes and a 32-byte stack frame, but
its regalloc remarks contain register copies, not `NumSpills`/`NumReloads`.
Private arrays also use thread-local storage; these data do not prove that
reconstruction crossed a register-allocation spill threshold. Current redistance
does have explicit spill/reload remarks. The reader preserves stack and actual
spill remarks separately so this distinction survives future comparisons.

The completed first-paging native capture is
`/tmp/fluid-metal-hardware/first-paging-frame128.gputrace`. It has 384 dispatches
and 127 distinct compiled pipelines, with 8,951 fine tiles, 1,325,283 particles
and 5,879 band tiles at frame 128. The full dispatch count **decreases from
388 to 384** across paging (both have 384 direct calls; pre-paging also has four
indirect calls). Initial dispatch-count growth is therefore not the explanation
for the historical +24.90 ms GPU regression. This does not imply zero dispatch
overhead or equivalent work per dispatch.

What the hardware evidence establishes:

1. Paging increases the compiled cost of addressing on the M1 Max. In the
   initial commit, redistance's static INT32 count triples while its FP32 count
   goes **1,496 → 1,422**. Reconstruction's INT32 count doubles while FP32 goes
   **407 → 380**. Device-load instruction sites also increase: redistance
   **87 → 234**, reconstruction **49 → 90**. These are compiled sites, not
   dynamic load counts or a claim that every branch runs for every vertex.
2. The compiler allocates more temporary registers and reports new spill
   storage in those kernels. Apple's M1 guidance explains how increased
   register demand reduces the number of resident threads, limiting latency
   hiding. Actual reconstruction occupancy falls **25.52% → 19.07%** in the
   mostly exclusive frame-128 counter comparison. This is consistent evidence
   of worse execution efficiency; it does not measure a specific occupancy
   ceiling or assign the full timing regression to spills.
3. The page-directory buffer load precedes slot resolution and the dependent
   texture load. Compiler block locations in current reconstruction repeatedly
   point to directory indexing and the two canonical vertex-slot fallback
   searches. `instructionLocations` in the reader reports those static block
   locations and checks their aggregate against the pipeline instruction count.
   They are not instruction-level timing samples. The aggregate agrees for
   every distinct captured pipeline in all three captures, excluding Apple's
   separate constant programs.
4. Cache/MMU limiters do not rise in the useful reconstruction comparison.
   There is no evidence there for a newly saturated memory system. The
   practical target is the repeated address/control graph and its live state,
   not merely relocating the page-table words. A compiled plan must let the
   consumers skip slot searches, field-selection logic and coordinate
   reconstruction, with its rebuild cost counted in the complete frame.

On Apple GPUs, integer/complex ALU operations have different throughput from
floating-point math, and register allocations are block-granular; small source
edits need not increase occupancy. Those architecture facts explain why a
cache-only change or a small instruction reduction is not sufficient evidence
of a frame-time win. They do not substitute for a synchronized A/B benchmark.
The remaining per-stage priorities are to remove the proven address expansion
from reconstruction/redistance, audit the later reconstruction growth, and
measure advection's active paths separately from its very large cold paths.
No numerical tolerance, iteration count, resolution or physics was changed in
this investigation. The last accepted frame-128 result remains **73.646 ms**;
**<50 ms is not achieved**.

Diagnostic validation: `npm run check:types` passed; `npm run test:unit` passed
978, skipped 74, failed 0. The native capture addon compiles. The reader joins
every direct/indirect dispatch in these captures to compiler stats, and its
instruction-location aggregation was checked against the compiler totals.
