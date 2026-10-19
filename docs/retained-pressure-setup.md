# Regional propagation and retained pressure setup

The default reuses the pressure hierarchy's geometry and coefficients only
when its producers explicitly hold them. This removes 11 settled-step launches.
Exact regional propagation is implemented and tested, but remains opt-in through
`cycles.retainRegionalSetup`: measured active-frame latency regresses, so reduced
work counts do not justify enabling it by default. It does not change the
pressure equations, iteration schedule,
acceptance threshold, time step, particle budget or tier classification.

## Dependency contract

The existing pressure-only arena keeps each level's raw phi, continued phi,
open/face fractions and baked coefficients. No non-pressure stage writes these
ranges. The root producer compares current phi and fractions bit for bit with
the retained values. A separate exact record compares each block's page
identity and all 27 neighbours, including free and reclaimed pages.

Changes propagate through downsampling, continuation and coefficient baking.
Each stage rebuilds a conservative neighbour closure and publishes whether its
outputs actually changed. Unchanged output stops propagation to deeper levels.
The unit of this evidence is a 32h page block, distinct from the simulation's
16h heat regions. No approximate equality, heat threshold or delayed host
receipt decides whether a coefficient is valid.

The kernels share neighbour checks across each block's run in a workgroup,
including coarse levels where a group spans several blocks or the boundary
between cell and wall rows. An
initial implementation repeated those checks per lane: despite eliminating
settled rebuild work, it regressed the 128-cell rest case by about 0.25 ms in
median paired time. That version was rejected. Its report is retained as
`artifacts/region-tiers/retained-per-lane-rest-128.json`.

When the frame explicitly holds the surface and suppresses census/remap, with
no moved body or pressure kick, it also passes that producer guarantee to the
pressure setup. This removes 11 hierarchy dispatches: three downsample, four
continuation and four bake launches. Root scratch initialization and current
job classification still run. First setup and the diagnostic full-rebuild arm
always build everything. Wakeup restores a full rebuild by default; the
experimental mode restores
current-frame GPU comparison and regional propagation.

RHS, bounds, rounding, residuals, pressure acceptance and job classification are
fresh every solve. In particular, the dynamic job list is not retained merely
because geometry is unchanged. The new evidence requires 148 bytes per pool
block; geometry/coefficient storage already existed. There is no new production
readback, indirect dispatch, runtime pipeline compilation or per-frame JS array.

## Validation and measurement

`tests/uniform-pressure-retained-setup.test.ts` compares every word of the
hierarchy with an unconditional rebuild from identical inputs. It covers
unchanged geometry, explicit held inputs, a changed cell on a block face,
restored values, page identity changes, page release/reclaim, changed neighbours
and moving solid geometry. Both GPU tests passed on Metal. The geometry/topology
test also passed with an
odd 11-block capacity, exercising partial workgroups and coarse cell/wall
boundaries.

The experimental regional comparison uses `--retained-setup`; the final default
comparison uses `--held-setup`: both arms include the
earlier quiet-particle and empty-band improvements. The reference disables only
retained pressure setup. The same shader arithmetic builds its operators in
full; change comparisons are bypassed. Both arms keep small diagnostic block
counters and topology snapshots. Each frame alternates execution order and is
fenced; diagnostics and construction are outside the measured interval.

Reports include every frame, exact final velocity/volume/phi/presentation-pressure hashes,
recorded work and band-residual histories, and setup block visits. Final physics
statistics also include the accepted root residual; the current runner adds
per-frame root residual and cycle counts for subsequent runs. The final default
runner also hashes actual root pressure because the presentation pressure
texture can be a placeholder. The current runner compares stored pages by page
identity, excluding unused capacity: allocation order may differ between arms. Visits are summed across
levels; baking's full-rebuild count includes free-block mask initialization.
The comparison fails if final fields or recorded histories differ.
Separate startup, pre-activation (10–27), activation (28–39) and later (40+)
windows expose p95, maximum and worst paired penalties. Only the splash fixture
actually adds a source at frame 30. The activation window includes its source
preparation in the preceding advance; these are control intervals in the rest
and dam cases.

Runs use pinned Node 22.22.1 and an immutable source snapshot at
`artifacts/region-tiers/retained-final-source`. The width experiment uses
`retained-wide-source`, and the final default ablation uses `retained-held-source`,
with the same solver base and only the documented setup policy changed. Other activity on the M1 Max makes
timings provisional; paired order checks and repeated activation runs help
identify contention but cannot remove it. These are simulation-step wall times,
not renderer-inclusive frame times or isolated GPU stage timings.

```sh
node --import tsx tools/benchmark-region-tier-pair.ts --retained-setup --scene=rest --length=128 --steps=100
node --import tsx tools/benchmark-region-tier-pair.ts --retained-setup --scene=splash --length=512 --steps=100
node --import tsx tools/benchmark-region-tier-pair.ts --retained-setup --scene=splash --length=512 --steps=100 --reverse-order
```

`--splash-count=N` adds equal drops at fixed spatial spacing, two steps apart,
to increase active complexity independently of pool length. The source API
permits only one drop per step; the initial simultaneous-drop attempt was
rejected by that existing validation and is preserved as a failed report. `--pool-blocks=N` changes
reserved capacity independently of the physical scene. `--depth=N` adds settled
water across the other horizontal axis at the same cell size and waterline. These are fixture
controls; the production implementation has no scene-specific branches.

Active frames still encode the hierarchy dispatches, with unchanged blocks
returning before geometry work. This reduces rebuild cost and settled scheduling
cost; it does not remove the full active-frame scheduling floor, the global
coarse pressure solve or its existing coarse-grid capacity limit.

## Measured scaling

The 100-step length sweep uses a fixed 32-cell depth and one identical sphere
at frame 30. Each case was repeated with reversed execution order. Positive
values below mean a reduction in the median of paired step times, frames
40–100; negative values mean a regression. All 14 completed narrow-pool runs
(rest and splash at three lengths in both orders, warm and hot at length 128)
matched final fields and recorded work/residual histories exactly.

| Fixture | Length | Standard order saving, ms | Reverse order saving, ms |
| --- | ---: | ---: | ---: |
| Rest | 128 | 0.217 | 0.209 |
| Rest | 256 | 0.204 | 0.166 |
| Rest | 512 | 0.134 | -0.004 |
| Splash | 128 | -0.048 | -0.052 |
| Splash | 256 | -0.055 | -0.156 |
| Splash | 512 | -0.242 | -0.107 |
| Warm wave | 128 | -0.149 | — |
| Hot dam | 128 | -0.047 | — |

The settled 128/256 cases show a repeatable scheduling saving. Active narrow
pools do **not** show an end-to-end speedup: regional evidence and branching
still cost more than the small geometry rebuild they replace. Large outliers
make means misleading, especially the 512-cell cases; no outliers were removed.

The structural work reduction is much clearer. At frame 40 of the single
splash, visits are:

| Pool length | Full downsample / extend / bake | Retained downsample / extend / bake |
| ---: | --- | --- |
| 128 | 24 / 32 / 40 | 18 / 24 / 24 |
| 256 | 48 / 64 / 72 | 18 / 24 / 24 |
| 512 | 96 / 128 / 136 | 18 / 24 / 24 |

After settling, retained visits are zero at every length. As the disturbance
spreads, visits increase: by frame 100 the 512-cell splash uses 30 / 40 / 40.
This is local propagation rather than a global rebuild on wakeup. It is not
constant total frame cost: detection still scans inputs, active dispatches
cover capacity, and pressure correction has global dependencies.

## Activation tails

The actual source preparation and wakeup are included in frames 28–39. Across
the six single-splash runs the largest paired candidate penalty in that window
was 0.84–16.10 ms. The largest was in the heavily disturbed standard 512 run;
its reverse repeat was 0.84 ms. The reverse 128/256 runs still had 5.63/6.62 ms
paired penalties. Even rest controls had isolated penalties up to 9.05 ms.
Those observations prevent a claim of spike-free activation on this machine.
The implementation has no wake-time allocation or compilation and the recorded
geometry visits stay local, but runtime/workload outliers and the broader
activation path remain visible in the wall-clock measurements.

## Wider hierarchy and default policy

At 512 × 128 horizontal cells (130 reserved blocks), the same sphere changes
more neighbouring pages across depth. Frame 40's full visits are
384 / 512 / 520; regional visits are 72 / 96 / 96. By frame 100 they are
96 / 128 / 128. Both orderings match fields, per-frame root and band residuals,
cycle counts and tier/work histories exactly. Their median paired savings are
**-0.178 and -0.180 ms**: a repeatable active-step regression even with four
times the hierarchy. Raw reports are `final-retained-wide-{standard,reverse}-splash-512x128.json`.

Consequently, `retainSetup` defaults to true but `retainRegionalSetup` defaults
to false. A producer-held frame skips the hierarchy launches; an active frame
uses the full rebuild. No threshold based on scene name, pool dimensions or
measured heat selects a numerically weaker path. The regional implementation
and diagnostic switch remain available for the next iteration.

The next structural work should remove the cost of discovering and scheduling
changes: attach exact change evidence to producers, consume compact block work
lists, and batch small hierarchy stages. Repeating neighbourhood queries across
capacity-sized dispatches is not yet a latency improvement. These measurements
do not support enabling the regional path merely because it does less arithmetic.

## Final default ablation

With regional propagation disabled, the length-128 rest run saves **0.285 ms**
in paired median time (mean 4.726 → 4.439 ms). The splash's active interval is
effectively unchanged: paired median saving +0.047 ms, mean 12.233 → 12.286 ms;
these small conflicting summaries do not establish an active-step gain.
Both match final fields including the actual root pressure buffer, plus all
per-frame residual, cycle and work histories. These reports are
`final-held-rest-128.json` and `final-held-splash-128.json`.

The default splash's activation maximum is 13.376 → 13.126 ms, with a worst
paired penalty of 0.332 ms. The rest control nevertheless has a 30.355 ms
candidate outlier in that same window (25.081 ms paired penalty). That is
preserved in the report and is another reason not to certify p99 behaviour
from these short, shared-machine measurements.

A four-drop test, spaced two steps apart, exhausted the original 34-block
reservation at frame 84. Histories matched through the completed frames; it
is an existing hard capacity boundary, not a completed correctness or latency
result. The failure is retained as
`final-retained-staggered4-splash-512.json.failed.json`. A larger reservation is
used separately to measure active complexity without conflating it with that
capacity failure.

## Checks

- `npm run check:types`: passed.
- `npm run test:unit`: 971 passed, 60 skipped, zero failures.
- Metal retained hierarchy audits: geometry/topology and moving solids passed;
  the odd-capacity geometry/topology repeat passed separately.
- `git diff --check`: passed.

The normal unit suite skips GPU-dependent tests; the Metal audits above were
run explicitly with `WEBGPU_NODE_MODULE`. An accidental GPU-enabled full unit
run was stopped and is not counted as a completed validation result.

## Increased active complexity and page activation

The four-drop run completes with a 66-block reservation and 143,413 particles
at frame 100. Frame 40's full visits are 96 / 128 / 264; regional visits are
36 / 48 / 48. By frame 100 they are 102 / 136 / 264 versus 48 / 64 / 64.
Compared with one splash, more active pages increase the regional closure;
unused reservation does not become geometry rebuild work in the regional path.
The median paired active-step saving is **-0.111 ms**, so this remains experimental.
The activation window has a worst paired penalty of 6.508 ms; this broader
stress case does not establish smooth activation tails either.

The first pressure-buffer hash flagged unequal physical storage. Investigation
confirmed pages 64, 65 and 70 occupied different block slots between solvers;
all stored pages' pressure values match exactly when indexed by page identity.
The benchmark now records each page's block and hash, sorts by page identity,
and excludes unused slots. This is a comparison of identical physical state,
not a relaxed numeric tolerance. Velocity, volume, phi, all per-frame root and
band residuals, cycles and work histories also match exactly. The completed
report is `final-retained-staggered4-pool66-canonical-splash-512.json`; the raw
storage-hash failure remains in `final-retained-staggered4-pool66-splash-512.json`.

The independent reservation sweep keeps the rest pool at length 128 and grows
capacity from 10 to 34 blocks. Full setup visits become 24 / 32 / 136; the
default held path remains 0 / 0 / 0. Its paired median saving is 0.215 ms
(mean 5.296 → 4.916 ms), with exact page-indexed root pressure, other fields
and all recorded histories. See `final-held-reserve34-rest-128.json`. This
confirms that unused reservation does not restore settled hierarchy rebuilds;
it does not remove reservation-dependent work elsewhere in the solver.
