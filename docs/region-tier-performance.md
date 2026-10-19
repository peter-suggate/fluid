# Region-tier performance experiments

The subsequent structural pressure work is documented in
[Regional propagation and retained pressure setup](retained-pressure-setup.md).

The fixtures in `tools/region-tier-scenes.ts` run the production NB-FLIP solver,
including its normal tier classifier, adaptive ownership, particles, pressure
acceptance and source handling. They do not force tier words or disable stages.

| Fixture | Physical input | Expected work |
| --- | --- | --- |
| `cold` | Level pool, no gravity | All stored regions cool to COLD; no particles or fine band. Isolates the retained-frame floor. |
| `rest` | Same level pool, normal gravity | Hydrostatic COLD state; tests whether numerical noise wakes expensive work. |
| `warm` | 5 mm cosine wave on the pool | Mostly WARM/COLD, coarse transport and pressure; zero particles is plausible and measured rather than assumed. |
| `splash` | Resting pool, 10 cm radius sphere added at 0.5 s | Cold startup, source wakeup, impact, then a local HOT patch and WARM surroundings. |
| `hot` | Half-length, 70 cm high dam | Broad motion and surface detail; a control for regressions on active workloads. |

All use 25 mm cells, 0.8 m height/depth, a 1/60 s timestep, no rigid bodies,
viscosity or surface tension. The first four start at a 0.4 m waterline. Length
is a multiple of 32 cells, at least 64. The splash stays 0.6 m from the left
wall when length grows: its size, fall distance and resolution stay fixed.
Tier counts include stored air regions; they are not liquid-volume fractions.
The dam reserves four page blocks per 32-cell length segment, plus the two
sentinels, because its crest's support grows above the original box. Without
that explicit headroom the 128-cell fixture exhausted the default pool at
frame 27. That was an invalid performance fixture, not a slow completed frame.

The expected cost is a fixed scheduling floor plus global coarse pressure,
WARM transport, HOT detail and particle work. Adding settled water should add
mostly coarse work, not the cost of another splash. COLD does not imply zero
cost: pressure coupling and wakeup evidence still run. A case earns its name
through observed tier/particle/band counts, not its scene label.

## Reproduction

```sh
node --import tsx tools/benchmark-region-tiers.ts --scene=cold --length=128 --steps=120
node --import tsx tools/benchmark-region-tiers.ts --scene=splash --length=256 --steps=120
node --import tsx tools/benchmark-region-tiers.ts --scene=warm --steps=48 --from=40 --profile
node --import tsx tools/benchmark-region-tier-pair.ts --scene=warm --steps=140
node --import tsx tools/benchmark-region-tier-pair.ts --scene=splash --steps=120
```

`WEBGPU_NODE_MODULE` defaults to the repository's Dawn installation and the
backend defaults to Metal. Measurements fence each step; construction,
rendering, diagnostic reads and final field downloads are excluded. The report
contains raw per-frame observations, machine and source identifiers. `--from`
(default 40) selects the summary window, not a change to simulation startup.
The splash still lands at frame 30. Use the raw series to distinguish startup,
impact and recovery instead of interpreting their average as a steady state.

`--profile` records compute passes, dispatches and launched workgroups by label.
These are command counts, not per-stage time. Profile runs include CPU observer
overhead and must be kept separate from uninstrumented comparisons.
`--timestamps` additionally attempts hardware phase timing. This machine has
returned zero/invalid timestamp boundaries; those errors remain explicit and
must not be replaced with invented GPU timings. Wall measurements remain the
primary performance evidence. `--fingerprint` hashes the final velocity,
volume, vertex-phi and pressure textures outside the measured window.

The paired harness uses two solvers on one device, alternates which runs first,
and fences one before starting the other. The baseline restores the previous
empty-band predicate and 4,096-workgroup particle grid, using the same compiled
shaders and physics parameters as the candidate. Both arms run every startup
and source event. It records both run orders, work counts, physics statistics
and final field hashes. A source change during a run invalidates the report.
Do not run GPU tests or another benchmark concurrently.
Use an absolute Node executable when launching through another shell: the
machine used here selected Node 22 in the workspace and Node 25 in a nested
shell. Every report records the actual runtime. Source snapshots for this run
are under `artifacts/region-tiers/source-snapshot`.

## Changes under test

* Do not launch the fine pressure band when the conservative frame ceiling is
  WARM, WARM uses 4h, and COLD regions are frozen. Both kinds of tile already
  fail the GPU membership predicate. Still clear the header and slot map.
  Unknown evidence, host triggers, promotions and live particle evidence keep
  the conservative HOT path. The fine-WARM and unfrozen settings are preserved.
* Use 64 direct workgroups for particle/order kernels when the frame ceiling
  is below HOT, instead of 4,096. Every affected kernel strides its actual GPU
  count; newly appearing work is processed completely. This changes
  parallelism, not the particle budget, seeding rules, or solver tolerances.
  HOT and unknown frames keep the existing throughput grid.

## Measurements

Measured on an Apple M1 Max, 2026-10-18. Other GPU workloads were observed on
the machine, so absolute timings and small differences are provisional. Source
snapshots prevent concurrent edits from changing the code under test; they do
not isolate GPU contention. Reports with `sourceUnchanged: false` are discarded.

The two original paired runs (Node 25.8.1, frames 40 onward) gave:

| Case | Baseline mean | Candidate mean | Median paired saving | Final fields / work history |
| --- | ---: | ---: | ---: | --- |
| Small wave, 128 cells, 140 steps | 9.261 ms | 9.001 ms | 0.238 ms | Identical / identical |
| Splash, 128 cells, 120 steps | 13.080 ms | 13.176 ms | 0.025 ms | Identical / identical |

The wave improvement is modest; restricting the comparison to its 79 measured
WARM-ceiling frames gives a 0.311 ms median paired saving. The splash result is
noise-level, with inconsistent run-order means: it is **not evidence of a
splash speedup**. The live particles keep the HOT host ceiling, so the original
two changes mostly benefit the quieter workloads. Field comparisons hash all
downloaded velocity, volume, vertex-phi and pressure texels, not only aggregate
volume. See `stable-paired-warm-128.json` and `stable-paired-splash-128.json`.

The fixed-runtime control runs (Node 22.22.1, frames 40–120) gave:

| Case | Median fenced step | Mean COLD / WARM / HOT regions | Particles / band |
| --- | ---: | --- | --- |
| Zero-gravity cold, 128 cells | 4.294 ms | 32 / 0 / 0 | 0 / 0 |
| Hydrostatic rest, 128 cells | 4.491 ms | 64 / 0 / 0 | 0 / 0 |
| Zero-gravity cold, 256 cells | 4.488 ms | 64 / 0 / 0 | 0 / 0 |
| Local splash, 256 cells | 12.829 ms | 89.0 / 18.4 / 20.6 | 36,837 / 77 at frame 120 |

The cold controls have zero measured volume drift. The hydrostatic control's
maximum speed is about 3.3e-6 m/s. Doubling the cold pool length doubles the
stored COLD region count but adds only about 0.2 ms to the median in these runs.
The 256-cell splash still keeps roughly 89 regions COLD while only about 39
are WARM/HOT: the intended spatial separation is real. Compare splash sizes
before the outgoing wave reaches the shorter pool's far wall; their complete
later trajectories need not have identical particle counts.

Two additional paired experiments used Node 22.22.1. Reducing the HOT grid to
1,024 groups while particle count was at most 65,536 gave no splash benefit
(12.312 ms baseline mean, 12.344 ms candidate, -0.005 ms median paired saving).
Forcing 64 groups throughout the active dam reached 154,458 particles at frame
100 with identical final field hashes and work histories, demonstrating that
the small grid still processes work beyond its first stride. It was slower:
16.615 ms baseline mean versus 17.346 ms candidate, a 0.743 ms median paired
penalty. Both experiments kept unchanged source hashes. They support keeping
the existing 4,096-group HOT grid; neither experimental override is used in
production. Reports: `candidate-hot-grid-1024.json` and
`stress-small-grid-hot.json`. Reproduce the latter with:

```sh
node --import tsx tools/benchmark-region-tier-pair.ts --scene=hot --steps=100 --stress-small-grid
```

The command-only wave profile (`commands-warm.json`, frame 40) has 127 compute
passes and 418 dispatches despite zero particles and zero fine tiles. The
pressure schedule alone encodes 36 V-cycle dispatches and 50 Full-cycle
dispatches, plus setup/checkpoints; crossing search runs three times (24
dispatches), and there are 35 compute-copy passes. These counts identify work
to investigate, not measured percentages of GPU time. The remaining floor is
substantial even when expensive spatial detail is absent. The next larger
optimization should measure reductions in that fixed scheduling/retained-state
work while preserving pressure acceptance and wakeup, rather than assume that
making more regions COLD will eliminate it.

## Validation

`npm run check:types` passed. `npm run test:unit` passed with 963 tests passing,
57 skipped and no failures. The added tests cover WARM-band omission, retained
membership clearing, fine-WARM/unfrozen behavior and conservative launch
selection after host triggers, live particle evidence and promotions.
The GPU evidence above comes from the production-solver fixtures and paired
field comparisons; the skipped GPU tests in the unit suite are not claimed as
executed. Timings exclude rendering and remain provisional under the concurrent
machine load explicitly accepted for this run.


## Shared regional frame plan: direct launches (2026-10-19)

`UniformRegionalFramePlan` is sealed once after the frame head's final
remap/promotions and current velocity certificate. It carries surface,
topology and velocity change domains, their 27-region neighbourhoods, a
support union, and shared fine/coarse geometry tile lists. Every consumer
uses direct, bounded, grid-striding launches. Particle merging also uses
bounded direct launches; there are no indirect dispatch calls in the uniform
solver.

Topology producers publish a separate saturating revision alongside phi
revisions. The plan reads eleven history words per region, rather than
re-comparing 192 tile metadata words. Non-COLD regions are potential writers
for the entire frame, even when their input revisions match. Page release or
relocation marks the surviving neighbours of the old world location. Tile
ranges are reserved once per region and filled cooperatively, avoiding a
serial global atomic for every tile.

Surface model preparation/support and geometry consume the shared lists;
exact stage revisions still decide whether individual retained results are
valid. Transport reads the controller's surface models and support tables
directly. This removes three bulk copies (2,872 bytes per region copied each
frame) and 2,876 bytes per region of duplicate allocation. The new plan and
topology revisions have their own allocation cost. Solid evidence and deferred
entries share one existing scratch binding to fit the ten-buffer device limit.
NB-FLIP copies the final tier snapshot once per planned frame instead of once
per consumer.

Crossing list construction now shares the snapshot dispatch: both depend on
the completed crossing-seed stage. Neighbour validation and the X/Y/Z nearest
search remain separate because their dependencies require synchronization.
This removes one dispatch per crossing search, including regions beyond the
bounded retained cache. A fully held frame omits geometry preparation,
evaluation/mirroring, coarse-phi publication, revision export and rebuilding
the regional lists; the host must already certify unchanged inputs and valid
geometry. Activation invalidates that schedule before the next advance.

### Measurements and limits

Frozen-source runs alternate both arms on one device, with reversed starting
order, 80 frames, and frames 40–80 used for the steady summary. **The ablation
compares the shared plan enabled/disabled. Shared-table access, crossing fusion
and direct particle sorting are common to both arms**, so this is not a total
before/after estimate for all changes. The frozen snapshot's source hash is
recorded in each artifact and all four runs report `sourceUnchanged: true`.
Other processes continued using the CPU/GPU as requested.

| Scene | Plan off mean ms | Plan on mean ms | Paired median penalty ms |
| --- | ---: | ---: | ---: |
| region-tier-warm-128 | 18.509 | 17.296 | -0.333 |
| region-tier-warm-512 | 9.182 | 9.218 | +0.035 |
| region-tier-splash-128 | 13.043 | 15.787 | +0.518 |
| region-tier-splash-512 | 12.814 | 12.879 | +0.076 |

These timings do not establish an overall active-frame speedup. In the less
noisy 512-cell runs, the extra plan costs about 0.035 ms (WARM) and 0.076 ms
(splash) at the paired median. The 128-cell runs contain substantial external
contention: the splash candidate has an 89.5 ms frame. Outliers are retained,
not discarded. The 512-cell splash activation window (frames 28–39) has a
worst paired penalty of 0.424 ms; the 128-cell repeat has 5.271 ms. This is not
sufficient evidence to claim activation spikes have been eliminated.

All four paired runs have bit-identical final phi, velocity, volume and
pressure fingerprints (including page-normalized pressure-root storage),
and identical tier/particle/refinement/pressure-residual histories. The
single changed-region GPU fixture selects 27 neighbour regions at both
128×64×64 and 512×64×64, admits HOT changes immediately, and tests page release
and frozen schedules. This proves the list domain scales with the change;
it does not prove total frame time does. The head still reads O(region
capacity) compact metadata. Pressure remains globally coupled, and crossing
search, support/authority setup and several clears still retain stage-wide
scheduling costs. Those are the remaining consumers to carry the shared
change domains into; merely reducing their launch grids will not remove
that overhead.

Artifacts: `artifacts/regional-work/isolated-{warm,splash}-{128,512}.json`.
The earlier `final-warm-128.json` run failed the source-change guard and must
not be used as a performance result. Early profiler totals also omitted
indirect dispatches; the profiler now counts them explicitly and paired
profile runs assert zero.

Reproduce a paired run (keep sources frozen while measuring):

```sh
node --import tsx tools/benchmark-region-tier-pair.ts --regional-work \
  --scene=splash --length=512 --steps=80 --from=40 --reverse-order
```

Add `--profile` for command counts; treat that as a separate diagnostic run.


The final diagnostic runs (`profile-warm-128.json`,
`profile-splash-128.json`) verify zero indirect dispatches in both arms.
At the held splash frame 20, the shared schedule reduces **62 to 52
dispatches, 41 to 37 passes, and 12,090 to 3,761 launched workgroups**.
At WARM frame 48 both arms submit 357 dispatches/129 passes; at active splash
frame 48 both submit 443 dispatches/133 passes. The plan's additional head
work currently offsets its active-frame scheduling savings. This is a
concrete remaining limitation, not a claim of a dramatic active-frame gain.

Validation after implementation: `npm run check:types` passed;
`npm run test:unit` passed (974 pass, 68 skipped); nine focused GPU tests
passed with none skipped. These cover the regional plan, retained surface
models, slope bounds, particle ordering (including concentrated activation),
crossing samples, crossing-cache invalidation, nearest minima and list fusion.
The paired solver measurements use the frozen source hash in their artifacts;
concurrent shader-specialization work subsequently changed other live-source
code and is not part of those measurements.

## Retained geometry setup in the shared plan (2026-10-19 follow-up)

The plan now maintains the bounded geometry cache in its existing head seed
launch. A single workgroup keeps live physical-block assignments, releases
retired assignments and fills vacancies. Surviving regions keep their snapshot
slots across ownership adoption. Cache allocation marks producer work and
invalidates the assigned snapshot before reuse. It costs O(cache capacity)
when full, or O(cache capacity + physical blocks) while filling vacancies;
it does not perform a block-table rank search for every cache slot.

After sealing, geometry control consumes the shared surface neighbourhood:
one workgroup checks a region's tile snapshots and commits its dirty,
topology and revision evidence. Exact producer topology and phi revisions
bypass unchanged tile metadata and texture gathers. Unknown/saturated phi
revisions still require comparison. The path removes the global dirty,
topology and mapping clears, the private revision copy, cache-rank selection
and the separate finish dispatch. Initialization, forced refresh and the
metadata-only diagnostic retain their full fallback.

Geometry control and evaluation share a compute pass, with separate ordered
dispatches for the actual data dependency. The evaluation scatters pending
normal invalidation to changed regions' neighbours; the relayout/body path
also emits this once per changed region using a consumed atomic marker.
There is no full-region normal candidate scan on those paths. Owner-rank tags
still update after adoption; the sealed frame does not scan them again.
The post-surface volume mirror consumes the same tile list and writes only
regions with changed geometry. Relayout still mirrors all owners because
remapping may have written volume outside the geometry change list.

All launches remain direct and bounded. The new diagnostic
`--regional-geometry` compares these changes, including retained cache
assignment, against the previous full geometry setup while both arms use the
shared frame plan. `--surface-tension=0.072` exercises the normal consumers.
No numerical tolerance, solver iteration, particle policy or scene-specific
execution rule changed.

### Follow-up measurements

Frozen snapshot: `/var/folders/yg/x12gwkfx6934zcd17yn0tdq00000gn/T/fluid-geometry-measured-zpb6jie3`.
Artifacts: `artifacts/regional-work/geometry-measured-*.json`. Each records its
source hash and verifies unchanged source. Both arms alternate first execution;
the reversed-order repeats reverse that alternation. The table uses the common
frames 40–80, including for the first WARM run that continued to frame 100.
Profiler runs are excluded from this timing table. Other machine workloads
continued; the heavily contended first WARM run is retained, not discarded.

| Case | Full setup mean ms | Shared setup mean ms | Paired median saved ms |
| --- | ---: | ---: | ---: |
| WARM 128, contended | 15.939 | 16.165 | +0.116 |
| WARM 128, reversed order | 8.800 | 8.673 | +0.168 |
| WARM 512 | 9.377 | 9.229 | -0.001 |
| One splash, 128 | 12.553 | 12.564 | +0.014 |
| One splash, 512 | 12.847 | 12.751 | +0.090 |
| Four splashes, 512 / 64-block pool | 15.429 | 15.269 | +0.135 |
| One splash, 512, reversed order | 13.309 | 13.041 | +0.165 |
| Four splashes, 512 / 64-block pool, reversed order | 16.165 | 15.943 | +0.275 |
| One splash, 128 / surface tension 0.072 | 12.317 | 12.338 | -0.005 |

The gains are small, not a demonstrated large whole-frame speedup. The quieter
moderate WARM repeat saves 0.168 ms at the paired median; the longer WARM case
is effectively flat. One to four splashes increases the candidate mean from
12.751 to 15.269 ms while the four-splash case reaches 90,153 particles and
534 fine tiles. The pool budget differs in that stress case, so it is not a
pure particle-count regression fit. Increasing pool length from 128 to 512
keeps the single-splash candidate mean near 12.6–12.8 ms in these runs.

The 100-frame pilot of the 512-cell splash exhausted its unchanged default
32-resident-block page budget at frame 84. The valid comparisons use 80 frames;
the four-splash stress fixture explicitly reserves 64 pool blocks. This is a
fixture capacity constraint, not evidence of a new numerical failure.

The WARM frame-48 command trace goes from **357 dispatches / 129 passes /
160,467 workgroups** to **351 / 127 / 154,345**. This removes whole dispatches
and two pass boundaries, including one compute-copy pass. It is still only
six dispatches out of the complete solver schedule: crossing reconstruction,
transport/support setup and globally coupled pressure work remain significant.

All completed comparisons have bit-identical final phi, velocity, volume,
display pressure and page-normalized root pressure, and identical work/residual
histories. Surface tension at 0.072 N/m also matches exactly. Focused GPU tests
compare the production full and regional controllers from identical snapshots,
including cache overflow, subthreshold phi changes, saturated revisions, sparse
high physical blocks, vacancy reuse, and 125-vertex h/4h closures. The plan tests
cover immediate activation, old-page neighbours and constant 27-region local
domains as the world grows. All six focused GPU tests pass. Required checks:
`npm run check:types` passes; `npm run test:unit` passes (976 passed, 68 skipped).
The GPU tests above were executed separately without skips.

The 512-cell splash frame-48 trace goes from **443 dispatches / 133 passes /
279,669 workgroups** to **437 / 131 / 261,340**. At activation frame 30 it goes
from 390 / 134 / 260,077 to 384 / 132 / 241,748. The held frame 20 remains
52 dispatches / 37 passes in both arms: those stages were already omitted.
Every profiled frame records zero indirect dispatches.

Activation timing needs qualification. With the original order, the first
particle frame (30) is 1.509 ms slower in the 512-cell candidate; reversing
execution order makes it 1.294 ms faster. The single-splash activation window
(frames 28–39) has baseline/candidate maxima of 13.248/13.792 ms in the original
run and 16.754/14.085 ms in the reversed run. This does not support a consistent
new activation cliff, but the concurrent load prevents a tight latency bound.
The reversed single-splash steady run saves 0.165 ms at the paired median
(13.309 ms baseline mean, 13.041 ms candidate mean), with exact fields/history.

Reproduction (Node 22 with the installed WebGPU provider):

```sh
node --import tsx tools/benchmark-region-tier-pair.ts --regional-geometry --scene=splash --length=512 --steps=80 --from=40
```

Add `--reverse-order` for the alternating-order control, `--profile` for actual
command counts, or `--splash-count=4 --pool-blocks=64` for increased active work.
Use a frozen source snapshot when other agents/processes edit the workspace.

The four-splash reversed repeat saves 0.275 ms at the paired median. Its
activation maxima are 18.317/19.623 ms (baseline/candidate), with a worst paired
penalty of 1.544 ms. The original four-splash run's maxima were 14.575/14.340 ms,
but its worst paired penalty was 4.379 ms before particles appeared. These
outliers remain reported: neither order establishes a bounded activation cost.
All eleven final measured artifacts (including command profiles and the
surface-tension case) pass exact field/history and unchanged-source checks.


## Shared crossing/air domains and occupied particle bins (2026-10-19)

The shared frame seal now owns the crossing and air-support domains. After
particle movement publishes support, two producers build the exact requested
tile masks, crossing neighbour regions, and surface/particle air neighbourhood.
All three crossing searches consume that same domain. Ownership/solid closure
validation runs once per frame; exact phi revisions still validate each search.
The first producer shares its compute pass with the first search. Cache selection
and the X nearest sweep share a dispatch; dependent Y/Z sweeps remain separate.

Air support combines the coarse and fine seed launches, then computes the three
separable dilation axes and final gate in workgroup memory. It preserves the
live-page mask at each axis, including holes. Its 14³ tile gathers resolve 125
region addresses once, instead of searching the page table for every gather.
Coarse air clearing consumes the seed-region list. This replaces eight air
stage dispatches with four; unchanged regions preserve their output planes.

Particle ordering also consistently uses the occupied-tile domain at every
pool size. It clears the prior occupied cells, scans tile totals (64 times fewer
inputs than cells), and derives cell offsets only in occupied tiles. A listed
consumer adds the final parent prefix, eliminating the finest whole-domain add
pass. Empty WARM frames clear no cell banks. The previous size threshold remains
only in the diagnostic reference; the production algorithm is unified. The
bounded run sort and parallel merge preserve canonical particle order. Occupied
clears and offsets use at most 128 workgroups, independent of the particle grid.
This removes the moderate-case overlaunch of two 4,096-group tile stages when a
small splash makes the particle launch policy HOT; every listed tile is still
processed by a striding loop. A preliminary corrected batch (`frame-plan-verified-*`)
exposed that overlaunch and is superseded by `frame-plan-measured-*`.

All launches are direct. Fixed launch grids stride the lists, so unexpected
activation is handled immediately without indirect dispatch, delayed host counts,
or a scene-specific shortcut. A global metadata producer and tile-total scan
remain: this reduces full-domain work, but does not make the whole solver
strictly proportional to splash size.

### Validation and measurement method

Required type checking and the complete unit suite pass (978 passed, 73 skipped).
Seven focused regional GPU tests and both particle-order GPU tests pass without
skips. Tests exercise request retirement/activation, sparse holes, retained
crossing reuse, exact fused air propagation, and canonical FLIP/APIC records
through crowded cells with 131,072 particles.

The paired benchmark compares the legacy cell-scan policy plus full crossing/air
setup with the unified occupied policy plus shared domains. It asserts each
constructed arm's ordering policy, checks source stability, compares exact final
phi/velocity/volume/display pressure and normalized root pressure, and checks
every frame's work/residual history. Both arms alternate first execution.
Native GPU timestamps are collected in separate diagnostic runs.

An initial batch named `frame-plan-final-*` did not apply the legacy ordering
switch through the solver's scoped device wrapper: both arms used occupied
ordering. Those files measure crossing/air only, and are excluded from the
complete-pipeline table below. The corrected switch uses the device's shared
compilation authority; construction now asserts the requested policy.

The measurements use a frozen implementation snapshot while other tasks continue
editing the workspace. The live files later gained unrelated buried-vertex
shader and wall-contact cleanup changes; those concurrent changes are preserved
and are not part of these performance comparisons. Other GPU and CPU workloads remain active. Means,
paired medians, order reversals and activation outliers are reported separately;
these runs cannot establish an isolated-machine latency bound.

### Measured domain and command scaling

At frame 48, increasing the single-splash world length from 128 to 512 leaves
its domain exactly unchanged: **432 requested tiles, 30 crossing regions,
64 air regions**, while the pool grows from 80 to 272 regions. Four splashes
in the larger reserved pool use 1,464 requested tiles, 78 crossing regions,
and 112 air regions out of 512. These are read after the measured frame for
diagnostics; they never control host launch sizing.

The dense particle prefix scans 327,680 / 1,114,112 cell entries in the two
WARM sizes. The new path scans 5,120 / 17,408 tile totals, plus cells only in
occupied tiles. Prefix workgroups fall from 5,161 / 17,545 to 41 / 141.
The empty occupied-cell consumer adds 64 bounded groups and immediately returns.
The old full-bin clears write 2.64 / 8.98 MB each frame; the new empty occupied
clear writes no cell banks. With particles present, clearing and cell-offset work
scale with occupied tiles, irrespective of crowded-cell particle counts.

Complete frame-48 command counts (including compute copies):

| Case | Baseline dispatches / passes / groups | Candidate dispatches / passes / groups |
| --- | ---: | ---: |
| WARM 128 | 351 / 127 / 145,849 | 346 / 129 / 130,819 |
| WARM 512 | 351 / 127 / 188,754 | 348 / 129 / 161,345 |
| One splash 128 | 437 / 131 / 210,235 | 432 / 133 / 195,333 |
| One splash 512 | 437 / 131 / 253,276 | 434 / 133 / 225,995 |
| Four splashes 512 / 64-block pool | 457 / 131 / 301,257 | 454 / 133 / 258,257 |

Dispatch count is still only three to five lower, and the occupied clear adds
pass boundaries (two net additional passes). This is **not** a large scheduling
reduction. The material improvement here is removed cell-domain traffic and
repeated regional validation; fusion reduces the air stage from eight dispatches
to four. No profiled frame uses indirect dispatch. The remaining WARM submission
floor includes 35 compute-copy dispatches, 22 extension dispatches and 14 ownership
census dispatches; counts alone do not establish their time cost.

### Paired frame timings and activation

Apple M1 Max, Node 22.22.1, Metal; frames 40–80. Frozen snapshot:
`/var/folders/yg/x12gwkfx6934zcd17yn0tdq00000gn/T/fluid-frame-plan-final-1yc9klty`.
Artifacts: `artifacts/regional-work/frame-plan-measured-*.json`. Positive saved
values mean the candidate is faster. Native timestamp runs are excluded here.

| Case | Baseline mean ms | Candidate mean ms | Paired median saved ms |
| --- | ---: | ---: | ---: |
| WARM 128 | 9.902 | 9.440 | +0.327 |
| WARM 128, reversed | 9.331 | 9.011 | +0.199 |
| WARM 512 | 13.720 | 16.339 | +0.531 |
| One splash 128 | 17.838 | 16.597 | -0.330 |
| One splash 128, reversed | 20.376 | 20.530 | -0.062 |
| One splash 512 | 13.946 | 13.790 | +0.099 |
| One splash 512, reversed | 13.563 | 13.411 | +0.256 |
| Four splashes 512 / 64-block pool | 16.823 | 16.669 | +0.130 |
| One splash 128 / surface tension 0.072 | 13.647 | 13.615 | +0.032 |

These are modest gains, not a large whole-frame improvement. WARM's two 128
runs save 0.20–0.33 ms at the paired median, while the 512 run saves 0.53 ms.
The moderate splash runs remain 0.06–0.33 ms slower at the paired median;
the reversed run is also slightly slower in the mean. This regression is not
resolved by reducing the occupied-stage overlaunch. All outliers remain included. The four-splash candidate takes 16.67 ms versus 13.79 ms for
one splash, reaching 90,153 versus 23,704 particles and 534 versus 136 fine tiles.
Its larger reserved pool prevents page exhaustion; this is not a pure
particle-count regression fit.

In the single-splash 512 activation window (frames 28–39), baseline/candidate
maxima are 14.214/14.121 ms and 15.049/14.946 ms with reversed order. The first
particle frame is 0.533/0.241 ms slower in the candidate in those two runs.
Four-splash activation maxima are 16.081/15.532 ms; its first particle frame
is 0.213 ms faster and worst paired penalty is 0.516 ms. There is no large
repeatable activation cliff in these samples, but these measurements do not
bound activation latency under contention. WARM-512's unrelated activation-window
outlier reaches 73 ms in the baseline, illustrating the limitation.

### Native GPU stage timings

Separate `*-timed.json` runs timestamp the selected passes on the GPU. Values
below are means over frames 40–80; timestamps are quantized at approximately
0.032768 ms, so a zero sample is below resolution, not free work. Crossing
includes all three searches and the shared domain producer. The particle
baseline measures its prefix pass; the candidate includes its occupied clear
and prefix/cell-offset passes. The baseline's full-bin compute clear is outside
that measured subset.

| Stage | WARM 128 baseline → candidate ms | Splash 512 baseline → candidate ms |
| --- | ---: | ---: |
| Particle prefix / listed setup | 0.062 → 0.021 | 0.094 → 0.038 |
| Crossing searches and domain | 0.318 → 0.219 | 0.513 → 0.527 |
| Air support and clearing | 0.133 → 0.128 | 0.145 → 0.110 |

WARM crossing and particle setup improve; WARM air is effectively flat. Splash
air and particle setup improve, but its crossing subset is slightly slower.
The measured subsets total about 0.513 → 0.368 ms for WARM and 0.753 → 0.676 ms
for splash. Those savings explain why eliminating large scans does not yet
produce a large overall speedup: most frame cost lies outside these passes,
and the submission floor remains high.

Reproduce with the installed Node 22 WebGPU provider:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/benchmark-region-tier-pair.ts --regional-pipeline --scene=splash --length=512 --steps=80 --from=40 --profile
```

Use `--reverse-order` for the paired-order control, `--timings` for a separate
native timestamp run, `--splash-count=4 --pool-blocks=64` for active-work scaling,
or `--surface-tension=0.072` to exercise the normal-dependent surface path.

All eleven final measured artifacts (nine plain command-profile comparisons and
two timestamp diagnostics) pass bit-exact final-field and every-frame work/history
checks, unchanged-source checks, and zero-indirect-dispatch checks. The final
integrated workspace rerun passes all seven regional GPU tests; the particle
ordering GPU rerun passes both tests. Type checking and the unit suite also pass.
The moderate-splash regression and high submission count remain open performance
limitations; this change establishes regional data-work scaling, not the requested
large end-to-end speedup.

## Shared stage scheduling and fused regional air retirement (2026-10-19)

The sealed regional plan now enables a shared compute schedule for the remainder
of the frame. Logical stages and compute copies share native WebGPU passes;
dispatch boundaries still synchronize their dependent reads and writes. Native
buffer/texture copies, readbacks, query resolution, explicit timestamp passes,
render passes and submission close the shared pass. The frame head stays outside
this schedule until the plan is sealed. No indirect dispatch or new count
readback is used.

The first implementation cleared every binding at every logical stage. That
added hundreds of binding commands and largely erased the host-side saving.
The final scheduler clears only old bindings that the next stage does not
replace, immediately before its first dispatch. Pressure binding caches now
include the logical-stage epoch: pass object identity alone is insufficient
when several stages share it. Timestamped diagnostic stages remain separate.

Air retirement also loses whole stages. The seed producer resets its changed
word, and propagation retires the canonical fine/coarse vertices in the same
regional workgroup. It consumes its local six-cubed dilation for positive-phi
protection, rather than another workgroup's output. Negative-phi protection
still reads the completed physical seed plane, including the coarse interpolation
footprint at wall contacts. Coarse vertices outside the resident fine ring read
the canonical base home. This removes both clear dispatches and their preceding
one-word compute clear: four air dispatches become two, with unchanged sparse-hole
and canonical-writer semantics. Fine and coarse jobs share the same implementation.

Diagnostic switches on `UniformRegionalFramePlan` separate scheduling from air
fusion. `benchmark-region-tier-pair.ts --frame-schedule` compares the previous
shared-domain implementation with the new schedule and air fusion. Add
`--schedule-only` or `--air-only` for ablation. Add `--regional-pipeline` to compare
against the earlier legacy particle scan/full crossing-and-air pipeline as well.
The artifact now records synchronous encode time alongside fenced frame time.

Validation: `npm run check:types` and `npm run test:unit` pass (978 passed, 76
skipped). Fourteen focused GPU tests pass, including exact fused air clearing
across sparse holes, COLD regions, both vertex widths and repeated activation;
binding-layout changes/native copy/timestamp fences; regional lists and crossing
reuse; and FLIP/APIC canonical ordering through 131,072-particle crowded cells.
The paired runs compare exact final phi, velocity, volume, displayed and canonical
root pressure, plus every frame's tiers, particle/fine/band counts, residuals and
root cycles. Every measured command stream contains zero indirect dispatches.

Measurements use a frozen source copy at
`/var/folders/yg/x12gwkfx6934zcd17yn0tdq00000gn/T/fluid-frame-schedule-final-k2atq662`.
Other CPU/GPU work remained active. The first measurement batch also overlapped
our required checks; those artifacts are retained, and a second batch runs after
our checks finish. Construction and diagnostic readbacks are outside the frame
timing. Absolute times and isolated maxima are not reliable throughput claims
under this load; paired medians, both execution orders and command counts are
reported separately.

A follow-up removes unchanged far-air work inside that fused producer: values
already at or above the far marker return before neighbourhood queries, revision
checks and writes. Coarse loads choose their canonical home directly, and local
positive-phi protection no longer calculates a global tile address. The exact
air test includes already-retired values above the threshold. Its updated source
is frozen at
`/var/folders/yg/x12gwkfx6934zcd17yn0tdq00000gn/T/fluid-frame-schedule-retained-uiz1euqf`;
`frame-retained-*` artifacts measure this final version. Earlier
`frame-schedule-*` artifacts are exploratory measurements before this refinement.

### Final measurements

Eighty steps per paired run, reporting steps 40–80; alternating execution order,
with the moderate splash repeated in reverse order. Frame 48 command counts:

| Scene | Passes, before → after | Dispatches, before → after | Paired median frame saving | Paired median encode saving |
|---|---:|---:|---:|---:|
| WARM, length 128 | 129 → 44 | 346 → 343 | 0.001 ms | 0.279 ms |
| WARM, length 512 | 129 → 44 | 348 → 345 | 0.150 ms | 0.273 ms |
| Splash, length 128 | 133 → 44 | 432 → 429 | 0.333 ms | 0.375 ms |
| Splash 128, reverse order | 133 → 44 | 432 → 429 | 0.016 ms | 0.305 ms |
| Splash, length 512 | 133 → 44 | 434 → 431 | 0.127 ms | 0.382 ms |
| Four splashes, length 512 / 64 blocks | 133 → 44 | 454 → 451 | 0.018 ms | 0.607 ms |
| Splash 128 vs earlier full pipeline | 131 → 44 | 437 → 429 | -0.001 ms | 0.258 ms |
| Same full comparison, reverse order | 131 → 44 | 437 → 429 | 0.140 ms | 0.228 ms |

Positive savings favour the candidate. Values near zero are effectively neutral
under this load. Against the earlier full pipeline the moderate regression is
not reproduced: mean savings are 0.100 and 0.301 ms, but the paired medians are
flat and +0.140 ms. This is evidence of improved scheduling, not a large or
noise-free end-to-end throughput gain. The earlier candidate’s WARM regression
was not retained as a claimed win; after the far-air refinement WARM128’s paired
median is flat (its mean remains 0.170 ms slower).

The air-only WARM128 timestamp diagnostic measured mean air-stage time of
0.139 → 0.094 ms. Timestamps were quantized in roughly 0.065536 ms increments
and included large external-load outliers, so this supports direction rather
than precise microsecond attribution. The pass-sharing ablation initially
provided little wall-time gain; eliminating eager binding resets made host
encoding consistently cheaper.

At frame 48 the single splash consumes 30 crossing regions and 64 air regions
at both lengths 128 and 512, despite region capacity growing 80 → 272. Requested
tiles are 404 and 428. Four splashes consume 78 crossing regions, 112 air regions
and 1,421 requested tiles in a 512-region pool. The shared scheduling overhead
remains 44 passes; pressure, transport, global metadata and residual work still
grow with the pool. This is not yet a wholly local-cost solver.

Activation (frames 28–39) remains exact in all histories. The moderate run’s
maximum frame time was 15.02 → 19.31 ms; in reverse order it was 14.44 → 12.93 ms.
The worst paired candidate penalty was 8.34 ms in the first order and 0.92 ms in
the second. Four-splash activation maxima were 21.05 → 19.23 ms. These do not
establish an activation-time bound or justify claiming that all spikes are gone.
The direct striding lists process new work immediately; neither their cardinality
nor solver convergence changed.

All eight final `frame-retained-schedule-*` artifacts and the air-only diagnostic
pass exact field/history and unchanged-source checks. The final and preceding
source versions also have identical moderate-splash reference field hashes.
The remaining dispatch cost is substantive: a representative splash still has
429 dispatches, including 76 band-solve dispatches, pressure cycles, extension
and the fixed merge envelope. Pass sharing alone cannot remove those numerical
dependencies; the next gains must remove work or fuse genuinely local dependencies.
