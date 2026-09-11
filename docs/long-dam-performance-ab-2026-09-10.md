# Long dam: historical performance and algorithm A/B

The reported regression is reproduced. On the same M1 Max, uninstrumented,
queue-complete production advances average **62.38 ms on September 5 versus
127.59 ms on September 10: 2.05× slower**. The main causes are face preparation
and conservative transport. Together they explain **80% of the added GPU time
in steps 9–24**, and **73% in steps 41–56**. The investigation initially left production unchanged. The subsequent
user-requested cutover now restores the September 6 face, transport and
sharpening algorithms; see the cutover receipt below.

## Controls and scope

| Revision | Meaning | Resolved selector |
| --- | --- | --- |
| `e43ea418` | September 5, before coarse-first existed | activity |
| `cd38e8b4` | September 6, first coarse-first implementation | coarse-first |
| `28c95c47` | September 10 investigation baseline | coarse-first |

All runs use long dam, balanced quality, B8/P8, the 192×96×32 finest lattice,
12.5 mm cells, the 2.4×1.2×0.4 m tank, and **1/30 s per advance**. Gamma
diffusion and sharpening are on; pressure has the same 128-iteration ceiling
and 0.001 relative tolerance. Each artifact contains the full resolved values.
The September 5 scene's old 4 ms scene timestep is explicitly overridden.

Hardware: Apple M1 Max, 32 GPU cores; Dawn Metal through the same installed
`webgpu` 0.4.0 package and Node 25.8.1. Historical revisions run in detached
worktrees sharing those dependencies. Dawn runs are sequential under the
repository's exclusive GPU lease. The Fluid Lab browser viewport was unloaded
for all final measurements.

The stage runs use hardware timestamps and exclude construction. Eight warmup
advances precede 48 measured advances, allowing comparison of steps 9–24
(0.300–0.800 s) and 41–56 (1.367–1.867 s). Production throughput is measured
separately with instrumentation off, no diagnostics within the measured window,
and one advance per completion fence. It includes command encoding, submission,
and queue completion, but excludes rendering and construction.

Historical runs evolve their own fields and topology. They measure the shipped
algorithm at matched simulated times, not identical numerical states. The
within-frame replay below provides the stronger control for face-kernel costs.

## Stage differences

These are **arithmetic means**, so differences can be added. The earlier
sharpening stage included capacity repair and scalar-mask publication; today's
three separate stages are grouped to match it. The small historical symmetry
publication bypass remains in “Other.”

| GPU work, ms/advance | Sep 5, steps 9–24 | Current, steps 9–24 | Added time |
| --- | ---: | ---: | ---: |
| Face preparation | 19.45 | 59.60 | +40.15 |
| Conservative mass/gamma/momentum transport | 6.33 | 17.31 | +10.97 |
| Sharpening + capacity repair + scalar publication | 4.09 | 11.07 | +6.98 |
| Pressure solve | 15.32 | 16.63 | +1.31 |
| Presentation publication | 1.22 | 2.67 | +1.46 |
| Other stages | 10.34 | 13.39 | +3.05 |
| **Total** | **56.75** | **120.67** | **+63.91** |

| GPU work, ms/advance | Sep 5, steps 41–56 | Current, steps 41–56 | Added time |
| --- | ---: | ---: | ---: |
| Face preparation | 12.34 | 53.32 | +40.98 |
| Conservative transport | 4.53 | 13.36 | +8.83 |
| Sharpening + capacity repair + scalar publication | 3.99 | 13.16 | +9.18 |
| Pressure solve | 24.47 | 28.87 | +4.40 |
| Presentation publication | 2.38 | 2.80 | +0.43 |
| Other stages | 9.13 | 14.01 | +4.88 |
| **Total** | **56.83** | **125.53** | **+68.70** |

The early-window stage medians from separate adjacent controls were 56.49 ms
for September 5, 62.78 ms for the first coarse-first revision, 124.32 ms for
current coarse-first, and 114.03 ms for current with the activity selector.
Thus introducing or selecting coarse-first alone does not explain the
regression. Compared with the first coarse-first revision, face preparation
increases from 18.42 to 62.39 ms median, while average accepted rows grow only
about 7.5%.

Uninstrumented production repeats measured approximately 61.93/62.38 ms for
September 5 and 126.76/127.59 ms for current. The final pristine-source pair
has medians of 62.13 and 132.54 ms. These are simulation advances, not browser
FPS or the performance of a particular render mode.

## Work growth versus cost per operation

| Mean work | Sep 5, steps 9–24 | Current, steps 9–24 | Sep 5, steps 41–56 | Current, steps 41–56 |
| --- | ---: | ---: | ---: | ---: |
| Accepted cells | 72,603 | 81,732 | 89,488 | 110,911 |
| Accepted rows | 221,153 | 249,183 | 275,537 | 340,671 |
| Pressure cells | 30,077 | 28,547 | 44,666 | 44,580 |
| Pressure rows | 93,639 | 89,392 | 143,870 | 144,260 |
| Executed pressure iterations | 65.5 | 72.5 | 101 | 102 |
| End-frame committed bricks | 12.4 | 15.7 | 4.6 | 6.4 |

Accepted work grows 13% early and 24% later, substantially less than the
3.1×/4.3× increase in face-stage cost. Pressure work and iterations are almost
identical later. More topology work is real, but the dominant increase lies
inside interpolation and transport. End-frame commits are next-frame pressure
inputs; they are not attributed to the preceding pressure-topology stage.

Separate counting-only runs instrumented **step 24** on September 6 and
current. Their atomic counters are excluded from performance measurements.

| Actual face work in step 24 | Sep 6 coarse-first | Current coarse-first |
| --- | ---: | ---: |
| Supported faces traced | 248,587 | 250,259 |
| RK2 substeps | 1,267,265 | 1,217,481 |
| Average substeps per supported face | 5.10 | 4.86 |
| RK vector samples | 2,534,530 | 2,434,962 |
| Terminal collocated samples, including fallback | 248,587 | 320,401 |
| Native staggered donor queries | absent | 799,180 |
| Native queries falling back to collocated sampling | absent | 171,879 |
| Regular adaptive dual-cell visits | no geometric locator | 2,716,142 |
| Mixed adaptive dual-cell visits | no geometric locator | 53,919 |
| Newton iterations | absent | 264,438 |

The older RK count is derived as twice its directly counted substeps; its
2,783,117 total samples exactly equal RK samples plus terminal samples.
Current directly counts both categories. Locator visits can exceed query
counts because a query can walk to another dual cell. About **98.05% of
current visits are regular**. Only 250 of the 250,259 supported faces have
exactly zero initial velocity, and 95% require multiple RK substeps. This
is not predominantly more faces, more substeps, or a large stationary domain.

## What changed algorithmically

**Face preparation.** The older path sampled the extended collocated velocity
with eight regular-lattice points, mapping those points to sparse owners. RK2
uses two velocity samples per substep, then a final velocity sample at the
departure. Its cost was bounded regular interpolation plus sparse addressing.

Native staggered sampling was added in `663838a0` and extended in `e894aa04`.
It preserves projected face detail: averaging opposing faces to cell centers
and interpolating back would otherwise apply a smoothing filter even at zero
transport time. Wet faces now sample up to eight native donor faces at their
departure. Finding a donor resolves its owner, scans that cell's incidence
rows, checks axis/center/acceptance, and checks incident-cell width and wetness.
Unsupported corners individually invoke the collocated interpolant. The census
shows 21.5% of native donor queries taking that fallback.

**The common interpolation basis.** `57b6ae39` introduced the geometric
adaptive dual-cell locator. Instead of substituting coarse owners into a
fictitious regular lattice, it uses actual donor centers, preserving the
first spatial moment and avoiding coarse-cell dead zones. At mixed junctions
the eight nodes can form a collapsed wedge or pyramid. The locator walks
vertices, inverts a trilinear geometric map with safeguarded Newton iterations,
and validates its residual before publishing positive weights. Regular cells
skip Newton but still resolve and inspect donor geometry to establish that
they are regular. This repeated regular work dominates query volume.

`e894aa04` then introduced relative-coordinate mass tracing and ordered
reductions to preserve reflection behavior. Those numerical choices matter:
reassociating weights is not automatically equivalent.

**Conservative transport.** The trace/scatter/gather transaction still moves
density, gamma and all three momentum components together, with fixed-point
deficit receipts for conservation. Its backward and forward characteristics
now use the same more expensive geometric interpolation machinery. Accepted
cell count and flow also change, so the whole-stage difference cannot be
assigned solely to one helper without a separate transport replay.

**Sharpening and repair.** The sharpening trace now differentiates the common
adaptive density interpolant using centered density samples, including seam
alignment in its dose calculation. Those samples inherit the geometric lookup
cost. Capacity repair and final scalar publication were split into named
stages; treating those names as wholly new work would overstate the change.
The grouped stage still grows significantly. Today's zero-transfer exits and
first-interior-gradient optimization already reduce some of this work.

**Topology and presentation.** Coarse-first adds energy, curvature and incoming
liquid tests; evolving support changes the number and kinds of cells. Its
receiver search is bounded by a radius-cubed neighborhood. These algorithms
are meaningful improvements, but measured planning/activity/transfer costs
are secondary here. Presentation publication contributes only 1.46 ms of the
early increase and 0.43 ms later. Renderer traversal, meshing and shading are
not part of these Dawn advance measurements.

## Same-input face replay

To isolate sampling costs, an experimental checkout executes eight face passes
in one solver's step 24: control, collocated endpoint, lattice interpolant,
both substitutions, then the reverse sequence. Face support is published once.
Every pass reads the same source fields and accepted topology, and overwrites
the destination face array. Hardware timestamps cover each pass independently;
face-array copies are outside those intervals. The last control restores the
original output before conservative transport runs.

| Face kernel variant | Mean of forward/reverse pass, ms | Face words different from control |
| --- | ---: | ---: |
| Current | 80.12 | 0 |
| Collocated terminal sample; current characteristic tracing | 58.85 | 101,750 |
| Historical-style lattice sampling; native terminal sampling retained | 58.95 | 165,105 |
| Both substitutions | 39.42 | 231,955 |

The first and last controls are **byte-identical across both copied face
arrays**. The replay reports zero validation errors and healthy authority
receipts after restoration. The two substitutions roughly halve the cost in
combination, but they materially change face values. They are attribution
experiments, **not optimization candidates or a justified revert**. The lattice
variant substitutes the old style of interpolation into today's code; it is
not a complete reconstruction of the historical algorithm.

An earlier attempt compared independently evolved processes with a final-frame
switch. Their incoming field hashes differed before the switch, so those
artifacts are excluded from exact-state causal conclusions. The same-frame
replay avoids relying on cross-process bitwise determinism.

## Optimization targets that preserve the improvements

1. Specialize the regular interpolation path while preserving actual donor
   IDs, the geometric branch, weight arithmetic and ordered sums. Separate
   reusable accepted-topology geometry from position-dependent evaluation.
   Cover all RK substeps: a one-substep-only shortcut misses most work.
2. Compile or cache **incidence-first native donor identity**, including its
   generation and support semantics, instead of rediscovering it per corner.
   Coincident face centers are insufficient: the previous direct-address
   shortcut selected a one-sided row where incidence selected a two-sided row.
3. Reuse exact interpolation work in conservative transport and sharpening
   after establishing per-stage field equivalence. Those stages provide the
   next largest opportunity after face preparation.

Accepted-row face dispatch is already compact (`2c6dd0a3`); simply reducing
immutable-template dispatch lanes is not the remaining main opportunity.
Previous rejected midpoint arithmetic and donor shortcuts are documented in
[the earlier face/sharpening investigation](sparse-cm12-face-sharpening-performance-2026-09-10.md).
Its actual-scene weight differences explain why manufactured interpolation
tests, mass totals, or one scene's hashes alone are insufficient.

## Artifacts and reproduction

All raw receipts, logs, source patches and archived probes are in
[`artifacts/long-dam-ab-2026-09-10/`](../artifacts/long-dam-ab-2026-09-10/).
The primary data are `sep05-pristine-56.json`, `current-pristine-56.json`,
`sep05-production-repeat.json`, `current-production-pristine.json`,
`sep06-face-census.json`, `current-face-census.json`, and
`current-face-replay-combined.json`.

Rebuild and validate the CPU-only rollup with:

```sh
python3 tools/summarize-sparse-cm12-history-ab.py \
  artifacts/long-dam-ab-2026-09-10 \
  --out=artifacts/long-dam-ab-2026-09-10/summary.json
```

For a fresh historical stage run, create a detached worktree at the chosen
revision, provide the same `node_modules` and the stage probe's required
`artifacts/sparse-cm12-ocean-b16-p16-stage-cost-baseline.json`, then copy the
archived revision-appropriate probe into its `tools/probe-long-dam-history.ts`.
The archived probes add resolved-value overrides and per-frame work receipts;
the pristine runs do not modify any simulation source. In that worktree run:

```sh
WEBGPU_NODE_MODULE=/absolute/path/to/shared/node_modules/webgpu/index.js \
FLUID_WEBGPU_BACKEND=metal node --import tsx tools/probe-long-dam-history.ts \
  --scene=long-dam --brick-fine=8 --presentation-page=8 --time-step=paper \
  --warmup=8 --frames=48 --final-qa=0 --quiet=1 --out=/absolute/path/result.json
```

Use `FLUID_AB_VALUES='{"selectorMode":"activity"}'` for the current activity
control. The archived `benchmark-long-dam-history.ts` extends the existing
production benchmark to this scene and explicitly pins 1/30 s; use warmup 8,
frames 16 and queue depth 1. The census patches apply to their named revisions;
the replay patch applies to `28c95c47` and uses the archived replay probe with
warmup 23 and one measured frame. Their timings must not be substituted for
production timing. Execution manifests retain the actual commands.

Initial `*-r1` stage receipts had the paused browser still rendering and are
excluded from final timings. Final pristine runs, repeated production timings,
and the same-frame replay provide the reported evidence. Full catalog QA was
disabled for timing; hardware partition closure, pressure authority receipts,
and WebGPU validation were checked. This investigation does not certify the
full numerical regression suite. No production simulation change was made,
so the post-refactor Dawn gate was not triggered; a future implementation must
pass the unchanged gate and the relevant field-equivalence checks.


## First implementation experiment: invocation-local owner lookup cache

Tested a one-entry cache of the most recently queried logical brick in the
face tracing invocation. It retained the existing accepted range, rung,
clipped dimensions and cell availability predicates. Crossing a logical brick
reloaded the cache; missing authored ownership retained the original dynamic
fallback. Donor geometry, geometric search, interpolation weights, native
incidence-first sampling and reductions were unchanged.

The same-frame step-24 replay alternated control/candidate in ABBAABBA order.
All eight passes consumed the same source fields and topology; the last
control restored the destination before downstream transport.

| Accepted-face preparation | Four-pass mean |
| --- | ---: |
| Original | 79.1675 ms |
| Owner cache | 83.3782 ms |

Every replay had **zero differing face words**, including the restored
control. Diagnostic and pressure-receipt checks passed with no WebGPU errors.
The candidate was nevertheless **5.32% slower**, so it was rejected and has
not been applied to production. These are isolated replay timings, not full
frame timings. Additional private state and control flow are plausible causes
of the regression; this test does not directly measure register occupancy.

The experiment remains reproducible in `/tmp/fluid-ab-owner-cache` at
`28c95c47`, with `owner-cache-replay.patch`, `owner-cache-replay.json`, its log,
and `owner-cache-summary.json` archived alongside the earlier receipts.
Use the same replay command above (23 warm-up frames, one measured frame).
No numerical approximation or timing ceiling was introduced. Since the
candidate was rejected before promotion, the full post-refactor gate has not
been run for it.

The initial response to this result proposed geometry or identity preparation.
The user redirected the work toward the actual old implementation and explicit
instrumentation before selecting another optimization. That inspection is now
recorded in [the adaptive-mass code map](adaptive-mass-code-map-2026-09-10.md).

The new same-input replay runs the original September 6 face helpers in
26.7 ms versus 78.5 ms for current helpers. A regular-only attempt spends
71.1 ms before its 10.2 ms fallback, while retaining 98.8% of current
cell-centre helper calls and 98.5% of native donor queries. This supplies a
concrete explanation for why removing Newton alone did not recover old speed.
The new map inventories all main adaptive-mass structures and stage changes,
including their recorded correctness motivations and remaining uncertainties.


## Production cutover: September 6 transport restored

The working tree now uses the old algorithm as the default, with no feature
flag or alternate geometric/native-face path. This is an intentional numerical
change, not an exact-output optimization of September 10.

Restored from `cd38e8b4` in `webgpu-sparse-cm12-resident.wgsl.ts`:

- Two-point face support, fixed-lattice collocated velocity interpolation,
  absolute-position RK2 tracing and collocated terminal face sampling.
- Source-width scalar donor interpolation, sequential eight-corner sums,
  cached departures, gamma/beta transport, conservative scatter and gather.
- Fine-lattice sharpening density samples, centred gradient traces and raw
  neighbouring densities in the sharpening incidence stencil.
- The old transport's unsupported-deficit self-return and nonnegative output
  clamp. These replace the newer transport failure-on-unsupported-deficit and
  negative-output behavior; the general failure latch remains.

The current accepted-row dispatch, topology/ownership storage and services,
pressure operator, velocity extension, capacity repair, live controls, solid
trace stop, and presentation pipeline remain. Thus this recreates the old hot
pipeline within the newer surrounding system, rather than checking out the
entire old repository. The geometric locator, relative-coordinate interpolation
and native face reconstruction have been removed from the production shader.

### Measured result

Same long-dam B8/P8 configuration, eight warmup frames and sixteen measured
frames (steps 9–24). The September 10 column comes from the earlier pristine
capture; the restored column is a new complete evolving-scene run. These are
not same-state replay timings. The new GPU median reported by the probe is
**63.9631 ms**, compared with the previously recorded current-code **124.32 ms**.
The following means sum the same stage partition in both receipts:

| GPU stage, mean ms | September 10 baseline | Restored old path |
| --- | ---: | ---: |
| All stages | 120.668 | 63.144 |
| Face preparation | 59.601 | 18.932 |
| Conservative transport | 17.306 | 7.107 |
| Surface sharpening | 7.623 | 1.462 |
| Pressure solve | 16.630 | 15.630 |
| Capacity repair | 3.056 | 3.088 |
| Presentation | 2.675 | 2.961 |

The summed stage cost falls **47.7% (1.91× faster)**. Face, transport and
sharpening account for nearly all the recovered time. The new receipt is
[old-transport-restored-24.json](../artifacts/long-dam-ab-2026-09-10/old-transport-restored-24.json).
It reports no WebGPU validation errors and valid frame/topology receipts for
all 24 advances. This is a short simulation timing result, not a rendering
benchmark or a long-run physical acceptance result.

### Validation and known limitations

- Effective-transport CPU contracts: **12/12 passed**.
- Production face interpolation/transport and sharpening GPU fixtures:
  **2/2 passed**. The native staggered-identity fixture was replaced because
  native reconstruction was intentionally removed. The all-incident-child
  support assertion remains a regression detector, adapted to call the restored
  production face entry point; its expectation was not relaxed.
- Ran the full **`npm run test:dawn:sparse-cm12`** with unchanged lanes and
  thresholds. It is **not green**. Eight of seventeen lanes passed: failure
  halt, mixed-ratio topology, clipped transfer, bounded generation storage,
  hydrostatic adaptivity, mini32 correctness, region surface and mini32
  performance. Mini32 measured **30.212 ms**, below the unchanged 40 ms ceiling.
- Symmetric expansion failed at **0.29648298**, versus the 0.006 ceiling and
  the earlier current baseline of 0.014906. The user explicitly directed us
  to set symmetry aside for now and focus on recreating the faster pipeline.
- Topology page-budget and mini64 performance hit their 30-second limits.
  Mini64 surface exhausted the remaining suite budget. The five following
  far-wall/live-edit/open-world lanes did not run before the unchanged
  180-second suite limit. These are unverified, not passes.
- Repository type-checking still reports errors outside the changed files;
  none of the reported errors are in the cutover files.

Receipts: [full regression](../artifacts/long-dam-ab-2026-09-10/old-transport-regression.json),
[regression log](../artifacts/long-dam-ab-2026-09-10/old-transport-regression.log),
[kernel checks](../artifacts/long-dam-ab-2026-09-10/old-transport-kernels.log).
GPU runs stopped after production cutover to leave the device available for
user UI testing. Reload an existing scene to construct the restored shaders.


## Follow-up: cheap coarse-surface improvements

The subsequent [cheap-surface trial](adaptive-mass-cheap-surface-trial-2026-09-10.md)
puts complete incident face support and physical-width face sampling into
production. Its uncontaminated ABBA comparison measures 61.96 ms for the
support-only control and 62.95 ms with physical-width sampling. The report also
records the external-GPU-contaminated captures, compiled-patch prototype and
unchanged regression-gate failures.
