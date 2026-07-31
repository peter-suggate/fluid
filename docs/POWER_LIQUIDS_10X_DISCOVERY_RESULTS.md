# Power Liquids 10x discovery results

Date: 2026-07-30. This is the result ledger for
`POWER_LIQUIDS_10X_DISCOVERY_EXPERIMENTS.md`.

The correctness authority remains Aanjaneya et al. (2017), as transcribed in
`papers/aanjaneya-2017-power-liquids.txt`. None of the diagnostic switches is
enabled by default. Every product-path run kept the existing topology,
restriction, generation, band-capacity, structured-publication, and pressure
tripwires enabled. X-3, X-4, and X-9 are deliberately quality-invalid bounds;
their output says so and is not a shippable result.

## Decision summary

| experiment | measured result | decision |
|---|---:|---|
| X-1 still-water ledger | clean wall 128.48 vs 221.59 ms/advance; isolated non-pressure `S/K = 1.20` | fund Bet R |
| X-2 redundancy | maintenance/structured rows 98–99% exact; flags 62–66%; phi 0% | delta-repair maintenance; lean rebuild phi |
| X-3 dirty oracle | 33.85 vs 114.53 ms/advance, 70.4% lower; invalid from the first missing coefficient dependencies | fund sound per-reason dirty cones |
| X-4 maintenance freeze | 67.01 vs 114.53 ms/advance, 41.5% lower | maintenance is the program |
| X-5 rooflines | 0.353/0.432/0.833/1.657 ms for stream/gather/flood/backtrace at 896K voxels | Bet L is useful but not sufficient alone |
| X-6 critical path | 143.49 ms path / 151.83 ms wall = 0.945 | fund Bet C |
| X-7 dt elasticity | 185.95/162.85/179.18 ms at dt 4/2/1 ms; elasticity 0.0268 | rebuild-dominated flat |
| X-8 cost vs change | wall correlates with dirty/change in this 120-step window; band correlation only 0.487 | inconclusive; retain both size and change predictors |
| X-9 clean-room spike | 0.4305 and 0.4282 ms/advance, six kernels, 14K pages | the 10x exists for the isolated slice |

The combined reading is not “skip everything.” It is:

1. Preserve the paper's complete Power-diagram projection and free-surface
   construction, but carry the 98–99% identical maintenance and structured
   records and repair exact deltas.
2. Do not use exact or `1e-4`-quantized hash carry for phi: both lanes measured
   essentially zero redundant phi pages. Phi needs the lean packed-stream path.
3. Replace the naive membership-only dirty oracle with sound, per-promotion
   dependency radii or fingerprint-gated promotion. The naive ring is a large
   ceiling and a correctness failure, not an implementation candidate.
4. Shorten the serial phase graph. The measured RAW/WAW spine owns 94.5% of
   wall, so phase packing and overlap are first-class work rather than cleanup.

## X-1 — hydrostatic ledger

The `hydrostatic` lane uses the authored quarter-cell-cut
`hydrostatic-power-large-offset` oracle with the large lane's dt, leaf size,
band, and fine factor. Two clean 240-step runs produced:

| lane | clean ms/advance | validation errors |
|---|---:|---:|
| hydrostatic | 128.48 | 0 |
| large churn | 221.59 | 0 |

The counter-start-gated xctrace path solved two real capture races: counter
attachment used to start after the bounded Metal label stream ended, while an
immediate attach could delay solver construction beyond that same stream. The
new gate starts counters, then releases recurring work; report reduction also
prefers the authored semantic boundary when one edge occurrence is clipped.
Both reports contain exactly one complete representative advance with full
label isolation.

| family | still ms / calls | churn ms / calls |
|---|---:|---:|
| pressure + SPGrid | 149.05 / 1165 | 168.96 / 1209 |
| octree topology | 3.51 / 19 | 20.29 / 19 |
| fine transport | 2.98 / 13 | 2.43 / 13 |
| fine topology / summary / volume | 27.16 / 51 | 18.75 / 52 |
| fine JFA | 37.42 / 33 | 6.58 / 33 |
| air support | 1.48 / 8 | 19.42 / 8 |
| structured velocity | 27.77 / 52 | 14.92 / 52 |
| other | 2.49 / 34 | 3.40 / 34 |

These isolated numbers rank work; encoder isolation inflates absolute time.
The matched non-pressure ratio is `102.81 / 85.78 = 1.20`, above the 0.6 rule.
The still-water row census further shows only two membership-changing bootstrap
generations: 197 of the first 200 generations have zero affected rows. The
frame nevertheless continues to encode about 171 compute passes per advance.

Artifacts:

- `artifacts/discovery/x1-hydrostatic-v3/report.html`
- `artifacts/discovery/x1-hydrostatic-v3/summary.json`
- `artifacts/discovery/x1-large-v2/report.html`
- `artifacts/discovery/x1-large-v2/summary.json`

## X-2 — frame redundancy

`FLUID_REDUNDANCY_CENSUS=1` is a lagged, diagnostic-only GPU census. Fine
pages follow logical brick identity across A/B physical-page recycling. Row
families follow the exact new-to-old row map across compaction. Packed
structured A/B banks read their accepted and candidate bank selectors from
their distinct control ABIs. Float families additionally hash a `1e-4`
quantized representation.

Median exact-identical fractions over 500 steps:

| family | large | mini | decision |
|---|---:|---:|---|
| fine phi | 0.000 | 0.000 | kill hash carry at this tolerance |
| fine sample/CPT flags | 0.664 | 0.621 | inconclusive |
| power descriptors | 0.987 | 0.991 | fund exact delta repair |
| power topology metrics | 0.987 | 0.991 | fund exact delta repair |
| structured boundary liquid mask | 0.987 | 0.991 | fund exact delta repair |
| structured cell velocities | 0.985 | 0.983 | fund exact delta repair |
| structured row geometry | 0.987 | 0.991 | fund exact delta repair |

The analyzer emits early/middle/late medians and separate exact/epsilon
decisions. Run it with:

```sh
npm run experiment:power-liquids-redundancy -- --run --lane=large --steps=500
npm run experiment:power-liquids-redundancy -- --run --lane=mini --steps=500
```

## X-3 — dirty oracle

The census now emits distinct dirty and affected counters plus the six fine
promotion reasons. `FLUID_DIRTY_ORACLE=membership` narrows both coarse and
fine classifications to membership delta plus one ring.

At 120 large-lane steps, the matched clean control is 114.53 ms/advance and
the probe is 33.85 ms/advance (70.4% lower). It becomes invalid early: air
support and structured dynamics reject, the fine band reaches the INVALID
sentinel, and pressure executes zero iterations. Those failures remain in
`probeFailures`/diagnostics; `--quality-invalid-probe` only allows the run to
finish and print the bound. The >15% decision rule is decisively met, but the
one-ring classifier itself is rejected.

## X-4 — maintenance freeze

`FLUID_FREEZE_TOPOLOGY_AFTER=N` skips recurring coarse/fine topology,
restriction, summaries, descriptor/topology/boundary/owner, and air-support
maintenance after generation N while continuing transport, redistance,
volume correction, and pressure on the frozen sets.

At 120 steps with `N=60`, the matched result is 67.01 vs 114.53 ms/advance,
41.5% lower. The frozen half retains 3,806 active fine pages and valid
structured/boundary controls. The final authority correctly reports that fine
generation 60 is not current at step 120. This is a quality-invalid maintenance
tax bound and clears the plan's 40% “maintenance is the program” threshold.

## X-5 — speed-of-light kernels

The standalone 14,000-page / 896,000-voxel runner contains four irreducible
kernels: packed stream, pre-resolved 8-tap gather, 28-candidate B4 flood, and
four-substep staged backtrace. Dawn/Metal timestamp markers were unreliable,
so the published default is conservative serialized submit-to-fence wall.

| kernel | median ms | µs/Mvoxel | effective GB/s |
|---|---:|---:|---:|
| stream | 0.353 | 393.5 | 20.3 |
| 8-tap gather | 0.432 | 482.5 | 82.9 |
| 28-candidate flood | 0.833 | 929.5 | 124.8 |
| four-step backtrace | 1.657 | 1849.1 | 157.9 |

## X-6 — critical path

The data-flow manifest now retains ordered dispatches and buffer read/write
sets. `FLUID_GPU_DATA_FLOW_MANIFEST=1` independently enables the audit; it no
longer silently depends on fine timestamps. Raising the timestamp query set to
4,096 captured all 171 passes with zero overflow. The resulting manifest has
2,152 ordered dispatch nodes.

- wall: 151.83 ms/advance
- summed work: 190.96 ms/advance
- longest RAW/WAW path: 143.49 ms/advance
- path/wall: 0.945
- implied parallelism: 1.33
- timestamp bracket sum/span: 1.033 (aggregate overlap exists; endpoints are
  not retained per pass)

Decision: fund critical-path shortening. Artifact:
`artifacts/discovery/x6-large-v4.json`.

## X-7 and X-8 — change elasticity

X-7 keeps physical time fixed and derives exact step counts. Over 0.5 seconds:

| dt | steps | ms/advance |
|---:|---:|---:|
| 0.004 | 125 | 185.95 |
| 0.002 | 250 | 162.85 |
| 0.001 | 500 | 179.18 |

Log elasticity is 0.0268: rebuild-dominated flat.

X-8 joins progress windows to census samples, accounting for the two bootstrap
publications and excluding the `UINT_MAX` unbounded-displacement sentinel. On
the 120-step large run, Pearson correlations are band 0.487, dirty rows 0.976,
membership changes 0.891, and displacement 0.884. This run does not isolate
existence from motion, so its decision is correctly `inconclusive` rather than
forcing Bet R. The turnkey command is:

```sh
npm run experiment:power-liquids-cost-change -- --run --steps=120
```

## X-9 — clean-room existence proof

The standalone six-kernel pipeline performs classify, carried worksets,
four-substep staged advection, two packed B4 floods, and summarize for 60
advances with frozen velocity. The default was moved from the unstable
synthetic 16,384-page boundary to the real observed 14,000-page population.

Two independent runs produced medians of 0.4305 and 0.4282 ms/advance. This is
a frozen-coupling, no-pressure, no-topology-churn lower bound, not correct
Power Liquids physics. It nevertheless clears the <=2 ms existence rule by a
wide margin and proves the hardware can execute the recommended packed band
pipeline fast enough.

## Product work funded by the matrix

1. Sound per-reason dirty radii and fingerprints, with the current tripwires
   remaining fail-closed.
2. Persistent exact carry for descriptors, topology metrics, row geometry,
   boundary masks, and structured velocities.
3. Packed lean rebuild for phi and likely CPT flags rather than assuming
   coherence that X-2 did not measure.
4. Phase packing and dependency shortening along the X-6 spine.
5. Re-derive every winning probe under the normal GPU-resident scheduling and
   generation contracts before shipping it.

### Sound dirty-oracle follow-through

The quality-invalid `FLUID_DIRTY_ORACLE=membership` arm has been deleted from
both the pressure-row and fine-page classifiers. Coarse promotion is now
unconditionally gated by the existing exact structural and wet/dry decision
fingerprints. In the fine lane, Section 5 interface membership is used only to
allocate the new SPGrid band; after exact desired/current identity assignment,
transported closest-point repairs, additions, and retirements author complete
dirty and JFA-support dependency cones. Malformed producers, wrapped pair
counts, generation mismatches, topology errors, capacity overflow, and failed
downstream publication still reject the generation.

`FLUID_FINE_REASON_CONES=0` retains the previous sound broad-interface cone as
a clean measurement control. It does not revive the invalid membership oracle.
The reproducible large-lane comparison is:

```sh
npm run measure:power-liquids-reason-cones -- --steps=120
```

It runs normal-scheduling control/candidate/control/candidate, leaves every
tripwire enabled, requires exact terminal-counter parity with the clean
control, and reports both correctness and median wall time.

The 2026-07-31 120-step large-lane run passed every tripwire and matched the
clean control's terminal counters exactly (6,690 active/desired fine bricks,
2/10 pressure iterations). Median wall was 39.458 ms/advance for the broad
clean control and 39.429 ms/advance for reason cones, a noise-level -0.07%.
This validates the sound replacement but does not claim a large-lane speedup:
on this window the GPU fingerprint usually proves broad membership and exact
repair membership identical, so the product correctly reuses the already
scattered broad cone without adding a dispatch.

## Product follow-through: exact image tasks and phase packing

The first follow-through pass implemented exact A/B compiled SPGrid images.
The inactive generation maps current rows to predecessors through the compact
row delta, rejects structurally dirty rows, and remaps every carried destination
through the exact old-to-new map. Previously unresolved skip/error codes are
always rebuilt. This fixed the stage-28 stale-edge failure and completed the
60-step large lane with zero validation errors.

Cross-generation image carry did not pay for its own exact remapping. In a
matched interleave it measured 38.30/38.63 ms disabled and 38.73/38.80 ms
enabled after predecessor validation was hoisted to one workgroup lookup.
Production therefore uses a single image bank; carry and its doubled allocation
are available only with `FLUID_SPGRID_PERSISTENT_IMAGES=1`.

The useful part was producer-authored image work. Structured row publication
now emits exact union-row and transition-row dispatch records. Operator-image
compilation consumes one 32-lane task per accepted row; adjoint compilation
consumes one 64-lane task per transition row in three fixed waves. This removes
the old `capacity * 19` and `capacity * 144` launch shapes without adding a
storage-to-indirect pass boundary.

The large-capacity single-workgroup MGPCG arm was also carried through to a
correct measurement. With `FLUID_OCTREE_PERSISTENT_MGPCG_LARGE=1`, it reduced
MGPCG from 1,675 dispatches and 44 passes per advance to one dispatch/pass and
kept every tripwire clean, but slowed 38.3 to 74.82 ms/advance because one
workgroup leaves the large GPU idle. It remains an explicit discovery arm;
capacity-large systems retain the row-parallel solver.

Phase packing produced a shipping win in structured boundary publication.
Only prepare writes indirect arguments; classify, free-surface resolve,
solid resolve, commit, rebuild, and workset publication exchange plain storage.
Keeping the required prepare boundary while packing the later stages reduced
the stage from seven compute passes to two. Two paired large-lane comparisons
measured 39.52 -> 38.63 and 38.35 -> 38.08 ms/advance, a mean saving of
0.58 ms/advance, with zero validation errors. The packed path is default;
`FLUID_STRUCTURED_BOUNDARY_COMPACT_PASS=0` retains the control.

Both packed and unpacked boundary paths completed the 500-step minimal dam with
identical late state and identical quality diagnostics, including the same
817/430/6/1 ceiling-pixel failures. Those failures and the variational-residual
gate are therefore a pre-existing long-lane baseline issue, not a phase-pack
divergence. Publication, topology, restriction, redistance, and pressure
controls remained valid through all 500 advances in both arms.

Three adjacent fence probes were not promoted: removing the structured
scatter-to-Section-6.3 label boundary regressed by about 0.36 ms/advance,
packing reconstruction into its consumers was neutral (about 0.06 ms within
run noise), and packing Section 5 fixed-point publication into reconstruction
reduced two passes but was wall-neutral (38.325 vs 38.35 ms/advance). Their
switches remain opt-in measurement arms.

## Verification

- `npx tsc --noEmit` and `git diff --check` pass.
- The focused SPGrid, Section 5, structured-boundary,
  structured-publication, persistent MGPCG, and B4 transport suites pass (93
  tests in the latest combined run;
  GPU-only cases are covered separately).
- Dawn/Metal constructs every production SPGrid setup bind group with
  validation enabled.
- The final 60-step production large lane reports 166.1 compute passes per
  advance (down from 171.1), zero validation errors, 3/10 pressure iterations,
  and complete fine-band residency.
- The 500-step minimal gate advances cleanly but fails pre-existing late
  variational/ceiling quality thresholds in both packed and unpacked controls.
- The exact UI two-step gate advances and renders cleanly but fails the same
  pre-existing descriptor/topology diagnostic counts (32/1,633) with boundary
  packing and persistent carry explicitly disabled. These unresolved baseline
  gates prevent claiming full acceptance-suite green.
