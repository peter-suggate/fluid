# NB-FLIP hot-kernel experiments — 2 November 2026

This extends the [ten-section study](nbflip-ten-experiments-2026-11-02.md) to actual scene inputs for particle movement, transfer, pressure, and surface reconstruction. The controlled surface fixture from that study is not evidence of a whole-scene improvement.

## Measurement contract

Host: Apple M1 Max, 32 GiB, Metal/Dawn, Node 22.22.1. GPU runs use `tools/gpu/fluid-gpu-run.sh timing` exclusively. The frozen baseline includes the working tree's existing contact/body-window changes; its uniform-method source hash is `ea43ea10c3b8f4fd020e4de7cc602040c19b6af3631175bb31c47ca1bf389cf3`. Other work in this checkout is preserved.

Artifacts live in `artifacts/nbflip-hot-kernels-2026-11-02/`. `baseline/` and candidate directories are complete source snapshots, not patch labels. Reports record source hashes, settings, raw samples, output differences and errors.

Three levels of measurement answer different questions:

* **Kernel replay:** capture a real dispatch at frames 60 and 120, restore all inputs, alternate candidates and two unchanged controls, compare every declared writable word, then restore the original output before continuing the original trajectory. Timestamp quantization is disabled. Accumulating kernels run once per sample, never repeatedly against their previous outputs. Advection additionally compares exact multisets of complete particle records and queued walk indices, retaining duplicates; counters and other words remain exact.
* **Captured GPU frame:** replay every recorded dispatch/copy/write/submission from a real frame, including worklist producers and extra argument-building passes. Time encoding through GPU completion; exclude restores/comparisons and freeze CPU planning. Unchanged controls themselves have substantial downstream differences and timing variation, so these runs are exploratory, not correctness proofs or a basis for shipping a marginal win.
* **Independent whole-scene simulation:** alternate baseline/candidate frames, or twelve-frame blocks using the existing receipt ring. Time CPU advance through completion, including normal planning/readbacks. Exclude construction, renderer and final diagnostics. Both arms use identical scene/settings/timestep/adaptive-policy code. Early paired runs and kernel replays inherited a device feature list without `float32-filterable`; they exercise the supported fallback path. Final UI runs use `fluidExecutionDeviceFeatures`, the same feature selection as the renderer, and record granted features. Do not mix these configurations when comparing absolute timings. `--features=legacy` reproduces the early timestamp/subgroup-only feature set; the default now matches the UI. Independent trajectories vary even in unchanged controls; volume and residual receipts are recorded, not treated as bitwise equivalence.

All three scenes use the normal balanced UI settings at 1/60 s: `sparse-cm12-long-dam-break`, `nbflip-figure-9-teaser`, and `nbflip-simple-breaking-dam`. A diagnostic-enabled UI-feature dam replay verified exact transfer output at frame 60, then exhausted its 17-block page pool at frame 75 before the requested frame 120. That partial run is not a completed frame-120 check. The separate pour scene failed before the measurement window because its baseline page pool exhausted (also with a 32-block override). No result from that failed scene is included.

## Whole-scene profile

Instrumented stage timings (ms), useful for attribution rather than uninstrumented frame latency:

| Scene/frame | Particle movement | Pressure solve | Transfer | Surface reconstruction | Canonical metric/publication |
| --- | ---: | ---: | ---: | ---: | ---: |
| Long dam / 59 | 6.226 | 4.588 | 1.114 | 0.590 | 0.721 |
| Long dam / 119 | 2.884 | 4.129 | 0.786 | — | 0.721 |
| Teaser / 59 | 37.234 | 25.989 | 5.137 | 12.928 | 13.821 |
| Teaser / 119 | 25.345 | 14.880 | 7.384 | 4.948 | 7.024 |

The first dam profile used quantized timestamps; the subsequent replay and teaser profile disable quantization. Pressure includes many passes: reducing the first `pRelax` dispatch is not equivalent to reducing the entire pressure solve. Tests also cover `pDownA`, `pDownB`, `pUpA`, and `pUpB`.

## All ten sections on the hot path

| Section | Experiment actually run | Outcome / decision |
| --- | --- | --- |
| 1. Memory layout | Separation reads the existing compact position stream instead of full-record positions, with and without a separate liveness load. Both are tile-sorted; the baseline is AoS within each slab. Pressure stages also test a local contiguous iterate tile. | Compact positions plus a separate record load is slower (teaser 4.96 → 7.42 ms at frame 60); eliminating that load is near noise (4.96 → 4.88 ms). No global particle-layout conversion is justified. |
| 2. Bandwidth | Advection caches the heat sample when rigid correction leaves its position unchanged; an independent variant skips the surface-shift machinery for zero displacement. | Heat cache 3.72 → 3.80 ms; zero-shift 3.72 → 4.19 ms on dam frame 60. Exact particle multisets, no speed win. |
| 3. Unnecessary work | Build separation/movement indirect arguments from the existing active slab/spill counts; include the argument builder and existing slab producers in whole-frame replay. | Dam frame 120 mean candidate 37.52 ms vs controls 38.70/39.50 ms, but controls themselves vary widely and downstream output is nondeterministic. Exploratory only; not promoted. Existing dispatches already stride active lists, with a 1,024-group ceiling. |
| 4. Divergence | Pressure wall active-set solve exits once its six-bit mask stabilizes; compare six-face unrolling separately. Movement tests 32/64/128/256-lane groups against the existing spatial slab jobs. | Stable-mask pressure is exact and faster for some dispatches (dam ~0.028 → 0.025 ms), but no demonstrated combined frame gain. Unrolling generally loses. Retain spatial grouping and pressure logic. |
| 5. Pass fusion | Replace `arenaDistance` + `arenaPublishCanonical` with the existing fused `arenaDistanceCanonical`, retaining finish and all other frame work. | Dam frame mean 37.49 ms vs 41.24/38.41 ms controls; substantial A/A variance and output differences prevent promotion. Separate metric-only calls are not conflated with canonical publication. |
| 6. Kernel splitting | Specialize the ordinary transfer path without diagnostic `bandPhi`/`particleDepth` state; prepared product candidate retains a diagnostic twin and APIC's original group size. | Strong same-state kernel gain (~33% in teaser), but whole-scene results conflict. Assess separately from the workgroup-only candidate; do not infer a frame win from the kernel gain. |
| 7. Atomics | Guard zero global maxima; aggregate transfer's three counts and maximum in workgroup memory, flushing at most four global atomics per group. | Exact results. On teaser frame 120, lean 128-lane transfer is 1.770 ms and lean+aggregation 1.749 ms; too little incremental gain to justify the extra barriers/state. Momentum already accumulates fixed-point values locally. |
| 8. Neighbourhood reuse | Stage two 4³ pressure iterate tiles in 128 shared floats, use local red/black neighbors and preserve boundary loads. | Exact, but teaser frame 120 0.053 → 0.062 ms. Barriers/staging cost more than saved reads. Existing transfer and surface halo reuse remain. |
| 9. Occupancy | Transfer 192/96/128 lanes; mover 64/32/128/256; surface metric/publication 256/128/64; pressure coarse offset/half-sweep unrolling to shorten private array lifetimes. | Transfer 128 is the leading candidate. Mover 128 gives small isolated wins but no combined frame win. Surface publication gains fail parity. Coarse pressure changes provide no robust gain. No register/spill hardware counters were collected. |
| 10. CPU–GPU coordination | Full solver per-step fences versus its existing two-frame receipt ring; then compare actual source candidates in twelve-frame pipelined blocks. | Previous three-arm full-solver run measured 14–16% better median throughput from the existing ring, with the same 656 mapped bytes/step. Already available in production. Required planning/error receipts are retained. |

These are distinct experiments, not claims that every possible AoS/SoA/AoSoA conversion or every grouping policy has been exhausted. The earlier fixture study separately measures explicit SoA packing/conversion costs.

## Lane addresses and traffic accounting

The following is source-level traffic accounting, **not measured DRAM bandwidth**. Caches, transaction sizes, masked loads and compiler elimination change physical traffic.

* **Particle advection:** adjacent lanes read `particles[job.y + off + lane]`. FLIP's record stride is 48 bytes; APIC's is 112 bytes. A successful full-record move requests one record read and one record write (96/224 bytes), before grid samples, sorting, metadata and deferred contact work. Slab output reservations are aggregated by destination brick; resulting writes are dense but atomic ranks can permute lanes. A pure SoA layout would improve individual field streams, but this kernel uses nearly the whole record and shares the bank with transfer, sorting and snapshots.
* **Separation:** neighbor records are addressed through cell runs. AoS positions have a 48/112-byte stride; the existing compact stream is `NB_LINKS + 4*j + xyz`, a 16-byte stride with a packed fourth word. Three coordinates request 12 bytes per neighbor; checking `particles[j].position.w` adds a separate 4-byte request to the record bank. The measured two-stream variant is substantially worse. No extra conversion was needed for this experiment because the compact stream already exists.
* **Transfer:** consecutive lanes own consecutive sorted records. It reads position/velocity/mode data, writes `before.w` (4 bytes), compact motion (12), and a packed support/heat word (4). Momentum scatter contributes 81 nodes × two 32-bit local atomics per supported particle. A tile flush has up to 756 node pairs × 8 bytes of global atomic payload (6,048 bytes), amortized across its particles, plus touched-tile flags. It stages 216 liquid words and 27 topology words, and holds 6,048 bytes of momentum/weight accumulators in workgroup memory. Changing lanes changes scheduling; it does not reduce this algorithmic byte count.
* **Pressure:** spatial rows use packed index/topology data and field planes. A six-face row can request six 4-byte pressure values (24 bytes), in addition to coefficients, addressing and outputs. The local-tile experiment adds 128 × 4 = 512 bytes of shared state per workgroup, plus initialization and barriers; boundary values remain global. Coarse solvers already stage a halo, so another array/unroll is not automatically useful.
* **Surface:** lanes stride consecutive local vertex numbers within an 8³ brick and 9³ window. The metric keeps a 2,048-word workgroup candidate queue (8,192 bytes), 256 found values (1,024 bytes), and window/neighbor data. The split metric-to-publication handoff costs one 4-byte write plus one 4-byte read per computed metric vertex. The tested fused entry still writes bank A for later consumers, so it avoids the publication's 4-byte metric reread rather than the entire 8-byte handoff. It also combines the metric's private/shared state with publication work; no metric buffer allocation is eliminated. Surface publication also routes writes to canonical owners; changing its group size cannot be accepted solely because the writes are faster.

## Correctness exclusions

At teaser frame 120, publication with 128 lanes changed 93 texture words, with maximum absolute difference about 0.045, while both unchanged controls matched. At frame 60 even the unchanged controls differed, and a smaller-group candidate differed by over 1.0. Simple dam frame 120 also rejects 128-lane publication (72 words, maximum difference about 0.111). These are output texels, not unused scratch. No surface workgroup/fusion change is shipped on this evidence.

Movement shape and load experiments preserve exact complete-record multisets and every other compared word. Transfer/pressure variants compare every declared writable word exactly. Whole-frame replay differences are reported rather than hidden behind a loose epsilon.

## Reproduction

```sh
# Actual scene dispatches, saved source and exact transfer/pressure parity.
bash tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/probe-nbflip-variant-replay.ts \
  --root=artifacts/nbflip-hot-kernels-2026-11-02/baseline \
  --scene=nbflip-figure-9-teaser --frames=60,120 \
  --entries=transferDeposit,pRelax --variants=aa,hotA,hotB,hotC,aa --features=legacy \
  --repeats=9 --inner=1 --assert-parity --dump=artifacts/nbflip-hot-kernels-2026-11-02/reproduce

# Movement requires the explicit exact-multiset contract.
# Use --entries=advectParticlesDeferred --particle-parity on the same replay tool.

# Whole GPU frame, including existing producers and extra argument build.
bash tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiment-nbflip-hot-frame.ts \
  --root=artifacts/nbflip-hot-kernels-2026-11-02/baseline \
  --scene=sparse-cm12-long-dam-break --frame=120 --repeats=12 \
  --active-dispatch=true --out=artifacts/nbflip-hot-kernels-2026-11-02/reproduce-indirect

# Full simulation with source snapshots (no runtime shader rewrite).
bash tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/benchmark-nbflip-hot-paired.ts reproduce \
  --baseline=artifacts/nbflip-hot-kernels-2026-11-02/baseline \
  --candidate=artifacts/nbflip-hot-kernels-2026-11-02/candidate-shape \
  --scene=nbflip-figure-9-teaser --steps=180 --from=25 \
  --out=artifacts/nbflip-hot-kernels-2026-11-02/paired
```

Historical source-rewrite recipes intentionally fail if their expected source changes. Use the frozen baseline with `--root` to replay them. Workgroup-only production checks use `--variants=aa,transfer192,aa`, which compares the parameterized candidate against the original 192-lane shape; `--traces=true` enables diagnostics and `--transfer=apic` covers the APIC path.

## Implemented change

`uniform-particle-brick.wgsl.ts` makes the transfer workgroup size and every dependent staging/particle/flush stride share the `PB_LANES` override. `UniformNarrowBandFlip` specializes it to 128 for FLIP and keeps 192 for APIC. The default shader value remains 192. There is no new buffer, worklist, dispatch, diagnostic mode, or numerical change. Fixed-point momentum accumulation and diagnostic counting are retained.

The diagnostic-specialized, aggregated-counter, advection, pressure and surface candidates remain experiments; they are not bundled into this patch. The 128-lane-only snapshot is `candidate-shape/`; the rejected diagnostic-specialized snapshot is `candidate-transfer/`.

## Final whole-scene results and validation

The 600-step teaser run uses UI device features (including `float32-filterable`) and discards the first 24 steps. Over 576 measured steps, mean complete simulation-step time is **124.087 → 121.883 ms (1.78% faster)**. Paired median savings are 2.022 ms; the trimmed mean saves 2.207 ms. Savings hold in both execution orders: 1.888 ms when baseline runs first and 2.520 ms when candidate runs first. Both arms allocate exactly 3,451,731,096 tracked GPU bytes at peak.

Both finish without GPU/pipeline errors and with identical volume sum 1,865,698.375, drift 0.0000527310 and front position 2.838542 m. Their independent trajectories differ: 984,162 vs 972,519 final particles, 8,160 vs 7,978 fine tiles, and residuals 0.12844 vs 0.26203. The measured end-to-end difference therefore includes trajectory-dependent work; exact same-state transfer comparisons separately establish numerical parity and kernel performance.

An unchanged-vs-unchanged UI-feature teaser control over 156 measured steps differs by 1.258 ms (1.03%) in the same direction, with different particle/tile counts and residuals. This is material trajectory/run variation: the observed 1.78% whole-scene saving is not a precise causal effect estimate. The kernel parity measurements and same-state frame comparison must be read alongside it. The workgroup-only full-frame replay (`teaser-shape-frame/`, 36 rotated trials) has baseline medians 146.242/146.598 ms and candidate 146.156 ms. The paired median saving against the mean of the controls is 0.285 ms (~0.2%). Means are 146.681/149.630 vs 147.270 ms, with a 254 ms baseline-B outlier and a 169 ms candidate outlier. The candidate therefore does not beat both controls by mean. This supports only a small/noisy full-frame benefit, not a large or precisely established frame-speed claim. The workgroup change is retained for its exact-output kernel improvement, positive observed long-scene results, effectively neutral repeated simple-dam result, and absence of added runtime storage or passes. All timing claims are limited to this M1 Max; other adapters remain unmeasured. The final UI-feature long-dam run measures 22.303 → 22.085 ms over 216 steps (0.98% faster). The simple dam run measures 26.080 → 26.339 ms over 216 steps (**0.99% slower**), despite a slightly lower unpaired candidate median. Its paired median and trimmed mean also worsen; it is not counted as a win. The repeat reverses direction: 25.793 → 25.528 ms (1.03% faster). Pooling the two equal-length simple-dam runs yields 25.9364 → 25.9334 ms, effectively unchanged (+0.012%). Both runs are retained; neither is selected as representative alone. APIC retains 192 lanes and passed diagnostic-enabled exact replay at frame 30. The product-source FLIP transfer passed exact diagnostic-enabled replay at frames 30/60: 128-lane control medians average 0.248/0.753 ms versus 192-lane 0.305/0.868 ms. Normal UI-feature teaser transfer also passed exact parity at frames 60/120: 128-lane control medians average 1.605/2.095 ms versus the 192-lane reference 2.212/2.731 ms, **27.4%/23.3% faster**. No declared writable output differed. These are simulation timings, not rendered FPS.


| UI-feature full simulation | Measured steps | Baseline ms/step | Candidate ms/step | Observed saving |
| --- | ---: | ---: | ---: | ---: |
| Teaser, 600 steps | 576 | 124.087 | 121.883 | +1.78% |
| Long dam, 240 steps | 216 | 22.303 | 22.085 | +0.98% |
| Simple dam, run 1 | 216 | 26.080 | 26.339 | -0.99% |
| Simple dam, run 2 | 216 | 25.793 | 25.528 | +1.03% |
| Teaser unchanged control | 156 | 121.910 | 120.651 | +1.03% |

Positive savings mean faster. The unchanged control contains no optimization and quantifies run/trajectory variation. Rendering is excluded from every row.


Validation: `npm run test:unit` passed (1,447 passed, 264 skipped, zero failures; 1,711 total). GPU parity experiments above run separately from the unit suite's environment-gated skips. Two harness-only type annotations were fixed. An incremental shared-checkout run encountered slot/page-pool API mismatches during concurrent work. **Fresh `npm run check:types -- --incremental false` passed both in the current checkout and in the frozen validation snapshot.** `git diff --check` also passed.
