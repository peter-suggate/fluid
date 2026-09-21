# Uniform Geometric on a large sparse scene — xctrace GPU capture

Successor to `xctrace-capture.md` (mini64, 64³). Same method (`uniform-volume`,
"Uniform Geometric"), same machine (Apple M1 Max, 4 GPU partitions), same
xctrace 27.0 (27A266a), Node v22.22.1, darwin 25.6.0, repo revision
`918fc273` (working tree dirty — see *Anomalies*).

The large scene is **`cm12-figure-7`**: the CM12 paper's ball-drop into an
empty sealed tank, **128×128×128**, 6.4 m cube, h = 0.05 m, a 20-cell-radius
ball released at y = 90. ≈1.6 % of the domain is liquid. It is the smallest
catalogued scene above the `max(nx,ny,nz) > 64` threshold at
`lib/methods/uniform/webgpu-uniform-reference.ts:974` that turns on
`volumePageConfig.work` — the compact 4³ tile work list, the indirect volume
dispatch and the tile-cached sharpening arm. Every number below is therefore in
the per-cell-bound paged regime that mini64 never enters.

Data only. No recommendations.

---

## 1. Commands

Reproduce (each was run in the foreground, one GPU process at a time, holding
the repo-wide WebGPU lease):

```bash
# Capture 2 — shipping graph, traced, NOT isolated (75 paper steps)
#   (driven through scratch launch-capture.sh; the equivalent single command)
xcrun xctrace record --template "Metal System Trace" \
  --output artifacts/xctrace-uniform-geometric-fig7-launch-plain/launch.trace \
  --no-prompt --target-stdout - \
  --env FLUID_SCENE=cm12-figure-7 --env FLUID_LANE=uniform-one-step \
  --env FLUID_METHOD=uniform-volume --env FLUID_EXPECT_GRID=128,128,128 \
  --env FLUID_MAX_DT=0.03333333333333333 --env FLUID_TARGET_S=2.5 \
  --env FLUID_ORACLE_STEPS=75 --env FLUID_EXPECT_EXACT_STEPS=75 \
  --env FLUID_WEBGPU_DAWN_FEATURES=skip_validation,use_user_defined_labels_in_backend \
  --env FLUID_PERFORMANCE_PROFILE=1 --env FLUID_GPU_COMMAND_AUDIT=1 \
  --env FLUID_QUALITY=balanced --env FLUID_TRIPWIRES=1 \
  --launch -- node --import tsx tools/run-webgpu-smoke-isolated-worker.ts

# Capture 3 — isolated Metal GPU counter run (the generalised profiler preset)
node --import tsx tools/profile-mini-dam-xctrace.ts \
  --uniform-mini-64 --method=uniform-volume \
  --uniform-scene=cm12-figure-7 --expect-grid=128,128,128 \
  --steps=40 --counter-seconds=5 --frames=2 \
  --out=artifacts/xctrace-uniform-geometric-fig7

# Limiter rebuild of the same trace (no new GPU run)
node --import tsx tools/profile-mini-dam-xctrace.ts \
  --uniform-mini-64 --method=uniform-volume \
  --uniform-scene=cm12-figure-7 --expect-grid=128,128,128 \
  --steps=40 --counter-seconds=5 --frames=2 --reuse-tables \
  --out=artifacts/xctrace-uniform-geometric-fig7-limiters

# Per-level multigrid labels (in-process pass timestamps, no xctrace)
FLUID_UNIFORM_MG_LEVEL_LABELS=1 FLUID_GPU_PASS_TIMESTAMPS=1 \
FLUID_GPU_ISOLATE_PASS_ENCODERS=1 FLUID_GPU_ISOLATE_PASS_LABELS=1 \
FLUID_GPU_PASS_TIMESTAMP_COMMAND_BUFFERS=2 \
FLUID_GPU_PASS_TIMESTAMP_QUERY_CAPACITY=4096 \
FLUID_GPU_PASS_TIMESTAMP_LABEL_PREFIXES=Uniform \
  node --import tsx tools/run-webgpu-smoke.ts    # once per scene
```

No `MTL_CAPTURE_ENABLED`, no `xctrace --window`, no browser, no render lanes,
no git operations.

## 2. Tool and harness edits

| file:line | edit | default behaviour |
|---|---|---|
| `tools/profile-mini-dam-xctrace.ts:150-162` | new `--uniform-scene`, `--uniform-lane`, `--expect-grid` flags, validated (`nx,ny,nz`) and required once a non-default scene is named | `--uniform-scene` defaults to `minimal-power-dam-break-64`, `--expect-grid` to `64,64,64`, `--uniform-lane` to `uniform-one-step` — `npm run profile:uniform-mini-dam-64-xctrace` unchanged |
| `tools/profile-mini-dam-xctrace.ts:465,466,471` | those three flags become `FLUID_SCENE`, `FLUID_LANE`, `FLUID_EXPECT_GRID` in the inherited lane environment (and reach the manifest at `:605`, the console line at `:1001`) | identical strings for the old defaults |
| `lib/harness/scene-webgpu-smoke-catalog.ts:44,686` | `"cm12-figure-7"` added to `sceneWebGPUSmokeIds` and a one-lane suite (`uniform-one-step`, paper 1/30 s, grid/finite acceptance only) | no existing lane touched; no test pins `sceneWebGPUSmokeIds` |
| `lib/harness/webgpu-smoke-executor.ts:10,507` | `uniformVolumeMethod` installed in `availableMethods` so `FLUID_METHOD=uniform-volume` is runnable on any catalogued scene | lanes select by authored id, so a lane that never names it is unaffected |
| `lib/methods/uniform/webgpu-uniform-pressure-multigrid.ts:27-28,476` | **the one permitted solver edit** — `MG_LEVEL_LABELS` (module-scope read of `FLUID_UNIFORM_MG_LEVEL_LABELS === "1"`, guarded by `typeof process !== "undefined"`) appends ` L<activeLevel>` to the multigrid compute-pass label | flag unset ⇒ the label is byte-identical to before |

## 3. Artifacts

| path | size | what |
|---|--:|---|
| `artifacts/xctrace-uniform-geometric-fig7-launch-plain/` | 47 MB | 75-step traced, non-isolated launch trace (capture 2) |
| `artifacts/xctrace-uniform-geometric-fig7/` | 261 MB | 40-step isolated Metal GPU counter run (capture 3) |
| `artifacts/xctrace-uniform-geometric-fig7-limiters/` | 47 MB | limiter rebuild of the same trace (17 retained counters) |

`artifacts/` is gitignored. Every exported XML was deleted immediately after
reduction; `find artifacts -name '*.xml'` returns only the `.trace` bundles'
own internal `schema.xml` files. Disk before and after each capture: **88 GiB
free of 926 GiB** (no capture moved the figure by a whole GiB).

## 4. Integrity panel

Read these before quoting any number below.

| | mini64 (prior report) | fig7 (this report) |
|---|--:|--:|
| grid | 64³ | 128³ |
| frames analysed (counter run) | 2 | **2** |
| which advances | budget-heavy pair (1452 passes/adv vs 947 run mean) | **advances 1–2 of 40** — verified: `gpuPassTimestamps.byLabel` for the first two command buffers gives `mgSmoothColour` 632/advance, identical to the xctrace task table |
| scene phase covered by those frames | steady dam collapse | **free fall** (ball released, not yet in contact) |
| frame anchor | `Uniform Sec. 3.3 rho-prime and face authority` | same |
| attribution | full, 82 exact buckets, 0 composite | full, **86 exact buckets, 0 composite**, composite share 8e-16 |
| counter run untraced baseline | 36.77 ms/adv | **65.22 ms/adv** |
| counter run traced wall | 79.98 ms/adv | **119.67 ms/adv** |
| **counter-run distortion** | **2.18×** | **1.83×** |
| counter series retained / source | 17 / 31 | 17 / 31 |
| counter timestamp stride | 18 | 18 |
| retained sample interval | 180 µs | 180 µs |
| exclusive coverage | 0.8595 | **0.9999** |
| contention (other processes on the GPU) | WindowServer 23.24 ms / 621 intervals, Codex 9.50 ms / 365 | **WindowServer 0.223 ms / 2, Cursor Helper 0.088 / 9, Codex 0.026 / 4 — negligible** |
| GPU slot detection confidence | 0.081 | 0.121 (still low; occupancy is read from the `Compute Occupancy` counter, not from slot counts) |
| CPU samples in the counter run | 0 | 0 (Metal GPU Counters template carries no time profile — CPU comes from capture 2) |
| shader-profiler rows in `summary.shaders` | 0 | 0 (the raw `metal-shader-profiler-intervals` table did return 14 089 rows for fig7, unlike mini64, but the reducer still produced no per-shader attribution) |

Isolation distortions that ride on every capture-3 number:

* 1 WebGPU pass = 1 Metal encoder (938 of each per advance) instead of Dawn's
  packed 27.1 compute encoders; encoder launch overhead is therefore paid 938
  times, and inter-encoder idle rises from 19.4 ms to **60.75 ms per advance**.
* The timestamp resolve adds **938 diagnostic blits and 3.28 ms per advance**.
* Traced-vs-untraced wall on the same binary: **1.83×**.

Capture-2 (shipping-graph) numbers carry no isolation distortion; their only
distortion is xctrace itself: **74.61 ms/advance traced vs 65.22–67.42 ms/advance
untraced ⇒ ≈1.12×**.

---

## B. Shipping graph (capture 2 — traced, not isolated)

75 paper steps, 74 anchor-delimited advances, **72 interior advances analysed**
(indices 1–72; index 0 is the bootstrap advance with 218 encoders and is
excluded). 4 251 labelled compute intervals.

### B.1 Wall baselines (untraced unless stated)

| run | scene | steps | ms/advance | passes/advance |
|---|---|--:|--:|--:|
| untraced, shipping env | mini64 | 12 | 38.75 | 869.5 |
| untraced, shipping env | mini64 | 60 | 34.10–36.48 (10 runs, 5 env arms × 2 reps) | 931.6–947.2 |
| untraced, isolated | mini64 | 12 | 70.33 (1.81×) | 869.5 |
| traced, not isolated | mini64 | 12 | 48.17 | 869.5 |
| untraced, shipping env | **fig7** | 12 | **76.92** | 938.0 |
| untraced, shipping env | **fig7** | 40 | **65.22** | 986.3 |
| untraced, shipping env | **fig7** | 60 | **63.88–67.42** | 991.7 |
| untraced, isolated | **fig7** | 12 | **97.08** (1.26×) | 938.0 |
| traced, not isolated | **fig7** | 75 | **74.61** | 980.9 |
| traced + counters, isolated | **fig7** | 40 | **119.67** (1.83×) | 986.3 |

### B.2 GPU busy, idle and encoders per advance

| | mini64 | fig7 | fig7/mini64 |
|---|--:|--:|--:|
| frame wall (GPU-timeline, traced) | 45.34 ms | **73.97 ms** | 1.63× |
| GPU busy | 21.79 ms | **55.50 ms** | 2.55× |
| **busy share of wall** | **48.1 %** | **75.0 %** | |
| interior idle | 23.6 ms | **19.38 ms** | |
| largest single gap | — | **17.94 ms**, at 54.5 ms into the advance | |
| tail gap | — | **0.013 ms** | |
| Metal encoders / advance | 52.2 (26.1 compute + 26.1 blit) | **54.2 (27.1 compute + 27.1 blit)** | |
| WebGPU compute passes / advance | 755.4 | **983.7** | 1.30× |
| **passes per Metal compute encoder** | 28.93 | **36.28** | |
| command buffers / advance | 2 | **2** | |
| pressure encoders / advance | 12 | **12.1** | |
| pressure GPU ms / advance | 8.96 | **22.46** | 2.51× |
| pressure passes / advance | 652 | **877.7** | |
| `mgSmoothColour` passes / advance | 488 | **667** | |
| pressure inter-encoder gap (median / total) | 49.3 µs / 0.565 ms | **53.8 µs / 0.607 ms** | |
| pressure encoder duration (median / p90 / max) | 124 / 1520 / 5180 µs | **601 / 5670 / 10 030 µs** | |

Whole-trace encoder census (all 75 advances, including bootstrap): 4 248
encoders — 2 118 compute, 2 130 blit, **1 823 of the 2 118 compute encoders
carry more than one WebGPU pass label**; 73 577 passes implied by the labels.
Command-buffer submissions: 89, of which 74 carry exactly 54 encoders.

### B.3 Scene phases

| phase | advances | wall ms | busy ms | busy % | idle ms | largest gap ms | passes/adv | pressure ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| free fall | 1–24 (n=23) | 66.57 | 48.20 | 72.4 % | 19.28 | 17.87 | 965.0 | 21.32 |
| impact | 25–39 (n=15) | 76.27 | 57.23 | 75.0 % | 19.96 | 18.53 | 1094.0 | 22.53 |
| transition | 40–43 (n=4) | 80.46 | 58.30 | 72.5 % | 23.13 | 21.62 | 1287.5 | 23.80 |
| spread | 44–72 (n=29) | 77.78 | 60.01 | 77.2 % | 18.66 | 17.22 | 967.5 | 23.14 |
| **all interior** | **1–72** | **73.97** | **55.50** | **75.0 %** | **19.38** | **17.94** | **1010.8** | **22.46** |

Pass count peaks in the four-advance transition (1287.5/advance) because the
pressure cycle budget spikes there; GPU busy keeps climbing into spread even as
the pass count falls back, i.e. the spread advances run *fewer, larger* passes.

### B.4 Per-encoder table, shipping graph (fig7, ms/advance descending)

Labels are Dawn's concatenation; `…(+N)` is the number of further WebGPU passes
merged into the same Metal encoder.

```
   9.877 ms   1.04 enc/adv    25.0 pass/adv  med  9740 us  Uniform Sec. 3.3 seed active front …(+23)
   9.425 ms   1.00 enc/adv   643.0 pass/adv  med  9460 us  Uniform CM11a mgMeasureFineResidual …(+642)
   5.480 ms   1.00 enc/adv    22.0 pass/adv  med  5535 us  Uniform CM11a mgBuildFinestTopology …(+21)
   4.665 ms   8.00 enc/adv   160.0 pass/adv  med   599 us  Uniform CM11a mgMeasureFineResidual …(+19)
   4.578 ms   1.00 enc/adv    14.0 pass/adv  med  4465 us  Total surface volume: begin …(+13)
   3.417 ms   2.00 enc/adv     6.0 pass/adv  med  1910 us  uvNormalizeRows …(+2)
   2.829 ms   1.00 enc/adv     5.0 pass/adv  med  2830 us  Uniform Geometric two-level uvTwoLevelSeed …(+4)
   2.307 ms   1.00 enc/adv     5.0 pass/adv  med  2555 us  uvNormalizeRows …(+4)
   2.278 ms   1.00 enc/adv     3.0 pass/adv  med  2400 us  Uniform pressure projection …(+2)
   1.680 ms   1.00 enc/adv     1.0 pass/adv  med  1750 us  Uniform semi-Lagrangian velocity advection and body forces
   1.620 ms   1.00 enc/adv    34.0 pass/adv  med  1345 us  Cache sharpening cell geometry …(+33)
   1.595 ms   1.00 enc/adv     3.0 pass/adv  med  1690 us  uvBuildEdges …(+2)
   1.486 ms   1.00 enc/adv     3.0 pass/adv  med  1480 us  Phi support census …(+2)
   1.013 ms   1.00 enc/adv     1.0 pass/adv  med  1010 us  Refresh corrected surface targets
   0.834 ms   1.00 enc/adv     1.0 pass/adv  med   856 us  Blit Command 16
   0.822 ms   1.00 enc/adv     2.0 pass/adv  med   827 us  Surface-deficit page sums …(+1)
   0.323 ms   0.07 enc/adv    44.7 pass/adv  med  3440 us  Uniform CM11a mgMeasureFineResidual …(+643)
   0.288 ms   1.00 enc/adv     7.0 pass/adv  med   291 us  Uniform CM11a mgMeasureFineResidual …(+6)
   0.111 ms   1.00 enc/adv     1.0 pass/adv  med   111 us  Classify 4h sharpening work
   (remaining rows are Blit Commands and the two page-marking encoders, all < 0.15 ms)
```

Which stages share an encoder (advance 38, wall 72.27 ms, busy 55.16 ms;
`+gap` is the idle *before* the encoder):

| start ms | gap µs | duration µs | passes | encoder contents (first … last) |
|--:|--:|--:|--:|---|
| 0.000 | 0 | 2870 | 5 | `uvTwoLevelSeed` & DilateX & DilateY & DilateZ & `Sec. 3.3 rho-prime and face authority` |
| 2.919 | 49 | 9000 | 24 | `Sec. 3.3 seed active front` & prepare initial dispatch & FIM 1 … `Redistance page vertex phi` |
| 11.979 | 15 | 47 | 2 | `uvMarkTransportPages` & `Compact volume pages` |
| 12.234 | 33 | 1840 | 3 | `uvBuildEdges` & `uvFinishDonorSums` & `uvFallback` |
| 14.259 | 32 | 1910 | 3 | `uvNormalizeRows` & `uvFinishDonorSums` & `uvNormalizeDonors` |
| 16.364 | 33 | 1910 | 3 | (same triple again) |
| 18.474 | 37 | 2550 | 5 | `uvNormalizeRows` & `uvFinishDonorSums` & `uvNormalizeDonors` & `uvGather` & `Surface volume capacities` |
| 21.071 | 47 | 4200 | 14 | `Total surface volume: begin` & seed & dilate … apply |
| 25.377 | 23 | 1020 | 1 | `Refresh corrected surface targets` |
| 26.463 | 15 | 111 | 1 | `Classify 4h sharpening work` |
| 26.625 | 15 | 52 | 2 | `uvMarkSharpenPages` & `Compact volume pages` |
| 26.791 | 33 | 1750 | 34 | `Cache sharpening cell geometry` & face geometry & `uvPrepareSharpen` … `uvCommitSharpen` |
| 28.624 | 16 | 1910 | 1 | `Uniform semi-Lagrangian velocity advection and body forces` |
| 30.589 | 11 | 829 | 2 | `Surface-deficit page sums` & `Surface-deficit global balance` |
| 31.456 | 38 | 5540 | 22 | `mgBuildFinestTopology` & `mgBuildFinestRhs` & `mgDownsampleTopology` … `mgBakeCoefficients` |
| 37.052 | 12 | 8860 | **643** | `mgMeasureFineResidual` & `mgCheckCycleConvergence` & `mgCopyPressure` … `mgCopyPressure` |
| 45.968 … 50.966 | 10–17 each | 298–624 each | 20 each (7 in the last) | eight more V-cycle encoders, each `mgMeasureFineResidual` & `mgCheckCycleConvergence` & `mgSaveAccepted` … `mgSmoothColour` |
| 51.319 | 16 | 2540 | 3 | `Uniform pressure projection` & `Uniform Geometric surface publication` & `Uniform diagnostics reduction` |
| **70.683** | **16 824** | 6 | 1 | `Blit Command 0` — the next command buffer |
| 70.711 | 22 | 1510 | 3 | `Phi support census` & `Phi support reduction` & `Phi support closure` |

**The idle is one gap, not many.** Interior gaps between solver encoders are
10–49 µs. A single **16.8–17.9 ms** hole sits immediately after
`Uniform diagnostics reduction` (the last pass of the solver command buffer) and
before `Blit Command 0` of the next command buffer. Tail gap is 0.013 ms.
Section D identifies the mechanism.

### B.5 Main-thread CPU (capture 2, `time-profile` with `--stacks`)

Window: 5 326 ms of anchored stepping covering 72 advances.

| | mini64 | fig7 |
|---|--:|--:|
| main-thread CPU per advance | 23.9 ms | **32.81 ms** |
| running share of wall | — | **44.3 %** |
| `__kdebug_trace64` (the trace's own cost) | 28 % of CPU | **16.2 %** of CPU |
| CPU per advance discounting `__kdebug_trace64` | ≈17 ms | **≈27.5 ms** |

Groups (stack-classified, same classifier as the mini64 report):

| group | mini64 | fig7 |
|---|--:|--:|
| Dawn native (validation, encoding, state) | 37.9 % | **40.2 %** |
| Dawn Metal backend / driver | 41.2 % | **31.9 %** |
| Node / V8 / JS | 21.0 % | **27.7 %** |

Top leaves (fig7): `__kdebug_trace64` 16.2 %, `kevent` 3.4 %,
`<deduplicated_symbol>` 3.2 %, `_xzm_free` 3.0 %,
`dawn::native::ErrorMonad::IsError()` 2.0 %, `_xzm_xzone_malloc_tiny` 1.7 %,
`__kdebug_trace_string` 1.4 %, `pthread_mutex_lock` 1.2 %,
`BindGroupTracker::ApplyBindGroupImpl` 0.8 %.

---

## 5. Isolated counter run (capture 3) — full per-task table

**86 tasks, 938 firings/advance, 51.684 ms attributed busy/advance.** Frames
analysed: the first two advances after gate release (free fall). 36 of the 86
tasks received at least one counter sample at the 180 µs retained interval;
tasks with fewer than one sample show `—`. Distortion on every row: encoder
isolation (938 encoders instead of 27.1), +3.28 ms/advance of resolve blits,
1.83× wall.

| # | task | family | firings/adv | ms/adv | % busy | mean us | occ % | ALU % | LLC % | top limiter | 2nd limiter | read GB/s | write GB/s | imbalance | counter samples |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|---|---|--:|--:|--:|--:|
| 1 | Uniform CM11a mgSmoothColour | pressure V-cycles | 632 | 14.5775 | 28.21% | 23.1 | 25.7 | 12.0 | 12.4 | MMU 22.7% | GPU Last Level Cache 17.6% | 32.32 | 5.43 | 1.03 | 52 |
| 2 | Total surface volume: measure | surface publication | 2 | 3.4350 | 6.65% | 1717.5 | 20.7 | 27.7 | 52.9 | GPU Last Level Cache 52.9% | ALU 29.4% | 17.48 | 10.94 | 1.00 | 10 |
| 3 | Uniform CM11a mgBuildFinestTopology | pressure setup | 1 | 3.4050 | 6.59% | 3405.0 | 22.7 | 31.4 | 4.9 | ALU 33.4% | Threadgroup/Imageblock Load 11.0% | 1.48 | 3.29 | 1.00 | 13 |
| 4 | Advect page vertex phi | phi transport + redistance | 1 | 3.3000 | 6.39% | 3300.0 | 20.8 | 36.5 | 9.0 | ALU 40.3% | GPU Last Level Cache 9.0% | 2.28 | 0.33 | 1.00 | 5 |
| 5 | Uniform Sec. 3.3 rho-prime and face authority | extension (Sec. 3.3 FIM) | 1 | 2.1300 | 4.12% | 2130.0 | 25.0 | 45.4 | 4.7 | ALU 49.8% | GPU Last Level Cache 4.7% | 2.21 | 4.73 | 1.00 | 6 |
| 6 | uvNormalizeRows | conservative volume gather | 3 | 1.8212 | 3.52% | 607.1 | 45.2 | 7.8 | 81.0 | GPU Last Level Cache 84.8% | Buffer Read 79.1% | 90.69 | 59.55 | 1.00 | 4 |
| 7 | Uniform semi-Lagrangian velocity advection and body forces | velocity advection + two-level maps | 1 | 1.6500 | 3.19% | 1650.0 | 14.5 | 27.9 | 14.0 | Texture Sample 50.0% | Texture Filtering 48.4% | 3.24 | 3.14 | 1.00 | 4 |
| 8 | Phi support census | surface publication | 1 | 1.3700 | 2.65% | 1370.0 | 10.3 | 15.0 | 2.3 | ALU 15.2% | Texture Sample 8.0% | 4.75 | 0.20 | 1.00 | 2 |
| 9 | uvFinishDonorSums | conservative volume gather | 4 | 1.2416 | 2.40% | 310.4 | 58.5 | 7.1 | 40.5 | Buffer Read 83.9% | MMU 71.9% | 110.42 | 2.25 | 1.01 | 6 |
| 10 | Uniform CM11a mgBuildFinestRhs | pressure setup | 1 | 1.1350 | 2.20% | 1135.0 | 17.8 | 17.0 | 14.3 | ALU 17.7% | GPU Last Level Cache 14.3% | 5.47 | 6.54 | 1.04 | 4 |
| 11 | Uniform pressure projection | projection | 1 | 1.1050 | 2.14% | 1105.0 | 16.2 | 21.2 | 15.8 | ALU 22.1% | GPU Last Level Cache 15.8% | 4.85 | 17.53 | 1.03 | 7 |
| 12 | uvNormalizeDonors | conservative volume gather | 3 | 1.0207 | 1.98% | 340.2 | — | — | — | — | — | — | — | — | 0 |
| 13 | Refresh corrected surface targets | surface publication | 1 | 1.0150 | 1.96% | 1015.0 | 18.7 | 23.8 | 4.2 | Texture Sample 81.0% | Texture Filtering 40.5% | 4.75 | 2.17 | 1.00 | 2 |
| 14 | uvBuildEdges | conservative volume gather | 1 | 0.9503 | 1.84% | 950.3 | 27.0 | 32.9 | 50.4 | GPU Last Level Cache 50.5% | ALU 36.8% | 8.63 | 25.67 | 1.00 | 2 |
| 15 | Uniform CM11a mgMeasureFineResidual | pressure finish + convergence | 11 | 0.7337 | 1.42% | 66.7 | 21.4 | 9.8 | 8.4 | MMU 17.9% | GPU Last Level Cache 13.6% | 24.54 | 0.06 | 1.18 | 3 |
| 16 | Uniform Sec. 3.3 hierarchy restrict 1 | extension (Sec. 3.3 FIM) | 1 | 0.7041 | 1.36% | 704.1 | — | — | — | — | — | — | — | — | 0 |
| 17 | Uniform Geometric two-level uvTwoLevelSeed | velocity advection + two-level maps | 1 | 0.6803 | 1.32% | 680.3 | 33.3 | 11.0 | 51.1 | Texture Cache 99.8% | Texture Sample 99.5% | 152.16 | 0.06 | 1.00 | 4 |
| 18 | Uniform Sec. 3.3 FIM indirect update 2 | extension (Sec. 3.3 FIM) | 1 | 0.6522 | 1.26% | 652.2 | 11.6 | 15.8 | 29.8 | GPU Last Level Cache 30.1% | ALU 16.6% | 22.79 | 4.50 | 1.01 | 2 |
| 19 | Surface-deficit page sums | surface publication | 1 | 0.6246 | 1.21% | 624.6 | 15.6 | 23.9 | 5.8 | ALU 24.7% | Texture Sample 13.6% | 6.41 | 0.06 | 1.04 | 4 |
| 20 | Total surface volume: dilate | surface publication | 4 | 0.6076 | 1.18% | 151.9 | 75.6 | 70.2 | 41.3 | ALU 95.9% | MMU 44.0% | 108.57 | 0.06 | 1.00 | 5 |
| 21 | Uniform CM11a mgResidual | pressure V-cycles | 22 | 0.5822 | 1.13% | 26.5 | 28.3 | 15.9 | 18.2 | MMU 20.8% | GPU Last Level Cache 19.7% | 48.06 | 7.95 | 1.01 | 1 |
| 22 | Redistance page vertex phi | phi transport + redistance | 1 | 0.5596 | 1.08% | 559.6 | 28.0 | 55.7 | 0.1 | ALU 96.3% | Texture Sample 34.9% | 0.04 | 0.10 | 1.01 | 1 |
| 23 | Uniform CM11a mgProlongateAdd | pressure V-cycles | 21 | 0.4571 | 0.88% | 21.8 | 47.4 | 54.7 | 10.5 | ALU 67.0% | Texture Sample 58.1% | 15.61 | 11.23 | 1.00 | 1 |
| 24 | uvGather | conservative volume gather | 1 | 0.4129 | 0.80% | 412.9 | 16.5 | 29.1 | 38.5 | GPU Last Level Cache 38.6% | Buffer Read 32.3% | 18.47 | 12.66 | 1.00 | 3 |
| 25 | uvProposeSharpen | sharpening | 8 | 0.4082 | 0.79% | 51.0 | 17.2 | 3.7 | 38.2 | GPU Last Level Cache 38.3% | Buffer Read 35.4% | 42.71 | 20.21 | 1.05 | 4 |
| 26 | Uniform Sec. 3.3 FIM indirect update 1 | extension (Sec. 3.3 FIM) | 1 | 0.4039 | 0.78% | 403.9 | — | — | — | — | — | — | — | — | 0 |
| 27 | uvCommitSharpen | sharpening | 8 | 0.3536 | 0.68% | 44.2 | 21.5 | 24.4 | 48.2 | Buffer Read 64.8% | GPU Last Level Cache 49.1% | 96.63 | 6.58 | 1.00 | 3 |
| 28 | Uniform CM11a mgSaveAccepted | pressure finish + convergence | 9 | 0.3494 | 0.68% | 38.8 | — | — | — | — | — | — | — | — | 0 |
| 29 | Uniform CM11a mgBakeCoefficients | pressure setup | 7 | 0.3371 | 0.65% | 48.1 | — | — | — | — | — | — | — | — | 0 |
| 30 | Uniform CM11a mgRestoreRejected | pressure finish + convergence | 10 | 0.3272 | 0.63% | 32.7 | 7.9 | 5.4 | 0.3 | ALU 5.4% | Buffer Read 1.0% | 0.09 | 0.06 | 1.13 | 1 |
| 31 | Uniform diagnostics reduction | other | 1 | 0.3270 | 0.63% | 327.0 | 41.3 | 21.6 | 25.3 | Texture Sample 78.4% | Texture Filtering 37.9% | 47.35 | 0.06 | 1.00 | 4 |
| 32 | Uniform Sec. 3.3 hierarchy prolong 7 | extension (Sec. 3.3 FIM) | 1 | 0.3063 | 0.59% | 306.3 | — | — | — | — | — | — | — | — | 0 |
| 33 | uvLimitSharpen | sharpening | 8 | 0.2721 | 0.53% | 34.0 | 16.3 | 7.2 | 38.8 | Buffer Read 50.0% | GPU Last Level Cache 39.3% | 63.28 | 5.91 | 1.00 | 1 |
| 34 | Uniform CM11a mgRestrictResidual | pressure V-cycles | 27 | 0.2701 | 0.52% | 10.0 | — | — | — | — | — | — | — | — | 0 |
| 35 | Uniform CM11a mgDownsampleTopology | pressure setup | 6 | 0.2599 | 0.50% | 43.3 | 24.2 | 5.3 | 23.1 | Texture Sample 63.7% | Texture Cache 62.3% | 56.20 | 4.66 | 1.00 | 2 |
| 36 | Uniform CM11a mgDownsampleSubtract | pressure V-cycles | 21 | 0.2416 | 0.47% | 11.5 | 42.4 | 14.8 | 50.2 | Texture Sample 93.3% | Texture Cache 70.1% | 116.33 | 6.00 | — | 1 |
| 37 | Uniform CM11a mgProlongateAssign | pressure V-cycles | 6 | 0.2394 | 0.46% | 39.9 | — | — | — | — | — | — | — | — | 0 |
| 38 | Uniform Sec. 3.3 seed active front | extension (Sec. 3.3 FIM) | 1 | 0.2351 | 0.46% | 235.1 | — | — | — | — | — | — | — | — | 0 |
| 39 | Uniform CM11a mgExtrapolatePhiOneCell | pressure setup | 7 | 0.2243 | 0.43% | 32.0 | 41.9 | 21.1 | 27.3 | MMU 38.5% | GPU Last Level Cache 34.1% | 65.19 | 11.92 | — | 1 |
| 40 | Uniform Sec. 3.3 hierarchy prolong 6 | extension (Sec. 3.3 FIM) | 1 | 0.2006 | 0.39% | 200.6 | — | — | — | — | — | — | — | — | 0 |
| 41 | Surface-deficit global balance | surface publication | 1 | 0.1782 | 0.34% | 178.2 | 0.1 | 0.1 | 0.1 | GPU Last Level Cache 0.1% | MMU 0.1% | 0.38 | 0.06 | 4.00 | 1 |
| 42 | uvPrepareSharpen | sharpening | 8 | 0.1769 | 0.34% | 22.1 | — | — | — | — | — | — | — | — | 0 |
| 43 | Uniform CM11a mgDownsampleMinimum | pressure setup | 6 | 0.1479 | 0.29% | 24.7 | — | — | — | — | — | — | — | — | 0 |
| 44 | Uniform CM11a mgClearPressure | pressure setup | 22 | 0.1385 | 0.27% | 6.3 | 0.3 | 0.1 | 0.2 | GPU Last Level Cache 0.1% | ALU 0.1% | 0.06 | 0.15 | 1.02 | 2 |
| 45 | Total surface volume: seed | surface publication | 1 | 0.1365 | 0.26% | 136.5 | — | — | — | — | — | — | — | — | 0 |
| 46 | Uniform CM11a mgCopyPressure | pressure setup | 3 | 0.1323 | 0.26% | 44.1 | — | — | — | — | — | — | — | — | 0 |
| 47 | Uniform Geometric surface publication | surface publication | 1 | 0.1295 | 0.25% | 129.5 | 17.9 | 13.7 | 6.5 | Texture Sample 18.8% | ALU 15.3% | 6.35 | 5.13 | — | 1 |
| 48 | Total surface volume: metric | surface publication | 1 | 0.1135 | 0.22% | 113.5 | — | — | — | — | — | — | — | — | 0 |
| 49 | Classify 4h sharpening work | sharpening | 1 | 0.1107 | 0.21% | 110.7 | 32.5 | 53.8 | 30.8 | Texture Sample 100.0% | ALU 57.5% | 40.65 | 0.06 | 1.00 | 1 |
| 50 | Phi support reduction | surface publication | 1 | 0.1070 | 0.21% | 107.0 | — | — | — | — | — | — | — | — | 0 |
| 51 | Uniform Sec. 3.3 hierarchy restrict 2 | extension (Sec. 3.3 FIM) | 1 | 0.1058 | 0.21% | 105.8 | — | — | — | — | — | — | — | — | 0 |
| 52 | Uniform CM11a mgSolveCoarsest | pressure V-cycles | 7 | 0.1047 | 0.20% | 15.0 | — | — | — | — | — | — | — | — | 0 |
| 53 | Cache sharpening face geometry | sharpening | 1 | 0.0895 | 0.17% | 89.5 | — | — | — | — | — | — | — | — | 0 |
| 54 | Total surface volume: apply | surface publication | 1 | 0.0788 | 0.15% | 78.8 | — | — | — | — | — | — | — | — | 0 |
| 55 | Uniform CM11a mgAddPressure | pressure finish + convergence | 1 | 0.0605 | 0.12% | 60.5 | — | — | — | — | — | — | — | — | 0 |
| 56 | Surface volume capacities | surface publication | 1 | 0.0584 | 0.11% | 58.4 | — | — | — | — | — | — | — | — | 0 |
| 57 | Total surface volume: reduce | surface publication | 2 | 0.0573 | 0.11% | 28.6 | — | — | — | — | — | — | — | — | 0 |
| 58 | Uniform CM11a mgCheckCycleConvergence | pressure finish + convergence | 10 | 0.0572 | 0.11% | 5.7 | — | — | — | — | — | — | — | — | 0 |
| 59 | Uniform Sec. 3.3 resolve converged front | extension (Sec. 3.3 FIM) | 1 | 0.0554 | 0.11% | 55.4 | — | — | — | — | — | — | — | — | 0 |
| 60 | Total surface volume: solve | surface publication | 2 | 0.0527 | 0.10% | 26.4 | — | — | — | — | — | — | — | — | 0 |
| 61 | Uniform CM11a mgShiftMinimum | pressure setup | 1 | 0.0514 | 0.10% | 51.4 | — | — | — | — | — | — | — | — | 0 |
| 62 | Uniform Sec. 3.3 hierarchy restrict 6 | extension (Sec. 3.3 FIM) | 1 | 0.0459 | 0.09% | 45.9 | — | — | — | — | — | — | — | — | 0 |
| 63 | Uniform Sec. 3.3 hierarchy restrict 4 | extension (Sec. 3.3 FIM) | 1 | 0.0393 | 0.08% | 39.3 | — | — | — | — | — | — | — | — | 0 |
| 64 | Uniform Sec. 3.3 hierarchy restrict 3 | extension (Sec. 3.3 FIM) | 1 | 0.0390 | 0.07% | 39.0 | — | — | — | — | — | — | — | — | 0 |
| 65 | Uniform Sec. 3.3 hierarchy restrict 5 | extension (Sec. 3.3 FIM) | 1 | 0.0387 | 0.07% | 38.7 | 0.0 | 0.0 | 0.0 | GPU Last Level Cache 0.0% | MMU 0.0% | 0.02 | 0.06 | 4.00 | 1 |
| 66 | uvMarkTransportPages | phi transport + redistance | 1 | 0.0373 | 0.07% | 37.3 | — | — | — | — | — | — | — | — | 0 |
| 67 | uvMarkSharpenPages | sharpening | 1 | 0.0373 | 0.07% | 37.3 | — | — | — | — | — | — | — | — | 0 |
| 68 | Uniform Sec. 3.3 hierarchy prolong 5 | extension (Sec. 3.3 FIM) | 1 | 0.0346 | 0.07% | 34.6 | — | — | — | — | — | — | — | — | 0 |
| 69 | uvFallback | conservative volume gather | 1 | 0.0344 | 0.07% | 34.4 | 8.4 | 6.4 | 2.4 | ALU 9.0% | Buffer Read 3.0% | 6.35 | 0.06 | 1.01 | 1 |
| 70 | Uniform Sec. 3.3 publish 4h face table | surface publication | 1 | 0.0325 | 0.06% | 32.5 | — | — | — | — | — | — | — | — | 0 |
| 71 | Uniform Sec. 3.3 hierarchy prolong 2 | extension (Sec. 3.3 FIM) | 1 | 0.0285 | 0.06% | 28.4 | — | — | — | — | — | — | — | — | 0 |
| 72 | Cache sharpening cell geometry | sharpening | 1 | 0.0284 | 0.06% | 28.4 | — | — | — | — | — | — | — | — | 0 |
| 73 | Uniform Sec. 3.3 hierarchy prolong 4 | extension (Sec. 3.3 FIM) | 1 | 0.0281 | 0.05% | 28.1 | — | — | — | — | — | — | — | — | 0 |
| 74 | Uniform Sec. 3.3 hierarchy prolong 3 | extension (Sec. 3.3 FIM) | 1 | 0.0279 | 0.05% | 27.9 | — | — | — | — | — | — | — | — | 0 |
| 75 | Uniform Sec. 3.3 hierarchy restrict 7 | extension (Sec. 3.3 FIM) | 1 | 0.0273 | 0.05% | 27.3 | — | — | — | — | — | — | — | — | 0 |
| 76 | Compact volume pages | conservative volume gather | 2 | 0.0239 | 0.05% | 12.0 | — | — | — | — | — | — | — | — | 0 |
| 77 | Total surface volume: begin | surface publication | 1 | 0.0194 | 0.04% | 19.4 | — | — | — | — | — | — | — | — | 0 |
| 78 | Phi support closure | surface publication | 1 | 0.0178 | 0.03% | 17.8 | — | — | — | — | — | — | — | — | 0 |
| 79 | Uniform Geometric two-level uvTwoLevelDilateX | velocity advection + two-level maps | 1 | 0.0150 | 0.03% | 15.0 | — | — | — | — | — | — | — | — | 0 |
| 80 | Uniform Sec. 3.3 hierarchy prolong 1 | extension (Sec. 3.3 FIM) | 1 | 0.0145 | 0.03% | 14.5 | — | — | — | — | — | — | — | — | 0 |
| 81 | Uniform Geometric two-level uvTwoLevelDilateZ | velocity advection + two-level maps | 1 | 0.0133 | 0.03% | 13.3 | — | — | — | — | — | — | — | — | 0 |
| 82 | Uniform Geometric two-level uvTwoLevelDilateY | velocity advection + two-level maps | 1 | 0.0121 | 0.02% | 12.1 | — | — | — | — | — | — | — | — | 0 |
| 83 | Uniform Sec. 3.3 prepare initial active dispatch | extension (Sec. 3.3 FIM) | 1 | 0.0054 | 0.01% | 5.4 | — | — | — | — | — | — | — | — | 0 |
| 84 | Uniform Sec. 3.3 prepare active dispatch 2 | extension (Sec. 3.3 FIM) | 1 | 0.0054 | 0.01% | 5.4 | — | — | — | — | — | — | — | — | 0 |
| 85 | Uniform Sec. 3.3 prepare active dispatch 3 | extension (Sec. 3.3 FIM) | 1 | 0.0051 | 0.01% | 5.1 | — | — | — | — | — | — | — | — | 0 |
| 86 | Uniform CM11a mgFinishSafety | pressure finish + convergence | 1 | 0.0039 | 0.01% | 3.9 | — | — | — | — | — | — | — | — | 0 |

### 5.1 Stage-family aggregates (fig7, isolated)

| stage family | labels | firings/adv | ms/adv | % of busy | ms-weighted occ % | ms-weighted ALU % |
|---|--:|--:|--:|--:|--:|--:|
| pressure V-cycles | 7 | 736 | 16.473 | 31.87% | 26.7 | 13.4 |
| surface publication | 17 | 23 | 8.034 | 15.54% | 22.0 | 27.1 |
| pressure setup | 9 | 54 | 5.831 | 11.28% | 21.9 | 25.6 |
| conservative volume gather | 7 | 15 | 5.505 | 10.65% | 42.1 | 14.9 |
| extension (Sec. 3.3 FIM) | 22 | 22 | 5.133 | 9.93% | 21.6 | 37.9 |
| phi transport + redistance | 3 | 3 | 3.897 | 7.54% | 21.8 | 39.3 |
| velocity advection + two-level maps | 5 | 5 | 2.371 | 4.59% | 20.0 | 23.0 |
| pressure finish + convergence | 6 | 42 | 1.532 | 2.96% | 17.2 | 8.4 |
| sharpening | 8 | 36 | 1.477 | 2.86% | 19.8 | 15.8 |
| projection | 1 | 1 | 1.105 | 2.14% | 16.2 | 21.2 |
| other | 1 | 1 | 0.327 | 0.63% | 41.3 | 21.6 |
| **total (attributed)** | 86 | 938 | 51.684 | 100.00% | 26.1 | 20.5 |

### 5.2 Per-firing duration buckets

| per-firing duration | mini64 firings/adv | mini64 ms/adv | mini64 % busy | fig7 firings/adv | fig7 ms/adv | fig7 % busy |
|---|--:|--:|--:|--:|--:|--:|
| <20 us | 1277 | 14.209 | 36.0% | 586 | 6.505 | 12.6% |
| 20-50 us | 100.5 | 3.528 | 8.9% | 259.5 | 8.245 | 16.0% |
| 50-200 us | 50 | 5.153 | 13.0% | 56 | 7.237 | 14.0% |
| 200 us - 1 ms | 20.5 | 9.257 | 23.4% | 26.5 | 11.153 | 21.6% |
| >1 ms | 4 | 7.371 | 18.6% | 10 | 18.545 | 35.9% |

| pressure-only duration | mini64 firings/adv | mini64 ms/adv | fig7 firings/adv | fig7 ms/adv |
|---|--:|--:|--:|--:|
| <20 us | 1255.5 | 13.912 | 575.5 | 6.383 |
| 20-50 us | 61.5 | 2.249 | 217 | 6.843 |
| 50-200 us | 26 | 2.760 | 32.5 | 4.958 |
| 200 us - 1 ms | 6.5 | 2.306 | 5 | 1.112 |
| >1 ms | 1.5 | 2.537 | 3 | 5.645 |
| **total** | 1351 | 23.764 | 833 | 24.941 |

---

## 6. Side-by-side: mini64 vs fig7 (isolated regime)

| stage family | mini64 ms/adv | mini64 % busy | fig7 ms/adv | fig7 % busy | fig7/mini64 | mini64 firings/adv | fig7 firings/adv |
|---|--:|--:|--:|--:|--:|--:|--:|
| pressure V-cycles | 18.437 | 48.18% | 16.473 | 31.87% | 0.89x | 1220.5 | 736 |
| surface publication | 2.196 | 5.74% | 8.034 | 15.54% | 3.66x | 23 | 23 |
| pressure setup | 1.965 | 5.13% | 5.831 | 11.28% | 2.97x | 80 | 54 |
| conservative volume gather | 4.614 | 12.06% | 5.505 | 10.65% | 1.19x | 15 | 15 |
| extension (Sec. 3.3 FIM) | 2.232 | 5.83% | 5.133 | 9.93% | 2.30x | 20 | 22 |
| phi transport + redistance | 3.492 | 9.12% | 3.897 | 7.54% | 1.12x | 3 | 3 |
| velocity advection + two-level maps | 1.454 | 3.80% | 2.371 | 4.59% | 1.63x | 5 | 5 |
| pressure finish + convergence | 0.631 | 1.65% | 1.532 | 2.96% | 2.43x | 49.5 | 42 |
| sharpening | 1.659 | 4.34% | 1.477 | 2.86% | 0.89x | 34 | 36 |
| projection | 1.529 | 3.99% | 1.105 | 2.14% | 0.72x | 1 | 1 |
| other | 0.057 | 0.15% | 0.327 | 0.63% | 5.74x | 1 | 1 |
| **total attributed busy** | 38.267 | 100% | 51.684 | 100% | 1.35x | 1685 | 938 |

Shipping-regime headline, both scenes:

| | mini64 | fig7 |
|---|--:|--:|
| untraced wall / advance (shipping env, 60 steps) | 34.10–36.48 ms | 63.88–67.42 ms |
| GPU busy / advance (traced, not isolated) | 21.79 ms | 55.50 ms |
| busy share of frame wall | 48.1 % | 75.0 % |
| main-thread CPU / advance | 23.9 ms (≈17 ms discounted) | 32.81 ms (≈27.5 ms discounted) |
| WebGPU passes / advance | 755.4 | 983.7 |
| Metal compute encoders / advance | 26.1 | 27.1 |
| pressure share of passes | 86 % | 89 % |
| pressure share of GPU busy | 41.1 % | 40.5 % |

Read the family table with the caveat that the two isolated captures sit at
different pressure budgets — mini64's captured pair ran 1220.5 V-cycle firings
per advance, fig7's 736. The `pressure V-cycles` row is therefore the one row
where the scene comparison is confounded by the cycle budget; every other row
is a like-for-like firing count (`surface publication` 23 vs 23,
`conservative volume gather` 15 vs 15, `phi transport + redistance` 3 vs 3).

---

## C. Per-level multigrid

`FLUID_UNIFORM_MG_LEVEL_LABELS=1` with encoder+label isolation and the
in-process pass-timestamp audit (no xctrace, no counter distortion; the only
distortion is encoder isolation). mini64 has levels L0…L5, fig7 L0…L6 — the
128³ lattice adds exactly one coarsening rung.

### mini64 — 2 captured advances of 20; 2292 measured passes; summed 44.10 ms; coverage 34.8%; capacity overflows 0

**mgSmoothColour** — 848 firings/advance, 12.071 ms/advance

| level | firings/adv | ms/adv | % of family | mean us/firing | min us | max us |
|--:|--:|--:|--:|--:|--:|--:|
| L0 | 176 | 3.1470 | 26.1% | 17.9 | 9.6 | 50.1 |
| L1 | 96 | 1.4234 | 11.8% | 14.8 | 13.8 | 15.9 |
| L2 | 144 | 1.8566 | 15.4% | 12.9 | 12.0 | 13.9 |
| L3 | 192 | 2.6412 | 21.9% | 13.8 | 12.4 | 188.8 |
| L4 | 240 | 3.0027 | 24.9% | 12.5 | 11.8 | 13.8 |

**mgResidual** — 32 firings/advance, 0.484 ms/advance

| level | firings/adv | ms/adv | % of family | mean us/firing | min us | max us |
|--:|--:|--:|--:|--:|--:|--:|
| L0 | 4 | 0.1522 | 31.5% | 38.0 | 36.7 | 39.1 |
| L1 | 4 | 0.0537 | 11.1% | 13.4 | 12.9 | 14.0 |
| L2 | 6 | 0.0700 | 14.5% | 11.7 | 11.2 | 12.2 |
| L3 | 8 | 0.0961 | 19.9% | 12.0 | 11.6 | 12.5 |
| L4 | 10 | 0.1119 | 23.1% | 11.2 | 10.7 | 11.8 |

**mgRestrictResidual** — 40 firings/advance, 0.324 ms/advance

| level | firings/adv | ms/adv | % of family | mean us/firing | min us | max us |
|--:|--:|--:|--:|--:|--:|--:|
| L1 | 4 | 0.0426 | 13.2% | 10.7 | 10.3 | 11.1 |
| L2 | 6 | 0.0477 | 14.7% | 7.9 | 7.6 | 8.7 |
| L3 | 8 | 0.0641 | 19.8% | 8.0 | 7.5 | 8.9 |
| L4 | 10 | 0.0803 | 24.8% | 8.0 | 7.6 | 8.5 |
| L5 | 12 | 0.0892 | 27.5% | 7.4 | 7.0 | 7.9 |

**mgProlongateAdd** — 30 firings/advance, 0.347 ms/advance

| level | firings/adv | ms/adv | % of family | mean us/firing | min us | max us |
|--:|--:|--:|--:|--:|--:|--:|
| L0 | 2 | 0.0628 | 18.1% | 31.4 | 31.3 | 31.6 |
| L1 | 4 | 0.0450 | 13.0% | 11.3 | 10.9 | 12.3 |
| L2 | 6 | 0.0592 | 17.1% | 9.9 | 9.6 | 10.3 |
| L3 | 8 | 0.0803 | 23.1% | 10.0 | 9.8 | 10.6 |
| L4 | 10 | 0.0996 | 28.7% | 10.0 | 9.3 | 10.9 |

**mgProlongateAssign** — 10 firings/advance, 0.141 ms/advance

| level | firings/adv | ms/adv | % of family | mean us/firing | min us | max us |
|--:|--:|--:|--:|--:|--:|--:|
| L0 | 2 | 0.0611 | 43.4% | 30.6 | 30.4 | 30.8 |
| L1 | 2 | 0.0216 | 15.3% | 10.8 | 10.6 | 10.9 |
| L2 | 2 | 0.0193 | 13.7% | 9.6 | 9.5 | 10.0 |
| L3 | 2 | 0.0196 | 13.9% | 9.8 | 9.6 | 10.0 |
| L4 | 2 | 0.0194 | 13.8% | 9.7 | 9.1 | 10.2 |

**mgDownsampleSubtract** — 30 firings/advance, 0.268 ms/advance

| level | firings/adv | ms/adv | % of family | mean us/firing | min us | max us |
|--:|--:|--:|--:|--:|--:|--:|
| L1 | 2 | 0.0277 | 10.3% | 13.8 | 13.5 | 14.3 |
| L2 | 4 | 0.0354 | 13.2% | 8.9 | 8.5 | 9.5 |
| L3 | 6 | 0.0522 | 19.5% | 8.7 | 8.4 | 9.1 |
| L4 | 8 | 0.0675 | 25.2% | 8.4 | 8.1 | 8.9 |
| L5 | 10 | 0.0850 | 31.7% | 8.5 | 8.2 | 8.9 |

**mgSolveCoarsest** — 12 firings/advance, 1.383 ms/advance

| level | firings/adv | ms/adv | % of family | mean us/firing | min us | max us |
|--:|--:|--:|--:|--:|--:|--:|
| L5 | 12 | 1.3832 | 100.0% | 115.3 | 32.9 | 276.4 |

**mgClearPressure** — 32 firings/advance, 0.197 ms/advance

| level | firings/adv | ms/adv | % of family | mean us/firing | min us | max us |
|--:|--:|--:|--:|--:|--:|--:|
| L1 | 2 | 0.0138 | 7.0% | 6.9 | 6.8 | 7.0 |
| L2 | 4 | 0.0251 | 12.7% | 6.3 | 6.0 | 6.8 |
| L3 | 6 | 0.0385 | 19.5% | 6.4 | 6.0 | 6.9 |
| L4 | 8 | 0.0507 | 25.7% | 6.3 | 6.0 | 6.8 |
| L5 | 12 | 0.0691 | 35.1% | 5.8 | 5.4 | 6.5 |

Whole-run firings/advance by level (GPU command audit, all advances):

| entry point | levels |
|---|---|
| mgSmoothColour | L0:165.2, L1:74.4, L2:111.6, L3:148.8, L4:186 |
| mgMeasureFineResidual | L0:11.55 |
| mgCheckCycleConvergence | L0:10.55 |
| mgRestoreRejected | L0:10.55 |
| mgSaveAccepted | L0:9.55 |
| mgClearPressure | L1:1.55, L2:3.1, L3:4.65, L4:6.2, L5:9.3 |
| mgRestrictResidual | L1:3.1, L2:4.65, L3:6.2, L4:7.75, L5:9.3 |
| mgSolveCoarsest | L5:9.3 |
| mgDownsampleSubtract | L1:1.55, L2:3.1, L3:4.65, L4:6.2, L5:7.75 |
| mgProlongateAdd | L0:1.55, L1:3.1, L2:4.65, L3:6.2, L4:7.75 |
| mgResidual | L0:3.1, L1:3.1, L2:4.65, L3:6.2, L4:7.75 |
| mgCopyPressure | L0:4.1 |
| mgAddPressure | L0:1.55 |
| mgDownsampleMinimum | L1:1.55, L2:1.55, L3:1.55, L4:1.55, L5:1.55 |
| mgProlongateAssign | L0:1.55, L1:1.55, L2:1.55, L3:1.55, L4:1.55 |
| mgShiftMinimum | L0:1.55 |
| mgBakeCoefficients | L0:1, L1:1, L2:1, L3:1, L4:1, L5:1 |
| mgBuildFinestRhs | L0:1 |
| mgBuildFinestTopology | L0:1 |
| mgDownsampleTopology | L1:1, L2:1, L3:1, L4:1, L5:1 |
| mgExtrapolatePhiOneCell | L0:1, L1:1, L2:1, L3:1, L4:1, L5:1 |
| mgFinishSafety | L0:1 |

### fig7 — 2 captured advances of 30; 1726 measured passes; summed 67.14 ms; coverage 43.1%; capacity overflows 0

**mgSmoothColour** — 632 firings/advance, 14.870 ms/advance

| level | firings/adv | ms/adv | % of family | mean us/firing | min us | max us |
|--:|--:|--:|--:|--:|--:|--:|
| L0 | 152 | 8.3254 | 56.0% | 54.8 | 29.1 | 227.4 |
| L1 | 48 | 1.1402 | 7.7% | 23.8 | 20.5 | 33.4 |
| L2 | 72 | 1.0285 | 6.9% | 14.3 | 12.8 | 17.6 |
| L3 | 96 | 1.3178 | 8.9% | 13.7 | 12.0 | 17.8 |
| L4 | 120 | 1.6402 | 11.0% | 13.7 | 12.2 | 17.4 |
| L5 | 144 | 1.4175 | 9.5% | 9.8 | 8.2 | 13.4 |

**mgResidual** — 22 firings/advance, 0.574 ms/advance

| level | firings/adv | ms/adv | % of family | mean us/firing | min us | max us |
|--:|--:|--:|--:|--:|--:|--:|
| L0 | 2 | 0.3199 | 55.7% | 159.9 | 143.6 | 176.5 |
| L1 | 2 | 0.0409 | 7.1% | 20.5 | 19.8 | 21.6 |
| L2 | 3 | 0.0393 | 6.8% | 13.1 | 12.3 | 15.0 |
| L3 | 4 | 0.0533 | 9.3% | 13.3 | 11.5 | 15.0 |
| L4 | 5 | 0.0691 | 12.0% | 13.8 | 12.1 | 16.1 |
| L5 | 6 | 0.0519 | 9.0% | 8.7 | 7.3 | 10.9 |

**mgRestrictResidual** — 27 firings/advance, 0.280 ms/advance

| level | firings/adv | ms/adv | % of family | mean us/firing | min us | max us |
|--:|--:|--:|--:|--:|--:|--:|
| L1 | 2 | 0.0702 | 25.1% | 35.1 | 33.5 | 37.8 |
| L2 | 3 | 0.0314 | 11.2% | 10.5 | 10.3 | 10.9 |
| L3 | 4 | 0.0334 | 11.9% | 8.3 | 7.8 | 10.2 |
| L4 | 5 | 0.0426 | 15.2% | 8.5 | 7.1 | 10.5 |
| L5 | 6 | 0.0471 | 16.8% | 7.8 | 7.0 | 9.8 |
| L6 | 7 | 0.0555 | 19.8% | 7.9 | 7.2 | 10.3 |

**mgProlongateAdd** — 21 firings/advance, 0.464 ms/advance

| level | firings/adv | ms/adv | % of family | mean us/firing | min us | max us |
|--:|--:|--:|--:|--:|--:|--:|
| L0 | 1 | 0.2021 | 43.5% | 202.1 | 191.5 | 212.7 |
| L1 | 2 | 0.0638 | 13.7% | 31.9 | 31.7 | 32.2 |
| L2 | 3 | 0.0370 | 8.0% | 12.3 | 11.0 | 14.3 |
| L3 | 4 | 0.0446 | 9.6% | 11.2 | 9.9 | 13.7 |
| L4 | 5 | 0.0528 | 11.4% | 10.6 | 9.5 | 12.8 |
| L5 | 6 | 0.0642 | 13.8% | 10.7 | 9.7 | 14.1 |

**mgProlongateAssign** — 6 firings/advance, 0.245 ms/advance

| level | firings/adv | ms/adv | % of family | mean us/firing | min us | max us |
|--:|--:|--:|--:|--:|--:|--:|
| L0 | 1 | 0.1720 | 70.1% | 172.0 | 169.2 | 174.8 |
| L1 | 1 | 0.0321 | 13.1% | 32.1 | 30.8 | 33.4 |
| L2 | 1 | 0.0107 | 4.4% | 10.7 | 10.7 | 10.7 |
| L3 | 1 | 0.0106 | 4.3% | 10.7 | 10.4 | 10.9 |
| L4 | 1 | 0.0100 | 4.1% | 10.0 | 10.0 | 10.1 |
| L5 | 1 | 0.0100 | 4.1% | 10.0 | 9.7 | 10.3 |

**mgDownsampleSubtract** — 21 firings/advance, 0.262 ms/advance

| level | firings/adv | ms/adv | % of family | mean us/firing | min us | max us |
|--:|--:|--:|--:|--:|--:|--:|
| L1 | 1 | 0.0657 | 25.1% | 65.8 | 65.0 | 66.5 |
| L2 | 2 | 0.0287 | 11.0% | 14.3 | 13.2 | 17.1 |
| L3 | 3 | 0.0291 | 11.1% | 9.7 | 8.7 | 11.8 |
| L4 | 4 | 0.0382 | 14.6% | 9.6 | 8.5 | 11.8 |
| L5 | 5 | 0.0449 | 17.1% | 9.0 | 8.0 | 11.0 |
| L6 | 6 | 0.0556 | 21.2% | 9.3 | 8.1 | 10.8 |

**mgSolveCoarsest** — 7 firings/advance, 0.109 ms/advance

| level | firings/adv | ms/adv | % of family | mean us/firing | min us | max us |
|--:|--:|--:|--:|--:|--:|--:|
| L6 | 7 | 0.1090 | 100.0% | 15.6 | 14.0 | 17.2 |

**mgClearPressure** — 22 firings/advance, 0.150 ms/advance

| level | firings/adv | ms/adv | % of family | mean us/firing | min us | max us |
|--:|--:|--:|--:|--:|--:|--:|
| L1 | 1 | 0.0128 | 8.5% | 12.8 | 10.8 | 14.8 |
| L2 | 2 | 0.0130 | 8.6% | 6.5 | 6.4 | 6.7 |
| L3 | 3 | 0.0193 | 12.8% | 6.4 | 6.2 | 6.8 |
| L4 | 4 | 0.0265 | 17.6% | 6.6 | 6.0 | 8.4 |
| L5 | 5 | 0.0345 | 22.9% | 6.9 | 5.4 | 8.9 |
| L6 | 7 | 0.0443 | 29.5% | 6.3 | 5.6 | 8.5 |

Whole-run firings/advance by level (GPU command audit, all advances):

| entry point | levels |
|---|---|
| mgSmoothColour | L0:154.4, L1:52.8, L2:79.2, L3:105.6, L4:132, L5:158.4 |
| mgMeasureFineResidual | L0:11.1 |
| mgCheckCycleConvergence | L0:10.1 |
| mgRestoreRejected | L0:10.1 |
| mgSaveAccepted | L0:9.1 |
| mgClearPressure | L1:1.1, L2:2.2, L3:3.3, L4:4.4, L5:5.5, L6:7.7 |
| mgRestrictResidual | L1:2.2, L2:3.3, L3:4.4, L4:5.5, L5:6.6, L6:7.7 |
| mgSolveCoarsest | L6:7.7 |
| mgDownsampleSubtract | L1:1.1, L2:2.2, L3:3.3, L4:4.4, L5:5.5, L6:6.6 |
| mgProlongateAdd | L0:1.1, L1:2.2, L2:3.3, L3:4.4, L4:5.5, L5:6.6 |
| mgResidual | L0:2.2, L1:2.2, L2:3.3, L3:4.4, L4:5.5, L5:6.6 |
| mgCopyPressure | L0:3.2 |
| mgAddPressure | L0:1.1 |
| mgDownsampleMinimum | L1:1.1, L2:1.1, L3:1.1, L4:1.1, L5:1.1, L6:1.1 |
| mgProlongateAssign | L0:1.1, L1:1.1, L2:1.1, L3:1.1, L4:1.1, L5:1.1 |
| mgShiftMinimum | L0:1.1 |
| mgBakeCoefficients | L0:1, L1:1, L2:1, L3:1, L4:1, L5:1, L6:1 |
| mgBuildFinestRhs | L0:1 |
| mgBuildFinestTopology | L0:1 |
| mgDownsampleTopology | L1:1, L2:1, L3:1, L4:1, L5:1, L6:1 |
| mgExtrapolatePhiOneCell | L0:1, L1:1, L2:1, L3:1, L4:1, L5:1, L6:1 |
| mgFinishSafety | L0:1 |

Three facts the level split makes visible:

1. **`mgSmoothColour` is the frame in both scenes, but for different reasons.**
   On mini64 the 848 firings/advance spread almost evenly across L0–L4
   (3.15 / 1.42 / 1.86 / 2.64 / 3.00 ms) and *no* level costs more than 18 µs
   per firing — the smoother is launch-bound at every level. On fig7 the 632
   firings cost 14.87 ms of which **L0 alone is 8.33 ms (56 %)** at
   **54.8 µs/firing** — three to four times mini64's per-firing cost, because
   L0 is now 128³.
2. **Coarse levels never get cheaper than ≈6–10 µs per firing** on either
   scene. L5 on fig7 (a 4³-ish lattice) still costs 9.84 µs per smoother firing
   and L6 restrict costs 7.93 µs. That floor is the dispatch, not the work.
3. **`mgSolveCoarsest` behaves oppositely on the two scenes.** mini64: 12
   firings, 1.383 ms, 115 µs mean with a 276 µs max — the coarsest solve is
   1.4 ms/advance. fig7: 7 firings, 0.109 ms, 15.6 µs mean — the extra rung
   makes the coarsest level small enough that the serial solve collapses to
   nothing.

---

## D. Why the GPU starves — CPU/GPU serialisation

### D.1 The mechanism, from code

`FLUID_AWAIT_EVERY_STEPS` (default 30, `lib/harness/webgpu-smoke-executor.ts:687`)
is **dead for this method**. Two lines earlier in the advance loop:

```
lib/harness/webgpu-smoke-executor.ts:3179    await solver.awaitFrameCompletion?.();
lib/methods/uniform/webgpu-uniform-reference.ts:2551
    async awaitFrameCompletion(): Promise<void> { await this.device.queue.onSubmittedWorkDone(); }
```

The harness awaits `awaitFrameCompletion` **unconditionally after every
accepted advance**, and the Uniform Geometric solver implements it as a bare
full queue drain. The every-30-steps `awaitAdvanceCompletion` fence at `:2311`
is therefore always a no-op — the queue was already empty one line earlier.
The candidates in the brief are all cleared:

* `FLUID_PERFORMANCE_PROFILE=1` self-reject (`:689-695`) — cleared by ablation
  below.
* the 12-byte `readPressureCycleDemand` mapAsync — cleared by the `yield` arm
  below.
* `readStats` per step — not called per step in these lanes.
* the GPU command audit wrapper — cleared by ablation.
* CPU encode ≥ GPU time — false: CPU is 8.6–11.2 ms against 21.7–51.4 ms of GPU.

`framePending` is hard-coded `false` (`webgpu-uniform-reference.ts:2550`), and
the renderer's only `awaitFrameCompletion` call (`lib/core/webgpu-renderer.ts:2336`)
is in solver **retirement**, not the per-frame path — so this serialisation is
harness-specific, not something the app pays.

### D.2 Harness environment ablation (mini64, 60 steps, interleaved, 2 reps)

| env arm | rep 1 | rep 2 |
|---|--:|--:|
| minimal (profile/audit/tripwires all off) | 36.23 | 35.50 |
| + tripwires | 35.78 | 35.90 |
| + GPU command audit | 36.43 | 36.48 |
| + performance profile | 35.52 | 35.70 |
| full (all on) | 36.00 | 36.07 |

**Spread 2.8 %, no ordering.** None of the profiling or audit switches costs
anything measurable. The serialisation is structural.

### D.3 Direct measurement — scratch driver, three arms

`overlap-probe.mts` (modelled on `tools/benchmark-uniform-pressure-tolerance-dawn.ts`)
builds the solver via `uniformVolumeMethod.createSolverAsync`, warms 2
advances, then times **the same 58 advances (indices 3–60)** in each arm. One
arm per process so every arm sees the same trajectory.

* `drain` — no fence at all, one trailing `onSubmittedWorkDone()`.
* `yield` — no fence, but one macrotask turn per step so Dawn's event callbacks
  (including the lagged pressure-cycle readback) can run while the GPU runs on.
* `fenced` — `onSubmittedWorkDone()` after every advance, i.e. the harness's
  own cadence.

All three counted `beginComputePass` (the wrapper costs < 0.5 % of CPU).

| scene | arm | CPU ms/adv (inside `advanceTo`) | loop ms/adv | trailing drain ms | **total ms/adv** | passes/adv |
|---|---|--:|--:|--:|--:|--:|
| mini64 | drain | 8.615 | 8.620 | 761.31 | **21.746** | 753 |
| mini64 | yield | 8.690 | 8.800 | 751.79 | **21.762** | 753 |
| mini64 | fenced | 10.323 | 33.870 | 0.02 | **33.870** | 937.8 |
| fig7 | drain | 10.768 | 10.770 | 2354.01 | **51.356** | 938 |
| fig7 | yield | 11.184 | 11.491 | 2307.60 | **51.278** | 938 |
| fig7 | fenced | 11.203 | 64.236 | 0.01 | **64.236** | 993.5 |

Harness, same tree, same 60-step lane, measured in the same series:

| scene | env | simulationWall ms | **ms/advance** | passes/adv |
|---|---|--:|--:|--:|
| mini64 | minimal | 2046 | **34.10** | — |
| mini64 | full | 2068 | **34.47** | 931.6 |
| fig7 | minimal | 3833 | **63.88** | — |
| fig7 | full | 3912 | **65.20** | 991.7 |

### D.4 Answers

**A clean advance costs `max(CPU, GPU)`. A harness advance costs `CPU + GPU`.**

* Unfenced, the CPU runs 58 advances of encode in 0.50 s (mini64) / 0.62 s
  (fig7) and then waits 0.76 s / 2.35 s for the GPU. Total per advance equals
  the GPU time alone: 21.7 ms and 51.4 ms. **CPU encode of advance N+1 fully
  overlaps GPU execution of advance N.**
* Fenced, the same advances cost 33.87 ms and 64.24 ms — CPU (10.3 / 11.2) plus
  GPU, additively, with the small extra from the higher pressure budget the
  fence's resolved readback unlocks.
* The harness lands on 34.10–34.47 ms and 63.88–65.20 ms, i.e. **on the fenced
  arm, within 1 %**, not on the unfenced one.
* The 16.8–17.9 ms idle gap in the shipping trace (section B.4) is the same
  effect seen from the GPU side: the GPU finishes the solver command buffer and
  waits for the main thread to re-encode.

**True CPU ms/advance inside `advanceTo`, no audit wrappers, no tracing:**

| scene | CPU ms/advance | passes encoded | **µs of CPU per encoded pass** |
|---|--:|--:|--:|
| mini64 | 8.615 | 753 | **11.44** |
| fig7 | 10.768 | 938 | **11.48** |
| mini64 (fenced, 938 passes) | 10.323 | 937.8 | **11.01** |
| fig7 (fenced, 993 passes) | 11.203 | 993.5 | **11.28** |

**≈11.2 µs of main-thread CPU per encoded compute pass**, flat across scene
size, pass count and fencing — with Dawn validation **on** (the driver does not
set `skip_validation`). The harness's traced figure of 32.81 ms/advance is that
same ≈11 ms plus the tracer's `__kdebug_trace64` and the harness's own
per-advance bookkeeping.

**The mapAsync is not the cause.** The `yield` arm resolves the lagged
`readPressureCycleDemand` readback every step without fencing the queue and is
within 0.1 % of `drain` on both scenes (21.76 vs 21.75; 51.28 vs 51.36).

---

## 7. Anomalies

1. **Another session edited the working tree three times during this series.**
   All of the files below are that session's, not mine.
   * 19:29:19 — `lib/methods/uniform/uniform-volume.wgsl.ts` and
     `webgpu-uniform-reference.wgsl.ts` (a `bodySolidFractionAt` split and a
     `faceOpenFraction` out-of-bounds change, behind `FLUID_UNIFORM_AB_BASELINE`
     with the **new** arm as default). This changed the physics at the domain
     boundary: an identical mini64 `yield` arm measured 961.9 passes/advance and
     `maxSpeed` 1.865 m/s before, 753 and 3.637 m/s after. **Every part-D number
     in section D.3 was re-measured after this change so all arms and both
     harness baselines share one tree** (verified by md5 before and after the
     arm series: unchanged). The pre-change series told the same story with ~5 %
     larger numbers (mini64 drain 23.41 / fenced 35.52 / harness 35.5–36.5;
     fig7 drain 54.53 / fenced 67.33 / harness 67.42).
   * 19:32:15 — `lib/methods/uniform/uniform-ab-switch.ts` created.
   * 19:32:49 — `webgpu-uniform-pressure-multigrid.ts` gained a second,
     unrelated uncommitted edit: a `BATCH_PASSES` arm that merges every
     multigrid dispatch into one shared compute pass, **on by default**
     (`uniformAbOn("batch")` returns true unless `FLUID_UNIFORM_AB_OFF` names
     it). It was written on top of my label edit and reads it
     (`const batch = BATCH_PASSES && !MG_LEVEL_LABELS;`). A second rep of the
     harness baselines started 6 s before that write and measured 5.6 ms/advance
     at 116 passes/advance on mini64 — **that rep was discarded**; only rep 1,
     which completed before the write, is reported. **Every capture and every
     number in this report predates the batching change.**
   * Captures 2 and 3 predate all three writes.
2. **The two isolated counter captures sit at different pressure budgets.**
   mini64's analysed pair ran 1452 passes/advance against a 947 whole-run mean;
   fig7's ran 938 against 986. The `pressure V-cycles` family row is the only
   side-by-side row this confounds.
3. **The fig7 counter window lands on free fall, not steady state.** The
   profiler's gated counter path (`counterStartGate`, forced on by
   `--uniform-mini-64`) always places the window on the first advances after
   gate release, and `--frames=2` keeps two of them. Verified by matching the
   in-process `gpuPassTimestamps.byLabel` firing counts (first two command
   buffers) against the xctrace task table — `mgSmoothColour` 632/advance in
   both. Free-fall advances are the *cheapest* phase (48.20 ms busy vs 60.01 in
   spread), so the isolated table understates spread-phase cost by ≈24 %. The
   75-step shipping trace covers all four phases and is the phase-resolved
   source.
4. `metal-shader-profiler-intervals` returned **14 089 rows** for fig7 (mini64
   returned none), yet `summary.shaders` is still an empty array — the reducer
   does not consume that table. No per-shader attribution exists in either
   report.
5. **`uvNormalizeRows` and `uvFinishDonorSums` are the bandwidth outliers**:
   90.7 / 110.4 GB/s read against a frame mean of 33.4 GB/s, with LLC
   utilisation 81 % and 40.5 % and `Buffer Read` / `GPU Last Level Cache`
   limiters at 79–85 %. They cost 3.06 ms/advance between them on 7 firings.
6. **`Total surface volume: dilate`** is the only task on either scene that
   looks compute-saturated: occupancy 75.6 %, ALU 70.2 %, `ALU Limiter` 95.9 %,
   108.6 GB/s read, over 4 firings and 0.61 ms/advance. Every other task in the
   top 20 sits at 10–45 % occupancy.
7. **fig7 has essentially no GPU contention** (0.34 ms total across three
   foreign processes) where mini64's capture had 32.7 ms of WindowServer and
   Codex traffic and only 0.86 exclusive coverage. fig7's counter numbers are
   the cleaner of the two.
8. The `>1 ms` duration bucket holds **10 firings out of 938 (1.07 %) but
   35.9 % of fig7's busy time**, against 4 firings / 18.6 % on mini64. Going
   from 64³ to 128³ moved the frame from "a thousand tiny launches" toward "ten
   big kernels plus a thousand tiny launches": the `<20 µs` bucket fell from
   36.0 % to 12.6 % of busy while its firing count roughly halved (1277 → 586).
