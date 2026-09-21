# Uniform Geometric on an M1 Max: xctrace GPU capture

Method `uniform-volume` ("Uniform Geometric"), scene `minimal-power-dam-break-64`, grid 64x64x64,
paper time step 1/30 s. Captured 2026-09-21 on an Apple M1 Max with Instruments/xctrace.
Everything below is measurement; there are no recommendations in this document.

Companion machine-readable file: `xctrace-capture.json` (this directory).

## Headline numbers

| | |
|---|---|
| shipping wall (untraced, non-isolated, 12 steps) | **38.75 ms/advance** |
| GPU busy in that regime | **21.79 ms/advance** = 48% of wall |
| main-thread CPU in that regime | **~23.9 ms/advance** = 53% of wall |
| WebGPU compute passes/advance | **755** (12-step window), 858 run-average at 120 steps |
| Metal compute encoders/advance, shipping | **26** -- Dawn packs **28.9 passes per encoder** (max 466) |
| Metal compute encoders/advance, isolated | **756** (1:1) |
| pressure share of passes | **86-93%**; `mgSmoothColour` alone is 56-71% of all passes |
| pressure GPU busy, shipping | **8.96 ms/advance in 12 Metal encoders**, 0.56 ms inter-encoder idle |
| pressure GPU busy, isolated | 12.58 ms/advance in 654 encoders, **25.1 ms** inter-encoder idle |
| mean compute occupancy (counter run) | **13.5%**; ALU **10.1%** |
| GPU compute capacity used over the frame | **4.1%** |
| one idle stall per advance across the diagnostics readback fence | **~40 ms** (counter run), 30% of frame wall |
| per-WGSL-entry-point GPU time | **not obtainable** without `MTL_CAPTURE_ENABLED` (section 8) |

The single largest finding for the mid-task question: the ~13 us "pass floor" is a **profiling
artefact**. Dawn merges the pressure multigrid's ~652 per-dispatch compute passes into **12** Metal
encoders in the shipping graph; encoder isolation splits them 1:1 and manufactures 24.6 ms/advance
of inter-dispatch GPU idle that does not exist in production.

### Is this steady state?

Yes. The run is 120 advances x 1/30 s = 4.0 s of simulated time; the audit reports `front_m 0.4`
(the domain extent) already at advance 12, so the dam has reached the far wall well before the
window. The counter window sits 3.62-7.62 s into a 14.50 s stepping phase, covering 24 advances
(20% of the run) at roughly 1.0-2.1 s of simulated time -- collapsed and sloshing, not the initial
release. `maxSpeed 20.1 m/s`, `volumeDrift -1.0e-4`, `uniformCM11aConverged true`.

---

## 1. Tooling changes made to reach this method

`npm run profile:uniform-mini-dam-64-xctrace` sets `FLUID_METHOD=uniform` (base CM11a). Two edits
were needed and no solver or shader code was touched.

| file:line | change | why |
|---|---|---|
| `lib/harness/webgpu-smoke-executor.ts:507` (import at :10) | added `uniformVolumeMethod` to `availableMethods` | `FLUID_METHOD=uniform-volume` previously threw "Unknown FLUID_METHOD". `methodsForScenario` filters by `FLUID_METHOD` when it is set, so scene lanes that never name the method are unaffected. The method's harness plugin already overrides `methodId: "uniform-volume"` (`lib/methods/uniform/uniform-volume-method.ts:34`), which satisfies the executor's `plugin.methodId !== candidate.id` assertion. |
| `tools/profile-mini-dam-xctrace.ts` (import :108, `--method` flag :166, `uniformGeometric` :177, `uniformMini64Dt` :439, lane env :443-457) | `--uniform-mini-64 --method=uniform-volume` switches the lane's cadence to `UNIFORM_PAPER_DT_S` (1/30 s) and drops `FLUID_UNIFORM_TIME_STEP` / `FLUID_UNIFORM_DENSITY_POSTPROCESSING` | see below |

**`FLUID_UNIFORM_TIME_STEP` and `FLUID_UNIFORM_DENSITY_POSTPROCESSING` do not apply to Uniform
Geometric.** `resolveUniformGeometricValues` (`lib/methods/uniform/uniform-geometric-parameters.ts`)
rebuilds the value bag from the geometric parameter list alone, and that list declares no
`timeStep`, so the solver always runs the paper 1/30 s step; `densityPostProcessing: false` is
hard-wired in `lib/methods/uniform/uniform-geometric-options.ts:36`. Forwarding the dense lane's
4 ms cadence is not merely inert -- it would reject 8 advances in 9 and trip
`FLUID_MAX_CONSECUTIVE_REJECTED_ADVANCES=8` before Instruments saw a frame. The runs below use the
method's shipping defaults; the logs confirm `lastDt_s 0.0333` and `rejectedAdvanceAttempts 0`.

**Label-isolation prefix.** `FLUID_GPU_PASS_TIMESTAMP_LABEL_PREFIXES` only gates the in-process
timestamp-query pairs. xctrace attribution comes from `isolateComputePassEncoders`
(`lib/harness/webgpu-pass-encoder-isolation.ts`), which is *not* prefix-gated, so all geometric
stage labels are attributed. Confirmed: 82 exact stage buckets, 0 composite.

**Method actually running.** `traced.log` carries `"method":"uniform-volume"` on 95 records; the
trace contains the geometric stage labels `Advect page vertex phi`, `Redistance page vertex phi`,
`uvGather`/`uvBuildEdges`/`uvNormalizeRows`/`uvNormalizeDonors`/`uvFinishDonorSums`,
`Classify 4h sharpening work`, `uvProposeSharpen`/`uvLimitSharpen`/`uvCommitSharpen`,
`Uniform Geometric two-level uvTwoLevel*` and `Uniform Geometric surface publication`.

**Working-tree note.** `lib/methods/uniform/uniform-geometric-options.ts` was edited by another
session *during* this capture series (line 14 became `volumePages: values.pageSize === "16" ? 16 : 32`).
All four 12-step runs nevertheless agree at `computePasses: 10434`, so the encoded graph did not
change between them. No git operation of any kind was performed.

---

## 2. Exact commands

Capture 1 -- counter run (attach), the primary measurement:

```bash
node --import tsx tools/profile-mini-dam-xctrace.ts --uniform-mini-64 --method=uniform-volume \
  --steps=120 --counter-seconds=4 --out=artifacts/xctrace-uniform-geometric-mini-64
```

Limiter re-export from the *same retained trace* (no new GPU run), then report rebuild:

```bash
# one-off script (scratchpad): re-exports gpu-counter-value from the retained .trace,
# keeping the 5 default counters + all 11 "* Limiter" series + LLC Utilization at
# timestamp stride 18, writing artifacts/xctrace-uniform-geometric-mini-64-limiters/.
# The other tables (encoders, gpu-intervals, command buffers, logs) were copied across
# from the original capture directory unchanged; no new GPU run was performed.
node --import tsx tools/profile-mini-dam-xctrace.ts --uniform-mini-64 --method=uniform-volume \
  --steps=120 --counter-seconds=4 --out=artifacts/xctrace-uniform-geometric-mini-64-limiters --reuse-tables
```

Capture 2 -- launch-mode Metal System Trace, **label/encoder isolation ON** (12 steps), and the
non-isolated control, both via `xcrun xctrace record --template "Metal System Trace" --launch --
node --import tsx tools/run-webgpu-smoke-isolated-worker.ts` with the lane environment inlined as
`--env` (full scripts: `scratchpad/capture2.sh`, `scratchpad/isolation-ab.sh`). Shared lane env:

```
FLUID_METHOD=uniform-volume  FLUID_SCENE=minimal-power-dam-break-64  FLUID_LANE=uniform-one-step
FLUID_MAX_DT=0.0333333  FLUID_TARGET_S=0.4  FLUID_ORACLE_STEPS=12  FLUID_EXPECT_GRID=64,64,64
FLUID_WEBGPU_DAWN_FEATURES=skip_validation,use_user_defined_labels_in_backend
FLUID_PERFORMANCE_PROFILE=1  FLUID_GPU_COMMAND_AUDIT=1  FLUID_GPU_FINE_TIMESTAMPS=0
isolated arm:     FLUID_GPU_PASS_TIMESTAMPS=1 FLUID_GPU_ISOLATE_PASS_LABELS=1 FLUID_GPU_ISOLATE_PASS_ENCODERS=1
non-isolated arm: FLUID_GPU_PASS_TIMESTAMPS=0 FLUID_GPU_ISOLATE_PASS_LABELS=0 FLUID_GPU_ISOLATE_PASS_ENCODERS=0
```

`MTL_CAPTURE_ENABLED` was never set and `xctrace --window` was never used. Every run held the
repo-wide WebGPU lease exclusively; three other live sessions contended for it and were waited on,
never killed.

Artifacts (all under `artifacts/`, gitignored):

| path | size | what |
|---|--:|---|
| `artifacts/xctrace-uniform-geometric-mini-64/` | 253 MB | capture 1, original 5-counter reduction |
| `artifacts/xctrace-uniform-geometric-mini-64-limiters/` | 43 MB | capture 1 re-exported with limiters -- **quoted below** |
| `artifacts/xctrace-uniform-geometric-mini-64-launch/` | 286 MB | capture 2, isolated launch trace |
| `artifacts/xctrace-uniform-geometric-mini-64-launch-plain/` | 37 MB | capture 2 control, non-isolated launch trace |

All exported XML was deleted after reduction. `df`: 91 GiB free at the end (95 GiB at the start).

---

## 3. Measurement integrity

| quantity | value |
|---|---|
| frames analysed | **2** (anchor `Uniform Sec. 3.3 rho-prime and face authority`) |
| anchor firings inside the counter window | 24 of 120 advances = 20% of the run |
| counter window position | 3.62 s -- 7.62 s into a 14.50 s stepping phase |
| attribution mode | **full** -- 82 exact stage buckets, 0 composite, 0.00 ms composite |
| untraced wall (120 steps, same process, non-isolated) | **36.77 ms/advance** |
| traced wall (isolated + counters) | **79.98 ms/advance** (**2.18x**) |
| encoders per advance (analysed frames) | 1452, carrying 1452 labelled WebGPU compute passes |
| command buffers per advance | 122 command buffers for 120 advances = **1.017/advance** (120 step buffers + 2 diagnostics readbacks) |
| isolation blits per advance | 1453, costing 6.12 ms/advance (excluded from stage totals) |
| counter coverage | **17 of 31 series**, every 18th hardware timestamp (180 us effective) |
| uncontended counter samples | 373 of 1120 |
| exclusive GPU coverage | **0.8595** -- 86.0% of our GPU time was uncontended |
| contention in-window | WindowServer (433) 23.24 ms / 621 intervals; Codex (Service) (1279) 9.50 ms / 365 intervals |
| slot detection | `slotsPerPartition 512` at confidence **0.081**, `totalSlots 0`, `totalThreads 0` |
| CPU samples in capture 1 | **0** (time-profile is `fullDiagnosticsOnly`) |
| shader-profiler rows in capture 1 | **0** |

Four integrity caveats that bound every number in section 5.

1. **Only 2 frames.** The retained detailed frames are a representative pair from the middle third
   of the 24 in-window advances. Their encoder counts differ (1685 and 1219), so per-advance task
   counts are the mean of two unlike advances, not a modal figure.
2. **The counter window landed on heavy advances.** Over the whole 120-advance run the audit
   records 102 942 compute passes = **858/advance** and `mgSmoothColour` 68 280 = **569/advance**.
   The two analysed advances carry **1452 passes** and **1028 `mgSmoothColour`** -- 69% and 81%
   above the run mean, because the lagged pressure-cycle budget is high there. Their wall is
   133.32 ms against a run mean of ~121 ms/advance (14.50 s / 120), i.e. only ~10% above mean
   wall. Pressure *share* is therefore overstated relative to a run-average advance by roughly the
   ratio of the extra passes (~594 extra x ~22 us busy+gap ~ 13 ms of the 133 ms).
3. **Slot detection failed** (confidence 0.081, totals 0), so the report cannot convert occupancy
   into **resident threads**. The brief asked for a resident-thread column; it is not obtainable
   from this trace and the column is omitted rather than guessed.
4. **2.18x distortion.** Everything in section 5 is measured under full encoder isolation, which
   section 6 shows is the dominant distortion (1.81x on its own, before Instruments). Ratios
   between stages that fire once per advance are trustworthy; the pressure family's share is
   inflated because its cost is dominated by per-pass floor, which isolation creates. The size of
   that inflation is measurable from section 6: in the shipping graph pressure is 8.96 of
   21.79 ms busy = **41% of GPU busy**, against **59%** (22.6 of 38.4 ms) in the isolated counter
   run. Read the pressure families' shares in section 5 as an upper bound.

---

## 4. Frame level (capture 1, isolated + counters)

| quantity | value |
|---|---|
| frame wall | **133.32 ms/advance** (p10 119.20, p90 147.44) |
| GPU busy | **40.63 ms/advance = 30.5% of frame wall** |
| attributed interval time | 38.27 ms/advance (incl. 0.13 ms overlapping records) |
| GPU idle gap | **92.70 ms/advance across 2209 gaps** |
| largest gap | **one 40.97 ms / 38.74 ms stall per advance** (the next largest is 0.84 ms) |
| isolation blits | 6.12 ms/advance (1453 clears), excluded from stage totals |
| mean compute occupancy | **13.5%** |
| mean ALU utilisation | **10.1%** |
| mean read / write bandwidth | 13.73 / 5.82 GB/s |
| per-partition occupancy | 0.1367, 0.1351, 0.1312, 0.1386 (four partitions, evenly loaded) |
| GPU-time-area occupied | busy x occupancy / wall = 40.63 x 0.135 / 133.32 = **4.1%** of the M1 Max's compute capacity over the frame |

Channels (all our work is compute; the 44.366 ms is 38.27 ms of solver stages plus 6.12 ms of
isolation blits, and the 5817 intervals are 2 x (1452 passes + 1453 blits)):

| channel | ms/advance | intervals |
|---|--:|--:|
| Compute | 44.366 | 5817 |
| Fragment | 0.021 | 3 |
| Vertex | 0.004 | 1 |

The two retained advances:

| advance | wall ms | busy ms | gap ms | encoders | passes | occupancy | counter samples |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 150.97 | 45.41 | 105.56 | 1685 | 1685 | 11.6% | 196 |
| 1 | 115.67 | 35.84 | 79.84 | 1219 | 1219 | 15.7% | 177 |

**CPU encode time per advance is not available from capture 1** (`cpu.samples: 0`; the time-profile
table is gated behind `--full-diagnostics`). Section 7 substitutes the non-isolated launch trace.

The single big gap is the tell: at 133 ms wall and 40.6 ms busy, 92.7 ms of each advance is GPU
idle, of which ~40 ms sits in **one contiguous stall per advance** rather than being spread across
the 1104 other gaps in that advance (which total ~50 ms). In both analysed advances that stall
falls in exactly the same place -- immediately after `Uniform diagnostics reduction` and before
`Phi support census`, i.e. across the diagnostics readback fence.

---

## 5. Full per-task table (capture 1, sorted by GPU ms/advance)

82 rows, untruncated. `fire/adv` is the mean over the two analysed advances. Occupancy, ALU, LLC,
limiters and bandwidth come from the 180 us counter overlay, so short stages have few or no
samples (`ctr samples` column); a `--` means the stage never overlapped a retained timestamp.
Limiter percentages are the hardware "% of peak" figures. For the stages that dominate the frame
they are all small, which is itself the finding: `mgSmoothColour` (39% of busy) tops out at
Texture Sample 7.4% and LLC 7.1%, `Advect page vertex phi` at ALU 28.1%. The only genuinely
limiter-bound rows are the two conservative-volume normalisation stages -- `uvNormalizeDonors`
(Buffer Write **97.9%**, LLC 95.1%, occupancy 89.1%, 99.3 / 61.0 GB/s read/write) and
`uvNormalizeRows` (LLC **78.8%**, MMU 65.6%, 67.0 / 44.4 GB/s) -- which together are 3.05 ms/advance,
8.0% of busy. `imbal` is peak/mean across the four GPU partitions; 4.00 means the stage ran on one
partition only.

| # | task label | family | fire/adv | ms/adv | % busy | mean us/fire | occ % | ALU % | LLC % | top limiter | 2nd limiter | read GB/s | write GB/s | placement peak/mean | imbal | ctr samples |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|---|---|--:|--:|---|--:|--:|
| 1 | Uniform CM11a mgSmoothColour | pressure V-cycles | 1028 | 14.9145 | 38.98% | 14.5 | 7.3 | 4 | 6.9 | Texture Sample 7.4% | GPU Last Level Cache 7.1% | 9.10 | 1.22 | 0.090 / 0.074 | 1.21 | 166 |
| 2 | Advect page vertex phi | phi transport + redistance | 1 | 2.9592 | 7.73% | 2959.2 | 16.5 | 26.2 | 16.8 | ALU 28.1% | GPU Last Level Cache 17.3% | 6.02 | 1.91 | 0.193 / 0.172 | 1.13 | 16 |
| 3 | uvNormalizeRows | conservative volume gather | 3 | 2.0999 | 5.49% | 700.0 | 43.2 | 5.1 | 75.8 | GPU Last Level Cache 78.8% | MMU 65.6% | 67.02 | 44.38 | 0.438 / 0.430 | 1.02 | 20 |
| 4 | Uniform pressure projection | projection | 1 | 1.5286 | 4.00% | 1528.6 | 15.6 | 13.6 | 16.9 | GPU Last Level Cache 17.0% | ALU 13.9% | 3.87 | 2.04 | 0.181 / 0.127 | 1.43 | 15 |
| 5 | Uniform CM11a mgSolveCoarsest | pressure V-cycles | 15 | 1.4076 | 3.68% | 93.8 | 0.2 | 0.1 | 0.1 | ALU 0.1% | GPU Last Level Cache 0.1% | 0.03 | 0.06 | 0.009 / 0.002 | 4.00 | 18 |
| 6 | Uniform semi-Lagrangian velocity advection and body forces | velocity advection + two-level maps | 1 | 1.2992 | 3.40% | 1299.2 | 14.4 | 27.1 | 21.1 | Texture Sample 48.0% | Texture Filtering 44.3% | 4.72 | 1.43 | 0.145 / 0.144 | 1.01 | 14 |
| 7 | Total surface volume: measure | surface publication | 2 | 1.1092 | 2.90% | 554.6 | 18 | 28.9 | 34.3 | GPU Last Level Cache 34.3% | ALU 32.3% | 7.15 | 3.20 | 0.208 / 0.139 | -- | 10 |
| 8 | uvBuildEdges | conservative volume gather | 1 | 0.9830 | 2.57% | 983.0 | 26 | 24.3 | 49.3 | GPU Last Level Cache 49.3% | ALU 26.7% | 4.35 | 14.48 | 0.269 / 0.262 | 1.03 | 12 |
| 9 | uvNormalizeDonors | conservative volume gather | 3 | 0.9465 | 2.47% | 315.5 | 89.1 | 2.3 | 95 | Buffer Write 97.9% | GPU Last Level Cache 95.1% | 99.26 | 60.98 | 0.952 / 0.914 | 1.04 | 6 |
| 10 | Uniform CM11a mgBuildFinestRhs | pressure setup | 1 | 0.7508 | 1.96% | 750.8 | 18.2 | 21.7 | 17.2 | ALU 22.6% | GPU Last Level Cache 17.2% | 3.97 | 3.13 | 0.207 / 0.132 | -- | 8 |
| 11 | Uniform Sec. 3.3 FIM indirect update 2 | extension (Sec. 3.3 FIM) | 1 | 0.7186 | 1.88% | 718.6 | 2.7 | 4.4 | 6.4 | GPU Last Level Cache 6.5% | ALU 4.4% | 0.05 | 0.48 | 0.039 / 0.027 | 1.42 | 1 |
| 12 | Uniform CM11a mgResidual | pressure V-cycles | 40 | 0.6842 | 1.79% | 17.1 | 5.5 | 3.2 | 5.6 | GPU Last Level Cache 5.6% | Texture Sample 5.6% | 8.03 | 1.90 | 0.202 / 0.068 | 2.99 | 7 |
| 13 | uvProposeSharpen | sharpening | 8 | 0.6812 | 1.78% | 85.2 | 22.3 | 24.3 | 44.2 | GPU Last Level Cache 44.5% | Buffer Read 37.7% | 38.61 | 24.88 | 0.323 / 0.237 | 1.36 | 8 |
| 14 | Redistance page vertex phi | phi transport + redistance | 1 | 0.5123 | 1.34% | 512.4 | 19.8 | 47 | 0.2 | ALU 71.5% | Texture Sample 18.9% | 0.09 | 0.19 | 0.287 / 0.189 | 1.52 | 5 |
| 15 | Uniform CM11a mgBuildFinestTopology | pressure setup | 1 | 0.4986 | 1.30% | 498.6 | 22.9 | 32.2 | 6.4 | ALU 33.8% | Threadgroup/Imageblock Load 9.5% | 1.45 | 2.94 | 0.229 / 0.172 | -- | 5 |
| 16 | Uniform CM11a mgDownsampleSubtract | pressure V-cycles | 37.5 | 0.4341 | 1.13% | 11.6 | 0.3 | 0.2 | 0.5 | GPU Last Level Cache 0.5% | Texture Sample 0.3% | 0.66 | 0.09 | 0.005 / 0.002 | -- | 5 |
| 17 | Uniform CM11a mgProlongateAdd | pressure V-cycles | 37.5 | 0.4285 | 1.12% | 11.4 | 4.9 | 4.6 | 1.1 | ALU 5.7% | Texture Sample 4.6% | 1.50 | 0.75 | 0.123 / 0.031 | -- | 5 |
| 18 | Uniform CM11a mgRestrictResidual | pressure V-cycles | 50 | 0.4028 | 1.05% | 8.1 | 0.1 | 0.2 | 0.8 | Texture Sample 1.8% | MMU 1.8% | 1.05 | 1.49 | 0.003 / 0.001 | 4.00 | 4 |
| 19 | uvGather | conservative volume gather | 1 | 0.3885 | 1.01% | 388.5 | 18.6 | 26.9 | 50.4 | Buffer Read 62.6% | GPU Last Level Cache 50.6% | 33.84 | 1.55 | 0.187 / 0.140 | -- | 5 |
| 20 | uvCommitSharpen | sharpening | 8 | 0.3684 | 0.96% | 46.0 | 23.1 | 24.4 | 33.5 | ALU 34.5% | GPU Last Level Cache 33.6% | 39.32 | 7.83 | 0.298 / 0.132 | -- | 3 |
| 21 | Uniform Sec. 3.3 rho-prime and face authority | extension (Sec. 3.3 FIM) | 1 | 0.3657 | 0.96% | 365.7 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 22 | uvLimitSharpen | sharpening | 8 | 0.3345 | 0.87% | 41.8 | 17.5 | 7.9 | 30.9 | Buffer Read 41.4% | GPU Last Level Cache 31.0% | 34.56 | 8.31 | 0.175 / 0.117 | -- | 2 |
| 23 | Uniform CM11a mgMeasureFineResidual | pressure finish + convergence | 12.5 | 0.2949 | 0.77% | 23.6 | 14.2 | 7.5 | 13.4 | GPU Last Level Cache 13.7% | Texture Sample 13.6% | 17.00 | 1.42 | 0.186 / 0.095 | -- | 6 |
| 24 | Uniform Sec. 3.3 FIM indirect update 1 | extension (Sec. 3.3 FIM) | 1 | 0.2511 | 0.66% | 251.1 | 11.9 | 12.1 | 19.2 | GPU Last Level Cache 19.6% | ALU 12.9% | 18.78 | 3.86 | 0.120 / 0.119 | 1.00 | 1 |
| 25 | Uniform CM11a mgClearPressure | pressure setup | 40 | 0.2410 | 0.63% | 6.0 | 2.3 | 2.1 | 2.3 | ALU 2.6% | GPU Last Level Cache 2.4% | 4.38 | 1.07 | 0.116 / 0.029 | 3.98 | 5 |
| 26 | Total surface volume: dilate | surface publication | 4 | 0.2025 | 0.53% | 50.6 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 27 | Phi support census | surface publication | 1 | 0.1747 | 0.46% | 174.8 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 28 | uvFinishDonorSums | conservative volume gather | 4 | 0.1657 | 0.43% | 41.4 | 32.9 | 11.5 | 28.6 | Buffer Read 63.3% | MMU 38.9% | 67.34 | 0.05 | 0.329 / 0.329 | 1.00 | 2 |
| 29 | Uniform CM11a mgProlongateAssign | pressure V-cycles | 12.5 | 0.1654 | 0.43% | 13.2 | 14.1 | 16.4 | 2.6 | ALU 20.6% | Texture Sample 14.7% | 1.55 | 2.81 | 0.281 / 0.070 | -- | 2 |
| 30 | Uniform Sec. 3.3 hierarchy prolong 6 | extension (Sec. 3.3 FIM) | 1 | 0.1629 | 0.43% | 162.9 | 28.8 | 49.7 | 25.5 | ALU 57.3% | Texture Sample 30.5% | 18.77 | 10.83 | 0.288 / 0.288 | 1.00 | 1 |
| 31 | Surface-deficit page sums | surface publication | 1 | 0.1602 | 0.42% | 160.2 | 11.1 | 14 | 5.9 | ALU 14.7% | Texture Sample 8.0% | 3.38 | 0.06 | 0.111 / 0.037 | -- | 1 |
| 32 | Refresh corrected surface targets | surface publication | 1 | 0.1580 | 0.41% | 158.0 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 33 | uvPrepareSharpen | sharpening | 8 | 0.1575 | 0.41% | 19.7 | 13 | 18.7 | 31.5 | Buffer Read 33.2% | GPU Last Level Cache 31.7% | 44.03 | 2.60 | 0.130 / 0.130 | 1.00 | 1 |
| 34 | Uniform CM11a mgDownsampleMinimum | pressure setup | 12.5 | 0.1513 | 0.40% | 12.1 | 2.6 | 7.9 | 7.1 | Texture Sample 25.6% | Texture Cache 12.1% | 16.79 | 2.35 | 0.075 / 0.020 | -- | 3 |
| 35 | Uniform Sec. 3.3 hierarchy restrict 1 | extension (Sec. 3.3 FIM) | 1 | 0.1425 | 0.37% | 142.5 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 36 | Total surface volume: begin | surface publication | 1 | 0.1268 | 0.33% | 126.8 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 37 | Uniform CM11a mgSaveAccepted | pressure finish + convergence | 10.5 | 0.1229 | 0.32% | 11.7 | 8.6 | 3.1 | 1.7 | ALU 3.9% | MMU 3.1% | 3.04 | 1.76 | 0.138 / 0.086 | 1.60 | 2 |
| 38 | Uniform Geometric two-level uvTwoLevelSeed | velocity advection + two-level maps | 1 | 0.1186 | 0.31% | 118.6 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 39 | Uniform CM11a mgRestoreRejected | pressure finish + convergence | 11.5 | 0.1088 | 0.28% | 9.5 | 17.3 | 8.5 | 5.2 | ALU 11.1% | MMU 5.2% | 8.09 | 8.00 | 0.212 / 0.115 | -- | 2 |
| 40 | Uniform Sec. 3.3 seed active front | extension (Sec. 3.3 FIM) | 1 | 0.0975 | 0.26% | 97.5 | 39.7 | 47.6 | 65.7 | GPU Last Level Cache 65.7% | Texture Sample 57.7% | 34.09 | 15.26 | 0.402 / 0.397 | 1.01 | 1 |
| 41 | Classify 4h sharpening work | sharpening | 1 | 0.0902 | 0.24% | 90.2 | 19 | 17.5 | 12.8 | Texture Sample 29.9% | ALU 19.8% | 19.90 | 6.87 | 0.190 / 0.048 | -- | 1 |
| 42 | Uniform CM11a mgBakeCoefficients | pressure setup | 6 | 0.0898 | 0.24% | 15.0 | 6.2 | 3.8 | 4.6 | GPU Last Level Cache 5.1% | MMU 4.6% | 9.17 | 2.33 | 0.072 / 0.064 | 1.13 | 3 |
| 43 | Uniform CM11a mgExtrapolatePhiOneCell | pressure setup | 6 | 0.0722 | 0.19% | 12.0 | 0.2 | 2.3 | 1.5 | Texture Write 4.6% | ALU 3.0% | 2.48 | 0.64 | 0.002 / 0.001 | -- | 1 |
| 44 | Uniform CM11a mgCheckCycleConvergence | pressure finish + convergence | 11.5 | 0.0691 | 0.18% | 6.0 | 2.3 | 0.8 | 0.2 | ALU 0.8% | GPU Last Level Cache 0.2% | 0.08 | 0.06 | 0.047 / 0.012 | -- | 2 |
| 45 | Uniform CM11a mgCopyPressure | pressure setup | 6 | 0.0684 | 0.18% | 11.4 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 46 | Uniform CM11a mgDownsampleTopology | pressure setup | 5 | 0.0627 | 0.16% | 12.6 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 47 | Uniform diagnostics reduction | other | 1 | 0.0573 | 0.15% | 57.3 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 48 | Uniform Sec. 3.3 hierarchy restrict 4 | extension (Sec. 3.3 FIM) | 1 | 0.0561 | 0.15% | 56.1 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 49 | Uniform Sec. 3.3 hierarchy restrict 2 | extension (Sec. 3.3 FIM) | 1 | 0.0560 | 0.15% | 56.0 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 50 | Uniform Sec. 3.3 hierarchy restrict 5 | extension (Sec. 3.3 FIM) | 1 | 0.0538 | 0.14% | 53.8 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 51 | Uniform Sec. 3.3 hierarchy restrict 3 | extension (Sec. 3.3 FIM) | 1 | 0.0510 | 0.13% | 51.0 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 52 | Uniform Sec. 3.3 hierarchy prolong 5 | extension (Sec. 3.3 FIM) | 1 | 0.0455 | 0.12% | 45.5 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 53 | Uniform Sec. 3.3 hierarchy prolong 4 | extension (Sec. 3.3 FIM) | 1 | 0.0390 | 0.10% | 39.0 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 54 | Uniform Sec. 3.3 hierarchy prolong 2 | extension (Sec. 3.3 FIM) | 1 | 0.0385 | 0.10% | 38.5 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 55 | Uniform Sec. 3.3 hierarchy prolong 3 | extension (Sec. 3.3 FIM) | 1 | 0.0365 | 0.10% | 36.5 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 56 | Total surface volume: solve | surface publication | 2 | 0.0351 | 0.09% | 17.5 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 57 | Uniform Sec. 3.3 hierarchy restrict 6 | extension (Sec. 3.3 FIM) | 1 | 0.0347 | 0.09% | 34.7 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 58 | Uniform Sec. 3.3 resolve converged front | extension (Sec. 3.3 FIM) | 1 | 0.0335 | 0.09% | 33.5 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 59 | Uniform CM11a mgAddPressure | pressure finish + convergence | 2.5 | 0.0315 | 0.08% | 12.6 | 25.2 | 17.3 | 19.1 | Texture Sample 23.2% | ALU 20.7% | 32.81 | 11.07 | 0.252 / 0.063 | -- | 1 |
| 60 | Uniform Sec. 3.3 hierarchy prolong 1 | extension (Sec. 3.3 FIM) | 1 | 0.0302 | 0.08% | 30.3 | 0 | 0 | 0.1 | GPU Last Level Cache 0.1% | MMU 0.0% | 0.05 | 0.04 | 0.001 / 0.000 | 4.00 | 1 |
| 61 | Uniform CM11a mgShiftMinimum | pressure setup | 2.5 | 0.0302 | 0.08% | 12.1 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 62 | Total surface volume: reduce | surface publication | 2 | 0.0282 | 0.07% | 14.1 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 63 | Uniform Sec. 3.3 publish 4h face table | surface publication | 1 | 0.0282 | 0.07% | 28.2 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 64 | uvMarkSharpenPages | sharpening | 1 | 0.0271 | 0.07% | 27.1 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 65 | Surface-deficit global balance | surface publication | 1 | 0.0264 | 0.07% | 26.4 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 66 | Uniform Geometric surface publication | surface publication | 1 | 0.0251 | 0.07% | 25.1 | 22.9 | 20.4 | 13.8 | Texture Sample 36.9% | ALU 23.3% | 15.61 | 11.21 | 0.229 / 0.057 | -- | 1 |
| 67 | Total surface volume: seed | surface publication | 1 | 0.0243 | 0.06% | 24.3 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 68 | Total surface volume: metric | surface publication | 1 | 0.0232 | 0.06% | 23.2 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 69 | Phi support reduction | surface publication | 1 | 0.0222 | 0.06% | 22.2 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 70 | uvMarkTransportPages | phi transport + redistance | 1 | 0.0201 | 0.05% | 20.1 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 71 | Total surface volume: apply | surface publication | 1 | 0.0195 | 0.05% | 19.5 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 72 | Phi support closure | surface publication | 1 | 0.0175 | 0.05% | 17.5 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 73 | uvFallback | conservative volume gather | 1 | 0.0174 | 0.04% | 17.4 | 0.3 | 0 | 0.1 | GPU Last Level Cache 0.1% | MMU 0.0% | 0.04 | 0.06 | 0.003 / 0.001 | -- | 1 |
| 74 | Surface volume capacities | surface publication | 1 | 0.0150 | 0.04% | 15.0 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 75 | Uniform Geometric two-level uvTwoLevelDilateZ | velocity advection + two-level maps | 1 | 0.0138 | 0.04% | 13.8 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 76 | Compact volume pages | conservative volume gather | 2 | 0.0135 | 0.03% | 6.8 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 77 | Uniform Geometric two-level uvTwoLevelDilateY | velocity advection + two-level maps | 1 | 0.0125 | 0.03% | 12.5 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 78 | Uniform Geometric two-level uvTwoLevelDilateX | velocity advection + two-level maps | 1 | 0.0104 | 0.03% | 10.4 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 79 | Uniform Sec. 3.3 prepare active dispatch 3 | extension (Sec. 3.3 FIM) | 1 | 0.0066 | 0.02% | 6.6 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 80 | Uniform Sec. 3.3 prepare initial active dispatch | extension (Sec. 3.3 FIM) | 1 | 0.0064 | 0.02% | 6.4 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 81 | Uniform Sec. 3.3 prepare active dispatch 2 | extension (Sec. 3.3 FIM) | 1 | 0.0062 | 0.02% | 6.2 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| 82 | Uniform CM11a mgFinishSafety | pressure finish + convergence | 1 | 0.0039 | 0.01% | 3.9 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |

**Multigrid levels cannot be separated.** Every smoother dispatch across all levels carries the
single label `Uniform CM11a mgSmoothColour` (`webgpu-uniform-pressure-multigrid.ts` labels by
kernel, not by level), so the 1028 firings/advance cannot be grouped by level from this trace. The
same applies to `mgResidual`, `mgRestrictResidual`, `mgProlongateAdd` and `mgDownsampleSubtract`.
The only level-resolved rows are `mgSolveCoarsest` (coarsest only) and `mgBuildFinest*` (finest only).

Short labels `measure`, `dilate`, `begin`, `solve`, `reduce`, `seed`, `metric`, `apply` are the
report's stripped form of `Total surface volume: <x>`; they are spelled out above.

### Stage-family aggregates

| family | labels | fire/adv | ms/adv | % of busy | ms-weighted occ % | ms-weighted ALU % |
|---|--:|--:|--:|--:|--:|--:|
| pressure V-cycles | 7 | 1220.5 | 18.437 | 48.18% | 6.4 | 3.6 |
| conservative volume gather | 7 | 15 | 4.614 | 12.06% | 46.4 | 10.7 |
| phi transport + redistance | 3 | 3 | 3.492 | 9.12% | 17 | 29.3 |
| extension (Sec. 3.3 FIM) | 20 | 20 | 2.232 | 5.83% | 10.7 | 15 |
| surface publication | 17 | 23 | 2.196 | 5.74% | 17.2 | 26.9 |
| pressure setup | 9 | 80 | 1.965 | 5.13% | 14.7 | 19.2 |
| sharpening | 6 | 34 | 1.659 | 4.34% | 20.4 | 20 |
| projection | 1 | 1 | 1.529 | 3.99% | 15.6 | 13.6 |
| velocity advection + two-level maps | 5 | 5 | 1.454 | 3.80% | 14.4 | 27.1 |
| pressure finish + convergence | 6 | 49.5 | 0.631 | 1.65% | 12.9 | 6.6 |
| other | 1 | 1 | 0.057 | 0.15% | -- | -- |

### Distribution of per-firing durations

2902 intervals across the two analysed advances, totalling 38.361 ms/advance (0.1% off the
report's 38.267 ms; see the exclusion note below).

| bucket | firings/adv | % of firings | ms/adv | % of GPU busy |
|---|--:|--:|--:|--:|
| <20 us | 1277.0 | 88.01% | 14.209 | 37.04% |
| 20-50 us | 100.5 | 6.93% | 3.528 | 9.20% |
| 50-200 us | 50.0 | 3.45% | 5.153 | 13.43% |
| 200 us - 1 ms | 20.0 | 1.38% | 9.080 | 23.67% |
| >1 ms | 3.5 | 0.24% | 6.390 | 16.66% |

Pressure stages only (`Uniform CM11a mg*` + `Uniform pressure projection`), 1350 firings/adv
totalling 22.606 ms/adv -- which reconciles with the 22.562 ms the four pressure families sum to
in the table above:

| bucket | firings/adv | ms/adv |
|---|--:|--:|
| <20 us | 1255.5 | 13.912 |
| 20-50 us | 61.5 | 2.249 |
| 50-200 us | 26.0 | 2.760 |
| 200 us - 1 ms | 6.0 | 2.130 |
| >1 ms | 1.0 | 1.556 |

**88.0% of all firings are under 20 us and they account for 37% of GPU busy time; 98.3% of that
sub-20-us population is pressure, and 75.6% of it is `mgSmoothColour` alone.** At the other end,
3.5 firings/advance longer than 1 ms carry 16.7% of busy time: `Advect page vertex phi` 2.98 ms,
`Uniform pressure projection` 1.56 ms, `Uniform semi-Lagrangian velocity advection and body
forces` 1.31 ms, and a single `uvBuildEdges` firing (0.55 ms/advance averaged over the two
advances) that crosses 1 ms in only one of them.

*Exclusion.* Two Metal interval records were dropped from this distribution because each fully
contains many others and is therefore a resumed/coalesced execution slice rather than a real
firing: one `mgResidual` record of 1962 us spanning 29 other intervals, and one `mgSmoothColour`
record of 353 us spanning 8. The report's own per-task table already discounts them (raw label sum
39.518 ms/adv vs reported 38.267 ms/adv, of which 0.98 ms is that one `mgResidual` record). This
is the report's "6 resumed execution slices coalesced" note made explicit.

---

## 6. Dawn's WebGPU-pass to Metal-encoder mapping (the mid-task question)

Both launch traces encode the *identical* graph: `gpuCommandAudit` reports `computePasses: 10434`,
`commandBuffers: 14`, `indirectDispatches: 48` in all four 12-step runs, and the encoder-list
census implies 10 441 WebGPU compute passes in both traces (10 434 audited + 7 bootstrap/sentinel).

### 6.1 Encoders and command buffers per advance

Nine interior advances (anchor-delimited) from each launch trace:

| | isolated | non-isolated | ratio |
|---|--:|--:|--:|
| WebGPU compute passes / advance (from labels) | 755.9 | 755.4 | 1.00 |
| Metal **compute** encoders / advance | **755.9** | **26.1** | 29.0x |
| Metal blit encoders / advance | 753.4 | 26.1 | |
| Metal encoders / advance (total) | 1509.3 | 52.2 | 28.9x |
| **WebGPU passes per Metal compute encoder** | **1** | **28.93** | |
| command buffers / advance | 1 | 1 | |
| `num-encoders` on each steady step's command buffer | 1506 -- 1509 | 52 -- 56 | |

Whole-trace encoder census (all submissions, incl. bootstrap):

| | isolated | non-isolated |
|---|--:|--:|
| Metal encoders created | 20897 | 790 |
| ... blit | 10454 | 399 |
| ... compute | 10441 | 391 |
| ... compute encoders carrying **more than one** WebGPU pass | **0** | **294** of 391 |
| WebGPU passes implied by the concatenated labels | 10441 | 10441 |

In the shipping (non-isolated) graph Dawn packs **28.93 WebGPU compute passes into each Metal
compute encoder**, 294 of 391 compute encoders hold more than one pass (median 5 passes per
encoder, p90 22, **maximum 466** -- one Metal encoder absorbing 466 consecutive WebGPU compute
passes, labels concatenated with " & "), and the encoder boundaries fall exactly where the solver
itself issues a blit -- 399 blit encoders interleaved with 391 compute encoders. Isolation turns
this into strict 1:1 (10 441 compute encoders, 0 merged) plus one 4-byte clear blit each.

### 6.2 Wall clock: how much isolation itself costs

All at 12 steps, same process, same encoded graph (`computePasses: 10434` in every arm),
`simulationWall_ms / 12`:

| arm | ms/advance | vs shipping |
|---|--:|--:|
| untraced, non-isolated (**shipping**) | **38.75** | 1.00x |
| untraced, **isolated** | **70.33** | **1.81x** |
| traced (Metal System Trace), non-isolated | 48.17 | 1.24x |
| traced (Metal System Trace), **isolated** | 99.33 | 2.56x |
| traced + counters, isolated, 120 steps (capture 1) | 79.98 (baseline 36.77) | 2.18x |

Isolation alone costs **+81%** on this 755-pass frame, and Instruments adds a further ~25-30% on
top. For reference the skill quotes ~1% for the CM12 lane, which has 168 passes per advance.

### 6.3 Pressure bracket, per-encoder durations and gaps

Pressure = `Uniform CM11a mg*` and `Uniform pressure projection`, nine interior advances:

| | isolated | non-isolated |
|---|--:|--:|
| pressure WebGPU passes / advance | 653.7 | 652 |
| pressure **Metal encoders** / advance | **653.7** | **12** |
| pressure GPU busy / advance | **12.576 ms** | **8.963 ms** |
| per-encoder duration -- median | **12.29 us** | **124.12 us** |
| per-encoder duration -- p90 | 15.12 us | 1520.0 us |
| per-encoder duration -- mean / min / max | 19.24 / 3.92 / 1560 us | 746.88 / 77.33 / 5180 us |
| inter-encoder GPU idle gap -- median | 24.25 us | 49.29 us |
| inter-encoder GPU idle gap -- p90 | 61.62 us | 63.00 us |
| inter-encoder gap -- mean / max | 38.50 / 833 us | 51.40 / 162 us |
| **total idle gap inside the pressure bracket** | **25.12 ms/advance** | **0.565 ms/advance** |
| gap count sampled | 5873 | 99 |
| whole-advance GPU busy | 29.23 ms | 21.79 ms |
| whole-advance wall | 88.18 ms | 45.34 ms |

The ~12 us median "pass floor" seen under isolation is an artefact of isolation: the same 652
pressure dispatches, packed by Dawn into 12 Metal encoders, run in 8.96 ms of GPU busy time
with 0.56 ms of inter-encoder idle, against 12.58 ms busy and 25.1 ms idle when each dispatch gets its
own encoder. Isolation adds 3.61 ms of busy and 24.6 ms of idle to the pressure bracket alone, i.e.
28.2 ms of the 42.8 ms/advance that isolation adds to the frame.

There is **no measurable per-dispatch gap in the shipping graph at all**. Dispatches inside one
Metal encoder are back-to-back and invisible on the Metal GPU track; the only idle the hardware
shows is the 49.3 us median between the 12 pressure *encoders*, totalling 0.56 ms/advance. The
~13 us floor and the 25 ms of inter-dispatch idle are both produced by the profiling
configuration, not by the solver's 652 dispatches.

One caveat in the other direction: the non-isolated regime cannot separate the 652 dispatches, so
its 8.96 ms of pressure busy is a bracket total, not a sum of per-dispatch costs. Any per-dispatch
attribution in section 5 necessarily comes from the isolated capture and carries its inflation.

### 6.4 Pressure cycles actually encoded (the lagged budget)

Counted, not planned. The plan maximum quoted in the brief is 2187 passes/advance.

| window | compute passes / advance | pressure passes / advance | `mgSmoothColour` / advance |
|---|--:|--:|--:|
| launch traces, 9 interior advances of a 12-step run | 755.4 | **652** | 488 |
| 12-step run, whole-run audit average | 869.5 | 768.5 (88.4% of passes) | 578.0 |
| 120-step run, whole-run audit average | 857.9 | -- | 569.0 |
| capture 1's two analysed advances | 1452 | **1351** | **1028** |

So the lagged budget spends roughly **650-1350 pressure passes per advance**, 30-62% of the 2187
plan maximum, and it varies by more than 2x across the run. `mgSmoothColour` alone is 56-71% of
all compute passes in every window measured.

---

## 7. CPU side

Capture 1 carries no CPU samples. The table below is from the **non-isolated launch trace**
(`time-profile`, 2831 samples total), restricted to the main thread inside the anchor-delimited
stepping window (461.4 ms, 243 main-thread samples, all in `Running` state).

Instruments samples the main thread every 1 ms while it is running (measured median inter-sample
delta 1000 us), so the sample count is directly a CPU-time measurement:

| quantity | value |
|---|--:|
| window | 461.4 ms, ~10.2 advances at 45.34 ms/advance |
| main-thread CPU time in window | 243 ms |
| **main-thread CPU per advance (encode + submit + JS)** | **~23.9 ms** |
| main thread running share of wall | **52.7%** |
| GPU busy per advance (same trace) | 21.79 ms |

So in the *shipping* (non-isolated) regime CPU encode and GPU execution are roughly equal in
magnitude -- ~24 ms of main-thread work against ~22 ms of GPU busy inside a 45 ms advance -- and
neither saturates the frame. Subtracting the tracing overhead below puts real CPU work nearer
17 ms/advance.

| group | samples | share of main-thread running time |
|---|--:|--:|
| Dawn Metal backend / driver | 100 | 41.2% |
| Dawn native (validation, encoding, state) | 92 | 37.9% |
| Node / V8 / JS | 51 | 21.0% |

Top leaf symbols:

| symbol | samples | share |
|---|--:|--:|
| `__kdebug_trace64` | 68 | 28.0% |
| `<deduplicated_symbol>` | 9 | 3.7% |
| `kevent` | 8 | 3.3% |
| `_xzm_free` | 7 | 2.9% |
| `dawn::native::ErrorMonad::IsError() const` | 5 | 2.1% |
| `_xzm_xzone_malloc_freelist_outlined` | 4 | 1.6% |
| `v8::internal::LookupIterator::State v8::internal::LookupIterator::LookupInRegularHolder<false>(v8::internal::Tagged<v8::internal::Map>, v8::internal::Tagged<v8::internal::JSReceiver>)` | 4 | 1.6% |
| `__kdebug_trace_string` | 4 | 1.6% |
| `_xzm_xzone_malloc_tiny` | 4 | 1.6% |
| `dawn::native::BindGroupBase::GetBindingAsTextureView(dawn::detail::TypedIntegerImpl<dawn::native::BindingIndexT, unsigned int>) const` | 3 | 1.2% |
| `malloc_type_malloc` | 3 | 1.2% |
| `pthread_mutex_lock` | 3 | 1.2% |
| `dawn::native::SyncScopeUsageTracker::AddBindGroup(dawn::native::BindGroupBase*)` | 3 | 1.2% |
| `v8::internal::(anonymous namespace)::Invoke(v8::internal::Isolate*, v8::internal::(anonymous namespace)::InvokeParams const&)` | 3 | 1.2% |
| `node::errors::TryCatchScope::~TryCatchScope()` | 3 | 1.2% |

`__kdebug_trace64` at 28% is Instruments' own signpost emission inside the Dawn Metal backend --
it is the tracing overhead, not solver work; it does not exist in an untraced run. Discounting it,
the remaining main-thread time splits roughly evenly between Dawn's native layer (bind-group and
encoder bookkeeping, `ErrorMonad::IsError`, `BindGroupBase::GetBindingAsTextureView`) and
Node/V8. No single JS frame dominates.

---

## 8. Per-WGSL-entry-point GPU time: not obtainable

Requested as capture 2's deliverable. It cannot be produced on this machine without
`MTL_CAPTURE_ENABLED`, which the brief forbids.

| table | rows exported | note |
|---|--:|---|
| `metal-shader-profiler-intervals` | **0** (2096 B, schema only) | empty in *launch* mode too, not only when attaching |
| `gpu-shader-profiler-interval` | 0 (749 B) | |
| `gpu-shader-profiler-sample` | 0 (532 B) | |
| `metal-gpu-counter-profile` | 0 (338 B) | |
| `metal-shader-profiler-shader-list` | 1020 rows (308 ours) | **no timing**: 88 distinct shader labels, 158 distinct compute pipelines, PC ranges only |

The skill's trap 5 predicts empty shader intervals when counters force an attach; this capture
shows they are also empty under `--launch` with the plain Metal System Trace template. The
practical substitute is already in section 5: under full isolation one Metal encoder = one
dispatch = one pipeline, so the 82-row task table *is* a per-entry-point breakdown, at the cost of
the 1.81x isolation distortion.

---

## 9. Capture 3 (larger scene): skipped — now done, see `xctrace-capture-large.md`

**Superseded 2026-09-21.** The tool work described below was done:
`tools/profile-mini-dam-xctrace.ts` now takes `--uniform-scene`, `--uniform-lane`
and `--expect-grid` (defaults unchanged), and `cm12-figure-7` (128³) was added to
the smoke catalog. The large-scene captures, the per-level multigrid tables and
the CPU/GPU serialisation measurement live in
`docs/research/uniform-geometric-gpu-utilization-2026-09-21/xctrace-capture-large.md`.
The original note follows.

`tools/profile-mini-dam-xctrace.ts` has no `--scene` or `--grid` flag; `--uniform-mini-64` hard-codes
`FLUID_SCENE=minimal-power-dam-break-64`, `FLUID_LANE=uniform-one-step` and
`FLUID_EXPECT_GRID=64,64,64`. Driving `large-power-dam-break` or
`mass-conserving-figure-9-dam-break` would need a second lane preset (scene + a lane that accepts
the method + expected grid + a step budget that keeps a 4 s counter window inside the stepping
phase) plus a fresh untraced baseline and a counter run. That is more than the "modest tool work"
the brief allows, and the WebGPU lease was contended by three other live sessions throughout.
Skipped, per the brief's own instruction to say so.

---

## 10. Anomalies

1. **Capture 2's isolated launch run took 31.1 s to construct** (`construction_ms: 31132`) against
   742-790 ms for the other three 12-step runs. Pipeline construction under Metal System Trace with
   isolation on; the stepping phase itself was unaffected (`simulationWall_ms 1192`).
2. **`gpuPassBoundaryAudit.labelIsolated` reports `false` in every arm**, including the isolated
   ones. That field tracks the pass-boundary *broker*, a different mechanism from
   `isolateComputePassEncoders`. The encoder isolation demonstrably ran (10 454 blit encoders
   against 399 in the control). The field is misleading as an isolation indicator.
3. **Acceptance findings are red in every log** (`uniform-64-grid`, `uniform-64-pressure-converged`,
   `pack.*.method-evidence`, ...). They are keyed to `methods.uniform.*` and the run publishes under
   `methods.uniform-volume.*`; `FLUID_PERFORMANCE_PROFILE=1` sets
   `qualityGates: "final-authority-only"` so they are not enforced. They are not evidence of a
   broken run: `uniformCM11aConverged: true`, `volumeDrift -1.0e-4`, `rejectedAdvanceAttempts 0`.
4. **`uniformFIMConverged: false`** with `uniformFIMTerminalActiveFaces: 17218` and
   `uniformFIMExecutedPasses: 2` in every run -- the Sec. 3.3 extension hits its pass budget rather
   than converging. Noted as observed state, not diagnosed.
5. **`mgSolveCoarsest` reports occupancy 0.2% and ALU 0.1%** over 15 firings/advance at 94 us
   mean, on 18 counter samples -- the samples exist, so this is a genuinely near-empty coarse grid
   rather than a missing sample. It still costs 1.41 ms/advance (3.7% of busy), all of it latency.
   `mgRestrictResidual` (0.1% occupancy) and `mgDownsampleSubtract` (0.3%) are the same shape, and
   all three report `imbal` 4.00 or a single populated partition, i.e. the work fits on one
   partition.
6. **`uvNormalizeDonors` is the only high-occupancy stage** (89%) in the whole table; the next
   highest is `uvNormalizeRows` at 43%. Everything else, including every pressure stage, sits
   under 30%.
7. **One ~40 ms GPU idle stall per advance** (section 4), always across the diagnostics readback
   fence, is 43% of the 92.7 ms gap budget and 30% of the 133 ms frame wall. It is present under
   isolation + counters; the non-isolated launch trace shows 45.34 ms wall against 21.79 ms busy,
   so a stall of the same shape (~23 ms) exists there too.
