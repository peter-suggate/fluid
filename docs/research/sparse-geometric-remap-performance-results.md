# Sparse geometric remap performance results

## S0: native stage-timing baseline

This checkpoint establishes the native, no-UI timing baseline before S1. All
four arms used one freshly built release binary and the same source snapshot.
The in-frame values below come from monotonic timers inside `World` and the
cellwise remap; they exclude process startup, scene parsing, JSON serialization,
and the TypeScript wrapper.

Configuration for both scenes was `dt = 1/30 s`, 256 pressure iterations, and
relative pressure tolerance `1e-6`. Remap requested one trace segment and one
edge interval with `BandProjection`. The exact live catalog scenes were
`cm12-figure-7` and `coarse-first-pool-impact-half`.

### Provenance

- Git commit: `7f6a2cdc196c0703892400a31fd5aa45a84c08b7`
- Native binary: `rust/target/release/examples/verify_world`
- Native binary SHA-256: `ab235cf13446a560e3e9f5e6979cbc0e1c08c3ba5b1c72d1a5302abfacb7ca50`
- `adaptive_remap.rs`: `74a88761287b11fdaae3cfd2b836f4f8ddc21e9989eead513eabcc2512c40114`
- `numerics.rs`: `72568d3e15744f329cc0b3161af64ed4fcc3efba4a3be512815cca2a8d4d1933`
- `world.rs`: `b17bee75cc9aab35ffdd4ab505bb2a02468b3fb7fb04b3d8e0b0968f29f527f6`
- `band_projection.rs`: `b73a0fbe0c4c1ff85f23489b818ce83ee7582c74f1cb09b3fcf56c895aaeb127`
- `resolution.rs`: `3c529adf10859e8d2f8dbac3bb5a274dd138e7a77e87f5e2da1e99df482d1770`
- `verify_world.rs`: `fc9c155b2af0a319ca87fc2348a2723b6ce6fe80818191e2f234380b74d3bd66`

The working tree was intentionally not clean, so the file and binary hashes,
rather than the commit alone, identify the measured implementation.

### Mean stage time

All values are milliseconds per simulated frame. `other` closes the timing
partition, so the named World stages plus `other` equal `totalAdvance` exactly.

| scene and arm | total | field build | primary pressure | support and transfer | post-support pressure | transport | post-transport | resolution and publication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Figure 7 baseline, 30 frames | 68.652 | 4.938 | 7.591 | 6.400 | 0 | 22.713 | 0.777 | 26.231 |
| Figure 7 remap, 30 frames | 332.900 | 2.978 | 6.978 | 6.148 | 1.351 | 298.243 | 0.618 | 16.585 |
| half-pool baseline, 10 frames | 15.146 | 1.439 | 0.765 | 4.691 | 0 | 1.378 | 0.233 | 6.640 |
| half-pool remap, 10 frames | 171.480 | 1.419 | 2.020 | 5.831 | 0.782 | 151.154 | 0.371 | 9.903 |

The remap-only split is exclusive. `streamfunctionExtension` includes extension
solve and streamfunction assembly, excluding harmonic continuation.

| scene | total remap | base field | receiver closure | extension and assembly | harmonic fill | diagnostics | trace | geometry and refinement | gather | commit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Figure 7 | 296.315 | 0.713 | 2.026 | 18.778 | 163.064 | 1.344 | 99.535 | 9.653 | 1.054 | 0.003 |
| half-pool | 150.320 | 0.458 | 1.072 | 3.004 | 3.367 | 1.362 | 129.804 | 10.451 | 0.695 | 0.002 |

### Selected frame costs and work

| scene | frame | baseline total ms | baseline microsteps | remap total ms | traces | RK evaluations | receiver-band folds raw/corrected |
|---|---:|---:|---:|---:|---:|---:|---:|
| Figure 7 | 1 | 7.783 | 1 | 248.358 | 9,875 | 39,500 | 0 / 0 |
| Figure 7 | 5 | 8.286 | 3 | 235.467 | 9,167 | 183,340 | 0 / 0 |
| Figure 7 | 10 | 31.100 | 5 | 254.830 | 9,469 | 340,884 | 3 / 6 |
| Figure 7 | 20 | 61.229 | 9 | 308.024 | 10,927 | 786,744 | 18 / 29 |
| Figure 7 | 25 | 70.629 | 12 | 763.754 | 11,785 | 4,808,280 | 0 / 0 |
| Figure 7 | 28 | 181.899 | 33 | 696.181 | 12,762 | 3,471,264 | 21 / 22 |
| Figure 7 | 30 | 166.854 | 23 | 400.881 | 11,903 | 1,237,912 | 14 / 22 |
| half-pool | 1 | 3.802 | 1 | 74.741 | 6,681 | 267,240 | 0 / 0 |
| half-pool | 5 | 13.146 | 3 | 44.938 | 6,953 | 139,060 | 0 / 2 |
| half-pool | 10 | 15.041 | 5 | 205.274 | 9,087 | 1,490,268 | 5 / 2 |

Figure 7 baseline used 334 material microsteps in total, with a maximum of 51
in one frame. The half-pool baseline used 29, with a maximum of five. Remap
used zero material microsteps. Maximum remap work was 4,808,280 RK evaluations
for Figure 7 and 4,634,028 for the half-pool.

All four arms completed. Remap frames committed with no numerical fault or
velocity fallback; material-relevant fold checks were zero; gather total and
worst-donor residuals stayed inside their published bounds. Receiver-band fold
counts in the table are diagnostics and can be nonzero where the folded region
cannot reach donor PLIC material; they were recorded rather than treated as
zero or silently omitted.

### Reproduction

```bash
cargo build --manifest-path rust/Cargo.toml --release -p fluid-core --example verify_world

node --import tsx tools/benchmark-cellwise-remap-native.ts \
  --scene=cm12-figure-7 --mode=baseline --arm=s0-figure7-baseline --frames=30
node --import tsx tools/benchmark-cellwise-remap-native.ts \
  --scene=cm12-figure-7 --mode=cellwise-remap --arm=s0-figure7-remap --frames=30
node --import tsx tools/benchmark-cellwise-remap-native.ts \
  --scene=coarse-first-pool-impact-half --mode=baseline --arm=s0-pool-baseline --frames=10
node --import tsx tools/benchmark-cellwise-remap-native.ts \
  --scene=coarse-first-pool-impact-half --mode=cellwise-remap --arm=s0-pool-remap --frames=10
```

Machine-readable reports are retained locally under
`artifacts/sparse-geometric-remap-native/s0/`:

- `s0-figure7-baseline.json`
- `s0-figure7-remap.json`
- `s0-pool-baseline.json`
- `s0-pool-remap.json`

### Limits

This is one sample per arm on one development host. P95 values in the raw
reports are percentiles across frames with different physical work, not repeat
samples of the same frame. Wrapper times include startup and JSON overhead and
are not used for comparisons. No UI or Dawn process ran. This checkpoint does
not establish a 90-frame Figure 7 result, and it does not measure later S1-S5
implementations.

## S1 and S2: failed go/no-go measurements

The standalone S1 checkpoint removed harmonic continuation and made a trace
outside represented streamfunction support fail closed. Its exact half-pool
lane rejected frame 1 with 2,555 support-exit samples. This disproved the S1
premise that the old zero fallback counter showed that harmonic continuation
was unused: the old full-domain active mask had hidden those exits.

S2 then restricted tracing to subfaces touching the receiver band, retained a
two-interval floor without the former additional doubling, removed the fifth
RK diagnostic sample, and removed the full-cell differential census. The joint
S1+S2 checkpoint still failed its go/no-go lanes:

| lane | first rejected frame | support exits | traced subfaces | unique traces | RK evaluations | mean total/remap/trace ms through rejection |
|---|---:|---:|---:|---:|---:|---:|
| half-pool | 8 | 3,171 | 2,205 | 4,439 | 390,632 | 52.412 / 27.666 / 18.275 |
| Figure 7 | 3 | 64 | 1,764 | 3,431 | 41,172 | 60.096 / 37.732 / 4.352 |

Both rejected receipts reported zero harmonic time, zero fifth-sample counter,
and no completed gather or material commit. The pool exit began at `[45,44]`
on an intra-brick subface between represented dry extension cells and reached
`[48.0179392560,44.8644774551]`, just beyond the represented page. The Figure
7 exit began at `[48,104]` on a brick-face subface between represented dry
cells and reached `[47.9999830981,104.1113041509]` above represented support.
These are real dry outer-band paths into SparseAir, rather than ambiguous
source-face ownership lookups. Whether those paths can contribute material
must be decided before expanding support or relaxing the exit gate.

The diagnostic rebuild used binary SHA-256
`99ed3275d00b719388b944716961534747e801f661923b8ea3c8b6ed47721a2f`.
Source hashes were captured before each run: `adaptive_remap.rs`
`f506002910cb2a6856424b28708e254960fabbc3eba5902939c73f6ded00e39f`,
`numerics.rs` `be614fa62dedc882375172716506a6180ac60efcf38c2f787a6f85c7db6eaf1f`,
and `world.rs`
`b17bee75cc9aab35ffdd4ab505bb2a02468b3fb7fb04b3d8e0b0968f29f527f6`.
Compact reports are retained locally under
`artifacts/sparse-geometric-remap-native/s1-s2/` as
`pool-f8-failed.json` and `figure7-f3-failed.json`. The detailed pool stage
capture is `/tmp/s1-s2-pool-f8-full-stages.json`; that temporary path is
diagnostic evidence on this host, not a durable fixture.

The joint checkpoint is a failed go/no-go result. It provides no Figure 7
frame-25, frame-28, or frame-30 gradient prerequisite data, no sustained-impact
claim, and no reason to publish production artifacts.
