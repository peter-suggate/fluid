# Narrowband FLIP kernel optimization

The follow-up [ten executed experiments](nbflip-ten-experiments-2026-11-02.md)
adds measured comparisons for every section below, including layout conversion,
indirect launches, cost queues, fusion, splitting and full-solver coordination.

The current solver already uses spatial particle slabs, compact surface jobs,
shared-memory particle deposits, a deferred contact walk, and asynchronous
receipts. Optimize those complete producer/consumer paths before changing the
particle ABI or introducing another spatial sort. Preserve the numerical
operator, collision behavior, and authored geometry.

None of the tested rewrites established a consistent improvement on the measured
workloads, so this change adds benchmark coverage and retains the runtime kernels.

This audit describes the working tree on 2 November 2026. Existing particle
contact, body-window, and fine-slot work is outside this change. Byte counts
below are logical accesses derived from the shaders, not measured DRAM traffic.

## Memory layout and bandwidth

| Kernel | Addresses visited by neighbouring lanes | Layout decision and logical traffic |
| --- | --- | --- |
| `advectParticlesDeferred` | Consecutive records within a brick slab; a FLIP record is 48 B, APIC is 112 B. Position components therefore have a 48/112 B lane stride. Velocity samples gather spatial neighbours. | Full-record AoS serves a mover that updates the whole record. SoA could improve position-only scans, but adds streams to movement and sorting. The existing separate 16 B position/state stream already serves those scans. Compare complete move + sort + publication before changing the ABI. |
| `surfaceSplatOccupied` | Lane `i` reads position at `NB_LINKS + 4*i + {0,1,2}` and hot state at `+3`. Lanes scatter into two 343-word shared arrays, then lanes flush consecutive window indices. | The 16 B compact AoS consumes all four words for a contributing sample. SoA offers little payload reduction here. A job reads at most 512 such samples: 8 KiB. Initializing both windows writes 2,744 B to workgroup memory, and each window is scanned at publication. Sparse/noncontributing vertices skip global updates. |
| `transferDeposit` | Consecutive particle records in an occupied tile; each sample scatters to 27 nodes for each of three MAC components. Flush lanes traverse consecutive component-major slots. Global destination words are six-word cell records: momentum/weight pairs for three axes. | 81 contributions and 162 local integer atomics per supported sample. This is an operation count, not a DRAM byte count. Component SoA would coalesce the flush better, but requires migrating its readers too. The current tile window amortizes global atomics across particles. |
| `pRelax` | Each half of a 64-lane group visits one 4³ tile, one red/black colour at a time. The x coordinate advances in colour order; six neighbour addresses depend on row kinds and brick ownership. | Pressure fields are already separate planes in an arena. A six-band-neighbour row reads kinds, RHS, diagonal, six iterates and writes one iterate: 40 B before address metadata. With six stored coefficients this rises to 64 B; unit coefficients come from uniforms. Cache reuse and special row types change actual traffic. Keep scalar planes; benchmark local iterate staging with the same sweep semantics. |
| `arenaDistanceCanonical` | A group visits one brick; lanes visit nearby vertices and queued candidate cells. Candidate corners and neighbouring records are shared spatially, but the exact distance solve has variable work. | Tiled execution fits the access pattern. Another full per-brick packet must pay for its halo and candidate construction. Split or shrink private candidate state only after measuring the complete distance/publication chain. |

FLIP already packs sparse age and ballistic mode into `before.w`; merge mode
also packs mass there. APIC needs its affine rows. Dropping these fields or
reducing their precision would change behavior. Prefer delayed loads and reuse
of existing streams over another persistent mirror.

## The ten optimization questions

| Question | Current implementation | Action and acceptance condition |
| --- | --- | --- |
| 1. Memory layout | AoS particle records, compact position/state stream, plane-based pressure, tiled deposits and surface scatter. | Keep stage-specific layouts. Measure conversion and sorting costs with any AoS/SoA experiment; scalar-channel coalescing alone is insufficient. |
| 2. Bandwidth | Surface scatter reads 16 B per contributing particle; P2G and motion need broader records. Surface jobs reuse the expired motion bank instead of allocating another buffer. | Keep hot fields narrow and load metadata after liveness/support tests. Attribute device traffic separately from workgroup accesses and atomics. |
| 3. Unnecessary work | Particle and surface kernels stride current GPU job counts through fixed launch grids; the normal particle grid is 4,096 groups, the quiet grid 64; surface planning and scatter each launch 1,024 groups. Some fallback cell/list kernels scan live tiles. | Compare a GPU-built indirect launch with the existing fixed grid, including argument construction and indirect validation. Empty launched groups are overhead, but these kernels do not process every allocated particle slot. |
| 4. Divergence | Slab jobs group particles spatially; surface jobs cap each tile chunk at 512 samples. Motion, collision and exact-distance work still vary. | Retain spatial sorting already paid for by the solver. Measure job-size distributions before adding cost buckets; sorting by material adds no demonstrated benefit to these fluid kernels. |
| 5. Pass fusion | Occupied-tile surface selection feeds bounded scatter jobs; canonical distance also publishes surface data. Pressure reads a stable previous plane across workgroups. | Do not fuse pressure sweeps across a required global dependency. Fuse only values whose consumers stay in the same group or read immutable inputs; count extra halo work and register pressure. |
| 6. Kernel splitting | `advectWalk` consumes a compact list of deferred collision walks, keeping the trace sampler out of that kernel. | Further splitting must offset enqueue traffic, another dispatch, and reloaded particle state. Keep the existing contact changes separate from this experiment. |
| 7. Atomics | Motion reserves output ranges once per neighbour brick per group turn. P2G uses local atomics and flushes reduced node sums. Surface distance also reduces locally, but previously issued an atomic minimum for every in-radius contribution. | Test a monotonic load guard on the two surface windows. No new queue, scan, buffer, or dispatch is required. List aggregation is a separate candidate: changing list order needs set/output validation rather than raw list-byte parity. |
| 8. Neighbourhood reuse | Surface scatter shares two 7³ windows: 2,744 B total. P2G shares two 756-word arrays: 6,048 B before metadata. Pressure still gathers neighbouring iterates. | Benchmark pressure tile staging against its exact cross-tile input plane. A one-cell halo on an 8³ brick is 1,000 samples for 512 interior cells, so low occupancy can erase reuse savings. |
| 9. Occupancy | Surface scatter uses 128 lanes; deposit uses 192; canonical distance uses 256. Each has different live state and reuse. | Compare 64/128/256 surface lanes on identical inputs. Faster local atomics may alter the best shape. Lane count alone is not evidence of higher hardware occupancy; register/spill counters are needed for that claim. |
| 10. CPU–GPU coordination | Receipts map asynchronously in a ring; frame N consumes a fixed-lag plan rather than whichever readback finishes first. Counts and iterative work already live largely on the GPU. | Preserve fixed-lag determinism and failure receipts. Evaluate indirect sizing on the GPU; do not turn diagnostic profiling into an awaited readback on the shipping frame path. |

## Measurements on the M1 Max

The dam-break scene at frame 119 records 17.96 ms across instrumented passes.
The largest stages are pressure (4.129 ms) and particle motion (2.687 ms),
followed by canonical surface distance/publication (0.786 ms). Complete surface
reconstruction is 0.590 ms. These timestamps include profiling overhead and
65.536 µs quantization; they are stage attribution, not clean frame latency.

Changing surface scatter from 128 to 64 lanes preserves every writable output
word but loses on actual simulation states:

| Dam-break frame | 128 lanes ms | 64 lanes ms | Result |
| --- | ---: | ---: | --- |
| 1 | 0.0051 | 0.0051 | No meaningful difference |
| 60 | 0.1659 | 0.1864 | 12.4% slower |
| 120 | 0.1311 | 0.1403 | 7.0% slower |

The replay rotates order across eleven trials, repeats each idempotent kernel
64 times per timed pass, and restores all writable resources before each
candidate. All three states pass strict bitwise comparison. A separate earlier
run also found 64 lanes slower at frame 60 (0.1731 → 0.1966 ms). Keep 128 lanes.

The focused benchmark uses the production planner and scatter on a synthetic
512-tile lattice with controlled populations, mixed hot/cold samples, and a
scalar per-cell oracle. World address lookup is stubbed, so these timings cannot
replace real-scene replay. Stable runs use duplicate baseline pipelines, fifteen
rotating trials, and 64 repetitions of reset + plan + scatter; list construction
is included. The early short exploratory runs had large timing variance and
are superseded by these runs.

| Experiment | Exact result | Measured decision |
| --- | --- | --- |
| Load guard before shared `atomicMin` | Matches scalar distance and hot-distance outputs | Reject: slower across all five populations in the stable run; at 512 particles/tile, 0.1516 → 0.1587 ms. |
| 64/128/256 lanes | All match the scalar oracle | 64 helps sparse synthetic tiles but regresses the real scene; 256 also loses on sparse cases. Keep 128. |
| One global job-range reservation per workgroup | Matches scalar outputs | No useful chain gain: 0.0440 ms unchanged at 64 particles/tile; 0.1505 → 0.1516 ms at 512. Barriers/local reduction offset saved global atomics here. |
| 256/512/1024/2048 particles per job | All match the scalar oracle through 4,097 particles/tile | No consistent replacement for 512. A 1,024 chunk is slower at 1,025 particles/tile (0.2990 → 0.3205 ms), while 2,048 gives only a small case-specific gain. |
| Omit wholly held cold tiles in the planner | Matches scalar outputs | Exploratory timings did not establish a reliable gain; not promoted. |

The atomic guard is logically safe because window keys only decrease between
initialization and flush: a stale larger load can only trigger an extra atomic.
Its failure is a performance result. More checks and divergent branches can cost
more than the atomics they avoid. Likewise, fewer global reservations do not
necessarily pay for their workgroup barriers.

Reproduce the controlled chain sweep:

```sh
bash tools/gpu/fluid-gpu-run.sh timing \
  node --import tsx tools/benchmark-nbflip-splat.ts \
  --variants=aa,splatguard,splat64,splat256,splatguard64,aa \
  --repeats=15 --inner=64
```

Reproduce real-scene parity and collect attribution on a separate frame:

```sh
bash tools/gpu/fluid-gpu-run.sh timing \
  node --import tsx tools/probe-nbflip-variant-replay.ts \
  --scene=sparse-cm12-long-dam-break --frames=1,60,120 \
  --entries=surfaceSplatOccupied --variants=aa,splat64 \
  --repeats=11 --inner=64 --assert-parity --profile-frames=119
```

`--assert-parity` fails on any differing writable word or a requested entry
that was never captured. Shader hashes and pipeline constants identify generated
source. Replay and frame profiling must use different frames because both own
timestamp queries; the harness rejects overlaps before solver construction.

Evidence is in `artifacts/nbflip-opt-2026-11-02/`: `splat-scene-final.jsonl`,
`splat-shapes-confirm.jsonl`, `splat-plan.jsonl`, and `splat-chunks.jsonl`.
The initial full-scene attempt was stopped during prolonged Metal compilation;
a later cached run completed. An intervening run hit overlapping profiler and
replay timestamps at frame 120; `splat-scene-final.jsonl` is the completed run.

## Pressure and particle follow-ups

The pressure tile prototype keeps two 4³ tiles in a 128-word shared iterate
array (512 B). Same-tile black rows read the freshly updated red values there;
other tiles still read the immutable previous plane. It replaces the storage
barrier with a workgroup barrier and preserves every output word in the two
frozen-state replays. Frame 60 worsens from 0.0184 to 0.0205 ms per `pRelax`
dispatch; frame 120 moves from 0.0205 to 0.0195 ms, one effective timestamp
quantum in this 64-dispatch batch. This is not a consistent gain. Keep the
current smoother. Evidence: `pressure-tile.jsonl`.

A separate frame-119 profile with one pass per dispatch attributes 1.769 ms to
`advectParticlesDeferred` and 1.180 ms to `separationPrepass`. Pressure costs are
distributed: `pRelax`, `pDownA`, and `pUp3` each account for approximately
0.459 ms. Splitting passes increases instrumentation overhead, so its 20.45 ms
sum cannot be compared with the earlier unsplit frame sum as a regression.
Evidence: `kernel-profile.jsonl`.

The separation loop visits four neighbouring x rows, each with at most two
contiguous particle ranges. It reads a 16 B position/liveness vector at a 48 B
FLIP record stride (112 B for APIC), loading mass only for a neighbour in range.
A prototype removed the two-element private array and deferred the second
range lookup until after the first. It preserves exact outputs but slows frame
60 from 2.8590 to 2.9942 ms and gives no useful gain at frame 120. Shortening live
state alone is insufficient; the changed scheduling also postpones independent
address loads. Evidence: `separation-runs.jsonl`. A second version kept both lookups early
but replaced the private array with two named ranges and a `select`; it was
indistinguishable from duplicate baselines at both frames and also remains a
probe (`separation-select.jsonl`).

The leading next experiment is to share immutable neighbour-range and position
loads among spatially adjacent separation samples, while preserving per-sample
accumulation order, liveness checks, and the merged-particle mass path. Include
packet construction, tail handling, and the movement fallback in its cost.
Avoid changing the pressure operator or taking extra CPU readbacks merely to
reduce dispatch counts.

## Validation

- `npm run check:types`: passed.
- `npm run test:unit`: 1,447 passed, 264 skipped, zero failures.
- Actual-scene GPU replays: strict equality of every declared writable word for
  surface scatter (frames 1, 60, 120), pressure tile staging (60, 120), and both
  separation variants (60, 120); no uncaptured-entry success or ignored mismatch.
- Synthetic surface sweeps: scalar-oracle parity across sparse and dense runs,
  mixed hot/cold samples, world boundaries, multiple jobs per tile, and repeated
  workgroup-window reuse. Counts reached 4,097 samples per tile.

## Source map

- `lib/methods/uniform/uniform-narrow-band-apic.wgsl.ts`: particle ABI and packed state.
- `lib/methods/uniform/uniform-narrow-band-flip.ts`: compact streams, slab jobs, deferred walks, output reservations and launch grids.
- `lib/methods/uniform/uniform-narrow-band-splat.wgsl.ts`: surface job construction and scatter.
- `lib/methods/uniform/uniform-particle-brick.wgsl.ts`: P2G accumulation and flush.
- `lib/methods/uniform/uniform-pressure-arena.ts`: row loads and red/black plane contract.
- `lib/methods/uniform/uniform-surface-brick-arena.wgsl.ts`: canonical distance and publication.
- `lib/methods/uniform/uniform-mixed-frame.ts`: asynchronous receipts and fixed-lag planning.
