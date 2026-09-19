# Uniform Geometric bounded donor accumulation

2026-09-20, Dawn/Metal. Production implementation following the
[figure 9 investigation](uniform-geometric-figure9-transport-cost.md).

## Result and limitation

The floating atomic CAS retry loop is replaced in the production Uniform
Geometric solver, including optional liquid capacity balancing. There is one
accumulation scheme, with no contention-triggered fallback or new UI control.
This bounds contribution work; it does **not** fix the pressure instability or
the subsequent disappearance of the level-set surface.

The original figure 9 baseline already loses its represented surface at frame
99 (3.3 s), retaining 335900.50 cell-volumes against 335902 initially. The new
implementation loses its surface at frame 100, retaining 335900.51. Recorded
raw displacement jumps to millions of cells in the baseline and trillions in
this new trajectory. The surface's represented volume becomes zero while the
conservative volume persists. This is not evidence of a usable post-instability
simulation, nor of a successful pressure correction. Timing the empty-surface
frames as ordinary performance would be misleading.

## Implementation

Each nonnegative f32 contribution is encoded exactly as an integer multiple of
2^-149. Six 32-bit limbs accumulate with native integer atomics. Carry
propagation is bounded by the six limbs; there are no failed-CAS retries and no
contributions dropped or quantized away. A separate decode rounds the exact
sum to f32, ties to even. This changes rounding from the old order-dependent
floating scatter and can change later nonlinear trajectories.

Transport weights are bounded by approximately one, and at most nine per cell
contribute. A 192-bit accumulator has ample headroom for every supported edge
buffer size. Additional allocation is 24 bytes/cell: 24 MiB for figure 9's
128 x 128 x 64 grid, 48 MiB at 128 cubed. Allocation reporting includes this.
The existing 80-byte/cell edge-buffer limit check also bounds this smaller
buffer. Donor passes reuse the rigid-exchange binding slot with dedicated
scratch; they do not execute rigid coupling.

Performance recovery comes from two changes:

- Store limbs in six contiguous planes, rather than six interleaved words per
  donor. This substantially reduces ordinary-case memory traffic cost.
- Accumulate while building edges and normalizing rows, removing four separate
  full edge-table reads per step. Decode writes only the donor-sum scratch
  plane, preserving extension data and the liquid-balancing header.

Reverse linked donor lists were also prototyped. They improved ordinary-case
transport time, but a concentrated donor left a long serial pointer walk (about
23.5 ms per sum on the captured pathological table). That prototype is not in
production.

## Measurements

Sequential runs on the same machine, default scene controls, dt=1/30 s.
These are single-run medians, not statistically established universal costs.

| Scene / frames | Original frame time | New fused frame time | Difference |
|---|---:|---:|---:|
| Figure 9, 5–60 | 61.57 ms | 63.01 ms | +2.3% |
| Figure 9, 60–90 | 76.48 ms | 77.23 ms | +1.0% |
| Figure 12, 5–60 | 108.51 ms | 112.38 ms | +3.6% |

Before fusing edge construction and row normalization, the integer version's
figure 9 medians were 64.09 and 78.05 ms respectively. Fusion recovered about
0.8–1.1 ms per frame. The initially discussed approximately 15% overhead was
an isolated donor-sum microbenchmark, not total frame time.

On the frozen figure 9 frame-99 edge table, the production accumulator plus
decode takes 1.245 ms median over five runs and exactly matches the CPU integer
oracle for all 1,048,576 donor outputs. The earlier CAS microbenchmark on that
table took roughly 43–48 ms. The original complete-scene stall was much more
variable, reaching 4.285 seconds across four donor sums in one run. Frozen
replay isolates accumulation from the pressure-driven trajectory changes.

## Validation

`tools/bench-uniform-donor-sum-dawn.ts` executes the production WGSL helper on
Dawn and compares every output bit with an independent BigInt sum and rounding
oracle. Ordinary fan-in, 589,824 contributions into one donor, subnormals,
cross-limb carries, and rounding ties all pass five repeats without mismatch.
The captured figure 9 frame-99 replay also passes five repeats.

The volume and boundary Dawn suite passes 18 checks. An additional overlay
Dawn check fails at frame zero with "vertex capacity missing" both before and
after this change; this was reproduced by restoring the pre-accumulator
production files. No threshold was relaxed or failing Dawn check deleted.
Repository typecheck still reports unrelated existing errors; none refer to
the changed files after fixing adapter narrowing in the timing probe.

Reproduce (run sequentially, with no browser GPU run):

```sh
node --import tsx tools/bench-uniform-donor-sum-dawn.ts
node --import tsx tools/probe-uniform-geometric-transport-cost-dawn.ts --capture=99 --dump-edges=on --out=/tmp/donor-capture
node --import tsx tools/bench-uniform-donor-sum-dawn.ts --edges=/tmp/donor-capture/99-edges.bin
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test --test-concurrency=1 tests/uniform-volume-dawn.test.ts tests/uniform-geometric-boundary-dawn.test.ts
```

The JSON receipt alongside this report preserves timing traces, accumulator
checks and the surface-collapse evidence. The precise failing pressure
hierarchy correction remains unresolved; bounded accumulation must not be
mistaken for correcting it.
