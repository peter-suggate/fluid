# Uniform Geometric sharpening performance

Target: every non-pressure UI stage below 5 ms. This change is progress toward
that target, not completion. Production CM12 Figure 9, 128×128×64, Dawn/Metal,
frames 121–180 after 120 warm-up advances, without rendering or a browser
simulation. Pressure settings and all eight sharpening sweeps are unchanged.

## Root causes and changes

The shared transport/sharpening arena used 40-byte cell records. Neighboring
SIMD lanes loading a proposal or limiter therefore touched strided words.
Store the ten members in contiguous planes within each existing 64-cell brick.
Allocation size, field values, and arithmetic stay unchanged. The original
layout remains selected when brick padding is disabled.

Skip proposals when neither endpoint can receive volume, and skip limiter
loads for zero raw flux. Reuse cached positive-face eligibility during commit;
negative owners still require an activity check before reading their scratch.
Inactive tiles can retain transport data and must never supply proposals.

## Measurements

Matched per-pass profiling before/after the planar layout and initial zero-flux
shortcut, summed over eight sweeps:

| Kernel | Control | Optimized |
|---|---:|---:|
| Propose | 3.47 ms | 1.34 ms |
| Limit | 2.10 ms | 1.25 ms |
| Commit | 3.54 ms | 2.10 ms |

Final normal stage timestamps, with the additional commit eligibility shortcut:

| Stage | Median |
|---|---:|
| Vertex level set | 8.32 ms |
| Conservative transport | 4.98 ms |
| Gather and surface-volume correction | 4.98 ms |
| Sharpening | 5.83 ms |
| Velocity advection | 3.51 ms |
| Velocity extension components | 0.66 + 4.62 + 2.42 ms |
| Pressure components | 2.23 + 5.90 + 0.20 + 2.23 ms |

These are sampled timings, not guaranteed ceilings. Subsequent control timings
varied substantially with machine load. Per-pass timestamps add overhead and
must not replace normal stage timestamps on the same pass. The benchmark uses
separate modes. Do not compare instrumented per-pass stage totals directly to
ordinary production stage totals.

Control and final optimized runs produced identical SHA-256 hashes of the
complete published fields after 180 frames:

- Volume: `c5f976b542173397a3ecd823c05572bf9f8fd92063dc04ad14eb0d0b1fdf0b54`
- Vertex phi: `67459637885df32c544e763017f66518fa34d7922c48fd62b2783269cd0dcccc`
- Velocity: `4b9731a06c2468a9a36ff79856d5ba5d4bf8b0a6a8061913c785b8fa2bf557d8`

## Reproduce

Run serially with no browser simulation or other Dawn process. The probe takes
the repository GPU lease. Repeat in control/optimized/optimized/control order
for stronger timing confidence.

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_UNIFORM_AB_OFF=sharpenflux,edgeplanes UNIFORM_BENCH_OUTPUT=/tmp/sharpen-control.json node --import tsx tools/benchmark-uniform-sharpening-dawn.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js UNIFORM_BENCH_OUTPUT=/tmp/sharpen-optimized.json node --import tsx tools/benchmark-uniform-sharpening-dawn.ts
```

`UNIFORM_BENCH_PASSES=on` selects individual volume-kernel timings instead of
normal stage timings. Override scene, frames, and warm-up with
`UNIFORM_BENCH_SCENE`, `UNIFORM_BENCH_FRAMES`, and `UNIFORM_BENCH_WARMUP`.

## Next steps

1. Vertex redistancing: isolated cost was about 5.44 ms plus 2.82 ms advection.
   Investigate shared neighborhood loads while preserving arithmetic and contours.
2. Aggregate velocity extension: about 7.7 ms, mainly its narrow-band front and
   hierarchy/shell work.
3. Finish sharpening below 5 ms by profiling limiter/commit traffic and dispatch
   overhead, retaining all eight sweeps and conservation checks.
4. Give transport/gather margin below 5 ms, then validate later trajectories,
   representative scenes, and browser timings without weakening quality or gates.

## Validation

- `uniform-scratch-storage-dawn.test.ts`: passed shared-versus-separate storage
  parity through the 40-frame ball-drop trajectory.
- `uniform-volume-tile-work-dawn.test.ts`: passed dense/tiled parity, stale
  scratch, tile activation, and live toggling.
- Figure 9 control/optimized final-field hashes match after 180 frames.
- `npm run check:types` reports existing errors in unrelated sparse-CM12 tests
  and tools; none are in the changed files.

This change touches Uniform Geometric only; the Sparse CM12 regression gate
was not required for this scope.
