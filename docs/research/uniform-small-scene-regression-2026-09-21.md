# Water-box paging regression: stage-by-stage investigation

The regression is real. The scene is 24×16×16 (6,144 cells), with one 32³
accepted domain page. Queue-fenced simulation frames increased from about
17–18 ms at `5d4d31f2` (Before paging attempt) to 37–38 ms in the current tree.
These are isolated Dawn/Metal simulation measurements, not browser FPS.

Each arm advances the authored scene for 24 frames; the first four are excluded.
The browser simulation was unloaded throughout. Hardware stage timestamps
include dispatch/driver gaps; they are quantized, so small differences near
0.066 ms should not be overinterpreted. CPU diagnostics readback occurs after
the timed frame. All runs completed without uncaptured GPU errors.

| Stage | Before paging, ms | Current, ms |
|---|---:|---:|
| Interface authority | 0.33 | 0.33 |
| Extension front | 0.36 | 0.79 |
| Hierarchy fill + transport pack | 0.43 | 0.59 |
| Phi transport + redistance | 0.26 | 0.59 |
| Volume transport setup | 0.39 | 0.52 |
| Volume gather | 0.39 | 0.52 |
| Sharpening | 0.72 | 1.11 |
| Velocity advection + forces | 0.39 | 1.31 |
| Pressure topology + RHS | 0.52 | 0.79 |
| Pressure Full-Cycles | 3.70 | 9.83 |
| Pressure V-Cycles | omitted by feedback | 2.36 |
| Pressure finish | 1.18 | 1.38 |
| Projection | 0.33 | 0.46 |
| Phi publication | 0.07 | 0.07 |
| Diagnostic reduction | 0.07 | 0.07 |

These are per-stage medians, not an additive wall-time decomposition.

## Controlled ablations

- **Pressure direct dispatch:** current page fields and full schedule retained;
  queue-fenced time falls from about 37.5 to 32.1 ms. Indirect launch cost is real.
- **One-cycle cap:** about 22.8 ms. This is a diagnostic lower-budget variant,
  not a safe production fix. The old lagged controller typically allowed demand
  plus one cycle of headroom; it did not hard-cap every future frame at one.
- **Disable root read audit:** advection 1.31→0.85 ms, phi 0.59→0.39 ms.
  The membership test at every texture read is costly even when no fault occurs.
- **Dense physical field backing, same generated accessors:** extension front
  0.79→0.46 ms, hierarchy fill 0.59→0.46 ms. Advection remains 1.31 ms.
  This changes fluid/extension backing, not the independently paged pressure hierarchy.
- **Native dense field access, bypass the entire field shader adapter:** retain
  the current page-domain scheduler and paged pressure solver. Advection is
  0.29 ms, phi 0.20 ms, extension front 0.33 ms and hierarchy fill 0.33 ms.
  The generic adapter is a major non-pressure regression. This bundled oracle
  removes its addressing, bounds/membership wrappers, dimension metadata
  substitution, and loop rewriting; it does not isolate one machine instruction.
- **Direct root domain launches:** sharpening 1.11→0.85 ms. Combined with no
  read audit: sharpening 0.79 ms and projection 0.33 ms, near their old costs.
- **Literal loop bounds:** total time essentially unchanged; advection is
  slightly worse (1.57 ms). The startup-loop fix is not the broad regression.
- **Tight domain/pressure workgroup extents:** essentially no wall-time benefit.
  Padding exists, but clipping rejected lanes does not remove the launch floor.
- **Historical solver with liquid/pressure windows disabled:** advection remains
  about 0.33 ms. Restoring the liquid window is not necessary to recover that cost.

## Where the architecture fell short

The domain page is 32³, but physical texture pages are **16³**. This scene's
cell field occupies two physical pages; its 25×17×17 vertex field and 26×18×18
extension field each occupy eight. The displayed one-page count therefore does
not mean native local texture access. Every sampler is routed through a generic
field adapter, and root reads additionally query accepted membership.

The page cutover also unconditionally disables the previous lagged cycle
budget and encodes all 3 Full-Cycles + 4 V-Cycles. The sample converges after
one Full-Cycle, but 1,159 pressure passes remain encoded. Zero-work indirect
launches still have overhead. Shader-side early exits also leave launch overhead.

Finally, converting otherwise small fixed passes to indirect dispatches added
cost throughout transport and sharpening without reducing useful work here.

## Required direction

1. Recover demand-based omission of whole pressure-cycle sequences. The old
   mechanism used asynchronous CPU feedback from prior completed solves. Restoring
   it literally must be reconciled with the GPU-only frame requirement. A GPU-only
   replacement needs coarser/persistent solver kernels; per-pass indirect gates
   are demonstrably not equivalent in cost.
2. Make within-page sampling cheap, with compatible domain/field layouts and
   specialized accessors. Restrict membership validation to appropriate dependency
   boundaries/debug audits rather than taxing every production interpolation tap.
3. Reduce indirect launch granularity for small work, while retaining GPU page
   membership as domain authority. Do not restore the liquid-window architecture.
4. Gate subsequent changes against this scene and MiniDam64, including numerical
   correctness. No ablation here is promoted as a validated production replacement.

The current benchmark is `tools/probe-uniform-small-frame-cost-dawn.ts` (QA-only
hooks and variants). Raw samples, source hashes, and scope notes are in the
adjacent JSON report. The historical driver used the same loop, omitting the
then-unavailable `awaitFrameCompletion()` and retaining the queue fence.
