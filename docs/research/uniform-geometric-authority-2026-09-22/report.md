# Uniform Geometric: reuse pressure inputs and parallelize the tile census

On `cm12-figure-7-256`, the final Dawn/Metal comparison reduces mean GPU frame time from **222.08 to 219.38 ms (1.2%)**. Peak requested GPU resources remain **2,971,243,288 bytes**, identical in both arms. This is a modest gain; the initial estimate of a large static-geometry opportunity was too optimistic.

| Work | Baseline ms | Optimized ms | Reduction |
| --- | ---: | ---: | ---: |
| Whole GPU frame | 222.08 | 219.38 | 1.2% |
| Interface authority, including census/classification | 24.10 | 21.94 | 9.0% |
| Pressure topology and RHS | 21.86 | 20.81 | 4.8% |
| Free fall | 154.71 | 152.26 | 1.6% |
| Impact and spread | 259.51 | 256.67 | 1.1% |

The two targeted phases together fall from 45.96 to 42.74 ms (7.0%). An earlier comparison measured 222.09 to 220.27 ms overall, with about 3.2 ms saved in these phases. Individual stage timings vary; the final repeated pair above is the reported result. Queue-fenced wall time falls from 230.83 to 228.35 ms. Both runs use 60 advances at 1/30 s; the mean excludes frames 1–4. Free fall is frames 5–24, impact/spread is 25–60. Stage timestamps only; no xctrace.

The implementation makes two changes:

- The finest pressure RHS reads interface phi and cell capacity from the topology already built earlier in the same step. It reuses those values in classification, divergence and volume correction. Exterior halo rows keep their original continuation query; topology's exterior halo phi is deliberately different. Since geometry and phi cannot change between these passes, live edits, terrain and moving bodies require no persistent-cache invalidation.
- The tile seed census uses one 4×4×4 workgroup per tile. Adjacent lanes read adjacent cells and collectively read the tile's vertices, replacing a serial scan of 64 cells and up to 125 vertices in each lane. Workgroup OR and nonnegative maximum reductions preserve the exact class bits and displacement bound. Dilation, active sets, numerical algorithms and precision are unchanged. No new buffers or textures are allocated.

The serial census and raw pressure RHS remain available as QA controls (`tileSeedForQA: "serial"`, `pressureAuthorityForQA: "raw"`) and through the A/B switches below.

The original static-cache hypothesis was checked and rejected. Positive-face fractions were already retained between static steps. A cell-capacity cache in the existing geometry texture's unused channel slowed the full run from 222.40 to 233.13 ms; additional texture traffic outweighed the saved geometry queries. A one-word-per-tile static flag cache added 1 MiB without a measurable improvement over pressure reuse alone. Neither persistent cache is included in the final code. Per-pass probing also showed that the large “interface authority” interval includes the support census and tile classification; density authority itself was only about 1 ms in early frames.

Validation:

- [Baseline](baseline-256.json) and [optimized](optimized-256.json) 256³ runs have zero WebGPU validation errors. Every recorded `work` field matches exactly across all 60 frames, including volume, maximum speed, pressure residuals, cycle decisions and work counts. Final volume sum is 267719.4365234375 in both arms.
- The new operator regression compares eight fields on identical physical inputs through fractional terrain, wall insertion/undo, and moving-body insertion/removal. The final 68×64×60 fixture exercises partial atlas pages and shared scratch storage while keeping the two-level census enabled. The test asserts that its telemetry is populated, so an automatic fallback cannot silently bypass coverage. All 96 field comparisons were bit-for-bit identical. A 64³ development run also matched exactly. [Fine/shell tile counts and maximum displacement](census-regression.txt) match exactly. A separate 66×64×63 development run exercised pressure reuse with the census automatically disabled, as required for dimensions that are not multiples of four.
- [Native, tiled and logical pressure-layout regressions](authority-and-layout-regressions.txt) pass for partial pages and the long dam, including liquid insertion.
- [Six phi-band/page-layout unit tests](unit-regressions.txt) pass. `git diff --check` passes. Type checking reports existing unrelated repository errors; none concern the changed files.

Reproduce the final timing pair sequentially with no browser simulation or other Dawn process running:

```bash
FLUID_UNIFORM_AB_OFF=pressureauthority,tileseed node --import tsx tools/profile-uniform-geometric-dawn.ts --frames=60 --max-gpu-bytes=3000000000 --out=/tmp/authority-baseline.json
node --import tsx tools/profile-uniform-geometric-dawn.ts --frames=60 --max-gpu-bytes=3000000000 --out=/tmp/authority-optimized.json

WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" node --import tsx --test --test-concurrency=1 tests/uniform-pressure-authority-dawn.test.ts tests/uniform-pressure-layout-dawn.test.ts
```

For finer pass attribution, `tools/probe-uniform-geometric-transport-cost-dawn.ts` now accepts `--passes=<regex>`; `--passes=.` captures all compute passes. The main frame profile and the allocation ceiling remain the acceptance measurements.
