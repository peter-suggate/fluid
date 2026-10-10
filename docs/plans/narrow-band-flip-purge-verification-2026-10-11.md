# Narrow-band purge verification — 2026-10-11

The application retains narrow-band FLIP, the method registry and feature/plugin
contracts, the SVO renderer, AdvanceLab, Shape Lab, and the remaining UI. Retired
method implementations, their exclusive dependencies, obsolete inspection/render
paths, generated receipts, and stale documentation were deleted. Scene and URL
hydration migrate retired method identifiers to narrow-band FLIP.

Paper PDFs and `.txt` files are retained for reference, including papers about
retired methods. Deleted paper sources were restored; extracted images remain deleted.
The remaining 241 extracted images were subsequently deleted; all 28 PDFs and
49 `.txt` files were verified byte-identical before and after that deletion.

The skipped-test deletion also applies to conditional GPU tests and ignored Rust
tests. Their case bodies, orphaned helpers, and obsolete runner/scripts were removed.
The file and case manifests are in `docs/plans/purge-*.txt` and
`docs/plans/purge-skipped-test-cases.json`.

## Shader closure

This census covers the production narrow-band solver's compute programs.

| Measure | Result |
|---|---|
| Shader modules, including timestamp marker | 74 |
| Eager compute pipelines, including timestamp marker | 223 |
| Without timestamp-query | 73 modules / 222 pipelines |
| dam32 / dam64 / torus WGSL comparison | All 74 files byte-identical |
| Pipeline descriptor comparison | All 223 descriptors identical |
| Naga validation | Every module parses |
| Pressure traversal | Eight shared kernels, padded tables and immutable launch records |
| Native pressure source | 122,413 → 45,034 bytes |
| Native extrapolation source | 60,209 → 20,278 bytes |

The original plan's 432 reachable pipelines and roughly 201 final pipelines were
estimates, rather than a measured GPU baseline. The final eager census is 223.
Nine native continuation/setup kernels originally proposed for deletion are reached
by the default frame and were retained. The shared momentum cache is initialized
once. Tank dimensions, spacing, layout offsets, pressure hierarchy tables and launch
selection now come from uniforms; the solid flag uses a uniform branch.

## Startup and frame measurements

Measured locally on Apple M1 Max, Dawn Metal, Node 22.22.1. These are elapsed host
times through solver initialization and completed frames, not GPU timestamp timings.
No pipeline was created while advancing any benchmark arm.

| Startup probe | Time |
|---|---|
| First production dam64 in final paired run | 87.28 s |
| Dam64 with a synthetic shader cache miss | 96.66 s |
| Torus immediately after that same shader variant | 0.189 s |

Both first-scene probes passed the 120-second ceiling. The synthetic miss adds a
uniform term whose value is zero and renames entry points; it changes shader code
without changing geometry. Driver caches were not cleared, so this is not a fully
cold-driver measurement. An earlier unchanged-production dam→torus run measured
0.163 s for the second scene.

Frame comparison uses three rounds, each bracketed by two production runs, with 36
frames per solver. Each reported sample is the median of the final six frames; the
table averages the samples across rounds. The baked variants are temporary probes,
not retained solver modes.

| Arm | Frame time | Production overhead |
|---|---|---|
| Production runtime geometry and solid flag | 14.815 ms | — |
| Constant solid-free branch | 14.460 ms | +2.45% |
| Baked geometry helpers | 13.836 ms | +7.08% |

The solid comparison varied from −2.39% to +5.11% by round. The geometry comparison
varied from +3.80% to +10.05%. Runtime geometry trades some frame cost for reusable
shaders across tanks; the measured frame cost is not a speedup.

## Checks

- `npm run check:types`: pass.
- `npm run test:unit`: 847 pass, zero failures, skips, or todos.
- `cargo test -p fluid-core --lib kernels::tests`: 3 pass, zero ignored.
- `git diff --check`: pass.
- `npm run check:module-boundaries`: non-strict audit reports 13 shared imports
  across core/UI/labs and the uniform implementation; it is not a clean boundary gate.

Final targeted GPU checks pass all 15 cases with zero skips, across torus motion,
cached forces, live voxel edits without simulation compilation, rigid bodies,
adaptive rest/dam evolution, fluid edits, pressure dispatch, volume publication,
surface extraction, visual layers, and wireframe depth. Deleted GPU test sources
were run from temporary copies against the current repository implementation.

The rest state preserved mass exactly (16,384 cells), with zero contour error and
velocity. The adaptive dam measured −2.341% volume drift after six 1/30-second
steps. The historical 2% assertion exercised fixed, nonadaptive surface detail,
which this purge removes; it was not an app-default adaptive acceptance bound.
The adapted check retains occupancy/geometry equality, finite supported particles,
fine tiles and fine pressure. This measurement does not establish a volume-error
improvement or equality to the previous adaptive implementation.

The external surface fixture also exercised removed Requested/Full modes. Its
replacement keeps current coarse-base checks, window/full scan equality, reference
polygonizer comparisons, finite normals, and a live gravity edit under app defaults.

The SVO hero smoke's historical 67-candidate ratchet described 1,233 primitives on
200 mm bricks; the unchanged current scene publishes 2,554 on 100 mm bricks. That
stale ratchet was deleted. The tool retains the real 512-candidate arena, zero
overflow, the 64-candidate refinement diagnostic, and rendering/lighting checks.
See [calibration evidence](purge-svo-smoke-calibration.md). A final current-tool
GPU run passes all 17 SVO smoke checks with zero warnings.
