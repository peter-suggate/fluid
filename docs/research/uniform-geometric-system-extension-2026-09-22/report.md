# Uniform Geometric system-build and extension work, 2026-09-22

Dawn/Metal, `cm12-figure-7-256`, 256³, default method settings. Sixty advances of 1/30 s; means below exclude frames 1–4. Each run held the exclusive WebGPU lease. These are production stage timestamps, not xctrace or timings with per-pass instrumentation.

| GPU stage | Baseline ms | Optimized ms | Saved ms |
| --- | ---: | ---: | ---: |
| Full frame | 218.657 | 210.071 | 8.586 (3.93%) |
| Topology + RHS pyramid | 20.812 | 15.559 | 5.253 (25.24%) |
| Narrow-band FIM front | 12.632 | 10.465 | 2.167 (17.15%) |
| Hierarchy fill + transport shell | 9.249 | 9.084 | 0.165 (1.78%) |

Free fall: 151.388 → 145.434 ms. Impact/spread: 256.029 → 245.980 ms. An earlier run of the accepted implementation measured 210.110 ms overall, including 10.556 ms FIM and 15.575 ms system build.

Peak requested GPU resources: **2,972,293,940 bytes**, below the 3,000,000,000-byte limit. The full-capacity shell index list adds about 1 MiB; the balance reduction adds 2 KiB. This measures application resource requests, not driver residency or process RSS.

## Changes

- Bake the seven-cell liquid mask before pressure face coefficients. Explicitly zero and return for records outside every liquid row's face stencil. Adjacent air owners are retained. This remains a dense launch, with much less arithmetic and topology sampling in empty regions.
- Parallelize the surface-deficit global sum with 1024-record chunks, then reduce those partials. Partial outputs follow the live input records and have separately reserved capacity, including page-domain schedules. The final surplus/deficit formula is unchanged.
- Build an indirect list of existing shell tiles. Seed, FIM updates, resolve, final prolongation, and packing use that list. Two dispatch axes support full tile capacity; zero tiles produce zero finest work. Coarse restriction/prolongation still fills the globally required far-air hierarchy. Page/window schedules retain their existing dispatches.
- Return each converged neighbor's updated distance with its convergence result, avoiding a second Godunov solve of that neighbor. Input parity and the rule that newly activated faces wait until the next sweep are preserved. No additional convergence field or precision reduction.

A cooperative shared-memory cache was tried and rejected: FIM increased from 12.482 to 15.492 ms. Computing the tile halo eagerly cost more than the saved queries. Returning the distance directly reduced FIM to 10.556 ms in the following run; final confirmation was 10.465 ms.

## Numerical scope

The parallel reduction changes FP32 summation grouping. Uninterrupted 60-frame final volume changed from 267719.4365234375 to 267720.01416015625 cell-volumes, a relative difference of 0.000216%. Pressure cycle budget decisions differ on nine frames, so this is an end-to-end comparison of the default simulations, not a claim of identical solver work. Extension list/reuse experiments with the same reduction produced identical final volume and trajectory diagnostics. Both final profiles reported zero WebGPU validation errors.

## Reproduction

```sh
FLUID_UNIFORM_AB_OFF=maskfirst,balancetree,shelllist,frontreuse \
  node --import tsx tools/profile-uniform-geometric-dawn.ts \
  --frames=60 --max-gpu-bytes=3000000000 --out=/tmp/baseline.json
node --import tsx tools/profile-uniform-geometric-dawn.ts \
  --frames=60 --max-gpu-bytes=3000000000 --out=/tmp/optimized.json
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
  node --import tsx --test --test-concurrency=1 \
  tests/uniform-system-extension-dawn.test.ts \
  tests/uniform-nearest-extension-dawn.test.ts \
  tests/uniform-pressure-layout-dawn.test.ts
```

The saved baseline predates renaming the diagnostic switch `frontcache` to `frontreuse`; its captured `abOff` uses the old name. All four optimizations were disabled in that run. Raw profile records are adjacent to this report.

## Validation

All three Dawn regression tests passed, with zero validation errors:

- A 68×64×60 terrain fixture exercises shared scratch, a partially filled page grid, shell dispatch, pressure masking and the parallel balance tree through wall insertion/undo and a moving rigid body. Eight physical fields over twelve steps pass the existing numerical tolerances (96 comparisons), resynchronizing physical input after each step. Maximum relative L2 difference is 7.08e-8; maximum absolute difference is 0.00390625 in pressure.
- Nearest-source extension checks predicted velocity and varying three-component velocities against full-shell compact/unfused packing. The 512-tile full shell exercises both indirect dispatch axes; the empty shell emits no finest work. The falling drop remains isolated from the remote pool.
- Pressure-layout equivalence passes partial-page and long-dam fixtures, including live liquid insertion.

`git diff --check` passes. `npx tsc --noEmit` still reports existing errors in unrelated Sparse tests/probes; it reports none in the Uniform code or tests. The output is retained in `typecheck.txt`. No Sparse simulation code was changed by this work.
