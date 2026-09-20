# Total surface volume: default in 2D and 3D

The user's preferred **Total area only** 2D control is now available in the 3D Uniform Geometric solver as **Total surface volume**. Following user review, it defaults On in both 2D and 3D; the 3D control changes live. No regional correction or phi seeding is enabled.

Open `http://localhost:3000/scene?scene=sparse-cm12-ladder-long-dam&method=uniform-volume`.

Press **S** to open Simulation pipeline, then **Level set + volume → Vertex level set → Total surface volume → On / Off**. Reset between runs for an initial-condition comparison; the switch itself does not reset time. The existing “Seed from V”, “Follow V”, and “Compaction” controls remain off.

## Implementation

- Runs after conservative V gather, before sharpening and pressure. Refreshes the cached phi targets after changing phi.
- Builds a geometric band from cells crossed by the current zero contour, then dilates four vertex edges with a taper. Computes local physical gradient magnitude to convert normal displacement to phi change, including stretched phi.
- Estimates surface volume by six linear tetrahedra per cell, weighted by open capacity. This is a continuous volume estimator, rather than counting negative cell centres. It approximates the trilinear contour; it is not an exact rendered-mesh volume integral.
- Computes two successively refined 17-sample volume-versus-displacement curves, reduces them deterministically on the GPU, and interpolates a global scalar displacement. The displacement is bounded to one minimum cell width per step.
- Applies that scalar through the tapered metric band. It cannot seed remote liquid. V is never written by this operation. The original advected-phi scratch remains available to diagnostics.
- Uses the full domain for totals, irrespective of solve-window size. Sources skip the correction for that step. The scalar constraint is global across disconnected liquid bodies.
- All reduction and shift selection stays on the GPU. No per-step CPU readback is introduced. Scratch buffers and the output texture allocate on first use; Off encodes no correction work. Shared parameter defaults are On. The 2D lab defaults to Total area and preserves explicit Off in its URL.

## Long-dam check

Actual registered scene, 192 × 96 × 32, default Uniform Geometric values, 120 steps at 1/30 s. Raw conservative volume is summed from the texture; represented surface volume is the solver's existing, independent diagnostic. See `baseline.json` and `total-volume.json`.

| At four seconds | Off | On |
| --- | ---: | ---: |
| Initial V | 40,960 cells | 40,960 cells |
| Raw final V | 40,959.963 | 40,960.019 |
| Represented surface volume | 28,051.09 | 40,843.82 |
| Represented / initial volume | 68.48% | 99.72% |
| Negative cell centres (secondary check) | 27,951 | 40,939 |

The end-to-end median step observations were 66.76 ms Off and 61.30 ms On. These are **not** a correction-cost benchmark: the altered surface changes pressure and active-work scheduling, and the runs were sequential. These measurements preceded isolating the correction output from the advected-phi diagnostic scratch; that final change adds one lazily allocated vertex texture without changing the numerical operations.

## Validation

- Dedicated GPU tests: fractional planes in X/Y/Z with anisotropic cells and stretched phi; repeat application to a matching surface; unchanged V and far air; curved surfaces with and without solid capacity, checked against independent trilinear quadrature; empty-surface no-seeding behavior.
- Existing full Uniform Geometric GPU tests pass, plus a new live-toggle integration case recovering a shrunken planar pool while keeping V unchanged.
- Scalar/SIMD WASM builds and 2D worker acceptance pass after regenerating the shared parameter contract. The shared option also maps to the existing area-only profile in the native 2D backend.
- GPU compilation-policy checks pass. Full TypeScript checking reports only the pre-existing unrelated sparse test/tool errors; no changed-file errors.
- Browser: 3D long dam ran with the control On; On/Off controls and URL persistence verified. The user approved promotion to the default after reviewing both dimensions.

Reproduce:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
  node --import tsx --test --test-concurrency=1 \
  tests/uniform-surface-volume-dawn.test.ts tests/uniform-volume-dawn.test.ts

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
  node --import tsx tools/probe-uniform-volume-dawn.ts \
  --scene=sparse-cm12-ladder-long-dam --frames=120 --total-surface-volume=on
```

Use `--total-surface-volume=off` for the baseline. Run Dawn only while the browser simulation is released. The Sparse CM12 regression gate is not applicable to this Uniform-only numerical change; no sparse implementation or shared presentation code was changed.
