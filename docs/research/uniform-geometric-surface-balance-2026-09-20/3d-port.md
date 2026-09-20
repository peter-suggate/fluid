# 3D Uniform Geometric surface-deficit balancing

Enabled by default in the 3D constructor and shared method parameters. The live
On/Off control is in Advance pipeline → Pressure → System build. The 2D lab
retains its separate experimental checkbox and prior default.

Before pressure setup, sum over pressure-liquid cells with positive capacity:

- Positive expansion volume: `min(0.5 * max(V - capacity, 0), capacity)`.
- Deficit: `max(surfaceTarget - V, 0)` in cells where `V <= capacity`.
- Global rate: `min(sumPositive / sumDeficit, 1)`, or zero without a deficit.
- Pressure source: `(positive - rate * deficit) / dt`.

The positive source is unchanged. Conservative V is not edited. The 3D port
uses the existing gammaA surface target from transport (refreshed after total
surface-volume correction), rather than introducing another contour integration.
This matches the intended inexpensive approximation; the 2D contour integration
and 3D surface target are not numerically identical.

Two GPU passes run over the solve window and reduce partial sums. There is no
CPU readback, connectivity search, new storage binding, or extra cell-sized
field. A tail of the existing scratch buffer stores two floats per 4³ workgroup
plus two header words (32,776 bytes at 64³). Off clears the scalar rate before
the pressure solve. The existing ordinary Uniform solver is unaffected.

Validation:

- Dedicated Dawn source tests: 5 passed, including unchanged positive source,
  net cancellation, surface-weighted/capped contraction, no-source/no-deficit
  cases, and live Off/On without stale correction.
- Uniform geometric numerical/hydrostatic and boundary tests: 18 passed.
- MiniDam64, 60 steps per mode, shared defaults including solve-window scheduling:
  finite output for both. Queue-fenced median wall time after ten warmup steps:
  Off 37.226 ms, On 37.535 ms; p95 46.197 / 46.893 ms. A short smoke comparison,
  not an isolated GPU cost measurement or long-run settling assessment.
- Rebuilt scalar/SIMD Wasm and passed the 2D scene/controller integration suite.
- Browser: MiniDam64 64³ loaded, default On visible, live Off/On tested, one
  successful advance to 0.0333 s; left paused with On selected.
- Typecheck still reports existing unrelated sparse tests/tools errors, none in
  the changed files. Generated Rust parameter contract and diff whitespace pass.

Global balancing can couple disconnected pools, and insufficient deficit leaves
some expansion uncompensated. This port does not claim to eliminate all late
energy gain or resolve the observed slow 2D MiniDam64 settling.

Reproduce: `WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test tests/uniform-surface-deficit-balance-dawn.test.ts`
and `node --import tsx tools/probe-uniform-surface-deficit-dawn.ts`.
Stop browser GPU simulation before running Dawn.
