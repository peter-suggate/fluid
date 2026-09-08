# Immutable native uniform VEX map research receipt

This isolated tool recognizes an exactly uniform f32 velocity on the actual
accepted, full-fine, open, VEX-valid native centers. It produces a GPU affine
departure map and physical-center ownership mask. It does not change production
density, transport, topology, or presentation, and does not yet drive a new field.
Nonuniform velocity, including shear and rotation, is explicitly unsupported.

`GPUUniformVexMapCompiler.encodeSnapshotAndCompile` copies the native topology,
frame control, VEX validity and velocity before gather. A small shader copies
the actual uniform parameter block, whose production buffer has no COPY_SRC
usage. The compiler requires sealed FCA authority, coherent source/candidate
generations, full-fine unique ownership, and bit-identical finite velocities.
Invalid VEX cells remain absent. Map and coverage buffers are the output
authority; their host decoders and copied-header capture are QA only.

The departure convention is `oldPosition = B * newPosition + translation`.
Uniform recognition sets B to identity and applies the native constant-velocity
substep recurrence using GPU dt and h. This is a physical uniform-map
certificate; normalized native interpolation can introduce f32 rounding for
arbitrary uniform vectors, so the API does not promise bit-identical general
native tracer positions. Solid/world clipping and uncovered interpolation
donors cannot be inferred from recognition alone.

Validation on Metal, serialized under the repository WebGPU lease:

- CPU oracle and swept-donor coverage checks: 24 passed.
- Synthetic GPU compiler: 21 subcases passed, including one-ulp nonuniformity,
  stale/unsealed/incomplete authority, NaNs, duplicate owners, partial solids,
  invalid VEX values, and an unsubmitted command buffer. Live native buffers are
  overwritten after capture in the same command buffer to check immutability.
- Actual resident capture: two native steps passed in 5.689 seconds. The fixture
  is 32³, h=0.05 m, dt=1/30 s, sphere R=0.25 m, width=0.05 m, imposed velocity
  [0.75,0,0] m/s. Both maps have B=I and translation
  [-0.02500000037252903,0,0] m. Accepted source frames 1 and 2 have sealed
  candidate generations 2 and 3; topology generations are 1 and 2.
- All trilinear native-center donors over complete wet-intersecting cubes and
  their swept trajectories are present: 2,108 / 2,024 required centers versus
  5,736 / 4,100 valid VEX centers. This capture-only geometric proof uses the
  known analytic fixture; the compiler itself accepts no sphere or ROI input.

Receipts: `/tmp/fluid-uniform-vex-map-gpu-2.log`,
`/tmp/fluid-native-uniform-vex-capture-2.log`, and
`artifacts/retained-imposed-flow/native-uniform-vex-map/` (raw map, coverage,
copied headers, provenance, and per-step JSON). Commands:

```sh
node --import tsx --test tests/uniform-vex-map-oracle.test.ts tests/native-uniform-vex-coverage.test.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal node --import tsx --test --test-concurrency=1 tests/uniform-vex-map-gpu-dawn.test.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal node --import tsx --test --test-concurrency=1 tests/native-uniform-vex-capture-dawn.test.ts
```

The next research step must consume both GPU outputs directly, prove whole
candidate-sweep coverage without an analytic ROI, and retain the current-field
wet-source coverage and integral acceptance transaction. This receipt alone
does not establish that coupling or a production surface authority.
