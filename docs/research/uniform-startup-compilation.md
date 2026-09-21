# Uniform page shader startup compilation

2026-09-21: reproduced the water-box-dam-break startup stall in Dawn/Metal.
The solver completed initialization, but individual shader compilations took
seconds to tens of seconds. The first reproduction took 84.97 seconds; its
semi-Lagrangian advection, correction, and projection pipelines took 15.62,
12.45, and 12.66 seconds respectively. Rigid-body integration took 0.30 seconds.
The UI nevertheless continued naming rigid-body integration because the task
runner only reported completions after every task in its compile batch finished.

Changes:

- Specialize known page/dense backing choices when generating field accessors.
  Keep numerical logical dimensions in uniform metadata, as before.
- Supply the existing small quadrature and stencil loop counts through immutable
  layout metadata. This prevents expansion of nested page/geometry sampling
  loops in the Metal compiler. Iteration counts and order remain unchanged.
  Metadata is initialized once; this adds no per-frame CPU decisions or readbacks.
- Report independent pipeline completions immediately and display the first
  unfinished task. Aggregate partial progress monotonically across the batch.

Measured initialization was 68.23 seconds after backing specialization, 47.29
seconds after the first loop changes, and 7.21 seconds on the final follow-up.
These are sequential local observations, **not an isolated cold-cache benchmark**:
Metal caches compiled shaders, and unchanged pipelines benefit on later runs.
The first loop experiment reduced advection compilation to 3.50 seconds and
correction to 2.84 seconds. Frame execution performance was not benchmarked here.

Reproduce with all browser simulations closed (the probe takes the WebGPU lease):

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
  node --import tsx tools/probe-uniform-startup-dawn.ts
```

`SCENE` selects the scene; `REPORT` selects the JSON output (default
`/tmp/uniform-startup-profile.json`). The probe records driver pipeline times
excluding the compilation manager's queue wait, solver initialization, and the
first completed frame. It fails on uncaptured GPU validation errors.
