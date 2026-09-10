# Sparse CM12 face and sharpening cost investigation

## Reproduction

Use the native Dawn/Metal production-default probe, at the paper timestep
(1/30 s), with three warm-up frames and twelve measured frames:

```sh
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
node --import tsx tools/probe-sparse-cm12-stage-cost.ts \
  --scene=mini32 --production-defaults=1 --warmup=3 --frames=12 \
  --final-qa=0 --final-scalar-hash=1 --out=/tmp/cm12-stage-cost.json
```

Keep GPU runs exclusive. Compare adjacent controls, because unrelated GPU load
can change the absolute frame time substantially. Whole-frame and individual
stage medians are separate measurements; do not add medians to infer FPS.

## Cause

The rollback base `e894aa04` already contained September 8 changes to face
interpolation. The September 6 path used a simple eight-corner interpolant;
the retained path locates a geometric adaptive dual cell and can solve its
inverse using Newton iterations. Regular leaf interiors went through that
same general call graph.

The face dispatch already scanned all accepted interiors on September 6.
Its older dirty-row mask was removed in August (`37b0423e`), so restoring a
recently deleted dirty-face mask is not an available fix.

Native staggered sampling also scanned cell incidence to rediscover an address
already provided by IBO/ITR. The interior entry point serialized three axes
inside each invocation, though their destination rows are independent.

## Landed changes

- Reuse accepted face addresses, retaining incidence lookup for unsupported
  geometry and the original wetness/width checks.
- Dispatch the three interior axes independently. An adjacent axis-off/on
  comparison measured **11.141 / 7.406 ms** for interiors, **2.097 / 2.163 ms**
  for seams, and **47.841 / 43.450 ms** for the complete advance.
- Avoid sharpening gradients after its density or distance stop, avoid
  zero fixed-point transfers, reuse the source's cardinal density, and
  validate each dose's incidence range once.
- At regular interior source centers, form the first sharpening gradient
  from the six normal neighbors, with the same two half-weight samples per
  axis and normalized solid handling. The paired trace timing was
  **3.867 / 3.408 ms**; full mini32 density and gamma hashes stayed identical.
- Sample only the required component at terminal face departures; retain the
  full velocity vector throughout RK2 characteristic tracing.
- Expose individual face interior, seam, sparse-air, dynamic-row, sharpening
  dose, and sharpening trace timings in the existing stage profiler.

Those changes retained bit-identical final mini32 density and gamma fields:

```text
density a907505c3cbc28e6b55a2fe356644c3fd3fa512ae15e645f4291432b34d62dad
gamma   5915fc810d7cd4cb1929f3da441b55ab4da6e360a44cad56fd1ea4fad9eed3c8
```

The native face tests cover zero-time staggered identity at widths 1/2/4/8,
accepted-address equivalence, dry velocity extension, mixed-width support,
and transverse advection. The sharpening seam test covers affine directional
derivatives and reflections at 2:1 boundaries. A separate native oracle compares
the regular interior gradient against ordinary eight-corner interpolation in
108 cases spanning widths 1/2/4/8, solid masks, and boundary/clipped fallbacks.

## Acceptance

The measured 43.4 ms combined result still exceeds the unchanged 40 ms mini32
regression ceiling. It is an intermediate result, not restored baseline
performance. Further regular-kernel specialization is being measured.

The intermediate full canonical run exhausted its unchanged 180-second budget:
4 lanes passed, 7 timed out, and 6 were not reached. No timing ceiling or lane
was weakened. The build and focused native face/sharpening tests passed.
