# Fluid Lab Metal

This is the native, Eulerian-only macOS backend for Fluid Lab. It is an AppKit
and MetalKit application with runtime-compiled Metal Shading Language kernels;
it does not embed a browser or translate WebGPU calls.

## Run

From the repository root:

```bash
npm run native:run
```

Build an ad-hoc signed application bundle:

```bash
npm run native:app
open "native/.build/Fluid Lab Metal.app"
```

The app defaults to the `Ultra` preset (about 1.2 million cells on the standard
scene); the 2-million-cell `M1 Max` tier remains available. Drag empty space to
orbit, directly drag a rigid body to select and move it, scroll to zoom, and use File > Open Scene
to load any version 1.0.0 browser scene JSON.

The scrollable inspector exposes container dimensions, fill, gravity,
viscosity, surface tension, initial and wall conditions, all four rigid body
shapes, density/size editing, quality, scientific overlays, glass visibility,
and camera presets. Native menus provide scene save and run export.

## Architecture

The browser and native applications read the exact same
`Resources/default-scene.json`. The scene schema, SI units, dam-break geometry,
VOF threshold, pressure sign convention, boundary rules, and validation
contract are shared semantically. Platform code is deliberately separate:

```text
shared JSON scene + physics contract
          |                     |
     TypeScript/WGSL       Swift/MSL
          |                     |
        WebGPU             Metal private buffers
                                |
                  compute raymarch -> CAMetalDrawable
```

The Metal backend uses linear private buffers for the solver, cached limiter,
curvature, and analytic cut-cell auxiliaries, serialized GPU simulation frames,
compact asynchronous diagnostics, and shared body impulse buffers. A frame uses one command buffer
and no full simulation-state readback. The drawable is written by compute, so
there is no intermediate render target or fullscreen copy. MSL fast math is
enabled for the interactive `f32` path.

Physics includes midpoint RK2 transport, conservative donor/receiver VOF
limiting, molecular and Smagorinsky viscosity, balanced-force surface tension,
ghost-fluid pressure coefficients, weighted cut-cell Jacobi projection,
CFL/capillary step limiting, exact plane/cube and plane/face fractions for
moving sphere/box/capsule/cylinder SDFs, aperture-aware VOF transport, and a
bounded conservative cover/uncover remap. Pressure traction and torque use the
same embedded geometry as projection and feed an entirely GPU-resident rigid
integrator. Grabbed bodies become kinematic moving boundaries during mouse
gestures, retaining swept velocity so they displace liquid rather than passing
through it silently. Rigid bodies use quaternion integration and container contacts.

Presentation includes the glass tank, floor grid, front/back water thickness,
analytic normals, Fresnel reflection, Beer–Lambert absorption, scattering,
analytic sphere/box/capsule/cylinder intersections, submerged shading, selection
highlighting, scientific grid mode, tone mapping, and quality-aware Retina
render scaling.

Quality presets are 110k, 500k, 1.2m, and 2m target cells. The M1 Max preset
uses 96 weighted-Jacobi pressure passes and 128-thread 3D threadgroups. Ultra
and M1 Max use reduced Retina presentation resolution while retaining the full
physics grid. The sidebar reports command-buffer GPU time, simulation rate,
volume drift, and maximum speed.

## Verification

```bash
npm run native:test
npm run native:smoke
```

`native:test` checks the shared scene/schema and M1 Max preset contract without
requiring an external test framework. `native:smoke` creates the Metal device,
compiles every runtime MSL pipeline, advances submerged light and dense bodies,
and rejects non-finite state, excessive speed, volume drift, or inverted
buoyancy response.
The interactive launch additionally validates drawable presentation.

The product is intentionally Eulerian-only. The CPU Eulerian oracle remains in
the browser test suite; no particle fluid code is compiled or exposed by either
runtime.
