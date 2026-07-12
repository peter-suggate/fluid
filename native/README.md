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
scene); the 2-million-cell `M1 Max` tier remains available. Drag to orbit,
Option-drag the selected rigid body, scroll to zoom, and use File > Open Scene
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

The Metal backend uses linear private buffers for the solver, cached limiter
and curvature auxiliaries, three in-flight frame contexts, compact asynchronous
diagnostics, and shared body impulse buffers. A frame uses one command buffer
and no full simulation-state readback. The drawable is written by compute, so
there is no intermediate render target or fullscreen copy. MSL fast math is
enabled for the interactive `f32` path.

Physics includes midpoint RK2 transport, conservative donor/receiver VOF
limiting, molecular and Smagorinsky viscosity, balanced-force surface tension,
ghost-fluid pressure coefficients, weighted Jacobi projection, CFL/capillary
step limiting, and immersed moving solids with paired linear/angular impulses.
Rigid bodies use native quaternion integration and container contacts.

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
compiles all MSL, creates every pipeline, runs three headless Eulerian steps
including rigid coupling and diagnostics, and gates finite state plus volume
drift below one percent. The interactive launch additionally validates drawable
presentation.

The product is intentionally Eulerian-only. The CPU Eulerian oracle remains in
the browser test suite; no particle fluid code is compiled or exposed by either
runtime.
