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

The app defaults to the `M1 Max` quality preset (about 2 million cells on the
standard scene). Drag to orbit, scroll to zoom, use Pause/Run to control time,
and use File > Open Scene to load any version 1.0.0 browser scene JSON.

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

The Metal backend uses linear private buffers rather than 3D textures for the
solver. This gives predictable coalesced addressing for the seven-point
pressure stencil on Apple GPUs and avoids texture format restrictions. A frame
uses one command buffer and no simulation-state CPU readbacks. The drawable is
written by compute, so there is no intermediate render target or fullscreen
copy. MSL fast math is enabled for the interactive `f32` path.

Quality presets are 110k, 500k, 1.2m, and 2m target cells. The M1 Max preset
uses 96 weighted-Jacobi pressure passes and 128-thread 3D threadgroups. The
sidebar reports command-buffer GPU time rather than CPU submission time.

## Verification

```bash
npm run native:test
npm run native:smoke
```

`native:test` checks the shared scene/schema and M1 Max preset contract without
requiring an external test framework. `native:smoke` creates the Metal device, compiles all MSL at runtime, creates
every compute pipeline, allocates private solver state, and uploads the shared
scene. The interactive launch additionally validates drawable presentation.

The product is intentionally Eulerian-only. The CPU Eulerian oracle remains in
the browser test suite; no particle fluid code is compiled or exposed by either
runtime.
