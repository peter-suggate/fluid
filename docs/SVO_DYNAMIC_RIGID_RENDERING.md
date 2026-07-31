# Dynamic rigid rendering inside the SVO path

## Retained architecture

Rigid bodies remain part of the SVO presentation pipeline rather than becoming
an unlit overlay. The renderer precompiles both exact primary-discovery paths:

- an analytic per-ray body loop for small body counts;
- instanced conservative proxy rasterization for larger body counts.

The live `BodyGPU` allocation is bound directly as read-only storage by the
small raster pipeline and remains a uniform in the wide SVO shader. Moving a
body therefore changes GPU buffer contents only. It does not rebuild the SVO,
recreate a pipeline, or compile WGSL.

The measured strategy crossover is four bodies. Switching across it selects an
already-created pipeline and invalidates exact primary reuse; it does not
compile. One-body scenes such as hose-tank keep the analytic path, while the
six-body dam scene uses raster discovery.

The raster arm accelerates primary visibility only. Lighting remains integrated
with the SVO renderer: direct shadows and AO test the live analytic bodies, and
environmental GI clips each tetrahedral radiance cone at the nearest
current-frame rigid hit. A clipped cone retains radiance accumulated in front
of the body but contributes zero residual visibility behind it. Rigid receivers
ignore their own owner ID; static receivers test every live body.

Capsule and cylinder intersections retain both quadratic roots, so rays that
begin inside a body find the positive exit rather than leaking through it.

## One-intersection certificate bridge

The depth-tested raster pass uses Metal's baseline 32-byte color-attachment
budget exactly:

- `rgba32uint` packed surface: 16 bytes;
- `rgba16uint` identity/media: 8 bytes;
- `rg32uint` rigid winner certificate: 8 bytes.

The certificate contains exact `f32` ray distance plus an oct12 world normal,
four-bit owner, three-bit feature, and motion-valid bit. A second proxy-covered
pass only unpacks the winning certificate into the existing split geometry and
identity targets. It performs no second intersection and no full-screen scan.
The resulting hit then enters the existing cone prepass, material/PBR shading,
motion, AO, glass, and current rigid shadow overlay.

There is no temporal history, jitter, dithering, camera restriction, or stale
body cache in this path.

## Dawn measurements

Apple M1 Max Metal, 660x662, canonical-parametric traversal, split full-res
relight, cone scale 0.5, shadows and AO enabled. These were short fresh-process
runs under desktop GPU contention, so the within-pair direction is more useful
than the absolute wall time.

| Scene | Bodies | Analytic reduced median | Raster reduced median | Selection |
|---|---:|---:|---:|---|
| hose-tank | 1 | 22.377 ms | 23.849 ms | analytic |
| dam-break-boxes | 6 | 17.553 ms | 16.859 ms | raster |

The six-body raster arm is 3.95% faster in the interleaved comparison. The
adaptive one-body production smoke reproduced the analytic control's configured
image, packed-surface, identity, and hardware-depth hashes.

Replacing exact bridge normals with the certificate changed only 0.195% of
configured hose pixels relative to the prior exact-reintersection raster arm;
RGB RMSE was `4.57e-6` and maximum absolute channel error was `7.32e-4`.

The current-frame rigid GI overlay plus deterministic cone fan-out was measured
again in the hose-tank scene with glass enabled and no temporal accumulation.
Keeping first light samples contiguous lets GLOBAL dispatch only the live
one-sample prefix; light-count and lighting-mode changes update a uniform and
dispatch extent without compiling a shader. The interleaved reduced result was
`18.481 ms` median and `19.910 ms` p95, versus `35.511 ms` for scale 1. The
configured image hash (`0xd220b8b2`), packed surface (`0x8cbfb0ab`), identity
(`0xe0afaa95`), depth (`0x2653efd5`), and reduced quality statistics were
unchanged by active-layer elision.

## Next structural bottleneck

Shadows remain the dominant cost (roughly 9-11 ms in hose attribution). The
static-shadow field now has a GPU-executed, revision-checked publication and
dirty-invalidation ABI. The next performance-producing step is a conservative
SVO proof classifier that publishes useful visible/occluded page-light
certificates; unknown, dirty, mixed, or stale channels must continue to trace
the exact current frame, with live rigid blockers applied afterward.
