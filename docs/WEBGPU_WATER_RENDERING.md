# WebGPU water rendering

Water presentation uses one GPU-resident raster pipeline. Renderer
initialization fails visibly if an adapter cannot create that required pipeline.

## Frame pipeline

1. **Surface extraction (compute, three stages in one pass).** The solver's
   current volume texture is sampled directly. A lean classification sweep
   loads each cube's eight corners and appends surface-crossing cubes to a
   GPU worklist with one atomic add per surface cube; keeping triangle
   emission out of the sweep kernels preserves the occupancy that hides the
   classification load latency. Restricted tall-cell fields scan only the
   moving cubic band in interior columns; a separate full-height perimeter
   scan preserves the side-wall interfaces needed for closed-volume optics.
   Uniform and adaptive full-height fields retain the exhaustive scan. A
   one-thread prepare kernel then sizes an indirect dispatch, and the
   polygonise stage runs one thread per worklist cube: six marching
   tetrahedra per cube avoid ambiguous marching-cubes saddle cases, and each
   workgroup reserves one contiguous vertex block (two global atomics per
   workgroup — no per-triangle contention) before threads emit into private
   slices. The mesh renders from the same buffer's indirect draw count.
   Vertex normals use the analytic derivative of the cube's already-loaded
   trilinear field, requiring no additional volume samples. There is no
   GPU-to-CPU readback. The remaining known inefficiency is structural:
   tetrahedra emit unindexed triangles, so each surface vertex is duplicated
   across its incident triangles; an indexed table-based marching-cubes
   variant would reduce both vertex traffic and interface-pass work.
2. **Caustic projection (currently disabled).** Surface triangles can be refracted from
   the directional light onto the tank floor and additively accumulated.
   Sampling uses a five-tap reconstruction filter in the scene pass. Both the
   projection pass and sampling branch are disabled while this effect is retuned.
3. **Dry scene (HDR raster).** Sky, floor, tank edges, and rigid bodies are
   rendered without water. Linear scene distance is stored in alpha so objects
   in front of the water remain correctly visible.
4. **Front/back interfaces (two raster passes).** The extracted mesh is rendered
   with opposing cull modes into world-position and level-set-gradient normal
   buffers. Separate depth buffers select the nearest interface of each kind.
5. **Optical composite.** Three image-space secant-style updates follow the
   refracted ray from the front interface toward the back buffer. The exit ray
   then samples the dry scene. Shading includes two-interface Snell refraction,
   Schlick Fresnel (`F0 = 0.02037`), Beer-Lambert absorption, restrained
   in-scattering, environment/scene reflection, and explicit total internal
   reflection. Rear tank seams live in the refracted scene; the near glass pane
   is composited last with its own Fresnel and edge highlights so it correctly
   remains in front of water and submerged objects.

   Rigid-body contacts use a hybrid resolver. The raster interfaces remain the
   fast global representation, but pixels whose analytic rigid depth is within
   1.5 finest cells of the raster front re-intersect the resident trilinear
   liquid field with four bounded Newton updates. Exact primitive depth then
   decides whether water or solid owns the pixel. After refraction, an analytic
   rigid hit that precedes the back interface terminates the water thickness as
   an opaque solid contact; it is not treated as a water-air exit. Pixels away
   from bodies perform no volume samples beyond the ordinary raster composite.

Fullscreen shader UVs have Y=1 at the top of a render target, while texture
sampling uses Y=0 there. The composite converts that coordinate for every
intermediate read and world-space projection. The shared final upscaler retains
the same coordinate conversion.

The virtual extraction lattice closes side and top contacts, but deliberately
extends the lowest liquid sample through the floor. A solid floor is not a
water-air surface and extracting it as one creates a large coplanar optical
sheet. Rays that leave through the floor are terminated analytically against
the tank bounds in the composite shader.

Surface extraction and the caustic map are cached by solver revision. Rapid
solver revisions are coalesced so mesh extraction runs at no more than 30 Hz;
camera and optical rendering remain at display rate. Raster optics renders at
72% internal resolution before linear upscale.

When timestamp queries are supported, the live profiler reports extraction,
dry-scene rasterization, the two interface passes, optical compositing, and
final upscale separately. A paused or unchanged solver reports zero extraction
work rather than carrying a stale extraction sample into the current frame.

## Resource bounds

The append buffer is sized from grid surface area rather than volume:

```text
max vertices = clamp(32 × (NxNy + NxNz + NyNz), 262144, 2097152)
```

At 32 bytes per vertex, extraction is capped at 64 MiB. The classify worklist
holds one 8-byte entry per surface-crossing cube and is sized at one third of
the vertex capacity (at most 5.4 MiB): every appended cube emits at least one
triangle, so the worklist can only clip on fields that would clip the vertex
buffer as well. If an adversarial field exceeds either capacity, whole
triangles are dropped; the indirect count never exceeds the allocation. The
benchmark reports this explicitly: the balanced 20 m deep-water case has
4,041,660 uncapped vertices and therefore clips the 2,097,152-vertex
production buffer. Ordinary shallow scenes remain below the cap.

Full-resolution transient targets are five `rgba16float` textures and two
`depth24plus` textures. The scene remains linear HDR until the final composite,
where tone mapping and display gamma are applied once.

## Selection and validation

Both CPU-uploaded uniform fields and GPU solver textures use the same raster
extraction path; restricted tall-cell fields are unpacked in the compute shader
with the solver's column-base texture.

The shader-only validation command accepts a Naga executable:

```sh
NAGA=/path/to/naga npm run test:water-shaders
```

The ordinary application build and unit suite do not require Naga.

## Extraction performance benchmark

`benchmark:water` measures the full-volume and surface-bounded extraction
paths against the same real tall-cell textures on the same WebGPU adapter. It
requires hardware timestamp queries and does not use CPU submission time as a
GPU proxy:

```sh
WEBGPU_NODE_MODULE=/path/to/webgpu/index.js npm run benchmark:water
```

The default run covers settled fill, dam break, and the 20 m deep-water case.
It performs 20 untimed warm-up pairs and 80 timed pairs, alternating A/B order
to reduce thermal and ordering bias. Each JSON result includes the complete
sample distribution, median and p95 GPU time, a deterministic bootstrap 95%
confidence interval for median speedup, logical and padded dispatch counts,
adapter/backend metadata, and an exact uncapped emitted-vertex-count
comparison. The count path is a pipeline-constant specialization of the same
shader, so deep scenes remain comparable even when the production append
buffer would clip both paths. A count mismatch or WebGPU validation error
fails the run.

Use `FLUID_BENCH_WARMUPS`, `FLUID_BENCH_ITERATIONS`,
`FLUID_BENCH_SCENES`, `FLUID_QUALITY`, and `FLUID_WEBGPU_BACKEND` to control a
run. Benchmark reports should retain the raw JSON rather than copying only the
headline speedup.
