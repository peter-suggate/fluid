# First Light — alpine tarn

The live scene `alpine-lake-sunrise` uses the production uniform narrow-band FLIP method. Its elliptical terrain basin closes inside the 64 × 8 × 80 m domain. Water is initialized at 2.3 m above the domain floor and has four 0.5 m cells of depth at the centre. A weak initial horizontal velocity perturbs the mirror surface. The Blender environment does not contain an alternate liquid simulation or a proxy water plane.

Export a stepped solver surface with the shared exporter, then render:

```sh
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
  node --import tsx tools/blender/export-nbflip-water.ts \
  --scene=alpine-lake-sunrise --time=0.5

'/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender' -b --python-exit-code 1 \
  --python tools/blender/alpine_lake.py -- \
  --water-mesh artifacts/water-scenes/alpine-lake-sunrise/water.obj \
  --output artifacts/water-scenes/alpine-lake-sunrise/alpine-lake.png \
  --width 1800 --height 1200 --samples 192
```

The script writes an editable `.blend` beside the PNG. `--no-render` saves the scene without launching Cycles. Output dimensions, sample count and paths are configurable. Missing liquid data raises an error; the script never silently manufactures substitute water.

The OBJ uses Blender Z-up metres, mapped from application `(x,y,z)` to `(x,-z,y)`. The importer uses forward Y/up Z, welds coincident triangle vertices at 10 micrometres, and smooths vertex normals without moving the fluid surface. It prints imported world bounds for comparison with the exporter report.

The visual bed mirrors the runtime terrain equation, including the sign flip in its small sinusoidal term. Geometry beyond the basin is artistic scenery: fractured snowy massifs, subalpine firs, mineral stones and localized volumetric mist. Cycles uses water IOR 1.333, absorption, eight transmission bounces, a high-altitude physical sky and low warm sun. The desktop scene has its own low shore camera and the imported environment described below; the large-scale Cycles environment is an offline look-development variant.

The scene-specific tests check containment, resolved depth, validation and scene-file round trips. The `.json` accompanying the water export records the actual simulation time and source geometry; preserve it with renders.

## Interactive environment

`export-alpine-runtime.py` authors a closed, eroded mountain range roughly 150 metres high, with photographed mineral strata and snow on gentler upper slopes. A low connecting skirt starts at the far shore. The source GLB retains its authored triangles and image materials for Blender. The interactive renderer reconstructs an isosurface from the 1.25 m FVX samples and keeps geometry and visibility entirely voxel based; it does not load GLB triangles. Its fallback shading uses the sampled material palette. The optional shared **Lighting → Image textures** control now reads an FVM1 optical companion for the ridge, applying the original color, roughness and normal images through voxel-sampled affine UV mappings. This restores image detail while retaining the same 1.25 m geometry. The shared renderer reconstructs optical attributes from the eight valid neighboring palette samples in linear space, while retaining the elected material ID for visibility and ownership. This removes palette-election discontinuities but cannot recover unsampled detail by itself. FVM image attributes recover texture detail; they do not increase geometric resolution. Two new alpine fir assets deform the source forest cedars coherently, breaking their regular horizontal tiers while retaining closed foliage and branches. These use 4 cm and 3 cm sample spacing with explicit component-union solid semantics. This preserves overlapping closed leaf envelopes instead of creating odd/even overlap holes. Their source GLB geometry and materials remain unchanged. The source cedars remain unchanged. Rounded pebbles and dry-bank boulders reuse the Blender forest library. The editable source is `assets/alpine-lake/alpine-runtime.blend`. A separate `alpine-shore-scree.blend` contains 1,011 closed rounded cobbles in an irregular dry-bank deposit, using the same photographed mineral texture and the exported terrain heights. The shore asset uses 10 cm FVX samples (2,127 bricks, about 2.7 MB compressed), adding one instance to the scene. Its provenance file records the actual positions and source terrain hash. Run `export-alpine-live-layout.ts` before regenerating the shore to refresh those height samples.

```sh
'/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender' -b --python-exit-code 1 \
  --python tools/blender/export-alpine-runtime.py
'/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender' -b --python-exit-code 1 \
  --python tools/blender/export-alpine-shore.py
'/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender' -b --python-exit-code 1 \
  --python tools/blender/bake_mesh_volume.py -- --collection alpine-lake
'/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender' -b --python-exit-code 1 \
  --python tools/blender/bake_surface_mapping.py -- --collection alpine-lake \
  --asset alpine-mountain-ridge --output-dir artifacts/water-scenes/alpine-lake-sunrise/ridge-fvm-staging --publish
npm run generate:assets
```

The general **Frame → Render resolution** control supports 200% supersampling (four rendered pixels per output pixel). The capture harness exposes the same choice with `--render-scale=2`; it retains the requested output dimensions and filters the higher-resolution scene. This is a shared quality/performance choice, particularly relevant for fine foliage.

`lib/core/alpine-lake-scenery.ts` places the ridge, firs and shoreline stones. The runtime camera stands at `(0, 4.5, 37)` metres, just above the near bank, facing the far-shore ridge. Static scenery contains no water mesh; every visible live liquid surface is produced by the running narrow-band FLIP solver. Tiny decorative pebbles do not alter the simulation's continuous terrain collider.

## Rock texture provenance

Cycles look development uses **Aerial Rocks 02**, a CC0 photogrammetric material by Rob Tuytel / [Poly Haven](https://polyhaven.com/a/aerial_rocks_02), at its measured 50 m tile scale. Color, OpenGL normal, displacement and roughness inputs and checksums are retained in `assets/alpine-lake/textures`; the color, normal and roughness maps are connected to the rock shader. Generated `.blend` files pack those images. Powered by Poly Haven. The live ridge uses a reproducibly neutralized variant of the photographed color map (15% original chroma, 85% luminance, 0.8 intensity), with original normals and roughness. Source GLB materials embed these images; the common FVX baker samples them into the material palette and shading normals used by the live renderer. The FVM companion packages those same images as raw GPU-ready mip levels; SHA-256 binds its coordinates to the exact FVX payload. The generic renderer never reads source GLB geometry. See [FVM_FORMAT.md](FVM_FORMAT.md) for the backward-compatible optical ABI. Disabling image textures (`svoSurfaceTextures=0`) returns to the filtered FVX palette. The shared baker now estimates each cell’s texture footprint from the triangle UV/world Jacobian, using linear-color and scalar-roughness area mips and filtered normal vectors. The shoreline and ridge use this filtering; other previously baked assets retain their recorded texture policy until rebaked. `tools/blender/test_texture_filter.py` verifies analytic checker means, world-scale invariance, normal cancellation, and non-power-of-two images.

## Compare the live layout in Cycles

For renderer diagnosis, export the actual interactive graph and render its same GLBs, placements, physical terrain, camera, authored light palette and stepped fluid contour:

```sh
node --import tsx tools/blender/export-alpine-live-layout.ts
'/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender' -b --threads 4 \
  --python-exit-code 1 --python tools/blender/render-alpine-live-reference.py -- \
  --water-mesh artifacts/water-scenes/alpine-lake-sunrise/water.obj \
  --output artifacts/water-scenes/alpine-lake-sunrise/cycles-live-reference.png \
  --width 1000 --height 700 --samples 64
```

This reference deliberately adds no extra trees, mist or distant photograph. Full source GLB mesh detail replaces the live FVX reconstruction, and Cycles traces actual reflection/refraction rays. The terrain has the identical geometry with a neutral mineral material because its live procedural terrain shader is renderer-native. The diagnostic sky matches the authored palette; the optional live cloud field is renderer-native and is not reproduced in this reference. The runtime's linear exposure multiplier is converted to Cycles EV with `log2(exposure)`. Each render now writes a JSON sidecar with source-asset hashes and the supplied water export’s complete provenance. The water is a recorded solver run, which may predate later solver edits; matching simulation times across separate captures does not imply identical fluid geometry. The standalone artistic `alpine_lake.py` composition remains separate; it supports `--device CPU` when a Metal driver cannot render its shader set.

Candidate source exports can use `--output-dir` and `--source-dir` to render and inspect new assets before updating the live collection. The latest source comparison retains the exact camera and real water and exposes remaining gaps: repeated tree crowns, coarse snow boundaries, and the smooth mineral terrain proxy. These references are progress evidence, not a claim that photorealism has been achieved.

The initially enlarged 600 m ridge exceeded the general renderer’s construction budget despite a coarse source FVX. The integrated range is uniformly scaled to one quarter of that candidate: source GLB vertices, FVX origin, cell/band widths and metre-valued signed distances all scale together; UVs, materials, normals and sparse indices remain invariant. `rescale-alpine-ridge.py --factor .25` performed that verified conversion without resampling. The source exporter now directly produces the final scale. Do not rerun that one-time conversion on the already scaled assets.

Foliage bake experiments remain staged under `artifacts/water-scenes/alpine-lake-sunrise/`. The generic volume baker accepts `--solid-rule component-union`, or manifest `solidRule: "component-union"`, for intersecting disconnected closed shells. It welds coincident vertices only to identify topology, then uses independent component BVHs and the minimum signed distance; source positions and corner attributes remain unchanged. Concave cavities within a connected shell remain empty. The default `even-odd` rule retains previous asset semantics. Leaf envelopes in the original fir bake overlap: 655 of 3,000 deterministic leaf-center probes were omitted by global parity despite lying inside the component union. A staged 8 cm union candidate and a 4 cm default-parity candidate measure this separately from resolution; neither is automatically published.

The diagnostic reference accepts `--sun-angle RADIANS` (default `0.065`, matching the numeric live shadow-aperture default). Use `--sun-angle 0.009250245` to reproduce the earlier physical 0.53-degree sun reference. Each render records the chosen angular diameter; the two renderers still use different shadow filtering and visibility methods.

The final fir volumes use the validated finer component-union candidates: fir0 has 9,919 bricks (23.24 MB compressed), fir1 has 7,971 bricks (19.76 MB compressed). They share unchanged source GLBs with the Cycles reference. The previous production FVX files remain archived for comparison; resolution increases GPU geometry and memory cost. The original-resolution union and finer even-odd experiments remain available as diagnostic artifacts.
