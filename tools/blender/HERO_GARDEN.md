# Blender hero garden

The Garden library's **Porcelain pond · Blender garden** (`hero-garden-hose-x10-blender`)
is a separate scene. The original X10 remains available. The new composition retains
its 81 dense planting stands and replaces all visible solid scenery with Blender
meshes: the continuous pond casting and ground, four sculpted boulders, five stepping
discs, four grades of river pebble, oak plus seven miniature woody species, grass,
succulents, and a curved rubber hose with a hollow metal nozzle. Living green and
silver sage foliage, brown bark, earthy soil, honey sandstone coping, warm
limestone and a brass nozzle bring color around the sand glazed porcelain basin. The
closed pond-and-ground mesh has separate soil, coping and basin face materials. The source GLBs and baked volume
palettes carry the same materials.

The editable, assembled scene is `assets/hero-garden-blender/hero-garden.blend`.
Each reusable source collection is named after its GLB. Instances preserve the
same local origins and rotations in Blender and the application. The water object
is in a clearly labelled preview-only collection and is never exported: the
application uses its existing live water and hose jet.

## Oak fidelity

The oak uses the original X10 `heroGardenCloudTree()` specimen as its blueprint:
all 1,532 bowed branch centerlines and radii, all 768 oriented foliage spray
positions, its 0.85 m scale recipe and the original planted base at (0.53, -0.15).
Blender rebuilds the wood as closed tapered meshes and fills each original spray
with twelve small, lobed oak leaves. This restores the winding trunk, recursive
forks and rounded, fine canopy in place of the earlier radial tree with oversized
lanceolate leaves. The imported oak is sampled at 2.5 mm. Its green/brown palette
continues the colored garden; the rest of the composition and fluid are unchanged.

The old generic oak is evaluated only to preserve the existing seeded random
stream, then its meshes are replaced before export. Oak leaf scatter uses each
original spray's seed, independently of every other garden asset.

## Rebuild

Use Blender 5.2 LTS. All randomness is seeded. From the repository root:

```sh
node --import tsx tools/blender/export-hero-layout.ts
BLENDER='/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender'
"$BLENDER" -b --python-exit-code 1 --python tools/blender/hero_garden.py -- --samples 48
python3 tools/blender/bake-hero-assets.py
npm run check:types
npm run test:unit
```

`--no-render` skips only the Cycles presentation. The bake uses three processes by
default (`--jobs 1` for lower memory use). `--resume` reuses existing volumes and
should only be used to resume an interrupted bake with unchanged GLBs. The aggregate
metadata file is published only after every bake succeeds. GLBs, compressed FVX
volumes, the instance table and metadata are runtime assets and must stay together.
The layout exporter shares the original pond's profile, waterline and planting
stations rather than attempting to infer them from a rendered image.

## Rendering and physics

Source samples are 1.5625–6.25 mm for props and the pond, and 25 mm for the flat outer
ground. One continuous Blender ground surround replaces the repeated slabs, with
a central opening tucked underneath the pond casting. Its one-cell distance band
keeps empty source space from enlarging its render bounds. The old 12.5 mm ground
tile asset remains available for previously saved scenes but is no longer instanced.
Each volume is loaded once and shared across its instances. Source mesh
normals and per-material roughness/metallicity survive conversion. These assets
are independent of the fluid solver, which now defaults to 6.25 mm: twice the
previous linear resolution, or eight times as many cells (192 × 128 × 144).
Scenery detail remains 3.125 mm. The original terrain and stepping
colliders still contain the water; the Blender pond finish sits 2 mm above the
collision surface. The original procedural backdrop is replaced by authored ground
and planting instances. The new camera pulls back to show the whole crown.

The scenery graph declares `terrainSurface: "scenery"`. The renderer omits the
duplicate collision heightfield and analytic horizon plane, preventing white
terrain from showing through the Blender finish. CPU and GPU terrain publication,
render occupancy and residual caching respect the same policy. Physics retains
the original heightfield, and later voxel edits remain visible.

The source preview is `public/assets/hero-garden-blender/hero-garden-preview.png`.

### Raster ambient occlusion

Use **Rasterized** primary visibility and **RASTER + AO** lighting, with AO on
and GI off. This is the default product mode. The Blender garden's
`lighting.rasterAoContrast: 2` strengthens the existing horizon and coarse AO
visibility after reconstruction. It adds no samples, cone marches or lighting
passes. Open areas remain unoccluded, and the normal AO strength control still
scales the effect. Other scenes retain contrast 1.

For an on/off/before/after check, set `FLUID_SVO_DRY_FRAME_RASTER_AO_COMPARE=1`.
The comparison requires mesh raster primary, raster-AO lighting and GI off; it
rejects other configurations instead of producing a misleading capture.

For a production renderer capture:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
FLUID_SVO_DRY_FRAME_SCENE_MODULE=tools/preview/hero-blender.ts \
FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT=1 \
FLUID_SVO_DRY_FRAME_TRAVERSAL=raster-primary \
FLUID_SVO_DRY_FRAME_SURFACE_MESH=1 FLUID_SVO_DRY_FRAME_MESHER=dual-marching-cubes \
FLUID_SVO_DRY_FRAME_CONE_TRACING=raster-ao FLUID_SVO_DRY_FRAME_GI=0 \
FLUID_SVO_DRY_FRAME_CONE_SCALE=1 \
FLUID_SVO_DRY_FRAME_RASTER_AO_COMPARE=1 \
FLUID_SVO_DRY_FRAME_WIDTH=1200 FLUID_SVO_DRY_FRAME_HEIGHT=750 \
FLUID_SVO_DRY_FRAME_WARMUPS=1 FLUID_SVO_DRY_FRAME_CYCLES=1 \
FLUID_SVO_DRY_FRAME_OUT=artifacts/hero-garden-blender/frame.json \
node --import tsx tools/benchmark-svo-dry-frame-gpu.ts
```

This dry harness verifies solid scenery; the running scene supplies the fluid.
Tests verify catalog discovery, all 81 stands, signed asset interiors and pond/nozzle
cavities, source-to-runtime material colors, separate dirt and coping surfaces,
asset routing, and unchanged fluid geometry across four lattices. Regression
checks cover terrain visibility versus collision ownership, appended voxel edits,
and continuous ground across the former tile boundaries.

## Verification and rendering cost

The scene uses 26 assets in 394 instances, including all 81 original density stands.
The published library retains 27 assets, including the legacy ground tile. The
1200 × 750 production capture uses 6.25 mm fluid cells (192 × 128 × 144) and
3.125 mm scenery. It built at the requested refinement depth of one and reported
no GPU validation errors. Native imported fields supply the visible surfaces;
the DMC diagnostics report ready with no overflow. On M1 Max it allocates about
1.93 GB of renderer resources; the latest raster-AO world build took about 12.2 s.
The short, settled-frame comparison is visual verification, not a reliable AO
cost benchmark or sustained simulation FPS measurement. The contrast adjustment
adds one scalar power operation per shaded receiver, with no extra visibility
samples or passes. Visual inspection confirms the white terrain show-through and
repeated floor grid are gone.

The earlier experiment with 6.25 mm **scenery** introduced pond surface artifacts,
so scenery stays at 3.125 mm. Fluid resolution and source asset sample pitches are
independent. Runtime owner-material fallbacks use each imported asset's own palette.

The latest capture and report are in `artifacts/hero-garden-blender/raster-ao/`
(local, ignored), with AO off, original contrast and stronger contrast images in
`occlusion/`. The report verifies raster primary, mesh surfaces, raster AO on,
GI off and zero GPU validation errors. `npm run check:types` and
`npm run test:unit` passed: 955 checks,
51 skips. The Blender source preview includes illustrative water; the production
dry capture intentionally omits the simulated water.
