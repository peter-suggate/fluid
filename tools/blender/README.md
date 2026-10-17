# Pool party asset pipeline

Blender 5.2.2 LTS (Apple Silicon) is installed at
`/Applications/Blender 5.2 LTS.app`; the pre-existing Blender 4.0.2 installation
is retained. The installer came from the official Blender release server and
matched its published SHA-256:
`dc4125399b8bfefe283cc1624d6cfc7809d1cac20ace51072127eb371f31f210`.

The four initial assets are a teak lounger with sewn cushions and a towel, an
eight-panel garden parasol, an insulated cooler, and a ceramic refreshment set.
The editable source is `assets/pool-party/pool-party-assets.blend`. Each asset
has its own origin-centred collection. The separate `pool-party-preview.blend`
contains the Cycles presentation stage. Exported meshes, packed textures,
volumes and the preview image live in `public/assets/pool-party/`.

## Rebuild

From the repository root:

```sh
BLENDER='/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender'
"$BLENDER" --background --python-exit-code 1 --python tools/blender/pool_party_assets.py -- --samples 96
"$BLENDER" --background --python-exit-code 1 --python tools/blender/pool_party_landscape.py -- --samples 64
"$BLENDER" --background --python-exit-code 1 --python tools/blender/bake_mesh_volume.py
node --import tsx tools/blender/check-pool-assets.ts
npm run check:types
npm run test:unit
```

The baker also accepts `-- --asset teak-lounger --cell 0.00625`. Its default voxel
spacings are 10 mm for the lounger, parasol and cooler, and 5 mm
for the ceramic set. The diving assembly, pool ladder and stone paver use
6.25 mm; the turf uses 3.125 mm to resolve its narrow curved blades. These are independent of the pool's 50 mm fluid lattice.
Changing `--cell` changes the sampled geometry; features below that spacing
cannot be recovered just by retaining their normals.

## What survives voxelization

The baker imports the exported GLB back into Blender, evaluates its actual
triangles, checks that every component is watertight, and computes signed
distance using nearest-triangle queries and ray parity. Components combine by
the minimum signed field. It retains an eight-cell brick wherever the solid or
its three-cell distance band can reach it. Adjacent bricks share identical
9³ node samples. Coordinates are metres, Y up, with `(x, z, -y)` converting
Blender coordinates to glTF/application coordinates.

Each 16-byte node carries:

| Channel | Representation | Purpose |
| --- | --- | --- |
| Signed distance | float32, metres, negative inside | Position and occupancy of the reconstructed surface |
| Geometric normal | two octahedral uint16 coordinates | Source distance gradient direction for fitting and geometric shading |
| Shading normal | two octahedral uint16 coordinates | Interpolated source corner normal plus tangent normal map, independent of geometry |
| Material | uint32 palette index | Material identity at the closest surface |

The file starts with `FVX1`, a little-endian uint32 JSON byte length, the padded
UTF-8 JSON header, and fixed-size brick records. A brick starts with three
int32 cell-origin coordinates. Node order is `x + 9*(y + 9*z)`. The complete
file is gzip compressed. `decodeMeshVolume` validates dimensions, payload
length, node ranges, material IDs, duplicates and shared faces.

`sampleMeshVolume` returns the trilinear distance and its analytic derivative
alongside the retained geometric/shading normals. `meshVolumeToSolid` imports
the geometry into the existing `SolidWorld` patch and `ProceduralSolidBoundary`
contracts. The existing smooth mesher can therefore reconstruct its surfaces
from a continuous field, rather than the occupancy bitmask. This adapter uses
the derivative of the sampled field; source shading normals remain in FVX for
the material/render path. Normals alone cannot repair a blocky silhouette.

The baker samples base-colour textures at the closest triangle's UV in linear
light and retains roughness and metallic in a shared PBR palette. Roughness
textures use eight quantisation buckets, while untextured roughness stays exact.
Colour variants are quantised to five bits per sRGB channel. The runtime packs each node into
eight bytes: float32 distance plus an oct8 shading normal and uint16 local
material index. Instances share the same packed volume and palette. Tangent normal maps
are sampled through the actual triangle UV basis and baked into the independent
shading-normal channel. The geometric gradient and distance remain unchanged.
The original embedded texture images and UVs remain in GLB.

## Live scene and lighting

The actual `backyard-pool-party` preset now places three imported loungers, the
parasol, cooler and refreshments, plus the Blender diving assembly and pool
ladder. The patio uses 160 instances of one honed travertine slab, and the lawn
uses 128 instances of one metre-square turf patch. Quarter turns vary the
repetition without duplicating sample data. Their sampled fields participate in scenery
voxelization and use source PBR materials. A depth-tested imported-field pass
preserves the source asset's finer detail when the surrounding visual lattice
is coarser. It uses the same lights, shadows and display grade as the rest of
the scene. Smooth normals survive independently of surface occupancy. Imported receivers
also trace their native sampled fields for direct-light visibility. Their coarse
opacity voxels are excluded from that one shadow walk, while ordinary world
voxels and moving bodies still occlude. This prevents the parasol from shadowing
its own upper fabric in patches and preserves its real shadow on the chairs.

The pool preset uses a warm afternoon sun, cool sky fill, warm terrace bounce
and an AgX display transform to approach `pool-party-assets-preview.png`.
AgX compresses highlights and saturation instead of clipping pale fabric into
flat white. Benchmark PNGs now apply the scene's actual display grade, matching
the application's presentation path. Cycles still has more complete light
transport; this is a real-time voxel approximation, not pixel parity with Cycles.

The visual lattice can run at 6.25 mm (environment refinement depth 3) while the
fluid lattice remains 50 mm. Asset sampling is independent again: the current
volumes use 3.125–10 mm spacing, so choosing a 6.25 mm world lattice does not invent
new sub-voxel detail in a 10 mm source bake. Re-bake an individual asset with
`--cell 0.00625` when finer source geometry is needed. The cooler is a closed
exterior prop; the pitcher and tumblers have hollow interiors.

## Memory

The largest allocations are world voxel payloads and fitted-surface maintenance,
not the textures. The initial furniture-only 6.25 mm construction profile reported roughly 3,795 MiB of
payload and 3,961 MiB of maintenance, versus 47.6 MiB for the four unique imported
GPU volume images. These are components of the source allocation, not extra
costs to add to its reported total. The profiler's cumulative buffer requests
include temporary buffers and must not be read as peak resident memory.

The implementation reduces these costs in four places:

- Fitted points use 8 bytes instead of 16, with 10 bits per local coordinate.
  Quantisation stays finer than the mesh's output coordinate precision.
- Candidate scratch is capped at 32 MiB and reused in ordered chunks instead
  of reserving 512 primitive indices for every world brick.
- Proven constant regions stay coarse. Planar boundaries coarsen by at most
  two levels; thin slabs, edges, gaps and curved props remain refined.
- Directional visibility uses a tagged 64 MiB working-set cache, replacing the
  full-world texture that requested 4.79 GB in this scene. Hash collisions are
  cache misses and trace normally; they cannot reuse another voxel's shadow.

Large CPU-to-GPU table updates are split into 1 MiB writes to bound staging
allocations. The initial furniture-only world construction at depth 3 used 964,810 leaves. That source alone
occupied about 9.2 GiB; rendering and the fluid solver need additional
memory. The full 6.25 mm mesh also reserved about 2 GiB in the wide-view capture.
This remains a high-memory mode, not a claim that the entire app fits in 9.2 GiB.

## Reproduce and inspect

Record source allocations without permitting a lower-resolution fallback:

```sh
FLUID_WEBGPU_BACKEND=metal FLUID_POOL_DEPTH=3 \
node --import tsx tools/blender/profile-pool-memory.ts
```

Render the actual pool preset through the production GPU renderer:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_WEBGPU_BACKEND=metal \
FLUID_SVO_DRY_FRAME_SCENE_MODULE=tools/preview/pool-party-live.ts \
FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT=3 \
FLUID_SVO_DRY_FRAME_WIDTH=1000 FLUID_SVO_DRY_FRAME_HEIGHT=720 \
FLUID_SVO_DRY_FRAME_WARMUPS=1 FLUID_SVO_DRY_FRAME_CYCLES=1 \
FLUID_SVO_DRY_FRAME_TRAVERSAL=raster-primary \
FLUID_SVO_DRY_FRAME_SURFACE_MESH=1 FLUID_SVO_DRY_FRAME_MESHER=dual-marching-cubes \
FLUID_SVO_DRY_FRAME_CONE_SCALE=0.5 \
FLUID_SVO_DRY_FRAME_RAW_OUT=artifacts/pool-party-live/frame.raw \
FLUID_SVO_DRY_FRAME_OUT=artifacts/pool-party-live/frame.json \
node --import tsx tools/benchmark-svo-dry-frame-gpu.ts
```

`tools/preview/pool-party-detail.ts` changes only the camera for a close view of
the same scene. `tools/preview/imported-pool-asset.ts` isolates one asset selected
by `FLUID_POOL_ASSET` for inspecting material and silhouette defects. JSON reports
record GPU validation errors, mesh status, visual/fluid spacing and timings;
`reference.png` is the full-rate graded render, while `reduced.png` uses the
chosen reduced lighting rate.

Validation on Apple M1 Max / Metal: the full scene constructed and rendered at
6.25 mm with no resolution fallback, no GPU validation errors, and a ready DMC
mesh with no overflow. The ordinary 25 mm visual setting was also checked with
close views of the same props. Captures are dry geometry diagnostics: they omit
the evolving water while exercising the production asset and lighting path.

## Diving equipment and landscape

`pool_party_landscape.py` creates `assets/pool-party/pool-party-landscape.blend`
and a separate Cycles preview. It preserves the furniture manifest entries;
regenerating the furniture also preserves landscape entries.

The diving assembly has powder-coated supports, flanged anchors, polished 316
steel rails with continuous bends, rubber stair treads, a fibreglass springboard,
and a mineral-grit walking surface. Its origin is the original board centre in
XZ, on the patio in Y. The existing simulation collider and downhill slope are
unchanged; the cosmetic board envelope encloses it. The opposite pool ladder
has curved rails, escutcheons, wall standoffs and anti-slip steps.

The travertine slab has 5 mm eased edges and authored albedo, roughness and
normal textures. Pores change shading, not the flat geometric gradient. Texture
sampling is bilinear in the correct colour space, including glTF packed-channel
roughness. The turf contains 1,568 closed curved blades per square metre, six
blade colours and a thin root mat. Blade tips narrower than the source sampling
pitch soften or disappear; this is finite-resolution geometry, not infinite
detail. Blade clusters are partitioned into local BVHs to bound bake work.

The baker indexes expanded part bounds per brick instead of scanning every
object for every sample. The lookup depends only on the global lattice-node
coordinate, preserving identical samples on shared brick faces. All repeated
pavers/turf share their packed GPU images and material palettes. Their detail
never expands the fluid grid. Native volume rendering still adds real GPU work;
sharing data does not make hundreds of grass patches free to shade.

`tools/preview/pool-party-diving.ts` and `tools/preview/pool-party-ground.ts`
inspect these additions in the actual pool preset using production lighting.

The imported assets also carry a small shared CPU prefix index of occupied
source bricks. Empty parts of a tall assembly's bounds do not force refinement.
The stone baker checks every sampled node in its inset planar region before
publishing a slab certificate: only those flat interiors coarsen, while bevels
and joints retain the fine world lattice. Certificates rotate only by axis-aligned
quarter turns; oblique placements conservatively remain fine.

For curved imported assets, the native field pass remains the visual surface at
its authored 3.125–10 mm pitch. Its redundant world occupancy/shadow proxy may
use cells up to four times coarser, capped at 25 mm. This avoids storing grass
blades twice at full resolution. Ordinary intersecting geometry and edits still
prevent coarsening; the 50 mm fluid representation is unchanged. Native surfaces
and shadows between imported objects continue to trace the fine source fields.
Ordinary voxel shadow receivers use the bounded occupancy approximation.

All eight unique packed source images total 217.5 MiB per image arena. Stone is
4.91 MiB shared across 160 slabs, and turf is 47.21 MiB shared across 128 patches;
naively duplicating just those images would cost about 6.67 GiB. Prefix indices,
instance records, world occupancy and other rendering buffers are additional.

Final M1 Max / Metal verification with the landscape: refinement depth 3 was
requested and built, with a 6.25 mm visual lattice and 50 mm fluid configuration.
The dry capture reports no GPU validation errors, a ready DMC mesh, no overflow
and no fallback. It builds 788,839 occupied world bricks (795,477 capacity with
reserve), versus 964,810 previously. Reported source allocations decrease from
9.21 GiB to 7.74 GiB despite adding the landscape, about 16%. This includes
3.03 GiB payload and 3.36 GiB maintenance; the mesh separately reserves 2 GiB,
and the evolving fluid solver is not part of this dry-render measurement.

The ordinary 25 mm world setting was also checked with the 6.25 mm native diving
assembly. Native shadow rays retain a small bias; ordinary voxel shadow rays
clear their coarser cell uncertainty so the underlying board collider does not
leave dashed self-shadows on the imported deck. Close grass still has finite
voxel tips and pixel aliasing; the real-time renderer is not Cycles image parity.
The Blender source preview is `pool-party-landscape-preview.png`; live captures
are `artifacts/pool-party-diving/reference.png` and
`artifacts/pool-party-ground-6mm/reference.png`, with JSON reports beside them.

Validation: eight GLB/FVX assets pass sign, cavity, gradient, shared-node and
SolidWorld-adapter checks. Tests cover independent stone normal/roughness maps,
shared image packing, rotated planar certificates, sparse support under rotation,
and conservative proxy refinement. `npm run check:types` and `npm run test:unit`
pass (931 tests: 878 passed, 53 GPU-only tests skipped).

### Version control

Commit the origin-space `.blend` sources, authoring/baking scripts, GLB exports,
FVX payloads and their JSON metadata/manifest. The application and tests need
the exported assets without requiring a local Blender installation. Textures
are packed into the Blender sources and GLBs. Standalone PNG texture exports,
rendered previews, presentation-only `*preview.blend` files, Blender backups
and `verification.json` are generated locally and ignored. Runtime captures
remain under the already ignored `artifacts/` directory.

### Architecture and pool surround

`pool_party_architecture.py` adds six reusable 6.25 mm source fields: cedar fence
panels, stucco window bays, straight and corner limestone coping, glazed mosaic
waterline strips, and a slatted teak serving table. The scene instances these
instead of the old flat windows, fence pickets, continuous coping bars and
box table. The physical pool and diving collider remain unchanged.

Wood and mineral surfaces carry albedo, roughness and tangent normal maps.
Manufactured edges, sill projections, mullions, roof flashing, fasteners, tile
relief and furniture joints are geometry. Window glazing currently uses opaque
tinted PBR: the renderer has no imported transmission or interior refraction.
All repeated panels/stones share the same source samples and material palette;
none of this detail is inserted into the 50 mm fluid grid.

Rebuild this pass from the repository root:

```sh
BLENDER='/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender'
"$BLENDER" --background --python-exit-code 1 --python tools/blender/pool_party_architecture.py
for asset in cedar-fence-panel stucco-window-bay limestone-coping limestone-coping-corner ceramic-waterline teak-serving-table; do
  "$BLENDER" --background --python-exit-code 1 --python tools/blender/bake_mesh_volume.py -- --asset "$asset" || break
done
node --import tsx tools/blender/check-pool-assets.ts
```

Editable sources are `assets/pool-party/pool-party-architecture.blend` and its
separate presentation file. `tools/preview/pool-party-architecture.ts` and
`tools/preview/pool-party-service.ts` inspect the actual scene under production
lighting; the Cycles presentation is not a real-time renderer screenshot.

The architecture capture at `artifacts/pool-party-architecture-6mm/reference.png`
builds the requested refinement depth 3: 6.25 mm visual cells with the 50 mm fluid
configuration. It reports a ready mesh, zero overflow, zero fallback and no GPU
validation errors. Source allocation is 7.22 GiB (previous landscape pass:
7.74 GiB), with 711,940 occupied world bricks instead of 788,839. The changed
architecture and shared imported proxies reduce world storage despite adding
174.2 MiB of source samples. All fourteen unique packed images total 391.7 MiB
per arena; the window bay is 117.5 MiB shared by three placements, and the fence
panel is 19.25 MiB shared by nineteen placements. The mesh reserves a separate
2 GiB. These are dry rendering measurements, without the evolving fluid solver.

All fourteen GLB/FVX assets pass the sign, cavity, gradient, shared-node and
SolidWorld-adapter verification. The service close-up also checks the production
25 mm world setting with the new geometry still drawn at its native 6.25 mm.
