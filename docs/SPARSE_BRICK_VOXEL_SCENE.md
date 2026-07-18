# Hybrid sparse-brick scene

The interactive WebGPU application has one simulation representation: the
octree solver publishes fluid state and conservative solid proxies into a
shared sparse-brick ABI. Presentation is deliberately hybrid. Water is derived
from the octree level set, while glass, rigid bodies, and terrain retain their
analytic or mesh raster surfaces so their silhouettes, normals, and thin
features are not limited by voxel resolution.

Other fluid solvers remain available to offline comparison tools, but the UI
method control resolves to the voxel octree.

## Representation boundary

| Scene element | Simulation / modelling authority | Smooth presentation |
| --- | --- | --- |
| Fluid | Octree level set, velocity, and liquid fraction in sparse bricks | Surface extracted from the octree level set |
| Rigid bodies | Authored analytic primitive plus conservative voxel owner/material proxy | Smooth analytic geometry in the dry-scene pass |
| Terrain | Authored heightfield plus conservative voxel solid proxy | Raster terrain surface |
| Container walls | Analytic boundary plus conservative voxel boundary proxy | Raster glass interfaces |
| Room and props | Sparse voxels for the room shell and every authored prop | Existing procedural environment shading |

The voxel proxies are used for fluid boundary conditions, collision queries,
CSG, occupancy, and unified spatial inspection. They are not substituted for
the authored presentation surfaces in smooth mode. A moving rigid body keeps
its analytic transform and its voxel proxy synchronized from the same GPU scene
state; neither representation is reconstructed from the other after motion.

## Runtime path

1. `WebGPUOctreeProjection` remains the fluid authority and cold-starts its
   pressure octree over the complete domain on the GPU.
2. `GPUFluidBrickResidency` classifies every 8³ solver brick from the resident
   level set. Disconnected bodies produce independent core bricks; a signed-
   distance halo keeps interpolation and pressure stencils valid. Activation,
   hysteretic retirement, and indirect worklists remain GPU-owned.
3. Subsequent pressure topology and solid rebuilds consume that same active
   core/halo worklist. The scene page table maps every logical solver brick to
   its finest sparse-tree leaf, so pressure scheduling and scene publication
   share one evolving residency decision without a CPU readback.
4. `OctreeSparseBrickWorld` publishes the final substep into 8³ bricks. Its
   topology, signed distances, solid fraction, velocity, liquid fraction,
   material ID, and owner ID remain GPU-resident. Only core/halo payloads are
   refreshed; retired payloads are cleared while static environment material
   is preserved.
5. The smooth viewport presents opaque rigid bodies and terrain in the dry
   scene. Water and smooth glass contribute optical interfaces. A shared
   lighting/material contract shades opaque surfaces, and the optical composite
   resolves reflection, refraction, absorption, and interface ordering.
6. Raw-voxel and brick-grid views compact the same sparse publication on the
   GPU and issue indirect draws. Raw mode shows active payload cells in every
   terminal leaf; grid mode shows terminal leaf-brick bounds. Internal branch
   nodes are topology, not draw instances. Neither mode reads topology or
   counts back to the CPU.

Raw-voxel inspection shows liquid payload, not allocation: empty halo cells are
not painted as water. Brick-grid inspection exposes allocation state instead;
core fluid bricks are blue and stencil-halo bricks are purple. A source brick
therefore visibly changes from core to halo/vacant as its liquid migrates.

Room-shell voxels remain modelled and contribute leaf topology, but raw mode
discards their fragments so an enclosing front wall cannot conceal desks,
chairs, fixtures, and other interior props. Brick-grid mode still exposes the
shell's allocated leaf regions.

The tree uses one finest-level lattice for the whole scene. Fluid-domain bricks
stay at solver resolution while off-tank room and prop regions coarsen by at
most one octree level. This makes authored scene geometry four times finer
linearly than the initial three-level-coarsened representation. Branches
intersecting the solver domain are recursively
split, so the fluid publication has no resolution gaps. Empty room volume is
not densely allocated.

Environment catalogs cover every visible authored prop. For example, the night
laboratory catalog includes the complete desk and bench, legs, shelf,
controller, lamp, stool/chair, counter, monitor, keyboard, instruments,
bottles, and four ceiling lights. The room shell is included as thin floor,
wall, and ceiling slabs. Garden terrain stays under its analytic heightfield
authority rather than being falsely represented by a filled bounding box.

## Unified lighting and optics

Hybrid means different geometry producers, not different looks. Smooth solids,
the extracted water surface, glass, and voxel debug rendering use stable
material IDs and authored linear colors. Production surface shaders share the
same light direction, exposure, environment response, and material parameters.

The production frame is ordered as follows:

1. Present opaque surfaces into the HDR dry-scene color and depth targets.
2. Generate water and glass front/back interface data with their authored
   normals and indices of refraction.
3. Resolve those interfaces in the optical composite against the dry scene,
   applying the same environment lighting and linear color pipeline.
4. Apply the selected raw-voxel or brick-grid inspection pass only when that
   debug view is requested.

Glass remains an interface material rather than a volume of visible cubes.
Rigid bodies and terrain retain smooth authored normals and exact silhouettes.
Their voxel material and owner IDs still make the simulation proxies traceable
to the raster objects.

## Material and color contract

Material zero is empty. Stable IDs are defined in `lib/voxel-scene.ts` for
container glass, terrain, water, and every rigid primitive. Environment proxy
IDs begin at 32 and remain stable within the catalog through stable owner keys.
Colors are authored and uploaded as linear RGB values; they must not receive an
additional sRGB conversion. Prop colors, emission, and roughness use the exact
authored values. Procedural room patterns use representative shell materials,
because a sparse solid proxy encodes material identity rather than a baked
screen-space texture. Smooth water and glass use optical material parameters,
while raw mode uses explicit inspection colors chosen for legibility.

The same material ID must identify an object in its raster draw, voxel proxy,
debug draw, and optical interface. Owner IDs provide the per-instance link for
rigid bodies. This keeps color and material selection deterministic across
smooth and inspection modes without requiring the presentation geometry to be
voxelized.

## Validation

- `npm run test:unit` checks Morton addressing, topology packing, overflow,
  scene coverage, material stability, camera projection, UI modes, and resource
  lifetime.
- `npm run test:water-shaders` validates the shared smooth lighting and optical
  WGSL.
- `WEBGPU_NODE_MODULE=... npm run test:webgpu:voxel-scene` runs the real octree
  fluid regression, reads back sparse records for QA, checks colors and finite
  bounds, and separately submits raw-voxel and brick-grid draws on the real
  adapter. Readback exists only in the smoke harness.
- `WEBGPU_NODE_MODULE=... npm run test:webgpu:garden-brick-migration` starts the
  garden dam break with exactly one core fluid brick, verifies that multiple
  neighboring core bricks activate, and checks that the original brick has no
  liquid payload after the release.
- The production smooth smoke exercises the hybrid renderer: raster dry scene,
  octree-derived water, raster glass interfaces, and optical composition. It
  must not use voxel cubes as the smooth presentation for glass, rigid bodies,
  or terrain.
- The night-laboratory real-GPU smoke requires nonzero environment voxels and
  verifies that the full furniture catalog is published with finite bounds and
  valid material IDs.
