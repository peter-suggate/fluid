# Shape lab: analytic → mesh → voxels

**Date:** 2026-08-08
**State:** design. Nothing built. Every measured number below was taken at HEAD
`bf3c360` by direct probe (§2); every file:line was verified by read.
**Ask:** make the shape lab generate rastered triangle meshes so a form can be
iterated at interactive rates, with a split pane whose second half voxelizes on
click.
**Decisions taken with the ask:** the pipeline is **analytic ⇒ mesh ⇒ voxels**
(a chain, not two views of one thing); scope stays the `hero-garden-hose`
specimens; meshes are on-screen only, no export.

---

## 1. TL;DR

Today the lab has two representations and they are siblings: the CPU tracer
either intersects the SDF (*Exact*) or tests the SDF at voxel centres
(*Voxels*). Production does the same — `sampleSparseScenePrimitiveCell`
(`lib/webgpu-sparse-scene-proxies.ts:1347`) reads the field directly. **There is
no mesh anywhere in this codebase**: no index buffer in `lib/`, no triangle
voxelizer, no triangle scene geometry of any kind.

This inserts one, as a real stage with its own input boundary:

| stage | what it is | produced by | today |
|---|---|---|---|
| **analytic** | the authored field — `SvoPrimitiveDescriptor[]` | `createHeroGardenHoseSceneWithSet` → `buildEnvironmentProxyCatalog` → `svoDescriptorForEnvironmentProxy` | exists, unchanged |
| **mesh** | triangles on that field's zero level set | **new** — GPU isosurface extraction, *or* an import | does not exist |
| **voxels** | occupancy at `shapeLabLeaf_m(depth)` | **new** — GPU triangle voxelization | exists, but fed from the field, not from a mesh |

Two consequences, and the second is the reason this is worth more than a speedup:

1. **Interactivity comes from the mesh being cached.** Extraction runs on a
   parameter edit; orbit, pan and zoom then cost nothing. That is the whole
   interactive loop, and it cannot be had on the CPU (§2).
2. **A mesh stage is what unblocks the oak article.**
   `docs/oak-tree-species-plan.md` ruled out all three of the 80.lv piece's
   load-bearing techniques, foliage cards first, on the grounds that *"a plane
   has no interior. It cannot be voxelized, has no SDF."* That is true of a
   field-fed voxelizer and false of a mesh-fed one: a card voxelizes to a shell
   of authored thickness. The lab is where that gets proven before the renderer
   commits to it.

---

## 2. The measurement that decides the architecture

Probed at HEAD, depth 3 (0.781 mm leaf), specimen `oak`
(`tmp/probe-specimens.ts`, `tmp/probe-mesh-cost.ts`):

```
document at depth 3: 37 ms
oak     records=652  expand=92ms   smooth-union-cluster:507  field-program:145   env=[0.019..0.424]m
stone   records=137  expand=29ms   rounded-cylinder:9 cone:9 smooth-union-cluster:119
plant   records=121  expand= 6ms   ellipsoid:3 cone:118
hose    records=  6  expand= 2ms   smooth-union-cluster:5 capsule:1
layout  records= 24  expand= 0ms   ellipsoid:4 cone:20
```

**The oak publishes zero analytic primitives.** 507 clusters and 145 field
programs — both marched kinds. So "generate a triangle mesh" for the specimen
the ask actually names cannot be parametric meshing of boxes and cones; it is
isosurface extraction from an SDF, for every triangle.

CPU sample cost, same records, median record of each kind:

| kind | µs / sample |
|---|---|
| `smooth-union-cluster` | **1.71** |
| `field-program` | **7.55** |

Summed per-record grid budgets over the oak's 652 envelopes:

| mesh cell | dense samples | bounding-shell band |
|---|---|---|
| 1 mm | 1 685 M | 49.5 M |
| 2 mm | 218 M | 12.9 M |
| 4 mm | 29.4 M | 3.5 M |
| 8 mm | 4.3 M | 1.0 M |

A 4 mm *narrow band* — the cheapest thing that still shows an oak — is ~3.5 M
samples ≈ **10 s single-threaded, ~1.3 s across the eight-worker pool**, and
that ignores the coarse pass needed to find the band. There is no CPU
configuration that meshes this specimen inside a drag.

**Therefore extraction is a compute pass, or it does not exist.** Every other
choice in this document follows from that one.

### WP0: the one number still unmeasured

GPU cost of a field-program tape evaluation. The whole budget rests on it and
nothing in the tree records it. Cheapest read: the live-maintenance benchmark
(`tools/benchmark-svo-live-maintenance.ts`) already drives
`SparseSceneProxyVoxelizer` over exactly these records, and voxelization *is*
"SDF evals over a grid" — its throughput is a direct proxy. Take that number
before sizing anything in §4.

---

## 3. Shape of the redesign

### 3.1 The mesh is an interface, not an artifact

```ts
interface ShapeLabMesh {
  readonly positions: GPUBuffer;   // vec3f world metres
  readonly normals:   GPUBuffer;   // vec3f, the field's own gradient
  readonly draw:      GPUBuffer;   // indirect args, written by the extractor
  readonly bounds:    ShapeLabBounds;
  readonly provenance: "extracted" | "imported";
}
```

Everything downstream — the raster pane, the voxelizer, the wipe — consumes
`ShapeLabMesh` and knows nothing about where the triangles came from. That is
what makes the later moves cheap: a SpeedTree/Blender oak, alpha-tested foliage
cards, a decimated LOD all arrive as `provenance: "imported"` and the voxel
stage is already written for them. Getting this boundary right is worth more
than any single stage behind it.

### 3.2 Extraction

Reuse the shape of the water pipeline's own polygoniser
(`lib/webgpu-water-pipeline.ts:578`, `polygoniseMain`): **six tetrahedra sharing
cube diagonal 0–6** — no lookup tables, no ambiguous saddle cases, as its own
comment says — with a workgroup vertex accumulator, one atomic block
allocation, and an indirect draw. It is proven in this tree; it builds the fluid
surface every frame. The distance function it evaluates is `svoPrimitiveWGSL`
(`lib/svo-primitive-abi.ts:2715`) plus `svoClusterArenaDecodeWGSL` and
`svoFieldProgramWGSL` — the shipped ABI, so nothing here becomes a second
definition of any shape. That constraint is the existing lab's and it holds.

Four things differ from the water case:

1. **Per record, not per domain.** Smooth-union blending is *within* a cluster
   record and never across records, so extracting each record over its own
   envelope is exact, needs no global acceleration structure, and parallelises
   trivially. The union of shells is correct under opaque shading — sheets
   buried inside another record are occluded.
2. **Screen-space grid sizing.** The load-bearing performance idea. A record's
   resolution comes from its *projected* size, not from a world cell size: the
   0.019 m twig that is 20 px across gets 16³, the trunk gets 128³. Extraction
   re-runs on a parameter edit and on zoom crossing a 2× hysteresis band —
   **never on orbit or pan**. This is what turns §2's numbers from impossible
   into routine.
3. **Block classification first.** One SDF eval at each 4³ block centre, reject
   when `|d| > blockHalfDiagonal`, polygonise only survivors. Same two-level
   shape as `extractMain` → `polygoniseMain`.
4. **A stated budget.** One triangle cap allocated across records by projected
   area, with a readout that says so when it binds. Silent truncation reads as
   "covered everything" when it did not.

Vertex normals are the SDF's own central difference at the vertex — the ABI's
gradient, not a triangle-area average — so the mesh reads as the *form* rather
than as its tessellation. Not a detail: the cluster normal was 55 % of a hero
frame, and a wrong normal has twice been mistaken here for wrong geometry.

### 3.3 Voxelization, fed by the mesh

- **Conservative surface voxelization** of the extracted triangles at
  `shapeLabLeaf_m(depth)`, one dispatch per triangle batch, into a hashed brick
  set. Parity fill for closed records; **authored shell thickness for open
  ones** — which is the card case, and the reason to build it in this direction.
- **Rendered** by a DDA over that brick set in a fragment shader, shaded by the
  face the ray crossed. That is exactly the law `traceShapeLabTile`'s voxel mode
  already states, and it keeps the terracing honest: the hero frame's chunk is
  six-axis face normals, so a voxel pane that shaded smoothly would draw a
  picture the renderer never produces.
- **On demand**, per the ask: a *Voxelize* button plus a staleness badge when
  the mesh has moved under it. Once built, orbit is free.

**And a second source, labelled.** A `from field` toggle that runs the shipped
occupancy predicate instead — `sparseSceneProxyVoxelizationShaderFor()` for the
WGSL, `sampleSparseScenePrimitiveCell` for the CPU mirror. Production is
field-fed today; a mesh-fed lab that could not show the difference would be
asserting the round trip is free rather than measuring it. It also gives the lab
one thing it has never had: two independent packers over one record set, which
is the exact class of bug where a scene draws perfectly and refuses to voxelize.

### 3.4 Layout

Split pane, draggable divider, **one shared camera**:

- **left** — the live stage. Mesh by default; Analytic (the existing CPU trace)
  available as the reference with no resampling in front of it.
- **right** — the derived stage. Voxels, generated on click.
- **wipe** — one image over the other in a single frame. Same pixels, same
  camera. This is the comparison that decides "is this form surviving the
  lattice"; two viewports at slightly different sub-pixel offsets is not.
- Stage chips across the top read `analytic → mesh → voxels`, each with its
  count and its cost, so the pipeline is the UI's own structure.

The specimen list, depth ladder, parameter tree and override export stay exactly
where they are — that is the part of today's lab that works. **Mesh detail is a
new and separate axis from refinement depth** and the UI must not blur them:
depth rebuilds the document because generators derive their legibility floors
from the leaf; mesh detail is a viewer setting that changes no geometry.

---

## 4. Work packages

| WP | content | gate |
|---|---|---|
| **WP0** | Measure GPU field-program eval throughput via the live-maintenance benchmark. | a number in this doc |
| **WP1** | Device acquisition, shared camera, split-pane shell, wipe. Existing CPU trace ported into the left pane **unchanged**. | today's lab, new frame, no regression |
| **WP2** | Extraction: block classify → marching tets → indirect draw → clay raster. Screen-space grid sizing, triangle budget readout. | oak at 60 fps orbit; re-extract < 200 ms |
| **WP3** | Mesh-fed voxelization + brick DDA + face-normal shading; Voxelize button, staleness badge. | voxelize < 300 ms at depth 3 |
| **WP4** | `from field` toggle through the shipped predicate; A/B readout of what the mesh round trip lost. | the two agree, or the disagreement is named |
| **WP5** | Zero-thickness cards: authored shell voxelization. | a card voxelizes to a shell |

WP5 is the one that changes what the renderer can accept. WP1–WP4 are the lab.

---

## 5. Risks

- **GPU tape cost is unmeasured** (WP0). If it is 10× the assumption, §3.2's
  budgets move — the design survives, the resolutions do not.
- **Per-record extraction leaves overlapping shells.** Correct under opaque
  clay; not correct under any later transparency or ambient occlusion.
- **A mesh-fed voxel stage diverges from production**, which is field-fed. That
  divergence is the experiment, but it must be labelled in the UI or the lab
  starts lying about what the renderer will do.
- **Arena headroom is per-specimen, not per-set.** 145 field programs against
  `SPARSE_SCENE_FIELD_PROGRAM_CAPACITY = 256`, and 507 clusters against
  `SPARSE_SCENE_CLUSTER_CAPACITY = 4_096`: fine for the oak alone, thin if the
  lab ever expands the whole set at once.
- **The CPU pool stops being the interactive path.** Keep it — it is written, it
  is the only view with no resampling in front of it, and it is the no-WebGPU
  fallback — but stop treating its latency as something to optimise.
