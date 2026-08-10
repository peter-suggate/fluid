# Hero garden hose scene — recursive-shape reimplementation handoff

## Outcome

Rebuild `hero-garden-hose` around the supplied voxel-garden references, with the
dry scene as the only acceptance lane for this pass. Fluid, the stream, splash,
ripples, refraction and caustics are explicitly out of scope.

The old oak implementation has been deleted. Its replacement is an authored
hierarchy of individually selectable shapes whose frontier can be refined
recursively in Shape Lab; do not restore the oak, rename it, or add a
compatibility adapter.

## Implementation status — complete 2026-08-09

The dry pass described here is implemented. The hero now has a flat exterior,
scene-selected flat voxel face normals, a six-sweep structural tree, eight
authored recursive crown roots and 80 independently addressable active foliage
leaves. Shape Lab supports isolated, parent and assembled contexts plus refine,
collapse, regenerate, duplicate and delete operations. Active foliage leaves
compile to dense bounded voxel-lattice records; recursive parents remain
organizational only.

The final production-path 800×460 GPU render publishes 337 primitives, builds
14,245 derived-lighting pages, reports zero failure-tint and fallback pixels,
and has a 10.23 ms one-sample frame median on the recorded M1 Max run. Fluid
remains intentionally untouched and out of scope.

This handoff supersedes the tree/canopy geometry direction in
`docs/hero-fidelity-1000x-handoff.md`. Its camera, grade, measurement lane and
lighting work remain useful evidence, but the old bonsai/oak form is no longer a
constraint.

## What the references require

Treat the two supplied images as two resolutions of one art direction, not as
two different scenes:

- The pond is the dominant low oval, seen from a high three-quarter camera.
- All terrain outside the pond is a flat warm-ivory plane. The pond wall and lip
  may step in height; the surrounding ground must not swell into landscape
  mounds or terraced beds.
- Geometry reads as cubes. Silhouettes staircase, visible faces are planar and
  axis-aligned, and lighting changes discretely from face to face. A smooth SDF
  silhouette with a voxel texture is not a match.
- The tree is a broad cloud tree, not a botanical oak. Several pale trunks split
  into visible arms, then disappear into a wide, shallow, contiguous crown.
- The coarse reference explains the macro construction: a small number of
  overlapping crown masses. The fine reference explains the recursion: each
  mass resolves into smaller blocky lobes, then terminal cubic clusters. There
  are no individually modeled leaves.
- The crown is denser at the top, irregular at the perimeter, and open enough at
  the underside to reveal the major branches. It must not become a sphere cloud,
  a Romanesco cone field, or a noisy shell full of black one-voxel pits.
- The set is monochrome warm plaster/porcelain except for the green hose and
  brass ferrule. Form and soft contact shadow do most of the separation.
- Lighting is high-key and warm with a broad directional key, bright fill,
  shallow soft shadows, visible contact darkening and a warm seamless backdrop.

Before fidelity work begins, check the supplied target images into a stable
`output/reference/` path or designate an existing checked-in plate as canonical.
Do not build a scoring lane against a temporary clipboard path.

## Non-negotiable architecture

### The saved document owns every editable shape

Recursion may *propose* children, but it may not hide them inside generator
output. Once a shape is refined, its child shapes are written into the scenery
document with stable IDs and ordinary JSON parameters. Shape Lab must be able to
select, isolate, tune, delete and re-seed any child without re-running the rest
of the tree.

That rules out:

- one generator node that emits the entire tree;
- one field-program tape containing the whole crown;
- an array of anonymous branch orders hidden from Shape Lab;
- IDs derived from floating-point values or mutable array positions;
- automatic regeneration of edited descendants when an ancestor changes.

Add one declarative scenery node kind rather than forcing this through an opaque
species interface:

```ts
interface SceneryRecursiveShapeNode extends SceneryNodeBase {
  readonly kind: "recursive-shape";
  readonly family: "foliage-pad";
  readonly seed: number;
  readonly form: FoliagePadForm;
  readonly split: FoliageSplitRule;
  readonly material: SceneryMaterial;
  readonly children?: readonly SceneryRecursiveShapeNode[];
}
```

Its render rule is deliberately simple:

- No children: the node is an active leaf and emits exactly one cluster or
  field-program record compiled from `form`.
- Has children: the node is an organizational/refinement node and emits no
  geometry of its own; its descendants render instead.
- Shape Lab may display a non-rendered parent envelope as a translucent guide.
- Voxel-detail changes never create or remove authored children. Geometry LOD
  and voxel LOD are separate controls.

This replacement semantics avoids double volume and prevents a broad parent
blob from defeating perimeter refinement. It also gives refinement a clean undo:
remove `children` and the parent shape becomes the active leaf again.

Use hierarchical slot IDs such as:

```text
tree
  tree/structure/trunk
  tree/structure/arm-a
  tree/foliage/crown-a
    tree/foliage/crown-a/a
    tree/foliage/crown-a/b
      tree/foliage/crown-a/b/a
```

Child suffixes are allocated once from deterministic slots (`a`, `b`, `c`, …)
and are never renumbered. Reordering siblings must not change identity or GPU
owner order.

### Shape form stays small and art-directable

`FoliagePadForm` should describe one visible crown mass, not the recursion:

```ts
interface FoliagePadForm {
  readonly radii_m: readonly [number, number, number];
  readonly flatten: number;
  readonly edgeLobes: number;
  readonly lobeDepth: number;
  readonly topBias: number;
  readonly undersideCut: number;
  readonly blockJitter: number;
}
```

Compile this to a small, bounded field program or cluster primitive. Keep the
parameters perceptual: width, thickness, perimeter breakup, top fullness and
underside clearance. Do not expose the raw tape as the normal authoring surface.
Every parameter needs a defined range and a visual responsibility; if two
controls move the same silhouette feature, combine them.

`FoliageSplitRule` describes only how the lab proposes the next frontier:

```ts
interface FoliageSplitRule {
  readonly pattern: "fan" | "ring" | "cap";
  readonly childCount: number;       // normally 3–5
  readonly childScale: number;       // normally 0.48–0.68
  readonly spread: number;
  readonly overlap: number;          // target 20–40% of child width
  readonly verticalBias: number;
  readonly flattening: number;
  readonly jitter: number;
}
```

Running **Refine** materializes children from this rule and the node seed. After
that, child transforms and forms are independent authored data. Changing the
parent rule must not silently rewrite them; provide an explicit destructive
**Regenerate children** command with confirmation if that workflow is needed.

### Recursive crown construction

Build the crown top-down:

1. Author 6–10 macro pads at the terminal major branches. They establish the
   whole silhouette and must already read as the reference at thumbnail size.
2. Refine only the pads whose perimeter or underside lacks structure. A normal
   macro pad becomes 3–5 overlapping children at 48–68% scale.
3. Refine selected children once more for the fine reference. Stop at depth 2–3
   or when the child diameter would be less than three presentation voxels.
4. Terminal nodes are compact cubic foliage clusters, not leaves. Their visible
   block size should be approximately the finest presentation leaf.
5. Preserve connectivity: every child overlaps its parent envelope before the
   split and at least one sibling after it. Reject or flag isolated islands.
6. Bias mass upward and outward, keep the underside flatter, and leave deliberate
   negative windows around the primary branch forks. Porosity is authored at
   macro scale, never produced by one-voxel subtraction noise.

The recursive frontier will be uneven by design. Silhouette pads may reach depth
3 while an interior bridge remains at depth 1. Uniform subdivision produces a
procedural texture; selective refinement produces an authored tree.

### Structure is also made of individual shapes

Use explicit tapered-sweep or short swept-tube nodes for the trunk and each major
arm. Begin with one flared trunk splitting into 4–6 upward/outward arms. Secondary
branches exist only where they are visible beneath or between crown pads.

Each structural run gets its own stable ID and local transform. Group them under
`tree/structure`, root the entire tree once at the far-right pond bank, and keep
the branch/crown relationship authored: a crown macro pad names or is grouped
with its supporting branch tip. Do not build a botanical growth simulation.

## Shape Lab work comes first

The current lab groups only top-level nodes by the first ID segment and replaces
only top-level nodes. Implement nested-shape editing before attempting the final
tree; otherwise the new hierarchy will be technically declarative but still
effectively opaque.

Update `lib/shape-lab/specimens.ts` to walk `walkSceneryNodes` and publish a tree
of specimens containing at least:

- `id`, `parentId`, `depth`, `nodePath` and node kind;
- whether the node is rendered, refined or a group;
- child count and stable document order;
- the ancestor transforms needed to reproduce the assembled placement.

Replace the top-level override map with a recursive immutable `replaceNodeAtPath`
operation. Isolation must prune unrelated siblings while retaining the selected
node's ancestor transform chain, terrain anchor, palette and shell. Filtering
catalog spans only by the selected nested node ID is insufficient if it discards
the transforms that place it.

Update `components/ShapeLab.tsx` with:

- a collapsible hierarchy rather than a flat specimen dropdown for recursive
  shapes;
- **Isolated**, **Parent context** and **Assembled tree** preview modes;
- a parent-envelope guide toggle;
- local/world transform editing;
- **Refine**, **Collapse**, **Duplicate**, **Delete**, **Re-seed** and explicit
  **Regenerate children** actions;
- side-by-side coarse and fine voxel previews using the same authored frontier;
- a copyable whole-node JSON patch, including children and stable IDs;
- deep links to nested IDs via the existing `specimen` query parameter.

The normal parameter form may recurse through plain objects, but identity fields
remain hidden. Arrays of actual child nodes belong in the hierarchy, not as a
wall of anonymous numeric controls.

Shape Lab correctness gate: isolating any leaf shape and rendering it with its
ancestor chain must match the pixels and world-space bounds of that same shape in
the assembled tree.

## Scene reimplementation order

### R0 — lock the dry target and baseline

- Select one canonical reference image and register the existing hero camera to
  it. Keep the current `heroGardenCamera` as the first candidate; change it only
  through the existing camera solver plus visual review.
- Capture the current dry frame, proxy count, busiest-brick count, build time,
  GPU memory and fidelity-region scores before further geometry changes.
- Add a mask for pond, exterior ground, tree silhouette, tree interior and hose.
- Keep `water: false` throughout this handoff.

Exit: the same command produces a deterministic reference-sized PNG and scored
regions from a clean checkout.

### R1 — flatten all terrain outside the pond

- In `HERO_GARDEN_VESSEL_AUTHORED`, remove the two raised `terraces` from the
  hero form. Do not merely reduce their height: the target is one plane.
- Keep the pond's plan outline, basin, wall, lip and left-hand shallow entry as
  pond geometry. Outside the outer rim band, `pondVesselHeightAt` must converge
  exactly to `groundHeight_m` plus only the tiny authored surface relief.
- Re-seat or remove props that depended on terrace height. Tree root placement,
  rosettes, stones and hose supports must query the flattened baked terrain.
- Add profile tests sampling radial lines on all four sides. Beyond the rim foot,
  height spread should be at most the relief budget, not a hidden plateau.

Exit: a grazing-light render shows no mound, terrace contour, fault line or slope
outside the pond.

### R2 — add an explicit flat-voxel presentation mode

The reference's cubes are a shading and sampling contract, not extra box props.
Add a scene-level SVO surface style such as `"voxel-flat"` and use it only for
this hero scene initially.

- The primary hit normal is the axis-aligned voxel face normal from traversal.
- Do not substitute analytic/SDF normals, interpolate normals across occupied
  cells, or normal-map the silhouette in this mode.
- Quantization must be stable in world space across camera motion.
- Keep silhouettes stair-stepped and keep each face constant-normal.
- Lighting visibility and GI still use the same occupied voxels; this mode changes
  presentation normals, not collision or ownership.
- Choose a presentation leaf by comparison with both references. Start at the
  existing 6.25 mm authored cell and test 6.25/3.125 mm before changing domain
  scale. Finer is not automatically closer if the visible cubes disappear.

Exit: a normal-debug render contains only six axis directions on dry opaque
surfaces, and a slow camera orbit does not shimmer or re-grid the cubes.

### R3 — land recursive shapes and nested Shape Lab editing

- Add `SceneryRecursiveShapeNode` to `lib/scenery-graph.ts`, validation, cloning,
  node walking and expansion.
- Put pure split/compile logic in
  `lib/voxel-scenery/recursive-foliage.ts`; no scene imports.
- Add the hierarchy, recursive replacement and preview modes to Shape Lab.
- Build a three-level synthetic specimen first. Do not debug schema, editor and
  art direction simultaneously on the hero tree.

Exit: refine/collapse/save/reload preserves stable IDs and pixels; every active
leaf is individually selectable and editable.

### R4 — author the cloud tree in Shape Lab

- Add a new scene-owned `tree` group at approximately the deleted tree slot on
  the far-right bank, leaning gently over the pond.
- Author the trunk and 4–6 major arms first; approve their silhouette without
  foliage.
- Author the 6–10 macro crown pads; approve the thumbnail silhouette.
- Selectively refine the frontier to depth 2–3, starting with the outer contour
  and branch windows, then filling only visibly empty interior regions.
- Export the authored hierarchy into the hero graph. Keep all hero-specific form
  data with the hero scene or a dedicated plain-data module; do not promote it to
  a reusable species until a second scene proves the abstraction.

Exit: the tree reads as a broad contiguous cloud at thumbnail size, exposes its
major branch forks, has no floating foliage islands, and every visible pad can be
opened directly in Shape Lab.

### R5 — recompose dry props and hose

- Compare pond size, lip thickness, stepping-disc path, mushroom forms, plants,
  tree root and hose arc against the registered reference.
- Preserve the hose's green body and brass ferrule as the only strong hue. Keep
  it segmented by the presentation lattice rather than replacing it with a
  staircase of independently authored blocks.
- Remove props that compete with the reference. Do not use extra terrain height
  to fix composition; move the prop.

Exit: masked silhouettes for pond, tree and hose are within the agreed image-space
tolerance before any light or grade retune.

### R6 — lighting fidelity last

The implemented rig in `lib/hero-garden-scene.ts` uses warm key
`[1, 0.925, 0.78]`, intensity `3.2`, environment diffuse `1.22`, specular
`1.05`, ACES exposure `0.2200`, and white balance `[1.22, 1.0, 0.82]`. Geometry
and flat face normals materially changed the light transport from the earlier
baseline.

Tune in this order:

1. key direction from cast-shadow direction;
2. key angular softness/visibility settings from penumbra width;
3. diffuse environment and derived bounce from shadow-floor brightness;
4. contact visibility from trunk/ground and pad/pad contacts;
5. material roughness from highlight width;
6. exposure and white balance with `solve:hero-grade` only after the ratios hold.

Never fix black canopy pits with exposure or fill. They indicate undersized
geometric gaps or excessive contact occlusion. Never fix a geometry silhouette
with shadow softness.

Exit: neutral-region ΔE and gradient scores do not regress from the current
fidelity gate, while the tree-region mean lightness and gradient distribution
move toward the canonical plate.

## File-level implementation map

| Area | Change |
|---|---|
| `lib/scenery-graph.ts` | Add recursive-shape schema, guards, validation and walk semantics |
| `lib/scenery-expand.ts` | Emit only active recursive leaves; preserve composed transforms and owner IDs |
| `lib/voxel-scenery/recursive-foliage.ts` | Pure foliage-pad compiler, deterministic split rules, connectivity/bounds helpers |
| `lib/hero-garden-scene.ts` | Flatten exterior, add authored tree hierarchy, opt into flat-voxel style, final light values |
| `lib/shape-lab/specimens.ts` | Nested specimens, path replacement, isolated subtree expansion |
| `components/ShapeLab.tsx` | Hierarchy UI, context modes, recursive editing commands and deep links |
| SVO primary/shading modules | Scene-controlled axis-face normal path with no smoothing |
| `tests/recursive-foliage.test.ts` | Determinism, split ranges, connectivity, bounds and frontier semantics |
| `tests/shape-lab-recursive-shape.test.ts` | Nested isolation, edits, IDs, save round-trip and coarse/fine parity |
| hero terrain/render tests | Exterior flatness, six-axis normals, counts, bounds and fidelity masks |

## Required invariants and gates

- A recursive shape tree is plain JSON and survives `serializeScene` /
  `parseScene` byte-for-byte in identity and document order.
- Refining node `x` changes only the subtree rooted at `x`.
- Editing an ancestor transform moves descendants without changing descendant
  local data or IDs.
- A refined node emits descendants and never its own parent geometry.
- Every active foliage leaf publishes exactly one bounded shape record.
- All active leaves lie inside the authored tree bound and inside the 1.2 m hero
  container headroom.
- Every foliage connected component touches its intended crown component; no
  isolated island survives validation.
- The tree stays below record, field-program arena and busiest-brick budgets.
  Record the new ceilings from a measured accepted tree; do not inherit an old
  oak number.
- Flat-voxel normal debug shows only the six signed axes.
- Dry render is deterministic at the canonical resolution and registered camera.
- Coarse and fine preview modes preserve macro silhouette. A useful automated
  guard is silhouette IoU between presentation rungs after downsampling; choose
  the threshold from the accepted tree, then pin it.
- Fidelity changes are accepted from contact sheets and regional scores together.
  A scalar score is a guard, not an art director.

## Recommended commit sequence

1. Recursive node schema, expansion and unit tests.
2. Nested Shape Lab hierarchy/isolation/editing.
3. Flat exterior terrain and profile tests.
4. Flat-voxel presentation mode and normal-debug test.
5. Structural tree plus macro crown.
6. Selective foliage refinement and performance gates.
7. Dry prop composition.
8. Lighting/grade retune and final fidelity baseline.

Keep these commits separable. In particular, do not combine the renderer normal
change with the first foliage art pass; otherwise a bad silhouette and a bad
lighting response cannot be diagnosed independently.

## Explicit non-goals

- Fluid and every water effect.
- Restoring or migrating the deleted oak.
- Botanical branch simulation or individual leaves.
- Runtime wind/sway until the static reference passes.
- A general-purpose procedural-tree species before the hero tree is accepted.
- Per-pixel smoothing that hides the voxel faces.
- Recursion that exists only inside a generator or field-program tape.

The completion test is straightforward: a tree-shaped hierarchy is visible in
Shape Lab, any foliage pad can be isolated and refined without changing its
siblings, the dry hero frame has flat exterior terrain and explicit cube faces,
and the registered composition and lighting read as the supplied references.
