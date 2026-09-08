# Oak v2: recursive branches and twig-scale foliage

This supersedes the canopy-form direction in the historical
`hero-garden-recursive-foliage-handoff.md`. The hero scene retains its placement,
porcelain materials, editable scenery nodes, and existing analytic primitive ABI.

## Construction

`lib/core/voxel-scenery/oak-v2.ts` builds a deterministic, crown-guided branch
hierarchy. Eight boughs distribute 96 shoot sites through a broad, rising crown.
A spatial binary partition routes branches toward their descendant sites.
Each site then branches three more times: fork planes rotate between generations,
lengths shrink by 0.62, and foliage occurs only at the 768 terminal tips. This is
a geometric fractal construction, not a biological growth simulation or a claim
that one mathematical tree is the uniquely correct oak.

The planner publishes 1,532 wood sweeps and 768 small foliage fields. All 2,300
shapes have stable editable IDs and serialize as ordinary scenery nodes. There
is no canopy-sized rendered envelope. A versioned `oak` recipe is saved alongside
the ordinary children; rendering reads those children, and only explicit growth
edits regenerate them.
The existing `heroGardenCloudTree` scene entry point now places this v2 specimen.

At a fork, child radii satisfy

```
r_child_base = r_parent_tip * sqrt(child_terminal_count / parent_terminal_count)
sum(r_child_base²) = r_parent_tip²
```

Wood tapers between forks. Every child begins at the exact parent endpoint;
round-cone sweeps share occupied endpoint spheres. Sweep envelopes include all
control spheres and the smooth-union allowance. At default scale the thinnest
wood radius is at least 1 mm, larger than half the depth-3 voxel diagonal
(0.677 mm). Thus a closest voxel-centre sample exists inside that wood core.
This is a sampling guarantee, not a promise that every twig covers a screen pixel.

Foliage uses the existing bounded two-frequency density field, at the size of a
small twig spray. The seed phase is selected with a bounded deterministic search
so each terminal point is *inside occupied density*, rather than merely inside
an empty acceleration envelope. Noise supplies fine breakup; recursive branches
and separated sprays supply the larger gaps. The ABI's distance lower bound,
normal computation, and GPU code are unchanged.

This approach draws on crown-guided tree construction and the pipe-model radius
assignment described by [Runions, Lane and Prusinkiewicz (2007)](https://algorithmicbotany.org/papers/colonization.egwnp2007.html).
It uses deterministic spatial partitioning rather than their space-colonization
growth algorithm. The broad crown is consistent with the [white oak form](https://www.arborday.org/perspectives/featured-tree-white-oak).

## Refinement and editing

Voxel depth and twig depth are independent. The shipped geometry is identical
at voxel depths 0–3 (6.25, 3.125, 1.5625, 0.78125 mm on the hero lattice).
`twigDepth` is a bounded planner parameter, 0–3, useful for studying the geometric
recursion. Increasing voxel depth does not add or remove document shapes.

Open the voxel tool chooser and choose **Add tree** to plant at the view centre.
Select an oak and open its contextual **Tree settings** to experiment with
growth. The surface action **Prop → Fractal oak** plants at the picked point.
Move and scale trees with the existing object handles. The contextual editor
and generic object inspector consume the same growth control declarations.

Twenty numeric controls are grouped into Specimen, Crown, Branches, Twigs, and
Foliage. Four growth presets (Fractal oak, Open crown, Spreading oak, Fine tracery),
repeatable seed changes, Natural/Clay materials, and Leaves/Branches views provide
starting points for comparison. The default recipe reproduces the v2 specimen.
Presets retain seed, size, placement, colour and the current Leaves/Branches view.
Integer controls are bounded; the largest recipe emits 3,448 primitives.

Sliders preview their value locally and regenerate on release. Each committed
change uses ordinary scene history. Save scene and Export JSON include recipe,
seed, materials, transforms and geometry; import does not regenerate the tree.
Explicit growth edits replace generated children, including any hand-sculpted
foliage, and undo restores them. Colour edits recolour existing children without
regenerating. These trees are decorative scenery and do not block water; growth
edits do not reset the running solver.

**Voxel comparison** changes the scene's sampling depth 0–3 while retaining the
authored tree, terrain and voxel edits. It requires water off, following the
renderer’s existing refinement constraint. Unlike the preset's **Re-author at**
control, it does not call the scene factory. The panel reports the requested
voxel pitch; finer samples need more GPU memory.

Scenery is part of baked planar topology, so regeneration requests a replacement
display world instead of staging incompatible bounds into the old one. Sparse
CM12 keeps its fluid solver while its display sidecar is replaced. Failed or
superseded sidecar candidates cannot destroy the retained solver. Ordinary voxel
strokes and fluid uniform edits keep the same scenery construction key.
Superseded initialization publishes a cancellation event that retires only its
resource activity, preserving the active runtime status and usable generation.
It does not trigger the simulation's terminal stop path.

The legacy three-dial canopy URL projection is omitted/ignored for recipe-bearing
oaks because it cannot round-trip independently adjustable density settings.
Legacy/static pads retain their old curves. Shape Lab's foliage-only refinement
inherits the v2 tags; it remains a foliage edit, not a woody growth operation.
Old saved v2 groups without a recipe remain ordinary editable scenery: growth
settings cannot be reliably recovered from their baked geometry.

## Reproducible checks

```
node --import tsx --test tests/oak-*.test.ts
npm run check:scenery
node --import tsx tools/check-oak-voxel-convergence.ts
node --import tsx tools/render-oak-voxels.ts
node --import tsx tools/check-oak-render.ts
```

The unit tests check multiple seeds, tree connectivity, fork area conservation,
valid sweep envelopes, occupied foliage attachment, deterministic JSON and IDs,
finite scale/depth inputs, a bounded record count, positive taper, resolvable wood
cores, geometric recursion, canopy query round-trips, and Shape Lab dial scale.
The editor tests additionally exercise every parameter endpoint and combined
extremes, all presets, primitive budgets, recipe validation, save/import,
transform/material preservation, generic inspector declarations, and undo/redo.
The renderer/history lifecycle tests cover sidecar replacement, rejection and
supersession, retain the fluid solver, and verify that undo keeps the selected
tree and running timeline.

The editor/lifecycle validation run passes 43 focused tests and all 11 scenery
checks. The broader Sparse CM12 Dawn gate remains incomplete: its latest shared
run passed five lanes, timed out six, and left six unrun at the unchanged
180-second budget. No numerical assertion failed in that run. Full scene
regeneration can still take several seconds; cancellation safety does not make
voxelization instantaneous. Evidence is in `artifacts/oak-v2/editor/`.
The integrated browser pass verifies Tools → Add tree, contextual selection and
Tree settings, and rapid fork-depth 3 → 2 → Undo 3 → Redo 2 without a terminal
halt. The full production build also passes; the project-wide typecheck still
has unrelated errors recorded in the evidence directory.

`check-oak-voxel-convergence.ts` independently integrates the field sign at six
spatially separated foliage samples, using a half-cell reference beyond depth 3.
`render-oak-voxels.ts` produces independent centre-sampled front-view voxel
images. Neither substitutes for GPU conservative coverage, screen-space LOD,
shadows or GI.

`check-oak-render.ts` waits for the repository GPU lease, then renders all four
depths serially through the production dry-scene path. It never removes another
owner's lock. Reports and full-rate reference images are under
`artifacts/oak-v2/production-depth*/`. The existing benchmark also renders its
half-rate lighting comparison; use `reference.png` when assessing the tree.

For an analytic geometry view:

```
OAK_COLOR=1 node --import tsx tools/render-oak-analytic.ts
OAK_COLOR=1 OAK_BARE=1 OAK_ANALYTIC_OUT=artifacts/oak-v2/skeleton.png node --import tsx tools/render-oak-analytic.ts
```

This uses the ABI's CPU ray-intersection mirror and its bounded march, not a
photorealistic renderer. Use the independent occupancy integration to assess
sampling; do not infer exact occupied volume from a bounded ray-march image.

The old seven-shape baseline is retained as
`tools/preview/fixtures/oak-v1.json`. Set `OAK_TREE_JSON` to that path for either
`render-oak-analytic.ts` or the production `tools/preview/oak-v2.ts` scene module.

## Validation results (2026-09-08)

The final hero placement uses `scale_m: 0.85` to retain the previous specimen's
approximate footprint and fit the existing scene camera. All 13 targeted tests
and all 11 `check:scenery` subjects pass. The project-wide typecheck still reports
unrelated errors elsewhere in this working tree; no oak, canopy-control, or
new-tool errors remain.

Production Dawn/Metal, Apple M1 Max, 800×650, two warmups and five measured cycles:

| Voxel depth | Cell size | GPU median | Reported scene allocation |
| --- | --- | --- | --- |
| 0 | 6.25 mm | 14.88 ms | 175.9 MB |
| 1 | 3.125 mm | 18.55 ms | 250.7 MB |
| 2 | 1.5625 mm | 21.43 ms | 556.8 MB |
| 3 | 0.78125 mm | 28.70 ms | 1,684.2 MB |

These are whole dry-garden costs with the benchmark's half-rate cone lighting,
not isolated tree costs or a fluid-simulation benchmark. The saved reference
images use full-rate lighting. All four reports have zero failure-tint pixels,
zero black radiance pages, and the expected built maximum depth. Reconstruction
fallback is nonzero (approximately 1.3–7%); it is not a failure-tint result and
has not been hidden. No renderer budgets or timing ceilings were changed.

At the final 0.85 scale, six foliage-volume probes have relative midpoint
integration errors of 5.93%, 1.02%, 0.111%, and 0.0589% at depths 0–3 against the
0.390625 mm reference. This certifies convergence for those sampled fields, not
all possible seeds, viewing directions, or voxels. The CPU branch/foliage views
show a recognizable open-crowned tree and multiscale forks; visual acceptance
remains an aesthetic judgment rather than a proof of botanical realism.

Depth 2 is a useful interactive tradeoff: the recursive structure is present at
every depth, while depth 3 mostly resolves finer edges at a substantial memory
cost. A future optimization should preserve the accepted geometry and compare
actual production images; do not collapse the crown back into large envelopes.

The saved old-tree baseline was also rendered at depths 0 and 3 with the same
camera, viewport, lighting and measurement settings:

| Depth | Old tree GPU median | V2 GPU median | Old allocation | V2 allocation |
| --- | --- | --- | --- | --- |
| 0 | 17.24 ms | 14.88 ms | 194.1 MB | 175.9 MB |
| 3 | 28.31 ms | 28.70 ms | 1,689.0 MB | 1,684.2 MB |

The depth-3 difference is 0.39 ms (about 1.4%) in these short, separate runs;
it is not a statistically established speed difference. The large depth-3
allocation is also present with the old tree. V2 therefore does not introduce
a material allocation increase in this scene despite the larger primitive count.
Both baseline runs also have zero failure tints and black radiance pages.

To repeat this comparison after the v2 matrix:

```
OAK_TREE_JSON=tools/preview/fixtures/oak-v1.json OAK_RENDER_DEPTHS=0,3 OAK_RENDER_OUT_PREFIX=artifacts/oak-v2/baseline node --import tsx tools/check-oak-render.ts
```
