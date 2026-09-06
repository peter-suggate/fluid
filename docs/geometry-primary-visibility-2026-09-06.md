# Geometry-derived primary visibility: render ownership

Target: 2× primary-ray throughput at 1920×1080, unchanged visual quality, without scene-specific shortcuts or indirect scheduling.

## Finding

The planar-leaf classifier in `webgpu-svo-sparse-bricks.ts` used every initial SolidWorld physics patch as a visibility blocker. The renderer already excludes some of those patches from both its analytic catalogue and its voxel residual. For example, canonical tank walls presented only as an outline remain physical walls for the fluid solver but have no filled surface in the primary-ray field.

Counting these physics-only patches as visible detail prevents overlapping analytic geometry from becoming planar terminals. This also matters to the primary entry prepass: it already skips planar terminals, but ordinary voxel leaves contribute padded node bounds. Incorrect voxel classification can therefore force residual traversal around an otherwise analytic floor.

The supplied heatmap shows concentrated work near the tank base and broad block-shaped regions. It is a useful before-state, but does not by itself identify traversal cost or prove this mismatch is its dominant cause.

## Change

`svoPlanarSolidWorldBlockers` uses the existing render ownership catalogue. It omits a patch only when the patch is excluded from the voxel residual **and** has no analytic source. Accepted planar patches retain their source indices; visible residuals remain blockers. Terrain classification is unchanged. No scene ID, floor height, camera assumption, shader change, or new dispatch is introduced.

The canonical physics world is unchanged. The existing topology stamp requires a world rebuild when authored geometry affecting planar terminals changes.

## Validation

All 24 tests passed across `svo-planar-boundary`, `svo-render-solid-field`, `svo-primary-reuse-gate`, and `svo-surface-style`. The new behavioral regression checks an analytic floor overlapping physics-only walls, explicit glass, a cut face, a thick solid, and a second analytic surface. It also checks source-index preservation after filtering.

The repository typecheck still reports errors in other solver/harness work; it reports none in the three changed files.

The 1080p before/after GPU benchmark could not start because another CM12 run owned the repository-wide GPU lease. No performance gain or image equivalence is claimed yet. Frozen source copies and the blocked-run log are in `artifacts/geometry-visibility-2026-09-06/`. The before copy uses the original constructor; the after copy uses the corrected classifier.

This is an enabling classification correction, not evidence that the 2× target has been reached. Further measurements should compare primary pass time, G-buffer outputs, planar/voxel terminal counts, and the primary work map under identical cameras.
