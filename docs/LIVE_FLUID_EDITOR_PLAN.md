# Live fluid tools: contextual UX and implementation contract

Status: three shape plugins implemented; backend integration and browser acceptance
are in progress, 2026-09-08.

## Interaction

Use the existing compact Tools chooser and active-tool card. Fluid tools form a
single group alongside the solid editing plugins. Selecting a tool closes the
chooser and enters EDIT; Done returns to navigation. The host renders plugin
metadata rather than branching on a tool name.

The first suite is **Water ball**, **Water cube** and **Water torus**, each with
a **Remove water** toggle for subtracting the same shape from current water. Press and drag preview
one bounded volume; release commits it once. Escape, blur or switching tools
cancels the uncommitted preview. These are direct edits, not continuous emitters.
Hover shows the actual shape and location the operation will use, with the
fluid highlight tone. Ball and torus wire loops are sampled from the same exact
analytic descriptor sent to the solver; the torus opening is visible. The cube
uses that descriptor's box outline.

Add/remove brushes are a follow-up only if the host and backend agree on a
bounded, deduplicated descriptor union. Interpolating a path into repeated
injections must not accidentally add water more than once in overlapping regions
or when the pointer is stationary. Do not count these brushes as shipped before
their operation and preview semantics are implemented and verified.

The primary controls are **Diameter · voxels** for a ball, **Edge · voxels** for
a cube, and **Outer diameter · voxels** / **Tube thickness · voxels** for a torus.
**Remove water** is visible beside these controls; the default operation adds.
For a torus, outer diameter and tube thickness correspond to dimensions visible
in the preview. Major radius is
`(outer diameter - tube thickness) / 2` and minor radius is `tube thickness / 2`.
Require a positive inner opening. Default shapes must span enough simulation
cells to remain recognizable.

Placement uses a frozen horizontal plane. **Height above floor · voxels** is a
visible primary control measuring the shape bottom. Each plugin supplies a
scene-aware default: two cells above the authored tank fill when space permits,
or two cells above the floor for an empty tank. The default is capped so the
whole shape fits beneath the tank top. It follows the authored fill fraction,
not a GPU measurement of the evolved water surface. An explicitly entered
height is preserved.

The generic optional `defaults(scene, resolvedValues)` hook owns this behavior
in the plugin; the shelf and gesture host resolve the same values without
fluid-tool ID branches. Ball and cube default to six cells across; torus defaults
to eight cells across and two cells of tube thickness. Moving water cannot pull
the cursor target. The complete shape must fit inside the tank; outside proposals are
rejected rather than clipped. Backend work and storage limits are also checked
before injection. Unsupported solvers show an availability reason; they do not
silently restart the scene as a fallback.

## Research basis

Houdini exposes bounded volume sourcing and separate add/subtract operations,
including nonnegative subtraction. This supports treating add and remove as
explicit volume operations rather than editing an SDF or a rendering mesh.
[Houdini Volume Source](https://www.sidefx.com/docs/houdini/nodes/dop/volumesource.html)

Blender exposes primitive creation with shape-specific dimensions, including
torus major/minor radii. Our outer-size/thickness proposal translates those
parameters into dimensions visible in the placement preview.
[Blender mesh primitives](https://docs.blender.org/manual/en/latest/modeling/meshes/primitives.html)

These references inform the interaction, not the simulation algorithm. A fluid
volume fraction is not a signed distance field. Injection and removal must obey
the resident solver's mass, occupancy and retained-support conventions.

## Live execution and history

The backend accepts a bounded analytic shape plus add/remove operation, applies
it to the current resident fluid, and publishes through the normal simulation
path. The simulation clock and unrelated flow continue. Preview and operation
share the same descriptor. Capacity and support checks precede mutation; failure
must leave the accepted field unchanged.

A live drop is an event in the running fluid. Once that water has moved, removing
the original source volume does not reverse the drop: it may remove different
water. The present engine has no transported source provenance, so live fluid
events must not advertise exact Undo/Redo. Existing document history retains its
meaning; the fluid tool should explicitly explain that Remove water edits the
current volume. Canceling a preview before release publishes no fluid edit.
After an event has been accepted, cancel does not rewind advected liquid.

Saving stores the authored scene recipe, not a checkpoint of the current fluid.
Unless a separate authored-source path is implemented, transient live drops do
not appear in the saved initial water setup. This distinction must be visible
when using the fluid tools and described in the editing guide. Do not manufacture
an authored-history entry that re-seeds the running solver.

## Acceptance

- Each Ball, Cube and Torus descriptor produces the intended shape; the torus
  center stays empty. Size bounds reject nonfinite or degenerate input.
- Adding to empty cells increases liquid; removing intersects current liquid and
  clamps at zero. Neither operation adds liquid inside a solid collider.
- Preview movement and pointer-up commit exactly one accepted stamp. Escape,
  blur and tool switch before release publish nothing. Any later brush extension
  must also deduplicate its accepted path.
- Rejection leaves mass/state unchanged; all accepted operations retain the
  resident world identity and advance the existing clock without reset.
- Concurrent edits serialize; unavailable/failed workers clear pending UI state.
- The contextual card, preview and operation agree; all three drops work in the
  browser with motion continuing. Brushes require separate acceptance if added.
- Notices accurately distinguish transient fluid events, document history and
  saved initial conditions. No exact advected-water Undo claim is made.

## Verification status

The focused UI/plugin/transaction CPU run passed 34 tests. A separate combined
backend run passed 14 tests; these are separate run counts and must not be added
together because their coverage can overlap. Native shape acceptance passes
in `tests/live-fluid-shapes-dawn.test.ts`: all three add/remove pairs, torus
opening, solid exclusion, atomic bounds rejection and continued clock are checked.
Browser acceptance remains pending: empty-world renderer publication and
subsequent solid insertion exposed separate issues recorded in
[the acceptance log](VOXEL_EDITOR_BROWSER_QA.md).

The concurrent solid-tool browser blocker was traced to GPU floating-point
complement arithmetic: `1 - f32(255) / 255` produced a tiny negative open fraction
for a fully closed cell. Taking the integer complement before conversion fixes
that arithmetic; all 256 q8 fractions and the exact 149-patch replay pass native
checks. Strict failure checks remain in place. This does not establish browser
acceptance or displaced-water conservation.
