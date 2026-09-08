# Live voxel editor

Choose **Tools** from the compact strip beside the viewport. Selecting a tool closes the chooser and enters EDIT mode. Hover over the viewport to see its proposed bounds; hold the primary mouse button and drag to edit. Accepted samples change solid geometry while the fluid continues running. Releasing completes one history entry.

For water scenes, use **Sparse CM12**. A new scene starts as a dry room with its fluid system disabled; wait for its live scene resources to become ready before stamping. New/Open/Import replace the document and restart its initial setup. They are scene-loading operations, not live strokes.

## Choose a tool

| Tool | Gesture | Result |
| --- | --- | --- |
| **Build** | Paint while dragging | Accumulates solid stamps along your path. |
| **Carve** | Paint while dragging | Accumulates cuts along your path. Depth goes into the starting face. |
| **Box** | Drag out a rectangle | Extrudes its footprint by Depth. Drag back to shrink it before release. |
| **Cut** | Drag out a rectangle | Removes its footprint through Depth. Drag back to shrink the cut before release. |
| **Sphere** | Click to place; drag to reposition | Adds one voxel sphere. Dragging moves the proposed sphere; it does not paint a trail. Width controls its diameter in cells. |
| **Drill** | Click to place; drag to reposition | Cuts one cylinder along the starting face normal. Width controls diameter; Depth controls length. |
| **Wall** | Drag from start to end | Adds a straight wall with Width and Depth as its cross-section. Moving the endpoint replaces the current proposal. |
| **Channel** | Drag from start to end | Cuts a straight channel with Width and Depth as its cross-section. Moving the endpoint replaces the current proposal. |

The active-tool card shows primary settings. Open **More** for targeting, construction height and symmetry; badges keep nondefault options visible when More is closed. **Done** finishes tool use. Closing the chooser leaves the current tool armed. Tool settings are captured when a stroke starts, so change settings between strokes.

## Aim and size the edit

- **Solid faces:** the face under the initial press determines the editing plane for the entire stroke. Build operations extend outward; subtraction goes inward. This works on all six face directions. Dragging onto a different face does not change the plane halfway through the stroke.
- **Empty space:** when no editable solid is hit, the ray meets a horizontal plane at **Empty-space height · voxels**. Height zero is the container floor level. Negative values work below it. A ray parallel to the plane, or looking away from it, cannot place an edit there.
- **Edit tank walls:** off by default, so the front walls do not intercept attempts to work inside the tank. The floor remains a usable target. Turn this on to explicitly target and alter authored tank-wall voxels.
- **Width · voxels:** 1–16 cells, for brushes, round stamps and lines. A width of one is a single-cell footprint.
- **Depth · voxels:** 1–32 cells, perpendicular to the starting face. Sphere uses its Width on all three axes.
- **Empty-space height · voxels:** −64 to 128 cells. This setting applies only when the initial ray does not hit an editable solid.
- **Mirror X:** applies the complete stroke and its reflection across the container's central X plane. It mirrors construction and subtraction alike.

Measurements are integer lattice cells, not metres. Physical cell size belongs to the scene. Round tools produce voxel approximations; they do not create infinitely smooth surfaces.

## Navigate, cancel and undo

Hold **Shift before pressing**, then drag to **pan** without starting a voxel stroke. Middle-button dragging also pans. For orbiting, switch to **LOOK** with the mode control (or Tab when focus is outside a text field), then primary-button drag. Use the wheel to zoom. Pressing Shift after a voxel stroke has begun does not convert that stroke into navigation.

**Escape** or **Ctrl/⌘ Z during a stroke** cancels it and attempts to restore the pre-stroke solids. Losing pointer capture or leaving the browser window also cancels. Changing tools or leaving EDIT commits the accepted stroke. Once a stroke finishes, Undo restores its previous document and Redo reapplies it; voxel-only history changes in Sparse CM12 preserve the running timeline.

Restoring solid can be rejected if water has entered a newly carved space.
The accepted edit stays in history, and a notice explains the blocked restoration.
Remove water from that space before retrying.

While **Applying stroke…** is visible, the final update may still be awaiting validation. Save, Export, Undo and Redo wait until it finishes. If validation rejects a sample, the last accepted geometry remains. If the runtime fails, that accepted document remains available to save once the pending stroke settles.

## Create and keep scenes

1. Open **Scene**, then choose **New scene** for an empty room, or **Open scene** for a preset or browser-saved scene. In the chooser, search by name and press Enter, or click a tile.
2. Edit, enter a **Scene name**, and select **Save scene**. Saves belong to this browser's local scene library. Saving the same name replaces that named entry.
3. Use **Export JSON** for a portable file. **Import JSON** validates a chosen file and replaces the current document; an invalid file leaves the document and history unchanged.

A saved/exported scene contains authored voxel edits and the scene's initial water setup. It is not a checkpoint of the current fluid motion, solver memory, or undo history. Reopening starts from that authored setup. JSON contains the scene document; the active solver/quality controls are not part of this file. Use a named save or exported file to retain new work rather than relying on a starter's URL.

The browser library stores compact documents and shares identical named-save
and autosave payloads. Existing saved documents remain readable. The library
holds at most 64 entries, including its working autosave. If that limit or the
browser's storage quota prevents a save, Save reports failure and preserves
existing saves; it never evicts an older scene to make room. Saving over an
existing name remains available when the entry limit is reached. Export JSON
keeps a portable copy. Browser verification of the large-scene save/reopen fix
is still pending.

## Fluid stamps

The **Fluid** group adds **Water ball**, **Water cube** and **Water torus**. Water
must already be enabled with a ready Sparse CM12 solver. These tools are
transient edits to the moving fluid, separate from the solid tools above.

A new scene starts with water disabled. Choose **Scene → Enable water** to
initialize fluid from the scene's starting setup. The action appears only in
dry scenes. Enabling water leaves playback paused at time zero; wait for
readiness, then use Play for moving water. This is initial setup, separate from
the subsequent live stamp operations.

Choose a shape, adjust its diameter/edge in voxels, and drag its outline into
position. Release to apply it once. The torus exposes outer diameter and tube
thickness; its horizontal ring keeps a visible opening. The preview uses the
fluid highlight colour and the same shape descriptor as the operation.

**Height above floor · voxels** stays visible and sets the shape's bottom. Its
default places the drop two cells above the authored tank fill when space
permits, capped so the whole shape fits inside the tank. Empty tanks start at
two cells above the floor. An explicitly entered height is preserved. The
default follows the initial fill, so adjust it if the running water has moved.
**Remove water** subtracts current water within the same shape.

Escape, losing focus or changing tools before release cancels the preview and
adds no water. After release, accepted liquid joins the current flow. It is not
an exact Undo/Redo operation: water may have moved away from where it was added.
The document's existing Undo history remains separate. Live fluid stamps are
not included in Save/Export's initial water setup and do not become continuous
emitters.

## Scene and object controls

When no voxel tool is active, **Scene settings** and selected-object settings start collapsed. Select an oak to reveal **Tree settings**; use **Tools → Add tree** to create one at the view centre. Changing the selection closes the old object's details.

In EDIT, selectable objects under the pointer show faint dotted selection bounds.
An idle tool preview can coexist with those bounds; an active stroke owns the
highlight. LOOK hides the hover bounds.

## Limits and rejected edits

Voxel tools also target terrain-backed solids. Terrain heights remain the original authored surface; fill and clear patches form a bounded live overlay, including Undo. Editing the terrain recipe or changing its lattice remains a separate scene rebuild. Fluid-enabled methods other than Sparse CM12 disable the tools. The renderer and solver must both be ready to accept a sample.

A proposal is limited to **32,768 voxel operations** and **4,096 patches**. Mirrored and overlapping patches count toward the work budget; the limit is not simply the final number of unique occupied voxels. Interpolation across a brush jump or straight-line stroke is limited to **256 cells** on its longest axis. Fixed GPU page, region and live-solid capacities also apply and may reject an edit before these brush limits are reached.

Adding solid into occupied water is rejected before publication because static editing does not yet conservatively displace that water. Remove water from the intended space, or place the solid in an empty area. Clearing solid remains available. The GPU checks current occupancy and applies accepted geometry in one ordered transaction while simulation continues. The document follows its asynchronous acceptance receipt.

Rejected proposals are not silently clipped. Reduce the width/depth or span, or release and continue with another stroke. These tools add and subtract voxel solids; they do not currently offer smoothing, arbitrary rotations, or moving existing voxel selections as objects.

## Hero garden availability

`hero-garden-hose-x10` is intentionally a dry rendering stress scene. Its fluid
tools become usable after **Scene → Enable water** initializes a supported
solver. The former blanket terrain guard also disabled all eight solid tools;
the live terrain-overlay implementation removes that restriction without
rebaking terrain heights. Native checks verify fill, deep carve and Undo against
GPU occupancy while retaining the refined terrain source. Browser checks also
verify Box fill, Undo/Redo and deep Cut with Undo on this scene.

The first pointer hover previously regenerated the scenery catalogue on the UI
thread. The worker now transfers its existing catalogue, and the picker adopts
it before probing objects. Procedural terrain picking samples the ray locally
instead of baking the heightfield, and tank-shell picking uses direct predicates.
Fresh first-drag browser checks pass in both garden variants. Fluid add/remove
and large-scene save/reopen also pass. See the
[browser QA record](VOXEL_EDITOR_BROWSER_QA.md) for the current evidence.


## Regression checks

Run `npm run test:dawn:voxel-editor` for the repeatable production-path editor
matrix. It exercises all plugins, atomic acceptance, history/cancellation,
fluid insertion/removal and surface clearing. Keep the required
`npm run test:dawn:sparse-cm12` simulation gate as well. Run Dawn sequentially,
with GPU browser tabs closed. See [acceptance coverage](VOXEL_EDITOR_ACCEPTANCE_COVERAGE.md)
for exact assertions, current results, and the browser checks still needed for
pointer routing and presentation.
