"use client";

import { EditorControlGroupRows } from "./EntityOptions";
import { isEditableOak, oakTreeControlGroups } from "../lib/core/oak-tree-controls";
import { sceneSceneryGraph } from "../lib/core/scenery-edit";
import { sceneryIdFromSelection, scenerySelectionId } from "../lib/core/editor-scenery";
import { useSession } from "../lib/core/session/session-context";
import { simulation } from "../lib/core/simulation/controller";
import { terrainHeightAt } from "../lib/core/terrain";
import { Field, Select } from "./ui";

// The tree editor also works with sessions predating asynchronous voxel strokes.
const voxelStrokePending = (state: object) => "voxelStrokePending" in state && state.voxelStrokePending === true;

/** Tree growth is a scenery edit, sharing selection, history, and scene persistence. */
export function OakTreeEditor({ contextual = false }: { contextual?: boolean }) {
  const session = useSession();
  const scene = session.scene(state => state.scene);
  const selection = session.ui(state => state.selection);
  const pending = session.ui(voxelStrokePending);
  const trees = sceneSceneryGraph(scene).nodes.filter(isEditableOak);
  const id = selection?.kind === "scenery" ? sceneryIdFromSelection(selection.id) : undefined;
  const selected = trees.find(tree => tree.id === id);
  const groups = selected ? oakTreeControlGroups(scene, selected.id) : [];
  return <section className="oak-tree-editor" aria-label="Fractal oak editor">
    {!contextual && <><div className="oak-tree-heading"><strong>Fractal oaks</strong>
      <button type="button" disabled={pending} onClick={() => {
        session.ui.getState().setVoxelTool(undefined);
        const { x, z } = session.ui.getState().camera.target_m;
        simulation.addScenery("oak-v2", { x, y: terrainHeightAt(scene.terrain, x, z), z }, { x: 0, y: 1, z: 0 }, session.id);
      }}>Add oak</button>
    </div>
    <p className="voxel-tool-help">Add at the view centre, then move with the object handles. Use Prop → Fractal oak to plant at a picked surface.</p>
    {trees.length > 0 && <Field label="Tree" disabled={pending}>
      <Select ariaLabel="Tree to edit" value={selected?.id ?? ""} disabled={pending}
        options={[{ value: "", label: "Select an oak…" }, ...trees.map(tree => ({ value: tree.id, label: tree.id }))]}
        onChange={value => {
          session.ui.getState().setVoxelTool(undefined);
          session.ui.getState().select(value ? { kind: "scenery", id: scenerySelectionId(value) } : undefined);
        }} />
    </Field>}</>}
    {selected && <EditorControlGroupRows key={selected.id} groups={groups} entityLabel="Oak" />}
  </section>;
}
