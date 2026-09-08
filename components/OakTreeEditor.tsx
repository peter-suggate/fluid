"use client";

import { useId, useState } from "react";
import type { EditorField } from "../lib/core/editor-entity";
import { normalizeControlNumber } from "../lib/framework/controls";
import { isEditableOak, oakTreeControlGroups, withTreePreviewDepth } from "../lib/core/oak-tree-controls";
import { sceneSceneryGraph } from "../lib/core/scenery-edit";
import { sceneryIdFromSelection, scenerySelectionId } from "../lib/core/editor-scenery";
import { useSession } from "../lib/core/session/session-context";
import { simulation } from "../lib/core/simulation/controller";
import { terrainHeightAt } from "../lib/core/terrain";

// The tree editor also works with sessions predating asynchronous voxel strokes.
const voxelStrokePending = (state: object) => "voxelStrokePending" in state && state.voxelStrokePending === true;

function TreeField({ field, commit }: { field: EditorField; commit: (value: number) => void }) {
  const inputId = useId();
  const [draft, setDraft] = useState(String(field.value));
  const accept = () => {
    const value = draft.trim() ? Number(draft) : field.value;
    const next = normalizeControlNumber(value, field.value, field, ["oak-seed", "oak-twigDepth", "oak-boughsPerTier", "oak-sitesPerBough"].includes(field.id));
    setDraft(String(next));
    if (next !== field.value) commit(next);
  };
  return <div className="oak-tree-field" title={field.hint}>
    <label htmlFor={inputId}>{field.label}{field.unit && <span>{field.unit}</span>}</label>
    <div className="oak-tree-inputs">
      {field.id !== "oak-seed" && <input type="range" aria-label={`${field.label} slider`}
        min={field.min} max={field.max} step={field.step} value={Number(draft) || field.min}
        onChange={event => setDraft(event.target.value)} onPointerUp={accept} onKeyUp={accept} onBlur={accept} />}
      <input id={inputId} aria-label={field.label} type="number" min={field.min} max={field.max} step={field.step}
        value={draft} onChange={event => setDraft(event.target.value)} onBlur={accept}
        onKeyDown={event => {
          if (event.key === "Enter") event.currentTarget.blur();
          if (event.key === "Escape") { setDraft(String(field.value)); event.stopPropagation(); }
        }} />
    </div>
  </div>;
}

/** Tree growth is a scenery edit, sharing selection, history, and scene persistence. */
export function OakTreeEditor() {
  const session = useSession();
  const scene = session.scene(state => state.scene);
  const selection = session.ui(state => state.selection);
  const pending = session.ui(voxelStrokePending);
  const trees = sceneSceneryGraph(scene).nodes.filter(isEditableOak);
  const id = selection?.kind === "scenery" ? sceneryIdFromSelection(selection.id) : undefined;
  const selected = trees.find(tree => tree.id === id);
  const groups = selected ? oakTreeControlGroups(scene, selected.id) : [];
  const fluidEnabled = scene.systems?.fluid !== false;
  const previewDepth = Math.round(Math.log2(scene.voxelDomain.finestCellSize_m / (scene.voxelDomain.detailCellSize_m ?? scene.voxelDomain.finestCellSize_m)));
  const commit = (label: string, apply: () => Partial<typeof scene>) => {
    if (voxelStrokePending(session.ui.getState())) return;
    const patch = apply();
    simulation.beginEdit(label, session.id);
    simulation.commitEdit(patch, { reseed: true }, session.id);
  };
  return <section className="oak-tree-editor" aria-label="Fractal oak editor">
    <div className="oak-tree-heading"><strong>Fractal oaks</strong>
      <button type="button" disabled={pending} onClick={() => {
        session.ui.getState().setVoxelTool(undefined);
        const { x, z } = session.ui.getState().camera.target_m;
        simulation.addScenery("oak-v2", { x, y: terrainHeightAt(scene.terrain, x, z), z }, { x: 0, y: 1, z: 0 }, session.id);
      }}>Add oak</button>
    </div>
    <p className="voxel-tool-help">Add at the view centre, then move with the object handles. Use Prop → Fractal oak to plant at a picked surface.</p>
    {trees.length > 0 && <label>Tree<select aria-label="Tree to edit" value={selected?.id ?? ""} disabled={pending}
      onChange={event => {
        session.ui.getState().setVoxelTool(undefined);
        session.ui.getState().select(event.target.value ? { kind: "scenery", id: scenerySelectionId(event.target.value) } : undefined);
      }}>
      <option value="">Select an oak…</option>
      {trees.map(tree => <option key={tree.id} value={tree.id}>{tree.id}</option>)}
    </select></label>}
    {selected && <div key={selected.id}>
      <details>
        <summary>Voxel comparison</summary>
        <p className="voxel-tool-help">Compare the current geometry at depths 0–3. Applies to the whole scene and keeps your tree edits.</p>
        {fluidEnabled && <button type="button" disabled={pending} onClick={() => simulation.setFluidSystem(false, session.id)}>Turn water off to compare</button>}
        <div className="voxel-tool-buttons">{[0, 1, 2, 3].map(depth => <button type="button" key={depth}
          aria-label={`Voxel depth ${depth}`} aria-pressed={!fluidEnabled && previewDepth === depth} disabled={pending || fluidEnabled}
          title={`${(scene.voxelDomain.finestCellSize_m * 1000 / 2 ** depth).toFixed(3)} mm voxels`}
          onClick={() => commit(`Set scenery voxel depth ${depth}`, () => withTreePreviewDepth(scene, depth))}>{depth}</button>)}</div>
        <p className="voxel-tool-help">{(scene.voxelDomain.finestCellSize_m * 1000 / 2 ** (fluidEnabled ? 0 : previewDepth)).toFixed(3)} mm voxels. Finer depths use more GPU memory. This is independent of fork generations.</p>
      </details>
      {groups.map(group => <details key={group.id} open={group.id === "oak-specimen" ? true : undefined}>
        <summary>{group.label}</summary>
        <fieldset disabled={pending}>
          {group.choices?.map(choice => <div key={choice.id} className="oak-tree-choice">
            <span>{choice.label}</span><div className="voxel-tool-buttons">
              {choice.options.map(option => <button type="button" key={option.id} aria-pressed={choice.value === option.id} title={option.hint}
                onClick={() => commit(`Set ${selected.id} ${choice.label}`, option.apply)}>{option.label}</button>)}
            </div>
          </div>)}
          {group.fields?.map(field => <TreeField key={`${field.id}:${field.value}`} field={field}
            commit={value => commit(`Set ${selected.id} ${field.label}`, () => field.apply(value))} />)}
        </fieldset>
        {group.summary && <p className="voxel-tool-help">{group.summary}</p>}
      </details>)}
    </div>}
  </section>;
}
