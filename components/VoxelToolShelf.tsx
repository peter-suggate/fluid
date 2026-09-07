"use client";

import { useState } from "react";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { toolValues } from "../lib/core/voxel-editor/plugin";
import { useSession } from "../lib/core/session/session-context";
import { simulation } from "../lib/core/simulation/controller";
import { serializeScene } from "../lib/core/model";

/** All tool-specific UI comes from the same plugin that implements its stroke. */
export function VoxelToolShelf() {
  const session = useSession();
  const id = session.ui((state) => state.voxelToolId);
  const stored = session.ui((state) => state.voxelToolValues);
  const method = session.method((state) => state.methodId);
  const plugin = voxelTools.get(id);
  const values = plugin ? toolValues(plugin, stored[plugin.id]) : {};
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("Voxel scene");
  const exportScene = () => {
    const url = URL.createObjectURL(new Blob([serializeScene(session.scene.getState().scene)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = `${name.trim().replace(/[^a-z0-9-]/gi, "-") || "voxel-scene"}.json`;
    anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <aside className="voxel-tool-shelf" aria-label="Voxel editing">
    <button type="button" aria-expanded={open} onClick={() => {
      setOpen(!open); if (open) session.ui.getState().setVoxelTool(undefined);
    }}>Voxels {open ? "−" : "+"}</button>
    {open && <div className="voxel-tool-panel">
      {method !== "adaptive-mass" && <p>Choose Sparse CM12 to edit solids while water runs.</p>}
      {[...new Set(voxelTools.tools.map((tool) => tool.ui.group))].map((group) => <fieldset key={group}>
        <legend>{group}</legend>
        <div className="voxel-tool-buttons">{voxelTools.tools.filter((tool) => tool.ui.group === group).map((tool) =>
          <button type="button" key={tool.id} aria-pressed={id === tool.id} title={tool.ui.hint}
            disabled={method !== "adaptive-mass"} onClick={() => session.ui.getState().setVoxelTool(id === tool.id ? undefined : tool.id)}>
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden><path d={tool.ui.icon} /></svg>
            {tool.ui.label}
          </button>)}</div>
      </fieldset>)}
      {plugin && <>
        <p>{plugin.ui.hint}</p>
        {plugin.ui.controls.map((control) => <label key={control.id}>{control.label}
          <input type="number" min={control.min} max={control.max} step={control.step} value={values[control.id]}
            onChange={(event) => session.ui.getState().setVoxelToolValue(plugin.id, control.id, event.target.valueAsNumber)} />
        </label>)}
        <p className="voxel-tool-help">Shift-drag to navigate · Escape cancels the stroke. Water keeps moving.</p>
      </>}
      <div className="voxel-tool-buttons">
        <button type="button" onClick={() => simulation.undo(session.id)}>Undo</button>
        <button type="button" onClick={() => simulation.redo(session.id)}>Redo</button>
      </div>
      <label>Scene name<input value={name} maxLength={80} onChange={(event) => setName(event.target.value)} /></label>
      <div className="voxel-tool-buttons">
        <button type="button" onClick={() => simulation.saveNamedScene(name, session.id)}>Save scene</button>
        <button type="button" onClick={exportScene}>Export JSON</button>
      </div>
      <p className="voxel-tool-help">Saved scenes reopen from their initial water setup.</p>
    </div>}
  </aside>;
}
