"use client";

import { useId, useRef, useState } from "react";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { toolValues } from "../lib/core/voxel-editor/plugin";
import { useSession } from "../lib/core/session/session-context";
import { simulation } from "../lib/core/simulation/controller";
import { serializeScene } from "../lib/core/model";
import { OakTreeEditor } from "./OakTreeEditor";

/** All tool-specific UI comes from the same plugin that implements its stroke. */
export function VoxelToolShelf() {
  const session = useSession();
  const panelId = useId();
  const canUndo = session.history((state) => state.past.length > 0);
  const canRedo = session.history((state) => state.future.length > 0);
  const id = session.ui((state) => state.voxelToolId);
  const stored = session.ui((state) => state.voxelToolValues);
  const scene = session.scene((state) => state.scene);
  const fileInput = useRef<HTMLInputElement>(null);
  const method = session.method((state) => state.methodId);
  const plugin = voxelTools.get(id);
  const unavailableReason = plugin?.unavailable({ scene, methodId: method });
  const allUnavailable = voxelTools.tools.every((tool) => tool.unavailable({ scene, methodId: method }));
  const values = plugin ? toolValues(plugin, stored[plugin.id]) : {};
  const [open, setOpen] = useState(false);
  const [subject, setSubject] = useState<"voxels" | "trees">("voxels");
  const [name, setName] = useState("Voxel scene");
  const exportScene = () => {
    const url = URL.createObjectURL(new Blob([serializeScene(session.scene.getState().scene)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = `${name.trim().replace(/[^a-z0-9-]/gi, "-") || "voxel-scene"}.json`;
    anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <aside className="voxel-tool-shelf" aria-label="Voxel editing">
    <button type="button" aria-expanded={open} aria-controls={panelId} onClick={() => {
      setOpen(!open); if (open) session.ui.getState().setVoxelTool(undefined);
    }}>Voxels {open ? "−" : "+"}</button>
    {open && <div id={panelId} className="voxel-tool-panel">
      <div className="voxel-tool-buttons">
        <button type="button" onClick={() => { session.ui.getState().setVoxelTool(undefined); simulation.newScene(undefined, session.id); }}>New scene</button>
        <button type="button" onClick={() => { session.ui.getState().setVoxelTool(undefined); session.ui.getState().setSceneSelectorOpen(true); }}>Open scene</button>
      </div>
      <div className="voxel-tool-buttons" aria-label="Editing subject">
        <button type="button" aria-pressed={subject === "voxels"} onClick={() => setSubject("voxels")}>Voxel tools</button>
        <button type="button" aria-pressed={subject === "trees"} onClick={() => {
          setSubject("trees"); session.ui.getState().setVoxelTool(undefined);
        }}>Trees</button>
      </div>
      {subject === "trees" ? <OakTreeEditor /> : <>
      {unavailableReason ? <p role="status">{unavailableReason}</p> : allUnavailable &&
        <p role="status">Voxel tools are unavailable in this scene. Hover over a tool for details.</p>}
      {[...new Set(voxelTools.tools.map((tool) => tool.ui.group))].map((group) => <fieldset key={group}>
        <legend>{group}</legend>
        <div className="voxel-tool-buttons">{voxelTools.tools.filter((tool) => tool.ui.group === group).map((tool) =>
          <button type="button" key={tool.id} aria-pressed={id === tool.id} title={tool.unavailable({ scene, methodId: method }) ?? tool.ui.hint}
            disabled={Boolean(tool.unavailable({ scene, methodId: method }))} onClick={() => session.ui.getState().setVoxelTool(id === tool.id ? undefined : tool.id)}>
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden><path d={tool.ui.icon} /></svg>
            {tool.ui.label}
          </button>)}</div>
      </fieldset>)}
      {plugin && <>
        <p>{plugin.ui.hint}</p>
        {plugin.ui.controls.map((control) => <label key={control.id}>{control.label}
          {control.kind === "toggle" ? <input type="checkbox" checked={values[control.id] === 1}
            onChange={(event) => session.ui.getState().setVoxelToolValue(plugin.id, control.id, event.target.checked ? 1 : 0)} />
            : <input type="number" min={control.min} max={control.max} step={control.step} value={values[control.id]}
              onChange={(event) => {
                if (Number.isFinite(event.target.valueAsNumber)) session.ui.getState().setVoxelToolValue(plugin.id, control.id, event.target.valueAsNumber);
              }} />}
        </label>)}
        <p className="voxel-tool-help">Shift-drag to navigate · Escape or Ctrl/⌘ Z cancels the current stroke. Water keeps moving.</p>
      </>}
      </>}
      <div className="voxel-tool-buttons">
        <button type="button" disabled={!canUndo} onClick={() => simulation.undo(session.id)}>Undo</button>
        <button type="button" disabled={!canRedo} onClick={() => simulation.redo(session.id)}>Redo</button>
      </div>
      <label>Scene name<input value={name} maxLength={80} onChange={(event) => setName(event.target.value)} /></label>
      <div className="voxel-tool-buttons">
        <button type="button" onClick={() => simulation.saveNamedScene(name, session.id)}>Save scene</button>
        <button type="button" onClick={exportScene}>Export JSON</button>
        <button type="button" onClick={() => fileInput.current?.click()}>Import JSON</button>
      </div>
      <input ref={fileInput} type="file" accept="application/json,.json" hidden onChange={async (event) => {
        const file = event.target.files?.[0]; event.target.value = "";
        if (!file) return;
        try {
          const contents = await file.text();
          session.ui.getState().setVoxelTool(undefined);
          simulation.importScene(file.name, contents, session.id);
        }
        catch { session.runtime.getState().setNotice("Could not read the scene file.", "warn"); }
      }} />
      <p className="voxel-tool-help">Saved scenes reopen from their initial water setup.</p>
    </div>}
  </aside>;
}
