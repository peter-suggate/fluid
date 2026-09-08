"use client";

import { useEffect, useId, useRef, useState } from "react";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { toolValues, type ToolControl } from "../lib/core/voxel-editor/plugin";
import { useSession } from "../lib/core/session/session-context";
import { simulation } from "../lib/core/simulation/controller";
import { serializeScene } from "../lib/core/model";
import { enableWaterLockReason } from "../lib/core/scene-fluid-readiness";
import { terrainHeightAt } from "../lib/core/terrain";

/** Plugins own the tools; this host only arranges their contextual controls. */
export function VoxelToolShelf() {
  const session = useSession();
  const panelId = useId();
  const root = useRef<HTMLElement>(null);
  const sceneButton = useRef<HTMLButtonElement>(null);
  const toolsButton = useRef<HTMLButtonElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const pending = session.ui((state) => state.voxelStrokePending);
  const mode = session.ui((state) => state.viewportMode);
  const canUndo = session.history((state) => state.past.length > 0);
  const canRedo = session.history((state) => state.future.length > 0);
  const id = session.ui((state) => state.voxelToolId);
  const stored = session.ui((state) => state.voxelToolValues);
  const scene = session.scene((state) => state.scene);
  const method = session.method((state) => state.methodId);
  const gpuInfo = session.diagnostics((state) => state.gpuInfo);
  const svoReadiness = session.diagnostics((state) => state.resourceReadiness.svo);
  const waterLockReason = enableWaterLockReason(gpuInfo, svoReadiness);
  const plugin = voxelTools.get(id);
  const unavailableReason = plugin?.unavailable({ scene, methodId: method });
  const allUnavailable = voxelTools.tools.every((tool) => tool.unavailable({ scene, methodId: method }));
  const values = plugin ? toolValues(plugin, stored[plugin.id], scene) : {};
  const [popover, setPopover] = useState<"scene" | "tools" | undefined>();
  const [name, setName] = useState("Voxel scene");
  useEffect(() => {
    if (!popover) return;
    const outside = (event: PointerEvent) => {
      if (!(event.target instanceof Node) || root.current?.contains(event.target)) return;
      // A dismissal is not a stroke. Capture it before the canvas can begin editing.
      event.preventDefault();
      event.stopPropagation();
      setPopover(undefined);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setPopover(undefined);
      // Let the gesture host receive Escape: closing a menu must not swallow cancellation.
      if (!session.ui.getState().voxelStrokePending) {
        (popover === "scene" ? sceneButton : toolsButton).current?.focus();
      }
    };
    document.addEventListener("pointerdown", outside, true);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside, true);
      document.removeEventListener("keydown", escape);
    };
  }, [popover, session]);
  const exportScene = () => {
    if (session.ui.getState().voxelStrokePending) return;
    const url = URL.createObjectURL(new Blob([serializeScene(session.scene.getState().scene)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = `${name.trim().replace(/[^a-z0-9-]/gi, "-") || "voxel-scene"}.json`;
    anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const controlInput = (control: ToolControl) => <label key={control.id} className="voxel-context-control">
    <span>{control.label}</span>
    {control.kind === "toggle" ? <input type="checkbox" checked={values[control.id] === 1}
      onChange={(event) => session.ui.getState().setVoxelToolValue(plugin!.id, control.id, event.target.checked ? 1 : 0)} />
      : <input type="number" min={control.min} max={control.max} step={control.step} value={values[control.id]}
        onChange={(event) => {
          if (Number.isFinite(event.target.valueAsNumber)) session.ui.getState().setVoxelToolValue(plugin!.id, control.id, event.target.valueAsNumber);
        }} />}
  </label>;
  const advanced = plugin?.ui.controls.filter(control => control.presentation === "advanced") ?? [];
  const changedAdvanced = advanced.filter(control => values[control.id] !== control.initial);
  return <aside ref={root} className="voxel-context" aria-label="Voxel editing"
    onPointerDown={event => event.stopPropagation()} onWheel={event => event.stopPropagation()}>
    <div className="voxel-context-bar">
      <button ref={sceneButton} type="button" aria-expanded={popover === "scene"} aria-controls={`${panelId}-scene`}
        onClick={() => setPopover(popover === "scene" ? undefined : "scene")}>Scene <span aria-hidden>⌄</span></button>
      <button ref={toolsButton} type="button" aria-expanded={popover === "tools"} aria-controls={`${panelId}-tools`}
        onClick={() => setPopover(popover === "tools" ? undefined : "tools")}>Tools <span aria-hidden>⌄</span></button>
      <button type="button" aria-label="Undo" title="Undo (Ctrl/⌘ Z)" disabled={pending || !canUndo} onClick={() => simulation.undo(session.id)}>↶</button>
      <button type="button" aria-label="Redo" title="Redo (Ctrl/⌘ Shift Z)" disabled={pending || !canRedo} onClick={() => simulation.redo(session.id)}>↷</button>
    </div>
    {popover === "scene" && <section id={`${panelId}-scene`} className="voxel-context-popover" aria-label="Scene document">
      <div className="voxel-context-heading"><strong>Scene</strong><button type="button" aria-label="Close scene menu" onClick={() => { setPopover(undefined); sceneButton.current?.focus(); }}>×</button></div>
      <div className="voxel-context-actions">
        <button type="button" disabled={pending} onClick={() => { setPopover(undefined); session.ui.getState().setVoxelTool(undefined); simulation.newScene(undefined, session.id); }}>New scene</button>
        <button type="button" disabled={pending} onClick={() => { setPopover(undefined); session.ui.getState().setVoxelTool(undefined); session.ui.getState().setSceneSelectorOpen(true); }}>Open scene</button>
      </div>
      {scene.systems?.fluid === false && <button type="button" disabled={pending || Boolean(waterLockReason)}
        title={waterLockReason ?? "Initialize water from this scene’s starting setup."} onClick={() => {
          const latest = session.diagnostics.getState();
          if (session.ui.getState().voxelStrokePending || enableWaterLockReason(latest.gpuInfo, latest.resourceReadiness.svo)) return;
          setPopover(undefined);
          session.ui.getState().setVoxelTool(undefined);
          simulation.setFluidSystem(true, session.id);
        }}>Enable water</button>}
      <label className="voxel-context-name">Scene name<input value={name} maxLength={80} onChange={(event) => setName(event.target.value)} /></label>
      <button className="voxel-context-save" type="button" disabled={pending} onClick={() => simulation.saveNamedScene(name, session.id)}>Save scene</button>
      <div className="voxel-context-actions">
        <button type="button" disabled={pending} onClick={exportScene}>Export JSON</button>
        <button type="button" disabled={pending} onClick={() => fileInput.current?.click()}>Import JSON</button>
      </div>
      <p>Saved scenes reopen from their initial water setup.</p>
    </section>}
    {popover === "tools" && <section id={`${panelId}-tools`} className="voxel-context-popover" aria-label="Choose an editing tool">
      <div className="voxel-context-heading"><strong>Editing tools</strong><button type="button" aria-label="Close tool chooser" onClick={() => { setPopover(undefined); toolsButton.current?.focus(); }}>×</button></div>
      {allUnavailable && <p role="status">Voxel tools are unavailable in this scene. Start a new voxel scene to build.</p>}
      {[...new Set(voxelTools.tools.map((tool) => tool.ui.group))].map((group) => <fieldset key={group}>
        <legend>{group}</legend>
        <div className="voxel-context-choices">{voxelTools.tools.filter((tool) => tool.ui.group === group).map((tool) =>
          <button type="button" key={tool.id} aria-pressed={id === tool.id} title={tool.unavailable({ scene, methodId: method }) ?? tool.ui.hint}
            disabled={pending || Boolean(tool.unavailable({ scene, methodId: method }))} onClick={() => {
              session.ui.getState().setVoxelTool(tool.id); setPopover(undefined);
            }}>
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden><path d={tool.ui.icon} /></svg>
            {tool.ui.label}
          </button>)}</div>
      </fieldset>)}
      <button className="voxel-context-add-tree" type="button" disabled={pending} title="Add an oak at the view centre, then adjust it with the object handles."
        onClick={() => {
          setPopover(undefined); session.ui.getState().setVoxelTool(undefined);
          session.ui.getState().setViewportMode("interact");
          const { x, z } = session.ui.getState().camera.target_m;
          simulation.addScenery("oak-v2", { x, y: terrainHeightAt(scene.terrain, x, z), z }, { x: 0, y: 1, z: 0 }, session.id);
        }}>＋ Add tree</button>
    </section>}
    {plugin && mode === "interact" && !popover && <section key={plugin.id} className="voxel-context-card" aria-label={`${plugin.ui.label} tool settings`}>
      <div className="voxel-context-heading"><strong>{plugin.ui.label}</strong><button type="button" onClick={() => session.ui.getState().setVoxelTool(undefined)}>Done</button></div>
      <p>{unavailableReason ?? plugin.ui.hint}</p>
      <div className="voxel-context-primary">{plugin.ui.controls.filter(control => control.presentation !== "advanced").map(controlInput)}</div>
      {changedAdvanced.length > 0 && <div className="voxel-context-badges" aria-label="Active advanced settings">{changedAdvanced.map(control =>
        <span key={control.id}>{control.label}{control.kind === "toggle" ? (values[control.id] === 1 ? " on" : " off") : ` ${values[control.id]}`}</span>)}</div>}
      <details className="voxel-context-more"><summary>More</summary>
        {advanced.map(controlInput)}
        <p>Shift-drag to navigate. Escape or Ctrl/⌘ Z cancels the stroke. Water keeps moving.</p>
      </details>
      {plugin.ui.notice && <p>{plugin.ui.notice}</p>}
    </section>}
    {pending && <p className="voxel-context-pending" role="status">Applying stroke… Save and history become available when it finishes.</p>}
    <input ref={fileInput} type="file" accept="application/json,.json" hidden onChange={async (event) => {
      const file = event.target.files?.[0]; event.target.value = "";
      if (!file) return;
      try {
        const contents = await file.text();
        if (session.ui.getState().voxelStrokePending) {
          session.runtime.getState().setNotice("Finish the voxel stroke before importing a scene.", "warn");
          return;
        }
        session.ui.getState().setVoxelTool(undefined);
        simulation.importScene(file.name, contents, session.id);
      }
      catch { session.runtime.getState().setNotice("Could not read the scene file.", "warn"); }
    }} />
  </aside>;
}
