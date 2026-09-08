"use client";

import { useSession } from "../lib/core/session/session-context";
import { toolValues, type ToolControl } from "../lib/core/voxel-editor/plugin";
import { voxelTools } from "../lib/core/voxel-editor/registry";

/**
 * The armed sculpt tool's card, and nothing else.
 *
 * This used to be a persistent shelf: a Scene / Tools / Undo / Redo bar with
 * two popovers, standing in the corner of every viewport whether or not anyone
 * was editing. Those capabilities are contextual now — the tools are wedges on
 * the scene's ring, the document verbs sit on the same ring and on the
 * container strip while the tank is selected, and undo/redo were always on the
 * keyboard — so what remains here is the one thing that genuinely follows a
 * mode: the card describing the tool that is armed right now. Done, or Escape
 * through the gesture host, puts it away and the card goes with it.
 *
 * Plugins own the tools; this host only arranges the armed one's declared
 * controls, so adding a tool never touches this file.
 */
export function VoxelToolShelf() {
  const session = useSession();
  const pending = session.ui((state) => state.voxelStrokePending);
  const mode = session.ui((state) => state.viewportMode);
  const id = session.ui((state) => state.voxelToolId);
  const stored = session.ui((state) => state.voxelToolValues);
  const scene = session.scene((state) => state.scene);
  const method = session.method((state) => state.methodId);
  const plugin = voxelTools.get(id);
  const unavailableReason = plugin?.unavailable({ scene, methodId: method });
  const values = plugin ? toolValues(plugin, stored[plugin.id], scene) : {};
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
  if (!plugin && !pending) return null;
  return <aside className="voxel-context" aria-label="Voxel editing"
    onPointerDown={event => event.stopPropagation()} onWheel={event => event.stopPropagation()}>
    {plugin && mode === "interact" && <section key={plugin.id} className="voxel-context-card" aria-label={`${plugin.ui.label} tool settings`}>
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
  </aside>;
}
