"use client";

import { Fragment } from "react";
import { MousePointer2 } from "lucide-react";
import { voxelToolRail } from "../lib/core/editor-voxel-tool-actions";
import { useSession } from "../lib/core/session/session-context";
import { adjustableControl, toolValues, type ToolControl, type VoxelToolPlugin } from "../lib/core/voxel-editor/plugin";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { EditorActionPathGlyph } from "./EditorActionIcon";
import { Stepper, ToggleButton } from "./ui";

/**
 * The sculpt tools, standing down the left edge of EDIT.
 *
 * They were rows on the container's strip and a card in the top-left corner:
 * two places, neither of them where a tool lives. The strip is anchored to a
 * projected corner of the tank, so the tools moved with every orbit, hid behind
 * a chevron per group, and vanished altogether the moment one was armed —
 * which is exactly when a reader wants the one beside it. The card then spent
 * a column of typed number fields on what a hand already knows how to say with
 * a bracket key.
 *
 * So: one mark per tool, in one fixed place, each on the digit of its position.
 * The armed mark is lit, and the only thing that opens is a single line beside
 * it — the tool's own numbers and switches, each of which is also a key (see
 * `useVoxelToolGesture`). Clicking the lit mark, or the pointer at the head of
 * the rail, or Escape, puts the tool away.
 *
 * This is the bounded exception to "nothing permanent" that a palette earns:
 * sculpting is a run of tool changes, and a ring summoned per change costs more
 * than a column that is only here while EDIT is. LOOK draws none of it.
 *
 * Everything drawn is the registry's declaration — groups, order, icon, hint,
 * controls, availability — so a plugin added there takes the next place and the
 * next digit without touching this file. A tool that cannot run here is dimmed,
 * not omitted, and still arms: its line says why.
 */

/** The keys that step a number, as its tip names them. */
function adjustKeys(plugin: VoxelToolPlugin, control: ToolControl): string | undefined {
  if (adjustableControl(plugin, false) === control) return "[ ]  ·  ⌥ scroll";
  if (adjustableControl(plugin, true) === control) return "⇧ [ ]  ·  ⇧⌥ scroll";
  return undefined;
}

/** The armed tool's numbers and switches, on one line beside its mark. */
function ToolOptions({ plugin }: { plugin: VoxelToolPlugin }) {
  const session = useSession();
  const stored = session.ui((state) => state.voxelToolValues);
  const scene = session.scene((state) => state.scene);
  const methodId = session.method((state) => state.methodId);
  const values = toolValues(plugin, stored[plugin.id], scene);
  const unavailable = plugin.unavailable({ scene, methodId });
  const set = (control: ToolControl, value: number) =>
    session.ui.getState().setVoxelToolValue(plugin.id, control.id, value);
  const numbers = plugin.ui.controls.filter((control) => control.kind !== "toggle");
  const toggles = plugin.ui.controls.filter((control) => control.kind === "toggle");
  return <div className="tool-options" role="group" aria-label={`${plugin.ui.label} options`}
    data-testid="voxel-tool-options">
    <div className="tool-options-line">
      <strong>{plugin.ui.label}</strong>
      {numbers.map((control) => {
        const keys = adjustKeys(plugin, control);
        return <label key={control.id} className="tool-option" title={keys ? `${control.label}  ·  ${keys}` : control.label}>
          <span>{control.short ?? control.label}</span>
          <Stepper value={values[control.id] ?? control.initial} min={control.min} max={control.max} step={control.step}
            ariaLabel={control.label} onChange={(value) => set(control, value)} />
        </label>;
      })}
      {toggles.map((control) => <ToggleButton key={control.id}
        pressed={values[control.id] === 1}
        ariaLabel={control.label}
        hint={control.shortcut ? `${control.label}  ·  ${control.shortcut.toUpperCase()}` : control.label}
        onChange={(on) => set(control, on ? 1 : 0)}
      >{control.short ?? control.label}{control.shortcut && <kbd>{control.shortcut.toUpperCase()}</kbd>}</ToggleButton>)}
    </div>
    {unavailable && <p className="tool-options-note is-blocked">{unavailable}</p>}
    {!unavailable && plugin.ui.notice && <p className="tool-options-note">{plugin.ui.notice}</p>}
  </div>;
}

export function VoxelToolRail() {
  const session = useSession();
  const mode = session.ui((state) => state.viewportMode);
  const armedId = session.ui((state) => state.voxelToolId);
  const pending = session.ui((state) => state.voxelStrokePending);
  const scene = session.scene((state) => state.scene);
  const methodId = session.method((state) => state.methodId);
  if (mode !== "interact") return null;
  const groups = voxelToolRail(scene, methodId);
  const armed = voxelTools.get(armedId);
  // A tool armed from elsewhere — a water shape, from its own row — is a guest:
  // it has no place here, but its line still needs somewhere to hang.
  const guest = armed && !groups.some((group) => group.tools.some((tool) => tool.id === armed.id)) ? armed : undefined;
  const arm = (id: string | undefined) => session.ui.getState().setVoxelTool(id === armedId ? undefined : id);
  // A mark never takes focus from the canvas on a click: the keys that drive
  // the tool — Space among them — must keep reaching the viewport, not a button.
  const keepFocus = (event: React.MouseEvent) => event.preventDefault();
  return <nav className="tool-rail" aria-label="Sculpt tools" data-testid="voxel-tool-rail"
    onPointerDown={(event) => event.stopPropagation()} onWheel={(event) => event.stopPropagation()}>
    <div className="tool-rail-slot">
      <button type="button" className="tool-rail-key" aria-pressed={!armed} aria-label="Select"
        data-testid="voxel-tool-rail-select" onMouseDown={keepFocus}
        onClick={() => session.ui.getState().setVoxelTool(undefined)}>
        <MousePointer2 width={15} height={15} strokeWidth={1.7} aria-hidden />
      </button>
      <span className="tool-rail-tip"><strong>Select<kbd>esc</kbd></strong>
        <small>Click a thing to select it, drag across solids to box them, drag the room to orbit.</small></span>
    </div>
    {groups.map((group) => <Fragment key={group.id}>
      <i className="tool-rail-rule" aria-hidden />
      {group.tools.map((tool) => {
        const lit = tool.id === armedId;
        return <div key={tool.id} className="tool-rail-slot" data-armed={lit || undefined}>
          <button type="button" className="tool-rail-key" aria-pressed={lit} aria-label={tool.label}
            aria-keyshortcuts={tool.key} data-unavailable={tool.unavailable ? "true" : undefined}
            data-testid={`voxel-tool-rail-${tool.id}`} onMouseDown={keepFocus} onClick={() => arm(tool.id)}>
            <EditorActionPathGlyph path={tool.iconPath} size={16} />
            {tool.key && <kbd aria-hidden>{tool.key}</kbd>}
          </button>
          {lit && armed
            ? <ToolOptions plugin={armed} />
            : <span className="tool-rail-tip"><strong>{tool.label}{tool.key && <kbd>{tool.key}</kbd>}</strong>
              <small>{tool.hint}</small></span>}
        </div>;
      })}
    </Fragment>)}
    {guest && <>
      <i className="tool-rail-rule" aria-hidden />
      <div className="tool-rail-slot" data-armed>
        <button type="button" className="tool-rail-key" aria-pressed aria-label={`${guest.ui.label} — put away`}
          data-testid={`voxel-tool-rail-${guest.id}`} onMouseDown={keepFocus} onClick={() => arm(guest.id)}>
          <EditorActionPathGlyph path={guest.ui.icon} size={16} />
        </button>
        <ToolOptions plugin={guest} />
      </div>
    </>}
    {pending && <p className="tool-rail-pending" role="status">Applying…</p>}
  </nav>;
}
