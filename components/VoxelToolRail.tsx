"use client";

import { Fragment } from "react";
import { MousePointer2, RotateCcw } from "lucide-react";
import { VOXEL_RAIL_CONTROLS, voxelToolRail } from "../lib/core/editor-voxel-tool-actions";
import { simulation } from "../lib/core/simulation/controller";
import { useSession } from "../lib/core/session/session-context";
import { SHARED_TOOL_VALUES_ID, adjustableControl, controlValueOwner, controlValues, storedControlValues,
  storedToolValues, toolValues, type ToolChoiceControl, type ToolControl, type ToolNumberControl,
  type VoxelToolPlugin } from "../lib/core/voxel-editor/plugin";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { EditorActionPathGlyph } from "./EditorActionIcon";
import { VoxelSelectionBar } from "./VoxelSelectionBar";
import { Stepper, Swatch, ToggleButton } from "./ui";

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
 * Above the tools stands the one thing they share and the hand changes between
 * strokes: what a brush does with the cells it covers. It is a switch on the
 * rail rather than a control on each line so that it does not move when the
 * tool does, and its lit mark is the colour the stroke's ghost is drawn in.
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

/** A control as a line draws it: its resolved value, and the owner that value is stored under. */
interface LineControl<Control extends ToolControl = ToolControl> {
  readonly control: Control;
  readonly owner: string;
  readonly value: number;
  /** The keys that step it, for a number the gesture host sizes. */
  readonly keys?: string;
}

const withKey = (label: string, shortcut: string | undefined) =>
  shortcut ? `${label}  ·  ${shortcut.toUpperCase()}` : label;

/**
 * Declared controls, drawn: numbers, then choices, then switches.
 *
 * The one place a control's kind becomes a widget, for a tool's line and the
 * selection's alike. A choice is a row of marks of which one is down — a
 * swatch where the option is a colour, its icon or its name otherwise — and a
 * choice with nothing to choose from draws nothing.
 */
function ControlLine({ controls }: { controls: readonly LineControl[] }) {
  const session = useSession();
  const set = (item: LineControl, value: number) =>
    session.ui.getState().setVoxelToolValue(item.owner, item.control.id, value);
  const numbers = controls.filter((item): item is LineControl<ToolNumberControl> =>
    item.control.kind === undefined || item.control.kind === "number");
  const choices = controls.filter((item): item is LineControl<ToolChoiceControl> => item.control.kind === "choice");
  const toggles = controls.filter((item): item is LineControl<ToolNumberControl> => item.control.kind === "toggle");
  return <>
    {numbers.map((item) => {
      const control = item.control;
      return <label key={control.id} className="tool-option"
        title={item.keys ? `${control.label}  ·  ${item.keys}` : control.label}>
        <span>{control.short ?? control.label}</span>
        <Stepper value={item.value} min={control.min} max={control.max} step={control.step}
          ariaLabel={control.label} onChange={(value) => set(item, value)} />
      </label>;
    })}
    {choices.map((item) => {
      const control = item.control;
      if (control.options.length === 0) return null;
      return <div key={control.id} className="tool-choice" role="group" aria-label={control.label}>
        {control.options.map((option) => <ToggleButton key={option.value}
          className={option.swatch ? "tool-choice-swatch" : undefined}
          pressed={item.value === option.value}
          ariaLabel={`${control.label}: ${option.label}`}
          hint={withKey(option.label, option.shortcut)}
          // One of them is always down: pressing the chosen mark leaves it chosen.
          onChange={() => set(item, option.value)}
        >{option.swatch ? <Swatch color={option.swatch} />
          : option.icon ? <EditorActionPathGlyph path={option.icon} size={12} /> : option.label}
          {option.shortcut && <kbd>{option.shortcut.toUpperCase()}</kbd>}</ToggleButton>)}
      </div>;
    })}
    {toggles.map((item) => {
      const control = item.control;
      return <ToggleButton key={control.id}
        pressed={item.value === 1}
        ariaLabel={control.label}
        hint={withKey(control.label, control.shortcut)}
        onChange={(on) => set(item, on ? 1 : 0)}
      >{control.short ?? control.label}{control.shortcut && <kbd>{control.shortcut.toUpperCase()}</kbd>}</ToggleButton>;
    })}
  </>;
}

/** The armed tool's numbers, choices and switches, on one line beside its mark. */
function ToolOptions({ plugin }: { plugin: VoxelToolPlugin }) {
  const session = useSession();
  const stored = session.ui((state) => state.voxelToolValues);
  const scene = session.scene((state) => state.scene);
  const methodId = session.method((state) => state.methodId);
  const clipboard = session.ui((state) => state.voxelClipboard);
  const values = toolValues(plugin, storedToolValues(plugin, stored), scene);
  const unavailable = plugin.unavailable({ scene, methodId, clipboard });
  // The rail's own switch is drawn on the rail, and a control put away by the
  // values that stand (a material while erasing) keeps its value and its place.
  const shown = plugin.ui.controls.filter((control) => control.surface !== "rail" && control.when?.(values) !== false);
  return <div className="tool-options" role="group" aria-label={`${plugin.ui.label} options`}
    data-testid="voxel-tool-options">
    <div className="tool-options-line">
      <strong>{plugin.ui.label}</strong>
      <ControlLine controls={shown.map((control) => ({
        control, owner: controlValueOwner(plugin.id, control), value: values[control.id] ?? control.initial,
        keys: adjustKeys(plugin, control),
      }))} />
    </div>
    {unavailable && <p className="tool-options-note is-blocked">{unavailable}</p>}
    {!unavailable && plugin.ui.notice && <p className="tool-options-note">{plugin.ui.notice}</p>}
  </div>;
}

/**
 * A shared choice as a switch on the rail: its options as adjacent marks, one
 * of them down, each in its own tone and on its own letter.
 *
 * `idle` is the armed tool not reading it — a selection tool, a water shape.
 * The switch still stands and still answers, because what it holds is the
 * editor's and the next brush will read it; it is only drawn quietly.
 */
function RailSwitch({ control, idle }: { control: ToolChoiceControl; idle: boolean }) {
  const session = useSession();
  const stored = session.ui((state) => state.voxelToolValues);
  const value = controlValues([control], storedControlValues(SHARED_TOOL_VALUES_ID, [control], stored))[control.id];
  return <div className="tool-rail-switch" role="group" aria-label={control.label} data-idle={idle || undefined}
    data-testid={`voxel-tool-rail-${control.id}`}>
    {control.options.map((option) => <div key={option.value} className="tool-rail-slot">
      <button type="button" className={`tool-rail-key${option.tone ? ` tone-${option.tone}` : ""}`}
        aria-pressed={value === option.value} aria-label={`${control.label}: ${option.label}`}
        aria-keyshortcuts={option.shortcut} data-testid={`voxel-tool-rail-${control.id}-${option.value}`}
        // As a tool's mark: the canvas keeps the focus, and with it the keys.
        onMouseDown={(event) => event.preventDefault()}
        onClick={() => session.ui.getState().setVoxelToolValue(SHARED_TOOL_VALUES_ID, control.id, option.value)}>
        {option.icon ? <EditorActionPathGlyph path={option.icon} size={14} /> : option.label.slice(0, 1)}
        {option.shortcut && <kbd aria-hidden>{option.shortcut.toUpperCase()}</kbd>}
      </button>
      <span className="tool-rail-tip">
        <strong>{option.label}{option.shortcut && <kbd>{option.shortcut.toUpperCase()}</kbd>}</strong>
        {option.hint && <small>{option.hint}</small>}
      </span>
    </div>)}
  </div>;
}

export function VoxelToolRail() {
  const session = useSession();
  const mode = session.ui((state) => state.viewportMode);
  const armedId = session.ui((state) => state.voxelToolId);
  const pending = session.ui((state) => state.voxelStrokePending);
  const clipboard = session.ui((state) => state.voxelClipboard);
  // The region and the selection that names it land together (`selectVoxelRegion`); read as one.
  const selected = session.ui((state) => state.selection?.kind === "voxel-region" ? state.voxelRegion : undefined);
  const scene = session.scene((state) => state.scene);
  const methodId = session.method((state) => state.methodId);
  if (mode !== "interact") return null;
  const groups = voxelToolRail(scene, methodId, clipboard);
  const armed = voxelTools.get(armedId);
  const switches = VOXEL_RAIL_CONTROLS.filter((control): control is ToolChoiceControl => control.kind === "choice");
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
      {selected
        ? <VoxelSelectionBar region={selected} />
        : <span className="tool-rail-tip"><strong>Select<kbd>esc</kbd></strong>
          <small>Click a thing to select it, drag across solids to box them, drag the room to orbit.</small></span>}
    </div>
    {switches.map((control) => <Fragment key={control.id}>
      <i className="tool-rail-rule" aria-hidden />
      <RailSwitch control={control} idle={armed !== undefined && !armed.ui.controls.includes(control)} />
    </Fragment>)}
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
    {/* Last, apart from the tools: it ends every edit rather than making one.
        Document edits never enter the link, so the overrides chip cannot do
        this; it is one undo step (`revertScene`). */}
    <i className="tool-rail-rule" aria-hidden />
    <div className="tool-rail-slot">
      <button type="button" className="tool-rail-key" aria-label="Revert scene" disabled={pending}
        data-testid="voxel-tool-rail-revert" onMouseDown={keepFocus}
        onClick={() => simulation.revertScene(session.id)}>
        <RotateCcw width={15} height={15} strokeWidth={1.7} aria-hidden />
      </button>
      <span className="tool-rail-tip"><strong>Revert scene</strong>
        <small>Discard every edit and reopen the scene as authored. Camera and solver settings stay; undo brings the edits back.</small></span>
    </div>
    {pending && <p className="tool-rail-pending" role="status">Applying…</p>}
  </nav>;
}
