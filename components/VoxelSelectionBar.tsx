"use client";

import { useMemo, type MouseEvent } from "react";
import { performEditorAction } from "../lib/core/editor-action-runtime";
import { voxelRegionVerbEntries, voxelRegionVerbGroups, voxelVerbOfferedVariants,
  type VoxelSelectionRegion } from "../lib/core/editor-voxel-region";
import { useSession } from "../lib/core/session/session-context";
import { controlValueOwner, controlValues, type ToolChoiceControl, type ToolControl,
  type ToolNumberControl } from "../lib/core/voxel-editor/plugin";
import { verbValuesOwner, voxelRegionVerbTarget, voxelVerbKeyLabel } from "../lib/core/voxel-editor/verb";
import { EditorActionPathGlyph } from "./EditorActionIcon";
import { Button, Stepper, Swatch, ToggleButton } from "./ui";

/** A verb's control as the bar draws it: its resolved value, and the owner that value is stored under. */
interface BarControl<Control extends ToolControl = ToolControl> {
  readonly control: Control;
  readonly owner: string;
  readonly value: number;
}

/** A hint with the key that does the same thing. */
const withKey = (hint: string, chord: string | undefined) => chord ? `${hint}  ·  ${voxelVerbKeyLabel(chord)}` : hint;

/**
 * A press on the bar leaves focus where it was. The verbs' keys answer while
 * no field has focus, and a button that took it would turn the next Space into
 * a second press of itself. A field still takes focus: that is how it is typed in.
 */
const keepFocus = (event: MouseEvent) => {
  if (!(event.target instanceof HTMLInputElement)) event.preventDefault();
};

/**
 * What the standing voxel selection can be told to do, beside the Select mark
 * — where an armed tool's options hang, because a selection is what the
 * pointer tool is holding.
 *
 * The registry's verbs, a row to each group they declare. A verb is a button;
 * one that runs several ways (an axis, a direction) is a mark for each way.
 * Every control the verbs declare is drawn once, on the row of the first verb
 * that reads it. A verb that cannot run here is dimmed and its reason is
 * written under the rows, never hidden.
 */
export function VoxelSelectionBar({ region }: { region: VoxelSelectionRegion }) {
  const session = useSession();
  const stored = session.ui((state) => state.voxelToolValues);
  const pending = session.ui((state) => state.voxelStrokePending);
  const clipboard = session.ui((state) => state.voxelClipboard);
  const scene = session.scene((state) => state.scene);
  const methodId = session.method((state) => state.methodId);
  // Asking a verb whether it applies counts the solids in the box; once per selection and document.
  const entries = useMemo(() => voxelRegionVerbEntries(voxelRegionVerbTarget({ scene, methodId, region, clipboard })),
    [scene, methodId, region, clipboard]);
  const drawn = new Set<string>();
  const rows = voxelRegionVerbGroups(entries).map((row) => {
    const controls: BarControl[] = [];
    for (const { verb } of row.entries) for (const control of verb.controls ?? []) {
      const owner = controlValueOwner(verbValuesOwner(verb), control);
      if (drawn.has(`${owner}:${control.id}`)) continue;
      drawn.add(`${owner}:${control.id}`);
      controls.push({ control, owner, value: controlValues([control], stored[owner])[control.id]! });
    }
    return { ...row, controls };
  });
  const set = (item: BarControl, value: number) =>
    session.ui.getState().setVoxelToolValue(item.owner, item.control.id, value);
  const run = (verbId: string, variantId?: string) =>
    performEditorAction({ kind: "voxel-verb", verbId, variantId }, session);
  const reasons = [...new Set(entries.flatMap((entry) => entry.unavailable ?? []))];
  return <div className="tool-options selection-bar" role="group" aria-label="Selection"
    data-testid="voxel-selection-options" onMouseDown={keepFocus}>
    {rows.map((row) => <div key={row.group} className="tool-options-line" role="group" aria-label={row.group}>
      <strong>{row.group}</strong>
      {row.entries.map(({ verb, unavailable }) => {
        const off = pending || unavailable !== undefined;
        const variants = voxelVerbOfferedVariants(verb, "bar");
        if (variants.length === 0) {
          const chord = verb.ui.keys?.[0];
          return <Button key={verb.id} tone={verb.ui.tone === "danger" ? "danger" : "default"}
            disabled={off} hint={unavailable ?? verb.ui.hint} testId={`voxel-verb-${verb.id}`}
            onClick={() => run(verb.id)}>
            <EditorActionPathGlyph path={verb.ui.icon} size={12} />{verb.ui.label}
            {chord && <kbd>{voxelVerbKeyLabel(chord)}</kbd>}
          </Button>;
        }
        return <div key={verb.id} className="selection-bar-ways" role="group" aria-label={verb.ui.label}>
          <span title={unavailable ?? verb.ui.hint}><EditorActionPathGlyph path={verb.ui.icon} size={12} />{verb.ui.label}</span>
          {variants.map((variant) => <Button key={variant.id} disabled={off}
            ariaLabel={`${verb.ui.label} ${variant.label}`}
            hint={unavailable ?? withKey(variant.hint ?? verb.ui.hint, variant.keys?.[0])}
            testId={`voxel-verb-${verb.id}-${variant.id}`}
            onClick={() => run(verb.id, variant.id)}>{variant.label}</Button>)}
        </div>;
      })}
      {row.controls.map((item) => {
        const { control } = item, key = `${item.owner}:${control.id}`;
        if (control.kind === "choice") {
          const chosen = item as BarControl<ToolChoiceControl>;
          if (chosen.control.options.length === 0) return null;
          return <div key={key} className="tool-choice" role="group" aria-label={control.label}>
            {chosen.control.options.map((option) => <ToggleButton key={option.value}
              className={option.swatch ? "tool-choice-swatch" : undefined}
              pressed={item.value === option.value}
              ariaLabel={`${control.label}: ${option.label}`}
              hint={option.label}
              // One of them is always down: pressing the chosen mark leaves it chosen.
              onChange={() => set(item, option.value)}
            >{option.swatch ? <Swatch color={option.swatch} />
              : option.icon ? <EditorActionPathGlyph path={option.icon} size={12} /> : option.label}</ToggleButton>)}
          </div>;
        }
        const number = control as ToolNumberControl;
        if (number.kind === "toggle") {
          return <ToggleButton key={key} pressed={item.value === 1} ariaLabel={control.label} hint={control.label}
            onChange={(on) => set(item, on ? 1 : 0)}>{number.short ?? control.label}</ToggleButton>;
        }
        return <label key={key} className="tool-option" title={control.label}>
          <span>{number.short ?? control.label}</span>
          <Stepper value={item.value} min={number.min} max={number.max} step={number.step}
            ariaLabel={control.label} onChange={(value) => set(item, value)} />
        </label>;
      })}
    </div>)}
    {reasons.map((reason) => <p key={reason} className="tool-options-note is-blocked">{reason}</p>)}
  </div>;
}
