import type { EditorAction, EditorActionIcon, EditorActionTone } from "./editor-action";
import type { SceneDescription } from "./model";
import { materialControl, operationControl, toolOperation } from "./voxel-editor/controls";
import { adjustableControl, type ToolControl, type ToolHintKey, type ToolHintPhase, type ToolValues,
  type VoxelToolPlugin } from "./voxel-editor/plugin";
import { voxelTools } from "./voxel-editor/registry";
import type { VoxelClipboard } from "./voxel-editor/verb";

/**
 * The sculpt tools, arranged for the surfaces that offer them.
 *
 * The registry owns the tools; this only arranges them, exactly as the old
 * shelf's chooser did — one entry per declared group. A plugin added to the
 * registry appears on every surface without a second table, and no tool id is
 * dispatched on: each entry's effect carries the id back to the one performer.
 *
 * Two surfaces compose from here: the scene's ring draws the groups as wedges
 * (`voxelSculptActions`), and the EDIT rail stands the brushes and the
 * selection tools down the viewport's left edge (`voxelToolRail`) — the same
 * groups, the same availability, one composition. A tool that does not apply right now is drawn
 * with its own reason as the hint — never omitted — so the reason ("Choose
 * Sparse CM12 to edit solids while water runs") teaches the way in rather than
 * hiding it.
 */
const GROUP_FACES: Record<string, {
  readonly id: string;
  readonly label: string;
  readonly icon: EditorActionIcon;
  readonly tone: EditorActionTone;
  readonly hint: string;
  /** Stands on the EDIT rail. Water shapes do not: to a reader they are water, and the Drop-water row owns them. */
  readonly rail?: boolean;
}> = {
  Brush: {
    rail: true,
    id: "sculpt-brush",
    label: "Brushes",
    icon: "solid",
    tone: "body",
    hint: "Attach, erase or paint solids, terrain and scenery: the pen and shapes",
  },
  Select: {
    rail: true,
    id: "sculpt-select",
    label: "Select voxels",
    icon: "region",
    tone: "region",
    hint: "Select solid voxels: click a connected piece or drag a box",
  },
  // Not on the rail: the Paste verb arms it, and it stands as the rail's guest while it does.
  Clipboard: {
    id: "sculpt-clipboard",
    label: "Paste",
    icon: "solid",
    tone: "body",
    hint: "Stamp the copied voxels where you click",
  },
  Fluid: {
    id: "sculpt-water",
    label: "Water shapes",
    icon: "water-ball",
    tone: "fluid",
    hint: "Drag shaped bodies of water into the live solve",
  },
};

export interface VoxelToolEntry {
  readonly id: string;
  readonly label: string;
  /** The plugin's own 24-box SVG path. */
  readonly iconPath: string;
  /** The plugin's hint, or — when it cannot run here — its own reason why not. */
  readonly hint: string;
  readonly unavailable?: string;
}

export interface VoxelToolGroup {
  readonly id: string;
  /** The registry's group name; `"Fluid"` is the water-shape group. */
  readonly group: string;
  readonly label: string;
  readonly icon: EditorActionIcon;
  readonly tone: EditorActionTone;
  readonly hint: string;
  readonly tools: readonly VoxelToolEntry[];
}

/** `clipboard` is what the editor last copied: a tool that stamps it cannot run without one. */
export function voxelToolGroups(
  scene: SceneDescription,
  methodId: string | undefined,
  clipboard?: VoxelClipboard,
): readonly VoxelToolGroup[] {
  const groups = [...new Set(voxelTools.tools.map((tool) => tool.ui.group))];
  return groups.map((group) => {
    const face = GROUP_FACES[group]
      ?? { id: `sculpt-${group.toLowerCase()}`, label: group, icon: "solid" as const, tone: "body" as const, hint: group };
    return {
      ...face,
      group,
      tools: voxelTools.tools
        .filter((tool) => tool.ui.group === group)
        .map((tool) => {
          const unavailable = tool.unavailable({ scene, methodId: methodId ?? "", clipboard });
          return {
            id: tool.id,
            label: tool.ui.label,
            iconPath: tool.ui.icon,
            hint: unavailable ?? tool.ui.hint,
            unavailable,
          };
        }),
    };
  });
}

export interface VoxelRailTool extends VoxelToolEntry {
  /** The key that arms it: the letter it declares, or else its place on the rail, counted from one. Absent past nine. */
  readonly key?: string;
}

/**
 * The EDIT rail: the solid tools, group by group, each on the digit of its place.
 *
 * The key is the position rather than a letter because the rail is always in
 * view while it means anything — the third mark is `3`, and there is nothing to
 * memorize. It is derived here, from the same order the rail draws, so a tool
 * added to the registry takes the next digit without a table.
 */
export function voxelToolRail(
  scene: SceneDescription,
  methodId: string | undefined,
  clipboard?: VoxelClipboard,
): readonly (Omit<VoxelToolGroup, "tools"> & { readonly tools: readonly VoxelRailTool[] })[] {
  let place = 0;
  return voxelToolGroups(scene, methodId, clipboard)
    .filter((group) => GROUP_FACES[group.group]?.rail)
    .map((group) => ({ ...group, tools: group.tools.map((tool) => {
      place += 1;
      return { ...tool, key: voxelTools.get(tool.id)?.ui.key?.toUpperCase() ?? (place <= 9 ? String(place) : undefined) };
    }) }));
}

const RAIL_TOOL_IDS: readonly string[] = [...new Set(voxelTools.tools.map((tool) => tool.ui.group))]
  .filter((group) => GROUP_FACES[group]?.rail)
  .flatMap((group) => voxelTools.tools.filter((tool) => tool.ui.group === group).map((tool) => tool.id));

/**
 * The rail tool a key arms: the letter a tool declares (`ui.key`), or a digit
 * by place. The scene is not asked: a tool that cannot run still arms and says
 * why. The registry has already refused a letter the editor or a control
 * answers to, so this cannot shadow one.
 */
export function voxelRailToolForKey(key: string): string | undefined {
  if (/^[1-9]$/.test(key)) return RAIL_TOOL_IDS[Number(key) - 1];
  const letter = key.toLowerCase();
  return /^[a-z]$/.test(letter) ? RAIL_TOOL_IDS.find((id) => voxelTools.get(id)!.ui.key === letter) : undefined;
}

/** The key that arms a tool when it stands on the rail: its letter, or its digit. */
export function voxelRailKeyForTool(toolId: string): string | undefined {
  const letter = RAIL_TOOL_IDS.includes(toolId) ? voxelTools.get(toolId)!.ui.key : undefined;
  if (letter) return letter;
  const place = RAIL_TOOL_IDS.indexOf(toolId);
  return place >= 0 && place < 9 ? String(place + 1) : undefined;
}

/**
 * The controls the rail draws once, above its tools, rather than on each
 * tool's line: every `surface: "rail"` control a rail tool lists, by first
 * appearance. They are shared by declaration, so one value stands for all.
 */
export const VOXEL_RAIL_CONTROLS: readonly ToolControl[] = [...new Set(RAIL_TOOL_IDS
  .flatMap((id) => voxelTools.get(id)!.ui.controls.filter((control) => control.surface === "rail")))];

/**
 * The controls whose keys are live while this tool is armed: its own, and the
 * rail's while it stands there. The rail's switch is in view under every rail
 * tool, so its letters answer under every one of them — a selection tool
 * included — rather than only under the tools that read it.
 */
export function voxelToolKeyControls(toolId: string | undefined): readonly ToolControl[] {
  const tool = voxelTools.get(toolId);
  if (!tool) return [];
  const rail = RAIL_TOOL_IDS.includes(tool.id)
    ? VOXEL_RAIL_CONTROLS.filter((control) => !tool.ui.controls.includes(control)) : [];
  return [...tool.ui.controls, ...rail];
}

/**
 * One line saying what the hand can do while a tool is armed, most used first.
 *
 * Composed from the tool's own declaration — which choice its letters switch,
 * what Alt does to it, which numbers its brackets step — so the line cannot
 * promise a key the gesture host would not take. Its toggles are not repeated:
 * each already shows its letter on the tool's line. The camera clauses are the
 * host's and constant.
 */
export function voxelToolLegend(toolId: string | undefined): string | undefined {
  const tool = voxelTools.get(toolId);
  if (!tool) return undefined;
  const name = (control: { readonly label: string }) => control.label.split(" · ")[0]!.toLowerCase();
  const first = adjustableControl(tool, false), second = adjustableControl(tool, true);
  const switches = voxelToolKeyControls(tool.id).flatMap((control) => {
    if (control.kind !== "choice") return [];
    const keys = control.options.flatMap((option) => option.shortcut ? [option.shortcut.toUpperCase()] : []);
    return keys.length > 0 ? [`${keys.join(" ")} ${name(control)}`] : [];
  });
  return [
    ...switches,
    // In the tool's own words, limits included: under paint the modifier samples, and its line says so.
    tool.ui.opposite && `⌥ ${tool.ui.opposite}`,
    first && `[ ] ${name(first)}`,
    second && second !== first && `⇧[ ] ${name(second)}`,
    "space-drag orbits",
    "right-click cancels the stroke, then puts the tool away",
  ].filter(Boolean).join(" · ");
}

/**
 * The controls a tool's rail line draws, and how many wait behind its "more".
 *
 * The rail's own switch is drawn on the rail, and a control put away by the
 * values that stand (a material while erasing) keeps its value and its place.
 * An advanced control shows only with the line expanded, unless it holds
 * something other than its initial value: a mirror that is on is changing every
 * stroke, and must be in view to be turned off.
 */
export function voxelToolLineControls(plugin: VoxelToolPlugin, values: ToolValues, expanded: boolean):
  { readonly shown: readonly ToolControl[]; readonly advanced: number } {
  const drawn = plugin.ui.controls.filter((control) => control.surface !== "rail" && control.when?.(values) !== false);
  const advanced = drawn.filter((control) => control.tier === "advanced");
  return {
    shown: drawn.filter((control) => control.tier !== "advanced" || expanded
      || (values[control.id] ?? control.initial) !== control.initial),
    advanced: advanced.length,
  };
}

/**
 * What a hint's named key reads as, for this tool with these values — or
 * nothing, where the tool does not have it. Resolved from the declaration, as
 * the legend above is, so a hint cannot promise a key the host would not take.
 */
function hintKey(tool: VoxelToolPlugin, key: ToolHintKey, values: ToolValues): string | undefined {
  const name = (control: { readonly label: string }) => control.label.split(" · ")[0]!.toLowerCase();
  const letters = (control: ToolControl) => control.kind === "choice"
    ? control.options.flatMap((option) => option.shortcut ? [option.shortcut.toUpperCase()] : []).join(" ")
    : control.shortcut?.toUpperCase() ?? "";
  switch (key) {
    case "operation": {
      const control = voxelToolKeyControls(tool.id).find((candidate) => candidate === operationControl);
      return control && `${letters(control)} attach · erase · paint`;
    }
    case "adjust": { const control = adjustableControl(tool, false); return control && `[ ] ${name(control)}`; }
    case "adjustSecondary": {
      const control = adjustableControl(tool, true);
      return control && control !== adjustableControl(tool, false) ? `⇧[ ] ${name(control)}` : undefined;
    }
    case "opposite": {
      if (!tool.ui.opposite) return undefined;
      const painting = tool.ui.controls.includes(operationControl) && tool.ui.controls.includes(materialControl)
        && toolOperation(values) === "paint";
      return painting ? "Alt-click picks a colour" : "Alt inverts";
    }
    // The scene focus (lib/core/editor-focus): the lock keys work under every tool that resolves through it.
    case "lock": return "X Y Z lock an axis · ⇧ a plane";
    case "cancel": return "esc cancels";
    case "undo": return "⌘Z undoes";
    default: {
      const control = tool.ui.controls.find((candidate) => `control:${candidate.id}` === key);
      const keys = control && letters(control);
      return keys ? `${keys} ${name(control)}` : undefined;
    }
  }
}

/**
 * The one line a tool teaches in this phase: the keys that apply right now,
 * then its own words. Undefined where it declares nothing for the phase.
 */
export function voxelToolHint(toolId: string | undefined, phase: ToolHintPhase, values: ToolValues): string | undefined {
  const tool = voxelTools.get(toolId);
  const declared = tool?.ui.hints?.[phase];
  const hint = typeof declared === "function" ? declared(values) : declared;
  if (!tool || !hint) return undefined;
  // Keys first, then the words: the keys are what the eye looks for once the words are known.
  const parts = [...(hint.keys ?? []).map((key) => hintKey(tool, key, values)), hint.text].filter(Boolean);
  return parts.length ? parts.join(" · ") : undefined;
}

export function voxelSculptActions(
  scene: SceneDescription,
  methodId: string | undefined,
  clipboard?: VoxelClipboard,
): readonly EditorAction[] {
  return voxelToolGroups(scene, methodId, clipboard).map((group) => ({
    id: group.id,
    label: group.label,
    icon: group.icon,
    tone: group.tone,
    hint: group.hint,
    children: group.tools.map((tool) => ({
      id: `voxel-tool-${tool.id}`,
      label: tool.label,
      iconPath: tool.iconPath,
      tone: group.tone,
      enabled: !tool.unavailable,
      hint: tool.hint,
      effect: { kind: "voxel-tool" as const, toolId: tool.id },
    })),
  }));
}
