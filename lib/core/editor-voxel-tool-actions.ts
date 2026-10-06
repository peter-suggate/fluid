import type { EditorAction, EditorActionIcon, EditorActionTone } from "./editor-action";
import type { SceneDescription } from "./model";
import { adjustableControl } from "./voxel-editor/plugin";
import { voxelTools } from "./voxel-editor/registry";

/**
 * The sculpt tools, arranged for the surfaces that offer them.
 *
 * The registry owns the tools; this only arranges them, exactly as the old
 * shelf's chooser did — one entry per declared group. A plugin added to the
 * registry appears on every surface without a second table, and no tool id is
 * dispatched on: each entry's effect carries the id back to the one performer.
 *
 * Two surfaces compose from here: the scene's ring draws the groups as wedges
 * (`voxelSculptActions`), and the EDIT rail stands the solid tools down the
 * viewport's left edge (`voxelToolRail`) — the same groups, the same
 * availability, one composition. A tool that does not apply right now is drawn
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
  Construct: {
    rail: true,
    id: "sculpt-build",
    label: "Build",
    icon: "solid",
    tone: "body",
    hint: "Sculpt solid voxels: brush, box, sphere and wall strokes",
  },
  Subtract: {
    rail: true,
    id: "sculpt-carve",
    label: "Carve",
    icon: "erase",
    tone: "body",
    hint: "Cut into solids: carve, cut, drill and channel strokes",
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

export function voxelToolGroups(
  scene: SceneDescription,
  methodId: string | undefined,
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
          const unavailable = tool.unavailable({ scene, methodId: methodId ?? "" });
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
  /** The digit that arms it: its place on the rail, counted from one. Absent past nine. */
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
): readonly (Omit<VoxelToolGroup, "tools"> & { readonly tools: readonly VoxelRailTool[] })[] {
  let place = 0;
  return voxelToolGroups(scene, methodId)
    .filter((group) => GROUP_FACES[group.group]?.rail)
    .map((group) => ({ ...group, tools: group.tools.map((tool) => {
      place += 1;
      return { ...tool, key: place <= 9 ? String(place) : undefined };
    }) }));
}

const RAIL_TOOL_IDS: readonly string[] = [...new Set(voxelTools.tools.map((tool) => tool.ui.group))]
  .filter((group) => GROUP_FACES[group]?.rail)
  .flatMap((group) => voxelTools.tools.filter((tool) => tool.ui.group === group).map((tool) => tool.id));

/** The rail tool a digit arms, by place. The scene is not asked: a tool that cannot run still arms and says why. */
export function voxelRailToolForKey(key: string): string | undefined {
  return /^[1-9]$/.test(key) ? RAIL_TOOL_IDS[Number(key) - 1] : undefined;
}

/** The digit that arms a tool, when it stands on the rail. */
export function voxelRailKeyForTool(toolId: string): string | undefined {
  const place = RAIL_TOOL_IDS.indexOf(toolId);
  return place >= 0 && place < 9 ? String(place + 1) : undefined;
}

/**
 * One line saying what the hand can do while a tool is armed, most used first.
 *
 * Composed from the tool's own declaration — which numbers its brackets step —
 * so the line cannot promise a key the gesture host would not take. Its
 * toggles are not repeated: each already shows its letter on the tool's line.
 * The camera clauses are the host's and constant.
 */
export function voxelToolLegend(toolId: string | undefined): string | undefined {
  const tool = voxelTools.get(toolId);
  if (!tool) return undefined;
  const name = (control: { readonly label: string }) => control.label.split(" · ")[0]!.toLowerCase();
  const first = adjustableControl(tool, false), second = adjustableControl(tool, true);
  return [
    "⌥ opposite",
    first && `[ ] ${name(first)}`,
    second && second !== first && `⇧[ ] ${name(second)}`,
    "space-drag orbits",
    "right-click cancels",
  ].filter(Boolean).join(" · ");
}

export function voxelSculptActions(
  scene: SceneDescription,
  methodId: string | undefined,
): readonly EditorAction[] {
  return voxelToolGroups(scene, methodId).map((group) => ({
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
