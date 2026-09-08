import type { EditorAction, EditorActionIcon, EditorActionTone } from "./editor-action";
import type { SceneDescription } from "./model";
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
 * (`voxelSculptActions`), and the container strip draws them as rows while the
 * viewport is in EDIT — the same groups, the same availability, one
 * composition. A tool that does not apply right now is drawn with its own
 * reason as the hint — never omitted — so the reason ("Choose Sparse CM12 to
 * edit solids while water runs") teaches the way in rather than hiding it.
 */
const GROUP_FACES: Record<string, {
  readonly id: string;
  readonly label: string;
  readonly icon: EditorActionIcon;
  readonly tone: EditorActionTone;
  readonly hint: string;
}> = {
  Construct: {
    id: "sculpt-build",
    label: "Build",
    icon: "solid",
    tone: "body",
    hint: "Sculpt solid voxels: brush, box, sphere and wall strokes",
  },
  Subtract: {
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
