import type { EditorAction, EditorActionIcon, EditorActionTone } from "./editor-action";
import type { SceneDescription } from "./model";
import { voxelTools } from "./voxel-editor/registry";

/**
 * The sculpt tools, as wedges on the scene's ring.
 *
 * The registry owns the tools; this only arranges them, exactly as the old
 * shelf's chooser did — one wedge per declared group, each opening the ring of
 * that group's tools. A plugin added to the registry appears here without a
 * second table, and no tool id is dispatched on: the wedge's effect carries the
 * id back to the one performer.
 *
 * A tool that does not apply right now is drawn disabled with its own reason as
 * the hint — the ring's standing rule, so the menu keeps a learnable shape and
 * the reason ("Choose Sparse CM12 to edit solids while water runs") teaches the
 * way in rather than hiding it.
 */
const GROUP_WEDGES: Record<string, {
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

export function voxelSculptActions(
  scene: SceneDescription,
  methodId: string | undefined,
): readonly EditorAction[] {
  const groups = [...new Set(voxelTools.tools.map((tool) => tool.ui.group))];
  return groups.map((group) => {
    const wedge = GROUP_WEDGES[group]
      ?? { id: `sculpt-${group.toLowerCase()}`, label: group, icon: "solid" as const, tone: "body" as const, hint: group };
    return {
      id: wedge.id,
      label: wedge.label,
      icon: wedge.icon,
      tone: wedge.tone,
      hint: wedge.hint,
      children: voxelTools.tools
        .filter((tool) => tool.ui.group === group)
        .map((tool) => {
          const unavailable = tool.unavailable({ scene, methodId: methodId ?? "" });
          return {
            id: `voxel-tool-${tool.id}`,
            label: tool.ui.label,
            iconPath: tool.ui.icon,
            tone: wedge.tone,
            enabled: !unavailable,
            hint: unavailable ?? tool.ui.hint,
            effect: { kind: "voxel-tool" as const, toolId: tool.id },
          };
        }),
    };
  });
}
