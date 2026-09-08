"use client";

import { useState } from "react";
import { performEditorAction } from "../lib/core/editor-action-runtime";
import { voxelToolGroups, type VoxelToolGroup } from "../lib/core/editor-voxel-tool-actions";
import { useSession } from "../lib/core/session/session-context";
import { EditorActionGlyph, EditorActionPathGlyph } from "./EditorActionIcon";
import {
  ToolstripMenuButton,
  ToolstripMenuItem,
  ToolstripRow,
  useToolstripSection,
} from "./toolstrip";

/**
 * The sculpt tools, as rows on the EDIT strip.
 *
 * The making rows above answer "what can I add"; these answer "how do I shape
 * what is here", and they were unreachable from the column — arming a stroke
 * meant knowing to right-click first. The ring keeps its wedges; a row and a
 * wedge arming one tool is one state seen from two places, exactly as the
 * making rows argue.
 *
 * Everything drawn here is the registry's declaration: the groups, their order,
 * each tool's label, icon and availability. Clicking a row arms its group's
 * first runnable tool; the chevron is the rest of the group. A tool that
 * cannot run here still arms — its card says why, which is the reason it can
 * be reached at all.
 *
 * The Fluid group is not drawn: its shapes are the Drop-water row's own menu,
 * because to a reader they are water, not sculpture.
 */
function SculptGroupRow({ group }: { group: VoxelToolGroup }) {
  const session = useSession();
  const [picking, setPicking] = useState(false);
  const { claim } = useToolstripSection(group.id, () => setPicking(false));
  const pick = (open: boolean) => {
    claim(open);
    setPicking(open);
  };
  const first = group.tools.find((tool) => !tool.unavailable) ?? group.tools[0];
  if (!first) return null;
  return <ToolstripRow
    icon={<EditorActionGlyph name={group.icon} />}
    name={group.label}
    hint={first.unavailable ?? group.hint}
    testId={`scene-${group.id}-row`}
    onClick={() => performEditorAction({ kind: "voxel-tool", toolId: first.id }, session)}
    after={<ToolstripMenuButton
      label={`${group.label} tools`}
      hint="Every stroke in this group. Choosing one arms it."
      open={picking}
      testId={`scene-${group.id}-pick`}
      onOpen={pick}
    >
      {group.tools.map((tool) => <ToolstripMenuItem
        key={tool.id}
        icon={<EditorActionPathGlyph path={tool.iconPath} size={13} />}
        label={tool.label}
        title={tool.hint}
        testId={`scene-${group.id}-pick-${tool.id}`}
        onClick={() => {
          pick(false);
          performEditorAction({ kind: "voxel-tool", toolId: tool.id }, session);
        }}
      />)}
    </ToolstripMenuButton>}
  />;
}

export function SculptRows() {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  const methodId = session.method((state) => state.methodId);
  const groups = voxelToolGroups(scene, methodId).filter((group) => group.group !== "Fluid");
  return <>{groups.map((group) => <SculptGroupRow key={group.id} group={group} />)}</>;
}
