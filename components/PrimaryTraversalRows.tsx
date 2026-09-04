"use client";

import { useState, type ComponentType } from "react";
import { Activity, Boxes, Eye, Grid3X3, Network, ScanLine } from "lucide-react";
import { useSession } from "../lib/core/session/session-context";
import {
  SVO_PRIMARY_WORK_STAGE_VIEWS,
  SVO_RENDER_STAGE_DEFINITIONS,
  svoRenderStageUsesPrimaryWorkMap,
  type SvoRenderStageView,
} from "../lib/svo/svo-render-diagnostics";
import {
  ToolstripMenuButton,
  ToolstripMenuItem,
  ToolstripRow,
  useToolstripSection,
} from "./toolstrip";

type WorkView = typeof SVO_PRIMARY_WORK_STAGE_VIEWS[number];
type WorkIcon = ComponentType<{
  width?: number;
  height?: number;
  strokeWidth?: number;
  "aria-hidden"?: boolean;
}>;

const ICONS: Readonly<Record<WorkView, WorkIcon>> = {
  "primary-work": Activity,
  "primary-voxel-cells": Grid3X3,
  "primary-node-visits": Network,
  "primary-leaf-visits": Boxes,
  "primary-entry-tail": ScanLine,
};

const DEFAULT_WORK_VIEW: WorkView = "primary-work";

/**
 * Exact primary-ray work as one ordinary scene-toolstrip row.
 *
 * This deliberately has the same anatomy as the field-view row: the current
 * picture is the toggle, and its chevron swaps that picture for another. The
 * work counter plane is the expensive state, so selecting Scene turns it off;
 * changing between work pictures keeps the one instrumented primary variant.
 */
export function PrimaryTraversalRow() {
  const session = useSession();
  const stageView = session.ui((state) => state.svoStageView);
  const setStageView = session.ui((state) => state.setSvoStageView);
  const active = svoRenderStageUsesPrimaryWorkMap(stageView);
  const [lastView, setLastView] = useState<WorkView>(DEFAULT_WORK_VIEW);
  const [picking, setPicking] = useState(false);
  const { claim } = useToolstripSection("primary-ray-work", () => setPicking(false));
  const pick = (open: boolean) => {
    claim(open);
    setPicking(open);
  };
  const shown = active ? stageView as WorkView : lastView;
  const ShownIcon = ICONS[shown];

  const choose = (view: SvoRenderStageView) => {
    if (svoRenderStageUsesPrimaryWorkMap(view)) setLastView(view as WorkView);
    setStageView(view);
    pick(false);
  };

  return <ToolstripRow
    icon={<ShownIcon width={14} height={14} strokeWidth={1.7} aria-hidden />}
    name={active ? SVO_RENDER_STAGE_DEFINITIONS[shown].label : "Primary ray work"}
    hint={active
      ? "Click to restore the scene. Exact counters change the measured frame."
      : `Click to show ${SVO_RENDER_STAGE_DEFINITIONS[shown].label.toLowerCase()}. Exact counters change the measured frame.`}
    active={active}
    testId="primary-traversal-toggle"
    onClick={() => setStageView(active ? "off" : shown)}
    after={<ToolstripMenuButton
      label="Primary traversal visual"
      hint="Live exact node, leaf and voxel-cell counts; all use the hero-garden-hose p99 reference scale."
      open={picking}
      testId="primary-traversal-chevron"
      onOpen={pick}
    >
      <ToolstripMenuItem
        icon={<Eye width={13} height={13} strokeWidth={1.7} aria-hidden />}
        label="Scene"
        note="no counters"
        title="Restore the composited frame and the normal uncounted primary renderer."
        active={!active}
        testId="primary-traversal-pick-off"
        onClick={() => choose("off")}
      />
      {SVO_PRIMARY_WORK_STAGE_VIEWS.map((view) => {
        const Icon = ICONS[view];
        const definition = SVO_RENDER_STAGE_DEFINITIONS[view];
        return <ToolstripMenuItem
          key={view}
          icon={<Icon width={13} height={13} strokeWidth={1.7} aria-hidden />}
          label={definition.label}
          title={definition.description}
          active={stageView === view}
          testId={`primary-traversal-pick-${view}`}
          onClick={() => choose(view)}
        />;
      })}
    </ToolstripMenuButton>}
  />;
}
