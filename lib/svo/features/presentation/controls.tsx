"use client";

import { RailRange } from "../../../../components/rail";
import type { SvoFeatureControlContext, SvoStageControls } from "../../pipeline/control-context";

export function renderPresentControls({ tuning, updateTuning, modified, resetTuning }: Pick<SvoFeatureControlContext, "tuning" | "updateTuning" | "modified" | "resetTuning">): SvoStageControls {
  return { node: <>
      <RailRange label="Render resolution" unit="%" value={tuning.resolutionScale * 100}
        min={35} max={200} step={1} digits={0}
        onChange={(value) => updateTuning("resolutionScale", value / 100)}
        modified={modified("resolutionScale")} onReset={resetTuning("resolutionScale")}
        hint="Above 100% supersamples geometry and reflections for smoother fine detail. 200% renders four pixels per display pixel and costs roughly four times the pixel work. Extra supersampling is capped at eight megapixels and device texture limits." />
    </> };
}
