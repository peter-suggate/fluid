"use client";

import { PipeRange } from "../../../../components/PipeControls";
import type { SvoFeatureControlContext } from "../../pipeline/control-context";

export function renderPresentControls({ tuning, updateTuning, modified, resetTuning }: Pick<SvoFeatureControlContext, "tuning" | "updateTuning" | "modified" | "resetTuning">) {
  return (<div className="pipe-fields">
      <PipeRange label="Render resolution" unit="%" value={tuning.resolutionScale * 100}
        min={35} max={100} step={1} digits={0}
        onChange={(value) => updateTuning("resolutionScale", value / 100)}
        modified={modified("resolutionScale")} onReset={resetTuning("resolutionScale")}
        hint="Pixel-linear: the frame costs roughly 19–22 ms per megapixel with cones and 13–15 without, so this is the one dial that moves every pass at once." />
    </div> );
}
