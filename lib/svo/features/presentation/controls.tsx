"use client";

import { PipeChoice, PipeRange } from "../../../../components/PipeControls";
import type { SvoFeatureControlContext } from "../../pipeline/control-context";
import type { SvoOccluderGhosting } from "../../pipeline/svo-render-tuning";

const OCCLUDER_GHOSTING_OPTIONS: ReadonlyArray<{ value: SvoOccluderGhosting; label: string; hint: string }> = [
  { value: "auto", label: "Scene", hint: "Follow the scene: on for an authored voxel vessel, off elsewhere." },
  { value: "on", label: "On", hint: "Every opaque voxel standing between the camera and the water is drawn translucent." },
  { value: "off", label: "Off", hint: "Solids hide the water behind them, as authored." },
];

export function renderPresentControls({ tuning, updateTuning, modified, resetTuning }: Pick<SvoFeatureControlContext, "tuning" | "updateTuning" | "modified" | "resetTuning">) {
  return (<div className="pipe-fields">
      <PipeRange label="Render resolution" unit="%" value={tuning.resolutionScale * 100}
        min={35} max={100} step={1} digits={0}
        onChange={(value) => updateTuning("resolutionScale", value / 100)}
        modified={modified("resolutionScale")} onReset={resetTuning("resolutionScale")}
        hint="Pixel-linear: the frame costs roughly 19–22 ms per megapixel with cones and 13–15 without, so this is the one dial that moves every pass at once." />
      <PipeChoice label="See-through solids" value={tuning.occluderGhosting} options={OCCLUDER_GHOSTING_OPTIONS}
        onChange={(value) => updateTuning("occluderGhosting", value)} />
      <PipeRange label="See-through opacity" unit="%" value={tuning.occluderGhostOpacity * 100}
        min={0} max={90} step={5} digits={0} disabled={tuning.occluderGhosting === "off"}
        onChange={(value) => updateTuning("occluderGhostOpacity", value / 100)}
        modified={modified("occluderGhostOpacity")} onReset={resetTuning("occluderGhostOpacity")}
        hint="How much of a ghosted voxel stays over the water it would otherwise hide. A voxel with no water behind it along the view ray is never ghosted." />
    </div> );
}
