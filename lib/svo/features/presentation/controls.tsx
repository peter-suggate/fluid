"use client";

import { useSvoRenderTuningValue } from "../../pipeline/tuning-binding";
import { RailRange } from "../../../../components/rail";
import type { SvoFeatureControlContext, SvoStageControls } from "../../pipeline/control-context";
import { LensBloomControls } from "../../../features/lens-bloom/ui";

export function renderPresentControls({ tuning, updateTuning, modified, resetTuning }: Pick<SvoFeatureControlContext, "tuning" | "updateTuning" | "modified" | "resetTuning">): SvoStageControls {
  return { node: <>
      <RailRange label="Render resolution" unit="%" value={tuning.resolutionScale * 100}
        min={35} max={200} step={1} digits={0}
        onChange={(value) => updateTuning("resolutionScale", value / 100)}
        modified={modified("resolutionScale")} onReset={resetTuning("resolutionScale")}
        hint="Above 100% supersamples geometry and reflections for smoother fine detail. 200% renders four pixels per display pixel and costs roughly four times the pixel work. Extra supersampling is capped at eight megapixels and device texture limits." />
      <LensBloomControls />
    </> };
}

/** Directly discoverable from the right-hand Performance menu. */
export function RenderResolutionControlRow(){
  const {value,set,modified,reset}=useSvoRenderTuningValue("resolutionScale");
  return <RailRange label="Render resolution" unit="%" value={value*100} min={35} max={200} step={1} digits={0}
    onChange={next=>set(next/100)} modified={modified} onReset={reset} testId="render-resolution"
    hint="Above 100% supersamples geometry and reflections. 200% requests four pixels per display pixel. Extra supersampling is capped at eight megapixels and device texture limits."/>;
}
