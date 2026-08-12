"use client";

import { simulation } from "@/lib/simulation/controller";
import { useDisplayScene } from "@/lib/stores/scene-draft-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { sceneScaleOption, sceneScaleSummary, type SceneScaleAxis, type SceneScaleFactor } from "@/lib/scene-scale";
import { fluidBodyBox, fluidWaterVolume_m3 } from "@/lib/editor-fluid-body";

/**
 * Scale and water-volume controls, over the scene rather than inside the
 * configuration panel: these are the two edits worth making dozens of times
 * while watching the result, and a popover you have to open first is the wrong
 * home for them.
 *
 * Shown only while the tank or the water body is selected. The viewport is
 * already ringed with chrome on all four edges, and a permanent fourth cluster
 * is how a workbench stops being readable — so these ride the selection they
 * belong to. Selecting the tank is what puts its handles on the scene, so "edit
 * the world" is one decision rather than several scattered ones.
 *
 * The two scale rows are deliberately separate because they cost different
 * things, and the row says which: a world step keeps the lattice and re-seeds
 * the live solver, a detail step moves it and rebuilds the arenas. A step that
 * cannot be taken is disabled with the reason in its tooltip rather than
 * failing on click.
 */

const AXES: ReadonlyArray<{
  axis: SceneScaleAxis;
  label: string;
  hint: string;
}> = [
  {
    axis: "world",
    label: "WORLD",
    hint: "Container extents, fluid, and cell size together — lattice dimensions hold and the live solver re-seeds at the new metre scale.",
  },
  {
    axis: "detail",
    label: "DETAIL",
    hint: "Cell size at a fixed world size — eight times the cells per step, so the arenas are rebuilt. Pipelines are cached, so no shaders recompile.",
  },
];

export function SceneScaleOverlay() {
  const shapeMode = useUIStore((state) =>
    state.selection?.kind === "tank" || state.selection?.kind === "fluid-body");
  // The display scene, so the litres and the extents count up as a shape drag
  // is happening rather than jumping when it lands.
  const scene = useDisplayScene();
  const summary = sceneScaleSummary(scene);
  // Two questions, and they stopped having the same answer once a scene could
  // hold several bodies: the readout reports all the water in the document,
  // while the buttons scale the reservoir — which a seed-authored scene like the
  // twin dams simply does not have, so they disable rather than reshape a body
  // that is not there. Each body's own handles still resize it in the scene.
  const body = fluidBodyBox(scene);
  const water_m3 = fluidWaterVolume_m3(scene);
  const fluidEnabled = scene.systems?.fluid !== false;
  const [nx, ny, nz] = summary.dimensions;
  if (!shapeMode) return null;

  const step = (axis: SceneScaleAxis, factor: SceneScaleFactor) => {
    const option = sceneScaleOption(summary, axis, factor);
    const [ax, ay, az] = option.dimensions;
    return (
      <button
        key={`${axis}-${factor}`}
        type="button"
        disabled={!option.available}
        data-testid={`scene-scale-${axis}-${factor === 2 ? "up" : "down"}`}
        title={option.available
          ? `${factor === 2 ? "Double" : "Halve"} the ${axis === "world" ? "world and fluid size" : "cell size"} · ${ax}×${ay}×${az} cells`
          : `Unavailable — ${option.blocked}`}
        onClick={() => simulation.scaleScene(axis, factor)}
      >{factor === 2 ? "×2" : "÷2"}</button>
    );
  };

  return (
    <div className="scene-scale-overlay" data-testid="scene-scale-overlay">
      {AXES.map((entry) => (
        <div className="scene-scale-row" key={entry.axis} title={entry.hint}>
          <span className="scene-scale-label">{entry.label}</span>
          {step(entry.axis, 0.5)}
          <output>{entry.axis === "world"
            ? summary.extents_m.map((extent) => extent.toFixed(2)).join(" × ")
            : `${nx}×${ny}×${nz}`}</output>
          {step(entry.axis, 2)}
        </div>
      ))}
      {fluidEnabled && <div
        className="scene-scale-row"
        title="Grow or shrink the initial water body about its own centre. Drag the box handles in the scene to reshape it instead."
      >
        <span className="scene-scale-label">WATER</span>
        <button
          type="button"
          disabled={!body}
          data-testid="fluid-body-shrink"
          title="Halve the water body"
          onClick={() => simulation.scaleFluidBody(0.5)}
        >÷2</button>
        <output>{water_m3 > 0 ? `${(water_m3 * 1000).toFixed(1)} L` : "none"}</output>
        <button
          type="button"
          disabled={!body}
          data-testid="fluid-body-grow"
          title="Double the water body"
          onClick={() => simulation.scaleFluidBody(2)}
        >×2</button>
      </div>}
    </div>
  );
}
