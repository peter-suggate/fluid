"use client";

import { simulation } from "@/lib/simulation/controller";
import { useSceneStore } from "@/lib/stores/scene-store";
import { sceneScaleOption, sceneScaleSummary, type SceneScaleAxis, type SceneScaleFactor } from "@/lib/scene-scale";
import { fluidBodyBox, fluidBodyBoxVolume_m3 } from "@/lib/editor-fluid-body";

/**
 * Scale and water-volume controls, over the scene rather than inside the
 * configuration panel: these are the two edits worth making dozens of times
 * while watching the result, and a popover you have to open first is the wrong
 * home for them.
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
    hint: "Container extents and cell size together — the lattice, the arenas, and the compiled pipelines all hold, so this re-seeds instead of rebuilding.",
  },
  {
    axis: "detail",
    label: "DETAIL",
    hint: "Cell size at a fixed world size — eight times the cells per step, so the arenas are rebuilt. Pipelines are cached, so no shaders recompile.",
  },
];

export function SceneScaleOverlay() {
  const scene = useSceneStore((state) => state.scene);
  const summary = sceneScaleSummary(scene);
  const body = fluidBodyBox(scene);
  const fluidEnabled = scene.systems?.fluid !== false;
  const [nx, ny, nz] = summary.dimensions;

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
          ? `${factor === 2 ? "Double" : "Halve"} the ${axis === "world" ? "world size" : "cell size"} · ${ax}×${ay}×${az} cells`
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
        <output>{body ? `${(fluidBodyBoxVolume_m3(body) * 1000).toFixed(1)} L` : "none"}</output>
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
