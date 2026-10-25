"use client";

import type { ReactNode } from "react";
import { PerformanceControls, PerformanceLegend } from "../spatial-performance/ui";
import { RailChoice, RailKey, RailRange, RailRule, RailSelect, RailSub, RailToggle } from "../../../components/rail";
import { getMethod } from "../../core/method-registry";
import { useSession } from "../../core/session/session-context";
import {
  HEAT_LEGEND, IMPORTANCE_VIEW_OPTIONS, MIXED_TILE_LEGEND, VISUAL_LAYERS, importanceLegend, importanceView, isSliceLayer, layerOpacity,
  legacyVisualLayers, setImportanceView, layerView, setLayerView, toggleVisualLayer,
  type ImportanceView, type LayerView, type VisualLayerId, type VisualLayerState,
} from "../../core/visual-layers";

const PAGE_LEGEND = [
  { color: "#b39bea", label: "Resident" },
  { color: "#5c616b", label: "Solid" },
] as const;

interface VisualLayerBinding {
  readonly state: VisualLayerState;
  /** Layers this solver publishes no source for. */
  readonly hidden?: readonly VisualLayerId[];
  readonly onChange: (state: VisualLayerState) => void;
  readonly plane?: {
    axis: string;
    slice: number;
    setAxis: (axis: "x" | "y" | "z") => void;
    setSlice: (n: number) => void;
  };
}

/**
 * The pane's layer state, and where a change to it is written: the layers ride
 * the structure overlay on a slice plane, so a change also puts the overlay on
 * a plane if it was off or volumetric.
 */
function useVisualLayerBinding(): VisualLayerBinding {
  const session = useSession();
  const methodId = session.method((state) => state.methodId);
  const overlayMode = session.ui((state) => state.gridOverlayMode);
  const overlayAxis = session.ui((state) => state.gridOverlayAxis);
  const overlaySlice = session.ui((state) => state.gridOverlaySlice);
  const setOverlayAxis = session.ui((state) => state.setGridOverlayAxis);
  const setOverlaySlice = session.ui((state) => state.setGridOverlaySlice);
  const layers = session.ui(state => state.visualLayers);
  return {
    state: layers ?? { ...legacyVisualLayers(overlayMode), visible: overlayAxis !== "off" },
    onChange: visualLayers => session.ui.setState({
      visualLayers,
      gridOverlayMode: "structure",
      gridOverlayAxis: overlayAxis === "off" || overlayAxis === "volume" ? "z" : overlayAxis,
    }),
    plane: { axis: overlayAxis, slice: overlaySlice, setAxis: setOverlayAxis, setSlice: setOverlaySlice },
    // Layers the method publishes no source for would draw nothing or stale data.
    hidden: getMethod(methodId).capabilities?.visualLayers?.hidden ?? [],
  };
}

/** A shown layer's colour key, where its colours need one. */
function layerKey(id: VisualLayerId, state: VisualLayerState): ReactNode {
  switch (id) {
    case "performance": return <PerformanceLegend state={state} />;
    case "pages": return <RailKey label="Domain page states" entries={PAGE_LEGEND}
      hint="Current allocation. Resident pages may also support pressure and the interface. Air pages are hidden. 3D boxes show through the scene so the allocation remains visible." />;
    case "heat": return <RailKey label="Region work tiers" entries={HEAT_LEGEND}
      hint="Each 16h region's work tier this frame (the region heat), labelled with the resolution it is computed at (h, 4h, or 16h for a frozen cold region) once the region is large enough on screen. 3D boxes show through the scene; cold regions are faintest." />;
    case "tiles": return <RailKey label="Mixed Uniform tile reasons" entries={MIXED_TILE_LEGEND}
      hint="Why each h tile is h (the frame head's census), and which velocity sampler it took." />;
    case "importance": return <RailKey label="Mixed Uniform detail importance" entries={importanceLegend(importanceView(state))}
      hint="The frame head census's detail importance. Every score is its measure over its threshold: 1 triggers." />;
    default: return null;
  }
}

const sliceLayers = (hidden: readonly VisualLayerId[]) =>
  VISUAL_LAYERS.filter(layer => isSliceLayer(layer.id) && !hidden.includes(layer.id));

/**
 * The layer card's lines: whether the selected layers are drawn, the plane
 * they are drawn on, and every layer the solver publishes as a switch, its
 * opacity, view and key set in under it while it is on.
 *
 * The particle display is a layer too, placed in this card by the surface
 * display (`ParticleDisplayLines`).
 */
export function VisualLayerLines({ state, onChange, plane, hidden = [] }: VisualLayerBinding) {
  const layers = sliceLayers(hidden);
  const shown = (id: VisualLayerId) => state.visible && state.enabled.includes(id);
  const sliced = plane && state.visible && layers.some(l => state.enabled.includes(l.id) && layerView(state, l.id) === "slice");
  return <>
    <RailToggle label="Show layers" checked={state.visible} hint="Hide or restore the selected layers."
      testId="visual-layers" onChange={visible => onChange({ ...state, visible })} />
    {sliced && <>
      <RailChoice<"x" | "y" | "z"> label="Plane" ariaLabel="Field view plane"
        value={plane.axis === "off" || plane.axis === "volume" ? "z" : plane.axis as "x" | "y" | "z"}
        options={[{ value: "x", label: "X" }, { value: "y", label: "Y" }, { value: "z", label: "Z" }]}
        onChange={plane.setAxis} />
      <RailRange label="Slice" ariaLabel="Field slice" value={plane.slice} min={0} max={1} step={0.005}
        display={`${Math.round(plane.slice * 100)}%`} onInput={plane.setSlice} />
    </>}
    <RailRule label="Fields" />
    {layers.map(layer => {
      const selected = state.enabled.includes(layer.id);
      const opacity = layerOpacity(state, layer.id);
      return <div key={layer.id} className="rail-group">
        <RailToggle label={layer.label} swatch={layer.color} checked={selected} hint={layer.description}
          testId={`visual-layer-${layer.id}`} onChange={() => onChange(toggleVisualLayer(state, layer.id))} />
        {selected && <RailSub>
          <RailRange label="Opacity" ariaLabel={`${layer.label} opacity`} value={opacity} min={0} max={1} step={0.05}
            display={`${Math.round(opacity * 100)}%`}
            onInput={value => onChange({ ...state, opacity: { ...state.opacity, [layer.id]: value } })} />
          {plane && <RailChoice<LayerView> label="View" ariaLabel={`${layer.label} view`} value={layerView(state, layer.id)}
            options={[{ value: "slice", label: "Slice" }, { value: "3d", label: "3D" }]}
            onChange={view => onChange(setLayerView(state, layer.id, view))} />}
          {layer.id === "importance" && <RailSelect<ImportanceView> label="Criterion" ariaLabel="Detail importance criterion"
            hint="What the importance layer shows: every tile's highest-scoring criterion, or one criterion's score as a heat map."
            value={importanceView(state)} options={IMPORTANCE_VIEW_OPTIONS} testId="visual-layer-importance-view"
            onChange={view => onChange(setImportanceView(state, view))} />}
          {layer.id === "performance" && <PerformanceControls state={state} onChange={onChange} />}
          {shown(layer.id) && layerKey(layer.id, state)}
        </RailSub>}
      </div>;
    })}
  </>;
}

/** The keys of every shown layer, each under its layer's name. */
export function VisualLayerLegend({ state, hidden = [] }: Pick<VisualLayerBinding, "state" | "hidden">) {
  if (!state.visible) return null;
  const keyed = sliceLayers(hidden).filter(layer => state.enabled.includes(layer.id))
    .map(layer => ({ layer, key: layerKey(layer.id, state) })).filter(entry => entry.key !== null);
  return <>{keyed.map(({ layer, key }) => <div key={layer.id} className="rail-legend-group">
    <strong>{layer.label}</strong>
    {key}
  </div>)}</>;
}

/** The layer card, bound to the pane's layer state. */
export function VisualLayersControl() {
  return <VisualLayerLines {...useVisualLayerBinding()} />;
}

/** The keys the rail keeps in view while its cards are closed. */
export function VisualLayerLegendControl() {
  const { state, hidden } = useVisualLayerBinding();
  return <VisualLayerLegend state={state} hidden={hidden} />;
}
