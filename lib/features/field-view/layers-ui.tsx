"use client";

import { useState } from "react";
import { Eye } from "lucide-react";
import {
  ToolstripRow, ToolstripMenuButton, ToolstripMenuItem, useToolstripSection,
} from "../../../components/toolstrip";
import { Choice, ControlRow, Select, Slider, Value } from "../../../components/ui";
import {
  IMPORTANCE_VIEW_OPTIONS, MIXED_TILE_LEGEND, VISUAL_LAYERS, importanceLegend, importanceView, isSliceLayer, layerOpacity,
  setImportanceView, pageView, setPageView, toggleVisualLayer,
  type ImportanceView, type PageView, type VisualLayerId, type VisualLayerState,
} from "../../core/visual-layers";

const PAGE_LEGEND = [
  { color: "#b39bea", label: "Resident" },
  { color: "#5c616b", label: "Solid" },
] as const;

interface VisualLayerRowsProps {
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
 * The same multi-select instrument in the 2D lab and 3D studio. It lists what
 * can be sampled on the slice plane (pages also offer 3D in the studio);
 * the particle display is a setting of its own
 * beside the surface mode (`SurfaceDisplayRow`).
 */
export function VisualLayerRows({ state, onChange, plane, hidden = [] }: VisualLayerRowsProps) {
  const [open, setOpen] = useState(false);
  const layers = VISUAL_LAYERS.filter(layer => isSliceLayer(layer.id) && !hidden.includes(layer.id));
  const enabled = layers.filter(layer => state.enabled.includes(layer.id)).length;
  const { claim } = useToolstripSection("visual-layers", () => setOpen(false));
  const menu = <ToolstripMenuButton
    label="Visual layers"
    hint="Combine any layers."
    open={open}
    onOpen={value => { claim(value); setOpen(value); }}
  >
    {layers.map(layer => {
      const selected = state.enabled.includes(layer.id);
      return <div key={layer.id} role="none" className="visual-layer-option">
        <ToolstripMenuItem
          multiple
          note={selected ? "✓" : undefined}
          label={layer.label}
          swatch={layer.color}
          title={layer.description}
          active={selected}
          onClick={() => onChange(toggleVisualLayer(state, layer.id))}
          testId={`visual-layer-${layer.id}`}
        />
        {selected && <ControlRow>
          <Slider
            min={0} max={1} step={0.05}
            value={layerOpacity(state, layer.id)}
            ariaLabel={`${layer.label} opacity`}
            onInput={value => onChange({
              ...state, opacity: { ...state.opacity, [layer.id]: value },
            })}
          />
          <Value value={`${Math.round(layerOpacity(state, layer.id) * 100)}%`} />
        </ControlRow>}
        {selected && plane && layer.id === "pages" && <Choice<PageView>
          ariaLabel="Domain pages view"
          value={pageView(state)}
          options={[{ value: "slice", label: "Slice" }, { value: "3d", label: "3D" }]}
          onChange={view => onChange(setPageView(state, view))}
        />}
        {selected && layer.id === "importance" && <Select<ImportanceView>
          ariaLabel="Detail importance criterion"
          hint="What the importance layer shows: every tile's highest-scoring criterion, or one criterion's score as a heat map."
          value={importanceView(state)}
          options={IMPORTANCE_VIEW_OPTIONS}
          onChange={view => onChange(setImportanceView(state, view))}
          testId="visual-layer-importance-view"
        />}
      </div>;
    })}
  </ToolstripMenuButton>;

  // A layer's key stands under the row, wrapped to the column's width: beside
  // the plane controls it made the row as wide as the picture.
  const shown = (id: VisualLayerId) => state.visible && !hidden.includes(id) && state.enabled.includes(id);
  const legend = (label: string, title: string, entries: ReadonlyArray<{ color: string; label: string; title?: string; wide?: boolean }>) =>
    <div className="toolstrip-legend" aria-label={label} title={title}>
      {entries.map(entry => <span key={entry.label} title={entry.title}>
        <i aria-hidden className={entry.wide ? "is-wide" : undefined} style={{ background: entry.color }} />{entry.label}
      </span>)}
    </div>;
  const planeControls = plane && state.visible && layers.some(l => state.enabled.includes(l.id) && (l.id !== "pages" || pageView(state) === "slice")) ? <>
    <Choice<"x" | "y" | "z">
      ariaLabel="Field view plane"
      value={plane.axis === "off" || plane.axis === "volume" ? "z" : plane.axis as "x" | "y" | "z"}
      options={[
        { value: "x", label: "X" }, { value: "y", label: "Y" }, { value: "z", label: "Z" },
      ]}
      onChange={plane.setAxis}
    />
    <Slider
      min={0} max={1} step={0.005}
      value={plane.slice}
      ariaLabel="Field slice"
      onInput={plane.setSlice}
    />
    <Value value={`${Math.round(plane.slice * 100)}%`} />
  </> : null;
  return <><ToolstripRow
    icon={<Eye size={14} />}
    name="Visual layers"
    hint="Hide or restore the selected layers."
    active={state.visible && enabled > 0}
    testId="visual-layers"
    onClick={() => onChange({ ...state, visible: !state.visible })}
    after={<>{menu}<span className="toolstrip-name">{enabled} {enabled === 1 ? "layer" : "layers"}</span></>}
  >
    {planeControls}
  </ToolstripRow>
  {shown("pages") && legend("Domain page states", "Current allocation. Resident pages may also support pressure and the interface. Air pages are hidden. 3D boxes show through the scene so the allocation remains visible.", PAGE_LEGEND)}
  {shown("tiles") && legend("Mixed Uniform tile reasons", "Why each h tile is h (the frame head's census), and which velocity sampler it took.", MIXED_TILE_LEGEND)}
  {shown("importance") && legend("Mixed Uniform detail importance", "The frame head census's detail importance. Every score is its measure over its threshold: 1 triggers.", importanceLegend(importanceView(state)))}
  </>;
}
