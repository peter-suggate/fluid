"use client";

import { useState } from "react";
import { Eye } from "lucide-react";
import {
  ToolstripRow, ToolstripMenuButton, ToolstripMenuItem, useToolstripSection,
} from "../../../components/toolstrip";
import { Choice, ControlRow, Select, Slider, Value } from "../../../components/ui";
import {
  IMPORTANCE_VIEW_OPTIONS, MIXED_TILE_LEGEND, PARTICLE_VIEW_OPTIONS, VISUAL_LAYERS, importanceLegend, importanceView, layerOpacity,
  particleLegend, particleView, setImportanceView, setParticleView, toggleVisualLayer,
  type ImportanceView, type ParticleView, type VisualLayerId, type VisualLayerState,
} from "../../core/visual-layers";

const PAGE_LEGEND = [
  { color: "#1fc7a6", label: "Transport" }, { color: "#f29c29", label: "Sharpen only" }, { color: "#9475c2", label: "Resident" },
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

/** The same multi-select instrument in the 2D lab and 3D studio. */
export function VisualLayerRows({ state, onChange, plane, hidden = [] }: VisualLayerRowsProps) {
  const [open, setOpen] = useState(false);
  const { claim } = useToolstripSection("visual-layers", () => setOpen(false));
  const menu = <ToolstripMenuButton
    label="Visual layers"
    hint="Combine any layers."
    open={open}
    onOpen={value => { claim(value); setOpen(value); }}
  >
    {VISUAL_LAYERS.filter(layer => !hidden.includes(layer.id)).map(layer => {
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
  const planeControls = plane && state.visible ? <>
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
  // On the strip beside the plane, not in the menu: it is switched back and
  // forth against the moving picture, and the menu covers the picture.
  const particleControls = shown("particles") ? <Choice<ParticleView>
    ariaLabel="Particle view"
    value={particleView(state)}
    options={PARTICLE_VIEW_OPTIONS}
    onChange={view => onChange(setParticleView(state, view))}
  /> : null;
  // One child or none: the row reads any child as its open state.
  const controls = planeControls || particleControls ? <>{planeControls}{particleControls}</> : null;
  return <><ToolstripRow
    icon={<Eye size={14} />}
    name="Visual layers"
    hint="Hide or restore the selected layers."
    active={state.visible && state.enabled.length > 0}
    testId="visual-layers"
    onClick={() => onChange({ ...state, visible: !state.visible })}
    after={<>{menu}<span className="toolstrip-name">{state.enabled.length} {state.enabled.length === 1 ? "layer" : "layers"}</span></>}
  >
    {controls}
  </ToolstripRow>
  {shown("pages") && legend("Domain page states", "Last-step volume work. Resident pages may also support pressure and the interface. Absent pages are hidden.", PAGE_LEGEND)}
  {shown("tiles") && legend("Mixed Uniform tile reasons", "Why each h tile is h (the frame head's census), and which velocity sampler it took.", MIXED_TILE_LEGEND)}
  {shown("particles") && legend("Particle speed", "The method's own particles as spheres, painted by speed.", particleLegend(particleView(state)))}
  {shown("importance") && legend("Mixed Uniform detail importance", "The frame head census's detail importance. Every score is its measure over its threshold: 1 triggers.", importanceLegend(importanceView(state)))}
  </>;
}
