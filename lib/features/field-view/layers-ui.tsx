"use client";

import { useState } from "react";
import { Eye } from "lucide-react";
import {
  ToolstripRow, ToolstripMenuButton, ToolstripMenuItem,
  ToolstripChoice, ToolstripScrub, useToolstripSection,
} from "../../../components/toolstrip";
import {
  VISUAL_LAYERS, layerOpacity, toggleVisualLayer, type VisualLayerState,
} from "../../core/visual-layers";

interface VisualLayerRowsProps {
  readonly state: VisualLayerState;
  readonly onChange: (state: VisualLayerState) => void;
  readonly plane?: {
    axis: string;
    slice: number;
    setAxis: (axis: "x" | "y" | "z") => void;
    setSlice: (n: number) => void;
  };
}

/** The same multi-select instrument in the 2D lab and 3D studio. */
export function VisualLayerRows({ state, onChange, plane }: VisualLayerRowsProps) {
  const [open, setOpen] = useState(false);
  const { claim } = useToolstripSection("visual-layers", () => setOpen(false));
  const menu = <ToolstripMenuButton
    label="Visual layers"
    hint="Combine any layers."
    open={open}
    onOpen={value => { claim(value); setOpen(value); }}
  >
    {VISUAL_LAYERS.map(layer => {
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
        {selected && <ToolstripScrub
          min={0} max={1} step={0.05}
          value={layerOpacity(state, layer.id)}
          ariaLabel={`${layer.label} opacity`}
          readout={`${Math.round(layerOpacity(state, layer.id) * 100)}%`}
          onChange={value => onChange({
            ...state, opacity: { ...state.opacity, [layer.id]: value },
          })}
        />}
      </div>;
    })}
  </ToolstripMenuButton>;

  return <ToolstripRow
    icon={<Eye size={14} />}
    name="Visual layers"
    hint="Hide or restore the selected layers."
    active={state.visible && state.enabled.length > 0}
    testId="visual-layers"
    onClick={() => onChange({ ...state, visible: !state.visible })}
    after={<>{menu}<span className="toolstrip-name">{state.enabled.length} {state.enabled.length === 1 ? "layer" : "layers"}</span></>}
  >
    {plane && state.visible && <>
      <ToolstripChoice
        ariaLabel="Field view plane"
        value={plane.axis === "off" || plane.axis === "volume" ? "z" : plane.axis}
        options={[
          { value: "x", label: "X" }, { value: "y", label: "Y" }, { value: "z", label: "Z" },
        ]}
        onChange={value => plane.setAxis(value as "x" | "y" | "z")}
      />
      <ToolstripScrub
        min={0} max={1} step={0.005}
        value={plane.slice}
        ariaLabel="Field slice"
        readout={`${Math.round(plane.slice * 100)}%`}
        onChange={plane.setSlice}
      />
    </>}
  </ToolstripRow>;
}
