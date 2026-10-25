"use client";

import { useSession } from "../../../core/session/session-context";
import type { FeatureControlViewProps } from "../../../framework/ui/slot";
import { ChoiceField, RangeField, ResetButton, Slider, Switch, SwitchField, Value } from "../../../../components/ui";
import { useSvoRenderTuningValue } from "../../pipeline/tuning-binding";
import { SVO_LIGHTING_VISIBILITY_FEATURE, SVO_LIGHTING_VISIBILITY_OPTIONS } from "./definition";
import { SVO_AO_TAP_LEGEND } from "../diagnostics/svo-render-diagnostics";

export function SvoAoTapOverlayControlRow({ control }: FeatureControlViewProps) {
  const session = useSession();
  const view = session.ui(state => state.svoStageView);
  const setView = session.ui(state => state.setSvoStageView);
  const raster = session.ui(state => state.svoConeTracingMode === "raster-ao");
  const enabled = view === "raster-ao-taps";
  return <>
    <div className="view-rail-toggle" title={control.hint} data-testid="svo-ao-tap-overlay">
      <span>{control.label}</span>
      <Switch checked={enabled} disabled={!raster && !enabled} ariaLabel={control.label}
        onChange={value => setView(value ? "raster-ao-taps" : "off")} />
    </div>
    {enabled && <div className="view-rail-ao-legend" aria-label="AO tap budget legend">
      {SVO_AO_TAP_LEGEND.map(stop => <span key={stop.at}><i style={{backgroundColor:stop.color}} aria-hidden />{stop.label}</span>)}
    </div>}
    <p className="tool-options-note">{!raster ? "Select RASTER + AO to inspect its tap budget." : "Scene-normalized budget per receiver, before sharing or rejected samples. Coarse voxel AO is separate."}</p>
  </>;
}

export function SvoLightingVisibilityControlRow() {
  const session = useSession();
  const value = session.ui((state) => state.svoConeTracingMode);
  const onChange = session.ui((state) => state.setSvoConeTracingMode);
  return <ChoiceField label={SVO_LIGHTING_VISIBILITY_FEATURE.controls[0].label}
    value={value} onChange={onChange} options={SVO_LIGHTING_VISIBILITY_OPTIONS} />;
}

export function SvoShadowsControlRow() {
  const session = useSession();
  const checked = session.ui((state) => state.svoShadowsEnabled);
  const onChange = session.ui((state) => state.setSvoShadowsEnabled);
  const disabled = session.ui((state) => state.svoConeTracingMode === "off");
  return <SwitchField label={SVO_LIGHTING_VISIBILITY_FEATURE.controls[1].label} checked={checked} onChange={onChange} disabled={disabled} />;
}

export function SvoAmbientOcclusionControlRow() {
  const session = useSession();
  const checked = session.ui((state) => state.svoAmbientOcclusionEnabled);
  const onChange = session.ui((state) => state.setSvoAmbientOcclusionEnabled);
  const disabled = session.ui((state) => state.svoConeTracingMode === "off");
  return <SwitchField label={SVO_LIGHTING_VISIBILITY_FEATURE.controls[2].label} checked={checked} onChange={onChange} disabled={disabled} />;
}

export function SvoLatticeVisibilityControlRow() {
  const session = useSession();
  const checked = session.ui((state) => state.svoLatticeVisibilityEnabled);
  const onChange = session.ui((state) => state.setSvoLatticeVisibilityEnabled);
  // Only the cone arm samples reduced visibility; EXACT and OFF have no source to choose.
  const disabled = session.ui((state) => state.svoConeTracingMode !== "cones");
  const control = SVO_LIGHTING_VISIBILITY_FEATURE.controls[3];
  return <SwitchField label={control.label} hint={control.hint} checked={checked} onChange={onChange} disabled={disabled}
    testId="svo-lattice-visibility" />;
}

/**
 * The distant contact-AO budget. Expanded, a slider among the pipeline's
 * lighting fields; compact, a slider on the view rail's line that holds its
 * shape while it moves — the readout is a fixed width and the reset keeps its
 * place when there is nothing to reset. The budget is a live uniform, so the
 * frame follows the thumb. Only RASTER + AO samples it, so under any other
 * source it stands dimmed and says why.
 */
export function SvoFarAoTapsControlRow({ control, placement }: FeatureControlViewProps) {
  const session = useSession();
  const disabled = session.ui((state) => state.svoConeTracingMode !== "raster-ao");
  const { value, set, modified, reset } = useSvoRenderTuningValue("rasterFarAoTaps");
  const min = control.min ?? 0, max = control.max ?? 36, step = control.step ?? 4;
  if (placement.presentation !== "compact") {
    return <RangeField label={control.label} unit={control.unit} value={value} min={min} max={max} step={step} digits={0}
      disabled={disabled} onChange={set} modified={modified} onReset={reset} hint={control.hint} testId="svo-far-ao-taps" />;
  }
  return <>
    <div className="view-rail-range" title={control.hint} data-testid="svo-far-ao-taps">
      <span>{control.label}</span>
      <Slider value={value} min={min} max={max} step={step} disabled={disabled} ariaLabel={control.label}
        onInput={set} onChange={set} />
      <Value value={value} digits={0} unit={control.unit} />
      <ResetButton onReset={reset} ariaLabel={`Reset ${control.label}`}
        className={modified ? undefined : "is-idle"} />
    </div>
    {disabled && <p className="tool-options-note">Only the RASTER + AO visibility source samples screen-space AO.</p>}
  </>;
}

/**
 * Whether a still camera reuses the last frame's raster shadow and AO
 * visibility. Compact, a switch whose knob stands where the rail's readouts
 * end, so a line of toggles and sliders keeps one right edge.
 */
export function SvoStillViewReuseControlRow({ control, placement }: FeatureControlViewProps) {
  const session = useSession();
  const disabled = session.ui((state) => state.svoConeTracingMode !== "raster-ao");
  const { value, set } = useSvoRenderTuningValue("rasterStillViewReuse");
  if (placement.presentation !== "compact") {
    return <SwitchField label={control.label} hint={control.hint} checked={value} onChange={set} disabled={disabled}
      testId="svo-still-view-reuse" />;
  }
  return <div className="view-rail-toggle" title={control.hint} data-testid="svo-still-view-reuse">
    <span>{control.label}</span>
    <Switch checked={value} onChange={set} disabled={disabled} ariaLabel={control.label} hint={control.hint} />
  </div>;
}
