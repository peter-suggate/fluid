"use client";

import { useSession } from "../../../core/session/session-context";
import type { FeatureControlViewProps } from "../../../framework/ui/slot";
import { RailKey, RailRange, RailSelect, RailToggle } from "../../../../components/rail";
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
    <RailToggle label={control.label} checked={enabled} disabled={!raster && !enabled} testId="svo-ao-tap-overlay"
      hint={`${control.hint} Scene-normalized budget per receiver, before sharing or rejected samples. Coarse voxel AO is separate.`}
      onChange={value => setView(value ? "raster-ao-taps" : "off")} />
    {enabled && <RailKey label="AO tap budget legend" entries={SVO_AO_TAP_LEGEND.map(stop => ({ color: stop.color, label: stop.label }))} />}
    {!raster && <p className="tool-options-note">Select RASTER + AO to inspect its tap budget.</p>}
  </>;
}

export function SvoLightingVisibilityControlRow() {
  const session = useSession();
  const value = session.ui((state) => state.svoConeTracingMode);
  const onChange = session.ui((state) => state.setSvoConeTracingMode);
  return <RailSelect label={SVO_LIGHTING_VISIBILITY_FEATURE.controls[0].label}
    value={value} onChange={onChange} options={SVO_LIGHTING_VISIBILITY_OPTIONS} />;
}

export function SvoShadowsControlRow() {
  const session = useSession();
  const checked = session.ui((state) => state.svoShadowsEnabled);
  const onChange = session.ui((state) => state.setSvoShadowsEnabled);
  const disabled = session.ui((state) => state.svoConeTracingMode === "off");
  return <RailToggle label={SVO_LIGHTING_VISIBILITY_FEATURE.controls[1].label} checked={checked} onChange={onChange} disabled={disabled} />;
}

export function SvoAmbientOcclusionControlRow() {
  const session = useSession();
  const checked = session.ui((state) => state.svoAmbientOcclusionEnabled);
  const onChange = session.ui((state) => state.setSvoAmbientOcclusionEnabled);
  const disabled = session.ui((state) => state.svoConeTracingMode === "off");
  return <RailToggle label={SVO_LIGHTING_VISIBILITY_FEATURE.controls[2].label} checked={checked} onChange={onChange} disabled={disabled} />;
}

export function SvoLatticeVisibilityControlRow() {
  const session = useSession();
  const checked = session.ui((state) => state.svoLatticeVisibilityEnabled);
  const onChange = session.ui((state) => state.setSvoLatticeVisibilityEnabled);
  // Only the cone arm samples reduced visibility; EXACT and OFF have no source to choose.
  const disabled = session.ui((state) => state.svoConeTracingMode !== "cones");
  const control = SVO_LIGHTING_VISIBILITY_FEATURE.controls[3];
  return <RailToggle label={control.label} hint={control.hint} checked={checked} onChange={onChange} disabled={disabled}
    testId="svo-lattice-visibility" />;
}

/**
 * The distant contact-AO budget, a slider that holds its shape while it moves
 * — the readout is a fixed width and the reset keeps its place when there is
 * nothing to reset. The budget is a live uniform, so the frame follows the
 * thumb. Only RASTER + AO samples it, so under any other source it stands
 * dimmed and says why.
 */
export function SvoFarAoTapsControlRow({ control }: FeatureControlViewProps) {
  const session = useSession();
  const disabled = session.ui((state) => state.svoConeTracingMode !== "raster-ao");
  const { value, set, modified, reset } = useSvoRenderTuningValue("rasterFarAoTaps");
  const min = control.min ?? 0, max = control.max ?? 36, step = control.step ?? 4;
  return <>
    <RailRange label={control.label} value={value} min={min} max={max} step={step} digits={0} unit={control.unit}
      disabled={disabled} hint={control.hint} testId="svo-far-ao-taps"
      onInput={set} onChange={set} modified={modified} onReset={reset} />
    {disabled && <p className="tool-options-note">Only the RASTER + AO visibility source samples screen-space AO.</p>}
  </>;
}

/** Whether a still camera reuses the last frame's raster shadow and AO visibility. */
export function SvoStillViewReuseControlRow({ control }: FeatureControlViewProps) {
  const session = useSession();
  const disabled = session.ui((state) => state.svoConeTracingMode !== "raster-ao");
  const { value, set } = useSvoRenderTuningValue("rasterStillViewReuse");
  return <RailToggle label={control.label} checked={value} onChange={set} disabled={disabled} hint={control.hint}
    testId="svo-still-view-reuse" />;
}
