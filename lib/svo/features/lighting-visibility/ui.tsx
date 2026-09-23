"use client";

import { useSession } from "../../../core/session/session-context";
import { ChoiceField, SwitchField } from "../../../../components/ui";
import { SVO_LIGHTING_VISIBILITY_FEATURE, SVO_LIGHTING_VISIBILITY_OPTIONS } from "./definition";

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
