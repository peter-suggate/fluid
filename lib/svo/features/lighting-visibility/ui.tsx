"use client";

import { useSession } from "../../../core/session/session-context";
import { ChoiceField, Switch } from "../../../../components/ui";
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
  return <Switch label={SVO_LIGHTING_VISIBILITY_FEATURE.controls[1].label} checked={checked} onChange={onChange} disabled={disabled} />;
}

export function SvoAmbientOcclusionControlRow() {
  const session = useSession();
  const checked = session.ui((state) => state.svoAmbientOcclusionEnabled);
  const onChange = session.ui((state) => state.setSvoAmbientOcclusionEnabled);
  const disabled = session.ui((state) => state.svoConeTracingMode === "off");
  return <Switch label={SVO_LIGHTING_VISIBILITY_FEATURE.controls[2].label} checked={checked} onChange={onChange} disabled={disabled} />;
}
