"use client";

import { useSession } from "../../../core/session/session-context";
import { ChoiceField } from "../../../../components/ui";
import { SVO_RADIANCE_RECONSTRUCTION_FEATURE, SVO_RADIANCE_RECONSTRUCTION_OPTIONS } from "./definition";

export function SvoRadianceReconstructionControlRow() {
  const session = useSession();
  const tuning = session.ui(state => state.svoRenderTuning);
  const lighting = session.ui(state => state.svoConeTracingMode);
  const setTuning = session.ui(state => state.setSvoRenderTuning);
  return <ChoiceField label={SVO_RADIANCE_RECONSTRUCTION_FEATURE.controls[0].label}
    value={tuning.coneRadianceReconstruction} disabled={tuning.coneLightingScale === 1 || lighting !== "cones"}
    options={SVO_RADIANCE_RECONSTRUCTION_OPTIONS}
    onChange={coneRadianceReconstruction => setTuning(current => ({ ...current, coneRadianceReconstruction }))} />;
}
