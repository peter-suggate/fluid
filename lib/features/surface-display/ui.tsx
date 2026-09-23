"use client";
import { Waves } from "lucide-react";
import { useSession } from "../../core/session/session-context";
import { ToolstripRow } from "../../../components/toolstrip";
import { Choice } from "../../../components/ui";
import { surfaceDisplayFeature, type FluidSurfaceRenderMode } from "./definition";
export function SurfaceDisplayRow() {
  const session = useSession();
  const value = session.ui(state => state.fluidSurfaceRenderMode);
  const change = session.ui(state => state.setFluidSurfaceRenderMode);
  const control = surfaceDisplayFeature.controls[0];
  return <ToolstripRow name={control.label} hint={control.hint} testId="fluid-surface-render-row" icon={<Waves width={14} height={14} />}>
    <Choice<FluidSurfaceRenderMode> ariaLabel="Fluid surface render mode" value={value} options={control.options} onChange={change} />
  </ToolstripRow>;
}
