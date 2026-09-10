"use client";

import { ArrowDown } from "lucide-react";
import { ToolstripNumber, ToolstripRow } from "../../../components/toolstrip";
import { useSession } from "../../core/session/session-context";
import { simulation } from "../../core/simulation/controller";
import { gravityFeature } from "./definition";
import { GRAVITY_DIRECTIONS, gravityDirection, setGravityDirection, gravityEnabled, setGravity, toggleGravity } from "./state";

export function GravityRow() {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  const enabled = gravityEnabled(scene.fluid.gravity_m_s2);
  const control = gravityFeature.controls[0];
  const toggle = () => {
    const current = session.scene.getState().scene;
    simulation.beginEdit(gravityEnabled(current.fluid.gravity_m_s2) ? "Disable gravity" : "Enable gravity", session.id);
    simulation.commitEdit({ fluid: toggleGravity(current.fluid) }, { reseed: true }, session.id);
  };
  return <ToolstripRow
    icon={<ArrowDown width={14} height={14} strokeWidth={1.7} aria-hidden />}
    name={control.label}
    hint={control.hint}
    testId="scene-gravity-row"
    after={<div className="toolstrip-choice">
      <button type="button" className={enabled ? "active" : ""}
        aria-label={control.label} aria-pressed={enabled}
        title={enabled ? "Disable gravity" : "Enable gravity"}
        data-testid="scene-gravity-toggle" onClick={toggle}
      >Gravity {enabled ? "on" : "off"}</button>
      <GravityDirectionControl />
    </div>}
  />;
}

/** Expanded presentation uses the same pure edit command as the body editor. */
export function GravityYRow() {
  const session = useSession();
  const gravity = session.scene(state => state.scene.fluid.gravity_m_s2);
  const control = gravityFeature.controls[1];
  return <ToolstripRow icon={<ArrowDown width={14} height={14} aria-hidden />}
    name={control.label} hint="Vertical acceleration shared by every simulation method."
    after={<ToolstripNumber value={gravity.y} min={control.min} max={control.max}
      step={control.step} ariaLabel={`${control.label} (${control.unit})`}
      onCommit={value => {
        const fluid = session.scene.getState().scene.fluid;
        if (value === fluid.gravity_m_s2.y) return;
        simulation.beginEdit(`Set ${control.label}`, session.id);
        simulation.commitEdit({ fluid: setGravity(fluid, { ...fluid.gravity_m_s2, y: value }) },
          { reseed: true }, session.id);
      }} />}
  />;
}

function GravityDirectionControl() {
  const session = useSession();
  const fluid = session.scene(state => state.scene.fluid);
  const methodId = session.method(state => state.methodId);
  if (methodId !== "adaptive-mass") return null;
  const direction = gravityDirection(fluid);
  return <select className="toolstrip-gravity-direction" aria-label="Gravity direction" title="World direction; keeps gravity strength and remembers the choice while off" value={direction}
      data-testid="scene-gravity-direction" onChange={event => {
        const current = session.scene.getState().scene.fluid;
        simulation.beginEdit("Change gravity direction", session.id);
        simulation.commitEdit({ fluid: setGravityDirection(current, event.target.value) }, { reseed: true }, session.id);
      }}>
      {direction === "custom" && <option value="custom" disabled>Custom direction</option>}
      {GRAVITY_DIRECTIONS.map(choice => <option key={choice.id} value={choice.id}>{choice.label}</option>)}
    </select>;
}
