"use client";

import { ArrowDown } from "lucide-react";
import { ToolstripRow } from "../../../components/toolstrip";
import { NumberInput, Select, ToggleButton } from "../../../components/ui";
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
    after={<>
      <ToggleButton pressed={enabled} onChange={toggle} ariaLabel={control.label}
        hint={enabled ? "Disable gravity" : "Enable gravity"} testId="scene-gravity-toggle"
      >Gravity {enabled ? "on" : "off"}</ToggleButton>
      <GravityDirectionControl />
    </>}
  />;
}

/** Expanded presentation uses the same pure edit command as the body editor. */
export function GravityYRow() {
  const session = useSession();
  const gravity = session.scene(state => state.scene.fluid.gravity_m_s2);
  const control = gravityFeature.controls[1];
  return <ToolstripRow icon={<ArrowDown width={14} height={14} aria-hidden />}
    name={control.label} hint="Vertical acceleration shared by every simulation method."
    after={<NumberInput value={gravity.y} min={control.min} max={control.max}
      step={control.step} ariaLabel={`${control.label} (${control.unit})`}
      onChange={value => {
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
  if (methodId !== "adaptive-mass" && methodId !== "adaptive-volume") return null;
  const direction = gravityDirection(fluid);
  // A vector no listed direction matches reads as "Custom direction", disabled:
  // it can be kept but not chosen, since there is no one vector it would mean.
  return <Select<string> ariaLabel="Gravity direction"
    hint="World direction; keeps gravity strength and remembers the choice while off" value={direction}
    customLabel="Custom direction" testId="scene-gravity-direction"
    options={GRAVITY_DIRECTIONS.map(choice => ({ value: choice.id, label: choice.label }))}
    onChange={value => {
      const current = session.scene.getState().scene.fluid;
      simulation.beginEdit("Change gravity direction", session.id);
      simulation.commitEdit({ fluid: setGravityDirection(current, value) }, { reseed: true }, session.id);
    }} />;
}
