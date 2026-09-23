"use client";

import { useState } from "react";
import { Activity } from "lucide-react";
import { ADAPTIVITY_PARAMS, ADAPTIVITY_MODES, adaptivityPrimaryControls, adaptivityControlEnabled } from "./definition";
import { useSession } from "../../../../core/session/session-context";
import { resolvedMethodValues } from "../../../../core/stores/method-store";
import { simulation } from "../../../../core/simulation/controller";
import { ToolstripMenuButton, ToolstripMenuItem, ToolstripRow,
  useToolstripSection } from "../../../../../components/toolstrip";
import { NumberInput, Select } from "../../../../../components/ui";

/** The criterion and its primary live dials belong beside the solver. */
export function AdaptiveMassToolstripRow() {
  const session = useSession();
  const methodState = session.method();
  const values = resolvedMethodValues(methodState);
  const [open, setOpen] = useState(false);
  const { claim } = useToolstripSection("adaptive-criterion", () => setOpen(false));
  if (methodState.methodId !== "adaptive-volume") return null;
  const pick = (next: boolean) => { claim(next); setOpen(next); };
  const modes = ADAPTIVITY_MODES;
  const mode = modes.find(option => option.value === values.selectorMode) ?? modes[0]!;
  const primary = adaptivityPrimaryControls(values);
  return <ToolstripRow
    icon={<Activity width={14} height={14} strokeWidth={1.7} aria-hidden />}
    name="Fluid adaptivity"
    hint="Choose the refinement criterion. E is finest kinetic energy, κ is curvature tolerance, and T is impact lookahead in seconds. Full controls are in Activity + resolution."
    testId="scene-adaptivity-row"
    after={<>
      <ToolstripMenuButton label="Adaptive criterion" open={open} onOpen={pick}
        testId="scene-adaptivity-pick">
        {modes.map(option => <ToolstripMenuItem key={option.value} label={option.label}
          title={option.label} active={mode.value === option.value}
          testId={`scene-adaptivity-pick-${option.value}`}
          onClick={() => {
            simulation.setMethodParam("adaptive-volume", "selectorMode", option.value, session.id);
            pick(false);
          }} />)}
      </ToolstripMenuButton>
      <span className="toolstrip-name">{mode.label}</span>
      <div className="toolstrip-dimensions">
        {primary.map(({ key, tag }) => {
          const spec = ADAPTIVITY_PARAMS.find(param => param.key === key);
          if (spec?.kind !== "number") return null;
          return <NumberInput key={key} tag={tag} value={Number(values[key])}
            min={spec.min} max={spec.max} step={spec.step ?? 0.01}
            ariaLabel={`${spec.label}${spec.unit ? ` (${spec.unit})` : ""}`}
            onChange={value => simulation.setMethodParam("adaptive-volume", key, value, session.id)} />;
        })}
      </div>
    </>}
  />;
}

/** Any panel can render an individual setting through the same session command. */
export function AdaptiveMassControlRow({control}: {control: import("../../../../framework/composition").ResolvedControl}) {
  const session = useSession();
  const method = session.method();
  const values = resolvedMethodValues(method);
  if (method.methodId !== "adaptive-volume" || !control.setting) return null;
  const key = control.setting;
  if (!adaptivityControlEnabled(key, values)) return null;
  const commit = (value: string | number) => simulation.setMethodParam("adaptive-volume", key, value, session.id);
  return <label className="toolstrip-row" title={control.hint}>
    <span>{control.label}</span>
    {control.kind === "choice" ? <Select ariaLabel={control.label} value={String(values[key])}
      options={control.options ?? []} onChange={commit} />
      : <NumberInput unit={control.unit} value={Number(values[key])}
        min={control.min} max={control.max} step={control.step ?? 0.01}
        ariaLabel={control.label} onChange={commit} />}
  </label>;
}
