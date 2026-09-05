"use client";

import { useState } from "react";
import { Activity } from "lucide-react";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { useSession } from "../lib/core/session/session-context";
import { resolvedMethodValues } from "../lib/core/stores/method-store";
import { simulation } from "../lib/core/simulation/controller";
import { ToolstripMenuButton, ToolstripMenuItem, ToolstripNumber, ToolstripRow,
  useToolstripSection } from "./toolstrip";

/** The criterion and its primary live dials belong beside the solver. */
export function AdaptiveMassToolstripRow() {
  const session = useSession();
  const methodState = session.method();
  const values = resolvedMethodValues(methodState);
  const [open, setOpen] = useState(false);
  const { claim } = useToolstripSection("adaptive-criterion", () => setOpen(false));
  if (methodState.methodId !== "adaptive-mass") return null;
  const pick = (next: boolean) => { claim(next); setOpen(next); };
  const modes = [
    { value: "coarse-first", label: "Coarse first", hint: "Coarse still water; energy, curvature and approaching motion refine it." },
    { value: "activity", label: "Causal + proof", hint: "Legacy causal activity and accepted-surface proofs." },
    { value: "surface", label: "Surface distance", hint: "Always keep the liquid interface fine." },
  ];
  const mode = modes.find(option => option.value === values.selectorMode) ?? modes[0]!;
  const primary = mode.value === "coarse-first"
    ? [{ key: "energyThreshold", tag: "E" }, { key: "curvatureTolerance", tag: "κ" },
      { key: "anticipationSeconds", tag: "T" }]
    : mode.value === "activity"
      ? [{ key: "finestTravelCells", tag: "U" }, { key: "surfaceDisplacementToleranceCells", tag: "Δ" }]
      : [];
  return <ToolstripRow
    icon={<Activity width={14} height={14} strokeWidth={1.7} aria-hidden />}
    name="Fluid adaptivity"
    hint="Choose the refinement criterion. E is finest kinetic energy, κ is curvature tolerance, and T is impact lookahead in seconds. Full controls are in Activity + resolution."
    testId="scene-adaptivity-row"
    after={<>
      <ToolstripMenuButton label="Adaptive criterion" open={open} onOpen={pick}
        testId="scene-adaptivity-pick">
        {modes.map(option => <ToolstripMenuItem key={option.value} label={option.label}
          title={option.hint} active={mode.value === option.value}
          testId={`scene-adaptivity-pick-${option.value}`}
          onClick={() => {
            simulation.setMethodParam("adaptive-mass", "selectorMode", option.value, session.id);
            pick(false);
          }} />)}
      </ToolstripMenuButton>
      <span className="toolstrip-name">{mode.label}</span>
      <div className="toolstrip-dimensions">
        {primary.map(({ key, tag }) => {
          const spec = adaptiveMassMethod.params.find(param => param.key === key);
          if (spec?.kind !== "number") return null;
          return <ToolstripNumber key={key} tag={tag} value={Number(values[key])}
            min={spec.min} max={spec.max} step={spec.step ?? 0.01}
            ariaLabel={`${spec.label}${spec.unit ? ` (${spec.unit})` : ""}`}
            onCommit={value => simulation.setMethodParam("adaptive-mass", key, value, session.id)} />;
        })}
      </div>
    </>}
  />;
}
