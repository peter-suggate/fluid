"use client";

import { RailLine } from "../../../components/rail";
import { Stepper } from "../../../components/ui";
import { useSession } from "../../core/session/session-context";
import { simulation } from "../../core/simulation/controller";
import { effectiveSimulationStep_s, methodPinsSimulationStep } from "../../core/simulation-step";
import { useSafeBrowserGPUBringup } from "../../core/use-safe-browser-gpu-bringup";
import { simulationClockFeature } from "./definition";

/** The step-size control's range, in milliseconds. */
const STEP_MIN_MS = 1;
const STEP_MAX_MS = 50;

/**
 * The fixed simulation step, shown as the *effective* step: when a method pins
 * the clock (the uniform paper profile), the number shown is the one the
 * simulation actually takes, and editing it releases the pin rather than
 * presenting a number that refuses to move.
 *
 * Halved and doubled rather than nudged by a millisecond: the step is a
 * log-scale knob — 4, 8, 16, 33 ms are the settings a reader moves between —
 * and printed to a tenth at most, because a second decimal is noise.
 */
export function SimulationStepLine() {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  const methodId = session.method((state) => state.methodId);
  const quality = session.method((state) => state.quality);
  const overrides = session.method((state) => state.overrides);
  // Bounded bring-up admits exactly the authored scene; its step is not ours to move.
  const locked = useSafeBrowserGPUBringup() !== false;
  const method = { methodId, quality, overrides };
  const step_ms = effectiveSimulationStep_s(scene, method) * 1000;
  const pinned = methodPinsSimulationStep(scene, method);
  const control = simulationClockFeature.controls[0];
  return <RailLine label={pinned ? `${control.label} · pinned` : control.label} testId="transport-step"
    hint={pinned
      ? "The active method pins the clock to this step; editing it releases the method to the scene-authored step."
      : control.hint}>
    <Stepper
      value={step_ms}
      onChange={(value_ms) => simulation.setStepSize(value_ms / 1000, session.id)}
      min={STEP_MIN_MS} max={STEP_MAX_MS}
      factor={2} grain={0.1}
      unit={control.unit}
      editable
      disabled={locked}
      ariaLabel="Simulation step in milliseconds"
      decreaseHint="Halve the simulation step"
      increaseHint="Double the simulation step"
    />
  </RailLine>;
}
