"use client";

import { useState } from "react";
import { PipelineCost } from "../../../components/PipelineCost";
import { useFluidPipelineCost, useSolverSetup } from "../../../components/fluid-pipeline-model";
import { RailChoice, RailLine, RailRange, RailReadout } from "../../../components/rail";
import { UniformDetailLines, UniformDetailStatus } from "../../../components/UniformCoarseControl";
import { NumberInput, Stepper } from "../../../components/ui";
import { useSession } from "../../core/session/session-context";
import { simulation } from "../../core/simulation/controller";
import { effectiveSimulationStep_s, methodPinsSimulationStep } from "../../core/simulation-step";
import { useSafeBrowserGPUBringup } from "../../core/use-safe-browser-gpu-bringup";
import { simulationFeature } from "./definition";

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
  const control = simulationFeature.controls[0];
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

const UNIFORM_METHODS = new Set(["uniform-volume", "uniform-narrow-band-flip"]);

/** What an advance costs: the total, then a line per band. */
export function SimulationCostLines() {
  const { methodId } = useSolverSetup();
  const cost = useFluidPipelineCost();
  return <>
    {UNIFORM_METHODS.has(methodId) && <UniformDetailStatus />}
    <RailReadout label="Advance" value={cost.advance} hint={cost.source} testId="fluid-advance-cost" />
    {cost.declared && <PipelineCost id="fluid-pipeline" bands={cost.bands} />}
  </>;
}

/** Rust/Wasm runtime values the method lets a reader move live. */
const CPU_PARAM_KEYS = new Set(["pressureIterations", "pressureRelativeTolerance", "selectorMode", "energyThreshold", "curvatureTolerance"]);

/** Whether More was left open, for the life of the page. */
let moreOpen = false;

/** Everything else the solve is set to, folded: grid, backend, budgets, detail coverage. */
export function SimulationMoreLines() {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  const patchScene = session.scene((state) => state.patchScene);
  const patchNumerics = session.scene((state) => state.patchNumerics);
  const info = session.diagnostics((state) => state.gpuInfo);
  const { methodId, method, values, backend, cpuPhysics } = useSolverSetup();
  const [open, setOpen] = useState(moreOpen);
  const setParam = (key: string, value: string | number) => simulation.setMethodParam(methodId, key, value, session.id);
  return <details className="rail-more" open={open} onToggle={(event) => { moreOpen = event.currentTarget.open; setOpen(moreOpen); }}>
    <summary>More</summary>
    {open && <>
      {info && <RailReadout label="Grid" testId="grid-readout"
        value={`${info.nx}×${info.ny}×${info.nz} · ${(info.allocatedBytes / 1073741824).toFixed(2)} GiB`}
        hint={`${info.cellCount.toLocaleString()} ${cpuPhysics ? "active sparse cells" : "samples"} · ${(info.allocatedBytes / 1048576).toFixed(1)} MiB allocated`} />}
      {backend && <RailChoice label={backend.label} value={String(values[backend.key])} testId="physics-backend-control"
        options={backend.options.map((option) => ({ ...option }))} onChange={(value) => setParam(backend.key, value)} />}
      {cpuPhysics && method.params.filter((spec) => CPU_PARAM_KEYS.has(spec.key)).map((spec) => spec.kind === "select"
        ? <RailChoice key={spec.key} label={spec.label} value={String(values[spec.key])} hint={spec.hint}
          options={spec.options.map((option) => ({ ...option }))} onChange={(value) => setParam(spec.key, value)} />
        : <RailRange key={spec.key} label={spec.label} unit={spec.unit} hint={spec.hint}
          value={Number(values[spec.key])} min={spec.min} max={spec.max} step={spec.step} digits={spec.digits ?? 0}
          onChange={(value) => setParam(spec.key, value)} />)}
      <RailLine label="Oracle cell" hint="Nominal resolution of the scene, in metres.">
        <NumberInput value={scene.nominalResolution.length_m} step={0.0025} min={0.0125} max={0.08} digits={4} unit="m"
          ariaLabel="Oracle cell" onChange={(value) => patchScene({ nominalResolution: { length_m: value } })} />
      </RailLine>
      {!cpuPhysics && <RailLine label="PCG budget" hint="Maximum pressure iterations per advance.">
        <NumberInput value={scene.numerics.pressureMaxIterations} step={20} min={8} max={1000} digits={0}
          ariaLabel="PCG budget" onChange={(value) => patchNumerics({ pressureMaxIterations: Math.round(value) })} />
      </RailLine>}
      {UNIFORM_METHODS.has(methodId) && <UniformDetailLines />}
    </>}
  </details>;
}
