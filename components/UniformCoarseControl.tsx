"use client";

import { useMemo } from "react";
import { useSession } from "../lib/core/session/session-context";
import { simulation } from "../lib/core/simulation/controller";
import { resolvedMethodValues } from "../lib/core/stores/method-store";
import { sceneLatticeDimensions, solidVoxelEditsForScene } from "../lib/core/scene-lattice";
import { sceneHasTerrain } from "../lib/core/terrain";
import { Field, FieldList, FieldNote, Button, ControlRow } from "./ui";

/** Requests are acknowledged by the active solver resolution. */
export function UniformCoarseControl() {
  const session = useSession();
  const method = session.method();
  const scene = session.scene(state => state.scene);
  const info = session.diagnostics(state => state.gpuInfo);
  const status = session.diagnostics(state => state.gpuStatus);
  // Eligibility may scan authored solid voxels; telemetry refreshes must not
  // repeat that work for an unchanged scene.
  const geometry = useMemo(() => {
    if (method.methodId !== "uniform-volume") return { dimensions: [], supported: false };
    const dimensions = sceneLatticeDimensions(scene);
    const supported = (scene.container.shape ?? "box") === "box"
      && !sceneHasTerrain(scene) && !scene.rigidBodies.length && !scene.fluid.inflow
      && !solidVoxelEditsForScene(scene).length
      && dimensions.every(n => n >= 32 && n % 4 === 0);
    return { dimensions, supported };
  }, [scene, method.methodId]);
  if (method.methodId !== "uniform-volume") return null;
  const values = resolvedMethodValues(method);
  const prepared = values.prepareCoarseSimulation === "on";
  const requested = values.coarseSimulation === "on";
  const active = info?.uniformSimulationCellScale === 4;
  const spansPages = geometry.dimensions.some(n => n > Number(values.pageSize));
  const supported = geometry.supported;
  const pending = requested !== active;
  const ready = status.state === "ready" && info?.uniformCoarsePrepared;
  return <FieldList testId="uniform-coarse-control"><Field label="Coarse simulation" hint="Run the whole simulation with cells four times wider. Switching preserves the liquid and simulation time.">
    <ControlRow>
      <Button disabled={!prepared && (!supported || !spansPages)}
        onClick={() => {
          // Clear the resolution request in the same event before rebuilding, so
          // preparing again always starts in fine mode.
          simulation.setMethodParam(method.methodId, "coarseSimulation", "off", session.id);
          simulation.setMethodParam(method.methodId, "prepareCoarseSimulation", prepared ? "off" : "on", session.id);
        }}>
        {prepared ? "Disable preparation (restart)" : "Prepare coarse switch (restart)"}
      </Button>
      {prepared && <Button disabled={!ready || pending}
        onClick={() => simulation.setMethodParam(method.methodId, "coarseSimulation", active ? "off" : "on", session.id)}>
        {pending ? "Switch pending" : active ? "Switch to fine" : "Switch to coarse"}
      </Button>}
    </ControlRow>
  </Field>
      <FieldNote>{!supported
        ? "Requires a box without terrain, bodies, solid edits or inflow; each grid axis must be at least 32 cells and divisible by four."
        : !spansPages ? "For a 32³ scene, choose 16³ pages in solver setup before preparing. Otherwise use a larger grid."
        : prepared && pending ? "Switch requested; waiting for the current simulation step to finish."
        : active ? "Coarse active · 4× cell width. Switching to fine preserves time; lost detail is reconstructed."
        : prepared ? "Fine active. Switch either way without restarting."
        : "Preparation restarts the scene and reserves about 1–3% more GPU memory."}</FieldNote>
  </FieldList>;
}
