"use client";

import { useSession } from "../lib/core/session/session-context";
import { Field, FieldList, FieldNote } from "./ui";

/** Resolution is an accepted frame property, never a preparation/activation switch. */
export function UniformCoarseControl() {
  const session = useSession();
  const method = session.method();
  const scene = session.scene(state => state.scene);
  const info = session.diagnostics(state => state.gpuInfo);
  if (method.methodId !== "uniform-volume") return null;
  const regions = scene.fluid.refinementRegions?.length ?? 0;
  const ready = info?.uniformMixedGeneration !== undefined;
  const coarse = info?.uniformMixedCoarseTiles ?? 0;
  return <FieldList testId="uniform-resolution-control">
    <Field label="Simulation resolution" hint="Draw a resolution region to use larger cells. Edits preserve the running simulation.">
      <span>{!ready ? "Initializing resolution" : coarse ? "Mixed resolution" : "Fine resolution"}</span>
    </Field>
    <FieldNote>{ready
      ? `${info.uniformMixedFineTiles ?? 0} fine tiles · ${info.uniformMixedTransitionTiles ?? 0} transition tiles · ${coarse} coarse tiles`
      : "Resolution status appears when the simulation is ready."}</FieldNote>
    <FieldNote>{regions
      ? "Move, resize or remove regions to change resolution. Edits take effect after the current simulation step."
      : "No regions: fine cells throughout. Use the Region tool to draw a coarse area."}</FieldNote>
  </FieldList>;
}
