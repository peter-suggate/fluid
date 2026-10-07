import type { FluidPipelineGraph } from "../../core/fluid-pipeline";
import { MAC_PHASES } from "../mac-shared/pipeline";

export const APIC_PHASES = {
  transport: { ...MAC_PHASES.transport, label: "APIC particle to grid transfer" },
  surface: { ...MAC_PHASES.surface, label: "APIC geometry and grid forces" },
  pressure: { ...MAC_PHASES.pressure, label: "APIC incompressibility projection solve" },
  projection: { ...MAC_PHASES.projection, label: "APIC grid to particle transfer and advection" },
  publication: { ...MAC_PHASES.publication, label: "APIC surface publication" },
};
const summaries = [
  "Gather particle mass and momentum onto staggered MAC faces using quadratic B-spline weights and GPU particle bins. APIC includes affine gradients; PIC/FLIP saves the transferred velocity before grid forces.",
  "Classify the reconstructed liquid interface, apply gravity and viscosity, and prepare the capillary pressure boundary.",
  "Solve the ghost-fluid Poisson system with multigrid-preconditioned CG and verify a fresh divergence residual before accepting the step.",
  "Project and extend velocity, update particles using APIC, PIC or the blended FLIP grid velocity change, advect with RK2 and solid collision checks, then rebuild the surface and measure the material ledger.",
  "Publish the reconstructed vertex surface, liquid fractions and diagnostic fields through the shared water renderer only after an accepted frame.",
];
export const APIC_PIPELINE: FluidPipelineGraph = Object.freeze({
  methodId: "particle-apic", bands: [{ id: "advance", label: "APIC particles and MAC grid" }],
  stages: Object.values(APIC_PHASES).map((phase, i) => ({ id: phase.id, label: phase.label, band: "advance",
    side: i % 2 ? "right" as const : "left" as const, phaseLabels: [phase.label],
    tip: { summary: summaries[i] }, state: () => "on" as const, chip: () => "",
    controls: i === 0 ? [
      { kind: "param-range" as const, param: "cfl", label: "Advection CFL", min: 0.1, max: 4, step: 0.1, digits: 1, unit: "cells", editable: true },
    ] : i === 2 ? [
      { kind: "param-range" as const, param: "pressureTolerance", label: "Residual ceiling", min: 0.00001, max: 100, step: 0.00001, digits: 5, unit: "s⁻¹", editable: true },
      { kind: "param-range" as const, param: "pressureRelativeReduction", label: "Initial residual fraction", min: 0, max: 1, step: 0.01, digits: 2, unit: "×", editable: true },
      { kind: "param-range" as const, param: "pressureLimit", label: "MG-PCG iteration limit", min: 1, max: 128, step: 1, unit: "iterations", editable: true },
    ] : undefined,
  })),
});
