import type { FluidPipelineGraph } from "../../../core/fluid-pipeline";
import type { GPUTimestampPhase } from "../../../core/performance-trace";

export const MAC_PHASES = {
  transport: { id: "velocity-advection", label: "MAC velocity and level-set transport" },
  surface: { id: "fine-sdf-redistance", label: "MAC surface reinitialization and forces" },
  pressure: { id: "pressure-solve", label: "MAC ghost-fluid pressure solve" },
  projection: { id: "velocity-projection", label: "MAC projection and velocity extension" },
  publication: { id: "adaptive-publication", label: "MAC publication and diagnostics" },
} satisfies Record<string, GPUTimestampPhase>;

const summaries = [
  "RK2 backtracing with bounded MacCormack correction, or first-order semi-Lagrangian transport. Face velocities and vertex level set advance from the same old velocity.",
  "Reinitialize the level set while freezing vertices beside a zero crossing. Apply gravity and explicit molecular viscosity; surface tension enters the pressure boundary.",
  "Symmetric seven-point MAC Poisson operator with subcell ghost-fluid boundaries. Diagonally preconditioned conjugate gradients, verified with a fresh residual before projection. Negative pressures are permitted.",
  "Subtract the same discrete pressure gradient used to build the Poisson operator, then extend valid face velocities four cells into air.",
  "Publish the native vertex level set, derived liquid fraction, positive MAC faces, pressure and divergence through the shared renderer interfaces. Measure volume drift without correcting it.",
];
export const MAC_PIPELINE: FluidPipelineGraph = Object.freeze({
  methodId: "uniform-mac",
  bands: [{ id: "advance", label: "Uniform MAC / level set" }],
  stages: Object.values(MAC_PHASES).map((phase, i) => ({
    id: phase.id, label: phase.label, band: "advance", side: i % 2 ? "right" as const : "left" as const,
    phaseLabels: [phase.label], tip: { summary: summaries[i] }, state: () => "on" as const, chip: () => "",
  })),
});
