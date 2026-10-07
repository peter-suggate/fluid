import type { FluidPipelineGraph } from "../../core/fluid-pipeline";
import type { GPUTimestampPhase } from "../../core/performance-trace";
export const SPH_PHASES = {
  substeps: { id: "velocity-advection", label: "SPH particle substeps" },
  publication: { id: "adaptive-publication", label: "SPH surface reconstruction and publication" },
} satisfies Record<string, GPUTimestampPhase>;
export const SPH_PIPELINE: FluidPipelineGraph = Object.freeze({
  methodId: "particle-sph", bands: [{ id: "advance", label: "Traditional weakly compressible SPH" }],
  stages: [
    { id: SPH_PHASES.substeps.id, label: SPH_PHASES.substeps.label, band: "advance", side: "left" as const,
      phaseLabels: [SPH_PHASES.substeps.label], state: () => "on" as const, chip: () => "",
      tip: { summary: "Build neighbor bins, sum poly6 density, evaluate explicit spiky pressure, viscosity and surface tension, choose a GPU stability-limited dt, and integrate particles. Each substep is one compute pass; no iterative pressure solve." },
      controls: [
        { kind: "param-range" as const, param: "soundSpeed", label: "Sound speed", min: 1, max: 100, step: 1, unit: "m/s", editable: true },
        { kind: "param-range" as const, param: "artificialViscosity", label: "Acoustic damping", min: 0, max: 1, step: 0.01, digits: 2, unit: "α", editable: true },
        { kind: "param-range" as const, param: "cfl", label: "Timestep safety", min: 0.05, max: 0.3, step: 0.01, digits: 2, editable: true },
      ] },
    { id: SPH_PHASES.publication.id, label: SPH_PHASES.publication.label, band: "advance", side: "right" as const,
      phaseLabels: [SPH_PHASES.publication.label], state: () => "on" as const, chip: () => "",
      tip: { summary: "Reconstruct the particle surface once per advance and publish shared water fields. Measure material conservation, escaped volume, maximum density compression and reconstructed volume independently." } },
  ],
});
