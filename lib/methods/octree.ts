import { WebGPUUniformEulerianSolver } from "../webgpu-uniform-eulerian";
import { numberValue, type MethodParamSpec, type MethodParamValues, type SimulationMethod } from "./types";
import type { GPUQuality } from "../tall-cell-grid";

const params: MethodParamSpec[] = [
  { kind: "number", key: "pressureIterations", label: "Pressure effort", unit: "equiv. sweeps", min: 16, max: 400, step: 8, digits: 0, default: 128, tier: "coarse", hint: "Jacobi-equivalent pressure effort. The parallel Chebyshev solver maps four equivalent sweeps to one polynomial pass; rigid pressure impulses use a frame-lagged partitioned exchange so the accelerated path stays active." },
  { kind: "number", key: "surfaceColumns", label: "Finest columns", unit: "columns", min: 1_000, max: 20_000, step: 500, digits: 0, default: 2_500, tier: "fine", hint: "Finest x/z lattice shared by the authoritative level set and octree owner map." },
  { kind: "number", key: "adaptivity", label: "Octree adaptivity", unit: "", min: 0, max: 1, step: 0.1, digits: 1, default: 1, tier: "coarse", hint: "Debug quality/performance sweep: 0 forces finest pressure cells everywhere; 1 enables full signed-distance-graded coarsening." },
  { kind: "select", key: "secondaryParticles", label: "Secondary liquid", default: "on", tier: "coarse", options: [{ value: "on", label: "Spray droplets" }, { value: "off", label: "Off" }], hint: "One-way GPU droplets preserve escaped splash detail without changing liquid mass or pressure." },
  { kind: "number", key: "secondaryParticleCapacity", label: "Particle budget", unit: "particles", min: 4_096, max: 65_536, step: 1_024, digits: 0, default: 16_384, tier: "fine", hint: "Fixed GPU ring capacity. Full rings overwrite the oldest slots without allocating or reading back." },
  { kind: "select", key: "maximumLeafSize", label: "Maximum leaf", default: "8", tier: "fine", options: [{ value: "2", label: "2³ cells" }, { value: "4", label: "4³ cells" }, { value: "8", label: "8³ cells" }, { value: "16", label: "16³ cells" }, { value: "32", label: "32³ cells" }], hint: "Largest dyadic pressure cell. Interface bands stay fine while distant bulk air, water, and solid regions can collapse to much larger cells, then enforce 2:1 balance." },
  { kind: "number", key: "interfaceRefinementBandCells", label: "Interface refinement band", unit: "cells", min: 0, max: 32, step: 1, digits: 0, default: 4, tier: "fine", hint: "Pure air, water, or solid leaves farther than this many finest cells from liquid or solid interfaces may stay at the maximum leaf size. Lower values make all bulk regions coarser." },
  { kind: "select", key: "pressureWarmStart", label: "Pressure warm start", default: "on", tier: "fine", options: [{ value: "on", label: "On (previous field)" }, { value: "off", label: "Off (cold start)" }], hint: "Seed each compacted leaf solve with the previous step's pressure instead of clearing to zero, so the polynomial refines an already-good field. The legacy dense ladder always cold-starts." }
];

const maximumLeafSize = (value: unknown): 2 | 4 | 8 | 16 | 32 => {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : 4;
  return numeric >= 32 ? 32 : numeric >= 16 ? 16 : numeric >= 8 ? 8 : numeric <= 2 ? 2 : 4;
};

const leafSolver = (value: unknown): "auto" | "dense" | "compact" | "chebyshev" | "megakernel" =>
  value === "dense" || value === "compact" || value === "chebyshev" || value === "megakernel" ? value : "auto";

const options = (quality: GPUQuality, values: MethodParamValues) => ({
  densitySharpening: false,
  velocityTransport: "maccormack" as const,
  pressureIterations: numberValue(values, params, "pressureIterations"),
  secondaryParticles: values.secondaryParticles !== "off" && values.secondaryParticles !== false,
  secondaryParticleCapacity: numberValue(values, params, "secondaryParticleCapacity"),
  tallCellSettings: { surfaceColumns: numberValue(values, params, "surfaceColumns") },
  octree: {
    pressureIterations: numberValue(values, params, "pressureIterations"),
    maximumLeafSize: maximumLeafSize(values.maximumLeafSize ?? 8),
    adaptivity: numberValue(values, params, "adaptivity"),
    interfaceRefinementBandCells: numberValue(values, params, "interfaceRefinementBandCells"),
    jacobiRelaxation: 0.8,
    extrapolationSweeps: 4,
    leafSolver: leafSolver(values.leafSolver),
    pressureWarmStart: values.pressureWarmStart !== "off" && values.pressureWarmStart !== false
  }
});

export const octreeMethod: SimulationMethod = {
  id: "octree",
  label: "GPU octree",
  shortLabel: "Octree",
  badge: "GPU OCTREE",
  description: "Fully GPU-resident 3D adaptive pressure cells driven by an independently transported signed-distance level set.",
  detail: "pressure-only dyadic octree, GPU-resident signed-distance sizing, finest interface band, coarse deep-liquid cells, strict 2:1 smoothing, level-set ghost-fluid free surfaces, conservative affine pressure prolongation to dense transport faces, finite-volume coarse/fine flux matching, row-parallel Chebyshev-Jacobi pressure solve, frame-lagged variational volume-of-solid rigid-body coupling with geometric displacement, and no topology readbacks",
  backend: "webgpu",
  qualityLabels: { balanced: "2.5k finest columns", high: "7k finest columns", ultra: "12.5k finest columns" },
  params,
  pressureMapping: "Pressure effort maps to a quarter as many Chebyshev-Jacobi polynomial passes; solved leaf fluxes are prolonged to interior fine faces without changing the leaf balance, and topology rebuild, solve, and traversal never leave the GPU.",
  presetFor: (quality) => ({
    pressureIterations: quality === "balanced" ? 128 : quality === "high" ? 320 : 400,
    surfaceColumns: quality === "balanced" ? 2_500 : quality === "high" ? 7_000 : 12_500,
    adaptivity: 1,
    secondaryParticles: "on",
    secondaryParticleCapacity: 16_384,
    maximumLeafSize: "8",
    interfaceRefinementBandCells: 4
  }),
  createSolver: (device, scene, quality, values, onRigidLoads) => new WebGPUUniformEulerianSolver(device, scene, quality, onRigidLoads, options(quality, values)),
  createSolverAsync: (device, scene, quality, values, onRigidLoads, onProgress) => WebGPUUniformEulerianSolver.createAsync(
    device, scene, quality, onRigidLoads, options(quality, values),
    (label, completed, total) => onProgress({ phase: label.includes("octree") || label.includes("adaptive") ? "adaptive-topology" : "solver-pipelines", label, completed, total })
  )
};
