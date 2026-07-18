import { WebGPUUniformEulerianSolver } from "../webgpu-uniform-eulerian";
import { numberValue, type MethodParamSpec, type MethodParamValues, type SimulationMethod } from "./types";
import type { GPUQuality } from "../tall-cell-grid";

const params: MethodParamSpec[] = [
  { kind: "number", key: "pressureIterations", label: "Pressure effort", unit: "equiv. sweeps", min: 16, max: 400, step: 8, digits: 0, default: 128, tier: "coarse", hint: "Jacobi-equivalent pressure effort. The parallel Chebyshev solver maps four equivalent sweeps to one polynomial pass; rigid pressure impulses use a frame-lagged partitioned exchange so the accelerated path stays active." },
  { kind: "number", key: "surfaceColumns", label: "Finest columns", unit: "columns", min: 1_000, max: 20_000, step: 500, digits: 0, default: 2_500, tier: "fine", hint: "Finest x/z lattice shared by the authoritative level set and octree owner map." },
  { kind: "number", key: "adaptivity", label: "Octree adaptivity", unit: "", min: 0, max: 1, step: 0.1, digits: 1, default: 1, tier: "coarse", hint: "Debug quality/performance sweep: 0 forces finest pressure cells everywhere; 1 enables full signed-distance-graded coarsening." },
  { kind: "select", key: "secondaryParticles", label: "Secondary liquid", default: "on", tier: "coarse", options: [{ value: "on", label: "Spray droplets" }, { value: "off", label: "Off" }], hint: "One-way GPU droplets preserve escaped splash detail without changing liquid mass or pressure." },
  { kind: "number", key: "secondaryParticleCapacity", label: "Particle budget", unit: "particles", min: 4_096, max: 65_536, step: 1_024, digits: 0, default: 16_384, tier: "fine", hint: "Fixed GPU ring capacity. Full rings overwrite the oldest slots without allocating or reading back." },
  { kind: "number", key: "secondaryParticleSurfaceCorrection", label: "Particle surface correction", unit: "", min: 0, max: 1, step: 0.1, digits: 1, default: 0, tier: "fine", hint: "Optionally folds only near-interface spray markers back into phi. Each substep is capped to 0.2 cell and detached droplets remain render-only; zero preserves the proven one-way path." },
  { kind: "select", key: "maximumLeafSize", label: "Maximum leaf", default: "8", tier: "fine", options: [{ value: "2", label: "2³ cells" }, { value: "4", label: "4³ cells" }, { value: "8", label: "8³ cells" }, { value: "16", label: "16³ cells" }, { value: "32", label: "32³ cells" }], hint: "Largest dyadic pressure cell. Interface bands stay fine while distant bulk air, water, and solid regions can collapse to much larger cells, then enforce 2:1 balance." },
  { kind: "number", key: "interfaceRefinementBandCells", label: "Interface refinement band", unit: "cells", min: 0, max: 32, step: 1, digits: 0, default: 4, tier: "fine", hint: "Pure air, water, or solid leaves farther than this many finest cells from liquid or solid interfaces may stay at the maximum leaf size. Lower values make all bulk regions coarser." },
  { kind: "number", key: "surfaceDetailStrength", label: "Dynamic surface detail", unit: "", min: 0, max: 1, step: 0.1, digits: 1, default: 0, tier: "fine", hint: "Widens the finest pressure band by up to eight cells where the surface is sharply curved or locally straining. Zero is the proven fixed-band path." },
  { kind: "select", key: "sparseSurfaceBand", label: "Sparse surface field", default: "authoritative", tier: "fine", options: [{ value: "authoritative", label: "Adaptive detail patches" }, { value: "mirror", label: "Mirror / parity" }, { value: "off", label: "Dense only" }], hint: "Allocates fine two-sided phi pages only where curvature or velocity strain needs detail. Calm planar regions retain the coarse field; mirror mode validates sparse sampling without independent transport." },
  { kind: "select", key: "surfaceRefinementFactor", label: "Surface refinement", default: "2", tier: "fine", options: [{ value: "1", label: "1× parity" }, { value: "2", label: "2× linear" }, { value: "4", label: "4× experimental" }], hint: "Fine samples per transport-cell edge inside resident surface pages. Memory follows active surface area instead of refining the full 3D domain." },
  { kind: "number", key: "sparseSurfaceBandCells", label: "Fine surface support", unit: "fine cells", min: 2, max: 16, step: 1, digits: 0, default: 4, tier: "fine", hint: "Minimum signed-distance support on both sides of phi=0. Velocity backtrace and stencil margins are added automatically." },
  { kind: "number", key: "sparseSurfacePageFraction", label: "Surface page budget", unit: "domain fraction", min: 0.1, max: 1, step: 0.05, digits: 2, default: 0.75, tier: "fine", hint: "Hard physical pool as a fraction of the virtual fine page lattice. Exhaustion is reported and atomically falls back to dense extraction; it never indexes beyond the pool." },
  { kind: "select", key: "pressureWarmStart", label: "Pressure warm start", default: "on", tier: "fine", options: [{ value: "on", label: "On (previous field)" }, { value: "off", label: "Off (cold start)" }], hint: "Seed each compacted leaf solve with the previous step's pressure instead of clearing to zero, so the polynomial refines an already-good field. The legacy dense ladder always cold-starts." }
];

const maximumLeafSize = (value: unknown): 2 | 4 | 8 | 16 | 32 => {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : 4;
  return numeric >= 32 ? 32 : numeric >= 16 ? 16 : numeric >= 8 ? 8 : numeric <= 2 ? 2 : 4;
};

const leafSolver = (value: unknown): "auto" | "dense" | "compact" | "chebyshev" | "megakernel" =>
  value === "dense" || value === "compact" || value === "chebyshev" || value === "megakernel" ? value : "auto";

const surfaceRefinementFactor = (value: unknown): 1 | 2 | 4 => Number(value) >= 4 ? 4 : Number(value) <= 1 ? 1 : 2;
const sparseSurfaceBand = (value: unknown): "off" | "mirror" | "authoritative" => value === "off" || value === "mirror" ? value : "authoritative";

const options = (quality: GPUQuality, values: MethodParamValues) => ({
  densitySharpening: false,
  velocityTransport: "maccormack" as const,
  pressureIterations: numberValue(values, params, "pressureIterations"),
  secondaryParticles: values.secondaryParticles !== "off" && values.secondaryParticles !== false,
  secondaryParticleCapacity: numberValue(values, params, "secondaryParticleCapacity"),
  secondaryParticleSurfaceCorrection: numberValue(values, params, "secondaryParticleSurfaceCorrection"),
  tallCellSettings: { surfaceColumns: numberValue(values, params, "surfaceColumns") },
  octree: {
    pressureIterations: numberValue(values, params, "pressureIterations"),
    maximumLeafSize: maximumLeafSize(values.maximumLeafSize ?? 8),
    adaptivity: numberValue(values, params, "adaptivity"),
    interfaceRefinementBandCells: numberValue(values, params, "interfaceRefinementBandCells"),
    surfaceDetailStrength: numberValue(values, params, "surfaceDetailStrength"),
    sparseSurfaceBand: sparseSurfaceBand(values.sparseSurfaceBand),
    surfaceRefinementFactor: surfaceRefinementFactor(values.surfaceRefinementFactor),
    sparseSurfaceBandCells: numberValue(values, params, "sparseSurfaceBandCells"),
    sparseSurfacePageFraction: numberValue(values, params, "sparseSurfacePageFraction"),
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
    secondaryParticleSurfaceCorrection: 0,
    maximumLeafSize: "8",
    interfaceRefinementBandCells: 4,
    surfaceDetailStrength: 0,
    sparseSurfaceBand: "authoritative",
    surfaceRefinementFactor: "2",
    sparseSurfaceBandCells: 4,
    sparseSurfacePageFraction: 0.75
  }),
  createSolver: (device, scene, quality, values, onRigidLoads) => new WebGPUUniformEulerianSolver(device, scene, quality, onRigidLoads, options(quality, values)),
  createSolverAsync: (device, scene, quality, values, onRigidLoads, onProgress) => WebGPUUniformEulerianSolver.createAsync(
    device, scene, quality, onRigidLoads, options(quality, values),
    (label, completed, total) => onProgress({ phase: label.includes("octree") || label.includes("adaptive") ? "adaptive-topology" : "solver-pipelines", label, completed, total })
  )
};
