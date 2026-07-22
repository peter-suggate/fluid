import { WebGPUUniformEulerianSolver } from "../webgpu-uniform-eulerian";
import { numberValue, type MethodParamSpec, type MethodParamValues, type SimulationMethod } from "./types";
import type { GPUQuality } from "../tall-cell-grid";
import type { SceneDescription } from "../model";
import { sceneHasTerrain } from "../terrain";
import { fencedSparseAuthorityBringupEnabled } from "../gpu-startup";

const params: MethodParamSpec[] = [
  { kind: "select", key: "globalFineLevelSetFactor", label: "Surface tracking", default: "4", tier: "coarse", options: [{ value: "4", label: "4× fine band · paper default" }, { value: "off", label: "Coarse octree only · faster" }, { value: "8", label: "8× fine band · experimental" }], hint: "The paper tracks the interface on a separate 4× or 8× sparse narrow band. Coarse-only still uses the power-diagram pressure solve, but transports and redistances octree phi directly and cannot preserve sub-cell surface detail." },
  { kind: "select", key: "maximumLeafSize", label: "Largest pressure cell", default: "16", tier: "fine", options: [{ value: "2", label: "2³ finest cells" }, { value: "4", label: "4³ finest cells" }, { value: "8", label: "8³ finest cells" }, { value: "16", label: "16³ finest cells" }, { value: "32", label: "32³ finest cells" }], hint: "Largest dyadic octree cell away from interfaces. The topology remains strictly 2:1 graded for valid power-diagram stencils." },
  { kind: "number", key: "interfaceRefinementBandCells", label: "Pressure refinement band", unit: "finest cells", min: 0, max: 32, step: 1, digits: 0, default: 4, tier: "fine", hint: "Keeps the octree pressure grid fine around liquid and solid interfaces. This is distinct from the separate high-resolution surface-tracking band." }
];

const maximumLeafSize = (value: unknown): 2 | 4 | 8 | 16 | 32 => {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : 4;
  return numeric >= 32 ? 32 : numeric >= 16 ? 16 : numeric >= 8 ? 8 : numeric <= 2 ? 2 : 4;
};

const leafSolver = (value: unknown): "auto" | "dense" | "compact" | "chebyshev" | "mgpcg" | "megakernel" =>
  value === "dense" || value === "compact" || value === "chebyshev" || value === "mgpcg" || value === "megakernel" ? value : "auto";

const surfaceRefinementFactor = (value: unknown): 1 | 2 | 4 => Number(value) >= 4 ? 4 : Number(value) <= 1 ? 1 : 2;
const globalFineLevelSetFactor = (value: unknown): 4 | 8 | undefined =>
  Number(value) >= 8 ? 8 : Number(value) >= 4 ? 4 : undefined;
const powerDiagramProjection = (value: unknown): "off" | "mirror" | "authoritative" =>
  value === "authoritative" || value === "mirror" ? value : "off";
const powerMultigridHierarchy = (value: unknown): "aggregate-galerkin" | "paper-pyramid" =>
  value === "paper-pyramid" ? "paper-pyramid" : "aggregate-galerkin";
const sparseSurfaceBand = (value: unknown): "off" | "mirror" | "authoritative" => value === "authoritative" || value === "mirror" ? value : "off";
const brickAtlasMode = (value: unknown): "off" | "mirror" | "authoritative" =>
  value === "off" || value === false ? "off" : value === "authoritative" ? "authoritative" : "mirror";

const requiresCompatibilityGeometry = (scene: SceneDescription) => sceneHasTerrain(scene)
  || (scene.fluid.initialBrickSeeds_m?.length ?? 0) > 0;

const options = (scene: SceneDescription, quality: GPUQuality, values: MethodParamValues) => ({
  densitySharpening: false,
  velocityTransport: "maccormack" as const,
  brickSparseVelocityAdvection: values.brickSparseAdvection !== "off" && values.brickSparseAdvection !== false,
  brickSparseTransportPreparation: values.brickSparseTransport !== "off" && values.brickSparseTransport !== false,
  brickSparseOccupancyFluxPreparation: values.brickSparseOccupancyFlux !== "off" && values.brickSparseOccupancyFlux !== false,
  hydrostaticSplit: values.hydrostaticSplit === "on" || values.hydrostaticSplit === true,
  // Browser safe mode and the Dawn diagnostic can explicitly request phase
  // fences. The solver additionally forces them when timestamp-query is active.
  fencedInitialSparseAuthority: fencedSparseAuthorityBringupEnabled(
    typeof location === "undefined" ? "" : location.search,
  ) || (typeof process !== "undefined" && process.env?.FLUID_SAFE_BRINGUP === "1"),
  pressureIterations: numberValue(values, params, "pressureIterations"),
  // Compact power-face authority has no compatible secondary-particle
  // sampler. Keep the unreachable legacy component explicitly disabled.
  secondaryParticles: false,
  secondaryParticlesEnabled: false,
  octree: {
    pressureIterations: numberValue(values, params, "pressureIterations"),
    powerPcgIterationCap: numberValue(values, params, "powerPcgIterationCap"),
    powerBoundarySmoothingIterations: numberValue(values, params, "powerBoundarySmoothingIterations"),
    powerMultigridHierarchy: powerMultigridHierarchy(values.powerMultigridHierarchy),
    faceVelocityMirror: values.faceVelocityMirror === true || values.faceVelocityMirror === "on",
    faceVelocityRhs: values.faceVelocityRhs === true || values.faceVelocityRhs === "on",
    faceVelocityTransport: values.faceVelocityTransport !== false && values.faceVelocityTransport !== "off",
    maximumLeafSize: maximumLeafSize(values.maximumLeafSize ?? 16),
    adaptivity: numberValue(values, params, "adaptivity"),
    interfaceRefinementBandCells: numberValue(values, params, "interfaceRefinementBandCells"),
    surfaceDetailStrength: numberValue(values, params, "surfaceDetailStrength"),
    sparseSurfaceBand: sparseSurfaceBand(values.sparseSurfaceBand),
    surfaceRefinementFactor: surfaceRefinementFactor(values.surfaceRefinementFactor),
    // Terrain and explicitly seeded/imported shapes still enter through the
    // compatibility bootstrap. Do not allocate the factor-4 authority merely
    // because the balanced dam-break preset requests it.
    globalFineLevelSetFactor: requiresCompatibilityGeometry(scene)
      ? undefined : globalFineLevelSetFactor(values.globalFineLevelSetFactor),
    powerDiagramProjection: powerDiagramProjection(values.powerDiagramProjection),
    sparseSurfaceBandCells: numberValue(values, params, "sparseSurfaceBandCells"),
    sparseSurfacePageFraction: numberValue(values, params, "sparseSurfacePageFraction"),
    brickAtlas: brickAtlasMode(values.brickAtlas),
    brickPreActivation: values.brickPreActivation !== "off" && values.brickPreActivation !== false,
    brickSparseSurface: values.brickSparseSurface !== "off" && values.brickSparseSurface !== false,
    brickSparseExtrapolation: values.brickSparseExtrapolation !== "off" && values.brickSparseExtrapolation !== false,
    jacobiRelaxation: 0.8,
    extrapolationSweeps: 4,
    leafSolver: leafSolver(values.leafSolver),
    pressureWarmStart: values.pressureWarmStart !== "off" && values.pressureWarmStart !== false
  }
});

export const octreeMethod: SimulationMethod = {
  id: "octree",
  label: "Power-diagram octree",
  shortLabel: "Power octree",
  badge: "POWER OCTREE",
  description: "GPU-resident adaptive power cells with the paper's sparse-pyramid pressure solve and optional high-resolution interface band.",
  detail: "2:1-graded dyadic octree, power-cell face velocities, Section 4.3 sparse-pyramid hybrid PCG, factor-4 signed-distance narrow band by default, frame-lagged variational rigid-body coupling, and no topology readbacks",
  backend: "webgpu",
  qualityLabels: { balanced: "paper defaults", high: "paper defaults", ultra: "paper defaults" },
  showQualityControl: false,
  params,
  pressureMapping: "The paper's sparse-grid V-cycle preconditions the Section 4.3 power-diagram PCG solve. Eight symmetric boundary sweeps match the paper; a 128-iteration recorded tail provides a fail-closed safety cap beyond its reported 6–10 typical iterations.",
  presetFor: () => ({
    powerMultigridHierarchy: "paper-pyramid",
    powerPcgIterationCap: 128,
    powerBoundarySmoothingIterations: 8,
    pressureIterations: 128,
    adaptivity: 1,
    maximumLeafSize: "16",
    interfaceRefinementBandCells: 4,
    surfaceDetailStrength: 0,
    faceVelocityTransport: "on",
    sparseSurfaceBand: "off",
    surfaceRefinementFactor: "2",
    globalFineLevelSetFactor: "4",
    powerDiagramProjection: "authoritative",
    leafSolver: "auto",
    sparseSurfaceBandCells: 4,
    sparseSurfacePageFraction: 0.75,
    hydrostaticSplit: "off",
    brickAtlas: "mirror",
    brickSparseSurface: "on",
    brickSparseAdvection: "on",
    brickSparseTransport: "off",
    brickSparseOccupancyFlux: "off",
    brickSparseExtrapolation: "off",
    brickPreActivation: "on"
  }),
  createSolver: (device, scene, quality, values, onRigidLoads) => new WebGPUUniformEulerianSolver(device, scene, quality, onRigidLoads, options(scene, quality, values)),
  createSolverAsync: (device, scene, quality, values, onRigidLoads, onProgress, signal) => WebGPUUniformEulerianSolver.createAsync(
    device, scene, quality, onRigidLoads, options(scene, quality, values),
    (label, completed, total, phase, taskId) => onProgress({ phase: phase === "adaptive-topology" || phase === "secondary-particles" || phase === "allocation" || phase === "warmup" ? phase : "solver-pipelines", taskId, label, completed, total }),
    signal,
  )
};
