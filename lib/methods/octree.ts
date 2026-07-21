import { WebGPUUniformEulerianSolver } from "../webgpu-uniform-eulerian";
import { numberValue, type MethodParamSpec, type MethodParamValues, type SimulationMethod } from "./types";
import type { GPUQuality } from "../tall-cell-grid";
import type { SceneDescription } from "../model";
import { sceneHasTerrain } from "../terrain";

const params: MethodParamSpec[] = [
  { kind: "number", key: "pressureIterations", label: "Pressure effort", unit: "iterations", min: 16, max: 400, step: 8, digits: 0, default: 128, tier: "coarse", hint: "The paper-authoritative Section 4.3 solve is capped at 16 PCG iterations (reported convergence: 6–10); converged GPU iterations become no-ops. The same effort remains the Chebyshev rollback budget." },
  { kind: "select", key: "leafSolver", label: "Pressure solver", default: "auto", tier: "fine", options: [{ value: "auto", label: "Auto · Section 4.3 for power" }, { value: "mgpcg", label: "Section 4.3 hybrid" }, { value: "chebyshev", label: "Chebyshev rollback" }], hint: "Auto selects the paper's Section 4.3 hybrid PCG preconditioner only when power authority is admitted. Unsupported scenes fail closed to the compact Chebyshev rollback." },
  { kind: "number", key: "surfaceColumns", label: "Finest columns", unit: "columns", min: 384, max: 20_000, step: 24, digits: 0, default: 384, tier: "fine", hint: "Finest x/z lattice shared by the authoritative level set and octree owner map. The balanced 384-column dam-break bring-up grid is exactly cubic (24 x 18 x 16), as required by the power catalog and global fine lattice." },
  { kind: "number", key: "adaptivity", label: "Octree adaptivity", unit: "", min: 0, max: 1, step: 0.1, digits: 1, default: 1, tier: "coarse", hint: "Debug quality/performance sweep: 0 forces finest pressure cells everywhere; 1 enables full signed-distance-graded coarsening." },
  { kind: "select", key: "secondaryParticles", label: "Secondary liquid", default: "off", tier: "coarse", update: "runtime", options: [{ value: "on", label: "Spray droplets" }, { value: "off", label: "Off" }], hint: "One-way GPU droplets preserve escaped splash detail without changing liquid mass or pressure." },
  { kind: "number", key: "secondaryParticleCapacity", label: "Particle budget", unit: "particles", min: 4_096, max: 65_536, step: 1_024, digits: 0, default: 16_384, tier: "fine", hint: "Fixed GPU ring capacity. Full rings overwrite the oldest slots without allocating or reading back." },
  { kind: "number", key: "secondaryParticleSurfaceCorrection", label: "Particle surface correction", unit: "", min: 0, max: 1, step: 0.1, digits: 1, default: 0, tier: "fine", hint: "Optionally folds only near-interface spray markers back into phi. Each substep is capped to 0.2 cell and detached droplets remain render-only; zero preserves the proven one-way path." },
  { kind: "select", key: "maximumLeafSize", label: "Maximum leaf", default: "16", tier: "fine", options: [{ value: "2", label: "2³ cells" }, { value: "4", label: "4³ cells" }, { value: "8", label: "8³ cells" }, { value: "16", label: "16³ cells" }, { value: "32", label: "32³ cells" }], hint: "Largest dyadic pressure cell. Interface bands stay fine while distant bulk air, water, and solid regions can collapse to much larger cells, then enforce 2:1 balance." },
  { kind: "number", key: "interfaceRefinementBandCells", label: "Interface refinement band", unit: "cells", min: 0, max: 32, step: 1, digits: 0, default: 4, tier: "fine", hint: "Pure air, water, or solid leaves farther than this many finest cells from liquid or solid interfaces may stay at the maximum leaf size. Lower values make all bulk regions coarser." },
  { kind: "number", key: "surfaceDetailStrength", label: "Dynamic surface detail", unit: "", min: 0, max: 1, step: 0.1, digits: 1, default: 0, tier: "fine", hint: "Widens the finest pressure band by up to eight cells where the surface is sharply curved or locally straining. Zero is the proven fixed-band path." },
  { kind: "select", key: "faceVelocityTransport", label: "Adaptive velocity", default: "on", tier: "fine", options: [{ value: "on", label: "Compact octree faces" }, { value: "off", label: "Dense compatibility" }], hint: "Transports and projects velocity directly on canonical octree faces. Unsupported terrain, hydrostatic, and rigid host-cutover cases retain the dense compatibility path." },
  { kind: "select", key: "sparseSurfaceBand", label: "Sparse surface field", default: "off", tier: "fine", options: [{ value: "authoritative", label: "Adaptive detail patches" }, { value: "mirror", label: "Mirror / parity" }, { value: "off", label: "Dense only" }], hint: "Optional fine two-sided phi pages for curvature or velocity-strain detail. Off keeps the octree's nominal-resolution level set authoritative; mirror and authoritative modes remain available for comparison." },
  { kind: "select", key: "surfaceRefinementFactor", label: "Surface refinement", default: "2", tier: "fine", options: [{ value: "1", label: "1× parity" }, { value: "2", label: "2× linear" }, { value: "4", label: "4× experimental" }], hint: "Fine samples per transport-cell edge inside resident surface pages. Memory follows active surface area instead of refining the full 3D domain." },
  { kind: "select", key: "globalFineLevelSetFactor", label: "Global fine lattice", default: "4", tier: "fine", options: [{ value: "off", label: "Leaf-page rollback" }, { value: "4", label: "4× indexed narrow band" }, { value: "8", label: "8× experimental" }], hint: "The product path uses a 4× globally indexed sparse narrow band, independent of octree rows. Compact coarse octree phi supplies sign and distance outside the valid band; factor 8 remains experimental." },
  { kind: "select", key: "powerDiagramProjection", label: "Power projection", default: "authoritative", tier: "fine", options: [{ value: "off", label: "Axis rollback" }, { value: "mirror", label: "Power mirror" }, { value: "authoritative", label: "Power authority" }], hint: "The cubic dam-break uses catalog power faces and the Section 4.3 hybrid solver. Terrain, imported/seeded geometry, anisotropic grids, and invalid topology, transfer, aperture, solve, or publication fail closed to the retained axis generation." },
  { kind: "number", key: "sparseSurfaceBandCells", label: "Fine surface support", unit: "fine cells", min: 2, max: 16, step: 1, digits: 0, default: 4, tier: "fine", hint: "Minimum signed-distance support on both sides of phi=0. Velocity backtrace and stencil margins are added automatically." },
  { kind: "number", key: "sparseSurfacePageFraction", label: "Surface page budget", unit: "domain fraction", min: 0.1, max: 1, step: 0.05, digits: 2, default: 0.75, tier: "fine", hint: "Hard physical pool as a fraction of the virtual fine page lattice. Exhaustion is reported and atomically falls back to dense extraction; it never indexes beyond the pool." },
  { kind: "select", key: "pressureWarmStart", label: "Pressure warm start", default: "on", tier: "fine", options: [{ value: "on", label: "On (previous field)" }, { value: "off", label: "Off (cold start)" }], hint: "Seed each compacted leaf solve with the previous step's pressure instead of clearing to zero, so the polynomial refines an already-good field. The legacy dense ladder always cold-starts." },
  { kind: "select", key: "hydrostaticSplit", label: "Hydrostatic reference", default: "off", tier: "fine", options: [{ value: "off", label: "Absolute pressure" }, { value: "on", label: "Deviation pressure A/B" }], hint: "Experimental octree-only A/B for body-free tank fills without inflow. Subtracts a fixed rest-surface reference, with perturbation pressure carrying surface displacement. Unsupported scenes fail closed to absolute pressure." },
  { kind: "select", key: "brickAtlas", label: "Brick atlas ownership", default: "mirror", tier: "fine", options: [{ value: "mirror", label: "Mirror + validate" }, { value: "authoritative", label: "Authoritative A/B" }, { value: "off", label: "Off" }], hint: "Compatibility-path A/B for pooled phi/velocity atlas tiles. Compact face authority keeps only the deep-liquid residency worklist and suppresses this duplicate payload; FLUID_OCTREE_COMPACT_BRICK_ATLAS=1 restores it for diagnostics." },
  { kind: "select", key: "brickSparseSurface", label: "Brick-sparse surface", default: "on", tier: "fine", options: [{ value: "on", label: "Resident worklist" }, { value: "off", label: "Dense A/B" }], hint: "Runs coarse phi advection, redistancing, and volume correction only over velocity-swept resident bricks. The dense texture is retained as a compatibility mirror while remaining kernels migrate." },
  { kind: "select", key: "brickSparseAdvection", label: "Brick-sparse velocity", default: "on", tier: "fine", options: [{ value: "on", label: "Resident worklist" }, { value: "off", label: "Dense A/B" }], hint: "Dispatches velocity predictor, reverse, and MacCormack correction over the GPU-authored wet-domain brick list; retired bricks are explicitly zeroed." },
  { kind: "select", key: "brickSparseTransport", label: "Brick-sparse transport prep", default: "off", tier: "fine", options: [{ value: "off", label: "Dense (ocean default)" }, { value: "on", label: "Resident worklist A/B" }], hint: "Builds current and predicted padded transport fields only for wet-domain bricks. The widened full-footprint ocean retains most bricks, so this remains an opt-in A/B until a sparse-domain benchmark proves a win." },
  { kind: "select", key: "brickSparseOccupancyFlux", label: "Brick-sparse occupancy/flux", default: "off", tier: "fine", options: [{ value: "off", label: "Dense (A/B baseline)" }, { value: "on", label: "Resident worklist A/B" }], hint: "Reduces column occupancy and conservative VOF flux limits over wet-domain bricks while retaining their dense compatibility textures. Column maxima use an area-only atomic resolve; retired flux cells restore invalid-neighbor limits." },
  { kind: "select", key: "brickSparseExtrapolation", label: "Brick-sparse extrapolation", default: "off", tier: "fine", options: [{ value: "off", label: "Dense (A/B baseline)" }, { value: "on", label: "Resident worklist A/B" }], hint: "Runs the post-projection velocity extrapolation seed and narrow-band sweeps over the GPU-authored wet-domain brick list. Retired velocity bricks are explicitly zeroed; this remains opt-in until benchmarked." },
  { kind: "select", key: "brickPreActivation", label: "Brick pre-activation", default: "on", tier: "fine", options: [{ value: "on", label: "Velocity swept" }, { value: "off", label: "Phi band only" }], hint: "Widens brick residency support by the velocity swept per step and activates the downstream face-neighbor of interface bricks, so a moving front never advects into an unscheduled brick." }
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
  pressureIterations: numberValue(values, params, "pressureIterations"),
  // The spray component is allocated once; visibility/simulation is a live
  // runtime setting so toggling it never rebuilds the pressure solver.
  secondaryParticles: true,
  secondaryParticlesEnabled: values.secondaryParticles !== "off" && values.secondaryParticles !== false,
  secondaryParticleCapacity: numberValue(values, params, "secondaryParticleCapacity"),
  secondaryParticleSurfaceCorrection: numberValue(values, params, "secondaryParticleSurfaceCorrection"),
  tallCellSettings: { surfaceColumns: numberValue(values, params, "surfaceColumns") },
  octree: {
    pressureIterations: numberValue(values, params, "pressureIterations"),
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
  label: "GPU octree",
  shortLabel: "Octree",
  badge: "GPU OCTREE",
  description: "Fully GPU-resident 3D adaptive pressure cells driven by an independently transported signed-distance level set.",
  detail: "pressure-only dyadic octree, GPU-resident factor-4 signed-distance narrow band, compact face transport, catalog power cells, Section 4.3 hybrid PCG with fail-closed axis/Chebyshev rollback, strict 2:1 smoothing, frame-lagged variational rigid-body coupling, and no topology readbacks",
  backend: "webgpu",
  qualityLabels: { balanced: "384-column cubic safety grid", high: "7k finest columns", ultra: "12.5k finest columns" },
  params,
  pressureMapping: "Admitted power authority uses the paper's Section 4.3 hybrid PCG with a 16-iteration hard cap and GPU convergence. Unsupported scenes interpret pressure effort as the retained Chebyshev rollback budget; neither solve reads topology or row counts back.",
  presetFor: (quality) => ({
    pressureIterations: quality === "balanced" ? 128 : quality === "high" ? 320 : 400,
    // The default 1.2 x 0.9 x 0.8 m dam-break box maps exactly to
    // 24 x 18 x 16 cells at 384 x/z columns. This keeps the paper's cubic
    // power/fine lattice invariant while bringing the first factor-4 run up
    // with 15.6x fewer base cells than the 60 x 45 x 40 production grid.
    surfaceColumns: quality === "balanced" ? 384 : quality === "high" ? 7_000 : 12_500,
    adaptivity: 1,
    secondaryParticles: "off",
    secondaryParticleCapacity: 16_384,
    secondaryParticleSurfaceCorrection: 0,
    maximumLeafSize: "16",
    interfaceRefinementBandCells: 4,
    surfaceDetailStrength: 0,
    faceVelocityTransport: "on",
    sparseSurfaceBand: "off",
    surfaceRefinementFactor: "2",
    // Bring the complete paper path up on the bounded cubic grid first. The
    // existing high/ultra allocations remain explicit rollback presets until
    // their staged memory and endurance gates have passed.
    globalFineLevelSetFactor: quality === "balanced" ? "4" : "off",
    powerDiagramProjection: quality === "balanced" ? "authoritative" : "off",
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
  runtimeParamKeys: ["secondaryParticles"],
  createSolver: (device, scene, quality, values, onRigidLoads) => new WebGPUUniformEulerianSolver(device, scene, quality, onRigidLoads, options(scene, quality, values)),
  createSolverAsync: (device, scene, quality, values, onRigidLoads, onProgress, signal) => WebGPUUniformEulerianSolver.createAsync(
    device, scene, quality, onRigidLoads, options(scene, quality, values),
    (label, completed, total, phase, taskId) => onProgress({ phase: phase === "adaptive-topology" || phase === "secondary-particles" || phase === "allocation" || phase === "warmup" ? phase : "solver-pipelines", taskId, label, completed, total }),
    signal,
  )
};
