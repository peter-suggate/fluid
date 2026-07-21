import type { PerformanceSnapshot } from "./stores/diagnostics-store";
import type { GPUPhysicsStageId } from "./webgpu-eulerian";

export type PerformanceStage = {
  key: string;
  label: string;
  shortLabel: string;
  value: number;
  className: string;
  timer: "physics" | "render" | "async";
  group: "compute" | "graphics" | "transfer";
  description: string;
  reads: string[];
  writes: string[];
  dependsOn: string[];
  active: boolean;
  sync?: string;
};

type PhysicsStageInput = {
  methodId: string;
  snapshot: PerformanceSnapshot;
  contextMatches: boolean;
  pressureSolver?: string;
  topologyPath?: "inline" | "async";
};

const active = (snapshot: PerformanceSnapshot, id: GPUPhysicsStageId) => snapshot.gpuActiveStages.includes(id);
const value = (matches: boolean, measured: number) => matches ? measured : 0;

/** Method-specific queue order and resource flow for the live physics implementation. */
export function physicsPerformanceStages({ methodId, snapshot, contextMatches, pressureSolver, topologyPath = "inline" }: PhysicsStageInput): PerformanceStage[] {
  const stage = (definition: Omit<PerformanceStage, "timer">): PerformanceStage => ({ ...definition, timer: "physics" });
  const overhead = (dependsOn: string) => stage({
    key: "overhead", label: "Copies, clears + queue gaps", shortLabel: "GAPS", value: value(contextMatches, snapshot.gpuOverhead_ms), className: "stage-overhead", group: "transfer", active: true,
    description: "Measured GPU time outside the named timestamp regions: texture and buffer copies, clears, transitions, and unclassified queue work.", reads: ["GPU queue"], writes: ["staging and working resources"], dependsOn: [dependsOn], sync: "Residual measured time; it is included in capacity calculations."
  });
  const diagnostics = (dependsOn: string) => stage({
    key: "diagnostics", label: "Diagnostics reductions", shortLabel: "REDUCE", value: value(contextMatches, snapshot.gpuDiagnostics_ms), className: "stage-diagnostics", group: "transfer", active: active(snapshot, "diagnostics"),
    description: "Reduces stability, conservation, and pressure-quality signals for asynchronous readback.", reads: ["φ / VOF", "velocity u", "pressure p"], writes: ["diagnostic summary"], dependsOn: [dependsOn], sync: "Small summaries are copied back asynchronously."
  });
  const rigid = (dependsOn: string) => stage({
    key: "rigid", label: "Rigid-body coupling", shortLabel: "COUPLE", value: value(contextMatches, snapshot.gpuRigid_ms), className: "stage-rigid", group: "compute", active: active(snapshot, "rigidCoupling"),
    description: "Applies solid occupancy and exchanges impulses between the liquid and active rigid bodies.", reads: ["fluid state", "body transforms and velocities"], writes: ["fluid momentum", "body impulses"], dependsOn: [dependsOn]
  });
  const spray = (dependsOn: string) => stage({
    key: "spray-sim", label: "Spray breakup + transport", shortLabel: "SPRAY SIM", value: value(contextMatches, snapshot.gpuSpraySimulation_ms), className: "stage-surface-update", group: "compute", active: active(snapshot, "spray"),
    description: "Advects the bounded secondary-liquid ring, emits spatially coherent breakup events, and classifies drops, ligaments, and thin sheets for optical rendering.", reads: ["signed distance φ", "projected velocity", "spray particle ring"], writes: ["spray particle ring", "breakup shape metadata"], dependsOn: [dependsOn]
  });

  if (methodId === "octree") {
    const pressureLabel = pressureSolver ? `Octree leaf pressure · ${pressureSolver}` : "Octree leaf pressure · Chebyshev-Jacobi";
    const splitPowerPressure = active(snapshot, "powerAssembly") && active(snapshot, "pressureSolve");
    const splitPowerProjection = active(snapshot, "powerProjection") && active(snapshot, "velocityProjection");
    const pressureStages = splitPowerPressure ? [
      stage({
        key: "power-assembly", label: "Power rows + operator assembly", shortLabel: "POWER BUILD", value: value(contextMatches, snapshot.gpuPowerAssembly_ms), className: "stage-pressure", group: "compute", active: true,
        description: "Compacts the published liquid frontier, captures the Cartesian L1 preconditioner rows, transports and constrains generalized face velocities, generates power descriptors, and assembles the second-order L2 operator and right-hand side.", reads: ["balanced octree", "fine / coarse φ", "predicted face velocity", "solid apertures"], writes: ["compact pressure rows", "power descriptors and faces", "L1 / L2 operators"], dependsOn: ["advection"], sync: "One hardware timestamp boundary partitions the existing pressure interval; no extra readback is introduced."
      }),
      stage({
        key: "pressure", label: pressureLabel, shortLabel: "PRESSURE", value: value(contextMatches, snapshot.gpuPressureSolve_ms), className: "stage-pressure", group: "compute", active: true,
        description: "Runs the selected pressure iteration, megakernel, or Section 4.3 MGPCG/V-cycle path through its final iterate and residual reduction.", reads: ["assembled L1 / L2 operators", "pressure warm start", "rigid coupling state"], writes: ["octree pressure p", "solver residuals"], dependsOn: ["power-assembly"]
      }),
    ] : [stage({
      key: "pressure", label: pressureLabel, shortLabel: "LEAF SOLVE", value: value(contextMatches, snapshot.gpuPressure_ms), className: "stage-pressure", group: "compute", active: active(snapshot, "pressure"),
      description: "Compacts liquid rows, generates power descriptors and generalized faces, assembles the second-order power operator and Section 4.3 first-order preconditioner, then runs the selected pressure solve through its final iterate. This range includes the complete PCG/V-cycle work rather than stopping at assembly.", reads: ["balanced octree", "power-face velocity", "fine / coarse φ", "solid apertures"], writes: ["power-face operator", "octree pressure p", "solver residuals"], dependsOn: ["advection"]
    })];
    const projectionStages = splitPowerProjection ? [
      stage({
        key: "power-projection", label: "Generalized power-face projection", shortLabel: "POWER PROJECT", value: value(contextMatches, snapshot.gpuPowerProjection_ms), className: "stage-projection", group: "compute", active: true,
        description: "Applies the solved pressure jump to generalized faces, publishes the canonical adaptive normal velocities, enforces power-face solid constraints, and reconstructs and fast-marches the regular fine-band velocities used by interface transport.", reads: ["power faces", "leaf pressure p", "solid constraints", "predicted face velocity"], writes: ["projected power faces", "canonical adaptive velocity", "fine-band velocity"], dependsOn: ["pressure"], sync: "Shares one timestamp boundary with compatibility projection; the two values exactly partition the existing projection interval."
      }),
      stage({
        key: "projection", label: "Compatibility-field projection", shortLabel: "FIELD PROJECT", value: value(contextMatches, snapshot.gpuVelocityProjection_ms), className: "stage-projection", group: "compute", active: true,
        description: "Reconstructs and projects the regular compatibility velocity field, applies body coupling and pressure impulses, and records projection parity diagnostics.", reads: ["leaf pressure p", "projected power faces", "rigid coupling state"], writes: ["regular velocity samples", "body pressure impulses", "projection parity"], dependsOn: ["power-projection"]
      }),
    ] : [stage({
      key: "projection", label: "Finite-volume octree projection", shortLabel: "PROJECT", value: value(contextMatches, snapshot.gpuProjection_ms), className: "stage-projection", group: "compute", active: active(snapshot, "projection"),
      description: "Applies the solved pressure jump to the same generalized power faces used by assembly, reconstructs canonical octree and regular-face velocities, enforces solid constraints, and publishes the projected field consumed by fine-interface transport.", reads: ["power faces", "leaf pressure p", "solid constraints", "predicted face velocity"], writes: ["projected power faces", "canonical face velocity", "regular velocity samples"], dependsOn: ["pressure"]
    })];
    return [
      stage({
        key: "topology", label: "Owner residency + balanced octree", shortLabel: "OCTREE", value: value(contextMatches, snapshot.gpuLayerConstruction_ms), className: "stage-topology", group: "compute", active: active(snapshot, "topology"),
        description: "Evolves owner-page residency and indirect worklists, selects changed topology, resets and refines resident tiles from sparse signed-distance summaries, runs every 2:1 balance round, and republishes the persistent liquid-leaf frontier.", reads: ["coarse / global-fine φ", "owner pages", "surface candidates", "adaptivity"], writes: ["balanced octree owners", "resident worklists", "liquid-leaf frontier"], dependsOn: ["uploads"], sync: "Timestamped across the complete rebuild, including every CFL substep, with no topology readback."
      }),
      stage({
        key: "advection", label: "Velocity transport preparation + advection", shortLabel: "ADVECT", value: value(contextMatches, snapshot.gpuAdvection_ms), className: "stage-advection", group: "compute", active: active(snapshot, "advection"),
        description: "Builds signed-distance occupancy and an air-extended transport field, then advances the dense velocity predictor with bounded MacCormack transport.", reads: ["projected velocity", "signed distance φ"], writes: ["transport velocity", "predicted velocity"], dependsOn: ["topology"]
      }),
      ...pressureStages,
      ...projectionStages,
      stage({
        key: "extrapolation", label: "Narrow-band velocity extrapolation", shortLabel: "EXTEND", value: value(contextMatches, snapshot.gpuExtrapolation_ms), className: "stage-extrapolation", group: "compute", active: active(snapshot, "extrapolation"),
        description: "Extends projected velocity through the air-side interface band so the following level-set transport can sample newly exposed cells.", reads: ["projected velocity", "signed distance φ"], writes: ["extrapolated velocity"], dependsOn: ["projection"]
      }),
      stage({
        key: "materialization", label: "Adaptive overlay materialization", shortLabel: "MAP FIELDS", value: value(contextMatches, snapshot.gpuMaterialization_ms), className: "stage-materialization", group: "compute", active: active(snapshot, "materialization"),
        description: "Materializes the resident owner map, pressure ownership, mapped pressure, and projected divergence into 3D textures for adaptive diagnostics and overlays.", reads: ["octree owners", "leaf pressure p", "projected velocity", "signed distance φ"], writes: ["topology overlay", "pressure overlay", "divergence overlay"], dependsOn: ["extrapolation"]
      }),
      stage({
        key: "surface-update", label: "Level-set transport + volume control", shortLabel: "SURFACE φ", value: value(contextMatches, snapshot.gpuSurfaceUpdate_ms), className: "stage-surface-update", group: "compute", active: active(snapshot, "surfaceUpdate"),
        description: "Adapts surface pages, seeds and transports the global-fine level set from reconstructed power velocity, redistances it, applies volume correction, restricts a corrected coarse φ directory, publishes sparse sizing summaries, and commits or rolls back the generation transactionally.", reads: ["projected / extrapolated power velocity", "current fine and coarse φ", "surface pages"], writes: ["next global-fine φ", "corrected coarse φ", "surface summaries", "publication generation"], dependsOn: ["materialization"]
      }),
      rigid("surface-update"),
      spray("rigid"),
      stage({
        key: "fluid-residency", label: "Fluid brick residency", shortLabel: "RESIDENCY", value: value(contextMatches, snapshot.gpuFluidResidency_ms), className: "stage-residency", group: "compute", active: active(snapshot, "fluidResidency"),
        description: "Classifies finest-level bricks as fluid core, stencil halo, newly activated, or retired and emits GPU-owned indirect worklists for the next adaptive rebuild.", reads: ["advected signed distance φ", "previous brick states"], writes: ["core / halo residency", "active and retired worklists"], dependsOn: ["spray-sim"]
      }),
      stage({
        key: "sparse-publication", label: "Sparse scene fluid publication", shortLabel: "PUBLISH", value: value(contextMatches, snapshot.gpuSparsePublication_ms), className: "stage-publication", group: "compute", active: active(snapshot, "sparsePublication"),
        description: "Publishes the current fluid generation into the shared sparse scene representation consumed by SVO traversal and inspection, including compact payloads and retirement-safe records.", reads: ["resident worklists", "fine / coarse φ", "projected velocity", "solid occupancy"], writes: ["SVO fluid payload", "voxel and brick records", "scene generation"], dependsOn: ["fluid-residency"]
      }),
      diagnostics("sparse-publication"),
      overhead("diagnostics")
    ];
  }

  if (methodId === "quadtree-tall-cell") {
    const pressureLabel = pressureSolver ? `Variational pressure + projection · ${pressureSolver}` : "Variational pressure + projection";
    const stages: PerformanceStage[] = [];
    if (topologyPath === "inline") stages.push(stage({
        key: "topology", label: "Adaptive topology update", shortLabel: "TOPOLOGY", value: value(contextMatches, snapshot.gpuLayerConstruction_ms), className: "stage-topology", group: "compute", active: active(snapshot, "topology"),
        description: "Evaluates the sizing field, refines and 2:1-smooths the quadtree, builds vertical tall segments, and packs the live variational system.", reads: ["projected velocity", "signed distance φ", "explicit refinement"], writes: ["quadtree leaves", "tall segments", "pressure DOFs and faces"], dependsOn: ["uploads"], sync: "Resident updates run inline; overflow and rigid-coupled rebuilds continue on the asynchronous topology path."
      }));
    stages.push(
      stage({
        key: "advection", label: "Backing-field advection", shortLabel: "ADVECT", value: value(contextMatches, snapshot.gpuAdvection_ms), className: "stage-advection", group: "compute", active: active(snapshot, "advection"),
        description: "Builds occupancy and transport views, then advances the cubic backing velocity field with the selected transport scheme.", reads: ["velocity u", "VOF volume", "solid occupancy"], writes: ["predicted velocity", "transport flux scales"], dependsOn: [topologyPath === "inline" ? "topology" : "uploads"]
      }),
      stage({
        key: "pressure", label: pressureLabel, shortLabel: "PRESSURE+PROJECT", value: value(contextMatches, snapshot.gpuPressure_ms), className: "stage-pressure", group: "compute", active: active(snapshot, "pressure"),
        description: "Refreshes adaptive coefficients once, solves pressure to the relative residual tolerance with PCG, projects face fluxes, maps velocity back to the cubic field, and evaluates post-projection divergence. An experimental fixed-pass Chebyshev path remains selectable.", reads: ["adaptive DOFs and faces", "predicted velocity", "solid fractions", "rigid velocity"], writes: ["pressure p", "projected and extrapolated velocity", "rigid impulse"], dependsOn: ["advection"], sync: "PCG uses scalar reductions and exact same-step rigid coupling. Experimental Chebyshev removes the hot-loop reductions and publishes Kᵀp after the solve."
      }),
      stage({
        key: "surface-update", label: "Surface transport + redistance", shortLabel: "SURFACE φ", value: value(contextMatches, snapshot.gpuSurfaceUpdate_ms), className: "stage-surface-update", group: "compute", active: active(snapshot, "surfaceUpdate"),
        description: "Materializes flux-consistent transport velocity, advects the level set, restores signed distance, optionally culls debris, and reduces surface volume diagnostics.", reads: ["projected face fluxes", "signed distance φ", "VOF volume"], writes: ["advected signed distance φ", "surface diagnostics"], dependsOn: ["pressure"]
      }),
      rigid("surface-update"),
      diagnostics("rigid"),
      overhead("diagnostics")
    );
    return stages;
  }

  if (methodId === "tall-cell") {
    const pressureLabel = pressureSolver ? `Tall-cell multigrid pressure · ${pressureSolver}` : "Tall-cell multigrid pressure";
    return [
      stage({
        key: "preparation", label: "Velocity extension + φ maintenance", shortLabel: "PREP", value: value(contextMatches, snapshot.gpuPreparation_ms), className: "stage-preparation", group: "compute", active: active(snapshot, "preparation"),
        description: "Extends velocity out of the liquid through the tall-cell hierarchy and periodically reinitializes the narrow-band signed-distance field.", reads: ["packed velocity", "column bases", "signed distance φ"], writes: ["extended velocity", "maintained φ"], dependsOn: ["uploads"]
      }),
      stage({
        key: "advection", label: "Level-set + velocity advection", shortLabel: "ADVECT", value: value(contextMatches, snapshot.gpuAdvection_ms), className: "stage-advection", group: "compute", active: active(snapshot, "advection"),
        description: "Advects the level set through its planned substeps and advances velocity on the packed restricted tall-cell grid.", reads: ["extended velocity", "signed distance φ", "packed volume"], writes: ["advected φ", "predicted velocity and volume"], dependsOn: ["preparation"]
      }),
      stage({
        key: "remesh", label: "Tall-cell remesh + field transfer", shortLabel: "REMESH", value: value(contextMatches, snapshot.gpuRemeshing_ms), className: "stage-remesh", group: "compute", active: active(snapshot, "remeshing"),
        description: "Plans and smooths new column bases, remaps fluid fields, and transfers the pressure warm start when the remesh cadence fires.", reads: ["advected φ and volume", "old column bases", "old pressure"], writes: ["new column bases", "remapped fields"], dependsOn: ["advection"], sync: "Visible every frame; idle means the configured remesh cadence did not fire in this sample."
      }),
      rigid("remesh"),
      stage({
        key: "pressure", label: pressureLabel, shortLabel: "PRESSURE", value: value(contextMatches, snapshot.gpuPressure_ms), className: "stage-pressure", group: "compute", active: active(snapshot, "pressure"),
        description: "Builds divergence and runs the restricted full-cycle multigrid pressure solve, including an optional second defect-correction solve.", reads: ["remapped predictor", "packed volume", "tall-cell coefficients"], writes: ["pressure p"], dependsOn: ["rigid"], sync: pressureSolver?.includes("defect correction") ? "This aggregate contains both pressure solves." : "Fixed V-cycle work; residual is measured after the solve."
      }),
      stage({
        key: "projection", label: "Tall-cell velocity projection", shortLabel: "PROJECT", value: value(contextMatches, snapshot.gpuProjection_ms), className: "stage-projection", group: "compute", active: active(snapshot, "projection"),
        description: "Applies the tall-cell pressure gradient and transfers the projected velocity and volume into the next state.", reads: ["pressure p", "predicted velocity"], writes: ["projected velocity", "next volume"], dependsOn: ["pressure"]
      }),
      diagnostics("projection"),
      overhead("diagnostics")
    ];
  }

  if (methodId === "uniform") {
    const pressureLabel = pressureSolver ? `Uniform pressure · ${pressureSolver}` : "Uniform Jacobi pressure";
    return [
      stage({
        key: "advection", label: "VOF + velocity advection", shortLabel: "ADVECT", value: value(contextMatches, snapshot.gpuAdvection_ms), className: "stage-advection", group: "compute", active: active(snapshot, "advection"),
        description: "Builds occupancy and flux transport views, then advances velocity and conservative volume on the full-depth cubic grid.", reads: ["velocity u", "VOF volume", "solid occupancy"], writes: ["predicted velocity", "advected VOF"], dependsOn: ["uploads"]
      }),
      stage({
        key: "conditioning", label: "VOF density sharpening", shortLabel: "SHARPEN", value: value(contextMatches, snapshot.gpuConditioning_ms), className: "stage-conditioning", group: "compute", active: active(snapshot, "conditioning"),
        description: "Computes, scatters, and resolves conservative density-sharpening deposits before projection.", reads: ["advected VOF", "transport flux scales"], writes: ["conditioned VOF", "deposit buffer"], dependsOn: ["advection"], sync: "Idle when density sharpening is disabled."
      }),
      stage({
        key: "pressure", label: pressureLabel, shortLabel: "PRESSURE", value: value(contextMatches, snapshot.gpuPressure_ms), className: "stage-pressure", group: "compute", active: active(snapshot, "pressure"),
        description: "Runs the configured weighted-Jacobi sweeps for the cubic-grid pressure system.", reads: ["conditioned VOF", "predicted velocity", "solid SDF"], writes: ["pressure p"], dependsOn: ["conditioning"]
      }),
      stage({
        key: "projection", label: "Uniform velocity projection", shortLabel: "PROJECT", value: value(contextMatches, snapshot.gpuProjection_ms), className: "stage-projection", group: "compute", active: active(snapshot, "projection"),
        description: "Applies the cubic-grid pressure gradient to make the velocity field divergence-free.", reads: ["pressure p", "predicted velocity"], writes: ["projected velocity"], dependsOn: ["pressure"]
      }),
      rigid("projection"),
      diagnostics("rigid"),
      overhead("diagnostics")
    ];
  }

  return [];
}

type AdaptiveTopologyStageInput = {
  snapshot: PerformanceSnapshot;
  contextMatches: boolean;
};

/**
 * Event-driven topology rebuild phases. These are deliberately a separate
 * lane: repeating their last duration for every fluid advance would inflate
 * realtime GPU demand and make the capacity model wrong.
 */
export function adaptiveTopologyPerformanceStages({ snapshot, contextMatches }: AdaptiveTopologyStageInput): PerformanceStage[] {
  const available = contextMatches && (snapshot.adaptiveRebuildCompletedCount > 0 || snapshot.adaptiveRebuildPending);
  const stage = (definition: Omit<PerformanceStage, "timer" | "active">): PerformanceStage => ({ ...definition, timer: "async", active: available });
  return [
    stage({ key: "adaptive-gpu-build", label: "GPU quadtree construction", shortLabel: "GPU BUILD", value: value(contextMatches, snapshot.adaptiveGPUConstructionKernel_ms), className: "stage-topology", group: "compute", description: "Evaluates the sizing field, refines leaves, enforces 2:1 balance, and emits compact adaptive topology on the GPU.", reads: ["signed distance φ", "velocity u", "refinement controls"], writes: ["balanced quadtree leaves"], dependsOn: ["projected state"], sync: "Exact timestamp-query kernel time; runs only when an asynchronous rebuild is kicked." }),
    stage({ key: "adaptive-gpu-pack", label: "GPU sparse topology pack", shortLabel: "GPU PACK", value: value(contextMatches, snapshot.adaptiveGPUSparsePack_ms), className: "stage-topology", group: "transfer", description: "Compacts sparse GPU topology output into the readback representation used by the host builder.", reads: ["balanced quadtree leaves"], writes: ["packed topology readback"], dependsOn: ["adaptive-gpu-build"], sync: "Asynchronous wall phase; may overlap other queued work." }),
    stage({ key: "adaptive-topology-pack", label: "CPU topology unpack", shortLabel: "UNPACK", value: value(contextMatches, snapshot.adaptiveCPUTopologyPack_ms), className: "stage-preparation", group: "transfer", description: "Validates and expands the compact topology payload into host-side leaf records.", reads: ["packed topology readback"], writes: ["host leaf records"], dependsOn: ["adaptive-gpu-pack"] }),
    stage({ key: "adaptive-redistance", label: "CPU surface redistance", shortLabel: "REDIST φ", value: value(contextMatches, snapshot.adaptiveCPURedistance_ms), className: "stage-surface-update", group: "compute", description: "Restores a usable signed-distance field for topology decoding when the rebuild path requires it.", reads: ["surface field"], writes: ["redistanced φ"], dependsOn: ["adaptive-topology-pack"] }),
    stage({ key: "adaptive-decode", label: "CPU quadtree decode", shortLabel: "DECODE", value: value(contextMatches, snapshot.adaptiveCPUQuadtreeDecode_ms), className: "stage-preparation", group: "compute", description: "Decodes balanced leaves, neighbor relations, and level metadata.", reads: ["host leaf records", "redistanced φ"], writes: ["decoded quadtree"], dependsOn: ["adaptive-redistance"] }),
    stage({ key: "adaptive-tall-grid", label: "Tall-cell grid construction", shortLabel: "TALL GRID", value: value(contextMatches, snapshot.adaptiveCPUTallGrid_ms), className: "stage-remesh", group: "compute", description: "Extrudes adaptive horizontal leaves into vertical tall segments and assigns stored samples.", reads: ["decoded quadtree", "surface geometry"], writes: ["adaptive tall segments"], dependsOn: ["adaptive-decode"] }),
    stage({ key: "adaptive-assembly", label: "Variational system assembly", shortLabel: "ASSEMBLE", value: value(contextMatches, snapshot.adaptiveCPUVariationalAssembly_ms), className: "stage-pressure", group: "compute", description: "Builds pressure degrees of freedom, cut-face couplings, rigid constraints, and the adaptive sparse operator.", reads: ["adaptive tall segments", "solid geometry"], writes: ["variational pressure system"], dependsOn: ["adaptive-tall-grid"] }),
    stage({ key: "adaptive-system-pack", label: "Pressure system pack", shortLabel: "SYS PACK", value: value(contextMatches, snapshot.adaptiveCPUSystemPack_ms), className: "stage-pressure", group: "transfer", description: "Packs sparse rows, faces, constraints, and indirect dispatch metadata into GPU-ready arrays.", reads: ["variational pressure system"], writes: ["packed pressure buffers"], dependsOn: ["adaptive-assembly"] }),
    stage({ key: "adaptive-factor", label: "Preconditioner factorization", shortLabel: "IC FACTOR", value: value(contextMatches, snapshot.adaptiveCPUICFactorization_ms), className: "stage-conditioning", group: "compute", description: "Builds the selected incomplete-Cholesky or compatible adaptive pressure preconditioner.", reads: ["packed pressure matrix"], writes: ["preconditioner factors"], dependsOn: ["adaptive-system-pack"] }),
    stage({ key: "adaptive-upload", label: "Adaptive resource upload", shortLabel: "UPLOAD", value: value(contextMatches, snapshot.adaptiveCPUResourceUpload_ms), className: "stage-overhead", group: "transfer", description: "Creates and uploads the replacement topology, pressure, face, and factor resources before the atomic solver swap.", reads: ["packed pressure buffers", "preconditioner factors"], writes: ["next GPU projection resources"], dependsOn: ["adaptive-factor"], sync: "The completed resource set is swapped only at a safe advance boundary." })
  ];
}
