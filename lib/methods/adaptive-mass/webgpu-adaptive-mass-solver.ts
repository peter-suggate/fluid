import { GPUInitializationTaskRunner } from "../../core/gpu-initialization";
import type { GPUQuality } from "../../core/gpu-quality";
import type {
  GPUInitializationReporter,
  GPUSolverInstance,
} from "../../core/method-contract";
import type { SceneDescription } from "../../core/model";
import type { RigidBodyState } from "../../core/rigid-body";
import { sceneLatticeDimensions } from "../../core/scene-lattice";
import type { GPUEulerianInfo, GPURigidLoad } from "../../core/webgpu-eulerian";
import type { AdaptiveMassSolverOptions } from "./method";
import {
  initializeSparseBrickAtlasFromScene,
  sparseBrickAtlasStats,
  type SparseAdaptiveMassAtlas,
  type SparseBrickVec3,
} from "./sparse-brick-atlas";
import { advanceSparseBrickAtlasTranslation } from "./sparse-brick-translation";
import {
  materializeAdaptiveMassPresentationAtlas,
  WebGPUAdaptiveMassAtlasPresentation,
  type AdaptiveMassAtlasMaterialization,
} from "./webgpu-adaptive-mass-atlas-presentation";

/**
 * Browser milestone for the adaptive-mass method.
 *
 * Physics authority is a compact CPU f64 atlas for now; WebGPU owns the
 * consumer textures and the ordinary renderer lifecycle. This intentionally
 * makes the general scene/topology boundary usable before claiming that the
 * final GPU page pool, dynamic activity policy, full force step, or iterative
 * composite projection has landed.
 */
export class WebGPUAdaptiveMassSolver implements GPUSolverInstance {
  readonly info: GPUEulerianInfo;
  readonly volumeTexture: GPUTexture;
  readonly surfaceFieldTexture: GPUTexture;
  readonly gridCellTexture: GPUTexture;

  private atlas: SparseAdaptiveMassAtlas;
  private lastTime_s = 0;
  private initialMassFineCells: number;
  private disposed = false;

  private constructor(
    private readonly scene: SceneDescription,
    private readonly options: AdaptiveMassSolverOptions,
    private readonly presentation: WebGPUAdaptiveMassAtlasPresentation,
    atlas: SparseAdaptiveMassAtlas,
    quality: GPUQuality,
  ) {
    this.atlas = atlas;
    this.volumeTexture = presentation.densityTexture;
    this.surfaceFieldTexture = presentation.levelSetTexture;
    this.gridCellTexture = presentation.gridCellTexture;
    const stats = sparseBrickAtlasStats(atlas);
    this.initialMassFineCells = stats.integratedMassFineCells;
    const [nx, ny, nz] = atlas.dimensions;
    const representedFraction = stats.leafCount / Math.max(1, stats.equivalentFinestCellCount);
    const cellSize_m = Math.min(
      scene.container.width_m / nx,
      scene.container.height_m / ny,
      scene.container.depth_m / nz,
    );
    this.info = {
      nx,
      ny,
      nz,
      storedNy: ny,
      cellCount: stats.leafCount,
      equivalentUniformCells: stats.equivalentFinestCellCount,
      compressionRatio: representedFraction,
      activeCompressionRatio: representedFraction,
      activeSampleCount: stats.leafCount,
      regularLayers: ny,
      maximumNeighborDelta: 1,
      gridKind: "octree",
      cellSize_m,
      pressureIterations: 0,
      pressureSolver: "M1 composite GᵀWG seam operator (live iterative solve pending)",
      allocatedBytes: presentation.allocatedBytes + stats.leafCount * 16,
      quality,
      volumeCellSum: stats.integratedMassFineCells,
      representedVolumeCellSum: stats.integratedMassFineCells,
      representedVolumeDrift: 0,
      volumeTelemetrySource: "adaptive-conservative-mass",
      fluidBrickCapacity: stats.logicalBrickCount,
      fluidBrickResidentCount: stats.residentBrickCount,
      fluidBrickCoreCount: stats.residentBrickCount,
      fluidBrickHaloCount: 0,
      fluidBrickGeneration: stats.generation,
      quadtreeMaximumFluidScale: 2,
      quadtreeMaximumNeighborRatio: 2,
      submittedTime_s: 0,
      simulatedTime_s: 0,
      completedTime_s: 0,
      simulationLag_s: 0,
      encodedSteps: 0,
      lastSubsteps: 1,
      maximumTallCellHeight: 2,
      surfaceField: "levelset",
      volumeControl: false,
      hostSimulationSizedWorkItems: stats.leafCount,
      hostSchedulingUsesReadback: false,
    };
  }

  static async createAsync(
    device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    _onRigidLoads: ((loads: GPURigidLoad[]) => void) | undefined,
    options: AdaptiveMassSolverOptions,
    onProgress: GPUInitializationReporter,
    signal: AbortSignal = new AbortController().signal,
  ): Promise<WebGPUAdaptiveMassSolver> {
    const runner = new GPUInitializationTaskRunner(onProgress, signal);
    let dimensions: SparseBrickVec3 | undefined;
    let atlas: SparseAdaptiveMassAtlas | undefined;
    let presentation: WebGPUAdaptiveMassAtlasPresentation | undefined;
    let materialization: AdaptiveMassAtlasMaterialization | undefined;
    try {
      await runner.run([{
        id: "adaptive-mass.plan",
        phase: "planning",
        label: "Bound the arbitrary-scene presentation lattice",
        run: () => { dimensions = boundedPresentationDimensions(scene); },
      }, {
        id: "adaptive-mass.atlas",
        phase: "adaptive-topology",
        label: "Build resident 4³/8³ sparse bricks",
        dependencies: ["adaptive-mass.plan"],
        run: () => {
          atlas = initializeSparseBrickAtlasFromScene(scene, {
            finestDimensions: dimensions!,
            maximumFinestCells: 64 ** 3,
            fineHalf: {
              axis: options.seamAxis === "x" ? 0 : options.seamAxis === "y" ? 1 : 2,
              side: options.fineSide,
            },
          });
          materialization = atlasMaterialization(atlas, scene);
        },
      }, {
        id: "adaptive-mass.presentation",
        phase: "allocation",
        label: "Allocate adaptive water and ownership textures",
        dependencies: ["adaptive-mass.atlas"],
        run: () => { presentation = new WebGPUAdaptiveMassAtlasPresentation(device, dimensions!); },
      }, {
        id: "adaptive-mass.upload",
        phase: "upload",
        label: "Publish sparse atlas generation zero",
        dependencies: ["adaptive-mass.presentation"],
        run: () => { presentation!.upload(materialization!); },
      }, {
        id: "adaptive-mass.warmup",
        phase: "warmup",
        label: "Fence adaptive presentation generation zero",
        dependencies: ["adaptive-mass.upload"],
        run: () => device.queue.onSubmittedWorkDone(),
      }]);
      return new WebGPUAdaptiveMassSolver(scene, options, presentation!, atlas!, quality);
    } catch (error) {
      presentation?.destroy();
      throw error;
    }
  }

  advanceTo(time_s: number, _bodies: RigidBodyState[]): boolean {
    void _bodies;
    if (this.disposed || !Number.isFinite(time_s) || time_s <= this.lastTime_s + 1e-9) return false;
    const dt_s = Math.min(this.scene.numerics.maxDt_s, time_s - this.lastTime_s);
    if (!(dt_s > 0)) return false;
    const nextTime_s = this.lastTime_s + dt_s;
    const axis = this.options.seamAxis === "x" ? 0 : this.options.seamAxis === "y" ? 1 : 2;
    // A bounded reversible translation is the first interactive transport
    // acceptance motion. It repeatedly crosses 4/8 seams without pretending
    // the general force/free-surface step is already complete.
    const direction = Math.floor(nextTime_s / 1.5) % 2 === 0 ? 1 : -1;
    const displacement: [number, number, number] = [0, 0, 0];
    displacement[axis] = direction * Math.min(0.35, 0.75 * dt_s);
    this.atlas = advanceSparseBrickAtlasTranslation(this.atlas, {
      displacementFine: displacement,
    }).atlas;
    const materialization = atlasMaterialization(this.atlas, this.scene);
    this.presentation.upload(materialization);
    this.lastTime_s = nextTime_s;
    const stats = sparseBrickAtlasStats(this.atlas);
    this.info.submittedTime_s = nextTime_s;
    this.info.simulatedTime_s = nextTime_s;
    this.info.simulationLag_s = Math.max(0, time_s - nextTime_s);
    this.info.lastDt_s = dt_s;
    this.info.encodedSteps = (this.info.encodedSteps ?? 0) + 1;
    this.info.volumeCellSum = stats.integratedMassFineCells;
    this.info.representedVolumeCellSum = stats.integratedMassFineCells;
    this.info.representedVolumeDrift = stats.integratedMassFineCells - this.initialMassFineCells;
    this.info.fluidBrickGeneration = stats.generation;
    this.info.hostSimulationSizedWorkItems = stats.leafCount;
    return true;
  }

  async readStats(): Promise<GPUEulerianInfo> {
    return { ...this.info };
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.presentation.destroy();
  }
}

function boundedPresentationDimensions(scene: SceneDescription): SparseBrickVec3 {
  const authored = sceneLatticeDimensions(scene);
  const scale = Math.min(1, 64 / Math.max(...authored));
  return authored.map((value) => Math.max(8, Math.min(64,
    Math.max(1, Math.round(value * scale / 8)) * 8))) as [number, number, number];
}

function atlasMaterialization(
  atlas: SparseAdaptiveMassAtlas,
  scene: SceneDescription,
): AdaptiveMassAtlasMaterialization {
  const finestCell_m = Math.min(
    scene.container.width_m / atlas.dimensions[0],
    scene.container.height_m / atlas.dimensions[1],
    scene.container.depth_m / atlas.dimensions[2],
  );
  return materializeAdaptiveMassPresentationAtlas({
    dimensions: atlas.dimensions,
    emptyLevelSet: 4 * finestCell_m,
    densityProxyBand: 4 * finestCell_m,
    bricks: atlas.bricks.map((brick) => ({
      originFine: brick.coordinate.map((value) => value * 8) as [number, number, number],
      resolution: brick.resolution,
      fineSpan: 8,
      density: brick.density,
    })),
  });
}
