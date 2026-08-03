import type { SceneDescription } from "./model";
import type { GPUEulerianInfo } from "./webgpu-eulerian";
import type { GPUSolverInstance } from "./methods/types";
import { createTallCellLayout, type GPUQuality } from "./tall-cell-grid";
import { OctreeSparseBrickWorld } from "./webgpu-octree-sparse-bricks";
import type { ResourcePluginDefinition } from "./resource-readiness";
import type { SparseScenePrimitiveUpdate } from "./webgpu-sparse-scene-proxies";

/** Colocated lifecycle declaration for the renderer-owned live scene. */
export const liveSvoSceneResourcePlugin: ResourcePluginDefinition = Object.freeze({
  id: "scene.live-svo-source",
  lane: "svo",
  label: "Live sparse scene source",
  provides: ["live-scene"] as const,
  blocks: "nothing",
  phaseCopy: {
    allocation: "Allocating fixed live-scene arenas.",
  },
});

export interface LiveSvoSceneOptions {
  /**
   * Renderer-only construction override used by topology experiments. It does
   * not mutate `scene.voxelDomain` and therefore cannot change the simulation
   * contract of a fluid-enabled authored scene.
   */
  renderBrickSize?: 4 | 8;
  /** Additional subdivision of authored-environment bricks. */
  environmentBrickRefinementLevels?: number;
  /** Experimental in-place diffuse feedback; disabled in production. */
  radianceFeedback?: boolean;
}

export function liveSvoRenderBrickSize(
  scene: Pick<SceneDescription, "voxelDomain">,
  options: LiveSvoSceneOptions = {},
): 4 | 8 {
  const brickSize = options.renderBrickSize ?? scene.voxelDomain.brickSize_cells;
  if (brickSize !== 4 && brickSize !== 8) throw new RangeError("Live SVO render brick size must be 4 or 8");
  return brickSize;
}

export type LiveSvoSceneProgress = (progress: {
  phase: "allocation";
  taskId: string;
  label: string;
  completed: number;
  total: number;
}) => void;

/**
 * Renderer-only sparse world. It implements the renderer's narrow solver
 * source interface so the established SVO attachment path can be reused, but
 * it owns no transport, projection, level-set, or t=0 fluid authority.
 */
export class WebGPULiveSvoScene implements GPUSolverInstance {
  readonly info: GPUEulerianInfo;
  readonly volumeTexture: GPUTexture;
  readonly surfaceFieldTexture: GPUTexture;
  readonly columnBaseTexture: GPUTexture;
  readonly velocityTexture: GPUTexture;
  readonly sparseVoxelSceneSource;
  readonly initialSparseAuthorityReady = true;

  private readonly world: OctreeSparseBrickWorld;
  private accountedWorldBytes: number;
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    options: LiveSvoSceneOptions,
  ) {
    const layout = createTallCellLayout(scene, quality, device.limits.maxTextureDimension3D, {
      // Renderer-only presentation needs a lattice, not a stored liquid band.
      regularLayers: 2,
      liquidHalo: 0,
      airHalo: 0,
    });
    const dimensions = [layout.nx, layout.fineNy, layout.nz] as const;
    const textureUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST;
    this.volumeTexture = device.createTexture({
      label: "Live SVO empty-fluid field",
      size: dimensions,
      dimension: "3d",
      format: "r32float",
      usage: textureUsage,
    });
    this.surfaceFieldTexture = this.volumeTexture;
    this.velocityTexture = device.createTexture({
      label: "Live SVO zero-velocity field",
      size: dimensions,
      dimension: "3d",
      format: "rgba32float",
      usage: textureUsage,
    });
    this.columnBaseTexture = device.createTexture({
      label: "Live SVO column fallback",
      size: [layout.nx, layout.nz],
      format: "r32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.world = new OctreeSparseBrickWorld(device, scene, dimensions, {
      brickSize: liveSvoRenderBrickSize(scene, options),
      environmentBrickRefinementLevels: options.environmentBrickRefinementLevels,
      radianceFeedback: options.radianceFeedback,
      haloCells: 0,
      brickPreActivation: false,
    });
    this.sparseVoxelSceneSource = this.world.sceneSource;
    this.accountedWorldBytes = this.world.allocatedBytes;
    const h = Math.min(layout.cellSize_m.x, layout.cellSize_m.y, layout.cellSize_m.z);
    this.info = {
      nx: layout.nx,
      ny: layout.fineNy,
      nz: layout.nz,
      storedNy: 0,
      cellCount: 0,
      equivalentUniformCells: layout.equivalentUniformCellCount,
      compressionRatio: layout.equivalentUniformCellCount,
      activeCompressionRatio: layout.equivalentUniformCellCount,
      activeSampleCount: 0,
      regularLayers: 0,
      maximumNeighborDelta: 0,
      gridKind: "octree",
      initialSparseAuthorityReady: true,
      initialRasterSurfaceReady: true,
      initialRasterSurfaceState: "gpu-authoritative",
      initialRasterSurfaceDiagnostic: "Live scene source ready; fluid authority intentionally absent",
      cellSize_m: h,
      pressureIterations: 0,
      pressureSolver: "disabled · renderer-only scene",
      allocatedBytes: this.world.allocatedBytes,
      quality,
      encodedSteps: 0,
      submittedTime_s: 0,
      simulatedTime_s: 0,
      completedTime_s: 0,
      simulationLag_s: 0,
      stabilityFlags: [],
    };
  }

  /**
   * Raw-voxel and brick-grid records are intentionally absent from normal SVO
   * startup. Materialize them on first inspection, matching the dynamic
   * octree solver's lazy debug-source contract without starting a fluid
   * solver for a renderer-only scene.
   */
  get sparseVoxelRenderSource() {
    const source = this.world.inspectionSource;
    if (!source) void this.world.ensureInspectionSource();
    const worldBytes = this.world.allocatedBytes;
    this.info.allocatedBytes += worldBytes - this.accountedWorldBytes;
    this.accountedWorldBytes = worldBytes;
    return source;
  }

  static async create(
    device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    progress: LiveSvoSceneProgress,
    signal?: AbortSignal,
    options: LiveSvoSceneOptions = {},
  ): Promise<WebGPULiveSvoScene> {
    progress({ phase: "allocation", taskId: "live-svo.allocate", label: "Allocate live sparse scene", completed: 0, total: 1 });
    if (signal?.aborted) throw new DOMException("GPU initialization superseded", "AbortError");
    const source = new WebGPULiveSvoScene(device, scene, quality, options);
    await source.world.initializePipelines();
    progress({ phase: "allocation", taskId: "live-svo.allocate", label: "Live sparse scene arenas ready", completed: 1, total: 1 });
    return source;
  }

  stageSceneUpdate(scene: SceneDescription): void { this.world.stageSceneUpdate(scene); }
  stageLivePrimitiveUpdates(updates: readonly SparseScenePrimitiveUpdate[]): boolean {
    return this.world.stageLivePrimitiveUpdates(updates);
  }
  encodeSceneMaintenance(encoder: GPUCommandEncoder): boolean { return this.world.encodeSceneMaintenance(encoder); }

  advanceTo(): boolean { return false; }
  readStats(): Promise<GPUEulerianInfo> { return Promise.resolve({ ...this.info }); }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.world.destroy();
    this.volumeTexture.destroy();
    this.velocityTexture.destroy();
    this.columnBaseTexture.destroy();
  }
}
