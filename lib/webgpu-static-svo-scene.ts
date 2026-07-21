import type { SceneDescription } from "./model";
import type { GPUEulerianInfo } from "./webgpu-eulerian";
import type { GPUSolverInstance, MethodParamValues } from "./methods/types";
import { createTallCellLayout, type GPUQuality } from "./tall-cell-grid";
import { OctreeSparseBrickWorld } from "./webgpu-octree-sparse-bricks";

export type StaticSvoSceneProgress = (progress: {
  phase: "allocation" | "warmup";
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
export class WebGPUStaticSvoScene implements GPUSolverInstance {
  readonly info: GPUEulerianInfo;
  readonly volumeTexture: GPUTexture;
  readonly surfaceFieldTexture: GPUTexture;
  readonly columnBaseTexture: GPUTexture;
  readonly velocityTexture: GPUTexture;
  readonly sparseVoxelSceneSource;
  readonly initialSparseAuthorityReady = true;

  private readonly world: OctreeSparseBrickWorld;
  private readonly solidCells: GPUBuffer;
  private accountedWorldBytes: number;
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    values: MethodParamValues,
  ) {
    const requestedColumns = Number(values.surfaceColumns);
    const layout = createTallCellLayout(scene, quality, device.limits.maxTextureDimension3D, {
      ...(Number.isFinite(requestedColumns) && requestedColumns > 0 ? { surfaceColumns: requestedColumns } : {}),
      // Static presentation needs a lattice, not a stored liquid band.
      regularLayers: 2,
      liquidHalo: 0,
      airHalo: 0,
    });
    const dimensions = [layout.nx, layout.fineNy, layout.nz] as const;
    const textureUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST;
    this.volumeTexture = device.createTexture({
      label: "Static SVO empty-fluid field",
      size: dimensions,
      dimension: "3d",
      format: "r32float",
      usage: textureUsage,
    });
    this.surfaceFieldTexture = this.volumeTexture;
    this.velocityTexture = device.createTexture({
      label: "Static SVO zero-velocity field",
      size: dimensions,
      dimension: "3d",
      format: "rgba32float",
      usage: textureUsage,
    });
    this.columnBaseTexture = device.createTexture({
      label: "Static SVO column fallback",
      size: [layout.nx, layout.nz],
      format: "r32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.solidCells = device.createBuffer({
      label: "Static SVO zero-solid fallback",
      size: Math.max(8, layout.nx * layout.fineNy * layout.nz * 8),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Texture memory starts at zero, which is a liquid interface. Publish a
    // strictly positive field so the structural source contains no fluid.
    const floatsPerRow = Math.ceil(layout.nx * Float32Array.BYTES_PER_ELEMENT / 256) * 256 / Float32Array.BYTES_PER_ELEMENT;
    const emptyPhi = new Float32Array(floatsPerRow * layout.fineNy * layout.nz);
    emptyPhi.fill(8 * Math.min(layout.cellSize_m.x, layout.cellSize_m.y, layout.cellSize_m.z));
    device.queue.writeTexture(
      { texture: this.volumeTexture },
      emptyPhi,
      { bytesPerRow: floatsPerRow * Float32Array.BYTES_PER_ELEMENT, rowsPerImage: layout.fineNy },
      dimensions,
    );

    this.world = new OctreeSparseBrickWorld(device, scene, dimensions, {
      brickSize: 8,
      haloCells: 0,
      brickAtlas: "off",
      bulkResidencyOnly: false,
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
      initialRasterSurfaceDiagnostic: "Static SVO scene ready; fluid authority intentionally bypassed",
      cellSize_m: h,
      pressureIterations: 0,
      pressureSolver: "disabled · static renderer",
      allocatedBytes: this.world.allocatedBytes + this.solidCells.size,
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
    const source = this.world.ensureInspectionSource();
    const worldBytes = this.world.allocatedBytes;
    this.info.allocatedBytes += worldBytes - this.accountedWorldBytes;
    this.accountedWorldBytes = worldBytes;
    return source;
  }

  static async create(
    device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    values: MethodParamValues,
    progress: StaticSvoSceneProgress,
    signal?: AbortSignal,
  ): Promise<WebGPUStaticSvoScene> {
    progress({ phase: "allocation", taskId: "static-svo.allocate", label: "Allocate static sparse garden", completed: 0, total: 2 });
    if (signal?.aborted) throw new DOMException("GPU initialization superseded", "AbortError");
    const source = new WebGPUStaticSvoScene(device, scene, quality, values);
    try {
      progress({ phase: "warmup", taskId: "static-svo.publish", label: "Publish static sparse garden", completed: 1, total: 2 });
      const encoder = device.createCommandEncoder({ label: "Publish renderer-only SVO scene" });
      source.world.encode(encoder, {
        levelSet: source.volumeTexture,
        velocity: source.velocityTexture,
        solidCells: source.solidCells,
      });
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      if (signal?.aborted) throw new DOMException("GPU initialization superseded", "AbortError");
      progress({ phase: "warmup", taskId: "static-svo.publish", label: "Static sparse garden ready", completed: 2, total: 2 });
      return source;
    } catch (error) {
      source.destroy();
      throw error;
    }
  }

  advanceTo(): boolean { return false; }
  readStats(): Promise<GPUEulerianInfo> { return Promise.resolve({ ...this.info }); }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.world.destroy();
    this.volumeTexture.destroy();
    this.velocityTexture.destroy();
    this.columnBaseTexture.destroy();
    this.solidCells.destroy();
  }
}
