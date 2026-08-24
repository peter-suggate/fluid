import type { RenderFrameSeam } from "../core/render-frame-stages";
import type { SceneDescription } from "../core/model";
import type { GPUEulerianInfo } from "../core/webgpu-eulerian";
import type { GPUSolverInstance } from "../core/method-contract";
import { sceneLatticeDimensions } from "../core/scene-lattice";
import type { GPUQuality } from "../core/gpu-quality";
import {
  OCTREE_SPARSE_BRICK_WORLD_STAGES,
  OctreeSparseBrickWorld,
  type OctreeSparseBrickWorldProgress,
  type SparseSceneSolidReach,
} from "./webgpu-svo-sparse-bricks";
import {
  buildSvoScenePrimitives,
  svoDescriptorForEnvironmentProxy,
  svoScenePrimitiveSolidReach,
} from "./svo-scene-primitives";
import type { CooperativeBuildOptions } from "../core/cooperative-build";
import type { ResourcePluginDefinition } from "../core/resource-readiness";
import type { SparseScenePrimitiveUpdate } from "../core/webgpu-sparse-scene-proxies";

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
  /**
   * Extra octree levels for dense environment regions on a scene the solver
   * does not own. See `OctreeSparseBrickWorld`'s option of the same name.
   */
  environmentRefinementDepth?: number;
  /**
   * Whether the surface refinement rule may halt on a flat node. See
   * `OctreeSparseBrickWorldOptions`' option of the same name; off by default,
   * because the depth it saves is saved exactly where curvature is lowest and
   * the coarse leaf that results is a visible flat facet.
   */
  environmentPlanarRefinementExemption?: boolean;
  /** Experimental in-place diffuse feedback; disabled in production. */
  radianceFeedback?: boolean;
  /**
   * Build the compact hierarchy and the wide-fanout snapshot. See
   * `OctreeSparseBrickWorldOptions`' option of the same name; off by default
   * because nothing the renderer ships binds them, so only a lane that selects
   * the `compact`/`wide`/`hybrid` dry-scene traversal needs to ask.
   */
  derivedTraversalStructures?: boolean;
  /**
   * Longest the build may hold the thread between event-loop yields.
   *
   * Only `create` reads it, and only to hand it to the cooperative driver. Left
   * unset the driver's own default applies, which is what the renderer wants;
   * a lane that measures the trade sets it explicitly.
   */
  buildSliceBudget_ms?: number;
}

export function liveSvoRenderBrickSize(
  scene: Pick<SceneDescription, "voxelDomain">,
  options: LiveSvoSceneOptions = {},
): 4 | 8 {
  const brickSize = options.renderBrickSize ?? scene.voxelDomain.brickSize_cells;
  if (brickSize !== 4 && brickSize !== 8) throw new RangeError("Live SVO render brick size must be 4 or 8");
  return brickSize;
}

/**
 * The scene's exact solids for brick selection, or nothing.
 *
 * Only a dry world may narrow its claim this way: a solver writes cells no
 * authored solid touches, so the geometric answer is not the whole one there.
 * A build that fails is not an error here — the claim simply stays the AABB it
 * has always been, which is what shipped.
 */
function liveSvoSceneSolidReach(scene: SceneDescription): readonly SparseSceneSolidReach[] | undefined {
  if (scene.systems?.fluid !== false) return undefined;
  try {
    return svoScenePrimitiveSolidReach(buildSvoScenePrimitives(scene));
  } catch {
    return undefined;
  }
}

/**
 * The one stage this layer owns before the world's own list begins.
 *
 * Typed as an `OctreeSparseBrickWorldStage` would be a lie — the world does not
 * do this — so it is its own constant, and `create` recognises it by identity
 * when it maps stages onto the shared counter.
 */
const LIVE_SVO_SOLID_REACH_STAGE = "Resolve the scene's exact solids" as const;

/**
 * Solid reach, the world's planning stages, then shader compilation.
 *
 * Derived from the world's declared list rather than written beside it, so a
 * stage added there widens this bar instead of silently overflowing it.
 */
const LIVE_SVO_BUILD_STAGE_COUNT = OCTREE_SPARSE_BRICK_WORLD_STAGES.length + 2;

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

  /**
   * The refinement depth asked for, and the one that was actually built.
   *
   * They differ only after `create` has stepped the ladder down past a depth
   * this device could not allocate. Kept on the instance because the renderer's
   * readiness status is the only place a user finds out, and "the scene is at
   * 3.125 mm because 1.5625 mm did not fit" is a different fact from "the scene
   * failed" — which is what an unhandled throw here used to mean, all the way
   * out to a terminated render worker and a frozen viewport.
   */
  requestedRefinementDepth = 0;
  builtRefinementDepth = 0;

  private readonly world: OctreeSparseBrickWorld;
  private destroyed = false;

  /**
   * The expensive half of the build, as an interruptible step.
   *
   * Split out of the constructor because a constructor cannot `await`, and
   * everything that makes a refined scene slow is in here: the exact-solid
   * reach and then the sparse world itself. What remains in the constructor is
   * three 1x1x1 textures and an info record — microseconds, and never worth
   * interrupting.
   */
  private static async buildWorld(
    device: GPUDevice,
    scene: SceneDescription,
    options: LiveSvoSceneOptions,
    reportStage: OctreeSparseBrickWorldProgress,
    interrupt: CooperativeBuildOptions,
  ): Promise<{ dimensions: readonly [number, number, number]; world: OctreeSparseBrickWorld }> {
    const dimensions = sceneLatticeDimensions(scene, device.limits.maxTextureDimension3D);
    // The exact solids behind the AABBs the brick claim is made from, so the
    // claim can skip the corners of a box its geometry never enters. Built
    // here rather than in the world because `svo-scene-primitives` already
    // depends on the world's module and the reverse edge would be a cycle;
    // the build itself is the renderer's own, interned by content hash.
    //
    // Hoisted out of the options literal so it can be announced: it evaluates
    // one exact distance function per candidate brick, and at environment
    // refinement depth 3 that is 18 % of the build — the second most expensive
    // thing here after the octree plan itself.
    reportStage({ stage: LIVE_SVO_SOLID_REACH_STAGE, completed: 0, total: 1 });
    const sceneSolids = liveSvoSceneSolidReach(scene);
    const world = await OctreeSparseBrickWorld.create(device, scene, dimensions, {
      brickSize: liveSvoRenderBrickSize(scene, options),
      environmentBrickRefinementLevels: options.environmentBrickRefinementLevels,
      environmentRefinementDepth: options.environmentRefinementDepth,
      environmentPlanarRefinementExemption: options.environmentPlanarRefinementExemption,
      radianceFeedback: options.radianceFeedback,
      derivedTraversalStructures: options.derivedTraversalStructures,
      progress: reportStage,
      sceneSolids,
      // The same build, one proxy at a time: what the curvature-driven
      // refinement rule evaluates when `FLUID_SVO_REFINEMENT_MODE=surface`.
      // Bound here for the reason directly above — the world cannot import it.
      environmentDescriptorFor: (primitive) => svoDescriptorForEnvironmentProxy(primitive),
      haloCells: 0,
      brickPreActivation: false,
    }, interrupt);
    return { dimensions, world };
  }

  private constructor(
    private readonly device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    dimensions: readonly [number, number, number],
    world: OctreeSparseBrickWorld,
  ) {
    /**
     * The lattice, and *only* the lattice.
     *
     * This used to be a `createTallCellLayout` call, which was 95 % of the wall
     * clock of a scene build — 121 s of 127 s on `hero-garden-hose` at
     * refinement depth 0, measured with `--cpu-prof`. Every millisecond of it
     * was discarded: the layout's expensive half is the initial *liquid* state
     * (`requiredInitialRegularLayers` and the column-base scan both walk all
     * `nx * fineNy * nz` cells asking `initialWet`, which asks the ground for
     * its height at up to four cells per visit), and this class owns no liquid
     * at all. Its three fluid textures are 1x1x1 stubs on purpose, `advanceTo`
     * returns false, and the five numbers actually read off the layout below —
     * `nx`, `fineNy`, `nz`, the cell size and the uniform cell count — are the
     * O(1) prefix of that function, computed before it touches a cell.
     *
     * `sceneLatticeDimensions` is that prefix, already extracted and already
     * the agreed rounding (`lib/scene-lattice.ts`): same floor of 8, same
     * `Math.round`, same device cap. So this is the same lattice, without the
     * 5.3 M-cell scan of a fluid state that does not exist.
     *
     * Computed by `buildWorld` now — the world has to be told the lattice
     * before this object exists — and handed in.
     */
    const cellSize_m = {
      x: scene.container.width_m / dimensions[0],
      y: scene.container.height_m / dimensions[1],
      z: scene.container.depth_m / dimensions[2],
    };
    const equivalentUniformCellCount = dimensions[0] * dimensions[1] * dimensions[2];
    const textureUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST;
    /**
     * The three fluid fields, at one texel each.
     *
     * `GPUSolverInstance` requires a `volumeTexture`, and the renderer binds
     * whatever this class hands it — so these have to exist. What they do not
     * have to be is dense. This class owns no transport and no projection:
     * `advanceTo` returns false, nothing here ever holds a `writeTexture` or a
     * storage write against them, and a texture WebGPU zero-initialises and
     * nobody writes reads back zero at every coordinate forever. Allocated at
     * the lattice they were 20 B/cell — `r32float` plus `rgba32float` over
     * `nx * fineNy * nz`, about 106 MB on the hero garden's container at
     * 6.25 mm — spent on a constant.
     *
     * The reads are unchanged because every one of them lands on zero either
     * way. The lattice the consumers iterate is `u.gridInfo.xyz` from
     * `info.nx/ny/nz`, not `textureDimensions`, so the dispatch shape is the
     * same; a live scene reports `gridKind: "octree"`, which selects the mode-3
     * branch of `fieldCell`/`fluidSample`/`velocitySample` where the load is
     * direct and the tall-cell `textureDimensions` branches are unreachable;
     * and a `textureLoad` outside a texture is defined to return either zeros
     * or some in-bounds texel, which here is also zero.
     *
     * The renderer already agrees with this: its own fallbacks for a solver
     * that declines these fields (`velocityFallbackTexture`,
     * `columnBaseTexture`) are 1³ and 1×1 for exactly the same reason.
     */
    const stub = [1, 1, 1] as const;
    this.volumeTexture = device.createTexture({
      label: "Live SVO empty-fluid field",
      size: stub,
      dimension: "3d",
      format: "r32float",
      usage: textureUsage,
    });
    this.surfaceFieldTexture = this.volumeTexture;
    this.velocityTexture = device.createTexture({
      label: "Live SVO zero-velocity field",
      size: stub,
      dimension: "3d",
      format: "rgba32float",
      usage: textureUsage,
    });
    this.columnBaseTexture = device.createTexture({
      label: "Live SVO column fallback",
      size: [1, 1],
      format: "r32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.world = world;
    this.sparseVoxelSceneSource = this.world.sceneSource;
    const h = Math.min(cellSize_m.x, cellSize_m.y, cellSize_m.z);
    this.info = {
      nx: dimensions[0],
      ny: dimensions[1],
      nz: dimensions[2],
      storedNy: 0,
      cellCount: 0,
      equivalentUniformCells: equivalentUniformCellCount,
      compressionRatio: equivalentUniformCellCount,
      activeCompressionRatio: equivalentUniformCellCount,
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
   * Build the sparse world, stepping the refinement depth down until it fits.
   *
   * Each extra level multiplies the occupied brick count by up to eight, so the
   * depth control is the one knob in the product that can ask for an allocation
   * the device cannot serve. Without the ladder, that request is a throw — and
   * a throw here resolves to `state:"unavailable"`, which cancels the
   * viewport's animation loop and terminates the render worker. There is no
   * retry after that and no frame after that: the tab is bricked until reload.
   *
   * So an unaffordable depth degrades to the finest one that fits instead, and
   * says so through `builtRefinementDepth`. The ladder is bounded by the depth
   * itself — at most three extra attempts, and depth zero is the shipping
   * allocation, whose failure is a real failure and rethrows unchanged.
   *
   * An abort is never a capacity answer and is never retried.
   */
  static async create(
    device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    progress: LiveSvoSceneProgress,
    signal?: AbortSignal,
    options: LiveSvoSceneOptions = {},
  ): Promise<WebGPULiveSvoScene> {
    const requested = Math.max(0, Math.trunc(options.environmentRefinementDepth ?? 0));
    for (let depth = requested; ; depth -= 1) {
      if (signal?.aborted) throw new DOMException("GPU initialization superseded", "AbortError");
      /**
       * Every stage of one rung, as one monotonic count.
       *
       * The build used to report `completed: 0` and then `completed: 1` with
       * everything expensive in between, which on a refined scene is minutes of
       * a bar that does not move. The stages are the world's own declared list
       * plus the two this layer owns — the solid reach that precedes it and the
       * shader compilation that follows — and they are reported as *started*,
       * so the label always names work in flight.
       *
       * The rung is part of the label rather than the count. A ladder that
       * steps down restarts the stages, and a bar that ran to 6 of 7 and then
       * went back to 1 of 7 with no explanation is worse than no bar; "at
       * refinement depth 2" is that explanation.
       */
      const rung = depth === requested ? "" : ` at refinement depth ${depth}`;
      const total = LIVE_SVO_BUILD_STAGE_COUNT;
      const reportStage: OctreeSparseBrickWorldProgress = (stage) => progress({
        phase: "allocation",
        taskId: "live-svo.allocate",
        label: `${stage.stage}${rung}`,
        // The solid reach is stage 0 and the world's list follows it, so the
        // world's own 0-based index is offset by one here.
        completed: stage.stage === LIVE_SVO_SOLID_REACH_STAGE ? 0 : stage.completed + 1,
        total,
      });
      progress({
        phase: "allocation",
        taskId: "live-svo.allocate",
        label: `Allocate live sparse scene${rung}`,
        completed: 0,
        total,
      });
      let source: WebGPULiveSvoScene | undefined;
      let world: OctreeSparseBrickWorld | undefined;
      try {
        // The world first, in slices. `OctreeSparseBrickWorld.create` destroys
        // its own partial state on abort, so the only window this has to cover
        // is a built world whose wrapper throws — hence `world` is tracked
        // separately from `source` right up to the statement that adopts it.
        const built = await WebGPULiveSvoScene.buildWorld(
          device, scene, { ...options, environmentRefinementDepth: depth }, reportStage,
          { signal, sliceBudget_ms: options.buildSliceBudget_ms });
        world = built.world;
        source = new WebGPULiveSvoScene(device, scene, quality, built.dimensions, built.world);
        world = undefined;
        progress({
          phase: "allocation",
          taskId: "live-svo.allocate",
          label: `Compile live scene programs${rung}`,
          completed: total - 1,
          total,
        });
        await source.world.initializePipelines();
        if (signal?.aborted) throw new DOMException("GPU initialization superseded", "AbortError");
      } catch (error) {
        // Whatever was allocated before the throw is released before the next
        // rung, or a three-step ladder would hold four worlds at once.
        source?.destroy();
        world?.destroy();
        if (depth <= 0 || signal?.aborted || (error instanceof DOMException && error.name === "AbortError")) throw error;
        progress({
          phase: "allocation",
          taskId: "live-svo.allocate",
          label: `Refinement depth ${depth} does not fit (${error instanceof Error ? error.message : "allocation failed"}); retrying at ${depth - 1}`,
          completed: 0,
          total,
        });
        continue;
      }
      source.requestedRefinementDepth = requested;
      source.builtRefinementDepth = depth;
      progress({
        phase: "allocation",
        taskId: "live-svo.allocate",
        label: depth === requested
          ? "Live sparse scene arenas ready"
          : `Live sparse scene arenas ready at refinement depth ${depth} (requested ${requested})`,
        completed: total,
        total,
      });
      return source;
    }
  }

  stageSceneUpdate(scene: SceneDescription): void { this.world.stageSceneUpdate(scene); }
  stageLivePrimitiveUpdates(updates: readonly SparseScenePrimitiveUpdate[]): boolean {
    return this.world.stageLivePrimitiveUpdates(updates);
  }
  encodeSceneMaintenance(encoder: GPUCommandEncoder, seam?: RenderFrameSeam<"world">): boolean {
    return this.world.encodeSceneMaintenance(encoder, false, seam);
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
  }
}
