import type {
  GPUSolverInstance,
  InjectedLiquidBall,
  MethodParamValues,
} from "../core/method-contract";
import type { SceneDescription } from "../core/model";
import type { GPUQuality } from "../core/gpu-quality";
import type { GPUEulerianInfo } from "../core/webgpu-eulerian";
import type { RigidBodyState } from "../core/rigid-body";
import { boundingRadius } from "../core/rigid-body";
import type { GPURigidBodyPose } from "../core/webgpu-rigid-body";
import { GPU_RIGID_BODY_CAPACITY, GPU_RIGID_RENDER_FLOATS } from "../core/webgpu-rigid-body";
import { SCENE_SHAPE_PALETTE_LINEAR, sceneShapeCode,
  sceneShapeRenderHalfExtent_m } from "../core/scene-shape";
import { PhysicsWasmClient, type PhysicsWasmClientOptions } from "./client";
import { decodeFluid3DPublication, type Fluid3DPublicationView } from "./fluid3d-view";
import type { PhysicsCommandReceipt, PhysicsPublication } from "./protocol";

/** Construction options passed unchanged to Rust's 3D world constructor. */
export interface RustFluid3DWorldOptions {
  readonly quality: GPUQuality;
  readonly methodValues: MethodParamValues;
  readonly [key: string]: unknown;
}

/** Narrow client seam used by the adapter and its headless protocol tests. */
export interface RustFluid3DClient {
  load(scene: unknown, options: unknown): Promise<PhysicsCommandReceipt>;
  advance(dt_s: number, viewMask?: number): Promise<PhysicsPublication>;
  applyCommand(command: unknown, viewMask?: number): Promise<PhysicsCommandReceipt | PhysicsPublication>;
  snapshot(viewMask?: number): Promise<PhysicsPublication>;
  destroy(): Promise<void>;
}

export interface RustFluid3DSolverCreateOptions {
  readonly wasm?: PhysicsWasmClientOptions;
  /** Test/integration seam; production omits it and receives a real worker client. */
  readonly client?: RustFluid3DClient;
  readonly viewMask?: number;
}

export interface RustRigidConstraint {
  readonly id: string;
  readonly held: boolean;
  readonly position_m: RigidBodyState["position_m"];
  readonly orientation: RigidBodyState["orientation"];
  readonly linearVelocity_m_s: RigidBodyState["linearVelocity_m_s"];
  readonly angularVelocity_rad_s: RigidBodyState["angularVelocity_rad_s"];
}

export interface RustRigidConstraintCommand {
  readonly type: "set-rigid-constraints";
  readonly constraints: readonly RustRigidConstraint[];
}

/** Only explicit hand constraints cross from the host into Rust pose authority. */
export class RustRigidConstraintTracker {
  private readonly heldById = new Map<string, string>();

  update(bodies: readonly RigidBodyState[]): RustRigidConstraintCommand | undefined {
    const constraints: RustRigidConstraint[] = [];
    for (const body of bodies) {
      const id = body.description.id;
      const constraint: RustRigidConstraint = {
        id, held: body.held === true,
        position_m: { ...body.position_m }, orientation: { ...body.orientation },
        linearVelocity_m_s: { ...body.linearVelocity_m_s },
        angularVelocity_rad_s: { ...body.angularVelocity_rad_s },
      };
      if (constraint.held) {
        const encoded = JSON.stringify(constraint);
        if (this.heldById.get(id) !== encoded) constraints.push(constraint);
        this.heldById.set(id, encoded);
      } else if (this.heldById.delete(id)) {
        constraints.push(constraint);
      }
    }
    return constraints.length ? { type: "set-rigid-constraints", constraints } : undefined;
  }
}

const finiteNumber = (value: unknown, fallback: number) =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

const paddedRowBytes = (bytes: number) => Math.ceil(bytes / 256) * 256;

const isPhysicsPublication = (
  value: PhysicsCommandReceipt | PhysicsPublication,
): value is PhysicsPublication => value.bytes instanceof Uint8Array
  && typeof value.release === "function";

/** Project Rust's sparse simulation census into the renderer's shared info contract. */
export function fluid3DEulerianInfo(
  view: Fluid3DPublicationView,
  quality: GPUQuality,
): GPUEulerianInfo {
  const [nx, ny, nz] = view.scene.dimensions;
  const densePresentationCells = nx * ny * nz;
  const activeCells = finiteNumber(view.stats.activeCells, densePresentationCells);
  const equivalentUniformCells = finiteNumber(
    view.stats.equivalentUniformCells, densePresentationCells,
  );
  const compressionRatio = finiteNumber(
    view.stats.compressionRatio,
    activeCells / Math.max(1, equivalentUniformCells),
  );
  const gpuTextureBytes = densePresentationCells * (4 + 4 + 16);
  const uploadScratchBytes = paddedRowBytes(nx * 16) * ny * nz;
  const rigidRenderBytes = GPU_RIGID_BODY_CAPACITY * GPU_RIGID_RENDER_FLOATS * 4;
  return {
    nx, ny, nz, storedNy: ny, cellCount: activeCells, equivalentUniformCells,
    compressionRatio, activeCompressionRatio: compressionRatio,
    activeSampleCount: activeCells, regularLayers: ny,
    maximumNeighborDelta: finiteNumber(view.stats.maximumNeighborDelta, 1),
    gridKind: "octree", cellSize_m: view.scene.cellSizeM[1], quality,
    pressureIterations: finiteNumber(view.stats.pressureIterations, 0),
    pressureSolver: typeof view.stats.pressureSolver === "string"
      ? view.stats.pressureSolver : "Rust CPU",
    // Rust's linear-memory census already includes its dense publication
    // vectors. Add only allocations owned by this presentation adapter.
    allocatedBytes: finiteNumber(view.stats.allocatedBytes, 0)
      + gpuTextureBytes + uploadScratchBytes + rigidRenderBytes,
    surfaceField: "levelset",
    // Preserve the physics-step count separately from Rust's surface
    // publication generation: paused edits can change the latter alone.
    encodedSteps: view.revision.frame,
    surfaceRevision: view.revision.surfaceRevision,
    submittedTime_s: view.revision.time,
    completedTime_s: view.revision.time,
  };
}

/** Pack Rust poses into the renderer's established 16-float RenderBody ABI. */
export function packRustRigidRenderRecords(
  authoredBodies: SceneDescription["rigidBodies"],
  poses: readonly GPURigidBodyPose[],
  selected = -1,
): Float32Array {
  const values = new Float32Array(GPU_RIGID_BODY_CAPACITY * GPU_RIGID_RENDER_FLOATS);
  for (let index = 0; index < Math.min(authoredBodies.length,
    poses.length, GPU_RIGID_BODY_CAPACITY); index += 1) {
    const description = authoredBodies[index]!;
    const pose = poses[index]!;
    const shape = sceneShapeCode(description.shape);
    const half = sceneShapeRenderHalfExtent_m(description.shape, description.dimensions_m);
    const color = SCENE_SHAPE_PALETTE_LINEAR[shape]!;
    values.set([pose.position_m.x, pose.position_m.y, pose.position_m.z,
      boundingRadius(description), half[0], half[1], half[2], shape,
      pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z,
      color[0], color[1], color[2], index === selected ? 1 : 0,
    ], index * GPU_RIGID_RENDER_FLOATS);
  }
  return values;
}

/**
 * GPU texture owner for a Rust/Wasm 3D simulation.
 *
 * Every physical mutation goes through `client`. WebGPU is used only to upload
 * completed immutable publications into the renderer's established textures.
 */
export class RustFluid3DGPUSolverAdapter implements GPUSolverInstance {
  readonly volumeTexture: GPUTexture;
  readonly surfaceFieldTexture: GPUTexture;
  readonly velocityTexture: GPUTexture;
  readonly rigidRenderBuffer: GPUBuffer;
  readonly fluidDomain: NonNullable<GPUSolverInstance["fluidDomain"]>;
  readonly simulationReady = true;

  private readonly dimensions: readonly [number, number, number];
  private readonly uploadScratch: Uint8Array;
  private readonly uploadSize: GPUExtent3DDict;
  private readonly viewMask: number;
  private maximumAdvanceDt_s: number;
  private currentInfo: GPUEulerianInfo;
  private presentedTime_s: number;
  private requestedTime_s: number;
  private frameWork?: Promise<void>;
  private commandWork?: Promise<void>;
  private failure?: Error;
  private disposed = false;
  private rigidBodies: readonly GPURigidBodyPose[] = [];
  private selectedRigidBody = -1;
  private readonly rigidConstraints = new RustRigidConstraintTracker();
  private runtimeValuesStamp: string;
  private sceneStamp: string;
  private topologyFrozen = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly client: RustFluid3DClient,
    first: Fluid3DPublicationView,
    viewMask: number,
    private readonly quality: GPUQuality,
    private authoredBodies: SceneDescription["rigidBodies"],
    scene: SceneDescription,
    methodValues: MethodParamValues,
  ) {
    this.dimensions = first.scene.dimensions;
    this.fluidDomain = {
      origin_m: first.scene.originM,
      cellSize_m: first.scene.cellSizeM,
      dimensions: first.scene.dimensions,
    };
    this.viewMask = viewMask >>> 0;
    this.maximumAdvanceDt_s = first.scene.dtS;
    this.sceneStamp = JSON.stringify(scene);
    this.runtimeValuesStamp = JSON.stringify(methodValues);
    const [nx, ny, nz] = this.dimensions;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
    this.volumeTexture = device.createTexture({
      label: "Rust CPU fluid density presentation",
      size: [nx, ny, nz], dimension: "3d", format: "r32float", usage,
    });
    this.surfaceFieldTexture = device.createTexture({
      label: "Rust CPU fluid signed-distance presentation",
      size: [nx, ny, nz], dimension: "3d", format: "r32float", usage,
    });
    this.velocityTexture = device.createTexture({
      label: "Rust CPU fluid velocity presentation",
      size: [nx, ny, nz], dimension: "3d", format: "rgba32float", usage,
    });
    this.rigidRenderBuffer = device.createBuffer({
      label: "Rust CPU rigid-body presentation",
      size: GPU_RIGID_BODY_CAPACITY * GPU_RIGID_RENDER_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.uploadScratch = new Uint8Array(paddedRowBytes(nx * 16) * ny * nz);
    this.uploadSize = { width: nx, height: ny, depthOrArrayLayers: nz };
    this.presentedTime_s = first.revision.time;
    this.requestedTime_s = this.presentedTime_s;
    this.currentInfo = this.infoFrom(first);
    this.rigidBodies = first.rigidBodies;
    this.uploadRigidBodies();
    try { this.upload(first); } finally { first.release(); }
  }

  static async create(
    device: GPUDevice,
    scene: SceneDescription,
    worldOptions: RustFluid3DWorldOptions,
    options: RustFluid3DSolverCreateOptions = {},
  ): Promise<RustFluid3DGPUSolverAdapter> {
    const client = options.client ?? await PhysicsWasmClient.create(options.wasm);
    try {
      const receipt = await client.load(scene, { ...worldOptions, dimension: 3 });
      if (receipt.dimension !== 3) {
        throw new Error("Rust CPU backend loaded a world that is not three-dimensional");
      }
      const publication = await client.snapshot(options.viewMask ?? 0xffffffff);
      const first = decodeFluid3DPublication(publication);
      return new RustFluid3DGPUSolverAdapter(
        device, client, first, options.viewMask ?? 0xffffffff, worldOptions.quality,
        scene.rigidBodies, scene, worldOptions.methodValues,
      );
    } catch (error) {
      await client.destroy();
      throw error;
    }
  }

  get info(): GPUEulerianInfo { return this.currentInfo; }
  get framePending(): boolean {
    return this.frameWork !== undefined || this.commandWork !== undefined;
  }

  advanceTo(time_s: number, bodies: RigidBodyState[]): boolean {
    this.assertLive();
    // A drag is an explicit kinematic constraint. Queue it before checking the
    // simulation clock so paused manipulation still reaches Rust. Ordinary
    // unheld host poses remain presentation state and never overwrite Rust.
    const constraints = this.rigidConstraints.update(bodies);
    if (constraints) void this.queueCommand(constraints);
    if (this.framePending || !(time_s > this.requestedTime_s) || !Number.isFinite(time_s)) return false;
    const submittedFrom_s = this.requestedTime_s;
    const requestedSpan_s = time_s - submittedFrom_s;
    const dt_s = Math.min(this.maximumAdvanceDt_s, requestedSpan_s);
    const admittedTime_s = submittedFrom_s + dt_s;
    this.requestedTime_s = admittedTime_s;
    this.currentInfo.submittedTime_s = admittedTime_s;
    const work = this.client.advance(dt_s, this.viewMask)
      .then(publication => {
        this.adoptPublication(publication, true);
      });
    const guarded = work.catch(error => {
      const detail = error instanceof Error ? error.message : String(error);
      this.failure = new Error(`Rust CPU advance failed for target ${time_s} s from submitted ${submittedFrom_s} s `
        + `(admitted ${admittedTime_s} s, presented ${this.presentedTime_s} s, dt ${dt_s} s, maximum step ${this.maximumAdvanceDt_s} s): ${detail}`, {
        cause: error,
      });
      this.requestedTime_s = this.presentedTime_s;
      throw this.failure;
    }).finally(() => {
      if (this.frameWork === guarded) this.frameWork = undefined;
    });
    this.frameWork = guarded;
    return true;
  }

  async awaitFrameCompletion(): Promise<void> {
    await Promise.all([this.commandWork, this.frameWork]);
  }

  async assertSimulationHealthy(): Promise<void> {
    await this.awaitFrameCompletion();
    if (this.failure) throw this.failure;
  }

  async readStats(): Promise<GPUEulerianInfo> {
    await this.awaitFrameCompletion();
    if (this.failure) throw this.failure;
    return { ...this.currentInfo };
  }

  async readRigidBodyPoses(): Promise<GPURigidBodyPose[]> {
    await this.awaitFrameCompletion();
    return this.rigidBodies.map(pose => ({
      position_m: { ...pose.position_m }, orientation: { ...pose.orientation },
    }));
  }

  setSelectedRigidBody(index: number): void {
    this.selectedRigidBody = Number.isSafeInteger(index) ? index : -1;
    this.uploadRigidBodies();
  }

  injectLiquidBall(ball: InjectedLiquidBall): void {
    void this.queueCommand({ type: "inject-liquid", drop: {
      centre_m: ball.centre_m,
      radius_m: ball.radius_m,
      ...(ball.halfHeight_m === undefined ? {} : { halfHeight_m: ball.halfHeight_m }),
    } });
  }

  setTopologyFrozen(frozen: boolean): void {
    if (frozen === this.topologyFrozen) return;
    this.topologyFrozen = frozen;
    void this.queueCommand({ type: "set-topology-frozen", frozen });
  }

  stageSceneUpdate(scene: SceneDescription): void {
    const stamp = JSON.stringify(scene);
    if (stamp === this.sceneStamp) return;
    this.sceneStamp = stamp;
    void this.queueCommand({ type: "set-scene", scene }, () => {
      this.authoredBodies = scene.rigidBodies;
    });
  }

  applyRuntimeValues(values: MethodParamValues): void {
    const stamp = JSON.stringify(values);
    if (stamp === this.runtimeValuesStamp) return;
    this.runtimeValuesStamp = stamp;
    void this.queueCommand({ type: "set-runtime-values", values });
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.volumeTexture.destroy();
    this.surfaceFieldTexture.destroy();
    this.velocityTexture.destroy();
    this.rigidRenderBuffer.destroy();
    void this.client.destroy();
  }

  private queueCommand(command: unknown, beforeUpload?: () => void): Promise<boolean> {
    this.assertLive();
    const work = this.client.applyCommand(command, this.viewMask).then(result => {
      if (!isPhysicsPublication(result)) {
        throw new Error("Rust CPU command did not return its immutable publication");
      }
      this.adoptPublication(result, false, beforeUpload);
      return true;
    }).catch(error => {
      this.failure = error instanceof Error ? error : new Error(String(error));
      return false;
    });
    const pending = work.then(() => undefined).finally(() => {
      if (this.commandWork === pending) this.commandWork = undefined;
    });
    this.commandWork = pending;
    return work;
  }

  private adoptPublication(
    publication: PhysicsPublication,
    advanced: boolean,
    beforeUpload?: () => void,
  ): void {
    const view = decodeFluid3DPublication(publication);
    try {
      if (this.disposed) return;
      this.assertSameLattice(view);
      beforeUpload?.();
      this.upload(view);
      if (advanced) this.presentedTime_s = view.revision.time;
      this.maximumAdvanceDt_s = view.scene.dtS;
      this.currentInfo = this.infoFrom(view);
      this.currentInfo.submittedTime_s = this.requestedTime_s;
      this.rigidBodies = view.rigidBodies;
      this.uploadRigidBodies();
    } finally {
      view.release();
    }
  }

  private upload(view: Fluid3DPublicationView): void {
    this.uploadTexture(this.volumeTexture, view.density, 1);
    this.uploadTexture(this.surfaceFieldTexture, view.surfacePhi, 1);
    if (view.velocity) this.uploadTexture(this.velocityTexture, view.velocity, 4);
  }

  private uploadRigidBodies(): void {
    const values = packRustRigidRenderRecords(
      this.authoredBodies, this.rigidBodies, this.selectedRigidBody,
    );
    this.device.queue.writeBuffer(this.rigidRenderBuffer, 0, values.buffer as ArrayBuffer);
  }

  private uploadTexture(texture: GPUTexture, values: Float32Array<ArrayBuffer>, channels: 1 | 4): void {
    const [nx, ny, nz] = this.dimensions;
    const rowBytes = nx * channels * 4;
    const bytesPerRow = paddedRowBytes(rowBytes);
    const source = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
    for (let z = 0; z < nz; z += 1) {
      for (let y = 0; y < ny; y += 1) {
        const sourceOffset = (z * ny + y) * rowBytes;
        const targetOffset = (z * ny + y) * bytesPerRow;
        this.uploadScratch.set(source.subarray(sourceOffset, sourceOffset + rowBytes), targetOffset);
      }
    }
    this.device.queue.writeTexture(
      { texture }, this.uploadScratch.buffer as ArrayBuffer,
      { bytesPerRow, rowsPerImage: ny }, this.uploadSize,
    );
  }

  private assertSameLattice(view: Fluid3DPublicationView): void {
    if (view.scene.dimensions.some((value, axis) => value !== this.dimensions[axis])) {
      throw new RangeError("Rust CPU backend changed presentation dimensions without a solver rebuild");
    }
  }

  private infoFrom(view: Fluid3DPublicationView): GPUEulerianInfo {
    return fluid3DEulerianInfo(view, this.quality);
  }

  private assertLive(): void {
    if (this.disposed) throw new Error("Rust CPU fluid solver is destroyed");
    if (this.failure) throw this.failure;
  }
}
