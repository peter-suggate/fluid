/** Lane P's rigid-body host for the brick engine (M5 design note §7.1, A3, A4): the shared WebGPURigidBodySystem, built
 * as the old host builds it (webgpu-uniform-reference.ts:932-946), and the BrickBodiesBinding the engine binds.
 *
 * - Buffers: the system's state buffer is `bodies` (A3); `bodyCuts` is this host's cut atlas (packRigidBodyCutAtlas,
 *   written only when the signed cut list changes, after pbCheckCutAtlas); `bodyExchange` is the system's exchange.
 * - sync(-1): the t = 0 roster (initializeRigidBodies, the prescribed sample at 0), as the old construction syncs it.
 *   sync(n): the roster the host offered (offer: the app's advanceTo bodies), the prescribed sample at the step's end
 *   (n + 1) dt with dt (UniformPrescribedSolidMotion, the old advance's), into syncBodies, which writes the state only
 *   on a change (its unchangedSinceSync fast path). Bodies are GPU-owned between syncs.
 * - integrate(n): rigidSystem.encode(encoder, dt, h^3, 1, h) (tables, then integrate), then the exchange's clear, so
 *   the next couple starts from zero. Off (the old host's uncoupled bodies) when the coupling is off.
 * - Explicit refusals: more than GPU_RIGID_BODY_CAPACITY bodies, a rounded plate (the brick stamp has no plate), a cut
 *   atlas pbCheckCutAtlas rejects; motor assemblies with static voxels or terrain (the old host's procedural collision
 *   source, which the brick engine does not have yet): brickRigidBodyGaps names them to the scene adapter. Motor
 *   assemblies in a tank with no static voxel contact the container planes (the shell).
 * - Terrain: the old host's texture, the ground's columns in cells (brickTerrainColumns, the same bake), so the
 *   integrator's terrainPlane contact is the old one's. The old host attaches its sculpt source too, so its contact is
 *   editedTerrainContact; with no sculpt cut (a sculpted ground is the scene adapter's gap) that is staticSurfaceContact
 *   at the centre distance n.x - w, the same penetration as staticPlaneContact's to rounding. A scene with no terrain
 *   binds one zero texel, as the old host's.
 * - Save/load/undo: the rigid state is the system's (stateBuffer), not the engine's; save() and restore() copy it. */
import type { SceneDescription, RigidBodyDescription } from "../../core/model";
import { sceneHasTerrain } from "../../core/terrain";
import type { BrickTerrainColumns } from "./brick-terrain.wgsl";
import { solidVoxelEditsForScene } from "../../core/scene-lattice";
import { initializeRigidBodies, type RigidBodyState } from "../../core/rigid-body";
import { WebGPURigidBodySystem, GPU_RIGID_STATE_BYTES } from "../../core/webgpu-rigid-body";
import { GPU_RIGID_EXCHANGE_BYTES } from "../../core/webgpu-eulerian";
import { GPU_RIGID_BODY_CAPACITY } from "../../core/rigid-capacity";
import { RIGID_BODY_CUT_ATLAS_WIDTH, packRigidBodyCutAtlas } from "../../core/solid-cuts";
import { UniformPrescribedSolidMotion } from "../uniform/uniform-prescribed-solid-motion";
import { BODY_FLAG } from "./brick-abi";
import { pbCheckCutAtlas } from "./bodies-shapes.wgsl";
import type { BrickBodiesBinding, BrickBodiesHeader, Vec3 } from "./bodies-binding.wgsl";

export interface BrickRigidBodiesOptions {
  /** The lattice's low corner (m) and cell (m): BrickSceneSetup.origin_m and h_m. */
  readonly frame_m: Vec3;
  readonly h_m: number;
  readonly dt_s: number;
  readonly density_kg_m3: number;
  /** The method's rigidCoupling value (default on): off runs neither couple nor the integrator. */
  readonly coupling?: boolean;
  /** The ground's columns (BrickSceneSetup.terrain): required when the scene has terrain (sceneHasTerrain). */
  readonly terrain?: BrickTerrainColumns;
}

/** Why a scene's bodies cannot run on the brick engine's body stages (empty when they can). */
export function brickRigidBodyGaps(scene: SceneDescription): string[] {
  const gaps: string[] = [], bodies = scene.rigidBodies ?? [];
  if (bodies.length === 0) return gaps;
  if (bodies.length > GPU_RIGID_BODY_CAPACITY) gaps.push(`${bodies.length} rigid bodies exceed the rigid capacity ${GPU_RIGID_BODY_CAPACITY}`);
  // Motor assemblies: the old host gives them the fluid's solid mask and the tank's shell as their solid
  // (rigidAssemblySolidWGSL). With no static voxel, that mask is empty and the shell is the container's walls and floor
  // (and lid, when closed): the rigid system's container planes, which solve an assembly piece by piece the same way
  // (staticPlaneContact -> assemblyPlaneContact). With static voxels, the mask must be bound as the rigid system's
  // procedural source, and it is not yet.
  const motor = bodies.find(b => b.motorAttachment !== undefined);
  // With terrain, the old host's assembly solid drops its floor for the ground (rigidAssemblySolidLayout floor:
  // !terrainContact): an assembly then contacts the ground through the same procedural source.
  if (motor && sceneHasTerrain(scene)) gaps.push(`rigid body ${motor.id}: motor assemblies contact the ground through the old host's procedural source, not wired here (M5 step 3)`);
  if (motor && solidVoxelEditsForScene(scene).length > 0) gaps.push(`rigid body ${motor.id}: motor assemblies contact the static voxels through the old host's procedural source, not wired here (M5 step 3)`);
  const plate = bodies.find(b => b.shape === "rounded-plate");
  if (plate) gaps.push(`rigid body ${plate.id}: a rounded plate has no brick stamp`);
  return gaps;
}

export class BrickRigidBodies implements BrickBodiesBinding {
  readonly frame_m: Vec3;
  readonly h_m: number;
  readonly openTop: boolean;
  readonly density_kg_m3: number;
  readonly system: WebGPURigidBodySystem;
  readonly exchange: GPUBuffer;
  readonly cuts: GPUBuffer;
  /** The rigid system's terrain binding: one zero texel (a scene with no terrain reads every column at zero, as the old
   * host's 1x1 texture; the integrator's terrain flag is off, sceneHasTerrain). */
  private readonly terrain: GPUTexture;
  get bodies(): GPUBuffer { return this.system.stateBuffer; }
  /** Optional: a buffer (COPY_DST, at least GPU_RIGID_EXCHANGE_BYTES) the exchange is copied into each step before the
   * integrator, for observers (tools/nbflip-rigid-gates.ts). Never set on the app's path. */
  tap?: GPUBuffer;

  private roster: readonly RigidBodyState[];
  private readonly motion = new UniformPrescribedSolidMotion();
  private readonly coupling: boolean;
  private cutsSigned: readonly RigidBodyDescription[] = [];
  private cutsKey = "";
  private shapeKey = "";
  private generation = 0;
  private header: BrickBodiesHeader = { count: 0, flags: 0, generation: 0 };

  private constructor(private readonly device: GPUDevice, private readonly scene: SceneDescription, private readonly options: BrickRigidBodiesOptions) {
    const gaps = brickRigidBodyGaps(scene);
    if (gaps.length) throw new Error(`Brick rigid bodies: ${gaps.join("; ")}`);
    if (!(options.dt_s > 0 && options.h_m > 0)) throw new Error(`Brick rigid bodies: dt ${options.dt_s} s and h ${options.h_m} m must be positive`);
    this.frame_m = options.frame_m; this.h_m = options.h_m; this.density_kg_m3 = options.density_kg_m3;
    this.openTop = scene.container.top !== "closed";
    this.coupling = options.coupling !== false;
    this.exchange = device.createBuffer({ label: "nb-brick.rigid exchange", size: GPU_RIGID_EXCHANGE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.cuts = device.createBuffer({ label: "nb-brick.body cuts", size: 16 * RIGID_BODY_CUT_ATLAS_WIDTH, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    // The old host's construction, less the procedural (sculpt) collision source: the scene adapter refuses scenes that
    // need it.
    const ground = sceneHasTerrain(scene) ? options.terrain : undefined;
    if (sceneHasTerrain(scene) && !ground) throw new Error("Brick rigid bodies: the scene has terrain; pass its columns (BrickSceneSetup.terrain)");
    const [nx, nz] = ground?.columns ?? [1, 1];
    this.terrain = device.createTexture({ label: ground ? "nb-brick.rigid terrain" : "nb-brick.rigid terrain (none)", size: [nx, nz], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    if (ground) device.queue.writeTexture({ texture: this.terrain }, Float32Array.from(ground.heights), { bytesPerRow: 4 * nx, rowsPerImage: nz }, [nx, nz]);
    this.system = new WebGPURigidBodySystem(device, scene, this.exchange, this.terrain, [nx, nz]);
    this.roster = initializeRigidBodies(scene.rigidBodies ?? []);
  }

  static async create(device: GPUDevice, scene: SceneDescription, options: BrickRigidBodiesOptions): Promise<BrickRigidBodies> {
    const host = new BrickRigidBodies(device, scene, options);
    const signal = new AbortController().signal;
    for (const task of host.system.initializationTasks()) await task.run(signal);
    return host;
  }

  /** The app's roster for the next step (advanceTo's bodies). */
  offer(bodies: readonly RigidBodyState[]) { this.roster = bodies; }

  sync(step: number): BrickBodiesHeader {
    if (!Number.isInteger(step) || step < -1) throw new Error(`Brick rigid bodies: sync step ${step}`);
    const roster = this.roster;
    if (roster.length > GPU_RIGID_BODY_CAPACITY) throw new Error(`Brick rigid bodies: ${roster.length} bodies exceed the rigid capacity ${GPU_RIGID_BODY_CAPACITY}`);
    const dt = this.options.dt_s;
    const active = this.scene.interaction?.entities.length ? roster
      : step < 0 ? this.motion.sample(roster, 0) : this.motion.sample(roster, dt, (step + 1) * dt);
    const plate = active.find(b => b.description.shape === "rounded-plate");
    if (plate) throw new Error(`Brick rigid bodies: rigid body ${plate.description.id}: a rounded plate has no brick stamp`);
    this.syncCuts(active.map(b => b.description));
    this.system.syncBodies(active);
    const shapeKey = JSON.stringify(active.map(b => [b.description.id, b.description.shape, b.description.dimensions_m, b.description.fluidBoundary ?? "", b.description.cuts ?? []]));
    if (shapeKey !== this.shapeKey) { this.shapeKey = shapeKey; this.generation = (this.generation + 1) >>> 0; }
    let flags = this.coupling && active.length > 0 ? BODY_FLAG.COUPLING : 0;
    if (active.some(b => (b.description.cuts?.length ?? 0) > 0)) flags |= BODY_FLAG.CUTS;
    if (active.some(b => b.description.fluidBoundary === "interior")) flags |= BODY_FLAG.INVERTED;
    if (this.header.count !== active.length || this.header.flags !== flags || this.header.generation !== this.generation)
      this.header = { count: active.length, flags, generation: this.generation };
    return this.header;
  }

  integrate(step: number): GPUCommandBuffer | undefined {
    if (!this.coupling || this.header.count === 0) return undefined;
    const h = this.options.h_m, encoder = this.device.createCommandEncoder({ label: `nb-brick.step${step}.rigid` });
    if (this.tap) encoder.copyBufferToBuffer(this.exchange, 0, this.tap, 0, GPU_RIGID_EXCHANGE_BYTES);
    this.system.encode(encoder, this.options.dt_s, h * h * h, 1, h);
    encoder.clearBuffer(this.exchange);
    return encoder.finish();
  }

  /** The setBodyCuts cache (uniform-mixed-solid.wgsl.ts:458-468): the atlas only when the signed cut list changes. */
  private syncCuts(bodies: readonly RigidBodyDescription[]) {
    const signed = this.cutsSigned;
    if (bodies.length === signed.length && bodies.every((b, i) => b === signed[i])) return;
    const key = JSON.stringify(bodies.map(b => [b.id, b.cuts ?? []]));
    if (key !== this.cutsKey) {
      const atlas = packRigidBodyCutAtlas(bodies, GPU_RIGID_BODY_CAPACITY);
      pbCheckCutAtlas(atlas, bodies.length);
      this.device.queue.writeBuffer(this.cuts, 0, atlas);
      this.cutsKey = key;
    }
    this.cutsSigned = [...bodies];
  }

  get renderBuffer(): GPUBuffer { return this.system.renderBuffer; }
  get motionBuffer(): GPUBuffer { return this.system.motionBuffer; }
  readPoses() { return this.system.readPoses(); }

  /** The rigid state after a fence (save points, undo): the system's state buffer. */
  async save(): Promise<Uint32Array<ArrayBuffer>> {
    const staging = this.device.createBuffer({ size: GPU_RIGID_STATE_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(this.system.stateBuffer, 0, staging, 0, GPU_RIGID_STATE_BYTES);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(staging.getMappedRange().slice(0)); staging.unmap(); staging.destroy();
    return out;
  }
  restore(state: Uint32Array<ArrayBuffer>) {
    if (state.byteLength !== GPU_RIGID_STATE_BYTES) throw new Error(`Brick rigid bodies: restore has ${state.byteLength} bytes, the state ${GPU_RIGID_STATE_BYTES}`);
    this.device.queue.writeBuffer(this.system.stateBuffer, 0, state);
    this.device.queue.writeBuffer(this.exchange, 0, new Uint32Array(GPU_RIGID_EXCHANGE_BYTES / 4));
  }

  destroy() { this.system.destroy(); this.exchange.destroy(); this.cuts.destroy(); this.terrain.destroy(); }
}
