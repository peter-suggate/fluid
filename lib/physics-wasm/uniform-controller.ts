import { PhysicsWasmClient, type PhysicsWasmClientOptions } from "./client";
import { detectPhysicsWasmCapabilities } from "./capabilities";
import { decodePhysicsPublication, PhysicsPlane } from "./publication";
import type { PhysicsPublication, PhysicsRevision } from "./protocol";
import type {
  SceneDescription,
  RigidBodyDescription,
  Vec3,
} from "../core/model";
import { sceneLatticeDimensions } from "../core/scene-lattice";
import { sampleSolidWorld, solidWorldForScene } from "../core/solid-world";
import { sceneHasTerrain } from "../core/terrain";
import {
  uniformInitialVolume,
  uniformVolumeInitialPhi,
} from "../methods/uniform/uniform-volume-initial";
import { resolveUniformGeometricValues } from "../methods/uniform/uniform-geometric-parameters";
import { NARROW_BAND_FINE_ADVECTION, narrowBandControlValues } from "../methods/uniform/uniform-narrow-band-controls";
import { scenerySliceFraction, uniformLabSlice } from "./scenery-slice";

export const UNIFORM_LAB_METHOD = "uniform-narrow-band-flip";
const shared = resolveUniformGeometricValues();
/** The 3D method's values the native port takes (NarrowBandFlipOptions):
 * its fixed pressure schedule, its one extension sweep, its join padding and
 * its volume control. 2D runs the fixed particle band on the mixed 4h/h frame. */
export const UNIFORM_LAB_VALUES = Object.freeze({
  rigidCoupling: shared.rigidCoupling as string,
  pressureResidualTolerance: shared.pressureResidualTolerance as number,
  extensionFrontSweeps: 1,
  pressureFullCycles: 3,
  pressureVCycles: 4,
  pressureSweeps: 6,
  fineGridPadding: 0,
  phiFineAdvection: NARROW_BAND_FINE_ADVECTION.default as "cubic" | "trilinear" | "adaptive",
  volumeControlSeconds: narrowBandControlValues().volumeControlSeconds,
});
/** The method controls the lab lets a viewer change. */
export type UniformLabControls = Pick<typeof UNIFORM_LAB_VALUES, "phiFineAdvection" | "volumeControlSeconds">;

export function uniformLabSceneLimitation(
  scene: SceneDescription,
): string | undefined {
  const [nx, ny] = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  if (nx < 2 || ny < 2 || nx * ny > 4_194_304)
    return "This scene exceeds the 2D grid budget.";
  if (nx % 4 !== 0 || ny % 4 !== 0 || nx < 8 || ny < 8)
    return "Narrow-band FLIP needs a lattice of whole 4h tiles, at least two a side.";
  return undefined;
}
/** XY slice with scenery baked into the static fluid capacity. `volume` is
 * the Uniform Geometric scene runner's V; narrow-band FLIP takes the rest. */
export function uniformLabSeed(scene: SceneDescription, sliceDepth_m?: number) {
  const limitation = uniformLabSceneLimitation(scene);
  if (limitation) throw new Error(limitation);
  const dimensions = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  const [nx, ny, nz] = dimensions;
  const { index: z, z_m } = uniformLabSlice(scene, nz, sliceDepth_m);
  const cellSize = [
    scene.container.width_m / nx,
    scene.container.height_m / ny,
  ];
  const { volume, terrain } = uniformInitialVolume(scene, dimensions, true, z);
  const solid = solidWorldForScene(scene);
  const scenery = scenerySliceFraction(scene, dimensions, z_m);
  const capacity = Array.from({ length: nx * ny }, (_, i) => {
    const x = i % nx,
      y = Math.floor(i / nx);
    const open = 1 - sampleSolidWorld(solid, [x, y, z]).solidFraction;
    const terrainFraction = sceneHasTerrain(scene)
      ? Math.min(1, Math.max(0, terrain[x + nx * z]! / cellSize[1]! - y))
      : 0;
    return Math.min(open * (1 - terrainFraction), 1 - scenery[i]!);
  });
  return {
    dimensions: [nx, ny],
    cellSize,
    volume: Array.from(volume, (value, i) => Math.min(value, capacity[i]!)),
    capacity,
    phi: [...uniformVolumeInitialPhi(scene, dimensions, z + 0.5)],
    gravity: [scene.fluid.gravity_m_s2.x, scene.fluid.gravity_m_s2.y],
    density: scene.fluid.density_kg_m3,
    viscosity: scene.fluid.dynamicViscosity_Pa_s,
    surfaceTension: scene.fluid.surfaceTension_N_m,
    openTop: scene.container.top === "open",
  };
}

/** Floats a published sample holds: position in cells, velocity in m/s,
 * depth in cells (negative inside), ballistic. */
export const UNIFORM_PARTICLE_STRIDE = 6;
export interface UniformView {
  readonly revision: PhysicsRevision;
  readonly nx: number;
  readonly ny: number;
  readonly cellSize: readonly [number, number];
  readonly volume: Float32Array;
  readonly capacity: Float32Array;
  readonly phi: Float32Array;
  readonly velocity: Float32Array;
  readonly pressure: Float32Array;
  readonly lowX: Float32Array;
  readonly lowY: Float32Array;
  readonly released: Uint8Array;
  readonly tiles: Uint8Array;
  readonly particles: Float32Array;
  readonly receipt: Readonly<Record<string, unknown>>;
}
function particles(plane: Float32Array<ArrayBuffer>) {
  if (plane.length % UNIFORM_PARTICLE_STRIDE !== 0)
    throw new Error("Invalid narrow-band FLIP particle plane");
  return plane;
}
export function createUniformView(source: PhysicsPublication): UniformView {
  const publication = decodePhysicsPublication(source);
  try {
    const m = publication.metadata;
    if (m.method !== UNIFORM_LAB_METHOD)
      throw new Error("Expected narrow-band FLIP publication");
    const dims = m.dimensions as number[],
      spacing = m.cellSize as number[];
    if (
      !Array.isArray(dims) ||
      dims.length !== 2 ||
      !dims.every((n) => Number.isSafeInteger(n) && n >= 2) ||
      !Array.isArray(spacing) ||
      spacing.length !== 2 ||
      !spacing.every((h) => Number.isFinite(h) && h > 0)
    )
      throw new Error("Invalid uniform lattice metadata");
    const [nx, ny] = dims as [number, number];
    const f32 = (
      id: (typeof PhysicsPlane)[keyof typeof PhysicsPlane],
      count?: number,
    ) => {
      const p = publication.plane(id);
      if (!(p instanceof Float32Array) || (count !== undefined && p.length !== count))
        throw new Error(`Invalid uniform plane ${id}`);
      return p.slice();
    };
    const u8 = (
      id: (typeof PhysicsPlane)[keyof typeof PhysicsPlane],
      count?: number,
    ) => {
      const p = publication.plane(id);
      if (
        !(p instanceof Uint8Array) ||
        (count !== undefined && p.length !== count)
      )
        throw new Error(`Invalid uniform plane ${id}`);
      return p.slice();
    };
    return {
      revision: { ...source.revision },
      nx,
      ny,
      cellSize: spacing as [number, number],
      volume: f32(PhysicsPlane.Density, nx * ny),
      capacity: f32(PhysicsPlane.Capacity, nx * ny),
      phi: f32(PhysicsPlane.RdfVertices, (nx + 1) * (ny + 1)),
      velocity: f32(PhysicsPlane.CellVelocity, 2 * nx * ny),
      pressure: f32(PhysicsPlane.Pressure, nx * ny),
      lowX: f32(PhysicsPlane.UniformLowX, ny),
      lowY: f32(PhysicsPlane.UniformLowY, nx),
      released: u8(PhysicsPlane.UniformReleased, nx * ny),
      tiles: u8(PhysicsPlane.UniformTiles),
      particles: particles(f32(PhysicsPlane.UniformParticles)),
      receipt: m.receipt as Record<string, unknown>,
    };
  } finally {
    publication.release();
  }
}
export class UniformLabController {
  private constructor(private readonly client: PhysicsWasmClient) {}
  static async create(options: PhysicsWasmClientOptions = {}) {
    return new UniformLabController(
      await PhysicsWasmClient.create({
        artifact: detectPhysicsWasmCapabilities().simd ? "simd" : "scalar",
        ...options,
      }),
    );
  }
  async load(scene: SceneDescription, controls: Partial<UniformLabControls> = {}, sliceDepth_m?: number) {
    // The method's liquid is its surface: the seed's cell volumes stay behind.
    const uniformSeed: Partial<ReturnType<typeof uniformLabSeed>> = uniformLabSeed(scene, sliceDepth_m);
    delete uniformSeed.volume;
    await this.client.load(scene, {
      method: UNIFORM_LAB_METHOD,
      dimension: 2,
      methodValues: { ...UNIFORM_LAB_VALUES, ...controls },
      uniformSeed,
    });
    return createUniformView(await this.client.snapshot());
  }
  async advance(dt: number) {
    return createUniformView(await this.client.advance(dt));
  }
  async injectLiquid(centre_m: readonly [number, number], radius_m: number) {
    return this.command({
      type: "inject-liquid",
      drop: { centre_m, radius_m },
    });
  }
  async addRigidBody(body: RigidBodyDescription, held = false) {
    return this.command({ type: "add-rigid-body", body, held });
  }
  async removeRigidBody(id: string) {
    return this.command({ type: "remove-rigid-body", id });
  }
  async setRigidPose(
    id: string,
    position_m: Vec3,
    velocity_m_s: Vec3,
    held: boolean,
  ) {
    return this.command({
      type: "set-rigid-pose",
      id,
      position_m,
      velocity_m_s,
      held,
    });
  }
  private async command(command: unknown) {
    const publication = await this.client.applyCommand(command, 0xffffffff);
    if (!("bytes" in publication))
      throw new Error("Narrow-band FLIP edit did not publish fields");
    return createUniformView(publication as PhysicsPublication);
  }
  destroy() {
    return this.client.destroy();
  }
}
