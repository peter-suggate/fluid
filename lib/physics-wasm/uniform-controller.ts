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

/** The only lab override: window scheduling is deferred, as requested. */
export const UNIFORM_LAB_VALUES = Object.freeze(
  resolveUniformGeometricValues({ activeRegion: "off" }),
);

export function uniformLabSceneLimitation(
  scene: SceneDescription,
): string | undefined {
  if (scene.fluid.inflow)
    return "Inflow scenes are not available in Uniform 2D yet.";
  const [nx, ny] = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  if (nx < 2 || ny < 2 || nx * ny > 4_194_304)
    return "This scene exceeds the Uniform 2D grid budget.";
  return undefined;
}
/** Central XY slice, sampled by the same initial-field functions as 3D. */
export function uniformLabSeed(scene: SceneDescription) {
  const limitation = uniformLabSceneLimitation(scene);
  if (limitation) throw new Error(limitation);
  const dimensions = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  const [nx, ny, nz] = dimensions;
  const z = Math.floor(nz / 2);
  const cellSize = [
    scene.container.width_m / nx,
    scene.container.height_m / ny,
  ];
  const { volume, terrain } = uniformInitialVolume(scene, dimensions, true, z);
  const solid = solidWorldForScene(scene);
  const capacity = Array.from({ length: nx * ny }, (_, i) => {
    const x = i % nx,
      y = Math.floor(i / nx);
    const open = 1 - sampleSolidWorld(solid, [x, y, z]).solidFraction;
    const terrainFraction = sceneHasTerrain(scene)
      ? Math.min(1, Math.max(0, terrain[x + nx * z]! / cellSize[1]! - y))
      : 0;
    return open * (1 - terrainFraction);
  });
  return {
    dimensions: [nx, ny],
    cellSize,
    volume: [...volume],
    capacity,
    phi: [...uniformVolumeInitialPhi(scene, dimensions, nz / 2)],
    gravity: [scene.fluid.gravity_m_s2.x, scene.fluid.gravity_m_s2.y],
    density: scene.fluid.density_kg_m3,
    viscosity: scene.fluid.dynamicViscosity_Pa_s,
    surfaceTension: scene.fluid.surfaceTension_N_m,
    openTop: scene.container.top === "open",
  };
}

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
  readonly receipt: Readonly<Record<string, unknown>>;
}
export function createUniformView(source: PhysicsPublication): UniformView {
  const publication = decodePhysicsPublication(source);
  try {
    const m = publication.metadata;
    if (m.method !== "uniform-volume")
      throw new Error("Expected Uniform Geometric publication");
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
      count: number,
    ) => {
      const p = publication.plane(id);
      if (!(p instanceof Float32Array) || p.length !== count)
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
      receipt: m.receipt as Record<string, unknown>,
    };
  } finally {
    publication.release();
  }
}
export type SurfaceExperiment = "off" | "regional" | "regional-area" | "area-only";
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
  async load(scene: SceneDescription, experiment: SurfaceExperiment = "area-only") {
    await this.client.load(scene, {
      method: "uniform-volume",
      dimension: 2,
      methodValues: UNIFORM_LAB_VALUES,
      uniformSeed: uniformLabSeed(scene),
    });
    if (experiment !== "area-only") return this.command({ type: "set-surface-experiment", profile: experiment });
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
      throw new Error("Uniform edit did not publish fields");
    return createUniformView(publication as PhysicsPublication);
  }
  destroy() {
    return this.client.destroy();
  }
}
