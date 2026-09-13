import { decodePhysicsPublication, PhysicsPlane } from "./publication";
import type { PhysicsPublication, PhysicsRevision } from "./protocol";
import type { GPURigidBodyPose } from "../core/webgpu-rigid-body";

export interface Fluid3DSceneMetadata {
  readonly dimensions: readonly [number, number, number];
  readonly originM: readonly [number, number, number];
  readonly cellSizeM: readonly [number, number, number];
  readonly dtS: number;
}

export interface Fluid3DPublicationView {
  readonly revision: PhysicsRevision;
  readonly scene: Fluid3DSceneMetadata;
  readonly density: Float32Array<ArrayBuffer>;
  readonly surfacePhi: Float32Array<ArrayBuffer>;
  readonly velocity?: Float32Array<ArrayBuffer>;
  readonly tracers?: Float32Array<ArrayBuffer>;
  readonly tracersEnabled: boolean;
  readonly stats: Readonly<Record<string, unknown>>;
  readonly rigidBodies: readonly GPURigidBodyPose[];
  release(): void;
}

const finiteTriplet = (value: unknown, positive = false): value is [number, number, number] =>
  Array.isArray(value) && value.length === 3
  && value.every(item => typeof item === "number" && Number.isFinite(item)
    && (!positive || item > 0));

const positiveIntegerTriplet = (value: unknown): value is [number, number, number] =>
  finiteTriplet(value, true) && value.every(Number.isSafeInteger);
const finiteRecord = (value: unknown, keys: readonly string[]): value is Record<string, number> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)
  && keys.every(key => typeof (value as Record<string, unknown>)[key] === "number"
    && Number.isFinite((value as Record<string, number>)[key]));

/** Strict read-only projection of a Rust 3D presentation publication. */
export function decodeFluid3DPublication(source: PhysicsPublication): Fluid3DPublicationView {
  const decoded = decodePhysicsPublication(source);
  try {
    if (source.revision.dimension !== 3) throw new TypeError("Fluid 3D publication has a non-3D revision");
    const rawScene = decoded.metadata.scene;
    if (!rawScene || typeof rawScene !== "object" || Array.isArray(rawScene)) {
      throw new TypeError("Fluid 3D publication has no scene metadata");
    }
    const sceneRecord = rawScene as Record<string, unknown>;
    if (!positiveIntegerTriplet(sceneRecord.dimensions)
      || !finiteTriplet(sceneRecord.originM)
      || !finiteTriplet(sceneRecord.cellSizeM, true)
      || typeof sceneRecord.dtS !== "number" || !(sceneRecord.dtS > 0)
      || !Number.isFinite(sceneRecord.dtS)) {
      throw new TypeError("Fluid 3D publication scene metadata is invalid");
    }
    const count = sceneRecord.dimensions[0] * sceneRecord.dimensions[1] * sceneRecord.dimensions[2];
    if (!Number.isSafeInteger(count)) throw new RangeError("Fluid 3D publication dimensions overflow");
    const density = decoded.plane(PhysicsPlane.Density3D);
    const surfacePhi = decoded.plane(PhysicsPlane.SurfacePhi3D);
    const velocity = decoded.plane(PhysicsPlane.Velocity3D);
    const tracers = decoded.plane(PhysicsPlane.Tracers);
    if (!(density instanceof Float32Array) || density.length !== count) {
      throw new RangeError(`Fluid 3D density plane must contain ${count} float32 values`);
    }
    if (!(surfacePhi instanceof Float32Array) || surfacePhi.length !== count) {
      throw new RangeError(`Fluid 3D surface plane must contain ${count} float32 values`);
    }
    if (velocity !== undefined
      && (!(velocity instanceof Float32Array) || velocity.length !== 4 * count)) {
      throw new RangeError(`Fluid 3D velocity plane must contain ${4 * count} float32 values`);
    }
    const tracersEnabled = decoded.metadata.tracersEnabled === true;
    if (tracersEnabled
      && (!(tracers instanceof Float32Array) || tracers.length % 4 !== 0)) {
      throw new RangeError("Enabled Fluid 3D tracers require float32 vec4 slots");
    }
    if (!tracersEnabled && tracers !== undefined) {
      throw new RangeError("Disabled Fluid 3D tracers must not publish a tracer plane");
    }
    const stats = decoded.metadata.stats;
    const rigidBodies = Array.isArray(decoded.metadata.rigidBodies)
      ? decoded.metadata.rigidBodies.map((body, index) => {
        if (!body || typeof body !== "object" || Array.isArray(body)) {
          throw new TypeError(`Fluid 3D rigid body ${index} is invalid`);
        }
        const record = body as Record<string, unknown>;
        const position = record.position_m;
        const orientation = record.orientation;
        if (!finiteRecord(position, ["x", "y", "z"])
          || !finiteRecord(orientation, ["w", "x", "y", "z"])) {
          throw new TypeError(`Fluid 3D rigid body ${index} has no pose`);
        }
        return {
          position_m: { x: position.x, y: position.y, z: position.z },
          orientation: { w: orientation.w, x: orientation.x, y: orientation.y, z: orientation.z },
        } satisfies GPURigidBodyPose;
      }) : [];
    return {
      revision: source.revision,
      scene: {
        dimensions: sceneRecord.dimensions,
        originM: sceneRecord.originM,
        cellSizeM: sceneRecord.cellSizeM,
        dtS: sceneRecord.dtS,
      },
      density,
      surfacePhi,
      velocity: velocity as Float32Array<ArrayBuffer> | undefined,
      tracers: tracers as Float32Array<ArrayBuffer> | undefined,
      tracersEnabled,
      rigidBodies,
      stats: stats && typeof stats === "object" && !Array.isArray(stats)
        ? stats as Readonly<Record<string, unknown>> : {},
      release: () => decoded.release(),
    };
  } catch (error) {
    decoded.release();
    throw error;
  }
}
