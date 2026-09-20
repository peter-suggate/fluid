import { parsePhysicsReceipt, type PhysicsPublication } from "./protocol";

export const PHYSICS_PUBLICATION_MAGIC = "FLUIDCPU";
export const PHYSICS_PUBLICATION_VERSION = 1;
export const PHYSICS_PUBLICATION_HEADER_BYTES = 32;
export const PHYSICS_PUBLICATION_DIRECTORY_BYTES = 16;

export const PhysicsPlane = Object.freeze({
  Density: 1, Gamma: 2, Capacity: 3, CellVelocity: 4, FaceVelocity: 5,
  Pressure: 6, PressureRhs: 7, PressureDiagonal: 8, PressureMember: 9,
  PressureRowMember: 10, ExtensionDepth: 11, InterfaceNormal: 12,
  InterfaceOffset: 13, LowFlux: 14, HighFlux: 15, LimitedFlux: 16,
  CapacityBefore: 17, CapacityAfter: 18, CapacityRate: 19, SourceRate: 20,
  InflowCoverage: 21, CharacteristicClearance: 22, Tracers: 30,
  RdfVertices: 31, RdfSegments: 32, GraphJson: 100, SceneJson: 101,
  DensityBefore: 40, VelocityXBeforePressure: 41, VelocityYBeforePressure: 42,
  BrickResolutionBefore: 43, BrickActivity: 44, MaterialFine: 45,
  CapacityFine: 46,
  Density3D: 60, SurfacePhi3D: 61, Velocity3D: 62,
  UniformLowX: 70, UniformLowY: 71, UniformReleased: 72, UniformTiles: 73,
} as const);
export type PhysicsPlaneId = typeof PhysicsPlane[keyof typeof PhysicsPlane];
export type PhysicsPlaneValue = Float32Array<ArrayBuffer> | Uint8Array<ArrayBuffer>;

const F32_PLANES = new Set<number>([1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16,
  17, 18, 19, 20, 21, 22, 30, 31, 32, 40, 41, 42, 44, 45, 46, 60, 61, 62, 70, 71]);
const U8_PLANES = new Set<number>([9, 10, 11, 43, 72, 73]);
const JSON_PLANES = new Set<number>([100, 101]);
const KNOWN_PLANES = new Set([...F32_PLANES, ...U8_PLANES, ...JSON_PLANES]);
const decoder = new TextDecoder("utf-8", { fatal: true });
const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
const REVISION_KEYS = ["schemaVersion", "dimension", "runEpoch", "commandSequence", "frame",
  "time", "injections", "topologyGeneration", "fieldRevision", "surfaceRevision", "memoryEpoch"] as const;

const u32 = (view: DataView, at: number) => view.getUint32(at, true);

export class DecodedPhysicsPublication {
  private released = false;
  constructor(
    readonly metadata: Readonly<Record<string, unknown>>,
    private readonly planes: ReadonlyMap<PhysicsPlaneId, PhysicsPlaneValue>,
    private readonly source: PhysicsPublication,
  ) {}

  plane(id: PhysicsPlaneId): PhysicsPlaneValue | undefined {
    if (this.released) throw new Error("Physics publication has been released");
    return this.planes.get(id);
  }

  jsonPlane(id: typeof PhysicsPlane.GraphJson | typeof PhysicsPlane.SceneJson): unknown {
    const bytes = this.plane(id);
    if (!bytes || !(bytes instanceof Uint8Array)) return undefined;
    return JSON.parse(decoder.decode(bytes));
  }

  release(): void {
    if (this.released) return;
    this.released = true;
    this.source.release();
  }
}

export function decodePhysicsPublication(source: PhysicsPublication): DecodedPhysicsPublication {
  if (!littleEndian) throw new Error("Zero-copy physics publications require a little-endian host");
  const bytes = source.bytes;
  if (bytes.byteOffset !== 0) throw new RangeError("Physics publication must start at its ArrayBuffer origin");
  if (bytes.byteLength < PHYSICS_PUBLICATION_HEADER_BYTES) throw new RangeError("Physics publication header is truncated");
  const magic = decoder.decode(bytes.subarray(0, 8));
  if (magic !== PHYSICS_PUBLICATION_MAGIC) throw new TypeError("Physics publication magic is invalid");
  const buffer = bytes.buffer as ArrayBuffer;
  const view = new DataView(buffer, 0, bytes.byteLength);
  const version = u32(view, 8), totalBytes = u32(view, 12), count = u32(view, 16);
  const metadataOffset = u32(view, 20), metadataLength = u32(view, 24), reserved = u32(view, 28);
  if (version !== PHYSICS_PUBLICATION_VERSION) throw new RangeError(`Unsupported physics publication version ${version}`);
  if (totalBytes !== bytes.byteLength) throw new RangeError("Physics publication length does not match its header");
  if (reserved !== 0) throw new RangeError("Physics publication reserved header word is nonzero");
  const directoryEnd = PHYSICS_PUBLICATION_HEADER_BYTES + count * PHYSICS_PUBLICATION_DIRECTORY_BYTES;
  if (!Number.isSafeInteger(directoryEnd) || directoryEnd > bytes.byteLength
    || metadataOffset !== directoryEnd || metadataLength > bytes.byteLength - metadataOffset) {
    throw new RangeError("Physics publication directory or metadata range is invalid");
  }
  const metadataEnd = metadataOffset + metadataLength;
  let metadata: unknown;
  try { metadata = JSON.parse(decoder.decode(bytes.subarray(metadataOffset, metadataEnd))); }
  catch (error) { throw new TypeError("Physics publication metadata is not valid UTF-8 JSON", { cause: error }); }
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
    throw new TypeError("Physics publication metadata must be a JSON object");
  }
  const metadataRecord = metadata as Record<string, unknown>;
  const metadataRevision = parsePhysicsReceipt(JSON.stringify(metadataRecord.revision));
  for (const key of REVISION_KEYS) {
    if (metadataRevision[key] !== source.revision[key]) {
      throw new RangeError(`Physics publication metadata revision disagrees on ${key}`);
    }
  }
  const planes = new Map<PhysicsPlaneId, PhysicsPlaneValue>();
  const occupied: Array<readonly [number, number]> = [[0, metadataEnd]];
  for (let entry = 0; entry < count; entry++) {
    const at = PHYSICS_PUBLICATION_HEADER_BYTES + entry * PHYSICS_PUBLICATION_DIRECTORY_BYTES;
    const id = u32(view, at), kind = u32(view, at + 4), offset = u32(view, at + 8);
    const elementCount = u32(view, at + 12);
    if (!KNOWN_PLANES.has(id)) throw new RangeError(`Unknown physics plane id ${id}`);
    if (planes.has(id as PhysicsPlaneId)) throw new RangeError(`Duplicate physics plane id ${id}`);
    const expectedKind = F32_PLANES.has(id) ? 1 : U8_PLANES.has(id) ? 2 : 3;
    if (kind !== expectedKind) throw new TypeError(`Physics plane ${id} has kind ${kind}; expected ${expectedKind}`);
    if (offset % 64 !== 0) throw new RangeError(`Physics plane ${id} is not 64-byte aligned`);
    const byteLength = elementCount * (kind === 1 ? 4 : 1);
    if (!Number.isSafeInteger(byteLength) || offset < metadataEnd || byteLength > bytes.byteLength - offset) {
      throw new RangeError(`Physics plane ${id} range is invalid`);
    }
    const end = offset + byteLength;
    if (occupied.some(([begin, occupiedEnd]) => offset < occupiedEnd && begin < end)) {
      throw new RangeError(`Physics plane ${id} overlaps another publication range`);
    }
    occupied.push([offset, end]);
    const value = kind === 1
      ? new Float32Array(buffer, offset, elementCount)
      : new Uint8Array(buffer, offset, elementCount);
    planes.set(id as PhysicsPlaneId, value);
  }
  return new DecodedPhysicsPublication(metadataRecord, planes, source);
}
