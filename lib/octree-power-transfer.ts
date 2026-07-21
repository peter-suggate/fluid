/** Conservative CPU oracle for dynamic power-face topology transfer (WP9). */

import type { PowerVec3 } from "./octree-power-geometry";

/**
 * A transfer key is exactly four unsigned 32-bit radix words (128 bits).
 * Tuple comparison is unsigned lexicographic in word order. A least-significant
 * digit radix sort therefore processes words 3, 2, 1, then 0.
 */
export type OctreePowerTransferKey = readonly [number, number, number, number];

/** A site key is its packed 10/10/10-bit origin plus its dyadic size exponent. */
export interface OctreePowerSiteTransferKey {
  readonly packedOrigin: number;
  readonly sizeExponent: number;
}

export interface OctreePowerFaceTransferIdentity {
  readonly key: OctreePowerTransferKey;
  /** -1 means the supplied site order was reversed to form the canonical key. */
  readonly orientation: 1 | -1;
}

export interface OctreePowerTransferFace {
  /** Exact ordered-site/local-face identity. Normal orientation follows this key. */
  readonly key: OctreePowerTransferKey;
  /**
   * Optional stable refinement lineage. One parent and all of its children use
   * the same aggregate key. It is temporary rebuild metadata, not face state.
   */
  readonly aggregateKey?: OctreePowerTransferKey;
  readonly area: number;
  readonly centroid: PowerVec3;
  /** Finite, non-zero normal. Projection uses its normalized direction. */
  readonly normal: PowerVec3;
  readonly normalVelocity: number;
  readonly boundary?: boolean;
}

export type OctreePowerTransferMode = "exact" | "prolongation" | "restriction" | "trace-back";

export interface OctreePowerFaceTransferRecord {
  readonly newFace: number;
  readonly mode: OctreePowerTransferMode;
  readonly oldFaces: readonly number[];
  /** Area-integrated flux inherited from oldFaces (zero for trace-back). */
  readonly inheritedFlux: number;
  /** Weighted detail flux; zero to roundoff for prolongation. */
  readonly detailFlux: number;
}

export interface OctreePowerTransferDiagnostics {
  readonly oldBoundaryFlux: number;
  readonly newBoundaryFlux: number;
  readonly oldInternalFlux: number;
  readonly newInternalFlux: number;
  readonly exactFaceCount: number;
  readonly prolongedFaceCount: number;
  readonly restrictedFaceCount: number;
  readonly traceBackFaceCount: number;
}

export interface OctreePowerTopologyTransfer {
  readonly velocities: readonly number[];
  readonly records: readonly OctreePowerFaceTransferRecord[];
  readonly diagnostics: OctreePowerTransferDiagnostics;
}

export interface OctreePowerTopologyTransferOptions {
  /** Backtrace duration. Zero still samples the old field at the new centroid. */
  readonly dt?: number;
  /**
   * Sparse old-generation full-velocity sampler. The oracle first samples the
   * centroid for the trace direction, then samples centroid - dt * velocity.
   */
  readonly sampleVelocity?: (point: PowerVec3) => PowerVec3;
  /** Finite fallback used when no sampler is supplied. Defaults to zero. */
  readonly fallbackVelocity?: PowerVec3;
}

const U32_MAX = 0xffff_ffff;
const POSITIVE_EXPONENT_SENTINEL = 0x3f;

function u32(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > U32_MAX) {
    throw new RangeError(`${label} must be an unsigned 32-bit integer`);
  }
  return value >>> 0;
}

function validateKey(key: OctreePowerTransferKey, label: string): void {
  if (key.length !== 4) throw new RangeError(`${label} must contain exactly four u32 words`);
  key.forEach((word, index) => u32(word, `${label}[${index}]`));
}

function finiteVector(value: PowerVec3, label: string): PowerVec3 {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
  return value;
}

function keyString(key: OctreePowerTransferKey): string {
  validateKey(key, "Power transfer key");
  // Fixed-width hexadecimal is a collision-free textual representation of the
  // tuple; unlike delimiter-free decimal concatenation it preserves u32 words.
  return key.map((word) => (word >>> 0).toString(16).padStart(8, "0")).join(":");
}

export function compareOctreePowerTransferKeys(a: OctreePowerTransferKey, b: OctreePowerTransferKey): number {
  validateKey(a, "Left power transfer key");
  validateKey(b, "Right power transfer key");
  for (let word = 0; word < 4; word += 1) {
    const difference = (a[word] >>> 0) - (b[word] >>> 0);
    if (difference !== 0) return difference < 0 ? -1 : 1;
  }
  return 0;
}

/** Packs keys without coercing high-bit words through signed Int32 semantics. */
export function packOctreePowerTransferKeys(keys: readonly OctreePowerTransferKey[]): Uint32Array {
  const packed = new Uint32Array(keys.length * 4);
  keys.forEach((key, index) => {
    validateKey(key, `Power transfer key ${index}`);
    packed.set(key, index * 4);
  });
  return packed;
}

/** Creates the stable origin/exponent site key used by the 128-bit face key. */
export function createOctreePowerSiteTransferKey(origin: PowerVec3, size: number): OctreePowerSiteTransferKey {
  if (origin.length !== 3 || origin.some((component) => !Number.isSafeInteger(component) || component < 0 || component > 1023)) {
    throw new RangeError("Power transfer site origin must contain three integers in [0, 1023]");
  }
  const sizeExponent = Math.log2(size);
  if (!Number.isSafeInteger(size) || size < 1 || !Number.isInteger(sizeExponent)) {
    throw new RangeError("Power transfer site size must be a positive dyadic integer");
  }
  if (sizeExponent >= POSITIVE_EXPONENT_SENTINEL) throw new RangeError("Power transfer site size exponent exceeds the key ABI");
  return {
    packedOrigin: (origin[0] | (origin[1] << 10) | (origin[2] << 20)) >>> 0,
    sizeExponent,
  };
}

function compareSites(a: OctreePowerSiteTransferKey, b: OctreePowerSiteTransferKey): number {
  const originDifference = (a.packedOrigin >>> 0) - (b.packedOrigin >>> 0);
  return originDifference || a.sizeExponent - b.sizeExponent;
}

/**
 * Constructs the documented face-key ABI:
 *
 *   word 0: canonical negative packed origin
 *   word 1: canonical positive packed origin, or UINT32_MAX for a boundary
 *   word 2: negative exponent[0:5], positive exponent[6:11], boundary[12],
 *           reserved zero[13:15], catalog-local signature[16:31]
 *   word 3: boundary identity (zero for an interior face)
 *
 * localFaceSignature must be invariant to swapping the two incident sites.
 */
export function createOctreePowerFaceTransferKey(
  first: OctreePowerSiteTransferKey,
  second: OctreePowerSiteTransferKey | undefined,
  localFaceSignature: number,
  boundaryIdentity = 0,
): OctreePowerFaceTransferIdentity {
  const validateSite = (site: OctreePowerSiteTransferKey, label: string) => {
    u32(site.packedOrigin, `${label} packed origin`);
    if (!Number.isSafeInteger(site.sizeExponent) || site.sizeExponent < 0 || site.sizeExponent >= POSITIVE_EXPONENT_SENTINEL) {
      throw new RangeError(`${label} size exponent must fit six bits without using the boundary sentinel`);
    }
  };
  validateSite(first, "First power site");
  if (second) validateSite(second, "Second power site");
  if (!Number.isSafeInteger(localFaceSignature) || localFaceSignature < 0 || localFaceSignature > 0xffff) {
    throw new RangeError("Power local-face signature must be an unsigned 16-bit integer");
  }
  const boundary = second === undefined;
  if (!boundary && boundaryIdentity !== 0) throw new RangeError("Interior power-face keys cannot carry a boundary identity");
  const identity = u32(boundaryIdentity, "Power boundary identity");
  let negative = first, positive = second, orientation: 1 | -1 = 1;
  if (positive && compareSites(negative, positive) > 0) {
    [negative, positive] = [positive, negative];
    orientation = -1;
  }
  const positiveExponent = positive?.sizeExponent ?? POSITIVE_EXPONENT_SENTINEL;
  const metadata = (negative.sizeExponent | (positiveExponent << 6) | (Number(boundary) << 12)
    | (localFaceSignature << 16)) >>> 0;
  return {
    key: [negative.packedOrigin >>> 0, positive?.packedOrigin ?? U32_MAX, metadata, boundary ? identity : 0],
    orientation,
  };
}

function normalDirection(face: OctreePowerTransferFace, label: string): PowerVec3 {
  finiteVector(face.centroid, `${label} centroid`);
  finiteVector(face.normal, `${label} normal`);
  if (!Number.isFinite(face.area) || face.area <= 0) throw new RangeError(`${label} area must be finite and positive`);
  if (!Number.isFinite(face.normalVelocity)) throw new RangeError(`${label} normal velocity must be finite`);
  validateKey(face.key, `${label} key`);
  if (face.aggregateKey) validateKey(face.aggregateKey, `${label} aggregate key`);
  const keyIsBoundary = ((face.key[2] >>> 12) & 1) !== 0;
  if (face.boundary !== undefined && face.boundary !== keyIsBoundary) {
    throw new RangeError(`${label} boundary classification disagrees with its stable key`);
  }
  const magnitude = Math.hypot(...face.normal);
  if (!(magnitude > 0) || !Number.isFinite(magnitude)) throw new RangeError(`${label} normal must be non-zero`);
  return [face.normal[0] / magnitude, face.normal[1] / magnitude, face.normal[2] / magnitude];
}

function weightedFlux(faces: readonly OctreePowerTransferFace[], indices: readonly number[]): number {
  return indices.reduce((sum, index) => sum + faces[index].area * faces[index].normalVelocity, 0);
}

function classifyFlux(faces: readonly OctreePowerTransferFace[], velocities?: readonly number[]): readonly [number, number] {
  let boundary = 0, internal = 0;
  faces.forEach((face, index) => {
    const flux = face.area * (velocities?.[index] ?? face.normalVelocity);
    if (((face.key[2] >>> 12) & 1) !== 0) boundary += flux; else internal += flux;
  });
  return [boundary, internal];
}

/**
 * Transfers one compact power-face generation to the next without consulting
 * any dense field. Exact identities win first. Refinement/coarsening is then
 * resolved only through explicit aggregate keys; all other connectivity uses
 * the supplied trace-back sampler (or a finite zero fallback).
 */
export function buildOctreePowerTopologyTransfer(
  previous: readonly OctreePowerTransferFace[],
  next: readonly OctreePowerTransferFace[],
  options: OctreePowerTopologyTransferOptions = {},
): OctreePowerTopologyTransfer {
  const dt = options.dt ?? 0;
  if (!Number.isFinite(dt) || dt < 0) throw new RangeError("Power topology transfer dt must be finite and non-negative");
  const fallback = finiteVector(options.fallbackVelocity ?? [0, 0, 0], "Power topology transfer fallback velocity");
  previous.forEach((face, index) => normalDirection(face, `Previous power face ${index}`));
  const nextNormals = next.map((face, index) => normalDirection(face, `Next power face ${index}`));

  const oldByExact = new Map<string, number>();
  previous.forEach((face, index) => {
    const key = keyString(face.key);
    if (oldByExact.has(key)) throw new RangeError(`Duplicate previous power-face key ${key}`);
    oldByExact.set(key, index);
  });
  const nextKeys = new Set<string>();
  next.forEach((face) => {
    const key = keyString(face.key);
    if (nextKeys.has(key)) throw new RangeError(`Duplicate next power-face key ${key}`);
    nextKeys.add(key);
  });

  const velocities = new Array<number>(next.length);
  const records = new Array<OctreePowerFaceTransferRecord | undefined>(next.length);
  const usedOld = new Set<number>();
  for (let newFace = 0; newFace < next.length; newFace += 1) {
    const oldFace = oldByExact.get(keyString(next[newFace].key));
    if (oldFace === undefined) continue;
    // Direct assignment is intentionally un-arithmetic: unchanged topology is
    // bitwise stable for both Float32-origin and Float64-origin JS numbers.
    velocities[newFace] = previous[oldFace].normalVelocity;
    records[newFace] = { newFace, mode: "exact", oldFaces: [oldFace], inheritedFlux: previous[oldFace].area * previous[oldFace].normalVelocity, detailFlux: 0 };
    usedOld.add(oldFace);
  }

  const oldGroups = new Map<string, number[]>();
  previous.forEach((face, index) => {
    if (usedOld.has(index) || !face.aggregateKey) return;
    const key = keyString(face.aggregateKey);
    const group = oldGroups.get(key) ?? [];
    group.push(index); oldGroups.set(key, group);
  });
  const nextGroups = new Map<string, number[]>();
  next.forEach((face, index) => {
    if (records[index] || !face.aggregateKey) return;
    const key = keyString(face.aggregateKey);
    const group = nextGroups.get(key) ?? [];
    group.push(index); nextGroups.set(key, group);
  });

  const sampleProjected = (newFace: number): number => {
    const face = next[newFace], normal = nextNormals[newFace];
    let fullVelocity = fallback;
    if (options.sampleVelocity) {
      const advecting = finiteVector(options.sampleVelocity(face.centroid), `Trace direction for next power face ${newFace}`);
      const departure: PowerVec3 = [
        face.centroid[0] - dt * advecting[0],
        face.centroid[1] - dt * advecting[1],
        face.centroid[2] - dt * advecting[2],
      ];
      fullVelocity = finiteVector(options.sampleVelocity(departure), `Trace-back velocity for next power face ${newFace}`);
    }
    const projected = fullVelocity[0] * normal[0] + fullVelocity[1] * normal[1] + fullVelocity[2] * normal[2];
    if (!Number.isFinite(projected)) throw new Error(`Trace-back projection for next power face ${newFace} is non-finite`);
    return projected;
  };

  for (const [key, newFaces] of nextGroups) {
    const oldFaces = oldGroups.get(key) ?? [];
    if (oldFaces.length === 1 && newFaces.length > 1) {
      const inheritedFlux = weightedFlux(previous, oldFaces);
      const totalArea = newFaces.reduce((sum, index) => sum + next[index].area, 0);
      const raw = newFaces.map(sampleProjected);
      const rawMean = newFaces.reduce((sum, index, local) => sum + next[index].area * raw[local], 0) / totalArea;
      const base = inheritedFlux / totalArea;
      let emittedFlux = 0;
      newFaces.forEach((newFace, local) => {
        const velocity = base + raw[local] - rawMean;
        velocities[newFace] = velocity;
        emittedFlux += next[newFace].area * velocity;
      });
      // Report the actual floating residual. Mathematically this is zero; the
      // value is useful for Float32 implementation parity tolerances.
      const detailFlux = emittedFlux - inheritedFlux;
      newFaces.forEach((newFace) => {
        records[newFace] = { newFace, mode: "prolongation", oldFaces: [...oldFaces], inheritedFlux, detailFlux };
      });
      continue;
    }
    if (oldFaces.length > 1 && newFaces.length === 1) {
      const newFace = newFaces[0];
      const inheritedFlux = weightedFlux(previous, oldFaces);
      velocities[newFace] = inheritedFlux / next[newFace].area;
      records[newFace] = { newFace, mode: "restriction", oldFaces: [...oldFaces], inheritedFlux, detailFlux: 0 };
    }
  }

  for (let newFace = 0; newFace < next.length; newFace += 1) {
    if (records[newFace]) continue;
    velocities[newFace] = sampleProjected(newFace);
    records[newFace] = { newFace, mode: "trace-back", oldFaces: [], inheritedFlux: 0, detailFlux: 0 };
  }
  if (velocities.some((value) => !Number.isFinite(value))) throw new Error("Power topology transfer produced a non-finite velocity");

  const [oldBoundaryFlux, oldInternalFlux] = classifyFlux(previous);
  const [newBoundaryFlux, newInternalFlux] = classifyFlux(next, velocities);
  const completeRecords = records as OctreePowerFaceTransferRecord[];
  return {
    velocities,
    records: completeRecords,
    diagnostics: {
      oldBoundaryFlux, newBoundaryFlux, oldInternalFlux, newInternalFlux,
      exactFaceCount: completeRecords.filter((record) => record.mode === "exact").length,
      prolongedFaceCount: completeRecords.filter((record) => record.mode === "prolongation").length,
      restrictedFaceCount: completeRecords.filter((record) => record.mode === "restriction").length,
      traceBackFaceCount: completeRecords.filter((record) => record.mode === "trace-back").length,
    },
  };
}
