/** Sparse, page-backed mip data derived from a complete unified-octree generation. */

export const SVO_NODE_MIP_LAYOUT = Object.freeze({
  interiorSize: 8,
  apron: 1,
  physicalSize: 10,
  channelCount: 4,
  bytesPerTexel: 4,
  bytesPerPage: 4_000,
  keyWords: 4,
  keyBytes: 16,
  directoryBytesPerPage: 32,
} as const);

export const SVO_NODE_MIP_LANES = Object.freeze({
  solidMean: 0,
  solidMaximum: 1,
  fluidMean: 2,
  fluidMaximum: 3,
} as const);

export type SvoNodeMipCoordinate = readonly [number, number, number];

export interface SvoNodeMipPageKey {
  /** Complete source-octree publication generation. */
  generation: number;
  /** Zero is the finest virtual mip page level. */
  level: number;
  /** Virtual page coordinate at `level`. */
  coordinate: SvoNodeMipCoordinate;
}

export interface SvoNodeMipAtlasShape {
  pages: SvoNodeMipCoordinate;
  texels: SvoNodeMipCoordinate;
  capacity: number;
}

export interface SvoNodeMipPagePlan {
  key: SvoNodeMipPageKey;
  keyString: string;
  slot: number;
  atlasPage: SvoNodeMipCoordinate;
  atlasTexelOrigin: SvoNodeMipCoordinate;
}

export interface SvoNodeMipPyramidPlan {
  generation: number;
  pages: readonly SvoNodeMipPagePlan[];
  atlas: SvoNodeMipAtlasShape;
  requestedPageCount: number;
  residentPageCount: number;
  overflowPageCount: number;
  pagePayloadBytes: number;
  /** Physical atlas allocation, including unused slots introduced by its 3D shape. */
  atlasBytes: number;
  directoryBytes: number;
  allocatedBytes: number;
  complete: boolean;
}

export interface SvoNodeMipPlanOptions {
  generation: number;
  /** Finest-level virtual pages. Ancestor pages are inserted automatically. */
  occupiedPages: readonly SvoNodeMipCoordinate[];
  levelCount: number;
  capacity?: number;
  atlasPages?: SvoNodeMipCoordinate;
}

export type SvoNodeMipRgba8 = readonly [number, number, number, number];

export interface SvoNodeMipPublication {
  completeGeneration: number;
  plan: SvoNodeMipPyramidPlan;
}

export interface SvoNodeMipPublicationCandidate {
  generation: number;
  plan: SvoNodeMipPyramidPlan;
  directoryComplete: boolean;
  payloadComplete: boolean;
  apronsComplete: boolean;
}

export type SvoNodeMipPublicationDecision =
  | { published: true; reason: "published"; visible: SvoNodeMipPublication }
  | { published: false; reason: "generation-order" | "incomplete-plan" | "incomplete-directory" | "incomplete-payload" | "incomplete-aprons"; visible?: SvoNodeMipPublication };

const UINT32_MAX = 0xffff_ffff;
const MORTON_AXIS_BITS = 21;
const MORTON_AXIS_MAX = (1 << MORTON_AXIS_BITS) - 1;

function uint32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > UINT32_MAX) throw new RangeError(`${label} must fit uint32`);
  return value >>> 0;
}

function coordinate(value: SvoNodeMipCoordinate, label: string): SvoNodeMipCoordinate {
  if (value.length !== 3 || value.some((component) => !Number.isSafeInteger(component) || component < 0)) {
    throw new RangeError(`${label} must contain three non-negative safe integers`);
  }
  return value;
}

/** 63-bit Morton code; 21 coordinate bits per axis avoids lossy JS bitwise arithmetic. */
export function encodeSvoNodeMipMorton(value: SvoNodeMipCoordinate): bigint {
  coordinate(value, "SVO node-mip coordinate");
  if (value.some((component) => component > MORTON_AXIS_MAX)) {
    throw new RangeError(`SVO node-mip coordinates must fit ${MORTON_AXIS_BITS} bits per axis`);
  }
  let result = 0n;
  for (let bit = 0; bit < MORTON_AXIS_BITS; bit += 1) {
    const shift = BigInt(bit * 3);
    result |= BigInt((value[0] >>> bit) & 1) << shift;
    result |= BigInt((value[1] >>> bit) & 1) << (shift + 1n);
    result |= BigInt((value[2] >>> bit) & 1) << (shift + 2n);
  }
  return result;
}

export function decodeSvoNodeMipMorton(morton: bigint): SvoNodeMipCoordinate {
  if (morton < 0n || morton >= (1n << 63n)) throw new RangeError("SVO node-mip Morton code must fit 63 bits");
  const result = [0, 0, 0];
  for (let bit = 0; bit < MORTON_AXIS_BITS; bit += 1) {
    const shift = BigInt(bit * 3);
    result[0] += Number((morton >> shift) & 1n) * 2 ** bit;
    result[1] += Number((morton >> (shift + 1n)) & 1n) * 2 ** bit;
    result[2] += Number((morton >> (shift + 2n)) & 1n) * 2 ** bit;
  }
  return result as unknown as SvoNodeMipCoordinate;
}

/** Stable CPU/GPU ABI: generation, level, Morton low word, Morton high word. */
export function packSvoNodeMipPageKey(key: SvoNodeMipPageKey): Uint32Array {
  const generation = uint32(key.generation, "SVO node-mip generation");
  const level = uint32(key.level, "SVO node-mip level");
  const morton = encodeSvoNodeMipMorton(key.coordinate);
  return new Uint32Array([generation, level, Number(morton & 0xffff_ffffn), Number(morton >> 32n)]);
}

export function unpackSvoNodeMipPageKey(words: ArrayLike<number>): SvoNodeMipPageKey {
  if (words.length < SVO_NODE_MIP_LAYOUT.keyWords) throw new RangeError("SVO node-mip key requires four words");
  const generation = uint32(words[0], "Packed SVO node-mip generation");
  const level = uint32(words[1], "Packed SVO node-mip level");
  const morton = BigInt(uint32(words[2], "Packed SVO node-mip Morton low word"))
    | BigInt(uint32(words[3], "Packed SVO node-mip Morton high word")) << 32n;
  return { generation, level, coordinate: decodeSvoNodeMipMorton(morton) };
}

export function svoNodeMipPageKey(key: SvoNodeMipPageKey): string {
  const words = packSvoNodeMipPageKey(key);
  return `${words[0]}:${words[1]}:${words[3].toString(16).padStart(8, "0")}${words[2].toString(16).padStart(8, "0")}`;
}

function automaticAtlasShape(capacity: number): SvoNodeMipCoordinate {
  if (capacity === 0) return [0, 0, 0];
  const x = Math.ceil(Math.cbrt(capacity));
  const y = Math.ceil(Math.sqrt(capacity / x));
  const z = Math.ceil(capacity / (x * y));
  return [x, y, z];
}

export function svoNodeMipAtlasPageCoordinate(slot: number, atlasPages: SvoNodeMipCoordinate): SvoNodeMipCoordinate {
  coordinate(atlasPages, "SVO node-mip atlas page dimensions");
  const capacity = atlasPages[0] * atlasPages[1] * atlasPages[2];
  if (!Number.isSafeInteger(slot) || slot < 0 || slot >= capacity) throw new RangeError("SVO node-mip atlas slot is outside the atlas");
  return [slot % atlasPages[0], Math.floor(slot / atlasPages[0]) % atlasPages[1], Math.floor(slot / (atlasPages[0] * atlasPages[1]))];
}

export function planSvoNodeMipPyramid(options: SvoNodeMipPlanOptions): SvoNodeMipPyramidPlan {
  const generation = uint32(options.generation, "SVO node-mip generation");
  if (!Number.isSafeInteger(options.levelCount) || options.levelCount < 1 || options.levelCount > 32) {
    throw new RangeError("SVO node-mip level count must be an integer in [1, 32]");
  }
  const all = new Map<string, SvoNodeMipPageKey>();
  for (const input of options.occupiedPages) {
    let current = [...coordinate(input, "Occupied SVO node-mip page")] as [number, number, number];
    for (let level = 0; level < options.levelCount; level += 1) {
      const key = { generation, level, coordinate: current as SvoNodeMipCoordinate };
      all.set(svoNodeMipPageKey(key), key);
      current = current.map((component) => Math.floor(component / 2)) as [number, number, number];
    }
  }
  // Level-major Morton order permits a binary-search directory lookup without
  // consuming another storage binding in the dry renderer.
  const ordered = [...all.entries()].sort((a, b) => a[1].level - b[1].level
    || (encodeSvoNodeMipMorton(a[1].coordinate) < encodeSvoNodeMipMorton(b[1].coordinate) ? -1 : 1));
  const requestedPageCount = ordered.length;
  const explicitShape = options.atlasPages ? coordinate(options.atlasPages, "SVO node-mip atlas page dimensions") : undefined;
  const explicitCapacity = explicitShape ? explicitShape[0] * explicitShape[1] * explicitShape[2] : Number.MAX_SAFE_INTEGER;
  const requestedCapacity = options.capacity ?? requestedPageCount;
  if (!Number.isSafeInteger(requestedCapacity) || requestedCapacity < 0) throw new RangeError("SVO node-mip capacity must be a non-negative safe integer");
  const capacity = Math.min(requestedCapacity, explicitCapacity);
  const residentPageCount = Math.min(requestedPageCount, capacity);
  const atlasPages = explicitShape ?? automaticAtlasShape(residentPageCount);
  const physical = SVO_NODE_MIP_LAYOUT.physicalSize;
  const pages = ordered.slice(0, residentPageCount).map(([keyString, key], slot): SvoNodeMipPagePlan => {
    const atlasPage = svoNodeMipAtlasPageCoordinate(slot, atlasPages);
    return { key, keyString, slot, atlasPage, atlasTexelOrigin: atlasPage.map((component) => component * physical) as unknown as SvoNodeMipCoordinate };
  });
  const pagePayloadBytes = residentPageCount * SVO_NODE_MIP_LAYOUT.bytesPerPage;
  const atlasBytes = atlasPages[0] * atlasPages[1] * atlasPages[2] * SVO_NODE_MIP_LAYOUT.bytesPerPage;
  const directoryBytes = residentPageCount * SVO_NODE_MIP_LAYOUT.directoryBytesPerPage;
  return {
    generation, pages,
    atlas: {
      pages: atlasPages,
      texels: atlasPages.map((component) => component * physical) as unknown as SvoNodeMipCoordinate,
      capacity: atlasPages[0] * atlasPages[1] * atlasPages[2],
    },
    requestedPageCount, residentPageCount,
    overflowPageCount: requestedPageCount - residentPageCount,
    pagePayloadBytes, atlasBytes, directoryBytes,
    allocatedBytes: atlasBytes + directoryBytes,
    complete: requestedPageCount === residentPageCount,
  };
}

function byte(value: number): number {
  if (!Number.isFinite(value)) throw new RangeError("SVO node-mip lane must be finite");
  return Math.max(0, Math.min(255, Math.round(value)));
}

/** Reduces eight RGBA8 children. Mean lanes are averaged; maximum lanes are conservative maxima. */
export function reduceSvoNodeMipChildren(children: readonly SvoNodeMipRgba8[]): SvoNodeMipRgba8 {
  if (children.length !== 8) throw new RangeError("SVO node-mip reduction requires eight children");
  let solidMean = 0, fluidMean = 0, solidMaximum = 0, fluidMaximum = 0;
  for (const child of children) {
    if (child.length !== 4) throw new RangeError("SVO node-mip child requires four lanes");
    const lanes = child.map(byte);
    solidMean += lanes[0]; fluidMean += lanes[2];
    solidMaximum = Math.max(solidMaximum, lanes[1]); fluidMaximum = Math.max(fluidMaximum, lanes[3]);
  }
  return [Math.round(solidMean / 8), solidMaximum, Math.round(fluidMean / 8), fluidMaximum];
}

export function svoNodeMipTexelOffset(x: number, y: number, z: number): number {
  const size = SVO_NODE_MIP_LAYOUT.physicalSize;
  if (![x, y, z].every((value) => Number.isInteger(value) && value >= 0 && value < size)) throw new RangeError("SVO node-mip physical texel is outside its page");
  return ((z * size + y) * size + x) * SVO_NODE_MIP_LAYOUT.channelCount;
}

export interface SvoNodeMipVirtualTexelAddress {
  page: SvoNodeMipCoordinate;
  texel: SvoNodeMipCoordinate;
}

/** Resolves an apron coordinate (-1 or 8) to the adjacent same-level virtual page. */
export function resolveSvoNodeMipVirtualTexel(
  page: SvoNodeMipCoordinate,
  texel: readonly [number, number, number],
): SvoNodeMipVirtualTexelAddress | undefined {
  coordinate(page, "SVO node-mip virtual page");
  if (texel.some((component) => !Number.isInteger(component))) throw new RangeError("SVO node-mip virtual texel must contain integers");
  const n = SVO_NODE_MIP_LAYOUT.interiorSize;
  const resultPage = [0, 0, 0];
  const resultTexel = [0, 0, 0];
  for (let axis = 0; axis < 3; axis += 1) {
    const pageOffset = Math.floor(texel[axis] / n);
    resultPage[axis] = page[axis] + pageOffset;
    if (resultPage[axis] < 0) return undefined;
    resultTexel[axis] = ((texel[axis] % n) + n) % n;
  }
  return { page: resultPage as unknown as SvoNodeMipCoordinate, texel: resultTexel as unknown as SvoNodeMipCoordinate };
}

export type SvoNodeMipApronSampler = (address: SvoNodeMipVirtualTexelAddress) => SvoNodeMipRgba8 | undefined;

/** Fills aprons from same-level neighbours and clamps only where a neighbour is not resident. */
export function createSvoNodeMipPageWithApron(
  pageCoordinate: SvoNodeMipCoordinate,
  interior: Uint8Array,
  sampleNeighbour: SvoNodeMipApronSampler,
): Uint8Array {
  const result = createSvoNodeMipPage(interior);
  const n = SVO_NODE_MIP_LAYOUT.interiorSize;
  for (let z = 0; z < n + 2; z += 1) for (let y = 0; y < n + 2; y += 1) for (let x = 0; x < n + 2; x += 1) {
    if (x > 0 && x < n + 1 && y > 0 && y < n + 1 && z > 0 && z < n + 1) continue;
    const address = resolveSvoNodeMipVirtualTexel(pageCoordinate, [x - 1, y - 1, z - 1]);
    const sampled = address && sampleNeighbour(address);
    if (sampled) result.set(sampled.map(byte), svoNodeMipTexelOffset(x, y, z));
  }
  return result;
}

/** Creates a physical 10^3 page and fills its one-texel apron by clamping its 8^3 interior. */
export function createSvoNodeMipPage(interior: Uint8Array): Uint8Array {
  const n = SVO_NODE_MIP_LAYOUT.interiorSize;
  const channels = SVO_NODE_MIP_LAYOUT.channelCount;
  if (interior.byteLength !== n * n * n * channels) throw new RangeError("SVO node-mip interior must contain 8^3 RGBA8 texels");
  const result = new Uint8Array(SVO_NODE_MIP_LAYOUT.bytesPerPage);
  for (let z = 0; z < n + 2; z += 1) for (let y = 0; y < n + 2; y += 1) for (let x = 0; x < n + 2; x += 1) {
    const sourceX = Math.max(0, Math.min(n - 1, x - 1));
    const sourceY = Math.max(0, Math.min(n - 1, y - 1));
    const sourceZ = Math.max(0, Math.min(n - 1, z - 1));
    const source = ((sourceZ * n + sourceY) * n + sourceX) * channels;
    result.set(interior.subarray(source, source + channels), svoNodeMipTexelOffset(x, y, z));
  }
  return result;
}

/** Publication is atomic: incomplete candidate state never replaces the last complete generation. */
export function publishSvoNodeMipGeneration(
  visible: SvoNodeMipPublication | undefined,
  candidate: SvoNodeMipPublicationCandidate,
): SvoNodeMipPublicationDecision {
  uint32(candidate.generation, "Candidate SVO node-mip generation");
  if (candidate.plan.generation !== candidate.generation || (visible && candidate.generation <= visible.completeGeneration)) {
    return { published: false, reason: "generation-order", visible };
  }
  if (!candidate.plan.complete) return { published: false, reason: "incomplete-plan", visible };
  if (!candidate.directoryComplete) return { published: false, reason: "incomplete-directory", visible };
  if (!candidate.payloadComplete) return { published: false, reason: "incomplete-payload", visible };
  if (!candidate.apronsComplete) return { published: false, reason: "incomplete-aprons", visible };
  return { published: true, reason: "published", visible: { completeGeneration: candidate.generation, plan: candidate.plan } };
}
