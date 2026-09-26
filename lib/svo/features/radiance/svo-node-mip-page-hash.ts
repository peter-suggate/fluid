/**
 * The node-mip pyramid's page table: an open-addressed hash from
 * `(level, page coordinate)` to atlas slot, sized by the pages it holds.
 *
 * It replaces a *dense* direct table — one r32uint texel for every page the
 * domain's page grid could hold, one Z slab per level — whose size followed the
 * world's extent rather than its occupancy. That table had two ceilings, both
 * about the span and neither about the scene: its extent had to fit
 * `maxTextureDimension3D` per axis (and 192 MB), and its per-level slab offsets
 * lived in fixed twelve-entry uniform arrays, so a world wider than 2^11 finest
 * pages (25.6 m at a 1.5625 mm leaf) withdrew its whole derived-lighting path
 * for want of an *addressing* structure.
 *
 * A hash has neither. Its capacity is a power of two at least
 * `1 / SVO_NODE_MIP_PAGE_HASH.maximumLoadFactor` times the page capacity, so it
 * costs 16 B x 4 = 64 B per addressable page — about 3 % of the opacity and
 * radiance atlas bytes that page already costs — whatever the span. The level is
 * part of the key, so there is no per-level table to cap the level count.
 *
 * Layout: a 2D `rgba32uint` texture, `width x height` entries, both powers of
 * two, filled row-major. An entry is `(x | level << 24, y, z, slot + 1)`; a zero
 * `w` is an empty entry. The shader derives the capacity and the row width from
 * `textureDimensions`, so the table and its readers cannot disagree about shape
 * and no uniform carries it. Linear probing; the builder rejects (throws) a
 * table any of whose keys sits more than `maximumProbes` entries from its home,
 * so a lookup is bounded by that constant and a miss terminates at the first
 * empty entry — at a load factor of a quarter, after 1.4 probes on average.
 *
 * Slots are exactly what the dense table held, so every consumer resolves the
 * same slot for the same page and a frame is byte-identical across the swap.
 */
export const SVO_NODE_MIP_PAGE_HASH = Object.freeze({
  format: "rgba32uint" as GPUTextureFormat,
  bytesPerEntry: 16,
  wordsPerEntry: 4,
  maximumLoadFactor: 0.25,
  /** Longest probe sequence a built table may contain; the WGSL loop bound. */
  maximumProbes: 32,
  minimumCapacity: 64,
  /** Row width ceiling. Rows then wrap, so `maxTextureDimension2D` bounds rows, not entries. */
  maximumWidth: 4096,
  /** Coordinates share the first key word with the level. */
  coordinateBits: 24,
  maximumLevels: 32,
} as const);

/**
 * Most levels a node-mip pyramid may have.
 *
 * Nothing in the page table caps it any more; the remaining per-level arrays are
 * the derived planner's worklist sections (sized to this) and the directory's
 * level starts (whose last entry covers every level above it, so they stay
 * correct past twelve). Twenty levels is 2^19 finest pages a side: 6.5 km at a
 * 1.5625 mm leaf, and the tree's own 21-bit Morton keys run out at about the
 * same place.
 */
export const SVO_NODE_MIP_MAXIMUM_LEVELS = 20;

export interface SvoNodeMipPageHashShape {
  /** Entries; `width * height`, a power of two. */
  capacity: number;
  width: number;
  height: number;
  bytes: number;
}

export interface SvoNodeMipPageHashEntry {
  level: number;
  coordinate: readonly [number, number, number];
  slot: number;
}

function nextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) result *= 2;
  return result;
}

/** Table shape for a fixed page capacity. Throws when the device cannot hold it. */
export function svoNodeMipPageHashShape(pageCapacity: number, maximumTextureDimension2D = 8_192): SvoNodeMipPageHashShape {
  if (!Number.isSafeInteger(pageCapacity) || pageCapacity < 0) throw new RangeError("Node-mip page-hash capacity must be a non-negative safe integer");
  const capacity = nextPowerOfTwo(Math.max(SVO_NODE_MIP_PAGE_HASH.minimumCapacity,
    Math.ceil(pageCapacity / SVO_NODE_MIP_PAGE_HASH.maximumLoadFactor)));
  const widthLimit = 2 ** Math.floor(Math.log2(Math.max(1, Math.min(SVO_NODE_MIP_PAGE_HASH.maximumWidth, maximumTextureDimension2D))));
  const width = Math.min(capacity, widthLimit);
  const height = capacity / width;
  if (height > maximumTextureDimension2D) {
    throw new RangeError(`Node-mip page hash for ${pageCapacity} pages needs ${height} rows, exceeding the ${maximumTextureDimension2D} texture limit`);
  }
  return { capacity, width, height, bytes: capacity * SVO_NODE_MIP_PAGE_HASH.bytesPerEntry };
}

/** 32-bit finalizer (murmur3 fmix). Mirrored exactly by `svoNodeMipPageHashWGSL`. */
function mix32(value: number): number {
  let h = value >>> 0;
  h ^= h >>> 16; h = Math.imul(h, 0x85ebca6b) >>> 0;
  h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35) >>> 0;
  h ^= h >>> 16;
  return h >>> 0;
}

export function svoNodeMipPageHashHome(level: number, x: number, y: number, z: number): number {
  return mix32((Math.imul(x, 0x9e3779b1) ^ Math.imul(y, 0x85ebca77) ^ Math.imul(z, 0xc2b2ae3d) ^ Math.imul(level + 1, 0x27d4eb2f)) >>> 0);
}

/**
 * Fill a table. Every entry's key must be unique; a duplicate, an out-of-range
 * coordinate or a probe sequence past `maximumProbes` is a fault in the plan or
 * the shape, and throws rather than dropping a page that would then read as air.
 */
export function buildSvoNodeMipPageHash(
  entries: Iterable<SvoNodeMipPageHashEntry>,
  shape: SvoNodeMipPageHashShape,
): { words: Uint32Array<ArrayBuffer>; entries: number; longestProbe: number } {
  const words = new Uint32Array(shape.capacity * SVO_NODE_MIP_PAGE_HASH.wordsPerEntry);
  const mask = shape.capacity - 1;
  const coordinateLimit = 2 ** SVO_NODE_MIP_PAGE_HASH.coordinateBits;
  let count = 0, longestProbe = 0;
  for (const { level, coordinate: [x, y, z], slot } of entries) {
    if (!Number.isSafeInteger(level) || level < 0 || level >= SVO_NODE_MIP_PAGE_HASH.maximumLevels) throw new RangeError(`Node-mip page-hash level ${level} is out of range`);
    for (const component of [x, y, z]) {
      if (!Number.isSafeInteger(component) || component < 0 || component >= coordinateLimit) throw new RangeError(`Node-mip page-hash coordinate ${component} is out of range`);
    }
    if (!Number.isSafeInteger(slot) || slot < 0 || slot >= 0xffffffff) throw new RangeError(`Node-mip page-hash slot ${slot} is out of range`);
    const key0 = (x | (level << SVO_NODE_MIP_PAGE_HASH.coordinateBits)) >>> 0;
    let index = svoNodeMipPageHashHome(level, x, y, z) & mask;
    for (let probe = 0; ; probe += 1) {
      if (probe >= SVO_NODE_MIP_PAGE_HASH.maximumProbes || probe >= shape.capacity) {
        throw new RangeError(`Node-mip page hash probe sequence exceeds ${SVO_NODE_MIP_PAGE_HASH.maximumProbes} at ${count} of ${shape.capacity} entries`);
      }
      const base = index * SVO_NODE_MIP_PAGE_HASH.wordsPerEntry;
      if (words[base + 3] === 0) {
        words[base] = key0; words[base + 1] = y; words[base + 2] = z; words[base + 3] = slot + 1;
        longestProbe = Math.max(longestProbe, probe + 1);
        break;
      }
      if (words[base] === key0 && words[base + 1] === y && words[base + 2] === z) {
        throw new RangeError(`Node-mip page hash received duplicate page ${level}:${x},${y},${z}`);
      }
      index = (index + 1) & mask;
    }
    count += 1;
  }
  if (count > shape.capacity * SVO_NODE_MIP_PAGE_HASH.maximumLoadFactor) {
    throw new RangeError(`Node-mip page hash holds ${count} pages, above its ${shape.capacity * SVO_NODE_MIP_PAGE_HASH.maximumLoadFactor} load ceiling`);
  }
  return { words, entries: count, longestProbe };
}

/** CPU mirror of `svoNodeMipPageHashFind`, for oracles and tools. */
export function findSvoNodeMipPageHash(
  words: Uint32Array,
  shape: SvoNodeMipPageHashShape,
  level: number,
  coordinate: readonly [number, number, number],
): number | undefined {
  const [x, y, z] = coordinate;
  const key0 = (x | (level << SVO_NODE_MIP_PAGE_HASH.coordinateBits)) >>> 0;
  const mask = shape.capacity - 1;
  let index = svoNodeMipPageHashHome(level, x, y, z) & mask;
  for (let probe = 0; probe < SVO_NODE_MIP_PAGE_HASH.maximumProbes; probe += 1) {
    const base = index * SVO_NODE_MIP_PAGE_HASH.wordsPerEntry;
    if (words[base + 3] === 0) return undefined;
    if (words[base] === key0 && words[base + 1] === y && words[base + 2] === z) return words[base + 3] - 1;
    index = (index + 1) & mask;
  }
  return undefined;
}

/**
 * Binding-free lookup. Returns the slot, or `0xffffffff` for a page the table
 * does not hold. Coordinates at or past 2^24 cannot be keys and miss directly.
 */
export const svoNodeMipPageHashWGSL = /* wgsl */ `
fn svoNodeMipPageHashMix(value:u32)->u32{var h=value;h^=h>>16u;h*=0x85ebca6bu;h^=h>>13u;h*=0xc2b2ae35u;h^=h>>16u;return h;}
fn svoNodeMipPageHashFind(table:texture_2d<u32>,level:u32,coordinate:vec3u)->u32{
  if(any(coordinate>=vec3u(${2 ** SVO_NODE_MIP_PAGE_HASH.coordinateBits}u))||level>=${SVO_NODE_MIP_PAGE_HASH.maximumLevels}u){return 0xffffffffu;}
  let dimensions=textureDimensions(table);let widthShift=firstTrailingBit(dimensions.x);let mask=dimensions.x*dimensions.y-1u;
  let key0=coordinate.x|(level<<${SVO_NODE_MIP_PAGE_HASH.coordinateBits}u);
  var index=svoNodeMipPageHashMix((coordinate.x*0x9e3779b1u)^(coordinate.y*0x85ebca77u)^(coordinate.z*0xc2b2ae3du)^((level+1u)*0x27d4eb2fu))&mask;
  for(var probe=0u;probe<${SVO_NODE_MIP_PAGE_HASH.maximumProbes}u;probe+=1u){
    let entry=textureLoad(table,vec2u(index&(dimensions.x-1u),index>>widthShift),0);
    if(entry.w==0u){return 0xffffffffu;}
    if(entry.x==key0&&entry.y==coordinate.y&&entry.z==coordinate.z){return entry.w-1u;}
    index=(index+1u)&mask;
  }
  return 0xffffffffu;
}
`;
