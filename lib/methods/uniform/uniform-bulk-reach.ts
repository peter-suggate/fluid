/** The tiles of a 16h region's 3^3 neighbourhood its surface can reach
 * (BULK16 plan, "Eligibility behind a closed solid").
 *
 * The eligibility test (uniformBulkRegionClass, bulkSelect, bulkPromote)
 * fails a region for AIR, a band tile or a shallow tile anywhere in its
 * 12^3 tiles (the region's own tiles, 4 on every side). Air a static wall
 * hides is not a margin: water reaches it only around the wall. A tile is
 * reached when a face path of open tiles inside the neighbourhood joins it
 * to one of the region's own tiles. A path that leaves the neighbourhood is
 * at least 4 tiles long on each side, so an unreached tile is never nearer
 * the region than the 4-tile margin the test keeps. Open means not wholly
 * static solid: a cut tile passes the path, and a moving body is never a
 * wall here. Reach depends on static solids alone, so it is computed once
 * at construction and again on a solid edit (WebGPUUniformReferenceSolver
 * bulkReachMasks). A region needs a mask only where some open tile of its
 * neighbourhood is not reached; every other region reaches all of it.
 *
 * Mask bit i is neighbourhood tile d = (i % 12, i / 12 % 12, i / 144) - 4
 * from the region's tile origin, BULK_REACH_WORDS words. The GPU table: a
 * hash of `capacity` keys (key + 1, 0 empty, linear probe) then the masks,
 * BULK_REACH_WORDS a key slot. */

export type BulkReachVec = readonly [number, number, number];
export const BULK_REACH_SIDE = 12;
export const BULK_REACH_TILES = BULK_REACH_SIDE ** 3;
export const BULK_REACH_WORDS = BULK_REACH_TILES / 32;

/** The neighbourhood bit of offset d (-4..7 on each axis). */
export function bulkReachBit(d: BulkReachVec): number { return (d[0] + 4) + BULK_REACH_SIDE * ((d[1] + 4) + BULK_REACH_SIDE * (d[2] + 4)); }
export function bulkReached(mask: Uint32Array | undefined, d: BulkReachVec): boolean {
  if (!mask) return true;
  const i = bulkReachBit(d);
  return ((mask[i >> 5]! >>> (i & 31)) & 1) !== 0;
}

/** Region r's reach mask over its neighbourhood from closed(t) (a box tile
 * wholly static solid, or past the box), undefined when it reaches every
 * open tile. Own tiles are always reached; a closed tile is never entered. */
export function uniformBulkRegionReach(closed: (t: BulkReachVec) => boolean, region: BulkReachVec): Uint32Array | undefined {
  const S = BULK_REACH_SIDE, mask = new Uint32Array(BULK_REACH_WORDS), stack: number[] = [];
  const shut = new Uint8Array(BULK_REACH_TILES);
  let open = 0;
  for (let i = 0; i < BULK_REACH_TILES; i++) {
    const x = i % S, y = Math.floor(i / S) % S, z = Math.floor(i / (S * S));
    shut[i] = closed([4 * region[0] + x - 4, 4 * region[1] + y - 4, 4 * region[2] + z - 4]) ? 1 : 0;
    if (!shut[i]) open++;
    if (x >= 4 && x < 8 && y >= 4 && y < 8 && z >= 4 && z < 8) { mask[i >> 5]! |= 1 << (i & 31); stack.push(i); }
  }
  let reached = 0;
  for (const i of stack) if (!shut[i]) reached++;
  while (stack.length) {
    const i = stack.pop()!, x = i % S, y = Math.floor(i / S) % S, z = Math.floor(i / (S * S));
    for (const [dx, dy, dz] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] as const) {
      const u = x + dx, v = y + dy, w = z + dz;
      if (u < 0 || v < 0 || w < 0 || u >= S || v >= S || w >= S) continue;
      const j = u + S * (v + S * w);
      if (shut[j] || ((mask[j >> 5]! >>> (j & 31)) & 1)) continue;
      mask[j >> 5]! |= 1 << (j & 31); reached++; stack.push(j);
    }
  }
  return reached === open ? undefined : mask;
}

/** The table key of box region r (each axis below 1024). */
export function bulkReachKey(r: BulkReachVec): number {
  if (r.some(v => !Number.isInteger(v) || v < 0 || v >= 1024)) throw new Error("A BULK16 reach key needs box regions in 0..1023");
  return (r[0] | (r[1] << 10) | (r[2] << 20)) + 1;
}
function slotOf(key: number, capacity: number): number { return (Math.imul(key, 0x9e3779b1) >>> 0) >>> (32 - Math.log2(capacity)); }

/** The key slots a table of n masks takes: a power of two over twice n, so
 * a probe always meets an empty slot. 0 without masks (no table). */
export function bulkReachCapacity(n: number): number { return n === 0 ? 0 : 2 ** Math.ceil(Math.log2(2 * n + 2)); }
export function bulkReachTableWords(capacity: number): number { return capacity * (1 + BULK_REACH_WORDS); }

/** The GPU table of masks (keys by bulkReachKey) in `capacity` slots. */
export function bulkReachTable(masks: ReadonlyMap<number, Uint32Array>, capacity: number): Uint32Array<ArrayBuffer> {
  if (capacity === 0 && masks.size === 0) return new Uint32Array(0);
  if (masks.size >= capacity) throw new Error(`BULK16 reach: ${masks.size} masks over the ${capacity} key slots allocated at construction (a solid edit that walls off more regions than the scene began with)`);
  const out = new Uint32Array(bulkReachTableWords(capacity));
  for (const [key, mask] of masks) {
    let s = slotOf(key, capacity);
    while (out[s] !== 0) s = (s + 1) & (capacity - 1);
    out[s] = key; out.set(mask, capacity + BULK_REACH_WORDS * s);
  }
  return out;
}

/** The mask of `key` in a table (bReachSlot's probe), undefined: none. */
export function bulkReachLookup(table: Uint32Array, capacity: number, key: number): Uint32Array | undefined {
  if (capacity === 0) return undefined;
  let s = slotOf(key, capacity);
  for (let n = 0; n < capacity; n++) {
    if (table[s] === key) return table.subarray(capacity + BULK_REACH_WORDS * s, capacity + BULK_REACH_WORDS * (s + 1));
    if (table[s] === 0) return undefined;
    s = (s + 1) & (capacity - 1);
  }
  return undefined;
}

/** WGSL: bReachSlot(region) the mask slot of box region `region`, or
 * capacity where it reaches its whole neighbourhood; word(i) reads table
 * word i. */
export function bulkReachWGSL(capacity: number, word: (i: string) => string): string {
  if (capacity === 0) return /* wgsl */`const B_REACH_SLOTS:u32=0u;
fn bReachSlot(region:vec3i)->u32{return 0u;}
fn bReachWord(slot:u32,w:u32)->u32{return 0xffffffffu;}`;
  return /* wgsl */`const B_REACH_SLOTS:u32=${capacity}u;
fn bReachSlot(region:vec3i)->u32{
 if(any(region<vec3i(0))||any(region>=vec3i(1024))){return B_REACH_SLOTS;}
 let key=(u32(region.x)|(u32(region.y)<<10u)|(u32(region.z)<<20u))+1u;
 var s=(key*0x9e3779b1u)>>${32 - Math.log2(capacity)}u;
 for(var n=0u;n<B_REACH_SLOTS;n++){
  let k=${word("s")};
  if(k==key){return s;}
  if(k==0u){return B_REACH_SLOTS;}
  s=(s+1u)&(B_REACH_SLOTS-1u);
 }
 return B_REACH_SLOTS;
}
// Word w of slot's mask; every bit where the region reaches its whole neighbourhood.
fn bReachWord(slot:u32,w:u32)->u32{if(slot>=B_REACH_SLOTS){return 0xffffffffu;}return ${word(`B_REACH_SLOTS+${BULK_REACH_WORDS}u*slot+w`)};}`;
}
