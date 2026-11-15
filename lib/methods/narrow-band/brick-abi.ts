/**
 * The brick ABI of NB-FLIP's brick engine (docs/plans/nbflip-port/brick-abi.md, lane E).
 *
 * Every constant of the ABI doc lives here, and brickAbiWGSL() gives every
 * kernel the same constants and accessors. Changes go through lane E: the doc
 * and this file change together.
 *
 * Storage: sparse 8^3 bricks of kinds F (h), C (a page at 4h), B (BULK16) and
 * K (a static-solid cut), each slot at a stride of 520 words in channel-major
 * pools; a two-level directory (a toroidal window of page words, then 64 child
 * words a C slot); 32-word records; lists in slot order with indirect
 * arguments; particle bins per F slot.
 */

/** Words a brick slot of a channel: 512 cells and 8 words of padding, so the
 * 27 neighbours of a brick do not alias the same cache sets (the arena
 * measured 2.5x slower staging at 512). */
export const BRICK_STRIDE = 520;
export const BRICK_CELLS = 512;
export const RECORD_WORDS = 32;
export const CHILD_WORDS = 64;
export const CMASK_WORDS = 16;
/** The none word: not resident (directory, child, neighbour, cut slot). */
export const NONE = 0xffffffff;
/** Neighbour word kinds (bit 31 set): SEAM | cslot, BULK | bslot; NONE. */
export const SEAM = 0x80000000;
export const BULK = 0xa0000000;
/** A page word mid-claim inside setUpdate. */
export const CLAIM = 0xc0000000;
export const SLOT_MASK = 0x1fffffff;

/** The window's most pages an axis (keys carry 10 bits an axis). */
export const WINDOW_MAX_PAGES = 1024;

/** Vertex phi (h) of a brick that enters as air, beyond every band (the old
 * engine's "past the reach a vertex keeps its sign at six cells"). */
export const PHI_AIR = 6;
/** φ (h) where no slot stores a vertex: the old vertex field's empty value, fluidLatticeFarPhi in h (the lattice's largest extent). */
export const phiFar = (lattice: readonly number[]) => Math.max(...lattice);
/** The reach box's stencil margin in cells (brick-abi.md §5.4): with no travel
 * it gives the design's one-brick ring, and never more than
 * ceil(bound dt / 8h) + 1 bricks. */
export const REACH_MARGIN = 7;
/** A candidate key's flag (bit 31): the residency request already took its directory word NONE -> CLAIM. */
export const KEY_CLAIMED = 0x80000000;
/** Words an F slot of the particle support (NB_S_PSUPPORT, brick-abi.md §5.4): the support mask of the brick's 8
 * tiles, then six travel maxima (-xyz, +xyz; f32 bits, cells), then a pad. */
export const PSUPPORT_WORDS = 8;

/** Bricks an axis that hold the lattice's vertices 0..n (the skirt included). */
export const bricksPerAxis = (lattice: readonly number[]): [number, number, number] => lattice.map(n => Math.floor(n / 8) + 1) as [number, number, number];

export const RING_SLOTS = 4;
export const RING_WORDS = 256;
export const HEADER_WORDS = 256;

/** Control header words (set buffer, brick-abi.md §5.1). */
export const H = {
  FAIL: 0, FAIL_INFO: 1, STEP: 2, STATE: 3,
  TOP_F: 4, TOP_C: 5, TOP_B: 6, TOP_K: 7,
  HIGH_F: 8, HIGH_C: 9, HIGH_B: 10, HIGH_K: 11,
  LIVE_F: 12, LIVE_C: 13, LIVE_B: 14, LIVE_K: 15,
  REFUSED_F: 16, REFUSED_C: 17, REFUSED_B: 18, REFUSED_K: 19,
  PCOUNT0: 20, PCOUNT1: 21, P_OVERFLOW: 22, P_UNBINNED: 23,
  N_ENTER_F: 24, N_LEAVE_F: 25, N_ENTER_C: 26, N_LEAVE_C: 27, N_CAND: 28, N_CHANGED: 29,
  /** setTransfer's listed tiles: appended, then the count its jobs run (brick-abi.md §2.5). */
  N_TLIST: 30, N_TLIST_RUN: 31,
  LIST_COUNT: 32,
  LANE_F: 64, LANE_P: 128,
  PEAK_F: 192, PEAK_C: 193, PEAK_B: 194, PEAK_K: 195, PEAK_PARTICLES: 196, PEAK_CAND: 197,
  /** setCensus's counters (brick-abi.md §5.1; brick-census.wgsl.ts CENSUS_H). */
  CENSUS_CLAMPED: 224, CENSUS_CAND: 225, CENSUS_REACH: 226, CENSUS_CROSSING: 227,
  CENSUS_REQUIRED: 228, CENSUS_BAND: 229, CENSUS_REFINED: 230, CENSUS_TRIGGERED: 231,
  /** The release port (levelset-release.wgsl.ts; brick-abi.md §5.1), two copies alternating by the ring parity
   * (cfg.y): the plan's speed (f32 bits, m/s; nrReach), and 6 words each of the face directions' released wall reach
   * (f32 bits, cells; nrWallReach). */
  REL_SPEED0: 232, REL_SPEED1: 233, WALL_REACH0: 234, WALL_REACH1: 240,
  /** The head's residency (brick-residency.wgsl.ts; brick-abi.md §5.4): the largest region radius this census, in
   * tiles (the old pageSeed's closure radius), and the bricks and pages its request claimed. */
  RES_RADIUS: 246, RES_CLAIMED_F: 247, RES_CLAIMED_C: 248,
} as const;

/** Latched failure bits (H.FAIL), brick-abi.md §8. */
export const FAIL = {
  POOL_F: 1 << 0, POOL_C: 1 << 1, POOL_B: 1 << 2, POOL_K: 1 << 3,
  PARTICLES: 1 << 4, REACH: 1 << 5, LIST: 1 << 6, WINDOW: 1 << 7, UNSUPPORTED: 1 << 8,
  PIVOT: 1 << 9, SEGMENTS: 1 << 10,
  /** A tier change needs storage no resident brick or page holds (brick-abi.md §2.5). */
  TIER: 1 << 11,
} as const;
export const FAIL_POOL = FAIL.POOL_F | FAIL.POOL_C | FAIL.POOL_B | FAIL.POOL_K;
/** Reported, non-fatal state bits (H.STATE). */
export const STATE = { PRESSURE_CAP: 1 } as const;

/** F record word 31 (and C record word 31: LIQUID, SURFACE, SEAM, COLD, ENTERING, LEAVING). */
export const F_CLASS = {
  LIQUID: 1 << 0, SURFACE: 1 << 1, PARTICLES: 1 << 2, NEAR_SOLID: 1 << 3, NEAR_BODY: 1 << 4,
  SEAM: 1 << 5, SKIRT: 1 << 6, COLD: 1 << 7, CHANGED: 1 << 8,
  ENTERING: 1 << 30, LEAVING: 2 ** 31,
} as const;
/** Record words (F and C). */
export const REC = { SELF: 13, PARENT: 27, MASK: 28, KEY: 29, CUT: 30, CLASS: 31, C_CHILDREN: 27, C_LIQUID_TILES: 28 } as const;

/** FLAGS channel bits (cell, u32). */
export const FLAG = { OPEN: 1, LIQUID: 2, BAND: 4, VALID0: 8, VALID1: 16, VALID2: 32, OWNER_H: 64 } as const;

/** Lists (brick-abi.md §5.5): index, kind. */
export const LIST = {
  F_ALL: 0, F_LIQ: 1, F_SURF: 2, F_PART: 3, F_SEAM: 4, F_SOLID: 5, F_BODY: 6, F_COLD: 7,
  F_ENTER: 8, F_LEAVE: 9, F_CHANGED: 10,
  C_ALL: 11, C_LIQ: 12, C_SEAM: 13, C_ENTER: 14, C_LEAVE: 15,
  B_ALL: 16,
} as const;
export const LIST_SLOTS = 32;
export type ListName = keyof typeof LIST;
const listKind = (k: number): "F" | "C" | "B" => k <= 10 ? "F" : k <= 15 ? "C" : k === 16 ? "B" : "F";

/** Indirect arguments: 12 words a list (G1, G8, G64, pad), then FC_* at 384. Lanes: P from 512, F from 2048. */
export const ARGS = { PER_LIST: 12, G1: 0, G8: 3, G64: 6, FC_G1: 384, FC_G8: 387, FC_G64: 390, XFER_G1: 396, LANE_P: 512, LANE_F: 2048, WORDS: 4096 } as const;
/** Workgroups a dimension under the fold rule: x = min(n, FOLD), y = ceil(n / FOLD). */
export const FOLD = 32768;

/** Channels of each pool (brick-abi.md §2.3). pres is lane P's; cut is K-indexed. */
export const POOLS = {
  vel: ["U0", "U1", "U2", "UP0", "UP1", "UP2", "US0", "US1", "US2"],
  phi: ["PHI", "PHI1", "FLAGS"],
  pres: ["X", "R", "Z", "D", "Q", "RHS", "DIAG", "K0", "K1", "K2", "KU0", "KU1", "KU2", "PPHI", "B"],
  cut: ["FRAC", "AP0", "AP1", "AP2"],
} as const;
export type PoolName = keyof typeof POOLS;

/** Param ring slot words (brick-abi.md §5.2). */
export const RING = {
  STEP: 0, DT: 1, TIME: 2, FLAGS: 3, GRAVITY: 4, H: 7,
  VOLUME: 8, SOURCES: 16, BODIES: 64, BODY_WORDS: 16, BODY_RECORDS: 8, EDITS: 192, EDIT_COUNT: 222,
} as const;
export const RING_FLAG = { SOLID_EDIT: 1, BODIES: 2 } as const;

/** Per-scene capacities (scene.fluid.brickPools). */
export interface BrickCapacities {
  /** F bricks. */ readonly f: number;
  /** C pages. */ readonly c: number;
  /** B (BULK16) pages; 0 until M3. */ readonly b: number;
  /** K (cut) bricks. */ readonly k: number;
  /** Particles a bank. */ readonly particles: number;
  /** setTransfer's listed tiles a step (default 2 f; over it latches FAIL.LIST). */ readonly t?: number;
}

export interface BrickLayoutOptions extends BrickCapacities {
  /** The lattice in h cells. */
  readonly lattice: readonly [number, number, number];
  /** APIC records (28 floats) instead of FLIP's 12. */
  readonly apic?: boolean;
}

const align64 = (n: number) => Math.ceil(n / 64) * 64;
const pow2AtLeast = (n: number) => { let p = 1; while (p < n) p *= 2; return p; };

/** The window in pages an axis: a power of two, at least the lattice's pages plus one (the skirt). */
export function windowPages(lattice: readonly number[]): [number, number, number] {
  const w = lattice.map(n => pow2AtLeast(Math.floor(n / 32) + 1)) as [number, number, number];
  if (w.some(n => n > WINDOW_MAX_PAGES)) throw new Error(`Brick engine: a window of ${w.join("x")} pages, over ${WINDOW_MAX_PAGES} an axis`);
  return w;
}

/** The ABI's concrete layout for one scene: every region's word offset and every buffer's size. */
export class BrickLayout {
  readonly cap: Required<BrickCapacities>;
  readonly lattice: readonly [number, number, number];
  readonly window: readonly [number, number, number];
  readonly windowBits: readonly [number, number, number];
  readonly particleStride: number;
  /** set buffer regions (words). */
  readonly set: {
    readonly ring: number; readonly freeF: number; readonly freeC: number; readonly freeB: number; readonly freeK: number;
    readonly markF: number; readonly markC: number; readonly cand: number; readonly boundF: number; readonly boundC: number;
    readonly binStart: number; readonly binCount: number; readonly binCursor: number;
    /** Tiers (§2.5): the census's wanted h tiles (16 words a C slot), lane F's tile occupancy and the census's trigger bits (a word an F slot, bit j tile j). */
    readonly want: number; readonly tileOcc: number; readonly trig: number;
    /** setTransfer's listed tiles: 2 words an entry (C slot << 9 | tile, flags), cap.t entries. */
    readonly tlist: number;
    readonly lists: readonly number[]; readonly listCap: readonly number[];
    /** The 4h tiles' far-air gate (16 words a C slot, bit j tile j): written by lcClear, read by the next step's advect (§5.4). */
    readonly airGate: number;
    /** The release port (§5.4): the tiles' reach words (512 words a C slot each: nrReach's local, the spread's two
     * halves; the final is reach1), the step they were made for (a word a C slot: ring step + 1), and the released wall
     * face tiles (384 words a C slot, 64 a face direction). */
    readonly reach0: number; readonly reach1: number; readonly reachStep: number; readonly wallTile: number;
    /** The particle support (§5.4): PSUPPORT_WORDS an F slot, accumulated by F's pJoin, read and cleared by the head's resSeed. */
    readonly psupport: number;
    readonly words: number;
  };
  /** dir buffer regions (words). */
  readonly dir: { readonly page: number; readonly child: number; readonly frec: number; readonly crec: number; readonly cmask: number; readonly words: number };
  /** Words of each pool. */
  readonly poolWords: Readonly<Record<PoolName, number>>;
  readonly partsWords: number;
  /** setTransfer's face scratch: 384 floats a listed tile and face channel (U, US): positive faces, then low-wall rows. */
  readonly xferWords: number;

  constructor(options: BrickLayoutOptions) {
    const { f, c, b, k, particles } = options;
    for (const [name, v] of Object.entries({ f, c, b, k, particles }))
      if (!Number.isInteger(v) || v < (name === "b" || name === "k" ? 0 : 1)) throw new Error(`Brick engine: capacity ${name} = ${v}`);
    if (f >= SLOT_MASK || c >= SLOT_MASK) throw new Error("Brick engine: slot capacity over 2^29");
    const t = options.t ?? 2 * f;
    if (!Number.isInteger(t) || t < 1) throw new Error(`Brick engine: capacity t = ${t}`);
    this.cap = { f, c, b, k, particles, t };
    this.lattice = [...options.lattice] as [number, number, number];
    this.window = windowPages(options.lattice);
    this.windowBits = this.window.map(n => Math.log2(n)) as [number, number, number];
    this.particleStride = options.apic ? 28 : 12;
    let at = HEADER_WORDS;
    const take = (n: number) => { const o = at; at += align64(n); return o; };
    const ring = take(RING_SLOTS * RING_WORDS);
    const freeF = take(f), freeC = take(c), freeB = take(Math.max(b, 1)), freeK = take(Math.max(k, 1));
    const markF = take(f), markC = take(c), cand = take(f), boundF = take(f), boundC = take(c);
    const binStart = take(f), binCount = take(f), binCursor = take(f);
    const want = take(CMASK_WORDS * c), tileOcc = take(f), trig = take(f), tlist = take(2 * t);
    const listCap: number[] = [], lists: number[] = [];
    for (let j = 0; j < LIST_SLOTS; j++) {
      const kind = listKind(j), n = j > 16 ? 0 : kind === "F" ? f : kind === "C" ? c : b;
      listCap.push(n); lists.push(take(Math.max(n, 1)));
    }
    const airGate = take(CMASK_WORDS * c);
    const reach0 = take(512 * c), reach1 = take(512 * c), reachStep = take(c), wallTile = take(384 * c);
    const psupport = take(PSUPPORT_WORDS * f);
    this.set = { ring, freeF, freeC, freeB, freeK, markF, markC, cand, boundF, boundC, binStart, binCount, binCursor, want, tileOcc, trig, tlist, lists, listCap, airGate, reach0, reach1, reachStep, wallTile, psupport, words: at };
    const pageWords = this.window[0] * this.window[1] * this.window[2];
    let d = 0;
    const dtake = (n: number) => { const o = d; d += align64(n); return o; };
    const page = dtake(pageWords), child = dtake(CHILD_WORDS * c), frec = dtake(RECORD_WORDS * f), crec = dtake(RECORD_WORDS * c), cmask = dtake(CMASK_WORDS * c);
    this.dir = { page, child, frec, crec, cmask, words: d };
    const fc = (n: number) => n * BRICK_STRIDE * (f + c);
    this.poolWords = { vel: fc(POOLS.vel.length), phi: fc(POOLS.phi.length), pres: fc(POOLS.pres.length), cut: POOLS.cut.length * BRICK_STRIDE * Math.max(k, 1) };
    this.partsWords = 2 * particles * this.particleStride;
    this.xferWords = 2 * 384 * t;
  }

  /** A channel's first word in its pool (brick-abi.md §2.2). */
  base(pool: PoolName, kind: "F" | "C" | "K", channel: string): number {
    const names: readonly string[] = POOLS[pool], ch = names.indexOf(channel);
    if (ch < 0) throw new Error(`Brick engine: no channel ${channel} in pool ${pool}`);
    if (pool === "cut") { if (kind !== "K") throw new Error("Brick engine: the cut pool is K-indexed"); return ch * this.cap.k * BRICK_STRIDE; }
    if (kind === "K") throw new Error(`Brick engine: pool ${pool} has no K region`);
    return kind === "F" ? ch * this.cap.f * BRICK_STRIDE : names.length * this.cap.f * BRICK_STRIDE + ch * this.cap.c * BRICK_STRIDE;
  }

  pageIndex(px: number, py: number, pz: number): number {
    const [wx, wy, wz] = this.window, [bx, by] = this.windowBits;
    return (px & (wx - 1)) | ((py & (wy - 1)) << bx) | ((pz & (wz - 1)) << (bx + by));
  }

  /** Bytes the engine's core buffers take (set, dir, args, pools, particle banks). */
  get bytes(): number {
    return 4 * (this.set.words + this.dir.words + ARGS.WORDS + Object.values(this.poolWords).reduce((a, b) => a + b, 0) + this.partsWords + this.xferWords);
  }
}

export const brickKey = (bx: number, by: number, bz: number) => ((bx & 1023) | ((by & 1023) << 10) | ((bz & 1023) << 20)) >>> 0;
export const brickKeyCoord = (key: number): [number, number, number] => [key & 1023, (key >>> 10) & 1023, (key >>> 20) & 1023];
export const brickLocal = (bx: number, by: number, bz: number) => (bx & 3) | ((by & 3) << 2) | ((bz & 3) << 4);
export const cellLocal = (x: number, y: number, z: number) => (x & 7) | ((y & 7) << 3) | ((z & 7) << 6);
/** Neighbour index of an offset in (-1..1)^3. */
export const neighbourIndex = (dx: number, dy: number, dz: number) => (dx + 1) + 3 * (dy + 1) + 9 * (dz + 1);

export interface BrickAbiWGSLOptions {
  /** The set buffer's WGSL name, and whether this module binds it as array<atomic<u32>>. */
  readonly set?: { readonly name: string; readonly atomic: boolean };
  /** The dir buffer's WGSL name, and whether atomic. */
  readonly dir?: { readonly name: string; readonly atomic: boolean };
}

const u = (n: number) => `${n >>> 0}u`;

/** The ABI's constants and accessors for a kernel module. Declares no bindings:
 * the module declares set and dir under the names given here. */
export function brickAbiWGSL(layout: BrickLayout, options: BrickAbiWGSLOptions = {}): string {
  const L = layout, s = L.set, d = L.dir;
  const ld = (buf: { name: string; atomic: boolean }, at: string) => buf.atomic ? `atomicLoad(&${buf.name}[${at}])` : `${buf.name}[${at}]`;
  const lines: string[] = [];
  const c = (name: string, v: number) => lines.push(`const ${name}:u32=${u(v)};`);
  c("NB_STRIDE", BRICK_STRIDE); c("NB_NONE", NONE); c("NB_SEAM", SEAM); c("NB_BULK", BULK); c("NB_CLAIM", CLAIM); c("NB_SLOT_MASK", SLOT_MASK);
  c("NB_FCAP", L.cap.f); c("NB_CCAP", L.cap.c); c("NB_BCAP", L.cap.b); c("NB_KCAP", L.cap.k); c("NB_PCAP", L.cap.particles);
  c("NB_PSTRIDE", L.particleStride); c("NB_FOLD", FOLD);
  lines.push(`const NB_LATTICE:vec3i=vec3i(${L.lattice.join(",")});`);
  lines.push(`const NB_BRICKS:vec3i=vec3i(${bricksPerAxis(L.lattice).join(",")});`);
  lines.push(`const NB_PHI_AIR:f32=${PHI_AIR.toFixed(1)};`);
  // φ (h) where nothing is stored: the old vertex field's empty value, fluidLatticeFarPhi (the lattice's largest
  // extent; lib/core/initial-liquid-lattice.ts), in h. Stored air is NB_PHI_AIR (setInit).
  lines.push(`const NB_PHI_FAR:f32=${phiFar(L.lattice).toFixed(1)};`);
  lines.push(`const NB_REACH_MARGIN:f32=${REACH_MARGIN.toFixed(1)};`);
  lines.push("const NB_KEY_PAGE:u32=0x40000000u;");
  lines.push(`const NB_WINDOW:vec3u=vec3u(${L.window.map(n => u(n)).join(",")});`);
  lines.push(`const NB_WINDOW_BITS:vec3u=vec3u(${L.windowBits.map(n => u(n)).join(",")});`);
  for (const [name, v] of Object.entries(H)) c(`NB_H_${name}`, v);
  for (const [name, v] of Object.entries(FAIL)) c(`NB_FAIL_${name}`, v);
  for (const [name, v] of Object.entries(STATE)) c(`NB_STATE_${name}`, v);
  for (const [name, v] of Object.entries(F_CLASS)) c(`NB_CLASS_${name}`, v);
  for (const [name, v] of Object.entries(REC)) c(`NB_REC_${name}`, v);
  for (const [name, v] of Object.entries(FLAG)) c(`NB_FLAG_${name}`, v);
  for (const [name, v] of Object.entries(LIST)) c(`NB_LIST_${name}`, v);
  for (const [name, v] of Object.entries(ARGS)) c(`NB_ARGS_${name}`, v);
  for (const [name, v] of Object.entries(RING)) c(`NB_RING_${name}`, v);
  for (const [name, v] of Object.entries(RING_FLAG)) c(`NB_RING_FLAG_${name}`, v);
  c("NB_RING_BASE", s.ring); c("NB_RING_WORDS", RING_WORDS);
  c("NB_S_FREE_F", s.freeF); c("NB_S_FREE_C", s.freeC); c("NB_S_FREE_B", s.freeB); c("NB_S_FREE_K", s.freeK);
  c("NB_S_MARK_F", s.markF); c("NB_S_MARK_C", s.markC); c("NB_S_CAND", s.cand); c("NB_S_BOUND_F", s.boundF); c("NB_S_BOUND_C", s.boundC);
  c("NB_S_BIN_START", s.binStart); c("NB_S_BIN_COUNT", s.binCount); c("NB_S_BIN_CURSOR", s.binCursor);
  c("NB_S_WANT", s.want); c("NB_S_TILE_OCC", s.tileOcc); c("NB_S_TRIG", s.trig); c("NB_S_TLIST", s.tlist); c("NB_TCAP", L.cap.t); c("NB_S_AIR_GATE", s.airGate);
  c("NB_S_REACH0", s.reach0); c("NB_S_REACH1", s.reach1); c("NB_S_REACH_STEP", s.reachStep); c("NB_S_WALL_TILE", s.wallTile);
  c("NB_S_PSUPPORT", s.psupport); c("NB_PSUPPORT_WORDS", PSUPPORT_WORDS); c("NB_KEY_CLAIMED", KEY_CLAIMED);
  lines.push(`const NB_S_LIST=array<u32,${LIST_SLOTS}>(${s.lists.map(u).join(",")});`);
  lines.push(`const NB_S_LIST_CAP=array<u32,${LIST_SLOTS}>(${s.listCap.map(u).join(",")});`);
  c("NB_D_PAGE", d.page); c("NB_D_CHILD", d.child); c("NB_D_FREC", d.frec); c("NB_D_CREC", d.crec); c("NB_D_CMASK", d.cmask);
  for (const pool of Object.keys(POOLS) as PoolName[]) {
    for (const ch of POOLS[pool]) {
      if (pool === "cut") c(`NB_${pool.toUpperCase()}_${ch}`, L.base(pool, "K", ch));
      else { c(`NB_${pool.toUpperCase()}_F_${ch}`, L.base(pool, "F", ch)); c(`NB_${pool.toUpperCase()}_C_${ch}`, L.base(pool, "C", ch)); }
    }
  }
  lines.push(/* wgsl */`
// A slot's word of a channel (base: a channel constant above).
fn nbAt(base:u32,slot:u32,local:u32)->u32{return base+NB_STRIDE*slot+local;}
fn nbLocal(c:vec3i)->u32{return u32((c.x&7)|((c.y&7)<<3u)|((c.z&7)<<6u));}
fn nbLocalCoord(l:u32)->vec3i{return vec3i(i32(l&7u),i32((l>>3u)&7u),i32(l>>6u));}
fn nbBrickLocal(b:vec3i)->u32{return u32((b.x&3)|((b.y&3)<<2u)|((b.z&3)<<4u));}
fn nbKey(b:vec3i)->u32{return u32(b.x&1023)|(u32(b.y&1023)<<10u)|(u32(b.z&1023)<<20u);}
fn nbKeyCoord(k:u32)->vec3i{return vec3i(i32(k&1023u),i32((k>>10u)&1023u),i32((k>>20u)&1023u));}
fn nbPageIndex(p:vec3i)->u32{let q=vec3u(p)&(NB_WINDOW-vec3u(1u));return q.x|(q.y<<NB_WINDOW_BITS.x)|(q.z<<(NB_WINDOW_BITS.x+NB_WINDOW_BITS.y));}
fn nbNeighbourIndex(o:vec3i)->u32{return u32((o.x+1)+3*(o.y+1)+9*(o.z+1));}
fn nbNeighbourOffset(n:u32)->vec3i{return vec3i(i32(n%3u),i32((n/3u)%3u),i32(n/9u))-vec3i(1);}
// The fold rule: a list-indirect kernel's group index.
fn nbGroup(wid:vec3u)->u32{return wid.x+NB_FOLD*wid.y;}
// Inside the lattice (the box rule's domain; skirts lie beyond it on a + axis).
fn nbInLattice(c:vec3i)->bool{return all(c>=vec3i(0))&&all(c<NB_LATTICE);}
// A brick coordinate that can hold a slot (vertices 0..lattice, the skirt included).
fn nbBrickInRange(b:vec3i)->bool{return all(b>=vec3i(0))&&all(b<NB_BRICKS);}
`);
  if (options.dir) {
    const D = options.dir;
    lines.push(/* wgsl */`
fn nbPageWord(p:vec3i)->u32{return ${ld(D, "NB_D_PAGE+nbPageIndex(p)")};}
// The C slot of page p, or NB_NONE (air, or a BULK16 page).
fn nbCSlotOfPage(p:vec3i)->u32{let w=nbPageWord(p);return select(NB_NONE,w,w<NB_SEAM);}
fn nbChild(cslot:u32,brickLocal:u32)->u32{return ${ld(D, "NB_D_CHILD+64u*cslot+brickLocal")};}
// The F slot of brick b, or NB_NONE. Callers apply the box rule first (no negative coordinates).
fn nbFSlotOfBrick(b:vec3i)->u32{let cs=nbCSlotOfPage(b>>vec3u(2u));if(cs==NB_NONE){return NB_NONE;}return nbChild(cs,nbBrickLocal(b));}
fn nbFSlotOfCell(c:vec3i)->u32{return nbFSlotOfBrick(c>>vec3u(3u));}
fn nbFRec(slot:u32,k:u32)->u32{return ${ld(D, "NB_D_FREC+32u*slot+k")};}
fn nbCRec(slot:u32,k:u32)->u32{return ${ld(D, "NB_D_CREC+32u*slot+k")};}
fn nbFOrigin(slot:u32)->vec3i{return 8*nbKeyCoord(nbFRec(slot,NB_REC_KEY));}
fn nbCOrigin(slot:u32)->vec3i{return 32*nbKeyCoord(nbCRec(slot,NB_REC_KEY));}
// Tile t (C local) of C slot cslot is h-owned.
fn nbTileH(cslot:u32,t:u32)->bool{return ((${ld(D, "NB_D_CMASK+16u*cslot+(t>>5u)")}>>(t&31u))&1u)!=0u;}
`);
  }
  if (options.set) {
    const S = options.set;
    lines.push(/* wgsl */`
fn nbSetWord(i:u32)->u32{return ${ld(S, "i")};}
fn nbRing(slot:u32,k:u32)->u32{return ${ld(S, "NB_RING_BASE+NB_RING_WORDS*slot+k")};}
fn nbRingF(slot:u32,k:u32)->f32{return bitcast<f32>(nbRing(slot,k));}
fn nbListCount(k:u32)->u32{return min(${ld(S, "NB_H_LIST_COUNT+k")},NB_S_LIST_CAP[k]);}
fn nbListSlot(k:u32,e:u32)->u32{return ${ld(S, "NB_S_LIST[k]+e")};}
`);
    if (S.atomic) lines.push(/* wgsl */`
fn nbFail(bits:u32,info:u32){if((atomicOr(&${S.name}[NB_H_FAIL],bits)&bits)!=bits){atomicCompareExchangeWeak(&${S.name}[NB_H_FAIL_INFO],0u,info);}}
`);
  }
  return lines.join("\n");
}

/** Indirect argument triple of n workgroups under the fold rule. */
export const foldArgs = (n: number): [number, number, number] => n === 0 ? [0, 1, 1] : [Math.min(n, FOLD), Math.ceil(n / FOLD), 1];
