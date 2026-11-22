/**
 * LANE F OWNS THIS FILE. The brick pool as lane F's kernels read it (brick-abi.md §1, §2.3), and lane F's own
 * buffers (brick-abi.md §6.2: lane-owned, allocated zeroed).
 *
 * Box rule, in the ABI's units (cells, φ in h, m/s):
 * - A cell or vertex outside the bricks a slot can hold (any coordinate below 0, or past the skirt) has no slot.
 * - A brick no slot holds is air (§2.3: a non-resident brick is air outside the reach): φ reads NB_PHI_AIR_C (16 h,
 *   the old stored far air, as an entering brick holds it; a stated deviation from the old vertex field's empty value,
 *   the box's largest extent, which a read interpolating through an absent vertex would carry into the physics and
 *   make it depend on the box), the particle depth reads NB_PHI_AIR, and a face reads 0.
 * - A face on a box wall (index 0 or N along its axis) or outside the lattice across it is closed and reads 0.
 *   Readers never clamp an index into the box.
 *
 * Lane F's buffers (one binding each, regions below):
 * - `fcell`, per cell of an F slot (index 512 slot + local) unless noted:
 *   - CNT: the rebin's per-cell counts (pMove's atomic ranks; pScan reads and zeroes them);
 *   - RUN: the first record of the cell's run in the bank pScatter wrote (the cell-sorted bin);
 *   - NEAR, NEAR2: the redistance's nearest crossing cell (old NB_DEPTH_B, a packed cell or NONE) and its scratch;
 *   - KEY, HOT: phiUnion's splat keys at vertices (nbfNearestKey: the nearest sample, and the nearest hot one);
 *   - DEPTH: the particle depth at vertices (old NB_DEPTH_A, cells; "bank A");
 *   - MASK: crossing cells, 16 words a slot (bit l of word l / 32);
 *   - TILE: FT_WORDS words for each of a slot's 8 tiles (FT_* below);
 *   - SLOT: 1 + the key the slot held when F last initialised it (a slot whose record key differs is new);
 *   - BOUND: the slot's velocity bound as pScan saw it (BOUND_F before setMark clears it; m/s, f32 bits).
 *   - Tiered only, uCache's resolved copies of this step's opening state (no region all h): UNIT, 3 x 729 a slot
 *     (FS_UNIT_FACES: the 9^3 faces o .. o + 8, local r.x + 9 r.y + 81 r.z, the slot's 8^3 and at 8 the +x, +y, +z
 *     neighbours' as an apron, so a 2 x 2 x 2 stencil based in the slot reads one slot), the h tap of US at each face
 *     (F's face where F owns it, else the 4h cache's trilinear: the old unit velocity cache; 0 at a closed face); PHIU, the owner-resolved PHI at each vertex (lAdvect's reads); HMASK, a word a tile, bit k the h
 *     ownership of the tile t + (k % 3, k / 3 % 3, k / 9) - 1 (bit 13 the tile's own); CACHE4, 3 x 512 a C slot, the
 *     4h cache of US on each tile's lower plane (the old 4h velocity cache).
 *   - LIQ: each cell's liquid phase this step (fgLiquidCell: centre φ under -0.001 of a cell, in the lattice), 16
 *     words a slot (bit l of word l / 32), written by uTransfer (uKick at boot) after the level set's last φ write,
 *     read by the tail's extension (uExt4Seed, uExtend) in place of recomputing each cell's eight corners.
 *   - Tiered only, DIL4: a word for each tile of a C slot, written by lDilateC for the tiles no F slot holds (the
 *     4h tiles past the owners' margin): bits 0..11 the dilation's x pass (lDilate's DA), 12..23 its y pass (DB),
 *     24..31 the tile's source flags (FTF_*) from its eight C corners. lDilate reads it where a tile has no F slot.
 * - `fdep`: p2g's fixed-point deposits, 2 words (momentum i32, weight u32) per face of each axis.
 * - `fpart`, per record of a bank: RANK (2 words: the destination rank of the record and of its split half, or
 *   NONE) and MOVED (a third bank: the move's output, which pScatter files into the new bank).
 */
import { BOX_TOKEN, BRICK_CELLS, type BrickLayout, brickAbiWGSL, resolveBox } from "./brick-abi";
import { type BrickBinding, type BrickKernel, declareBindings } from "./brick-kernel";
import { brickOwnerWGSL } from "./brick-owner.wgsl";

/** Tile words (FT_*): per 4^3 tile, FT_WORDS words. */
export const FT = {
  /** activity H (f32), the census heat that cools at 4/s */ H: 0,
  /** target heat T (f32): max of H over the 27 tiles, the neighbours' at half */ T: 1,
  /** THETA (f32 bits, atomicMax): the max heat of samples whose 2-cell support reaches the tile */ THETA: 2,
  /** SURFACE_HEAT (f32 bits, atomicMax): the max heat of the tile's own samples */ SURF: 3,
  /** flag bits (FTF_*) */ FLAGS: 4,
  /** the dilation passes' ping-pong words */ DA: 5, DB: 6,
  /** distances after the dilation (packed 4-bit Chebyshev tile distances: 0 crossing, 1 air seed, 2 occupied) */ DIST: 7,
  /** the activity spread's ping-pong maxima (f32) */ HA: 8, HB: 9,
  /** the census heat this step (f32), before the spread */ HN: 10,
  WORDS: 12,
} as const;
/** Tile flag bits. */
export const FTF = { SURFACE_TILE: 1, CROSSING: 2, AIR_SEED: 4, OCCUPIED: 8, OPEN: 16 } as const;

export interface FluidLayout {
  /** The stages read through the tier owners and write only owned storage (FG_TIERED; brick-abi.md §2.5). */
  readonly tiered: boolean;
  readonly cells: number;
  readonly fcell: { readonly CNT: number; readonly RUN: number; readonly NEAR: number; readonly NEAR2: number; readonly KEY: number; readonly HOT: number; readonly DEPTH: number; readonly MASK: number; readonly TILE: number; readonly SLOT: number; readonly BOUND: number; readonly UNIT: number; readonly PHIU: number; readonly HMASK: number; readonly CACHE4: number; readonly LIQ: number; readonly DIL4: number; readonly RES: number; readonly words: number };
  readonly fdep: { readonly words: number };
  readonly fpart: { readonly RANK: number; readonly words: number };
  readonly fmoved: { readonly words: number };
  /** The 4h extension's state (tiered only; one word all h): per C slot, SEEDS then STATE, each 512 tiles x 3 axes
   * x (value, prior, distance), FX4_WORDS a slot. */
  readonly fext4: { readonly words: number };
}
/** Words a C slot holds in fext4: seeds and state, 512 tiles x 3 axes x 3 words each. */
export const FX4_WORDS = 2 * 512 * 3 * 3;

/** UNIT's faces a slot and axis: 9^3, the brick's 8^3 and the +side apron (uCache writes all of them in a near brick). */
export const FS_UNIT_FACES = 729;

export function fluidLayout(L: BrickLayout, options: { readonly tiered?: boolean } = {}): FluidLayout {
  const f = L.cap.f, cells = f * BRICK_CELLS, p = L.cap.particles;
  let at = 0;
  const take = (n: number) => { const o = at; at += n; return o; };
  const CNT = take(cells), RUN = take(cells), NEAR = take(cells), NEAR2 = take(cells), KEY = take(cells), HOT = take(cells), DEPTH = take(cells);
  const MASK = take(16 * f), TILE = take(8 * FT.WORDS * f), SLOT = take(f), BOUND = take(f);
  const tiered = options.tiered === true;
  // Tiered, uCache's resolved copies (the old engine's unit and 4h velocity caches and its vertex field).
  const UNIT = take(tiered ? 3 * FS_UNIT_FACES * f : 0), PHIU = take(tiered ? cells : 0), HMASK = take(tiered ? 8 * f : 0), CACHE4 = take(tiered ? 3 * 512 * L.cap.c : 0);
  const LIQ = take(16 * f);
  // Tiered, lDilateC's word for each tile of a C slot that no F slot holds.
  const DIL4 = take(tiered ? 512 * L.cap.c : 0);
  // Tiered, the analysis's re-resolve stamps (lResolveD): a C slot's word (FC_RES + cs) where lcClear changed one of
  // its vertices, an F slot's (FC_RES + NB_CCAP + slot) where lClear changed a non-owned vertex; the step's ring STEP + 1.
  const RES = take(tiered ? L.cap.c + f : 0);
  const fcell = { CNT, RUN, NEAR, NEAR2, KEY, HOT, DEPTH, MASK, TILE, SLOT, BOUND, UNIT, PHIU, HMASK, CACHE4, LIQ, DIL4, RES, words: Math.max(at, 1) };
  const fpart = { RANK: 0, words: 2 * p };
  return { tiered, cells, fcell, fdep: { words: 6 * cells }, fpart, fmoved: { words: L.particleStride * p }, fext4: { words: tiered ? FX4_WORDS * L.cap.c : 1 } };
}

/** Bindings lane F's kernels use. */
export const FB = {
  set: { name: "set", kind: "rw", type: "array<atomic<u32>>" },
  dir: { name: "dir", kind: "ro", type: "array<u32>" },
  velRW: { name: "vel", kind: "rw", type: "array<f32>" },
  velRO: { name: "vel", kind: "ro", type: "array<f32>" },
  phiRW: { name: "phi", kind: "rw", type: "array<u32>" },
  phiRO: { name: "phi", kind: "ro", type: "array<u32>" },
  parts: { name: "parts", kind: "rw", type: "array<Particle>" },
  partsRO: { name: "parts", kind: "ro", type: "array<Particle>" },
  fcell: { name: "fcell", kind: "rw", type: "array<atomic<u32>>" },
  /** fcell where the kernel writes none of it (fluidKernel picks it). */
  fcellRO: { name: "fcell", kind: "ro", type: "array<u32>" },
  fdep: { name: "fdep", kind: "rw", type: "array<atomic<u32>>" },
  // The deposit words as plain words: a kernel that alone touches each word (uTransfer's read and clear).
  fdepPlain: { name: "fdep", kind: "rw", type: "array<u32>" },
  fpart: { name: "fpart", kind: "rw", type: "array<u32>" },
  fmoved: { name: "fmoved", kind: "rw", type: "array<Particle>" },
  fext4: { name: "fext4", kind: "rw", type: "array<f32>" },
  /** The sources' static record (brick-source.wgsl.ts BRICK_SOURCE_BINDING: the drops' extrusion edges). */
  src: { name: "src", kind: "ro", type: "array<f32>" },
} as const satisfies Record<string, BrickBinding>;

const u = (n: number) => `${n >>> 0}u`;

/** Lane F's constants: buffer regions and tile words. */
export function fluidConstantsWGSL(F: FluidLayout): string {
  const lines: string[] = [`const FG_TIERED:bool=${F.tiered};`];
  for (const [k, v] of Object.entries(F.fcell)) if (k !== "words") lines.push(`const FC_${k}:u32=${u(v)};`);
  lines.push(`const FP_RANK:u32=${u(F.fpart.RANK)};`);
  for (const [k, v] of Object.entries(FT)) lines.push(`const FT_${k}:u32=${u(v)};`);
  for (const [k, v] of Object.entries(FTF)) lines.push(`const FTF_${k}:u32=${u(v)};`);
  return lines.join("\n");
}

/** A lane F kernel: the ABI prelude (set atomic as nbSet, dir plain), F's constants, its bindings, then the body. */
const FG_FACE_OPEN = "fn fgFaceOpen(c:vec3i,a:u32)->bool{var t=c;t[a]=0;return nbInWallBox(t)&&c[a]>0&&c[a]<NB_WALL_BOX[a];}";

export function fluidKernel(L: BrickLayout, F: FluidLayout, label: string, bindings: readonly BrickBinding[], body: string, entry = label): BrickKernel {
  // A kernel that never writes fcell binds it read-only and reads it with plain loads (FG_FCELL_RO_WGSL): the same
  // words (no lane of the dispatch writes them), without a device atomic a read.
  if (bindings.includes(FB.fcell) && body.includes(FG_FCELL_WGSL) && !/fcStore|&fcell\[/.test(body.split(FG_FCELL_WGSL).join(""))) {
    bindings = bindings.map(b => b === FB.fcell ? FB.fcellRO : b);
    body = body.split(FG_FCELL_WGSL).join(FG_FCELL_RO_WGSL);
  }
  // A module with vertex φ and no staged ring (FG_RING_WGSL) reads every vertex through its owner.
  const ring = (body.includes("fn fgPhiWord") && !body.includes("fn fgRingAt") ? FG_NO_RING_WGSL : "")
    + (body.includes("fn fsCacheOf") && !body.includes("fn fsCacheRawStaged") ? "\nfn fsCacheRawStaged(T:vec3i,a:u32)->f32{return fsCacheRaw(T,a);}" : "")
    + (body.includes("fn fsUnit(") && !body.includes("fn fsUnitAbsent") ? "\nfn fsUnitAbsent(c:vec3i,a:u32)->f32{return fsTapH(c,a,false);}" : "");
  // A detail engine's coupled box planes are open faces (brick-detail.wgsl.ts): they carry the parent's velocity, which
  // the forces relax to and the pressure prescribes (pressure-composite.wgsl.ts pressureDetailFaces).
  if (L.detail && body.includes(FG_FACE_OPEN)) body = body.replace(FG_FACE_OPEN, `fn fgFaceOpen(c:vec3i,a:u32)->bool{var t=c;t[a]=0;let lo=((${L.detail.coupled}u>>(2u*a))&1u)!=0u;let hi=((${L.detail.coupled}u>>(2u*a+1u))&1u)!=0u;return nbInWallBox(t)&&(c[a]>0||(lo&&c[a]==0))&&(c[a]<NB_WALL_BOX[a]||(hi&&c[a]==NB_WALL_BOX[a]));}`);
  return {
    label, entry, bindings,
    // The box hook (brick-abi.md §2, M3 padding): the walls and the cell domain at the box, the storage at the lattice.
    code: resolveBox(L, [brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, dir: { name: "dir", atomic: false } }),
      body.includes("struct Particle") ? "" : "struct Particle{position:vec4f,velocity:vec4f,before:vec4f}", fluidConstantsWGSL(F), declareBindings(bindings), body, ring].join("\n")),
  };
}

const FG_NO_RING_WGSL = "const FG_RING_STRICT:bool=false;\nfn fgRingAt(v:vec3i)->u32{return NB_NONE;}\nfn fgRingPhi(i:u32)->f32{return 0.0;}";

/** A staged ring of vertex φ (seep S2c): the n^3 vertices from `low`, each fgPhi (through its owner, the window's
 * lookups) once, in workgroup memory; fgPhi inside the ring reads it, outside it the owner. The values are fgPhi's
 * own, so results are bit-identical while the kernel writes no PHI before its last ring read. Every lane calls
 * fgRingStage(low, lid, lanes) with the same low (after the window's barrier), then a barrier, before any read.
 * `store` names the storage (default a ring of its own): "slot(j)=x" and "slot(j)" forms for a kernel that lends
 * other workgroup memory before its own use of it.
 *
 * Strict (the default): every PHI read of the kernel past the staging is inside the staged ring, so fgPhi reads the
 * ring alone and a read outside it (or with the ring off) latches FAIL.WINDOW (info 0x52490000, "RI") and reads 0:
 * no owner walk behind every ring read (lTiles 0.58 to 0.46 ms, pLiquid 0.52 to 0.43 on the pour, exact). The
 * staging reads through fgPhiFull. A kernel with reads past its ring (uTransfer's C pages) passes strict false. */
export const FG_RING_WGSL = (n: number, store?: { readonly write: (j: string, x: string) => string; readonly read: (j: string) => string } | null, strict = true) => /* wgsl */`
const FG_RING_STRICT:bool=${strict};
${store ? "" : `var<workgroup> fgRing:array<f32,${n ** 3}>;`}
var<private> fgRingLow:vec3i;
var<private> fgRingOn:bool=false;
fn fgRingAt(v:vec3i)->u32{
 let r=vec3u(v-fgRingLow);
 if(fgRingOn&&all(r<vec3u(${n}u))){return r.x+${n}u*(r.y+${n}u*r.z);}
 return NB_NONE;
}
fn fgRingPhi(i:u32)->f32{return ${store ? store.read("i") : "fgRing[i]"};}
fn fgRingStage(low:vec3i,lid:u32,lanes:u32){
 fgRingOn=false;fgRingLow=low;
 for(var j=lid;j<${n ** 3}u;j+=lanes){let v=low+vec3i(i32(j%${n}u),i32((j/${n}u)%${n}u),i32(j/${n * n}u));let i=fgPhiResolved(v);var x=0.0;if(i!=NB_NONE){x=bitcast<f32>(phi[i]);}else{x=fgPhiFull(v);}${store ? store.write("j", "x") : "fgRing[j]=x"};}
 fgRingOn=true;
}
// The ring's storage is lent back: later reads go to the owners.
fn fgRingClose(){fgRingOn=false;}`;


/** The tier owners (brick-owner.wgsl.ts), prefix fo: the tile part (no pool), φ (PHI; fo1 the PHI1 copy) and the
 * faces of each velocity channel (foU, foP, foS: U, UP, US). Every part is cut from brickOwnerWGSL's own text. */
const FO_ALL = brickOwnerWGSL({ phi: "phi", vel: "vel", prefix: "fo", box: BOX_TOKEN });
const cutOwner = (code: string, from: string, to?: string) => {
  const i = code.indexOf(from), j = to === undefined ? code.length : code.indexOf(to);
  if (i < 0 || j < 0) throw new Error(`fluid-grid: brickOwnerWGSL has no ${i < 0 ? from : to}`);
  return code.slice(i, j);
};
// The owners' directory reads go through fgTileCSlot (a tile's page's C slot), fgTileH (a tile is h-owned) and
// fgBrickSlot (an F slot), which a brick-centred workgroup serves from its staged window (FG_SLOT_WINDOW_WGSL).
const ownerReads = (code: string) => code
  .replace(/nbCSlotOfPage\((\w)>>vec3u\(3u\)\)/g, "fgTileCSlot($1)")
  .replace(/nbFSlotOfBrick\(v>>vec3u\(3u\)\)/g, "fgBrickSlot(v>>vec3u(3u))")
  .replace(/nbFSlotOfCell\(c\)/g, "fgBrickSlot(c>>vec3u(3u))");
const FO_TILE_RAW = ownerReads(cutOwner(FO_ALL, "fn foTileInLattice", "fn foCVertex"));
const FO_OWNER_H = /fn foTileOwnerH\(t:vec3i\)->bool\{[^]*?\n\}\n/;
if (!FO_OWNER_H.test(FO_TILE_RAW)) throw new Error("fluid-grid: brickOwnerWGSL's foTileOwnerH changed shape");
// fgTileHDir is the owners' own foTileOwnerH (renamed); foTileOwnerH reads through fgTileH.
const FO_TILE_WGSL = FO_TILE_RAW.replace(FO_OWNER_H, m => m.replace("fn foTileOwnerH", "fn fgTileHDir") + "fn foTileOwnerH(t:vec3i)->bool{return fgTileH(t);}\n");
const FO_PHI_PART = ownerReads(cutOwner(FO_ALL, "fn foCVertex", "fn foCFace"));
const FO_PHI_WGSL = FO_PHI_PART + FO_PHI_PART.replace(/\bfo(CVertex|FVertex|FPhiBase|VertexPhi)\b/g, "fo1$1")
  .replace(/\bNB_PHI_F_PHI\b/g, "NB_PHI_F_PHI1").replace(/\bNB_PHI_C_PHI\b/g, "NB_PHI_C_PHI1");
// The window's vertex authority (fgWinVertexCoarse: foVertexCoarse with each candidate tile's rule read from the
// window's staged bit, fgWinCoarse) and its φ (fgWinVertexPhi; fgWin1VertexPhi the PHI1 copy), cut from the owners'
// own text: the candidates, their order and the interpolant are the owners'.
const FO_COARSE_RULE = "if(foTileInLattice(t)&&fgTileCSlot(t)!=NB_NONE&&!foTileOwnerH(t)){return t;}";
const cutFunction = (code: string, name: string) => {
  const m = new RegExp(`fn ${name}\\([^]*?\\n\\}\\n`).exec(code);
  if (!m) throw new Error(`fluid-grid: brickOwnerWGSL has no ${name}`);
  return m[0];
};
const FO_WIN_COARSE = cutFunction(FO_TILE_RAW, "foVertexCoarse");
if (!FO_WIN_COARSE.includes(FO_COARSE_RULE)) throw new Error("fluid-grid: brickOwnerWGSL's foVertexCoarse tile rule changed shape");
const FO_WIN_PHI = cutFunction(FO_PHI_PART, "foVertexPhi");
if (!FO_WIN_PHI.includes("fn foVertexPhi(v:vec3i)") || !FO_WIN_PHI.includes("let T=foVertexCoarse(v);")) throw new Error("fluid-grid: brickOwnerWGSL's foVertexPhi changed shape");
const FG_WIN_VERTEX_WGSL = FO_WIN_COARSE.replace("fn foVertexCoarse", "fn fgWinVertexCoarse").replace(FO_COARSE_RULE, "if(fgWinCoarseBit(fgWinTileAt(t))!=0u){return t;}")
  + FO_WIN_PHI.replace("fn foVertexPhi", "fn fgWinVertexPhi").replace("foVertexCoarse(v)", "fgWinVertexCoarse(v)")
  + FO_WIN_PHI.replace("fn foVertexPhi", "fn fgWin1VertexPhi").replace("foVertexCoarse(v)", "fgWinVertexCoarse(v)").replace(/\bfo(CVertex|FVertex)\b/g, "fo1$1");
/** fgWinVertexPhi with its authority T given (the caller's fgWinVertexCoarse: one walk serves the owner test and the
 * read) and the C vertices read through lrCVertex, the caller's stage of them (lResolve): the same candidates, weights
 * and sum as fgWinVertexPhi. */
export const FG_WIN_VERTEX_PHI_AT_WGSL = FO_WIN_PHI.replace("fn foVertexPhi(v:vec3i)", "fn fgWinVertexPhiAt(v:vec3i,T:vec3i)")
  .replace("let T=foVertexCoarse(v);", "").replace("foCVertex(", "lrCVertex(");
if ((FG_WIN_VERTEX_PHI_AT_WGSL.match(/lrCVertex\(/g) ?? []).length !== 1 || FG_WIN_VERTEX_PHI_AT_WGSL.includes("foCVertex(")) throw new Error("fluid-grid: foVertexPhi's C reads changed shape");
const foFaces = (prefix: string, ch: string) => ownerReads(cutOwner(brickOwnerWGSL({ phi: "phi", vel: "vel", prefix, faceF: `NB_VEL_F_${ch}0`, faceC: `NB_VEL_C_${ch}0`, box: BOX_TOKEN }), `fn ${prefix}CFace`)
  .replace(new RegExp(`\\b${prefix}(TileInLattice|TileIndex|TileOwnerH|CellOwnerH)\\b`, "g"), "fo$1"));
const FO_VEL_WGSL = [foFaces("foU", "U"), foFaces("foP", "UP"), foFaces("foS", "US")].join("\n");

/** The directory's F slot of a brick: two dependent loads (page, then child). */
const FG_BRICK_SLOT_WGSL = /* wgsl */`
fn fgBrickSlotOfDir(b:vec3i)->u32{
 if(!nbBrickInRange(b)){return NB_NONE;}
 let s=nbFSlotOfBrick(b);return select(NB_NONE,s,s<NB_SEAM);
}`;

/** Cell addressing (needs `dir` and an fgBrickSlot). */
const FG_ADDRESS_WGSL = /* wgsl */`
// The F slot holding cell (or vertex) c, or NB_NONE.
fn fgSlot(c:vec3i)->u32{
 if(any(c<vec3i(0))){return NB_NONE;}
 return fgBrickSlot(c>>vec3u(3u));
}
fn fgUnit(a:u32)->vec3i{var e=vec3i(0);e[a]=1;return e;}
// Face a of cell c (its lower face) is open: not on a wall, and its cell row inside the lattice across the axis.
fn fgFaceOpen(c:vec3i,a:u32)->bool{var t=c;t[a]=0;return nbInWallBox(t)&&c[a]>0&&c[a]<NB_WALL_BOX[a];}
// A cell's tile within its brick (x + 2y + 4z) and a tile word's index.
fn fgTileOf(c:vec3i)->u32{return u32(((c.x>>2u)&1)|(((c.y>>2u)&1)<<1u)|(((c.z>>2u)&1)<<2u));}
fn fgTileWord(slot:u32,t:u32,w:u32)->u32{return FC_TILE+FT_WORDS*(8u*slot+t)+w;}
fn fgCell(slot:u32,c:vec3i)->u32{return 512u*slot+nbLocal(c);}
// The census's importance trigger (NB_S_TRIG, bit j tile j) of the tile holding cell c; none where no slot holds it.
fn fgTileTrig(c:vec3i)->bool{let s=fgSlot(c);if(s==NB_NONE){return false;}return ((nbSetWord(NB_S_TRIG+s)>>fgTileOf(c))&1u)!=0u;}
// The tile's thin trigger alone (TRIG bit 8 + tile, the old census's w1 bit 17).
fn fgTileThin(c:vec3i)->bool{let s=fgSlot(c);if(s==NB_NONE){return false;}return ((nbSetWord(NB_S_TRIG+s)>>(8u+fgTileOf(c)))&1u)!=0u;}
// The tile word w of the tile holding cell c (NB_NONE where no slot holds it).
fn fgTileAt(c:vec3i,w:u32)->u32{let s=fgSlot(c);if(s==NB_NONE){return NB_NONE;}return fgTileWord(s,fgTileOf(c),w);}
// A cell in one word: each axis takes its window's cell bits (NB_WINDOW_BITS + 5, a page 32 cells), which the layout
// keeps within 32 (BrickLayout). Ten bits an axis wrapped a lattice past 1024 cells (reservoir-seep-vast's 2016: the
// seep's jet past x = 1024 read its nearest surface cell 1024 cells back, 2026-11-18).
fn fgPack(c:vec3i)->u32{let b=NB_WINDOW_BITS+vec3u(5u);return u32(c.x)|(u32(c.y)<<b.x)|(u32(c.z)<<(b.x+b.y));}
fn fgUnpack(k:u32)->vec3i{let b=NB_WINDOW_BITS+vec3u(5u);return vec3i(i32(k&((1u<<b.x)-1u)),i32((k>>b.x)&((1u<<b.y)-1u)),i32(k>>(b.x+b.y)));}
${FO_TILE_WGSL}
// Owned storage (brick-abi.md §2.5): what an F kernel may write. All h (FG_TIERED false) every F word is owned. A
// vertex is F's where no incident tile is 4h-owned (the authority rule); a lower face where either cell beside it
// is h-owned; a cell where its tile is.
fn fgVertexOwned(v:vec3i)->bool{return !FG_TIERED||foVertexOwnerH(v);}
fn fgFaceOwned(c:vec3i,a:u32)->bool{return !FG_TIERED||foCellOwnerH(c)||foCellOwnerH(c-fgUnit(a));}
fn fgCellOwned(c:vec3i)->bool{return !FG_TIERED||foCellOwnerH(c);}
// Tiered: a lattice cell whose tile is 4h-owned (the old engine's cell of a width-4 owner).
fn fgCellCoarse(c:vec3i)->bool{return FG_TIERED&&nbInWallBox(c)&&!foCellOwnerH(c);}
`;

/** Slot lookup and cell addressing (needs `dir`): every lookup reads the directory. */
export const FG_SLOT_WGSL = [FG_BRICK_SLOT_WGSL, /* wgsl */`
fn fgBrickSlot(b:vec3i)->u32{return fgBrickSlotOfDir(b);}
fn fgTileCSlot(t:vec3i)->u32{return nbCSlotOfPage(t>>vec3u(3u));}
fn fgTileH(t:vec3i)->bool{return fgTileHDir(t);}
fn fgWinCovers(v:vec3i)->bool{return false;}
fn fgWinTileAt(t:vec3i)->u32{return NB_NONE;}
fn fgWinCoarseBit(i:u32)->u32{return 0u;}`, FG_ADDRESS_WGSL].join("\n");

/** Slot and owner lookup through a workgroup's window of B^3 bricks: their F slots and their (2B)^3 tiles' C slots
 * and h ownership (the tier owners' every directory read, seep S2c's staged owners). A lookup inside the window is
 * one workgroup read in place of the directory's dependent loads; outside it, or in a workgroup that staged none (a
 * C page of an FC pass), the directory. The directory is read-only in a kernel, so results are bit-identical.
 * Every lane calls fgWindowStageAt(low brick, lid, lanes) (or fgWindowStage(centre, lid, lanes): the 3^3 about
 * centre) with the same bricks, then a barrier, before any lookup. */
export const fgSlotWindowWGSL = (B: number) => [FG_BRICK_SLOT_WGSL, /* wgsl */`
var<workgroup> fgWinSlot:array<u32,${B ** 3}>;
// A tile's page's C slot (NB_NONE as 0x7fffffff) in bits 0..30, its h ownership in bit 31.
var<workgroup> fgWinTile:array<u32,${(2 * B) ** 3}>;
var<private> fgWinLow:vec3i;
var<private> fgWinOn:bool=false;
// Bit j set when window tile j is a 4h authority (foVertexCoarse's tile rule; workgroup memory starts zeroed, and a
// workgroup stages one window).
var<workgroup> fgWinCoarse:array<atomic<u32>,${Math.ceil((2 * B) ** 3 / 32)}>;
fn fgBrickSlot(b:vec3i)->u32{
 let r=vec3u(b-fgWinLow);
 if(fgWinOn&&all(r<vec3u(${B}u))){return fgWinSlot[r.x+${B}u*(r.y+${B}u*r.z)];}
 return fgBrickSlotOfDir(b);
}
fn fgWinTileAt(t:vec3i)->u32{
 let r=vec3u(t-2*fgWinLow);
 if(fgWinOn&&all(r<vec3u(${2 * B}u))){return r.x+${2 * B}u*(r.y+${2 * B}u*r.z);}
 return NB_NONE;
}
fn fgTileCSlot(t:vec3i)->u32{
 let i=fgWinTileAt(t);if(i==NB_NONE){return nbCSlotOfPage(t>>vec3u(3u));}
 let w=fgWinTile[i]&0x7fffffffu;return select(w,NB_NONE,w==0x7fffffffu);
}
fn fgTileH(t:vec3i)->bool{let i=fgWinTileAt(t);if(i==NB_NONE){return fgTileHDir(t);}return (fgWinTile[i]>>31u)!=0u;}
fn fgWindowStageAt(low:vec3i,lid:u32,lanes:u32){
 fgWinOn=false;fgWinLow=low;
 for(var j=lid;j<${B ** 3}u;j+=lanes){fgWinSlot[j]=fgBrickSlotOfDir(low+vec3i(i32(j%${B}u),i32((j/${B}u)%${B}u),i32(j/${B * B}u)));}
 for(var j=lid;j<${(2 * B) ** 3}u;j+=lanes){
  let t=2*low+vec3i(i32(j%${2 * B}u),i32((j/${2 * B}u)%${2 * B}u),i32(j/${4 * B * B}u));
  let cs=nbCSlotOfPage(t>>vec3u(3u));
  let h=fgTileHDir(t);
  fgWinTile[j]=select(cs,0x7fffffffu,cs==NB_NONE)|select(0u,0x80000000u,h);
  // Tile j is a 4h authority (foVertexCoarse's tile rule).
  if(foTileInLattice(t)&&cs!=NB_NONE&&!h){atomicOr(&fgWinCoarse[j>>5u],1u<<(j&31u));}
 }
 fgWinOn=true;
}
fn fgWinCoarseBit(i:u32)->u32{return (atomicLoad(&fgWinCoarse[i>>5u])>>(i&31u))&1u;}
// Vertex v lies in the lattice (foVertexPhi's FAR test: a brick on a lattice face holds vertices beyond it) and its
// candidate authority tiles (foVertexCoarse: v>>2 - 1 .. v>>2) in the window: fgWinVertexPhi reads its authority
// from the staged bits. Read after the barrier that follows the staging.
fn fgWinCovers(v:vec3i)->bool{return fgWinOn&&all(vec3u(v-8*fgWinLow-vec3i(4))<vec3u(${8 * B - 4}u))&&all(v<=NB_LATTICE);}
fn fgWindowStage(centre:vec3i,lid:u32,lanes:u32){fgWindowStageAt(centre-vec3i(1),lid,lanes);}`, FG_ADDRESS_WGSL].join("\n");
/** The 3^3 bricks about a brick. */
export const FG_SLOT_WINDOW_WGSL = fgSlotWindowWGSL(3);

/** fcell reads (needs `fcell`, atomic). */
const FG_FCELL_RW_WGSL = /* wgsl */`
fn fcLoad(i:u32)->u32{return atomicLoad(&fcell[i]);}
fn fcLoadF(i:u32)->f32{return bitcast<f32>(atomicLoad(&fcell[i]));}
fn fcStore(i:u32,v:u32){atomicStore(&fcell[i],v);}
fn fcStoreF(i:u32,v:f32){atomicStore(&fcell[i],bitcast<u32>(v));}
`;
const FG_FCELL_READS_WGSL = /* wgsl */`
// A tile word of cell c's tile, or the default where no slot holds it.
fn fgTileLoad(c:vec3i,w:u32,otherwise:u32)->u32{let i=fgTileAt(c,w);if(i==NB_NONE){return otherwise;}return fcLoad(i);}
fn fgTileFlags(c:vec3i)->u32{return fgTileLoad(c,FT_FLAGS,0u);}
fn fgTileDist(c:vec3i)->u32{return fgTileLoad(c,FT_DIST,0xffffffffu);}
// umTileSupport bit 1 (the extension's support: lDilate's frame-plan dilation, FT_DIST bit 12); none where no slot holds c.
fn fgTileSupported(c:vec3i)->bool{return ((fgTileLoad(c,FT_DIST,0u)>>12u)&1u)!=0u;}
// The packed Chebyshev tile distances (4 bits each, 15 for none within reach): 0 crossing, 1 air seed, 2 occupied.
fn fgDist(word:u32,k:u32)->u32{return (word>>(4u*k))&15u;}
// The redistance reach bits (old NB_BAND_REACH): bit 1 within 2 tiles of a crossing tile, bit 2 within 3.
fn fgReach1(c:vec3i)->bool{return fgDist(fgTileDist(c),0u)<=2u;}
fn fgReach2(c:vec3i)->bool{return fgDist(fgTileDist(c),0u)<=3u;}
fn fgTileHeat(c:vec3i)->f32{let i=fgTileAt(c,FT_T);if(i==NB_NONE){return 0.0;}return fcLoadF(i);}
`;
export const FG_FCELL_WGSL = FG_FCELL_RW_WGSL + FG_FCELL_READS_WGSL;
/** The same reads from a read-only fcell (FB.fcellRO: plain loads, no stores). */
const FG_FCELL_RO_WGSL = /* wgsl */`
fn fcLoad(i:u32)->u32{return fcell[i];}
fn fcLoadF(i:u32)->f32{return bitcast<f32>(fcell[i]);}
` + FG_FCELL_READS_WGSL;

/** Vertex φ (needs `phi`).
 *
 * Resolved φ (tiered): an F slot's PHI word holds every vertex of the slot at its owner-resolved value, its own
 * vertices as F writes them, the others (an incident tile 4h-owned: the tile's interpolant) as lResolve
 * (levelset-kernels.wgsl.ts) writes them after each writer of what they depend on (C PHI and the tile ownership).
 * So PHI at a vertex an F slot holds, inside the lattice, is a plain load equal to foVertexPhi's; any other vertex
 * reads through its owner. FG_PHI_RESOLVED off (FG_PHI_OWNER_WGSL) reads every vertex through its owner outside a
 * staged ring (lResolve itself; uAdvect, whose ring alone pays). The schedule's rule is checked in
 * fluid-staging.test.ts: no resolved read between a C PHI writer or the set update and the next lResolve. */
export const FG_PHI_WGSL = FO_PHI_WGSL + FG_WIN_VERTEX_WGSL + /* wgsl */`
const FG_PHI_RESOLVED:bool=true;
// The PHI word of vertex v where an F slot holds it inside the lattice (tiered), else NB_NONE.
fn fgPhiResolved(v:vec3i)->u32{
 if(FG_TIERED&&all(v>=vec3i(0))&&all(v<=NB_LATTICE)){let s=fgSlot(v);if(s!=NB_NONE){return NB_PHI_F_PHI+NB_STRIDE*s+nbLocal(v);}}
 return NB_NONE;
}
// Tiered: PHI and PHI1 read through the vertex authority (foVertexPhi: F's vertex, or the 4h tile's interpolant),
// PHI at a vertex an F slot holds as its resolved word.
fn fgPhiWord(base:u32,v:vec3i)->f32{
 if(base==NB_PHI_F_PHI){let r=fgRingAt(v);if(r!=NB_NONE){return fgRingPhi(r);}if(FG_RING_STRICT){nbFail(NB_FAIL_WINDOW,0x52490000u);return 0.0;}}
 if(FG_PHI_RESOLVED&&base==NB_PHI_F_PHI){let i=fgPhiResolved(v);if(i!=NB_NONE){return bitcast<f32>(phi[i]);}}
 if(FG_TIERED){let w=fgWinCovers(v);if(base==NB_PHI_F_PHI){if(w){return fgWinVertexPhi(v);}return foVertexPhi(v);}if(base==NB_PHI_F_PHI1){if(w){return fgWin1VertexPhi(v);}return fo1VertexPhi(v);}}
 let s=fgSlot(v);if(s==NB_NONE){return NB_PHI_AIR_C;}return bitcast<f32>(phi[base+NB_STRIDE*s+nbLocal(v)]);}
fn fgPhi(v:vec3i)->f32{return fgPhiWord(NB_PHI_F_PHI,v);}
// fgPhi outside any ring: the ring's own staging reads (a strict ring's fgPhi reads the ring alone).
fn fgPhiFull(v:vec3i)->f32{
 if(FG_PHI_RESOLVED){let i=fgPhiResolved(v);if(i!=NB_NONE){return bitcast<f32>(phi[i]);}}
 if(FG_TIERED){if(fgWinCovers(v)){return fgWinVertexPhi(v);}return foVertexPhi(v);}
 let s=fgSlot(v);if(s==NB_NONE){return NB_PHI_AIR_C;}return bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*s+nbLocal(v)]);}
// The eight vertex φ of cell c in corner order x + 2y + 4z.
fn fgCornersOf(base:u32,c:vec3i)->array<f32,8>{
 var v:array<f32,8>;
 for(var k=0u;k<8u;k++){v[k]=fgPhiWord(base,c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}
 return v;
}
fn fgCorners(c:vec3i)->array<f32,8>{return fgCornersOf(NB_PHI_F_PHI,c);}
// The cell-centre φ: the mean of the eight corners (brick-abi.md §6.1, the operator's rule).
fn fgCentrePhi(c:vec3i)->f32{let v=fgCorners(c);var s=0.0;for(var k=0u;k<8u;k++){s+=v[k];}return 0.125*s;}
// A liquid cell (the authority's phase, -0.001 of a cell, uniform-mixed-pressure-surface): centre φ under it.
fn fgLiquidCell(c:vec3i)->bool{if(!nbInWallBox(c)){return false;}return fgCentrePhi(c)< -0.001;}
// The point p (cells) held in the box.
fn fgHeld(p:vec3f)->vec3f{return clamp(p,vec3f(0.0),vec3f(NB_WALL_BOX));}
// Trilinear φ at p (cells), the cell holding p (held in the box): umSampleVertex.
fn fgPhiAtOf(base:u32,p:vec3f)->f32{
 let q=fgHeld(p);let c=min(vec3i(floor(q)),NB_WALL_BOX-vec3i(1));let t=q-vec3f(c);let v=fgCornersOf(base,c);
 let x00=mix(v[0],v[1],t.x);let x10=mix(v[2],v[3],t.x);let x01=mix(v[4],v[5],t.x);let x11=mix(v[6],v[7],t.x);
 return mix(mix(x00,x10,t.y),mix(x01,x11,t.y),t.z);
}
fn fgPhiAt(p:vec3f)->f32{return fgPhiAtOf(NB_PHI_F_PHI,p);}
// A 4h tile as one cell of width 4 (the old engine's width-4 owner): its eight C corners in corner order, its centre
// φ (their mean, fgCentrePhi's sum: the old 4h centre cache) and its phase (the authority's -0.001 of a cell).
fn fgTileCorners4(t:vec3i)->array<f32,8>{
 var v:array<f32,8>;
 for(var k=0u;k<8u;k++){v[k]=foCVertex(t+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}
 return v;
}
fn fgTileCentre4(t:vec3i)->f32{let v=fgTileCorners4(t);var s=0.0;for(var k=0u;k<8u;k++){s+=v[k];}return 0.125*s;}
// A cell's phase at its owner's width: its own (fgLiquidCell) in an h tile, its 4h tile's in a 4h one.
fn fgLiquidOwner(c:vec3i)->bool{if(fgCellCoarse(c)){return fgTileCentre4(c>>vec3u(2u))< -0.001;}return fgLiquidCell(c);}
`;
/** FG_PHI_WGSL with every read outside a staged ring through the owner (FG_PHI_RESOLVED off). */
export const FG_PHI_OWNER_WGSL = (() => {
  const from = "const FG_PHI_RESOLVED:bool=true;";
  if (FG_PHI_WGSL.split(from).length !== 2) throw new Error("fluid-grid: FG_PHI_WGSL has no single FG_PHI_RESOLVED");
  return FG_PHI_WGSL.replace(from, "const FG_PHI_RESOLVED:bool=false;");
})();

/** Face velocity reads (needs `vel`): base is a channel's first word (NB_VEL_F_U0, _UP0 or _US0). */
export const FG_VEL_WGSL = FO_VEL_WGSL + /* wgsl */`
// Tiered: the owner's face (foFaceU: F's h face, or the 4h tile's linear sample between its plane faces).
fn fgFace(base:u32,c:vec3i,a:u32)->f32{
 if(!fgFaceOpen(c,a)){return 0.0;}
 if(FG_TIERED){if(base==NB_VEL_F_U0){return foUFaceU(c,a);}if(base==NB_VEL_F_UP0){return foPFaceU(c,a);}if(base==NB_VEL_F_US0){return foSFaceU(c,a);}}
 let s=fgSlot(c);if(s==NB_NONE){return 0.0;}
 return vel[base+NB_STRIDE*(s+a*NB_FCAP)+nbLocal(c)];
}
// The MAC trilinear sample of axis a at p (cells; umSampleVelocity): face a of cell c sits at c + 0.5 off the axis,
// so the sample index is p - 0.5 off the axis, held to the box (0..N along a, 0..N-1 across).
fn fgVelAxis(base:u32,p:vec3f,a:u32)->f32{
 var s=p-vec3f(0.5);s[a]=p[a];
 var hi=vec3f(NB_WALL_BOX-vec3i(1));hi[a]=f32(NB_WALL_BOX[a]);
 s=clamp(s,vec3f(0.0),hi);
 let i0=vec3i(floor(s));let f=s-vec3f(i0);
 var v:array<f32,8>;
 for(var k=0u;k<8u;k++){let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));v[k]=fgFace(base,i0+bit,a);}
 // umVelocityLerp8's operand order (nbfFaceLerp8).
 let x0=(1.0-f.x)*v[0]+f.x*v[1];let x1=(1.0-f.x)*v[2]+f.x*v[3];
 let x2=(1.0-f.x)*v[4]+f.x*v[5];let x3=(1.0-f.x)*v[6]+f.x*v[7];
 return (1.0-f.z)*((1.0-f.y)*x0+f.y*x1)+f.z*((1.0-f.y)*x2+f.y*x3);
}
fn fgVel(base:u32,p:vec3f)->vec3f{return vec3f(fgVelAxis(base,p,0u),fgVelAxis(base,p,1u),fgVelAxis(base,p,2u));}
`;

/**
 * The momentum sampler (needs `vel`, `fcell`, FG_FCELL_WGSL and FG_VEL_WGSL): the old engine's h/4h sample of the
 * extended velocity US, umUnitSample (uniform-cached-momentum.wgsl.ts l.93) with umUnitWeight, the cached unit
 * interpolant and umUnitCoarse, which the momentum trace and the surface trace (umFastVelocity) share.
 * fgSampleVel(p, lo, hi): components lo..hi-1 at p (cells). All h (FG_TIERED false) it is fgVel(US) exactly.
 * Tiered, it takes one interpolant a sample, not the old blend: the fine one where the fine weight is over 0, else
 * the coarse one. In the fringe (0 < weight < 1) the fine taps away from the h owners are already the 4h cache's
 * trilinear (fsTapH), so the fine interpolant there is the coarse one up to the 4h cells' kinks. The blend cost
 * uAdvect 1.5 ms (2026-11-16, 10.31 -> 8.79 ms); the dam's D1-D8 against the canonical refs are unchanged
 * (lane F's progress log).
 * - The fine weight (umUnitWeight, umVelocitySamplingWeightsAt, uniform-mixed-velocity-sampling.wgsl.ts l.290):
 *   0 where the sample's tile lacks support bit 0 (an occupied tile within fineReach = 2 tiles: the frame plan's
 *   seed and dilation, FT_DIST's occupied distance); 1 in an h tile; else max over the h tiles of the 26 about it
 *   of max(0, 1 - d/2), d the Chebyshev distance (cells) from the held point to the h tile's box. A tile with an
 *   h neighbour lies in an F brick (the census's ring), so a tile no slot holds has weight 0.
 * - The fine interpolant: the MAC trilinear of h taps (fgVelAxis's clamp and order); a tap is F's face where either
 *   cell beside it is h-owned, else the 4h cache's trilinear at the face (umCompiledVelocityTap,
 *   uniform-compiled-velocity-taps.wgsl.ts: eighth-cell fractions, zero weights skipped, umVertexSum8's order).
 *   Taps all in h tiles (umUnitTaps) read F directly.
 * - The coarse interpolant (umSampleVelocity4In): the trilinear of the 4h cache on the tile lattice.
 * - The 4h cache of a tile plane (UniformMixedMomentumCache.cacheTexel): the C face between two 4h tiles, else the
 *   mean of its 16 F subfaces (y outer, x inner, over 16). Taps past the lattice take the edge tile's (SOLID); a
 *   wall plane reads 0 (F's closed-face rule); storage absent reads 0 (AIR).
 */
export const FG_SAMPLE_WGSL = /* wgsl */`
fn fsSupport(t:vec3i)->bool{return fgDist(fgTileDist(4*t),2u)<=2u;}
fn fsLerp8(v:array<f32,8>,f:vec3f)->f32{
 let x0=(1.0-f.x)*v[0]+f.x*v[1];let x1=(1.0-f.x)*v[2]+f.x*v[3];
 let x2=(1.0-f.x)*v[4]+f.x*v[5];let x3=(1.0-f.x)*v[6]+f.x*v[7];
 return (1.0-f.z)*((1.0-f.y)*x0+f.y*x1)+f.z*((1.0-f.y)*x2+f.y*x3);
}
fn fsBit(k:u32)->vec3i{return vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));}
// The 4h cache of tile T's lower plane along a (US), resolved through the owners (uCache's source).
fn fsCacheRaw(T:vec3i,a:u32)->f32{
 var hi=(NB_WALL_BOX>>vec3u(2u))-vec3i(1);hi[a]+=1;
 let t=clamp(T,vec3i(0),hi);let c=4*t;
 if(!fgFaceOpen(c,a)){return 0.0;}
 if(!foCellOwnerH(c)&&!foCellOwnerH(c-fgUnit(a))){return foSCFace(t,a);}
 let u=(a+1u)%3u;let v=(a+2u)%3u;var sum=0.0;
 for(var y=0;y<4;y++){for(var x=0;x<4;x++){var q=c;q[u]+=x;q[v]+=y;sum+=foSFFace(q,a);}}
 return sum/16.0;
}
// The same from uCache's CACHE4 (a page no slot holds reads 0, as the owners' readers do).
fn fsCache(T:vec3i,a:u32)->f32{
 var hi=(NB_WALL_BOX>>vec3u(2u))-vec3i(1);hi[a]+=1;
 let t=clamp(T,vec3i(0),hi);
 if(!fgFaceOpen(4*t,a)){return 0.0;}
 let cs=fgTileCSlot(t);if(cs==NB_NONE){return 0.0;}
 return fcLoadF(FC_CACHE4+512u*(3u*cs+a)+foTileIndex(t));
}
fn fsCacheOf(T:vec3i,a:u32,raw:bool)->f32{if(raw){return fsCacheRawStaged(T,a);}return fsCache(T,a);}
// Corner k of a trilinear has weight 0 where it lies on the far side of an axis whose fraction is 0: its tap is not
// read. fsLerp8 then adds 0 * 0 where it added 0 * v (v finite), equal up to the sign of a zero sum.
fn fsZeroWeight(z:vec3<bool>,k:u32)->bool{return any(z&(fsBit(k)==vec3i(1)));}
// The coarse interpolant at p. Its taps' tiles (fsCache's, held) lie between the held b and b + 1; where those share
// a page, one C slot serves all eight (fsFine's one lookup): eight lookups set uAdvect's register peak (Metal's cap
// 704 threads, two 256-lane workgroups a core, against 768 with this).
fn fsCoarse(p:vec3f,a:u32)->f32{
 var off=vec3f(0.5);off[a]=0.0;
 let q=p/4.0-off;let b=vec3i(floor(q));let f=q-vec3f(b);var v:array<f32,8>;let z=f==vec3f(0.0);
 var hi=(NB_WALL_BOX>>vec3u(2u))-vec3i(1);hi[a]+=1;
 let t0=clamp(b,vec3i(0),hi);let t1=clamp(b+vec3i(1),vec3i(0),hi);
 if(all((t0>>vec3u(3u))==(t1>>vec3u(3u)))){
  let cs=fgTileCSlot(t0);
  for(var k=0u;k<8u;k++){
   if(fsZeroWeight(z,k)){v[k]=0.0;continue;}
   let t=clamp(b+fsBit(k),vec3i(0),hi);
   if(!fgFaceOpen(4*t,a)||cs==NB_NONE){v[k]=0.0;continue;}
   v[k]=fcLoadF(FC_CACHE4+512u*(3u*cs+a)+foTileIndex(t));
  }
 }else{for(var k=0u;k<8u;k++){if(fsZeroWeight(z,k)){v[k]=0.0;continue;}v[k]=fsCache(b+fsBit(k),a);}}
 return fsLerp8(v,f);
}
// The 4h cache's trilinear at the h face (c, a): in eighths of a tile, 2c - 3 across and 2c along.
fn fsTap4(c:vec3i,a:u32,raw:bool)->f32{
 var q8=2*c-vec3i(3);q8[a]=2*c[a];
 let b=q8>>vec3u(3u);let f=vec3f(q8-8*b)/8.0;var t:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let corner=fsBit(k);let w=select(vec3f(1.0)-f,f,corner!=vec3i(0));let weight=w.x*w.y*w.z;
  if(weight>0.0){t[k]=weight*fsCacheOf(b+corner,a,raw);}
 }
 return ((t[0]+t[5])+(t[1]+t[4]))+((t[2]+t[7])+(t[3]+t[6]));
}
fn fsTapH(c:vec3i,a:u32,raw:bool)->f32{
 if(!fgFaceOpen(c,a)){return 0.0;}
 if(foCellOwnerH(c)||foCellOwnerH(c-fgUnit(a))){return foSFFace(c,a);}
 return fsTap4(c,a,raw);
}
// The h tap from uCache's UNIT; a brick no slot holds takes fsUnitAbsent. Only fsFine reads it, where the fine
// weight is over 0: p's tile has an F slot and an h owner of the opening tiers (HMASK, the tiers the head's setMark
// saw) lies within 2 cells, so every tap is within 3 cells of an h owner. Residency keeps the owners' bricks by
// the h ring tiered (R3: the 27 tiles about each, 4 cells; lane E's read table: the sampler's UNIT taps, 3), so no
// tap's brick lacks a slot (under the earlier flow + 9: 0 hits in uAdvect, lAdvect, lcAdvect, pMove and pSeed over the
// tiered 300 and the all-h 200 dam, 2026-11-16).
// fsUnitAbsent defaults to the 4h fallback (fsTapH); uAdvect takes 0, which is 1.2 ms of its time (a fallback, or
// any side effect there, keeps the compiler from simplifying the tap).
fn fsUnit(c:vec3i,a:u32)->f32{
 if(!fgFaceOpen(c,a)){return 0.0;}
 let s=fgSlot(c);if(s==NB_NONE){return fsUnitAbsent(c,a);}
 return fcLoadF(FC_UNIT+729u*(3u*s+a)+fsLocal9(c&vec3i(7)));
}
// UNIT's face r (0 .. 8 on each axis) of a slot.
fn fsLocal9(r:vec3i)->u32{return u32(r.x+9*r.y+81*r.z);}
// The fast path's residency audit (a no-op; a probe substitutes a counter): a tap c the base slot's apron served.
fn fsUnitAudit(i0:vec3i,c:vec3i,a:u32){}
// The fine interpolant at p. Its 2 x 2 x 2 taps lie in i0's slot's 9^3 UNIT (the apron holds the +side
// neighbours' taps, and 0 at a closed face), so one slot lookup serves all eight; i0's brick with no slot (the
// lattice's upper face) takes the tap-by-tap path.
fn fsFine(p:vec3f,a:u32)->f32{
 var s=p-vec3f(0.5);s[a]=p[a];
 var hi=vec3f(NB_WALL_BOX-vec3i(1));hi[a]=f32(NB_WALL_BOX[a]);
 s=clamp(s,vec3f(0.0),hi);
 let i0=vec3i(floor(s));let f=s-vec3f(i0);
 var v:array<f32,8>;let z=f==vec3f(0.0);
 let s0=fgSlot(i0);
 if(s0==NB_NONE){for(var k=0u;k<8u;k++){if(fsZeroWeight(z,k)){v[k]=0.0;continue;}v[k]=fsUnit(i0+fsBit(k),a);}}
 else{
  let base=FC_UNIT+729u*(3u*s0+a)+fsLocal9(i0&vec3i(7));
  for(var k=0u;k<8u;k++){if(fsZeroWeight(z,k)){v[k]=0.0;continue;}v[k]=fcLoadF(base+(k&1u)+9u*((k>>1u)&1u)+81u*(k>>2u));fsUnitAudit(i0,i0+fsBit(k),a);}
 }
 return fsLerp8(v,f);
}
// A tile's HMASK word (uCache), 0 where no F slot holds it.
fn fsHMask(t:vec3i)->u32{let c=4*t;let s=fgSlot(c);if(s==NB_NONE){return 0u;}return fcLoad(FC_HMASK+8u*s+fgTileOf(c));}
// The offsets k (x, y, z = k % 3, (k / 3) % 3, k / 9, each d + 1) with d on axis j: FS_REACH[3j + d + 1].
const FS_REACH:array<u32,9>=array<u32,9>(0x1249249u,0x2492492u,0x4924924u,0x1c0e07u,0xe07038u,0x70381c0u,0x1ffu,0x3fe00u,0x7fc0000u);
// The fine weight is over 0 (the tiered sampler's switch): p's tile has the support and an h owner n of the opening
// tiers (HMASK) whose tile box is within 2 cells of p, max(0, 1 - max(delta) / 2) > 0, which is max(delta) < 2,
// every axis's delta under 2. O(1): the owners within reach are the product of each axis's offsets with delta < 2.
fn fsFineAny(p:vec3f)->bool{
 let q=fgHeld(p);let t=min(vec3i(floor(q)),NB_WALL_BOX-vec3i(1))>>vec3u(2u);
 let m=fsHMask(t);if(m==0u){return false;}
 if(!fsSupport(t)){return false;}
 if((m&(1u<<13u))!=0u){return true;}
 var reach=0x7ffffffu;
 for(var j=0u;j<3u;j++){
  var r=0u;
  for(var d=-1;d<=1;d++){
   let origin=f32(t[j]+d)*4.0;let delta=max(0.0,max(origin-q[j],q[j]-origin-4.0));
   if(delta<2.0){r|=FS_REACH[3u*j+u32(d+1)];}
  }
  reach&=r;
 }
 return (m&reach)!=0u;
}
fn fgSampleVel(p:vec3f,lo:u32,hi:u32)->vec3f{
 var v=vec3f(0.0);
 if(!FG_TIERED){for(var a=lo;a<hi;a++){v[a]=fgVelAxis(NB_VEL_F_US0,p,a);}return v;}
 let fine=fsFineAny(p);
 for(var a=lo;a<hi;a++){if(fine){v[a]=fsFine(p,a);}else{v[a]=fsCoarse(p,a);}}
 return v;
}
`;

/** The particle depth (needs fcell and phi): particleDepthIn. Bank A's trilinear in a reach-1 tile, else φ pushed
 * to six cells (nbfFarDepth). */
export const FG_DEPTH_WGSL = /* wgsl */`
fn fgDepthVertex(v:vec3i)->f32{let s=fgSlot(v);if(s==NB_NONE){return NB_PHI_AIR;}return fcLoadF(FC_DEPTH+fgCell(s,v));}
fn fgDepth(p:vec3f)->f32{
 let c=vec3i(floor(p));
 if(!fgReach1(c)){let far=fgPhiAt(p);return select(max(far,6.0),min(far,-6.0),far<0.0);}
 let f=p-vec3f(c);var value=0.0;
 for(var k=0u;k<8u;k++){
  let corner=vec3u(k&1u,(k>>1u)&1u,k>>2u);let w=select(1.0-f,f,corner!=vec3u(0));
  value+=w.x*w.y*w.z*fgDepthVertex(c+vec3i(corner));
 }
 return value;
}
`;

/** A cell's run in the bank pScatter wrote (needs set and fcell): [start, end), empty where the slot's bin is. */
export const FG_RUN_WGSL = /* wgsl */`
// The run is held to the slot's bin, [BIN_START, min(BIN_START + BIN_COUNT, NB_PCAP)]: the identity when nothing
// overflowed, and a bound on every run loop (pSeparation, the held-mass scan, pMergeRun, vcTransferFace) when a
// FC_RUN word is not this rebin's.
fn fgRun(c:vec3i)->vec2u{
 if(!nbInLattice(c)){return vec2u(0u);}
 let s=fgSlot(c);if(s==NB_NONE){return vec2u(0u);}
 let count=nbSetWord(NB_S_BIN_COUNT+s);if(count==0u){return vec2u(0u);}
 let bs=min(nbSetWord(NB_S_BIN_START+s),NB_PCAP);let be=bs+min(count,NB_PCAP-bs);
 let l=nbLocal(c);let start=clamp(fcLoad(FC_RUN+512u*s+l),bs,be);
 var end=be;if(l<511u){end=clamp(fcLoad(FC_RUN+512u*s+l+1u),bs,be);}
 return vec2u(start,end);
}
`;
