/**
 * LANE F OWNS THIS FILE. The brick pool as lane F's kernels read it (brick-abi.md §1, §2.3), and lane F's own
 * buffers (brick-abi.md §6.2: lane-owned, allocated zeroed).
 *
 * Box rule, in the ABI's units (cells, φ in h, m/s):
 * - A cell or vertex outside the bricks a slot can hold (any coordinate below 0, or past the skirt) has no slot.
 * - A brick no slot holds is air (§2.3: a non-resident brick is air outside the reach): φ reads NB_PHI_FAR (the old
 *   vertex field's empty value, the lattice's largest extent in h; brick-abi.md §2.3), the particle depth reads
 *   NB_PHI_AIR, and a face reads 0.
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
 * - `fdep`: p2g's fixed-point deposits, 2 words (momentum i32, weight u32) per face of each axis.
 * - `fpart`, per record of a bank: RANK (2 words: the destination rank of the record and of its split half, or
 *   NONE) and MOVED (a third bank: the move's output, which pScatter files into the new bank).
 */
import { BRICK_CELLS, type BrickLayout, brickAbiWGSL } from "./brick-abi";
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
  readonly fcell: { readonly CNT: number; readonly RUN: number; readonly NEAR: number; readonly NEAR2: number; readonly KEY: number; readonly HOT: number; readonly DEPTH: number; readonly MASK: number; readonly TILE: number; readonly SLOT: number; readonly BOUND: number; readonly words: number };
  readonly fdep: { readonly words: number };
  readonly fpart: { readonly RANK: number; readonly words: number };
  readonly fmoved: { readonly words: number };
  /** The 4h extension's state (tiered only; one word all h): per C slot, SEEDS then STATE, each 512 tiles x 3 axes
   * x (value, prior, distance), FX4_WORDS a slot. */
  readonly fext4: { readonly words: number };
}
/** Words a C slot holds in fext4: seeds and state, 512 tiles x 3 axes x 3 words each. */
export const FX4_WORDS = 2 * 512 * 3 * 3;

export function fluidLayout(L: BrickLayout, options: { readonly tiered?: boolean } = {}): FluidLayout {
  const f = L.cap.f, cells = f * BRICK_CELLS, p = L.cap.particles;
  let at = 0;
  const take = (n: number) => { const o = at; at += n; return o; };
  const CNT = take(cells), RUN = take(cells), NEAR = take(cells), NEAR2 = take(cells), KEY = take(cells), HOT = take(cells), DEPTH = take(cells);
  const MASK = take(16 * f), TILE = take(8 * FT.WORDS * f), SLOT = take(f), BOUND = take(f);
  const fcell = { CNT, RUN, NEAR, NEAR2, KEY, HOT, DEPTH, MASK, TILE, SLOT, BOUND, words: at };
  const fpart = { RANK: 0, words: 2 * p };
  const tiered = options.tiered === true;
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
  fdep: { name: "fdep", kind: "rw", type: "array<atomic<u32>>" },
  fpart: { name: "fpart", kind: "rw", type: "array<u32>" },
  fmoved: { name: "fmoved", kind: "rw", type: "array<Particle>" },
  fext4: { name: "fext4", kind: "rw", type: "array<f32>" },
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
export function fluidKernel(L: BrickLayout, F: FluidLayout, label: string, bindings: readonly BrickBinding[], body: string, entry = label): BrickKernel {
  return {
    label, entry, bindings,
    code: [brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, dir: { name: "dir", atomic: false } }),
      body.includes("struct Particle") ? "" : "struct Particle{position:vec4f,velocity:vec4f,before:vec4f}", fluidConstantsWGSL(F), declareBindings(bindings), body].join("\n"),
  };
}


/** The tier owners (brick-owner.wgsl.ts), prefix fo: the tile part (no pool), φ (PHI; fo1 the PHI1 copy) and the
 * faces of each velocity channel (foU, foP, foS: U, UP, US). Every part is cut from brickOwnerWGSL's own text. */
const FO_ALL = brickOwnerWGSL({ phi: "phi", vel: "vel", prefix: "fo" });
const cutOwner = (code: string, from: string, to?: string) => {
  const i = code.indexOf(from), j = to === undefined ? code.length : code.indexOf(to);
  if (i < 0 || j < 0) throw new Error(`fluid-grid: brickOwnerWGSL has no ${i < 0 ? from : to}`);
  return code.slice(i, j);
};
const FO_TILE_WGSL = cutOwner(FO_ALL, "fn foTileInLattice", "fn foCVertex");
const FO_PHI_PART = cutOwner(FO_ALL, "fn foCVertex", "fn foCFace");
const FO_PHI_WGSL = FO_PHI_PART + FO_PHI_PART.replace(/\bfo(CVertex|FVertex|VertexPhi)\b/g, "fo1$1")
  .replace(/\bNB_PHI_F_PHI\b/g, "NB_PHI_F_PHI1").replace(/\bNB_PHI_C_PHI\b/g, "NB_PHI_C_PHI1");
const foFaces = (prefix: string, ch: string) => cutOwner(brickOwnerWGSL({ phi: "phi", vel: "vel", prefix, faceF: `NB_VEL_F_${ch}0`, faceC: `NB_VEL_C_${ch}0` }), `fn ${prefix}CFace`)
  .replace(new RegExp(`\\b${prefix}(TileInLattice|TileIndex|TileOwnerH|CellOwnerH)\\b`, "g"), "fo$1");
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
fn fgFaceOpen(c:vec3i,a:u32)->bool{var t=c;t[a]=0;return nbInLattice(t)&&c[a]>0&&c[a]<NB_LATTICE[a];}
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
fn fgPack(c:vec3i)->u32{return u32(c.x)|(u32(c.y)<<10u)|(u32(c.z)<<20u);}
fn fgUnpack(k:u32)->vec3i{return vec3i(i32(k&1023u),i32((k>>10u)&1023u),i32(k>>20u));}
${FO_TILE_WGSL}
// Owned storage (brick-abi.md §2.5): what an F kernel may write. All h (FG_TIERED false) every F word is owned. A
// vertex is F's where no incident tile is 4h-owned (the authority rule); a lower face where either cell beside it
// is h-owned; a cell where its tile is.
fn fgVertexOwned(v:vec3i)->bool{return !FG_TIERED||foVertexOwnerH(v);}
fn fgFaceOwned(c:vec3i,a:u32)->bool{return !FG_TIERED||foCellOwnerH(c)||foCellOwnerH(c-fgUnit(a));}
fn fgCellOwned(c:vec3i)->bool{return !FG_TIERED||foCellOwnerH(c);}
// Tiered: a lattice cell whose tile is 4h-owned (the old engine's cell of a width-4 owner).
fn fgCellCoarse(c:vec3i)->bool{return FG_TIERED&&nbInLattice(c)&&!foCellOwnerH(c);}
`;

/** Slot lookup and cell addressing (needs `dir`): every lookup reads the directory. */
export const FG_SLOT_WGSL = [FG_BRICK_SLOT_WGSL, "fn fgBrickSlot(b:vec3i)->u32{return fgBrickSlotOfDir(b);}", FG_ADDRESS_WGSL].join("\n");

/** Slot lookup through a workgroup's window of the 3^3 bricks about one brick (needs `dir`). A lookup inside the
 * window is one workgroup read in place of the directory's two dependent loads; outside it, the directory. The
 * slot is the same either way (the directory is read-only in a kernel), so results are bit-identical.
 * Every lane calls fgWindowStage(centre, lid) with the same centre, then a barrier, before any lookup. */
export const FG_SLOT_WINDOW_WGSL = [FG_BRICK_SLOT_WGSL, /* wgsl */`
var<workgroup> fgWinSlot:array<u32,27>;
var<private> fgWinLow:vec3i;
fn fgBrickSlot(b:vec3i)->u32{
 let r=vec3u(b-fgWinLow);
 if(all(r<vec3u(3u))){return fgWinSlot[r.x+3u*(r.y+3u*r.z)];}
 return fgBrickSlotOfDir(b);
}
fn fgWindowStage(centre:vec3i,lid:u32){
 fgWinLow=centre-vec3i(1);
 if(lid<27u){fgWinSlot[lid]=fgBrickSlotOfDir(fgWinLow+vec3i(i32(lid%3u),i32((lid/3u)%3u),i32(lid/9u)));}
}`, FG_ADDRESS_WGSL].join("\n");

/** fcell reads (needs `fcell`, atomic). */
export const FG_FCELL_WGSL = /* wgsl */`
fn fcLoad(i:u32)->u32{return atomicLoad(&fcell[i]);}
fn fcLoadF(i:u32)->f32{return bitcast<f32>(atomicLoad(&fcell[i]));}
fn fcStore(i:u32,v:u32){atomicStore(&fcell[i],v);}
fn fcStoreF(i:u32,v:f32){atomicStore(&fcell[i],bitcast<u32>(v));}
// A tile word of cell c's tile, or the default where no slot holds it.
fn fgTileLoad(c:vec3i,w:u32,otherwise:u32)->u32{let i=fgTileAt(c,w);if(i==NB_NONE){return otherwise;}return fcLoad(i);}
fn fgTileFlags(c:vec3i)->u32{return fgTileLoad(c,FT_FLAGS,0u);}
fn fgTileDist(c:vec3i)->u32{return fgTileLoad(c,FT_DIST,0xffffffffu);}
// The packed Chebyshev tile distances (4 bits each, 15 for none within reach): 0 crossing, 1 air seed, 2 occupied.
fn fgDist(word:u32,k:u32)->u32{return (word>>(4u*k))&15u;}
// The redistance reach bits (old NB_BAND_REACH): bit 1 within 2 tiles of a crossing tile, bit 2 within 3.
fn fgReach1(c:vec3i)->bool{return fgDist(fgTileDist(c),0u)<=2u;}
fn fgReach2(c:vec3i)->bool{return fgDist(fgTileDist(c),0u)<=3u;}
fn fgTileHeat(c:vec3i)->f32{let i=fgTileAt(c,FT_T);if(i==NB_NONE){return 0.0;}return fcLoadF(i);}
`;

/** Vertex φ (needs `phi`). */
export const FG_PHI_WGSL = FO_PHI_WGSL + /* wgsl */`
// Tiered: PHI and PHI1 read through the vertex authority (foVertexPhi: F's vertex, or the 4h tile's interpolant).
fn fgPhiWord(base:u32,v:vec3i)->f32{
 if(FG_TIERED){if(base==NB_PHI_F_PHI){return foVertexPhi(v);}if(base==NB_PHI_F_PHI1){return fo1VertexPhi(v);}}
 let s=fgSlot(v);if(s==NB_NONE){return NB_PHI_FAR;}return bitcast<f32>(phi[base+NB_STRIDE*s+nbLocal(v)]);}
fn fgPhi(v:vec3i)->f32{return fgPhiWord(NB_PHI_F_PHI,v);}
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
fn fgLiquidCell(c:vec3i)->bool{if(!nbInLattice(c)){return false;}return fgCentrePhi(c)< -0.001;}
// The point p (cells) held in the box.
fn fgHeld(p:vec3f)->vec3f{return clamp(p,vec3f(0.0),vec3f(NB_LATTICE));}
// Trilinear φ at p (cells), the cell holding p (held in the box): umSampleVertex.
fn fgPhiAtOf(base:u32,p:vec3f)->f32{
 let q=fgHeld(p);let c=min(vec3i(floor(q)),NB_LATTICE-vec3i(1));let t=q-vec3f(c);let v=fgCornersOf(base,c);
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
 var hi=vec3f(NB_LATTICE-vec3i(1));hi[a]=f32(NB_LATTICE[a]);
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
 * The momentum sampler (needs `vel`, `fcell`, FG_FCELL_WGSL and FG_VEL_WGSL): the old engine's blended h/4h sample
 * of the extended velocity US, umUnitSample (uniform-cached-momentum.wgsl.ts l.93) with umUnitWeight, the cached
 * unit interpolant and umUnitCoarse, which the momentum trace and the surface trace (umFastVelocity) share.
 * fgSampleVel(p, lo, hi): components lo..hi-1 at p (cells). All h (FG_TIERED false) it is fgVel(US) exactly.
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
// The 4h cache of tile T's lower plane along a (US).
fn fsCache(T:vec3i,a:u32)->f32{
 var hi=(NB_LATTICE>>vec3u(2u))-vec3i(1);hi[a]+=1;
 let t=clamp(T,vec3i(0),hi);let c=4*t;
 if(!fgFaceOpen(c,a)){return 0.0;}
 if(!foCellOwnerH(c)&&!foCellOwnerH(c-fgUnit(a))){return foSCFace(t,a);}
 let u=(a+1u)%3u;let v=(a+2u)%3u;var sum=0.0;
 for(var y=0;y<4;y++){for(var x=0;x<4;x++){var q=c;q[u]+=x;q[v]+=y;sum+=foSFFace(q,a);}}
 return sum/16.0;
}
// The coarse interpolant at p.
fn fsCoarse(p:vec3f,a:u32)->f32{
 var off=vec3f(0.5);off[a]=0.0;
 let q=p/4.0-off;let b=vec3i(floor(q));let f=q-vec3f(b);var v:array<f32,8>;
 for(var k=0u;k<8u;k++){v[k]=fsCache(b+fsBit(k),a);}
 return fsLerp8(v,f);
}
// The 4h cache's trilinear at the h face (c, a): in eighths of a tile, 2c - 3 across and 2c along.
fn fsTap4(c:vec3i,a:u32)->f32{
 var q8=2*c-vec3i(3);q8[a]=2*c[a];
 let b=q8>>vec3u(3u);let f=vec3f(q8-8*b)/8.0;var t:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let corner=fsBit(k);let w=select(vec3f(1.0)-f,f,corner!=vec3i(0));let weight=w.x*w.y*w.z;
  if(weight>0.0){t[k]=weight*fsCache(b+corner,a);}
 }
 return ((t[0]+t[5])+(t[1]+t[4]))+((t[2]+t[7])+(t[3]+t[6]));
}
fn fsTapH(c:vec3i,a:u32)->f32{
 if(!fgFaceOpen(c,a)){return 0.0;}
 if(foCellOwnerH(c)||foCellOwnerH(c-fgUnit(a))){return foSFFace(c,a);}
 return fsTap4(c,a);
}
// The fine interpolant at p.
fn fsFine(p:vec3f,a:u32)->f32{
 var s=p-vec3f(0.5);s[a]=p[a];
 var hi=vec3f(NB_LATTICE-vec3i(1));hi[a]=f32(NB_LATTICE[a]);
 s=clamp(s,vec3f(0.0),hi);
 let i0=vec3i(floor(s));let f=s-vec3f(i0);
 let low=i0>>vec3u(2u);let high=vec3i(ceil(s))>>vec3u(2u);var fine=true;
 for(var k=0u;k<8u;k++){if(!foTileOwnerH(select(low,high,fsBit(k)!=vec3i(0)))){fine=false;}}
 var v:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let c=i0+fsBit(k);
  if(fine){v[k]=select(0.0,foSFFace(c,a),fgFaceOpen(c,a));}else{v[k]=fsTapH(c,a);}
 }
 return fsLerp8(v,f);
}
fn fsFineWeight(p:vec3f)->f32{
 let q=fgHeld(p);let t=min(vec3i(floor(q)),NB_LATTICE-vec3i(1))>>vec3u(2u);
 if(!fsSupport(t)){return 0.0;}
 if(foTileOwnerH(t)){return 1.0;}
 var fine=0.0;
 for(var k=0u;k<27u;k++){
  if(k==13u){continue;}
  let n=t+vec3i(i32(k%3u),i32((k/3u)%3u),i32(k/9u))-vec3i(1);
  if(!foTileOwnerH(n)){continue;}
  let origin=vec3f(n)*4.0;let delta=max(vec3f(0.0),max(origin-q,q-origin-vec3f(4.0)));
  fine=max(fine,max(0.0,1.0-max(delta.x,max(delta.y,delta.z))/2.0));
 }
 return fine;
}
fn fgSampleVel(p:vec3f,lo:u32,hi:u32)->vec3f{
 var v=vec3f(0.0);
 if(!FG_TIERED){for(var a=lo;a<hi;a++){v[a]=fgVelAxis(NB_VEL_F_US0,p,a);}return v;}
 let fine=fsFineWeight(p);let coarse=max(0.0,1.0-fine);
 for(var a=lo;a<hi;a++){
  var value=0.0;
  if(fine>0.0){value+=fine*fsFine(p,a);}
  if(coarse>0.0){value+=coarse*fsCoarse(p,a);}
  v[a]=value;
 }
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
fn fgRun(c:vec3i)->vec2u{
 if(!nbInLattice(c)){return vec2u(0u);}
 let s=fgSlot(c);if(s==NB_NONE){return vec2u(0u);}
 let count=nbSetWord(NB_S_BIN_COUNT+s);if(count==0u){return vec2u(0u);}
 let l=nbLocal(c);let start=fcLoad(FC_RUN+512u*s+l);
 var end=nbSetWord(NB_S_BIN_START+s)+count;if(l<511u){end=fcLoad(FC_RUN+512u*s+l+1u);}
 return vec2u(start,end);
}
`;
