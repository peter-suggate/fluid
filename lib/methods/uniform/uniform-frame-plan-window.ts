import { UNIFORM_TILE_BITS_WORDS } from "./uniform-block-tables";
import { UNIFORM_STAGE_CERTIFICATE } from "./uniform-stage-grids";
import { UNIFORM_WINDOW_PAGES } from "./uniform-window-address";

/** The frame plan's support dilation and certificate tail (method param
 * framePlanPass, fixed for a solver's life).
 * - "chain": the per-axis launches over the resident tile keys: dilate0..2
 *   for the support; tileError, spread0..2, coarseBits, coarseBlocks and
 *   certify for the certificate (8 + 8 launches a frame on paddleboats).
 * - "window": a workgroup per half page (8x8x4 tiles) stages a tile window
 *   in threadgroup memory and runs the separable chain there, writing only
 *   its core. supportWindow replaces dilate0..2 (halo 3 tiles: exact for
 *   fineReach + shellReach <= 3, else the encode takes the chain);
 *   certifySpread replaces tileError, spread0/1, coarseBits and coarseBlocks
 *   (halo 10 tiles in x and y: exact for a certificate radius <= 10, else
 *   every reach saturates, which certifies no tile regular); certifyTiles
 *   replaces spread2 and certify. Unknown values throw. */
export const FRAME_PLAN_PASSES = ["chain", "window"] as const;
export type FramePlanPass = typeof FRAME_PLAN_PASSES[number];
export const framePlanPass = (value: unknown): FramePlanPass => {
  if (value === undefined) return "chain";
  if (typeof value === "string" && (FRAME_PLAN_PASSES as readonly string[]).includes(value)) return value as FramePlanPass;
  throw new Error(`Uniform frame plan framePlanPass must be one of ${FRAME_PLAN_PASSES.join(", ")}: ${JSON.stringify(value)}`);
};

/** Lanes a window group, its core (tiles), the two halos and the most
 * workgroups a window launch strides the items (two per resident block) with. */
export const FRAME_PLAN_WINDOW = { lanes: 256, core: [8, 8, 4] as const, supportHalo: 3, certificateHalo: 10, maxGroups: 512 } as const;
/** The support window: the core and supportHalo on every side. */
export const FRAME_PLAN_SUPPORT_EXTENT = FRAME_PLAN_WINDOW.core.map(n => n + 2 * FRAME_PLAN_WINDOW.supportHalo) as [number, number, number];
/** The certificate window: the core and certificateHalo in x and y only (z is certifyTiles's, from global memory). */
export const FRAME_PLAN_CERTIFICATE_EXTENT = [FRAME_PLAN_WINDOW.core[0] + 2 * FRAME_PLAN_WINDOW.certificateHalo, FRAME_PLAN_WINDOW.core[1] + 2 * FRAME_PLAN_WINDOW.certificateHalo, FRAME_PLAN_WINDOW.core[2]] as [number, number, number];
/** A window launch's workgroups for a pool of `blocks` blocks: a group an
 * item (half page) up to maxGroups, which then stride the items. */
export const framePlanWindowGroups = (blocks: number): number => Math.max(1, Math.min(FRAME_PLAN_WINDOW.maxGroups, 2 * blocks));
/** Every packed reach field (umPackReach) at UM_REACH_SATURATED. */
export const FRAME_PLAN_REACH_SATURATED = 0x3fffffff;
/** The packed reach's even fields (bits 0, 10, 20) and the guard bit above each. */
export const FRAME_PLAN_EVEN_FIELDS = 0x01f07c1f, FRAME_PLAN_FIELD_GUARDS = 0x02008020;
/** pwMaxReach on the host: the field-wise maximum of two packed reaches, bit 31 dropped. */
export function framePlanMaxReach(a: number, b: number): number {
  const E = FRAME_PLAN_EVEN_FIELDS, G = FRAME_PLAN_FIELD_GUARDS;
  const fields = (x: number, y: number) => { const m = ((((((x | G) >>> 0) - y) >>> 0) & G) >>> 5) * 31; return ((x & m) | (y & ~m & E)) >>> 0; };
  return (fields(a & E, b & E) | (fields((a >>> 5) & E, (b >>> 5) & E) << 5)) >>> 0;
}

/** The three window entries. The including module (uniform-mixed-frame-plan)
 * provides the topology (umTileKey, umKeyStored, umTileKind, umTileLive,
 * umTileWidth, umTileCoord, umBlockPage, atomic umSupport), the live lists,
 * policy, speeds, coarse, the reach packing, umCertificateRadius,
 * tileErrorOf, coarseAny, certifyCounts and certifyBases.
 * surfaceOnly: the support dilation's shell rule (dilate's). */
export function uniformFramePlanWindowWGSL(surfaceOnly: boolean): string {
  const [sx, sy, sz] = FRAME_PLAN_SUPPORT_EXTENT, hs = FRAME_PLAN_WINDOW.supportHalo;
  const [cx, cy, cz] = FRAME_PLAN_CERTIFICATE_EXTENT, hc = FRAME_PLAN_WINDOW.certificateHalo;
  const W = UNIFORM_TILE_BITS_WORDS, C = UNIFORM_STAGE_CERTIFICATE;
  return /* wgsl */ `
// ---- framePlanPass "window": a workgroup per half page (item 2b+half: block
// ulResidentBlock(b), tile z 4*half..4*half+3), a lane per core tile. Each
// group stages its window, runs the chain in threadgroup memory and writes
// only its core. A staged word: bit 31 set where the key is stored.
const PW_STORED:u32=0x80000000u;
const PW_SATURATED:u32=${FRAME_PLAN_REACH_SATURATED >>> 0}u;
const PW_LAST:i32=${8 * UNIFORM_WINDOW_PAGES - 1};
var<workgroup> pwSupport:array<u32,${sx * sy * sz}>;
var<workgroup> pwReach:array<u32,${cx * cy * cz}>;
var<workgroup> pwBits:array<atomic<u32>,8>;
fn pwCore(item:u32)->vec3i{return vec3i(umBlockPage(ulResidentBlock(item>>1u))<<vec3u(3u))+vec3i(0,0,4*i32(item&1u));}
// One dilate axis over count outputs from first, stride apart: the chain's
// dilate<axis>Tile of each, over the staged window. A source that is not
// stored adds nothing; an output is written only where stored.
fn pwDilate(first:u32,stride:u32,count:u32,fine:i32,reach:i32,mode:u32){
 var out:array<u32,8>;
 for(var c=0u;c<count;c++){
  let at=i32(first+c*stride);var flags=0u;
  for(var d=-reach;d<=reach;d++){
   let source=pwSupport[u32(at+d*i32(stride))];if((source&PW_STORED)==0u){continue;}
   ${surfaceOnly ? "flags|=select((source&1u)<<1u,source&3u,abs(d)<=fine);" : "flags|=source&select(2u,3u,abs(d)<=fine);"}
  }
  out[c]=flags|mode;
 }
 for(var c=0u;c<count;c++){let at=first+c*stride;if((pwSupport[at]&PW_STORED)!=0u){pwSupport[at]=PW_STORED|out[c];}}
}
// The support planes' last (umSupport[3T]) from the seed flags (umSupport[0,T)):
// dilate0..2 in one launch. Planes 1 and 2 are not written.
@compute @workgroup_size(${FRAME_PLAN_WINDOW.lanes}) fn supportWindow(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let fine=i32(policy.settings.x);let reach=fine+i32(policy.settings.y);let mode=policy.settings.z;
 for(var item=group.x;item<2u*ulResidentCount();item+=groups.x){
  let low=pwCore(item)-vec3i(${hs});
  for(var i=lane;i<${sx * sy * sz}u;i+=${FRAME_PLAN_WINDOW.lanes}u){
   let k=umTileKey(low+vec3i(vec3u(i%${sx}u,(i/${sx}u)%${sy}u,i/${sx * sy}u)));var w=0u;
   if(umKeyStored(k)){w=PW_STORED|atomicLoad(&umSupport[k]);}
   pwSupport[i]=w;
  }
  workgroupBarrier();
  // x: every window row (y, z), its core outputs.
  if(lane<${sy * sz}u){pwDilate(${sx}u*lane+${hs}u,1u,8u,fine,reach,mode);}
  workgroupBarrier();
  // y: the core x columns of every window z.
  if(lane<${8 * sz}u){pwDilate(${hs}u+(lane&7u)+${sx}u*(${hs}u+${sy}u*(lane>>3u)),${sx}u,8u,fine,reach,mode);}
  workgroupBarrier();
  // z: the core columns.
  if(lane<64u){pwDilate(${hs}u+(lane&7u)+${sx}u*(${hs}u+(lane>>3u))+${sx * sy * hs}u,${sx * sy}u,4u,fine,reach,mode);}
  workgroupBarrier();
  {
   let tile=512u*ulResidentBlock(item>>1u)+256u*(item&1u)+lane;
   let at=${hs}u+(lane&7u)+${sx}u*(${hs}u+((lane>>3u)&7u)+${sy}u*(${hs}u+(lane>>6u)));
   if(umTileLive(tile)){atomicStore(&umSupport[3u*UM_TILES+tile],pwSupport[at]&3u);}
  }
  workgroupBarrier();
 }
}
// One spread axis over the eight core outputs from first, stride apart: the
// chain's spread<axis>Tile of each. Staged words hold a stored key's reach,
// UM_REACH_SATURATED in every field where an AIR page holds the tile, and 0
// where nothing is stored or the tile is past the lattice: the field-wise
// maximum is the chain's. An output is written only where stored (an unstored
// tile keeps its staged word, as the chain rereads it). The maximum is taken
// packed (pwMaxReach): unpacking to two vec3u held certifySpread to 576
// threads a group, packed it compiles at 1024.
// The field-wise maximum of two packed reaches (umPackReach) in two words:
// the even fields (bits 0, 10, 20) and the odd ones shifted onto them, each
// 5-bit field with a free guard bit above it. Bit 31 (PW_STORED) is dropped.
const PW_EVEN:u32=${FRAME_PLAN_EVEN_FIELDS}u;const PW_GUARD:u32=${FRAME_PLAN_FIELD_GUARDS}u;
fn pwMaxFields(a:u32,b:u32)->u32{let m=((((a|PW_GUARD)-b)&PW_GUARD)>>5u)*31u;return (a&m)|(b&~m&PW_EVEN);}
fn pwMaxReach(a:u32,b:u32)->u32{return pwMaxFields(a&PW_EVEN,b&PW_EVEN)|(pwMaxFields((a>>5u)&PW_EVEN,(b>>5u)&PW_EVEN)<<5u);}
fn pwSpread(first:u32,stride:u32,reach:i32){
 var out:array<u32,8>;
 for(var c=0u;c<8u;c++){
  let at=i32(first+c*stride);var w=0u;
  for(var d=-reach;d<=reach;d++){w=pwMaxReach(w,pwReach[u32(at+d*i32(stride))]);}
  out[c]=w;
 }
 for(var c=0u;c<8u;c++){let at=first+c*stride;if((pwReach[at]&PW_STORED)!=0u){pwReach[at]=PW_STORED|out[c];}}
}
// tileError, spread0 and spread1 (into umSupport[T,2T), free under this pass:
// no support plane but the last is read), coarseBits and coarseBlocks in one
// launch. A radius past the halo saturates every reach.
@compute @workgroup_size(${FRAME_PLAN_WINDOW.lanes}) fn certifySpread(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let radius=umCertificateRadius(bitcast<f32>(atomicLoad(&umSupport[4u*UM_TILES])));
 let reach=i32(min(radius,${hc}u));let wide=radius>${hc}u;
 for(var item=group.x;item<2u*ulResidentCount();item+=groups.x){
  let b=ulResidentBlock(item>>1u);let tile=512u*b+256u*(item&1u)+lane;
  tileErrorOf(tile);
  if(lane<8u){atomicStore(&pwBits[lane],0u);}
  let low=pwCore(item)-vec3i(${hc},${hc},0);
  for(var i=lane;i<${cx * cy * cz}u;i+=${FRAME_PLAN_WINDOW.lanes}u){
   let p=low+vec3i(vec3u(i%${cx}u,(i/${cx}u)%${cy}u,i/${cx * cy}u));var w=0u;
   if(all(p>=vec3i(0))&&all(p<=vec3i(PW_LAST))){
    let k=umTileKey(p);
    if(umKeyStored(k)){w=PW_STORED|speeds[k];}else if(umTileKind(p)==UM_AIR){w=PW_SATURATED;}
   }
   pwReach[i]=w;
  }
  workgroupBarrier();
  // coarseBits: a bit per stored 4h tile, by key.
  if(umTileLive(tile)&&umTileWidth(tile)!=1u){atomicOr(&pwBits[lane>>5u],1u<<(lane&31u));}
  if(lane<${cy * cz}u){pwSpread(${cx}u*lane+${hc}u,1u,reach);}
  workgroupBarrier();
  if(lane<32u){pwSpread(${hc}u+(lane&7u)+${cx}u*(${hc}u+${cy}u*(lane>>3u)),${cx}u,reach);}
  workgroupBarrier();
  if(umTileLive(tile)){
   let at=${hc}u+(lane&7u)+${cx}u*(${hc}u+((lane>>3u)&7u)+${cy}u*(lane>>6u));
   atomicStore(&umSupport[UM_TILES+tile],select(pwReach[at]&PW_SATURATED,PW_SATURATED,wide));
  }
  // The half's eight tile-bit words. coarseBlocks' slab and neighbour
  // summaries are written all-set: coarseAny then tests every page of a box
  // on its own words, the same answer without the shortcut.
  if(lane<8u){coarse[${W}u*b+8u*(item&1u)+lane]=atomicLoad(&pwBits[lane]);}
  if(lane==0u&&(item&1u)==0u){coarse[${W}u*b+16u]=0xffu;coarse[${W}u*b+17u]=0x7ffffffu;}
  workgroupBarrier();
 }
}
// spread2 (umSupport[T,2T) along z into the reach) and certify in one launch.
@compute @workgroup_size(${FRAME_PLAN_WINDOW.lanes}) fn certifyTiles(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let radius=i32(umCertificateRadius(bitcast<f32>(atomicLoad(&umSupport[4u*UM_TILES]))));
 for(var item=group.x;item<2u*ulResidentCount();item+=groups.x){
  let tile=512u*ulResidentBlock(item>>1u)+256u*(item&1u)+lane;let live=umTileLive(tile);let fine=live&&umTileWidth(tile)==1u;
  if(lane<2u){atomicStore(&certifyCounts[lane],0u);}workgroupBarrier();
  var reach=0u;
  if(live){
   let p=vec3i(umTileCoord(tile));var plus=vec3u(0);var minus=vec3u(0);
   let first=clamp(p.z-radius,0,PW_LAST);let last=clamp(p.z+radius,0,PW_LAST);
   for(var z=first;z<=last;z++){
    let r=vec3i(p.x,p.y,z);let k=umTileKey(r);
    if(!umKeyStored(k)){
     if(umTileKind(r)==UM_AIR){plus=vec3u(UM_REACH_SATURATED);minus=plus;}
     continue;
    }
    let value=atomicLoad(&umSupport[UM_TILES+k]);plus=max(plus,umReachPlus(value));minus=max(minus,umReachMinus(value));
   }
   reach=umPackReach(plus,minus);
  }
  var list=1u;var local=0u;
  if(fine){
   let plus=umReachPlus(reach);let minus=umReachMinus(reach);
   let bounded=all(plus<vec3u(UM_REACH_SATURATED))&&all(minus<vec3u(UM_REACH_SATURATED));
   let p=vec3i(umTileCoord(tile));
   let regular=bounded&&!coarseAny(tile>>9u,p-vec3i(minus),p+vec3i(plus));
   list=select(2u,1u,regular);local=atomicAdd(&certifyCounts[list-1u],1u);
   speeds[UM_TILES+tile]=reach|(select(select(${C.coarseInReach}u,${C.saturated}u,!bounded),${C.regular}u,regular)<<30u);
  }else if(live){speeds[UM_TILES+tile]=reach;}
  workgroupBarrier();
  if(lane<2u){let count=atomicLoad(&certifyCounts[lane]);if(count>0u){certifyBases[lane]=atomicAdd(&umSupport[4u*UM_TILES+lane+1u],count);}}
  workgroupBarrier();
  if(fine){atomicStore(&umSupport[(4u+list)*UM_TILES+16u+certifyBases[list-1u]+local],tile);}
 }
}
`;
}
