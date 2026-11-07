/** ubaStage: the shared halo and seam staging of the brick-resident step
 * (docs/plans/brick-resident-step-2026-11-08.md §3.2). A fused brick pass
 * (one workgroup a brick) stages its 27-neighbour record once, then fills a
 * threadgroup window of (8+2k)³ owned indices per channel, each value by one
 * of four cases:
 * - the brick's own core or a neighbour's core, through the staged record
 *   (a live slot of the level, its tile live): the channel's core fn;
 * - the seam rule, where the record says the neighbour is stored but not
 *   live at this level (UBA_SEAM | block, or a live slot whose tile mask bit
 *   is clear):
 *   - level 0 (h), "prolong": from the 4h owner, by the channel's layout:
 *     - cell: the 4h cell's value plus its measured slope across the 4h
 *       neighbours (central where both are known, one-sided at an edge,
 *       none without): exact for any linear field, the rule d7fa24db gave
 *       the seam pressure;
 *     - face: linear between the 4h owner's two faces on the axis,
 *       constant across it, so the 4x4 h subfaces on a 4h face plane carry
 *       the 4h face's value and their V-weighted sum is its flux;
 *     - vertex: trilinear in the 4h cell's corners (the hanging vertex
 *       rule: on a 4h face bilinear in its corners, on an edge linear);
 *   - level 2 (4h), "restrict", where the 4h cell is h-owned: the V-weighted
 *     mean of its h values (cell: the 64 cells; face: the 4x4 subfaces on
 *     the face plane, the reflux of a coarse face; vertex: the coincident h
 *     vertex);
 *   - or a channel's own WGSL fn;
 * - outside every stored page (UB_NONE: SOLID or AIR by the page table): the
 *   channel's outside fn.
 * Stages never see a seam: they read the window.
 *
 * Addresses: a window index c is relative to the brick's first cell, in the
 * level's cells, -k .. 8+k an axis; p = origin + c is the owned index (a
 * cell, its +axis face, or lattice vertex p+1: the arena's owned-high
 * layouts). Every fn named here is WGSL the including module defines:
 * - core: fn(word:u32,base:u32,l:vec3u)->f32, the value of local owned
 *   index l (0..7) of neighbour slot word (level 0: a live slot; level 2: a
 *   stored block), base the word the resolver gave that neighbour; or
 *   {pool} for an arena pool channel;
 * - resolve: fn(word:u32,origin:vec3i)->u32, run once a neighbour (27
 *   lanes) at record time, origin its first cell: the address a core fn
 *   needs (a paged store's texel origin, say), so a staged value is one load;
 * - coarse (prolong): fn(c:vec3i)->vec2f, the 4h owned index c's (value,
 *   known>0);
 * - fine, weight (restrict): fn(p:vec3i)->f32 at h owned index p;
 * - fineOwned (level 2): fn(c:vec3i)->bool, whether 4h cell c is h-owned
 *   (default: its tile is live in the arena's level 0 table);
 * - outside: fn(p:vec3i)->f32 (default 0).
 *
 * Every entry WGSL calls (ubaRecord_K, ubaStage_K_<channel>, ubaStage_K)
 * holds barriers: call them from uniform control flow. Windows are separate
 * workgroup arrays, so a pass that stages only some channels pays only
 * their memory; the total is checked against UBA_STAGE_WORDS (32 KiB). */
import type { UniformArenaLayout, UniformBrickArena } from "./uniform-brick-arena";

/** Workgroup words a kernel's windows may take: 32 KiB, the M1 Max threadgroup limit. */
export const UBA_STAGE_WORDS=8192;
export type UbaStageLevel=0|2;
export interface UbaStageChannel{
 /** A WGSL identifier: the channel's accessors are ubaGet_K_<name> etc. */
 name:string;
 layout:UniformArenaLayout;
 /** The face component (layout "face"). */
 axis?:0|1|2;
 /** Owned indices staged around the brick, 0..8. */
 halo:number;
 core:string|{pool:string;component?:number;bank?:0|1};
 /** "prolong" (level 0), "restrict" (level 2), or a WGSL fn(p:vec3i,word:u32)->f32. */
 seam:"prolong"|"restrict"|{fn:string};
 coarse?:string;
 fine?:string;
 weight?:string;
 outside?:string;
}
export interface UbaStageOptions{
 level:UbaStageLevel;
 workgroup:number;
 channels:readonly UbaStageChannel[];
 resolve?:string;
 fineOwned?:string;
}
const IDENT=/^[A-Za-z_][A-Za-z0-9_]*$/;
/** The level's record base in ubaLevelRecord. */
const RECORD_LEVEL={0:"0u",2:"2u"} as const;

function prolongWGSL(fn:string,ch:UbaStageChannel):string{
 const coarse=ch.coarse;if(!coarse)throw new Error(`ubaStage ${ch.name}: prolong needs a coarse fn`);
 if(ch.layout==="cell")return /* wgsl */`
fn ${fn}(p:vec3i)->f32{
 let C=p>>vec3u(2u);let c0=${coarse}(C);if(c0.y<=0.0){return c0.x;}
 let x=(vec3f(p&vec3i(3))+vec3f(0.5))*0.25-vec3f(0.5);var v=c0.x;
 for(var a=0u;a<3u;a++){
  var e=vec3i(0);e[a]=1;let lo=${coarse}(C-e);let hi=${coarse}(C+e);var s=0.0;
  if(lo.y>0.0&&hi.y>0.0){s=0.5*(hi.x-lo.x);}else if(hi.y>0.0){s=hi.x-c0.x;}else if(lo.y>0.0){s=c0.x-lo.x;}
  v+=s*x[a];
 }
 return v;
}`;
 if(ch.layout==="face"){const a=ch.axis??0;return /* wgsl */`
fn ${fn}(p:vec3i)->f32{
 let C=p>>vec3u(2u);let hi=${coarse}(C);let q=(p[${a}]&3)+1;if(q==4){return hi.x;}
 var e=vec3i(0);e[${a}]=1;let lo=${coarse}(C-e);let w=f32(q)*0.25;
 if(lo.y>0.0&&hi.y>0.0){return mix(lo.x,hi.x,w);}if(lo.y>0.0&&hi.y<=0.0){return lo.x;}return hi.x;
}`;}
 return /* wgsl */`
fn ${fn}(p:vec3i)->f32{
 let q=p+vec3i(1);let b=(q>>vec3u(2u))-vec3i(1);let f=vec3f(q&vec3i(3))*0.25;var sum=0.0;var total=0.0;
 for(var k=0u;k<8u;k++){
  let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));let w3=select(vec3f(1.0)-f,f,bit==vec3i(1));let w=w3.x*w3.y*w3.z;
  if(w<=0.0){continue;}let v=${coarse}(b+bit);if(v.y>0.0){sum+=w*v.x;total+=w;}
 }
 return select(0.0,sum/total,total>0.0);
}`;
}
function restrictWGSL(fn:string,ch:UbaStageChannel):string{
 const fine=ch.fine;if(!fine)throw new Error(`ubaStage ${ch.name}: restrict needs a fine fn`);
 const weight=ch.weight?`${ch.weight}(q)`:"1.0";
 const mean=(count:number,at:string)=>/* wgsl */`
 var sum=0.0;var total=0.0;var plain=0.0;
 for(var i=0u;i<${count}u;i++){let q=${at};let w=${weight};let v=${fine}(q);sum+=w*v;total+=w;plain+=v;}
 return select(plain/${count}.0,sum/total,total>0.0);`;
 if(ch.layout==="cell")return `
fn ${fn}(c:vec3i)->f32{${mean(64,"4*c+vec3i(i32(i&3u),i32((i>>2u)&3u),i32(i>>4u))")}
}`;
 if(ch.layout==="face"){const a=ch.axis??0,u=(a+1)%3,v=(a+2)%3;return `
fn ${fn}(c:vec3i)->f32{${mean(16,`${fn}_part(c,i)`)}
}
fn ${fn}_part(c:vec3i,i:u32)->vec3i{var q=4*c;q[${a}]+=3;q[${u}]+=i32(i&3u);q[${v}]+=i32(i>>2u);return q;}`;}
 return `
fn ${fn}(c:vec3i)->f32{return ${fine}(4*c+vec3i(3));}`;
}

/** The staging helper of one kernel K (one call site). */
export function ubaStageWGSL(arena:UniformBrickArena,kernel:string,options:UbaStageOptions):string{
 const {level,workgroup,channels}=options,K=kernel;
 if(!IDENT.test(K))throw new Error(`ubaStage: kernel ${K} is not an identifier`);
 if(level!==0&&level!==2)throw new Error(`ubaStage: level ${level} (0 or 2)`);
 if(!Number.isInteger(workgroup)||workgroup<32)throw new Error(`ubaStage ${K}: a workgroup of ${workgroup} lanes stages no record (at least 32)`);
 const names=new Set<string>();let words=0;
 for(const ch of channels){
  if(!IDENT.test(ch.name)||names.has(ch.name))throw new Error(`ubaStage ${K}: channel ${ch.name} is not a fresh identifier`);
  names.add(ch.name);
  if(!Number.isInteger(ch.halo)||ch.halo<0||ch.halo>8)throw new Error(`ubaStage ${K}/${ch.name}: halo ${ch.halo} outside 0..8`);
  if(ch.seam==="prolong"&&level!==0)throw new Error(`ubaStage ${K}/${ch.name}: prolong stages h values (level 0)`);
  if(ch.seam==="restrict"&&level!==2)throw new Error(`ubaStage ${K}/${ch.name}: restrict stages 4h values (level 2)`);
  if(ch.layout==="face"&&ch.axis===undefined)throw new Error(`ubaStage ${K}/${ch.name}: a face channel names its axis`);
  words+=(8+2*ch.halo)**3;
 }
 if(words+128>UBA_STAGE_WORDS)throw new Error(`ubaStage ${K}: ${words} window words past the ${UBA_STAGE_WORDS}-word threadgroup budget (march z-slabs, or stage fewer channels a stage)`);
 const fineOwned=options.fineOwned??`ubaFineOwned_${K}`;
 const lines:string[]=[/* wgsl */`
var<workgroup> ubaR_${K}:array<u32,27>;
var<workgroup> ubaM_${K}:array<u32,27>;
var<workgroup> ubaB_${K}:array<u32,27>;
var<workgroup> ubaO_${K}:vec3i;
// The staged record: neighbour n's word (offset x+3y+9z of (-1..1)³), its live tile mask, its resolved base; the brick's first cell.
fn ubaNeighbour_${K}(n:u32)->u32{return ubaR_${K}[n];}
fn ubaMask_${K}(n:u32)->u32{return ubaM_${K}[n];}
fn ubaBase_${K}(n:u32)->u32{return ubaB_${K}[n];}
fn ubaFirst_${K}()->vec3i{return ubaO_${K};}
fn ubaRecord_${K}(slot:u32,lane:u32){
 workgroupBarrier();
 if(lane==27u){ubaO_${K}=vec3i(ubaLevelOrigin(${RECORD_LEVEL[level]},ubaLevelRecord(${RECORD_LEVEL[level]},slot,29u)));}
 workgroupBarrier();
 if(lane<27u){
  let w=ubaLevelRecord(${RECORD_LEVEL[level]},slot,lane);ubaR_${K}[lane]=w;var m=0u;
  ${level===0?"if(w<UBA_SEAM){m=ubaRecordWord(w,28u);}":"if(w!=UB_NONE_ID){m=0xffu;}"}
  ubaM_${K}[lane]=m;
  let o=ubaO_${K}+8*(vec3i(vec3u(lane%3u,(lane/3u)%3u,lane/9u))-vec3i(1));
  ubaB_${K}[lane]=${options.resolve?`${options.resolve}(w,o)`:"0u"};
 }
 workgroupBarrier();
}`];
 if(level===2&&!options.fineOwned)lines.push(/* wgsl */`
// A 4h cell is h-owned where its tile is live in the level 0 table.
fn ubaFineOwned_${K}(c:vec3i)->bool{
 let id=ubaBrickAt(4*c);if(id==UB_NONE_ID){return false;}
 let s=ubaSlot(id);if(s==UB_NONE_ID||(ubaClass(s)&UBA_LEAVING)!=0u){return false;}
 let t=(vec3u(4*c)>>vec3u(2u))&vec3u(1u);
 return ((ubaRecordWord(s,28u)>>(t.x|(t.y<<1u)|(t.z<<2u)))&1u)!=0u;
}`);
 for(const ch of channels){
  const e=8+2*ch.halo,e3=e**3,W=`ubaW_${K}_${ch.name}`;
  let core:string;
  if(typeof ch.core==="string")core=`${ch.core}(w,ubaB_${K}[n],l)`;
  else{const pool=arena.poolOf(ch.core.pool)||"";core=`ubaLoad${pool}(${arena.baseExpression(ch.core.pool,level,ch.core.component??0,ch.core.bank??0)},w,l.x+8u*(l.y+8u*l.z))`;}
  const outside=ch.outside?`${ch.outside}(p)`:"0.0";
  let seam:string;
  if(ch.seam==="prolong"){seam=`ubaProlong_${K}_${ch.name}(p)`;lines.push(prolongWGSL(`ubaProlong_${K}_${ch.name}`,ch));}
  else if(ch.seam==="restrict"){seam=`ubaRestrict_${K}_${ch.name}(p)`;lines.push(restrictWGSL(`ubaRestrict_${K}_${ch.name}`,ch));}
  else seam=`${ch.seam.fn}(p,w)`;
  const pick=level===0?`let t=(l.x>>2u)|((l.y>>2u)<<1u)|((l.z>>2u)<<2u);
  if(w<UBA_SEAM&&((ubaM_${K}[n]>>t)&1u)!=0u){v=${core};}
  else if(w==UB_NONE_ID){v=${outside};}
  else{v=${seam};}`:`if(w==UB_NONE_ID){v=${outside};}
  else if(${fineOwned}(p)){v=${seam};}
  else{v=${core};}`;
  lines.push(/* wgsl */`
var<workgroup> ${W}:array<f32,${e3}>;
const UBA_HALO_${K}_${ch.name}:i32=${ch.halo};
fn ubaAt_${K}_${ch.name}(c:vec3i)->u32{let q=vec3u(c+vec3i(${ch.halo}));return q.x+${e}u*(q.y+${e}u*q.z);}
fn ubaGet_${K}_${ch.name}(c:vec3i)->f32{return ${W}[ubaAt_${K}_${ch.name}(c)];}
fn ubaPut_${K}_${ch.name}(c:vec3i,v:f32){${W}[ubaAt_${K}_${ch.name}(c)]=v;}
fn ubaStage_${K}_${ch.name}(lane:u32){
 let origin=ubaO_${K};
 for(var k=lane;k<${e3}u;k+=${workgroup}u){
  let c=vec3i(vec3u(k%${e}u,(k/${e}u)%${e}u,k/${e*e}u))-vec3i(${ch.halo});let p=origin+c;
  let d=vec3u(c+vec3i(8))>>vec3u(3u);let n=d.x+3u*d.y+9u*d.z;let l=vec3u(c&vec3i(7));let w=ubaR_${K}[n];
  var v=0.0;
  ${pick}
  ${W}[k]=v;
 }
 workgroupBarrier();
}`);
 }
 lines.push(`
fn ubaStage_${K}(slot:u32,lane:u32){
 ubaRecord_${K}(slot,lane);
 ${channels.map(ch=>`ubaStage_${K}_${ch.name}(lane);`).join("\n ")}
}
`);
 return lines.join("\n");
}
