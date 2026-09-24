import { uniformAbOn } from "./uniform-ab-switch";
/** Page table for transient geometric transport / sharpening records. It lives
 * after the existing conditioning planes so no extra storage binding is needed.
 * Donor IDs remain logical cell IDs: paging must not change the transport graph.
 */
export interface UniformVolumePageShaderOptions { edge: 16 | 32; base: number; count: number; work?: boolean; nativeRecords?: boolean; }
/** One layout for the shared receiver list and its donor closure. All offsets
 * are words in the conditioning buffer; both lists hold logical 4³ tile IDs. */
export function uniformVolumeWorkLayout(base: number, pages: number, tiles: number) {
  const receiver = base + 8 + 2 * pages;
  const donor = receiver + 1 + tiles;
  return { receiver, donor, donorDispatch: donor + 1, words: 8 + 2 * pages + 5 + 2 * tiles };
}
/** balancesupport: without a work list the restricted balance entries compile
 * to no-ops and the host never encodes them. */
const UV_NO_WORK_LIST = " fn uvWorkListTile(n:u32)->u32{return 0xffffffffu;} fn uvBalanceWindowBase()->u32{return 0u;}";
export const UNIFORM_VOLUME_PAGE_ENTRIES = ["uvMarkTransportPages", "uvMarkSharpenPages", "uvCompactPages", "uvDonorDispatch"] as const;
export function uniformVolumePagesWGSL(options?: UniformVolumePageShaderOptions): string {
  // Native records in brick order (uvBrickOrder): the 32 lanes of a 4x4x2
  // receiver block then address 32 consecutive records instead of eight
  // four-record row fragments.
  const native = uniformAbOn("edgebricks") ? "return uvBrickOrder(i);" : "return i;";
  if (!options) return "fn uvEdgeAddress(i:u32)->u32{" + native + "} fn uvWorkId(g:vec3u)->vec3i{return activeId(g);} fn uvPageWorkEnabled()->bool{return false;} fn uvAppendDonorWork(t:vec3i){} fn uvDonorWorkId(g:vec3u)->vec3i{return vec3i(g);}" + UV_NO_WORK_LIST;
  const {edge,base,count,work,nativeRecords}=options;
  return /* wgsl */ `
fn uvPageWorkEnabled()->bool{return ${work ? "true" : "false"};}
const UV_PAGE_EDGE:u32=${edge}u;
const UV_PAGE_BASE:u32=${base}u;
const UV_PAGE_COUNT:u32=${count}u;
// Header: edge, pages X/Y/Z, used slots, sharpening mode, dispatch X/Y.
// Flags then logical-page -> slot, followed by a count and compact 4³ tile list.
fn uvWorkBase()->u32{return UV_PAGE_BASE+8u+2u*UV_PAGE_COUNT;}
// Both lists are compiled from the same topology. Receiver work is narrowed
// after extension; donor work retains its conservative scatter closure.
fn uvDonorWorkBase()->u32{let d=(vec3u(dims())+vec3u(3u))/4u;return uvWorkBase()+1u+d.x*d.y*d.z;}
${work ? `// The current receiver or sharpening list's n-th 4³ tile, or ~0 past its count.
fn uvWorkListTile(n:u32)->u32{
 if(n>=u32(atomicLoad(&sharpenDeposits[uvWorkBase()]))){return 0xffffffffu;}
 return u32(atomicLoad(&sharpenDeposits[uvWorkBase()+1u+n]));
}
// Six words after the donor list: the surface-deficit tile box (origin, dims).
fn uvBalanceWindowBase()->u32{let d=(vec3u(dims())+vec3u(3u))/4u;return uvDonorWorkBase()+4u+d.x*d.y*d.z;}` : UV_NO_WORK_LIST}
fn uvAppendDonorWork(t:vec3i){
 ${work ? `let b=uvDonorWorkBase();let n=u32(atomicAdd(&sharpenDeposits[b],1));
 let d=(vec3u(dims())+vec3u(3u))/4u;
 atomicStore(&sharpenDeposits[b+4u+n],t.x+i32(d.x)*(t.y+i32(d.y)*t.z));` : ''}
}
fn uvDonorWorkId(g:vec3u)->vec3i{
 ${work ? `if(uvTransportTiles()){
 let b=uvDonorWorkBase();let n=g.x/4u+65535u*(g.y/4u);
 if(n>=u32(atomicLoad(&sharpenDeposits[b]))){return dims();}
 let t=u32(atomicLoad(&sharpenDeposits[b+4u+n]));let d=(vec3u(dims())+vec3u(3u))/4u;
 return vec3i(vec3u(t%d.x,(t/d.x)%d.y,t/(d.x*d.y))*4u+g%vec3u(4u));
 }` : ''}
 return vec3i(g);
}
fn uvAppendWork(id:vec3i){
 ${work ? `let n=atomicAdd(&sharpenDeposits[uvWorkBase()],1);
 let d=(vec3u(dims())+vec3u(3u))/4u;let t=vec3u(id)/4u;
 atomicStore(&sharpenDeposits[uvWorkBase()+1u+u32(n)],i32(t.x+d.x*(t.y+d.y*t.z)));` : ''}
}
fn uvWorkId(g:vec3u)->vec3i{
 ${work ? `let index=g.x/4u+65535u*(g.y/4u);
 if(index>=u32(atomicLoad(&sharpenDeposits[uvWorkBase()]))){return dims();}
 let tile=u32(atomicLoad(&sharpenDeposits[uvWorkBase()+1u+index]));
 let d=(vec3u(dims())+vec3u(3u))/4u;
 return vec3i(vec3u(tile%d.x,(tile/d.x)%d.y,tile/(d.x*d.y))*4u+g%vec3u(4u));` : 'return activeId(g);'}
}
fn uvPageIndex(id:vec3i)->u32{
 let d=(vec3u(dims())+vec3u(UV_PAGE_EDGE-1u))/UV_PAGE_EDGE;
 let q=vec3u(id)/UV_PAGE_EDGE;return q.x+d.x*(q.y+d.y*q.z);
}
fn uvEdgeAddress(i:u32)->u32{
 ${nativeRecords ? native : `let id=uvCell(i);let page=uvPageIndex(id);
 let slot=u32(atomicLoad(&sharpenDeposits[UV_PAGE_BASE+8u+UV_PAGE_COUNT+page]));
 let q=vec3u(id)%UV_PAGE_EDGE;
 return slot*UV_PAGE_EDGE*UV_PAGE_EDGE*UV_PAGE_EDGE+q.x+UV_PAGE_EDGE*(q.y+UV_PAGE_EDGE*q.z);`}
}
fn uvMarkPage(id:vec3i){
 if(valid(id)){atomicStore(&sharpenDeposits[UV_PAGE_BASE+8u+uvPageIndex(id)],1);}
}
@compute @workgroup_size(4,4,4)
fn uvMarkTransportPages(@builtin(global_invocation_id)gid:vec3u){
 if(any(gid%vec3u(4u)!=vec3u(0u))){return;}
 let id=activeId(gid);if(!valid(id)||!uvInWindow(id)||uvTransportSkip(id)){return;}uvMarkPage(id);uvAppendWork(id);
}
fn uvPageSharpenActive(id:vec3i)->bool {
 if(!uvInWindow(id)){return false;}
 return atomicLoad(&sharpenDeposits[UV_PAGE_BASE+5u])==0 || atomicLoad(&sharpenDeposits[uvSharpenTileIndex(id)])!=0;
}
@compute @workgroup_size(4,4,4)
fn uvMarkSharpenPages(@builtin(global_invocation_id)gid:vec3u){
 if(any(gid%vec3u(4u)!=vec3u(0u))){return;}
 let id=activeId(gid);if(!valid(id)||!uvPageSharpenActive(id)){return;}
 uvMarkPage(id);uvAppendWork(id);
 // Limited flux reads the neighboring record. Preserve the existing window
 // behavior by backing every eligible neighbor as well as every writer.
 for(var a=0u;a<3u;a++){var e=vec3i(0);e[a]=1;
  if(valid(id-e)&&uvPageSharpenActive(id-e)){uvMarkPage(id-e);}
  if(valid(id+4*e)&&uvPageSharpenActive(id+4*e)){uvMarkPage(id+4*e);}}
}
// donorsampled: the dispatch for the donor tile list uvBuildEdgesSampled appended.
@compute @workgroup_size(1)
fn uvDonorDispatch(){
 ${work ? `let b=uvDonorWorkBase();let donors=u32(atomicLoad(&sharpenDeposits[b]));
 atomicStore(&sharpenDeposits[b+1u],i32(min(donors,65535u)));
 atomicStore(&sharpenDeposits[b+2u],i32((donors+65534u)/65535u));
 atomicStore(&sharpenDeposits[b+3u],1);` : ''}
}
@compute @workgroup_size(1)
fn uvCompactPages(){
 let d=(vec3u(dims())+vec3u(UV_PAGE_EDGE-1u))/UV_PAGE_EDGE;
 atomicStore(&sharpenDeposits[UV_PAGE_BASE],i32(UV_PAGE_EDGE));
 atomicStore(&sharpenDeposits[UV_PAGE_BASE+1u],i32(d.x));
 atomicStore(&sharpenDeposits[UV_PAGE_BASE+2u],i32(d.y));
 atomicStore(&sharpenDeposits[UV_PAGE_BASE+3u],i32(d.z));
 var used=0u;
 for(var page=0u;page<UV_PAGE_COUNT;page++){
  var slot=0xffffffffu;
  if(atomicLoad(&sharpenDeposits[UV_PAGE_BASE+8u+page])!=0){slot=used;used++;}
  atomicStore(&sharpenDeposits[UV_PAGE_BASE+8u+UV_PAGE_COUNT+page],bitcast<i32>(slot));
 }
 atomicStore(&sharpenDeposits[UV_PAGE_BASE+4u],i32(used));
 ${work ? `let groups=u32(atomicLoad(&sharpenDeposits[uvWorkBase()]));
 atomicStore(&sharpenDeposits[UV_PAGE_BASE+6u],i32(min(groups,65535u)));
 atomicStore(&sharpenDeposits[UV_PAGE_BASE+7u],i32((groups+65534u)/65535u));
 let b=uvDonorWorkBase();let donors=u32(atomicLoad(&sharpenDeposits[b]));
 atomicStore(&sharpenDeposits[b+1u],i32(min(donors,65535u)));
 atomicStore(&sharpenDeposits[b+2u],i32((donors+65534u)/65535u));
 atomicStore(&sharpenDeposits[b+3u],1);` : ''}
}
`;
}
