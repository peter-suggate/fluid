/** BULK16 page transitions on the GPU (docs/plans/bulk16-plan-2026-10-28.md, M2).
 *
 * A demote gives a whole-16h page's block back to the pool: its 4h faces
 * restrict by area to the 36 face means of its eight 16h cells, its root
 * pressures by volume to their eight means, and its 4h vertex block (9^3,
 * skirt included) is kept as it stood, since deep phi is a signed distance.
 * A promote claims a block again and prolongs them (bulk16TransferWGSL).
 *
 * Storage that moves with residency moves here too, exactly, with no
 * restriction: a vertex a demoted page held passes to the stored page that
 * holds it once the page is not stored (umWideVertexSlot's rule), and its
 * high plane's faces to the high neighbour's low-face rows (umFaceLow reads
 * them there once the page is not stored). A promote takes them back.
 *
 * Launches, all fixed grids compiled at setup:
 *  decide (before the pool commit): ask the pool to promote every live
 *    slot its frame asked to (upPromote), and claim a slot for every
 *    resident block whose eight regions are bulk members and give it to the
 *    pool (upDemote). The test cycle (cycle) promotes every live slot
 *    instead, and demotes only when none is live.
 *  demote (before the commit, after decide): each demoted block's transfer.
 *  map, seed (after the commit, before the certificate's pageMark): each
 *    promoted page's new block, its slot; its regions' certificate seed
 *    records, deep water's (seedRecords).
 *  promote, cells, liquid (after the page initialise of its claims): each
 *    promoted page's new block takes its slot's state, and the slot
 *    retires; then its 4h owners' cell fields, deep liquid's.
 * The slot store's begin and finish run around them (UniformBulkPages). */
import { BULK16_PAGE, BULK16_PHI_TEXELS, bulk16TransferWGSL } from "./uniform-bulk-pages-reference";
import { BULK16_STATUS, bulk16LifecycleWGSL, type UniformBulkPages } from "./uniform-bulk-pages";
import { UNIFORM_PAGE_POOL_RECORD, UNIFORM_PAGE_POOL_STATUS, uniformPagePoolWGSL, type UniformPagePool } from "./uniform-page-pool";
import { RESIDENCY_STAGES } from "./uniform-local-residency";
import { UNIFORM_BULK_REACH_CELLS } from "./uniform-bulk-tier-reference";
import { uniformMixedRegionWGSL } from "./uniform-mixed-layout-builder";

const GROUPS=64;
const PASS:GPUComputePassDescriptor={label:"Uniform BULK16 page transfer"};

export interface UniformBulkPageTransferFields{
 /** The live 4h faces' base block (rgba32float; a tile's four texels along z). */
 readonly velocityBase:GPUTexture;
 /** The live vertex phi's 4h base (r32float, 9^3 a block). */
 readonly phiBase:GPUTexture;
 /** The scratch fields' 4h bases. A deep vertex the advect holds keeps its
  * scratch texel (umHoldWarmDeep), and the forced faces are read before
  * this step forces them: a promote writes the image into both, as it
  * writes both low-face rows. */
 readonly phiScratchBase:GPUTexture;readonly velocityScratchBase:GPUTexture;
 /** The 4h owners' cell bases (r32float, 8^3 a block) that deep liquid
  * holds and a claim's AIR image does not: the geometry's centre phi and
  * fill target (gcEvaluate), the volume and its scratch, the phase. */
 readonly cells:{readonly centerPhi:GPUTexture;readonly target:GPUTexture;readonly volume:GPUTexture;readonly volumeScratch:GPUTexture;readonly phase:GPUTexture};
 /** Low-face rows (3072 a block): the live field's and the forced field's. */
 readonly negative:GPUBuffer;readonly negativeScratch:GPUBuffer;
 /** The root's presented pressure: 512 tile keys a block first. */
 readonly pressure:GPUBufferBinding;
 /** The bulk tier's record (flags: bit 0 member, a word a region). */
 readonly record:GPUBufferBinding;
}

/** The census's certificate seed records (UniformMixedDynamicClassifier.residencyRecords). */
export interface UniformBulkResidencyRecords{readonly census:GPUBuffer;readonly offset:number;readonly pages:number;readonly policy:GPUBuffer;readonly regions:{readonly buffer:GPUBuffer;readonly offset:number}}

/** A promoted region's seed records: those pageSeed writes for deep, still
 * water. Every tile wet: transport and pressure seed it (a staged census);
 * ownership seeds its static h tiles (umStaticFine); the legacy stage alone
 * otherwise. Deep 16h water holds no h tile, band tile, source or particle,
 * so no other seed. Its radii bound the travel a bulk member may have
 * (UNIFORM_BULK_REACH_CELLS a step) plus the stage's margin, as lrRadii
 * does; the next census rewrites them. */
function seedRecordsWGSL(records:UniformBulkResidencyRecords):string{
 const travel=Math.ceil(UNIFORM_BULK_REACH_CELLS/4),radius=records.offset+4*RESIDENCY_STAGES.length*records.pages;
 return /* wgsl */`
@group(0) @binding(14) var<storage,read_write> btCensus:array<atomic<u32>>;
@group(0) @binding(15) var<uniform> btPolicy:array<vec4u,7>;
@group(0) @binding(16) var<storage,read> btRegions:array<u32>;
fn umFramePage()->vec3i{return upFramePage();}
${uniformMixedRegionWGSL("btRegionsAt",i=>`btRegions[${records.regions.offset}u+${i}]`)}
@compute @workgroup_size(64) fn btSeedPromoted(@builtin(global_invocation_id) gid:vec3u){
 let staged=(btPolicy[5].x&32768u)!=0u;
 for(var i=gid.x;i<8u*upNewCount();i+=${64*GROUPS}u){
  let b=upNewBlock(i/8u);if(btMap[b]==0u){continue;}
  let region=8u*b+(i%8u);let o=i%8u;
  // The region's tiles (lane x+4y+16z) authored static h: ownership's seeds.
  let origin=8*upNewPage(i/8u)+4*vec3i(i32(o&1u),i32((o>>1u)&1u),i32(o>>2u));
  var fine=vec2u(0u);
  for(var lane=0u;lane<64u;lane++){
   let c=origin+vec3i(i32(lane&3u),i32((lane>>2u)&3u),i32(lane>>4u));
   if((btRegionsAt(vec3u(c))&1u)!=0u){fine[lane>>5u]|=1u<<(lane&31u);}
  }
  for(var stage=0u;stage<${RESIDENCY_STAGES.length}u;stage++){
   var mask=select(vec2u(0u),vec2u(0xffffffffu),select(stage==5u,stage==1u||stage==2u,staged));
   if(staged&&stage==0u){mask=fine;}
   let seeded=(mask.x|mask.y)!=0u;
   let margin=array<u32,${RESIDENCY_STAGES.length}>(${RESIDENCY_STAGES.map(s=>s.margin+"u").join(",")})[stage];
   // lrRadii's: none for ownership, else the travel bound plus the margin.
   let r=select(${travel}u+margin,0u,stage==0u);let radii=select(0u,r|(r<<8u)|(r<<16u),seeded&&stage!=0u);
   let at=${records.offset}u+4u*(stage*${records.pages}u+region);
   atomicStore(&btCensus[at],mask.x);atomicStore(&btCensus[at+1u],mask.y);atomicStore(&btCensus[at+2u],radii);atomicStore(&btCensus[at+3u],radii);
   if(seeded&&stage!=0u){atomicMax(&btCensus[${radius}u],r);}
  }
 }
}
`;
}

export function uniformBulkPageTransferWGSL(pool:UniformPagePool,store:UniformBulkPages,blockBits:number,promote:boolean):string{
 const P=BULK16_PAGE,G=UNIFORM_PAGE_POOL_RECORD.generation;
 return /* wgsl */`${uniformPagePoolWGSL(pool.layout,0,0,1)}
${bulk16LifecycleWGSL(store.layout,0,2)}
// read_write: it shares pool B with the store (binding 2), and one pass may
// not use a buffer both writable and read-only. Nothing here writes it.
@group(0) @binding(3) var<storage,read_write> btRecord:array<u32>;
@group(0) @binding(5) var<storage,read_write> btNegative:array<f32>;
@group(0) @binding(6) var<storage,read_write> btNegativeScratch:array<f32>;
@group(0) @binding(7) var<storage,read_write> btPressure:array<f32>;
// x: 1 while a cycle is asked for, 2 the runtime; y: the blocks an error
// promote may not take (promoteReserve); zw unused. gradient: rho g h (Pa a cell).
struct BTParams{mode:vec4u,gradient:vec4f}
@group(0) @binding(8) var<uniform> btParams:BTParams;
@group(0) @binding(9) var<storage,read_write> btMap:array<u32>;
@group(0) @binding(10) var btPhi:texture_storage_3d<r32float,read_write>;
${promote?"@group(0) @binding(11) var btVelocityOut:texture_storage_3d<rgba32float,write>;\n@group(0) @binding(12) var btPhiScratchOut:texture_storage_3d<r32float,write>;\n@group(0) @binding(13) var btVelocityScratchOut:texture_storage_3d<rgba32float,write>;":"@group(0) @binding(11) var btVelocity:texture_3d<f32>;"}
const BT_BITS:u32=${blockBits}u;const BT_PHI:u32=${P.phi}u;const BT_PRESSURE:u32=${P.pressure}u;const BT_VELOCITY:u32=${P.velocity}u;
fn btBlockCoord(b:u32)->vec3i{let m=(1u<<BT_BITS)-1u;return vec3i(vec3u(b&m,(b>>BT_BITS)&m,b>>(2u*BT_BITS)));}
fn btLocal(c:vec3u)->u32{return c.x|(c.y<<3u)|(c.z<<6u);}
// Tangential axes of a face normal, the 16h faces' order (umBulkRow).
fn btU(a:u32)->u32{return (a+1u)%3u;}fn btV(a:u32)->u32{return (a+2u)%3u;}
fn btTile(a:u32,n:u32,i:u32,j:u32)->vec3u{var c=vec3u(0u);c[a]=n;c[btU(a)]=i;c[btV(a)]=j;return c;}
// The +face texel of axis a of tile c in block b's face base (udFace).
fn btFaceTexel(b:u32,c:vec3u,a:u32)->vec3i{return btBlockCoord(b)*vec3i(8,8,32)+vec3i(vec3u(c.x,c.y,4u*c.z+1u+a));}
// The low-face row of block b on axis a at tile c (umLowFaceIndex of the
// face's cell above, c's origin: h-local tangential coordinates).
fn btLowRow(b:u32,a:u32,c:vec3u)->u32{let l=4u*c;let uv=select(select(l.xy,l.xz,a==1u),l.yz,a==0u);return ((b*3u+a)<<10u)|uv.x|(uv.y<<5u);}
fn btPhiTexel(b:u32,l:vec3u)->vec3i{return btBlockCoord(b)*9+vec3i(l);}
fn btPhiIndex(l:vec3u)->u32{return l.x+9u*(l.y+9u*l.z);}
fn btDemoting(b:u32)->bool{return upBulkWord(atomicLoad(&upWork[UP_WORK_KEEP+b]));}
fn btNew(b:u32)->bool{return upState[UP_AT_RECORDS+8u*b+${G}u]==upGeneration();}
// The block that holds page q now, else UP_NONE; with skip, not one that
// demotes in this commit (before it) or that the commit claimed (after it).
fn btStored(q:vec3i,skip:bool)->u32{
 let b=upBlockOf(q);if(b==UP_NONE||!skip){return b;}
 ${promote?"if(btNew(b)){return UP_NONE;}":"if(btDemoting(b)){return UP_NONE;}"}
 return b;
}
// umWideVertexSlot's rule over the pool's pages: the stored page and texel
// that hold 4h vertex v, w = UP_NONE where no stored page touches it.
fn btHolder(v:vec3i,skip:bool)->vec4u{
 let c=v-vec3i(1);let page=c>>vec3u(3u);let own=vec3u(c&vec3i(7))+vec3u(1u);
 let b=btStored(page,skip);if(b!=UP_NONE){return vec4u(own,b);}
 let plane=(v&vec3i(7))==vec3i(0);
 for(var k=1u;k<8u;k++){
  let step=vec3<bool>((k&1u)!=0u,(k&2u)!=0u,(k&4u)!=0u);
  if(any(step&!plane)){continue;}
  let s=btStored(page+vec3i(step),skip);
  if(s!=UP_NONE){return vec4u(select(own,vec3u(0u),step),s);}
 }
 return vec4u(own,UP_NONE);
}
`;
}

const DECIDE=/* wgsl */`
// The requests the frame listed before decide: what an error promote counts as taken.
@compute @workgroup_size(1) fn btDecideAsked(){atomicStore(&bulkPages[${BULK16_STATUS.promoteAsked}u],atomicLoad(&upWork[0u]));}
// Mode 2 (the runtime): promote every live slot its frame asked to (the
// slot's promote word, BULK16_PAGE.promote), and demote each resident page
// whose eight regions are all 16h members (membership already keeps h out
// of each region's 3^3). Mode 1 (the test cycle): promote every live slot,
// else demote as the runtime does.
@compute @workgroup_size(64) fn btDecidePages(@builtin(global_invocation_id) gid:vec3u){
 if(btParams.mode.x==0u){return;}
 let live=atomicLoad(&bulkPages[${BULK16_STATUS.live}u]);
 for(var i=gid.x;i<live;i+=${64*GROUPS}u){
  let slot=atomicLoad(&bulkPages[BP_LIVE+i]);
  let word=bpRead(slot,${BULK16_PAGE.promote}u);
  if(btParams.mode.x==1u||word!=0u){
   // V2d: an error promote (2) takes a block only past the reserve, counting
   // the requests listed before this pass (btDecideAsked) and the error
   // promotes before it; else it is
   // refused: its word cleared, so the next frame's error test asks again, and
   // the page stays a consistent 16h solve. Closure and scan promotes always ask.
   if(btParams.mode.x==2u&&word==2u&&atomicAdd(&bulkPages[${BULK16_STATUS.promoteClaims}u],1u)+atomicLoad(&bulkPages[${BULK16_STATUS.promoteAsked}u])+btParams.mode.y>=upState[${UNIFORM_PAGE_POOL_STATUS.free}u]+upState[${UNIFORM_PAGE_POOL_STATUS.pending}u]){
    bpWrite(slot,${BULK16_PAGE.promote}u,0u);atomicAdd(&bulkPages[${BULK16_STATUS.promoteRefused}u],1u);continue;
   }
   upPromote(vec3i(bitcast<i32>(bpRead(slot,0u)),bitcast<i32>(bpRead(slot,1u)),bitcast<i32>(bpRead(slot,2u))));
  }
 }
 if(btParams.mode.x==1u&&live>0u){return;}
 for(var i=gid.x;i<upResidentCount();i+=${64*GROUPS}u){
  let b=upResidentBlock(i);let q=upPageOf(b);
  if(upBlockOf(q)!=b){continue;}
  var whole=true;for(var o=0u;o<8u;o++){whole=whole&&(btRecord[8u*b+o]&1u)!=0u;}
  if(!whole){continue;}
  let slot=bpClaim(q);if(slot!=0u){upDemote(b,0x80000000u|slot);}
 }
}
`;

const DEMOTE=/* wgsl */`
fn bpFineFace(a:u32,n:u32,i:u32,j:u32)->f32{
 if(n>0u){return textureLoad(btVelocity,btFaceTexel(btBlock,btTile(a,n-1u,i,j),a),0)[a];}
 // The low plane: the low neighbour's high faces, or this block's low-face rows.
 var q=btPage;q[a]-=1;let low=upBlockOf(q);
 if(low!=UP_NONE){return textureLoad(btVelocity,btFaceTexel(low,btTile(a,7u,i,j),a),0)[a];}
 return btNegative[btLowRow(btBlock,a,btTile(a,0u,i,j))];
}
fn bpFinePressure(c:vec3u)->f32{return btPressure[512u*btBlock+btLocal(c)];}
var<private> btBlock:u32;var<private> btPage:vec3i;
@compute @workgroup_size(64) fn btDemotePages(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var i=group.x;i<upResidentCount();i+=${GROUPS}u){
  let b=upResidentBlock(i);let word=atomicLoad(&upWork[UP_WORK_KEEP+b]);
  if(!upBulkWord(word)){continue;}
  let slot=word&0x3fffffffu;btBlock=b;btPage=upPageOf(b);
  if(lane<36u){bpWrite(slot,BT_VELOCITY+lane,bitcast<u32>(bpRestrictFace(lane)));}
  else if(lane<44u){bpWrite(slot,BT_PRESSURE+lane-36u,bitcast<u32>(bpRestrictPressure(lane-36u)));}
  // The vertex block as it stands; a vertex it holds passes to the stored
  // page that holds it once this page is not stored.
  for(var t=lane;t<${BULK16_PHI_TEXELS}u;t+=64u){
   let l=vec3u(t%9u,(t/9u)%9u,t/81u);let value=textureLoad(btPhi,btPhiTexel(b,l)).x;
   bpWrite(slot,BT_PHI+t,bitcast<u32>(value));
   let v=8*btPage+vec3i(l);let now=btHolder(v,false);
   if(now.w!=b||any(now.xyz!=l)){continue;}
   let next=btHolder(v,true);
   if(next.w!=UP_NONE){textureStore(btPhi,btPhiTexel(next.w,next.xyz),vec4f(value,0.0,0.0,0.0));}
  }
  // The high plane's faces pass to a stored high neighbour's low-face rows.
  for(var t=lane;t<192u;t+=64u){
   let a=t/64u;let i=t%8u;let j=(t/8u)%8u;var q=btPage;q[a]+=1;let high=btStored(q,true);
   if(high==UP_NONE){continue;}
   let value=textureLoad(btVelocity,btFaceTexel(b,btTile(a,7u,i,j),a),0)[a];
   let row=btLowRow(high,a,btTile(a,0u,i,j));btNegative[row]=value;btNegativeScratch[row]=value;
  }
 }
}
`;

const PROMOTE=/* wgsl */`
// The new blocks that hold promoted pages, each its slot (zero for any other claim).
@compute @workgroup_size(256) fn btMapPromoted(@builtin(local_invocation_index) lane:u32){
 for(var i=lane;i<upNewCount();i+=256u){btMap[upNewBlock(i)]=0u;}
 storageBarrier();workgroupBarrier();
 let live=atomicLoad(&bulkPages[${BULK16_STATUS.live}u]);
 for(var i=lane;i<live;i+=256u){
  let slot=atomicLoad(&bulkPages[BP_LIVE+i]);
  let q=vec3i(bitcast<i32>(bpRead(slot,0u)),bitcast<i32>(bpRead(slot,1u)),bitcast<i32>(bpRead(slot,2u)));
  let b=upBlockOf(q);if(b!=UP_NONE&&btNew(b)){btMap[b]=slot;}
 }
}
fn bpFineFace(a:u32,n:u32,i:u32,j:u32)->f32{return 0.0;}
fn bpFinePressure(c:vec3u)->f32{return 0.0;}
@group(0) @binding(17) var btCentreOut:texture_storage_3d<r32float,write>;
@group(0) @binding(18) var btTargetOut:texture_storage_3d<r32float,write>;
@group(0) @binding(19) var btVolumeOut:texture_storage_3d<r32float,write>;
@group(0) @binding(20) var btPhaseOut:texture_storage_3d<r32float,write>;
@group(0) @binding(21) var btVolumeScratchOut:texture_storage_3d<r32float,write>;
fn btCellTexel(b:u32,c:vec3u)->vec3i{return btBlockCoord(b)*8+vec3i(c);}
fn btSum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
var<workgroup> btFace:array<f32,36>;var<workgroup> btP:array<f32,8>;var<workgroup> btShared:u32;
// The slot that holds page q's own vertex block now: a promoted page's
// (btMap) or a BULK16 page's; zero for any other.
fn btSlotOf(q:vec3i)->u32{
 let b=upBlockOf(q);if(b!=UP_NONE){return select(0u,btMap[b],btNew(b));}
 return bpSlot(upWord(q));
}
@compute @workgroup_size(64) fn btPromotePages(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var i=group.x;i<upNewCount();i+=${GROUPS}u){
  let b=upNewBlock(i);if(lane==0u){btShared=btMap[b];}
  let slot=workgroupUniformLoad(&btShared);
  if(slot==0u){continue;}
  let page=upPageOf(b);
  if(lane<36u){btFace[lane]=bitcast<f32>(bpRead(slot,BT_VELOCITY+lane));}else if(lane<44u){btP[lane-36u]=bitcast<f32>(bpRead(slot,BT_PRESSURE+lane-36u));}
  workgroupBarrier();
  var face:array<f32,36>;var pressure:array<f32,8>;
  for(var k=0u;k<36u;k++){face[k]=btFace[k];}for(var r=0u;r<8u;r++){pressure[r]=btP[r];}
  for(var t=lane;t<512u;t+=64u){
   let c=vec3u(t&7u,(t>>3u)&7u,t>>6u);
   btPressure[512u*b+t]=bpProlongPressure(&pressure,c,btParams.gradient.xyz);
  }
  // Faces: planes 1..8 in this block's base; the high plane from a stored
  // high neighbour's low-face rows where it holds them; the low plane in this
  // block's rows where no stored page lies below.
  for(var t=lane;t<1728u;t+=64u){
   let a=t/576u;let n=t%9u;let i=(t/9u)%8u;let j=(t/72u)%8u;
   var value=bpProlongFace(&face,a,n,i,j);
   if(n==0u){
    var q=page;q[a]-=1;
    if(upBlockOf(q)==UP_NONE){let row=btLowRow(b,a,btTile(a,0u,i,j));btNegative[row]=value;btNegativeScratch[row]=value;}
    continue;
   }
   if(n==8u){var q=page;q[a]+=1;let high=btStored(q,true);if(high!=UP_NONE){value=btNegative[btLowRow(high,a,btTile(a,0u,i,j))];}}
   var texel=vec4f(0.0);texel[a]=value;
   textureStore(btVelocityOut,btFaceTexel(b,btTile(a,n-1u,i,j),a),texel);textureStore(btVelocityScratchOut,btFaceTexel(b,btTile(a,n-1u,i,j),a),texel);
  }
  // Vertices this block holds: from the stored page that held each before
  // the commit, else from the slot that holds it in btHolder's rule over
  // the slots (its own page's, else a plane neighbour's in btHolder's
  // order; this block's own slot ends the search). Measured: a page asked
  // back above a BULK16 page read its wall-edge low-plane vertices from its
  // own slot, texels no page held, and dried those tiles.
  for(var t=lane;t<${BULK16_PHI_TEXELS}u;t+=64u){
   let l=vec3u(t%9u,(t/9u)%9u,t/81u);let v=8*page+vec3i(l);
   let now=btHolder(v,false);if(now.w!=b||any(now.xyz!=l)){continue;}
   let before=btHolder(v,true);var value=0.0;
   if(before.w!=UP_NONE){value=textureLoad(btPhi,btPhiTexel(before.w,before.xyz)).x;}
   else{
    let c=v-vec3i(1);let near=c>>vec3u(3u);let own=vec3u(c&vec3i(7))+vec3u(1u);
    let plane=(v&vec3i(7))==vec3i(0);var holder=btSlotOf(near);var at=own;
    for(var k=1u;k<8u&&holder==0u;k++){
     let step=vec3<bool>((k&1u)!=0u,(k&2u)!=0u,(k&4u)!=0u);
     if(any(step&!plane)){continue;}
     holder=btSlotOf(near+vec3i(step));at=select(own,vec3u(0u),step);
    }
    value=bitcast<f32>(bpRead(select(slot,holder,holder!=0u),BT_PHI+btPhiIndex(select(l,at,holder!=0u))));
   }
   textureStore(btPhi,btPhiTexel(b,l),vec4f(value,0.0,0.0,0.0));textureStore(btPhiScratchOut,btPhiTexel(b,l),vec4f(value,0.0,0.0,0.0));
  }
  workgroupBarrier();
  if(lane==0u){bpRetire(slot);}
  workgroupBarrier();
 }
}
// After btPromotePages: each promoted block's 4h owners as the geometry
// writes deep liquid (gcEvaluate): the centre phi, the mean of its eight
// corners in d4Sum8's order, and a fill of 1 (every corner below -1e-20,
// fully open: bulk membership). Its volume, volume scratch and phase: 1.
@compute @workgroup_size(64) fn btPromoteCells(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var i=group.x;i<upNewCount();i+=${GROUPS}u){
  let b=upNewBlock(i);if(lane==0u){btShared=btMap[b];}
  if(workgroupUniformLoad(&btShared)==0u){continue;}
  let page=upPageOf(b);
  for(var t=lane;t<512u;t+=64u){
   let c=vec3u(t&7u,(t>>3u)&7u,t>>6u);var centre:array<f32,8>;var negative=true;
   for(var j=0u;j<8u;j++){
    let v=8*page+vec3i(c)+vec3i(i32(j&1u),i32((j>>1u)&1u),i32(j>>2u));let at=btHolder(v,false);
    let value=textureLoad(btPhi,btPhiTexel(at.w,at.xyz)).x;negative=negative&&value< -1e-20;centre[j]=value*0.125;
   }
   let texel=btCellTexel(b,c);
   textureStore(btCentreOut,texel,vec4f(btSum8(centre)));textureStore(btTargetOut,texel,vec4f(select(0.0,1.0,negative)));textureStore(btVolumeOut,texel,vec4f(1.0));
  }
 }
}
@compute @workgroup_size(64) fn btPromoteLiquid(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var i=group.x;i<upNewCount();i+=${GROUPS}u){
  let b=upNewBlock(i);if(lane==0u){btShared=btMap[b];}
  if(workgroupUniformLoad(&btShared)==0u){continue;}
  for(var t=lane;t<512u;t+=64u){let texel=btCellTexel(b,vec3u(t&7u,(t>>3u)&7u,t>>6u));textureStore(btPhaseOut,texel,vec4f(1.0));textureStore(btVolumeScratchOut,texel,vec4f(1.0));}
 }
}
`;

export class UniformBulkPageTransfer{
 /** Test only (M2's lifecycle harness): demote every whole-16h page at one
  * commit, promote them all at the next, and again. No runtime setting. */
 cycle=false;
 /** V2d: the free blocks an error promote may not take, the pool rule's
  * ballistic columns and closure shell (the solver sets it); an error promote
  * past them is refused (BULK16_STATUS.promoteRefused). */
 promoteReserve=0;
 private readonly params:GPUBuffer;
 private readonly map:GPUBuffer;
 private readonly words=new ArrayBuffer(32);
 private pipelines?:{asked:GPUComputePipeline;decide:GPUComputePipeline;demote:GPUComputePipeline;map:GPUComputePipeline;seed:GPUComputePipeline;promote:GPUComputePipeline;cells:GPUComputePipeline;liquid:GPUComputePipeline};
 private groups?:{decide:GPUBindGroup;demote:GPUBindGroup;map:GPUBindGroup;seed:GPUBindGroup;promote:GPUBindGroup;cells:GPUBindGroup;liquid:GPUBindGroup};
 private records?:UniformBulkResidencyRecords;
 /** The census's certificate records, before initialize (the frame's registerCertificate). */
 attachRecords(records:UniformBulkResidencyRecords):void{this.records=records;}
 get allocatedBytes():number{return this.params.size+this.map.size;}
 constructor(private readonly device:GPUDevice,private readonly pool:UniformPagePool,private readonly store:UniformBulkPages,private readonly blockBits:number,private readonly fields:UniformBulkPageTransferFields){
  this.params=device.createBuffer({label:"Uniform BULK16 transfer params",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.map=device.createBuffer({label:"Uniform BULK16 promoted slots",size:4*pool.layout.blocks,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
 }
 async initialize():Promise<void>{
  const d=this.device,f=this.fields,records=this.records;
  if(!records)throw new Error("BULK16 transfer: no certificate records attached (attachRecords)");
  const buffer=(type:GPUBufferBindingType):Omit<GPUBindGroupLayoutEntry,"binding">=>({visibility:GPUShaderStage.COMPUTE,buffer:{type}});
  // Each entry point's layout holds only the bindings it uses: the storage
  // buffers and textures of one stage stay inside the per-stage limit.
  const kinds:Record<number,Omit<GPUBindGroupLayoutEntry,"binding">>={0:buffer("read-only-storage"),1:buffer("storage"),2:buffer("storage"),3:buffer("storage"),
   5:buffer("storage"),6:buffer("storage"),7:buffer("storage"),8:buffer("uniform"),9:buffer("storage"),
   10:{visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-write",format:"r32float",viewDimension:"3d"}},
   12:{visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
   13:{visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},14:buffer("storage"),15:buffer("uniform"),16:buffer("read-only-storage")};
  const cell={visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"r32float" as const,viewDimension:"3d" as const}};
  for(const binding of [17,18,19,20,21])kinds[binding]=cell;
  for(const [name,t] of Object.entries(f.cells))if(t.format!=="r32float")throw new Error(`BULK16 transfer: the ${name} base is ${t.format}, not r32float`);
  const sampled={visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}};
  const written={visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"rgba32float" as const,viewDimension:"3d" as const}};
  const resources:Record<number,GPUBindingResource>={0:{buffer:this.pool.state},1:{buffer:this.pool.work},2:this.store.binding,
   3:f.record,5:{buffer:f.negative},6:{buffer:f.negativeScratch},7:f.pressure,8:{buffer:this.params},9:{buffer:this.map},10:f.phiBase.createView(),11:f.velocityBase.createView(),12:f.phiScratchBase.createView(),13:f.velocityScratchBase.createView(),14:{buffer:records.census},15:{buffer:records.policy,size:112},16:{buffer:records.regions.buffer},
   17:f.cells.centerPhi.createView(),18:f.cells.target.createView(),19:f.cells.volume.createView(),20:f.cells.phase.createView(),21:f.cells.volumeScratch.createView()};
  const bind=(label:string,bindings:number[],velocity?:Omit<GPUBindGroupLayoutEntry,"binding">)=>{
   const layout=d.createBindGroupLayout({label,entries:bindings.map(binding=>({binding,...(binding===11?velocity!:kinds[binding]!)}))});
   return {layout:d.createPipelineLayout({label,bindGroupLayouts:[layout]}),group:d.createBindGroup({label,layout,entries:bindings.map(binding=>({binding,resource:resources[binding]!}))})};
  };
  const compile=async(code:string,label:string)=>{const m=d.createShaderModule({label,code});const e=(await m.getCompilationInfo()).messages.filter(x=>x.type==="error");if(e.length)throw new Error(`${label}\n${e.map(x=>`${x.lineNum}: ${x.message}`).join("\n")}`);return m;};
  const [early,late]=await Promise.all([
   compile(uniformBulkPageTransferWGSL(this.pool,this.store,this.blockBits,false)+bulk16TransferWGSL+DECIDE+DEMOTE,"Uniform BULK16 decide and demote"),
   compile(uniformBulkPageTransferWGSL(this.pool,this.store,this.blockBits,true)+bulk16TransferWGSL+PROMOTE+seedRecordsWGSL(records),"Uniform BULK16 promote")]);
  const stages={decide:bind("Uniform BULK16 decide",[0,1,2,3,8]),demote:bind("Uniform BULK16 demote",[0,1,2,5,6,7,10,11],sampled),
   map:bind("Uniform BULK16 promoted slots",[0,1,2,9]),seed:bind("Uniform BULK16 promoted seeds",[0,1,2,9,14,15,16]),promote:bind("Uniform BULK16 promote",[0,1,2,5,6,7,8,9,10,11,12,13],written),
   cells:bind("Uniform BULK16 promoted cells",[0,1,2,9,10,17,18,19]),liquid:bind("Uniform BULK16 promoted liquid",[0,1,2,9,20,21])};
  const pipeline=(stage:keyof typeof stages,module:GPUShaderModule,entryPoint:string)=>d.createComputePipelineAsync({label:`Uniform BULK16 ${stage}`,layout:stages[stage].layout,compute:{module,entryPoint}});
  const [asked,decide,demote,map,seed,promote,cells,liquid]=await Promise.all([pipeline("decide",early,"btDecideAsked"),pipeline("decide",early,"btDecidePages"),pipeline("demote",early,"btDemotePages"),pipeline("map",late,"btMapPromoted"),pipeline("seed",late,"btSeedPromoted"),pipeline("promote",late,"btPromotePages"),pipeline("cells",late,"btPromoteCells"),pipeline("liquid",late,"btPromoteLiquid")]);
  this.pipelines={asked,decide,demote,map,seed,promote,cells,liquid};
  this.groups={decide:stages.decide.group,demote:stages.demote.group,map:stages.map.group,seed:stages.seed.group,promote:stages.promote.group,cells:stages.cells.group,liquid:stages.liquid.group};
 }
 /** Before the pool commit: rho g h is the hydrostatic pressure step of a cell (Pa). */
 encodeBefore(encoder:GPUCommandEncoder,gradient:readonly [number,number,number]):void{
  const p=this.pipelines,g=this.groups;if(!p||!g)throw new Error("BULK16 transfer is not initialized");
  new Uint32Array(this.words,0,4).set([this.cycle?1:2,this.promoteReserve,0,0]);new Float32Array(this.words,16,4).set([...gradient,0]);
  this.device.queue.writeBuffer(this.params,0,this.words);
  this.store.encodeBegin(encoder);
  const pass=encoder.beginComputePass(PASS);
  pass.setBindGroup(0,g.decide);pass.setPipeline(p.asked);pass.dispatchWorkgroups(1);pass.setPipeline(p.decide);pass.dispatchWorkgroups(GROUPS);pass.setBindGroup(0,g.demote);pass.setPipeline(p.demote);pass.dispatchWorkgroups(GROUPS);pass.end();
 }
 /** After the commit, before the certificate's pageMark: the promoted pages' slots and seed records. */
 encodeClaimed(encoder:GPUCommandEncoder):void{
  const p=this.pipelines,g=this.groups;if(!p||!g)throw new Error("BULK16 transfer is not initialized");
  const pass=encoder.beginComputePass(PASS);
  pass.setBindGroup(0,g.map);pass.setPipeline(p.map);pass.dispatchWorkgroups(1);pass.setBindGroup(0,g.seed);pass.setPipeline(p.seed);pass.dispatchWorkgroups(GROUPS);pass.end();
 }
 /** After the page initialise of the commit's claims. */
 encodeAfter(encoder:GPUCommandEncoder):void{
  const p=this.pipelines,g=this.groups;if(!p||!g)throw new Error("BULK16 transfer is not initialized");
  const pass=encoder.beginComputePass(PASS);
  pass.setBindGroup(0,g.promote);pass.setPipeline(p.promote);pass.dispatchWorkgroups(GROUPS);
  pass.setBindGroup(0,g.cells);pass.setPipeline(p.cells);pass.dispatchWorkgroups(GROUPS);pass.setBindGroup(0,g.liquid);pass.setPipeline(p.liquid);pass.dispatchWorkgroups(GROUPS);pass.end();
  this.store.encodeFinish(encoder);
 }
 destroy():void{this.params.destroy();this.map.destroy();}
}
