import { uniformWindowAddressWGSL } from "./uniform-window-address";
import { UNIFORM_PAGE_TABLE_WORDS, uniformBlockNeighbourLayout } from "./uniform-lattice-address";

/** Sweeps of the 32h level over the pool's blocks (mixedSweepBlocks). A
 * resident page lies within three pages of a source: the residency closure
 * reaches 64 cells from a near tile, a near tile 20 cells from the surface.
 * Odd, so the last sweep writes the level's `up` textures. */
export const UNIFORM_EXTRAPOLATION_BLOCK_SWEEPS = 3;

/** Coarse nearest-source continuation for narrow-band FLIP: the 4h, 8h, 16h
 * and 32h levels of every pool block (8, 4, 2 and 1 cells a block axis),
 * block b's cells at umBlockCoord(b) in each level's textures. */
export function uniformVelocityExtrapolationShader(blocks: number, blockBits: number): string {
  return /* wgsl */ `
struct Params {
  dimsDt: vec4f,
  cellGravity: vec4f,
}

@group(0) @binding(0) var<uniform> params: Params;

@group(0) @binding(1) var primaryIn: texture_3d<f32>;

@group(0) @binding(2) var secondaryIn: texture_3d<f32>;

@group(0) @binding(3) var primaryOut: texture_storage_3d<rgba32float, write>;

@group(0) @binding(4) var sourceOrigins: texture_3d<u32>;

@group(0) @binding(5) var existingOrigins: texture_3d<u32>;

@group(0) @binding(6) var outputOrigins: texture_storage_3d<rgba32uint, write>;

// The page pool's tail from the page table's first word: the table, one
// record per block (its page, or FREE) and 27 neighbour words per block (the
// table words of the pages around its own).
@group(0) @binding(7) var<storage,read> pagePool: array<u32>;

const BLOCKS:u32=${blocks}u;
const BLOCK_BITS:u32=${blockBits}u;
const RECORDS:u32=${UNIFORM_PAGE_TABLE_WORDS}u;
const ROWS:u32=${uniformBlockNeighbourLayout(blocks).rows}u;
const FREE:u32=0xffffffffu;
${uniformWindowAddressWGSL("pagePool","0u","BLOCKS")}

fn blockStored(b:u32)->bool{return b-1u<BLOCKS-2u;}

fn blockCoord(b:u32)->vec3i{let m=(1u<<BLOCK_BITS)-1u;return vec3i(vec3u(b&m,(b>>BLOCK_BITS)&m,b>>(2u*BLOCK_BITS)));}

fn d4Sum8Vec2(value:array<vec2f,8>)->vec2f{
  let y0=(value[0]+value[5])+(value[1]+value[4]);
  let y1=(value[2]+value[7])+(value[3]+value[6]);
  return y0+y1;
}

fn componentKnown(state: vec4f, component: u32) -> bool {
  return (u32(round(state.w)) & (1u << component)) != 0u;
}

// A source bound is a cell of the page window, 11 bits an axis: x, y and the
// low ten bits of z in the origin texel's component, the eleventh bit of z in
// bit 24 + component of the texel's w.
fn sourceWord(p:vec3i)->u32{let q=vec3u(p);return q.x|(q.y<<11u)|((q.z&1023u)<<22u);}

fn sourcePoint(word:u32,high:u32)->vec3i{return vec3i(vec3u(word&2047u,(word>>11u)&2047u,(word>>22u)|(high<<10u)));}

fn sourceFace(word:u32,high:u32,component:u32)->vec3f{
  var point=vec3f(sourcePoint(word,high))+vec3f(0.5);point[component]+=0.5;return point;
}

fn highBit(w:u32,component:u32)->u32{return (w>>(24u+component))&1u;}

// One cell of a level: the lane's block, its page and the cell in the block.
// held: the block holds a page (a free block and the two sentinels do not).
struct LevelCell { held: bool, block: u32, page: vec3i, local: vec3i, edge: i32 }

// 64 cells a workgroup in block order; edge is the level's cells a block axis.
fn levelCell(group:u32,cell:u32,edge:u32)->LevelCell{
  let k=firstTrailingBit(edge);let n=64u*group+cell;let b=n>>(3u*k);let l=n&((1u<<(3u*k))-1u);
  let record=umBlockWord(min(b,BLOCKS-1u));
  return LevelCell(blockStored(b)&&record!=FREE,b,vec3i(vec3u(record&63u,(record>>6u)&63u,(record>>12u)&63u)),
    vec3i(vec3u(l&(edge-1u),(l>>k)&(edge-1u),l>>(2u*k))),i32(edge));
}

fn levelTexel(c:LevelCell)->vec3i{return blockCoord(c.block)*c.edge+c.local;}

// The +face centre of a level cell on its component, in cells.
fn faceLocation(c:LevelCell,component:u32)->vec3f{
  let g=c.page*c.edge+c.local;let size=32.0/f32(c.edge);
  var point=(vec3f(g)+vec3f(0.5))*size;point[component]=f32(g[component]+1)*size;
  return point;
}

// The texel of cell q of the level with edge cells a page (q from the page
// window's origin), seen from block home on page: home's own, or a neighbour
// page's through home's neighbour row. x < 0: that page is not stored, so the
// cell holds nothing.
fn tapTexel(home:u32,page:vec3i,q:vec3i,edge:i32)->vec3i{
  let k=firstTrailingBit(u32(edge));let d=(q>>vec3u(k))-page;var b=home;
  if(any(d!=vec3i(0))){b=pagePool[ROWS+27u*home+u32(d.x+3*d.y+9*d.z+13)];if(!blockStored(b)){return vec3i(-1);}}
  return blockCoord(b)*edge+(q&vec3i(edge-1));
}

// high: bit 0 the lower bound's eleventh z bit, bit 1 the upper bound's.
struct NearestSample { value: f32, weight: f32, lower: u32, upper: u32, high: u32 }

fn sampleOf(sum:vec2f,supportLower:vec3i,supportUpper:vec3i)->NearestSample{
  if(sum.y<=0.0){return NearestSample(0.0,0.0,0u,0u,0u);}
  return NearestSample(sum.x/sum.y,sum.y,sourceWord(supportLower),sourceWord(supportUpper),
    (u32(supportLower.z)>>10u)|((u32(supportUpper.z)>>10u)<<1u));
}

fn cellScale()->vec3f{let spacing=params.cellGravity.xyz;return spacing/min(spacing.x,min(spacing.y,spacing.z));}

fn boundDistance(location:vec3f,first:vec4u,last:vec4u,component:u32)->f32{
  let lo=sourceFace(first[component],highBit(first.w,component),component);
  let hi=sourceFace(last[component],highBit(last.w,component),component);
  let delta=(location-clamp(location,lo,hi))*cellScale();
  return (delta.x*delta.x+delta.z*delta.z)+delta.y*delta.y;
}

var<private> nearestSelection:u32;

// The nearest known value among the eight cells of the level with sourceEdge
// cells a page around cell c: the finer level's children (restriction), or
// the coarser level's cells around c (prolongation). Ties average.
fn nearestHierarchySample(c: LevelCell, sourceEdge: i32, component: u32, footprint: bool) -> NearestSample {
  nearestSelection=0u;
  let location = faceLocation(c, component);
  let g=c.page*c.edge+c.local;let ratio=f32(sourceEdge)/f32(c.edge);
  var sourcePosition = (vec3f(g)+vec3f(0.5))*ratio-vec3f(0.5);
  sourcePosition[component] = f32(g[component]+1)*ratio-1.0;
  let lower = select(vec3i(floor(sourcePosition)), 2*g, footprint);
  let half=vec3i(0,0,i32(textureDimensions(sourceOrigins).z/2u));
  var best = 1e30;
  var distances: array<f32,8>;
  var lowers: array<vec3i,8>;var uppers: array<vec3i,8>;
  var values: array<f32,8>;
  for(var k=0u;k<8u;k++){
    distances[k] = 1e30;
    let offset=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
    let fraction=fract(sourcePosition);
    let weights=select(vec3f(1.0)-fraction,fraction,offset==vec3i(1));
    if (!footprint && any(weights<=vec3f(0.0))) { continue; }
    let texel=tapTexel(c.block,c.page,lower+offset,sourceEdge);
    if (texel.x < 0) { continue; }
    let state = textureLoad(primaryIn,texel,0);
    if (!componentKnown(state,component)) { continue; }
    let first=textureLoad(sourceOrigins,texel,0);let last=textureLoad(sourceOrigins,texel+half,0);
    let distance=boundDistance(location,first,last,component);
    distances[k]=distance;values[k]=state[component];
    lowers[k]=sourcePoint(first[component],highBit(first.w,component));uppers[k]=sourcePoint(last[component],highBit(last.w,component));
    best=min(best,distance);
  }
  var contributions:array<vec2f,8>;
  var supportLower=vec3i(4096);var supportUpper=vec3i(-1);
  let epsilon=1e-6*max(1.0,best);
  for(var k=0u;k<8u;k++){
    contributions[k]=vec2f(0.0);
    if(distances[k]<1e29 && abs(distances[k]-best)<=epsilon){
      contributions[k]=vec2f(values[k],1.0);nearestSelection|=1u<<k;
      supportLower=min(supportLower,lowers[k]);supportUpper=max(supportUpper,uppers[k]);
    }
  }
  return sampleOf(d4Sum8Vec2(contributions),supportLower,supportUpper);
}

// The same choice among the 26 blocks around c's on the 32h level, summed in
// row order.
fn nearestBlockSample(c: LevelCell, component: u32) -> NearestSample {
  let location = faceLocation(c, component);
  let half=vec3i(0,0,i32(textureDimensions(sourceOrigins).z/2u));
  var best = 1e30;
  var distances: array<f32,27>;
  var lowers: array<vec3i,27>;var uppers: array<vec3i,27>;
  var values: array<f32,27>;
  for(var i=0u;i<27u;i++){
    distances[i] = 1e30;
    if (i == 13u) { continue; }
    let b=pagePool[ROWS+27u*c.block+i];
    if (!blockStored(b)) { continue; }
    let texel=blockCoord(b);let state = textureLoad(primaryIn,texel,0);
    if (!componentKnown(state,component)) { continue; }
    let first=textureLoad(sourceOrigins,texel,0);let last=textureLoad(sourceOrigins,texel+half,0);
    let distance=boundDistance(location,first,last,component);
    distances[i]=distance;values[i]=state[component];
    lowers[i]=sourcePoint(first[component],highBit(first.w,component));uppers[i]=sourcePoint(last[component],highBit(last.w,component));
    best=min(best,distance);
  }
  var sum=vec2f(0.0);
  var supportLower=vec3i(4096);var supportUpper=vec3i(-1);
  let epsilon=1e-6*max(1.0,best);
  for(var i=0u;i<27u;i++){
    if(distances[i]<1e29 && abs(distances[i]-best)<=epsilon){
      sum+=vec2f(values[i],1.0);
      supportLower=min(supportLower,lowers[i]);supportUpper=max(supportUpper,uppers[i]);
    }
  }
  return sampleOf(sum,supportLower,supportUpper);
}

// x: the value's bits; y, z: the source bounds; w: bit 0 known, bits 1 and 2
// the bounds' eleventh z bits, bits 8.. a prolongation's tap selection.
var<workgroup> mixedHierarchyLanes:array<vec4u,192>;

fn mixedHierarchyEntry(sample:NearestSample)->vec4u{
  return select(vec4u(0u),vec4u(bitcast<u32>(sample.value),sample.lower,sample.upper,1u|(sample.high<<1u)),sample.weight>0.0);
}

fn mixedExistingEntry(state:vec4f,first:vec4u,last:vec4u,component:u32)->vec4u{
  return vec4u(bitcast<u32>(state[component]),first[component],last[component],
    select(0u,1u,componentKnown(state,component))|(highBit(first.w,component)<<1u)|(highBit(last.w,component)<<2u));
}

fn mixedStoreOrigins(texel:vec3i,x:vec4u,y:vec4u,z:vec4u){
  let half=vec3i(0,0,i32(textureDimensions(outputOrigins).z/2u));
  let selection=((x.w>>8u)&255u)|(((y.w>>8u)&255u)<<8u)|(((z.w>>8u)&255u)<<16u);
  textureStore(outputOrigins,texel,vec4u(x.y,y.y,z.y,selection|(((x.w>>1u)&1u)<<24u)|(((y.w>>1u)&1u)<<25u)|(((z.w>>1u)&1u)<<26u)));
  textureStore(outputOrigins,texel+half,vec4u(x.z,y.z,z.z,(((x.w>>2u)&1u)<<24u)|(((y.w>>2u)&1u)<<25u)|(((z.w>>2u)&1u)<<26u)));
}

fn mixedStoreValues(texel:vec3i,x:vec4u,y:vec4u,z:vec4u){
  textureStore(primaryOut,texel,vec4f(bitcast<f32>(x.x),bitcast<f32>(y.x),bitcast<f32>(z.x),f32((x.w&1u)|((y.w&1u)<<1u)|((z.w&1u)<<2u))));
}

// A level from the next finer one, whose cells are in the same block.
@compute @workgroup_size(192)
fn mixedRestrictKnownVelocity(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let component=lane/64u;let cell=lane%64u;
  let c=levelCell(wid.x,cell,textureDimensions(primaryOut).x>>BLOCK_BITS);
  var entry=vec4u(0u);
  if(c.held){
    var result=nearestHierarchySample(c,2*c.edge,component,false);
    if(result.weight<=0.0&&component==1u){result=nearestHierarchySample(c,2*c.edge,component,true);}
    entry=mixedHierarchyEntry(result);
  }
  mixedHierarchyLanes[lane]=entry;
  workgroupBarrier();
  if(component!=0u||!c.held){return;}
  let x=mixedHierarchyLanes[cell];let y=mixedHierarchyLanes[cell+64u];let z=mixedHierarchyLanes[cell+128u];
  mixedStoreOrigins(levelTexel(c),x,y,z);mixedStoreValues(levelTexel(c),x,y,z);
}

// One sweep of the 32h level, a cell a block: a block without a value takes
// the nearest among its 26 neighbours'. ${UNIFORM_EXTRAPOLATION_BLOCK_SWEEPS} sweeps stand for every level
// above 32h.
@compute @workgroup_size(192)
fn mixedSweepBlocks(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let component=lane/64u;let cell=lane%64u;let c=levelCell(wid.x,cell,1u);
  var entry=vec4u(0u);
  if(c.held){
    let texel=levelTexel(c);let half=vec3i(0,0,i32(textureDimensions(sourceOrigins).z/2u));
    entry=mixedExistingEntry(textureLoad(primaryIn,texel,0),textureLoad(sourceOrigins,texel,0),textureLoad(sourceOrigins,texel+half,0),component);
    if((entry.w&1u)==0u){entry=mixedHierarchyEntry(nearestBlockSample(c,component));}
  }
  mixedHierarchyLanes[lane]=entry;
  workgroupBarrier();
  if(component!=0u||!c.held){return;}
  let x=mixedHierarchyLanes[cell];let y=mixedHierarchyLanes[cell+64u];let z=mixedHierarchyLanes[cell+128u];
  mixedStoreOrigins(levelTexel(c),x,y,z);mixedStoreValues(levelTexel(c),x,y,z);
}

// A 4h cell whose +face on the component is a closed wall (the page across
// it is SOLID) stays unknown: the wall face publishes its physical value, and
// no far lookup may take the wall layer's value for a face inside the tile.
fn closedFace(c:LevelCell,component:u32)->bool{
  if(c.edge!=8||c.local[component]!=7){return false;}
  return pagePool[ROWS+27u*c.block+13u+select(select(9u,3u,component==1u),1u,component==0u)]==0xffffffffu;
}

// A level's unknown values from the complete coarser level; a tap in another
// page goes through the block's neighbour row.
@compute @workgroup_size(192)
fn mixedProlongUnknownVelocity(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let component=lane/64u;let cell=lane%64u;
  let c=levelCell(wid.x,cell,textureDimensions(primaryOut).x>>BLOCK_BITS);
  var entry=vec4u(0u);
  if(c.held){
    let texel=levelTexel(c);let half=vec3i(0,0,i32(textureDimensions(existingOrigins).z/2u));
    entry=mixedExistingEntry(textureLoad(secondaryIn,texel,0),textureLoad(existingOrigins,texel,0),textureLoad(existingOrigins,texel+half,0),component);
    if((entry.w&1u)==0u&&!closedFace(c,component)){
      let result=nearestHierarchySample(c,c.edge/2,component,false);
      if(result.weight>0.0){entry=mixedHierarchyEntry(result);entry.w|=nearestSelection<<8u;}
    }
  }
  mixedHierarchyLanes[lane]=entry;
  workgroupBarrier();
  if(component!=0u||!c.held){return;}
  let x=mixedHierarchyLanes[cell];let y=mixedHierarchyLanes[cell+64u];let z=mixedHierarchyLanes[cell+128u];
  mixedStoreOrigins(levelTexel(c),x,y,z);mixedStoreValues(levelTexel(c),x,y,z);
}

// The same fill from the taps a prolongation recorded in its origins.
@compute @workgroup_size(192)
fn mixedReplayUnknownVelocity(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
 let component=lane/64u;let cell=lane%64u;
 let c=levelCell(wid.x,cell,textureDimensions(primaryOut).x>>BLOCK_BITS);var value=0.0;var mask=0u;
 if(c.held){
  let texel=levelTexel(c);let existing=textureLoad(secondaryIn,texel,0);value=existing[component];mask=u32(round(existing.w));
  if(!componentKnown(existing,component)){
   let selected=(textureLoad(existingOrigins,texel,0).w>>(8u*component))&255u;
   if(selected!=0u){
    let g=c.page*c.edge+c.local;
    var sourcePosition=(vec3f(g)+0.5)*0.5-0.5;
    sourcePosition[component]=f32(g[component]+1)*0.5-1.0;
    let lower=vec3i(floor(sourcePosition));var terms:array<vec2f,8>;
    for(var k=0u;k<8u;k++){
     terms[k]=vec2f(0);if((selected&(1u<<k))!=0u){let q=tapTexel(c.block,c.page,lower+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)),c.edge/2);terms[k]=vec2f(textureLoad(primaryIn,q,0)[component],1.0);}
    }
    let sum=d4Sum8Vec2(terms);value=sum.x/sum.y;mask|=1u<<component;
   }
  }
 }
 mixedHierarchyLanes[lane]=vec4u(bitcast<u32>(value),0u,0u,mask);workgroupBarrier();
 if(component!=0u||!c.held){return;}
 let x=mixedHierarchyLanes[cell];let y=mixedHierarchyLanes[cell+64u];let z=mixedHierarchyLanes[cell+128u];
 textureStore(primaryOut,levelTexel(c),vec4f(bitcast<f32>(x.x),bitcast<f32>(y.x),bitcast<f32>(z.x),f32(x.w|y.w|z.w)));
}
`;
}
