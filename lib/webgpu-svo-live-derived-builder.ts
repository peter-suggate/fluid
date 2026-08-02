import { SPARSE_BRICK_GPU_LAYOUT, type SparseBrickOctreeGPU, type SparseBrickSize } from "./sparse-brick-octree";
import { SVO_BRICK_LIFECYCLE } from "./svo-brick-occupancy";
import { SVO_NODE_MIP_LAYOUT } from "./svo-node-mip-pyramid";
import type { WebGpuLiveSvoNodeMipGpuTarget } from "./webgpu-svo-node-mip-pyramid";
import type { WebGpuLiveSvoTetrahedralRadianceGpuTarget } from "./webgpu-svo-tetrahedral-radiance";

export const LIVE_SVO_DERIVED_WORKLIST = Object.freeze({
  countWord: 0,
  generationWord: 1,
  levelWord: 2,
  recordOffsetWord: 3,
  dispatchIndirectOffsetBytes: 16,
  headerWords: 8,
  recordWords: 12,
  destinationSlotWord: 0,
  sourcePageCoordinateWord: 1,
  sourceLeafWord: 4,
  childSlotWord: 2,
  invalidIndex: 0xffff_ffff,
} as const);

/**
 * One fixed GPU-authored worklist per virtual mip level. Records must be unique
 * by destination slot. Level zero supplies its virtual page coordinate and
 * resolves the deepest active leaf per texel; parent records supply eight child
 * slots in xyz-bit order. Dispatch args cover count * 10^3 threads.
 */
export interface LiveSvoDerivedGpuWorklist {
  buffer: GPUBuffer;
  capacity: number;
  bindingOffsetBytes?: number;
  bindingSizeBytes?: number;
  indirectOffsetBytes?: number;
}

export interface WebGpuLiveSvoDerivedBuilderOptions {
  tree: SparseBrickOctreeGPU;
  nodeMips: WebGpuLiveSvoNodeMipGpuTarget;
  radiance: WebGpuLiveSvoTetrahedralRadianceGpuTarget;
  /** vec4f per material ID: linear emissive RGB plus unused alpha. */
  materialEmission: GPUBuffer;
  worklists: readonly LiveSvoDerivedGpuWorklist[];
  /** GPU publication generation used to certify newly allocated empty pages. */
  generationSource: LiveSvoDerivedGenerationSource;
  /** Number of address-plan slots resident in the fixed atlas. */
  plannedPageCount: number;
  /** Finest canonical tree level used for cell-to-leaf traversal. */
  finestLevel: number;
  /** Optional compact scratch page grid; defaults to the smallest grid that fits the largest level worklist. */
  scratchAtlasPages?: readonly [number, number, number];
  label?: string;
}

export interface LiveSvoDirtyLeafSource {
  buffer: GPUBuffer;
  countOffsetBytes: number;
  recordOffsetBytes: number;
  capacity: number;
  /** Source records contain leafIndex in word zero and use this stride. */
  recordStrideWords: number;
}

/** One nonzero, monotonically increasing generation shared by every dirty lane. */
export interface LiveSvoDerivedGenerationSource {
  buffer: GPUBuffer;
  offsetBytes: number;
}

export interface WebGpuLiveSvoDerivedWorklistPlannerOptions {
  tree: SparseBrickOctreeGPU;
  nodeMips: WebGpuLiveSvoNodeMipGpuTarget;
  /** Scene, fluid, and topology dirty lanes are unioned before any page build. */
  dirtyLeafSources: readonly LiveSvoDirtyLeafSource[];
  generationSource: LiveSvoDerivedGenerationSource;
  levelCount: number;
  finestLevel: number;
  pageCapacityPerLevel: number;
  label?: string;
}

/** CPU mirror of the GPU dirty-leaf coverage used by generic planning tests. */
export function liveSvoLeafBasePages(options: {
  coordinate: readonly [number, number, number];
  leafLevel: number;
  finestLevel: number;
  brickSize: SparseBrickSize;
}): readonly (readonly [number, number, number])[] {
  const { coordinate, leafLevel, finestLevel, brickSize } = options;
  if (!Number.isInteger(leafLevel) || !Number.isInteger(finestLevel) || leafLevel < 0 || leafLevel > finestLevel || finestLevel > 21) {
    throw new RangeError("Live derived leaf levels must be ordered integers in [0, 21]");
  }
  if (coordinate.some((value) => !Number.isSafeInteger(value) || value < 0)) {
    throw new RangeError("Live derived leaf coordinate must be nonnegative safe integers");
  }
  const scale = 2 ** (finestLevel - leafLevel);
  const first = coordinate.map((value) => Math.floor(value * scale * brickSize / SVO_NODE_MIP_LAYOUT.interiorSize));
  const last = coordinate.map((value) => Math.floor(((value + 1) * scale * brickSize - 1) / SVO_NODE_MIP_LAYOUT.interiorSize));
  const pages: [number, number, number][] = [];
  for (let z = first[2]; z <= last[2]; z += 1) for (let y = first[1]; y <= last[1]; y += 1) for (let x = first[0]; x <= last[0]; x += 1) {
    pages.push([x, y, z]);
  }
  return pages;
}

export const liveSvoDerivedWorklistWGSL = /* wgsl */ `
struct Params{source:vec4u,domain:vec4u,limits:vec4u,zOffsets:array<u32,12>,sections:array<u32,12>}
@group(0) @binding(0) var<storage,read> source:array<u32>;
@group(0) @binding(1) var<storage,read> control:array<u32>;
@group(0) @binding(2) var<storage,read> topology:array<u32>;
@group(0) @binding(3) var directTable:texture_3d<u32>;
@group(0) @binding(4) var<storage,read_write> claims:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> output:array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params:Params;
@group(0) @binding(7) var<storage,read> generationState:array<u32>;
const INVALID:u32=0xffffffffu;const ACTIVE:u32=${SVO_BRICK_LIFECYCLE.activeBit}u;const RECORD_WORDS:u32=${LIVE_SVO_DERIVED_WORKLIST.recordWords}u;
fn keyBit(lo:u32,hi:u32,bit:u32)->u32{if(bit>=32u){return(hi>>(bit-32u))&1u;}return(lo>>bit)&1u;}
fn morton(lo:u32,hi:u32,level:u32)->vec3u{var p=vec3u(0u);for(var bit=0u;bit<level;bit+=1u){let s=1u<<bit;p+=vec3u(keyBit(lo,hi,3u*bit),keyBit(lo,hi,3u*bit+1u),keyBit(lo,hi,3u*bit+2u))*s;}return p;}
fn tableSlot(level:u32,p:vec3u)->u32{if(level>=params.domain.y||p.x>=params.domain.z||p.y>=params.domain.w){return INVALID;}let dims=textureDimensions(directTable);let levelStart=params.zOffsets[level];var levelEnd=dims.z;if(level+1u<params.domain.y){levelEnd=params.zOffsets[level+1u];}if(p.z>=levelEnd-levelStart){return INVALID;}let z=levelStart+p.z;let value=textureLoad(directTable,vec3i(i32(p.x),i32(p.y),i32(z)),0).x;return select(INVALID,value-1u,value>0u);}
fn deepestLeaf(globalCell:vec3u)->u32{let brickSize=control[11];if(brickSize==0u||control[0]==0u){return INVALID;}let brick=globalCell/brickSize;var node=0u;var selected=INVALID;
 for(var level=0u;level<=params.domain.x&&node<control[0];level+=1u){let nodeBase=node*8u;let leaf=topology[nodeBase+6u];if(leaf<control[1]&&(topology[nodeBase+7u]&ACTIVE)!=0u){selected=leaf;}if(level==params.domain.x){break;}let bit=params.domain.x-level-1u;let octant=((brick.x>>bit)&1u)|(((brick.y>>bit)&1u)<<1u)|(((brick.z>>bit)&1u)<<2u);let mask=topology[nodeBase+3u]&0xffu;if((mask&(1u<<octant))==0u){break;}node=topology[nodeBase+4u]+countOneBits(mask&((1u<<octant)-1u));}
 return selected;}
fn emitPage(level:u32,page:vec3u,generation:u32){let slot=tableSlot(level,page);if(slot==INVALID||slot>=arrayLength(&claims)){return;}if(atomicExchange(&claims[slot],generation)==generation){return;}
 let section=params.sections[level];let recordIndex=atomicAdd(&output[section],1u);if(recordIndex>=params.limits.y){return;}let record=section+${LIVE_SVO_DERIVED_WORKLIST.headerWords}u+recordIndex*RECORD_WORDS;atomicStore(&output[record],slot);
 if(level==0u){atomicStore(&output[record+1u],page.x);atomicStore(&output[record+2u],page.y);atomicStore(&output[record+3u],page.z);for(var octant=0u;octant<8u;octant+=1u){let bit=vec3u(octant&1u,(octant>>1u)&1u,(octant>>2u)&1u);atomicStore(&output[record+4u+octant],deepestLeaf(page*${SVO_NODE_MIP_LAYOUT.interiorSize}u+bit*4u));}}
 else{for(var child=0u;child<8u;child+=1u){let bit=vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u);atomicStore(&output[record+2u+child],tableSlot(level-1u,page*2u+bit));}}
}
@compute @workgroup_size(64) fn populate(@builtin(global_invocation_id) gid:vec3u){let dirty=gid.x;let allLive=params.limits.w!=0u;let count=select(min(source[params.source.x],params.limits.x),min(control[1],params.limits.x),allLive);if(dirty>=count){return;}let generation=generationState[params.source.y];if(generation==0u){return;}
 var leaf=dirty;if(!allLive){leaf=source[params.source.z+dirty*params.source.w];}if(leaf>=control[1]){return;}let leafBase=control[16]+leaf*4u;let node=topology[leafBase];let leafLevel=topology[node*8u+2u];if(leafLevel>params.domain.x){return;}let scale=1u<<(params.domain.x-leafLevel);let brickOrigin=morton(topology[leafBase+2u],topology[leafBase+3u],leafLevel)*scale;let cellMinimum=brickOrigin*control[11];let cellMaximum=(brickOrigin+vec3u(scale))*control[11];let firstPage=cellMinimum/${SVO_NODE_MIP_LAYOUT.interiorSize}u;let lastPage=(cellMaximum-vec3u(1u))/${SVO_NODE_MIP_LAYOUT.interiorSize}u;
 for(var z=firstPage.z;z<=lastPage.z;z+=1u){for(var y=firstPage.y;y<=lastPage.y;y+=1u){for(var x=firstPage.x;x<=lastPage.x;x+=1u){var page=vec3u(x,y,z);for(var level=0u;level<params.domain.y;level+=1u){emitPage(level,page,generation);page>>=vec3u(1u);}}}}
}
@compute @workgroup_size(1) fn finalize(@builtin(global_invocation_id) gid:vec3u){let level=gid.x;if(level>=params.domain.y){return;}let section=params.sections[level];let count=min(atomicLoad(&output[section]),params.limits.y);atomicStore(&output[section+1u],generationState[params.source.y]);atomicStore(&output[section+2u],level);atomicStore(&output[section+3u],${LIVE_SVO_DERIVED_WORKLIST.headerWords}u);let groups=(count*${SVO_NODE_MIP_LAYOUT.physicalSize ** 3}u+255u)/256u;let x=min(groups,65535u);atomicStore(&output[section+4u],x);atomicStore(&output[section+5u],select(0u,(groups+x-1u)/x,x>0u));atomicStore(&output[section+6u],1u);}
`;

export const liveSvoDerivedBuildWGSL = /* wgsl */ `
struct Params{targetAtlasPages:vec4u,scratchAtlasPages:vec4u,limits:vec4u,laneOffsets:vec4u}
@group(0) @binding(0) var<storage,read> control:array<u32>;
@group(0) @binding(1) var<storage,read> topology:array<u32>;
@group(0) @binding(2) var<storage,read> payload:array<u32>;
@group(0) @binding(3) var<storage,read> worklist:array<u32>;
@group(0) @binding(4) var<storage,read> emission:array<vec4f>;
@group(0) @binding(5) var opacitySource:texture_3d<f32>;
@group(0) @binding(6) var opacityScratch:texture_storage_3d<rgba8unorm,write>;
@group(0) @binding(7) var radianceSource0:texture_3d<f32>;
@group(0) @binding(8) var radianceSource1:texture_3d<f32>;
@group(0) @binding(9) var radianceSource2:texture_3d<f32>;
@group(0) @binding(10) var radianceSource3:texture_3d<f32>;
@group(0) @binding(11) var radianceScratch:texture_storage_3d<rgba16float,write>;
@group(0) @binding(12) var<uniform> params:Params;
@group(0) @binding(13) var opacityValidity:texture_2d<u32>;
@group(0) @binding(14) var radianceValidity:texture_2d<u32>;
@group(0) @binding(15) var<storage,read_write> scratchValidity:array<u32>;

const INVALID:u32=0xffffffffu;
const PHYSICAL:u32=${SVO_NODE_MIP_LAYOUT.physicalSize}u;
const INTERIOR:u32=${SVO_NODE_MIP_LAYOUT.interiorSize}u;
const RECORD_WORDS:u32=${LIVE_SVO_DERIVED_WORKLIST.recordWords}u;
const ACTIVE:u32=${SVO_BRICK_LIFECYCLE.activeBit}u;
const TETRA0:vec3f=vec3f(.577350269,.577350269,.577350269);
const TETRA1:vec3f=vec3f(.577350269,-.577350269,-.577350269);
const TETRA2:vec3f=vec3f(-.577350269,.577350269,-.577350269);
const TETRA3:vec3f=vec3f(-.577350269,-.577350269,.577350269);

fn linearIndex(gid:vec3u,groups:vec3u)->u32{return gid.x+gid.y*groups.x*256u+gid.z*groups.x*groups.y*256u;}
fn slotOrigin(slot:u32)->vec3u{
  let pages=params.targetAtlasPages.xyz;
  return vec3u(slot%pages.x,(slot/pages.x)%pages.y,slot/(pages.x*pages.y))*PHYSICAL;
}
fn scratchOrigin(recordIndex:u32)->vec3u{
  let pages=params.scratchAtlasPages.xyz;
  return vec3u(recordIndex%pages.x,(recordIndex/pages.x)%pages.y,recordIndex/(pages.x*pages.y))*PHYSICAL;
}
fn physicalCoordinate(index:u32)->vec3u{return vec3u(index%PHYSICAL,(index/PHYSICAL)%PHYSICAL,index/(PHYSICAL*PHYSICAL));}
fn interiorCoordinate(physical:vec3u)->vec3u{return clamp(physical,vec3u(1u),vec3u(INTERIOR))-1u;}
fn keyBit(lo:u32,hi:u32,bit:u32)->u32{if(bit>=32u){return(hi>>(bit-32u))&1u;}return(lo>>bit)&1u;}
fn morton(lo:u32,hi:u32,level:u32)->vec3u{var p=vec3u(0u);for(var bit=0u;bit<level;bit+=1u){let s=1u<<bit;p+=vec3u(keyBit(lo,hi,3u*bit),keyBit(lo,hi,3u*bit+1u),keyBit(lo,hi,3u*bit+2u))*s;}return p;}
fn leafLocal(globalCell:vec3u,leaf:u32)->vec3u{let brickSize=control[11];let finest=params.scratchAtlasPages.w;let node=topology[control[16]+leaf*4u];let nodeBase=node*8u;let level=topology[nodeBase+2u];let scale=1u<<(finest-level);let origin=morton(topology[nodeBase],topology[nodeBase+1u],level)*scale*brickSize;return min((globalCell-origin)/scale,vec3u(brickSize-1u));}
fn leafVoxel(leafIndex:u32,local:vec3u)->u32{
  let leafBase=control[16]+leafIndex*4u;
  return topology[leafBase+1u]+local.x+local.y*control[11]+local.z*control[11]*control[11];
}
fn solidDistance(leafIndex:u32,local:vec3u)->f32{let voxel=leafVoxel(leafIndex,local);return min(bitcast<f32>(payload[voxel*4u+1u]),bitcast<f32>(payload[params.laneOffsets.z+voxel*4u+1u]));}
fn safeNormal(leafIndex:u32,local:vec3u)->vec3f{
  let lo=max(local,vec3u(1u))-1u;let hi=min(local+1u,vec3u(control[11]-1u));
  let gradient=vec3f(solidDistance(leafIndex,vec3u(hi.x,local.y,local.z))-solidDistance(leafIndex,vec3u(lo.x,local.y,local.z)),
    solidDistance(leafIndex,vec3u(local.x,hi.y,local.z))-solidDistance(leafIndex,vec3u(local.x,lo.y,local.z)),
    solidDistance(leafIndex,vec3u(local.x,local.y,hi.z))-solidDistance(leafIndex,vec3u(local.x,local.y,lo.z)));
  return select(vec3f(0.,1.,0.),normalize(gradient),dot(gradient,gradient)>1e-12);
}
fn childSample(record:u32,local:vec3u,sampleIndex:u32)->vec4f{
  let bit=vec3u(sampleIndex&1u,(sampleIndex>>1u)&1u,(sampleIndex>>2u)&1u);
  let fine=local*2u+bit;let octant=(fine.x/INTERIOR)|((fine.y/INTERIOR)<<1u)|((fine.z/INTERIOR)<<2u);
  let child=worklist[record+${LIVE_SVO_DERIVED_WORKLIST.childSlotWord}u+octant];
  if(child==INVALID){return vec4f(0.);}
  return textureLoad(opacitySource,vec3i(slotOrigin(child)+(fine%INTERIOR)+1u),0);
}
fn childRadiance(record:u32,local:vec3u,sampleIndex:u32,lobe:u32)->vec3f{
  let bit=vec3u(sampleIndex&1u,(sampleIndex>>1u)&1u,(sampleIndex>>2u)&1u);
  let fine=local*2u+bit;let octant=(fine.x/INTERIOR)|((fine.y/INTERIOR)<<1u)|((fine.z/INTERIOR)<<2u);
  let child=worklist[record+${LIVE_SVO_DERIVED_WORKLIST.childSlotWord}u+octant];
  if(child==INVALID){return vec3f(0.);}
  let coordinate=vec3i(slotOrigin(child)+(fine%INTERIOR)+1u);
  if(lobe==0u){return textureLoad(radianceSource0,coordinate,0).rgb;}
  if(lobe==1u){return textureLoad(radianceSource1,coordinate,0).rgb;}
  if(lobe==2u){return textureLoad(radianceSource2,coordinate,0).rgb;}
  return textureLoad(radianceSource3,coordinate,0).rgb;
}
fn writeRadiance(coordinate:vec3u,a:vec3f,b:vec3f,c:vec3f,d:vec3f){
  let depth=params.scratchAtlasPages.z*PHYSICAL;
  textureStore(radianceScratch,coordinate,vec4f(a,1.));textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth),vec4f(b,1.));
  textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth*2u),vec4f(c,1.));textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth*3u),vec4f(d,1.));
}
fn referencedChildrenReady(record:u32)->bool{
  for(var childIndex=0u;childIndex<8u;childIndex+=1u){let child=worklist[record+${LIVE_SVO_DERIVED_WORKLIST.childSlotWord}u+childIndex];
    if(child!=INVALID&&(textureLoad(opacityValidity,vec2i(i32(child),0),0).x==0u||textureLoad(radianceValidity,vec2i(i32(child),0),0).x==0u)){return false;}}
  return true;
}

@compute @workgroup_size(256)
fn buildPages(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
  let index=linearIndex(gid,groups);let texels=PHYSICAL*PHYSICAL*PHYSICAL;
  let recordIndex=index/texels;let count=min(worklist[0],params.limits.x);if(recordIndex>=count){return;}
  let physical=physicalCoordinate(index-recordIndex*texels);let local=interiorCoordinate(physical);
  let record=worklist[3]+recordIndex*RECORD_WORDS;let destination=scratchOrigin(recordIndex)+physical;
  if(worklist[2]==0u){
    if(all(physical==vec3u(0u))){scratchValidity[recordIndex]=worklist[1];}let page=vec3u(worklist[record+1u],worklist[record+2u],worklist[record+3u]);let octant=(local.x/4u)|((local.y/4u)<<1u)|((local.z/4u)<<2u);let leaf=worklist[record+${LIVE_SVO_DERIVED_WORKLIST.sourceLeafWord}u+octant];if(leaf>=control[1]){textureStore(opacityScratch,destination,vec4f(0.));writeRadiance(destination,vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.));return;}let sampleLocal=leafLocal(page*INTERIOR+local,leaf);
    let voxel=leafVoxel(leaf,sampleLocal);let dynamicSolid=clamp(bitcast<f32>(payload[voxel*4u+2u]),0.,1.);let sceneSolid=clamp(bitcast<f32>(payload[params.laneOffsets.z+voxel*4u+2u]),0.,1.);
    let fluid=clamp(bitcast<f32>(payload[params.laneOffsets.x+voxel*4u+3u]),0.,1.);let solid=1.-(1.-dynamicSolid)*(1.-sceneSolid);
    textureStore(opacityScratch,destination,vec4f(solid,select(0.,1.,solid>0.),clamp(fluid,0.,1.),select(0.,1.,fluid>0.)));
    let dynamicIdentity=payload[params.laneOffsets.y+voxel];let sceneIdentity=payload[params.laneOffsets.w+voxel];
    let material=select(dynamicIdentity&0xffffu,sceneIdentity&0xffffu,sceneSolid>=dynamicSolid&&sceneSolid>0.);var emitted=vec3f(0.);
    if(material<arrayLength(&emission)){emitted=max(emission[material].rgb,vec3f(0.));}
    let normal=safeNormal(leaf,sampleLocal);let covered=solid;
    writeRadiance(destination,emitted*covered*max(0.,dot(normal,TETRA0)),emitted*covered*max(0.,dot(normal,TETRA1)),
      emitted*covered*max(0.,dot(normal,TETRA2)),emitted*covered*max(0.,dot(normal,TETRA3)));
    return;
  }
  if(all(physical==vec3u(0u))){scratchValidity[recordIndex]=select(0u,worklist[1],referencedChildrenReady(record));}
  var mean=vec2f(0.);var maximum=vec2f(0.);var r0=vec3f(0.);var r1=vec3f(0.);var r2=vec3f(0.);var r3=vec3f(0.);
  for(var child=0u;child<8u;child+=1u){let value=childSample(record,local,child);mean+=value.xz/8.;maximum=max(maximum,value.yw);
    r0+=childRadiance(record,local,child,0u)/8.;r1+=childRadiance(record,local,child,1u)/8.;
    r2+=childRadiance(record,local,child,2u)/8.;r3+=childRadiance(record,local,child,3u)/8.;}
  textureStore(opacityScratch,destination,vec4f(mean.x,maximum.x,mean.y,maximum.y));writeRadiance(destination,r0,r1,r2,r3);
}`;

export const liveSvoDerivedCopyWGSL = /* wgsl */ `
struct Params{targetAtlasPages:vec4u,scratchAtlasPages:vec4u,limits:vec4u,laneOffsets:vec4u}
@group(0) @binding(0) var<storage,read> worklist:array<u32>;
@group(0) @binding(1) var opacityScratch:texture_3d<f32>;
@group(0) @binding(2) var opacityDestination:texture_storage_3d<rgba8unorm,write>;
@group(0) @binding(3) var radianceScratch:texture_3d<f32>;
@group(0) @binding(4) var radianceDestination:texture_storage_3d<rgba16float,write>;
@group(0) @binding(5) var opacityValidity:texture_storage_2d<r32uint,write>;
@group(0) @binding(6) var radianceValidity:texture_storage_2d<r32uint,write>;
@group(0) @binding(7) var<uniform> params:Params;
@group(0) @binding(8) var<storage,read> scratchValidity:array<u32>;
const PHYSICAL:u32=${SVO_NODE_MIP_LAYOUT.physicalSize}u;const RECORD_WORDS:u32=${LIVE_SVO_DERIVED_WORKLIST.recordWords}u;
fn linearIndex(gid:vec3u,groups:vec3u)->u32{return gid.x+gid.y*groups.x*256u+gid.z*groups.x*groups.y*256u;}
fn targetOrigin(slot:u32)->vec3u{let p=params.targetAtlasPages.xyz;return vec3u(slot%p.x,(slot/p.x)%p.y,slot/(p.x*p.y))*PHYSICAL;}
fn scratchOrigin(slot:u32)->vec3u{let p=params.scratchAtlasPages.xyz;return vec3u(slot%p.x,(slot/p.x)%p.y,slot/(p.x*p.y))*PHYSICAL;}
fn physical(index:u32)->vec3u{return vec3u(index%PHYSICAL,(index/PHYSICAL)%PHYSICAL,index/(PHYSICAL*PHYSICAL));}
@compute @workgroup_size(256) fn invalidatePages(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;let count=min(worklist[0],params.limits.x);if(i>=count){return;}let record=worklist[3]+i*RECORD_WORDS;let slot=worklist[record];textureStore(opacityValidity,vec2i(i32(slot),0),vec4u(0u));textureStore(radianceValidity,vec2i(i32(slot),0),vec4u(0u));}
@compute @workgroup_size(256) fn copyOpacity(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let index=linearIndex(gid,groups);let texels=PHYSICAL*PHYSICAL*PHYSICAL;let recordIndex=index/texels;let count=min(worklist[0],params.limits.x);if(recordIndex>=count){return;}
 let local=physical(index-recordIndex*texels);let record=worklist[3]+recordIndex*RECORD_WORDS;let slot=worklist[record];let source=vec3i(scratchOrigin(recordIndex)+local);let destination=vec3i(targetOrigin(slot)+local);
 textureStore(opacityDestination,destination,textureLoad(opacityScratch,source,0));if(all(local==vec3u(0u))&&scratchValidity[recordIndex]!=0u){textureStore(opacityValidity,vec2i(i32(slot),0),vec4u(scratchValidity[recordIndex]));}}
fn copyRadianceLobe(gid:vec3u,groups:vec3u,lobe:u32){let index=linearIndex(gid,groups);let texels=PHYSICAL*PHYSICAL*PHYSICAL;let recordIndex=index/texels;let count=min(worklist[0],params.limits.x);if(recordIndex>=count){return;}
 let local=physical(index-recordIndex*texels);let record=worklist[3]+recordIndex*RECORD_WORDS;let slot=worklist[record];let scratch=scratchOrigin(recordIndex)+local+vec3u(0u,0u,lobe*params.scratchAtlasPages.z*PHYSICAL);let destination=vec3i(targetOrigin(slot)+local);
 textureStore(radianceDestination,destination,textureLoad(radianceScratch,vec3i(scratch),0));if(lobe==3u&&all(local==vec3u(0u))&&scratchValidity[recordIndex]!=0u){textureStore(radianceValidity,vec2i(i32(slot),0),vec4u(scratchValidity[recordIndex]));}}
@compute @workgroup_size(256) fn copyRadiance0(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){copyRadianceLobe(gid,groups,0u);}
@compute @workgroup_size(256) fn copyRadiance1(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){copyRadianceLobe(gid,groups,1u);}
@compute @workgroup_size(256) fn copyRadiance2(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){copyRadianceLobe(gid,groups,2u);}
@compute @workgroup_size(256) fn copyRadiance3(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){copyRadianceLobe(gid,groups,3u);}`;

/**
 * New WebGPU textures are logically zero-initialized. This runtime pass
 * certifies those zero pages against the current live publication before any
 * occupied-page worklist invalidates and overwrites them. It is deliberately
 * independent of scene contents: every address-resident page starts as empty,
 * and ordinary dirty propagation owns every later edit.
 */
export const liveSvoDerivedEmptyInitializationWGSL = /* wgsl */ `
struct Params{targetAtlasPages:vec4u,scratchAtlasPages:vec4u,limits:vec4u,laneOffsets:vec4u}
@group(0) @binding(0) var opacityValidity:texture_storage_2d<r32uint,write>;
@group(0) @binding(1) var radianceValidity:texture_storage_2d<r32uint,write>;
@group(0) @binding(2) var<storage,read> generationState:array<u32>;
@group(0) @binding(3) var<uniform> params:Params;
@compute @workgroup_size(256) fn initializeValidEmptyPages(@builtin(global_invocation_id) gid:vec3u){
  let slot=gid.x;if(slot>=params.limits.w){return;}let generation=generationState[params.limits.z];if(generation==0u){return;}
  textureStore(opacityValidity,vec2i(i32(slot),0),vec4u(generation));
  textureStore(radianceValidity,vec2i(i32(slot),0),vec4u(generation));
}`;

interface LevelBindings { worklist: LiveSvoDerivedGpuWorklist; invalidate: GPUBindGroup; build: GPUBindGroup; opacityCopy: GPUBindGroup; radianceCopy: readonly GPUBindGroup[] }

interface LiveSvoDerivedBuilderPipelineState {
  emptyInitializationPipeline: GPUComputePipeline;
  emptyInitializationBindGroup: GPUBindGroup;
  buildPipeline: GPUComputePipeline;
  invalidatePipeline: GPUComputePipeline;
  opacityCopyPipeline: GPUComputePipeline;
  radianceCopyPipelines: readonly GPUComputePipeline[];
  levels: readonly LevelBindings[];
}

function compactScratchAtlasPages(capacity: number, maximumTextureDimension3D: number): readonly [number, number, number] {
  const physical = SVO_NODE_MIP_LAYOUT.physicalSize;
  const maximumXyPages = Math.floor(maximumTextureDimension3D / physical);
  const maximumZPages = Math.floor(maximumTextureDimension3D / (physical * 4));
  if (maximumXyPages < 1 || maximumZPages < 1 || capacity > maximumXyPages * maximumXyPages * maximumZPages) {
    throw new RangeError("Live SVO derived scratch work budget exceeds the device 3D texture limit");
  }
  const x = Math.min(capacity, maximumXyPages);
  const y = Math.min(Math.ceil(capacity / x), maximumXyPages);
  return [x, y, Math.ceil(capacity / (x * y))];
}

/** GPU-only incremental builder; work is exactly bounded by GPU worklist counts. */
export class WebGpuLiveSvoDerivedBuilder {
  readonly allocatedBytes: number;
  readonly scratchAtlasPages: readonly [number, number, number];
  private readonly plannedPageCount: number;
  private readonly opacityScratch: GPUTexture;
  private readonly radianceScratch: GPUTexture;
  private readonly scratchValidity: GPUBuffer;
  private readonly params: GPUBuffer;
  private pipelineState?: LiveSvoDerivedBuilderPipelineState;
  private pipelineInitialization?: Promise<void>;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly options: WebGpuLiveSvoDerivedBuilderOptions) {
    if (options.nodeMips.pageCapacity !== options.radiance.pageCapacity) throw new Error("Live opacity and radiance page capacities must match");
    if (options.nodeMips.atlasPages.some((value, axis) => value !== options.radiance.atlasPages[axis])) throw new Error("Live opacity and radiance atlas shapes must match");
    if (options.nodeMips.atlasPages.reduce((product, value) => product * value, 1) < options.nodeMips.pageCapacity) {
      throw new RangeError("Live derived target atlas does not contain its declared page capacity");
    }
    if (options.worklists.length < 1) throw new RangeError("At least one live derived worklist is required");
    if (!Number.isSafeInteger(options.plannedPageCount) || options.plannedPageCount < 1
      || options.plannedPageCount > options.nodeMips.pageCapacity) {
      throw new RangeError("Live derived planned page count must fit the target page atlas");
    }
    this.plannedPageCount = options.plannedPageCount;
    if (!Number.isInteger(options.generationSource.offsetBytes) || options.generationSource.offsetBytes < 0
      || options.generationSource.offsetBytes % 4 !== 0) {
      throw new RangeError("Live derived generation offset must be a nonnegative u32-aligned byte offset");
    }
    if (!Number.isInteger(options.finestLevel) || options.finestLevel < 0 || options.finestLevel > 21) {
      throw new RangeError("Live derived finest level must be an integer in [0, 21]");
    }
    const scratchCapacity = Math.max(...options.worklists.map(({ capacity }) => capacity));
    if (!Number.isSafeInteger(scratchCapacity) || scratchCapacity < 1 || scratchCapacity > options.nodeMips.pageCapacity) {
      throw new RangeError("Live derived worklist capacity must fit the target page atlas");
    }
    const maximumTextureDimension3D = Number(device.limits?.maxTextureDimension3D) || 2_048;
    this.scratchAtlasPages = options.scratchAtlasPages ?? compactScratchAtlasPages(scratchCapacity, maximumTextureDimension3D);
    if (this.scratchAtlasPages.some((value) => !Number.isSafeInteger(value) || value < 1)
      || this.scratchAtlasPages.reduce((product, value) => product * value, 1) < scratchCapacity
      || this.scratchAtlasPages[0] * SVO_NODE_MIP_LAYOUT.physicalSize > maximumTextureDimension3D
      || this.scratchAtlasPages[1] * SVO_NODE_MIP_LAYOUT.physicalSize > maximumTextureDimension3D
      || this.scratchAtlasPages[2] * SVO_NODE_MIP_LAYOUT.physicalSize * 4 > maximumTextureDimension3D) {
      throw new RangeError("Live derived scratch atlas cannot contain the bounded worklist on this device");
    }
    const label = options.label ?? "Live SVO derived content";
    const scratchTexels = this.scratchAtlasPages.map((value) => value * SVO_NODE_MIP_LAYOUT.physicalSize) as [number, number, number];
    this.opacityScratch = device.createTexture({ label: `${label} opacity scratch`, size: scratchTexels, dimension: "3d", format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    this.radianceScratch = device.createTexture({ label: `${label} radiance scratch`, size: [scratchTexels[0], scratchTexels[1], scratchTexels[2] * 4],
      dimension: "3d", format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    this.scratchValidity = device.createBuffer({ label: `${label} scratch page validity`, size: scratchCapacity * 4,
      usage: GPUBufferUsage.STORAGE });
    this.params = device.createBuffer({ label: `${label} parameters`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.params, 0, new Uint32Array([
      ...options.nodeMips.atlasPages, 0,
      ...this.scratchAtlasPages, options.finestLevel,
      options.nodeMips.pageCapacity, SVO_NODE_MIP_LAYOUT.physicalSize,
      options.generationSource.offsetBytes / 4, options.plannedPageCount,
      options.tree.velocityOffsetBytes / 4, options.tree.materialOwnerOffsetBytes / 4,
      options.tree.sceneGeometryOffsetBytes / 4, options.tree.sceneMaterialOwnerOffsetBytes / 4,
    ]));
    const texelCount = scratchTexels[0] * scratchTexels[1] * scratchTexels[2];
    this.allocatedBytes = texelCount * (4 + 4 * 8) + scratchCapacity * 4 + 64;
  }

  async initializePipelines(): Promise<void> {
    if (this.pipelineState) return;
    if (this.destroyed) throw new Error("Cannot initialize destroyed live SVO derived builder");
    if (!this.pipelineInitialization) {
      this.pipelineInitialization = this.compilePipelineState().then((state) => {
        if (this.destroyed) throw new Error("Live SVO derived builder was destroyed during pipeline initialization");
        this.pipelineState = state;
      });
    }
    try {
      await this.pipelineInitialization;
    } catch (error) {
      this.pipelineInitialization = undefined;
      throw error;
    }
  }

  private async compilePipelineState(): Promise<LiveSvoDerivedBuilderPipelineState> {
    const { device, options } = this;
    const label = options.label ?? "Live SVO derived content";
    const buildModule = device.createShaderModule({ label: `${label} build shader`, code: liveSvoDerivedBuildWGSL });
    const copyModule = device.createShaderModule({ label: `${label} copy shader`, code: liveSvoDerivedCopyWGSL });
    const emptyInitializationModule = device.createShaderModule({
      label: `${label} valid-empty initialization shader`, code: liveSvoDerivedEmptyInitializationWGSL,
    });
    const [emptyInitializationPipeline, buildPipeline, invalidatePipeline, opacityCopyPipeline, radianceCopyPipelines] = await Promise.all([
      device.createComputePipelineAsync({ label: `${label} valid-empty initialization pipeline`, layout: "auto",
        compute: { module: emptyInitializationModule, entryPoint: "initializeValidEmptyPages" } }),
      device.createComputePipelineAsync({ label: `${label} build pipeline`, layout: "auto", compute: { module: buildModule, entryPoint: "buildPages" } }),
      device.createComputePipelineAsync({ label: `${label} invalidate pipeline`, layout: "auto", compute: { module: copyModule, entryPoint: "invalidatePages" } }),
      device.createComputePipelineAsync({ label: `${label} opacity copy pipeline`, layout: "auto", compute: { module: copyModule, entryPoint: "copyOpacity" } }),
      Promise.all([0, 1, 2, 3].map((lobe) => device.createComputePipelineAsync({ label: `${label} radiance copy ${lobe} pipeline`,
        layout: "auto", compute: { module: copyModule, entryPoint: `copyRadiance${lobe}` } }))),
    ]);
    if (this.destroyed) throw new Error("Live SVO derived builder was destroyed during pipeline initialization");
    const emptyInitializationBindGroup = device.createBindGroup({
      layout: emptyInitializationPipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: options.nodeMips.pageValidity.view },
        { binding: 1, resource: options.radiance.pageValidity.view },
        { binding: 2, resource: { buffer: options.generationSource.buffer } },
        { binding: 3, resource: { buffer: this.params } },
      ],
    });
    const radianceScratchView = this.radianceScratch.createView({ dimension: "3d" });
    const targetRadianceViews = options.radiance.textures.map((texture) => texture.createView({ dimension: "3d" }));
    const levels = options.worklists.map((worklist) => ({ worklist,
      invalidate: device.createBindGroup({ layout: invalidatePipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: worklist.buffer, offset: worklist.bindingOffsetBytes ?? 0, size: worklist.bindingSizeBytes } },
        { binding: 5, resource: options.nodeMips.pageValidity.view }, { binding: 6, resource: options.radiance.pageValidity.view },
        { binding: 7, resource: { buffer: this.params } },
      ] }),
      build: device.createBindGroup({ layout: buildPipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: options.tree.control, size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes } }, { binding: 1, resource: { buffer: options.tree.topology, offset: options.tree.topologyOffsetBytes } },
        { binding: 2, resource: { buffer: options.tree.payload } }, { binding: 3, resource: { buffer: worklist.buffer, offset: worklist.bindingOffsetBytes ?? 0, size: worklist.bindingSizeBytes } },
        { binding: 4, resource: { buffer: options.materialEmission } }, { binding: 5, resource: options.nodeMips.texture.createView({ dimension: "3d" }) },
        { binding: 6, resource: this.opacityScratch.createView({ dimension: "3d" }) },
        ...targetRadianceViews.map((resource, index) => ({ binding: 7 + index, resource })),
        { binding: 11, resource: radianceScratchView }, { binding: 12, resource: { buffer: this.params } },
        { binding: 13, resource: options.nodeMips.pageValidity.view }, { binding: 14, resource: options.radiance.pageValidity.view },
        { binding: 15, resource: { buffer: this.scratchValidity } },
      ] }),
      opacityCopy: device.createBindGroup({ layout: opacityCopyPipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: worklist.buffer, offset: worklist.bindingOffsetBytes ?? 0, size: worklist.bindingSizeBytes } }, { binding: 1, resource: this.opacityScratch.createView({ dimension: "3d" }) },
        { binding: 2, resource: options.nodeMips.texture.createView({ dimension: "3d" }) },
        { binding: 5, resource: options.nodeMips.pageValidity.view }, { binding: 7, resource: { buffer: this.params } },
        { binding: 8, resource: { buffer: this.scratchValidity } },
      ] }),
      radianceCopy: radianceCopyPipelines.map((pipeline, lobe) => device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: worklist.buffer, offset: worklist.bindingOffsetBytes ?? 0, size: worklist.bindingSizeBytes } }, { binding: 3, resource: radianceScratchView },
        { binding: 4, resource: targetRadianceViews[lobe] }, { binding: 6, resource: options.radiance.pageValidity.view },
        { binding: 7, resource: { buffer: this.params } }, { binding: 8, resource: { buffer: this.scratchValidity } },
      ] })),
    }));
    return { emptyInitializationPipeline, emptyInitializationBindGroup, buildPipeline, invalidatePipeline,
      opacityCopyPipeline, radianceCopyPipelines, levels };
  }

  encode(encoder: GPUCommandEncoder, initializeEmpty = false): void {
    if (this.destroyed) return;
    const state = this.pipelineState;
    if (!state) throw new Error("Live SVO derived builder pipelines are not initialized");
    if (initializeEmpty) {
      const pass = encoder.beginComputePass({ label: "Certify live SVO address pages as empty" });
      pass.setPipeline(state.emptyInitializationPipeline); pass.setBindGroup(0, state.emptyInitializationBindGroup);
      pass.dispatchWorkgroups(Math.ceil(this.plannedPageCount / 256)); pass.end();
    }
    // All affected pages fail closed before any level is rebuilt.
    for (const level of state.levels) {
      const pass = encoder.beginComputePass({ label: "Invalidate live SVO derived pages" });
      pass.setPipeline(state.invalidatePipeline); pass.setBindGroup(0, level.invalidate);
      pass.dispatchWorkgroups(Math.ceil(level.worklist.capacity / 256)); pass.end();
    }
    // Finest-to-coarsest order is the caller's worklist order. Each copy pass
    // makes complete children visible before the next parent build begins.
    for (const level of state.levels) {
      const indirect = level.worklist.indirectOffsetBytes ?? (level.worklist.bindingOffsetBytes ?? 0) + LIVE_SVO_DERIVED_WORKLIST.dispatchIndirectOffsetBytes;
      const build = encoder.beginComputePass({ label: "Build live SVO derived pages" });
      build.setPipeline(state.buildPipeline); build.setBindGroup(0, level.build); build.dispatchWorkgroupsIndirect(level.worklist.buffer, indirect); build.end();
      const opacity = encoder.beginComputePass({ label: "Publish live SVO opacity pages" });
      opacity.setPipeline(state.opacityCopyPipeline); opacity.setBindGroup(0, level.opacityCopy); opacity.dispatchWorkgroupsIndirect(level.worklist.buffer, indirect); opacity.end();
      state.radianceCopyPipelines.forEach((pipeline, lobe) => {
        const radiance = encoder.beginComputePass({ label: `Publish live SVO radiance lobe ${lobe}` });
        radiance.setPipeline(pipeline); radiance.setBindGroup(0, level.radianceCopy[lobe]);
        radiance.dispatchWorkgroupsIndirect(level.worklist.buffer, indirect); radiance.end();
      });
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.opacityScratch.destroy(); this.radianceScratch.destroy(); this.scratchValidity.destroy(); this.params.destroy(); this.destroyed = true;
  }
}

function align256(value: number): number { return Math.ceil(value / 256) * 256; }

interface LiveSvoDerivedPlannerSourceAllocation {
  source: LiveSvoDirtyLeafSource;
  capacity: number;
  params: GPUBuffer;
}

interface LiveSvoDerivedPlannerPipelineState {
  populatePipeline: GPUComputePipeline;
  finalizePipeline: GPUComputePipeline;
  sources: readonly { capacity: number; bindGroup: GPUBindGroup }[];
  initial: { capacity: number; bindGroup: GPUBindGroup };
  finalizeBindGroup: GPUBindGroup;
}

/** GPU compaction/deduplication from a unified dirty-leaf stream into level worklists. */
export class WebGpuLiveSvoDerivedWorklistPlanner {
  readonly worklists: readonly LiveSvoDerivedGpuWorklist[];
  readonly allocatedBytes: number;
  private readonly arena: GPUBuffer;
  private readonly claims: GPUBuffer;
  private readonly sources: readonly LiveSvoDerivedPlannerSourceAllocation[];
  private readonly initial: LiveSvoDerivedPlannerSourceAllocation;
  private readonly sectionOffsetsBytes: readonly number[];
  private pipelineState?: LiveSvoDerivedPlannerPipelineState;
  private pipelineInitialization?: Promise<void>;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly options: WebGpuLiveSvoDerivedWorklistPlannerOptions) {
    if (!Number.isInteger(options.levelCount) || options.levelCount < 1 || options.levelCount > 12) throw new RangeError("Live derived level count must be in [1, 12]");
    if (!Number.isInteger(options.pageCapacityPerLevel) || options.pageCapacityPerLevel < 1) throw new RangeError("Live derived per-level capacity must be positive");
    if (options.dirtyLeafSources.length < 1) throw new RangeError("At least one live dirty-leaf source is required");
    for (const source of options.dirtyLeafSources) {
      if (!Number.isInteger(source.recordStrideWords) || source.recordStrideWords < 1) throw new RangeError("Dirty leaf stride must be positive");
      if (!Number.isInteger(source.capacity) || source.capacity < 1) throw new RangeError("Dirty leaf capacity must be positive");
      if ([source.countOffsetBytes, source.recordOffsetBytes].some((offset) => !Number.isInteger(offset) || offset < 0 || offset % 4 !== 0)) {
        throw new RangeError("Dirty leaf offsets must be nonnegative u32-aligned byte offsets");
      }
    }
    if (!Number.isInteger(options.generationSource.offsetBytes) || options.generationSource.offsetBytes < 0 || options.generationSource.offsetBytes % 4 !== 0) {
      throw new RangeError("Derived generation offset must be a nonnegative u32-aligned byte offset");
    }
    const sectionBytes = align256((LIVE_SVO_DERIVED_WORKLIST.headerWords + options.pageCapacityPerLevel * LIVE_SVO_DERIVED_WORKLIST.recordWords) * 4);
    this.sectionOffsetsBytes = Array.from({ length: options.levelCount }, (_, level) => level * sectionBytes);
    const label = options.label ?? "Live SVO derived worklists";
    this.arena = device.createBuffer({ label: `${label} arena`, size: sectionBytes * options.levelCount,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.claims = device.createBuffer({ label: `${label} slot generation claims`, size: options.nodeMips.pageCapacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.worklists = this.sectionOffsetsBytes.map((offset) => ({ buffer: this.arena, capacity: options.pageCapacityPerLevel,
      bindingOffsetBytes: offset, bindingSizeBytes: sectionBytes, indirectOffsetBytes: offset + LIVE_SVO_DERIVED_WORKLIST.dispatchIndirectOffsetBytes }));
    const createSource = (source: LiveSvoDirtyLeafSource, suffix: string) => {
      const params = device.createBuffer({ label: `${label} ${suffix} parameters`, size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      return { source, capacity: source.capacity, params };
    };
    this.sources = options.dirtyLeafSources.map((source, index) => createSource(source, `source ${index}`));
    const initialSource: LiveSvoDirtyLeafSource = { buffer: options.tree.control, countOffsetBytes: 0, recordOffsetBytes: 0,
      capacity: options.tree.leafCapacity, recordStrideWords: 1 };
    this.initial = createSource(initialSource, "initial all-leaves");
    this.configurePlan(options.nodeMips, options.finestLevel, options.levelCount);
    this.allocatedBytes = sectionBytes * options.levelCount + options.nodeMips.pageCapacity * 4 + 144 * (this.sources.length + 1);
  }

  async initializePipelines(): Promise<void> {
    if (this.pipelineState) return;
    if (this.destroyed) throw new Error("Cannot initialize destroyed live SVO derived worklist planner");
    if (!this.pipelineInitialization) {
      this.pipelineInitialization = this.compilePipelineState().then((state) => {
        if (this.destroyed) throw new Error("Live SVO derived worklist planner was destroyed during pipeline initialization");
        this.pipelineState = state;
      });
    }
    try {
      await this.pipelineInitialization;
    } catch (error) {
      this.pipelineInitialization = undefined;
      throw error;
    }
  }

  private async compilePipelineState(): Promise<LiveSvoDerivedPlannerPipelineState> {
    const { device, options } = this;
    const label = options.label ?? "Live SVO derived worklists";
    const plannerModule = device.createShaderModule({ label: `${label} shader`, code: liveSvoDerivedWorklistWGSL });
    const [populatePipeline, finalizePipeline] = await Promise.all([
      device.createComputePipelineAsync({ label: `${label} populate pipeline`, layout: "auto",
        compute: { module: plannerModule, entryPoint: "populate" } }),
      device.createComputePipelineAsync({ label: `${label} finalize pipeline`, layout: "auto",
        compute: { module: plannerModule, entryPoint: "finalize" } }),
    ]);
    if (this.destroyed) throw new Error("Live SVO derived worklist planner was destroyed during pipeline initialization");
    const bindSource = ({ source, capacity, params }: LiveSvoDerivedPlannerSourceAllocation) => ({ capacity,
      bindGroup: device.createBindGroup({ layout: populatePipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: source.buffer } },
        { binding: 1, resource: { buffer: options.tree.control, size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes } },
        { binding: 2, resource: { buffer: options.tree.topology, offset: options.tree.topologyOffsetBytes } },
        { binding: 3, resource: options.nodeMips.directPageTableTexture.createView({ dimension: "3d" }) },
        { binding: 4, resource: { buffer: this.claims } }, { binding: 5, resource: { buffer: this.arena } },
        { binding: 6, resource: { buffer: params } }, { binding: 7, resource: { buffer: options.generationSource.buffer } },
      ] }),
    });
    const sources = this.sources.map(bindSource);
    const initial = bindSource(this.initial);
    const finalizeBindGroup = device.createBindGroup({ layout: finalizePipeline.getBindGroupLayout(0), entries: [
      { binding: 5, resource: { buffer: this.arena } }, { binding: 6, resource: { buffer: this.sources[0].params } },
      { binding: 7, resource: { buffer: options.generationSource.buffer } },
    ] });
    return { populatePipeline, finalizePipeline, sources, initial, finalizeBindGroup };
  }

  configurePlan(target: WebGpuLiveSvoNodeMipGpuTarget, finestLevel = this.options.finestLevel, levelCount = this.options.levelCount): void {
    if (!Number.isInteger(finestLevel) || finestLevel < 0) throw new RangeError("Live derived finest level must be nonnegative");
    if (!Number.isInteger(levelCount) || levelCount < 1 || levelCount > this.options.levelCount) {
      throw new RangeError("Configured live derived levels must fit the fixed planner allocation");
    }
    const write = (params: GPUBuffer, source: LiveSvoDirtyLeafSource, allLive: boolean) => {
      const words = new Uint32Array(36);
      words.set([source.countOffsetBytes / 4, this.options.generationSource.offsetBytes / 4,
        source.recordOffsetBytes / 4, source.recordStrideWords], 0);
      words.set([finestLevel, levelCount, target.directPageTableDimensions[0], target.directPageTableDimensions[1]], 4);
      words.set([source.capacity, this.options.pageCapacityPerLevel, target.pageCapacity, allLive ? 1 : 0], 8);
      words.set(target.directPageTableLevelZOffsets.slice(0, 12), 12);
      words.set(this.sectionOffsetsBytes.map((offset) => offset / 4), 24);
      this.device.queue.writeBuffer(params, 0, words);
    };
    this.options.dirtyLeafSources.forEach((source, index) => write(this.sources[index].params, source, false));
    write(this.initial.params, { buffer: this.options.tree.control, countOffsetBytes: 0, recordOffsetBytes: 0,
      capacity: this.options.tree.leafCapacity, recordStrideWords: 1 }, true);
  }

  /** Compacts all supplied scene/fluid/topology streams into one deduplicated hierarchy. */
  encode(encoder: GPUCommandEncoder): void {
    this.encodeSources(encoder, false);
  }

  /** First publication: unions every currently live tree leaf with all supplied dirty streams entirely on GPU. */
  encodeInitial(encoder: GPUCommandEncoder): void {
    this.encodeSources(encoder, true);
  }

  private encodeSources(encoder: GPUCommandEncoder, includeAllLeaves: boolean): void {
    if (this.destroyed) return;
    const state = this.pipelineState;
    if (!state) throw new Error("Live SVO derived worklist planner pipelines are not initialized");
    for (const offset of this.sectionOffsetsBytes) encoder.clearBuffer(this.arena, offset, LIVE_SVO_DERIVED_WORKLIST.headerWords * 4);
    const populate = encoder.beginComputePass({ label: "Populate live SVO derived worklists" });
    populate.setPipeline(state.populatePipeline);
    if (includeAllLeaves) {
      populate.setBindGroup(0, state.initial.bindGroup); populate.dispatchWorkgroups(Math.ceil(state.initial.capacity / 64));
    }
    for (const source of state.sources) {
      populate.setBindGroup(0, source.bindGroup); populate.dispatchWorkgroups(Math.ceil(source.capacity / 64));
    }
    populate.end();
    const finalize = encoder.beginComputePass({ label: "Finalize live SVO derived worklists" });
    finalize.setPipeline(state.finalizePipeline); finalize.setBindGroup(0, state.finalizeBindGroup); finalize.dispatchWorkgroups(this.options.levelCount); finalize.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.arena.destroy(); this.claims.destroy(); this.sources.forEach(({ params }) => params.destroy()); this.initial.params.destroy(); this.destroyed = true;
  }
}
