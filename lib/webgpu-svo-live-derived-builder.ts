import {
  SPARSE_BRICK_GPU_LAYOUT,
  type SparseBrickOctreeGPU,
  type SparseBrickPlan,
  type SparseBrickSize,
} from "./sparse-brick-octree";
import { SVO_BRICK_LIFECYCLE } from "./svo-brick-occupancy";
import { svoEnvironmentLightingWGSL } from "./svo-environment-lighting";
import { svoLightWGSL } from "./svo-light-abi";
import { svoMaterialWGSL } from "./svo-material-abi";
import { SVO_NODE_MIP_LAYOUT } from "./svo-node-mip-pyramid";
import { svoTetrahedralRadianceWGSL } from "./svo-tetrahedral-radiance";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";
import { liveSvoDerivedPageValidityWGSL } from "./webgpu-svo-live-derived-cache";
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

export const LIVE_SVO_RADIANCE_FEEDBACK = Object.freeze({
  /**
   * Experimental only. Partitioned writes currently mutate the visible atlas
   * in place, so leaves and parents from different fixed-point iterations can
   * be sampled as one advertised generation. Re-enable only after feedback has
   * ping-pong storage and publishes complete phase rotations atomically.
   */
  enabledByDefault: false,
  phaseCount: 4,
  /**
   * At the 0.85 transport ceiling, 24 rotations bound the unpropagated fixed-
   * point residual below 2.1% (0.85^24). Static scenes then go exactly idle.
   */
  settleCycleCount: 24,
  settleFrameCount: 96,
  directionCount: 4,
  distanceCells: [2.5, 6, 18, 54] as const,
  /** Pure-white display materials remain contractive in the transport solve. */
  maximumTransportAlbedo: 0.85,
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
  /** Canonical SVO material table used for diffuse transport albedo. */
  materialPbr?: GPUBuffer;
  /** Selected image-free environment record used as the open-cone source. */
  environmentLighting?: GPUBuffer;
  /** Bounded authored light table; the first four records feed diffuse transport. */
  lights?: GPUBuffer;
  lightCount?: number;
  worklists: readonly LiveSvoDerivedGpuWorklist[];
  /** GPU publication generation used to certify newly allocated empty pages. */
  generationSource: LiveSvoDerivedGenerationSource;
  /** Number of address-plan slots resident in the fixed atlas. */
  plannedPageCount: number;
  /** Finest canonical tree level used for cell-to-leaf traversal. */
  finestLevel: number;
  worldOrigin_m?: readonly [number, number, number];
  cellSize_m?: readonly [number, number, number];
  /** Previous-frame radiance feedback. Enabled when both optional lighting buffers are supplied. */
  radianceFeedback?: boolean;
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

/**
 * Finest virtual pages touched by the canonical sparse topology. Coarse leaves
 * expand to every page in their extent, while empty space with no leaf never
 * consumes a physical opacity/radiance slot.
 */
export function liveSvoPlanBasePages(plan: SparseBrickPlan): readonly (readonly [number, number, number])[] {
  const unique = new Map<string, readonly [number, number, number]>();
  for (const leaf of plan.leaves) {
    const node = plan.nodes[leaf.nodeIndex];
    if (!node) throw new RangeError(`Live derived leaf ${leaf.index} references a missing node`);
    for (const page of liveSvoLeafBasePages({
      coordinate: [node.coordinate.x, node.coordinate.y, node.coordinate.z],
      leafLevel: node.level,
      finestLevel: plan.maximumDepth,
      brickSize: plan.brickSize,
    })) unique.set(page.join(","), page);
  }
  return [...unique.values()];
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
@compute @workgroup_size(64) fn populate(@builtin(global_invocation_id) gid:vec3u){let dirty=gid.x;let allLive=params.limits.w!=0u;let partitioned=params.limits.w==2u;let count=select(min(source[params.source.x],params.limits.x),min(control[1],params.limits.x),allLive);if(dirty>=count||(partitioned&&dirty%max(params.source.w,1u)!=params.source.z)){return;}let generation=generationState[params.source.y];if(generation==0u){return;}
 var leaf=dirty;if(!allLive){leaf=source[params.source.z+dirty*params.source.w];}if(leaf>=control[1]){return;}let leafBase=control[16]+leaf*4u;let node=topology[leafBase];let leafLevel=topology[node*8u+2u];if(leafLevel>params.domain.x){return;}let scale=1u<<(params.domain.x-leafLevel);let brickOrigin=morton(topology[leafBase+2u],topology[leafBase+3u],leafLevel)*scale;let cellMinimum=brickOrigin*control[11];let cellMaximum=(brickOrigin+vec3u(scale))*control[11];let firstPage=cellMinimum/${SVO_NODE_MIP_LAYOUT.interiorSize}u;let lastPage=(cellMaximum-vec3u(1u))/${SVO_NODE_MIP_LAYOUT.interiorSize}u;
 for(var z=firstPage.z;z<=lastPage.z;z+=1u){for(var y=firstPage.y;y<=lastPage.y;y+=1u){for(var x=firstPage.x;x<=lastPage.x;x+=1u){var page=vec3u(x,y,z);for(var level=0u;level<params.domain.y;level+=1u){emitPage(level,page,generation);page>>=vec3u(1u);}}}}
}
@compute @workgroup_size(1) fn finalize(@builtin(global_invocation_id) gid:vec3u){let level=gid.x;if(level>=params.domain.y){return;}let section=params.sections[level];let count=min(atomicLoad(&output[section]),params.limits.y);atomicStore(&output[section+1u],generationState[params.source.y]);atomicStore(&output[section+2u],level);atomicStore(&output[section+3u],${LIVE_SVO_DERIVED_WORKLIST.headerWords}u);let groups=(count*${SVO_NODE_MIP_LAYOUT.physicalSize ** 3}u+255u)/256u;let x=min(groups,65535u);atomicStore(&output[section+4u],x);atomicStore(&output[section+5u],select(0u,(groups+x-1u)/x,x>0u));atomicStore(&output[section+6u],1u);}
`;

export const liveSvoDerivedBuildWGSL = /* wgsl */ `
${liveSvoDerivedPageValidityWGSL}
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
    if(child!=INVALID&&(textureLoad(opacityValidity,svoDerivedPageValidityTexel(textureDimensions(opacityValidity),child),0).x==0u||textureLoad(radianceValidity,svoDerivedPageValidityTexel(textureDimensions(radianceValidity),child),0).x==0u)){return false;}}
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
    let voxel=leafVoxel(leaf,sampleLocal);let dynamicIdentity=payload[params.laneOffsets.y+voxel];let sceneIdentity=payload[params.laneOffsets.w+voxel];
    let dynamicCoverage=clamp(bitcast<f32>(payload[voxel*4u+2u]),0.,1.);let sceneCoverage=clamp(bitcast<f32>(payload[params.laneOffsets.z+voxel*4u+2u]),0.,1.);
    // Container glass remains structural geometry, but it is not an opacity
    // source: otherwise the cone hierarchy turns the vessel into a projected
    // cutout even though the exact/composite paths treat it as dielectric.
    let dynamicSolid=select(dynamicCoverage,0.,(dynamicIdentity&0xffffu)==${VOXEL_MATERIAL_IDS.containerGlass}u);let sceneSolid=select(sceneCoverage,0.,(sceneIdentity&0xffffu)==${VOXEL_MATERIAL_IDS.containerGlass}u);
    let fluid=clamp(bitcast<f32>(payload[params.laneOffsets.x+voxel*4u+3u]),0.,1.);let solid=1.-(1.-dynamicSolid)*(1.-sceneSolid);
    textureStore(opacityScratch,destination,vec4f(solid,select(0.,1.,solid>0.),clamp(fluid,0.,1.),select(0.,1.,fluid>0.)));
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

/**
 * A deliberately low-rate Jacobi-style diffuse solve. Base pages read the
 * previous complete radiance atlas and write emission + reflected incident
 * light into compact scratch; parent pages then reduce the newly published
 * children. The recurrence replaces, rather than adds to, previous radiance,
 * so energy cannot grow merely because more frames have elapsed.
 */
export const liveSvoRadianceFeedbackWGSL = /* wgsl */ `
${liveSvoDerivedPageValidityWGSL}
${svoMaterialWGSL}
${svoEnvironmentLightingWGSL}
${svoLightWGSL}
${svoTetrahedralRadianceWGSL}
struct Params{targetAtlasPages:vec4u,scratchAtlasPages:vec4u,limits:vec4u,laneOffsets:vec4u,direct:vec4u,zOffsets:array<u32,12>,mappingOrigin:vec4f,mappingCellSize:vec4f}
@group(0) @binding(0) var<storage,read> control:array<u32>;
@group(0) @binding(1) var<storage,read> topology:array<u32>;
@group(0) @binding(2) var<storage,read> payload:array<u32>;
@group(0) @binding(3) var<storage,read> worklist:array<u32>;
@group(0) @binding(4) var opacitySource:texture_3d<f32>;
@group(0) @binding(5) var radianceSource0:texture_3d<f32>;
@group(0) @binding(6) var radianceSource1:texture_3d<f32>;
@group(0) @binding(7) var radianceSource2:texture_3d<f32>;
@group(0) @binding(8) var radianceSource3:texture_3d<f32>;
@group(0) @binding(9) var radianceScratch:texture_storage_3d<rgba16float,write>;
@group(0) @binding(10) var<uniform> params:Params;
@group(0) @binding(11) var directTable:texture_3d<u32>;
@group(0) @binding(12) var<storage,read> materials:array<SvoMaterialRecord>;
@group(0) @binding(13) var<storage,read> environment:array<SvoEnvironmentLightingRecord>;
@group(0) @binding(14) var opacityValidity:texture_2d<u32>;
@group(0) @binding(15) var radianceValidity:texture_2d<u32>;
@group(0) @binding(16) var<storage,read_write> scratchValidity:array<u32>;
@group(0) @binding(17) var<storage,read> lights:array<SvoLightRecord>;
@group(0) @binding(18) var atlasSampler:sampler;

const INVALID:u32=0xffffffffu;const PHYSICAL:u32=${SVO_NODE_MIP_LAYOUT.physicalSize}u;const INTERIOR:u32=${SVO_NODE_MIP_LAYOUT.interiorSize}u;
const RECORD_WORDS:u32=${LIVE_SVO_DERIVED_WORKLIST.recordWords}u;const ACTIVE:u32=${SVO_BRICK_LIFECYCLE.activeBit}u;const PI:f32=3.141592653589793;
fn linearIndex(gid:vec3u,groups:vec3u)->u32{return gid.x+gid.y*groups.x*256u+gid.z*groups.x*groups.y*256u;}
fn slotOrigin(slot:u32)->vec3u{let p=params.targetAtlasPages.xyz;return vec3u(slot%p.x,(slot/p.x)%p.y,slot/(p.x*p.y))*PHYSICAL;}
fn scratchOrigin(recordIndex:u32)->vec3u{let p=params.scratchAtlasPages.xyz;return vec3u(recordIndex%p.x,(recordIndex/p.x)%p.y,recordIndex/(p.x*p.y))*PHYSICAL;}
fn physicalCoordinate(index:u32)->vec3u{return vec3u(index%PHYSICAL,(index/PHYSICAL)%PHYSICAL,index/(PHYSICAL*PHYSICAL));}
fn interiorCoordinate(physical:vec3u)->vec3u{return clamp(physical,vec3u(1u),vec3u(INTERIOR))-1u;}
fn keyBit(lo:u32,hi:u32,bit:u32)->u32{if(bit>=32u){return(hi>>(bit-32u))&1u;}return(lo>>bit)&1u;}
fn morton(lo:u32,hi:u32,level:u32)->vec3u{var p=vec3u(0u);for(var bit=0u;bit<level;bit+=1u){let s=1u<<bit;p+=vec3u(keyBit(lo,hi,3u*bit),keyBit(lo,hi,3u*bit+1u),keyBit(lo,hi,3u*bit+2u))*s;}return p;}
fn leafLocal(globalCell:vec3u,leaf:u32)->vec3u{let brickSize=control[11];let finest=params.scratchAtlasPages.w;let node=topology[control[16]+leaf*4u];let nodeBase=node*8u;let level=topology[nodeBase+2u];let scale=1u<<(finest-level);let origin=morton(topology[nodeBase],topology[nodeBase+1u],level)*scale*brickSize;return min((globalCell-origin)/scale,vec3u(brickSize-1u));}
fn leafVoxel(leafIndex:u32,local:vec3u)->u32{let leafBase=control[16]+leafIndex*4u;return topology[leafBase+1u]+local.x+local.y*control[11]+local.z*control[11]*control[11];}
fn solidDistance(leafIndex:u32,local:vec3u)->f32{let voxel=leafVoxel(leafIndex,local);return min(bitcast<f32>(payload[voxel*4u+1u]),bitcast<f32>(payload[params.laneOffsets.z+voxel*4u+1u]));}
fn safeNormal(leafIndex:u32,local:vec3u)->vec3f{let lo=max(local,vec3u(1u))-1u;let hi=min(local+1u,vec3u(control[11]-1u));let gradient=vec3f(
 solidDistance(leafIndex,vec3u(hi.x,local.y,local.z))-solidDistance(leafIndex,vec3u(lo.x,local.y,local.z)),
 solidDistance(leafIndex,vec3u(local.x,hi.y,local.z))-solidDistance(leafIndex,vec3u(local.x,lo.y,local.z)),
 solidDistance(leafIndex,vec3u(local.x,local.y,hi.z))-solidDistance(leafIndex,vec3u(local.x,local.y,lo.z)));
 return select(vec3f(0.,1.,0.),normalize(gradient),dot(gradient,gradient)>1e-12);}
fn writeRadiance(coordinate:vec3u,a:vec3f,b:vec3f,c:vec3f,d:vec3f){let depth=params.scratchAtlasPages.z*PHYSICAL;
 textureStore(radianceScratch,coordinate,vec4f(a,1.));textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth),vec4f(b,1.));
 textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth*2u),vec4f(c,1.));textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth*3u),vec4f(d,1.));}
fn tableSlot(level:u32,page:vec3u)->u32{if(level>=params.direct.x||page.x>=params.direct.y||page.y>=params.direct.z){return INVALID;}let dims=textureDimensions(directTable);let start=params.zOffsets[level];var end=dims.z;if(level+1u<params.direct.x){end=params.zOffsets[level+1u];}if(page.z>=end-start){return INVALID;}let encoded=textureLoad(directTable,vec3i(i32(page.x),i32(page.y),i32(start+page.z)),0).x;return select(INVALID,encoded-1u,encoded>0u);}
struct SceneSample{coverage:f32,radiance:SvoTetraRadiance,valid:u32}
fn sceneSample(positionIn:vec3f,level:u32)->SceneSample{if(any(positionIn<vec3f(0.0))){return SceneSample(0.,SvoTetraRadiance(vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.)),1u);}let scale=exp2(f32(level));let levelVoxel=positionIn/scale;let pageFloor=floor(levelVoxel/f32(INTERIOR));if(any(pageFloor<vec3f(0.0))||any(pageFloor>=vec3f(2097152.0))){return SceneSample(0.,SvoTetraRadiance(vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.)),1u);}let page=vec3u(pageFloor);let slot=tableSlot(level,page);if(slot==INVALID){return SceneSample(0.,SvoTetraRadiance(vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.)),1u);}let opacityDimensions=textureDimensions(opacityValidity);let radianceDimensions=textureDimensions(radianceValidity);if(textureLoad(opacityValidity,svoDerivedPageValidityTexel(opacityDimensions,slot),0).x==0u||textureLoad(radianceValidity,svoDerivedPageValidityTexel(radianceDimensions,slot),0).x==0u){return SceneSample(0.,SvoTetraRadiance(vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.)),0u);}let interiorTexel=levelVoxel-vec3f(page)*f32(INTERIOR)-vec3f(.5);let physical=vec3f(slotOrigin(slot))+vec3f(1.0)+clamp(interiorTexel,vec3f(-.5),vec3f(f32(INTERIOR)-.5));let uv=(physical+vec3f(.5))/vec3f(textureDimensions(opacitySource));return SceneSample(textureSampleLevel(opacitySource,atlasSampler,uv,0.0).x,SvoTetraRadiance(textureSampleLevel(radianceSource0,atlasSampler,uv,0.0).rgb,textureSampleLevel(radianceSource1,atlasSampler,uv,0.0).rgb,textureSampleLevel(radianceSource2,atlasSampler,uv,0.0).rgb,textureSampleLevel(radianceSource3,atlasSampler,uv,0.0).rgb),1u);}
fn hemisphereDirection(normal:vec3f,index:u32)->vec3f{var basis=svoTetraDirection0();if(index==1u){basis=svoTetraDirection1();}else if(index==2u){basis=svoTetraDirection2();}else if(index==3u){basis=svoTetraDirection3();}basis=select(-basis,basis,dot(basis,normal)>=0.0);return normalize(normal*.45+basis);}
fn incidentAlong(origin:vec3f,normal:vec3f,index:u32)->vec3f{let direction=hemisphereDirection(normal,index);var transmittance=1.0;var incident=vec3f(0.0);
 for(var step=0u;step<4u;step+=1u){var distance=2.5;if(step==1u){distance=6.0;}else if(step==2u){distance=18.0;}else if(step==3u){distance=54.0;}let level=min(step,params.direct.x-1u);let sample=sceneSample(origin+direction*distance,level);if(sample.valid==0u){return vec3f(0.0);}let coverage=clamp(sample.coverage,0.0,1.0);incident+=transmittance*svoTetraRadianceAlong(sample.radiance,-direction);transmittance*=1.0-coverage;if(transmittance<=0.01){break;}}
 return incident+transmittance*svoEnvironmentDiffuseIrradiance(environment[0],normal)/PI;}
fn directVisibility(originCells:vec3f,towardLight:vec3f,maximumDistance_m:f32)->f32{let cellScale=max(params.mappingCellSize.x,max(params.mappingCellSize.y,params.mappingCellSize.z));var transmittance=1.0;for(var step=0u;step<4u;step+=1u){var distance_m=cellScale*2.5;if(step==1u){distance_m=cellScale*6.0;}else if(step==2u){distance_m=cellScale*18.0;}else if(step==3u){distance_m=cellScale*54.0;}if(maximumDistance_m>0.0){distance_m=min(distance_m,maximumDistance_m*.9);}let position=originCells+towardLight*distance_m/params.mappingCellSize.xyz;let sample=sceneSample(position,min(step,params.direct.x-1u));if(sample.valid==0u){return 0.0;}transmittance*=1.0-clamp(sample.coverage,0.0,1.0);}return clamp(transmittance,0.0,1.0);}
fn directIncident(worldPosition:vec3f,originCells:vec3f,normal:vec3f)->vec3f{var result=vec3f(0.0);let count=min(min(params.direct.w,arrayLength(&lights)),4u);for(var index=0u;index<4u;index+=1u){if(index>=count){break;}let light=lights[index];let base=svoLightRadiance(light);var toward=light.directionCone.xyz;var maximumDistance_m=0.0;var attenuation=1.0;if(light.identity.x!=SVO_LIGHT_DIRECTIONAL){let offset=light.positionRange.xyz-worldPosition;let distanceSquared=dot(offset,offset);if(distanceSquared<=1e-10||(light.positionRange.w>0.0&&distanceSquared>=light.positionRange.w*light.positionRange.w)){continue;}maximumDistance_m=sqrt(distanceSquared);toward=offset/maximumDistance_m;let rangeFade=select(1.0,pow(clamp(1.0-maximumDistance_m/max(light.positionRange.w,1e-6),0.0,1.0),2.0),light.positionRange.w>0.0);attenuation=rangeFade/max(1.0,distanceSquared);if(light.identity.x==SVO_LIGHT_SPHERE_AREA){let area=4.0*PI*light.shape.x*light.shape.x;attenuation=rangeFade*area/max(area,distanceSquared);}else if(light.identity.x==SVO_LIGHT_RECTANGLE_AREA){let area=4.0*light.axisUWidth.w*light.axisVHeight.w;attenuation=rangeFade*max(dot(normalize(light.directionCone.xyz),-toward),0.0)*area/max(area,distanceSquared);}}let cosine=max(dot(normal,toward),0.0);if(cosine>0.0&&directVisibility(originCells,toward,maximumDistance_m)>0.0){result+=base*(attenuation*cosine/PI);}}return result;}
fn childRadiance(record:u32,local:vec3u,sampleIndex:u32,lobe:u32)->vec3f{let bit=vec3u(sampleIndex&1u,(sampleIndex>>1u)&1u,(sampleIndex>>2u)&1u);let fine=local*2u+bit;let octant=(fine.x/INTERIOR)|((fine.y/INTERIOR)<<1u)|((fine.z/INTERIOR)<<2u);let child=worklist[record+${LIVE_SVO_DERIVED_WORKLIST.childSlotWord}u+octant];if(child==INVALID){return vec3f(0.);}let coordinate=vec3i(slotOrigin(child)+(fine%INTERIOR)+1u);if(lobe==0u){return textureLoad(radianceSource0,coordinate,0).rgb;}if(lobe==1u){return textureLoad(radianceSource1,coordinate,0).rgb;}if(lobe==2u){return textureLoad(radianceSource2,coordinate,0).rgb;}return textureLoad(radianceSource3,coordinate,0).rgb;}
fn childrenReady(record:u32)->bool{for(var childIndex=0u;childIndex<8u;childIndex+=1u){let child=worklist[record+${LIVE_SVO_DERIVED_WORKLIST.childSlotWord}u+childIndex];if(child!=INVALID&&textureLoad(radianceValidity,svoDerivedPageValidityTexel(textureDimensions(radianceValidity),child),0).x==0u){return false;}}return true;}
@compute @workgroup_size(256) fn feedbackPages(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){let index=linearIndex(gid,groups);let texels=PHYSICAL*PHYSICAL*PHYSICAL;let recordIndex=index/texels;let count=min(worklist[0],params.limits.x);if(recordIndex>=count){return;}let physical=physicalCoordinate(index-recordIndex*texels);let local=interiorCoordinate(physical);let record=worklist[3]+recordIndex*RECORD_WORDS;let destination=scratchOrigin(recordIndex)+physical;
 if(worklist[2]>0u){if(all(physical==vec3u(0u))){scratchValidity[recordIndex]=select(0u,worklist[1],childrenReady(record));}var r0=vec3f(0.);var r1=vec3f(0.);var r2=vec3f(0.);var r3=vec3f(0.);for(var child=0u;child<8u;child+=1u){r0+=childRadiance(record,local,child,0u)/8.;r1+=childRadiance(record,local,child,1u)/8.;r2+=childRadiance(record,local,child,2u)/8.;r3+=childRadiance(record,local,child,3u)/8.;}writeRadiance(destination,r0,r1,r2,r3);return;}
 if(all(physical==vec3u(0u))){scratchValidity[recordIndex]=worklist[1];}let slot=worklist[record];let coverage=textureLoad(opacitySource,vec3i(slotOrigin(slot)+physical),0).x;let page=vec3u(worklist[record+1u],worklist[record+2u],worklist[record+3u]);let octant=(local.x/4u)|((local.y/4u)<<1u)|((local.z/4u)<<2u);let leaf=worklist[record+${LIVE_SVO_DERIVED_WORKLIST.sourceLeafWord}u+octant];if(coverage<=0.0||leaf>=control[1]){writeRadiance(destination,vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.));return;}let globalCell=page*INTERIOR+local;let sampleLocal=leafLocal(globalCell,leaf);let voxel=leafVoxel(leaf,sampleLocal);let dynamicIdentity=payload[params.laneOffsets.y+voxel];let sceneIdentity=payload[params.laneOffsets.w+voxel];let dynamicCoverage=clamp(bitcast<f32>(payload[voxel*4u+2u]),0.,1.);let sceneCoverage=clamp(bitcast<f32>(payload[params.laneOffsets.z+voxel*4u+2u]),0.,1.);let materialIndex=select(dynamicIdentity&0xffffu,sceneIdentity&0xffffu,sceneCoverage>=dynamicCoverage&&sceneCoverage>0.);if(materialIndex>=arrayLength(&materials)){writeRadiance(destination,vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.));return;}let material=materials[materialIndex];let normal=safeNormal(leaf,sampleLocal);let originCells=vec3f(globalCell)+vec3f(.5)+normal*1.5;let transportAlbedo=clamp(material.baseColorOpacity.rgb*(1.0-clamp(material.surface.x,0.0,1.0)),vec3f(0.0),vec3f(${LIVE_SVO_RADIANCE_FEEDBACK.maximumTransportAlbedo}));var incident=vec3f(0.0);for(var direction=0u;direction<${LIVE_SVO_RADIANCE_FEEDBACK.directionCount}u;direction+=1u){incident+=incidentAlong(originCells,normal,direction)/f32(${LIVE_SVO_RADIANCE_FEEDBACK.directionCount});}let worldPosition=params.mappingOrigin.xyz+(vec3f(globalCell)+vec3f(.5))*params.mappingCellSize.xyz;incident+=directIncident(worldPosition,originCells,normal);let outgoing=max(material.emissiveRoughness.rgb,vec3f(0.0))+transportAlbedo*incident;writeRadiance(destination,outgoing*coverage*max(0.,dot(normal,svoTetraDirection0())),outgoing*coverage*max(0.,dot(normal,svoTetraDirection1())),outgoing*coverage*max(0.,dot(normal,svoTetraDirection2())),outgoing*coverage*max(0.,dot(normal,svoTetraDirection3())));
}`;

export const liveSvoDerivedCopyWGSL = /* wgsl */ `
${liveSvoDerivedPageValidityWGSL}
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
@compute @workgroup_size(256) fn invalidatePages(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;let count=min(worklist[0],params.limits.x);if(i>=count){return;}let record=worklist[3]+i*RECORD_WORDS;let slot=worklist[record];textureStore(opacityValidity,svoDerivedPageValidityTexel(textureDimensions(opacityValidity),slot),vec4u(0u));textureStore(radianceValidity,svoDerivedPageValidityTexel(textureDimensions(radianceValidity),slot),vec4u(0u));}
@compute @workgroup_size(256) fn copyOpacity(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let index=linearIndex(gid,groups);let texels=PHYSICAL*PHYSICAL*PHYSICAL;let recordIndex=index/texels;let count=min(worklist[0],params.limits.x);if(recordIndex>=count){return;}
 let local=physical(index-recordIndex*texels);let record=worklist[3]+recordIndex*RECORD_WORDS;let slot=worklist[record];let source=vec3i(scratchOrigin(recordIndex)+local);let destination=vec3i(targetOrigin(slot)+local);
 textureStore(opacityDestination,destination,textureLoad(opacityScratch,source,0));if(all(local==vec3u(0u))&&scratchValidity[recordIndex]!=0u){textureStore(opacityValidity,svoDerivedPageValidityTexel(textureDimensions(opacityValidity),slot),vec4u(scratchValidity[recordIndex]));}}
fn copyRadianceLobe(gid:vec3u,groups:vec3u,lobe:u32){let index=linearIndex(gid,groups);let texels=PHYSICAL*PHYSICAL*PHYSICAL;let recordIndex=index/texels;let count=min(worklist[0],params.limits.x);if(recordIndex>=count){return;}
 let local=physical(index-recordIndex*texels);let record=worklist[3]+recordIndex*RECORD_WORDS;let slot=worklist[record];let scratch=scratchOrigin(recordIndex)+local+vec3u(0u,0u,lobe*params.scratchAtlasPages.z*PHYSICAL);let destination=vec3i(targetOrigin(slot)+local);
 textureStore(radianceDestination,destination,textureLoad(radianceScratch,vec3i(scratch),0));if(lobe==3u&&all(local==vec3u(0u))&&scratchValidity[recordIndex]!=0u){textureStore(radianceValidity,svoDerivedPageValidityTexel(textureDimensions(radianceValidity),slot),vec4u(scratchValidity[recordIndex]));}}
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
${liveSvoDerivedPageValidityWGSL}
struct Params{targetAtlasPages:vec4u,scratchAtlasPages:vec4u,limits:vec4u,laneOffsets:vec4u}
@group(0) @binding(0) var opacityValidity:texture_storage_2d<r32uint,write>;
@group(0) @binding(1) var radianceValidity:texture_storage_2d<r32uint,write>;
@group(0) @binding(2) var<storage,read> generationState:array<u32>;
@group(0) @binding(3) var<uniform> params:Params;
@compute @workgroup_size(256) fn initializeValidEmptyPages(@builtin(global_invocation_id) gid:vec3u){
  let slot=gid.x;if(slot>=params.limits.w){return;}let generation=generationState[params.limits.z];if(generation==0u){return;}
  textureStore(opacityValidity,svoDerivedPageValidityTexel(textureDimensions(opacityValidity),slot),vec4u(generation));
  textureStore(radianceValidity,svoDerivedPageValidityTexel(textureDimensions(radianceValidity),slot),vec4u(generation));
}`;

interface LevelBindings { worklist: LiveSvoDerivedGpuWorklist; invalidate: GPUBindGroup; build: GPUBindGroup; feedback?: GPUBindGroup; opacityCopy: GPUBindGroup; radianceCopy: readonly GPUBindGroup[] }

interface LiveSvoDerivedBuilderPipelineState {
  emptyInitializationPipeline: GPUComputePipeline;
  emptyInitializationBindGroup: GPUBindGroup;
  buildPipeline: GPUComputePipeline;
  feedbackPipeline?: GPUComputePipeline;
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
  readonly radianceFeedbackEnabled: boolean;
  private readonly plannedPageCount: number;
  private readonly opacityScratch: GPUTexture;
  private readonly radianceScratch: GPUTexture;
  private readonly scratchValidity: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly feedbackParams?: GPUBuffer;
  private readonly feedbackSampler?: GPUSampler;
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
    this.radianceFeedbackEnabled = options.radianceFeedback
      ?? Boolean(options.materialPbr && options.environmentLighting && options.lights);
    if (this.radianceFeedbackEnabled && (!options.materialPbr || !options.environmentLighting || !options.lights)) {
      throw new Error("Live SVO radiance feedback requires material, environment, and light buffers");
    }
    if (!Number.isSafeInteger(options.lightCount ?? 0) || (options.lightCount ?? 0) < 0) {
      throw new RangeError("Live SVO feedback light count must be a nonnegative safe integer");
    }
    if ((options.worldOrigin_m ?? [0, 0, 0]).some((value) => !Number.isFinite(value))
      || (options.cellSize_m ?? [1, 1, 1]).some((value) => !Number.isFinite(value) || value <= 0)) {
      throw new RangeError("Live SVO feedback mapping must have a finite origin and positive cell size");
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
    if (this.radianceFeedbackEnabled) {
      this.feedbackSampler = device.createSampler({ label: `${label} radiance feedback sampler`,
        addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge",
        minFilter: "linear", magFilter: "linear", mipmapFilter: "nearest" });
      this.feedbackParams = device.createBuffer({ label: `${label} radiance feedback parameters`, size: 160,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const feedbackData = new ArrayBuffer(160), feedbackWords = new Uint32Array(feedbackData), feedbackFloats = new Float32Array(feedbackData);
      feedbackWords.set([...options.nodeMips.atlasPages, 0], 0);
      feedbackWords.set([...this.scratchAtlasPages, options.finestLevel], 4);
      feedbackWords.set([options.nodeMips.pageCapacity, SVO_NODE_MIP_LAYOUT.physicalSize,
        options.generationSource.offsetBytes / 4, options.plannedPageCount], 8);
      feedbackWords.set([options.tree.velocityOffsetBytes / 4, options.tree.materialOwnerOffsetBytes / 4,
        options.tree.sceneGeometryOffsetBytes / 4, options.tree.sceneMaterialOwnerOffsetBytes / 4], 12);
      feedbackWords.set([options.worklists.length, options.nodeMips.directPageTableDimensions[0],
        options.nodeMips.directPageTableDimensions[1], options.lightCount ?? 0], 16);
      feedbackWords.set(options.nodeMips.directPageTableLevelZOffsets.slice(0, 12), 20);
      feedbackFloats.set([...(options.worldOrigin_m ?? [0, 0, 0]), 0], 32);
      feedbackFloats.set([...(options.cellSize_m ?? [1, 1, 1]), 0], 36);
      device.queue.writeBuffer(this.feedbackParams, 0, feedbackData);
    }
    const texelCount = scratchTexels[0] * scratchTexels[1] * scratchTexels[2];
    this.allocatedBytes = texelCount * (4 + 4 * 8) + scratchCapacity * 4 + 64 + (this.feedbackParams?.size ?? 0);
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
    const feedbackModule = this.radianceFeedbackEnabled
      ? device.createShaderModule({ label: `${label} radiance feedback shader`, code: liveSvoRadianceFeedbackWGSL })
      : undefined;
    const copyModule = device.createShaderModule({ label: `${label} copy shader`, code: liveSvoDerivedCopyWGSL });
    const emptyInitializationModule = device.createShaderModule({
      label: `${label} valid-empty initialization shader`, code: liveSvoDerivedEmptyInitializationWGSL,
    });
    const [emptyInitializationPipeline, buildPipeline, feedbackPipeline, invalidatePipeline, opacityCopyPipeline, radianceCopyPipelines] = await Promise.all([
      device.createComputePipelineAsync({ label: `${label} valid-empty initialization pipeline`, layout: "auto",
        compute: { module: emptyInitializationModule, entryPoint: "initializeValidEmptyPages" } }),
      device.createComputePipelineAsync({ label: `${label} build pipeline`, layout: "auto", compute: { module: buildModule, entryPoint: "buildPages" } }),
      feedbackModule ? device.createComputePipelineAsync({ label: `${label} radiance feedback pipeline`, layout: "auto",
        compute: { module: feedbackModule, entryPoint: "feedbackPages" } }) : Promise.resolve(undefined),
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
    const materialPbr = options.materialPbr, environmentLighting = options.environmentLighting, lights = options.lights;
    const feedbackParams = this.feedbackParams, feedbackSampler = this.feedbackSampler;
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
      feedback: feedbackPipeline && materialPbr && environmentLighting && lights && feedbackParams && feedbackSampler
        ? device.createBindGroup({ layout: feedbackPipeline.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: options.tree.control, size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes } },
          { binding: 1, resource: { buffer: options.tree.topology, offset: options.tree.topologyOffsetBytes } },
          { binding: 2, resource: { buffer: options.tree.payload } },
          { binding: 3, resource: { buffer: worklist.buffer, offset: worklist.bindingOffsetBytes ?? 0, size: worklist.bindingSizeBytes } },
          { binding: 4, resource: options.nodeMips.texture.createView({ dimension: "3d" }) },
          ...targetRadianceViews.map((resource, index) => ({ binding: 5 + index, resource })),
          { binding: 9, resource: radianceScratchView }, { binding: 10, resource: { buffer: feedbackParams } },
          { binding: 11, resource: options.nodeMips.directPageTableTexture.createView({ dimension: "3d" }) },
          { binding: 12, resource: { buffer: materialPbr } }, { binding: 13, resource: { buffer: environmentLighting } },
          { binding: 14, resource: options.nodeMips.pageValidity.view }, { binding: 15, resource: options.radiance.pageValidity.view },
          { binding: 16, resource: { buffer: this.scratchValidity } }, { binding: 17, resource: { buffer: lights } },
          { binding: 18, resource: feedbackSampler },
        ] }) : undefined,
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
    return { emptyInitializationPipeline, emptyInitializationBindGroup, buildPipeline, feedbackPipeline, invalidatePipeline,
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

  /** Publish one GPU-compacted phase of previous-frame diffuse feedback. */
  encodeRadianceFeedback(encoder: GPUCommandEncoder): void {
    if (this.destroyed || !this.radianceFeedbackEnabled) return;
    const state = this.pipelineState;
    if (!state?.feedbackPipeline) throw new Error("Live SVO radiance feedback pipeline is not initialized");
    for (const level of state.levels) {
      if (!level.feedback) continue;
      const indirect = level.worklist.indirectOffsetBytes
        ?? (level.worklist.bindingOffsetBytes ?? 0) + LIVE_SVO_DERIVED_WORKLIST.dispatchIndirectOffsetBytes;
      const feedback = encoder.beginComputePass({ label: "Feed back live SVO diffuse radiance" });
      feedback.setPipeline(state.feedbackPipeline); feedback.setBindGroup(0, level.feedback);
      feedback.dispatchWorkgroupsIndirect(level.worklist.buffer, indirect); feedback.end();
      state.radianceCopyPipelines.forEach((pipeline, lobe) => {
        const radiance = encoder.beginComputePass({ label: `Publish feedback SVO radiance lobe ${lobe}` });
        radiance.setPipeline(pipeline); radiance.setBindGroup(0, level.radianceCopy[lobe]);
        radiance.dispatchWorkgroupsIndirect(level.worklist.buffer, indirect); radiance.end();
      });
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.opacityScratch.destroy(); this.radianceScratch.destroy(); this.scratchValidity.destroy(); this.params.destroy(); this.feedbackParams?.destroy(); this.destroyed = true;
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
  feedback: { capacity: number; bindGroup: GPUBindGroup };
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
  private readonly feedback: LiveSvoDerivedPlannerSourceAllocation;
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
    this.feedback = createSource(initialSource, "temporal feedback leaves");
    this.configurePlan(options.nodeMips, options.finestLevel, options.levelCount);
    this.allocatedBytes = sectionBytes * options.levelCount + options.nodeMips.pageCapacity * 4 + 144 * (this.sources.length + 2);
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
    const feedback = bindSource(this.feedback);
    const finalizeBindGroup = device.createBindGroup({ layout: finalizePipeline.getBindGroupLayout(0), entries: [
      { binding: 5, resource: { buffer: this.arena } }, { binding: 6, resource: { buffer: this.sources[0].params } },
      { binding: 7, resource: { buffer: options.generationSource.buffer } },
    ] });
    return { populatePipeline, finalizePipeline, sources, initial, feedback, finalizeBindGroup };
  }

  configurePlan(target: WebGpuLiveSvoNodeMipGpuTarget, finestLevel = this.options.finestLevel, levelCount = this.options.levelCount): void {
    if (!Number.isInteger(finestLevel) || finestLevel < 0) throw new RangeError("Live derived finest level must be nonnegative");
    if (!Number.isInteger(levelCount) || levelCount < 1 || levelCount > this.options.levelCount) {
      throw new RangeError("Configured live derived levels must fit the fixed planner allocation");
    }
    const write = (params: GPUBuffer, source: LiveSvoDirtyLeafSource, allLiveMode: 0 | 1 | 2, phase = 0, phaseCount = 1) => {
      const words = new Uint32Array(36);
      words.set([source.countOffsetBytes / 4, this.options.generationSource.offsetBytes / 4,
        allLiveMode === 2 ? phase : source.recordOffsetBytes / 4,
        allLiveMode === 2 ? phaseCount : source.recordStrideWords], 0);
      words.set([finestLevel, levelCount, target.directPageTableDimensions[0], target.directPageTableDimensions[1]], 4);
      words.set([source.capacity, this.options.pageCapacityPerLevel, target.pageCapacity, allLiveMode], 8);
      words.set(target.directPageTableLevelZOffsets.slice(0, 12), 12);
      words.set(this.sectionOffsetsBytes.map((offset) => offset / 4), 24);
      this.device.queue.writeBuffer(params, 0, words);
    };
    this.options.dirtyLeafSources.forEach((source, index) => write(this.sources[index].params, source, 0));
    write(this.initial.params, { buffer: this.options.tree.control, countOffsetBytes: 0, recordOffsetBytes: 0,
      capacity: this.options.tree.leafCapacity, recordStrideWords: 1 }, 1);
    write(this.feedback.params, { buffer: this.options.tree.control, countOffsetBytes: 0, recordOffsetBytes: 0,
      capacity: this.options.tree.leafCapacity, recordStrideWords: 1 }, 2);
  }

  /** Compacts all supplied scene/fluid/topology streams into one deduplicated hierarchy. */
  encode(encoder: GPUCommandEncoder): void {
    this.encodeSources(encoder, "dirty");
  }

  /** First publication: unions every currently live tree leaf with all supplied dirty streams entirely on GPU. */
  encodeInitial(encoder: GPUCommandEncoder): void {
    this.encodeSources(encoder, "initial");
  }

  /** Compact one rotating subset of all live leaves for temporal GI feedback. */
  encodeRadianceFeedback(encoder: GPUCommandEncoder, phase: number, phaseCount = LIVE_SVO_RADIANCE_FEEDBACK.phaseCount): void {
    if (!Number.isSafeInteger(phaseCount) || phaseCount < 1 || !Number.isSafeInteger(phase) || phase < 0 || phase >= phaseCount) {
      throw new RangeError("Live SVO feedback phase must fit its positive phase count");
    }
    this.configurePlan(this.options.nodeMips, this.options.finestLevel, this.options.levelCount);
    const words = new Uint32Array([0, this.options.generationSource.offsetBytes / 4, phase, phaseCount]);
    this.device.queue.writeBuffer(this.feedback.params, 0, words);
    this.encodeSources(encoder, "feedback");
  }

  private encodeSources(encoder: GPUCommandEncoder, sourceMode: "dirty" | "initial" | "feedback"): void {
    if (this.destroyed) return;
    const state = this.pipelineState;
    if (!state) throw new Error("Live SVO derived worklist planner pipelines are not initialized");
    encoder.clearBuffer(this.claims);
    for (const offset of this.sectionOffsetsBytes) encoder.clearBuffer(this.arena, offset, LIVE_SVO_DERIVED_WORKLIST.headerWords * 4);
    const populate = encoder.beginComputePass({ label: "Populate live SVO derived worklists" });
    populate.setPipeline(state.populatePipeline);
    if (sourceMode === "initial") {
      populate.setBindGroup(0, state.initial.bindGroup); populate.dispatchWorkgroups(Math.ceil(state.initial.capacity / 64));
    }
    if (sourceMode === "feedback") {
      populate.setBindGroup(0, state.feedback.bindGroup); populate.dispatchWorkgroups(Math.ceil(state.feedback.capacity / 64));
    } else {
      for (const source of state.sources) {
        populate.setBindGroup(0, source.bindGroup); populate.dispatchWorkgroups(Math.ceil(source.capacity / 64));
      }
    }
    populate.end();
    const finalize = encoder.beginComputePass({ label: "Finalize live SVO derived worklists" });
    finalize.setPipeline(state.finalizePipeline); finalize.setBindGroup(0, state.finalizeBindGroup); finalize.dispatchWorkgroups(this.options.levelCount); finalize.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.arena.destroy(); this.claims.destroy(); this.sources.forEach(({ params }) => params.destroy()); this.initial.params.destroy(); this.feedback.params.destroy(); this.destroyed = true;
  }
}
