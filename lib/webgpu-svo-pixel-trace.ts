import {
  SVO_PIXEL_TRACE_GI_STATE,
  SVO_PIXEL_TRACE_DEFAULT_RECORD_CAPACITY,
  SVO_PIXEL_TRACE_FLAGS,
  SVO_PIXEL_TRACE_HEADER,
  SVO_PIXEL_TRACE_HEADER_WORDS,
  SVO_PIXEL_TRACE_KINDS,
  SVO_PIXEL_TRACE_MAGIC,
  SVO_PIXEL_TRACE_MAXIMUM_RECORD_CAPACITY,
  SVO_PIXEL_TRACE_RECORD_WORDS,
  SVO_PIXEL_TRACE_STATUS,
  SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS,
  decodeSvoPixelTrace,
  svoPixelTraceBufferBytes,
  svoPixelTraceTextureRows,
  type SvoPixelTrace,
} from "./svo-pixel-trace";

/**
 * The single-pixel probe entry point appended to the dry-scene fragment module.
 *
 * The probe is deliberately a *mirror*, not a second implementation: it walks
 * the canonical hierarchy with the same slab arithmetic, the same near-to-far
 * stack discipline, the same brick DDA, and the same cone step law as the
 * shipping shader, and it calls the shipping shader's own helpers wherever a
 * helper exists (`primitiveHit`, `dryLightSample`, `dryBiasedVisibilityRayUnit`,
 * `dryNodeMipAt`, `traceTerrain`, `nearestBody`). What it adds is a record per
 * unit of work. Keeping it separate is what lets the production entry points
 * stay byte-identical, which the frame fingerprint depends on.
 *
 * It runs as a fragment shader over a 1x1 target and guards on the integer
 * pixel coordinate, so exactly one invocation ever writes: helper invocations
 * in the rasterizer's 2x2 quad are excluded by the guard rather than by relying
 * on helper-invocation store semantics, and the append cursor needs no atomics.
 *
 * Records go to a write-only r32uint storage texture rather than a storage
 * buffer. The dry pass already binds ten storage buffers, which is the whole
 * per-stage budget browsers report on Apple silicon; storage textures are a
 * separate budget it does not use. Being write-only, the texture cannot carry
 * the request in, so the requested pixel arrives in a small uniform instead.
 */
export interface SvoPixelTraceProbeOptions {
  /** Bind group holding the trace storage buffer. Group 0 stays the dry scene's. */
  readonly group?: number;
  readonly recordCapacity?: number;
  /** Mirrors SVO_DRY_CONE_LOD_BLEND_BAND_WIDTH. */
  readonly coneLodBlendBandWidth: number;
  /** Mirrors SVO_DRY_SCENE_MAX_SHADED_LIGHTS. */
  readonly maximumShadedLights: number;
  /** Mirrors SVO_DRY_SCENE_AREA_LIGHT_SAMPLES. */
  readonly areaLightSamples: number;
  /** Mirrors SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES. */
  readonly stableOcclusionConeSamples: number;
  /** Mirrors SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT. */
  readonly primaryLeafVisitHardLimit: number;
  /** Mirrors SVO_DRY_VISIBILITY_FLAGS. */
  readonly visibilityFlags: {
    readonly exactShadow: number;
    readonly coneLightingRequested: number;
    readonly ambientOcclusion: number;
    readonly globalIllumination: number;
    readonly globalIlluminationOcclusion: number;
  };
  /** Mirrors SVO_DRY_SCENE_CAMERA_SETTLED_WGSL. */
  readonly cameraSettledExpression: string;
}

export function svoPixelTraceProbeRecordCapacity(options: Pick<SvoPixelTraceProbeOptions, "recordCapacity">): number {
  const capacity = options.recordCapacity ?? SVO_PIXEL_TRACE_DEFAULT_RECORD_CAPACITY;
  if (!Number.isSafeInteger(capacity) || capacity < 64 || capacity > SVO_PIXEL_TRACE_MAXIMUM_RECORD_CAPACITY) {
    throw new RangeError(`Pixel-trace probe capacity must be an integer from 64 to ${SVO_PIXEL_TRACE_MAXIMUM_RECORD_CAPACITY}`);
  }
  return capacity;
}

export function createSvoPixelTraceProbeWGSL(options: SvoPixelTraceProbeOptions): string {
  const group = options.group ?? 1;
  if (!Number.isInteger(group) || group < 1) throw new RangeError("Pixel-trace probe group must be a positive integer");
  const capacity = svoPixelTraceProbeRecordCapacity(options);
  if (!(options.coneLodBlendBandWidth > 0) || options.coneLodBlendBandWidth > 1) {
    throw new RangeError("Pixel-trace probe cone LOD blend band must be inside (0, 1]");
  }
  const header = SVO_PIXEL_TRACE_HEADER;
  const kinds = SVO_PIXEL_TRACE_KINDS;
  const flags = SVO_PIXEL_TRACE_FLAGS;
  const bandStart = (1 - options.coneLodBlendBandWidth).toFixed(8);
  const bandScale = (1 / options.coneLodBlendBandWidth).toFixed(8);
  return /* wgsl */ `
// ---------------------------------------------------------------------------
// Live pixel-trace probe. Records the work one camera ray performs.
// ---------------------------------------------------------------------------
@group(${group}) @binding(0) var probeRecords:texture_storage_2d<r32uint,write>;
// x,y requested pixel; z request token; w non-zero arms the probe.
@group(${group}) @binding(1) var<uniform> probeRequest:vec4u;

const PROBE_HEADER_WORDS:u32=${SVO_PIXEL_TRACE_HEADER_WORDS}u;
const PROBE_RECORD_WORDS:u32=${SVO_PIXEL_TRACE_RECORD_WORDS}u;
const PROBE_CAPACITY:u32=${capacity}u;
const PROBE_ROW_WORDS:u32=${SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS}u;
const PROBE_STACK_CAPACITY:u32=32u;

// Flat word index into the record texture. Rows are contiguous in the readback,
// so the host decodes exactly the word order written here.
fn probeWriteWord(wordIndex:u32,value:u32){
  textureStore(probeRecords,vec2u(wordIndex%PROBE_ROW_WORDS,wordIndex/PROBE_ROW_WORDS),vec4u(value,0u,0u,0u));
}
fn probeWriteFloat(wordIndex:u32,value:f32){probeWriteWord(wordIndex,bitcast<u32>(value));}

var<private> probeCursor:u32;
var<private> probeDropped:u32;
var<private> probeNodeVisits:u32;
var<private> probeLeafVisits:u32;
var<private> probeEmptyBrickSkips:u32;
var<private> probeVoxelWork:u32;
var<private> probeExactTests:u32;
var<private> probeMaximumDepth:u32;
var<private> probeShadowNodeVisits:u32;
var<private> probeShadowLeafVisits:u32;
var<private> probeShadowWork:u32;
var<private> probeMipSteps:u32;
var<private> probeFailure:u32;
var<private> probeShadedLights:u32;
var<private> probeGiState:u32;
var<private> probeGiConeTaps:u32;
var<private> probeGiConeCount:u32;
var<private> probeGiVisibility:f32;
var<private> probeGiRadiance:vec3f;

struct ProbeStackEntry{nodeIndex:u32,tEnter:f32,tExit:f32}
var<private> probeStack:array<ProbeStackEntry,32>;
var<private> probeStackSize:u32;

fn probeRecord(kind:u32,level:u32,detail:u32,recordFlags:u32,a:vec3f,b:vec3f,tEnter:f32,tExit:f32){
  let index=probeCursor;
  probeCursor+=1u;
  if(index>=PROBE_CAPACITY){probeDropped+=1u;return;}
  let base=PROBE_HEADER_WORDS+index*PROBE_RECORD_WORDS;
  probeWriteWord(base,kind);
  probeWriteWord(base+1u,level);
  probeWriteWord(base+2u,detail);
  probeWriteWord(base+3u,recordFlags|(min(probeStackSize,255u)<<24u));
  probeWriteFloat(base+4u,a.x);
  probeWriteFloat(base+5u,a.y);
  probeWriteFloat(base+6u,a.z);
  probeWriteFloat(base+7u,b.x);
  probeWriteFloat(base+8u,b.y);
  probeWriteFloat(base+9u,b.z);
  probeWriteFloat(base+10u,tEnter);
  probeWriteFloat(base+11u,tExit);
}

fn probeRecordBox(kind:u32,level:u32,detail:u32,recordFlags:u32,bounds:mat2x3f,tEnter:f32,tExit:f32){
  probeRecord(kind,level,detail,recordFlags,bounds[0],bounds[1],tEnter,tExit);
}

// Mirror of traceLeafPayload: the same 32-step DDA over the same brick lattice,
// with a record for every cell entered and every analytic test issued.
fn probeTraceLeafPayload(ro:vec3f,rd:vec3f,hit:SvoTraversalHit,bounds:mat2x3f)->DryHit{
  let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);
  var entry=max(hit.tEnter,0.0);
  let point=ro+rd*(entry+1e-5);
  var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));
  let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0));
  let nextBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ro)/rd,abs(rd)>vec3f(1e-9));
  let deltaT=select(vec3f(DRY_MISS),abs(extent/rd),abs(rd)>vec3f(1e-9));
  let tolerance=length(extent)*1.05;
  for(var iteration=0u;iteration<32u;iteration+=1u){
    if(any(cell<vec3i(0))||any(cell>=vec3i(i32(dry.mapping.brickSize)))||entry>hit.tExit){break;}
    probeVoxelWork+=1u;
    let cellMinimum=bounds[0]+vec3f(cell)*extent;
    let cellExit=min(min(nextT.x,nextT.y),min(nextT.z,hit.tExit));
    let payloadIndex=svoBrickVoxelIndex(hit.voxelOffset,vec3u(cell),dry.mapping.brickSize);
    var cellFlags=0u;
    var candidate=missHit();
    if(payloadIndex<arrayLength(&materialOwners)){
      let identity=materialOwners[payloadIndex];
      let owner=identity>>16u;
      if(owner>=dry.metadata.y&&!dryOpaqueOwnerSuppressed(owner)){
        let primitiveIndex=owner-dry.metadata.y;
        if(primitiveIndex<dry.metadata.x&&primitiveIndex<arrayLength(&primitives)){
          cellFlags|=${flags.tagged}u;
          probeExactTests+=1u;
          candidate=primitiveHit(primitives[primitiveIndex],ro,rd,max(0.0,entry-tolerance),cellExit+tolerance);
        }
      }
    }
    probeRecord(${kinds.brickCell}u,hit.level,payloadIndex,cellFlags,cellMinimum,cellMinimum+extent,entry,cellExit);
    if((cellFlags&${flags.tagged}u)!=0u){
      let exactFlags=select(0u,${flags.hit}u,candidate.t<DRY_MISS);
      probeRecord(${kinds.exactTest}u,hit.level,payloadIndex,exactFlags,cellMinimum,cellMinimum+extent,entry,cellExit);
    }
    if(candidate.t<DRY_MISS){return candidate;}
    let advance=min(nextT.x,min(nextT.y,nextT.z));
    if(nextT.x<=advance+1e-6){cell.x+=step.x;nextT.x+=deltaT.x;}
    if(nextT.y<=advance+1e-6){cell.y+=step.y;nextT.y+=deltaT.y;}
    if(nextT.z<=advance+1e-6){cell.z+=step.z;nextT.z+=deltaT.z;}
    entry=advance;
  }
  return missHit();
}

// Mirror of traceStatic composed with svoTraversalContinuationNext: one
// persistent near-to-far stack, leaves yielded nearest-first, tMin advanced
// past a brick that held no surface.
fn probeTraceStatic(ro:vec3f,rd:vec3f)->DryHit{
  if(publicationState[0]==0u||(publicationState[1]&REQUIRED_FIELDS)!=REQUIRED_FIELDS){probeFailure=max(probeFailure,2u);return missHit();}
  if(svoControl[12]!=0u){probeFailure=max(probeFailure,2u);return missHit();}
  let mapping=dryConfiguredMapping();
  if(mapping.nodeCount==0u){return missHit();}
  let maximumDepth=dryDiagnosticMaximumDepth();
  let leafBudget=clamp(dry.tuningCounts0.x,1u,${options.primaryLeafVisitHardLimit}u);
  let inverseDirection=1.0/rd;
  let rootBounds=svoRootBounds(mapping);
  var minimum=0.0;
  let rootInterval=svoRayAabbWithInverse(SvoRay(ro,minimum,rd,DRY_MISS),inverseDirection,rootBounds);
  if(rootInterval.x==0.0){return missHit();}
  var current=ProbeStackEntry(0u,rootInterval.y,rootInterval.z);
  var currentBounds=rootBounds;
  var currentBoundsValid=true;
  probeStackSize=0u;
  var leafVisit=0u;
  var guard=0u;
  loop{
    guard+=1u;
    if(guard>2048u){probeFailure=max(probeFailure,1u);break;}
    if(current.tExit<minimum){
      if(probeStackSize==0u){break;}
      probeStackSize-=1u;
      current=probeStack[probeStackSize];
      currentBoundsValid=false;
      continue;
    }
    if(current.nodeIndex>=mapping.nodeCount){probeFailure=2u;break;}
    let node=svoNodes[current.nodeIndex];
    probeNodeVisits+=1u;
    if(node.address.z>mapping.maximumDepth){probeFailure=2u;break;}
    if(node.address.z>maximumDepth){probeFailure=max(probeFailure,1u);break;}
    var bounds=currentBounds;
    if(!currentBoundsValid){bounds=svoNodeBounds(node,mapping);currentBounds=bounds;currentBoundsValid=true;}
    if(node.links.z!=SVO_INVALID){
      if(node.links.z>=mapping.leafCount){probeFailure=2u;break;}
      let leaf=svoLeaves[node.links.z];
      if(leaf.topology.x!=current.nodeIndex){probeFailure=2u;break;}
      probeLeafVisits+=1u;
      probeMaximumDepth=max(probeMaximumDepth,node.address.z);
      let leafHit=SvoTraversalHit(SVO_STATUS_HIT,1u,current.nodeIndex,node.links.z,leaf.topology.y,
        node.address.z,max(current.tEnter,minimum),current.tExit);
      let payload=probeTraceLeafPayload(ro,rd,leafHit,bounds);
      let occupied=payload.t<DRY_MISS;
      probeRecordBox(${kinds.leafBrick}u,node.address.z,node.links.z,
        select(${flags.emptySkip}u,${flags.hit}u,occupied),bounds,leafHit.tEnter,leafHit.tExit);
      if(occupied){return payload;}
      probeEmptyBrickSkips+=1u;
      minimum=leafHit.tExit+max(1e-5,length(dry.mapping.cellSize)*1e-3);
      leafVisit+=1u;
      if(leafVisit>=leafBudget){probeFailure=max(probeFailure,1u);break;}
      if(probeStackSize==0u){break;}
      probeStackSize-=1u;
      current=probeStack[probeStackSize];
      currentBoundsValid=false;
      continue;
    }
    let mask=node.address.w&0xffu;
    if(mask==0u){
      if(probeStackSize==0u){break;}
      probeStackSize-=1u;
      current=probeStack[probeStackSize];
      currentBoundsValid=false;
      continue;
    }
    if(node.links.x==SVO_INVALID||countOneBits(mask)!=node.links.y){probeFailure=2u;break;}
    probeRecordBox(${kinds.hierarchyNode}u,node.address.z,current.nodeIndex,0u,bounds,current.tEnter,current.tExit);
    // Child expansion. Rejected children are recorded where they are rejected;
    // accepted ones are recorded after the sort so the record order matches the
    // order traversal will actually visit them.
    var candidateIndex=array<u32,8>(0u,0u,0u,0u,0u,0u,0u,0u);
    var candidateOctant=array<u32,8>(0u,0u,0u,0u,0u,0u,0u,0u);
    var candidateEnter=array<f32,8>(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
    var candidateExit=array<f32,8>(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
    var candidateCount=0u;
    var invalid=false;
    for(var octant=0u;octant<8u;octant+=1u){
      if((mask&(1u<<octant))==0u){continue;}
      let childIndex=node.links.x+svoPopcountBefore(mask,octant);
      if(childIndex>=mapping.nodeCount){invalid=true;break;}
      let child=svoNodes[childIndex];
      if(child.address.z!=node.address.z+1u){invalid=true;break;}
      let childBounds=svoChildBounds(bounds,octant);
      let interval=svoRayAabbWithInverse(SvoRay(ro,minimum,rd,DRY_MISS),inverseDirection,childBounds);
      if(interval.x==0.0){
        probeRecordBox(${kinds.childRejected}u,child.address.z,childIndex,0u,childBounds,0.0,0.0);
        continue;
      }
      // Insertion sort by (tEnter, tExit, nodeIndex): the same comparison
      // sequence the shipping expansion uses, kept nearest-first here.
      var insertion=candidateCount;
      loop{
        if(insertion==0u){break;}
        let settledEnter=candidateEnter[insertion-1u];
        let settledExit=candidateExit[insertion-1u];
        let settledIndex=candidateIndex[insertion-1u];
        let ordered=settledEnter<interval.y
          ||(settledEnter==interval.y&&(settledExit<interval.z
          ||(settledExit==interval.z&&settledIndex<=childIndex)));
        if(ordered){break;}
        candidateIndex[insertion]=settledIndex;
        candidateOctant[insertion]=candidateOctant[insertion-1u];
        candidateEnter[insertion]=settledEnter;
        candidateExit[insertion]=settledExit;
        insertion-=1u;
      }
      candidateIndex[insertion]=childIndex;
      candidateOctant[insertion]=octant;
      candidateEnter[insertion]=interval.y;
      candidateExit[insertion]=interval.z;
      candidateCount+=1u;
    }
    if(invalid){probeFailure=2u;break;}
    if(candidateCount==0u){
      if(probeStackSize==0u){break;}
      probeStackSize-=1u;
      current=probeStack[probeStackSize];
      currentBoundsValid=false;
      continue;
    }
    if(probeStackSize+candidateCount-1u>PROBE_STACK_CAPACITY){probeFailure=max(probeFailure,1u);break;}
    for(var slot=0u;slot<8u;slot+=1u){
      if(slot>=candidateCount){break;}
      let childBounds=svoChildBounds(bounds,candidateOctant[slot]);
      probeRecordBox(${kinds.childAccepted}u,node.address.z+1u,candidateIndex[slot],
        select(${flags.deferred}u,${flags.descended}u,slot==0u),childBounds,candidateEnter[slot],candidateExit[slot]);
    }
    // LIFO: farthest pushed first so the nearest child is the next node read.
    var reverse=candidateCount;
    loop{
      if(reverse<=1u){break;}
      reverse-=1u;
      probeStack[probeStackSize]=ProbeStackEntry(candidateIndex[reverse],candidateEnter[reverse],candidateExit[reverse]);
      probeStackSize+=1u;
    }
    current=ProbeStackEntry(candidateIndex[0],candidateEnter[0],candidateExit[0]);
    currentBounds=svoChildBounds(bounds,candidateOctant[0]);
    currentBoundsValid=true;
  }
  return missHit();
}

// Mirror of dryConeVisibility's march. Step distances, LOD selection, coverage
// blending, receiver-plane suppression, and the emitter-anchored second phase
// are the shipping arithmetic; the record is the addition. Any change to the
// production step law belongs here too, or the drawn cone stops being the cone.
fn probeConeMarch(kind:u32,detail:u32,origin_m:vec3f,direction:vec3f,aperture:f32,maximumDistance_m:f32,surfaceNormal:vec3f,anchored:bool)->f32{
  if(!dryNodeMipReady()){return 1.0;}
  let minimumVoxel=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let tangent=tan(aperture*.5);
  var distance=minimumVoxel*.75;
  var transmittance=1.0;
  var pageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u);
  var pageCacheCoarse=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u);
  let shadowCone=dot(surfaceNormal,surfaceNormal)>.25;
  var budget=clamp(dry.tuningCounts0.y,1u,48u);
  let phaseSplit=select(maximumDistance_m,maximumDistance_m*.5,anchored);
  for(var stepIndex=0u;stepIndex<48u&&budget>0u&&distance<phaseSplit&&transmittance>.005;stepIndex+=1u){
    budget-=1u;
    probeMipSteps+=1u;
    let diameter=max(minimumVoxel,2.0*distance*tangent);
    let lod=svoNodeMipLod(diameter,minimumVoxel);
    let remaining=maximumDistance_m-distance;
    let stepWidth=min(diameter,remaining);
    let position=origin_m+direction*distance;
    let lookup=dryNodeMipAt(position,lod,&pageCache);
    if(lookup.valid==0u){return 1.0;}
    var selfWeight=1.0;
    if(shadowCone){selfWeight=max(clamp(dot(position-origin_m,surfaceNormal)/(1.5*diameter)-1.0,0.0,1.0),clamp((distance-12.0*minimumVoxel)/(12.0*minimumVoxel),0.0,1.0))*clamp(remaining/(1.5*diameter),0.0,1.0);}
    let blendWeight=clamp((fract(lod)-${bandStart})*${bandScale},0.0,1.0);
    var coverage=max(lookup.sample.solidMean,lookup.sample.solidMaximum*.15);
    if(blendWeight>0.0){
      let lookupCoarse=dryNodeMipAt(position,lod+1.0,&pageCacheCoarse);
      if(lookupCoarse.valid==0u){return 1.0;}
      coverage=mix(coverage,max(lookupCoarse.sample.solidMean,lookupCoarse.sample.solidMaximum*.15),blendWeight);
    }
    let conservativeCoverage=selfWeight*coverage;
    transmittance*=1.0-svoNodeMipCoverageOpacity(conservativeCoverage,stepWidth/diameter);
    probeRecord(kind,u32(max(floor(lod),0.0)),detail,0u,position,direction,diameter*.5,conservativeCoverage);
    distance+=max(stepWidth,minimumVoxel*.25);
  }
  if(anchored){
    var emitterOffset=minimumVoxel*3.0;
    for(var rung=0u;rung<48u&&budget>0u&&emitterOffset<maximumDistance_m-phaseSplit&&transmittance>.005;rung+=1u){
      budget-=1u;
      probeMipSteps+=1u;
      let rungDistance=maximumDistance_m-emitterOffset;
      let diameter=max(minimumVoxel,2.0*rungDistance*tangent);
      let lod=svoNodeMipLod(diameter,minimumVoxel);
      let remaining=emitterOffset;
      let stepWidth=emitterOffset*.5;
      let position=origin_m+direction*rungDistance;
      let lookup=dryNodeMipAt(position,lod,&pageCache);
      if(lookup.valid==0u){return 1.0;}
      var selfWeight=1.0;
      if(shadowCone){selfWeight=max(clamp(dot(position-origin_m,surfaceNormal)/(1.5*diameter)-1.0,0.0,1.0),clamp((rungDistance-12.0*minimumVoxel)/(12.0*minimumVoxel),0.0,1.0))*clamp(remaining/(1.5*diameter),0.0,1.0);}
      let blendWeight=clamp((fract(lod)-${bandStart})*${bandScale},0.0,1.0);
      var coverage=max(lookup.sample.solidMean,lookup.sample.solidMaximum*.15);
      if(blendWeight>0.0){
        let lookupCoarse=dryNodeMipAt(position,lod+1.0,&pageCacheCoarse);
        if(lookupCoarse.valid==0u){return 1.0;}
        coverage=mix(coverage,max(lookupCoarse.sample.solidMean,lookupCoarse.sample.solidMaximum*.15),blendWeight);
      }
      let conservativeCoverage=selfWeight*coverage;
      transmittance*=1.0-svoNodeMipCoverageOpacity(conservativeCoverage,stepWidth/diameter);
      probeRecord(kind,u32(max(floor(lod),0.0)),detail,${flags.emitterAnchored}u,position,direction,diameter*.5,conservativeCoverage);
      emitterOffset*=1.5;
    }
  }
  return clamp(transmittance,0.0,1.0);
}

// Mirror of the shadow half of shadeDryOpaque: one biased visibility ray per
// accepted light sample, plus the cone march that actually answers it.
fn probeLightVisibility(position:vec3f,geometricNormal:vec3f,ownerId:u32){
  let globalIllumination=(dry.materialPublication.w&${options.visibilityFlags.globalIllumination}u)!=0u;
  let lightCount=select(min(dryLighting.metadata.x,min(dry.tuningCounts0.z,${options.maximumShadedLights}u)),min(dryLighting.metadata.x,1u),globalIllumination);
  let coneRequested=(dry.materialPublication.w&${options.visibilityFlags.coneLightingRequested}u)!=0u
    &&(dry.materialPublication.w&${options.visibilityFlags.globalIllumination}u)==0u&&dryNodeMipReady();
  for(var lightIndex=0u;lightIndex<${options.maximumShadedLights}u;lightIndex+=1u){
    if(lightIndex>=lightCount){break;}
    let light=dryLighting.lights[lightIndex];
    if(light.identity.w!=dryLighting.metadata.y){continue;}
    let area=light.identity.x==SVO_LIGHT_SPHERE_AREA||light.identity.x==SVO_LIGHT_RECTANGLE_AREA;
    let sampleCount=select(select(1u,select(dry.tuningCounts1.x,dry.tuningCounts0.w,${options.cameraSettledExpression}),area),1u,globalIllumination);
    for(var sampleIndex=0u;sampleIndex<${options.areaLightSamples}u;sampleIndex+=1u){
      if(sampleIndex>=sampleCount){break;}
      let sample=dryLightSample(light,sampleIndex,position);
      if(sample.valid==0u||dot(geometricNormal,sample.towardLight)<=0.0){continue;}
      let maximumDistance=select(directionalLightSceneExitDistance(position,sample.towardLight),sample.finiteDistance_m,sample.finiteDistance_m>0.0);
      if(maximumDistance<=0.0){continue;}
      probeShadedLights=max(probeShadedLights,lightIndex+1u);
      let ray=dryBiasedVisibilityRayUnit(position,geometricNormal,sample.towardLight,maximumDistance,dry.mapping.cellSize,dry.tuningRays0.x);
      var transmittance=1.0;
      if(coneRequested){
        let coneCell_m=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
        let coneEscape_m=coneCell_m*dry.tuningRays1.z;
        let coneMaxRaw_m=max(0.0,ray.tMax_m-coneEscape_m*dot(geometricNormal,sample.towardLight));
        let coneMax_m=coneMaxRaw_m-select(0.0,dry.tuningRays1.w*coneCell_m,sample.finiteDistance_m>0.0);
        // detail names this march, not this light: an area light runs one cone
        // per sample, and the overlay stitches a cone from the taps that share a
        // detail, so two marches sharing a number would be drawn as one cone.
        transmittance=probeConeMarch(${kinds.coneSample}u,lightIndex*${options.areaLightSamples}u+sampleIndex,
          ray.origin_m+geometricNormal*coneEscape_m,
          sample.towardLight,dry.tuningRays1.y,coneMax_m,geometricNormal,sample.finiteDistance_m>0.0);
      }else if((dry.materialPublication.w&${options.visibilityFlags.exactShadow}u)!=0u){
        // No cone pyramid: the shipping path walks the hierarchy exactly. Run
        // the real traversal so the reported shadow work is the real cost, even
        // though the individual boxes it opens are not recorded as geometry.
        dryVisibilityIgnoredOwner=ownerId;
        let exact=svoTraceVisibility(ray,SvoVisibilityBudget(dry.tuningCounts1.w,dry.tuningCounts2.x,dry.tuningCounts2.y,dry.tuningCounts2.z),true,0.001,max(ray.originBias_m,1e-6));
        dryVisibilityIgnoredOwner=DRY_OWNER_NONE;
        probeShadowNodeVisits+=exact.nodeVisits;
        probeShadowLeafVisits+=exact.leafVisits;
        probeShadowWork+=exact.workItems;
        transmittance=dot(clamp(exact.transmittance,vec3f(0.0),vec3f(1.0)),vec3f(1.0/3.0));
      }
      // level carries transmittance in milli-units; the ray is marked hit when
      // more than half the light was lost along it.
      probeRecord(${kinds.shadowRay}u,u32(round(clamp(transmittance,0.0,1.0)*1000.0)),lightIndex,
        select(0u,${flags.hit}u,transmittance<.5),ray.origin_m,ray.origin_m+sample.towardLight*ray.tMax_m,
        0.0,ray.tMax_m);
    }
  }
}

// Mirror of dryContactVisibility's cone branch.
fn probeContactVisibility(position:vec3f,geometricNormal:vec3f,featureId:u32){
  if((dry.materialPublication.w&${options.visibilityFlags.ambientOcclusion}u)==0u){return;}
  if((dry.materialPublication.w&${options.visibilityFlags.coneLightingRequested}u)==0u||!dryNodeMipReady()){return;}
  let radius=dryContactVisibilityRadius();
  if(radius<=0.0){return;}
  let cellScale=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let origin=position+normalize(geometricNormal)*cellScale*.2;
  let coneSampleCount=select(dry.tuningCounts1.z,dry.tuningCounts1.y,${options.cameraSettledExpression});
  for(var sampleIndex=0u;sampleIndex<${options.stableOcclusionConeSamples}u;sampleIndex+=1u){
    if(sampleIndex>=coneSampleCount){break;}
    let direction=dryContactVisibilityDirection(geometricNormal,featureId,sampleIndex&1u);
    let rotated=select(direction,normalize(direction+cross(normalize(geometricNormal),direction)*.7),sampleIndex>=2u);
    probeConeMarch(${kinds.occlusionConeSample}u,sampleIndex,origin,rotated,dry.tuningRays1.x,radius,vec3f(0.0),false);
  }
}

// The shipping integrator remains authoritative for energy, opacity, validity,
// and termination. This replay emits only the geometry of exactly the taps it
// took. Avoiding a second texture-sampling integration also keeps diagnostic
// work proportional to the shipping gather instead of nearly doubling it.
fn probeGiConeMarch(detail:u32,origin:vec3f,directionIn:vec3f,maximumSteps:u32)->SvoTetraRadianceConeResult{
  let direction=normalize(directionIn);let minimumVoxel=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let maximumDistance=dryNodeMipSceneExitDistance(origin,direction);let config=SvoTetraRadianceConeConfig(origin,direction,dry.giCones.x,minimumVoxel,maximumDistance,maximumSteps,.995,.0039215686,1u);
  dryGiPageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u);
  let result=svoTetraRadianceConeTrace(config);let tangent=tan(dry.giCones.x*.5);var distance=minimumVoxel*.5;
  for(var boundedStep=0u;boundedStep<64u;boundedStep+=1u){
    if(boundedStep>=result.coneTaps||distance>=maximumDistance){break;}
    let diameter=max(minimumVoxel,2.0*distance*tangent);let lod=svoNodeMipLod(diameter,minimumVoxel);
    let voxelWidth=minimumVoxel*exp2(floor(lod));let step=min(max(voxelWidth,diameter*.5),maximumDistance-distance);
    let position=origin+direction*distance;probeMipSteps+=1u;
    probeRecord(${kinds.globalIlluminationConeSample}u,u32(max(floor(lod),0.0)),detail,0u,position,direction,diameter*.5,0.0);
    distance+=max(step,minimumVoxel*.25);
  }
  return result;
}

fn probeGlobalIllumination(position:vec3f,normal:vec3f){
  if((dry.materialPublication.w&${options.visibilityFlags.globalIllumination}u)==0u){return;}
  probeGiState=${SVO_PIXEL_TRACE_GI_STATE.enabled}u;
  if(!dryTetraRadianceReady()){return;}
  probeGiState|=${SVO_PIXEL_TRACE_GI_STATE.ready}u;
  let minimumVoxel=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let origin=position+normalize(normal)*minimumVoxel*max(dry.tuningRays1.z,1.0);
  let coneCount=clamp(u32(round(dry.giCones.y)),3u,4u);let perConeBudget=max(1u,min(64u,dry.tuningCounts0.y)/coneCount);
  probeGiConeCount=coneCount;var radiance=vec3f(0.0);var visibility=0.0;
  for(var coneIndex=0u;coneIndex<4u;coneIndex+=1u){
    if(coneIndex>=coneCount){break;}let weight=svoTetraRadianceHemisphereWeight(coneIndex,coneCount);
    let result=probeGiConeMarch(coneIndex,origin,svoTetraRadianceHemisphereDirection(normal,coneIndex,coneCount,0.0),perConeBudget);
    probeGiConeTaps+=result.coneTaps;radiance+=result.radiance*weight;visibility+=result.transmittance*weight;
  }
  let occlusionStrength=select(0.0,dry.giLighting.y,(dry.materialPublication.w&${options.visibilityFlags.globalIlluminationOcclusion}u)!=0u);
  probeGiRadiance=max(radiance,vec3f(0.0))*dry.giLighting.x;
  probeGiVisibility=mix(1.0,clamp(visibility,0.0,1.0),occlusionStrength);
}

@fragment fn dryProbeMain(input:VertexOut)->@location(0) vec4f{
  // One writer only. The probe target is 1x1, so every other invocation in the
  // rasterizer's quad returns here before it can touch storage. This is an
  // early return rather than a discard: a discarded invocation's stores are
  // implementation-defined, a returned one's plainly never happen.
  if(u32(input.position.x)!=0u||u32(input.position.y)!=0u){return vec4f(0.0);}
  probeCursor=0u;probeDropped=0u;probeStackSize=0u;
  probeNodeVisits=0u;probeLeafVisits=0u;probeEmptyBrickSkips=0u;probeVoxelWork=0u;probeExactTests=0u;
  probeMaximumDepth=0u;probeShadowNodeVisits=0u;probeShadowLeafVisits=0u;probeShadowWork=0u;
  probeMipSteps=0u;probeFailure=0u;probeShadedLights=0u;
  probeGiState=0u;probeGiConeTaps=0u;probeGiConeCount=0u;probeGiVisibility=1.0;probeGiRadiance=vec3f(0.0);
  dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassEnabled=0u;dryThickGlassFailure=0u;dryShadowTracingEnabled=1u;
  dryPrimaryNodeVisits=0u;dryPrimaryLeafVisits=0u;dryPrimaryEmptyBrickSkips=0u;dryPrimaryVoxelWorkItems=0u;
  dryPrimaryExactTests=0u;dryPrimaryMaximumDepth=0u;dryShadowNodeVisits=0u;dryShadowLeafVisits=0u;
  dryShadowWorkItems=0u;dryMipSteps=0u;dryTraversalFailure=0u;

  // The record texture is write-only, so the request arrives in the uniform.
  if(probeRequest.w==0u){return vec4f(0.0);}
  let pixel=vec2f(f32(probeRequest.x),f32(probeRequest.y));
  let viewport=max(uniforms.viewport.xy,vec2f(1.0));
  // Same camera as every production entry point, sampled at the requested pixel.
  let uv=vec2f((pixel.x+.5)/viewport.x,1.0-(pixel.y+.5)/viewport.y);
  let ndc=uv*2.0-1.0;
  let ro=uniforms.cameraPosition.xyz;
  let forward=normalize(uniforms.cameraTarget.xyz-ro);
  let right=normalize(cross(forward,vec3f(0,1,0)));
  let up=normalize(cross(right,forward));
  let rd=normalize(forward+right*ndc.x*viewport.x/viewport.y*.72+up*ndc.y*.72);

  let staticHit=probeTraceStatic(ro,rd);
  // Compose the authoritative hit from the instrumented static result plus the
  // same terrain and rigid intersectors as traceDrySolidScene. Calling
  // traceOpaqueScene here would traverse the complete static SVO a second time
  // and makes Dawn specialize two independent primary traversers into this one
  // diagnostic fragment pipeline.
  var opaque=staticHit;
  let terrain=traceTerrain(ro,rd);if(terrain.t<opaque.t){opaque=terrain;}
  let rigid=nearestBody(ro,rd);if(rigid.t<opaque.t){opaque=rigid;}
  var status=${SVO_PIXEL_TRACE_STATUS.miss}u;
  if(opaque.t<DRY_MISS){
    status=${SVO_PIXEL_TRACE_STATUS.hit}u;
    let position=ro+rd*opaque.t;
    let geometricNormal=normalize(opaque.normal);
    let cellScale=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
    probeRecord(${kinds.primaryHit}u,0u,opaque.ownerId,${flags.hit}u,position,geometricNormal,opaque.t,cellScale*.75);
    probeLightVisibility(position,geometricNormal,opaque.ownerId);
    probeContactVisibility(position,geometricNormal,opaque.featureId);
    probeGlobalIllumination(position,geometricNormal);
    probeWriteFloat(${header.hitNormal}u,geometricNormal.x);
    probeWriteFloat(${header.hitNormal + 1}u,geometricNormal.y);
    probeWriteFloat(${header.hitNormal + 2}u,geometricNormal.z);
  }
  if(probeFailure>0u){status=select(${SVO_PIXEL_TRACE_STATUS.exhausted}u,${SVO_PIXEL_TRACE_STATUS.invalid}u,probeFailure>=2u);}
  if(staticHit.t<DRY_MISS&&status==${SVO_PIXEL_TRACE_STATUS.miss}u){status=${SVO_PIXEL_TRACE_STATUS.hit}u;}

  // The host reads a texture it never wrote, so identity comes from the shader:
  // the magic word is what tells the decode this content is a trace at all.
  probeWriteWord(${header.magic}u,${SVO_PIXEL_TRACE_MAGIC}u);
  probeWriteWord(${header.pixelX}u,probeRequest.x);
  probeWriteWord(${header.pixelY}u,probeRequest.y);
  probeWriteWord(${header.requestToken}u,probeRequest.z);
  probeWriteWord(${header.status}u,status);
  probeWriteWord(${header.recordCount}u,probeCursor);
  probeWriteWord(${header.droppedRecords}u,probeDropped);
  probeWriteFloat(${header.rayOrigin}u,ro.x);
  probeWriteFloat(${header.rayOrigin + 1}u,ro.y);
  probeWriteFloat(${header.rayOrigin + 2}u,ro.z);
  probeWriteFloat(${header.rayDirection}u,rd.x);
  probeWriteFloat(${header.rayDirection + 1}u,rd.y);
  probeWriteFloat(${header.rayDirection + 2}u,rd.z);
  probeWriteFloat(${header.hitDistance}u,select(0.0,opaque.t,opaque.t<DRY_MISS));
  // The finest voxel width, so the overlay can draw a cone tap's mip footprint
  // at the width it really covered: level L is this doubled L times.
  probeWriteFloat(${header.minimumVoxel}u,max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z)));
  probeWriteWord(${header.nodeVisits}u,probeNodeVisits);
  probeWriteWord(${header.leafVisits}u,probeLeafVisits);
  probeWriteWord(${header.emptyBrickSkips}u,probeEmptyBrickSkips);
  probeWriteWord(${header.voxelWork}u,probeVoxelWork);
  probeWriteWord(${header.exactTests}u,probeExactTests);
  probeWriteWord(${header.maximumDepth}u,probeMaximumDepth);
  probeWriteWord(${header.shadowNodeVisits}u,probeShadowNodeVisits);
  probeWriteWord(${header.shadowLeafVisits}u,probeShadowLeafVisits);
  probeWriteWord(${header.shadowWork}u,probeShadowWork);
  probeWriteWord(${header.mipSteps}u,probeMipSteps);
  probeWriteWord(${header.traversalFailure}u,probeFailure);
  probeWriteWord(${header.hitOwnerId}u,opaque.ownerId);
  probeWriteWord(${header.hitMaterialId}u,dryResolvedMaterialId(opaque));
  probeWriteWord(${header.hitFeatureId}u,opaque.featureId);
  probeWriteWord(${header.shadedLights}u,probeShadedLights);
  probeWriteWord(${header.giConeTaps}u,probeGiConeTaps);
  probeWriteWord(${header.giConeCount}u,probeGiConeCount);
  probeWriteFloat(${header.giVisibility}u,probeGiVisibility);
  probeWriteFloat(${header.giRadiance}u,probeGiRadiance.x);
  probeWriteFloat(${header.giRadiance + 1}u,probeGiRadiance.y);
  probeWriteFloat(${header.giRadiance + 2}u,probeGiRadiance.z);
  probeWriteWord(${header.giState}u,probeGiState);
  return vec4f(0.0);
}
`;
}

/* ------------------------------------------------------------------------- */
/* Host side: the trace buffer, its per-request header, and the readback ring. */
/* ------------------------------------------------------------------------- */

export interface SvoPixelTraceRequest {
  readonly pixelX: number;
  readonly pixelY: number;
  readonly token: number;
}

/**
 * Owns the record texture the probe writes, the uniform that carries the
 * request in, and the mapped copies the records are read back through. Two
 * staging slots are enough: the probe runs at most once per frame and a request
 * is superseded rather than queued.
 */
export class SparseVoxelPixelTraceBuffers {
  readonly recordCapacity: number;
  readonly records: GPUTexture;
  readonly recordsView: GPUTextureView;
  readonly request: GPUBuffer;
  private readonly rows: number;
  private readonly staging: { buffer: GPUBuffer; pending: boolean }[];
  private readonly requestWords = new Uint32Array(new ArrayBuffer(16));
  private destroyed = false;

  constructor(private readonly device: GPUDevice, recordCapacity = SVO_PIXEL_TRACE_DEFAULT_RECORD_CAPACITY) {
    this.recordCapacity = svoPixelTraceProbeRecordCapacity({ recordCapacity });
    this.rows = svoPixelTraceTextureRows(this.recordCapacity);
    this.records = device.createTexture({
      label: "Sparse voxel pixel-trace records",
      size: [SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS, this.rows],
      format: "r32uint",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.recordsView = this.records.createView();
    this.request = device.createBuffer({
      label: "Sparse voxel pixel-trace request",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const size = svoPixelTraceBufferBytes(this.recordCapacity);
    this.staging = Array.from({ length: 2 }, (_, index) => ({
      buffer: device.createBuffer({
        label: `Sparse voxel pixel-trace readback ${index + 1}/2`,
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      pending: false,
    }));
  }

  /** Publish the request the next probe pass answers. */
  writeRequest(request: SvoPixelTraceRequest): void {
    if (this.destroyed) return;
    if (!Number.isSafeInteger(request.pixelX) || request.pixelX < 0
      || !Number.isSafeInteger(request.pixelY) || request.pixelY < 0) {
      throw new RangeError("Pixel-trace request coordinates must be non-negative safe integers");
    }
    this.requestWords.set([request.pixelX, request.pixelY, request.token >>> 0, 1]);
    this.device.queue.writeBuffer(this.request, 0, this.requestWords);
  }

  /** Copy the records into a free staging slot. False when both are in flight. */
  encodeReadback(encoder: GPUCommandEncoder): boolean {
    if (this.destroyed) return false;
    const slot = this.staging.find((candidate) => !candidate.pending);
    if (!slot) return false;
    slot.pending = true;
    // One row is exactly 1024 bytes, so the copy needs no padding and the
    // destination is the flat word array the decode expects.
    encoder.copyTextureToBuffer(
      { texture: this.records },
      { buffer: slot.buffer, bytesPerRow: SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS * 4, rowsPerImage: this.rows },
      { width: SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS, height: this.rows, depthOrArrayLayers: 1 },
    );
    this.pendingSlot = slot;
    return true;
  }

  private pendingSlot?: { buffer: GPUBuffer; pending: boolean };

  /**
   * Resolve the most recently encoded readback. `isCurrent` is re-checked after
   * the map completes so a trace whose camera or topology has moved on is
   * dropped rather than drawn against a stale scene.
   */
  async read(isCurrent: () => boolean): Promise<SvoPixelTrace | undefined> {
    const slot = this.pendingSlot;
    this.pendingSlot = undefined;
    if (!slot || this.destroyed) return undefined;
    try {
      await slot.buffer.mapAsync(GPUMapMode.READ);
      if (this.destroyed || !isCurrent()) return undefined;
      const bytes = slot.buffer.getMappedRange();
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      const trace = decodeSvoPixelTrace(words, floats);
      return trace && trace.status !== SVO_PIXEL_TRACE_STATUS.pending ? trace : undefined;
    } catch {
      return undefined;
    } finally {
      try { slot.buffer.unmap(); } catch { /* Device loss or destruction. */ }
      slot.pending = false;
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    try { this.records.destroy(); } catch { /* Device loss. */ }
    try { this.request.destroy(); } catch { /* Device loss. */ }
    for (const slot of this.staging) { try { slot.buffer.destroy(); } catch { /* Device loss. */ } }
  }
}
