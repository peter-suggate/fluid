import { makeOctreePowerCoarseLevelSetSampleWGSL } from "./octree-power-coarse-levelset-sample-abi";
import { fineLevelSetPackedSampleWGSL } from "./fine-levelset-packed-sample";
import {
  compactFineLevelSetPageLookupWGSL,
  makeCompactFineLevelSetPhiWGSL,
} from "./compact-fine-levelset-phi";

// The tagged sharp-corner polygonizer clips its two cap owners at exactly half
// a cube. Admit only scalar data whose four corresponding zero crossings agree
// with that geometry; moving approximate corners use the ordinary conforming
// tetrahedralization.
export const GLOBAL_FINE_SHARP_CORNER_HALF_CELL_EPSILON = 1e-3;

/**
 * Binding-minimal classifier for the Section 5 global narrow-band level set.
 * Fine phi is authoritative for every valid tagged sample. Missing or stale
 * fine samples query the compact coarse-octree directory; no dense phi texture
 * is reachable from this module.
 */
export const globalFineSurfaceClassificationShader = /* wgsl */ `
struct Uniforms{viewport:vec4f,cameraPosition:vec4f,cameraTarget:vec4f,container:vec4f,options:vec4f,gridInfo:vec4f,debug:vec4f}
struct IndirectArgs{vertexCount:atomic<u32>,instanceCount:u32,firstVertex:u32,firstInstance:u32,activeCubeCount:atomic<u32>,vertexAllocator:atomic<u32>,globalFineAuthorityLatch:atomic<u32>,meshPublicationGeneration:atomic<u32>}
struct FineParams{sampleDimensions:vec3u,brickResolution:u32,brickDimensions:vec3u,samplesPerBrick:u32,table:vec4u,settings:vec4f,cellAndDt:vec4f,sizing:vec4f,physical:vec4f}
@group(0)@binding(0)var<uniform>u:Uniforms;
@group(0)@binding(4)var<storage,read_write>drawArgs:IndirectArgs;
@group(0)@binding(5)var<storage,read_write>activeCubes:array<vec2u>;
@group(0)@binding(6)var<storage,read_write>cubeValues:array<vec4f>;
@group(0)@binding(8)var<storage,read>fineWorklist:array<u32>;
@group(0)@binding(9)var<storage,read>fineSamples:array<u32>;
@group(0)@binding(10)var<uniform>params:FineParams;
@group(0)@binding(12)var<storage,read>metadata:array<u32>;
${makeOctreePowerCoarseLevelSetSampleWGSL(16)}
@group(0)@binding(17)var<storage,read>fineTopologyControl:array<u32>;
const INVALID:u32=0xffffffffu;
const SHARP_CORNER_HALF_CELL_EPSILON:f32=${GLOBAL_FINE_SHARP_CORNER_HALF_CELL_EPSILON};
${fineLevelSetPackedSampleWGSL("fineSamples", false)}
fn validCurrentPublication()->bool{
  let rowCount=min(powerCoarseSamples.rowCount,arrayLength(&powerCoarseSamples.entries));
  let expectedWidth=params.settings.w*max(1.0,params.cellAndDt.x);
  let coarseGeneration=powerCoarseSamples.generation&0x3fffffffu;
  let coarsePublished=powerCoarseSamples.state==0x80000000u
    &&rowCount>0u&&powerCoarseSamples.rowCount==rowCount
    &&powerCoarseSamples.maximumLeafSize>0u&&(powerCoarseSamples.maximumLeafSize&(powerCoarseSamples.maximumLeafSize-1u))==0u
    &&all(powerCoarseSamples.dimensions*max(1u,u32(round(params.cellAndDt.x)))==params.sampleDimensions)
    &&powerCoarseSamples.physicalCellSize>0.0
    &&abs(powerCoarseSamples.physicalCellSize-expectedWidth)<=1e-5*max(powerCoarseSamples.physicalCellSize,expectedWidth);
  let generation=params.table.w&0x3fffffffu;
  // State 6 is coarse-1: the compact octree publication is the complete
  // moving surface. It deliberately has no fine worklist, pages or topology
  // transaction, so validation must finish before inspecting those sentinels.
  if(params.table.y==6u){return coarsePublished;}
  if(arrayLength(&fineWorklist)<7u){return false;}
  let count=fineWorklist[1];
  let finePublished=params.table.y==7u&&count<=params.table.x&&count<=params.table.z
    &&7u+count<=arrayLength(&fineWorklist)&&(fineWorklist[0]&0x3fffffffu)==generation
    &&fineWorklist[2]==params.table.z&&(fineWorklist[3]&3u)==3u
    &&fineWorklist[4]==(count+63u)/64u&&fineWorklist[5]==1u&&fineWorklist[6]==1u;
  // Sparse CM12 publishes a self-contained, key-sorted page set with a dry
  // support apron. It has no background-octree generation to pair and missing
  // logical pages are authoritative air, so the compact flag is its complete
  // publication transaction.
  if((fineWorklist[3]&0x80000000u)!=0u){return finePublished;}
  if(arrayLength(&fineTopologyControl)<8u){return false;}
  let topologyFlags=fineTopologyControl[0];let topologyReason=fineTopologyControl[7];
  let clean=topologyFlags==0u&&fineTopologyControl[4]==1u&&fineTopologyControl[5]==0u&&topologyReason==0u;
  // Aanjaneya et al. 2017 Section 5 uses separate background-octree and
  // fine-SPGrid meshes. A rejected adaptation intentionally retains the last
  // valid octree while the independently numbered fine SPGrid advances.
  let retained=fineTopologyControl[4]==1u&&fineTopologyControl[5]==1u
    &&topologyFlags!=0u&&topologyReason!=0u&&(topologyFlags&~0x1fu)==0u&&(topologyReason&~0x0fu)==0u;
  return finePublished&&coarsePublished&&((clean&&coarseGeneration==generation)||retained);
}
${compactFineLevelSetPageLookupWGSL}
fn coarsePhi(q:vec3i)->f32{
  if((fineWorklist[3]&0x80000000u)!=0u){return 4.0*params.settings.w;}
  return sampleCoarseOctreePhi(params.settings.xyz+(vec3f(q)+vec3f(0.5))*params.settings.w);
}
fn adaptiveNodalPublication()->bool{let count=min(powerCoarseSamples.rowCount,arrayLength(&powerCoarseSamples.entries));return count>0u&&(powerCoarseSamples.entries[0].flags&0x18000000u)==0x10000000u;}
fn finite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
${makeCompactFineLevelSetPhiWGSL("coarsePhi")}
fn fineOwnsCube(base:vec3i)->bool{let q=max(base-vec3i(1),vec3i(0));return all(q<vec3i(params.sampleDimensions))&&fineValid(vec3u(q));}
fn occupancy(value:f32)->f32{let band=4.0*u.container.y/max(f32(params.sampleDimensions.y),1.0);return clamp(0.5-value/band,0.0,1.0);}
// Adaptive nodal phi is a signed-distance scalar, not optical occupancy.
// Preserve its affine map through edge interpolation: clamping the two edge
// endpoints before solving the 0.5 crossing moves the zero set by an amount
// that grows with leaf/sample spacing (and makes the same plane look different
// beside span-1, span-2 and span-4 leaves). Values outside [0,1] are valid here;
// only their relation to 0.5 and their linear interpolation are consumed.
fn adaptiveContourValue(value:f32)->f32{let band=4.0*u.container.y/max(f32(params.sampleDimensions.y),1.0);return 0.5-value/band;}
// The ordinary halo represents one closed tank wall with an exterior air
// sample. At an x/z tank edge a single Cartesian-product halo cube cannot
// represent both perpendicular wall caps: marching tetrahedra turns the two
// planes into a diagonal chamfer. Boundary-mode bits mirror z (bit 0) and x
// (bit 1); mode 3 mirrors both after exact wall-plane records own the caps.
fn latticeForWall(p:vec3i,wallMode:u32)->f32{
  let dims=vec3i(params.sampleDimensions);
  if(p.y>=dims.y+1){return 0.0;}
  var x=p.x-1;var z=p.z-1;
  if(p.x<=0||p.x>=dims.x+1){
    if((wallMode&2u)==0u){return 0.0;}x=clamp(x,0,dims.x-1);
  }
  if(p.z<=0||p.z>=dims.z+1){
    if((wallMode&1u)==0u){return 0.0;}z=clamp(z,0,dims.z-1);
  }
  return occupancy(phi(vec3i(x,max(p.y-1,0),z)));
}
fn emitClassifiedCubeTagged(base:vec3i,scale:i32,lo:f32,hi:f32,a:vec4f,b:vec4f,tag:u32){
  if(lo>=0.5||hi<0.5){return;}atomicStore(&drawArgs.globalFineAuthorityLatch,1u);atomicMin(&drawArgs.vertexAllocator,0u);let slot=atomicAdd(&drawArgs.activeCubeCount,1u);
  if(slot>=arrayLength(&activeCubes)||slot*2u+1u>=arrayLength(&cubeValues)){return;}
  activeCubes[slot]=vec2u(u32(base.x)|(u32(base.z)<<16u),u32(base.y)|((u32(scale)|tag)<<16u));cubeValues[slot*2u]=a;cubeValues[slot*2u+1u]=b;
}
fn emitClassifiedCube(base:vec3i,scale:i32,lo:f32,hi:f32,a:vec4f,b:vec4f){
  emitClassifiedCubeTagged(base,scale,lo,hi,a,b,0u);
}
// Native adaptive cubes do not scan the optical ghost halo. Close the tank
// with explicit scalar-owned face records instead: reserved codes 252/253 select the low
// or high wall plane, while bits 14..15 select its normal axis. The four live
// face samples remain in the ordinary cube payload so emission can clip a
// fractional marching-squares contact polygon rather than invent a wet quad.
fn emitAdaptiveWallFace(base:vec3i,values:array<f32,8>,axis:u32,side:u32){
  var corners=vec4u(0u,1u,2u,3u);
  if(axis==0u){corners=select(vec4u(0u,3u,7u,4u),vec4u(1u,2u,6u,5u),side!=0u);}
  else if(axis==1u){corners=vec4u(0u,1u,5u,4u);}
  else{corners=select(vec4u(0u,1u,2u,3u),vec4u(4u,5u,6u,7u),side!=0u);}
  // An all-exact-zero face already has one native exact-plane owner below.
  // Require strict liquid evidence here so the wall path cannot duplicate it.
  var ownsLiquid=false;for(var i=0u;i<4u;i+=1u){ownsLiquid=ownsLiquid||values[corners[i]]>.5;}
  if(!ownsLiquid){return;}
  // Bits 8..13 carry log2(tangential scale)+1 for every wall-face record.
  // Adaptive nodal faces are unit scale.
  let tag=(1u<<8u)|(axis<<14u);
  emitClassifiedCubeTagged(base,i32(252u+side),0.,1.,vec4f(values[0],values[1],values[2],values[3]),vec4f(values[4],values[5],values[6],values[7]),tag);
}
// Native Sparse CM12 macro pages are sampled every 2*span finest cells. Do
// not reconstruct a closed tank wall by interpolating from the adjacent wet
// cell to a ghost-air sample across that whole distance: doing so moves the
// visible wall inward by scale/2 and makes a uniform pool look terraced. Emit
// the exact wall plane and use the interior boundary scalar only to clip its
// tangential extent at a real liquid-air interface.
fn emitScaledBoundaryWallFace(base:vec3i,scale:i32,axis:u32,side:u32){
  let dims=vec3i(params.sampleDimensions);
  let o=array<vec3i,8>(vec3i(0,0,0),vec3i(1,0,0),vec3i(1,1,0),vec3i(0,1,0),vec3i(0,0,1),vec3i(1,0,1),vec3i(1,1,1),vec3i(0,1,1));
  var values=array<f32,8>();var ownsLiquid=false;
  for(var i=0u;i<8u;i+=1u){
    var q=clamp(base+o[i]*scale-vec3i(1),vec3i(0),dims-vec3i(1));
    q[axis]=select(0,dims[axis]-1,side!=0u);
    values[i]=occupancy(phi(q));ownsLiquid=ownsLiquid||values[i]>=.5;
  }
  if(!ownsLiquid){return;}
  var wallBase=base;wallBase[axis]=select(0,dims[axis]-1,side!=0u);
  let scaleCode=1u+(31u-countLeadingZeros(u32(scale)));
  let tag=(scaleCode<<8u)|(axis<<14u);
  emitClassifiedCubeTagged(wallBase,i32(252u+side),0.,1.,vec4f(values[0],values[1],values[2],values[3]),vec4f(values[4],values[5],values[6],values[7]),tag);
}
fn classifyAdaptiveNodal(base:vec3i){
  if(any(base<vec3i(0))||any(base>=vec3i(params.sampleDimensions))){return;}
  let o=array<vec3i,8>(vec3i(0,0,0),vec3i(1,0,0),vec3i(1,1,0),vec3i(0,1,0),vec3i(0,0,1),vec3i(1,0,1),vec3i(1,1,1),vec3i(0,1,1));
  var raw=array<f32,8>();var v=array<f32,8>();var lo=1.0;var hi=0.0;var minimumSigned=3.402823e38;var maximumSigned=-3.402823e38;
  for(var i=0;i<8;i+=1){let signed=sampleAdaptiveCoarseOctreePhiAtGrid(vec3f(base+o[i]));if(!finite(signed)){return;}raw[i]=signed;minimumSigned=min(minimumSigned,signed);maximumSigned=max(maximumSigned,signed);v[i]=adaptiveContourValue(signed);lo=min(lo,v[i]);hi=max(hi,v[i]);}
  let dims=vec3i(params.sampleDimensions);
  if(base.x==0){emitAdaptiveWallFace(base,v,0u,0u);}if(base.x==dims.x-1){emitAdaptiveWallFace(base,v,0u,1u);}
  // The opaque floor is intentionally not a liquid-interface owner: a
  // coplanar floor cap corrupts the renderer's front/back layering. The x/z
  // owners below close their lower Cartesian edges without that overlap.
  if(base.z==0){emitAdaptiveWallFace(base,v,2u,0u);}if(base.z==dims.z-1){emitAdaptiveWallFace(base,v,2u,1u);}
  // Exact-zero geometry is owned only by the air-side cube. A whole zero
  // face gets one six-vertex Cartesian patch; an edge/corner-only touch has
  // zero area and publishes nothing. This removes the collapsed tetrahedra
  // that otherwise make an authored nodal plane cell-size dependent.
  if(minimumSigned==0.0&&maximumSigned>0.0){
    for(var axis=0u;axis<3u;axis+=1u){for(var side=0u;side<2u;side+=1u){var face=true;for(var corner=0u;corner<8u;corner+=1u){let coordinate=u32(o[corner][axis]);if(coordinate==side){face=face&&raw[corner]==0.0;}}if(face){let code=2u+side;let tag=(1u<<(8u+axis))|(axis<<14u);emitClassifiedCubeTagged(base,i32(code),lo,hi,vec4f(v[0],v[1],v[2],v[3]),vec4f(v[4],v[5],v[6],v[7]),tag);return;}}}
    return;
  }
  if(minimumSigned>=0.0||maximumSigned<0.0){return;}
  // A zero low-byte descriptor distinguishes the native nodal lattice from
  // the legacy optical ghost lattice. Emission still uses a unit span.
  emitClassifiedCubeTagged(base,0,lo,hi,vec4f(v[0],v[1],v[2],v[3]),vec4f(v[4],v[5],v[6],v[7]),0u);
}
fn classifyScaledForWall(base:vec3i,scale:i32,wallMode:u32){
  if(any(base<vec3i(0))||any(base>=vec3i(params.sampleDimensions+vec3u(1u)))){return;}
  let o=array<vec3i,8>(vec3i(0,0,0),vec3i(1,0,0),vec3i(1,1,0),vec3i(0,1,0),vec3i(0,0,1),vec3i(1,0,1),vec3i(1,1,1),vec3i(0,1,1));
  var v=array<f32,8>();var lo=1.0;var hi=0.0;for(var i=0;i<8;i+=1){v[i]=latticeForWall(base+o[i]*scale,wallMode);lo=min(lo,v[i]);hi=max(hi,v[i]);}
  emitClassifiedCube(base,scale,lo,hi,vec4f(v[0],v[1],v[2],v[3]),vec4f(v[4],v[5],v[6],v[7]));
}
fn cubeCornerIndex(x:u32,y:u32,z:u32)->u32{
  if(z==0u){return select(x,3u-x,y==1u);}return select(4u+x,7u-x,y==1u);
}
fn halfCellCrossing(inside:f32,outside:f32)->bool{
  let denominator=outside-inside;
  return finite(inside)&&finite(outside)&&inside<0.0&&outside>0.0&&denominator>0.0
    &&abs((-inside/denominator)-0.5)<=SHARP_CORNER_HALF_CELL_EPSILON;
}
fn emitSharpBoxPlane(base:vec3i,scale:i32,raw:ptr<function,array<f32,8>>,inside:vec3u,axis:u32,clipMask:u32){
  let o=array<vec3i,8>(vec3i(0,0,0),vec3i(1,0,0),vec3i(1,1,0),vec3i(0,1,0),vec3i(0,0,1),vec3i(1,0,1),vec3i(1,1,1),vec3i(0,1,1));
  var v=array<f32,8>();var lo=1.0;var hi=0.0;for(var i=0u;i<8u;i+=1u){let c=vec3u(o[i]);var source=inside;source[axis]=c[axis];v[i]=occupancy((*raw)[cubeCornerIndex(source.x,source.y,source.z)]);lo=min(lo,v[i]);hi=max(hi,v[i]);}
  let highBits=(inside.x<<11u)|(inside.y<<12u)|(inside.z<<13u);
  // Bits 8..10 clip the two tangential extents, 11..13 select their
  // low/high halves, and 14..15 identify the plane normal. A nonzero clip
  // mask is also the direct-quad discriminator in the scan/emission passes.
  emitClassifiedCubeTagged(base,scale,lo,hi,vec4f(v[0],v[1],v[2],v[3]),vec4f(v[4],v[5],v[6],v[7]),(clipMask<<8u)|highBits|(axis<<14u));
}
// Marching tetrahedra is not invariant under the box's reflection group at a
// sharp edge or trihedral corner: its fixed 0--6 body diagonal chamfers an
// exact L-shaped zero set. Recognize only the exact half-cell box patterns and
// give each Cartesian face its own rectangular owner. Moving/non-box
// interfaces stay on the ordinary conforming tetrahedral path.
fn classifySharpInteriorBoxFeature(base:vec3i,scale:i32)->bool{
  let dims=vec3i(params.sampleDimensions);if(base.x<=0||base.z<=0||base.x>=dims.x||base.z>=dims.z){return false;}
  let o=array<vec3i,8>(vec3i(0,0,0),vec3i(1,0,0),vec3i(1,1,0),vec3i(0,1,0),vec3i(0,0,1),vec3i(1,0,1),vec3i(1,1,1),vec3i(0,1,1));
  var raw=array<f32,8>();for(var i=0;i<8;i+=1){let p=base+o[i]*scale;raw[i]=phi(vec3i(p.x-1,max(p.y-1,0),p.z-1));}
  // One inside sample and seven outside samples is a trihedral box corner.
  for(var q=0u;q<8u;q+=1u){let inside=vec3u(q&1u,(q>>1u)&1u,(q>>2u)&1u);let qi=cubeCornerIndex(inside.x,inside.y,inside.z);if(raw[qi]>=0.0){continue;}var isolated=true;for(var z=0u;z<2u;z+=1u){for(var y=0u;y<2u;y+=1u){for(var x=0u;x<2u;x+=1u){let index=cubeCornerIndex(x,y,z);if(index!=qi&&raw[index]<=0.0){isolated=false;}}}}
    var half=true;for(var axis=0u;axis<3u;axis+=1u){var neighbor=inside;neighbor[axis]=1u-neighbor[axis];half=half&&halfCellCrossing(raw[qi],raw[cubeCornerIndex(neighbor.x,neighbor.y,neighbor.z)]);}
    if(isolated&&half){for(var axis=0u;axis<3u;axis+=1u){emitSharpBoxPlane(base,scale,&raw,inside,axis,7u^(1u<<axis));}return true;}
  }
  // Two adjacent inside samples and six outside samples form a sharp edge.
  for(var edgeAxis=0u;edgeAxis<3u;edgeAxis+=1u){for(var side=0u;side<4u;side+=1u){var inside=vec3u(0u);var bit=0u;for(var axis=0u;axis<3u;axis+=1u){if(axis!=edgeAxis){inside[axis]=(side>>bit)&1u;bit+=1u;}}var other=inside;other[edgeAxis]=1u;let i0=cubeCornerIndex(inside.x,inside.y,inside.z);let i1=cubeCornerIndex(other.x,other.y,other.z);if(raw[i0]>=0.0||raw[i1]>=0.0){continue;}var isolated=true;for(var z=0u;z<2u;z+=1u){for(var y=0u;y<2u;y+=1u){for(var x=0u;x<2u;x+=1u){let index=cubeCornerIndex(x,y,z);if(index!=i0&&index!=i1&&raw[index]<=0.0){isolated=false;}}}}var half=true;for(var axis=0u;axis<3u;axis+=1u){if(axis==edgeAxis){continue;}var n0=inside;n0[axis]=1u-n0[axis];var n1=other;n1[axis]=1u-n1[axis];half=half&&halfCellCrossing(raw[i0],raw[cubeCornerIndex(n0.x,n0.y,n0.z)])&&halfCellCrossing(raw[i1],raw[cubeCornerIndex(n1.x,n1.y,n1.z)]);}
    if(isolated&&half){for(var axis=0u;axis<3u;axis+=1u){if(axis!=edgeAxis){emitSharpBoxPlane(base,scale,&raw,inside,axis,7u^((1u<<axis)|(1u<<edgeAxis)));}}return true;}
  }}return false;
}
fn classifyScaled(base:vec3i,scale:i32){
  let dims=vec3i(params.sampleDimensions);
  let lowX=base.x==0;let highX=base.x+scale-1==dims.x;
  let lowZ=base.z==0;let highZ=base.z+scale-1==dims.z;
  let xWall=lowX||highX;let zWall=lowZ||highZ;
  if(lowX){emitScaledBoundaryWallFace(base,scale,0u,0u);}
  if(highX){emitScaledBoundaryWallFace(base,scale,0u,1u);}
  if(lowZ){emitScaledBoundaryWallFace(base,scale,2u,0u);}
  if(highZ){emitScaledBoundaryWallFace(base,scale,2u,1u);}
  // Mirror both horizontal wall axes after their exact plane owners have
  // been emitted. This preserves a free-surface cap in a wall cube without
  // also emitting the old scale-dependent ghost-air wall contour.
  if(xWall||zWall){classifyScaledForWall(base,scale,3u);return;}
  if(!xWall&&!zWall&&classifySharpInteriorBoxFeature(base,scale)){return;}
  classifyScaledForWall(base,scale,0u);
}
@compute @workgroup_size(256)
fn extractGlobalFineMain(@builtin(global_invocation_id)gid:vec3u){
  if(!validCurrentPublication()){return;}
  let stream=gid.x+gid.y*65535u*256u;let samples=params.samplesPerBrick;let work=stream/max(1u,samples);
  if(work>=fineWorklist[1]){return;}let id=fineWorklist[7u+work];let metadataBase=id*4u;
  if(id>=params.table.z||metadataBase+2u>=arrayLength(&metadata)||metadata[metadataBase]!=id||metadata[metadataBase+2u]!=params.table.w){return;}
  let key=metadata[id*4u+1u];let xy=max(1u,params.brickDimensions.x*params.brickDimensions.y);let bz=key/xy;let rem=key-bz*xy;let by=rem/params.brickDimensions.x;let bx=rem-by*params.brickDimensions.x;
  let localIndex=stream-work*samples;let r=max(1u,params.brickResolution);let local=vec3u(localIndex%r,(localIndex/r)%r,localIndex/max(1u,r*r));
  let source=metadata[metadataBase+3u];var span=1u;
  if((fineWorklist[3]&0x80000000u)!=0u){span=1u<<compactSourceSpanLog(source);}
  let sampleScale=select(1u,compactPagesPerSolverAxis()*span,span>1u);
  let q=vec3u(bx,by,bz)*r+local*sampleScale;
  if(any(q>=params.sampleDimensions)){return;}let index=id*samples+localIndex;if(index>=arrayLength(&fineSamples)||(finePackedFlags(index)&1u)==0u||!finite(finePackedPhi(index))){return;}let xb=array<i32,2>(i32(q.x+1u),0);let yb=array<i32,2>(i32(q.y+1u),0);let zb=array<i32,2>(i32(q.z+1u),0);
  let xn=select(1u,2u,q.x==0u);let yn=select(1u,2u,q.y==0u);let zn=select(1u,2u,q.z==0u);
  for(var zi=0u;zi<zn;zi+=1u){for(var yi=0u;yi<yn;yi+=1u){for(var xi=0u;xi<xn;xi+=1u){classifyScaled(vec3i(xb[xi],yb[yi],zb[zi]),i32(sampleScale));}}}
}
@compute @workgroup_size(256)
fn extractGlobalCoarseMain(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)local:u32){if(!validCurrentPublication()){return;}
  // Coarse-1 publishes an authoritative dense complement as well as the
  // compact wet-row directory. Classifying only row-owned lower anchors
  // misses every face whose lower sample lies in the dry complement (the two
  // low-side vertical faces of a box). Scan the small factor-1 lattice once so
  // every cube has exactly one Cartesian lower-anchor owner.
  if(params.table.y==6u){atomicStore(&drawArgs.meshPublicationGeneration,powerCoarseSamples.generation&0x3fffffffu);let stream=(group.x+group.y*65535u)*256u+local;if(adaptiveNodalPublication()){let cubeDims=params.sampleDimensions;let total=cubeDims.x*cubeDims.y*cubeDims.z;if(stream>=total){return;}let z=stream/(cubeDims.x*cubeDims.y);let remainder=stream-z*cubeDims.x*cubeDims.y;let y=remainder/cubeDims.x;let x=remainder-y*cubeDims.x;classifyAdaptiveNodal(vec3i(i32(x),i32(y),i32(z)));return;}let cubeDims=params.sampleDimensions+vec3u(1u);let total=cubeDims.x*cubeDims.y*cubeDims.z;if(stream>=total){return;}let z=stream/(cubeDims.x*cubeDims.y);let remainder=stream-z*cubeDims.x*cubeDims.y;let y=remainder/cubeDims.x;let x=remainder-y*cubeDims.x;classifyScaled(vec3i(i32(x),i32(y),i32(z)),1);return;}
  let slot=group.x+group.y*65535u;if(slot>=min(powerCoarseSamples.rowCount,arrayLength(&powerCoarseSamples.entries))){return;}let entry=powerCoarseSamples.entries[slot];if(entry.cellPlusOne==0u||(entry.flags&1u)==0u||entry.size==0u){return;}let d=powerCoarseSamples.dimensions;let cell=entry.cellPlusOne-1u;let origin=vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));let factor=max(1u,u32(round(params.cellAndDt.x)));let scale=entry.size*factor;let base=vec3i(origin*factor+vec3u(1u));
  // Coarse fallback is sampled on the same unit fine lattice as the narrow
  // band. Emitting one scaled tetrahedral cube beside unit fine cubes creates
  // non-conforming diagonals and visible T-junction slits. Each coarse leaf
  // instead owns the unit cube bases corresponding to its samples. A valid
  // lower-anchor sample makes the fine pass the unique cube owner; its other
  // corners may still use compact coarse phi through lattice().
  //
  // A leaf touching a low tank boundary additionally owns base zero on that
  // axis. Taking the Cartesian product closes wall/floor edges and corners.
  let lowX=select(0u,1u,origin.x==0u);let lowY=select(0u,1u,origin.y==0u);let lowZ=select(0u,1u,origin.z==0u);
  let sx=scale+lowX;let sy=scale+lowY;let sz=scale+lowZ;let total=sx*sy*sz;
  // A whole workgroup cooperates on one leaf. This keeps large production
  // leaves bounded per invocation instead of putting scale^3 work on one
  // shader lane (a browser watchdog risk at maximumLeafSize 16/32).
  for(var index=local;index<total;index+=256u){let xi=index%sx;let yi=(index/sx)%sy;let zi=index/(sx*sy);
    var bx=0;if(xi>=lowX){bx=base.x+i32(xi-lowX);}var by=0;if(yi>=lowY){by=base.y+i32(yi-lowY);}var bz=0;if(zi>=lowZ){bz=base.z+i32(zi-lowZ);}let candidate=vec3i(bx,by,bz);
    if(!fineOwnsCube(candidate)){classifyScaled(candidate,1);}
  }
}
`;
