import { makeOctreePowerCoarseLevelSetSampleWGSL } from "./webgpu-octree-power-coarse-levelset";

/**
 * Binding-minimal classifier for the Section 5 global narrow-band level set.
 * Fine phi is authoritative for every valid tagged sample. Missing or stale
 * fine samples query the compact coarse-octree directory; no dense phi texture
 * is reachable from this module.
 */
export const globalFineSurfaceClassificationShader = /* wgsl */ `
struct Uniforms{viewport:vec4f,cameraPosition:vec4f,cameraTarget:vec4f,container:vec4f,options:vec4f,gridInfo:vec4f,debug:vec4f}
struct IndirectArgs{vertexCount:atomic<u32>,instanceCount:u32,firstVertex:u32,firstInstance:u32,activeCubeCount:atomic<u32>,vertexAllocator:atomic<u32>,globalFineAuthorityLatch:atomic<u32>}
struct FineParams{sampleDimensions:vec3u,brickResolution:u32,brickDimensions:vec3u,samplesPerBrick:u32,table:vec4u,settings:vec4f,cellAndDt:vec4f,sizing:vec4f,physical:vec4f}
@group(0)@binding(0)var<uniform>u:Uniforms;
@group(0)@binding(4)var<storage,read_write>drawArgs:IndirectArgs;
@group(0)@binding(5)var<storage,read_write>activeCubes:array<vec2u>;
@group(0)@binding(6)var<storage,read_write>cubeValues:array<vec4f>;
@group(0)@binding(7)var<storage,read>pageTable:array<u32>;
@group(0)@binding(8)var<storage,read>fineWorklist:array<u32>;
@group(0)@binding(9)var<storage,read>finePhi:array<f32>;
@group(0)@binding(10)var<uniform>params:FineParams;
@group(0)@binding(11)var<storage,read>fineFlags:array<u32>;
@group(0)@binding(12)var<storage,read>metadata:array<u32>;
${makeOctreePowerCoarseLevelSetSampleWGSL(16)}
@group(0)@binding(17)var<storage,read>fineTopologyControl:array<u32>;
const INVALID:u32=0xffffffffu;
fn validCurrentPublication()->bool{
  if(arrayLength(&fineTopologyControl)<8u||arrayLength(&fineWorklist)<5u){return false;}
  let clean=fineTopologyControl[0]==0u&&fineTopologyControl[4]==1u&&fineTopologyControl[5]==0u&&fineTopologyControl[7]==0u;
  let count=fineWorklist[0];let generation=params.table.w&0x3fffffffu;
  let finePublished=count<=params.table.z&&(fineWorklist[1]&0x3fffffffu)==generation
    &&fineWorklist[2]==(count+63u)/64u&&fineWorklist[3]==1u&&fineWorklist[4]==1u;
  let capacity=min(powerCoarseSamples.hashCapacity,arrayLength(&powerCoarseSamples.entries));
  let expectedWidth=params.settings.w*max(1.0,params.cellAndDt.x);
  let coarsePublished=powerCoarseSamples.state==0x80000000u
    &&(powerCoarseSamples.generation&0x3fffffffu)==generation
    &&capacity>0u&&powerCoarseSamples.hashCapacity==capacity&&(capacity&(capacity-1u))==0u
    &&powerCoarseSamples.maximumLeafSize>0u&&(powerCoarseSamples.maximumLeafSize&(powerCoarseSamples.maximumLeafSize-1u))==0u
    &&all(powerCoarseSamples.dimensions*max(1u,u32(round(params.cellAndDt.x)))==params.sampleDimensions)
    &&powerCoarseSamples.physicalCellSize>0.0
    &&abs(powerCoarseSamples.physicalCellSize-expectedWidth)<=1e-5*max(powerCoarseSamples.physicalCellSize,expectedWidth);
  return clean&&finePublished&&coarsePublished;
}
fn pageHash(key:u32)->u32{return ((key^(key>>16u))*0x9e3779b1u)&(params.table.x-1u);}
fn pageLookup(key:u32)->u32{
  if(params.table.x==0u||(params.table.x&(params.table.x-1u))!=0u){return INVALID;}
  let start=pageHash(key);for(var probe=0u;probe<min(params.table.y,params.table.x);probe+=1u){
    let slot=(start+probe)&(params.table.x-1u);if(slot*2u+1u>=arrayLength(&pageTable)){return INVALID;}
    let stored=pageTable[slot*2u];if(stored==key){let id=pageTable[slot*2u+1u];
      if(id<params.table.z&&id*10u+2u<arrayLength(&metadata)&&metadata[id*10u+1u]==key&&metadata[id*10u+2u]==params.table.w){return id;}return INVALID;}
    if(stored==INVALID){return INVALID;}
  }return INVALID;
}
fn coarsePhi(q:vec3i)->f32{return sampleCoarseOctreePhi(params.settings.xyz+(vec3f(q)+vec3f(0.5))*params.settings.w);}
fn finite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn phi(qi:vec3i)->f32{
  if(any(qi<vec3i(0))||any(qi>=vec3i(params.sampleDimensions))){return coarsePhi(qi);}
  let q=vec3u(qi);let r=max(1u,params.brickResolution);let brick=q/r;let local=q-brick*r;
  if(any(brick>=params.brickDimensions)){return coarsePhi(qi);}
  let key=brick.x+params.brickDimensions.x*(brick.y+params.brickDimensions.y*brick.z);let id=pageLookup(key);
  if(id==INVALID){return coarsePhi(qi);}
  let index=id*params.samplesPerBrick+local.x+r*(local.y+r*local.z);
  if(index>=arrayLength(&finePhi)||index>=arrayLength(&fineFlags)||(fineFlags[index]&1u)==0u||!finite(finePhi[index])){return coarsePhi(qi);}
  return finePhi[index];
}
fn fineValid(q:vec3u)->bool{if(any(q>=params.sampleDimensions)){return false;}let r=max(1u,params.brickResolution);let brick=q/r;if(any(brick>=params.brickDimensions)){return false;}let local=q-brick*r;let key=brick.x+params.brickDimensions.x*(brick.y+params.brickDimensions.y*brick.z);let id=pageLookup(key);if(id==INVALID){return false;}let index=id*params.samplesPerBrick+local.x+r*(local.y+r*local.z);return index<arrayLength(&fineFlags)&&index<arrayLength(&finePhi)&&(fineFlags[index]&1u)!=0u&&finite(finePhi[index]);}
fn fineCubeFullyValid(base:vec3i,scale:i32)->bool{let o=array<vec3i,8>(vec3i(0,0,0),vec3i(1,0,0),vec3i(1,1,0),vec3i(0,1,0),vec3i(0,0,1),vec3i(1,0,1),vec3i(1,1,1),vec3i(0,1,1));for(var i=0;i<8;i+=1){let q=base+o[i]*scale-vec3i(1);if(any(q<vec3i(0))||any(q>=vec3i(params.sampleDimensions))||!fineValid(vec3u(q))){return false;}}return true;}
fn occupancy(value:f32)->f32{let band=4.0*u.container.y/max(f32(params.sampleDimensions.y),1.0);return clamp(0.5-value/band,0.0,1.0);}
fn lattice(p:vec3i)->f32{
  let dims=vec3i(params.sampleDimensions);
  if(p.x<=0||p.z<=0||p.x>=dims.x+1||p.z>=dims.z+1||p.y>=dims.y+1){return 0.0;}
  return occupancy(phi(vec3i(p.x-1,max(p.y-1,0),p.z-1)));
}
fn classifyScaled(base:vec3i,scale:i32){
  if(any(base<vec3i(0))||any(base>=vec3i(params.sampleDimensions+vec3u(1u)))){return;}
  let o=array<vec3i,8>(vec3i(0,0,0),vec3i(1,0,0),vec3i(1,1,0),vec3i(0,1,0),vec3i(0,0,1),vec3i(1,0,1),vec3i(1,1,1),vec3i(0,1,1));
  var v=array<f32,8>();var lo=1.0;var hi=0.0;for(var i=0;i<8;i+=1){v[i]=lattice(base+o[i]*scale);lo=min(lo,v[i]);hi=max(hi,v[i]);}
  if(lo>=0.5||hi<0.5){return;}atomicStore(&drawArgs.globalFineAuthorityLatch,1u);atomicMin(&drawArgs.vertexAllocator,0u);let slot=atomicAdd(&drawArgs.activeCubeCount,1u);
  if(slot>=arrayLength(&activeCubes)||slot*2u+1u>=arrayLength(&cubeValues)){return;}
  activeCubes[slot]=vec2u(u32(base.x)|(u32(base.z)<<16u),u32(base.y)|(u32(scale)<<16u));
  cubeValues[slot*2u]=vec4f(v[0],v[1],v[2],v[3]);cubeValues[slot*2u+1u]=vec4f(v[4],v[5],v[6],v[7]);
}
@compute @workgroup_size(256)
fn extractGlobalFineMain(@builtin(global_invocation_id)gid:vec3u){
  if(!validCurrentPublication()){return;}
  let stream=gid.x+gid.y*65535u*256u;let samples=params.samplesPerBrick;let id=stream/max(1u,samples);
  if(id>=params.table.z||id*10u+2u>=arrayLength(&metadata)||metadata[id*10u+2u]!=params.table.w){return;}
  let key=metadata[id*10u+1u];let xy=max(1u,params.brickDimensions.x*params.brickDimensions.y);let bz=key/xy;let rem=key-bz*xy;let by=rem/params.brickDimensions.x;let bx=rem-by*params.brickDimensions.x;
  let localIndex=stream-id*samples;let r=max(1u,params.brickResolution);let local=vec3u(localIndex%r,(localIndex/r)%r,localIndex/max(1u,r*r));let q=vec3u(bx,by,bz)*r+local;
  if(any(q>=params.sampleDimensions)){return;}let index=id*samples+localIndex;if(index>=arrayLength(&fineFlags)||index>=arrayLength(&finePhi)||(fineFlags[index]&1u)==0u||!finite(finePhi[index])){return;}let xb=array<i32,2>(i32(q.x+1u),0);let yb=array<i32,2>(i32(q.y+1u),0);let zb=array<i32,2>(i32(q.z+1u),0);
  let xn=select(1u,2u,q.x==0u);let yn=select(1u,2u,q.y==0u);let zn=select(1u,2u,q.z==0u);
  for(var zi=0u;zi<zn;zi+=1u){for(var yi=0u;yi<yn;yi+=1u){for(var xi=0u;xi<xn;xi+=1u){classifyScaled(vec3i(xb[xi],yb[yi],zb[zi]),1);}}}
}
@compute @workgroup_size(256)
fn extractGlobalCoarseMain(@builtin(global_invocation_id)gid:vec3u){let slot=gid.x;if(!validCurrentPublication()){return;}if(slot>=min(powerCoarseSamples.hashCapacity,arrayLength(&powerCoarseSamples.entries))){return;}let entry=powerCoarseSamples.entries[slot];if(entry.cellPlusOne==0u||(entry.flags&1u)==0u||entry.size==0u){return;}let d=powerCoarseSamples.dimensions;let cell=entry.cellPlusOne-1u;let origin=vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));let factor=max(1u,u32(round(params.cellAndDt.x)));let scale=entry.size*factor;let base=vec3i(origin*factor+vec3u(1u));if(fineCubeFullyValid(base,i32(scale))){return;}classifyScaled(base,i32(scale));}
`;
