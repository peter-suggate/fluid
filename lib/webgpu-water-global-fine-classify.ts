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
fn fineOwnsCube(base:vec3i)->bool{let q=max(base-vec3i(1),vec3i(0));return all(q<vec3i(params.sampleDimensions))&&fineValid(vec3u(q));}
fn occupancy(value:f32)->f32{let band=4.0*u.container.y/max(f32(params.sampleDimensions.y),1.0);return clamp(0.5-value/band,0.0,1.0);}
// The ordinary halo represents one closed tank wall with an exterior air
// sample. At an x/z tank edge a single Cartesian-product halo cube cannot
// represent both perpendicular wall caps: marching tetrahedra turns the two
// planes into a diagonal chamfer. Boundary modes 1 and 2 give that edge cube
// two explicit owners. The x-wall owner mirrors z through the edge; the
// z-wall owner mirrors x. Their zero sets meet on the exact tank corner.
fn latticeForWall(p:vec3i,wallMode:u32)->f32{
  let dims=vec3i(params.sampleDimensions);
  if(p.y>=dims.y+1){return 0.0;}
  var x=p.x-1;var z=p.z-1;
  if(p.x<=0||p.x>=dims.x+1){
    if(wallMode!=2u){return 0.0;}x=clamp(x,0,dims.x-1);
  }
  if(p.z<=0||p.z>=dims.z+1){
    if(wallMode!=1u){return 0.0;}z=clamp(z,0,dims.z-1);
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
fn classifyScaledForWall(base:vec3i,scale:i32,wallMode:u32){
  if(any(base<vec3i(0))||any(base>=vec3i(params.sampleDimensions+vec3u(1u)))){return;}
  let o=array<vec3i,8>(vec3i(0,0,0),vec3i(1,0,0),vec3i(1,1,0),vec3i(0,1,0),vec3i(0,0,1),vec3i(1,0,1),vec3i(1,1,1),vec3i(0,1,1));
  var v=array<f32,8>();var lo=1.0;var hi=0.0;for(var i=0;i<8;i+=1){v[i]=latticeForWall(base+o[i]*scale,wallMode);lo=min(lo,v[i]);hi=max(hi,v[i]);}
  emitClassifiedCube(base,scale,lo,hi,vec4f(v[0],v[1],v[2],v[3]),vec4f(v[4],v[5],v[6],v[7]));
}
fn cubeCornerIndex(x:u32,y:u32,z:u32)->u32{
  if(z==0u){return select(x,3u-x,y==1u);}return select(4u+x,7u-x,y==1u);
}
// A signed distance to an axis-aligned liquid box has one inside x/z corner,
// two equally distant side samples, and an extruded profile in y. A single
// tetrahedral scalar cube rounds that exact L-shaped zero set into the
// shrunken corner visible in the renderer. Recognize only this sharp,
// vertically extruded configuration and give each plane its own cube values,
// just as the tank-edge caps above have separate owners.
fn classifySharpInteriorXZCorner(base:vec3i,scale:i32)->bool{
  let dims=vec3i(params.sampleDimensions);if(base.x<=0||base.z<=0||base.x>=dims.x||base.z>=dims.z){return false;}
  let o=array<vec3i,8>(vec3i(0,0,0),vec3i(1,0,0),vec3i(1,1,0),vec3i(0,1,0),vec3i(0,0,1),vec3i(1,0,1),vec3i(1,1,1),vec3i(0,1,1));
  var raw=array<f32,8>();for(var i=0;i<8;i+=1){let p=base+o[i]*scale;raw[i]=phi(vec3i(p.x-1,max(p.y-1,0),p.z-1));}
  let bottom=array<u32,4>(0u,1u,4u,5u);let top=array<u32,4>(3u,2u,7u,6u);let tolerance=0.15*params.settings.w;
  var inside=INVALID;
  for(var q=0u;q<4u;q+=1u){let sx=q^1u;let sz=q^2u;let sd=q^3u;
    let extruded=abs(raw[bottom[q]]-raw[top[q]])<=tolerance&&abs(raw[bottom[sx]]-raw[top[sx]])<=tolerance&&abs(raw[bottom[sz]]-raw[top[sz]])<=tolerance&&abs(raw[bottom[sd]]-raw[top[sd]])<=tolerance;
    let sharp=raw[bottom[q]]<0.0&&raw[top[q]]<0.0&&raw[bottom[sx]]>0.0&&raw[top[sx]]>0.0&&raw[bottom[sz]]>0.0&&raw[top[sz]]>0.0&&raw[bottom[sd]]>0.0&&raw[top[sd]]>0.0;
    let symmetric=abs(raw[bottom[q]]+raw[bottom[sx]])<=tolerance&&abs(raw[bottom[q]]+raw[bottom[sz]])<=tolerance&&abs(raw[bottom[sx]]-raw[bottom[sz]])<=tolerance;
    if(extruded&&sharp&&symmetric){inside=q;break;}
  }
  if(inside==INVALID){return false;}
  let insideX=inside&1u;let insideZ=(inside>>1u)&1u;var vx=array<f32,8>();var vz=array<f32,8>();var xlo=1.0;var xhi=0.0;var zlo=1.0;var zhi=0.0;
  for(var i=0u;i<8u;i+=1u){let c=vec3u(o[i]);vx[i]=occupancy(raw[cubeCornerIndex(c.x,c.y,insideZ)]);vz[i]=occupancy(raw[cubeCornerIndex(insideX,c.y,c.z)]);xlo=min(xlo,vx[i]);xhi=max(xhi,vx[i]);zlo=min(zlo,vz[i]);zhi=max(zhi,vz[i]);}
  // Descriptor bits 8..9 select the cap plane; bits 10..11 select which
  // half-cube is liquid. Polygonisation clips only positions, preserving the
  // exact axis normal encoded by vx/vz.
  let sideTag=(insideX<<10u)|(insideZ<<11u);
  emitClassifiedCubeTagged(base,scale,xlo,xhi,vec4f(vx[0],vx[1],vx[2],vx[3]),vec4f(vx[4],vx[5],vx[6],vx[7]),(1u<<8u)|sideTag);
  emitClassifiedCubeTagged(base,scale,zlo,zhi,vec4f(vz[0],vz[1],vz[2],vz[3]),vec4f(vz[4],vz[5],vz[6],vz[7]),(2u<<8u)|sideTag);
  return true;
}
fn classifyScaled(base:vec3i,scale:i32){
  let dims=vec3i(params.sampleDimensions);let xWall=base.x==0||base.x==dims.x;let zWall=base.z==0||base.z==dims.z;
  if(xWall&&zWall){classifyScaledForWall(base,scale,1u);classifyScaledForWall(base,scale,2u);return;}
  if(!xWall&&!zWall&&classifySharpInteriorXZCorner(base,scale)){return;}
  classifyScaledForWall(base,scale,0u);
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
fn extractGlobalCoarseMain(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)local:u32){let slot=group.x+group.y*65535u;if(!validCurrentPublication()){return;}if(slot>=min(powerCoarseSamples.hashCapacity,arrayLength(&powerCoarseSamples.entries))){return;}let entry=powerCoarseSamples.entries[slot];if(entry.cellPlusOne==0u||(entry.flags&1u)==0u||entry.size==0u){return;}let d=powerCoarseSamples.dimensions;let cell=entry.cellPlusOne-1u;let origin=vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));let factor=max(1u,u32(round(params.cellAndDt.x)));let scale=entry.size*factor;let base=vec3i(origin*factor+vec3u(1u));
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
