/** CM11a topology and coefficient setup for the coarse mixed pressure continuation. */
export { UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE } from "./pressure-policy";
export const uniformPressureInPlaceSmootherWGSL = /* wgsl */ `
var<workgroup> mgTileLive:atomic<u32>;

var<workgroup> mgTileConstrained:atomic<u32>;

@compute @workgroup_size(4,4,4)
fn mgBuildSmoothTiles(@builtin(global_invocation_id) gid:vec3u,
 @builtin(workgroup_id) tile:vec3u,@builtin(local_invocation_index) lane:u32){
  if(mgBakedLiquid(vec3i(gid))){atomicStore(&mgTileLive,1u);}
  if(MG_CYCLE_TILES&&mgValid(vec3i(gid),mg.levelDims.xyz)&&textureLoad(mgMinimumIn,vec3i(gid),0).x> -3.0e38){atomicOr(&mgTileConstrained,select(1u,2u,mgInterior(vec3i(gid),mg.levelDims.xyz)));}
  let live=workgroupUniformLoad(&mgTileLive);
  let d=(mg.levelDims.xyz+vec3u(3))/4u;
  let index=tile.x+d.x*(tile.y+d.y*tile.z);
  if(lane==0u&&live!=0u){
    let slot=atomicAdd(&mgCycleDispatch[0],1u);
    atomicStore(&mgCycleDispatch[4u+slot],index);
  }


  if(MG_CYCLE_TILES){
    let constrained=workgroupUniformLoad(&mgTileConstrained);
    if(lane==0u){atomicStore(&mgCycleDispatch[4u+d.x*d.y*d.z+index],select(0u,1u,live!=0u)|select(0u,2u,constrained!=0u)|select(0u,4u,(constrained&2u)!=0u));}
  }
}

@group(1) @binding(19) var<storage,read_write> mgCycleTiles:array<atomic<u32>>;

override MG_HALO_ENTRIES:bool=false;

@compute @workgroup_size(64)
fn mgBuildCycleTiles(@builtin(global_invocation_id) gid:vec3u){
  let d=(mg.levelDims.xyz+vec3u(3))/4u;let n=d.x*d.y*d.z;
  let at=gid.x;if(at>=n){return;}
  let t=vec3i(i32(at%d.x),i32((at/d.x)%d.y),i32(at/(d.x*d.y)));
  let own=atomicLoad(&mgCycleDispatch[4u+n+at]);
  var near=false;
  for(var k=0u;k<27u;k+=1u){
    if(near){break;}
    let q=t+vec3i(i32(k%3u)-1,i32((k/3u)%3u)-1,i32(k/9u)-1);
    if(any(q<vec3i(0))||any(q>=vec3i(d))){continue;}
    let j=u32(q.x)+d.x*(u32(q.y)+d.y*u32(q.z));
    if((atomicLoad(&mgCycleDispatch[4u+n+j])&1u)!=0u){near=true;}
  }
  if(!near&&(own&2u)==0u){return;}
  if(MG_HALO_ENTRIES&&!near&&(own&4u)==0u){
    for(var a=0u;a<3u;a+=1u){
      if(t[a]==0){let slot=atomicAdd(&mgCycleTiles[4u+n],1u);atomicStore(&mgCycleTiles[5u+n+slot],8u*at+2u*a);}
      if(t[a]==i32(d[a])-1){let slot=atomicAdd(&mgCycleTiles[4u+n],1u);atomicStore(&mgCycleTiles[5u+n+slot],8u*at+2u*a+1u);}
    }
    return;
  }
  let slot=atomicAdd(&mgCycleTiles[0],1u);
  atomicStore(&mgCycleTiles[4u+slot],at);
}

override MG_ROW_SEGMENT:u32=8u;

override MG_VISIT_LANES:u32=256u;
`;
export const uniformPressureMultigridWGSL = /* wgsl */ `
struct UniformMGParams {
  fineDims: vec4u,
  levelDims: vec4u,
  coarseDims: vec4u,
  spacing: vec4f,
  control: vec4u,
}

@group(1) @binding(0) var<uniform> mg: UniformMGParams;

@group(1) @binding(5) var mgPhiIn: texture_3d<f32>;

@group(1) @binding(6) var mgPhiOut: texture_storage_3d<r32float,write>;

@group(1) @binding(7) var mgVolumeIn: texture_3d<f32>;

@group(1) @binding(8) var mgVolumeOut: texture_storage_3d<rgba32float,write>;

@group(1) @binding(11) var mgMinimumIn: texture_3d<f32>;

struct UniformCoarseRow {
  p: f32, low: f32, rhs: f32, phi: f32,
  minimum: f32, residual: f32, state: u32, padding: u32,
  topology: vec4f,
}

struct UniformPressureState {
  convergence: array<atomic<u32>,28>,
  rows: array<UniformCoarseRow>,
}

@group(1) @binding(13) var<storage,read_write> mgState: UniformPressureState;

@group(1) @binding(14) var mgCoefficientsIn: texture_3d<f32>;

@group(1) @binding(15) var mgCoefficientsOut: texture_storage_3d<rgba32float,write>;

fn mgSkipCycle()->bool{
  if(mg.levelDims.w==0u){return false;}
  if(atomicLoad(&mgState.convergence[16])!=0u){return true;}
  let recovery=atomicLoad(&mgState.convergence[22])!=0u;
  return select(recovery,!recovery,mg.levelDims.w==2u);
}

@group(1) @binding(18) var<storage,read_write> mgCycleDispatch:array<atomic<u32>>;

fn mgValid(p:vec3i,d:vec3u)->bool{return all(p>=vec3i(0))&&all(p<vec3i(d));}

fn mgClamp(p:vec3i,d:vec3u)->vec3i{return clamp(p,vec3i(0),vec3i(d)-vec3i(1));}

fn mgD4Sum6(value:array<f32,6>)->f32{return ((value[0]+value[1])+(value[4]+value[5]))+(value[2]+value[3]);}

fn mgD4Sum8(value:array<f32,8>)->f32{
  let y0=(value[0]+value[5])+(value[1]+value[4]);
  let y1=(value[2]+value[7])+(value[3]+value[6]);
  return y0+y1;
}

fn mgD4Sum8Vec4(value:array<vec4f,8>)->vec4f{
  let y0=(value[0]+value[5])+(value[1]+value[4]);
  let y1=(value[2]+value[7])+(value[3]+value[6]);
  return y0+y1;
}

fn mgActiveId(gid:vec3u)->vec3i{


  if(pressureWindowLattice()){return vec3i(gid);}




  let base=16u+10u*mg.fineDims.w;
  let packed=activeRegion[base+9u];
  if((packed&0x40000000u)!=0u&&any(gid>=vec3u(
    packed&1023u,(packed>>10u)&1023u,(packed>>20u)&1023u))){return vec3i(-1);}
  return vec3i(gid)+vec3i(vec3u(activeRegion[base],activeRegion[base+1u],activeRegion[base+2u]));
}

fn mgFineChild(coarse:vec3i,o:vec3i)->vec3i{
  let finePhysical=vec3i(mg.levelDims.xyz)-vec3i(2);
  let coarsePhysical=max(vec3i(mg.coarseDims.xyz)-vec3i(2),vec3i(1));
  let stride=max(finePhysical/coarsePhysical,vec3i(1));
  return clamp(stride*(coarse-vec3i(1))+o*(stride-vec3i(1)),vec3i(-1),finePhysical)+vec3i(1);
}

fn mgTopology(p:vec3i)->vec4f{return textureLoad(mgVolumeIn,mgClamp(p,mg.levelDims.xyz),0);}

fn mgPhi(p:vec3i)->f32{return textureLoad(mgPhiIn,mgClamp(p,mg.levelDims.xyz),0).x;}

fn mgLiquidPhi(phi:f32)->bool{
  if(MG_MIXED_SURFACE_THETA){return (phi< -0.001*min(mg.spacing.x,min(mg.spacing.y,mg.spacing.z)));}
  return phi<0.0;
}

fn mgLiquid(p:vec3i)->bool{return mgValid(p,mg.levelDims.xyz)&&mgLiquidPhi(mgPhi(p));}

fn mgInterior(p:vec3i,d:vec3u)->bool{return all(p>=vec3i(1))&&all(p<vec3i(d)-vec3i(1));}

fn mgFaceV(id:vec3i,neighbor:vec3i,axis:u32)->f32{
  let positive=neighbor[axis]>id[axis];
  return select(mgTopology(neighbor)[axis+1u],mgTopology(id)[axis+1u],positive);
}

const MG_MIXED_SURFACE_THETA:bool=false;

fn mgSurfaceTheta(liquidPhi:f32,airPhi:f32)->f32{
  if(MG_MIXED_SURFACE_THETA){return clamp(-liquidPhi/max(airPhi-liquidPhi,1e-30),0.001,1000.0);}
  return cm12GhostFluidTheta(liquidPhi,airPhi,1e-9);
}

fn mgTheta(liquidCell:vec3i,airCell:vec3i)->f32{
  let a=mgPhi(liquidCell);let b=mgPhi(airCell);
  return mgSurfaceTheta(a,b);
}

fn mgCoefficientRaw(id:vec3i,q:vec3i,axis:u32)->f32{
  if(!mgValid(q,mg.levelDims.xyz)){


    if(axis==1u&&id.y==i32(mg.levelDims.y)-1&&q.y==i32(mg.levelDims.y)&&params.boundary.w>0.5){
      let h=mg.spacing.y;let phi=mgPhi(id);let exteriorPhi=0.5*h;
      let theta=mgSurfaceTheta(phi,exteriorPhi);
      return mgTopology(id).z/(h*h*theta);
    }
    return 0.0;
  }
  let vf=mgFaceV(id,q,axis);if(vf<=1e-6){return 0.0;}
  let h=mg.spacing[axis];





  let idLiquid=mgLiquid(id);let qLiquid=mgLiquid(q);var theta=1.0;
  if(idLiquid&&!qLiquid){theta=mgTheta(id,q);}
  if(!idLiquid&&qLiquid){theta=mgTheta(q,id);}
  return vf/(h*h*theta);
}

fn mgBakedLiquid(p:vec3i)->bool{
  return mgValid(p,mg.levelDims.xyz)&&(u32(textureLoad(mgCoefficientsIn,p,0).w)&1u)!=0u;
}

override MG_CYCLE_TILES:bool=true;

override MG_REUSE_FINEST_AUTHORITY:bool=true;

fn mgAverageSubmerged(c:vec3i)->bool{
  if(!mgValid(c,mg.coarseDims.xyz)){return true;}
  var terms:array<f32,8>;var count=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let q=mgFineChild(c,vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u)));
    let open=mgTopology(q).x>1e-5;terms[corner]=select(0.0,mgPhi(q),open);count+=select(0.0,1.0,open);
  }
  return count==0.0||mgLiquidPhi(mgD4Sum8(terms)/count);
}

fn mgAirVanishes(c:vec3i)->bool{
  for(var k=0u;k<27u;k+=1u){
    if(k!=13u&&!mgAverageSubmerged(c+vec3i(vec3u(k%3u,(k/3u)%3u,k/9u))-vec3i(1))){return false;}
  }
  return true;
}

@compute @workgroup_size(4,4,4)
fn mgDownsampleTopology(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}
  var topologyTerms:array<vec4f,8>;var phiTerms:array<f32,8>;var openTerms:array<f32,8>;var openFlags:array<f32,8>;
  var positiveTerms:array<f32,8>;var positiveFlags:array<f32,8>;var negativeFlags:array<f32,8>;
  for(var corner=0u;corner<8u;corner+=1u){
    let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let q=mgFineChild(id,o);
    var phi=0.0;
    topologyTerms[corner]=mgTopology(q);phi=mgPhi(q);
    phiTerms[corner]=phi;




    let open=topologyTerms[corner].x>1e-5;
    openTerms[corner]=select(0.0,phi,open);openFlags[corner]=select(0.0,1.0,open);
    positiveTerms[corner]=select(0.0,phi,phi>=0.0&&open);positiveFlags[corner]=select(0.0,1.0,phi>=0.0&&open);negativeFlags[corner]=select(0.0,1.0,phi<0.0&&open);
  }
  let v=mgD4Sum8Vec4(topologyTerms);let openSum=mgD4Sum8(openTerms);let openCount=mgD4Sum8(openFlags);

  let phiSum=select(mgD4Sum8(phiTerms),openSum*8.0/max(openCount,1.0),openCount>0.0);
  let positiveSum=mgD4Sum8(positiveTerms);let positiveCount=mgD4Sum8(positiveFlags);let negativeCount=mgD4Sum8(negativeFlags);


  let mixed=positiveCount>0.0&&negativeCount>0.0;
  var usePositive=mixed&&mg.control.x>=mg.control.y;











  if(mixed&&!usePositive&&mg.control.z!=0u&&mg.control.x>=mg.control.z&&mgLiquidPhi(phiSum/8.0)&&mgAirVanishes(id)){usePositive=true;}
  let coarsePhi=select(phiSum/8.0,positiveSum/max(positiveCount,1.0),usePositive);







  var faceTerms:array<vec4f,8>;
  for(var corner=0u;corner<8u;corner+=1u){
    let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    faceTerms[corner]=vec4f(0.0,select(0.0,2.0*topologyTerms[corner].y,o.x==1),select(0.0,2.0*topologyTerms[corner].z,o.y==1),select(0.0,2.0*topologyTerms[corner].w,o.z==1));
  }
  let fv=mgD4Sum8Vec4(faceTerms);var topology=vec4f(v.x,fv.y,fv.z,fv.w)/8.0;
  textureStore(mgVolumeOut,id,topology);textureStore(mgPhiOut,id,vec4f(coarsePhi));
}

@compute @workgroup_size(4,4,4)
fn mgExtrapolatePhiOneCell(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  
  if(mgTopology(id).x>1e-5){textureStore(mgPhiOut,id,vec4f(mgPhi(id)));return;}
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var terms:array<f32,6>;var weights:array<f32,6>;
  for(var n=0;n<6;n+=1){let q=id+e[n];terms[n]=0.0;weights[n]=0.0;if(!mgValid(q,mg.levelDims.xyz)){continue;}let v=mgTopology(q).x;if(v>1e-5&&mgPhi(q)<0.0){terms[n]=v*mgPhi(q);weights[n]=v;}}
  let sum=mgD4Sum6(terms);let weight=mgD4Sum6(weights);
  textureStore(mgPhiOut,id,vec4f(select(mgPhi(id),sum/max(weight,1e-9),weight>0.0)));
}

override MG_MASK_FIRST:bool=true;

@compute @workgroup_size(4,4,4)
fn mgBakeCoefficients(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  var mask=select(0u,1u,mgLiquid(id));
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  for(var n=0u;n<6u;n+=1u){if(mgLiquid(id+e[n])){mask|=2u<<n;}}


  if(MG_MASK_FIRST&&mask==0u){textureStore(mgCoefficientsOut,id,vec4f(0.0));return;}
  let coefficients=vec3f(
    mgCoefficientRaw(id,id+vec3i(1,0,0),0u),
    mgCoefficientRaw(id,id+vec3i(0,1,0),1u),
    mgCoefficientRaw(id,id+vec3i(0,0,1),2u));
  textureStore(mgCoefficientsOut,id,vec4f(coefficients,f32(mask)));
}
`;
