/**
 * CM11a dense pressure-hierarchy fragment.
 *
 * This is appended to the uniform solver shader so the finest-level build
 * calls the same pressurePhi, faceOpenFraction, divergenceAt, and solid
 * helpers as projection.  The fragment intentionally contains no alternate
 * solid or free-surface discretization.
 */
export const UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE = 1e-4;

export const uniformPressureMultigridWGSL = /* wgsl */ `
struct UniformMGParams {
  fineDims: vec4u,
  levelDims: vec4u,
  coarseDims: vec4u,
  spacing: vec4f,
  control: vec4u,
};

@group(1) @binding(0) var<uniform> mg: UniformMGParams;
@group(1) @binding(1) var mgPressureIn: texture_3d<f32>;
@group(1) @binding(2) var mgPressureOut: texture_storage_3d<r32float,write>;
@group(1) @binding(3) var mgRhsIn: texture_3d<f32>;
@group(1) @binding(4) var mgRhsOut: texture_storage_3d<r32float,write>;
@group(1) @binding(5) var mgPhiIn: texture_3d<f32>;
@group(1) @binding(6) var mgPhiOut: texture_storage_3d<r32float,write>;
// x is cell V; y/z/w are the positive x/y/z face V values.
@group(1) @binding(7) var mgVolumeIn: texture_3d<f32>;
@group(1) @binding(8) var mgVolumeOut: texture_storage_3d<rgba32float,write>;
@group(1) @binding(9) var mgResidualIn: texture_3d<f32>;
@group(1) @binding(10) var mgResidualOut: texture_storage_3d<r32float,write>;
@group(1) @binding(11) var mgMinimumIn: texture_3d<f32>;
@group(1) @binding(12) var mgMinimumOut: texture_storage_3d<r32float,write>;
// 0..3: worst coarse projected divergence residual (s^-1), final convergence,
// max iterations, cap fail.
// 4..9: first failing invocation, |b|max, diag|p|max, |p|max,
// projected pressure-gap max, and normalized projected residual max.
// 10..11: final finest projected residual and pressure gap.
// 12..14: active/free coarsest rows and packed worst-row state for the first
// failing solve (lane | active<<16 | halo<<17).
@group(1) @binding(13) var<storage,read_write> mgConvergence:array<atomic<u32>,15>;

fn mgValid(p:vec3i,d:vec3u)->bool{return all(p>=vec3i(0))&&all(p<vec3i(d));}
fn mgClamp(p:vec3i,d:vec3u)->vec3i{return clamp(p,vec3i(0),vec3i(d)-vec3i(1));}
fn mgFineChild(coarse:vec3i,o:vec3i)->vec3i{
  let finePhysical=vec3i(mg.levelDims.xyz)-vec3i(2);
  return clamp(2*(coarse-vec3i(1))+o,vec3i(-1),finePhysical)+vec3i(1);
}
fn mgTopology(p:vec3i)->vec4f{return textureLoad(mgVolumeIn,mgClamp(p,mg.levelDims.xyz),0);}
fn mgPhi(p:vec3i)->f32{return textureLoad(mgPhiIn,mgClamp(p,mg.levelDims.xyz),0).x;}
fn mgP(p:vec3i)->f32{return textureLoad(mgPressureIn,mgClamp(p,mg.levelDims.xyz),0).x;}
fn mgLiquid(p:vec3i)->bool{return mgValid(p,mg.levelDims.xyz)&&mgPhi(p)<0.0;}
fn mgInterior(p:vec3i,d:vec3u)->bool{return all(p>=vec3i(1))&&all(p<vec3i(d)-vec3i(1));}
fn mgOpenTopHalo(p:vec3i,d:vec3u)->bool{
  return params.boundary.w>0.5&&p.y==i32(d.y)-1&&p.x>0&&p.x<i32(d.x)-1&&p.z>0&&p.z<i32(d.z)-1;
}
fn mgFaceV(id:vec3i,neighbor:vec3i,axis:u32)->f32{
  let positive=neighbor[axis]>id[axis];
  return select(mgTopology(neighbor)[axis+1u],mgTopology(id)[axis+1u],positive);
}
fn mgTheta(liquidCell:vec3i,airCell:vec3i)->f32{
  let a=mgPhi(liquidCell);let b=mgPhi(airCell);
  return clamp(abs(a)/max(abs(a)+abs(b),1e-9),0.05,1.0);
}
fn mgCoefficient(id:vec3i,q:vec3i,axis:u32)->f32{
  if(!mgValid(q,mg.levelDims.xyz)){
    // The authored +Y opening is atmospheric air. All other exterior faces
    // are covered walls, matching the fine-grid stencil helper.
    if(axis==1u&&id.y==i32(mg.levelDims.y)-1&&q.y==i32(mg.levelDims.y)&&params.boundary.w>0.5){
      let h=mg.spacing.y;let phi=mgPhi(id);let exteriorPhi=0.5*h;
      let theta=clamp(abs(phi)/max(abs(phi)+exteriorPhi,1e-9),0.05,1.0);
      return mgTopology(id).z/(h*h*theta);
    }
    return 0.0;
  }
  let vf=mgFaceV(id,q,axis);if(vf<=1e-6){return 0.0;}
  let h=mg.spacing[axis];
  return vf/(h*h*select(mgTheta(id,q),1.0,mgLiquid(q)));
}
fn mgApply(id:vec3i)->f32{
  if(!mgLiquid(id)){return 0.0;}
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var result=0.0;let centre=mgP(id);
  for(var n=0;n<6;n+=1){let q=id+e[n];let axis=u32(n/2);let a=mgCoefficient(id,q,axis);let neighbor=select(0.0,mgP(q),mgLiquid(q));result+=a*(centre-neighbor);}
  return result;
}

@compute @workgroup_size(4,4,4)
fn mgBuildFinestTopology(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}let simulation=id-vec3i(1);
  let h=mg.spacing.xyz;
  if(mgInterior(id,mg.levelDims.xyz)){
    let topology=vec4f(cellOpenFraction(simulation),pressureFaceVolumeFraction(simulation,0u),pressureFaceVolumeFraction(simulation,1u),pressureFaceVolumeFraction(simulation,2u));
    textureStore(mgPhiOut,id,vec4f(pressurePhi(simulation)));textureStore(mgVolumeOut,id,topology);return;
  }
  let openTop=mgOpenTopHalo(id,mg.levelDims.xyz);var topology=vec4f(select(0.0,1.0,openTop));
  // Low-side halo cells own the three missing negative face-centred dual
  // cells. Their closed, grid-aligned domain halves have V=1/2.
  if(id.x==0&&id.y>0&&id.y<i32(mg.levelDims.y)-1&&id.z>0&&id.z<i32(mg.levelDims.z)-1){topology.y=0.5;}
  if(id.y==0&&id.x>0&&id.x<i32(mg.levelDims.x)-1&&id.z>0&&id.z<i32(mg.levelDims.z)-1){topology.z=0.5;}
  if(id.z==0&&id.x>0&&id.x<i32(mg.levelDims.x)-1&&id.y>0&&id.y<i32(mg.levelDims.y)-1){topology.w=0.5;}
  textureStore(mgPhiOut,id,vec4f(0.5*min(h.x,min(h.y,h.z))));textureStore(mgVolumeOut,id,topology);
}

@compute @workgroup_size(4,4,4)
fn mgBuildFinestRhs(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}let simulation=id-vec3i(1);
  var rhs=0.0;var minimum=-3.402823e38;
  if(mgInterior(id,mg.levelDims.xyz)){
    let checkSolid=nearAnyBody(worldCell(simulation));rhs=params.physical.x*(divergenceAt(simulation,checkSolid)-volumeCorrectionDivergence(simulation))/params.dimsDt.w;
    minimum=select(-3.402823e38,0.0,cellInsideSolid(simulation)||cellInsideTerrain(simulation));
  }else if(!mgOpenTopHalo(id,mg.levelDims.xyz)){minimum=0.0;}
  // The hierarchy uses A p = b with A p = sum a(p_i-p_j). The existing fine
  // projection convention therefore publishes b=-rho div(u*)/dt.
  textureStore(mgRhsOut,id,vec4f(-rhs));
  textureStore(mgPressureOut,id,vec4f(0.0));
  textureStore(mgMinimumOut,id,vec4f(minimum));
}

@compute @workgroup_size(4,4,4)
fn mgDownsampleTopology(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}
  var v=vec4f(0.0);var phiSum=0.0;var positiveSum=0.0;var positiveCount=0.0;var negativeCount=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let q=mgFineChild(id,o);
    v+=mgTopology(q);let phi=mgPhi(q);phiSum+=phi;if(phi>=0.0){positiveSum+=phi;positiveCount+=1.0;}else{negativeCount+=1.0;}
  }
  // CM11a Eq. 15-16 and C=2 sign-aware phi rule. control.x is the
  // destination level and control.y is M-C.
  let mixed=positiveCount>0.0&&negativeCount>0.0;
  let usePositive=mixed&&mg.control.x>=mg.control.y;
  let coarsePhi=select(phiSum/8.0,positiveSum/max(positiveCount,1.0),usePositive);
  textureStore(mgVolumeOut,id,v/8.0);textureStore(mgPhiOut,id,vec4f(coarsePhi));
}

@compute @workgroup_size(4,4,4)
fn mgResidual(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  textureStore(mgResidualOut,id,vec4f(textureLoad(mgRhsIn,id,0).x-mgApply(id)));
}

// Cell-centred trilinear restriction: a coarse centre lies halfway between
// each of its eight fine centres.  CM11a uses the same trilinear operator for
// restriction and prolongation.
@compute @workgroup_size(4,4,4)
fn mgRestrictResidual(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}var value=0.0;
  for(var corner=0u;corner<8u;corner+=1u){let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));value+=textureLoad(mgResidualIn,mgFineChild(id,o),0).x;}
  textureStore(mgRhsOut,id,vec4f(value/8.0));
}

fn mgTrilinearPressure(fineId:vec3i)->f32{
  let q=(vec3f(fineId)-vec3f(0.5))*0.5+vec3f(0.5);let base=vec3i(floor(q));let f=fract(q);
  var value=0.0;var total=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let p=base+o;
    if(!mgValid(p,mg.levelDims.xyz)){continue;}
    let w=select(1.0-f.x,f.x,o.x==1)*select(1.0-f.y,f.y,o.y==1)*select(1.0-f.z,f.z,o.z==1);
    value+=w*mgP(p);total+=w;
  }
  return select(0.0,value/total,total>0.0);
}

@compute @workgroup_size(4,4,4)
fn mgProlongateAdd(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}
  textureStore(mgPressureOut,id,vec4f(textureLoad(mgResidualIn,id,0).x+mgTrilinearPressure(id)));
}

@compute @workgroup_size(4,4,4)
fn mgProlongateAssign(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}
  textureStore(mgPressureOut,id,vec4f(mgTrilinearPressure(id)));
}

@compute @workgroup_size(4,4,4)
fn mgCopyPressure(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  textureStore(mgPressureOut,id,vec4f(mgP(id)));
}

@compute @workgroup_size(4,4,4)
fn mgClearPressure(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  textureStore(mgPressureOut,id,vec4f(0.0));
}

@compute @workgroup_size(4,4,4)
fn mgClearMinimum(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  textureStore(mgMinimumOut,id,vec4f(-3.402823e38));
}

@compute @workgroup_size(4,4,4)
fn mgShiftMinimum(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  textureStore(mgMinimumOut,id,vec4f(textureLoad(mgMinimumIn,id,0).x-mgP(id)));
}

@compute @workgroup_size(4,4,4)
fn mgAddPressure(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  textureStore(mgPressureOut,id,vec4f(mgP(id)+textureLoad(mgResidualIn,id,0).x));
}

// The paper requires one layer of phi in solid cells on every level. The
// authoritative finest continuation is pressurePhi; coarse levels continue
// that field from their face-adjacent non-solid cells for exactly one pass.
@compute @workgroup_size(4,4,4)
fn mgExtrapolatePhiOneCell(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  if(mgTopology(id).x>1e-5){textureStore(mgPhiOut,id,vec4f(mgPhi(id)));return;}
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var sum=0.0;var weight=0.0;
  for(var n=0;n<6;n+=1){let q=id+e[n];if(!mgValid(q,mg.levelDims.xyz)){continue;}let v=mgTopology(q).x;if(v>1e-5&&mgPhi(q)<0.0){sum+=v*mgPhi(q);weight+=v;}}
  textureStore(mgPhiOut,id,vec4f(select(mgPhi(id),sum/max(weight,1e-9),weight>0.0)));
}

// CM11a Eq. 19-20. mgMinimumIn and mgPressureIn are the fine p_min and
// current p; mgMinimumOut is the next-coarser constraint field.
@compute @workgroup_size(4,4,4)
fn mgDownsampleSubtract(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}var lower=-3.402823e38;
  for(var corner=0u;corner<8u;corner+=1u){let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let q=mgFineChild(id,o);lower=max(lower,textureLoad(mgMinimumIn,q,0).x-mgP(q));}
  textureStore(mgMinimumOut,id,vec4f(lower));
}

@compute @workgroup_size(4,4,4)
fn mgDownsampleMinimum(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}var lower=-3.402823e38;
  for(var corner=0u;corner<8u;corner+=1u){let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let q=mgFineChild(id,o);lower=max(lower,textureLoad(mgMinimumIn,q,0).x);}
  textureStore(mgMinimumOut,id,vec4f(lower));
}

@compute @workgroup_size(4,4,4)
fn mgSmoothColour(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  let old=mgP(id);let coarseDone=(mg.control.w&2u)!=0u&&atomicLoad(&mgConvergence[1])!=0u;
  // Pass-through cells carry the CM11a Eq. 18 projection with them. Nothing
  // reads a wrong-colour or non-liquid cell between the two colour passes
  // (neighbour sums are mgLiquid-gated and an update never reads its own old
  // value), so projecting here leaves every sweep-exit value identical to the
  // former trailing mgProjectMinimum pass while deleting that pass outright.
  if(coarseDone||!mgLiquid(id)||u32((id.x+id.y+id.z)&1)!=mg.control.z){textureStore(mgPressureOut,id,vec4f(old));return;}
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var diagonal=0.0;var sum=0.0;
  for(var n=0;n<6;n+=1){let q=id+e[n];let a=mgCoefficient(id,q,u32(n/2));diagonal+=a;if(mgLiquid(q)){sum+=a*mgP(q);}}
  let p=select(0.0,(sum+textureLoad(mgRhsIn,id,0).x)/diagonal,diagonal>0.0);
  // CM11a Eq. 18 says that p_min is enforced while smoothing. Project the
  // newly updated colour before the opposite colour consumes it.
  textureStore(mgPressureOut,id,vec4f(max(p,textureLoad(mgMinimumIn,id,0).x)));
}

// CM11a's PRBGS sweep is two colour passes followed by this projection.
@compute @workgroup_size(4,4,4)
fn mgProjectMinimum(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  textureStore(mgPressureOut,id,vec4f(max(mgP(id),textureLoad(mgMinimumIn,id,0).x)));
}

var<workgroup> mgCoarseP:array<f32,256>;
var<workgroup> mgCoarsePLow:array<f32,256>;
var<workgroup> mgCoarseRhs:array<f32,256>;
var<workgroup> mgCoarsePhi:array<f32,256>;
var<workgroup> mgCoarseMin:array<f32,256>;
var<workgroup> mgCoarseTopology:array<vec4f,256>;
var<workgroup> mgCoarseResidualBits:atomic<u32>;
var<workgroup> mgCoarseMaxBBits:atomic<u32>;
var<workgroup> mgCoarseMaxDiagPBits:atomic<u32>;
var<workgroup> mgCoarseMaxPBits:atomic<u32>;
var<workgroup> mgCoarseMaxGapBits:atomic<u32>;
var<workgroup> mgCoarseActiveRows:atomic<u32>;
var<workgroup> mgCoarseFreeRows:atomic<u32>;
var<workgroup> mgCoarseWorstLane:atomic<u32>;
var<workgroup> mgCoarseRowResidual:array<f32,256>;
var<workgroup> mgCoarseRowState:array<u32,256>;
var<workgroup> mgCoarseConvergedFlag:u32;

fn mgTwoSum(a:f32,b:f32)->vec2f{
  let s=a+b;let bb=s-a;return vec2f(s,(a-(s-bb))+(b-bb));
}
fn mgDSAdd(a:vec2f,b:vec2f)->vec2f{
  let s=mgTwoSum(a.x,b.x);let t=mgTwoSum(a.y,b.y);let u=mgTwoSum(s.y,t.x);let v=mgTwoSum(s.x,u.x);
  return vec2f(v.x,v.y+u.y+t.y);
}
fn mgDSScale(a:vec2f,b:f32)->vec2f{
  let product=a.x*b;let error=fma(a.x,b,-product)+a.y*b;let sum=mgTwoSum(product,error);return sum;
}
fn mgDSDivide(a:vec2f,b:f32)->vec2f{
  let q=a.x/b;let remainder=mgDSAdd(a,-mgDSScale(vec2f(q,0.0),b));let correction=(remainder.x+remainder.y)/b;
  return mgTwoSum(q,correction);
}
fn mgCoarsePressure(index:u32)->vec2f{return vec2f(mgCoarseP[index],mgCoarsePLow[index]);}

fn mgCoarseIndex(p:vec3i)->u32{let d=vec3i(mg.levelDims.xyz);return u32(p.x+d.x*(p.y+d.y*p.z));}
fn mgCoarseCoefficient(id:vec3i,q:vec3i,axis:u32)->f32{
  let ci=mgCoarseIndex(id);let h=mg.spacing[axis];
  let d=vec3i(mg.levelDims.xyz);
  if(any(q<vec3i(0))||any(q>=d)){
    if(axis==1u&&id.y==d.y-1&&q.y==d.y&&params.boundary.w>0.5){
      let phi=mgCoarsePhi[ci];let theta=clamp(abs(phi)/max(abs(phi)+0.5*h,1e-9),0.05,1.0);
      return mgCoarseTopology[ci].z/(h*h*theta);
    }
    return 0.0;
  }
  let qi=mgCoarseIndex(q);let positive=q[axis]>id[axis];
  let vf=select(mgCoarseTopology[qi][axis+1u],mgCoarseTopology[ci][axis+1u],positive);
  if(vf<=1e-6){return 0.0;}let qPhi=mgCoarsePhi[qi];var theta=1.0;
  if(qPhi>=0.0){let phi=mgCoarsePhi[ci];theta=clamp(abs(phi)/max(abs(phi)+abs(qPhi),1e-9),0.05,1.0);}
  return vf/(h*h*theta);
}

// The hierarchy's short axis ends at two cells. The host limits the resulting
// rectangular coarsest grid to 256 cells so one workgroup can execute the
// paper's high-precision solve without a host readback.
@compute @workgroup_size(256)
fn mgSolveCoarsest(@builtin(local_invocation_index) lane:u32){
  let d=mg.levelDims.xyz;let count=d.x*d.y*d.z;let live=lane<count;
  let id=vec3i(i32(lane%d.x),i32((lane/d.x)%d.y),i32(lane/(d.x*d.y)));
  if(live){mgCoarseP[lane]=mgP(id);mgCoarsePLow[lane]=0.0;mgCoarseRhs[lane]=textureLoad(mgRhsIn,id,0).x;
    mgCoarsePhi[lane]=mgPhi(id);mgCoarseMin[lane]=textureLoad(mgMinimumIn,id,0).x;
    mgCoarseTopology[lane]=mgTopology(id);}workgroupBarrier();
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var converged=false;var iterations=0u;
  for(var iteration=0u;iteration<mg.control.w;iteration+=1u){
    if(!converged){iterations=iteration+1u;}
    for(var color=0u;color<2u;color+=1u){
      if(!converged&&live&&mgCoarsePhi[lane]<0.0&&u32((id.x+id.y+id.z)&1)==color){
        var diagonal=0.0;var sum=vec2f(0.0);
        for(var n=0;n<6;n+=1){let q=id+e[n];let a=mgCoarseCoefficient(id,q,u32(n/2));diagonal+=a;if(all(q>=vec3i(0))&&all(q<vec3i(d))){let qi=mgCoarseIndex(q);if(mgCoarsePhi[qi]<0.0){sum=mgDSAdd(sum,mgDSScale(mgCoarsePressure(qi),a));}}}
        if(diagonal>0.0){let next=mgDSDivide(mgDSAdd(sum,vec2f(mgCoarseRhs[lane],0.0)),diagonal);
          if(next.x+next.y<mgCoarseMin[lane]){mgCoarseP[lane]=mgCoarseMin[lane];mgCoarsePLow[lane]=0.0;}
          else{mgCoarseP[lane]=next.x;mgCoarsePLow[lane]=next.y;}}
      }
      workgroupBarrier();
    }
    if(!converged&&live&&mgCoarseP[lane]+mgCoarsePLow[lane]<mgCoarseMin[lane]){mgCoarseP[lane]=mgCoarseMin[lane];mgCoarsePLow[lane]=0.0;}workgroupBarrier();
    if(lane==0u&&!converged){atomicStore(&mgCoarseResidualBits,0u);atomicStore(&mgCoarseMaxBBits,0u);atomicStore(&mgCoarseMaxDiagPBits,0u);atomicStore(&mgCoarseMaxPBits,0u);atomicStore(&mgCoarseMaxGapBits,0u);atomicStore(&mgCoarseActiveRows,0u);atomicStore(&mgCoarseFreeRows,0u);atomicStore(&mgCoarseWorstLane,0xffffffffu);}if(live){mgCoarseRowResidual[lane]=0.0;mgCoarseRowState[lane]=0u;}workgroupBarrier();
    if(!converged&&live&&mgCoarsePhi[lane]<0.0){var applied=vec2f(0.0);var diagonal=0.0;
      for(var n=0;n<6;n+=1){let q=id+e[n];let a=mgCoarseCoefficient(id,q,u32(n/2));diagonal+=a;var neighbor=vec2f(0.0);if(all(q>=vec3i(0))&&all(q<vec3i(d))){let qi=mgCoarseIndex(q);if(mgCoarsePhi[qi]<0.0){neighbor=mgCoarsePressure(qi);}}applied=mgDSAdd(applied,mgDSScale(mgDSAdd(mgCoarsePressure(lane),-neighbor),a));}
      let linearResidualPair=mgDSAdd(vec2f(mgCoarseRhs[lane],0.0),-applied);let linearResidual=linearResidualPair.x+linearResidualPair.y;
      // A raw linear residual is non-zero at a legitimately active lower
      // bound. Measure the projected LCP fixed-point residual instead; away
      // from p_min it is exactly the linear residual in the paper's units.
      if(diagonal>0.0){let pressure=mgCoarseP[lane]+mgCoarsePLow[lane];let gap=max(0.0,pressure-mgCoarseMin[lane]);
        // Evaluate the projected fixed-point residual without adding a tiny
        // correction to a large pressure (which would round away in f32).
        let projectsToMinimum=linearResidual<0.0&&-linearResidual>=gap*diagonal;
        let lcpResidual=select(abs(linearResidual),gap*diagonal,projectsToMinimum);
        // A p=b is scaled by rho/dt in the pressure system. TallCells reports
        // its absolute infinity tolerance in s^-1, so convergence and the
        // public residual must use the equivalent divergence residual.
        let divergenceResidual=lcpResidual*params.dimsDt.w/params.physical.x;
        let rowActive=projectsToMinimum;
        mgCoarseRowResidual[lane]=divergenceResidual;mgCoarseRowState[lane]=select(0u,1u,rowActive);
        if(rowActive){atomicAdd(&mgCoarseActiveRows,1u);}else{atomicAdd(&mgCoarseFreeRows,1u);}
        atomicMax(&mgCoarseResidualBits,bitcast<u32>(divergenceResidual));atomicMax(&mgCoarseMaxBBits,bitcast<u32>(abs(mgCoarseRhs[lane])));
        atomicMax(&mgCoarseMaxDiagPBits,bitcast<u32>(diagonal*abs(pressure)));atomicMax(&mgCoarseMaxPBits,bitcast<u32>(abs(pressure)));
        atomicMax(&mgCoarseMaxGapBits,bitcast<u32>(lcpResidual/diagonal));}
    }
    workgroupBarrier();if(!converged&&live&&bitcast<u32>(mgCoarseRowResidual[lane])==atomicLoad(&mgCoarseResidualBits)){atomicMin(&mgCoarseWorstLane,lane);}workgroupBarrier();
    // The residual maximum is a workgroup atomic every lane reads after a
    // barrier, but WGSL's uniformity analysis cannot see that, so route the
    // verdict through workgroupUniformLoad (itself a barrier) to make the
    // break formally uniform. Post-convergence iterations computed nothing
    // (every phase above is gated on !converged), so leaving the loop early
    // is bit-identical; it just stops paying ~8 barrier waves per remaining
    // capped iteration on this one 256-lane workgroup.
    if(lane==0u){mgCoarseConvergedFlag=select(0u,1u,bitcast<f32>(atomicLoad(&mgCoarseResidualBits))<=${UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE});}
    if(workgroupUniformLoad(&mgCoarseConvergedFlag)==1u){converged=true;break;}
  }
  if(live){textureStore(mgPressureOut,id,vec4f(mgCoarseP[lane]+mgCoarsePLow[lane]));}
  if(lane==0u){atomicMax(&mgConvergence[0],atomicLoad(&mgCoarseResidualBits));atomicStore(&mgConvergence[1],select(0u,1u,converged));atomicMax(&mgConvergence[2],iterations);atomicMax(&mgConvergence[3],select(1u,0u,converged));
    if(!converged){let claimed=atomicCompareExchangeWeak(&mgConvergence[4],0u,mg.control.z);if(claimed.exchanged){let maxB=bitcast<f32>(atomicLoad(&mgCoarseMaxBBits));let maxDiagP=bitcast<f32>(atomicLoad(&mgCoarseMaxDiagPBits));
      atomicStore(&mgConvergence[5],atomicLoad(&mgCoarseMaxBBits));atomicStore(&mgConvergence[6],atomicLoad(&mgCoarseMaxDiagPBits));atomicStore(&mgConvergence[7],atomicLoad(&mgCoarseMaxPBits));atomicStore(&mgConvergence[8],atomicLoad(&mgCoarseMaxGapBits));
      let rawResidual=bitcast<f32>(atomicLoad(&mgCoarseResidualBits))*params.physical.x/max(params.dimsDt.w,1e-20);
      atomicStore(&mgConvergence[9],bitcast<u32>(rawResidual/max(maxB,max(maxDiagP,1e-20))));
      let worstLane=atomicLoad(&mgCoarseWorstLane);let worstId=vec3i(i32(worstLane%d.x),i32((worstLane/d.x)%d.y),i32(worstLane/(d.x*d.y)));
      let halo=!mgInterior(worstId,d);let packed=worstLane|(mgCoarseRowState[worstLane]<<16u)|(select(0u,1u,halo)<<17u);
      atomicStore(&mgConvergence[12],atomicLoad(&mgCoarseActiveRows));atomicStore(&mgConvergence[13],atomicLoad(&mgCoarseFreeRows));atomicStore(&mgConvergence[14],packed);}}}
}

@compute @workgroup_size(4,4,4)
fn mgMeasureFineResidual(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!mgValid(id,mg.levelDims.xyz)||!mgLiquid(id)){return;}
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var diagonal=0.0;for(var n=0;n<6;n+=1){diagonal+=mgCoefficient(id,id+e[n],u32(n/2));}
  if(diagonal<=0.0){return;}let residual=textureLoad(mgRhsIn,id,0).x-mgApply(id);let pressure=mgP(id);let minimum=textureLoad(mgMinimumIn,id,0).x;let gap=max(0.0,pressure-minimum);
  let projectsToMinimum=residual<0.0&&-residual>=gap*diagonal;let projected=select(abs(residual),gap*diagonal,projectsToMinimum);
  atomicMax(&mgConvergence[10],bitcast<u32>(projected*params.dimsDt.w/params.physical.x));atomicMax(&mgConvergence[11],bitcast<u32>(projected/diagonal));
}
`;
