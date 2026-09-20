/** One storage-backed, strided coarse pressure solver for every grid size. */
import { UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE } from "./pressure-policy";

/** Seven scalar fields plus aligned vec4 topology: 48 bytes per haloed cell. */
export const UNIFORM_CM11A_COARSE_ROW_BYTES = 48;

export const UNIFORM_CM11A_COARSE_HEADER_BYTES = 112;
export const uniformPressureStateWGSL = /* wgsl */ `
struct UniformCoarseRow {
  p: f32, low: f32, rhs: f32, phi: f32,
  minimum: f32, residual: f32, state: u32, padding: u32,
  topology: vec4f,
};
struct UniformPressureState {
  convergence: array<atomic<u32>,26>,
  padding: vec2u,
  rows: array<UniformCoarseRow>,
};
@group(1) @binding(13) var<storage,read_write> mgState: UniformPressureState;
`;

// Scratch shares the convergence binding; no extra storage-buffer slot is needed.
export const uniformCoarseSolverWGSL = /* wgsl */ `
var<workgroup> mgCoarseResidualBits:atomic<u32>;
var<workgroup> mgCoarseMaxBBits:atomic<u32>;
var<workgroup> mgCoarseMaxDiagPBits:atomic<u32>;
var<workgroup> mgCoarseMaxPBits:atomic<u32>;
var<workgroup> mgCoarseMaxGapBits:atomic<u32>;
var<workgroup> mgCoarseActiveRows:atomic<u32>;
var<workgroup> mgCoarseFreeRows:atomic<u32>;
var<workgroup> mgCoarseWorstLane:atomic<u32>;


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
fn mgD4Sum6DS(value:array<vec2f,6>)->vec2f{
  return mgDSAdd(mgDSAdd(mgDSAdd(value[0],value[1]),mgDSAdd(value[4],value[5])),mgDSAdd(value[2],value[3]));
}
fn mgCoarsePressure(index:u32)->vec2f{return vec2f(mgState.rows[index].p,mgState.rows[index].low);}

fn mgCoarseIndex(p:vec3i)->u32{let d=vec3i(mg.levelDims.xyz);return u32(p.x+d.x*(p.y+d.y*p.z));}
fn mgCoarseCoefficient(id:vec3i,q:vec3i,axis:u32)->f32{
  let ci=mgCoarseIndex(id);let h=mg.spacing[axis];
  let d=vec3i(mg.levelDims.xyz);
  if(any(q<vec3i(0))||any(q>=d)){
    if(axis==1u&&id.y==d.y-1&&q.y==d.y&&params.boundary.w>0.5){
      let phi=mgState.rows[ci].phi;let theta=cm12GhostFluidTheta(phi,0.5*h,1e-9);
      return mgState.rows[ci].topology.z/(h*h*theta);
    }
    return 0.0;
  }
  let qi=mgCoarseIndex(q);let positive=q[axis]>id[axis];
  let vf=select(mgState.rows[qi].topology[axis+1u],mgState.rows[ci].topology[axis+1u],positive);
  if(vf<=1e-6){return 0.0;}let qPhi=mgState.rows[qi].phi;var theta=1.0;
  if(qPhi>=0.0){let phi=mgState.rows[ci].phi;theta=cm12GhostFluidTheta(phi,qPhi,1e-9);}
  return vf/(h*h*theta);
}

// A single workgroup owns the whole solve. Each colour completes across all
// rows before the other colour can read it, including rows in later batches.
// storageBarrier publishes cell state; workgroupBarrier publishes the reductions.
// There is never more than one workgroup, so these barriers cover every row.
@compute @workgroup_size(256)
fn mgSolveCoarsest(@builtin(local_invocation_index) localLane:u32){
  if(localLane==0u){mgCycleStopped=select(0u,1u,mgSkipCycle());}
  if(workgroupUniformLoad(&mgCycleStopped)!=0u){return;}
  let d=mg.levelDims.xyz;let count=d.x*d.y*d.z;
  for(var lane=localLane;lane<count;lane+=256u){
  
  let id=vec3i(i32(lane%d.x),i32((lane/d.x)%d.y),i32(lane/(d.x*d.y)));
  mgState.rows[lane].p=mgP(id);mgState.rows[lane].low=0.0;mgState.rows[lane].rhs=textureLoad(mgRhsIn,id,0).x;
    mgState.rows[lane].phi=mgPhi(id);mgState.rows[lane].minimum=textureLoad(mgMinimumIn,id,0).x;
    mgState.rows[lane].topology=mgTopology(id);
  }storageBarrier();workgroupBarrier();
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var converged=false;var iterations=0u;
  for(var iteration=0u;iteration<mg.control.w;iteration+=1u){
    if(!converged){iterations=iteration+1u;}
    for(var color=0u;color<2u;color+=1u){
      for(var lane=localLane;lane<count;lane+=256u){
    let id=vec3i(i32(lane%d.x),i32((lane/d.x)%d.y),i32(lane/(d.x*d.y)));
      if(!converged&&mgState.rows[lane].phi<0.0&&u32((id.x+id.y+id.z)&1)==color){
        var diagonalTerms:array<f32,6>;var sumTerms:array<vec2f,6>;
        for(var n=0;n<6;n+=1){let q=id+e[n];let a=mgCoarseCoefficient(id,q,u32(n/2));diagonalTerms[n]=a;sumTerms[n]=vec2f(0.0);if(all(q>=vec3i(0))&&all(q<vec3i(d))){let qi=mgCoarseIndex(q);if(mgState.rows[qi].phi<0.0){sumTerms[n]=mgDSScale(mgCoarsePressure(qi),a);}}}
        let diagonal=mgD4Sum6(diagonalTerms);let sum=mgD4Sum6DS(sumTerms);
        if(diagonal>0.0){let next=mgDSDivide(mgDSAdd(sum,vec2f(mgState.rows[lane].rhs,0.0)),diagonal);
          if(next.x+next.y<mgState.rows[lane].minimum){mgState.rows[lane].p=mgState.rows[lane].minimum;mgState.rows[lane].low=0.0;}
          else{mgState.rows[lane].p=next.x;mgState.rows[lane].low=next.y;}}
      }
      }storageBarrier();workgroupBarrier();
    }
    for(var lane=localLane;lane<count;lane+=256u){
    let id=vec3i(i32(lane%d.x),i32((lane/d.x)%d.y),i32(lane/(d.x*d.y)));
    if(!converged&&mgState.rows[lane].p+mgState.rows[lane].low<mgState.rows[lane].minimum){mgState.rows[lane].p=mgState.rows[lane].minimum;mgState.rows[lane].low=0.0;}
    }storageBarrier();workgroupBarrier();
    if(localLane==0u&&!converged){atomicStore(&mgCoarseResidualBits,0u);atomicStore(&mgCoarseMaxBBits,0u);atomicStore(&mgCoarseMaxDiagPBits,0u);atomicStore(&mgCoarseMaxPBits,0u);atomicStore(&mgCoarseMaxGapBits,0u);atomicStore(&mgCoarseActiveRows,0u);atomicStore(&mgCoarseFreeRows,0u);atomicStore(&mgCoarseWorstLane,0xffffffffu);}
    for(var lane=localLane;lane<count;lane+=256u){
    let id=vec3i(i32(lane%d.x),i32((lane/d.x)%d.y),i32(lane/(d.x*d.y)));
    mgState.rows[lane].residual=0.0;mgState.rows[lane].state=0u;
    }storageBarrier();workgroupBarrier();
    for(var lane=localLane;lane<count;lane+=256u){
    let id=vec3i(i32(lane%d.x),i32((lane/d.x)%d.y),i32(lane/(d.x*d.y)));
    if(!converged&&mgState.rows[lane].phi<0.0){var appliedTerms:array<vec2f,6>;var diagonalTerms:array<f32,6>;
      for(var n=0;n<6;n+=1){let q=id+e[n];let a=mgCoarseCoefficient(id,q,u32(n/2));diagonalTerms[n]=a;var neighbor=vec2f(0.0);if(all(q>=vec3i(0))&&all(q<vec3i(d))){let qi=mgCoarseIndex(q);if(mgState.rows[qi].phi<0.0){neighbor=mgCoarsePressure(qi);}}appliedTerms[n]=mgDSScale(mgDSAdd(mgCoarsePressure(lane),-neighbor),a);}
      let applied=mgD4Sum6DS(appliedTerms);let diagonal=mgD4Sum6(diagonalTerms);
      let linearResidualPair=mgDSAdd(vec2f(mgState.rows[lane].rhs,0.0),-applied);let linearResidual=linearResidualPair.x+linearResidualPair.y;
      // A raw linear residual is non-zero at a legitimately active lower
      // bound. Measure the projected LCP fixed-point residual instead; away
      // from p_min it is exactly the linear residual in the paper's units.
      if(diagonal>0.0){let pressure=mgState.rows[lane].p+mgState.rows[lane].low;let gap=max(0.0,pressure-mgState.rows[lane].minimum);
        // Evaluate the projected fixed-point residual without adding a tiny
        // correction to a large pressure (which would round away in f32).
        let projectsToMinimum=linearResidual<0.0&&-linearResidual>=gap*diagonal;
        let lcpResidual=select(abs(linearResidual),gap*diagonal,projectsToMinimum);
        // A p=b is scaled by rho/dt in the pressure system. TallCells reports
        // its absolute infinity tolerance in s^-1, so convergence and the
        // public residual must use the equivalent divergence residual.
        let divergenceResidual=lcpResidual*params.dimsDt.w/params.physical.x;
        let rowActive=projectsToMinimum;
        mgState.rows[lane].residual=divergenceResidual;mgState.rows[lane].state=select(0u,1u,rowActive);
        if(rowActive){atomicAdd(&mgCoarseActiveRows,1u);}else{atomicAdd(&mgCoarseFreeRows,1u);}
        atomicMax(&mgCoarseResidualBits,bitcast<u32>(divergenceResidual));atomicMax(&mgCoarseMaxBBits,bitcast<u32>(abs(mgState.rows[lane].rhs)));
        atomicMax(&mgCoarseMaxDiagPBits,bitcast<u32>(diagonal*abs(pressure)));atomicMax(&mgCoarseMaxPBits,bitcast<u32>(abs(pressure)));
        atomicMax(&mgCoarseMaxGapBits,bitcast<u32>(lcpResidual/diagonal));}
    }
    }storageBarrier();workgroupBarrier();
    for(var lane=localLane;lane<count;lane+=256u){
    let id=vec3i(i32(lane%d.x),i32((lane/d.x)%d.y),i32(lane/(d.x*d.y)));
    if(!converged&&bitcast<u32>(mgState.rows[lane].residual)==atomicLoad(&mgCoarseResidualBits)){atomicMin(&mgCoarseWorstLane,lane);}
    }storageBarrier();workgroupBarrier();
    // The residual maximum is a workgroup atomic every lane reads after a
    // barrier, but WGSL's uniformity analysis cannot see that, so route the
    // verdict through workgroupUniformLoad (itself a barrier) to make the
    // break formally uniform. Post-convergence iterations computed nothing
    // (every phase above is gated on !converged), so leaving the loop early
    // is bit-identical; it avoids the remaining capped iterations and barriers.
    if(localLane==0u){mgCoarseConvergedFlag=select(0u,1u,bitcast<f32>(atomicLoad(&mgCoarseResidualBits))<=${UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE});}
    if(workgroupUniformLoad(&mgCoarseConvergedFlag)==1u){converged=true;break;}
  }
  for(var lane=localLane;lane<count;lane+=256u){
    let id=vec3i(i32(lane%d.x),i32((lane/d.x)%d.y),i32(lane/(d.x*d.y)));
  textureStore(mgPressureOut,id,vec4f(mgState.rows[lane].p+mgState.rows[lane].low));
  }
  if(localLane==0u){atomicMax(&mgState.convergence[0],atomicLoad(&mgCoarseResidualBits));atomicStore(&mgState.convergence[1],select(0u,1u,converged));atomicMax(&mgState.convergence[2],iterations);atomicMax(&mgState.convergence[3],select(1u,0u,converged));
    if(!converged){let claimed=atomicCompareExchangeWeak(&mgState.convergence[4],0u,mg.control.z);if(claimed.exchanged){let maxB=bitcast<f32>(atomicLoad(&mgCoarseMaxBBits));let maxDiagP=bitcast<f32>(atomicLoad(&mgCoarseMaxDiagPBits));
      atomicStore(&mgState.convergence[5],atomicLoad(&mgCoarseMaxBBits));atomicStore(&mgState.convergence[6],atomicLoad(&mgCoarseMaxDiagPBits));atomicStore(&mgState.convergence[7],atomicLoad(&mgCoarseMaxPBits));atomicStore(&mgState.convergence[8],atomicLoad(&mgCoarseMaxGapBits));
      let rawResidual=bitcast<f32>(atomicLoad(&mgCoarseResidualBits))*params.physical.x/max(params.dimsDt.w,1e-20);
      atomicStore(&mgState.convergence[9],bitcast<u32>(rawResidual/max(maxB,max(maxDiagP,1e-20))));
      let worstLane=atomicLoad(&mgCoarseWorstLane);let worstId=vec3i(i32(worstLane%d.x),i32((worstLane/d.x)%d.y),i32(worstLane/(d.x*d.y)));
      let halo=!mgInterior(worstId,d);let packed=worstLane|(mgState.rows[worstLane].state<<30u)|(select(0u,1u,halo)<<31u);
      atomicStore(&mgState.convergence[12],atomicLoad(&mgCoarseActiveRows));atomicStore(&mgState.convergence[13],atomicLoad(&mgCoarseFreeRows));atomicStore(&mgState.convergence[14],packed);}}}
}
`;
