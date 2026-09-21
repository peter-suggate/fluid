import { uniformCoarseSolverWGSL, uniformPressureStateWGSL } from "./uniform-coarse-solver.wgsl";
/**
 * CM11a dense pressure-hierarchy fragment.
 *
 * This is appended to the uniform solver shader so the finest-level build
 * calls the same pressurePhi, faceOpenFraction, divergenceAt, and solid
 * helpers as projection.  The fragment intentionally contains no alternate
 * solid or free-surface discretization.
 */
import { UNIFORM_CM11A_RECOVERY_SWEEPS, UNIFORM_CM11A_RECOVERY_REDUCTION } from "./pressure-policy";
export { UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE } from "./pressure-policy";

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
// failing solve (row | active<<30 | halo<<31).
// 15..18: cycle residual norm bits, stopped, completed Full-Cycles, V-Cycles.
// 19..25: best residual, rejected cycles, recovery sweeps, recovery mode,
// rejected candidate, initial residual, recovery exhausted.
${uniformPressureStateWGSL}
// Per-level immutable (+x,+y,+z) coefficients and liquid flag, baked once
// after topology and the one-cell phi continuation are complete.
@group(1) @binding(14) var mgCoefficientsIn: texture_3d<f32>;
@group(1) @binding(15) var mgCoefficientsOut: texture_storage_3d<rgba32float,write>;

@group(1) @binding(17) var<uniform> mgTolerance:vec4f;
var<workgroup> mgCycleStopped:u32;
fn mgSkipCycle()->bool{
  if(mg.levelDims.w==0u){return false;}
  if(atomicLoad(&mgState.convergence[16])!=0u){return true;}
  let recovery=atomicLoad(&mgState.convergence[22])!=0u;
  return select(recovery,!recovery,mg.levelDims.w==2u);
}

// Published only after accepted/rejected pressure has been committed. A
// converged solve launches zero workgroups for all subsequent cycle kernels.
// Save/restore and final diagnostics remain unconditional.
@group(1) @binding(18) var<storage,read_write> mgCycleDispatch:array<u32>;
@compute @workgroup_size(1)
fn mgPublishCycleDispatch(){
  let records=mg.control.z;
  let stopped=atomicLoad(&mgState.convergence[16])!=0u;
  let recovery=atomicLoad(&mgState.convergence[22])!=0u;
  for(var gate=1u;gate<=2u;gate++){
    let enabled=!stopped&&select(!recovery,recovery,gate==2u);
    for(var i=0u;i<records*3u;i++){
      mgCycleDispatch[gate*records*3u+i]=select(0u,mgCycleDispatch[i],enabled);
    }
  }
}

@compute @workgroup_size(1)
fn mgCheckCycleConvergence(){
  if(mgSkipCycle()){return;}
  let candidate=atomicLoad(&mgState.convergence[15]);
  if(mg.control.z==0u){
    atomicStore(&mgState.convergence[19],candidate);
    atomicStore(&mgState.convergence[24],candidate);
    return;
  }
  if(mg.control.z==4u){atomicAdd(&mgState.convergence[21],${UNIFORM_CM11A_RECOVERY_SWEEPS}u);}
  else{atomicAdd(&mgState.convergence[15u+mg.control.z],1u);}
  let accepted=candidate<0x7f800000u&&candidate<=atomicLoad(&mgState.convergence[19]);
  atomicStore(&mgState.convergence[23],select(1u,0u,accepted));
  if(accepted){atomicStore(&mgState.convergence[19],candidate);}
  else if(mg.control.z!=4u){
    atomicAdd(&mgState.convergence[20],1u);
    atomicStore(&mgState.convergence[22],1u);
    return;
  }
  let threshold=select(mgTolerance.x,min(mgTolerance.x,bitcast<f32>(atomicLoad(&mgState.convergence[24]))*${UNIFORM_CM11A_RECOVERY_REDUCTION}),mg.control.z==4u);
  if(threshold>0.0&&bitcast<f32>(atomicLoad(&mgState.convergence[19]))<=threshold){
    atomicStore(&mgState.convergence[16],1u);
  }
}

@compute @workgroup_size(4,4,4)
fn mgSaveAccepted(@builtin(global_invocation_id) gid:vec3u){
  if(atomicLoad(&mgState.convergence[23])!=0u){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  textureStore(mgPressureOut,id,vec4f(mgP(id)));
}
@compute @workgroup_size(4,4,4)
fn mgRestoreRejected(@builtin(global_invocation_id) gid:vec3u){
  if(mg.control.w==0u){
    if(atomicLoad(&mgState.convergence[23])==0u){return;}
    // Smoothing may transiently worsen the infinity norm. Continue its finite
    // working iterate privately; finish always restores the best field.
    if(mg.levelDims.w==2u&&atomicLoad(&mgState.convergence[15])<0x7f800000u){return;}
  }
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  textureStore(mgPressureOut,id,textureLoad(mgResidualIn,id,0));
}
@compute @workgroup_size(1)
fn mgFinishSafety(){
  atomicStore(&mgState.convergence[25],select(0u,1u,atomicLoad(&mgState.convergence[22])!=0u&&atomicLoad(&mgState.convergence[16])==0u));
}

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
  // A window-local lattice IS the window: its dispatches cover the whole
  // capacity with a static plan and there is no origin to add.
  if(pressureWindowLattice()){return vec3i(gid);}
  // Domain-lattice mode places the level inside the whole-domain lattice, so
  // the host's lag-padded group counts overrun the level's own box. Those
  // threads exit at the packed extent the finalize published for this level; a
  // level whose extent did not fit the word clears the flag and clips nothing.
  let base=16u+10u*mg.fineDims.w;
  let packed=activeRegion[base+9u];
  if((packed&0x40000000u)!=0u&&any(gid>=vec3u(
    packed&1023u,(packed>>10u)&1023u,(packed>>20u)&1023u))){return vec3i(-1);}
  return vec3i(gid)+vec3i(vec3u(activeRegion[base],activeRegion[base+1u],activeRegion[base+2u]));
}
// The map from a coarse cell to one of its fine children. A semi-coarsened
// level has a stride of 1 on any axis that did not halve, and there both child
// offsets land on the same fine cell: the eight-tap folds above then average
// four distinct cells at weight 2/8 each, which is the correct restriction, and
// the max-folds are unaffected by seeing a cell twice. Nothing downstream has
// to know whether an axis coarsened. Multiplying o by (stride-1) is that
// collapse: it is o at stride 2 and 0 at stride 1.
fn mgFineChild(coarse:vec3i,o:vec3i)->vec3i{
  let finePhysical=vec3i(mg.levelDims.xyz)-vec3i(2);
  let coarsePhysical=max(vec3i(mg.coarseDims.xyz)-vec3i(2),vec3i(1));
  let stride=max(finePhysical/coarsePhysical,vec3i(1));
  return clamp(stride*(coarse-vec3i(1))+o*(stride-vec3i(1)),vec3i(-1),finePhysical)+vec3i(1);
}
fn mgTopology(p:vec3i)->vec4f{return textureLoad(mgVolumeIn,mgClamp(p,mg.levelDims.xyz),0);}
fn mgPhi(p:vec3i)->f32{return textureLoad(mgPhiIn,mgClamp(p,mg.levelDims.xyz),0).x;}
fn mgP(p:vec3i)->f32{return textureLoad(mgPressureIn,mgClamp(p,mg.levelDims.xyz),0).x;}
fn mgLiquid(p:vec3i)->bool{return mgValid(p,mg.levelDims.xyz)&&mgPhi(p)<0.0;}
fn mgInterior(p:vec3i,d:vec3u)->bool{return all(p>=vec3i(1))&&all(p<vec3i(d)-vec3i(1));}
fn mgOpenTopHalo(p:vec3i,d:vec3u)->bool{
  return params.boundary.w>0.5&&p.y==i32(d.y)-1&&p.x>0&&p.x<i32(d.x)-1&&p.z>0&&p.z<i32(d.z)-1;
}
// Cells the finest build takes from simulation space. Without a window that is
// the interior, and the halo is the domain boundary. With one, every cell whose
// simulation coordinate is in the domain qualifies, halo included.
fn mgSimulationCell(p:vec3i,d:vec3u,simulation:vec3i)->bool{
  return mgInterior(p,d)||(pressureWindowLattice()&&valid(simulation));
}
fn mgFaceV(id:vec3i,neighbor:vec3i,axis:u32)->f32{
  let positive=neighbor[axis]>id[axis];
  return select(mgTopology(neighbor)[axis+1u],mgTopology(id)[axis+1u],positive);
}
fn mgTheta(liquidCell:vec3i,airCell:vec3i)->f32{
  let a=mgPhi(liquidCell);let b=mgPhi(airCell);
  return cm12GhostFluidTheta(a,b,1e-9);
}
fn mgCoefficientRaw(id:vec3i,q:vec3i,axis:u32)->f32{
  if(!mgValid(q,mg.levelDims.xyz)){
    // The authored +Y opening is atmospheric air. All other exterior faces
    // are covered walls, matching the fine-grid stencil helper.
    if(axis==1u&&id.y==i32(mg.levelDims.y)-1&&q.y==i32(mg.levelDims.y)&&params.boundary.w>0.5){
      let h=mg.spacing.y;let phi=mgPhi(id);let exteriorPhi=0.5*h;
      let theta=cm12GhostFluidTheta(phi,exteriorPhi,1e-9);
      return mgTopology(id).z/(h*h*theta);
    }
    return 0.0;
  }
  let vf=mgFaceV(id,q,axis);if(vf<=1e-6){return 0.0;}
  let h=mg.spacing[axis];
  // A positive MAC face is baked once by its lower-coordinate owner, but the
  // liquid can lie on either side. The old expression only applied theta when
  // id was liquid and q was air. An air-owned face with a positive-side
  // liquid therefore used theta=1 in the solve while projection used the real
  // theta, creating a correction up to 1/theta_min too large.
  let idLiquid=mgLiquid(id);let qLiquid=mgLiquid(q);var theta=1.0;
  if(idLiquid&&!qLiquid){theta=mgTheta(id,q);}
  if(!idLiquid&&qLiquid){theta=mgTheta(q,id);}
  return vf/(h*h*theta);
}
fn mgBakedLiquid(p:vec3i)->bool{
  return mgValid(p,mg.levelDims.xyz)&&textureLoad(mgCoefficientsIn,p,0).w>0.5;
}
fn mgCoefficient(id:vec3i,q:vec3i,axis:u32)->f32{
  if(!mgValid(q,mg.levelDims.xyz)){
    if(q[axis]>id[axis]){return textureLoad(mgCoefficientsIn,id,0)[axis];}
    return 0.0;
  }
  if(q[axis]>id[axis]){return textureLoad(mgCoefficientsIn,id,0)[axis];}
  return textureLoad(mgCoefficientsIn,q,0)[axis];
}
fn mgApply(id:vec3i)->f32{
  if(!mgBakedLiquid(id)){return 0.0;}
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var terms:array<f32,6>;let centre=mgP(id);
  for(var n=0;n<6;n+=1){let q=id+e[n];let axis=u32(n/2);let a=mgCoefficient(id,q,axis);let neighbor=select(0.0,mgP(q),mgBakedLiquid(q));terms[n]=a*(centre-neighbor);}
  return mgD4Sum6(terms);
}

@compute @workgroup_size(4,4,4)
fn mgBuildFinestTopology(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  let simulation=id-vec3i(1)+pressureWindowOrigin();
  let h=mg.spacing.xyz;
  // A halo cell whose SIMULATION coordinate is still inside the domain is not
  // a boundary at all: the window keeps at least its padding of air between
  // the liquid and its own edge, so that cell is ordinary far air and is
  // built exactly as an interior cell would be -- real open fractions, real
  // face V, real phi. That is stronger than the lid rule it replaces: it
  // stays correct when a solid, a terrain column or the domain's own free
  // surface happens to sit in the window's halo. Only a halo whose simulation
  // coordinate leaves the DOMAIN is a wall, and it keeps today's behaviour.
  if(mgSimulationCell(id,mg.levelDims.xyz,simulation)){
    let topology=vec4f(cellOpenFraction(simulation),pressureFaceVolumeFraction(simulation,0u),pressureFaceVolumeFraction(simulation,1u),pressureFaceVolumeFraction(simulation,2u));
    textureStore(mgPhiOut,id,vec4f(pressurePhi(simulation)));textureStore(mgVolumeOut,id,topology);return;
  }
  let openTop=mgOpenTopHalo(id,mg.levelDims.xyz);var topology=vec4f(select(0.0,1.0,openTop));
  // Low-side halo cells own the three missing negative face-centred dual
  // cells. Their closed, grid-aligned domain halves have V=1/2.
  if(id.x==0&&id.y>0&&id.y<i32(mg.levelDims.y)-1&&id.z>0&&id.z<i32(mg.levelDims.z)-1){topology.y=pressureFaceVolumeFraction(simulation,0u);}
  if(id.y==0&&id.x>0&&id.x<i32(mg.levelDims.x)-1&&id.z>0&&id.z<i32(mg.levelDims.z)-1){topology.z=pressureFaceVolumeFraction(simulation,1u);}
  if(id.z==0&&id.x>0&&id.x<i32(mg.levelDims.x)-1&&id.y>0&&id.y<i32(mg.levelDims.y)-1){topology.w=pressureFaceVolumeFraction(simulation,2u);}
  textureStore(mgPhiOut,id,vec4f(0.5*min(h.x,min(h.y,h.z))));textureStore(mgVolumeOut,id,topology);
}

@compute @workgroup_size(4,4,4)
fn mgBuildFinestRhs(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  let simulation=id-vec3i(1)+pressureWindowOrigin();
  var rhs=0.0;var minimum=-3.402823e38;
  if(mgSimulationCell(id,mg.levelDims.xyz,simulation)){
    minimum=select(-3.402823e38,0.0,cellInsideSolid(simulation)||cellInsideTerrain(simulation));
    if(pressureLiquid(simulation)){
      let checkSolid=nearAnyBody(worldCell(simulation));rhs=params.physical.x*(divergenceAt(simulation,checkSolid)-volumeCorrectionDivergence(simulation))/params.dimsDt.w;
    }
  }else if(!mgOpenTopHalo(id,mg.levelDims.xyz)){
    minimum=0.0;
    // The solid halo is a constrained pressure row, not prescribed p=0.
    // Its incident predicted wall flux must enter b, or positive contact
    // pressure cannot cancel velocity directed into the exterior solid.
    if(geometricVolumeEnabled()&&pressurePhi(simulation)<0.0){
      rhs=params.physical.x*divergenceAt(simulation,false)/params.dimsDt.w;
    }
  }
  // The hierarchy uses A p = b with A p = sum a(p_i-p_j). The existing fine
  // projection convention therefore publishes b=-rho div(u*)/dt.
  textureStore(mgRhsOut,id,vec4f(-rhs));
  textureStore(mgPressureOut,id,vec4f(0.0));
  textureStore(mgMinimumOut,id,vec4f(minimum));
}

@compute @workgroup_size(4,4,4)
fn mgDownsampleTopology(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}
  var topologyTerms:array<vec4f,8>;var phiTerms:array<f32,8>;var positiveTerms:array<f32,8>;var positiveFlags:array<f32,8>;var negativeFlags:array<f32,8>;
  for(var corner=0u;corner<8u;corner+=1u){
    let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let q=mgFineChild(id,o);
    topologyTerms[corner]=mgTopology(q);let phi=mgPhi(q);phiTerms[corner]=phi;positiveTerms[corner]=select(0.0,phi,phi>=0.0);positiveFlags[corner]=select(0.0,1.0,phi>=0.0);negativeFlags[corner]=select(1.0,0.0,phi>=0.0);
  }
  let v=mgD4Sum8Vec4(topologyTerms);let phiSum=mgD4Sum8(phiTerms);let positiveSum=mgD4Sum8(positiveTerms);let positiveCount=mgD4Sum8(positiveFlags);let negativeCount=mgD4Sum8(negativeFlags);
  // CM11a Eq. 15-16 and C=2 sign-aware phi rule. control.x is the
  // destination level and control.y is M-C.
  let mixed=positiveCount>0.0&&negativeCount>0.0;
  let usePositive=mixed&&mg.control.x>=mg.control.y;
  let coarsePhi=select(phiSum/8.0,positiveSum/max(positiveCount,1.0),usePositive);
  var topology=v/8.0;
  // The positive face components are overlapping dual-cell volumes. A dual
  // cell centred on a grid-aligned closed wall is half exterior at every
  // hierarchy level; averaging the adjacent interior face into it would make
  // the wall spuriously approach V=1 with each coarsening step.
  textureStore(mgVolumeOut,id,topology);textureStore(mgPhiOut,id,vec4f(coarsePhi));
}

@compute @workgroup_size(4,4,4)
fn mgResidual(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  // CM11a defines b only on pressure unknowns. Air rows have no diagonal in
  // A, so carrying their velocity divergence as b-Ap would inject arbitrary
  // forcing into restriction and eventually the coarsest solve.
  let residual=select(0.0,textureLoad(mgRhsIn,id,0).x-mgApply(id),mgBakedLiquid(id));
  textureStore(mgResidualOut,id,vec4f(residual));
}

// Cell-centred trilinear restriction: a coarse centre lies halfway between
// each of its eight fine centres.  CM11a uses the same trilinear operator for
// restriction and prolongation.
@compute @workgroup_size(4,4,4)
fn mgRestrictResidual(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}var terms:array<f32,8>;
  for(var corner=0u;corner<8u;corner+=1u){let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));terms[corner]=textureLoad(mgResidualIn,mgFineChild(id,o),0).x;}
  textureStore(mgRhsOut,id,vec4f(mgD4Sum8(terms)/8.0));
}

fn mgTrilinearPressure(fineId:vec3i)->f32{
  // Prolongation reads the coarse level (levelDims) into the fine one
  // (coarseDims), so the per-axis scale is the inverse of mgFineChild's
  // stride. An axis that did not coarsen scales by 1, which lands q on an
  // exact integer: fract is then 0 and that axis's far weight collapses to
  // zero, degenerating the trilinear tap into the bilinear one it should be.
  let finePhysical=vec3f(vec3i(mg.coarseDims.xyz)-vec3i(2));
  let coarsePhysical=vec3f(vec3i(mg.levelDims.xyz)-vec3i(2));
  let scale=coarsePhysical/max(finePhysical,vec3f(1.0));
  let q=(vec3f(fineId)-vec3f(0.5))*scale+vec3f(0.5);let base=vec3i(floor(q));let f=fract(q);
  var values:array<f32,8>;var weights:array<f32,8>;
  for(var corner=0u;corner<8u;corner+=1u){
    let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let p=base+o;
    // The persistent halo is implementation bookkeeping, not part of the
    // pressure grid. CM11/CM11a require out-of-grid trilinear samples to be
    // ignored and the remaining weights renormalized. Blending the halo's
    // zero pressure into a correction makes Full-Cycles amplify wall error.
    values[corner]=0.0;weights[corner]=0.0;if(!mgInterior(p,mg.levelDims.xyz)){continue;}
    let w=select(1.0-f.x,f.x,o.x==1)*select(1.0-f.y,f.y,o.y==1)*select(1.0-f.z,f.z,o.z==1);
    values[corner]=w*mgP(p);weights[corner]=w;
  }
  let value=mgD4Sum8(values);let total=mgD4Sum8(weights);
  return select(0.0,value/total,total>0.0);
}

@compute @workgroup_size(4,4,4)
fn mgProlongateAdd(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}
  textureStore(mgPressureOut,id,vec4f(textureLoad(mgResidualIn,id,0).x+mgTrilinearPressure(id)));
}

@compute @workgroup_size(4,4,4)
fn mgProlongateAssign(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}
  textureStore(mgPressureOut,id,vec4f(mgTrilinearPressure(id)));
}

@compute @workgroup_size(4,4,4)
fn mgCopyPressure(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  textureStore(mgPressureOut,id,vec4f(mgP(id)));
}

@compute @workgroup_size(4,4,4)
fn mgClearPressure(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  textureStore(mgPressureOut,id,vec4f(0.0));
}

@compute @workgroup_size(4,4,4)
fn mgClearMinimum(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  textureStore(mgMinimumOut,id,vec4f(-3.402823e38));
}

@compute @workgroup_size(4,4,4)
fn mgShiftMinimum(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  textureStore(mgMinimumOut,id,vec4f(textureLoad(mgMinimumIn,id,0).x-mgP(id)));
}

@compute @workgroup_size(4,4,4)
fn mgAddPressure(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  textureStore(mgPressureOut,id,vec4f(mgP(id)+textureLoad(mgResidualIn,id,0).x));
}

// The paper requires one layer of phi in solid cells on every level. The
// authoritative finest continuation is pressurePhi; coarse levels continue
// that field from their face-adjacent non-solid cells for exactly one pass.
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

@compute @workgroup_size(4,4,4)
fn mgBakeCoefficients(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  let coefficients=vec3f(
    mgCoefficientRaw(id,id+vec3i(1,0,0),0u),
    mgCoefficientRaw(id,id+vec3i(0,1,0),1u),
    mgCoefficientRaw(id,id+vec3i(0,0,1),2u));
  textureStore(mgCoefficientsOut,id,vec4f(coefficients,select(0.0,1.0,mgLiquid(id))));
}

// CM11a Eq. 19-20. mgMinimumIn and mgPressureIn are the fine p_min and
// current p; mgMinimumOut is the next-coarser constraint field.
@compute @workgroup_size(4,4,4)
fn mgDownsampleSubtract(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}var lower=-3.402823e38;
  for(var corner=0u;corner<8u;corner+=1u){let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let q=mgFineChild(id,o);lower=max(lower,textureLoad(mgMinimumIn,q,0).x-mgP(q));}
  textureStore(mgMinimumOut,id,vec4f(lower));
}

@compute @workgroup_size(4,4,4)
fn mgDownsampleMinimum(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}var lower=-3.402823e38;
  for(var corner=0u;corner<8u;corner+=1u){let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let q=mgFineChild(id,o);lower=max(lower,textureLoad(mgMinimumIn,q,0).x);}
  textureStore(mgMinimumOut,id,vec4f(lower));
}

@compute @workgroup_size(4,4,4)
fn mgSmoothColour(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  let old=mgP(id);let coarseDone=(mg.control.w&2u)!=0u&&atomicLoad(&mgState.convergence[1])!=0u;
  // Pass-through cells carry the CM11a Eq. 18 projection with them. Nothing
  // reads a wrong-colour or non-liquid cell between the two colour passes
  // (neighbour sums are mgLiquid-gated and an update never reads its own old
  // value), so projecting here leaves every sweep-exit value identical to the
  // former trailing mgProjectMinimum pass while deleting that pass outright.
  let colour=u32((id.x+id.y+select(id.z,0,depthSymmetry()))&1);
  if(coarseDone||!mgBakedLiquid(id)||colour!=mg.control.z){textureStore(mgPressureOut,id,vec4f(max(old,textureLoad(mgMinimumIn,id,0).x)));return;}
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var diagonalTerms:array<f32,6>;var sumTerms:array<f32,6>;
  for(var n=0;n<6;n+=1){let q=id+e[n];let a=mgCoefficient(id,q,u32(n/2));diagonalTerms[n]=a;sumTerms[n]=select(0.0,a*mgP(q),mgBakedLiquid(q));}
  let diagonal=mgD4Sum6(diagonalTerms);let sum=mgD4Sum6(sumTerms);
  let p=select(0.0,(sum+textureLoad(mgRhsIn,id,0).x)/diagonal,diagonal>0.0);
  // CM11a Eq. 18 says that p_min is enforced while smoothing. Project the
  // newly updated colour before the opposite colour consumes it.
  textureStore(mgPressureOut,id,vec4f(max(p,textureLoad(mgMinimumIn,id,0).x)));
}

${uniformCoarseSolverWGSL}

@compute @workgroup_size(4,4,4)
fn mgMeasureFineResidual(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  // Air values also participate in prolongation: never accept NaN/Inf there.
  let pressure=mgP(id);
  if((bitcast<u32>(pressure)&0x7f800000u)==0x7f800000u){
    if(mg.control.z==1u){atomicMax(&mgState.convergence[15],0x7f800000u);}
    else{atomicMax(&mgState.convergence[10],0x7f800000u);}return;
  }
  if(!mgBakedLiquid(id)){return;}
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var diagonalTerms:array<f32,6>;for(var n=0;n<6;n+=1){diagonalTerms[n]=mgCoefficient(id,id+e[n],u32(n/2));}let diagonal=mgD4Sum6(diagonalTerms);
  if(diagonal<=0.0){return;}let residual=textureLoad(mgRhsIn,id,0).x-mgApply(id);let minimum=textureLoad(mgMinimumIn,id,0).x;let gap=max(0.0,pressure-minimum);
  let projectsToMinimum=residual<0.0&&-residual>=gap*diagonal;let projected=max(select(abs(residual),gap*diagonal,projectsToMinimum),max(0.0,minimum-pressure)*diagonal);
  if(mg.control.z==1u){
    let norm=projected*params.dimsDt.w/params.physical.x;
    // Non-finite values must never be mistaken for convergence.
    atomicMax(&mgState.convergence[15u+0],select(0x7f800000u,bitcast<u32>(norm),norm>=0.0&&(bitcast<u32>(norm)&0x7f800000u)!=0x7f800000u&&(bitcast<u32>(residual)&0x7f800000u)!=0x7f800000u));
    return;
  }
  atomicMax(&mgState.convergence[10],bitcast<u32>(projected*params.dimsDt.w/params.physical.x));atomicMax(&mgState.convergence[11],bitcast<u32>(projected/diagonal));
}
`;
