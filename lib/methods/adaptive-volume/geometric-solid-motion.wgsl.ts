/** Discrete geometric conservation baseline using the existing sampled rigid
 * coverage and static Q8 geometry. This does not claim exact solid intersections. */
export interface GeometricSolidMotionLayout {
  readonly oldCapacityFloats: number;
  readonly newCapacityFloats: number;
  readonly oldRowOpenFloats: number;
  readonly oldRowPressureOpenFloats: number;
  readonly controlBaseWords: number;
}
export function createGeometricSolidMotionWGSL(layout?: GeometricSolidMotionLayout): string {
  if (!layout) return /* wgsl */ `
fn geometricSolidMotionActive()->bool{return false;}
fn geometricSolidClosingWetCell(cell:u32)->bool{return false;}
fn geometricSolidCapacityAt(cell:u32,fraction:f32)->f32{return cellVolume(cell)*physicalCellOpenFraction(cell);}
fn geometricSolidCurrentCapacity(cell:u32)->f32{return cellVolume(cell)*physicalCellOpenFraction(cell);}
fn geometricSolidRowOpen(row:u32)->f32{return physicalRowOpenFraction(row);}
fn geometricSolidRowPressureOpen(row:u32)->f32{return physicalRowPressureOpenFraction(row);}
fn geometricSolidCapacityRate(cell:u32)->f32{return 0.0;}
fn geometricSolidSetTransportFraction(fraction:f32){}
fn geometricSolidCommitFinal(){}
`;
  for (const [name, value] of Object.entries(layout)) {
    if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`Invalid solid motion ${name}`);
  }
  return /* wgsl */ `
const GSM_OLD_C:u32=${layout.oldCapacityFloats}u;
const GSM_NEW_C:u32=${layout.newCapacityFloats}u;
const GSM_OLD_A:u32=${layout.oldRowOpenFloats}u;
const GSM_OLD_P:u32=${layout.oldRowPressureOpenFloats}u;
const GSM_CONTROL:u32=${layout.controlBaseWords}u;
fn gsmActive()->bool{return atomicLoad(&conditioning[GSM_CONTROL])!=0;}
fn geometricSolidMotionActive()->bool{return gsmActive();}
fn geometricSolidClosingWetCell(cell:u32)->bool{
 return gsmActive()&&state[GSM_NEW_C+cell]<state[GSM_OLD_C+cell]
   &&state[destinationDensity()+cell]>0.0;
}
fn geometricSolidCapacityAt(cell:u32,fraction:f32)->f32{
 if(!gsmActive()){return cellVolume(cell)*physicalCellOpenFraction(cell);}
 // Endpoint branches preserve exactly closed geometry; mix can leave a final
 // subtraction residue, which cannot authorize liquid in a zero-capacity cell.
 if(fraction<=0.0){return state[GSM_OLD_C+cell];}
 if(fraction>=1.0){return state[GSM_NEW_C+cell];}
 return mix(state[GSM_OLD_C+cell],state[GSM_NEW_C+cell],fraction);
}
fn geometricSolidCurrentCapacity(cell:u32)->f32{
 let fraction=bitcast<f32>(atomicLoad(&conditioning[GSM_CONTROL+1u]));
 return geometricSolidCapacityAt(cell,fraction);
}
fn gsmMeanRowOpen(row:u32)->f32{return 0.5*(state[GSM_OLD_A+row]+physicalRowOpenFraction(row));}
fn geometricSolidRowOpen(row:u32)->f32{
 if(!gsmActive()){return physicalRowOpenFraction(row);}return gsmMeanRowOpen(row);
}
fn geometricSolidRowPressureOpen(row:u32)->f32{
 if(!gsmActive()){return physicalRowPressureOpenFraction(row);}
 // Physical flux, pressure operator, and projected velocity must use the
 // same aperture. The copied dual-volume sample is retained as a diagnostic,
 // not a second incompatible fluid-face area.
 return gsmMeanRowOpen(row);
}
fn geometricSolidCapacityRate(cell:u32)->f32{
 if(!gsmActive()||p.frame.x<=0.0){return 0.0;}
 return (state[GSM_NEW_C+cell]-state[GSM_OLD_C+cell])/p.frame.x;
}
fn geometricSolidSetTransportFraction(fraction:f32){
 if(gsmActive()){atomicStore(&conditioning[GSM_CONTROL+1u],bitcast<i32>(fraction));}
}
fn geometricSolidCommitFinal(){
 if(!gsmActive()){return;}
 atomicStore(&conditioning[GSM_CONTROL+1u],bitcast<i32>(1.0));
 atomicStore(&conditioning[GSM_CONTROL+2u],1);
 atomicStore(&conditioning[GSM_CONTROL],0);
}
@compute @workgroup_size(1)
fn beginGeometricSolidSnapshot(){
 // Called only after the previous accepted frame's publication and after
 // preflight, with old poses voxelized on the newly accepted topology.
 atomicStore(&conditioning[GSM_CONTROL],0);
 atomicStore(&conditioning[GSM_CONTROL+1u],0);
 atomicStore(&conditioning[GSM_CONTROL+2u],0);
}
@compute @workgroup_size(64)
fn snapshotGeometricSolidCells(@builtin(global_invocation_id)gid:vec3u){
 let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
 state[GSM_OLD_C+cell]=cellVolume(cell)*physicalCellOpenFraction(cell);
}
@compute @workgroup_size(64)
fn snapshotGeometricSolidRows(@builtin(global_invocation_id)gid:vec3u){
 let row=acceptedTemplateRowInvocation(gid.x);if(row==INVALID){return;}
 state[GSM_OLD_A+row]=physicalRowOpenFraction(row);
 state[GSM_OLD_P+row]=physicalRowPressureOpenFraction(row);
}
@compute @workgroup_size(64)
fn captureGeometricSolidCells(@builtin(global_invocation_id)gid:vec3u){
 let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
 state[GSM_NEW_C+cell]=cellVolume(cell)*physicalCellOpenFraction(cell);
}
@compute @workgroup_size(1)
fn activateGeometricSolidMotion(){
 atomicStore(&conditioning[GSM_CONTROL+1u],0);
 atomicStore(&conditioning[GSM_CONTROL],1);
}
@compute @workgroup_size(64)
fn reexpressGeometricSolidRows(@builtin(global_invocation_id)gid:vec3u){
 if(atomicLoad(&conditioning[GSM_CONTROL+2u])==0){return;}
 let row=acceptedTemplateRowInvocation(gid.x);if(row==INVALID){return;}
 let oldOpen=gsmMeanRowOpen(row);let newOpen=physicalRowOpenFraction(row);
 let wall=rowSolidVelocity(row);
 // Only this frame's projected destination uses the mean aperture. FCA
 // commits that bank afterward; the historical source has a different geometry.
 let at=destinationFaceVelocity()+row;
 var fluid=0.0;
 if(oldOpen>0.0){fluid=(state[at]-(1.0-oldOpen)*wall)/oldOpen;}
 state[at]=newOpen*fluid+(1.0-newOpen)*wall;
}
@compute @workgroup_size(1)
fn finishGeometricSolidPublication(){atomicStore(&conditioning[GSM_CONTROL+2u],0);}
`;
}
