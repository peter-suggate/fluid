/**
 * Binding-free geometric amount transfer, composed after geometricCutCellWGSL.
 * The caller supplies the accepted parent plane and original solid corner SDF.
 * Refinement intersects that SAME geometry with each child's physical bounds.
 * It does not resample solid SDF or replace volume with a clamped density.
 * Coarsening only sums accepted extensive amounts; accepting a parent plane
 * still requires an independent surface/solid representability decision.
 *
 * Not yet connected to the resident's CM12 density/gamma transfer buffers.
 */
export const geometricVolumeTransferWGSL = /* wgsl */ `
fn geometricRefineChildVolumes(parentWidths:vec3f,parentSolidSDF:array<f32,8>,
 parentNormal:vec3f,parentOffset:f32,childLower:vec3f,
 childUpper:vec3f)->GeometricCutVolumes{
  return geometricClippedBoxPrismVolumes(parentWidths,parentSolidSDF,
    parentNormal,parentOffset,childLower,childUpper);
}

// Complete eight-child family, with zero-volume entries for absent children.
// Pairwise accumulation uses a fixed sibling order and
// avoids a long serial sum. All values are physical volumes.
fn geometricCoarsenChildVolumes(children:array<GeometricCutVolumes,8>)->GeometricCutVolumes{
  var capacity=vec4f(0.0);var liquid=vec4f(0.0);
  for(var pair=0u;pair<4u;pair+=1u){
    capacity[pair]=children[2u*pair].capacity+children[2u*pair+1u].capacity;
    liquid[pair]=children[2u*pair].liquid+children[2u*pair+1u].liquid;
  }
  return GeometricCutVolumes((capacity.x+capacity.y)+(capacity.z+capacity.w),
    (liquid.x+liquid.y)+(liquid.z+liquid.w));
}
`;

/** Shared extensive-amount arithmetic for generation remaps. The allocation
 * follows the executing resident candidate transfer's compensated sum and
 * deterministic residual sweeps; only proposals are bounded, never accepted V. */
export const geometricAmountAllocationWGSL = /* wgsl */ `
fn transferAmountAdd(total:vec2f,value:f32)->vec2f{
 let sum=total.x+value;
 let error=select((value-sum)+total.x,(total.x-sum)+value,abs(total.x)>=abs(value));
 let tail=total.y+error;let result=sum+tail;
 return vec2f(result,(sum-result)+tail);
}
fn transferAmountTolerance(capacity:f32)->f32{return 9.5367431640625e-7*capacity;}
fn transferAmountValid(amount:f32,capacity:f32)->bool{
 let tolerance=transferAmountTolerance(capacity);
 return capacity>=0.0&&capacity<=3.402823466e38
   &&amount>=-tolerance&&amount<=capacity+tolerance;
}
`;
