import type { UniformPageEdge } from "./uniform-page-layout";

/** Research pressure-like seven-point operator, NOT the production pressure
 * discretization. Missing pages are explicitly zero Dirichlet for this fixture;
 * production must distinguish absent support, free surfaces and solid boundaries.
 * Resolve page neighbors only at a seam; interior samples are arithmetic.
 */
export function uniformPageStencilWGSL(edge: UniformPageEdge): string {
  if (edge !== 16 && edge !== 32) throw new RangeError("Unsupported page edge");
  return /* wgsl */ `
const B:u32=${edge}u;
const N:u32=${edge ** 3}u;
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
@group(0) @binding(2) var<storage,read> neighbors:array<u32>;
@group(0) @binding(3) var<storage,read> slots:array<u32>;
fn neighbor(slot:u32,local:u32,face:u32)->f32 {
  let page=neighbors[6u*slot+face];
  if(page==0xffffffffu){return 0.0;}
  return input[page*N+local];
}
@compute @workgroup_size(4,4,4)
fn paged(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_id)lane:vec3u){
  let slot=slots[wg.z];
  let q=vec3u(wg.x%(B/4u),wg.x/(B/4u),wg.y)*4u+lane;
  let local=q.x+B*(q.y+B*q.z);let i=slot*N+local;
  var sum=0.0;
  if(q.x>0u){sum+=input[i-1u];}else{sum+=neighbor(slot,local+B-1u,0u);}
  if(q.x+1u<B){sum+=input[i+1u];}else{sum+=neighbor(slot,local-(B-1u),1u);}
  if(q.y>0u){sum+=input[i-B];}else{sum+=neighbor(slot,local+B*(B-1u),2u);}
  if(q.y+1u<B){sum+=input[i+B];}else{sum+=neighbor(slot,local-B*(B-1u),3u);}
  if(q.z>0u){sum+=input[i-B*B];}else{sum+=neighbor(slot,local+B*B*(B-1u),4u);}
  if(q.z+1u<B){sum+=input[i+B*B];}else{sum+=neighbor(slot,local-B*B*(B-1u),5u);}
  output[i]=6.0*input[i]-sum;
}`;
}

/** Same arithmetic, workgroup size and boundary convention as the paged probe. */
export function uniformDenseStencilWGSL(side: number): string {
  if (!Number.isSafeInteger(side) || side < 4 || side % 4 !== 0 || side > 1024) throw new RangeError("Invalid dense stencil side");
  return /* wgsl */ `
const B:u32=${side}u;
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
@compute @workgroup_size(4,4,4)
fn dense(@builtin(global_invocation_id)q:vec3u){
  let i=q.x+B*(q.y+B*q.z);var sum=0.0;
  if(q.x>0u){sum+=input[i-1u];}
  if(q.x+1u<B){sum+=input[i+1u];}
  if(q.y>0u){sum+=input[i-B];}
  if(q.y+1u<B){sum+=input[i+B];}
  if(q.z>0u){sum+=input[i-B*B];}
  if(q.z+1u<B){sum+=input[i+B*B];}
  output[i]=6.0*input[i]-sum;
}`;
}
