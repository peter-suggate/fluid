/** GPU constrained thin-plate reconstruction on the common presentation lattice.
 * Unknowns are fine column heights; dyadic block means are accepted authorities.
 * P removes each block mean. CG solves P L^T W L P without changing those means.
 * L is the symmetric graph Laplacian; W selects complete curvature stencils.
 */
// Temporarily disabled: retain the local monotone column reconstruction.
// One switch gates host dispatch/storage and shader consumers/dependency closure.
export const SPARSE_CM12_COMMON_HEIGHT_ENABLED = false;
export const SPARSE_CM12_HEIGHT_ITERATIONS = 64;
export const SPARSE_CM12_HEIGHT_FIELDS = 9;
export const SPARSE_CM12_HEIGHT_HEADER_FLOATS = 16;
export const SPARSE_CM12_HEIGHT_ENTRY_POINTS = [
  "initializeCM12Height", "constrainCM12Height", "laplacianCM12Height",
  "laplacianCM12HeightDirection", "applyCM12HeightLaplacian",
  "initializeCM12HeightResidual", "reduceCM12HeightInitial",
  "projectCM12HeightOperator", "reduceCM12HeightAlpha",
  "updateCM12HeightResidual", "reduceCM12HeightBeta", "updateCM12HeightDirection",
  "finishCM12Height",
] as const;

// Integration supplies heightDomain(), heightBufferBase(), heightIsEnabled(),
// heightRawReceipt(q) -> (height, native column width), and fineSamples storage.
export function sparseCM12HeightReconstructionWGSL(brickWidth: number): string {
  return /* wgsl */ `
const HEIGHT_B:u32=${brickWidth}u;
const HEIGHT_COLUMNS:u32=${brickWidth ** 2}u;
var<workgroup>heightCache:array<vec4f,${brickWidth ** 2}>;
var<workgroup>heightReduce:array<f32,64>;
var<workgroup>heightActive:u32;
fn heightCount()->u32{return heightDomain().x*heightDomain().y;}
fn heightAt(q:vec2u)->u32{return q.x+heightDomain().x*q.y;}
fn heightRead(field:u32,q:vec2u)->f32{
  return bitcast<f32>(fineSamples[heightBufferBase()+field*heightCount()+heightAt(q)]);
}
fn heightWrite(field:u32,q:vec2u,value:f32){
  fineSamples[heightBufferBase()+field*heightCount()+heightAt(q)]=bitcast<u32>(value);
}
fn heightScalar(slot:u32)->f32{
  return bitcast<f32>(fineSamples[heightBufferBase()+9u*heightCount()+slot]);
}
fn heightStamp()->u32{return fineSamples[heightBufferBase()+9u*heightCount()+6u];}
fn heightSetScalar(slot:u32,value:f32){
  fineSamples[heightBufferBase()+9u*heightCount()+slot]=bitcast<u32>(value);
}
fn heightGroups()->vec2u{return (heightDomain()+vec2u(HEIGHT_B-1u))/HEIGHT_B;}
fn heightQ(group:vec3u,local:u32)->vec2u{
  return group.xy*HEIGHT_B+vec2u(local%HEIGHT_B,local/HEIGHT_B);
}
fn heightRunning(lane:u32)->bool{
  if(lane==0u){heightActive=select(0u,1u,heightIsEnabled()&&heightScalar(4u)>0.5);}
  return workgroupUniformLoad(&heightActive)!=0u;
}
fn heightSum(lane:u32,value:f32)->f32{
  heightReduce[lane]=value;workgroupBarrier();
  for(var stride=32u;stride>0u;stride/=2u){
    if(lane<stride){heightReduce[lane]+=heightReduce[lane+stride];}workgroupBarrier();
  }
  return heightReduce[0];
}
fn heightPartial(group:vec3u,lane:u32,value:f32){
  let sum=heightSum(lane,value);
  if(lane==0u){let index=group.x+heightGroups().x*group.y;
    fineSamples[heightBufferBase()+8u*heightCount()+index]=bitcast<u32>(sum);}
}
fn heightTotal(lane:u32)->f32{
  let groups=heightGroups();var value=0.0;
  for(var i=lane;i<groups.x*groups.y;i+=64u){
    value+=bitcast<f32>(fineSamples[heightBufferBase()+8u*heightCount()+i]);
  }
  return heightSum(lane,value);
}
fn heightCacheMean(local:u32,width:u32,component:u32)->f32{
  let q=vec2u(local%HEIGHT_B,local/HEIGHT_B);
  let first=(q/width)*width;var total=0.0;
  let anchor=heightCache[first.x+HEIGHT_B*first.y][component];
  for(var z=0u;z<width;z+=1u){for(var x=0u;x<width;x+=1u){
    total+=heightCache[first.x+x+HEIGHT_B*(first.y+z)][component]-anchor;
  }}
  return anchor+total/f32(width*width);
}
@compute @workgroup_size(64)
fn initializeCM12Height(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
  if(lane==0u&&all(group==vec3u(0))){heightSetScalar(4u,select(0.0,1.0,heightIsEnabled()));}
  for(var local=lane;local<HEIGHT_COLUMNS;local+=64u){
    let q=heightQ(group,local);if(any(q>=heightDomain())){continue;}
    let previous=heightRead(1u,q);let previouslyValid=heightRead(6u,q)>0.0;
    let receipt=heightRawReceipt(q);
    heightWrite(0u,q,receipt.x);heightWrite(7u,q,receipt.y);
    heightWrite(1u,q,select(receipt.x,previous,previouslyValid&&receipt.y>0.0));
  }
}
@compute @workgroup_size(64)
fn constrainCM12Height(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
  for(var local=lane;local<HEIGHT_COLUMNS;local+=64u){
    let q=heightQ(group,local);var value=vec4f(0.0);
    if(all(q<heightDomain())){value=vec4f(heightRead(0u,q),heightRead(7u,q),heightRead(1u,q),0.0);}
    heightCache[local]=value;
  }workgroupBarrier();
  for(var local=lane;local<HEIGHT_COLUMNS;local+=64u){
    let q=heightQ(group,local);if(any(q>=heightDomain())){continue;}
    let sample=heightCache[local];var width=HEIGHT_B;
    if(sample.y<=0.0){heightWrite(6u,q,0.0);continue;}
    loop{
      let first=(q%vec2u(HEIGHT_B))/width*width;var fits=true;
      for(var z=0u;z<width;z+=1u){for(var x=0u;x<width;x+=1u){
        let donor=heightCache[first.x+x+HEIGHT_B*(first.y+z)];
        fits=fits&&donor.y>=f32(width)&&abs(donor.x-sample.x)<1e-5;
      }}
      if(fits||width==1u){break;}width/=2u;
    }
    let mean=heightCacheMean(local,width,0u);
    heightWrite(0u,q,mean);heightWrite(6u,q,f32(width));
    heightWrite(1u,q,mean+sample.z-heightCacheMean(local,width,2u));
  }
}
fn heightLaplacian(q:vec2u,field:u32)->f32{
  if(heightRead(6u,q)<=0.0){return 0.0;}
  let center=heightRead(field,q);var result=0.0;
  for(var axis=0u;axis<2u;axis+=1u){for(var side=0u;side<2u;side+=1u){
    var next=vec2i(q);next[axis]+=select(-1,1,side!=0u);
    if(any(next<vec2i(0))||any(next>=vec2i(heightDomain()))){continue;}
    if(heightRead(6u,vec2u(next))>0.0){result+=heightRead(field,vec2u(next))-center;}
  }}return result;
}
// Natural boundaries: do not impose a flat contact angle on an affine field.
// The first L is masked to complete stencils; the second is its transpose L^T.
// This keeps the projected operator symmetric and positive semidefinite.
fn heightCurvature(q:vec2u,field:u32)->f32{
  if(any(q==vec2u(0))||any(q+vec2u(1)>=heightDomain())){return 0.0;}
  if(heightRead(6u,q+vec2u(1,0))<=0.0||heightRead(6u,q-vec2u(1,0))<=0.0
    ||heightRead(6u,q+vec2u(0,1))<=0.0||heightRead(6u,q-vec2u(0,1))<=0.0){return 0.0;}
  return heightLaplacian(q,field);
}
@compute @workgroup_size(64)
fn laplacianCM12Height(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
  for(var local=lane;local<HEIGHT_COLUMNS;local+=64u){let q=heightQ(group,local);
    if(all(q<heightDomain())){heightWrite(5u,q,heightCurvature(q,1u));}}
}
@compute @workgroup_size(64)
fn laplacianCM12HeightDirection(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
  if(heightScalar(4u)<0.5){return;}
  for(var local=lane;local<HEIGHT_COLUMNS;local+=64u){let q=heightQ(group,local);
    if(all(q<heightDomain())){heightWrite(5u,q,heightCurvature(q,3u));}}
}
@compute @workgroup_size(64)
fn applyCM12HeightLaplacian(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
  if(heightScalar(4u)<0.5){return;}
  for(var local=lane;local<HEIGHT_COLUMNS;local+=64u){let q=heightQ(group,local);
    if(all(q<heightDomain())){heightWrite(4u,q,heightLaplacian(q,5u));}}
}
fn heightProject(group:vec3u,lane:u32,initial:bool){
  for(var local=lane;local<HEIGHT_COLUMNS;local+=64u){let q=heightQ(group,local);
    var value=vec4f(0.0);if(all(q<heightDomain())){value=vec4f(heightRead(4u,q),0.0,0.0,0.0);}
    heightCache[local]=value;
  }workgroupBarrier();
  var dotValue=0.0;
  for(var local=lane;local<HEIGHT_COLUMNS;local+=64u){let q=heightQ(group,local);
    if(any(q>=heightDomain())){continue;}
    let width=u32(heightRead(6u,q));var value=0.0;
    if(width>0u){value=heightCache[local].x-heightCacheMean(local,width,0u);}
    if(initial){heightWrite(2u,q,-value);heightWrite(3u,q,-value);dotValue+=value*value;
    }else{heightWrite(4u,q,value);dotValue+=value*heightRead(3u,q);}
  }
  heightPartial(group,lane,dotValue);
}
@compute @workgroup_size(64)
fn initializeCM12HeightResidual(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
  heightProject(group,lane,true);
}
@compute @workgroup_size(64)
fn projectCM12HeightOperator(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
  if(!heightRunning(lane)){return;}heightProject(group,lane,false);
}
@compute @workgroup_size(64)
fn reduceCM12HeightInitial(@builtin(local_invocation_index)lane:u32){
  let rr=heightTotal(lane);if(lane==0u){heightSetScalar(0u,rr);
    heightSetScalar(5u,max(1e-12,rr*1e-10));heightSetScalar(4u,select(0.0,1.0,rr>1e-12));}
}
@compute @workgroup_size(64)
fn reduceCM12HeightAlpha(@builtin(local_invocation_index)lane:u32){
  if(!heightRunning(lane)){return;}
  let denominator=heightTotal(lane);if(lane==0u){
    heightSetScalar(1u,select(0.0,heightScalar(0u)/max(1e-30,denominator),denominator>1e-20));
    if(denominator<=1e-20){heightSetScalar(4u,0.0);}
  }
}
@compute @workgroup_size(64)
fn updateCM12HeightResidual(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
  if(!heightRunning(lane)){return;}let alpha=heightScalar(1u);var rr=0.0;
  for(var local=lane;local<HEIGHT_COLUMNS;local+=64u){let q=heightQ(group,local);
    if(any(q>=heightDomain())){continue;}
    let residual=heightRead(2u,q)-alpha*heightRead(4u,q);
    heightWrite(1u,q,heightRead(1u,q)+alpha*heightRead(3u,q));
    heightWrite(2u,q,residual);rr+=residual*residual;
  }heightPartial(group,lane,rr);
}
@compute @workgroup_size(64)
fn reduceCM12HeightBeta(@builtin(local_invocation_index)lane:u32){
  if(!heightRunning(lane)){return;}let rr=heightTotal(lane);
  if(lane==0u){heightSetScalar(2u,rr/max(1e-30,heightScalar(0u)));heightSetScalar(0u,rr);
    if(rr<=heightScalar(5u)){heightSetScalar(4u,0.0);}}
}
@compute @workgroup_size(64)
fn updateCM12HeightDirection(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
  if(heightScalar(4u)<0.5){return;}let beta=heightScalar(2u);
  for(var local=lane;local<HEIGHT_COLUMNS;local+=64u){let q=heightQ(group,local);
    if(all(q<heightDomain())){heightWrite(3u,q,heightRead(2u,q)+beta*heightRead(3u,q));}}
}
@compute @workgroup_size(64)
fn finishCM12Height(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
  if(lane==0u&&all(group==vec3u(0))){fineSamples[heightBufferBase()+9u*heightCount()+6u]=heightGeneration();}
  for(var local=lane;local<HEIGHT_COLUMNS;local+=64u){let q=heightQ(group,local);var value=0.0;
    if(all(q<heightDomain())){value=clamp(heightRead(1u,q),0.0,heightMaximum());}
    heightCache[local]=vec4f(value,0.0,0.0,0.0);
  }workgroupBarrier();
  for(var local=lane;local<HEIGHT_COLUMNS;local+=64u){let q=heightQ(group,local);
    if(any(q>=heightDomain())){continue;}let width=u32(heightRead(6u,q));if(width==0u){continue;}
    let mean=heightCacheMean(local,width,0u);let delta=heightRead(0u,q)-mean;
    let value=heightCache[local].x;
    let capacity=select(value,heightMaximum()-value,delta>=0.0);
    let meanCapacity=select(mean,heightMaximum()-mean,delta>=0.0);
    heightWrite(1u,q,value+delta*capacity/max(1e-20,meanCapacity));
  }
}
`;
}
