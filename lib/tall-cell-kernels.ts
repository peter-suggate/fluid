import { inflowBoundaryWGSL } from "./inflow-boundary";

export const tallCellComputeShader = /* wgsl */ `
struct Params {
  dimsDt: vec4f,
  cellGravity: vec4f,
  container: vec4f,
  physical: vec4f,
  boundary: vec4f,
  tall: vec4f,
  inflowPositionRadius: vec4f,
  inflowVelocityLength: vec4f,
  inflowTiming: vec4f,
}
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var pressureIn: texture_3d<f32>;
@group(0) @binding(3) var pressureOut: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var volumeIn: texture_3d<f32>;
@group(0) @binding(5) var volumeOut: texture_storage_3d<r32float, write>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var columnBaseIn: texture_2d<f32>;
@group(0) @binding(8) var columnBaseOut: texture_storage_2d<r32float, write>;
@group(0) @binding(9) var<storage,read_write> reductions:array<atomic<u32>,32>;
struct RigidBody {
  positionShape: vec4f,
  dimensions: vec4f,
  orientation: vec4f,
  linearVelocity: vec4f,
  angularVelocity: vec4f,
}
@group(0) @binding(10) var<storage,read> rigidBodies:array<RigidBody,12>;
@group(0) @binding(11) var<storage,read_write> rigidExchange:array<atomic<i32>>;
@group(0) @binding(12) var<storage,read_write> nextColumnBases:array<u32>;
	@group(0) @binding(13) var predictedVelocityIn:texture_3d<f32>;
	@group(0) @binding(14) var reversedVelocityIn:texture_3d<f32>;
	@group(0) @binding(15) var solidFractionIn:texture_3d<f32>;

fn packedDims()->vec3i{return vec3i(textureDimensions(volumeIn));}
fn fineDims()->vec3i{let d=packedDims();return vec3i(d.x,i32(round(params.boundary.w)),d.z);}
fn inflowGridDims()->vec3i{return fineDims();}
fn regularLayers()->i32{return i32(round(params.tall.x));}
fn baseAt(x:i32,z:i32)->i32{let d=packedDims();if(x<0||x>=d.x||z<0||z>=d.z){return 0;}return i32(round(textureLoad(columnBaseIn,vec2i(x,z),0).x));}
fn validPacked(q:vec3i)->bool{let d=packedDims();return all(q>=vec3i(0))&&all(q<d);}
fn validWorld(q:vec3i)->bool{let d=fineDims();return all(q>=vec3i(0))&&all(q<d);}
fn activeSample(id:vec3i)->bool{
  if(!validPacked(id)){return false;}let base=baseAt(id.x,id.z);
  if(id.y<2){return base>0;}return base+id.y-2<fineDims().y;
}
fn sampleY(id:vec3i)->f32{
  let base=baseAt(id.x,id.z);if(id.y==0){return 0.5;}if(id.y==1){return max(0.5,f32(base)-0.5);}return f32(base+id.y-2)+0.5;
}
fn samplePoint(id:vec3i)->vec3f{return vec3f(f32(id.x)+0.5,sampleY(id),f32(id.z)+0.5);}
fn worldFromPoint(p:vec3f)->vec3f{let h=params.cellGravity.xyz;return vec3f(-0.5*params.container.x+p.x*h.x,p.y*h.y,-0.5*params.container.z+p.z*h.z);}
${inflowBoundaryWGSL}
fn finiteScalar(value:f32)->bool{return value==value&&abs(value)<=3.402823e38;}
fn storeWorldLocation(id:vec3i,slot:u32){let q=vec3i(floor(samplePoint(id)));atomicStore(&reductions[slot],u32(max(q.x,0)));atomicStore(&reductions[slot+1u],u32(max(q.y,0)));atomicStore(&reductions[slot+2u],u32(max(q.z,0)));}
fn updatePositiveMaximum(value:f32,valueSlot:u32,locationSlot:u32,id:vec3i){
  if(!finiteScalar(value)||value<0.0){atomicAdd(&reductions[20],1u);return;}
  let bits=bitcast<u32>(value);let previous=atomicMax(&reductions[valueSlot],bits);if(bits>previous){storeWorldLocation(id,locationSlot);}
}
fn updatePositiveMaximumOnly(value:f32,valueSlot:u32){if(!finiteScalar(value)||value<0.0){atomicAdd(&reductions[20],1u);return;}atomicMax(&reductions[valueSlot],bitcast<u32>(value));}

fn volumeCell(q:vec3i)->f32{
  if(!validWorld(q)){return 0.0;}let base=baseAt(q.x,q.z);
  if(q.y<base&&base>0){return textureLoad(volumeIn,vec3i(q.x,0,q.z),0).x;}
  let packedY=2+q.y-base;if(packedY<2||packedY>=packedDims().y){return 0.0;}return textureLoad(volumeIn,vec3i(q.x,packedY,q.z),0).x;
}
fn validVelocityCell(q:vec3i)->vec3f{
  let base=baseAt(q.x,q.z);
  if(q.y<base&&base>0){let t=clamp(f32(q.y)/f32(max(base-1,1)),0.0,1.0);return mix(textureLoad(velocityIn,vec3i(q.x,0,q.z),0).xyz,textureLoad(velocityIn,vec3i(q.x,1,q.z),0).xyz,t);}
  let packedY=2+q.y-base;if(packedY<2||packedY>=packedDims().y){return vec3f(0.0);}return textureLoad(velocityIn,vec3i(q.x,packedY,q.z),0).xyz;
}
fn velocityCell(q:vec3i)->vec3f{
  let d=fineDims();let c=clamp(q,vec3i(0),d-vec3i(1));var v=validVelocityCell(c);
  // Restricted tall cells use the same wall convention as the cubic solver:
  // zero the outward normal, but retain tangential motion at a free-slip wall.
  if(q.x<0||q.x>=d.x){v.x=0.0;}if(q.y<0||q.y>=d.y){v.y=0.0;}if(q.z<0||q.z>=d.z){v.z=0.0;}return v;
}
fn velocityStateCell(q:vec3i)->vec4f{if(!validWorld(q)){return vec4f(0.0);}let base=baseAt(q.x,q.z);if(q.y<base&&base>0){let t=clamp(f32(q.y)/f32(max(base-1,1)),0.0,1.0);return mix(textureLoad(velocityIn,vec3i(q.x,0,q.z),0),textureLoad(velocityIn,vec3i(q.x,1,q.z),0),t);}let packedY=2+q.y-base;if(packedY<2||packedY>=packedDims().y){return vec4f(0.0);}return textureLoad(velocityIn,vec3i(q.x,packedY,q.z),0);}
fn diffusionVelocity(q:vec3i)->vec3f{
  if(validWorld(q)){return validVelocityCell(q);}let c=clamp(q,vec3i(0),fineDims()-vec3i(1));let v=validVelocityCell(c);
  if(params.boundary.y>0.5){return -v;}return v;
}
	fn pressureCell(q:vec3i)->f32{
  if(!validWorld(q)){return 0.0;}let base=baseAt(q.x,q.z);
  if(q.y<base&&base>0){let t=clamp(f32(q.y)/f32(max(base-1,1)),0.0,1.0);return mix(textureLoad(pressureIn,vec3i(q.x,0,q.z),0).x,textureLoad(pressureIn,vec3i(q.x,1,q.z),0).x,t);}
  let packedY=2+q.y-base;if(packedY<2||packedY>=packedDims().y){return 0.0;}return textureLoad(pressureIn,vec3i(q.x,packedY,q.z),0).x;
	}
	fn solidFractionCell(q:vec3i)->f32{
	  if(!validWorld(q)){return 1.0;}let base=baseAt(q.x,q.z);
	  if(q.y<base&&base>0){let t=clamp(f32(q.y)/f32(max(base-1,1)),0.0,1.0);return mix(textureLoad(solidFractionIn,vec3i(q.x,0,q.z),0).x,textureLoad(solidFractionIn,vec3i(q.x,1,q.z),0).x,t);}
	  let packedY=2+q.y-base;if(packedY<2||packedY>=packedDims().y){return 1.0;}return textureLoad(solidFractionIn,vec3i(q.x,packedY,q.z),0).x;
	}
fn sampleVolume(p:vec3f)->f32{
  let q=clamp(p-vec3f(0.5),vec3f(0.0),vec3f(fineDims()-vec3i(1)));let b=vec3i(floor(q));let f=fract(q);
  let a=mix(mix(volumeCell(b),volumeCell(b+vec3i(1,0,0)),f.x),mix(volumeCell(b+vec3i(0,1,0)),volumeCell(b+vec3i(1,1,0)),f.x),f.y);
  let c=mix(mix(volumeCell(b+vec3i(0,0,1)),volumeCell(b+vec3i(1,0,1)),f.x),mix(volumeCell(b+vec3i(0,1,1)),volumeCell(b+vec3i(1,1,1)),f.x),f.y);return mix(a,c,f.z);
}
fn sampleVelocity(p:vec3f)->vec3f{
  let q=clamp(p-vec3f(0.5),vec3f(0.0),vec3f(fineDims()-vec3i(1)));let b=vec3i(floor(q));let f=fract(q);
  let a=mix(mix(velocityCell(b),velocityCell(b+vec3i(1,0,0)),f.x),mix(velocityCell(b+vec3i(0,1,0)),velocityCell(b+vec3i(1,1,0)),f.x),f.y);
  let c=mix(mix(velocityCell(b+vec3i(0,0,1)),velocityCell(b+vec3i(1,0,1)),f.x),mix(velocityCell(b+vec3i(0,1,1)),velocityCell(b+vec3i(1,1,1)),f.x),f.y);return mix(a,c,f.z);
}
fn samplePressure(p:vec3f)->f32{
  let q=clamp(p-vec3f(0.5),vec3f(0.0),vec3f(fineDims()-vec3i(1)));let b=vec3i(floor(q));let f=fract(q);
  let a=mix(mix(pressureCell(b),pressureCell(b+vec3i(1,0,0)),f.x),mix(pressureCell(b+vec3i(0,1,0)),pressureCell(b+vec3i(1,1,0)),f.x),f.y);
  let c=mix(mix(pressureCell(b+vec3i(0,0,1)),pressureCell(b+vec3i(1,0,1)),f.x),mix(pressureCell(b+vec3i(0,1,1)),pressureCell(b+vec3i(1,1,1)),f.x),f.y);return mix(a,c,f.z);
}

fn quaternionRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.0*(q.x*uv+uuv);}
fn quaternionInverseRotate(q:vec4f,v:vec3f)->vec3f{return quaternionRotate(vec4f(q.x,-q.yzw),v);}
	fn insideRigid(body:RigidBody,world:vec3f)->bool{
  let p=quaternionInverseRotate(body.orientation,world-body.positionShape.xyz);let d=body.dimensions.xyz;let shape=i32(round(body.positionShape.w));
  if(shape==0){return length(p)<=d.x;}if(shape==1){return all(abs(p)<=0.5*d);}if(shape==2){let cy=clamp(p.y,-0.5*d.y,0.5*d.y);return length(vec3f(p.x,p.y-cy,p.z))<=d.x;}return p.x*p.x+p.z*p.z<=d.x*d.x&&abs(p.y)<=0.5*d.y;
	}
	fn rigidVelocityAt(world:vec3f)->vec4f{
	  let bodyCount=u32(round(params.boundary.z));for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}let body=rigidBodies[bodyIndex];if(insideRigid(body,world)){let arm=world-body.positionShape.xyz;return vec4f(body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm),f32(bodyIndex+1u));}}
	  return vec4f(0.0);
	}
	fn solidVelocityCell(q:vec3i)->vec3f{return rigidVelocityAt(worldFromPoint(vec3f(vec3i(q))+vec3f(0.5))).xyz;}

fn axisOffset(axis:u32)->vec3i{return select(select(vec3i(0,0,1),vec3i(0,1,0),axis==1u),vec3i(1,0,0),axis==0u);}
fn upwind(face:f32,negative:f32,positive:f32)->f32{return face*select(positive,negative,face>=0.0);}
fn rawVolumeFlux(q:vec3i,axis:u32,dt:f32)->f32{
  if(!validWorld(q)){return 0.0;}let offset=axisOffset(axis);if(!validWorld(q+offset)){return 0.0;}let speed=0.5*(velocityCell(q)[axis]+velocityCell(q+offset)[axis]);return dt/params.cellGravity.xyz[axis]*upwind(speed,volumeCell(q),volumeCell(q+offset));
}
fn outwardFlux(q:vec3i,dt:f32)->f32{
  let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);return max(rawVolumeFlux(q,0u,dt),0.0)+max(-rawVolumeFlux(q-ex,0u,dt),0.0)+max(rawVolumeFlux(q,1u,dt),0.0)+max(-rawVolumeFlux(q-ey,1u,dt),0.0)+max(rawVolumeFlux(q,2u,dt),0.0)+max(-rawVolumeFlux(q-ez,2u,dt),0.0);
}
fn inwardFlux(q:vec3i,dt:f32)->f32{
  let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);return max(-rawVolumeFlux(q,0u,dt),0.0)+max(rawVolumeFlux(q-ex,0u,dt),0.0)+max(-rawVolumeFlux(q,1u,dt),0.0)+max(rawVolumeFlux(q-ey,1u,dt),0.0)+max(-rawVolumeFlux(q,2u,dt),0.0)+max(rawVolumeFlux(q-ez,2u,dt),0.0);
}
fn donorScale(q:vec3i,dt:f32)->f32{return min(1.0,volumeCell(q)/max(outwardFlux(q,dt),1e-9));}
fn receiverScale(q:vec3i,dt:f32)->f32{return min(1.0,max(0.0,1.0-volumeCell(q))/max(inwardFlux(q,dt),1e-9));}
fn limitedInternalFlux(q:vec3i,axis:u32,dt:f32)->f32{let n=q+axisOffset(axis);let flux=rawVolumeFlux(q,axis,dt);if(flux>=0.0){return flux*min(donorScale(q,dt),receiverScale(n,dt));}return flux*min(donorScale(n,dt),receiverScale(q,dt));}
fn limitedFlux(q:vec3i,axis:u32,dt:f32)->f32{return limitedInternalFlux(q,axis,dt);}
fn advectedVolume(q:vec3i,dt:f32)->f32{
  let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);let transported=volumeCell(q)-(limitedFlux(q,0u,dt)-limitedFlux(q-ex,0u,dt)+limitedFlux(q,1u,dt)-limitedFlux(q-ey,1u,dt)+limitedFlux(q,2u,dt)-limitedFlux(q-ez,2u,dt));return clamp(transported+inflowReceiverSource(q,dt),0.0,1.0);
}
fn integratedFluxSegment(q:vec3i,axis:u32,start:i32,count:i32,dt:f32)->f32{if(count<=0){return 0.0;}var total=0.0;for(var y=0;y<count;y+=1){total+=limitedInternalFlux(q+vec3i(0,start+y,0),axis,dt);}return total;}
fn integratedSharedTallFlux(q:vec3i,axis:u32,height:i32,dt:f32)->f32{
  if(height<=0){return 0.0;}if(height<=48){return integratedFluxSegment(q,axis,0,height,dt);}var total=integratedFluxSegment(q,axis,0,16,dt)+integratedFluxSegment(q,axis,height-16,16,dt);let width=f32(height-32)/16.0;
  for(var sample=0;sample<16;sample+=1){let y=16+i32(floor((f32(sample)+0.5)*width));total+=width*limitedInternalFlux(q+vec3i(0,y,0),axis,dt);}return total;
}
fn integratedFaceFlux(q:vec3i,axis:u32,height:i32,dt:f32)->f32{
  if(height<=0){return 0.0;}let offset=axisOffset(axis);let neighbor=q+offset;if(!validWorld(q)||!validWorld(neighbor)){return 0.0;}let split=clamp(min(baseAt(q.x,q.z),baseAt(neighbor.x,neighbor.z)),0,height);var total=integratedSharedTallFlux(q,axis,split,dt);
  // Only the bounded D-height mismatch is expanded exactly. Its opposite side
  // consists of ordinary cells, so using their identical face flux preserves
  // pairwise cancellation across the tall/regular transition.
  for(var y=split;y<height;y+=1){total+=limitedFlux(q+vec3i(0,y,0),axis,dt);}return total;
}
fn advectedTallVolume(x:i32,z:i32,dt:f32)->f32{
  let base=baseAt(x,z);if(base<=0){return 0.0;}let ex=vec3i(1,0,0);let ez=vec3i(0,0,1);let q=vec3i(x,0,z);var amount=textureLoad(volumeIn,vec3i(x,0,z),0).x*f32(base);
  amount-=integratedFaceFlux(q,0u,base,dt)-integratedFaceFlux(q-ex,0u,base,dt)+integratedFaceFlux(q,2u,base,dt)-integratedFaceFlux(q-ez,2u,base,dt);
  amount-=limitedFlux(vec3i(x,base-1,z),1u,dt);return clamp(amount/f32(base),0.0,1.0);
}
fn volumeGradient(q:vec3i)->vec3f{let h=params.cellGravity.xyz;return vec3f(volumeCell(q+vec3i(1,0,0))-volumeCell(q-vec3i(1,0,0)),volumeCell(q+vec3i(0,1,0))-volumeCell(q-vec3i(0,1,0)),volumeCell(q+vec3i(0,0,1))-volumeCell(q-vec3i(0,0,1)))/(2.0*h);}
fn interfaceNormal(q:vec3i)->vec3f{let g=volumeGradient(q);return g/max(length(g),1e-6);}
fn curvature(q:vec3i)->f32{let h=params.cellGravity.xyz;return -((interfaceNormal(q+vec3i(1,0,0)).x-interfaceNormal(q-vec3i(1,0,0)).x)/(2.0*h.x)+(interfaceNormal(q+vec3i(0,1,0)).y-interfaceNormal(q-vec3i(0,1,0)).y)/(2.0*h.y)+(interfaceNormal(q+vec3i(0,0,1)).z-interfaceNormal(q-vec3i(0,0,1)).z)/(2.0*h.z));}
fn velocityLaplacian(q:vec3i)->vec3f{let h=params.cellGravity.xyz;let c=diffusionVelocity(q);return (diffusionVelocity(q+vec3i(1,0,0))-2.0*c+diffusionVelocity(q-vec3i(1,0,0)))/(h.x*h.x)+(diffusionVelocity(q+vec3i(0,1,0))-2.0*c+diffusionVelocity(q-vec3i(0,1,0)))/(h.y*h.y)+(diffusionVelocity(q+vec3i(0,0,1))-2.0*c+diffusionVelocity(q-vec3i(0,0,1)))/(h.z*h.z);}
fn strainMagnitude(q:vec3i)->f32{let h=params.cellGravity.xyz;let dx=(diffusionVelocity(q+vec3i(1,0,0))-diffusionVelocity(q-vec3i(1,0,0)))/(2.0*h.x);let dy=(diffusionVelocity(q+vec3i(0,1,0))-diffusionVelocity(q-vec3i(0,1,0)))/(2.0*h.y);let dz=(diffusionVelocity(q+vec3i(0,0,1))-diffusionVelocity(q-vec3i(0,0,1)))/(2.0*h.z);let sxy=0.5*(dx.y+dy.x);let sxz=0.5*(dx.z+dz.x);let syz=0.5*(dy.z+dz.y);return sqrt(2.0*(dx.x*dx.x+dy.y*dy.y+dz.z*dz.z+2.0*(sxy*sxy+sxz*sxz+syz*syz)));}

fn tracedVelocity(p:vec3f,signedDt:f32)->vec3f{let h=params.cellGravity.xyz;let first=sampleVelocity(p);let midpoint=p-0.5*first*signedDt/h;return sampleVelocity(p-sampleVelocity(midpoint)*signedDt/h);}
@compute @workgroup_size(4,4,4)
fn extrapolateVelocity(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));return;}let q=vec3i(floor(samplePoint(id)));if(isInflowVelocityCell(q)){textureStore(velocityOut,id,vec4f(applyInflowVelocity(q,textureLoad(velocityIn,id,0).xyz),1.0));return;}let alpha=textureLoad(volumeIn,id,0).x;if(alpha>=0.5){textureStore(velocityOut,id,vec4f(textureLoad(velocityIn,id,0).xyz,1.0));return;}let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));var sum=vec3f(0.0);var weight=0.0;for(var index=0;index<6;index+=1){let state=velocityStateCell(q+offsets[index]);if(state.w>0.5){sum+=state.xyz;weight+=1.0;}}if(weight>0.0){textureStore(velocityOut,id,vec4f(sum/weight,1.0));}else{textureStore(velocityOut,id,vec4f(textureLoad(velocityIn,id,0).xyz,0.0));}}
@compute @workgroup_size(4,4,4)
fn predictVelocity(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));return;}textureStore(velocityOut,id,vec4f(tracedVelocity(samplePoint(id),params.dimsDt.w),0.0));}
@compute @workgroup_size(4,4,4)
fn reverseVelocity(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));return;}textureStore(velocityOut,id,vec4f(tracedVelocity(samplePoint(id),-params.dimsDt.w),0.0));}

fn boundedMacCormack(id:vec3i,p:vec3f)->vec3f{
  let h=params.cellGravity.xyz;let first=sampleVelocity(p);let midpoint=p-0.5*first*params.dimsDt.w/h;let departure=p-sampleVelocity(midpoint)*params.dimsDt.w/h;
  let sampleCoordinate=clamp(departure-vec3f(0.5),vec3f(0.0),vec3f(fineDims()-vec3i(1)));let b=vec3i(floor(sampleCoordinate));
  var lower=velocityCell(b);var upper=lower;
  for(var corner:u32=1u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let value=velocityCell(b+offset);lower=min(lower,value);upper=max(upper,value);}
  let predicted=textureLoad(predictedVelocityIn,id,0).xyz;let original=textureLoad(velocityIn,id,0).xyz;let reversed=textureLoad(reversedVelocityIn,id,0).xyz;var corrected=predicted+0.5*(original-reversed);
  for(var component:u32=0u;component<3u;component+=1u){if(corrected[component]<lower[component]||corrected[component]>upper[component]){corrected[component]=predicted[component];}}
  return corrected;
}

@compute @workgroup_size(4,4,4)
fn finishAdvection(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));textureStore(volumeOut,id,vec4f(0.0));textureStore(pressureOut,id,vec4f(0.0));return;}
	  let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let p=samplePoint(id);let q=vec3i(floor(p));var v=boundedMacCormack(id,p);var alpha=advectedVolume(q,dt);if(id.y==0){alpha=advectedTallVolume(id.x,id.z,dt);}else if(id.y==1){alpha=textureLoad(volumeIn,vec3i(id.x,0,id.z),0).x;}
  // Euler's momentum equation is integrated in the liquid domain. Air values
  // are extrapolation support for the next trace and must not accumulate a
  // separate gravity impulse; doing so feeds a falling-air mode back through
  // the collocated interface stencil.
  if(alpha>=0.5){v.y+=params.cellGravity.w*dt;}if(alpha>0.001){let nu=params.physical.y/params.physical.x;v+=dt*nu*velocityLaplacian(q);if(alpha<0.999){v+=dt*params.boundary.x/params.physical.x*curvature(q)*volumeGradient(q);}}
  v=applyInflowVelocity(q,v);
  let d=fineDims();if(q.x+1>=d.x){v.x=0.0;}if(q.y+1>=d.y){v.y=0.0;}if(q.z+1>=d.z){v.z=0.0;}
  textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(alpha));textureStore(pressureOut,id,vec4f(0.0));
}

	fn centeredFaceVelocity(q:vec3i,axis:u32,direction:i32)->f32{let offset=axisOffset(axis)*direction;let neighbor=q+offset;if(!validWorld(neighbor)){return 0.0;}if(solidFractionCell(neighbor)>0.9){return solidVelocityCell(neighbor)[axis];}return 0.5*(velocityCell(q)[axis]+velocityCell(neighbor)[axis]);}
	fn positiveFaceVelocity(q:vec3i,axis:u32)->f32{let offset=axisOffset(axis);let neighbor=q+offset;if(!validWorld(q)||!validWorld(neighbor)){return 0.0;}if(solidFractionCell(neighbor)>0.9){return solidVelocityCell(neighbor)[axis];}if(solidFractionCell(q)>0.9){return solidVelocityCell(q)[axis];}return velocityCell(q)[axis];}
	fn divergenceAt(id:vec3i)->f32{let q=vec3i(floor(samplePoint(id)));let h=params.cellGravity.xyz;if(params.physical.z<=0.5){return (centeredFaceVelocity(q,0u,1)-centeredFaceVelocity(q,0u,-1))/h.x+(centeredFaceVelocity(q,1u,1)-centeredFaceVelocity(q,1u,-1))/h.y+(centeredFaceVelocity(q,2u,1)-centeredFaceVelocity(q,2u,-1))/h.z;}return (positiveFaceVelocity(q,0u)-positiveFaceVelocity(q-vec3i(1,0,0),0u))/h.x+(positiveFaceVelocity(q,1u)-positiveFaceVelocity(q-vec3i(0,1,0),1u))/h.y+(positiveFaceVelocity(q,2u)-positiveFaceVelocity(q-vec3i(0,0,1),2u))/h.z;}
@compute @workgroup_size(4,4,4)
fn buildPressureRhs(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}let wet=activeSample(id)&&textureLoad(volumeIn,id,0).x>=0.5;let rhs=select(0.0,params.physical.x*divergenceAt(id)/params.dimsDt.w,wet);textureStore(pressureOut,id,vec4f(rhs,0.0,0.0,0.0));}
fn interfaceFraction(a:f32,b:f32)->f32{return clamp((a-0.5)/max(abs(a-b),1e-6),0.05,1.0);}
	fn pressureTerm(ownAlpha:f32,otherAlpha:f32,otherPressure:f32,solidFraction:f32,coefficient:f32)->vec2f{if(otherAlpha>=0.5){let open=1.0-clamp(solidFraction,0.0,1.0);return vec2f(coefficient*open,coefficient*open*otherPressure);}return vec2f(coefficient/interfaceFraction(ownAlpha,otherAlpha),0.0);}

@compute @workgroup_size(4,4,4)
fn jacobi(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(pressureOut,id,vec4f(0.0));return;}let ownAlpha=textureLoad(volumeIn,id,0).x;if(ownAlpha<0.5){textureStore(pressureOut,id,vec4f(0.0));return;}
  let p=samplePoint(id);let h=params.cellGravity.xyz;let d=fineDims();var stencil=vec2f(0.0);
	  let q=vec3i(floor(p));if(q.x>0){let n=q-vec3i(1,0,0);stencil+=pressureTerm(ownAlpha,volumeCell(n),pressureCell(n),solidFractionCell(n),1.0/(h.x*h.x));}if(q.x+1<d.x){let n=q+vec3i(1,0,0);stencil+=pressureTerm(ownAlpha,volumeCell(n),pressureCell(n),solidFractionCell(n),1.0/(h.x*h.x));}if(q.z>0){let n=q-vec3i(0,0,1);stencil+=pressureTerm(ownAlpha,volumeCell(n),pressureCell(n),solidFractionCell(n),1.0/(h.z*h.z));}if(q.z+1<d.z){let n=q+vec3i(0,0,1);stencil+=pressureTerm(ownAlpha,volumeCell(n),pressureCell(n),solidFractionCell(n),1.0/(h.z*h.z));}
	  if(id.y==0){let distance=max(h.y,f32(baseAt(id.x,id.z)-1)*h.y);stencil+=pressureTerm(ownAlpha,textureLoad(volumeIn,vec3i(id.x,1,id.z),0).x,textureLoad(pressureIn,vec3i(id.x,1,id.z),0).x,textureLoad(solidFractionIn,vec3i(id.x,1,id.z),0).x,1.0/(distance*h.y));}
	  else if(id.y==1){let distance=max(h.y,f32(baseAt(id.x,id.z)-1)*h.y);stencil+=pressureTerm(ownAlpha,textureLoad(volumeIn,vec3i(id.x,0,id.z),0).x,textureLoad(pressureIn,vec3i(id.x,0,id.z),0).x,textureLoad(solidFractionIn,vec3i(id.x,0,id.z),0).x,1.0/(distance*h.y));if(activeSample(vec3i(id.x,2,id.z))){stencil+=pressureTerm(ownAlpha,textureLoad(volumeIn,vec3i(id.x,2,id.z),0).x,textureLoad(pressureIn,vec3i(id.x,2,id.z),0).x,textureLoad(solidFractionIn,vec3i(id.x,2,id.z),0).x,1.0/(h.y*h.y));}}
	  else {if(id.y>2||baseAt(id.x,id.z)>=2){let below=id-vec3i(0,1,0);stencil+=pressureTerm(ownAlpha,textureLoad(volumeIn,below,0).x,textureLoad(pressureIn,below,0).x,textureLoad(solidFractionIn,below,0).x,1.0/(h.y*h.y));}if(activeSample(id+vec3i(0,1,0))){let above=id+vec3i(0,1,0);stencil+=pressureTerm(ownAlpha,textureLoad(volumeIn,above,0).x,textureLoad(pressureIn,above,0).x,textureLoad(solidFractionIn,above,0).x,1.0/(h.y*h.y));}else if(q.y+1>=d.y){stencil+=pressureTerm(ownAlpha,0.0,0.0,1.0,1.0/(h.y*h.y));}}
  let rhs=params.physical.x*divergenceAt(id)/params.dimsDt.w;let old=textureLoad(pressureIn,id,0).x;let next=(stencil.y-rhs)/max(stencil.x,1e-9);textureStore(pressureOut,id,vec4f(mix(old,next,0.8),0.0,0.0,0.0));
}

@compute @workgroup_size(4,4,4)
fn project(@builtin(global_invocation_id) gid:vec3u){
	  let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));textureStore(volumeOut,id,vec4f(0.0));return;}let q=vec3i(floor(samplePoint(id)));let h=params.cellGravity.xyz;let scale=params.dimsDt.w/params.physical.x;var v=textureLoad(velocityIn,id,0).xyz;let offsets=array<vec3i,3>(vec3i(1,0,0),vec3i(0,1,0),vec3i(0,0,1));
	  if(params.physical.z<=0.5){let ownAlpha=volumeCell(q);let ownLiquid=ownAlpha>=0.5;if(ownLiquid&&solidFractionCell(q)<=0.9){let ownPressure=pressureCell(q);for(var axis=0u;axis<3u;axis+=1u){let plus=q+offsets[axis];let minus=q-offsets[axis];let plusValid=validWorld(plus);let minusValid=validWorld(minus);var pPlus=ownPressure;var pMinus=ownPressure;if(plusValid){let alpha=volumeCell(plus);if(alpha>=0.5){let solid=solidFractionCell(plus);pPlus=solid*ownPressure+(1.0-solid)*pressureCell(plus);}else{let theta=interfaceFraction(ownAlpha,alpha);pPlus=ownPressure*(1.0-1.0/theta);}}if(minusValid){let alpha=volumeCell(minus);if(alpha>=0.5){let solid=solidFractionCell(minus);pMinus=solid*ownPressure+(1.0-solid)*pressureCell(minus);}else{let theta=interfaceFraction(ownAlpha,alpha);pMinus=ownPressure*(1.0-1.0/theta);}}let sampleSpan=select(1.0,2.0,plusValid&&minusValid);v[axis]-=scale*(pPlus-pMinus)/(sampleSpan*h[axis]);}}v=applyInflowVelocity(q,v);textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x));return;}
	  for(var axis=0u;axis<3u;axis+=1u){let plus=q+offsets[axis];if(!validWorld(plus)){v[axis]=0.0;continue;}let ownAlpha=volumeCell(q);let plusAlpha=volumeCell(plus);let ownLiquid=ownAlpha>=0.5;let plusLiquid=plusAlpha>=0.5;if(!ownLiquid&&!plusLiquid){v[axis]=0.0;continue;}var ownPressure=select(0.0,pressureCell(q),ownLiquid);var plusPressure=select(0.0,pressureCell(plus),plusLiquid);var theta=1.0;if(ownLiquid!=plusLiquid){theta=select(interfaceFraction(plusAlpha,ownAlpha),interfaceFraction(ownAlpha,plusAlpha),ownLiquid);}if(solidFractionCell(plus)>0.9){plusPressure=ownPressure;}v[axis]-=scale*(plusPressure-ownPressure)/(h[axis]*theta);}v=applyInflowVelocity(q,v);textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x));
	}

	@compute @workgroup_size(4,4,4)
	fn coupleRigid(@builtin(global_invocation_id) gid:vec3u){
	  let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));textureStore(volumeOut,id,vec4f(0.0));textureStore(pressureOut,id,vec4f(0.0));return;}
	  let q=vec3i(floor(samplePoint(id)));let alpha=textureLoad(volumeIn,id,0).x;let oldV=textureLoad(velocityIn,id,0).xyz;var v=oldV;let h=params.cellGravity.xyz;let bodyCount=u32(round(params.boundary.z));var solid=0.0;
	  // Adaptive optical-layer mode evaluates every virtual cubic cell inside a
	  // tall cell. Its contribution is mapped to the two endpoint unknowns with
	  // the paper's (1-s) / s weights; exchange is accumulated once by id.y=0.
	  if(params.physical.z>0.5&&id.y<2){
	    let height=baseAt(id.x,id.z);var weightedDelta=vec3f(0.0);var coupledWeight=0.0;var basisWeight=0.0;var solidBasis=0.0;
	    for(var y=0;y<height;y+=1){
	      let q=vec3i(id.x,y,id.z);let world=worldFromPoint(vec3f(f32(id.x)+0.5,f32(y)+0.5,f32(id.z)+0.5));var virtualSolid=0.0;var owner=12u;
	      for(var corner:u32=0u;corner<8u;corner+=1u){let offset=vec3f(select(-0.4,0.4,(corner&1u)>0u),select(-0.4,0.4,(corner&2u)>0u),select(-0.4,0.4,(corner&4u)>0u))*h;let sample=world+offset;for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}if(insideRigid(rigidBodies[bodyIndex],sample)){virtualSolid+=0.125;owner=min(owner,bodyIndex);break;}}}
	      var s=0.5;if(height>1){s=1.0-f32(y)/f32(height-1);}let endpointWeight=select(s,1.0-s,id.y==1);basisWeight+=endpointWeight;solidBasis+=endpointWeight*virtualSolid;
	      if(owner<12u&&virtualSolid>0.0){let body=rigidBodies[owner];let arm=world-body.positionShape.xyz;let solidVelocity=body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm);let virtualVelocity=velocityCell(q);let weight=endpointWeight*virtualSolid;weightedDelta+=weight*(solidVelocity-virtualVelocity);coupledWeight+=weight;
	        if(id.y==0){let solidDensity=max(body.angularVelocity.w,1.0);let reducedDensity=params.physical.x*solidDensity/(params.physical.x+solidDensity);let fluidImpulse=reducedDensity*h.x*h.y*h.z*alpha*virtualSolid*(solidVelocity-virtualVelocity);let reaction=-fluidImpulse;let torque=cross(arm,reaction);let exchangeBase=owner*8u;atomicAdd(&rigidExchange[exchangeBase],i32(round(reaction.x*1e6)));atomicAdd(&rigidExchange[exchangeBase+1u],i32(round(reaction.y*1e6)));atomicAdd(&rigidExchange[exchangeBase+2u],i32(round(reaction.z*1e6)));atomicAdd(&rigidExchange[exchangeBase+3u],i32(round(torque.x*1e6)));atomicAdd(&rigidExchange[exchangeBase+4u],i32(round(torque.y*1e6)));atomicAdd(&rigidExchange[exchangeBase+5u],i32(round(torque.z*1e6)));atomicAdd(&rigidExchange[exchangeBase+6u],i32(round(alpha*virtualSolid*65536.0)));}
	      }
	    }
	    if(coupledWeight>0.0){v+=weightedDelta/coupledWeight;}solid=solidBasis/max(basisWeight,1e-6);
	  }else{
	    let p=samplePoint(id);let world=worldFromPoint(p);var owner=12u;
	    for(var corner:u32=0u;corner<8u;corner+=1u){let offset=vec3f(select(-0.4,0.4,(corner&1u)>0u),select(-0.4,0.4,(corner&2u)>0u),select(-0.4,0.4,(corner&4u)>0u))*h;let sample=world+offset;for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}if(insideRigid(rigidBodies[bodyIndex],sample)){solid+=0.125;owner=min(owner,bodyIndex);break;}}}
	    if(owner<12u&&solid>0.0){let body=rigidBodies[owner];let arm=world-body.positionShape.xyz;let solidVelocity=body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm);v=mix(v,solidVelocity,solid);let solidDensity=max(body.angularVelocity.w,1.0);let reducedDensity=params.physical.x*solidDensity/(params.physical.x+solidDensity);let fluidImpulse=reducedDensity*h.x*h.y*h.z*alpha*solid*(solidVelocity-oldV);let reaction=-fluidImpulse;let torque=cross(arm,reaction);let exchangeBase=owner*8u;atomicAdd(&rigidExchange[exchangeBase],i32(round(reaction.x*1e6)));atomicAdd(&rigidExchange[exchangeBase+1u],i32(round(reaction.y*1e6)));atomicAdd(&rigidExchange[exchangeBase+2u],i32(round(reaction.z*1e6)));atomicAdd(&rigidExchange[exchangeBase+3u],i32(round(torque.x*1e6)));atomicAdd(&rigidExchange[exchangeBase+4u],i32(round(torque.y*1e6)));atomicAdd(&rigidExchange[exchangeBase+5u],i32(round(torque.z*1e6)));atomicAdd(&rigidExchange[exchangeBase+6u],i32(round(alpha*solid*65536.0)));}
	  }
	  // The prescribed reservoir occupies the nozzle's open channel. The
	  // display cylinder is a filled rigid primitive, so carve its inlet cells
	  // back out of the pressure mask and let the boundary velocity win there.
	  if(isInflowVelocityCell(q)){solid=0.0;v=applyInflowVelocity(q,v);}
	  textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(alpha));textureStore(pressureOut,id,vec4f(solid));
	}

@compute @workgroup_size(4,4,4)
fn reduceBeforeProjection(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!validPacked(id)||!activeSample(id)){return;}let alpha=textureLoad(volumeIn,id,0).x;let velocityValue=textureLoad(velocityIn,id,0).xyz;let speed=length(velocityValue);
  if(!finiteScalar(alpha)||!all(velocityValue==velocityValue)||any(abs(velocityValue)>vec3f(3.402823e38))){atomicAdd(&reductions[20],1u);return;}
  if(alpha>=0.5&&solidFractionCell(vec3i(floor(samplePoint(id))))<=0.9){updatePositiveMaximum(abs(divergenceAt(id)),4u,11u,id);}else if(id.y>=2){updatePositiveMaximum(speed,6u,17u,id);}
}

@compute @workgroup_size(8,8,1)
fn planRemesh(@builtin(global_invocation_id) gid:vec3u){
  let d=packedDims();if(gid.x>=u32(d.x)||gid.y>=u32(d.z)){return;}let x=i32(gid.x);let z=i32(gid.y);let oldBase=baseAt(x,z);let layers=regularLayers();let fineY=fineDims().y;var lowest=fineY;var highest=-1;var previous=select(0.0,textureLoad(volumeIn,vec3i(x,0,z),0).x,oldBase>0);
  if(oldBase>0){let bottom=textureLoad(volumeIn,vec3i(x,0,z),0).x;let wet=bottom>=0.5;let sideChange=(x>0&&wet!=(volumeCell(vec3i(x-1,0,z))>=0.5))||(x+1<d.x&&wet!=(volumeCell(vec3i(x+1,0,z))>=0.5))||(z>0&&wet!=(volumeCell(vec3i(x,0,z-1))>=0.5))||(z+1<d.z&&wet!=(volumeCell(vec3i(x,0,z+1))>=0.5));if(sideChange){lowest=0;highest=max(highest,oldBase-1);}}
  for(var packedY:i32=2;packedY<d.y;packedY+=1){let worldY=oldBase+packedY-2;if(worldY>=fineY){break;}let alpha=textureLoad(volumeIn,vec3i(x,packedY,z),0).x;let wet=alpha>=0.5;let sideChange=(x>0&&wet!=(volumeCell(vec3i(x-1,worldY,z))>=0.5))||(x+1<d.x&&wet!=(volumeCell(vec3i(x+1,worldY,z))>=0.5))||(z>0&&wet!=(volumeCell(vec3i(x,worldY,z-1))>=0.5))||(z+1<d.z&&wet!=(volumeCell(vec3i(x,worldY,z+1))>=0.5));if(wet!=(previous>=0.5)||sideChange){lowest=min(lowest,worldY);highest=max(highest,worldY);}previous=alpha;}
	if(previous>=0.5&&oldBase+layers<fineY){let bandTop=oldBase+layers-1;lowest=min(lowest,bandTop);highest=max(highest,bandTop);}
  let h=params.cellGravity.xyz;let worldX=-0.5*params.container.x+(f32(x)+0.5)*h.x;let worldZ=-0.5*params.container.z+(f32(z)+0.5)*h.z;let bodyCount=u32(round(params.boundary.z));let maxBase=max(0,fineY-layers);var bodyLower=0;var bodyUpper=maxBase;
	  for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}let body=rigidBodies[bodyIndex];let radius=body.dimensions.w;if(abs(worldX-body.positionShape.x)<=radius&&abs(worldZ-body.positionShape.z)<=radius){let bottom=i32(floor((body.positionShape.y-radius)/h.y));let top=i32(ceil((body.positionShape.y+radius)/h.y));let intersectsBand=top>=oldBase&&bottom<oldBase+layers;if(top>=0&&bottom<fineY&&intersectsBand){bodyLower=max(bodyLower,top+1-layers);bodyUpper=min(bodyUpper,bottom);}}}
	  let delta=i32(round(params.tall.w));var desired=oldBase;if(highest>=0){let surfaceLower=highest+1+i32(round(params.tall.z))-layers;let surfaceUpper=lowest+1-i32(round(params.tall.y));let lower=max(surfaceLower,bodyLower);let upper=min(surfaceUpper,bodyUpper);desired=select(lower,clamp(oldBase,lower,upper),lower<=upper);}else if(bodyLower<=bodyUpper){desired=clamp(oldBase,bodyLower,bodyUpper);}desired=clamp(desired,max(0,oldBase-delta),min(maxBase,oldBase+delta));
  var neighborLower=0;var neighborUpper=maxBase;for(var n:u32=0u;n<4u;n+=1u){let offset=select(select(vec2i(0,1),vec2i(0,-1),n==2u),select(vec2i(1,0),vec2i(-1,0),n==0u),n<2u);let q=vec2i(x,z)+offset;if(q.x<0||q.x>=d.x||q.y<0||q.y>=d.z){continue;}let b=baseAt(q.x,q.y);neighborLower=max(neighborLower,b-delta);neighborUpper=min(neighborUpper,b+delta);}if(neighborLower<=neighborUpper){desired=clamp(desired,neighborLower,neighborUpper);}nextColumnBases[u32(x+d.x*z)]=u32(max(0,desired));
}

@compute @workgroup_size(4,4,4)
fn remap(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!validPacked(id)){return;}let d=packedDims();let newBase=i32(nextColumnBases[u32(id.x+d.x*id.z)]);var p=vec3f(f32(id.x)+0.5,0.5,f32(id.z)+0.5);if(id.y==1){p.y=max(0.5,f32(newBase)-0.5);}else if(id.y>=2){p.y=f32(newBase+id.y-2)+0.5;}let isActive=select(newBase>0,newBase+id.y-2<fineDims().y,id.y>=2);var alpha=0.0;var velocity=vec3f(0.0);if(isActive){if(id.y<2){let oldBase=baseAt(id.x,id.z);var oldAmount=select(0.0,textureLoad(volumeIn,vec3i(id.x,0,id.z),0).x*f32(oldBase),oldBase>0);for(var packedY:i32=2;packedY<d.y;packedY+=1){let worldY=oldBase+packedY-2;if(worldY>=fineDims().y){break;}oldAmount+=textureLoad(volumeIn,vec3i(id.x,packedY,id.z),0).x;}var regularAmount=0.0;for(var packedY:i32=2;packedY<d.y;packedY+=1){let worldY=newBase+packedY-2;if(worldY>=fineDims().y){break;}regularAmount+=volumeCell(vec3i(id.x,worldY,id.z));}alpha=max(0.0,(oldAmount-regularAmount)/f32(max(newBase,1)));}else{alpha=volumeCell(vec3i(id.x,newBase+id.y-2,id.z));}
    if(id.y<2&&newBase>0){var sumY=0.0;var sumYY=0.0;var sumV=vec3f(0.0);var sumYV=vec3f(0.0);for(var y:i32=0;y<newBase;y+=1){let fy=f32(y);let oldV=velocityCell(vec3i(id.x,y,id.z));sumY+=fy;sumYY+=fy*fy;sumV+=oldV;sumYV+=fy*oldV;}let n=f32(newBase);let denominator=n*sumYY-sumY*sumY;var slope=vec3f(0.0);if(denominator>1e-6){slope=(n*sumYV-sumY*sumV)/denominator;}let intercept=(sumV-slope*sumY)/n;let targetY=select(0.0,f32(newBase-1),id.y==1);velocity=intercept+slope*targetY;}else{velocity=sampleVelocity(p);}
  }textureStore(velocityOut,id,vec4f(velocity,0.0));textureStore(volumeOut,id,vec4f(alpha));textureStore(pressureOut,id,vec4f(0.0));if(id.y==0){textureStore(columnBaseOut,vec2i(id.x,id.z),vec4f(f32(newBase)));}
}

@compute @workgroup_size(4,4,4)
fn reduceDiagnostics(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!validPacked(id)||!activeSample(id)){return;}let alpha=textureLoad(volumeIn,id,0).x;let velocityValue=textureLoad(velocityIn,id,0).xyz;
  if(!finiteScalar(alpha)||!all(velocityValue==velocityValue)||any(abs(velocityValue)>vec3f(3.402823e38))){atomicAdd(&reductions[20],1u);return;}
  var weight=1.0;if(id.y==0){weight=f32(baseAt(id.x,id.z));}else if(id.y==1){weight=0.0;}atomicAdd(&reductions[0],u32(clamp(alpha*weight*256.0,0.0,4294967295.0)));atomicAdd(&reductions[7],u32(clamp(alpha,0.0,1.0)*weight*256.0));atomicMax(&reductions[3],u32(baseAt(id.x,id.z)));
  if(alpha>=0.5){
    atomicMax(&reductions[1],u32(id.x+1));updatePositiveMaximum(length(velocityValue),2u,8u,id);
    if(solidFractionCell(vec3i(floor(samplePoint(id))))<=0.9){updatePositiveMaximum(abs(divergenceAt(id)),5u,14u,id);let h=params.cellGravity.xyz;let cfl=max(abs(velocityValue.x)*params.dimsDt.w/h.x,max(abs(velocityValue.y)*params.dimsDt.w/h.y,abs(velocityValue.z)*params.dimsDt.w/h.z));updatePositiveMaximumOnly(cfl,30u);if(cfl>1.0){atomicAdd(&reductions[31],1u);}}
  }
}
`;
