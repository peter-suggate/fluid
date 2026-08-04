import { OCTREE_LOSASSO_AXIS_FACE_MAXIMUM_PROBES } from "./webgpu-octree-losasso-velocity-sampler";

/** Requires globals `p`, `coarse`, `faceGeometry`, `extendedVelocity`, `faceDirectory`. */
export const octreeLosassoVelocitySamplingWGSL = /* wgsl */ `
fn losassoFloorDiv256(value:i32)->vec2i{var carry=value/256;var digit=value-carry*256;
 if(digit<0){digit+=256;carry-=1;}return vec2i(carry,digit);}
fn losassoExactAdd(limbs:ptr<function,array<i32,36>>,value:f32){let bits=bitcast<u32>(value);let magnitude=bits&0x7fffffffu;
 if(magnitude==0u){return;}let exponent=(magnitude>>23u)&0xffu;let fraction=magnitude&0x7fffffu;
 let significand=select(fraction,0x800000u|fraction,exponent!=0u);let shift=select(3u,exponent+2u,exponent!=0u);
 let first=shift>>3u;let shifted=significand<<(shift&7u);let sign=select(1,-1,(bits&0x80000000u)!=0u);
 for(var digit=0u;digit<4u;digit+=1u){let limb=first+digit;let byte=i32((shifted>>(digit*8u))&0xffu);
  if(byte!=0&&limb<36u){(*limbs)[limb]+=sign*byte;}}}
fn losassoExactValue(input:array<i32,36>)->f32{var limbs=input;
 for(var limb=0u;limb+1u<36u;limb+=1u){let n=losassoFloorDiv256(limbs[limb]);limbs[limb]=n.y;limbs[limb+1u]+=n.x;}
 let negative=limbs[35]<0;if(negative){for(var limb=0u;limb<36u;limb+=1u){limbs[limb]=-limbs[limb];}
  for(var limb=0u;limb+1u<36u;limb+=1u){let n=losassoFloorDiv256(limbs[limb]);limbs[limb]=n.y;limbs[limb+1u]+=n.x;}}
 var magnitude=0.;for(var limb=0u;limb<36u;limb+=1u){magnitude+=ldexp(f32(limbs[limb]),-152+i32(8u*limb));}
 return select(magnitude,-magnitude,negative);}
fn losassoSymmetricWeight(w:vec3f)->f32{let lo=min(w.x,min(w.y,w.z));let hi=max(w.x,max(w.y,w.z));
 let mid=max(min(w.x,w.y),max(min(w.x,w.z),min(w.y,w.z)));return(lo*mid)*hi;}
fn losassoFaceHash(packedAxisSpan:u32,q:vec3u)->u32{
 var value=(packedAxisSpan+1u)*0x9e3779b1u;
 value=(value^q.x)*0x85ebca6bu;
 value=(value^q.y)*0xc2b2ae35u;
 return (value^q.z)*0x27d4eb2du;
}
fn losassoFace(axis:u32,q:vec3u)->u32{
 if(p.directoryCapacity==0u||(p.directoryCapacity&(p.directoryCapacity-1u))!=0u){return INVALID;}
 let mask=p.directoryCapacity-1u;var span=1u;var logSpan=0u;
 loop{
  var origin=q;for(var tangent=0u;tangent<3u;tangent+=1u){if(tangent!=axis){origin[tangent]=(q[tangent]/span)*span;}}
  let packed=axis|(logSpan<<2u);let hash=losassoFaceHash(packed,origin);
  for(var probe=0u;probe<${OCTREE_LOSASSO_AXIS_FACE_MAXIMUM_PROBES}u;probe+=1u){
   let record=faceDirectory[(hash+probe)&mask];if(record.x==0u){break;}
   let face=record.x-1u;
   if(record.y==hash&&face<coarse[2]&&face<arrayLength(&faceGeometry)
     &&all(faceGeometry[face]==vec4u(packed,origin))){return face;}
  }
  if(span>=p.maximumLeafSize){break;}span*=2u;logSpan+=1u;
 }
 return INVALID;
}
struct LosassoVelocitySample{value:vec3f,valid:u32}
fn losassoVelocityAt(position:vec3f)->LosassoVelocitySample{
 if(arrayLength(&coarse)<4u||coarse[3]!=1u||coarse[2]>arrayLength(&faceGeometry)
   ||coarse[2]>arrayLength(&extendedVelocity)){return LosassoVelocitySample(vec3f(0),0u);}
 let grid=(position-p.domainOrigin)/p.velocityCellSize;
 var velocity=vec3f(0);
 for(var axis=0u;axis<3u;axis+=1u){
  var staggered=grid-vec3f(.5);staggered[axis]=grid[axis];
  let base=vec3i(floor(staggered));let fraction=fract(staggered);var exact:array<i32,36>;
  for(var corner=0u;corner<8u;corner+=1u){
   let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
   let weights=vec3f(select(1.-fraction.x,fraction.x,(corner&1u)!=0u),
    select(1.-fraction.y,fraction.y,(corner&2u)!=0u),
    select(1.-fraction.z,fraction.z,(corner&4u)!=0u));
   let weight=losassoSymmetricWeight(weights);if(weight==0.){continue;}
   var coordinate=base+offset;var upper=p.velocityDimensions-vec3u(1);upper[axis]=p.velocityDimensions[axis];
   // Cell-centred fine samples within half a coarse cell of a wall require a
   // staggered tangential corner just outside the domain. Closed-Neumann
   // sampling extends that corner from the nearest boundary face; rejecting
   // it would discard the entire wall-contact portion of the fine band.
   for(var component=0u;component<3u;component+=1u){
    if(coordinate[component]<0){if(p.closed==0u){return LosassoVelocitySample(vec3f(0),0u);}coordinate[component]=0;}
    if(coordinate[component]>i32(upper[component])){if(p.closed==0u){return LosassoVelocitySample(vec3f(0),0u);}
     coordinate[component]=i32(upper[component]);}}
   let q=vec3u(coordinate);
   let face=losassoFace(axis,q);var value=0.;
   if(face==INVALID){let onWall=q[axis]==0u||q[axis]==p.velocityDimensions[axis];
    let openTop=axis==1u&&q.y==p.velocityDimensions.y&&p.openTop!=0u;
    if(p.closed==0u||!onWall||openTop){return LosassoVelocitySample(vec3f(0),0u);}
   }else{value=extendedVelocity[face];if(!finite(value)){return LosassoVelocitySample(vec3f(0),0u);}}
   losassoExactAdd(&exact,weight*value);
  }
  velocity[axis]=losassoExactValue(exact);
 }
 return LosassoVelocitySample(velocity,select(0u,1u,finite3(velocity)));
}
`;
