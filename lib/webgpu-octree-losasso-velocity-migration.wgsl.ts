/** Exact geometric migration of lagged wet-face velocities across topology ids. */
export const octreeLosassoVelocityMigrationWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;const LIMBS:u32=36u;const FACE_SEPARATED:u32=0x20000000u;
struct Params{dimensions:vec3u,maximumLeafSize:u32,faceCapacity:u32,directoryCapacity:u32,reserved0:u32,reserved1:u32}
struct Face{negativeRow:u32,positiveRow:u32,axis:u32,reserved:u32,area:f32,inverseDistance:f32,openFraction:f32,normalVelocity:f32}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read_write>oldControl:array<u32>;
@group(0)@binding(2)var<storage,read_write>oldGeometry:array<vec4u>;
@group(0)@binding(3)var<storage,read>sourceFaces:array<Face>;
@group(0)@binding(4)var<storage,read_write>oldDirectory:array<vec2u>;
@group(0)@binding(5)var<storage,read_write>oldVelocity:array<f32>;
@group(0)@binding(6)var<storage,read>newControl:array<u32>;
@group(0)@binding(7)var<storage,read>newGeometry:array<vec4u>;
@group(0)@binding(8)var<storage,read_write>newFaces:array<Face>;
@group(0)@binding(9)var<storage,read_write>newVelocity:array<f32>;
@group(0)@binding(10)var<storage,read>newDirectory:array<vec2u>;
@group(0)@binding(11)var<storage,read>sourceVelocity:array<f32>;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn hashFace(packed:u32,q:vec3u)->u32{var value=(packed+1u)*0x9e3779b1u;
 value=(value^q.x)*0x85ebca6bu;value=(value^q.y)*0xc2b2ae35u;return(value^q.z)*0x27d4eb2du;}
fn exactOld(packed:u32,q:vec3u)->u32{let capacity=oldControl[6];
 if(capacity==0u||capacity>p.directoryCapacity||(capacity&(capacity-1u))!=0u){return INVALID;}
 let hash=hashFace(packed,q);let mask=capacity-1u;for(var probe=0u;probe<32u;probe+=1u){
  let record=oldDirectory[(hash+probe)&mask];if(record.x==0u){return INVALID;}let face=record.x-1u;
  if(record.y==hash&&face<oldControl[2]&&face<arrayLength(&oldGeometry)){
   let geometry=oldGeometry[face];if((geometry.x&~FACE_SEPARATED)==packed&&all(geometry.yzw==q)){return face;}
  }}return INVALID;}
fn containingOld(axis:u32,q:vec3u)->u32{var span=1u;var logSpan=0u;loop{var origin=q;
 for(var component=0u;component<3u;component+=1u){if(component!=axis){origin[component]=(q[component]/span)*span;}}
 let face=exactOld(axis|(logSpan<<2u),origin);if(face!=INVALID){return face;}
 if(span>=p.maximumLeafSize){break;}span*=2u;logSpan+=1u;}return INVALID;}
struct OldSample{value:f32,valid:bool,separated:bool}
// Refinement creates faces on planes that were interior to the old parent and
// therefore have no old face id. Reconstruct them from that parent's bounding
// faces instead of manufacturing zero momentum. Per-subface sampling preserves
// the area weighting when either bound is split by a 2:1 transition.
fn coarsenedOld(axis:u32,q:vec3u)->OldSample{var span=2u;loop{
 let low=(q[axis]/span)*span;let high=low+span;
 if(q[axis]>low&&high<=p.dimensions[axis]){var lowQ=q;var highQ=q;lowQ[axis]=low;highQ[axis]=high;
  let lowFace=containingOld(axis,lowQ);let highFace=containingOld(axis,highQ);
  if(lowFace!=INVALID&&highFace!=INVALID&&lowFace<arrayLength(&oldVelocity)&&highFace<arrayLength(&oldVelocity)){
   let a=oldVelocity[lowFace];let b=oldVelocity[highFace];if(finite(a)&&finite(b)){
    let value=mix(a,b,f32(q[axis]-low)/f32(span));
    let separated=(lowFace<arrayLength(&oldGeometry)&&(oldGeometry[lowFace].x&FACE_SEPARATED)!=0u)
     ||(highFace<arrayLength(&oldGeometry)&&(oldGeometry[highFace].x&FACE_SEPARATED)!=0u);
    if(finite(value)){return OldSample(value,true,separated);}
   }
  }
 }
 if(span>=p.maximumLeafSize){break;}span*=2u;}return OldSample(0.,false,false);}
fn floorDiv256(value:i32)->vec2i{let carry=value>>8;return vec2i(carry,value-carry*256);}
fn addExact(limbs:ptr<function,array<i32,36>>,value:f32){let bits=bitcast<u32>(value);let magnitude=bits&0x7fffffffu;
 if(magnitude==0u){return;}let exponent=(magnitude>>23u)&0xffu;let fraction=magnitude&0x7fffffu;
 let significand=select(fraction,0x800000u|fraction,exponent!=0u);let shift=select(3u,exponent+2u,exponent!=0u);
 let first=shift>>3u;let shifted=significand<<(shift&7u);let sign=select(1,-1,(bits&0x80000000u)!=0u);
 for(var digit=0u;digit<4u;digit+=1u){let limb=first+digit;let byte=i32((shifted>>(digit*8u))&0xffu);
  if(byte!=0&&limb<LIMBS){(*limbs)[limb]+=sign*byte;}}}
fn exactValue(input:ptr<function,array<i32,36>>)->f32{for(var limb=0u;limb+1u<LIMBS;limb+=1u){
 let normalized=floorDiv256((*input)[limb]);(*input)[limb]=normalized.y;(*input)[limb+1u]+=normalized.x;}
 let negative=(*input)[LIMBS-1u]<0;if(negative){for(var limb=0u;limb<LIMBS;limb+=1u){(*input)[limb]=-(*input)[limb];}
  for(var limb=0u;limb+1u<LIMBS;limb+=1u){let normalized=floorDiv256((*input)[limb]);
   (*input)[limb]=normalized.y;(*input)[limb+1u]+=normalized.x;}}
 var magnitude=0.;for(var limb=0u;limb<LIMBS;limb+=1u){magnitude+=ldexp(f32((*input)[limb]),-152+i32(8u*limb));}
 return select(magnitude,-magnitude,negative);}
fn tangentA(axis:u32)->vec3u{return select(vec3u(1,0,0),vec3u(0,1,0),axis==0u);}
fn tangentB(axis:u32)->vec3u{return select(vec3u(0,0,1),vec3u(0,1,0),axis==2u);}
@compute @workgroup_size(64)
fn snapshotLosassoFaceState(@builtin(global_invocation_id)invocation:vec3u){let face=invocation.x;
 if(face<min(8u,min(arrayLength(&oldControl),arrayLength(&newControl)))){oldControl[face]=newControl[face];}
 if(face>=newControl[2]||face>=arrayLength(&oldGeometry)||face>=arrayLength(&newGeometry)
  ||face>=arrayLength(&sourceFaces)){return;}
 oldGeometry[face]=newGeometry[face];
 oldGeometry[face].x=(oldGeometry[face].x&~FACE_SEPARATED)|(sourceFaces[face].reserved&FACE_SEPARATED);}
@compute @workgroup_size(64)
fn snapshotLosassoFaceLookup(@builtin(global_invocation_id)invocation:vec3u){let item=invocation.x;
 if(item<min(oldControl[6],min(arrayLength(&oldDirectory),arrayLength(&newDirectory)))){
  oldDirectory[item]=newDirectory[item];}
 if(item<oldControl[2]&&item<arrayLength(&oldVelocity)&&item<arrayLength(&sourceVelocity)){
  oldVelocity[item]=sourceVelocity[item];}}
@compute @workgroup_size(64)
fn migrateLosassoLaggedVelocity(@builtin(global_invocation_id)invocation:vec3u){let face=invocation.x;
 if(face>=p.faceCapacity||face>=newControl[2]||face>=arrayLength(&newVelocity)){return;}
 // A rejected/absent old publication must not erase the target bank. Its last
 // complete value is a safer carry than creating a whole epoch at zero.
 if(arrayLength(&oldControl)<4u||oldControl[3]!=1u){return;}
 let record=newGeometry[face];let axis=record.x&3u;let span=1u<<(record.x>>2u);let origin=record.yzw;
 var exact:array<i32,36>;var count=0u;var separated=false;for(var b=0u;b<span;b+=1u){for(var a=0u;a<span;a+=1u){
  let q=origin+tangentA(axis)*vec3u(a)+tangentB(axis)*vec3u(b);let old=containingOld(axis,q);
  if(old!=INVALID){
   if(old<arrayLength(&oldVelocity)){let value=oldVelocity[old];if(finite(value)){addExact(&exact,value);count+=1u;}}
   if(old<arrayLength(&oldGeometry)&&(oldGeometry[old].x&FACE_SEPARATED)!=0u){separated=true;}
  }else{let reconstructed=coarsenedOld(axis,q);if(reconstructed.valid){addExact(&exact,reconstructed.value);count+=1u;
    separated=separated||reconstructed.separated;}}
 }}if(count>0u){newVelocity[face]=exactValue(&exact)/f32(count);}
 if(face<arrayLength(&newFaces)&&separated){newFaces[face].reserved|=FACE_SEPARATED;}}
`;
