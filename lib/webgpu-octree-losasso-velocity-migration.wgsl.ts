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
@group(0)@binding(12)var<storage,read_write>newStatus:array<atomic<u32>>;
@group(0)@binding(13)var<storage,read_write>migrationReceipt:array<atomic<u32>>;
@group(0)@binding(14)var<storage,read>newGraphControl:array<u32>;
@group(0)@binding(15)var<storage,read>newNodeDirectory:array<vec2u>;
@group(0)@binding(16)var<storage,read>newNodalVelocity:array<vec4u>;
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
fn prepareLosassoLaggedVelocityMigration(@builtin(global_invocation_id)invocation:vec3u){let face=invocation.x;
 if(face==0u){for(var word=0u;word<8u;word+=1u){atomicStore(&migrationReceipt[word],0u);}
  if(arrayLength(&newControl)>=5u){atomicStore(&migrationReceipt[0u],newControl[0u]);atomicStore(&migrationReceipt[1u],newControl[2u]);}}
 if(face<p.faceCapacity&&face<arrayLength(&newVelocity)&&face<arrayLength(&newStatus)){
  newVelocity[face]=0.;atomicStore(&newStatus[face],0u);}}
// Nodal completion is a fixed-point handoff: completing newly introduced
// faces can seed additional graph-node components, which can in turn complete
// the remaining faces on the next sweep. Preserve face status/value authority
// while rebuilding the coverage verdict for each sweep.
@compute @workgroup_size(1)
fn prepareLosassoVelocityMigrationCoverage(){
 atomicStore(&migrationReceipt[2u],0u);atomicStore(&migrationReceipt[3u],0u);
 atomicStore(&migrationReceipt[4u],0u);atomicStore(&migrationReceipt[5u],0u);
 atomicAdd(&migrationReceipt[6u],1u);atomicStore(&migrationReceipt[7u],0u);
}
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
 if(face>=p.faceCapacity||face>=newControl[2]||face>=arrayLength(&newVelocity)||face>=arrayLength(&newStatus)){return;}
 // The prepare pass cleared every target slot. An absent old publication is
 // represented by status zero and must be reconstructed from the candidate
 // nodal authority below; stale storage is never a velocity source.
 if(arrayLength(&oldControl)<4u||oldControl[3]!=1u){return;}
 let record=newGeometry[face];let axis=record.x&3u;let span=1u<<(record.x>>2u);let origin=record.yzw;
 var exact:array<i32,36>;var count=0u;for(var b=0u;b<span;b+=1u){for(var a=0u;a<span;a+=1u){
  let q=origin+tangentA(axis)*vec3u(a)+tangentB(axis)*vec3u(b);let old=containingOld(axis,q);
  if(old!=INVALID){
   if(old<arrayLength(&oldVelocity)){let value=oldVelocity[old];if(finite(value)){addExact(&exact,value);count+=1u;}}
  }else{let reconstructed=coarsenedOld(axis,q);if(reconstructed.valid){addExact(&exact,reconstructed.value);count+=1u;
   }}
 }}if(count>0u){let value=exactValue(&exact)/f32(count);if(finite(value)){newVelocity[face]=value;
   atomicStore(&newStatus[face],1u);}}
}
// Wall-separation metadata is migrated independently so the value pass keeps
// the portable eight-storage-buffer ceiling after adding explicit coverage.
@compute @workgroup_size(64)
fn migrateLosassoFaceSeparation(@builtin(global_invocation_id)invocation:vec3u){let face=invocation.x;
 if(face>=p.faceCapacity||face>=newControl[2u]||face>=arrayLength(&newGeometry)
  ||face>=arrayLength(&newFaces)||arrayLength(&oldControl)<4u||oldControl[3u]!=1u){return;}
 let record=newGeometry[face];let axis=record.x&3u;let span=1u<<(record.x>>2u);let origin=record.yzw;
 var separated=false;for(var b=0u;b<span;b+=1u){for(var a=0u;a<span;a+=1u){
  let q=origin+tangentA(axis)*vec3u(a)+tangentB(axis)*vec3u(b);let old=containingOld(axis,q);
  if(old!=INVALID&&old<arrayLength(&oldGeometry)){separated=separated||((oldGeometry[old].x&FACE_SEPARATED)!=0u);}
  else{separated=separated||coarsenedOld(axis,q).separated;}
 }}if(separated){newFaces[face].reserved|=FACE_SEPARATED;}}

fn exactCandidateNode(item:u32)->u32{if(arrayLength(&newGraphControl)<7u||newGraphControl[0u]==0u
 ||newGraphControl[3u]!=newGraphControl[0u]||newGraphControl[4u]!=0u){return INVALID;}
 let live=min(newGraphControl[2u],u32(arrayLength(&newNodeDirectory)));var low=0u;var high=live;
 while(low<high){let middle=low+(high-low)/2u;let record=newNodeDirectory[middle];
  if(record.x<item){low=middle+1u;}else{high=middle;}}
 if(low>=live){return INVALID;}let found=newNodeDirectory[low];
 return select(INVALID,found.y,found.x==item&&found.y<newGraphControl[2u]);}
@group(0)@binding(17)var<storage,read>newConstraints:array<u32>;
fn candidateComponent(item:u32,axis:u32)->vec2f{let node=exactCandidateNode(item);
 if(node==INVALID){return vec2f(0.);}let at=2u*node;if(at>=arrayLength(&newNodalVelocity)){return vec2f(0.);}
 // A face average consumes only its normal component. Air-side corner nodes
 // may legitimately leave an unrelated tangential component outside the
 // compact extension support; requiring the whole vector here rejects an
 // otherwise complete normal reconstruction.
 let packed=newNodalVelocity[at];let bit=1u<<axis;
 if((packed.w&bit)==0u){
  let base=12u*node;if(base+11u>=arrayLength(&newConstraints)){return vec2f(0.);}
  let count=newConstraints[base+1u];let denominator=newConstraints[base+2u];
  if((count!=2u&&count!=4u)||denominator==0u){return vec2f(0.);}var constrained=0.;
  for(var i=0u;i<count;i+=1u){let master=newConstraints[base+4u+i];let masterAt=2u*master;
   if(masterAt>=arrayLength(&newNodalVelocity)){return vec2f(0.);}let source=newNodalVelocity[masterAt];
   if((source.w&bit)==0u){return vec2f(0.);}let sourceBits=select(select(source.x,source.z,axis==2u),source.y,axis==1u);
   let sourceValue=bitcast<f32>(sourceBits);if(!finite(sourceValue)){return vec2f(0.);}
   constrained+=f32(newConstraints[base+8u+i])*sourceValue/f32(denominator);}
  return vec2f(constrained,select(0.,1.,finite(constrained)));}
 let bits=select(select(packed.x,packed.z,axis==2u),packed.y,axis==1u);let value=bitcast<f32>(bits);
 return vec2f(value,select(0.,1.,finite(value)));}
// Losasso et al. reconstruct newly introduced face velocities from the
// complete nodal field and average those values at the face center. This pass
// runs only at topology publication, after exact-node handoff and causal
// extension. Covered/coarsened faces retain the geometric migration above.
@compute @workgroup_size(64)
fn completeLosassoLaggedVelocityFromNodes(@builtin(global_invocation_id)invocation:vec3u){let face=invocation.x;
 if(face>=p.faceCapacity||face>=newControl[2u]||face>=arrayLength(&newGeometry)
  ||face>=arrayLength(&newVelocity)||face>=arrayLength(&newStatus)
  ||atomicLoad(&newStatus[face])==1u){return;}
 let record=newGeometry[face];let axis=record.x&3u;let span=1u<<(record.x>>2u);let origin=record.yzw;
 if(axis>2u||span==0u){atomicOr(&migrationReceipt[4u],2u);return;}
 let nd=p.dimensions+vec3u(1u);let ta=tangentA(axis)*vec3u(span);let tb=tangentB(axis)*vec3u(span);
 let q0=origin;let q1=origin+ta;let q2=origin+tb;let q3=origin+ta+tb;
 let a=candidateComponent(q0.x+nd.x*(q0.y+nd.y*q0.z),axis);
 let b=candidateComponent(q1.x+nd.x*(q1.y+nd.y*q1.z),axis);
 let c=candidateComponent(q2.x+nd.x*(q2.y+nd.y*q2.z),axis);
 let d=candidateComponent(q3.x+nd.x*(q3.y+nd.y*q3.z),axis);
 let samples=u32(a.y)+u32(b.y)+u32(c.y)+u32(d.y);
 let finalSweep=atomicLoad(&migrationReceipt[6u])>=4u;var closure=0.;
 if(samples<4u){if(samples==0u){atomicOr(&migrationReceipt[4u],4u);return;}
  atomicAdd(&migrationReceipt[7u],1u);
  if(finalSweep){closure=newVelocity[face];if(!finite(closure)){atomicOr(&migrationReceipt[4u],1u);return;}}
 }
 // Topology refinement can create a short dependency cycle: an air-side
 // corner receives its normal component from this new face, while this face
 // is reconstructed from its corners. Use the same known-only restriction as
 // adaptive nodal stencils for the first sweep. A later nodal reconstruction
 // and completion sweep revisits status-2 faces with the expanded authority.
 var exact:array<i32,36>;if(a.y!=0.){addExact(&exact,a.x);}if(b.y!=0.){addExact(&exact,b.x);}
 if(c.y!=0.){addExact(&exact,c.x);}if(d.y!=0.){addExact(&exact,d.x);}
 if(finalSweep){for(var missing=samples;missing<4u;missing+=1u){addExact(&exact,closure);}}
 let value=exactValue(&exact)/f32(select(samples,4u,finalSweep));
 if(!finite(value)){atomicOr(&migrationReceipt[4u],1u);return;}
 newVelocity[face]=value;atomicStore(&newStatus[face],2u);
}
@compute @workgroup_size(64)
fn countLosassoVelocityMigrationCoverage(@builtin(global_invocation_id)invocation:vec3u){let face=invocation.x;
 if(face>=p.faceCapacity||face>=newControl[2u]||face>=arrayLength(&newStatus)){return;}
 let status=atomicLoad(&newStatus[face]);if(status==1u){atomicAdd(&migrationReceipt[2u],1u);}
 else if(status==2u){atomicAdd(&migrationReceipt[3u],1u);}else{atomicOr(&migrationReceipt[4u],4u);}}
@compute @workgroup_size(1)
fn finishLosassoLaggedVelocityMigration(){let generation=atomicLoad(&migrationReceipt[0u]);
 let faces=atomicLoad(&migrationReceipt[1u]);let completed=atomicLoad(&migrationReceipt[2u])
  +atomicLoad(&migrationReceipt[3u]);var errors=atomicLoad(&migrationReceipt[4u]);
 if(generation==0u||arrayLength(&newControl)<5u||newControl[0u]!=generation
  ||newControl[2u]!=faces||completed!=faces||newControl[4u]!=0u){errors|=8u;}
 atomicStore(&migrationReceipt[4u],errors);atomicStore(&migrationReceipt[5u],select(0u,generation,errors==0u));
}
`;
