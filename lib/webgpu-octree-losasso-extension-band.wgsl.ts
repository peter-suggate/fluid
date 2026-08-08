import { fineLevelSetPackedSampleWGSL } from "./fine-levelset-packed-sample";
import { makeFineLevelSetSortedWorklistLookupWGSL } from
  "./webgpu-octree-fine-levelset-bricks";

/** Compact W=7 extension faces published from wet faces plus resident fine bricks. */
export const octreeLosassoExtensionBandWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;const MULTI_OWNER:u32=0xfffffffeu;const VALID:u32=1u;const SEED:u32=1u;const ACTIVE:u32=2u;
const ERROR_CAPACITY:u32=1u;const ERROR_DIRECTORY:u32=2u;const ERROR_WET_AUTHORITY:u32=4u;
struct Params{
 velocityDimensions:vec3u,maximumLeafSize:u32,
 band:vec4u, // compact face capacity, directory capacity, wet directory capacity, W
 velocityOriginCell:vec4f,
 brickDimensions:vec3u,brickResolution:u32,
 sampleDimensions:vec3u,samplesPerBrick:u32,
 domainOrigin:vec3f,fineCellWidth:f32,
 fineAuthority:vec4u, // page capacity, worklist capacity, header words, generation
 closedBoundaries:vec4u, // six low/high axis bits in x
}
struct FineLookupParams{brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,
 domainOrigin:vec3f,fineCellWidth:f32,pageCapacity:u32,worklistCapacity:u32,worklistHeaderWords:u32,generation:u32}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read>metadata:array<u32>;
@group(0)@binding(2)var<storage,read>worklist:array<u32>;
@group(0)@binding(3)var<storage,read>samples:array<u32>;
@group(0)@binding(4)var<storage,read>wetControl:array<u32>;
@group(0)@binding(5)var<storage,read>wetGeometry:array<vec4u>;
@group(0)@binding(7)var<storage,read>wetProjected:array<f32>;
@group(0)@binding(8)var<storage,read_write>control:array<atomic<u32>>;
@group(0)@binding(9)var<storage,read_write>metrics:array<vec4u>;
@group(0)@binding(10)var<storage,read_write>geometry:array<vec4u>;
@group(0)@binding(11)var<storage,read_write>adjacencyOffsets:array<u32>;
@group(0)@binding(12)var<storage,read_write>adjacency:array<u32>;
@group(0)@binding(13)var<storage,read_write>directory:array<atomic<u32>>;
@group(0)@binding(14)var<storage,read_write>seedVelocity:array<f32>;
@group(0)@binding(15)var<storage,read_write>wetExtended:array<f32>;
@group(0)@binding(16)var<storage,read_write>seedWetFace:array<u32>;
@group(0)@binding(17)var<storage,read_write>geometricClaims:array<atomic<u32>>;
@group(0)@binding(18)var<storage,read_write>denseWetFace:array<atomic<u32>>;
@group(0)@binding(19)var<storage,read_write>liveFaceDispatch:array<u32>;
@group(0)@binding(20)var<storage,read>coarsePhiDirectory:array<u32>;
@group(0)@binding(21)var<storage,read_write>stagedVelocityArena:array<u32>;
@group(0)@binding(24)var<storage,read_write>stagedNodalVelocityArena:array<vec4u>;
fn fineParams()->FineLookupParams{return FineLookupParams(p.brickDimensions,p.brickResolution,
 p.sampleDimensions,p.samplesPerBrick,p.domainOrigin,p.fineCellWidth,
 p.fineAuthority.x,p.fineAuthority.y,p.fineAuthority.z,p.fineAuthority.w);}
${makeFineLevelSetSortedWorklistLookupWGSL("fineParams()", "metadata", "worklist", "bandFineBrick")}
${fineLevelSetPackedSampleWGSL("samples", false, "bandFine")}
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn packBrick(q:vec3u)->u32{return q.x+p.brickDimensions.x*(q.y+p.brickDimensions.y*q.z);}
fn hashFace(packed:u32,q:vec3u)->u32{var value=(packed+1u)*0x9e3779b1u;
 value=(value^q.x)*0x85ebca6bu;value=(value^q.y)*0xc2b2ae35u;return(value^q.z)*0x27d4eb2du;}
fn geometricKey(record:vec4u)->u32{let axis=record.x&3u;let level=record.x>>2u;let q=record.yzw;let d=p.velocityDimensions;
 let countX=(d.x+1u)*d.y*d.z;let countY=d.x*(d.y+1u)*d.z;let perLevel=countX+countY+d.x*d.y*(d.z+1u);
 var dense=0u;if(axis==0u){dense=q.x+(d.x+1u)*(q.y+d.y*q.z);}
 else if(axis==1u){dense=countX+q.x+d.x*(q.y+(d.y+1u)*q.z);}
 else{dense=countX+countY+q.x+d.x*(q.y+d.y*q.z);}return level*perLevel+dense;}
fn claimGeometry(record:vec4u)->bool{let key=geometricKey(record)+1u;let hash=key*0x9e3779b1u;let mask=p.band.y-1u;
 for(var probe=0u;probe<32u;probe+=1u){let slot=(hash+probe)&mask;var claim=atomicCompareExchangeWeak(&geometricClaims[slot],0u,key);
  loop{if(claim.exchanged||claim.old_value!=0u){break;}claim=atomicCompareExchangeWeak(&geometricClaims[slot],0u,key);}
  if(claim.exchanged){return true;}if(claim.old_value==key){return false;}}
 atomicOr(&control[4],ERROR_DIRECTORY);return false;}
fn installFace(id:u32,record:vec4u){let hash=hashFace(record.x,record.yzw);let mask=p.band.y-1u;
 for(var probe=0u;probe<32u;probe+=1u){let slot=(hash+probe)&mask;
  var claim=atomicCompareExchangeWeak(&directory[2u*slot],0u,id+1u);
  loop{if(claim.exchanged||claim.old_value!=0u){break;}claim=atomicCompareExchangeWeak(&directory[2u*slot],0u,id+1u);}
  if(claim.exchanged){atomicStore(&directory[2u*slot+1u],hash);return;}
  let present=claim.old_value-1u;
  if(present<atomicLoad(&control[2])&&atomicLoad(&directory[2u*slot+1u])==hash
   &&all(geometry[present]==record)){return;}}
 atomicOr(&control[4],ERROR_DIRECTORY);}
fn appendWetFace(record:vec4u,metric:vec4u,wetFace:u32){if(!claimGeometry(record)){return;}let id=atomicAdd(&control[2],1u);
 if(id>=p.band.x){atomicOr(&control[4],ERROR_CAPACITY);return;}
 geometry[id]=record;metrics[id]=metric;seedWetFace[id]=wetFace;installFace(id,record);}
fn appendAirFace(record:vec4u,metric:vec4u){if(!claimGeometry(record)){return;}let id=atomicAdd(&control[2],1u);
 if(id>=p.band.x){atomicOr(&control[4],ERROR_CAPACITY);return;}
 geometry[id]=record;metrics[id]=metric;installFace(id,record);}

const LIMBS:u32=36u;
fn floorDiv256(value:i32)->vec2i{let carry=value>>8;return vec2i(carry,value-carry*256);}
fn exactAdd(limbs:ptr<function,array<i32,36>>,value:f32){let bits=bitcast<u32>(value);let magnitude=bits&0x7fffffffu;if(magnitude==0u){return;}
 let exponent=(magnitude>>23u)&0xffu;let fraction=magnitude&0x7fffffu;let significand=select(fraction,0x800000u|fraction,exponent!=0u);
 let shift=select(3u,exponent+2u,exponent!=0u);let first=shift>>3u;let shifted=significand<<(shift&7u);let sign=select(1,-1,(bits&0x80000000u)!=0u);
 for(var digit=0u;digit<4u;digit+=1u){let limb=first+digit;let byte=i32((shifted>>(digit*8u))&0xffu);if(byte!=0&&limb<LIMBS){(*limbs)[limb]+=sign*byte;}}}
fn exactValue(input:ptr<function,array<i32,36>>)->f32{for(var limb=0u;limb+1u<LIMBS;limb+=1u){let n=floorDiv256((*input)[limb]);(*input)[limb]=n.y;(*input)[limb+1u]+=n.x;}
 let negative=(*input)[LIMBS-1u]<0;if(negative){for(var limb=0u;limb<LIMBS;limb+=1u){(*input)[limb]=-(*input)[limb];}
  for(var limb=0u;limb+1u<LIMBS;limb+=1u){let n=floorDiv256((*input)[limb]);(*input)[limb]=n.y;(*input)[limb+1u]+=n.x;}}
 var magnitude=0.;for(var limb=0u;limb<LIMBS;limb+=1u){magnitude+=ldexp(f32((*input)[limb]),-152+i32(8u*limb));}return select(magnitude,-magnitude,negative);}
fn symmetricWeight(w:vec3f)->f32{let lo=min(w.x,min(w.y,w.z));let hi=max(w.x,max(w.y,w.z));
 let mid=max(min(w.x,w.y),max(min(w.x,w.z),min(w.y,w.z)));return(lo*mid)*hi;}
struct PhiSample{value:f32,valid:u32}
fn finePhiCells(positionCells:vec3f)->PhiSample{
 // Fine samples are cell centred, so a face centre in the outermost half cell
 // of a closed domain has no in-lattice trilinear support. Clamp the probe to
 // the sample-centre lattice: the membership decision tolerates the half-cell
 // bias, while rejecting the probe silently unpublished the wall/lid corner
 // faces and hard-failed transport as soon as a corner splash entered the band.
 let grid=clamp(positionCells*(p.velocityOriginCell.w/p.fineCellWidth)-vec3f(.5),
  vec3f(0.),vec3f(p.sampleDimensions)-vec3f(1.));
 let base=vec3i(floor(grid));let fraction=fract(grid);var exact:array<i32,36>;var exactWeight:array<i32,36>;
 for(var corner=0u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
  let w=vec3f(select(1.-fraction.x,fraction.x,(corner&1u)!=0u),select(1.-fraction.y,fraction.y,(corner&2u)!=0u),
   select(1.-fraction.z,fraction.z,(corner&4u)!=0u));let weight=symmetricWeight(w);if(weight==0.){continue;}let coordinate=base+offset;
  if(any(coordinate<vec3i(0))||any(coordinate>=vec3i(p.sampleDimensions))){continue;}let q=vec3u(coordinate);
  let brick=q/p.brickResolution;let page=bandFineBrick(packBrick(brick));if(page==INVALID){continue;}let local=q-brick*p.brickResolution;
  let index=page*p.samplesPerBrick+local.x+p.brickResolution*(local.y+p.brickResolution*local.z);
  if(index>=arrayLength(&samples)||(bandFinePackedFlags(index)&VALID)==0u){continue;}let sample=bandFinePackedPhi(index);
  if(!finite(sample)){continue;}exactAdd(&exact,weight*sample);exactAdd(&exactWeight,weight);}
 let total=exactValue(&exactWeight);if(!finite(total)||total<=0.){return PhiSample(0.,0u);}
 // This probe decides velocity-support membership; it is not a transported
 // phi sample. Sparse fine topology may legitimately end through one half of
 // its trilinear stencil while a resident transport sample still needs this
 // MAC face. Renormalizing the available valid corners gives the boundary
 // face its one-sided signed-distance estimate instead of punching a diagonal
 // hole in the W7 velocity graph. Exact accumulators retain D4 invariance.
 return PhiSample(exactValue(&exact)/total,1u);}
fn exactCompact(packed:u32,q:vec3u)->u32{let hash=hashFace(packed,q);let mask=p.band.y-1u;
 for(var probe=0u;probe<32u;probe+=1u){let slot=(hash+probe)&mask;let encoded=atomicLoad(&directory[2u*slot]);if(encoded==0u){return INVALID;}
  let face=encoded-1u;if(atomicLoad(&directory[2u*slot+1u])==hash&&face<atomicLoad(&control[2])&&all(geometry[face]==vec4u(packed,q))){return face;}}
 return INVALID;}
fn tangentA(axis:u32)->vec3u{return select(vec3u(1,0,0),vec3u(0,1,0),axis==0u);}
fn tangentB(axis:u32)->vec3u{return select(vec3u(0,0,1),vec3u(0,1,0),axis==2u);}
fn containingCompact(axis:u32,q:vec3u)->u32{var span=1u;var logSpan=0u;loop{var origin=q;
 for(var c=0u;c<3u;c+=1u){if(c!=axis){origin[c]=(q[c]/span)*span;}}let found=exactCompact(axis|(logSpan<<2u),origin);
 if(found!=INVALID){return found;}if(span>=p.maximumLeafSize){break;}span*=2u;logSpan+=1u;}return INVALID;}
// Equal-size wet seed faces on both sides identify a real coarse leaf. Its
// interior finest planes are reconstruction sites, not extension-band faces.
fn insideWetCoarseLeaf(axis:u32,q:vec3u)->bool{var span=2u;var logSpan=1u;loop{
 let low=(q[axis]/span)*span;let high=low+span;
 if(q[axis]>low&&high<=p.velocityDimensions[axis]){var lowQ=q;var highQ=q;lowQ[axis]=low;highQ[axis]=high;
  for(var tangent=0u;tangent<3u;tangent+=1u){if(tangent!=axis){
   lowQ[tangent]=(q[tangent]/span)*span;highQ[tangent]=lowQ[tangent];}}
  let packed=axis|(logSpan<<2u);let lowFace=exactCompact(packed,lowQ);let highFace=exactCompact(packed,highQ);
  if(lowFace!=INVALID&&highFace!=INVALID&&(metrics[lowFace].y&SEED)!=0u&&(metrics[highFace].y&SEED)!=0u
   &&(geometry[lowFace].x>>2u)==logSpan&&(geometry[highFace].x>>2u)==logSpan){return true;}}
 if(span>=p.maximumLeafSize){break;}span*=2u;logSpan+=1u;}return false;}
fn linearInvocation(group:vec3u,lane:u32)->u32{return 64u*(group.x+65535u*group.y)+lane;}

const STAGED_INVALID:u32=0x7fc00000u;const STAGED_BOUNDARY_ZERO:u32=0x7fc00001u;
fn stagedMacCount()->u32{let d=p.velocityDimensions;
 return(d.x+1u)*d.y*d.z+d.x*(d.y+1u)*d.z+d.x*d.y*(d.z+1u);}
fn stagedOwnerOffset()->u32{return 2u*stagedMacCount();}
fn stagedCellIndex(q:vec3u)->u32{return q.x+p.velocityDimensions.x*(q.y+p.velocityDimensions.y*q.z);}
fn stagedOwnerRow(q:vec3u)->u32{let at=stagedOwnerOffset()+stagedCellIndex(q);
 if(at>=arrayLength(&stagedVelocityArena)){return INVALID;}let encoded=stagedVelocityArena[at];
 return select(INVALID,encoded-1u,encoded!=0u);}
// Span-1 air records may also shadow either endpoint of a coarse bracket. Walk
// the same dyadic covering hierarchy as containingCompact, but ignore a record
// whose extension remained starved so reconstruction can reach the real wet
// face behind it.
fn containingFiniteVelocity(axis:u32,q:vec3u)->u32{var span=1u;var logSpan=0u;loop{var origin=q;
 for(var c=0u;c<3u;c+=1u){if(c!=axis){origin[c]=(q[c]/span)*span;}}
 let found=exactCompact(axis|(logSpan<<2u),origin);
 if(found!=INVALID&&found<arrayLength(&seedVelocity)&&finite(seedVelocity[found])){return found;}
 if(span>=p.maximumLeafSize){break;}span*=2u;logSpan+=1u;}return INVALID;}
// Resolve the wet pressure leaf adjacent to this MAC plane, preferring the
// coarsest neighbor exactly as Losasso section 3 prescribes. Interpolation is
// then between that leaf's own two normal faces, never an arbitrary dyadic
// bracket that merely happens to exist around the query.
fn stagedOwnerVelocity(owner:u32,axis:u32,q:vec3u)->u32{
 if(owner==INVALID){return STAGED_INVALID;}let entry=8u+8u*owner;
 if(entry+5u>=arrayLength(&coarsePhiDirectory)||(coarsePhiDirectory[entry+5u]&9u)!=9u){return STAGED_INVALID;}
 let cellPlusOne=coarsePhiDirectory[entry];let ownerSize=coarsePhiDirectory[entry+1u];
 if(cellPlusOne==0u||ownerSize==0u){return STAGED_INVALID;}
 let cell=cellPlusOne-1u;let origin=vec3u(cell%p.velocityDimensions.x,
  (cell/p.velocityDimensions.x)%p.velocityDimensions.y,cell/(p.velocityDimensions.x*p.velocityDimensions.y));
 var lowQ=q;for(var component=0u;component<3u;component+=1u){if(component!=axis){
  lowQ[component]=clamp(q[component],origin[component],min(p.velocityDimensions[component]-1u,
   origin[component]+ownerSize-1u));}}
 var highQ=lowQ;lowQ[axis]=origin[axis];highQ[axis]=origin[axis]+ownerSize;
 let lowFace=containingFiniteVelocity(axis,lowQ);let highFace=containingFiniteVelocity(axis,highQ);
 if(lowFace!=INVALID&&highFace!=INVALID){let normal=clamp(q[axis],origin[axis],origin[axis]+ownerSize);
  let fraction=f32(normal-origin[axis])/f32(ownerSize);
  let value=mix(seedVelocity[lowFace],seedVelocity[highFace],fraction);
  if(finite(value)){return bitcast<u32>(value);}}
 if(lowFace!=INVALID){return bitcast<u32>(seedVelocity[lowFace]);}
 if(highFace!=INVALID){return bitcast<u32>(seedVelocity[highFace]);}
 return STAGED_INVALID;}
fn stagedOwningLeafVelocity(axis:u32,q:vec3u)->u32{
 if(arrayLength(&coarsePhiDirectory)<8u||coarsePhiDirectory[0]!=0x80000000u){return STAGED_INVALID;}
 var owner=INVALID;var ownerSize=0u;
 for(var side=0u;side<2u;side+=1u){var cell=q;var inside=true;
  if(side==0u){if(cell[axis]==0u){inside=false;}else{cell[axis]-=1u;}}
  else if(cell[axis]>=p.velocityDimensions[axis]){inside=false;}
  if(inside){let row=stagedOwnerRow(cell);if(row!=INVALID){let entry=8u+8u*row;
   if(entry+5u<arrayLength(&coarsePhiDirectory)){let size=coarsePhiDirectory[entry+1u];
    if(size>ownerSize&&(coarsePhiDirectory[entry+5u]&9u)==9u){owner=row;ownerSize=size;}}}}}
 return stagedOwnerVelocity(owner,axis,q);}
// The transport stencil straddles the air side of the interface. During a
// topology handoff that site may precede W7 graph republication, but the wet
// pressure leaves and their real bounding faces are already authoritative.
// Blend the local 5x5x5 owner neighborhood so diagonal trilinear corners do
// not inherit nearest-leaf/Voronoi discontinuities. Only zero-coverage sites
// use longer 26-direction rays to preserve complete W7 reach.
fn stagedNearbyOwningLeafVelocity(axis:u32,q:vec3u)->u32{
 let d=p.velocityDimensions;let anchor=min(q,d-vec3u(1u));let fullReach=min(p.maximumLeafSize,p.band.w);
 let reach=min(2u,fullReach);
 var exact:array<i32,36>;var totalWeight=0u;
 let signedReach=i32(reach);
 for(var dz=-signedReach;dz<=signedReach;dz+=1){for(var dy=-signedReach;dy<=signedReach;dy+=1){
  for(var dx=-signedReach;dx<=signedReach;dx+=1){let distance=u32(max(abs(dx),max(abs(dy),abs(dz))));
   if(distance==0u){continue;}let weight=reach+1u-distance;let delta=vec3i(dx,dy,dz);
   let signedCell=vec3i(anchor)+delta;let inside=all(signedCell>=vec3i(0))&&all(signedCell<vec3i(d));
    if(inside){let cell=vec3u(signedCell);let value=stagedOwnerVelocity(stagedOwnerRow(cell),axis,q);
    if(value!=STAGED_INVALID&&value!=STAGED_BOUNDARY_ZERO){exactAdd(&exact,f32(weight)*bitcast<f32>(value));totalWeight+=weight;}}}}}
 if(totalWeight==0u){for(var distance=3u;distance<=fullReach;distance+=1u){let weight=fullReach+1u-distance;
  for(var direction=0u;direction<27u;direction+=1u){if(direction==13u){continue;}
   let delta=vec3i(i32(direction%3u)-1,i32((direction/3u)%3u)-1,i32(direction/9u)-1)*i32(distance);
   let signedCell=vec3i(anchor)+delta;let inside=all(signedCell>=vec3i(0))&&all(signedCell<vec3i(d));
   if(inside){let cell=vec3u(signedCell);let value=stagedOwnerVelocity(stagedOwnerRow(cell),axis,q);
    if(value!=STAGED_INVALID&&value!=STAGED_BOUNDARY_ZERO){exactAdd(&exact,f32(weight)*bitcast<f32>(value));totalWeight+=weight;}}}}}
 if(totalWeight>0u){let value=exactValue(&exact)/f32(totalWeight);if(finite(value)){return bitcast<u32>(value);}}
 return STAGED_INVALID;}
// A finest-lattice MAC plane can lie strictly inside a coarse leaf. There is
// deliberately no face record on that plane, but that is not missing velocity:
// Losasso reconstructs a temporary coarsened value from the leaf's two real
// bounding faces. Find the smallest dyadic bracket that owns the plane and
// interpolate the two face values at this tangential sub-position. At a 2:1
// transition containingFiniteVelocity resolves each bounding sample past any
// starved shadow to the face that covers exactly this sub-area.
fn stagedCoarsenedVelocity(axis:u32,q:vec3u)->u32{
 var span=2u;
 loop{
  let low=(q[axis]/span)*span;let high=low+span;
  if(q[axis]>low&&high<=p.velocityDimensions[axis]){
   var lowQ=q;var highQ=q;lowQ[axis]=low;highQ[axis]=high;
   let lowFace=containingFiniteVelocity(axis,lowQ);let highFace=containingFiniteVelocity(axis,highQ);
   if(lowFace!=INVALID&&highFace!=INVALID&&lowFace<arrayLength(&seedVelocity)
      &&highFace<arrayLength(&seedVelocity)){
    let a=seedVelocity[lowFace];let b=seedVelocity[highFace];
    if(finite(a)&&finite(b)){
     let fraction=f32(q[axis]-low)/f32(span);
     let value=mix(a,b,fraction);
     if(finite(value)){return bitcast<u32>(value);}
    }
   }
  }
  if(span>=p.maximumLeafSize){break;}span*=2u;
 }
 // A 2:1 transition or a free-surface boundary can leave the aligned pair
 // incomplete. Continue with the extension graph's own six-axis metric and
 // average the nearest finite shell. Along the normal this is the usual
 // one-sided constant extension; along a tangent it bridges a split face.
 var upper=p.velocityDimensions-vec3u(1u);upper[axis]=p.velocityDimensions[axis];
 let fallbackReach=min(p.maximumLeafSize,p.band.w);
 for(var distance=1u;distance<=fallbackReach;distance+=1u){
  var exact:array<i32,36>;var count=0u;
  for(var direction=0u;direction<6u;direction+=1u){let component=direction>>1u;var candidate=q;var inside=true;
   if((direction&1u)==0u){if(candidate[component]<distance){inside=false;}else{candidate[component]-=distance;}}
   else{if(candidate[component]+distance>upper[component]){inside=false;}else{candidate[component]+=distance;}}
   if(inside){let found=containingFiniteVelocity(axis,candidate);
    if(found!=INVALID){exactAdd(&exact,seedVelocity[found]);count+=1u;}}}
  if(count>0u){let value=exactValue(&exact)/f32(count);if(finite(value)){return bitcast<u32>(value);}}}
 return STAGED_INVALID;
}
fn stagedRawVelocity(axis:u32,q:vec3u)->u32{
 if(arrayLength(&control)<4u||atomicLoad(&control[3])!=VALID){return STAGED_INVALID;}
 let face=containingCompact(axis,q);if(face!=INVALID){let value=seedVelocity[face];
  if(finite(value)){return bitcast<u32>(value);}}
 let owningValue=stagedOwningLeafVelocity(axis,q);if(owningValue!=STAGED_INVALID){return owningValue;}
 let coveringFace=containingFiniteVelocity(axis,q);if(coveringFace!=INVALID){
  return bitcast<u32>(seedVelocity[coveringFace]);}
 let nearbyOwningValue=stagedNearbyOwningLeafVelocity(axis,q);
 if(nearbyOwningValue!=STAGED_INVALID){return nearbyOwningValue;}
 let lowWall=q[axis]==0u;let highWall=q[axis]==p.velocityDimensions[axis];
 let wallBit=2u*axis+select(1u,0u,lowWall);let closed=(p.closedBoundaries.x&(1u<<wallBit))!=0u;
 if((lowWall||highWall)&&closed){return STAGED_BOUNDARY_ZERO;}
 return stagedCoarsenedVelocity(axis,q);}

fn stagedMacIndex(axis:u32,q:vec3u)->u32{let d=p.velocityDimensions;
 let countX=(d.x+1u)*d.y*d.z;if(axis==0u){return q.x+(d.x+1u)*(q.y+d.y*q.z);}
 let countY=d.x*(d.y+1u)*d.z;if(axis==1u){return countX+q.x+d.x*(q.y+(d.y+1u)*q.z);}
 return countX+countY+q.x+d.x*(q.y+d.y*q.z);}
fn stagedNodalSample(item:u32)->vec4u{let d=p.velocityDimensions;let nd=d+vec3u(1u);
 if(item>=nd.x*nd.y*nd.z||arrayLength(&control)<4u||atomicLoad(&control[3])!=VALID){return vec4u(0u);}
 let node=vec3u(item%nd.x,(item/nd.x)%nd.y,item/(nd.x*nd.y));var result=vec3f(0.);var boundary=false;
 for(var axis=0u;axis<3u;axis+=1u){var exact:array<i32,36>;var count=0u;
  for(var quadrant=0u;quadrant<4u;quadrant+=1u){var coordinate=vec3i(node);var tangent=0u;var inside=true;
   for(var component=0u;component<3u;component+=1u){if(component==axis){continue;}
    coordinate[component]+=select(0,-1,(quadrant&(1u<<tangent))!=0u);
    inside=inside&&coordinate[component]>=0&&coordinate[component]<i32(d[component]);tangent+=1u;}
   if(!inside){continue;}let at=stagedMacIndex(axis,vec3u(coordinate));
   let rawAt=stagedMacCount()+at;if(rawAt>=arrayLength(&stagedVelocityArena)){return vec4u(0u);}
   let sample=stagedVelocityArena[rawAt];
   if(sample==STAGED_INVALID){return vec4u(0u);}let sampleBoundary=sample==STAGED_BOUNDARY_ZERO;
   let value=select(bitcast<f32>(sample),0.,sampleBoundary);
   if(!finite(value)){return vec4u(0u);}boundary=boundary||sampleBoundary;exactAdd(&exact,value);count+=1u;}
  if(count==0u){return vec4u(0u);}result[axis]=exactValue(&exact)/f32(count);}
 if(!all(result==result)||any(abs(result)>vec3f(3.402823e38))){return vec4u(0u);}
 return vec4u(bitcast<u32>(result.x),bitcast<u32>(result.y),bitcast<u32>(result.z),1u|select(0u,2u,boundary));}

@compute @workgroup_size(64)
fn stageLosassoVelocityLattice(@builtin(workgroup_id)group:vec3u,@builtin(num_workgroups)groups:vec3u,
 @builtin(local_invocation_index)lane:u32){let item=64u*(group.x+groups.x*group.y)+lane;let d=p.velocityDimensions;
 let countX=(d.x+1u)*d.y*d.z;let countY=d.x*(d.y+1u)*d.z;let countZ=d.x*d.y*(d.z+1u);
 if(item>=countX+countY+countZ||stagedMacCount()+item>=arrayLength(&stagedVelocityArena)){return;}
 var axis=0u;var local=item;var q=vec3u(0u);
 if(local<countX){q=vec3u(local%(d.x+1u),(local/(d.x+1u))%d.y,local/((d.x+1u)*d.y));}
 else{local-=countX;if(local<countY){axis=1u;q=vec3u(local%d.x,(local/d.x)%(d.y+1u),local/(d.x*(d.y+1u)));}
  else{axis=2u;local-=countY;q=vec3u(local%d.x,(local/d.x)%d.y,local/(d.x*d.y));}}
 let raw=stagedRawVelocity(axis,q);stagedVelocityArena[stagedMacCount()+item]=raw;
 if(raw==STAGED_INVALID){stagedVelocityArena[item]=STAGED_INVALID;return;}
 var value=select(bitcast<f32>(raw),0.,raw==STAGED_BOUNDARY_ZERO);
 let lowWall=q[axis]==0u;let highWall=q[axis]==d[axis];let wallBit=2u*axis+select(1u,0u,lowWall);
 let closed=(p.closedBoundaries.x&(1u<<wallBit))!=0u;
 if((lowWall||highWall)&&closed){var interiorQ=q;
  interiorQ[axis]=select(d[axis]-1u,1u,lowWall);let interiorFace=containingCompact(axis,interiorQ);
  if(interiorFace!=INVALID&&interiorFace<arrayLength(&seedVelocity)){let interior=seedVelocity[interiorFace];
   if(!finite(interior)){stagedVelocityArena[item]=STAGED_INVALID;return;}let away=select((interior<0.),(interior>0.),lowWall);
   if(away){value=interior;}}}
 stagedVelocityArena[item]=bitcast<u32>(value);}

@compute @workgroup_size(64)
fn stageLosassoPredictorVelocityLattice(@builtin(workgroup_id)group:vec3u,@builtin(num_workgroups)groups:vec3u,
 @builtin(local_invocation_index)lane:u32){let item=64u*(group.x+groups.x*group.y)+lane;let d=p.velocityDimensions;
 let countX=(d.x+1u)*d.y*d.z;let countY=d.x*(d.y+1u)*d.z;let countZ=d.x*d.y*(d.z+1u);
 if(item>=countX+countY+countZ||stagedMacCount()+item>=arrayLength(&stagedVelocityArena)){return;}
 var axis=0u;var local=item;var q=vec3u(0u);
 if(local<countX){q=vec3u(local%(d.x+1u),(local/(d.x+1u))%d.y,local/((d.x+1u)*d.y));}
 else{local-=countX;if(local<countY){axis=1u;q=vec3u(local%d.x,(local/d.x)%(d.y+1u),local/(d.x*(d.y+1u)));}
  else{axis=2u;local-=countY;q=vec3u(local%d.x,(local/d.x)%d.y,local/(d.x*d.y));}}
 stagedVelocityArena[stagedMacCount()+item]=stagedRawVelocity(axis,q);}

@compute @workgroup_size(64)
fn stageLosassoNodalVelocityLattice(@builtin(workgroup_id)group:vec3u,@builtin(num_workgroups)groups:vec3u,
 @builtin(local_invocation_index)lane:u32){let item=64u*(group.x+groups.x*group.y)+lane;
 let count=(p.velocityDimensions.x+1u)*(p.velocityDimensions.y+1u)*(p.velocityDimensions.z+1u);
 if(item<count&&item<arrayLength(&stagedNodalVelocityArena)){stagedNodalVelocityArena[item]=stagedNodalSample(item);}}

@compute @workgroup_size(64)
fn stageLosassoPredictorNodalVelocityLattice(@builtin(workgroup_id)group:vec3u,@builtin(num_workgroups)groups:vec3u,
 @builtin(local_invocation_index)lane:u32){let item=64u*(group.x+groups.x*group.y)+lane;
 let count=(p.velocityDimensions.x+1u)*(p.velocityDimensions.y+1u)*(p.velocityDimensions.z+1u);
 if(item<count&&count+item<arrayLength(&stagedNodalVelocityArena)){
  stagedNodalVelocityArena[count+item]=stagedNodalSample(item);}}

@compute @workgroup_size(64)
fn beginLosassoExtensionBand(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){let id=linearInvocation(group,lane);if(id==0u){for(var word=0u;word<8u;word+=1u){atomicStore(&control[word],0u);}
 atomicStore(&control[0],select(0u,wetControl[0],arrayLength(&wetControl)>0u));atomicStore(&control[5],p.fineAuthority.w);}
 if(id<p.band.y){atomicStore(&directory[2u*id],0u);atomicStore(&directory[2u*id+1u],0u);}
 if(id<p.band.y){atomicStore(&geometricClaims[id],0u);}
 if(id<p.band.x){metrics[id]=vec4u(0u);seedWetFace[id]=INVALID;seedVelocity[id]=0.;}}

@compute @workgroup_size(64)
fn clearLosassoStagedCoarseOwners(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
 let item=linearInvocation(group,lane);let volume=p.velocityDimensions.x*p.velocityDimensions.y*p.velocityDimensions.z;
 let at=stagedOwnerOffset()+item;if(item<volume&&at<arrayLength(&stagedVelocityArena)){stagedVelocityArena[at]=0u;}}

@compute @workgroup_size(64)
fn publishLosassoStagedCoarseOwners(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
 let row=linearInvocation(group,lane);if(arrayLength(&coarsePhiDirectory)<8u||coarsePhiDirectory[0]!=0x80000000u
  ||row>=coarsePhiDirectory[2]){return;}let entry=8u+8u*row;if(entry+5u>=arrayLength(&coarsePhiDirectory)){return;}
 let cellPlusOne=coarsePhiDirectory[entry];let size=coarsePhiDirectory[entry+1u];let flags=coarsePhiDirectory[entry+5u];
 if(cellPlusOne==0u||size==0u||(flags&9u)!=9u){return;}let cell=cellPlusOne-1u;
 let origin=vec3u(cell%p.velocityDimensions.x,(cell/p.velocityDimensions.x)%p.velocityDimensions.y,
  cell/(p.velocityDimensions.x*p.velocityDimensions.y));
 for(var z=0u;z<size;z+=1u){for(var y=0u;y<size;y+=1u){for(var x=0u;x<size;x+=1u){let q=origin+vec3u(x,y,z);
  if(any(q>=p.velocityDimensions)){continue;}let at=stagedOwnerOffset()+stagedCellIndex(q);
  if(at<arrayLength(&stagedVelocityArena)){stagedVelocityArena[at]=row+1u;}}}}}

@compute @workgroup_size(64)
fn publishLosassoWetSeedFaces(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){let wet=linearInvocation(group,lane);
 if(arrayLength(&wetControl)<4u||wetControl[3]!=1u){if(wet==0u){atomicOr(&control[4],ERROR_WET_AUTHORITY);}return;}
 if(wet>=wetControl[2]||wet>=arrayLength(&wetGeometry)){return;}let record=wetGeometry[wet];appendWetFace(record,vec4u(record.x&3u,SEED|ACTIVE,0u,0u),wet);}

@compute @workgroup_size(64)
fn publishLosassoAirBandFaces(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
 let fineFactor=max(1u,u32(round(p.velocityOriginCell.w/p.fineCellWidth)));
 let coarseSpan=(p.brickResolution+fineFactor-1u)/fineFactor;
 let tangentSpan=coarseSpan+2u;let facesPerAxis=(coarseSpan+1u)*tangentSpan*tangentSpan;
 let facesPerBrick=3u*facesPerAxis;let work=linearInvocation(group,lane);
 if(work>=min(worklist[1],p.fineAuthority.y)){return;}let page=worklist[7u+work];if(page>=p.fineAuthority.x||4u*page+2u>=arrayLength(&metadata)
  ||metadata[4u*page+2u]!=p.fineAuthority.w){return;}let key=metadata[4u*page+1u];let xy=p.brickDimensions.x*p.brickDimensions.y;
 let bz=key/xy;let remainder=key-bz*xy;let by=remainder/p.brickDimensions.x;let brick=vec3u(remainder-by*p.brickDimensions.x,by,bz);
 let cell=(brick*p.brickResolution)/fineFactor;
 if(any(cell>=p.velocityDimensions)){return;}for(var local=0u;local<facesPerBrick;local+=1u){let axis=local/facesPerAxis;let faceLocal=local%facesPerAxis;
 let normalSide=faceLocal/(tangentSpan*tangentSpan);let tangentCode=faceLocal%(tangentSpan*tangentSpan);
 var q=vec3i(cell);q[axis]+=i32(normalSide);var tangent=0u;
 for(var component=0u;component<3u;component+=1u){if(component!=axis){let digit=select(tangentCode%tangentSpan,tangentCode/tangentSpan,tangent!=0u);
   q[component]+=i32(digit)-1;tangent+=1u;}}
 var upper=vec3i(p.velocityDimensions)-vec3i(1);upper[axis]=i32(p.velocityDimensions[axis]);
 if(any(q<vec3i(0))||any(q>upper)){continue;}let faceQ=vec3u(q);var centre=vec3f(faceQ);
 for(var c=0u;c<3u;c+=1u){if(c!=axis){centre[c]+=.5;}}
 var sample=finePhiCells(centre);
 // The complete MAC stencil around a valid fine transport sample reaches one
 // coarse face past that sample's B4 page. At a diagonal sparse-topology edge,
 // every phi corner at that face may therefore be unresident even though the
 // originating page is valid and needs the face. Fall back to the nearest
 // sample-centre point inside that page. This is a one-sided membership
 // estimate only; transport still gathers phi strictly from resident pages.
 if(sample.valid==0u){let inset=.5*p.fineCellWidth/p.velocityOriginCell.w;
  let pageLow=vec3f(cell)+vec3f(inset);let pageHigh=vec3f(cell+vec3u(coarseSpan))-vec3f(inset);
  sample=finePhiCells(clamp(centre,pageLow,pageHigh));}
 let width=f32(p.band.w)*p.velocityOriginCell.w;
 // Fine phi, not the coarse pressure-row sign, owns the production surface.
 // A finest MAC face can therefore lie on the liquid side of that surface
 // without belonging to any coarse wet-row incidence.  Publish the complete
 // signed W7 support footprint; wet pressure faces were claimed first and
 // remain the projected seeds wherever the two authorities overlap.
 if(sample.valid==0u||abs(sample.value)>width){continue;}
 let layer=min(p.band.w,max(1u,u32(ceil(abs(sample.value)/p.velocityOriginCell.w))));
 appendAirFace(vec4u(axis,faceQ),vec4u(axis,ACTIVE,bitcast<u32>(abs(sample.value)),layer));}}

fn coarsePhi(position:vec3f)->PhiSample{
 let dimensions=p.velocityDimensions;let volume=dimensions.x*dimensions.y*dimensions.z;
 if(arrayLength(&coarsePhiDirectory)<8u||coarsePhiDirectory[0]!=0x80000000u
  ||(coarsePhiDirectory[1]&0x40000000u)==0u){return PhiSample(0.,0u);}
 let cell=vec3i(floor(position));if(any(cell<vec3i(0))||any(cell>=vec3i(dimensions))){return PhiSample(0.,0u);}let q=vec3u(cell);
 let capacity=(arrayLength(&coarsePhiDirectory)-8u)/8u;if(capacity<volume){return PhiSample(0.,0u);}
 let linearCell=q.x+dimensions.x*(q.y+dimensions.y*q.z);let entry=8u+8u*(capacity-volume+linearCell);
 if(entry+5u>=arrayLength(&coarsePhiDirectory)||coarsePhiDirectory[entry]!=linearCell+1u
  ||coarsePhiDirectory[entry+1u]!=1u||(coarsePhiDirectory[entry+5u]&9u)!=9u){return PhiSample(0.,0u);}
 let value=bitcast<f32>(coarsePhiDirectory[entry+2u]);return PhiSample(value,select(0u,1u,finite(value)));}

@compute @workgroup_size(64)
fn publishLosassoCoarseAirBandFaces(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
 let d=p.velocityDimensions;let countX=(d.x+1u)*d.y*d.z;let countY=d.x*(d.y+1u)*d.z;let countZ=d.x*d.y*(d.z+1u);
 let work=linearInvocation(group,lane);if(work>=countX+countY+countZ){return;}var axis=0u;var q=vec3u(0);var code=work;
 if(code<countX){q.x=code%(d.x+1u);code/=d.x+1u;q.y=code%d.y;q.z=code/d.y;}
 else{code-=countX;axis=1u;if(code<countY){q.x=code%d.x;code/=d.x;q.y=code%(d.y+1u);q.z=code/(d.y+1u);}
  else{code-=countY;axis=2u;q.x=code%d.x;code/=d.x;q.y=code%d.y;q.z=code/d.y;}}
 if(insideWetCoarseLeaf(axis,q)){return;}
 var centre=vec3f(q);for(var component=0u;component<3u;component+=1u){if(component!=axis){centre[component]+=.5;}}
 var lowPoint=centre;var highPoint=centre;lowPoint[axis]-=.25;highPoint[axis]+=.25;
 let low=coarsePhi(lowPoint);let high=coarsePhi(highPoint);
 let crossing=low.valid!=0u&&high.valid!=0u&&((low.value<0.)!=(high.value<0.));
 let close=(low.valid!=0u&&abs(low.value)<=p.velocityOriginCell.w)
  ||(high.valid!=0u&&abs(high.value)<=p.velocityOriginCell.w);
 if(!crossing&&!close){return;}
 appendAirFace(vec4u(axis,q),vec4u(axis,ACTIVE,0u,1u));}

@compute @workgroup_size(1)
fn prepareLosassoBandDispatch(){let count=min(atomicLoad(&control[2]),p.band.x);
 liveFaceDispatch[0]=(count+63u)/64u;liveFaceDispatch[1]=1u;liveFaceDispatch[2]=1u;}

fn dilateAirFace(id:u32,sourceLayer:u32,targetLayer:u32){let count=min(atomicLoad(&control[2]),p.band.x);if(id>=count){return;}
 let metric=metrics[id];if((metric.y&ACTIVE)==0u||(metric.y&SEED)!=0u||metric.w!=sourceLayer){return;}
 let record=geometry[id];let axis=record.x&3u;let q=record.yzw;
 // Multiple B4 pages or neighboring layer-4 faces may propose the same key.
 // After the preceding dispatch has settled, only the directory-visible
 // representative expands it, preventing duplicate layer-5 fanout at layer 6.
 if(exactCompact(record.x,q)!=id){return;}
 var upper=p.velocityDimensions-vec3u(1u);upper[axis]=p.velocityDimensions[axis];
 for(var direction=0u;direction<6u;direction+=1u){let c=direction>>1u;var neighbor=q;var inDomain=true;
  if((direction&1u)==0u){if(neighbor[c]==0u){inDomain=false;}else{neighbor[c]-=1u;}}
  else{if(neighbor[c]>=upper[c]){inDomain=false;}else{neighbor[c]+=1u;}}
  if(inDomain){let proposed=vec4u(axis,neighbor);if(exactCompact(axis,neighbor)==INVALID){
   appendAirFace(proposed,vec4u(axis,ACTIVE,bitcast<u32>(f32(targetLayer)*p.velocityOriginCell.w),targetLayer));}}}}
@compute @workgroup_size(64)fn dilateLosassoAirBand2(@builtin(workgroup_id)g:vec3u,@builtin(local_invocation_index)l:u32){dilateAirFace(linearInvocation(g,l),1u,2u);}
@compute @workgroup_size(64)fn dilateLosassoAirBand3(@builtin(workgroup_id)g:vec3u,@builtin(local_invocation_index)l:u32){dilateAirFace(linearInvocation(g,l),2u,3u);}
@compute @workgroup_size(64)fn dilateLosassoAirBand4(@builtin(workgroup_id)g:vec3u,@builtin(local_invocation_index)l:u32){dilateAirFace(linearInvocation(g,l),3u,4u);}
@compute @workgroup_size(64)fn dilateLosassoAirBand5(@builtin(workgroup_id)g:vec3u,@builtin(local_invocation_index)l:u32){dilateAirFace(linearInvocation(g,l),4u,5u);}
@compute @workgroup_size(64)fn dilateLosassoAirBand6(@builtin(workgroup_id)g:vec3u,@builtin(local_invocation_index)l:u32){dilateAirFace(linearInvocation(g,l),5u,6u);}
@compute @workgroup_size(64)fn dilateLosassoAirBand7(@builtin(workgroup_id)g:vec3u,@builtin(local_invocation_index)l:u32){dilateAirFace(linearInvocation(g,l),6u,7u);}

@compute @workgroup_size(64)
fn buildLosassoExtensionAdjacency(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){let id=linearInvocation(group,lane);let count=min(atomicLoad(&control[2]),p.band.x);
 if(id>=count){return;}let record=geometry[id];let axis=record.x&3u;let q=record.yzw;var upper=p.velocityDimensions-vec3u(1u);upper[axis]=p.velocityDimensions[axis];
 adjacencyOffsets[id]=6u*id;for(var direction=0u;direction<6u;direction+=1u){let c=direction>>1u;var neighbor=q;var inDomain=true;
  if((direction&1u)==0u){if(neighbor[c]==0u){inDomain=false;}else{neighbor[c]-=1u;}}
  else{if(neighbor[c]>=upper[c]){inDomain=false;}else{neighbor[c]+=1u;}}
  adjacency[6u*id+direction]=select(INVALID,containingCompact(axis,neighbor),inDomain);}
 if(id+1u==count){adjacencyOffsets[count]=6u*count;}}

@compute @workgroup_size(1)
fn finishLosassoExtensionBand(){let wetValid=arrayLength(&wetControl)>=4u&&wetControl[3]==1u;
 let count=atomicLoad(&control[2]);if(count>p.band.x){atomicOr(&control[4],ERROR_CAPACITY);atomicStore(&control[2],0u);}
 atomicStore(&control[3],select(0u,1u,wetValid&&atomicLoad(&control[4])==0u));}

@compute @workgroup_size(64)
fn clearLosassoDenseWetFaces(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){let id=linearInvocation(group,lane);
 if(id<arrayLength(&denseWetFace)){atomicStore(&denseWetFace[id],INVALID);}}

@compute @workgroup_size(64)
fn scatterLosassoDenseWetFaces(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){let wet=linearInvocation(group,lane);
 if(arrayLength(&wetControl)<4u||wetControl[3]!=1u||wet>=wetControl[2]||wet>=arrayLength(&wetGeometry)){return;}
 let record=wetGeometry[wet];let axis=record.x&3u;let logSpan=record.x>>2u;let span=1u<<logSpan;
 // Prefer the finest owner if transition records overlap. Compact face ids
 // occupy the low 27 bits, leaving five bits for the dyadic level priority.
 let encoded=(logSpan<<27u)|(wet+1u);for(var b=0u;b<span;b+=1u){for(var a=0u;a<span;a+=1u){let q=record.yzw
   +tangentA(axis)*vec3u(a)+tangentB(axis)*vec3u(b);let key=geometricKey(vec4u(axis,q));
  if(key<arrayLength(&denseWetFace)){atomicMin(&denseWetFace[key],encoded);}}}}

@compute @workgroup_size(64)
fn remapLosassoWetSeedFaces(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){let id=linearInvocation(group,lane);
 if(arrayLength(&wetControl)<4u||wetControl[3]!=1u||id>=min(atomicLoad(&control[2]),p.band.x)){return;}let record=geometry[id];let axis=record.x&3u;let span=1u<<(record.x>>2u);
 var owner=INVALID;var uniformOwner=true;for(var b=0u;b<span;b+=1u){for(var a=0u;a<span;a+=1u){let q=record.yzw
   +tangentA(axis)*vec3u(a)+tangentB(axis)*vec3u(b);let key=geometricKey(vec4u(axis,q));
  var encoded=INVALID;if(key<arrayLength(&denseWetFace)){encoded=atomicLoad(&denseWetFace[key]);}
  let wet=select(INVALID,(encoded&0x07ffffffu)-1u,encoded!=INVALID);
  if(wet==INVALID){uniformOwner=false;}else if(owner==INVALID){owner=wet;}else if(wet!=owner){uniformOwner=false;}}}
 // A coarsened/current face maps directly. A retired coarse record spanning
 // several refined wet faces carries a tagged multi-owner mapping; gathering
 // then averages every finest covered face with its physical area weight.
 // Missing coverage remains unseeded, while an invalid wet publication leaves
 // the prior mapping intact through the early return above.
 let seeded=owner!=INVALID;seedWetFace[id]=select(INVALID,select(MULTI_OWNER,owner,uniformOwner),seeded);
 metrics[id].y=select(metrics[id].y&~SEED,metrics[id].y|SEED,seeded);}

@compute @workgroup_size(64)
fn gatherLosassoProjectedSeeds(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){let id=linearInvocation(group,lane);if(id>=min(atomicLoad(&control[2]),p.band.x)){return;}
 let wet=seedWetFace[id];if(atomicLoad(&control[3])!=1u||wet==INVALID){seedVelocity[id]=0.;return;}
 if(wet==MULTI_OWNER){let record=geometry[id];let axis=record.x&3u;let span=1u<<(record.x>>2u);var exact:array<i32,36>;var count=0.;
  for(var b=0u;b<span;b+=1u){for(var a=0u;a<span;a+=1u){let q=record.yzw+tangentA(axis)*vec3u(a)+tangentB(axis)*vec3u(b);let key=geometricKey(vec4u(axis,q));
   if(key<arrayLength(&denseWetFace)){let encoded=atomicLoad(&denseWetFace[key]);if(encoded!=INVALID){let mapped=(encoded&0x07ffffffu)-1u;
    if(mapped<arrayLength(&wetProjected)){let value=wetProjected[mapped];if(finite(value)){exactAdd(&exact,value);count+=1.;}}}}}}
  seedVelocity[id]=select(0.,exactValue(&exact)/max(1.,count),count>0.);return;}
 if(wet>=arrayLength(&wetProjected)){seedVelocity[id]=0.;return;}
 let value=wetProjected[wet];seedVelocity[id]=select(0.,value,finite(value));}

@compute @workgroup_size(64)
fn publishLosassoWetExtended(@builtin(global_invocation_id)invocation:vec3u){let face=invocation.x;
 if(face>=wetControl[2]||face>=arrayLength(&wetProjected)||face>=arrayLength(&wetExtended)){return;}
 let value=wetProjected[face];wetExtended[face]=select(0.,value,finite(value));}
`;
