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
fn floorDiv256(value:i32)->vec2i{var carry=value/256;var digit=value-carry*256;if(digit<0){digit+=256;carry-=1;}return vec2i(carry,digit);}
fn exactAdd(limbs:ptr<function,array<i32,36>>,value:f32){let bits=bitcast<u32>(value);let magnitude=bits&0x7fffffffu;if(magnitude==0u){return;}
 let exponent=(magnitude>>23u)&0xffu;let fraction=magnitude&0x7fffffu;let significand=select(fraction,0x800000u|fraction,exponent!=0u);
 let shift=select(3u,exponent+2u,exponent!=0u);let first=shift>>3u;let shifted=significand<<(shift&7u);let sign=select(1,-1,(bits&0x80000000u)!=0u);
 for(var digit=0u;digit<4u;digit+=1u){let limb=first+digit;let byte=i32((shifted>>(digit*8u))&0xffu);if(byte!=0&&limb<LIMBS){(*limbs)[limb]+=sign*byte;}}}
fn exactValue(input:array<i32,36>)->f32{var limbs=input;for(var limb=0u;limb+1u<LIMBS;limb+=1u){let n=floorDiv256(limbs[limb]);limbs[limb]=n.y;limbs[limb+1u]+=n.x;}
 let negative=limbs[LIMBS-1u]<0;if(negative){for(var limb=0u;limb<LIMBS;limb+=1u){limbs[limb]=-limbs[limb];}
  for(var limb=0u;limb+1u<LIMBS;limb+=1u){let n=floorDiv256(limbs[limb]);limbs[limb]=n.y;limbs[limb+1u]+=n.x;}}
 var magnitude=0.;for(var limb=0u;limb<LIMBS;limb+=1u){magnitude+=ldexp(f32(limbs[limb]),-152+i32(8u*limb));}return select(magnitude,-magnitude,negative);}
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
 let total=exactValue(exactWeight);if(!finite(total)||total<=0.){return PhiSample(0.,0u);}
 // This probe decides velocity-support membership; it is not a transported
 // phi sample. Sparse fine topology may legitimately end through one half of
 // its trilinear stencil while a resident transport sample still needs this
 // MAC face. Renormalizing the available valid corners gives the boundary
 // face its one-sided signed-distance estimate instead of punching a diagonal
 // hole in the W7 velocity graph. Exact accumulators retain D4 invariance.
 return PhiSample(exactValue(exact)/total,1u);}
fn exactCompact(packed:u32,q:vec3u)->u32{let hash=hashFace(packed,q);let mask=p.band.y-1u;
 for(var probe=0u;probe<32u;probe+=1u){let slot=(hash+probe)&mask;let encoded=atomicLoad(&directory[2u*slot]);if(encoded==0u){return INVALID;}
  let face=encoded-1u;if(atomicLoad(&directory[2u*slot+1u])==hash&&face<atomicLoad(&control[2])&&all(geometry[face]==vec4u(packed,q))){return face;}}
 return INVALID;}
fn tangentA(axis:u32)->vec3u{return select(vec3u(1,0,0),vec3u(0,1,0),axis==0u);}
fn tangentB(axis:u32)->vec3u{return select(vec3u(0,0,1),vec3u(0,1,0),axis==2u);}
fn containingCompact(axis:u32,q:vec3u)->u32{var span=1u;var logSpan=0u;loop{var origin=q;
 for(var c=0u;c<3u;c+=1u){if(c!=axis){origin[c]=(q[c]/span)*span;}}let found=exactCompact(axis|(logSpan<<2u),origin);
 if(found!=INVALID){return found;}if(span>=p.maximumLeafSize){break;}span*=2u;logSpan+=1u;}return INVALID;}
fn linearInvocation(group:vec3u,lane:u32)->u32{return 64u*(group.x+65535u*group.y)+lane;}

@compute @workgroup_size(64)
fn beginLosassoExtensionBand(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){let id=linearInvocation(group,lane);if(id==0u){for(var word=0u;word<8u;word+=1u){atomicStore(&control[word],0u);}
 atomicStore(&control[0],select(0u,wetControl[0],arrayLength(&wetControl)>0u));atomicStore(&control[5],p.fineAuthority.w);}
 if(id<p.band.y){atomicStore(&directory[2u*id],0u);atomicStore(&directory[2u*id+1u],0u);}
 if(id<p.band.y){atomicStore(&geometricClaims[id],0u);}
 if(id<p.band.x){metrics[id]=vec4u(0u);seedWetFace[id]=INVALID;seedVelocity[id]=0.;}}

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
  seedVelocity[id]=select(0.,exactValue(exact)/max(1.,count),count>0.);return;}
 if(wet>=arrayLength(&wetProjected)){seedVelocity[id]=0.;return;}
 let value=wetProjected[wet];seedVelocity[id]=select(0.,value,finite(value));}

@compute @workgroup_size(64)
fn publishLosassoWetExtended(@builtin(global_invocation_id)invocation:vec3u){let face=invocation.x;
 if(face>=wetControl[2]||face>=arrayLength(&wetProjected)||face>=arrayLength(&wetExtended)){return;}
 let value=wetProjected[face];wetExtended[face]=select(0.,value,finite(value));}
`;
