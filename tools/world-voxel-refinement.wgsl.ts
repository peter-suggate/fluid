import { SVO_DRY_VISIBILITY_FLAGS } from "../lib/svo/features/shading/program";

/** Bounded reconstruction from the existing structural voxel distance lane.
 * No authored primitive evaluation or source triangle geometry is involved. */
export const worldVoxelRefinementWGSL = /* wgsl */ `
struct WorldVoxelSample { value:f32, material:u32, valid:u32, scale:f32 }
struct WorldVoxelRefined { t:f32, normal:vec3f, valid:u32 }
fn worldVoxelCorner(k:u32)->vec3f{return vec3f(f32(k&1u),f32((k>>1u)&1u),f32(k>>2u));}
fn worldVoxelPolynomial(values:ptr<function,array<f32,8>>,q:vec3f)->vec4f{
 var phi=0.0;var gradient=vec3f(0);
 for(var k=0u;k<8u;k+=1u){let c=worldVoxelCorner(k);let w=mix(vec3f(1)-q,q,c);let sign=c*2.0-vec3f(1);let v=(*values)[k];phi+=v*w.x*w.y*w.z;gradient+=v*sign*vec3f(w.y*w.z,w.x*w.z,w.x*w.y);}
 return vec4f(gradient,phi);
}
// One interpolation cell only: eight current equal-scale samples, a bracket,
// and bounded bisection. Missing storage, a material seam, excessive field
// jumps, a grazing/no-root chord or a root outside this cell keeps the old hit.
fn worldVoxelRefine(ro:vec3f,rd:vec3f,t:f32,material:u32)->WorldVoxelRefined{
 let fallback=WorldVoxelRefined(t,vec3f(0),0u);let point=ro+rd*t;
 let sample=worldVoxelSample(point);if(sample.valid==0u){return fallback;}
 let size=worldVoxelCellSize()*sample.scale;
 let base=floor((point-worldVoxelOrigin())/size-vec3f(.5));let lower=worldVoxelOrigin()+(base+vec3f(.5))*size;
 var values:array<f32,8>;
 for(var k=0u;k<8u;k+=1u){let s=worldVoxelSample(lower+worldVoxelCorner(k)*size);
  if(s.valid==0u||s.scale!=sample.scale||!(abs(s.value)<=4.0*length(size))||(s.material!=0u&&s.material!=material)){return fallback;}values[k]=s.value;
 }
 // SDF samples must not jump faster than the geometric distance across an
 // edge, allowing small packing/producer error. Discontinuous sentinels are
 // unavailable evidence, not a smooth surface to invent between them.
 for(var k=0u;k<8u;k+=1u){for(var axis=0u;axis<3u;axis+=1u){let bit=1u<<axis;if((k&bit)==0u&&abs(values[k]-values[k|bit])>1.25*size[axis]){return fallback;}}}
 var a=max(0.0,t-length(size));var b=t+length(size);
 for(var axis=0u;axis<3u;axis+=1u){
  if(abs(rd[axis])<1e-8){if(ro[axis]<lower[axis]||ro[axis]>lower[axis]+size[axis]){return fallback;}}
  else{let x=(lower[axis]-ro[axis])/rd[axis];let y=(lower[axis]+size[axis]-ro[axis])/rd[axis];a=max(a,min(x,y));b=min(b,max(x,y));}
 }
 if(!(a<b)){return fallback;}
 let qa=clamp((ro+rd*a-lower)/size,vec3f(0),vec3f(1));let qb=clamp((ro+rd*b-lower)/size,vec3f(0),vec3f(1));
 let start=worldVoxelPolynomial(&values,qa);let finish=worldVoxelPolynomial(&values,qb);if(start.w<0.0){return fallback;}
 // A trilinear field along a ray is cubic. Its derivative has at most two
 // roots: splitting there gives monotonic intervals and cannot skip a narrow
 // nearer solid crossing the way a fixed number of chord probes can.
 let direction=qb-qa;let d0=dot(start.xyz,direction);let d1=dot(finish.xyz,direction);
 let c2=3.0*(finish.w-start.w)-2.0*d0-d1;let c3=2.0*(start.w-finish.w)+d0+d1;
 var split=array<f32,3>(1.0,1.0,1.0);var splits=0u;
 if(abs(c3)>1e-8){let disc=4.0*c2*c2-12.0*c3*d0;if(disc>=0.0){let r=sqrt(disc);let x=(-2.0*c2-r)/(6.0*c3);let y=(-2.0*c2+r)/(6.0*c3);
  let first=min(x,y);let second=max(x,y);if(first>0.0&&first<1.0){split[splits]=first;splits+=1u;}if(second>0.0&&second<1.0&&abs(second-first)>1e-7){split[splits]=second;splits+=1u;}
 }}else if(abs(c2)>1e-8){let x=-d0/(2.0*c2);if(x>0.0&&x<1.0){split[splits]=x;splits+=1u;}}
 split[splits]=1.0;splits+=1u;var lo=a;var hi=a;var before=start.w;var found=false;
 for(var i=0u;i<splits;i+=1u){let at=mix(a,b,split[i]);let after=worldVoxelPolynomial(&values,mix(qa,qb,split[i])).w;
  if(before>=0.0&&after<=0.0){hi=at;found=true;break;}lo=at;before=after;
 }
 if(!found){return fallback;}
 for(var i=0u;i<12u;i+=1u){let mid=.5*(lo+hi);let phi=worldVoxelPolynomial(&values,clamp((ro+rd*mid-lower)/size,vec3f(0),vec3f(1))).w;if(phi>0.0){lo=mid;}else{hi=mid;}}
 let hit=.5*(lo+hi);let gradient=worldVoxelPolynomial(&values,clamp((ro+rd*hit-lower)/size,vec3f(0),vec3f(1))).xyz/size;
 if(dot(gradient,gradient)<1e-8||dot(gradient,rd)>=0.0){return fallback;}
 return WorldVoxelRefined(hit,normalize(gradient),1u);
}
`;

/** Point lookup follows the same current leaf/lifecycle contract as
 * meshRegionAt. Sample positions are voxel centres on the selected scale. */
export const worldVoxelRefinementSceneWGSL = /* wgsl */ `
fn worldVoxelCellSize()->vec3f{return dry.mapping.cellSize;}
fn worldVoxelOrigin()->vec3f{return dry.mapping.worldOrigin;}
fn worldVoxelSample(world:vec3f)->WorldVoxelSample{
 let p=(world-dry.mapping.worldOrigin)/dry.mapping.cellSize;let rootSize=f32((1u<<dry.mapping.maximumDepth)*dry.mapping.brickSize);
 if(any(p<vec3f(0))||any(p>=vec3f(rootSize))){return WorldVoxelSample();}
 var origin=vec3f(0);var size=rootSize;var index=0u;
 for(var level=0u;level<=dry.mapping.maximumDepth;level+=1u){
  if(index>=svoControlLoad(0u)){return WorldVoxelSample();}let node=svoNodeLoad(index);
  if(node.links.z!=SVO_INVALID){
   let leaf=svoLeafLoad(node.links.z).topology;
   if(leaf.x!=index||leaf.z!=0u||!svoBrickLifecycleCurrent(svoBrickLifecycleDecode(node.links.w))){return WorldVoxelSample();}
   let scale=size/f32(dry.mapping.brickSize);let local=vec3u(clamp(floor((p-origin)/scale),vec3f(0),vec3f(f32(dry.mapping.brickSize-1u))));
   let voxel=svoBrickVoxelIndex(leaf.y,local,dry.mapping.brickSize);if(voxel>=dryVoxelCapacity()){return WorldVoxelSample();}
   let descriptor=dry.payloadLanes1.w;let stride=(descriptor>>8u)&255u;let at=dry.payloadLanes1.x+voxel*stride;
   if(stride==0u||at>=arrayLength(&scenePayload)){return WorldVoxelSample();}
   let word=scenePayload[at];let value=select(bitcast<f32>(word),sceneDistanceOf(word,voxel),(descriptor&(1u<<24u))!=0u);
   return WorldVoxelSample(value,sceneIdentityMaterial(sceneIdentityAt(voxel)),1u,scale);
  }
  size*=.5;let upper=p>=origin+vec3f(size);let octant=select(0u,1u,upper.x)|select(0u,2u,upper.y)|select(0u,4u,upper.z);origin+=select(vec3f(0),vec3f(size),upper);
  let mask=node.address.w&255u;let bit=1u<<octant;if((mask&bit)==0u){return WorldVoxelSample();}index=node.links.x+countOneBits(mask&(bit-1u));
 }
 return WorldVoxelSample();
}
fn waterWorldRefinementEligible(hit:DryHit)->bool{
 return hit.t<DRY_MISS&&hit.fieldSource==DRY_GBUFFER_FIELD_VOXEL&&hit.ownerId==DRY_OWNER_NONE&&hit.motionKind==DRY_GBUFFER_MOTION_STATIC&&dry.meshFilterNormals.w>=2.0&&(dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.flatVoxelNormals}u)==0u&&!drySampledMaterial(hit.materialId);
}
fn waterWorldRefineOpaque(hit:DryHit,ro:vec3f,rd:vec3f)->DryHit{
 if(!waterWorldRefinementEligible(hit)){return hit;}
 let refined=worldVoxelRefine(ro,rd,hit.t,hit.materialId);if(refined.valid==0u){return hit;}
 // A coarse static face could have hidden a nearer analytic body. Moving the
 // receiver must preserve that body's authoritative intersection.
 if(refined.t>hit.t){let rigid=nearestBody(ro,rd);if(rigid.t<refined.t){return rigid;}}
 var result=hit;result.t=refined.t;result.normal=refined.normal;result.aux=vec3u(0u,0u,1u);return result;
}
`;
