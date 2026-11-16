/** Lane P's shared rigid-body WGSL for the brick engine (M5 design note, docs/plans/nbflip-port/m5-bodies-design-2026-11-17.md
 * §2.3, §4, §6, §9). The physics is the old uniform engine's, ported verbatim; only the addressing changes:
 * - the old engine read the pose from `umSolidBodies` and the body cuts from a texture; here both are storage arrays
 *   the kernel names (`PbBodySource`), so the module fixes no binding (E's ABI, amendments A1-A8, decides those);
 * - the old candidate sets came from the solid record's tile masks; here every query takes its candidate mask (the cut
 *   slot's BMASK) as an argument, filtered by the same reach tests;
 * - the static solid (voxels, boundary records, terrain) is reached through five hook functions the kernel supplies
 *   (`PbStaticHooks`), the functions E's solidStamp itself uses (A7).
 *
 * Old file:line for each function is given beside it (main at c95f17aa). tests: bodies-shapes.test.ts (R6/R7 parity
 * against the old WGSL, compiled from the old engine's own sources). Rounded plates (shape 5) are not ported: a kernel
 * refuses them through pbBodyShapeSupported. */
import { sceneShapeWgsl } from "../../core/scene-shape";
import { fluidRigidBoundaryWGSL } from "../../core/fluid-rigid-boundary.wgsl";
import { solidBoxCutWGSL } from "../../core/solid-cuts";
import { GPU_RIGID_BODY_CAPACITY } from "../../core/rigid-capacity";
import { SCENE_SHAPE_TABLE } from "../../core/scene-shape";

/** Where the module reads the bodies and the lattice. Each is a WGSL expression or variable name. */
export interface PbBodySource {
  /** `array<PBRigidBody, 24>` (or a runtime array of it): the rigid state buffer, 128 B a body (webgpu-rigid-body.ts:73-92). */
  readonly bodies: string;
  /** u32: the present body count; clamped to the capacity. */
  readonly count: string;
  /** `array<vec4f>`: the body-local cut atlas of packRigidBodyCutAtlas (solid-cuts.ts:32-38), a vec4f a texel. */
  readonly cuts: string;
  /** vec3f: the world position (m) of lattice cell (0,0,0)'s low corner (the old umSolidParams.frame.xyz). */
  readonly frame: string;
  /** vec3f: the cell size per axis (m) (the old umSolidParams.cellGravity.xyz and params.hDt.xyz). */
  readonly h: string;
}

/** The static solid, as the five functions the kernel defines (E's solidStamp functions, A7). Face functions take the
 * face on the +axis side of cell c, as the old engine's do. */
export interface PbStaticHooks {
  /** `(world:vec3f)->bool`: a static solid holds the point (umSolidVoxelAtWorld || umSolidTerrainAtWorld). */
  readonly solidAt: string;
  /** `(c:vec3i)->f32`: the cell's static open fraction (umCellOpenGeometry with no body). */
  readonly cellOpen: string;
  /** `(c:vec3i)->u32`: PB_STATIC_PLAIN, PB_STATIC_BOUNDARY (a boundary record) or PB_STATIC_VOXEL (a voxel, no record). */
  readonly cellKind: string;
  /** `(c:vec3i,axis:u32)->f32`: the static aperture of a face that is not a voxel-wall face (umPressureFaceV, no body). */
  readonly faceOpen: string;
  /** `(c:vec3i,axis:u32)->bool`: a procedural boundary touches the face's unit box (umBoundaryTouchesBox(q, 1)). */
  readonly faceBoxed: string;
}

export const PB_DEFAULT_STATICS: PbStaticHooks = {
  solidAt: "nbStaticSolidAt", cellOpen: "nbStaticCellOpen", cellKind: "nbStaticCellKind",
  faceOpen: "nbStaticFaceOpen", faceBoxed: "nbStaticFaceBoxed",
};

/** The shape tags the brick engine stamps: sphere, box, capsule, cylinder, cup (and their interior inversions). */
export const PB_SUPPORTED_SHAPE_CODES: readonly number[] = Object.freeze(
  Object.values(SCENE_SHAPE_TABLE).map(s => s.code).filter(code => code !== SCENE_SHAPE_TABLE["rounded-plate"].code));

/** The pose and shape queries (§9: F's move, bodyStamp, couple, setMark). */
export function pbBodyShapesWGSL(src: PbBodySource): string {
  const B = src.bodies, C = src.cuts, H = src.h, F = src.frame;
  const supported = PB_SUPPORTED_SHAPE_CODES.map(code => `code==${code}`).join("||");
  return /* wgsl */`
struct PBRigidBody {
 positionShape:vec4f,dimensions:vec4f,orientation:vec4f,linearVelocity:vec4f,
 angularVelocity:vec4f,inverseMassInertia:vec4f,angularMomentumRestitution:vec4f,material:vec4f,
}
${sceneShapeWgsl()}
${fluidRigidBoundaryWGSL}
${solidBoxCutWGSL}
const PB_BODY_CAPACITY:u32=${GPU_RIGID_BODY_CAPACITY}u;
fn pbBodyCount()->u32{return min(${src.count},PB_BODY_CAPACITY);}
fn pbBodyMaskAll()->u32{return (1u<<pbBodyCount())-1u;}
fn pbH()->vec3f{return ${H};}
// Unit axis vectors. The module writes no vector component through a dynamic index: on Metal (Dawn, 2026-11-17) such a
// write in a loop whose body calls the shape queries miscompiled (couple's face-area loop, ported verbatim, gave wrong
// areas while each call alone was right; bodies-shapes.test.ts holds a CPU oracle against it). An offset is added as
// c * e_axis instead: exact, as every other component adds 0.
fn pbE(axis:u32)->vec3f{return vec3f(select(0.0,1.0,axis==0u),select(0.0,1.0,axis==1u),select(0.0,1.0,axis==2u));}
fn pbEi(axis:u32)->vec3i{return vec3i(select(0,1,axis==0u),select(0,1,axis==1u),select(0,1,axis==2u));}
// The decoded shape tag is one the brick engine stamps (no rounded plate, no unknown tag): a kernel latches
// FAIL.UNSUPPORTED on false.
fn pbBodyShapeSupported(i:u32)->bool{let raw=i32(round(${B}[i].positionShape.w));let code=select(raw,-raw-1,raw<0);return ${supported};}
// umSolidWorldCell, umSolidFaceWorld, umSolidTraceWorld (uniform-mixed-solid.wgsl.ts:1644, 1793-1795).
fn pbWorldCell(id:vec3i)->vec3f{let h=${H};let f=${F};
 return vec3f(f.x+(f32(id.x)+0.5)*h.x,f.y+(f32(id.y)+0.5)*h.y,f.z+(f32(id.z)+0.5)*h.z);}
fn pbFaceWorld(id:vec3i,axis:u32)->vec3f{return pbWorldCell(id)+(0.5*${H}[axis])*pbE(axis);}
fn pbTraceWorld(p:vec3f)->vec3f{let h=${H};let f=${F};return vec3f(f.x+p.x*h.x,f.y+p.y*h.y,f.z+p.z*h.z);}
// umQuatRotate, umBodyLocal, umBodyNear (1703-1705).
fn pbQuatRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.0*(q.x*uv+uuv);}
fn pbBodyLocal(i:u32,world:vec3f)->vec3f{let b=${B}[i];return pbQuatRotate(vec4f(b.orientation.x,-b.orientation.yzw),world-b.positionShape.xyz);}
fn pbBodyNear(i:u32,world:vec3f,margin:f32)->bool{let b=${B}[i];return b.positionShape.w<0.0||distance(world,b.positionShape.xyz)<=b.dimensions.w+margin;}
// umApplyBodyCuts (1673-1681): the atlas header texel i is (first texel, cut count); three texels a cut.
fn pbApplyBodyCuts(i:u32,local:vec3f,distance:f32)->f32{
 let header=${C}[i];var result=distance;
 for(var k=0u;k<u32(header.y);k++){
  let at=u32(header.x)+3u*k;let center=${C}[at].xyz;
  let extent=${C}[at+1u].xyz;let q=${C}[at+2u];
  result=max(result,-solidBoxCutDistance(local,center,extent,q));
 }return result;
}
// umInsideBody and umBodyDistance with cuts (1706-1707); umBodyVelocity (1709).
fn pbBodyInside(i:u32,world:vec3f)->bool{if(!pbBodyNear(i,world,0.0)){return false;}let b=${B}[i];if(${C}[i].y==0.0){return fluidRigidInside(i32(round(b.positionShape.w)),b.dimensions.xyz,pbBodyLocal(i,world));}return pbBodyDistance(i,world)<=0.0;}
fn pbBodyDistance(i:u32,world:vec3f)->f32{let b=${B}[i];let local=pbBodyLocal(i,world);return pbApplyBodyCuts(i,local,fluidRigidDistance(i32(round(b.positionShape.w)),b.dimensions.xyz,local));}
fn pbBodyVelocity(i:u32,world:vec3f)->vec3f{let b=${B}[i];return b.linearVelocity.xyz+cross(b.angularVelocity.xyz,world-b.positionShape.xyz);}
fn pbBodyPositionOf(i:u32)->vec3f{return ${B}[i].positionShape.xyz;}
// umBodyCandidates over a candidate mask (1690): the bodies of mask within margin of world.
fn pbBodyCandidates(world:vec3f,margin:f32,mask:u32)->u32{
 var result=0u;var remaining=mask&pbBodyMaskAll();
 while(remaining!=0u){let i=firstTrailingBit(remaining);remaining&=remaining-1u;if(pbBodyNear(i,world,margin)){result|=1u<<i;}}
 return result;
}
// umBodyAtCandidates (1716-1723): the first body in roster order holding the point, -1 for none.
fn pbBodyAtCandidates(world:vec3f,candidates:u32)->i32{
 var remaining=candidates;
 while(remaining!=0u){let i=firstTrailingBit(remaining);remaining&=remaining-1u;
  if(pbBodyInside(i,world)){return i32(i);}
 }
 return -1;
}
// A brick's BMASK (§3): every body whose bounding sphere, dilated by reach, meets the box [lo, hi]; an inverted body's
// solid lies outside its sphere, so it is always a candidate.
fn pbBodyBoxCandidates(lo:vec3f,hi:vec3f,reach:f32)->u32{
 var mask=0u;
 for(var i=0u;i<pbBodyCount();i++){let b=${B}[i];let c=b.positionShape.xyz;
  if(b.positionShape.w<0.0||distance(clamp(c,lo,hi),c)<=b.dimensions.w+reach){mask|=1u<<i;}}
 return mask;
}
// umBodyFraction (uniform-rigid-coverage.wgsl.ts:10-20, no plate): body i's share of cell p's 8 corner samples.
fn pbBodyFraction(i:u32,p:vec3i)->f32{
 let h=${H};let centre=pbWorldCell(p);
 if(!pbBodyNear(i,centre,length(0.4*h))){return 0.0;}
 var inside=0.0;
 for(var corner=0u;corner<8u;corner++){
  let offset=vec3f(select(-0.4,0.4,(corner&1u)!=0u),select(-0.4,0.4,(corner&2u)!=0u),select(-0.4,0.4,(corner&4u)!=0u));
  if(pbBodyInside(i,centre+offset*h)){inside+=1.0;}
 }
 return inside/8.0;
}
// umCellBodyFraction (uniform-rigid-coverage.wgsl.ts:21-41, no plate) over a candidate mask: the union of the bodies'
// corner samples, with the 1.01 length(0.4h) clearance certificate.
fn pbCellBodyFraction(p:vec3i,candidates:u32)->f32{
 let h=${H};let centre=pbWorldCell(p);
 var mask=0u;
 var bodyCandidates=candidates&pbBodyMaskAll();while(bodyCandidates!=0u){let i=firstTrailingBit(bodyCandidates);bodyCandidates&=bodyCandidates-1u;
  if(!pbBodyNear(i,centre,length(0.4*h))){continue;}
  let d=pbBodyDistance(i,centre);let reach=1.01*length(0.4*h);
  if(d>reach){continue;}
  if(d< -reach&&pbBodyNear(i,centre,-reach)){return 1.0;}
  for(var corner=0u;corner<8u;corner++){
   let bit=1u<<corner;if((mask&bit)!=0u){continue;}
   let offset=vec3f(select(-0.4,0.4,(corner&1u)!=0u),select(-0.4,0.4,(corner&2u)!=0u),select(-0.4,0.4,(corner&4u)!=0u));
   if(pbBodyInside(i,centre+offset*h)){mask|=bit;}
  }
  if(mask==255u){return 1.0;}
 }
 return f32(countOneBits(mask))/8.0;
}
// umBodyFaceFraction (uniform-mixed-bodies.ts:18-25): body i's share of the +axis face of p's four transverse
// samples, bodies only (couple's face fraction, §6).
fn pbBodyFaceFraction(i:u32,p:vec3i,axis:u32)->f32{
 let world=pbFaceWorld(p,axis);let h=${H};var blocked=0.0;
 for(var k=0u;k<4u;k++){
  let a=(axis+1u)%3u;let b=(axis+2u)%3u;
  let sample=world+(select(-0.35,0.35,(k&1u)!=0u)*h[a])*pbE(a)+(select(-0.35,0.35,(k&2u)!=0u)*h[b])*pbE(b);
  blocked+=select(0.0,0.25,pbBodyInside(i,sample));
 }return blocked;
}
// umBoxGap and umBodyBoxTileGap (uniform-body-tile-bound.ts:27-50): the separating-axis lower bound on the distance
// from box body i to the world-axis box of centre middle and half extents a.
fn pbBoxGap(n:vec3f,c:vec3f,e:vec3f,a:vec3f,u0:vec3f,u1:vec3f,u2:vec3f)->f32{
 return abs(dot(n,c))-dot(e,abs(n))-(a.x*abs(dot(n,u0))+a.y*abs(dot(n,u1))+a.z*abs(dot(n,u2)));
}
fn pbBodyBoxTileGap(i:u32,middle:vec3f,a:vec3f)->f32{
 let b=${B}[i];let e=0.5*b.dimensions.xyz;let c=pbBodyLocal(i,middle);
 let q=vec4f(b.orientation.x,-b.orientation.yzw);
 let u0=pbQuatRotate(q,vec3f(1.0,0.0,0.0));let u1=pbQuatRotate(q,vec3f(0.0,1.0,0.0));let u2=pbQuatRotate(q,vec3f(0.0,0.0,1.0));
 var gap=max(max(pbBoxGap(vec3f(1.0,0.0,0.0),c,e,a,u0,u1,u2),pbBoxGap(vec3f(0.0,1.0,0.0),c,e,a,u0,u1,u2)),pbBoxGap(vec3f(0.0,0.0,1.0),c,e,a,u0,u1,u2));
 gap=max(gap,max(max(pbBoxGap(u0,c,e,a,u0,u1,u2),pbBoxGap(u1,c,e,a,u0,u1,u2)),pbBoxGap(u2,c,e,a,u0,u1,u2)));
 for(var k=0u;k<3u;k++){
  let axis=pbE(k);
  for(var j=0u;j<3u;j++){
   let v=cross(axis,select(select(u2,u1,j==1u),u0,j==0u));let l=length(v);
   if(l>1e-3){gap=max(gap,pbBoxGap(v/l,c,e,a,u0,u1,u2));}
  }
 }
 let p=c-clamp(c,-e,e);let l=length(p);
 if(l>0.0){gap=max(gap,pbBoxGap(p/l,c,e,a,u0,u1,u2));}
 return gap;
}
// umBodyMarkTile's criterion (uniform-mixed-bodies.ts:189-217), for E's setMark (§4): some body may touch the 4^3 tile
// whose low corner is low within horizon seconds. A marked tile is h-owned and its brick NEAR_BODY.
fn pbBodyMarkTile(low:vec3f,horizon:f32)->bool{
 let h=${H};let high=low+4.0*h;
 let middle=0.5*(low+high);let half=length(2.0*h);let cell=max(h.x,max(h.y,h.z));
 for(var i=0u;i<pbBodyCount();i++){
  let b=${B}[i];let centre=b.positionShape.xyz;let radius=max(b.dimensions.w,0.0);
  let travel=length(b.linearVelocity.xyz)*horizon;
  if(distance(clamp(centre,low,high),centre)>radius+travel+cell){continue;}
  let margin=(length(b.linearVelocity.xyz)+length(b.angularVelocity.xyz)*radius)*horizon+cell;
  if(b.positionShape.w<0.0){return true;}
  var lower=pbBodyDistance(i,middle)-half;
  if(lower<=margin&&i32(round(b.positionShape.w))==${SCENE_SHAPE_TABLE.box.code}){lower=max(lower,pbBodyBoxTileGap(i,middle,2.0*h));}
  if(lower<=margin){return true;}
 }
 return false;
}
// nbRigidContact (uniform-particle-contact.wgsl.ts:152-178), the candidate-mask form: a sample (lattice coordinates)
// a body has overtaken moves out to inset cells from its surface and loses its velocity into it, relative to the
// wall's (rotation included).
struct PBRigidContact{position:vec3f,velocity:vec3f}
fn pbRigidContact(point:vec3f,velocity:vec3f,inset:f32,mask:u32)->PBRigidContact{
 let h=${H};let cell=min(h.x,min(h.y,h.z));let skin=inset*cell;let step=0.01*cell;
 var world=pbTraceWorld(point);var v=velocity;
 var candidates=mask&pbBodyMaskAll();
 while(candidates!=0u){let i=firstTrailingBit(candidates);candidates&=candidates-1u;let before=world;
  if(!pbBodyNear(i,world,skin)){continue;}
  for(var iteration=0u;iteration<4u;iteration++){
   let distance=pbBodyDistance(i,world);if(distance>skin){break;}
   var gradient=vec3f(0);
   for(var axis=0u;axis<3u;axis++){
    let delta=step*pbE(axis);
    gradient=gradient+(pbBodyDistance(i,world+delta)-pbBodyDistance(i,world-delta))*pbE(axis);
   }
   var normal=vec3f(0,1,0);let g=length(gradient);if(g>1e-10){normal=gradient/g;}
   world+=(skin-distance)*normal;
   let wall=pbBodyVelocity(i,world);
   v-=min(dot(v-wall,normal),0.0)*normal;
  }
  // A contact can move into a body excluded at the original point: restore every later body, in roster order.
  if(any(world!=before)){candidates=pbBodyMaskAll()&~((1u<<(i+1u))-1u);}
 }
 return PBRigidContact(point+(world-pbTraceWorld(point))/h,v);
}
`;
}

/** The composite stamp formulas bodyStamp writes into a NEAR_BODY brick's cut slot (§2.3): FRAC, AP and W, static and
 * body together. Needs pbBodyShapesWGSL and the static hooks. */
export function pbBodyStampWGSL(hooks: PbStaticHooks = PB_DEFAULT_STATICS): string {
  const S = hooks;
  return /* wgsl */`
const PB_STATIC_PLAIN:u32=0u;const PB_STATIC_BOUNDARY:u32=1u;const PB_STATIC_VOXEL:u32=2u;
// FRAC: umCellOpenGeometry (uniform-mixed-solid.wgsl.ts:1728-1741). The static branch is the hook's open fraction:
// with no body it is umUnitBoxOpen's min(box.y, box.x) on a boundary-record cell and clamp(box.x) elsewhere.
fn pbStampCell(p:vec3i,mask:u32)->f32{
 let kind=${S.cellKind}(p);
 if(kind==PB_STATIC_VOXEL){return 0.0;}
 let body=pbCellBodyFraction(p,mask);let open=${S.cellOpen}(p);
 if(kind==PB_STATIC_BOUNDARY){return min(open,1.0-body);}
 return clamp((1.0-body)*open,0.0,1.0);
}
// umFaceOpen (1879-1908, no plate): the transverse aperture of the +axis face of id, 4 samples at +-0.35h, each solid
// where a static solid or a candidate body holds it. The old staticClear skip only avoided static probes that answer
// false, so the hook is asked at every sample.
fn pbFaceOpen(id:vec3i,axis:u32,candidates:u32)->f32{
 let world=pbFaceWorld(id,axis);let h=pbH();
 let tangentA=(axis+1u)%3u;let tangentB=(axis+2u)%3u;var solid=0.0;
 let sampleReach=1.01*0.35*length(vec2f(h[tangentA],h[tangentB]));
 var bodies=0u;var remaining=candidates;
 while(remaining!=0u){let i=firstTrailingBit(remaining);remaining&=remaining-1u;
  let d=pbBodyDistance(i,world);
  if(d< -sampleReach&&pbBodyNear(i,world,-sampleReach)){return 0.0;}
  if(d<=sampleReach){bodies|=1u<<i;}
 }
 for(var sampleIndex=0u;sampleIndex<4u;sampleIndex+=1u){
  let sampleWorld=world+(select(-0.35,0.35,(sampleIndex&1u)!=0u)*h[tangentA])*pbE(tangentA)+(select(-0.35,0.35,(sampleIndex&2u)!=0u)*h[tangentB])*pbE(tangentB);
  solid+=select(0.0,1.0,${S.solidAt}(sampleWorld)||pbBodyAtCandidates(sampleWorld,bodies)>=0);
 }
 return 1.0-solid*0.25;
}
// A voxel-wall face: either cell is a voxel with no boundary record.
fn pbStaticWallFace(id:vec3i,axis:u32)->bool{let neighbor=id+pbEi(axis);return ${S.cellKind}(id)==PB_STATIC_VOXEL||${S.cellKind}(neighbor)==PB_STATIC_VOXEL;}
// AP: umPressureFaceV (1915-1952) of the +axis face of id; fracLow and fracHigh are the two cells' pbStampCell.
// - no body within length(0.5h) of the face centre: a voxel-wall face (not boxed) averages the two cells' FRAC, any
//   other face takes the static aperture;
// - a body near: a voxel-wall face averages, any other face is umFaceOpen.
fn pbStampFace(id:vec3i,axis:u32,mask:u32,fracLow:f32,fracHigh:f32)->f32{
 let h=pbH();let candidates=pbBodyCandidates(pbFaceWorld(id,axis),length(0.5*h),mask);
 let wall=pbStaticWallFace(id,axis);
 if(candidates==0u){if(wall&&!${S.faceBoxed}(id,axis)){return 0.5*(fracLow+fracHigh);}return ${S.faceOpen}(id,axis);}
 if(wall){return 0.5*(fracLow+fracHigh);}
 return pbFaceOpen(id,axis,candidates);
}
// W: umSolidFaceVelocity (1817-1871) of the +axis face of id, the moving-wall velocity u_s (m/s): the mean rigid
// velocity at the face centre over its 8 dual-cell samples a body holds (a static sample counts as solid at rest),
// else the nearest body's within one cell; a voxel-wall face is at rest.
fn pbStampFaceVelocity(id:vec3i,axis:u32,mask:u32)->f32{
 if(pbBodyCount()==0u){return 0.0;}
 let world=pbFaceWorld(id,axis);let h=pbH();let reach=max(h.x,max(h.y,h.z));
 var candidates=0u;var near=mask&pbBodyMaskAll();
 while(near!=0u){let i=firstTrailingBit(near);near&=near-1u;if(pbBodyNear(i,world,2.0*reach)){candidates|=1u<<i;}}
 if(candidates==0u){return 0.0;}
 if(pbStaticWallFace(id,axis)){return 0.0;}
 let sampleReach=1.01*length(0.4*h);
 var sampled=0u;var owner=-1;var leading=true;
 var nearest=reach;var result=0.0;
 var remaining=candidates;
 while(remaining!=0u){let i=firstTrailingBit(remaining);remaining&=remaining-1u;
  let sd=pbBodyDistance(i,world);let d=abs(sd);
  if(d<=nearest){nearest=d;result=pbBodyVelocity(i,world)[axis];}
  if(sd<=sampleReach){
   if(leading&&sd< -sampleReach&&pbBodyNear(i,world,-sampleReach)){owner=i32(i);}
   leading=false;sampled|=1u<<i;
  }
 }
 var solid=0.0;var velocity=0.0;
 for(var sampleIndex=0u;sampleIndex<8u;sampleIndex+=1u){
  let sampleWorld=world+vec3f(select(-0.4,0.4,(sampleIndex&1u)!=0u)*h.x,select(-0.4,0.4,(sampleIndex&2u)!=0u)*h.y,select(-0.4,0.4,(sampleIndex&4u)!=0u)*h.z);
  if(${S.solidAt}(sampleWorld)){solid+=1.0;continue;}
  var body=owner;if(owner<0){body=pbBodyAtCandidates(sampleWorld,sampled);}
  if(body>=0){solid+=1.0;velocity+=pbBodyVelocity(u32(body),world)[axis];}
 }
 if(solid>0.0){return velocity/solid;}
 return result;
}
`;
}

/** couple's per-cell, per-body contribution (§6; uniform-mixed-bodies.ts:37-94). Needs pbBodyShapesWGSL. */
export const pbBodyCoupleWGSL = /* wgsl */`
// The exchange's fixed point (webgpu-eulerian.ts:443-471): words 0-5 x1e6, word 6 x65536.
fn pbFixed(x:f32)->i32{return i32(round(x*1000000.0));}
fn pbFixedWet(x:f32)->i32{return i32(round(x*65536.0));}
// umBodyOwnerWet with phi in h (the old /(4h) is /4 here): an h owner's wet fraction from the sum of its 8 corner phi.
fn pbOwnerWet(phiSum:f32)->f32{return clamp(0.5-0.125*phiSum/4.0,0.0,1.0);}
struct PBCouple{reaction:vec3f,torque:vec3f,wet:f32,near:bool}
// Cell id's pressure reaction on body owner and its displaced wet weight. pressure: p (Pa) at the cell, 0 where FRAC
// <= 1e-5 or the cell is not liquid (umBodyPressure); wet: pbOwnerWet; dt: the step. The reaction
// p (blocked+ - blocked-) h_a h_b dt is the transpose of the RHS moving-wall flux.
fn pbCoupleBody(owner:u32,id:vec3i,pressure:f32,wet:f32,dt:f32)->PBCouple{
 let h=pbH();let world=pbWorldCell(id);
 var out=PBCouple(vec3f(0.0),vec3f(0.0),0.0,false);
 if(!pbBodyNear(owner,world,length(h))){return out;}
 out.near=true;
 // Shape distances are exact (1-Lipschitz): a centre farther than half the cell's diagonal from the surface has every
 // face and corner sample on its own side, its face fractions cancel and its fraction is 0 or 1.
 let distance=pbBodyDistance(owner,world);let clear=abs(distance)>0.505*length(h);
 var area=vec3f(0.0);
 if(!clear){
  for(var axis=0u;axis<3u;axis++){
   let low=id-pbEi(axis);
   area=area+((pbBodyFaceFraction(owner,id,axis)-pbBodyFaceFraction(owner,low,axis))*h[(axis+1u)%3u]*h[(axis+2u)%3u])*pbE(axis);
  }
 }
 if(any(area!=vec3f(0.0))){
  out.reaction=pressure*area*dt;
  out.torque=cross(world-pbBodyPositionOf(owner),out.reaction);
 }
 var fraction=select(0.0,1.0,distance<0.0);if(!clear){fraction=pbBodyFraction(owner,id);}
 out.wet=wet*fraction;
 return out;
}
`;

