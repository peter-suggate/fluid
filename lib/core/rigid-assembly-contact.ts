import asset from "./imported-assets/paddleboats.generated.json";
import type { RigidBodyDescription } from "./model";
import { rigidAppearanceSkin } from "./rigid-appearance";
import { SCENE_SHAPES_BY_CODE, sceneShapeCode, sceneShapeRenderHalfExtent_m } from "./scene-shape";
import { convexHullVertices, type HullPoint } from "./convex-hull";
import { quaternionRotate } from "./rigid-body";
import { solidOccupancyMaskPage, solidOccupancyMaskVoxel, solidOccupancyMaskWGSL, type SolidOccupancyMaskWalls } from "./solid-occupancy-mask";

/**
 * Motor-assembly contact, resident in the GPU rigid kernel.
 *
 * A motor assembly is a free hull and the massless fixtures (paddle wheels)
 * attached to it. Its collision shape is the union of convex pieces in the
 * hull's frame — the hull's own core shape, the hull's authored envelope and
 * one swept cylinder per axle that encloses every phase of its wheel — and
 * every contact on any piece acts on the hull. The fixtures carry no mass and
 * displace no extra water; they are positioned by `attachMotor` as before.
 *
 * Contacts are support-function based, so the same pieces serve every static
 * surface (the support of a union is the maximum of its pieces' supports) and
 * every pair (each piece pair is a separating-axis test). A pair of bodies is
 * travel-bounded by a speculative pass on the velocities before positions are
 * integrated: a gap that the step's motion would close is closed exactly and
 * no further, so a 40 m/s hull cannot pass through another in one step. No
 * substeps: one integration per frame.
 *
 * Each WGSL function below has a TypeScript twin of the same name and the same
 * operation order further down (`rigidContactMirror`). The twins are the test
 * oracle the CPU suite exercises (`tests/rigid-assembly-contact.test.ts`) and
 * the GPU suite compares the kernel against (`tests/rigid-motor-gpu.test.ts`).
 * Change them together.
 */

/** Points of one support feature within this distance of its deepest share one averaged contact point. */
export const RIGID_SUPPORT_TIE_M = 0.005;
/** Sequential contact sweeps per step: the kernel's, and the mirror's. */
export const RIGID_CONTACT_ITERATIONS = 6;
/** Sequential-impulse rounds over an assembly's pieces against one static plane, within each sweep. */
export const RIGID_PLANE_ITERATIONS = 16;
/** First float of a motor record's axle envelope (`RigidMotor.spare0`). */
export const RIGID_MOTOR_AXLE_FLOAT = 24;
/** Separating-axis candidates per piece pair: 3 + 3 face axes, 9 edge crosses, the centre line and two axle radials. */
export const RIGID_SAT_AXES = 18;

/** Pairs farther apart than their reaches, their step's travel and this are not solved this step. */
export const RIGID_PAIR_MARGIN_M = 0.1;
/** Assemblies the contact tables hold; a roster with more fails explicitly. */
export const RIGID_TABLE_ASSEMBLIES = 4;
/** Pieces of one assembly the contact tables hold: its core, its hull and four axles. */
export const RIGID_TABLE_PIECES = 6;

const leafCorners = (nodes: readonly { lo: readonly number[]; hi: readonly number[]; part: number }[]): HullPoint[] =>
  nodes.filter((n) => n.part >= 0).flatMap((n) => Array.from({ length: 8 }, (_, i) =>
    [(i & 1 ? n.hi : n.lo)[0]!, (i & 2 ? n.hi : n.lo)[1]!, (i & 4 ? n.hi : n.lo)[2]!] as const));

/** Every exported BVH leaf corner of the hull: the cloud the envelope must enclose. */
export const RIGID_HULL_ENVELOPE_CLOUD: readonly HullPoint[] = leafCorners(asset.hull.nodes);
/**
 * The hull envelope's vertices, rounded to f32 so the kernel and its mirror
 * read the same numbers. The bow and stern supports lie outside the buoyancy
 * box; the leaf bounds enclose every analytic part.
 */
export const RIGID_HULL_ENVELOPE: readonly HullPoint[] = convexHullVertices(RIGID_HULL_ENVELOPE_CLOUD)
  .map((p) => p.map(Math.fround) as unknown as HullPoint);
const hullLow = [0, 1, 2].map((a) => Math.min(...RIGID_HULL_ENVELOPE.map((p) => p[a]!)));
const hullHigh = [0, 1, 2].map((a) => Math.max(...RIGID_HULL_ENVELOPE.map((p) => p[a]!)));
/** The envelope's box centre, the separating-axis centre line's end. */
export const RIGID_HULL_CENTRE: HullPoint = [0, 1, 2].map((a) => Math.fround(0.5 * (hullLow[a]! + hullHigh[a]!))) as unknown as HullPoint;
/** Bounds the envelope from the body centre. */
export const RIGID_HULL_RADIUS = Math.fround(Math.max(...RIGID_HULL_ENVELOPE.map((p) => Math.hypot(...p))) * (1 + 1e-6));

/** Appearance skins that carry the hull envelope. */
export const RIGID_HULL_SKINS = Object.freeze([6, 7]);

/** One axle's swept envelope, relative to its pivot along its unit axis. */
export interface RigidAxleEnvelope {
  readonly alongMin_m: number;
  readonly alongMax_m: number;
  readonly radius_m: number;
}

/**
 * The cylinder about a fixture's axle that encloses every phase of its
 * rendered parts: the axial extent of the parts, and their largest distance
 * from the axis. The wheel turns about that axis, so the envelope does not
 * depend on the motor angle.
 */
export function rigidAxleEnvelope(body: RigidBodyDescription): RigidAxleEnvelope {
  const m = body.motorAttachment;
  if (!m) throw new Error(`${body.id} has no motor attachment`);
  const half = rigidAppearanceSkin(body)?.half ?? sceneShapeRenderHalfExtent_m(body.shape, body.dimensions_m);
  const localPoints: HullPoint[] = body.appearance === "paddleboat-wheel"
    ? leafCorners(asset.wheel.nodes)
    : Array.from({ length: 8 }, (_, i) => [i & 1 ? half[0] : -half[0], i & 2 ? half[1] : -half[1], i & 4 ? half[2] : -half[2]] as const);
  const length = Math.hypot(m.axis.x, m.axis.y, m.axis.z);
  if (!(length > 0)) throw new Error(`${body.id} has a zero motor axis`);
  const axis = { x: m.axis.x / length, y: m.axis.y / length, z: m.axis.z / length };
  let alongMin = Infinity, alongMax = -Infinity, radius = 0;
  for (const local of localPoints) {
    const r = quaternionRotate(m.orientation, { x: local[0], y: local[1], z: local[2] });
    const p = { x: m.offset_m.x + r.x, y: m.offset_m.y + r.y, z: m.offset_m.z + r.z };
    const along = p.x * axis.x + p.y * axis.y + p.z * axis.z;
    alongMin = Math.min(alongMin, along); alongMax = Math.max(alongMax, along);
    radius = Math.max(radius, Math.hypot(p.x - axis.x * along, p.y - axis.y * along, p.z - axis.z * along));
  }
  return { alongMin_m: alongMin, alongMax_m: alongMax, radius_m: radius };
}

/**
 * The four `RigidMotor.spare0` floats of each body: [alongMin, alongMax,
 * radius, 1] for the first fixture of each distinct axle of a parent, zeros
 * otherwise. The blades of one wheel share one envelope; it is a contact
 * piece once.
 */
export function rigidAxleLanes(descriptions: readonly RigidBodyDescription[]): Float32Array[] {
  const seen = new Set<string>();
  return descriptions.map((body) => {
    const m = body.motorAttachment;
    if (!m) return new Float32Array(4);
    const e = rigidAxleEnvelope(body);
    const key = JSON.stringify([m.parentId, m.pivot_m, m.axis, e]);
    if (seen.has(key)) return new Float32Array(4);
    seen.add(key);
    return new Float32Array([e.alongMin_m, e.alongMax_m, e.radius_m, 1]);
  });
}

const f = (v: number) => {
  const s = Math.fround(v).toPrecision(9);
  return s.includes(".") || s.includes("e") ? s : `${s}.0`;
};
const v3 = (p: HullPoint) => `vec3f(${p.map(f).join(",")})`;

/**
 * Kernel source. Needs, from the rigid shader: `RigidBody`, `motors`,
 * `params`, `qRotate`/`qInverseRotate`, `safeNormalize`, `velocityAt`,
 * `angularTerm`, `applyImpulse`, `resolveStaticContactArm`, `rigidShapeTag`,
 * `rigidShapeSupportRadius` and `RIGID_SHAPE_BOX`.
 */
export function rigidAssemblyContactWGSL(capacity: number): string {
  return /* wgsl */ `
const RIGID_SUPPORT_TIE:f32=${f(RIGID_SUPPORT_TIE_M)};
const RIGID_PLANE_ITERATIONS:u32=${RIGID_PLANE_ITERATIONS}u;
const RIGID_HULL_ENVELOPE_COUNT:u32=${RIGID_HULL_ENVELOPE.length}u;
const RIGID_HULL_ENVELOPE=array<vec3f,${RIGID_HULL_ENVELOPE.length}>(${RIGID_HULL_ENVELOPE.map(v3).join(",")});
const RIGID_HULL_CENTRE=${v3(RIGID_HULL_CENTRE)};
const RIGID_HULL_RADIUS:f32=${f(RIGID_HULL_RADIUS)};
struct RigidSupport { point:vec3f, reach:f32 }
struct RigidPairContact { normal:vec3f, depth:f32, point:vec3f }
// The bound of each body's compound from its centre; zero for an ordinary body.
var<private> rigidAssembly:array<f32,${capacity}>;
fn rigidHasHull(index:u32)->bool{
  let skin=i32(round(motors[index].renderHalfShape.w));
  return motors[index].pivotParent.w<.5&&(${RIGID_HULL_SKINS.map((s) => `skin==${s}`).join("||")});
}
fn rigidAxleOf(index:u32,fixture:u32)->bool{
  let m=motors[fixture];
  return m.pivotParent.w>.5&&u32(round(m.pivotParent.w))==index+1u&&m.spare0.w>.5;
}
fn rigidAxleRadius(m:RigidMotor)->f32{
  return length(m.pivotParent.xyz)+max(abs(m.spare0.x),abs(m.spare0.y))+m.spare0.z;
}
fn rigidAssemblyBuild(count:u32){
  for(var index=0u;index<${capacity}u;index++){rigidAssembly[index]=0.0;}
  for(var index=0u;index<count;index++){
    if(rigidHasHull(index)){rigidAssembly[index]=max(rigidAssembly[index],max(bodies[index].dimensions.w,RIGID_HULL_RADIUS));}
    let m=motors[index];
    if(m.pivotParent.w>.5&&m.spare0.w>.5){
      let parent=u32(round(m.pivotParent.w))-1u;
      if(parent<count){rigidAssembly[parent]=max(rigidAssembly[parent],max(bodies[parent].dimensions.w,rigidAxleRadius(m)));}
    }
  }
}
fn rigidIsAssembly(index:u32)->bool{return rigidAssembly[index]>0.0;}
fn rigidContactRadius(body:RigidBody,index:u32)->f32{return select(body.dimensions.w,rigidAssembly[index],rigidIsAssembly(index));}
// Piece 0 is the body's own shape, 1 the hull envelope, 2+j the axle of fixture j.
fn rigidPieceExists(index:u32,piece:u32)->bool{
  if(piece==0u){return true;}
  if(piece==1u){return rigidHasHull(index);}
  return rigidAxleOf(index,piece-2u);
}
fn rigidBoxSupport(half:vec3f,u:vec3f)->vec3f{
  var p=vec3f(0);
  for(var axis=0u;axis<3u;axis++){
    if(2.0*half[axis]*abs(u[axis])>=RIGID_SUPPORT_TIE){p[axis]=select(-half[axis],half[axis],u[axis]>0.0);}
  }
  return p;
}
fn rigidCoreSupport(body:RigidBody,u:vec3f)->RigidSupport{
  let tag=rigidShapeTag(body);let reach=rigidShapeSupportRadius(tag,body.dimensions.xyz,u);
  if(tag==RIGID_SHAPE_BOX){return RigidSupport(rigidBoxSupport(.5*body.dimensions.xyz,u),reach);}
  return RigidSupport(u*reach,reach);
}
fn rigidHullReach(u:vec3f)->f32{
  var best=-1e30;
  for(var i=0u;i<RIGID_HULL_ENVELOPE_COUNT;i++){best=max(best,dot(RIGID_HULL_ENVELOPE[i],u));}
  return best;
}
fn rigidHullSupport(u:vec3f)->RigidSupport{
  let best=rigidHullReach(u);var sum=vec3f(0);var n=0.0;
  for(var i=0u;i<RIGID_HULL_ENVELOPE_COUNT;i++){
    let p=RIGID_HULL_ENVELOPE[i];
    if(dot(p,u)>=best-RIGID_SUPPORT_TIE){sum+=p;n+=1.0;}
  }
  return RigidSupport(sum/max(n,1.0),best);
}
fn rigidAxleSupport(m:RigidMotor,u:vec3f)->RigidSupport{
  let axis=safeNormalize(m.axisSpeed.xyz,vec3f(0,0,1));let e=m.spare0.xyz;
  let along=dot(axis,u);let radial=u-axis*along;let radialLength=length(radial);
  let reach=dot(m.pivotParent.xyz,u)+max(e.x*along,e.y*along)+e.z*radialLength;
  var t=.5*(e.x+e.y);if((e.y-e.x)*abs(along)>=RIGID_SUPPORT_TIE){t=select(e.x,e.y,along>0.0);}
  var rim=vec3f(0);if(2.0*e.z*radialLength>=RIGID_SUPPORT_TIE){rim=radial*(e.z/radialLength);}
  return RigidSupport(m.pivotParent.xyz+axis*t+rim,reach);
}
fn rigidPieceSupport(body:RigidBody,piece:u32,direction:vec3f)->RigidSupport{
  let u=qInverseRotate(body.orientation,direction);var local=RigidSupport(vec3f(0),0.0);
  if(piece==0u){local=rigidCoreSupport(body,u);}
  else if(piece==1u){local=rigidHullSupport(u);}
  else{local=rigidAxleSupport(motors[piece-2u],u);}
  return RigidSupport(qRotate(body.orientation,local.point),local.reach);
}
fn rigidPieceReach(body:RigidBody,piece:u32,direction:vec3f)->f32{
  let u=qInverseRotate(body.orientation,direction);
  if(piece==0u){return rigidShapeSupportRadius(rigidShapeTag(body),body.dimensions.xyz,u);}
  if(piece==1u){return rigidHullReach(u);}
  return rigidAxleSupport(motors[piece-2u],u).reach;
}
fn rigidPieceCentre(body:RigidBody,piece:u32)->vec3f{
  if(piece==0u){return body.positionShape.xyz;}
  if(piece==1u){return body.positionShape.xyz+qRotate(body.orientation,RIGID_HULL_CENTRE);}
  let m=motors[piece-2u];let axis=safeNormalize(m.axisSpeed.xyz,vec3f(0,0,1));
  return body.positionShape.xyz+qRotate(body.orientation,m.pivotParent.xyz+axis*(.5*(m.spare0.x+m.spare0.y)));
}
fn rigidPieceAxle(body:RigidBody,piece:u32)->vec3f{
  if(piece<2u){return vec3f(0);}
  return qRotate(body.orientation,safeNormalize(motors[piece-2u].axisSpeed.xyz,vec3f(0,0,1)));
}
fn rigidBodyAxis(body:RigidBody,axis:u32)->vec3f{
  var e=vec3f(0);e[axis]=1.0;return qRotate(body.orientation,e);
}
fn rigidSatAxis(a:RigidBody,ap:u32,b:RigidBody,bp:u32,delta:vec3f,k:u32)->vec3f{
  if(k<3u){return rigidBodyAxis(a,k);}
  if(k<6u){return rigidBodyAxis(b,k-3u);}
  if(k<15u){return cross(rigidBodyAxis(a,(k-6u)/3u),rigidBodyAxis(b,(k-6u)%3u));}
  if(k==15u){return delta;}
  var axle=rigidPieceAxle(a,ap);if(k==17u){axle=rigidPieceAxle(b,bp);}
  return delta-axle*dot(delta,axle);
}
// The shallowest overlap over the candidate axes: positive is a penetration
// along normal (from a toward b); negative is a gap no candidate can close.
fn rigidPieceContact(a:RigidBody,ap:u32,b:RigidBody,bp:u32)->RigidPairContact{
  let delta=rigidPieceCentre(b,bp)-rigidPieceCentre(a,ap);
  var bestDepth=1e30;var bestNormal=vec3f(0,1,0);
  for(var k=0u;k<${RIGID_SAT_AXES}u;k++){
    let axis=rigidSatAxis(a,ap,b,bp,delta,k);let axisLength=length(axis);
    if(axisLength<1e-6){continue;}
    let n=axis/axisLength;let pa=dot(a.positionShape.xyz,n);let pb=dot(b.positionShape.xyz,n);
    let forward=pa+rigidPieceReach(a,ap,n)-(pb-rigidPieceReach(b,bp,-n));
    let backward=pb+rigidPieceReach(b,bp,n)-(pa-rigidPieceReach(a,ap,-n));
    if(forward<bestDepth){bestDepth=forward;bestNormal=n;}
    if(backward<bestDepth){bestDepth=backward;bestNormal=-n;}
  }
  let sa=rigidPieceSupport(a,ap,bestNormal);let sb=rigidPieceSupport(b,bp,-bestNormal);
  return RigidPairContact(bestNormal,bestDepth,.5*(a.positionShape.xyz+sa.point+b.positionShape.xyz+sb.point));
}
fn rigidResolvePair(a:ptr<function,RigidBody>,b:ptr<function,RigidBody>,contact:RigidPairContact){
  let rho=max(params.step.y,1e-9);let inverseA=(*a).inverseMassInertia.x/rho;let inverseB=(*b).inverseMassInertia.x/rho;
  let inverseTotal=inverseA+inverseB;if(inverseTotal<=0.0||contact.depth<=0.0){return;}
  let normal=contact.normal;let armA=contact.point-(*a).positionShape.xyz;let armB=contact.point-(*b).positionShape.xyz;
  (*a).positionShape=vec4f((*a).positionShape.xyz-normal*contact.depth*inverseA/inverseTotal,(*a).positionShape.w);
  (*b).positionShape=vec4f((*b).positionShape.xyz+normal*contact.depth*inverseB/inverseTotal,(*b).positionShape.w);
  var relative=velocityAt(*b,armB)-velocityAt(*a,armA);let normalSpeed=dot(relative,normal);
  if(normalSpeed>=0.0){return;}
  let restitution=select(0.0,min((*a).angularMomentumRestitution.w,(*b).angularMomentumRestitution.w),-normalSpeed>0.5);
  let denominator=max(inverseTotal+angularTerm(*a,armA,normal)+angularTerm(*b,armB,normal),1e-9);
  let normalMagnitude=-(1.0+restitution)*normalSpeed/denominator;
  applyImpulse(a,-normal*normalMagnitude,armA);applyImpulse(b,normal*normalMagnitude,armB);
  relative=velocityAt(*b,armB)-velocityAt(*a,armA);let tangentVelocity=relative-normal*dot(relative,normal);let tangentSpeed=length(tangentVelocity);
  if(tangentSpeed<=1e-8){return;}
  let tangent=tangentVelocity/tangentSpeed;let friction=sqrt(max(0.0,(*a).material.x*(*b).material.x));
  let tangentDenominator=max(inverseTotal+angularTerm(*a,armA,tangent)+angularTerm(*b,armB,tangent),1e-9);
  let magnitude=clamp(-tangentSpeed/tangentDenominator,-friction*normalMagnitude,friction*normalMagnitude);
  applyImpulse(a,-tangent*magnitude,armA);applyImpulse(b,tangent*magnitude,armB);
}
// Before integration: limit the closing speed at the witness to what the gap
// admits in one step. The pair then meets at the step's end instead of
// passing through, whatever its speed.
fn rigidSpeculativePair(a:ptr<function,RigidBody>,b:ptr<function,RigidBody>,contact:RigidPairContact,dt:f32){
  let rho=max(params.step.y,1e-9);let inverseTotal=((*a).inverseMassInertia.x+(*b).inverseMassInertia.x)/rho;
  if(inverseTotal<=0.0||contact.depth>=0.0){return;}
  let normal=contact.normal;let armA=contact.point-(*a).positionShape.xyz;let armB=contact.point-(*b).positionShape.xyz;
  let normalSpeed=dot(velocityAt(*b,armB)-velocityAt(*a,armA),normal);let allowed=contact.depth/max(dt,1e-8);
  if(normalSpeed>=allowed){return;}
  let denominator=max(inverseTotal+angularTerm(*a,armA,normal)+angularTerm(*b,armB,normal),1e-9);
  let magnitude=(allowed-normalSpeed)/denominator;
  applyImpulse(a,-normal*magnitude,armA);applyImpulse(b,normal*magnitude,armB);
}
fn rigidPairUsesAssembly(aIndex:u32,bIndex:u32)->bool{return rigidIsAssembly(aIndex)||rigidIsAssembly(bIndex);}
fn solveAssemblyPair(aIndex:u32,bIndex:u32,speculative:bool,dt:f32){
  if(motors[aIndex].pivotParent.w>.5||motors[bIndex].pivotParent.w>.5){return;}
  var a=rigidLocal[aIndex];var b=rigidLocal[bIndex];
  if(a.inverseMassInertia.x+b.inverseMassInertia.x<=0.0){return;}
  let radiusA=rigidContactRadius(a,aIndex);let radiusB=rigidContactRadius(b,bIndex);
  var travel=0.0;
  if(speculative){travel=(length(b.linearVelocity.xyz-a.linearVelocity.xyz)+length(a.angularVelocity.xyz)*radiusA+length(b.angularVelocity.xyz)*radiusB)*dt;}
  if(distance(a.positionShape.xyz,b.positionShape.xyz)>radiusA+radiusB+travel){return;}
  // The pieces in slot order, which is piece order (rigidContactTableWGSL).
  let slotsA=rigidSlotCount(aIndex);let slotsB=rigidSlotCount(bIndex);
  for(var sa=0u;sa<slotsA;sa++){
    for(var sb=0u;sb<slotsB;sb++){
      let contact=rigidPieceContactAt(a,aIndex,sa,b,bIndex,sb,speculative);
      if(speculative){rigidSpeculativePair(&a,&b,contact,dt);}else{rigidResolvePair(&a,&b,contact);}
    }
  }
  rigidLocal[aIndex]=a;rigidLocal[bIndex]=b;
}
// What a body can reach this step: its bound, and the travel of its velocity and spin.
fn rigidLinkReach(body:RigidBody,index:u32)->f32{
  let radius=rigidContactRadius(body,index);
  return radius+(length(body.linearVelocity.xyz)+length(body.angularVelocity.xyz)*radius)*params.step.x;
}
// The pairs a step solves: within reach of each other, on the kicked
// velocities, with margin to spare. It contains the speculative pass's own
// travel test, and it is what splits the bodies into independent islands.
fn rigidLinked(a:RigidBody,aIndex:u32,b:RigidBody,bIndex:u32,margin:f32)->bool{
  return distance(a.positionShape.xyz,b.positionShape.xyz)<=rigidLinkReach(a,aIndex)+rigidLinkReach(b,bIndex)+margin;
}
// A static surface, given as a plane, against every piece of an assembly at
// once. The pieces are one rigid body, so their contacts are coupled: an
// impulse at the stern wheel moves the bow. They are solved together by
// sequential impulses with accumulated, clamped normal and friction impulses
// (Catto), RIGID_PLANE_ITERATIONS rounds. The body is first lifted by the
// deepest penetration; a penetrating piece then must not approach (or
// rebounds by its restitution above 0.5 m/s, as resolveStaticContactArm), and
// a piece clear by gap may approach by at most gap this step (speculative).
fn assemblyPlaneContact(body:ptr<function,RigidBody>,index:u32,normal:vec3f,offset:f32){
  _=assemblyPieceContact(body,index,normal,offset,0xffffffffu,false,vec3f(0),vec3f(0));
}
// The same solve over the pieces in mask, for a bounded face: each arm is
// clamped into [clampLo, clampHi] when bounded (rigidAssemblySolidWGSL).
// True when it moved the body or changed its velocity.
fn assemblyPieceContact(body:ptr<function,RigidBody>,index:u32,normal:vec3f,offset:f32,mask:u32,bounded:bool,clampLo:vec3f,clampHi:vec3f)->bool{
  let rho=max(params.step.y,1e-9);let inverseMass=(*body).inverseMassInertia.x/rho;
  if(inverseMass<=0.0){return false;}
  let dt=max(params.step.x,1e-8);
  let travel=(length((*body).linearVelocity.xyz)+length((*body).angularVelocity.xyz)*rigidAssembly[index])*params.step.x;
  if(offset-(dot(normal,(*body).positionShape.xyz)-rigidAssembly[index]-travel)<=0.0){return false;}
  // By piece slot: slot order is piece order.
  let count=rigidSlotCount(index);
  // A surface whose normal is no world axis has no tabulated supports. Its
  // reach is bounded by the tabulated ones (a support function is sublinear:
  // h(d) <= sum |d_a| h(sign(d_a) e_a)), and a body whose every piece is
  // clear by that bound and cannot close the gap this step is left as the
  // solve below would leave it, without the supports.
  let inward=-normal;
  if(!bounded&&rigidAxisCode(inward)==6u&&rigidSupportTabled(*body,index,0u)){
    var bound=0.0;
    for(var slot=0u;slot<count;slot++){
      let row=RT_SUPPORT+(index*RT_PIECES+slot)*6u;
      bound=max(bound,abs(inward.x)*rigidTable[row+select(0u,1u,inward.x>0.0)].w+abs(inward.y)*rigidTable[row+select(2u,3u,inward.y>0.0)].w+abs(inward.z)*rigidTable[row+select(4u,5u,inward.z>0.0)].w);
    }
    let gap=dot(normal,(*body).positionShape.xyz)-bound-offset;
    if(gap>0.0&&dot(normal,(*body).linearVelocity.xyz)-length((*body).angularVelocity.xyz)*rigidAssembly[index]>=-gap/dt){return false;}
  }
  var arms:array<vec3f,${RIGID_TABLE_PIECES}>;var targets:array<f32,${RIGID_TABLE_PIECES}>;var penetrations:array<f32,${RIGID_TABLE_PIECES}>;
  var normalImpulses:array<f32,${RIGID_TABLE_PIECES}>;var tangentImpulses:array<vec3f,${RIGID_TABLE_PIECES}>;
  var lift=0.0;var engaged=false;
  for(var piece=0u;piece<count;piece++){
    penetrations[piece]=-1e30;
    if(((mask>>rigidSlotPiece(index,piece))&1u)==0u){continue;}
    let support=rigidSupportAt(*body,index,piece,-normal);
    arms[piece]=support.point;penetrations[piece]=offset-(dot(normal,(*body).positionShape.xyz)-support.reach);
    if(bounded){arms[piece]=clamp((*body).positionShape.xyz+support.point,clampLo,clampHi)-(*body).positionShape.xyz;}
    lift=max(lift,penetrations[piece]);
  }
  if(lift>0.0){(*body).positionShape=vec4f((*body).positionShape.xyz+normal*(lift+1e-7),(*body).positionShape.w);}
  for(var piece=0u;piece<count;piece++){
    normalImpulses[piece]=0.0;tangentImpulses[piece]=vec3f(0);
    if(penetrations[piece]<=-1e29){continue;}
    let normalSpeed=dot(velocityAt(*body,arms[piece]),normal);
    if(penetrations[piece]>0.0){targets[piece]=select(0.0,-(*body).angularMomentumRestitution.w*normalSpeed,-normalSpeed>0.5);}
    else{targets[piece]=(penetrations[piece]-max(lift,0.0))/dt;}
    if(normalSpeed<targets[piece]){engaged=true;}
  }
  if(!engaged){return lift>0.0;}
  for(var round=0u;round<RIGID_PLANE_ITERATIONS;round++){
    for(var piece=0u;piece<count;piece++){
      if(penetrations[piece]<=-1e29){continue;}
      let arm=arms[piece];
      let denominator=max(inverseMass+angularTerm(*body,arm,normal),1e-9);
      let previous=normalImpulses[piece];
      normalImpulses[piece]=max(previous+(targets[piece]-dot(velocityAt(*body,arm),normal))/denominator,0.0);
      applyImpulse(body,normal*(normalImpulses[piece]-previous),arm);
      let relative=velocityAt(*body,arm);let tangentVelocity=relative-normal*dot(relative,normal);
      let tangentSpeed=length(tangentVelocity);if(tangentSpeed<=1e-8){continue;}
      let tangent=tangentVelocity/tangentSpeed;
      let tangentDenominator=max(inverseMass+angularTerm(*body,arm,tangent),1e-9);
      let old=tangentImpulses[piece];var next=old-tangent*(tangentSpeed/tangentDenominator);
      let bound=(*body).material.x*normalImpulses[piece];let nextLength=length(next);
      if(nextLength>bound){next=next*(bound/nextLength);}
      tangentImpulses[piece]=next;applyImpulse(body,next-old,arm);
    }
  }
  return true;
}
// How far the body reaches into a surface with this normal.
fn contactReach(body:RigidBody,index:u32,normal:vec3f)->f32{
  if(!rigidIsAssembly(index)){return supportRadius(body,normal);}
  let count=rigidSlotCount(index);var reach=0.0;
  for(var slot=0u;slot<count;slot++){reach=max(reach,rigidReachAt(body,index,slot,-normal));}
  return reach;
}
fn staticPlaneContact(body:ptr<function,RigidBody>,index:u32,normal:vec3f,offset:f32){
  if(rigidIsAssembly(index)){assemblyPlaneContact(body,index,normal,offset);}else{planeContact(body,normal,offset);}
}
// A sampled surface at signed distance centreDistance from the body centre.
fn staticSurfaceContact(body:ptr<function,RigidBody>,index:u32,normal:vec3f,centreDistance:f32){
  if(rigidIsAssembly(index)){assemblyPlaneContact(body,index,normal,dot(normal,(*body).positionShape.xyz)-centreDistance);return;}
  let radius=supportRadius(*body,normal);resolveStaticContact(body,normal,radius-centreDistance,radius);
}
`;
}

/**
 * Where the contact tables live in `rigidTable`, in vec4f rows. The host
 * writes the header at a roster sync (`rigidContactTableHeader`); the wide
 * `rigidTables` launch fills the rest each step and `integrate` reads it.
 *
 *   0                 (assemblies, 0, 0, 0)
 *   1 + slot          (body index of that assembly, 0, 0, 0)
 *   slots + 2 i       body i's pieces by slot: (p0..p3), (p4, p5, assembly slot or -1, pieces)
 *   body + 10 i       body i kicked (eight lanes), then its integrated position and orientation
 *   support + ...     [body][piece slot][-x +x -y +y -z +z]: (arm, reach) at the integrated orientation
 *   pair + ...        [phase][assembly slot][other body][slot][other's slot][axis]: (n, stamp), four reaches
 *
 * Phase 0 is the kicked pose (the speculative pass), phase 1 the integrated
 * one (the sweeps). A pair's rows are those of its lower-indexed assembly.
 */
export function rigidContactTableLayout(capacity: number) {
  const slots = 1 + RIGID_TABLE_ASSEMBLIES + 3, body = 64 * Math.ceil((slots + 2 * capacity) / 64);
  const support = body + 10 * capacity, pair = support + capacity * RIGID_TABLE_PIECES * 6;
  const pairThreads = 2 * capacity * RIGID_TABLE_PIECES * RIGID_TABLE_PIECES * RIGID_SAT_AXES;
  return {
    slots, body, support, pair,
    /** vec4f rows of the whole table. */
    rows: pair + 2 * RIGID_TABLE_ASSEMBLIES * pairThreads,
    /** Threads of `rigidTables` before the pair rows: one a body, one a support row. */
    soloThreads: capacity + capacity * RIGID_TABLE_PIECES * 6,
    /** Threads of `rigidTables` for one assembly's pair rows, both phases. */
    pairThreads,
  };
}
/** Threads the `rigidTables` launch needs for a roster with this many assemblies. */
export function rigidContactTableThreads(capacity: number, assemblies: number): number {
  const layout = rigidContactTableLayout(capacity);
  return layout.soloThreads + assemblies * layout.pairThreads;
}
export const RIGID_TABLE_WORKGROUP = 64;

/**
 * The table's header for a roster's motor records (`rigidMotorRecords`): the
 * assemblies, and each body's pieces in piece order. More assemblies or pieces
 * than the table holds, or a fixture on a fixture, fail explicitly.
 */
export function rigidContactTableHeader(motors: ArrayLike<number>, count: number, capacity: number): Float32Array<ArrayBuffer> {
  const layout = rigidContactTableLayout(capacity), header = new Float32Array(4 * layout.body);
  const w = { motors: decodeMirrorMotors(motors, count) } as MirrorWorld;
  let assemblies = 0;
  for (let index = 0; index < capacity; index++) {
    const pieces: number[] = []; let slot = -1;
    if (index < count) {
      const parent = Math.round(w.motors[index]!.pivotParent[3]) - 1;
      if (parent >= 0 && (parent >= count || w.motors[parent]!.pivotParent[3] > .5)) throw new Error(`Body ${index}'s motor parent ${parent} is not a free body of the roster`);
      pieces.push(0);
      if (rigidHasHull(w, index)) pieces.push(1);
      for (let fixture = 0; fixture < count; fixture++) if (rigidAxleOf(w, index, fixture)) pieces.push(2 + fixture);
      if (pieces.length > RIGID_TABLE_PIECES) throw new Error(`Body ${index} has ${pieces.length} contact pieces; the contact tables hold ${RIGID_TABLE_PIECES}`);
      if (pieces.length > 1) {
        if (assemblies >= RIGID_TABLE_ASSEMBLIES) throw new Error(`The contact tables hold ${RIGID_TABLE_ASSEMBLIES} motor assemblies; body ${index} is one more`);
        slot = assemblies++; header[4 * (1 + slot)] = index;
      }
    }
    const at = 4 * (layout.slots + 2 * index);
    for (let k = 0; k < RIGID_TABLE_PIECES; k++) header[at + k] = pieces[k] ?? 0;
    header[at + 6] = slot; header[at + 7] = pieces.length;
  }
  header[0] = assemblies;
  return header;
}

/**
 * The tables' readers and writers. Needs `rigidAssemblyContactWGSL`. A table
 * row is a cached value of a function above, keyed by the pose it was
 * computed at: a reader that finds its body at another pose (a contact moved
 * it, a speculative impulse turned it) evaluates the function instead, so the
 * tables change where the arithmetic runs and not what it computes.
 */
export function rigidContactTableWGSL(capacity: number): string {
  const layout = rigidContactTableLayout(capacity);
  return /* wgsl */ `
const RT_ASSEMBLIES:u32=${RIGID_TABLE_ASSEMBLIES}u;const RT_PIECES:u32=${RIGID_TABLE_PIECES}u;const RT_CAPACITY:u32=${capacity}u;
const RT_SLOTS:u32=${layout.slots}u;const RT_BODY:u32=${layout.body}u;const RT_SUPPORT:u32=${layout.support}u;const RT_PAIR:u32=${layout.pair}u;
const RT_SOLO_THREADS:u32=${layout.soloThreads}u;
const RIGID_PAIR_MARGIN:f32=${f(RIGID_PAIR_MARGIN_M)};
@group(0) @binding(12) var<storage,read_write> rigidTable:array<vec4f>;
// The bodies one integrate thread solves: its island, and the neighbours it reads.
var<private> rigidLocal:array<RigidBody,${capacity}>;
fn rigidSlotCount(index:u32)->u32{return u32(rigidTable[RT_SLOTS+2u*index+1u].w);}
fn rigidSlotPiece(index:u32,slot:u32)->u32{return u32(rigidTable[RT_SLOTS+2u*index+slot/4u][slot%4u]);}
fn rigidAssemblySlot(index:u32)->i32{return i32(rigidTable[RT_SLOTS+2u*index+1u].z);}
fn rigidTableBody(index:u32)->RigidBody{
  let at=RT_BODY+10u*index;
  return RigidBody(rigidTable[at],rigidTable[at+1u],rigidTable[at+2u],rigidTable[at+3u],rigidTable[at+4u],rigidTable[at+5u],rigidTable[at+6u],rigidTable[at+7u]);
}
// 0..5 for exactly -x +x -y +y -z +z, 6 for any other direction.
fn rigidAxisCode(direction:vec3f)->u32{
  let a=abs(direction);
  if(a.x==1.0&&a.y==0.0&&a.z==0.0){return select(0u,1u,direction.x>0.0);}
  if(a.y==1.0&&a.x==0.0&&a.z==0.0){return select(2u,3u,direction.y>0.0);}
  if(a.z==1.0&&a.x==0.0&&a.y==0.0){return select(4u,5u,direction.z>0.0);}
  return 6u;
}
fn rigidAxisVector(code:u32)->vec3f{var e=vec3f(0);e[code/2u]=select(-1.0,1.0,(code&1u)==1u);return e;}
fn rigidSupportTabled(body:RigidBody,index:u32,code:u32)->bool{
  return code<6u&&rigidAssemblySlot(index)>=0&&all(body.orientation==rigidTable[RT_BODY+10u*index+9u]);
}
// rigidPieceSupport of the body's piece in this slot.
fn rigidSupportAt(body:RigidBody,index:u32,slot:u32,direction:vec3f)->RigidSupport{
  let code=rigidAxisCode(direction);
  if(rigidSupportTabled(body,index,code)){let e=rigidTable[RT_SUPPORT+(index*RT_PIECES+slot)*6u+code];return RigidSupport(e.xyz,e.w);}
  return rigidPieceSupport(body,rigidSlotPiece(index,slot),direction);
}
// rigidPieceReach of the body's piece in this slot.
fn rigidReachAt(body:RigidBody,index:u32,slot:u32,direction:vec3f)->f32{
  let code=rigidAxisCode(direction);
  if(rigidSupportTabled(body,index,code)){return rigidTable[RT_SUPPORT+(index*RT_PIECES+slot)*6u+code].w;}
  return rigidPieceReach(body,rigidSlotPiece(index,slot),direction);
}
fn rigidTableFillSupport(body:RigidBody,index:u32,slot:u32,code:u32){
  let support=rigidPieceSupport(body,rigidSlotPiece(index,slot),rigidAxisVector(code));
  rigidTable[RT_SUPPORT+(index*RT_PIECES+slot)*6u+code]=vec4f(support.point,support.reach);
}
// The rows of assembly x's slot sx against body y's slot sy.
fn rigidPairRow(x:u32,sx:u32,y:u32,sy:u32,phase:u32)->u32{
  return RT_PAIR+((((phase*RT_ASSEMBLIES+u32(rigidAssemblySlot(x)))*RT_CAPACITY+y)*RT_PIECES+sx)*RT_PIECES+sy)*${2 * RIGID_SAT_AXES}u;
}
// One candidate axis of rigidPieceContact, the lower-indexed body first as
// solveAssemblyPair calls it: the unit axis, and the four reaches along it.
fn rigidTableFillPair(bx:RigidBody,x:u32,sx:u32,by:RigidBody,y:u32,sy:u32,phase:u32,k:u32){
  let row=rigidPairRow(x,sx,y,sy,phase)+2u*k;
  var first=bx;var pf=rigidSlotPiece(x,sx);var second=by;var ps=rigidSlotPiece(y,sy);
  if(y<x){first=by;second=bx;let swap=pf;pf=ps;ps=swap;}
  let delta=rigidPieceCentre(second,ps)-rigidPieceCentre(first,pf);
  let axis=rigidSatAxis(first,pf,second,ps,delta,k);let axisLength=length(axis);
  if(axisLength<1e-6){rigidTable[row]=vec4f(0,0,0,params.gravity.w);rigidTable[row+1u]=vec4f(0);return;}
  let n=axis/axisLength;
  rigidTable[row]=vec4f(n,params.gravity.w);
  rigidTable[row+1u]=vec4f(rigidPieceReach(first,pf,n),rigidPieceReach(second,ps,-n),rigidPieceReach(second,ps,n),rigidPieceReach(first,pf,-n));
}
// rigidPieceContact of two slots, a the lower-indexed body. An axis is read
// from the table while both bodies hold the orientation it was filled at
// (and, for the three axes through the piece centres, the position) and the
// row carries this step's stamp; any other axis is evaluated here. The
// witness point is found only where the resolver reads it.
fn rigidPieceContactAt(a:RigidBody,aIndex:u32,sa:u32,b:RigidBody,bIndex:u32,sb:u32,speculative:bool)->RigidPairContact{
  let ap=rigidSlotPiece(aIndex,sa);let bp=rigidSlotPiece(bIndex,sb);
  let ka=RT_BODY+10u*aIndex;let kb=RT_BODY+10u*bIndex;
  var oriented=false;var placed=false;var phase=1u;
  if(speculative){
    phase=0u;
    oriented=all(a.orientation==rigidTable[ka+2u])&&all(b.orientation==rigidTable[kb+2u]);
    placed=all(a.positionShape.xyz==rigidTable[ka].xyz)&&all(b.positionShape.xyz==rigidTable[kb].xyz);
  }else{
    oriented=all(a.orientation==rigidTable[ka+9u])&&all(b.orientation==rigidTable[kb+9u]);
    placed=all(a.positionShape.xyz==rigidTable[ka+8u].xyz)&&all(b.positionShape.xyz==rigidTable[kb+8u].xyz);
  }
  var row=0u;
  if(oriented){if(rigidAssemblySlot(aIndex)>=0){row=rigidPairRow(aIndex,sa,bIndex,sb,phase);}else{row=rigidPairRow(bIndex,sb,aIndex,sa,phase);}}
  let delta=rigidPieceCentre(b,bp)-rigidPieceCentre(a,ap);
  var bestDepth=1e30;var bestNormal=vec3f(0,1,0);
  for(var k=0u;k<${RIGID_SAT_AXES}u;k++){
    var n=vec3f(0);var reach=vec4f(0);var tabled=false;
    if(oriented&&(k<15u||placed)){
      let e=rigidTable[row+2u*k];
      if(e.w==params.gravity.w){tabled=true;n=e.xyz;reach=rigidTable[row+2u*k+1u];}
    }
    if(tabled){if(dot(n,n)<.5){continue;}}
    else{
      let axis=rigidSatAxis(a,ap,b,bp,delta,k);let axisLength=length(axis);
      if(axisLength<1e-6){continue;}
      n=axis/axisLength;
      reach=vec4f(rigidPieceReach(a,ap,n),rigidPieceReach(b,bp,-n),rigidPieceReach(b,bp,n),rigidPieceReach(a,ap,-n));
    }
    let pa=dot(a.positionShape.xyz,n);let pb=dot(b.positionShape.xyz,n);
    let forward=pa+reach.x-(pb-reach.y);let backward=pb+reach.z-(pa-reach.w);
    if(forward<bestDepth){bestDepth=forward;bestNormal=n;}
    if(backward<bestDepth){bestDepth=backward;bestNormal=-n;}
  }
  if((speculative&&bestDepth>=0.0)||(!speculative&&bestDepth<=0.0)){return RigidPairContact(bestNormal,bestDepth,vec3f(0));}
  let witnessA=rigidPieceSupport(a,ap,bestNormal);let witnessB=rigidPieceSupport(b,bp,-bestNormal);
  return RigidPairContact(bestNormal,bestDepth,.5*(a.positionShape.xyz+witnessA.point+b.positionShape.xyz+witnessB.point));
}
`;
}

/**
 * The voxel solid a motor assembly contacts, in the container's cells: the
 * fluid's solid mask (`SolidOccupancyMask`) inside the lattice, and around it
 * the container's one-cell shell (`boxSolidVoxelShell`) less the scenery
 * sculpt cuts the solver already bakes (`collisionSculptSample`'s field). The
 * mask's wall pages are fixed for the life of its image, so the shell is read
 * as the shell patches and the cut field instead: one edited surface, the
 * same one the ordinary bodies' terrain contact and the renderer cut.
 */
export interface RigidAssemblySolidLayout {
  /** The container lattice, cells an axis. */
  readonly cells: readonly [number, number, number];
  readonly origin_m: readonly [number, number, number];
  readonly cellSize_m: readonly [number, number, number];
  /** The mask's walls; their origin is the window page of cell (0, 0, 0). */
  readonly walls: SolidOccupancyMaskWalls;
  /** Whether the shell has its floor slab: false where a terrain plane is the floor. */
  readonly floor: boolean;
}
/** Whether the layout's walls are a tank's, the only walls with a box shell around them. */
export function rigidAssemblySolidTank(layout: RigidAssemblySolidLayout): boolean {
  return layout.walls.policy === 1 || layout.walls.policy === 2;
}
const RIGID_SOLID_PAGE = 32;
/** Face planes an assembly's voxel contact gathers a sweep; one past them is solved at once. */
export const RIGID_SOLID_PLANES = 16;

/**
 * Voxel-solid contact for assemblies (`assemblyVoxelContact`). Needs
 * `rigidAssemblyContactWGSL` and, over `buffer` (the mask image from word 0,
 * the sculpt arena's pointer in its last words), `sculptCollisionWGSL`.
 *
 * The region a step can reach — the pieces' bounds, swept by the body's
 * velocity and spin over one step — is split into boxes the contact solves:
 * a shell slab with no cut page in the region is one box; anywhere else the
 * solid cells are scanned in rows, and a run of solid cells whose side faces
 * are exposed alike is one box. A box contacts through the exposed face of
 * least penetration (assemblyPieceContact, over the pieces that can reach it
 * this step, their arms clamped onto the face); a box separated along a
 * face its neighbour covers is left to that neighbour.
 */
export function rigidAssemblySolidWGSL(layout: RigidAssemblySolidLayout, buffer: string, capacity: number): string {
  if (!rigidAssemblySolidTank(layout)) throw new Error(`Assembly voxel contact needs a tank's walls, not policy ${layout.walls.policy}`);
  const [nx, ny, nz] = layout.cells, top = layout.walls.policy === 1;
  const window = layout.walls.origin.map((p) => RIGID_SOLID_PAGE * p);
  const fv = (v: readonly number[]) => `vec3f(${v.map(f).join(",")})`;
  return /* wgsl */ `
${solidOccupancyMaskWGSL(buffer, "rigidMaskPage", "rigidMaskVoxel", false, layout.walls)}
const RS_N=vec3i(${nx},${ny},${nz});
const RS_WINDOW=vec3i(${window.join(",")});
const RS_ORIGIN=${fv(layout.origin_m)};
const RS_H=${fv(layout.cellSize_m)};
var<private> rsCutPage=vec3i(2147483647);var<private> rsCutSlot=0xffffffffu;
var<private> rsPieceLo:array<vec3f,${capacity + 2}>;var<private> rsPieceHi:array<vec3f,${capacity + 2}>;
fn rsFloor8(c:vec3i)->vec3i{return select(c/8,(c-vec3i(7))/8,c<vec3i(0));}
fn rsUnit(a:u32)->vec3i{var e=vec3i(0);e[a]=1;return e;}
fn rsCut(c:vec3i)->bool{
  if(!collisionSculptActive()){return false;}
  let page=rsFloor8(c);
  if(any(page!=rsCutPage)){rsCutPage=page;rsCutSlot=sculptFieldSlot(page);}
  if(rsCutSlot==0xffffffffu){return false;}
  return sculptFieldValue(rsCutSlot,vec3u(c-page*8))<0.0;
}
fn rsShell(c:vec3i)->bool{
  let inX=c.x>=0&&c.x<RS_N.x;let inY=c.y>=0&&c.y<RS_N.y;let inZ=c.z>=0&&c.z<RS_N.z;
  if((c.x==-1||c.x==RS_N.x)&&inY&&inZ){return true;}
  if((c.z==-1||c.z==RS_N.z)&&inX&&inY){return true;}
  ${layout.floor ? "if(c.y==-1&&inX&&inZ){return true;}" : ""}
  ${top ? "if(c.y==RS_N.y&&inX&&inZ){return true;}" : ""}
  return false;
}
fn rsSolid(c:vec3i)->bool{
  if(all(c>=vec3i(0))&&all(c<RS_N)){return rigidMaskVoxel(c+RS_WINDOW);}
  return rsShell(c)&&!rsCut(c);
}
fn rsCutNear(a:vec3i,b:vec3i)->bool{
  if(!collisionSculptActive()){return false;}
  let lo=rsFloor8(a);let hi=rsFloor8(b-vec3i(1));
  for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
    if(sculptFieldSlot(vec3i(x,y,z))!=0xffffffffu){return true;}
  }}}
  return false;
}
fn rsBounds(body:RigidBody,index:u32){
  let count=rigidSlotCount(index);
  for(var slot=0u;slot<count;slot++){
    let piece=rigidSlotPiece(index,slot);
    var lo=vec3f(0);var hi=vec3f(0);
    for(var a=0u;a<3u;a++){var e=vec3f(0);e[a]=1.0;
      lo[a]=body.positionShape[a]-rigidReachAt(body,index,slot,-e);hi[a]=body.positionShape[a]+rigidReachAt(body,index,slot,e);}
    rsPieceLo[piece]=lo;rsPieceHi[piece]=hi;
  }
}
// A box's contact is gathered onto its face's plane: the boxes of one face
// (the rows of a wall, the slices of a block) are one contact, its pieces
// and its clamp the union of theirs, solved once a sweep. A plane past the
// list's capacity is solved at once.
var<private> rsPlanes=0u;
var<private> rsPlaneFace:array<u32,${RIGID_SOLID_PLANES}>;var<private> rsPlaneCell:array<i32,${RIGID_SOLID_PLANES}>;var<private> rsPlaneMask:array<u32,${RIGID_SOLID_PLANES}>;
var<private> rsPlaneLo:array<vec3f,${RIGID_SOLID_PLANES}>;var<private> rsPlaneHi:array<vec3f,${RIGID_SOLID_PLANES}>;
fn rsSolve(body:ptr<function,RigidBody>,index:u32,face:u32,mask:u32,lo:vec3f,hi:vec3f){
  let a=face/2u;var normal=vec3f(0);var offset=-lo[a];normal[a]=-1.0;
  if(face%2u==1u){normal[a]=1.0;offset=hi[a];}
  var clampLo=lo;var clampHi=hi;clampLo[a]=-1e30;clampHi[a]=1e30;
  if(assemblyPieceContact(body,index,normal,offset,mask,true,clampLo,clampHi)){rsBounds(*body,index);}
}
fn rsBoxContact(body:ptr<function,RigidBody>,index:u32,first:vec3i,last:vec3i,exposed:u32){
  let lo=RS_ORIGIN+vec3f(first)*RS_H;let hi=RS_ORIGIN+vec3f(last)*RS_H;
  let count=rigidSlotCount(index);let dt=params.step.x;
  let sweep=(*body).linearVelocity.xyz*dt;let spin=length((*body).angularVelocity.xyz)*rigidAssembly[index]*dt;
  let centre=.5*(lo+hi);let half=.5*(hi-lo);
  var mask=0u;var low=vec3f(1e30);var high=vec3f(-1e30);
  for(var slot=0u;slot<count;slot++){
    let piece=rigidSlotPiece(index,slot);
    let pl=rsPieceLo[piece];let ph=rsPieceHi[piece];
    let sl=min(pl,pl+sweep)-vec3f(spin);let sh=max(ph,ph+sweep)+vec3f(spin);
    if(any(sh<lo)||any(sl>hi)){continue;}
    let n=safeNormalize(centre-rigidPieceCentre(*body,piece),vec3f(0,1,0));
    let far=dot((*body).positionShape.xyz,n)+rigidPieceReach(*body,piece,n);
    if(far+abs(dot(sweep,n))+spin<dot(centre,n)-dot(half,abs(n))){continue;}
    mask|=1u<<piece;low=min(low,pl);high=max(high,ph);
  }
  if(mask==0u){return;}
  var best=1e30;var bestFace=6u;var least=1e30;var leastFace=6u;
  for(var face=0u;face<6u;face++){
    let a=face/2u;var depth=high[a]-lo[a];if(face%2u==1u){depth=hi[a]-low[a];}
    if(depth<least){least=depth;leastFace=face;}
    if(((exposed>>face)&1u)!=0u&&depth<best){best=depth;bestFace=face;}
  }
  if(bestFace>5u||(least<0.0&&((exposed>>leastFace)&1u)==0u)){return;}
  let a=bestFace/2u;var plane=first;if(bestFace%2u==1u){plane=last;}let cell=plane[a];
  for(var k=0u;k<rsPlanes;k++){
    if(rsPlaneFace[k]==bestFace&&rsPlaneCell[k]==cell){
      rsPlaneMask[k]|=mask;rsPlaneLo[k]=min(rsPlaneLo[k],lo);rsPlaneHi[k]=max(rsPlaneHi[k],hi);return;
    }
  }
  if(rsPlanes<${RIGID_SOLID_PLANES}u){
    rsPlaneFace[rsPlanes]=bestFace;rsPlaneCell[rsPlanes]=cell;rsPlaneMask[rsPlanes]=mask;rsPlaneLo[rsPlanes]=lo;rsPlaneHi[rsPlanes]=hi;rsPlanes++;return;
  }
  rsSolve(body,index,bestFace,mask,lo,hi);
}
fn rsCell(u:u32,t:i32,v:u32,s:i32,w:u32,r:i32)->vec3i{var c=vec3i(0);c[u]=t;c[v]=s;c[w]=r;return c;}
fn rsSideFlags(c:vec3i,v:u32,w:u32)->u32{
  var flags=0u;let ev=rsUnit(v);let ew=rsUnit(w);
  if(!rsSolid(c-ev)){flags|=1u<<(2u*v);}
  if(!rsSolid(c+ev)){flags|=1u<<(2u*v+1u);}
  if(!rsSolid(c-ew)){flags|=1u<<(2u*w);}
  if(!rsSolid(c+ew)){flags|=1u<<(2u*w+1u);}
  return flags;
}
fn rsRuns(body:ptr<function,RigidBody>,index:u32,a:vec3i,b:vec3i,u:u32){
  let v=(u+1u)%3u;let w=(u+2u)%3u;var lo=a;var hi=b;
  for(var r=lo[w];r<hi[w];r++){for(var s=lo[v];s<hi[v];s++){
    var open=false;var start=0;var flags=0u;var lowExposed=false;
    var previous=rsSolid(rsCell(u,lo[u]-1,v,s,w,r));
    for(var t=lo[u];t<=hi[u];t++){
      let c=rsCell(u,t,v,s,w,r);var solid=false;var sides=0u;
      if(t<hi[u]){solid=rsSolid(c);if(solid){sides=rsSideFlags(c,v,w);}}
      if(open&&(!solid||sides!=flags)){
        var highExposed=!solid;if(t==hi[u]){highExposed=!rsSolid(c);}
        var faces=flags;if(lowExposed){faces|=1u<<(2u*u);}if(highExposed){faces|=1u<<(2u*u+1u);}
        let first=rsCell(u,start,v,s,w,r);let last=rsCell(u,t,v,s+1,w,r+1);
        rsBoxContact(body,index,first,last,faces);
        open=false;
      }
      if(solid&&!open){open=true;start=t;flags=sides;lowExposed=!previous;}
      previous=solid;
    }
  }}
}
fn rsSlab(slab:u32)->array<vec3i,2>{
  if(slab==0u){return array<vec3i,2>(vec3i(-1,0,0),vec3i(0,RS_N.y,RS_N.z));}
  if(slab==1u){return array<vec3i,2>(vec3i(RS_N.x,0,0),vec3i(RS_N.x+1,RS_N.y,RS_N.z));}
  if(slab==2u){return array<vec3i,2>(vec3i(0,0,-1),vec3i(RS_N.x,RS_N.y,0));}
  if(slab==3u){return array<vec3i,2>(vec3i(0,0,RS_N.z),vec3i(RS_N.x,RS_N.y,RS_N.z+1));}
  if(slab==4u){return array<vec3i,2>(vec3i(0,-1,0),vec3i(RS_N.x,0,RS_N.z));}
  return array<vec3i,2>(vec3i(0,RS_N.y,0),vec3i(RS_N.x,RS_N.y+1,RS_N.z));
}
fn assemblyVoxelContact(body:ptr<function,RigidBody>,index:u32){
  if((*body).inverseMassInertia.x<=0.0){return;}
  rsBounds(*body,index);rsPlanes=0u;
  let count=rigidSlotCount(index);let dt=params.step.x;
  let sweep=(*body).linearVelocity.xyz*dt;let spin=length((*body).angularVelocity.xyz)*rigidAssembly[index]*dt;
  var lo=vec3f(1e30);var hi=vec3f(-1e30);
  for(var slot=0u;slot<count;slot++){
    let piece=rigidSlotPiece(index,slot);
    lo=min(lo,rsPieceLo[piece]);hi=max(hi,rsPieceHi[piece]);
  }
  lo=min(lo,lo+sweep)-vec3f(spin)-RS_H;hi=max(hi,hi+sweep)+vec3f(spin)+RS_H;
  let c0=max(vec3i(floor((lo-RS_ORIGIN)/RS_H)),vec3i(-1));let c1=min(vec3i(floor((hi-RS_ORIGIN)/RS_H))+vec3i(1),RS_N+vec3i(1));
  for(var slab=0u;slab<6u;slab++){
    ${layout.floor ? "" : "if(slab==4u){continue;}"}
    ${top ? "" : "if(slab==5u){continue;}"}
    let box=rsSlab(slab);let a=max(box[0],c0);let b=min(box[1],c1);
    if(any(a>=b)){continue;}
    if(!rsCutNear(a,b)){rsBoxContact(body,index,box[0],box[1],63u);}
    else{var runAxis=0u;if(slab<2u){runAxis=2u;}rsRuns(body,index,a,b,runAxis);}
  }
  let i0=max(c0,vec3i(0));let i1=min(c1,RS_N);
  if(all(i0<i1)){
    let p0=(i0+RS_WINDOW)/${RIGID_SOLID_PAGE};let p1=(i1-vec3i(1)+RS_WINDOW)/${RIGID_SOLID_PAGE};
    for(var z=p0.z;z<=p1.z;z++){for(var y=p0.y;y<=p1.y;y++){for(var x=p0.x;x<=p1.x;x++){
      let page=vec3i(x,y,z)*${RIGID_SOLID_PAGE};
      if((rigidMaskPage(page)&0xffffu)==0u){continue;}
      rsRuns(body,index,max(i0,page-RS_WINDOW),min(i1,page+vec3i(${RIGID_SOLID_PAGE})-RS_WINDOW),0u);
    }}}
  }
  for(var k=0u;k<rsPlanes;k++){rsSolve(body,index,rsPlaneFace[k],rsPlaneMask[k],rsPlaneLo[k],rsPlaneHi[k]);}
}
`;
}

// ---------------------------------------------------------------------------
// TypeScript twin. Same functions, same operation order, f64 arithmetic.
// ---------------------------------------------------------------------------

export type V3 = [number, number, number];
export type V4 = [number, number, number, number];

/** The kernel's `RigidBody`, eight vec4 lanes (`GPU_RIGID_STATE_FLOATS`). */
export interface MirrorBody {
  positionShape: V4; dimensions: V4; orientation: V4; linearVelocity: V4;
  angularVelocity: V4; inverseMassInertia: V4; angularMomentumRestitution: V4; material: V4;
}
/** The kernel's `RigidMotor`, eight vec4 lanes (`RIGID_MOTOR_FLOATS`). */
export interface MirrorMotor {
  pivotParent: V4; offsetPhase: V4; axisSpeed: V4; orientation: V4; timing: V4; renderHalfShape: V4; spare0: V4; spare1: V4;
}
/** A static half-space: points with dot(normal, p) < offset are solid. */
export interface MirrorPlane { normal: V3; offset: number }
export interface MirrorWorld {
  bodies: MirrorBody[];
  motors: MirrorMotor[];
  /** params.step.y: the fluid density the inverse masses are scaled by. */
  rho: number;
  gravity: V3;
  /** params.gravity.w: motor time. */
  time_s: number;
  /** params.step.x: the step being taken (set by `stepRigidMirror`). */
  step_s: number;
  /** The static surfaces in the kernel's order. */
  planes: MirrorPlane[];
  /** Shape support maps by tag (`rigidShapeSupportRadius`). */
  supportRadius: (tag: number, d: V3, local: V3) => number;
  /** The voxel solid assemblies contact in place of the container's planes (`assemblyVoxelContact`). */
  solid?: MirrorSolid;
}
/** What `rigidAssemblySolidWGSL` reads, for the mirror. */
export interface MirrorSolid {
  readonly layout: RigidAssemblySolidLayout;
  /** The fluid's solid mask image, as the kernel reads it from word 0 of its boundary buffer. */
  readonly mask: ArrayLike<number>;
  /** `collisionSculptActive`. */
  readonly cutActive: boolean;
  /** Whether the sculpt field holds this page (`sculptFieldSlot`). */
  cutPage(page: readonly [number, number, number]): boolean;
  /** The field's sample at a cell's centre, in a page it holds (`sculptFieldValue`). */
  cutDistance(cell: readonly [number, number, number]): number;
  /** The world's leading planes that are the container's: an assembly takes this solid in their place. */
  readonly containerPlanes: number;
}

/**
 * `rigidShapeSupportRadius` from the shared shape table. Negative tags are
 * fluid-only interior boundaries and reach no static surface (the kernel's
 * dispatch returns zero for them); any other unknown code fails.
 */
export function mirrorShapeSupportRadius(tag: number, d: V3, local: V3): number {
  if (tag < 0) return 0;
  const kind = SCENE_SHAPES_BY_CODE.find((shape) => shape.code === tag);
  if (!kind) throw new Error(`Unknown rigid shape code ${tag}`);
  return kind.supportRadius_m({ x: d[0], y: d[1], z: d[2] }, { x: local[0], y: local[1], z: local[2] });
}

const BODY_LANES = ["positionShape", "dimensions", "orientation", "linearVelocity", "angularVelocity", "inverseMassInertia", "angularMomentumRestitution", "material"] as const;
const MOTOR_LANES = ["pivotParent", "offsetPhase", "axisSpeed", "orientation", "timing", "renderHalfShape", "spare0", "spare1"] as const;
const lanes = <T>(values: ArrayLike<number>, at: number, names: readonly string[]): T =>
  Object.fromEntries(names.map((name, i) => [name, [0, 1, 2, 3].map((k) => values[at + 4 * i + k]!)])) as T;
/** Decode `count` 32-float records, the GPU buffers' layout. */
export function decodeMirrorBodies(values: ArrayLike<number>, count: number): MirrorBody[] {
  return Array.from({ length: count }, (_, i) => lanes<MirrorBody>(values, 32 * i, BODY_LANES));
}
export function decodeMirrorMotors(values: ArrayLike<number>, count: number): MirrorMotor[] {
  return Array.from({ length: count }, (_, i) => lanes<MirrorMotor>(values, 32 * i, MOTOR_LANES));
}
export function encodeMirrorBodies(bodies: readonly MirrorBody[], out: Float64Array | Float32Array = new Float64Array(32 * bodies.length)) {
  bodies.forEach((body, i) => BODY_LANES.forEach((name, k) => out.set(body[name], 32 * i + 4 * k)));
  return out;
}

const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const mulv = (a: V3, b: V3): V3 => [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
const neg = (a: V3): V3 => [-a[0], -a[1], -a[2]];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const length = (a: V3) => Math.sqrt(dot(a, a));
const xyz = (v: V4): V3 => [v[0], v[1], v[2]];
const with3 = (a: V3, w: number): V4 => [a[0], a[1], a[2], w];
const clamp = (x: number, lo: number, hi: number) => Math.min(Math.max(x, lo), hi);

// The rigid shader's helpers.
export const qConjugate = (q: V4): V4 => [q[0], -q[1], -q[2], -q[3]];
export function qMultiply(a: V4, b: V4): V4 {
  const av: V3 = [a[1], a[2], a[3]], bv: V3 = [b[1], b[2], b[3]];
  return with3W(a[0] * b[0] - dot(av, bv), add(add(mul(bv, a[0]), mul(av, b[0])), cross(av, bv)));
}
const with3W = (w: number, v: V3): V4 => [w, v[0], v[1], v[2]];
export function qRotate(q: V4, v: V3): V3 {
  const u: V3 = [q[1], q[2], q[3]], uv = cross(u, v);
  return add(v, mul(add(mul(uv, q[0]), cross(u, uv)), 2));
}
export const qInverseRotate = (q: V4, v: V3): V3 => qRotate(qConjugate(q), v);
export function safeNormalize(v: V3, fallback: V3): V3 { const l = length(v); return l > 1e-8 ? mul(v, 1 / l) : fallback; }
function normalize4(q: V4): V4 { const l = Math.hypot(...q); return [q[0] / l, q[1] / l, q[2] / l, q[3] / l]; }
function inverseInertia(w: MirrorWorld, body: MirrorBody, v: V3): V3 {
  const rho = Math.max(w.rho, 1e-9), local = qInverseRotate(body.orientation, v);
  return qRotate(body.orientation, mul(mulv(local, [body.inverseMassInertia[1], body.inverseMassInertia[2], body.inverseMassInertia[3]]), 1 / rho));
}
const rigidShapeTag = (body: MirrorBody) => Math.round(body.positionShape[3]);
function supportRadius(w: MirrorWorld, body: MirrorBody, direction: V3): number {
  const local = qInverseRotate(body.orientation, safeNormalize(direction, [1, 0, 0]));
  return w.supportRadius(rigidShapeTag(body), xyz(body.dimensions), local);
}
const velocityAt = (body: MirrorBody, arm: V3): V3 => add(xyz(body.linearVelocity), cross(xyz(body.angularVelocity), arm));
const angularTerm = (w: MirrorWorld, body: MirrorBody, arm: V3, direction: V3) =>
  dot(cross(inverseInertia(w, body, cross(arm, direction)), arm), direction);
function applyImpulse(w: MirrorWorld, body: MirrorBody, impulse: V3, arm: V3) {
  const inverseMass = body.inverseMassInertia[0] / Math.max(w.rho, 1e-9);
  body.linearVelocity = with3(add(xyz(body.linearVelocity), mul(impulse, inverseMass)), body.linearVelocity[3]);
  body.angularMomentumRestitution = with3(add(xyz(body.angularMomentumRestitution), cross(arm, impulse)), body.angularMomentumRestitution[3]);
  body.angularVelocity = with3(inverseInertia(w, body, xyz(body.angularMomentumRestitution)), body.angularVelocity[3]);
}
export function resolveStaticContactArm(w: MirrorWorld, body: MirrorBody, normal: V3, penetration: number, arm: V3) {
  const inverseMass = body.inverseMassInertia[0] / Math.max(w.rho, 1e-9);
  if (inverseMass <= 0 || penetration <= 0) return;
  body.positionShape = with3(add(xyz(body.positionShape), mul(normal, penetration + 1e-7)), body.positionShape[3]);
  let relative = velocityAt(body, arm);
  const normalSpeed = dot(relative, normal); if (normalSpeed >= 0) return;
  const restitution = -normalSpeed > 0.5 ? body.angularMomentumRestitution[3] : 0;
  const denominator = Math.max(inverseMass + angularTerm(w, body, arm, normal), 1e-9);
  const normalMagnitude = -(1 + restitution) * normalSpeed / denominator;
  applyImpulse(w, body, mul(normal, normalMagnitude), arm);
  relative = velocityAt(body, arm); const tangentVelocity = sub(relative, mul(normal, dot(relative, normal)));
  const tangentSpeed = length(tangentVelocity); if (tangentSpeed <= 1e-8) return;
  const tangent = mul(tangentVelocity, 1 / tangentSpeed);
  const tangentDenominator = Math.max(inverseMass + angularTerm(w, body, arm, tangent), 1e-9);
  const tangentMagnitude = clamp(-tangentSpeed / tangentDenominator, -body.material[0] * normalMagnitude, body.material[0] * normalMagnitude);
  applyImpulse(w, body, mul(tangent, tangentMagnitude), arm);
}
const resolveStaticContact = (w: MirrorWorld, body: MirrorBody, normal: V3, penetration: number, radius: number) =>
  resolveStaticContactArm(w, body, normal, penetration, mul(normal, -radius));
function planeContact(w: MirrorWorld, body: MirrorBody, normal: V3, offset: number) {
  const inverseMass = body.inverseMassInertia[0] / Math.max(w.rho, 1e-9);
  if (inverseMass <= 0) return;
  const radius = supportRadius(w, body, normal), penetration = offset - (dot(normal, xyz(body.positionShape)) - radius);
  if (penetration <= 0) return;
  resolveStaticContact(w, body, normal, penetration, radius);
}
function solveBodyPair(w: MirrorWorld, aIndex: number, bIndex: number) {
  if (w.motors[aIndex]!.pivotParent[3] > .5 || w.motors[bIndex]!.pivotParent[3] > .5) return;
  const a = w.bodies[aIndex]!, b = w.bodies[bIndex]!, rho = Math.max(w.rho, 1e-9);
  const inverseA = a.inverseMassInertia[0] / rho, inverseB = b.inverseMassInertia[0] / rho, inverseTotal = inverseA + inverseB;
  if (inverseTotal <= 0) return;
  const delta = sub(xyz(b.positionShape), xyz(a.positionShape)), distance = length(delta);
  const normal: V3 = distance > 1e-8 ? mul(delta, 1 / distance) : [1, 0, 0];
  const radiusA = a.dimensions[3], radiusB = b.dimensions[3], penetration = radiusA + radiusB - distance;
  if (penetration <= 0) return;
  a.positionShape = with3(sub(xyz(a.positionShape), mul(normal, penetration * inverseA / inverseTotal)), a.positionShape[3]);
  b.positionShape = with3(add(xyz(b.positionShape), mul(normal, penetration * inverseB / inverseTotal)), b.positionShape[3]);
  const armA = mul(normal, radiusA), armB = mul(normal, -radiusB);
  let relative = sub(velocityAt(b, armB), velocityAt(a, armA)); const normalSpeed = dot(relative, normal);
  if (normalSpeed < 0) {
    const restitution = -normalSpeed > 0.5 ? Math.min(a.angularMomentumRestitution[3], b.angularMomentumRestitution[3]) : 0;
    const denominator = Math.max(inverseTotal + angularTerm(w, a, armA, normal) + angularTerm(w, b, armB, normal), 1e-9);
    const normalMagnitude = -(1 + restitution) * normalSpeed / denominator;
    applyImpulse(w, a, mul(normal, -normalMagnitude), armA); applyImpulse(w, b, mul(normal, normalMagnitude), armB);
    relative = sub(velocityAt(b, armB), velocityAt(a, armA));
    const tangentVelocity = sub(relative, mul(normal, dot(relative, normal))), tangentSpeed = length(tangentVelocity);
    if (tangentSpeed > 1e-8) {
      const tangent = mul(tangentVelocity, 1 / tangentSpeed), friction = Math.sqrt(Math.max(0, a.material[0] * b.material[0]));
      const tangentDenominator = Math.max(inverseTotal + angularTerm(w, a, armA, tangent) + angularTerm(w, b, armB, tangent), 1e-9);
      const magnitude = clamp(-tangentSpeed / tangentDenominator, -friction * normalMagnitude, friction * normalMagnitude);
      applyImpulse(w, a, mul(tangent, -magnitude), armA); applyImpulse(w, b, mul(tangent, magnitude), armB);
    }
  }
}

// The assembly functions.
interface MirrorSupport { point: V3; reach: number }
interface MirrorPairContact { normal: V3; depth: number; point: V3 }
const HULL: V3[] = RIGID_HULL_ENVELOPE.map((p) => [p[0], p[1], p[2]]);
const HULL_CENTRE: V3 = [...RIGID_HULL_CENTRE] as V3;
const BOX_TAG = sceneShapeCode("box");

export function rigidHasHull(w: MirrorWorld, index: number): boolean {
  const skin = Math.round(w.motors[index]!.renderHalfShape[3]);
  return w.motors[index]!.pivotParent[3] < .5 && RIGID_HULL_SKINS.includes(skin);
}
function rigidAxleOf(w: MirrorWorld, index: number, fixture: number): boolean {
  const m = w.motors[fixture]!;
  return m.pivotParent[3] > .5 && Math.round(m.pivotParent[3]) === index + 1 && m.spare0[3] > .5;
}
const rigidAxleRadius = (m: MirrorMotor) => length(xyz(m.pivotParent)) + Math.max(Math.abs(m.spare0[0]), Math.abs(m.spare0[1])) + m.spare0[2];
/** `rigidAssembly`: each body's compound bound, zero for an ordinary body. */
export function rigidAssemblyBuild(w: MirrorWorld): number[] {
  const count = w.bodies.length, assembly = new Array<number>(count).fill(0);
  for (let index = 0; index < count; index++) {
    if (rigidHasHull(w, index)) assembly[index] = Math.max(assembly[index]!, Math.max(w.bodies[index]!.dimensions[3], RIGID_HULL_RADIUS));
    const m = w.motors[index]!;
    if (m.pivotParent[3] > .5 && m.spare0[3] > .5) {
      const parent = Math.round(m.pivotParent[3]) - 1;
      if (parent < count) assembly[parent] = Math.max(assembly[parent]!, Math.max(w.bodies[parent]!.dimensions[3], rigidAxleRadius(m)));
    }
  }
  return assembly;
}
export function rigidPieceExists(w: MirrorWorld, index: number, piece: number): boolean {
  if (piece === 0) return true;
  if (piece === 1) return rigidHasHull(w, index);
  return rigidAxleOf(w, index, piece - 2);
}
function rigidBoxSupport(half: V3, u: V3): V3 {
  const p: V3 = [0, 0, 0];
  for (let axis = 0; axis < 3; axis++) if (2 * half[axis]! * Math.abs(u[axis]!) >= RIGID_SUPPORT_TIE_M) p[axis] = u[axis]! > 0 ? half[axis]! : -half[axis]!;
  return p;
}
function rigidCoreSupport(w: MirrorWorld, body: MirrorBody, u: V3): MirrorSupport {
  const tag = rigidShapeTag(body), reach = w.supportRadius(tag, xyz(body.dimensions), u);
  if (tag === BOX_TAG) return { point: rigidBoxSupport(mul(xyz(body.dimensions), .5), u), reach };
  return { point: mul(u, reach), reach };
}
function rigidHullReach(u: V3): number { let best = -1e30; for (const p of HULL) best = Math.max(best, dot(p, u)); return best; }
function rigidHullSupport(u: V3): MirrorSupport {
  const best = rigidHullReach(u); let sum: V3 = [0, 0, 0], n = 0;
  for (const p of HULL) if (dot(p, u) >= best - RIGID_SUPPORT_TIE_M) { sum = add(sum, p); n += 1; }
  return { point: mul(sum, 1 / Math.max(n, 1)), reach: best };
}
function rigidAxleSupport(m: MirrorMotor, u: V3): MirrorSupport {
  const axis = safeNormalize(xyz(m.axisSpeed), [0, 0, 1]), e = xyz(m.spare0);
  const along = dot(axis, u), radial = sub(u, mul(axis, along)), radialLength = length(radial);
  const reach = dot(xyz(m.pivotParent), u) + Math.max(e[0] * along, e[1] * along) + e[2] * radialLength;
  let t = .5 * (e[0] + e[1]); if ((e[1] - e[0]) * Math.abs(along) >= RIGID_SUPPORT_TIE_M) t = along > 0 ? e[1] : e[0];
  let rim: V3 = [0, 0, 0]; if (2 * e[2] * radialLength >= RIGID_SUPPORT_TIE_M) rim = mul(radial, e[2] / radialLength);
  return { point: add(add(xyz(m.pivotParent), mul(axis, t)), rim), reach };
}
export function rigidPieceSupport(w: MirrorWorld, body: MirrorBody, piece: number, direction: V3): MirrorSupport {
  const u = qInverseRotate(body.orientation, direction);
  const local = piece === 0 ? rigidCoreSupport(w, body, u) : piece === 1 ? rigidHullSupport(u) : rigidAxleSupport(w.motors[piece - 2]!, u);
  return { point: qRotate(body.orientation, local.point), reach: local.reach };
}
export function rigidPieceReach(w: MirrorWorld, body: MirrorBody, piece: number, direction: V3): number {
  const u = qInverseRotate(body.orientation, direction);
  if (piece === 0) return w.supportRadius(rigidShapeTag(body), xyz(body.dimensions), u);
  if (piece === 1) return rigidHullReach(u);
  return rigidAxleSupport(w.motors[piece - 2]!, u).reach;
}
function rigidPieceCentre(w: MirrorWorld, body: MirrorBody, piece: number): V3 {
  if (piece === 0) return xyz(body.positionShape);
  if (piece === 1) return add(xyz(body.positionShape), qRotate(body.orientation, HULL_CENTRE));
  const m = w.motors[piece - 2]!, axis = safeNormalize(xyz(m.axisSpeed), [0, 0, 1]);
  return add(xyz(body.positionShape), qRotate(body.orientation, add(xyz(m.pivotParent), mul(axis, .5 * (m.spare0[0] + m.spare0[1])))));
}
function rigidPieceAxle(w: MirrorWorld, body: MirrorBody, piece: number): V3 {
  if (piece < 2) return [0, 0, 0];
  return qRotate(body.orientation, safeNormalize(xyz(w.motors[piece - 2]!.axisSpeed), [0, 0, 1]));
}
function rigidBodyAxis(body: MirrorBody, axis: number): V3 { const e: V3 = [0, 0, 0]; e[axis] = 1; return qRotate(body.orientation, e); }
function rigidSatAxis(w: MirrorWorld, a: MirrorBody, ap: number, b: MirrorBody, bp: number, delta: V3, k: number): V3 {
  if (k < 3) return rigidBodyAxis(a, k);
  if (k < 6) return rigidBodyAxis(b, k - 3);
  if (k < 15) return cross(rigidBodyAxis(a, Math.floor((k - 6) / 3)), rigidBodyAxis(b, (k - 6) % 3));
  if (k === 15) return delta;
  const axle = k === 17 ? rigidPieceAxle(w, b, bp) : rigidPieceAxle(w, a, ap);
  return sub(delta, mul(axle, dot(delta, axle)));
}
export function rigidPieceContact(w: MirrorWorld, a: MirrorBody, ap: number, b: MirrorBody, bp: number): MirrorPairContact {
  const delta = sub(rigidPieceCentre(w, b, bp), rigidPieceCentre(w, a, ap));
  let bestDepth = 1e30, bestNormal: V3 = [0, 1, 0];
  for (let k = 0; k < RIGID_SAT_AXES; k++) {
    const axis = rigidSatAxis(w, a, ap, b, bp, delta, k), axisLength = length(axis);
    if (axisLength < 1e-6) continue;
    const n = mul(axis, 1 / axisLength), pa = dot(xyz(a.positionShape), n), pb = dot(xyz(b.positionShape), n);
    const forward = pa + rigidPieceReach(w, a, ap, n) - (pb - rigidPieceReach(w, b, bp, neg(n)));
    const backward = pb + rigidPieceReach(w, b, bp, n) - (pa - rigidPieceReach(w, a, ap, neg(n)));
    if (forward < bestDepth) { bestDepth = forward; bestNormal = n; }
    if (backward < bestDepth) { bestDepth = backward; bestNormal = neg(n); }
  }
  const sa = rigidPieceSupport(w, a, ap, bestNormal), sb = rigidPieceSupport(w, b, bp, neg(bestNormal));
  return { normal: bestNormal, depth: bestDepth, point: mul(add(add(xyz(a.positionShape), sa.point), add(xyz(b.positionShape), sb.point)), .5) };
}
function rigidResolvePair(w: MirrorWorld, a: MirrorBody, b: MirrorBody, contact: MirrorPairContact) {
  const rho = Math.max(w.rho, 1e-9), inverseA = a.inverseMassInertia[0] / rho, inverseB = b.inverseMassInertia[0] / rho;
  const inverseTotal = inverseA + inverseB; if (inverseTotal <= 0 || contact.depth <= 0) return;
  const normal = contact.normal, armA = sub(contact.point, xyz(a.positionShape)), armB = sub(contact.point, xyz(b.positionShape));
  a.positionShape = with3(sub(xyz(a.positionShape), mul(normal, contact.depth * inverseA / inverseTotal)), a.positionShape[3]);
  b.positionShape = with3(add(xyz(b.positionShape), mul(normal, contact.depth * inverseB / inverseTotal)), b.positionShape[3]);
  let relative = sub(velocityAt(b, armB), velocityAt(a, armA)); const normalSpeed = dot(relative, normal);
  if (normalSpeed >= 0) return;
  const restitution = -normalSpeed > 0.5 ? Math.min(a.angularMomentumRestitution[3], b.angularMomentumRestitution[3]) : 0;
  const denominator = Math.max(inverseTotal + angularTerm(w, a, armA, normal) + angularTerm(w, b, armB, normal), 1e-9);
  const normalMagnitude = -(1 + restitution) * normalSpeed / denominator;
  applyImpulse(w, a, mul(normal, -normalMagnitude), armA); applyImpulse(w, b, mul(normal, normalMagnitude), armB);
  relative = sub(velocityAt(b, armB), velocityAt(a, armA));
  const tangentVelocity = sub(relative, mul(normal, dot(relative, normal))), tangentSpeed = length(tangentVelocity);
  if (tangentSpeed <= 1e-8) return;
  const tangent = mul(tangentVelocity, 1 / tangentSpeed), friction = Math.sqrt(Math.max(0, a.material[0] * b.material[0]));
  const tangentDenominator = Math.max(inverseTotal + angularTerm(w, a, armA, tangent) + angularTerm(w, b, armB, tangent), 1e-9);
  const magnitude = clamp(-tangentSpeed / tangentDenominator, -friction * normalMagnitude, friction * normalMagnitude);
  applyImpulse(w, a, mul(tangent, -magnitude), armA); applyImpulse(w, b, mul(tangent, magnitude), armB);
}
function rigidSpeculativePair(w: MirrorWorld, a: MirrorBody, b: MirrorBody, contact: MirrorPairContact, dt: number) {
  const rho = Math.max(w.rho, 1e-9), inverseTotal = (a.inverseMassInertia[0] + b.inverseMassInertia[0]) / rho;
  if (inverseTotal <= 0 || contact.depth >= 0) return;
  const normal = contact.normal, armA = sub(contact.point, xyz(a.positionShape)), armB = sub(contact.point, xyz(b.positionShape));
  const normalSpeed = dot(sub(velocityAt(b, armB), velocityAt(a, armA)), normal), allowed = contact.depth / Math.max(dt, 1e-8);
  if (normalSpeed >= allowed) return;
  const denominator = Math.max(inverseTotal + angularTerm(w, a, armA, normal) + angularTerm(w, b, armB, normal), 1e-9);
  const magnitude = (allowed - normalSpeed) / denominator;
  applyImpulse(w, a, mul(normal, -magnitude), armA); applyImpulse(w, b, mul(normal, magnitude), armB);
}
function solveAssemblyPair(w: MirrorWorld, assembly: readonly number[], aIndex: number, bIndex: number, speculative: boolean, dt: number) {
  if (w.motors[aIndex]!.pivotParent[3] > .5 || w.motors[bIndex]!.pivotParent[3] > .5) return;
  const a = w.bodies[aIndex]!, b = w.bodies[bIndex]!;
  if (a.inverseMassInertia[0] + b.inverseMassInertia[0] <= 0) return;
  const radiusA = assembly[aIndex]! > 0 ? assembly[aIndex]! : a.dimensions[3], radiusB = assembly[bIndex]! > 0 ? assembly[bIndex]! : b.dimensions[3];
  let travel = 0;
  if (speculative) travel = (length(sub(xyz(b.linearVelocity), xyz(a.linearVelocity))) + length(xyz(a.angularVelocity)) * radiusA + length(xyz(b.angularVelocity)) * radiusB) * dt;
  if (length(sub(xyz(a.positionShape), xyz(b.positionShape))) > radiusA + radiusB + travel) return;
  const count = w.bodies.length;
  for (let ap = 0; ap < 2 + count; ap++) {
    if (!rigidPieceExists(w, aIndex, ap)) continue;
    for (let bp = 0; bp < 2 + count; bp++) {
      if (!rigidPieceExists(w, bIndex, bp)) continue;
      const contact = rigidPieceContact(w, a, ap, b, bp);
      if (speculative) rigidSpeculativePair(w, a, b, contact, dt); else rigidResolvePair(w, a, b, contact);
    }
  }
}
function rigidLinkReach(w: MirrorWorld, assembly: readonly number[], body: MirrorBody, index: number): number {
  const radius = assembly[index]! > 0 ? assembly[index]! : body.dimensions[3];
  return radius + (length(xyz(body.linearVelocity)) + length(xyz(body.angularVelocity)) * radius) * w.step_s;
}
function rigidLinked(w: MirrorWorld, assembly: readonly number[], a: MirrorBody, aIndex: number, b: MirrorBody, bIndex: number, margin: number): boolean {
  return length(sub(xyz(a.positionShape), xyz(b.positionShape))) <= rigidLinkReach(w, assembly, a, aIndex) + rigidLinkReach(w, assembly, b, bIndex) + margin;
}
function assemblyPlaneContact(w: MirrorWorld, assembly: readonly number[], body: MirrorBody, index: number, normal: V3, offset: number) {
  assemblyPieceContact(w, assembly, body, index, normal, offset, 0xffffffff, false, [0, 0, 0], [0, 0, 0]);
}
function assemblyPieceContact(w: MirrorWorld, assembly: readonly number[], body: MirrorBody, index: number, normal: V3, offset: number,
  mask: number, bounded: boolean, clampLo: V3, clampHi: V3): boolean {
  const rho = Math.max(w.rho, 1e-9), inverseMass = body.inverseMassInertia[0] / rho;
  if (inverseMass <= 0) return false;
  const dt = Math.max(w.step_s, 1e-8);
  const travel = (length(xyz(body.linearVelocity)) + length(xyz(body.angularVelocity)) * assembly[index]!) * w.step_s;
  if (offset - (dot(normal, xyz(body.positionShape)) - assembly[index]! - travel) <= 0) return false;
  const pieces = 2 + w.bodies.length;
  const arms: V3[] = [], targets: number[] = [], penetrations: number[] = [], normalImpulses: number[] = [], tangentImpulses: V3[] = [];
  let lift = 0, engaged = false;
  for (let piece = 0; piece < pieces; piece++) {
    penetrations[piece] = -1e30;
    if (!rigidPieceExists(w, index, piece) || ((mask >>> piece) & 1) === 0) continue;
    const support = rigidPieceSupport(w, body, piece, neg(normal));
    arms[piece] = support.point; penetrations[piece] = offset - (dot(normal, xyz(body.positionShape)) - support.reach);
    if (bounded) {
      const p = xyz(body.positionShape), world = add(p, support.point);
      arms[piece] = sub([0, 1, 2].map((a) => clamp(world[a]!, clampLo[a]!, clampHi[a]!)) as V3, p);
    }
    lift = Math.max(lift, penetrations[piece]!);
  }
  if (lift > 0) body.positionShape = with3(add(xyz(body.positionShape), mul(normal, lift + 1e-7)), body.positionShape[3]);
  for (let piece = 0; piece < pieces; piece++) {
    normalImpulses[piece] = 0; tangentImpulses[piece] = [0, 0, 0];
    if (penetrations[piece]! <= -1e29) continue;
    const normalSpeed = dot(velocityAt(body, arms[piece]!), normal);
    targets[piece] = penetrations[piece]! > 0 ? (-normalSpeed > 0.5 ? -body.angularMomentumRestitution[3] * normalSpeed : 0) : (penetrations[piece]! - Math.max(lift, 0)) / dt;
    if (normalSpeed < targets[piece]!) engaged = true;
  }
  if (!engaged) return lift > 0;
  for (let round = 0; round < RIGID_PLANE_ITERATIONS; round++) {
    for (let piece = 0; piece < pieces; piece++) {
      if (penetrations[piece]! <= -1e29) continue;
      const arm = arms[piece]!;
      const denominator = Math.max(inverseMass + angularTerm(w, body, arm, normal), 1e-9);
      const previous = normalImpulses[piece]!;
      normalImpulses[piece] = Math.max(previous + (targets[piece]! - dot(velocityAt(body, arm), normal)) / denominator, 0);
      applyImpulse(w, body, mul(normal, normalImpulses[piece]! - previous), arm);
      const relative = velocityAt(body, arm), tangentVelocity = sub(relative, mul(normal, dot(relative, normal)));
      const tangentSpeed = length(tangentVelocity); if (tangentSpeed <= 1e-8) continue;
      const tangent = mul(tangentVelocity, 1 / tangentSpeed);
      const tangentDenominator = Math.max(inverseMass + angularTerm(w, body, arm, tangent), 1e-9);
      const old = tangentImpulses[piece]!; let next = sub(old, mul(tangent, tangentSpeed / tangentDenominator));
      const bound = body.material[0] * normalImpulses[piece]!, nextLength = length(next);
      if (nextLength > bound) next = mul(next, bound / nextLength);
      tangentImpulses[piece] = next; applyImpulse(w, body, sub(next, old), arm);
    }
  }
  return true;
}
function staticPlaneContact(w: MirrorWorld, assembly: readonly number[], body: MirrorBody, index: number, normal: V3, offset: number) {
  if (assembly[index]! > 0) assemblyPlaneContact(w, assembly, body, index, normal, offset); else planeContact(w, body, normal, offset);
}

// `rigidAssemblySolidWGSL`'s twin.
type V3i = [number, number, number];
function rsFloor8(c: V3i): V3i { return c.map((v) => Math.floor(v / 8)) as V3i; }
function rsUnit(a: number): V3i { const e: V3i = [0, 0, 0]; e[a] = 1; return e; }
const addi = (a: V3i, b: V3i): V3i => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const subi = (a: V3i, b: V3i): V3i => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
function rsCut(s: MirrorSolid, c: V3i): boolean {
  if (!s.cutActive) return false;
  if (!s.cutPage(rsFloor8(c))) return false;
  return s.cutDistance(c) < 0;
}
function rsShell(s: MirrorSolid, c: V3i): boolean {
  const n = s.layout.cells;
  const inX = c[0] >= 0 && c[0] < n[0], inY = c[1] >= 0 && c[1] < n[1], inZ = c[2] >= 0 && c[2] < n[2];
  if ((c[0] === -1 || c[0] === n[0]) && inY && inZ) return true;
  if ((c[2] === -1 || c[2] === n[2]) && inX && inY) return true;
  if (s.layout.floor && c[1] === -1 && inX && inZ) return true;
  if (s.layout.walls.policy === 1 && c[1] === n[1] && inX && inZ) return true;
  return false;
}
function rsWindow(s: MirrorSolid): V3i { return s.layout.walls.origin.map((p) => RIGID_SOLID_PAGE * p) as V3i; }
function rsSolid(s: MirrorSolid, c: V3i): boolean {
  const n = s.layout.cells;
  if (c.every((v, a) => v >= 0 && v < n[a]!)) { const p = addi(c, rsWindow(s)); return solidOccupancyMaskVoxel(s.mask, p[0], p[1], p[2]); }
  return rsShell(s, c) && !rsCut(s, c);
}
function rsCutNear(s: MirrorSolid, a: V3i, b: V3i): boolean {
  if (!s.cutActive) return false;
  const lo = rsFloor8(a), hi = rsFloor8(subi(b, [1, 1, 1]));
  for (let z = lo[2]; z <= hi[2]; z++) for (let y = lo[1]; y <= hi[1]; y++) for (let x = lo[0]; x <= hi[0]; x++) if (s.cutPage([x, y, z])) return true;
  return false;
}
interface PieceBounds { lo: V3[]; hi: V3[] }
function rsBounds(w: MirrorWorld, body: MirrorBody, index: number, bounds: PieceBounds) {
  for (let piece = 0; piece < 2 + w.bodies.length; piece++) {
    if (!rigidPieceExists(w, index, piece)) continue;
    const lo: V3 = [0, 0, 0], hi: V3 = [0, 0, 0];
    for (let a = 0; a < 3; a++) {
      const e: V3 = [0, 0, 0]; e[a] = 1;
      lo[a] = body.positionShape[a]! - rigidPieceReach(w, body, piece, neg(e)); hi[a] = body.positionShape[a]! + rigidPieceReach(w, body, piece, e);
    }
    bounds.lo[piece] = lo; bounds.hi[piece] = hi;
  }
}
interface PlaneList { face: number[]; cell: number[]; mask: number[]; lo: V3[]; hi: V3[] }
function rsSolve(w: MirrorWorld, assembly: readonly number[], body: MirrorBody, index: number, bounds: PieceBounds, face: number, mask: number, lo: V3, hi: V3) {
  const a = face >> 1, normal: V3 = [0, 0, 0]; normal[a] = face % 2 === 1 ? 1 : -1;
  const offset = face % 2 === 1 ? hi[a]! : -lo[a]!;
  const clampLo: V3 = [...lo], clampHi: V3 = [...hi]; clampLo[a] = -1e30; clampHi[a] = 1e30;
  if (assemblyPieceContact(w, assembly, body, index, normal, offset, mask >>> 0, true, clampLo, clampHi)) rsBounds(w, body, index, bounds);
}
function rsBoxContact(w: MirrorWorld, assembly: readonly number[], body: MirrorBody, index: number, bounds: PieceBounds, planes: PlaneList,
  s: MirrorSolid, first: V3i, last: V3i, exposed: number) {
  const origin = s.layout.origin_m, h = s.layout.cellSize_m;
  const lo = [0, 1, 2].map((a) => Math.fround(origin[a]! + first[a]! * h[a]!)) as V3, hi = [0, 1, 2].map((a) => Math.fround(origin[a]! + last[a]! * h[a]!)) as V3;
  const count = w.bodies.length, dt = w.step_s;
  const sweep = mul(xyz(body.linearVelocity), dt), spin = length(xyz(body.angularVelocity)) * assembly[index]! * dt;
  const centre = mul(add(lo, hi), .5), half = mul(sub(hi, lo), .5);
  let mask = 0, low: V3 = [1e30, 1e30, 1e30], high: V3 = [-1e30, -1e30, -1e30];
  for (let piece = 0; piece < 2 + count; piece++) {
    if (!rigidPieceExists(w, index, piece)) continue;
    const pl = bounds.lo[piece]!, ph = bounds.hi[piece]!;
    const sl = [0, 1, 2].map((a) => Math.min(pl[a]!, pl[a]! + sweep[a]!) - spin), sh = [0, 1, 2].map((a) => Math.max(ph[a]!, ph[a]! + sweep[a]!) + spin);
    if ([0, 1, 2].some((a) => sh[a]! < lo[a]!) || [0, 1, 2].some((a) => sl[a]! > hi[a]!)) continue;
    const n = safeNormalize(sub(centre, rigidPieceCentre(w, body, piece)), [0, 1, 0]);
    const far = dot(xyz(body.positionShape), n) + rigidPieceReach(w, body, piece, n);
    if (far + Math.abs(dot(sweep, n)) + spin < dot(centre, n) - dot(half, n.map(Math.abs) as V3)) continue;
    mask |= 1 << piece; low = [0, 1, 2].map((a) => Math.min(low[a]!, pl[a]!)) as V3; high = [0, 1, 2].map((a) => Math.max(high[a]!, ph[a]!)) as V3;
  }
  if (mask === 0) return;
  let best = 1e30, bestFace = 6, least = 1e30, leastFace = 6;
  for (let face = 0; face < 6; face++) {
    const a = face >> 1, depth = face % 2 === 1 ? hi[a]! - low[a]! : high[a]! - lo[a]!;
    if (depth < least) { least = depth; leastFace = face; }
    if (((exposed >>> face) & 1) !== 0 && depth < best) { best = depth; bestFace = face; }
  }
  if (bestFace > 5 || (least < 0 && ((exposed >>> leastFace) & 1) === 0)) return;
  const a = bestFace >> 1, cell = (bestFace % 2 === 1 ? last : first)[a]!;
  for (let k = 0; k < planes.face.length; k++) {
    if (planes.face[k] === bestFace && planes.cell[k] === cell) {
      planes.mask[k] = (planes.mask[k]! | mask) >>> 0;
      planes.lo[k] = [0, 1, 2].map((i) => Math.min(planes.lo[k]![i]!, lo[i]!)) as V3; planes.hi[k] = [0, 1, 2].map((i) => Math.max(planes.hi[k]![i]!, hi[i]!)) as V3;
      return;
    }
  }
  if (planes.face.length < RIGID_SOLID_PLANES) {
    planes.face.push(bestFace); planes.cell.push(cell); planes.mask.push(mask >>> 0); planes.lo.push(lo); planes.hi.push(hi); return;
  }
  rsSolve(w, assembly, body, index, bounds, bestFace, mask, lo, hi);
}
const rsCell = (u: number, t: number, v: number, s: number, wAxis: number, r: number): V3i => { const c: V3i = [0, 0, 0]; c[u] = t; c[v] = s; c[wAxis] = r; return c; };
function rsSideFlags(s: MirrorSolid, c: V3i, v: number, wAxis: number): number {
  let flags = 0; const ev = rsUnit(v), ew = rsUnit(wAxis);
  if (!rsSolid(s, subi(c, ev))) flags |= 1 << (2 * v);
  if (!rsSolid(s, addi(c, ev))) flags |= 1 << (2 * v + 1);
  if (!rsSolid(s, subi(c, ew))) flags |= 1 << (2 * wAxis);
  if (!rsSolid(s, addi(c, ew))) flags |= 1 << (2 * wAxis + 1);
  return flags;
}
function rsRuns(w: MirrorWorld, assembly: readonly number[], body: MirrorBody, index: number, bounds: PieceBounds, planes: PlaneList, s: MirrorSolid, lo: V3i, hi: V3i, u: number) {
  const v = (u + 1) % 3, wAxis = (u + 2) % 3;
  for (let r = lo[wAxis]!; r < hi[wAxis]!; r++) for (let sv = lo[v]!; sv < hi[v]!; sv++) {
    let open = false, start = 0, flags = 0, lowExposed = false;
    let previous = rsSolid(s, rsCell(u, lo[u]! - 1, v, sv, wAxis, r));
    for (let t = lo[u]!; t <= hi[u]!; t++) {
      const c = rsCell(u, t, v, sv, wAxis, r);
      let solid = false, sides = 0;
      if (t < hi[u]!) { solid = rsSolid(s, c); if (solid) sides = rsSideFlags(s, c, v, wAxis); }
      if (open && (!solid || sides !== flags)) {
        let highExposed = !solid; if (t === hi[u]) highExposed = !rsSolid(s, c);
        let faces = flags; if (lowExposed) faces |= 1 << (2 * u); if (highExposed) faces |= 1 << (2 * u + 1);
        rsBoxContact(w, assembly, body, index, bounds, planes, s, rsCell(u, start, v, sv, wAxis, r), rsCell(u, t, v, sv + 1, wAxis, r + 1), faces);
        open = false;
      }
      if (solid && !open) { open = true; start = t; flags = sides; lowExposed = !previous; }
      previous = solid;
    }
  }
}
function rsSlab(n: readonly number[], slab: number): [V3i, V3i] {
  if (slab === 0) return [[-1, 0, 0], [0, n[1]!, n[2]!]];
  if (slab === 1) return [[n[0]!, 0, 0], [n[0]! + 1, n[1]!, n[2]!]];
  if (slab === 2) return [[0, 0, -1], [n[0]!, n[1]!, 0]];
  if (slab === 3) return [[0, 0, n[2]!], [n[0]!, n[1]!, n[2]! + 1]];
  if (slab === 4) return [[0, -1, 0], [n[0]!, 0, n[2]!]];
  return [[0, n[1]!, 0], [n[0]!, n[1]! + 1, n[2]!]];
}
/** `assemblyVoxelContact`: an assembly against the voxel solid, in place of the container's planes. */
export function assemblyVoxelContact(w: MirrorWorld, assembly: readonly number[], body: MirrorBody, index: number, s: MirrorSolid) {
  if (body.inverseMassInertia[0] <= 0) return;
  const bounds: PieceBounds = { lo: [], hi: [] }, planes: PlaneList = { face: [], cell: [], mask: [], lo: [], hi: [] };
  rsBounds(w, body, index, bounds);
  const count = w.bodies.length, dt = w.step_s, n = s.layout.cells, origin = s.layout.origin_m, h = s.layout.cellSize_m;
  const sweep = mul(xyz(body.linearVelocity), dt), spin = length(xyz(body.angularVelocity)) * assembly[index]! * dt;
  let lo: V3 = [1e30, 1e30, 1e30], hi: V3 = [-1e30, -1e30, -1e30];
  for (let piece = 0; piece < 2 + count; piece++) {
    if (!rigidPieceExists(w, index, piece)) continue;
    lo = [0, 1, 2].map((a) => Math.min(lo[a]!, bounds.lo[piece]![a]!)) as V3; hi = [0, 1, 2].map((a) => Math.max(hi[a]!, bounds.hi[piece]![a]!)) as V3;
  }
  lo = [0, 1, 2].map((a) => Math.min(lo[a]!, lo[a]! + sweep[a]!) - spin - h[a]!) as V3;
  hi = [0, 1, 2].map((a) => Math.max(hi[a]!, hi[a]! + sweep[a]!) + spin + h[a]!) as V3;
  const c0 = [0, 1, 2].map((a) => Math.max(Math.floor((lo[a]! - origin[a]!) / h[a]!), -1)) as V3i;
  const c1 = [0, 1, 2].map((a) => Math.min(Math.floor((hi[a]! - origin[a]!) / h[a]!) + 1, n[a]! + 1)) as V3i;
  for (let slab = 0; slab < 6; slab++) {
    if (slab === 4 && !s.layout.floor) continue;
    if (slab === 5 && s.layout.walls.policy !== 1) continue;
    const box = rsSlab(n, slab), a = [0, 1, 2].map((k) => Math.max(box[0][k]!, c0[k]!)) as V3i, b = [0, 1, 2].map((k) => Math.min(box[1][k]!, c1[k]!)) as V3i;
    if ([0, 1, 2].some((k) => a[k]! >= b[k]!)) continue;
    if (!rsCutNear(s, a, b)) rsBoxContact(w, assembly, body, index, bounds, planes, s, box[0], box[1], 63);
    else rsRuns(w, assembly, body, index, bounds, planes, s, a, b, slab < 2 ? 2 : 0);
  }
  const i0 = c0.map((v) => Math.max(v, 0)) as V3i, i1 = c1.map((v, a) => Math.min(v, n[a]!)) as V3i;
  if ([0, 1, 2].every((k) => i0[k]! < i1[k]!)) {
    const window = rsWindow(s);
    const p0 = [0, 1, 2].map((k) => Math.floor((i0[k]! + window[k]!) / RIGID_SOLID_PAGE)), p1 = [0, 1, 2].map((k) => Math.floor((i1[k]! - 1 + window[k]!) / RIGID_SOLID_PAGE));
    for (let z = p0[2]!; z <= p1[2]!; z++) for (let y = p0[1]!; y <= p1[1]!; y++) for (let x = p0[0]!; x <= p1[0]!; x++) {
      const page: V3i = [x * RIGID_SOLID_PAGE, y * RIGID_SOLID_PAGE, z * RIGID_SOLID_PAGE];
      if ((solidOccupancyMaskPage(s.mask, page[0], page[1], page[2]) & 0xffff) === 0) continue;
      rsRuns(w, assembly, body, index, bounds, planes, s, [0, 1, 2].map((k) => Math.max(i0[k]!, page[k]! - window[k]!)) as V3i,
        [0, 1, 2].map((k) => Math.min(i1[k]!, page[k]! + RIGID_SOLID_PAGE - window[k]!)) as V3i, 0);
    }
  }
  for (let k = 0; k < planes.face.length; k++) rsSolve(w, assembly, body, index, bounds, planes.face[k]!, planes.mask[k]!, planes.lo[k]!, planes.hi[k]!);
}

function motorAngle(m: MirrorMotor, time: number): number {
  const t = Math.max(0, time - m.timing[0]), ramp = Math.max(m.timing[1], 1e-6), u = Math.min(1, t / ramp);
  return m.offsetPhase[3] + m.axisSpeed[3] * (ramp * (u * u * u - .5 * u * u * u * u) + Math.max(0, t - ramp));
}
function attachMotor(w: MirrorWorld, index: number, time: number, dt: number) {
  const m = w.motors[index]!; if (m.pivotParent[3] < .5) return;
  const parent = w.bodies[Math.round(m.pivotParent[3]) - 1]!, child = w.bodies[index]!;
  const angle = motorAngle(m, time), half = .5 * angle, rotation: V4 = [Math.cos(half), ...mul(xyz(m.axisSpeed), Math.sin(half))] as V4;
  const offset = qRotate(rotation, xyz(m.offsetPhase)), arm = qRotate(parent.orientation, add(xyz(m.pivotParent), offset));
  const omega = mul(qRotate(parent.orientation, xyz(m.axisSpeed)), (angle - motorAngle(m, time - dt)) / Math.max(dt, 1e-8));
  child.positionShape = with3(add(xyz(parent.positionShape), arm), child.positionShape[3]);
  child.orientation = qMultiply(parent.orientation, qMultiply(rotation, m.orientation));
  child.linearVelocity = with3(add(add(xyz(parent.linearVelocity), cross(xyz(parent.angularVelocity), arm)), cross(omega, qRotate(parent.orientation, offset))), child.linearVelocity[3]);
  child.angularVelocity = with3(add(xyz(parent.angularVelocity), omega), child.angularVelocity[3]);
}

/**
 * The kernel's `integrate` for a dry world: no fluid exchange, the static
 * surfaces given as planes. Gravity kick, the pair links, the speculative
 * pair pass, integration, the contact sweeps over the movable bodies and
 * motor attachment, in the kernel's order.
 */
export function stepRigidMirror(w: MirrorWorld, dt: number): void {
  const count = w.bodies.length, rho = Math.max(w.rho, 1e-9);
  w.time_s += dt; w.step_s = dt;
  const assembly = rigidAssemblyBuild(w);
  const dynamic = (b: MirrorBody) => b.material[2] > .5 && b.material[2] < 1.5;
  for (const body of w.bodies) {
    if (!dynamic(body)) continue;
    const scaledInverseMass = body.inverseMassInertia[0], mass = scaledInverseMass > 0 ? rho / scaledInverseMass : 1e30;
    const acceleration = mul(mul(w.gravity, mass), 1 / Math.max(mass, 1e-8));
    body.linearVelocity = with3(add(xyz(body.linearVelocity), mul(acceleration, dt)), body.linearVelocity[3]);
    body.angularVelocity = with3(inverseInertia(w, body, xyz(body.angularMomentumRestitution)), body.angularVelocity[3]);
  }
  // The pairs this step solves (rigidLinked), and the bodies contacts may move:
  // free bodies with mass. The kernel solves each connected set of them in
  // its own thread; unlinked sets share nothing, so the order here is theirs.
  const free = (index: number) => w.motors[index]!.pivotParent[3] < .5;
  const movable = w.bodies.map((body, index) => free(index) && body.inverseMassInertia[0] > 0);
  const linked = w.bodies.map((a, aIndex) => w.bodies.map((b, bIndex) => aIndex < bIndex && free(aIndex) && free(bIndex)
    && (movable[aIndex]! || movable[bIndex]!) && rigidLinked(w, assembly, a, aIndex, b, bIndex, RIGID_PAIR_MARGIN_M)));
  for (let a = 0; a < count; a++) for (let b = a + 1; b < count; b++) if (linked[a]![b]! && (assembly[a]! > 0 || assembly[b]! > 0)) solveAssemblyPair(w, assembly, a, b, true, dt);
  for (const body of w.bodies) {
    if (!dynamic(body)) continue;
    body.positionShape = with3(add(xyz(body.positionShape), mul(xyz(body.linearVelocity), dt)), body.positionShape[3]);
    const derivative = qMultiply([0, ...xyz(body.angularVelocity)] as V4, body.orientation);
    body.orientation = normalize4(body.orientation.map((q, i) => q + .5 * dt * derivative[i]!) as V4);
  }
  const settled = w.bodies.map((b) => structuredClone(b));
  for (let iteration = 0; iteration < RIGID_CONTACT_ITERATIONS; iteration++) {
    for (let index = 0; index < count; index++) {
      if (!movable[index]!) continue;
      const voxel = w.solid !== undefined && assembly[index]! > 0;
      if (voxel) assemblyVoxelContact(w, assembly, w.bodies[index]!, index, w.solid!);
      w.planes.forEach((plane, k) => { if (!voxel || k >= w.solid!.containerPlanes) staticPlaneContact(w, assembly, w.bodies[index]!, index, plane.normal, plane.offset); });
    }
    for (let a = 0; a < count; a++) for (let b = a + 1; b < count; b++) {
      if (!linked[a]![b]!) continue;
      if (assembly[a]! > 0 || assembly[b]! > 0) solveAssemblyPair(w, assembly, a, b, false, dt); else solveBodyPair(w, a, b);
    }
  }
  // Contacts move only the movable. A held body is the host's to place (it
  // has no inverse mass), and a fixture is placed by its motor: both end the
  // step as integration left them, whatever a pair's impulse did to their
  // angular momentum meanwhile.
  for (let index = 0; index < count; index++) if (!movable[index]!) w.bodies[index] = settled[index]!;
  for (let index = 0; index < count; index++) attachMotor(w, index, w.time_s, dt);
}

/** The kernel's container walls (and floor, and a closed lid) as `planeContact` visits them. */
export function containerMirrorPlanes(container: { width_m: number; height_m: number; depth_m: number; top?: string }): MirrorPlane[] {
  const planes: MirrorPlane[] = [
    { normal: [1, 0, 0], offset: -.5 * container.width_m }, { normal: [-1, 0, 0], offset: -.5 * container.width_m },
    { normal: [0, 0, 1], offset: -.5 * container.depth_m }, { normal: [0, 0, -1], offset: -.5 * container.depth_m },
    { normal: [0, 1, 0], offset: 0 },
  ];
  if (container.top === "closed") planes.push({ normal: [0, -1, 0], offset: -container.height_m });
  return planes;
}
