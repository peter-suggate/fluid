import type { SceneDescription } from "./model";
import type { Quaternion, Vec3 } from "./model";
import { boundingRadius, primitiveVolume, type RigidBodyState } from "./rigid-body";
import { SVO_PRIMITIVE_MOTION_STRIDE_BYTES, svoPrimitiveMotionWGSL } from "../svo/svo-primitive-motion";
import { sceneHasTerrain } from "./terrain";
import {
  SCENE_SHAPES_BY_CODE,
  SCENE_SHAPE_PALETTE_LINEAR,
  sceneShapeCode,
  sceneShapePresentationWgsl,
  sceneShapeRenderHalfExtent_m,
  sceneShapeWgsl,
} from "./scene-shape";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";
import type { GPUInitializationTask } from "./gpu-initialization";

export const GPU_RIGID_BODY_CAPACITY = 12;
/**
 * The first owner id that is *not* a rigid body.
 *
 * Owner ids 0..GPU_RIGID_BODY_CAPACITY-1 name a slot in the rigid arenas, which
 * are sized at that capacity whether or not the scene fills them; everything
 * else in the scene — scenery proxies, glass, fixture lights — is numbered from
 * here.
 *
 * It used to be `scene.rigidBodies.length`, which made every scenery object's
 * owner id a function of how many solids the scene happened to hold. Adding one
 * renumbered the whole published render source, which is why the roster length
 * sat in the solver's seed tier and why dropping a body into running water
 * restarted the clock. Pinned, the two ranges are disjoint by construction
 * rather than by arithmetic. See `rigidAllocationKey`.
 */
export const SCENE_ENVIRONMENT_OWNER_BASE = GPU_RIGID_BODY_CAPACITY;
/**
 * What `material.z` says about a body's motion.
 *
 * Three states, not two, because "does it integrate?" and "is it a fixed part
 * of the scene?" stopped being the same question the moment a body could be
 * picked up. A held body must keep publishing the volume it displaces — that is
 * the whole point of dipping a cup — while refusing gravity, contacts and pair
 * impulses, and neither of the two states that existed can say that.
 */
export const GPU_RIGID_MOTION_LANES = Object.freeze({
  /** Authored static: no integration, authored waterline volume. */
  fixed: 0,
  /** Ordinary dynamic: integrates, publishes its live displaced volume. */
  dynamic: 1,
  /** In the user's hand: publishes displacement, integrates nothing. */
  held: 2,
});

/** The lane `syncBodies` packs for one body. */
export function rigidMotionLane(body: RigidBodyState): number {
  if (body.description.motion === "static") return GPU_RIGID_MOTION_LANES.fixed;
  return body.held ? GPU_RIGID_MOTION_LANES.held : GPU_RIGID_MOTION_LANES.dynamic;
}

export const GPU_RIGID_STATE_FLOATS = 32;
export const GPU_RIGID_STATE_BYTES = GPU_RIGID_BODY_CAPACITY * GPU_RIGID_STATE_FLOATS * 4;
export const GPU_RIGID_RENDER_FLOATS = 16;
export const GPU_RIGID_RENDER_BYTES = GPU_RIGID_BODY_CAPACITY * GPU_RIGID_RENDER_FLOATS * 4;
export const GPU_RIGID_MOTION_BYTES = GPU_RIGID_BODY_CAPACITY * SVO_PRIMITIVE_MOTION_STRIDE_BYTES;
export const GPU_RIGID_IMMERSED_VOLUME_BYTES = GPU_RIGID_BODY_CAPACITY * 4;
const GPU_RIGID_PICK_BYTES = 48;

export interface GPURigidBodyPick {
  bodyIndex: number;
  distance_m: number;
  position_m: Vec3;
  orientation: Quaternion;
}

/** Where one body actually is, as opposed to where the host last told it to be. */
export interface GPURigidBodyPose {
  position_m: Vec3;
  orientation: Quaternion;
}

/**
 * The compact SolidWorld occupancy projection consumed by moving-body contact.
 *
 * The adaptive solver supplies its existing SOC1 arena. This is deliberately a
 * structural view rather than another static-solid upload: fluid apertures and
 * rigid contacts consequently observe the same missing or occupied voxel.
 */
export interface GPURigidSolidWorldCollisionSource {
  readonly buffer: GPUBuffer;
  readonly baseWords: number;
  readonly directoryCapacity: number;
  readonly directoryBaseWords: number;
  readonly regionCapacity: number;
  readonly regionBaseWords: number;
  readonly regionWords: number;
  readonly entryWords: number;
  readonly pageBaseWords: number;
  readonly pageWords: number;
  readonly fractionPageWords: number;
  readonly origin_m: readonly [number, number, number];
  readonly cellSize_m: readonly [number, number, number];
}

/**
 * Whether two published pose tables say the same thing.
 *
 * Exact comparison, not a tolerance: the question is whether the readback
 * carries news, and the numbers come from the same decode of the same buffer,
 * so a body that has not moved compares equal bit for bit. A tolerance here
 * would instead make a slow drift invisible until it crossed the threshold, and
 * a gizmo that snaps every few seconds is worse than one that tracks.
 */
export function samePublishedBodyPoses(
  a: Readonly<Record<string, GPURigidBodyPose>>,
  b: Readonly<Record<string, GPURigidBodyPose>>,
): boolean {
  const keys = Object.keys(a);
  if (keys.length !== Object.keys(b).length) return false;
  return keys.every((id) => {
    const left = a[id], right = b[id];
    if (!right) return false;
    return left.position_m.x === right.position_m.x
      && left.position_m.y === right.position_m.y
      && left.position_m.z === right.position_m.z
      && left.orientation.w === right.orientation.w
      && left.orientation.x === right.orientation.x
      && left.orientation.y === right.orientation.y
      && left.orientation.z === right.orientation.z;
  });
}

/** A pose with the body it belongs to, as published from a drawn frame. */
export interface DrawnRigidBodyPose extends GPURigidBodyPose {
  id: string;
}

/** Decode `RenderBody` records into poses. Separated from WebGPU for tests. */
export function decodeGPURigidBodyPoses(values: Float32Array, count: number): GPURigidBodyPose[] {
  const poses: GPURigidBodyPose[] = [];
  for (let index = 0; index < Math.min(count, GPU_RIGID_BODY_CAPACITY); index += 1) {
    const offset = index * GPU_RIGID_RENDER_FLOATS;
    const record = values.subarray(offset, offset + GPU_RIGID_RENDER_FLOATS);
    // A partially published record is not a pose. Falling back to the host
    // mirror is wrong in a different direction, but it is at least a pose the
    // rest of the app already agrees on.
    if (record.length < 12 || !Array.from(record.subarray(0, 12)).every(Number.isFinite)) return poses;
    poses.push({
      position_m: { x: record[0], y: record[1], z: record[2] },
      orientation: { w: record[8], x: record[9], y: record[10], z: record[11] },
    });
  }
  return poses;
}

const rigidTerrainBindingWGSL = /* wgsl */ `
@group(0) @binding(4) var terrainHeights: texture_2d<f32>;
`;
const rigidTerrainContactWGSL = /* wgsl */ `
fn terrainPlane(position: vec3f) -> vec4f {
  let dims=vec2i(textureDimensions(terrainHeights));
  let fx=(position.x/params.container.x+.5)*f32(dims.x); let fz=(position.z/params.container.z+.5)*f32(dims.y);
  let cell=clamp(vec2i(floor(vec2f(fx,fz))),vec2i(0),dims-vec2i(1));
  let xm=max(0,cell.x-1);let xp=min(dims.x-1,cell.x+1);let zm=max(0,cell.y-1);let zp=min(dims.y-1,cell.y+1);
  let h=params.terrain.x; let center=textureLoad(terrainHeights,cell,0).x*h;
  let dx=max(params.container.x/f32(dims.x),1e-6);let dz=max(params.container.z/f32(dims.y),1e-6);
  let dhdx=(textureLoad(terrainHeights,vec2i(xp,cell.y),0).x-textureLoad(terrainHeights,vec2i(xm,cell.y),0).x)*h/(f32(xp-xm)*dx+1e-6);
  let dhdz=(textureLoad(terrainHeights,vec2i(cell.x,zp),0).x-textureLoad(terrainHeights,vec2i(cell.x,zm),0).x)*h/(f32(zp-zm)*dz+1e-6);
  let normal=normalize(vec3f(-dhdx,1,-dhdz)); return vec4f(normal,dot(normal,vec3f(position.x,center,position.z)));
}
`;

const rigidSolidWorldBindingWGSL = /* wgsl */ `
@group(0) @binding(9) var<storage, read> solidWorldOccupancy: array<u32>;
`;

const rigidPlaneContactWGSL = /* wgsl */ `
fn planeContact(body: ptr<function,RigidBody>, normal: vec3f, offset: f32) {
  let rho=max(params.step.y,1e-9); let inverseMass=(*body).inverseMassInertia.x/rho;
  if(inverseMass<=0.0){return;}
  let radius=supportRadius(*body,normal); let penetration=offset-(dot(normal,(*body).positionShape.xyz)-radius);
  if(penetration<=0.0){return;}
  resolveStaticContact(body,normal,penetration,radius);
}
`;

const rigidSolidWorldContactWGSL = (
  source: GPURigidSolidWorldCollisionSource,
): string => /* wgsl */ `
const RIGID_SOC_BASE:u32=${source.baseWords}u;
const RIGID_SOC_DIRECTORY_CAPACITY:u32=${source.directoryCapacity}u;
const RIGID_SOC_DIRECTORY_BASE:u32=${source.directoryBaseWords}u;
const RIGID_SOC_REGION_CAPACITY:u32=${source.regionCapacity}u;
const RIGID_SOC_REGION_BASE:u32=${source.regionBaseWords}u;
const RIGID_SOC_REGION_WORDS:u32=${source.regionWords}u;
const RIGID_SOC_PAGE_BASE:u32=${source.pageBaseWords}u;
const RIGID_SOC_ENTRY_WORDS:u32=${source.entryWords}u;
const RIGID_SOC_PAGE_WORDS:u32=${source.pageWords}u;
const RIGID_SOC_FRACTION_WORDS:u32=${source.fractionPageWords}u;
const RIGID_SOC_INVALID:u32=0xffffffffu;

fn rigidSolidOriginFine()->vec3i{
  return vec3i(bitcast<i32>(solidWorldOccupancy[RIGID_SOC_BASE+8u]),
    bitcast<i32>(solidWorldOccupancy[RIGID_SOC_BASE+9u]),
    bitcast<i32>(solidWorldOccupancy[RIGID_SOC_BASE+10u]));
}
fn rigidSolidHash(q:vec3i)->u32{
  var hash=0x811c9dc5u;
  hash=(hash^bitcast<u32>(q.x))*0x01000193u;hash^=hash>>16u;
  hash=(hash^bitcast<u32>(q.y))*0x01000193u;hash^=hash>>16u;
  hash=(hash^bitcast<u32>(q.z))*0x01000193u;hash^=hash>>16u;
  hash=hash*0x01000193u;hash^=hash>>16u;return hash|1u;
}
fn rigidSolidFloorDiv8(value:i32)->i32{
  return select(value/8,(value-7)/8,value<0);
}
fn rigidSolidPageAt(q:vec3i)->u32{
  let hash=rigidSolidHash(q);var slot=hash&(RIGID_SOC_DIRECTORY_CAPACITY-1u);
  for(var probe=0u;probe<RIGID_SOC_DIRECTORY_CAPACITY;probe+=1u){
    let at=RIGID_SOC_BASE+RIGID_SOC_DIRECTORY_BASE+slot*RIGID_SOC_ENTRY_WORDS;
    if(solidWorldOccupancy[at]==0u){return RIGID_SOC_INVALID;}
    if(solidWorldOccupancy[at]==2u&&solidWorldOccupancy[at+1u]==hash
      &&bitcast<i32>(solidWorldOccupancy[at+2u])==q.x
      &&bitcast<i32>(solidWorldOccupancy[at+3u])==q.y
      &&bitcast<i32>(solidWorldOccupancy[at+4u])==q.z){
      let page=solidWorldOccupancy[at+5u];
      return select(RIGID_SOC_INVALID,page,
        page<solidWorldOccupancy[RIGID_SOC_BASE+6u]);
    }
    slot=(slot+1u)&(RIGID_SOC_DIRECTORY_CAPACITY-1u);
  }
  return RIGID_SOC_INVALID;
}
fn rigidSolidSdfQ8(worldFine:vec3i)->i32{
  var regionOperation=-1;
  for(var region=0u;region<RIGID_SOC_REGION_CAPACITY;region+=1u){
    let at=RIGID_SOC_BASE+RIGID_SOC_REGION_BASE+region*RIGID_SOC_REGION_WORDS;
    let minimum=vec3i(bitcast<i32>(solidWorldOccupancy[at+1u]),
      bitcast<i32>(solidWorldOccupancy[at+2u]),
      bitcast<i32>(solidWorldOccupancy[at+3u]));
    let maximum=vec3i(bitcast<i32>(solidWorldOccupancy[at+4u]),
      bitcast<i32>(solidWorldOccupancy[at+5u]),
      bitcast<i32>(solidWorldOccupancy[at+6u]));
    if(all(worldFine>=minimum)&&all(worldFine<maximum)){
      regionOperation=i32(solidWorldOccupancy[at]);
    }
  }
  if(regionOperation==1){return -128;}
  if(regionOperation==0){return 32767;}
  let q=worldFine-rigidSolidOriginFine();
  let pageQ=vec3i(rigidSolidFloorDiv8(q.x),rigidSolidFloorDiv8(q.y),
    rigidSolidFloorDiv8(q.z));let local=q-pageQ*8;let page=rigidSolidPageAt(pageQ);
  if(page==RIGID_SOC_INVALID){return 512;}
  let voxel=u32(local.x+8*(local.y+8*local.z));
  let word=solidWorldOccupancy[RIGID_SOC_BASE+RIGID_SOC_PAGE_BASE
    +page*RIGID_SOC_PAGE_WORDS+RIGID_SOC_FRACTION_WORDS+(voxel>>1u)];
  let lane=(word>>(16u*(voxel&1u)))&0xffffu;
  return bitcast<i32>(lane<<16u)>>16;
}
fn voxelInteriorContact(center:vec3f,minimum:vec3f,maximum:vec3f)->vec4f{
  let toMinimum=center-minimum;let toMaximum=maximum-center;
  var distance=toMinimum.x;var normal=vec3f(-1,0,0);
  if(toMaximum.x<distance){distance=toMaximum.x;normal=vec3f(1,0,0);}
  if(toMinimum.y<distance){distance=toMinimum.y;normal=vec3f(0,-1,0);}
  if(toMaximum.y<distance){distance=toMaximum.y;normal=vec3f(0,1,0);}
  if(toMinimum.z<distance){distance=toMinimum.z;normal=vec3f(0,0,-1);}
  if(toMaximum.z<distance){distance=toMaximum.z;normal=vec3f(0,0,1);}
  return vec4f(normal,max(0.0,distance));
}
fn solidVoxelContact(body:ptr<function,RigidBody>){
  let rho=max(params.step.y,1e-9);
  if((*body).inverseMassInertia.x/rho<=0.0){return;}
  let origin=params.solidOrigin.xyz;let cell=max(params.solidCell.xyz,vec3f(1e-7));
  let broadRadius=max((*body).dimensions.w,1e-7);
  let bodyMinimum=(*body).positionShape.xyz-vec3f(broadRadius);
  let bodyMaximum=(*body).positionShape.xyz+vec3f(broadRadius);
  let solidOriginFine=rigidSolidOriginFine();
  let pageCapacity=solidWorldOccupancy[RIGID_SOC_BASE+6u];
  var bestPenetration=0.0;var bestNormal=vec3f(0,1,0);var bestRadius=0.0;
  // SOC1 contains only occupied pages. Walk its compact directory, reject
  // pages outside the body's broad phase, then visit non-zero exact fractions. This
  // stays complete for bodies of any size without scanning a cubic empty host
  // domain or imposing a collision-size cutoff.
  for(var slot=0u;slot<RIGID_SOC_DIRECTORY_CAPACITY;slot+=1u){
    let entry=RIGID_SOC_BASE+RIGID_SOC_DIRECTORY_BASE+slot*RIGID_SOC_ENTRY_WORDS;
    if(solidWorldOccupancy[entry]!=2u){continue;}
    let page=solidWorldOccupancy[entry+5u];if(page>=pageCapacity){continue;}
    let pageQ=vec3i(bitcast<i32>(solidWorldOccupancy[entry+2u]),
      bitcast<i32>(solidWorldOccupancy[entry+3u]),
      bitcast<i32>(solidWorldOccupancy[entry+4u]));
    let pageFine=solidOriginFine+pageQ*8;
    let pageMinimum=origin+vec3f(pageFine)*cell;let pageMaximum=pageMinimum+8.0*cell;
    if(any(bodyMaximum<pageMinimum)||any(bodyMinimum>pageMaximum)){continue;}
    let payload=RIGID_SOC_BASE+RIGID_SOC_PAGE_BASE+page*RIGID_SOC_PAGE_WORDS;
    for(var voxel=0u;voxel<512u;voxel+=1u){
        let fractionWord=voxel>>2u;if(fractionWord>=RIGID_SOC_FRACTION_WORDS){continue;}
        let word=solidWorldOccupancy[payload+fractionWord];
        let fraction=(word>>(8u*(voxel&3u)))&255u;
        if(fraction==0u){continue;}
        let q=pageFine+vec3i(i32(voxel&7u),i32((voxel>>3u)&7u),i32(voxel>>6u));
        let minimum=origin+vec3f(q)*cell;let maximum=minimum+cell;
        let closest=clamp((*body).positionShape.xyz,minimum,maximum);
        let delta=(*body).positionShape.xyz-closest;let separation=length(delta);
        if(separation>broadRadius){continue;}
        var normal=vec3f(0,1,0);var penetration=0.0;var radius=0.0;
        if(fraction<255u){
          let sx=f32(clamp(rigidSolidSdfQ8(q+vec3i(1,0,0)),-512,512)
            -clamp(rigidSolidSdfQ8(q-vec3i(1,0,0)),-512,512))/cell.x;
          let sy=f32(clamp(rigidSolidSdfQ8(q+vec3i(0,1,0)),-512,512)
            -clamp(rigidSolidSdfQ8(q-vec3i(0,1,0)),-512,512))/cell.y;
          let sz=f32(clamp(rigidSolidSdfQ8(q+vec3i(0,0,1)),-512,512)
            -clamp(rigidSolidSdfQ8(q-vec3i(0,0,1)),-512,512))/cell.z;
          let centre=0.5*(minimum+maximum);
          normal=safeNormalize(vec3f(sx,sy,sz),
            safeNormalize((*body).positionShape.xyz-centre,vec3f(0,1,0)));
          let sdfMetres=f32(clamp(rigidSolidSdfQ8(q),-512,512))/256.0
            *min(cell.x,min(cell.y,cell.z));
          let surface=centre-normal*sdfMetres;radius=supportRadius(*body,normal);
          penetration=radius-(dot(normal,(*body).positionShape.xyz)-dot(normal,surface));
        }else if(separation>1e-7){
          normal=delta/separation;radius=supportRadius(*body,normal);
          penetration=radius-separation;
        }else{
          let interior=voxelInteriorContact((*body).positionShape.xyz,minimum,maximum);
          normal=interior.xyz;radius=supportRadius(*body,normal);
          penetration=radius+interior.w;
        }
        if(penetration>bestPenetration){
          bestPenetration=penetration;bestNormal=normal;bestRadius=radius;
        }
      }
  }
  // Large exact collider boxes remain compact regions instead of expanding
  // their area into voxel pages. Resolve the same support contact analytically.
  for(var region=0u;region<RIGID_SOC_REGION_CAPACITY;region+=1u){
    let at=RIGID_SOC_BASE+RIGID_SOC_REGION_BASE+region*RIGID_SOC_REGION_WORDS;
    if(solidWorldOccupancy[at]!=1u){continue;}
    let minimumFine=vec3i(bitcast<i32>(solidWorldOccupancy[at+1u]),
      bitcast<i32>(solidWorldOccupancy[at+2u]),
      bitcast<i32>(solidWorldOccupancy[at+3u]));
    let maximumFine=vec3i(bitcast<i32>(solidWorldOccupancy[at+4u]),
      bitcast<i32>(solidWorldOccupancy[at+5u]),
      bitcast<i32>(solidWorldOccupancy[at+6u]));
    let minimum=origin+vec3f(minimumFine)*cell;
    let maximum=origin+vec3f(maximumFine)*cell;
    if(any(bodyMaximum<minimum)||any(bodyMinimum>maximum)){continue;}
    let closest=clamp((*body).positionShape.xyz,minimum,maximum);
    let delta=(*body).positionShape.xyz-closest;let separation=length(delta);
    if(separation>broadRadius){continue;}
    var normal=vec3f(0,1,0);var penetration=0.0;var radius=0.0;
    if(separation>1e-7){
      normal=delta/separation;radius=supportRadius(*body,normal);
      penetration=radius-separation;
    }else{
      let interior=voxelInteriorContact((*body).positionShape.xyz,minimum,maximum);
      normal=interior.xyz;radius=supportRadius(*body,normal);
      penetration=radius+interior.w;
    }
    if(penetration>bestPenetration){
      bestPenetration=penetration;bestNormal=normal;bestRadius=radius;
    }
  }
  if(bestPenetration>0.0){
    resolveStaticContact(body,bestNormal,bestPenetration,bestRadius);
  }
}
`;

const gpuRigidBodyShaderSource = (
  terrainContact: boolean,
  solidWorldContact?: GPURigidSolidWorldCollisionSource,
): string => /* wgsl */ `
${svoPrimitiveMotionWGSL}
struct RigidBody {
  positionShape: vec4f,
  dimensions: vec4f,
  orientation: vec4f,
  linearVelocity: vec4f,
  angularVelocity: vec4f,
  inverseMassInertia: vec4f,
  angularMomentumRestitution: vec4f,
  material: vec4f,
}
struct RenderBody { positionRadius: vec4f, halfShape: vec4f, orientation: vec4f, colorSelected: vec4f }
struct Params {
  step: vec4f,
  gravity: vec4f,
  ${solidWorldContact ? "staticContactUnused0: vec4f,\n  staticContactUnused1: vec4f," : "container: vec4f,\n  terrain: vec4f,"}
  coupling: vec4f,
  solidOrigin: vec4f,
  solidCell: vec4f,
}
struct PickParams { originCount: vec4f, direction: vec4f }
struct PickResult { index: u32, hit: u32, distance: f32, pad: f32, position: vec4f, orientation: vec4f }
@group(0) @binding(0) var<storage, read_write> bodies: array<RigidBody, 12>;
@group(0) @binding(1) var<storage, read_write> exchange: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> renderBodies: array<RenderBody, 12>;
@group(0) @binding(3) var<uniform> params: Params;
${terrainContact ? rigidTerrainBindingWGSL : ""}
@group(0) @binding(5) var<uniform> pickParams: PickParams;
@group(0) @binding(6) var<storage, read_write> pickResult: PickResult;
@group(0) @binding(7) var<storage, read_write> rigidMotion: array<SvoPrimitiveMotionRecord, 12>;
@group(0) @binding(8) var<storage, read_write> immersedVolumes: array<f32>;
${solidWorldContact ? rigidSolidWorldBindingWGSL : ""}

fn qConjugate(q: vec4f) -> vec4f { return vec4f(q.x, -q.yzw); }
fn qMultiply(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.x*b.x-dot(a.yzw,b.yzw), a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));
}
fn qRotate(q: vec4f, v: vec3f) -> vec3f {
  let uv=cross(q.yzw,v); return v+2.0*(q.x*uv+cross(q.yzw,uv));
}
fn qInverseRotate(q: vec4f, v: vec3f) -> vec3f { return qRotate(qConjugate(q),v); }
fn safeNormalize(v: vec3f, fallback: vec3f) -> vec3f { let l=length(v); return select(fallback,v/l,l>1e-8); }
fn inverseInertia(body: RigidBody, v: vec3f) -> vec3f {
  let rho=max(params.step.y,1e-9); let local=qInverseRotate(body.orientation,v);
  return qRotate(body.orientation,local*body.inverseMassInertia.yzw/rho);
}
${sceneShapeWgsl()}
${sceneShapePresentationWgsl()}
fn rigidShapeTag(body: RigidBody) -> i32 { return i32(round(body.positionShape.w)); }
fn supportRadius(body: RigidBody, directionWorld: vec3f) -> f32 {
  let direction=safeNormalize(directionWorld,vec3f(1,0,0)); let local=qInverseRotate(body.orientation,direction);
  return rigidShapeSupportRadius(rigidShapeTag(body),body.dimensions.xyz,local);
}
fn velocityAt(body: RigidBody, arm: vec3f) -> vec3f { return body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm); }
fn angularTerm(body: RigidBody, arm: vec3f, direction: vec3f) -> f32 {
  return dot(cross(inverseInertia(body,cross(arm,direction)),arm),direction);
}
fn applyImpulse(body: ptr<function,RigidBody>, impulse: vec3f, arm: vec3f) {
  let rho=max(params.step.y,1e-9); let inverseMass=(*body).inverseMassInertia.x/rho;
  (*body).linearVelocity=vec4f((*body).linearVelocity.xyz+impulse*inverseMass,(*body).linearVelocity.w);
  (*body).angularMomentumRestitution=vec4f((*body).angularMomentumRestitution.xyz+cross(arm,impulse),(*body).angularMomentumRestitution.w);
  (*body).angularVelocity=vec4f(inverseInertia(*body,(*body).angularMomentumRestitution.xyz),(*body).angularVelocity.w);
}
fn resolveStaticContact(body:ptr<function,RigidBody>,normal:vec3f,
  penetration:f32,radius:f32){
  let rho=max(params.step.y,1e-9);let inverseMass=(*body).inverseMassInertia.x/rho;
  if(inverseMass<=0.0||penetration<=0.0){return;}
  (*body).positionShape=vec4f((*body).positionShape.xyz
    +normal*(penetration+1e-7),(*body).positionShape.w);
  let arm=-normal*radius;var relative=velocityAt(*body,arm);
  let normalSpeed=dot(relative,normal);if(normalSpeed>=0.0){return;}
  let restitution=select(0.0,(*body).angularMomentumRestitution.w,-normalSpeed>0.5);
  let denominator=max(inverseMass+angularTerm(*body,arm,normal),1e-9);
  let normalMagnitude=-(1.0+restitution)*normalSpeed/denominator;
  applyImpulse(body,normal*normalMagnitude,arm);
  relative=velocityAt(*body,arm);let tangentVelocity=relative-normal*dot(relative,normal);
  let tangentSpeed=length(tangentVelocity);if(tangentSpeed<=1e-8){return;}
  let tangent=tangentVelocity/tangentSpeed;
  let tangentDenominator=max(inverseMass+angularTerm(*body,arm,tangent),1e-9);
  let tangentMagnitude=clamp(-tangentSpeed/tangentDenominator,
    -(*body).material.x*normalMagnitude,(*body).material.x*normalMagnitude);
  applyImpulse(body,tangent*tangentMagnitude,arm);
}
${solidWorldContact ? "" : rigidPlaneContactWGSL}
fn bodyVolume(body: RigidBody) -> f32 {
  return rigidShapeVolume(rigidShapeTag(body),body.dimensions.xyz);
}
${terrainContact ? rigidTerrainContactWGSL : ""}
${solidWorldContact ? rigidSolidWorldContactWGSL(solidWorldContact) : ""}
fn solveBodyPair(aIndex: u32,bIndex: u32) {
  var a=bodies[aIndex];var b=bodies[bIndex];let rho=max(params.step.y,1e-9);
  let inverseA=a.inverseMassInertia.x/rho;let inverseB=b.inverseMassInertia.x/rho;let inverseTotal=inverseA+inverseB;
  if(inverseTotal<=0.0){return;}
  let delta=b.positionShape.xyz-a.positionShape.xyz;let distance=length(delta);let normal=select(vec3f(1,0,0),delta/distance,distance>1e-8);
  let radiusA=a.dimensions.w;let radiusB=b.dimensions.w;let penetration=radiusA+radiusB-distance;
  if(penetration<=0.0){return;}
  a.positionShape=vec4f(a.positionShape.xyz-normal*penetration*inverseA/inverseTotal,a.positionShape.w);
  b.positionShape=vec4f(b.positionShape.xyz+normal*penetration*inverseB/inverseTotal,b.positionShape.w);
  let armA=normal*radiusA;let armB=-normal*radiusB;var relative=velocityAt(b,armB)-velocityAt(a,armA);let normalSpeed=dot(relative,normal);
  if(normalSpeed<0.0){
    let restitution=select(0.0,min(a.angularMomentumRestitution.w,b.angularMomentumRestitution.w),-normalSpeed>0.5);
    let denominator=max(inverseTotal+angularTerm(a,armA,normal)+angularTerm(b,armB,normal),1e-9);
    let normalMagnitude=-(1.0+restitution)*normalSpeed/denominator;
    applyImpulse(&a,-normal*normalMagnitude,armA);applyImpulse(&b,normal*normalMagnitude,armB);
    relative=velocityAt(b,armB)-velocityAt(a,armA);let tangentVelocity=relative-normal*dot(relative,normal);let tangentSpeed=length(tangentVelocity);
    if(tangentSpeed>1e-8){let tangent=tangentVelocity/tangentSpeed;let friction=sqrt(max(0.0,a.material.x*b.material.x));let tangentDenominator=max(inverseTotal+angularTerm(a,armA,tangent)+angularTerm(b,armB,tangent),1e-9);let magnitude=clamp(-tangentSpeed/tangentDenominator,-friction*normalMagnitude,friction*normalMagnitude);applyImpulse(&a,-tangent*magnitude,armA);applyImpulse(&b,tangent*magnitude,armB);}
  }
  bodies[aIndex]=a;bodies[bIndex]=b;
}
fn publish(index: u32) {
  let body=bodies[index];let shape=rigidShapeTag(body);
  let half=rigidShapeRenderHalf(shape,body.dimensions.xyz);
  let color=rigidShapePalette(shape);
  renderBodies[index]=RenderBody(vec4f(body.positionShape.xyz,body.dimensions.w),vec4f(half,body.positionShape.w),body.orientation,vec4f(color,body.material.w));
}
fn motionQuaternionXyzw(qWxyz:vec4f)->vec4f{return vec4f(qWxyz.yzw,qWxyz.x);}
fn motionTransformMatches(record:SvoPrimitiveMotionRecord,body:RigidBody)->bool{
  let positionMatches=distance(record.currentPositionDt.xyz,body.positionShape.xyz)<=1e-6;
  let oldQ=svoPrimitiveMotionQuaternionNormalize(record.currentOrientation);let bodyQ=svoPrimitiveMotionQuaternionNormalize(motionQuaternionXyzw(body.orientation));
  return positionMatches&&abs(dot(oldQ,bodyQ))>=1.0-1e-6;
}
fn motionMaterialId(shape:i32)->u32{
  ${SCENE_SHAPES_BY_CODE.map((kind) => `if(shape==${kind.code}){return ${VOXEL_MATERIAL_IDS[kind.name]}u;}`).join("")}
  return ${VOXEL_MATERIAL_IDS.sphere}u;
}
fn publishMotion(index:u32,previousBody:RigidBody,currentBody:RigidBody,dt:f32){
  let old=rigidMotion[index];let generation=bitcast<u32>(currentBody.material.y);let previousGeneration=old.publication.x;
  let currentQ=svoPrimitiveMotionQuaternionNormalize(motionQuaternionXyzw(currentBody.orientation));var previousQ=svoPrimitiveMotionQuaternionNormalize(motionQuaternionXyzw(previousBody.orientation));var flags=0u;
  if(dot(previousQ,currentQ)<0.0){previousQ=-previousQ;flags|=SVO_PRIMITIVE_MOTION_SHORTEST_FLIP;}
  let deltaPosition=currentBody.positionShape.xyz-previousBody.positionShape.xyz;let rotationDot=clamp(abs(dot(previousQ,currentQ)),0.0,1.0);let angularDisplacement=2.0*acos(rotationDot);let radius=max(currentBody.dimensions.w,1e-6);let maximumDisplacement=length(deltaPosition)+2.0*radius*sin(.5*angularDisplacement);let motionLimit=min(.5,2.0*max(params.coupling.w,1e-6));
  let generationContinuous=generation!=0u&&generation==previousGeneration;let revisionContinuous=generationContinuous&&motionTransformMatches(old,previousBody);let teleport=maximumDisplacement>motionLimit;let valid=dt>1e-8&&generationContinuous&&revisionContinuous&&!teleport;
  if(valid){flags|=SVO_PRIMITIVE_MOTION_VALID;}if(length(deltaPosition)<=1e-8&&angularDisplacement<=1e-8){flags|=SVO_PRIMITIVE_MOTION_STATIC;}if(revisionContinuous){flags|=SVO_PRIMITIVE_MOTION_REVISION_CONTINUOUS;}if(generationContinuous){flags|=SVO_PRIMITIVE_MOTION_GENERATION_CONTINUOUS;}if(teleport){flags|=SVO_PRIMITIVE_MOTION_TELEPORT;}
  let previousRevision=old.identityRevision.z;let currentRevision=previousRevision+1u;let linearVelocity=select(vec3f(0.0),currentBody.linearVelocity.xyz,valid);let angularVelocity=select(vec3f(0.0),currentBody.angularVelocity.xyz,valid);let shape=i32(round(currentBody.positionShape.w));
  rigidMotion[index]=SvoPrimitiveMotionRecord(vec4f(currentBody.positionShape.xyz,dt),vec4f(previousBody.positionShape.xyz,radius),currentQ,previousQ,vec4f(linearVelocity,maximumDisplacement),vec4f(angularVelocity,angularDisplacement),vec4u(index,(index<<16u)|motionMaterialId(shape),currentRevision,previousRevision),vec4u(generation,previousGeneration,flags,bitcast<u32>(motionLimit)));
}
@compute @workgroup_size(1)
fn pickRigidBody(@builtin(global_invocation_id) id: vec3u) {
  if(any(id!=vec3u(0))){return;}
  let count=u32(round(pickParams.originCount.w));
  let origin=pickParams.originCount.xyz;
  let direction=safeNormalize(pickParams.direction.xyz,vec3f(0,0,-1));
  var bestIndex=0xffffffffu;
  var bestDistance=1e30;
  for(var index=0u;index<12u;index++){
    if(index>=count){break;}
    let body=bodies[index];
    let relative=origin-body.positionShape.xyz;
    let projected=dot(relative,direction);
    let discriminant=projected*projected-(dot(relative,relative)-body.dimensions.w*body.dimensions.w);
    if(discriminant<0.0){continue;}
    let root=sqrt(discriminant);
    let nearDistance=-projected-root;
    let farDistance=-projected+root;
    let distance=select(farDistance,nearDistance,nearDistance>0.0);
    if(distance>0.0&&distance<bestDistance){bestDistance=distance;bestIndex=index;}
  }
  if(bestIndex==0xffffffffu){pickResult=PickResult(bestIndex,0u,0.0,0.0,vec4f(0),vec4f(1,0,0,0));return;}
  let body=bodies[bestIndex];
  pickResult=PickResult(bestIndex,1u,bestDistance,0.0,vec4f(body.positionShape.xyz,1),body.orientation);
}
@compute @workgroup_size(1)
fn integrate(@builtin(global_invocation_id) id: vec3u) {
  if(any(id!=vec3u(0))){return;}let count=u32(round(params.step.z));let dt=params.step.x;let rho=max(params.step.y,1e-9);let snapshots=max(params.step.w,1.0);
  var previousBodies:array<RigidBody,12>;for(var index=0u;index<12u;index++){previousBodies[index]=bodies[index];}
  for(var index=0u;index<12u;index++){if(index>=count){break;}var body=bodies[index];
    let base=index*12u;let wet=f32(atomicLoad(&exchange[base+6u]))/65536.0/snapshots;let displaced=min(max(0.0,wet*params.coupling.x),bodyVolume(body));
    // Static bodies retain the analytic authored-waterline volume uploaded by
    // syncBodies. Only an integrating pose needs a recurring immersed update.
    // A held body is on the moving side of that line and not the integrating
    // one: a cup carried into the tank has to keep displacing water even while
    // the hand holding it refuses gravity.
    if(body.material.z>0.5){if(index<arrayLength(&immersedVolumes)){immersedVolumes[index]=displaced;}}
    if(body.material.z>0.5&&body.material.z<1.5){let impulse=vec3f(f32(atomicLoad(&exchange[base])),f32(atomicLoad(&exchange[base+1u])),f32(atomicLoad(&exchange[base+2u])))*1e-6;let angularImpulse=vec3f(f32(atomicLoad(&exchange[base+3u])),f32(atomicLoad(&exchange[base+4u])),f32(atomicLoad(&exchange[base+5u])))*1e-6;let weighted=vec3f(f32(atomicLoad(&exchange[base+7u])),f32(atomicLoad(&exchange[base+8u])),f32(atomicLoad(&exchange[base+9u])))*1e-4/snapshots;let velocityWeight=f32(atomicLoad(&exchange[base+11u]))/65536.0/snapshots;let pressureCoupled=atomicLoad(&exchange[base+10u])!=0;let meanVelocity=select(vec3f(0),weighted/velocityWeight,velocityWeight>1e-8);let scaledInverseMass=body.inverseMassInertia.x;let mass=select(1e30,rho/scaledInverseMass,scaledInverseMass>0.0);let immersed=clamp(displaced/max(bodyVolume(body),1e-9),0.0,1.0);let relative=body.linearVelocity.xyz-meanVelocity;let speed=length(relative);let drag=-.5*rho*params.coupling.y*3.141592653589793*body.dimensions.w*body.dimensions.w*immersed*speed*relative;let buoyancy=select(-rho*displaced*params.gravity.xyz,vec3f(0),pressureCoupled);let added=params.coupling.z*rho*displaced;let acceleration=(mass*params.gravity.xyz+impulse/max(dt,1e-8)+drag+buoyancy)/max(mass+added,1e-8);body.linearVelocity=vec4f(body.linearVelocity.xyz+acceleration*dt,body.linearVelocity.w);body.angularMomentumRestitution=vec4f(body.angularMomentumRestitution.xyz+angularImpulse,body.angularMomentumRestitution.w);body.positionShape=vec4f(body.positionShape.xyz+body.linearVelocity.xyz*dt,body.positionShape.w);body.angularVelocity=vec4f(inverseInertia(body,body.angularMomentumRestitution.xyz),body.angularVelocity.w);let derivative=qMultiply(vec4f(0,body.angularVelocity.xyz),body.orientation);body.orientation=normalize(body.orientation+.5*dt*derivative);bodies[index]=body;}
  }
  for(var iteration=0u;iteration<6u;iteration++){for(var index=0u;index<12u;index++){if(index>=count){break;}var body=bodies[index];${solidWorldContact ? "solidVoxelContact(&body);" : "planeContact(&body,vec3f(1,0,0),-.5*params.container.x);planeContact(&body,vec3f(-1,0,0),-.5*params.container.x);planeContact(&body,vec3f(0,0,1),-.5*params.container.z);planeContact(&body,vec3f(0,0,-1),-.5*params.container.z);planeContact(&body,vec3f(0,1,0),0);if(params.terrain.y>.5){planeContact(&body,vec3f(0,-1,0),-params.container.y);}"}${terrainContact ? "let terrain=terrainPlane(body.positionShape.xyz);planeContact(&body,terrain.xyz,terrain.w);" : ""}bodies[index]=body;}for(var a=0u;a<12u;a++){if(a>=count){break;}for(var b=a+1u;b<12u;b++){if(b>=count){break;}solveBodyPair(a,b);}}}
  // A held body is the host's to place. Whatever the contact iterations did to
  // it — a pair impulse from a crate it was shoved into, an angular momentum it
  // would spin off the instant it is let go — is rolled back to the pose the
  // pointer authored. It still pushes its neighbours; nothing pushes back.
  for(var index=0u;index<12u;index++){if(index>=count){break;}if(bodies[index].material.z>1.5){bodies[index]=previousBodies[index];}}
  for(var index=0u;index<12u;index++){if(index<count){publish(index);publishMotion(index,previousBodies[index],bodies[index],dt);}else{renderBodies[index]=RenderBody(vec4f(0),vec4f(0),vec4f(1,0,0,0),vec4f(0));rigidMotion[index]=SvoPrimitiveMotionRecord();}}
}
`;

/** Heightfield-contact variant retained by solvers that have not cut over to SolidWorld. */
export const gpuRigidBodyShader = gpuRigidBodyShaderSource(true);



/** Authoritative GPU rigid state. CPU writes only explicit reset/edit/drag commands. */
export class WebGPURigidBodySystem {
  readonly stateBuffer: GPUBuffer;
  readonly renderBuffer: GPUBuffer;
  /** GPU-authored 128-byte records used by surface motion and swept preactivation. */
  readonly motionBuffer: GPUBuffer;
  /** Persistent displaced volume; survives exchange clears and drives the
   * fine-volume target as bodies enter or leave the liquid. */
  readonly immersedVolumeBuffer: GPUBuffer;
  private readonly stateScratch: GPUBuffer;
  private readonly renderScratch: GPUBuffer;
  private readonly motionScratch: GPUBuffer;
  private readonly paramsBuffer: GPUBuffer;
  private pipeline!: GPUComputePipeline;
  private bindGroup!: GPUBindGroup;
  private pickPipeline!: GPUComputePipeline;
  private pickBindGroup!: GPUBindGroup;
  private shaderModule?: GPUShaderModule;
  private readonly pipelinesDeferred: boolean;
  private readonly pickParamsBuffer: GPUBuffer;
  private readonly pickResultBuffer: GPUBuffer;
  private bodyIds: string[] = [];
  private structuralSignatures: string[] = [];
  private authoredTransformSignatures: string[] = [];
  private commandSignatures: string[] = [];
  private bodyCount = 0;
  private selectedIndex = -1;
  private motionGenerations: number[] = [];
  private solidWorldCollisionSource?: GPURigidSolidWorldCollisionSource;

  // No `deferPipelineCompilation` parameter: the two rigid pipelines are always
  // compiled through `initializationTasks`, because a scene that starts without
  // bodies is a scene that can still be given one, and the first body dropped
  // into a running bodyless scene used to reach `encode` with no pipeline at
  // all. Both callers passed their own defer flag through and it named nothing.
  constructor(private readonly device: GPUDevice, private scene: SceneDescription,
    readonly exchangeBuffer: GPUBuffer, private readonly terrainTexture?: GPUTexture) {
    this.pipelinesDeferred = true;
    this.stateBuffer = device.createBuffer({ label: "GPU authoritative rigid-body state", size: GPU_RIGID_STATE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.renderBuffer = device.createBuffer({ label: "GPU rigid-body render records", size: GPU_RIGID_RENDER_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.motionBuffer = device.createBuffer({ label: "GPU rigid primitive motion sidecar", size: GPU_RIGID_MOTION_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.immersedVolumeBuffer = device.createBuffer({ label: "GPU rigid immersed volumes",
      size: GPU_RIGID_IMMERSED_VOLUME_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.stateScratch = device.createBuffer({ label: "GPU rigid-body roster scratch", size: GPU_RIGID_STATE_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.renderScratch = device.createBuffer({ label: "GPU rigid render roster scratch", size: GPU_RIGID_RENDER_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.motionScratch = device.createBuffer({ label: "GPU rigid motion roster scratch", size: GPU_RIGID_MOTION_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.paramsBuffer = device.createBuffer({ label: "GPU rigid-body step parameters", size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.pickParamsBuffer = device.createBuffer({ label: "GPU rigid-body pick ray", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.pickResultBuffer = device.createBuffer({ label: "GPU rigid-body pick result", size: GPU_RIGID_PICK_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  }

  /** Adopt authored scene scalars and roster metadata without replacing GPU state. */
  setScene(scene: SceneDescription): void { this.scene = scene; }

  private descriptor(entryPoint: "integrate" | "pickRigidBody"): GPUComputePipelineDescriptor {
    if (!this.terrainTexture && !this.solidWorldCollisionSource) {
      throw new Error("Rigid-body static contact requires terrain or SolidWorld occupancy");
    }
    this.shaderModule ??= this.device.createShaderModule({
      label: "GPU resident rigid-body solver",
      code: gpuRigidBodyShaderSource(this.terrainTexture !== undefined,
        this.solidWorldCollisionSource),
    });
    return { label: entryPoint === "integrate"
      ? "GPU resident rigid-body integrate/contact" : "GPU resident rigid-body ray pick",
      layout: "auto", compute: { module: this.shaderModule, entryPoint } };
  }

  private createIntegrationBindings(): void {
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.stateBuffer } },
      { binding: 1, resource: { buffer: this.exchangeBuffer } },
      { binding: 2, resource: { buffer: this.renderBuffer } },
      { binding: 3, resource: { buffer: this.paramsBuffer } },
      { binding: 7, resource: { buffer: this.motionBuffer } },
      { binding: 8, resource: { buffer: this.immersedVolumeBuffer } },
    ];
    if (this.terrainTexture) entries.push({ binding: 4,
      resource: this.terrainTexture.createView() });
    if (this.solidWorldCollisionSource) entries.push({ binding: 9,
      resource: { buffer: this.solidWorldCollisionSource.buffer } });
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0), entries,
    });
  }

  /**
   * Attach the adaptive solver's resident SolidWorld cache before compilation.
   * The source remains GPU-resident and is updated in place by voxel edits.
   */
  setSolidWorldCollisionSource(source: GPURigidSolidWorldCollisionSource): void {
    if (this.shaderModule || this.pipeline) {
      throw new Error("SolidWorld collision source must be attached before rigid compilation");
    }
    if (![source.baseWords, source.directoryCapacity, source.directoryBaseWords,
      source.regionCapacity, source.regionBaseWords, source.regionWords,
      source.entryWords, source.pageBaseWords, source.pageWords, source.fractionPageWords]
      .every((value) => Number.isSafeInteger(value) && value >= 0)
      || source.directoryCapacity < 1
      || source.entryWords < 6
      || source.regionWords < 8
      || source.pageWords < source.fractionPageWords
      || (source.directoryCapacity & (source.directoryCapacity - 1)) !== 0
      || !source.origin_m.every(Number.isFinite)
      || !source.cellSize_m.every((value) => Number.isFinite(value) && value > 0)) {
      throw new RangeError("SolidWorld collision source is invalid");
    }
    this.solidWorldCollisionSource = source;
  }

  private createPickBindings(): void {
    this.pickBindGroup = this.device.createBindGroup({ layout: this.pickPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.stateBuffer } },
      { binding: 5, resource: { buffer: this.pickParamsBuffer } },
      { binding: 6, resource: { buffer: this.pickResultBuffer } },
    ] });
  }

  /**
   * Compile the two rigid pipelines, whether or not the scene has bodies yet.
   *
   * It used to skip an empty roster, which made the pipelines a *compiled policy
   * shaped by solid presence*: the first body dropped into a running scene then
   * reached `encode` with `this.pipeline` still undefined and killed the device
   * on `setPipeline(undefined)`. That gate is the only reason the roster length
   * had to force a rebuild, so removing it is what actually makes
   * `adoptsRigidRosterShape` true rather than merely declared.
   *
   * The cost is two small pipelines from one module for a scene that never
   * gains a body, next to the several hundred a solver already compiles.
   */
  initializationTasks(): GPUInitializationTask[] {
    if (!this.pipelinesDeferred) return [];
    return [
      { id: "rigid.pipeline.integrate", phase: "solver-pipelines",
        label: "Compile rigid-body integration",
        run: async () => { this.pipeline = await this.device.createComputePipelineAsync(
          this.descriptor("integrate")); this.createIntegrationBindings(); } },
      { id: "rigid.pipeline.pick", phase: "solver-pipelines", label: "Compile rigid-body picking",
        run: async () => { this.pickPipeline = await this.device.createComputePipelineAsync(
          this.descriptor("pickRigidBody")); this.createPickBindings(); } },
    ];
  }

  syncBodies(bodies: readonly RigidBodyState[]) {
    const active = bodies.slice(0, GPU_RIGID_BODY_CAPACITY);
    const ids = active.map((body) => body.description.id);
    const structural = active.map((body) => JSON.stringify([body.description.shape,body.description.dimensions_m,body.description.density_kg_m3,body.description.restitution,body.description.friction,body.description.motion]));
    const authored = active.map((body) => JSON.stringify([body.description.position_m,body.description.orientation,body.description.linearVelocity_m_s,body.description.angularVelocity_rad_s]));
    const commands = active.map((body) => JSON.stringify([body.position_m,body.orientation,body.linearVelocity_m_s,body.angularVelocity_rad_s,Boolean(body.held)]));
    if (JSON.stringify([ids,structural,authored,commands]) === JSON.stringify([this.bodyIds,this.structuralSignatures,this.authoredTransformSignatures,this.commandSignatures])) return;
    this.bodyCount = active.length;
    const stateStorage = new ArrayBuffer(GPU_RIGID_STATE_BYTES), state = new Float32Array(stateStorage), stateWords = new Uint32Array(stateStorage);
    const render = new Float32Array(GPU_RIGID_BODY_CAPACITY * GPU_RIGID_RENDER_FLOATS);
    const immersed = new Float32Array(GPU_RIGID_BODY_CAPACITY);
    const nextMotionGenerations = ids.map((id, index) => {
      const previous = this.bodyIds.indexOf(id);
      if (previous < 0) return 1;
      const discontinuous = structural[index] !== this.structuralSignatures[previous]
        || authored[index] !== this.authoredTransformSignatures[previous]
        || commands[index] !== this.commandSignatures[previous];
      const generation = this.motionGenerations[previous] || 1;
      if (!discontinuous) return generation;
      const next = (generation + 1) >>> 0;
      return next === 0 ? 1 : next;
    });
    // The shape table's palette, not a local copy of it: this line held the
    // fifth copy of the same four literals, so the day a fifth shape was added
    // `palette[shape]` became undefined and spreading it took the device down
    // the moment a cup was placed. Indexed by code, which is what the table
    // orders it by.
    const palette = SCENE_SHAPE_PALETTE_LINEAR;
    active.forEach((body, index) => {
      const o = index * GPU_RIGID_STATE_FLOATS, d = body.description.dimensions_m, q = body.orientation, shape = sceneShapeCode(body.description.shape);
      const rho = this.scene.fluid.density_kg_m3;
      // Same reason as the palette: this was a hand-rolled ternary chain that
      // fell through to the capsule's radius for anything it did not name, so a
      // cup was packed with a bounding sphere it does not have.
      const radius = boundingRadius(body.description);
      // A held body is immovable rather than weightless: zero inverse mass and
      // inertia is what makes `planeContact` return early and `applyImpulse` a
      // no-op, so a wall or a neighbouring crate cannot push the thing in the
      // user's hand. Gravity is refused separately, by the motion lane below.
      const inverseMass = body.held ? 0 : body.inverseMass_kg * rho;
      const inverseInertia = body.held
        ? { x: 0, y: 0, z: 0 }
        : { x: body.inverseInertiaBody_kg_m2.x*rho, y: body.inverseInertiaBody_kg_m2.y*rho, z: body.inverseInertiaBody_kg_m2.z*rho };
      state.set([body.position_m.x,body.position_m.y,body.position_m.z,shape,d.x,d.y,d.z,radius,q.w,q.x,q.y,q.z,body.linearVelocity_m_s.x,body.linearVelocity_m_s.y,body.linearVelocity_m_s.z,inverseMass,body.angularVelocity_rad_s.x,body.angularVelocity_rad_s.y,body.angularVelocity_rad_s.z,body.description.density_kg_m3,inverseMass,inverseInertia.x,inverseInertia.y,inverseInertia.z,body.angularMomentum_kg_m2_s.x,body.angularMomentum_kg_m2_s.y,body.angularMomentum_kg_m2_s.z,body.description.restitution,body.description.friction,0,rigidMotionLane(body),0],o);
      stateWords[o+29]=nextMotionGenerations[index];
      const half = sceneShapeRenderHalfExtent_m(body.description.shape, d);
      render.set([body.position_m.x,body.position_m.y,body.position_m.z,state[o+7],half[0],half[1],half[2],shape,q.w,q.x,q.y,q.z,...palette[shape],0],index*GPU_RIGID_RENDER_FLOATS);
      const volume = primitiveVolume(body.description.shape, body.description.dimensions_m);
      const waterline = this.scene.container.fillFraction * this.scene.container.height_m;
      if (body.description.shape === "sphere") {
        const radius = body.description.dimensions_m.x;
        const cap = Math.max(0, Math.min(2 * radius,
          waterline - (body.position_m.y - radius)));
        immersed[index] = Math.PI * cap * cap * (radius - cap / 3);
      } else {
        const radiusY = Math.max(1e-9, half[1]);
        immersed[index] = volume * Math.max(0, Math.min(1,
          (waterline - (body.position_m.y - radiusY)) / (2 * radiusY)));
      }
    });
    if (this.bodyIds.length === 0) {
      this.device.queue.writeBuffer(this.stateBuffer,0,state);this.device.queue.writeBuffer(this.renderBuffer,0,render);
      this.device.queue.writeBuffer(this.immersedVolumeBuffer,0,immersed);
    } else {
      const rosterChanged = ids.length !== this.bodyIds.length || ids.some((id,index) => id !== this.bodyIds[index]);
      if (rosterChanged) {
        const encoder=this.device.createCommandEncoder({label:"Compact GPU resident rigid-body roster"});
        ids.forEach((id,index) => { const previous=this.bodyIds.indexOf(id); if(previous<0)return; encoder.copyBufferToBuffer(this.stateBuffer,previous*GPU_RIGID_STATE_FLOATS*4,this.stateScratch,index*GPU_RIGID_STATE_FLOATS*4,GPU_RIGID_STATE_FLOATS*4);encoder.copyBufferToBuffer(this.renderBuffer,previous*GPU_RIGID_RENDER_FLOATS*4,this.renderScratch,index*GPU_RIGID_RENDER_FLOATS*4,GPU_RIGID_RENDER_FLOATS*4);encoder.copyBufferToBuffer(this.motionBuffer,previous*SVO_PRIMITIVE_MOTION_STRIDE_BYTES,this.motionScratch,index*SVO_PRIMITIVE_MOTION_STRIDE_BYTES,SVO_PRIMITIVE_MOTION_STRIDE_BYTES); });
        encoder.copyBufferToBuffer(this.stateScratch,0,this.stateBuffer,0,GPU_RIGID_STATE_BYTES);encoder.copyBufferToBuffer(this.renderScratch,0,this.renderBuffer,0,GPU_RIGID_RENDER_BYTES);encoder.copyBufferToBuffer(this.motionScratch,0,this.motionBuffer,0,GPU_RIGID_MOTION_BYTES);this.device.queue.submit([encoder.finish()]);
      }
      active.forEach((_body,index) => {
        const previous=this.bodyIds.indexOf(ids[index]);
        const stateOffset=index*GPU_RIGID_STATE_FLOATS,renderOffset=index*GPU_RIGID_RENDER_FLOATS;
        if(previous<0 || authored[index] !== this.authoredTransformSignatures[previous] || (!rosterChanged && commands[index] !== this.commandSignatures[previous] && structural[index] === this.structuralSignatures[previous])) {
          this.device.queue.writeBuffer(this.stateBuffer,stateOffset*4,state.subarray(stateOffset,stateOffset+GPU_RIGID_STATE_FLOATS));this.device.queue.writeBuffer(this.renderBuffer,renderOffset*4,render.subarray(renderOffset,renderOffset+GPU_RIGID_RENDER_FLOATS));return;
        }
        if(structural[index] !== this.structuralSignatures[previous]) {
          this.device.queue.writeBuffer(this.stateBuffer,(stateOffset+3)*4,state.subarray(stateOffset+3,stateOffset+8));
          this.device.queue.writeBuffer(this.stateBuffer,(stateOffset+15)*4,state.subarray(stateOffset+15,stateOffset+16));
          this.device.queue.writeBuffer(this.stateBuffer,(stateOffset+19)*4,state.subarray(stateOffset+19,stateOffset+24));
          this.device.queue.writeBuffer(this.stateBuffer,(stateOffset+27)*4,state.subarray(stateOffset+27,stateOffset+31));
          this.device.queue.writeBuffer(this.renderBuffer,(renderOffset+3)*4,render.subarray(renderOffset+3,renderOffset+8));
          this.device.queue.writeBuffer(this.renderBuffer,(renderOffset+12)*4,render.subarray(renderOffset+12,renderOffset+15));
        }
      });
    }
    this.bodyIds=ids;this.structuralSignatures=structural;this.authoredTransformSignatures=authored;this.commandSignatures=commands;this.motionGenerations=nextMotionGenerations;
  }

  setSelectedIndex(index: number) {
    const next = index >= 0 && index < this.bodyCount ? index : -1;
    if (next === this.selectedIndex) return;
    const write = (bodyIndex: number, selected: number) => { if (bodyIndex < 0) return; this.device.queue.writeBuffer(this.stateBuffer, bodyIndex * GPU_RIGID_STATE_FLOATS * 4 + 31 * 4, new Float32Array([selected])); this.device.queue.writeBuffer(this.renderBuffer, bodyIndex * GPU_RIGID_RENDER_FLOATS * 4 + 15 * 4, new Float32Array([selected])); };
    write(this.selectedIndex,0);write(next,1);this.selectedIndex=next;
  }

  /**
   * The poses the renderer is drawing this frame.
   *
   * `syncBodies` only ever writes *toward* the GPU: once a run starts, gravity,
   * buoyancy and contacts all happen here, and the host roster keeps whatever
   * pose the last explicit command left it with. Every gesture that has to line
   * up with the image on screen — the grab that opens a drag, which needs the
   * body's centre to hold its offset from the cursor — must read the poses back
   * rather than trust that roster, or it starts the drag from a body that
   * settled seconds ago.
   *
   * User-triggered and bounded, like `pick`: one 768-byte copy per gesture.
   */
  async readPoses(): Promise<GPURigidBodyPose[] | undefined> {
    if (this.bodyCount === 0) return undefined;
    const readback = this.device.createBuffer({ label: "GPU rigid-body pose readback", size: GPU_RIGID_RENDER_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read GPU resident rigid-body poses" });
    encoder.copyBufferToBuffer(this.renderBuffer, 0, readback, 0, GPU_RIGID_RENDER_BYTES);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      return decodeGPURigidBodyPoses(new Float32Array(readback.getMappedRange()), this.bodyCount);
    } catch { return undefined; }
    finally { if (readback.mapState === "mapped") readback.unmap(); readback.destroy(); }
  }

  /** A bounded, user-triggered readback used only to begin mouse interaction. */
  async pick(origin: Vec3, direction: Vec3): Promise<GPURigidBodyPick | undefined> {
    if (this.bodyCount === 0) return undefined;
    this.device.queue.writeBuffer(this.pickParamsBuffer,0,new Float32Array([origin.x,origin.y,origin.z,this.bodyCount,direction.x,direction.y,direction.z,0]));
    const readback=this.device.createBuffer({label:"GPU rigid-body pick readback",size:GPU_RIGID_PICK_BYTES,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const encoder=this.device.createCommandEncoder({label:"Pick GPU resident rigid body"});
    const pass=encoder.beginComputePass({label:"Ray-pick GPU resident rigid bodies"});pass.setPipeline(this.pickPipeline);pass.setBindGroup(0,this.pickBindGroup);pass.dispatchWorkgroups(1);pass.end();
    encoder.copyBufferToBuffer(this.pickResultBuffer,0,readback,0,GPU_RIGID_PICK_BYTES);this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const bytes=readback.getMappedRange(),words=new Uint32Array(bytes),values=new Float32Array(bytes);
      if(words[1]===0||words[0]>=this.bodyCount)return undefined;
      return {bodyIndex:words[0],distance_m:values[2],position_m:{x:values[4],y:values[5],z:values[6]},orientation:{w:values[8],x:values[9],y:values[10],z:values[11]}};
    } catch { return undefined; }
    finally { if(readback.mapState==="mapped")readback.unmap();readback.destroy(); }
  }

  encode(encoder: GPUCommandEncoder, dt_s: number, cellVolume_m3: number, snapshotCount = 1, cellHeight_m = 1) {
    const c=this.scene.container,g=this.scene.fluid.gravity_m_s2;
    const solid=this.solidWorldCollisionSource;
    this.device.queue.writeBuffer(this.paramsBuffer,0,new Float32Array([dt_s,this.scene.fluid.density_kg_m3,this.bodyCount,Math.max(1,snapshotCount),g.x,g.y,g.z,0,c.width_m,c.height_m,c.depth_m,this.terrainTexture&&sceneHasTerrain(this.scene)?1:0,cellHeight_m,c.top === "closed"?1:0,0,0,cellVolume_m3,.9,.5,Math.max(cellHeight_m,1e-6),...(solid?.origin_m??[0,0,0]),solid?1:0,...(solid?.cellSize_m??[1,1,1]),0]));
    const pass=encoder.beginComputePass({label:"GPU resident rigid-body integration and contacts"});pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.bindGroup);pass.dispatchWorkgroups(1);pass.end();
  }

  destroy() { this.stateBuffer.destroy();this.renderBuffer.destroy();this.motionBuffer.destroy();this.immersedVolumeBuffer.destroy();this.stateScratch.destroy();this.renderScratch.destroy();this.motionScratch.destroy();this.paramsBuffer.destroy();this.pickParamsBuffer.destroy();this.pickResultBuffer.destroy(); }
}
