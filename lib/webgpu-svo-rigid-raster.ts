import { SVO_GBUFFER_FIELD_SOURCES, SVO_GBUFFER_MOTION_KINDS, svoGBufferWGSL } from "./svo-gbuffer";
import { svoPrimitiveMotionWGSL } from "./svo-primitive-motion";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";

/**
 * Standalone ABI for moving rigid bodies rendered into the dry split-geometry
 * bridge. The pass deliberately owns no history: the caller binds the live
 * camera uniform and live BodyGPU buffer, then draws 36 vertices per body.
 */
export const SVO_RIGID_RASTER_CONTRACT = Object.freeze({
  maximumBodies: 12,
  bodyStrideBytes: 64,
  bodyBufferBytes: 12 * 64,
  bodySourcePolicy: "bind-live-buffer-directly" as const,
  bodyBufferRequiredUsage: Object.freeze(["UNIFORM", "STORAGE", "COPY_DST"] as const),
  verticesPerProxy: 36,
  uniformBinding: 0,
  bodyBinding: 1,
  rigidMotionBinding: 14,
  uniformBufferType: "uniform" as const,
  bodyBufferType: "read-only-storage" as const,
  rigidMotionBufferType: "uniform" as const,
  splitOutputGroup: 1,
  packedSurfaceReadBinding: 0,
  identityMediaReadBinding: 1,
  splitGeometryBinding: 2,
  splitIdentityBinding: 3,
  primaryGeometryReadBinding: 4,
  splitBridgeWorkgroupSize: 8,
  splitBridgeEntryPoint: "rigidSplitBridgeFragment" as const,
  splitBridgeVisibilityAuthority: "current-frame-depth-tested-certificate" as const,
  splitBridgeUsesDepthAttachment: false,
  primaryGeometryFormat: "rg32uint" as const,
  primaryGeometryEncoding: "bitcast-f32-t-plus-oct12-normal-owner-feature-motion" as const,
  // Compatibility values for integrations that have not yet removed their
  // depth attachment. `always` leaves the packed winner guard authoritative.
  splitBridgeDepthCompare: "always" as const,
  splitBridgeDepthWriteEnabled: false,
  geometryFormat: "rgba32float" as const,
  identityFormat: "rg32uint" as const,
  packedSurfaceFormat: "rgba32uint" as const,
  identityMediaFormat: "rgba16uint" as const,
  colorAttachmentBytesPerSample: 32,
  depthFormat: "depth32float" as const,
  depthClearValue: 0,
  depthCompare: "greater" as const,
  depthWriteEnabled: true,
  reversedZNear_m: 0.01,
  bodyMaterialMarker: 0x8000_0000,
});

/**
 * Dedicated group-zero layout for the rigid pass. Do not reuse the dry
 * megakernel layout: its binding 1 is a uniform buffer, while this shader
 * deliberately consumes the same live BodyGPU allocation as read-only
 * storage. Binding the live allocation directly is exact for the current
 * frame and avoids a redundant (and, without COPY_SRC, invalid) mirror copy.
 */
export function svoRigidRasterInputBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return [
    {
      binding: SVO_RIGID_RASTER_CONTRACT.uniformBinding,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
      buffer: { type: SVO_RIGID_RASTER_CONTRACT.uniformBufferType },
    },
    {
      binding: SVO_RIGID_RASTER_CONTRACT.bodyBinding,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
      buffer: { type: SVO_RIGID_RASTER_CONTRACT.bodyBufferType },
    },
    {
      binding: SVO_RIGID_RASTER_CONTRACT.rigidMotionBinding,
      visibility: GPUShaderStage.FRAGMENT,
      buffer: { type: SVO_RIGID_RASTER_CONTRACT.rigidMotionBufferType },
    },
  ];
}

/** Diagnostic full-screen bridge resources; production uses the certificate bridge below. */
export function svoRigidRasterOutputBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return [
    {
      binding: SVO_RIGID_RASTER_CONTRACT.packedSurfaceReadBinding,
      visibility: GPUShaderStage.COMPUTE,
      texture: { sampleType: "uint" },
    },
    {
      binding: SVO_RIGID_RASTER_CONTRACT.identityMediaReadBinding,
      visibility: GPUShaderStage.COMPUTE,
      texture: { sampleType: "uint" },
    },
    {
      binding: SVO_RIGID_RASTER_CONTRACT.splitGeometryBinding,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: "write-only", format: SVO_RIGID_RASTER_CONTRACT.geometryFormat },
    },
    {
      binding: SVO_RIGID_RASTER_CONTRACT.splitIdentityBinding,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: "write-only", format: SVO_RIGID_RASTER_CONTRACT.identityFormat },
    },
  ];
}

/**
 * Coverage-scaled bridge inputs. Split geometry and identity are render
 * attachments, not storage bindings: the second proxy pass samples only the
 * depth-tested current-frame certificate produced by the first pass.
 */
export function svoRigidRasterCoverageBridgeBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return [
    {
      binding: SVO_RIGID_RASTER_CONTRACT.primaryGeometryReadBinding,
      visibility: GPUShaderStage.FRAGMENT,
      texture: { sampleType: "uint" },
    },
  ];
}

export const SVO_RIGID_RASTER_FEATURES = Object.freeze({
  smooth: 0,
  boxX: 1,
  boxY: 2,
  boxZ: 3,
  cylinderSide: 4,
  cylinderCap: 5,
});

/**
 * Required production ordering. The raster pass replaces only primary
 * intersection discovery; it must not become an unlit overlay. Its records
 * enter the same reduced visibility, PBR, motion, AO, and shadow consumers as
 * an analytic rigid hit does today.
 */
export const SVO_RIGID_RASTER_INTEGRATION = Object.freeze({
  stage: "inside-svo-renderer-after-static-primary-before-cone-prepass" as const,
  splitBridge: "second-proxy-raster-pass-unpacks-current-frame-winner-certificate" as const,
  deferredMaterial: "resolve-body-material-marker-from-current-BodyGPU" as const,
  motion: "resolve-current-surface-velocity-from-rigidMotion" as const,
  ambientOcclusion: "include-current-rigid-dynamic-occupancy-overlay" as const,
  shadows: "multiply-static-svo-visibility-by-current-rigid-shadow-overlay" as const,
  history: "none" as const,
});

/** CPU mirror of the two uints written to the existing split identity bridge. */
export function packSvoRigidRasterSplitIdentity(bodyIndex: number, featureId: number): readonly [number, number] {
  if (!Number.isInteger(bodyIndex) || bodyIndex < 0 || bodyIndex >= SVO_RIGID_RASTER_CONTRACT.maximumBodies) {
    throw new RangeError(`Rigid raster body index must be from 0 to ${SVO_RIGID_RASTER_CONTRACT.maximumBodies - 1}`);
  }
  if (!Number.isInteger(featureId) || featureId < 0 || featureId > 0xf) {
    throw new RangeError("Rigid raster feature ID must fit four bits");
  }
  const material = (SVO_RIGID_RASTER_CONTRACT.bodyMaterialMarker | bodyIndex) >>> 0;
  const metadata = (bodyIndex
    | featureId << 16
    | SVO_GBUFFER_FIELD_SOURCES.analyticPrimitive << 20
    | SVO_GBUFFER_MOTION_KINDS.rigid << 24) >>> 0;
  return [material, metadata];
}

/**
 * BodyGPU and the first five uniform lanes exactly mirror webgpu-renderer.ts.
 * Binding the existing buffers therefore needs no upload, conversion, or
 * camera cache. BodyGPU is exposed as read-only storage to avoid dynamically
 * indexing a uniform array in the fragment stage on Metal.
 *
 * Production packedSurface, identityMedia, and the 8-byte exact-depth/octahedral
 * normal certificate total the baseline 32-byte attachment limit exactly. A
 * second proxy raster pass reads that depth-tested certificate and writes
 * split `rgba32float(normal.xyz, t)` plus `rg32uint(materialWord,
 * metadataWord)` as ordinary color attachments. It rejects every fragment
 * whose instance does not own the certificate and performs no second analytic
 * intersection. Work therefore scales with body proxy coverage rather than
 * the viewport, and storage side effects never race late depth.
 */
export const svoRigidRasterShader = /* wgsl */ `
const RIGID_RASTER_MAX_BODIES:u32=${SVO_RIGID_RASTER_CONTRACT.maximumBodies}u;
const RIGID_RASTER_VERTEX_COUNT:u32=${SVO_RIGID_RASTER_CONTRACT.verticesPerProxy}u;
const RIGID_RASTER_NEAR_M:f32=${SVO_RIGID_RASTER_CONTRACT.reversedZNear_m};
const RIGID_RASTER_MISS:f32=3.402823e38;
const RIGID_RASTER_FIELD_ANALYTIC:u32=${SVO_GBUFFER_FIELD_SOURCES.analyticPrimitive}u;
const RIGID_RASTER_MOTION_RIGID:u32=${SVO_GBUFFER_MOTION_KINDS.rigid}u;
const RIGID_RASTER_FEATURE_SMOOTH:u32=${SVO_RIGID_RASTER_FEATURES.smooth}u;
const RIGID_RASTER_FEATURE_BOX_X:u32=${SVO_RIGID_RASTER_FEATURES.boxX}u;
const RIGID_RASTER_FEATURE_CYLINDER_SIDE:u32=${SVO_RIGID_RASTER_FEATURES.cylinderSide}u;
const RIGID_RASTER_FEATURE_CYLINDER_CAP:u32=${SVO_RIGID_RASTER_FEATURES.cylinderCap}u;
const RIGID_RASTER_MEDIUM_AIR:u32=0u;
const RIGID_RASTER_MEDIUM_OPAQUE:u32=3u;
const RIGID_RASTER_MATERIAL_SPHERE:u32=${VOXEL_MATERIAL_IDS.sphere}u;
const RIGID_RASTER_MATERIAL_BOX:u32=${VOXEL_MATERIAL_IDS.box}u;
const RIGID_RASTER_MATERIAL_CAPSULE:u32=${VOXEL_MATERIAL_IDS.capsule}u;
const RIGID_RASTER_MATERIAL_CYLINDER:u32=${VOXEL_MATERIAL_IDS.cylinder}u;

${svoPrimitiveMotionWGSL}
${svoGBufferWGSL}

// Prefix-compatible with the production Uniforms arena. options.z is the live
// body count; viewport/camera lanes are rewritten by the renderer every frame.
struct RigidRasterUniforms{
  viewport:vec4f,
  cameraPosition:vec4f,
  cameraTarget:vec4f,
  container:vec4f,
  options:vec4f,
}
// Existing 64-byte BodyGPU ABI; orientation is quaternion wxyz.
struct BodyGPU{
  positionRadius:vec4f,
  halfSizeShape:vec4f,
  orientation:vec4f,
  colorSelected:vec4f,
}
@group(0) @binding(0) var<uniform> rigidUniforms:RigidRasterUniforms;
@group(0) @binding(1) var<storage,read> rigidBodies:array<BodyGPU>;
@group(0) @binding(14) var<uniform> rigidMotion:array<SvoPrimitiveMotionRecord,12>;
@group(1) @binding(0) var rigidPackedSurfaceRead:texture_2d<u32>;
@group(1) @binding(1) var rigidIdentityMediaRead:texture_2d<u32>;
@group(1) @binding(2) var rigidSplitGeometryWrite:texture_storage_2d<rgba32float,write>;
@group(1) @binding(3) var rigidSplitIdentityWrite:texture_storage_2d<rg32uint,write>;
@group(1) @binding(4) var rigidPrimaryGeometryRead:texture_2d<u32>;

fn rigidQrot(q:vec4f,v:vec3f)->vec3f{
  let a=cross(q.yzw,v);return v+2.0*(q.x*a+cross(q.yzw,a));
}
fn rigidQinv(q:vec4f,v:vec3f)->vec3f{return rigidQrot(vec4f(q.x,-q.yzw),v);}
fn rigidPackPrimaryGeometryMetadata(normalIn:vec3f,bodyIndex:u32,featureId:u32,motionValid:u32)->u32{
  let normal=normalize(normalIn);let inverseL1=1.0/(abs(normal.x)+abs(normal.y)+abs(normal.z));var oct=normal.xy*inverseL1;
  if(normal.z<0.0){let original=oct;oct=(vec2f(1.0)-abs(original.yx))*select(vec2f(-1.0),vec2f(1.0),original>=vec2f(0.0));}
  let encoded=vec2u(round(clamp(oct*0.5+0.5,vec2f(0.0),vec2f(1.0))*4095.0));return encoded.x|(encoded.y<<12u)|((bodyIndex&15u)<<24u)|((featureId&7u)<<28u)|((motionValid&1u)<<31u);
}
fn rigidUnpackPrimaryNormal(packed:u32)->vec3f{
  let encoded=vec2f(f32(packed&0xfffu),f32((packed>>12u)&0xfffu))/4095.0*2.0-1.0;var normal=vec3f(encoded,1.0-abs(encoded.x)-abs(encoded.y));
  if(normal.z<0.0){let original=normal.xy;normal=vec3f((vec2f(1.0)-abs(original.yx))*select(vec2f(-1.0),vec2f(1.0),original>=vec2f(0.0)),normal.z);}return normalize(normal);
}
fn rigidProxyExtent(body:BodyGPU)->vec3f{
  let shape=i32(round(body.halfSizeShape.w));let radius=max(body.halfSizeShape.x,1e-6);
  var extent=max(body.halfSizeShape.xyz,vec3f(1e-6));
  if(shape==0){extent=vec3f(radius);}
  else if(shape==2){extent=vec3f(radius,max(body.halfSizeShape.y,0.0)+radius,radius);}
  else if(shape==3){extent=vec3f(radius,max(body.halfSizeShape.y,1e-6),radius);}
  // A tiny world-space expansion makes the proxy conservative at silhouettes;
  // the analytic fragment test remains the sole visibility authority.
  return extent+vec3f(1e-5);
}

fn rigidProxyCorner(index:u32)->vec3f{
  let corners=array<vec3f,8>(
    vec3f(-1,-1,-1),vec3f(1,-1,-1),vec3f(1,1,-1),vec3f(-1,1,-1),
    vec3f(-1,-1,1),vec3f(1,-1,1),vec3f(1,1,1),vec3f(-1,1,1));
  let indices=array<u32,36>(
    0u,2u,1u,0u,3u,2u,4u,5u,6u,4u,6u,7u,
    0u,4u,7u,0u,7u,3u,1u,2u,6u,1u,6u,5u,
    0u,1u,5u,0u,5u,4u,3u,7u,6u,3u,6u,2u);
  return corners[indices[index]];
}

struct RigidRasterVertexOut{
  @builtin(position) position:vec4f,
  @location(0) @interpolate(flat) bodyIndex:u32,
}
@vertex fn rigidRasterVertex(
  @builtin(vertex_index) vertexIndex:u32,
  @builtin(instance_index) instanceIndex:u32,
)->RigidRasterVertexOut{
  var output:RigidRasterVertexOut;output.bodyIndex=instanceIndex;
  let bodyCount=min(u32(round(max(rigidUniforms.options.z,0.0))),min(RIGID_RASTER_MAX_BODIES,arrayLength(&rigidBodies)));
  if(instanceIndex>=bodyCount||vertexIndex>=RIGID_RASTER_VERTEX_COUNT){output.position=vec4f(2.0,2.0,0.0,1.0);return output;}
  let body=rigidBodies[instanceIndex];
  let local=rigidProxyCorner(vertexIndex)*rigidProxyExtent(body);
  let world=body.positionRadius.xyz+rigidQrot(body.orientation,local);
  let camera=rigidUniforms.cameraPosition.xyz;let forward=normalize(rigidUniforms.cameraTarget.xyz-camera);
  let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));
  let relative=world-camera;let viewDepth=max(dot(relative,forward),RIGID_RASTER_NEAR_M);
  let aspect=rigidUniforms.viewport.x/max(rigidUniforms.viewport.y,1.0);
  // This is the exact inverse of the production ray construction. Proxy depth
  // only participates in clipping; the fragment writes the analytic depth.
  output.position=vec4f(dot(relative,right)/(aspect*.72),dot(relative,up)/.72,0.0,viewDepth);
  return output;
}

fn rigidRasterRay(pixelCoordinate:vec2f)->mat2x3f{
  let pixelUv=vec2f(pixelCoordinate.x/max(rigidUniforms.viewport.x,1.0),1.0-pixelCoordinate.y/max(rigidUniforms.viewport.y,1.0));let ndc=pixelUv*2.0-1.0;
  let origin=rigidUniforms.cameraPosition.xyz;let forward=normalize(rigidUniforms.cameraTarget.xyz-origin);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));
  let direction=normalize(forward+right*ndc.x*rigidUniforms.viewport.x/max(rigidUniforms.viewport.y,1.0)*.72+up*ndc.y*.72);return mat2x3f(origin,direction);
}

struct RigidRasterHit{t:f32,normal:vec3f,featureId:u32,valid:u32}
fn rigidMiss()->RigidRasterHit{return RigidRasterHit(RIGID_RASTER_MISS,vec3f(0,1,0),RIGID_RASTER_FEATURE_SMOOTH,0u);}
fn rigidSphereHit(origin:vec3f,direction:vec3f,radius:f32)->RigidRasterHit{
  let halfB=dot(origin,direction);let c=dot(origin,origin)-radius*radius;let discriminant=halfB*halfB-c;
  if(discriminant<0.0){return rigidMiss();}let root=sqrt(max(discriminant,0.0));var t=-halfB-root;
  if(t<=1e-4){t=-halfB+root;}if(t<=1e-4){return rigidMiss();}
  return RigidRasterHit(t,normalize(origin+direction*t),RIGID_RASTER_FEATURE_SMOOTH,1u);
}
fn rigidBoxHit(origin:vec3f,direction:vec3f,extent:vec3f)->RigidRasterHit{
  var nearT=-RIGID_RASTER_MISS;var farT=RIGID_RASTER_MISS;
  for(var axis=0u;axis<3u;axis+=1u){
    if(abs(direction[axis])<=1e-9){if(origin[axis]<-extent[axis]||origin[axis]>extent[axis]){return rigidMiss();}}
    else{let first=(-extent[axis]-origin[axis])/direction[axis];let second=(extent[axis]-origin[axis])/direction[axis];nearT=max(nearT,min(first,second));farT=min(farT,max(first,second));if(nearT>farT){return rigidMiss();}}
  }
  let t=select(nearT,farT,nearT<=1e-4);if(t<=1e-4){return rigidMiss();}
  let point=origin+direction*t;let q=abs(point/max(extent,vec3f(1e-6)));var axis=0u;if(q.y>q.x){axis=1u;}if(q.z>q[axis]){axis=2u;}
  var normal=vec3f(0.0);normal[axis]=select(-1.0,1.0,point[axis]>=0.0);
  return RigidRasterHit(t,normal,RIGID_RASTER_FEATURE_BOX_X+axis,1u);
}
fn rigidCylinderSideHit(origin:vec3f,direction:vec3f,radius:f32,halfHeight:f32)->RigidRasterHit{
  let a=dot(direction.xz,direction.xz);if(a<=1e-12){return rigidMiss();}
  let halfB=dot(origin.xz,direction.xz);let c=dot(origin.xz,origin.xz)-radius*radius;let discriminant=halfB*halfB-a*c;
  if(discriminant<0.0){return rigidMiss();}let root=sqrt(max(discriminant,0.0));var best=rigidMiss();
  for(var rootIndex=0u;rootIndex<2u;rootIndex+=1u){let signedRoot=select(-root,root,rootIndex!=0u);let t=(-halfB+signedRoot)/a;let y=origin.y+direction.y*t;
    if(t>1e-4&&t<best.t&&abs(y)<=halfHeight){let point=origin+direction*t;best=RigidRasterHit(t,normalize(vec3f(point.x,0,point.z)),RIGID_RASTER_FEATURE_CYLINDER_SIDE,1u);}}
  return best;
}
fn rigidCylinderHit(origin:vec3f,direction:vec3f,radius:f32,halfHeight:f32)->RigidRasterHit{
  var best=rigidCylinderSideHit(origin,direction,radius,halfHeight);
  if(abs(direction.y)>1e-9){for(var sideIndex=0u;sideIndex<2u;sideIndex+=1u){let side=select(-1.0,1.0,sideIndex!=0u);let t=(side*halfHeight-origin.y)/direction.y;let point=origin+direction*t;
      if(t>1e-4&&t<best.t&&dot(point.xz,point.xz)<=radius*radius){best=RigidRasterHit(t,vec3f(0,side,0),RIGID_RASTER_FEATURE_CYLINDER_CAP,1u);}}}
  return best;
}
fn rigidCapsuleHit(origin:vec3f,direction:vec3f,radius:f32,halfHeight:f32)->RigidRasterHit{
  var best=rigidCylinderSideHit(origin,direction,radius,halfHeight);best.featureId=RIGID_RASTER_FEATURE_SMOOTH;
  for(var sideIndex=0u;sideIndex<2u;sideIndex+=1u){let side=select(-1.0,1.0,sideIndex!=0u);let center=vec3f(0,side*halfHeight,0);let offset=origin-center;let halfB=dot(offset,direction);let c=dot(offset,offset)-radius*radius;let discriminant=halfB*halfB-c;
    if(discriminant>=0.0){let root=sqrt(max(discriminant,0.0));for(var rootIndex=0u;rootIndex<2u;rootIndex+=1u){let signedRoot=select(-root,root,rootIndex!=0u);let t=-halfB+signedRoot;let point=origin+direction*t;let onOuterHemisphere=select(point.y<=-halfHeight,point.y>=halfHeight,side>0.0);
        if(t>1e-4&&t<best.t&&onOuterHemisphere){best=RigidRasterHit(t,normalize(point-center),RIGID_RASTER_FEATURE_SMOOTH,1u);}}}}
  return best;
}
fn rigidBodyHit(origin:vec3f,direction:vec3f,body:BodyGPU)->RigidRasterHit{
  let localOrigin=rigidQinv(body.orientation,origin-body.positionRadius.xyz);let localDirection=normalize(rigidQinv(body.orientation,direction));
  let shape=i32(round(body.halfSizeShape.w));var hit=rigidMiss();
  if(shape==0){hit=rigidSphereHit(localOrigin,localDirection,max(body.halfSizeShape.x,1e-6));}
  else if(shape==1){hit=rigidBoxHit(localOrigin,localDirection,max(body.halfSizeShape.xyz,vec3f(1e-6)));}
  else if(shape==2){hit=rigidCapsuleHit(localOrigin,localDirection,max(body.halfSizeShape.x,1e-6),max(body.halfSizeShape.y,0.0));}
  else if(shape==3){hit=rigidCylinderHit(localOrigin,localDirection,max(body.halfSizeShape.x,1e-6),max(body.halfSizeShape.y,1e-6));}
  if(hit.valid!=0u){hit.normal=normalize(rigidQrot(body.orientation,hit.normal));}return hit;
}

fn rigidResolvedMaterialId(body:BodyGPU)->u32{
  let shape=i32(round(body.halfSizeShape.w));if(shape==0){return RIGID_RASTER_MATERIAL_SPHERE;}if(shape==1){return RIGID_RASTER_MATERIAL_BOX;}if(shape==2){return RIGID_RASTER_MATERIAL_CAPSULE;}return RIGID_RASTER_MATERIAL_CYLINDER;
}
struct RigidRasterMotion{velocity_m_s:vec3f,generation:u32,valid:u32}
fn rigidRasterMotionSurface(bodyIndex:u32,body:BodyGPU,materialId:u32,worldSurfacePosition_m:vec3f)->RigidRasterMotion{
  let record=rigidMotion[bodyIndex];let generation=svoPrimitiveMotionGeneration(record);
  let identityValid=record.identityRevision.x==bodyIndex&&svoPrimitiveMotionOwnerId(record)==bodyIndex&&svoPrimitiveMotionMaterialId(record)==materialId&&generation!=0u;
  let transformValid=distance(record.currentPositionDt.xyz,body.positionRadius.xyz)<=1e-5;
  let velocity=svoPrimitiveMotionVelocityAt(record,worldSurfacePosition_m);let valid=identityValid&&transformValid&&velocity.valid!=0u;
  return RigidRasterMotion(select(vec3f(0.0),velocity.velocity_m_s,valid),generation,select(0u,1u,valid));
}

struct RigidRasterFragmentOut{
  @location(0) packedSurface:vec4u,
  @location(1) identityMedia:vec4u,
  @location(2) primaryGeometry:vec2u,
  @builtin(frag_depth) hardwareDepth:f32,
}
@fragment fn rigidRasterFragment(input:RigidRasterVertexOut)->RigidRasterFragmentOut{
  let bodyCount=min(u32(round(max(rigidUniforms.options.z,0.0))),min(RIGID_RASTER_MAX_BODIES,arrayLength(&rigidBodies)));
  if(input.bodyIndex>=bodyCount){discard;}
  let ray=rigidRasterRay(input.position.xy);let origin=ray[0];let direction=ray[1];let forward=normalize(rigidUniforms.cameraTarget.xyz-origin);
  let body=rigidBodies[input.bodyIndex];let hit=rigidBodyHit(origin,direction,body);if(hit.valid==0u){discard;}
  let materialId=rigidResolvedMaterialId(body);let worldSurfacePosition_m=origin+direction*hit.t;let motion=rigidRasterMotionSurface(input.bodyIndex,body,materialId,worldSurfacePosition_m);
  let media=select(vec2u(RIGID_RASTER_MEDIUM_OPAQUE,RIGID_RASTER_MEDIUM_AIR),vec2u(RIGID_RASTER_MEDIUM_AIR,RIGID_RASTER_MEDIUM_OPAQUE),dot(direction,hit.normal)<0.0);
  let flags=select(0u,SVO_GBUFFER_MOTION_VALID,motion.valid!=0u)|select(SVO_GBUFFER_HARD_FEATURE,0u,hit.featureId==RIGID_RASTER_FEATURE_SMOOTH)
    |svoGBufferProducerFlags(SVO_GBUFFER_PRODUCER_RIGID);
  let targets=svoGBufferSurface(vec3f(0.0),hit.t,hit.normal,hit.normal,vec4u(materialId,input.bodyIndex,media.x,media.y),motion.velocity_m_s,RIGID_RASTER_MOTION_RIGID,RIGID_RASTER_FIELD_ANALYTIC,motion.generation,flags,hit.featureId);
  let viewDepth=hit.t*max(dot(direction,forward),1e-6);return RigidRasterFragmentOut(targets.packedSurface,targets.identityMedia,
    vec2u(bitcast<u32>(hit.t),rigidPackPrimaryGeometryMetadata(hit.normal,input.bodyIndex,hit.featureId,motion.valid)),clamp(RIGID_RASTER_NEAR_M/viewDepth,0.0,1.0));
}

struct RigidSplitBridgeFragmentOut{
  @location(0) geometry:vec4f,
  @location(1) identity:vec4u,
}
@fragment fn rigidSplitBridgeFragment(input:RigidRasterVertexOut)->RigidSplitBridgeFragmentOut{
  let dimensions=textureDimensions(rigidPrimaryGeometryRead);let coordinate=vec2i(input.position.xy);
  if(any(coordinate<vec2i(0))||any(coordinate>=vec2i(dimensions))){discard;}
  let geometry=textureLoad(rigidPrimaryGeometryRead,coordinate,0);if(geometry.x==0u){discard;}let owner=(geometry.y>>24u)&15u;if(owner!=input.bodyIndex){discard;}
  let feature=(geometry.y>>28u)&7u;let motionValid=geometry.y>>31u;let metadata=owner|(feature<<16u)|(RIGID_RASTER_FIELD_ANALYTIC<<20u)|(RIGID_RASTER_MOTION_RIGID<<24u)|(motionValid<<26u);
  return RigidSplitBridgeFragmentOut(
    vec4f(rigidUnpackPrimaryNormal(geometry.y),bitcast<f32>(geometry.x)),
    vec4u(0x80000000u|owner,metadata,0u,0u));
}

// Retained as a diagnostic fallback. Production should use the coverage
// bridge above; this entry necessarily scans the full viewport.
@compute @workgroup_size(${SVO_RIGID_RASTER_CONTRACT.splitBridgeWorkgroupSize},${SVO_RIGID_RASTER_CONTRACT.splitBridgeWorkgroupSize}) fn rigidSplitBridge(@builtin(global_invocation_id) globalId:vec3u){
  let dimensions=textureDimensions(rigidPackedSurfaceRead);if(any(globalId.xy>=dimensions)){return;}let coordinate=vec2i(globalId.xy);
  let packedSurface=textureLoad(rigidPackedSurfaceRead,coordinate,0);let motionKind=packedSurface.z>>30u;if(motionKind!=RIGID_RASTER_MOTION_RIGID){return;}
  let identityMedia=textureLoad(rigidIdentityMediaRead,coordinate,0);let bodyIndex=identityMedia.y;
  let bodyCount=min(u32(round(max(rigidUniforms.options.z,0.0))),min(RIGID_RASTER_MAX_BODIES,arrayLength(&rigidBodies)));if(bodyIndex>=bodyCount){return;}
  let body=rigidBodies[bodyIndex];if(identityMedia.x!=rigidResolvedMaterialId(body)){return;}
  let ray=rigidRasterRay(vec2f(globalId.xy)+vec2f(.5));let hit=rigidBodyHit(ray[0],ray[1],body);if(hit.valid==0u){return;}
  let metadata=(bodyIndex&0xffffu)|((hit.featureId&15u)<<16u)|(RIGID_RASTER_FIELD_ANALYTIC<<20u)|(RIGID_RASTER_MOTION_RIGID<<24u);
  textureStore(rigidSplitGeometryWrite,coordinate,vec4f(hit.normal,hit.t));textureStore(rigidSplitIdentityWrite,coordinate,vec4u(0x80000000u|bodyIndex,metadata,0u,0u));
}
`;
