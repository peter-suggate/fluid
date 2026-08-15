import { cameraApertureShaderLibrary } from "./webgpu-camera";
import { environmentShaderLibrary } from "./webgpu-environments";
import { unifiedLightingShaderLibrary, waterSceneOpticsShaderLibrary } from "./webgpu-lighting";

/**
 * Analytic intersection of the rigid roster, shared by every raster-path pass
 * that has to answer "what body is along this ray?".
 *
 * The including shader must expose the standard `u: Uniforms` binding — only
 * `options.z`, the body count, is read — and a `bodies: array<BodyGPU,12>`
 * binding whose element layout is (positionRadius, halfSizeShape, orientation,
 * colorSelected). Both are the conventions `environmentShaderLibrary` already
 * establishes, and both are already true of every shader that binds the
 * renderer's rigid buffer.
 *
 * `boxHit` lives here rather than beside its callers because `bodyRigidHit` is
 * its first consumer and WGSL requires declaration before use; a shader that
 * wants the slab test for its own tank bounds gets it from the same include.
 */
export const rigidBodyRaymarchShaderLibrary = /* wgsl */ `
fn boxHit(ro:vec3f,rd:vec3f,mn:vec3f,mx:vec3f)->vec2f{let inv=1.0/rd;let a=(mn-ro)*inv;let b=(mx-ro)*inv;let near3=min(a,b);let far3=max(a,b);return vec2f(max(max(near3.x,near3.y),near3.z),min(min(far3.x,far3.y),far3.z));}
fn qrot(q:vec4f,v:vec3f)->vec3f{let a=cross(q.yzw,v);return v+2.0*(q.x*a+cross(q.yzw,a));}
fn qinv(q:vec4f,v:vec3f)->vec3f{return qrot(vec4f(q.x,-q.yzw),v);}
struct RigidHit { t:f32, n:vec3f }
/** A hit that also names the body, for a caller that has to shade it. */
struct RigidPick { t:f32, n:vec3f, index:u32 }
fn sphereRigidHit(ro:vec3f,rd:vec3f,center:vec3f,radius:f32)->RigidHit{
  let oc=ro-center;let b=dot(oc,rd);let discriminant=b*b-dot(oc,oc)+radius*radius;
  if(discriminant<0.0){return RigidHit(1e20,vec3f(0,1,0));}
  let root=sqrt(discriminant);var t=-b-root;if(t<=1e-4){t=-b+root;}
  if(t<=1e-4){return RigidHit(1e20,vec3f(0,1,0));}
  return RigidHit(t,normalize(ro+rd*t-center));
}
fn cylinderRigidHit(ro:vec3f,rd:vec3f,radius:f32,halfHeight:f32,capped:bool)->RigidHit{
  var best=RigidHit(1e20,vec3f(0,1,0));let a=dot(rd.xz,rd.xz);
  if(a>1e-8){let b=dot(ro.xz,rd.xz);let c=dot(ro.xz,ro.xz)-radius*radius;let discriminant=b*b-a*c;
    if(discriminant>=0.0){let root=sqrt(discriminant);var t=(-b-root)/a;if(t<=1e-4){t=(-b+root)/a;}let y=ro.y+rd.y*t;
      if(t>1e-4&&abs(y)<=halfHeight){let p=ro+rd*t;best=RigidHit(t,normalize(vec3f(p.x,0,p.z)));}}}
  if(capped&&abs(rd.y)>1e-8){for(var side=-1.0;side<=1.0;side+=2.0){let t=(side*halfHeight-ro.y)/rd.y;let p=ro+rd*t;
    if(t>1e-4&&t<best.t&&dot(p.xz,p.xz)<=radius*radius){best=RigidHit(t,vec3f(0,side,0));}}}
  return best;
}
fn bodyRigidHit(ro:vec3f,rd:vec3f,body:BodyGPU)->RigidHit{
  let o=qinv(body.orientation,ro-body.positionRadius.xyz);let d=qinv(body.orientation,rd);let shape=i32(round(body.halfSizeShape.w));var hit=RigidHit(1e20,vec3f(0,1,0));
  if(shape==0){hit=sphereRigidHit(o,d,vec3f(0),body.halfSizeShape.x);}
  else if(shape==1){let interval=boxHit(o,d,-body.halfSizeShape.xyz,body.halfSizeShape.xyz);var t=interval.x;if(t<=1e-4){t=interval.y;}
    if(t>1e-4&&interval.x<=interval.y){let p=o+d*t;let q=abs(p/max(body.halfSizeShape.xyz,vec3f(1e-5)));var n=vec3f(0,0,sign(p.z));
      if(q.x>=q.y&&q.x>=q.z){n=vec3f(sign(p.x),0,0);}else if(q.y>=q.z){n=vec3f(0,sign(p.y),0);}hit=RigidHit(t,n);}}
  else if(shape==2){hit=cylinderRigidHit(o,d,body.halfSizeShape.x,body.halfSizeShape.y,false);let upper=sphereRigidHit(o,d,vec3f(0,body.halfSizeShape.y,0),body.halfSizeShape.x);let lower=sphereRigidHit(o,d,vec3f(0,-body.halfSizeShape.y,0),body.halfSizeShape.x);if(upper.t<hit.t){hit=upper;}if(lower.t<hit.t){hit=lower;}}
  else{hit=cylinderRigidHit(o,d,body.halfSizeShape.x,body.halfSizeShape.y,true);}
  return RigidHit(hit.t,normalize(qrot(body.orientation,hit.n)));
}
fn pickNearestRigid(ro:vec3f,rd:vec3f)->RigidPick{var best=RigidPick(1e20,vec3f(0,1,0),0u);for(var i=0u;i<12u;i+=1u){if(i>=u32(round(u.options.z))){break;}let hit=bodyRigidHit(ro,rd,bodies[i]);if(hit.t<best.t){best=RigidPick(hit.t,hit.n,i);}}return best;}
fn nearestRigid(ro:vec3f,rd:vec3f)->RigidHit{let pick=pickNearestRigid(ro,rd);return RigidHit(pick.t,pick.n);}
`;

/**
 * Beyond this the pick missed every body. Mirrors the marcher's own sentinel.
 *
 * Interpolated through `toExponential`, never through the default number
 * formatting: `String(1e19)` is "10000000000000000000", which WGSL reads as an
 * abstract-int literal too large to represent and rejects at compile time.
 */
export const RIGID_SCENE_MISS_DISTANCE_M = 1e19;

/**
 * How a body's surface answers light on the fluid-only path.
 *
 * A dielectric with a slightly soft finish, matching what the SVO path's
 * published per-shape records land on for the same palette. Nothing here is
 * scene-authored: a fluid-only scene has no material publication to read, and
 * inventing one would put a second material authority beside the SVO's.
 */
export const RIGID_SCENE_SURFACE = Object.freeze({
  roughness: 0.42,
  dielectricF0: 0.04,
  /** The SVO path's selection tint, so a selected body reads the same in both. */
  selectionEmission: [0.12, 0.42, 0.32] as const,
});

/**
 * The rigid roster drawn into the dry-scene attachment, for scenes that run no
 * dry scene at all.
 *
 * A `fluid-only` presentation has no SVO world, so `webgpu-svo-rigid-raster` —
 * the only other thing that draws a body — never runs, and the attachment the
 * optical composite reads is a flat clear. The composite consults the roster
 * for contact and for the interior exit, but it never *shades* one: a body
 * outside the water was therefore invisible, which made dropping one into a
 * tank a thing you could only see once it landed.
 *
 * This is the whole of the fix and deliberately not more. It writes the same
 * two things the SVO dry scene writes — linear radiance in RGB, ray distance in
 * alpha — so `resolvedDrySceneDepth` sorts bodies against the water with no
 * change to the composite, and refraction picks them up through the ordinary
 * `sceneTexture` read. Twelve analytic primitives and one fullscreen triangle:
 * a fluid-only scene stays a fluid-only scene.
 */
export const fluidOnlyRigidSceneShader = /* wgsl */ `
struct Uniforms { viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f, options:vec4f, gridInfo:vec4f, debug:vec4f, environment:vec4f, terrainMeta:vec4f, terrainFeatures:array<vec4f,16> }
struct BodyGPU { positionRadius:vec4f, halfSizeShape:vec4f, orientation:vec4f, colorSelected:vec4f }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var<storage,read> bodies:array<BodyGPU,12>;
${waterSceneOpticsShaderLibrary(0, 2, 3)}
${cameraApertureShaderLibrary("u")}
${rigidBodyRaymarchShaderLibrary}
${environmentShaderLibrary}
${unifiedLightingShaderLibrary}
// The same key the water shades with, resolved the same way: an authored
// document light wins, otherwise the environment preset's sun. Without this the
// highlight on a floating body would disagree with the highlight on the water
// it is floating in.
fn rigidKeyDirection()->vec3f{return select(environmentLightDirection(),waterAuthoredKeyDirection(),waterKeyAuthored());}
fn rigidKeyColor()->vec3f{return select(environmentLightColor(),waterKeyRadiance(),waterKeyAuthored());}
struct VOut{@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn vertexMain(@builtin(vertex_index)i:u32)->VOut{var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));var o:VOut;o.position=vec4f(p[i],0,1);o.uv=p[i]*.5+.5;return o;}
@fragment fn fragmentMain(input:VOut)->@location(0) vec4f{
  let ndc=input.uv*2.0-1.0;let ro=u.cameraPosition.xyz;
  let forward=normalize(u.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));
  let aperture=cameraTanHalfFov();
  let rd=normalize(forward+right*ndc.x*u.viewport.x/max(u.viewport.y,1.0)*aperture+up*ndc.y*aperture);
  let pick=pickNearestRigid(ro,rd);
  // A miss keeps the attachment's clear: the background belongs to whoever
  // cleared it, and writing one here would make this pass the backdrop's
  // author as well as the roster's.
  if(pick.t>=${RIGID_SCENE_MISS_DISTANCE_M.toExponential()}){discard;}
  let body=bodies[pick.index];
  let position=ro+rd*pick.t;
  let base=clamp(body.colorSelected.xyz,vec3f(0.0),vec3f(1.0));
  let emissive=body.colorSelected.w*vec3f(${RIGID_SCENE_SURFACE.selectionEmission.join(",")});
  let f0=vec3f(${RIGID_SCENE_SURFACE.dielectricF0});
  let material=unifiedMaterial(base,${RIGID_SCENE_SURFACE.roughness},emissive,0.0,f0,1.0,vec3f(0.0),0.0);
  let viewDirection=normalize(-rd);
  let direct=shadeUnifiedSurface(material,unifiedLightingInput(pick.n,viewDirection,rigidKeyDirection(),rigidKeyColor()));
  // Uniform incoming radiance L leaves a diffuse surface at albedo*L, so the
  // environment term is not divided by pi the way the direct lobe is.
  let environmentBrdf=unifiedEnvironmentBrdf(max(dot(pick.n,viewDirection),0.0),${RIGID_SCENE_SURFACE.roughness},f0);
  let diffuseEnergy=max(vec3f(0.0),vec3f(1.0)-environmentBrdf);
  let ambient=base*diffuseEnergy*environmentLight(pick.n)*environmentContactShadow(position,pick.n);
  let reflected=environmentLight(reflect(rd,pick.n))*environmentBrdf;
  return vec4f(max(direct+ambient+reflected,vec3f(0.0)),pick.t);
}
`;
