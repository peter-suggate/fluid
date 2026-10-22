import { RASTER_MAX_SPOT_LIGHTS, rasterSpotLights } from "./raster-spot-lights";
import { importedSurfaceShadowWGSL, importedSurfaceShadowGateWGSL } from "./imported-surface-shadow";
import { SVO_SURFACE_MESH_MODE, SVO_SURFACE_MESH_STATE as M } from "../primary-visibility/svo-surface-mesh";

/** Preview backend. Fixed world-space coverage is deliberately independent of the camera. */
export const SVO_RASTER_SHADOW_SIZE = 2048;
/**
 * Quads one partial redraw of the shadow maps can list: the casters over the
 * rectangle an edit's bricks land in. A redraw that would list more draws the
 * whole arena and lets each vertex test its own quad.
 */
export const SVO_RASTER_SHADOW_CASTER_CAPACITY = 1 << 22;
/** How the shadow passes of a frame redraw their maps; the cache check decides it on the GPU. */
const REDRAW = Object.freeze({ nothing: 0, whole: 1, listed: 2, tested: 3 } as const);
/** Words of the cache buffer: two indirect draws, the revisions last drawn, then the caster list's count and overflow flag. */
const CACHE = Object.freeze({ clearDraw: 0, casterDraw: 4, revisions: 8, listed: 16, overflowed: 17, bytes: 128 } as const);
export interface RasterAoPipelines {
  shadow: GPURenderPipeline;
  nativeShadow?: GPURenderPipeline;
  terrain: GPURenderPipeline;
  casters: GPUComputePipeline;
  ao: GPUComputePipeline;
}

const paramsWGSL = /* wgsl */ `
struct RasterSpot { positionNear:vec4f, directionTan:vec4f, rangeRadiusIndex:vec4f }
struct RasterLightingParams { control:vec4u, farCenterExtent:vec4f, depthCell:vec4f, sun:vec4f, spots:array<RasterSpot,${RASTER_MAX_SPOT_LIGHTS}> }
@group(2) @binding(0) var<uniform> rasterLighting:RasterLightingParams;
`;

/** Depth across a planar receiver is affine in shadow-map UV, including a
 * perspective map's post-divide depth. Both blocker search and filtering must
 * compare against that plane, not the centre pixel's constant depth. */
export const rasterReceiverPlaneWGSL = /* wgsl */ `
// Hardware comparison filtering uses one reference depth for four texels.
// Cover that footprint's slope as well as correcting the offset between taps.
fn rasterReceiverFootprintBias(gradient:vec2f,size:f32)->f32{
  return dot(abs(gradient),vec2f(1.0/size));
}
fn rasterSunReceiverGradient(normal:vec3f,extent:f32,depthSpan:f32)->vec2f{
  if(abs(normal.z)<1e-6){return vec2f(0);}
  return vec2f(normal.x,-normal.y)*(2.0*extent/(normal.z*depthSpan));
}
fn rasterSpotReceiverGradient(normal:vec3f,planeDistance:f32,near:f32,far:f32,tanHalfAngle:f32)->vec2f{
  if(abs(planeDistance)<1e-6){return vec2f(0);}
  return vec2f(normal.x,-normal.y)*(-2.0*tanHalfAngle*near*far/((far-near)*planeDistance));
}
`;

/** Shared by the caster and receiver: conventional [0,1] shadow depth, reversed camera depth unchanged. */
export const rasterAoConsumerWGSL = /* wgsl */ `
${paramsWGSL}
${rasterReceiverPlaneWGSL}
@group(2) @binding(1) var rasterSunDepth:texture_depth_2d_array;
@group(2) @binding(2) var rasterSunSampler:sampler_comparison;
@group(2) @binding(3) var rasterContact:texture_2d<f32>;
@group(2) @binding(5) var rasterShadowVisibility:texture_2d<f32>;
var<private> rasterResolvedVisibility:vec2f;
var<private> rasterPixel:vec2f;
fn rasterSunDirection()->vec3f{return normalize(rasterLighting.sun.xyz);}
fn rasterSunBasis()->mat3x3f{
  let sun=rasterSunDirection();let helper=select(vec3f(0,1,0),vec3f(1,0,0),abs(sun.y)>.95);
  let right=normalize(cross(helper,sun));return mat3x3f(right,cross(sun,right),sun);
}
fn rasterMapCenter(layer:u32)->vec3f{return select(uniforms.container.xyz*.5,rasterLighting.farCenterExtent.xyz,layer==1u);}
fn rasterMapExtent(layer:u32)->f32{return select(max(length(uniforms.container.xyz)*.7,.1),rasterLighting.farCenterExtent.w,layer==1u);}
fn rasterShadowPosition(world:vec3f,layer:u32)->vec3f{
  let basis=rasterSunBasis();let delta=world-rasterMapCenter(layer);let extent=rasterMapExtent(layer);
  return vec3f(dot(delta,basis[0])/extent,dot(delta,basis[1])/extent,.5-dot(delta,basis[2])/rasterLighting.depthCell.x);
}
fn rasterSpotBasis(spot:RasterSpot)->mat3x3f{
  let forward=spot.directionTan.xyz;let helper=select(vec3f(0,1,0),vec3f(1,0,0),abs(forward.y)>.95);
  let right=normalize(cross(forward,helper));return mat3x3f(right,cross(right,forward),forward);
}
fn rasterSpotClip(world:vec3f,spot:RasterSpot)->vec4f{
  let basis=rasterSpotBasis(spot);let delta=world-spot.positionNear.xyz;
  let z=dot(delta,basis[2]);let near=spot.positionNear.w;let far=spot.rangeRadiusIndex.x;
  return vec4f(dot(delta,basis[0])/spot.directionTan.w,dot(delta,basis[1])/spot.directionTan.w,far*(z-near)/(far-near),z);
}
fn rasterSpotDistance(depth:f32,spot:RasterSpot)->f32{
  let near=spot.positionNear.w;let far=spot.rangeRadiusIndex.x;
  return near*far/max(far-depth*(far-near),1e-6);
}
fn rasterSpotSlot(lightIndex:u32)->i32{
  for(var slot=0u;slot<rasterLighting.control.z;slot++){
    if(u32(rasterLighting.spots[slot].rangeRadiusIndex.z)==lightIndex){return i32(slot);}
  }return -1;
}
fn rasterSpotVisibility(position:vec3f,normal:vec3f,slot:u32)->f32{
  let spot=rasterLighting.spots[slot];let basis=rasterSpotBasis(spot);
  let distance=dot(position-spot.positionNear.xyz,basis[2]);
  let texel=2.0*max(distance,spot.positionNear.w)*spot.directionTan.w/f32(${SVO_RASTER_SHADOW_SIZE});
  let biased=position+normal*(texel*1.5+rasterLighting.depthCell.y*(.18+dry.tuningRays0.x));
  let clip=rasterSpotClip(biased,spot);if(clip.w<=spot.positionNear.w){return 1.0;}
  let p=clip.xyz/clip.w;let uv=p.xy*vec2f(.5,-.5)+.5;
  if(any(uv<vec2f(0))||any(uv>vec2f(1))||p.z>=1.0){return 1.0;}
  let offsets=array<vec2f,12>(vec2f(0,0),vec2f(-.32,.23),vec2f(.25,-.39),vec2f(-.43,-.32),vec2f(.60,.15),vec2f(-.09,.67),vec2f(-.69,.12),vec2f(.09,-.76),vec2f(.55,.63),vec2f(-.57,.62),vec2f(-.64,-.64),vec2f(.79,-.48));
  let lightNormal=transpose(basis)*normal;
  let gradient=rasterSpotReceiverGradient(lightNormal,dot(normal,biased-spot.positionNear.xyz),
    spot.positionNear.w,spot.rangeRadiusIndex.x,spot.directionTan.w);
  let receiver=clip.w-texel*.3;
  let depth=spot.rangeRadiusIndex.x*(receiver-spot.positionNear.w)/((spot.rangeRadiusIndex.x-spot.positionNear.w)*receiver)
    -rasterReceiverFootprintBias(gradient,f32(${SVO_RASTER_SHADOW_SIZE}));
  let radius=spot.rangeRadiusIndex.y;
  // Compare against the receiver plane, but measure the blocker at its actual
  // depth. A slope-corrected depth is not a physical occluder distance.
  // Perspective depth is nonlinear. Recover axial blocker distance before
  // estimating the finite emitter's contact-hardening penumbra in world units.
  let search=clamp(radius/max(texel,1e-6)*(clip.w-spot.positionNear.w)/spot.positionNear.w,2.0,96.0);
  var blocker=0.0;var count=0.0;
  for(var i=0u;i<12u;i++){
    let c=clamp(vec2i(uv*f32(${SVO_RASTER_SHADOW_SIZE})+offsets[i]*search),vec2i(0),vec2i(${SVO_RASTER_SHADOW_SIZE - 1}));
    let offset=(vec2f(c)+.5)/f32(${SVO_RASTER_SHADOW_SIZE})-uv;
    let raw=textureLoad(rasterSunDepth,c,i32(slot+2u),0);if(raw>=1.0){continue;}
    let d=raw-dot(gradient,offset);
    if(d<depth){blocker+=rasterSpotDistance(raw,spot);count+=1.0;}
  }
  if(count==0.0){return 1.0;}
  let blockerDistance=blocker/count;
  let penumbra=clamp(radius*(clip.w-blockerDistance)/max(blockerDistance*texel,1e-6),1.0,96.0);
  var visibility=0.0;
  for(var i=0u;i<12u;i++){
    let offset=offsets[i]*penumbra/f32(${SVO_RASTER_SHADOW_SIZE});
    visibility+=textureSampleCompareLevel(rasterSunDepth,rasterSunSampler,uv+offset,i32(slot+2u),depth+dot(gradient,offset));
  }return visibility/12.0;
}
fn rasterPcf(position:vec3f,normal:vec3f,layer:u32)->f32{
  let texel=2.0*rasterMapExtent(layer)/f32(${SVO_RASTER_SHADOW_SIZE});
  // World-space normal offset and a small depth bias, scaled with shadow texels.
  let biased=position+normal*(texel*1.5+rasterLighting.depthCell.y*(.18+dry.tuningRays0.x));
  let p=rasterShadowPosition(biased,layer);let uv=p.xy*vec2f(.5,-.5)+.5;
  if(any(uv<vec2f(0))||any(uv>vec2f(1))||p.z<=0.0||p.z>=1.0){return 1.0;}
  // Blocker search and a bounded PCSS filter retain contact hardening. The
  // softness control is the same angular diameter used by the cone reference.
  let offsets=array<vec2f,12>(vec2f(.13,.09),vec2f(-.32,.23),vec2f(.25,-.39),vec2f(-.43,-.32),vec2f(.60,.15),vec2f(-.09,.67),vec2f(-.69,.12),vec2f(.09,-.76),vec2f(.55,.63),vec2f(-.57,.62),vec2f(-.64,-.64),vec2f(.79,-.48));
  let angular=tan(dry.tuningRays1.y*.5);
  let search=max(2.0,length(uniforms.container.xyz)*angular/texel);
  let gradient=rasterSunReceiverGradient(transpose(rasterSunBasis())*normal,rasterMapExtent(layer),rasterLighting.depthCell.x);
  let bias=texel*.3/rasterLighting.depthCell.x+rasterReceiverFootprintBias(gradient,f32(${SVO_RASTER_SHADOW_SIZE}));var blocker=0.0;var count=0.0;
  for(var i=0u;i<12u;i++){
    let c=clamp(vec2i(uv*f32(${SVO_RASTER_SHADOW_SIZE})+offsets[i]*search),vec2i(0),vec2i(${SVO_RASTER_SHADOW_SIZE - 1}));
    let offset=(vec2f(c)+.5)/f32(${SVO_RASTER_SHADOW_SIZE})-uv;
    let raw=textureLoad(rasterSunDepth,c,i32(layer),0);if(raw>=1.0){continue;}
    let d=raw-dot(gradient,offset);
    if(d<p.z-bias){blocker+=raw;count+=1.0;}
  }
  if(count==0.0){return 1.0;}
  let penumbra=clamp((p.z-blocker/count)*rasterLighting.depthCell.x*angular/texel,1.0,48.0);
  let disk=array<vec2f,24>(vec2f(0.144338,0.000000),vec2f(-0.184342,0.168873),vec2f(0.028217,-0.321513),vec2f(0.232351,0.303061),vec2f(-0.426393,-0.075423),vec2f(0.403917,-0.256939),vec2f(-0.135102,0.502574),vec2f(-0.257655,-0.496099),vec2f(0.559008,0.204149),vec2f(-0.581555,0.240057),vec2f(0.280348,-0.599087),vec2f(0.207170,0.660490),vec2f(-0.624412,-0.361860),vec2f(0.732507,-0.161040),vec2f(-0.447038,0.635865),vec2f(-0.103276,-0.796974),vec2f(0.634013,0.534347),vec2f(-0.853183,0.035282),vec2f(0.622332,-0.619303),vec2f(-0.041636,0.900426),vec2f(-0.592151,-0.709594),vec2f(0.938032,0.126211),vec2f(-0.794793,0.552996),vec2f(0.217183,-0.965401));
  var visibility=0.0;
  for(var i=0u;i<24u;i++){
    let offset=disk[i]*penumbra/f32(${SVO_RASTER_SHADOW_SIZE});
    visibility+=textureSampleCompareLevel(rasterSunDepth,rasterSunSampler,uv+offset,i32(layer),p.z+dot(gradient,offset)-bias);
  }return visibility/24.0;
}
fn rasterSunVisibility(position:vec3f,normal:vec3f)->f32{
  let near=rasterShadowPosition(position,0u);let edge=max(abs(near.x),abs(near.y));
  if(edge<.85){return rasterPcf(position,normal,0u);}
  let far=rasterPcf(position,normal,1u);if(edge>=.98){return far;}
  return mix(rasterPcf(position,normal,0u),far,smoothstep(.85,.98,edge));
}
fn rasterOctEncode(n:vec3f)->vec2f{
  let p=n/max(abs(n.x)+abs(n.y)+abs(n.z),1e-6);
  return select((1.0-abs(p.yx))*select(vec2f(-1),vec2f(1),p.xy>=vec2f(0)),p.xy,p.z>=0.0);
}
fn rasterOctDecode(p:vec2f)->vec3f{
  var n=vec3f(p,1.0-abs(p.x)-abs(p.y));let t=clamp(-n.z,0.0,1.0);
  n.x+=select(t,-t,n.x>=0.0);n.y+=select(t,-t,n.y>=0.0);return normalize(n);
}
fn rasterVisibilityAt(position:vec3f,normal:vec3f)->vec2f{
  let depth=length(position-uniforms.cameraPosition.xyz);let q=rasterPixel*.5-.75;let base=vec2i(floor(q));let f=fract(q);
  var total=vec2f(0.0);var weight=0.0;let dims=vec2i(textureDimensions(rasterContact));
  for(var y=0;y<2;y++){for(var x=0;x<2;x++){
    let c=clamp(base+vec2i(x,y),vec2i(0),dims-1);let v=textureLoad(rasterContact,c,0);
    let bilinear=select(1.0-f.x,f.x,x==1)*select(1.0-f.y,f.y,y==1);
    let tolerance=max(depth*.005,rasterLighting.depthCell.y*.5);
    let w=bilinear*exp(-abs(v.y-depth)/max(tolerance,1e-4))*pow(max(dot(normal,rasterOctDecode(v.zw)),0.0),8.0);
    total+=vec2f(v.x,textureLoad(rasterShadowVisibility,c,0).x)*w;weight+=w;
  }}
  if(weight>1e-4){return total/weight;}
  // A subpixel voxel face may be absent from all four reduced samples. Fully
  // lit is not a valid reconstruction there: it punches white pinholes into
  // both shadows and AO. Evaluate only those receivers at their actual pixel.
  var contact=1.0;var shadow=1.0;
  if((dry.materialPublication.w&1u)!=0u){contact=min(rasterHorizonVisibility(rasterPixel,position,normal),rasterCoarseVisibility(position,normal));}
  if((dry.materialPublication.w&2u)!=0u){shadow=rasterSunVisibility(position,normal);}
  return vec2f(contact,shadow);
}
fn rasterContactVisibility()->vec3f{
  if((dry.materialPublication.w&1u)==0u){return vec3f(1);}
  // Contrast the existing horizon/coarse result, without adding visibility
  // work. Open sky stays at one, and the AO strength switch still wins.
  let visibility=pow(clamp(rasterResolvedVisibility.x,0.0,1.0),max(1.0,rasterLighting.depthCell.z));
  return vec3f(mix(1.0,visibility,dry.tuningRays0.w));
}
`;

/** Local estimator shared with the GPU reconstruction regression fixture. */
export const rasterHorizonWGSL = /* wgsl */ `
fn rasterWorldAt(pixel:vec2f,depth:f32)->vec3f{
  let camera=dryRasterPrimaryCamera();let ndc=pixel/uniforms.viewport.xy*vec2f(2,-2)+vec2f(-1,1);
  return camera[0]+normalize(camera[1]+camera[2]*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+camera[3]*ndc.y*cameraTanHalfFov())*depth;
}
// Stable local contact AO, shared by the reduced pass and uncovered receivers.
// Integrate the maximum obscured elevation per azimuth. Squaring each horizon
// erased moderate contact angles; apply contrast to visibility after averaging.
fn rasterHorizonVisibility(pixel:vec2f,position:vec3f,normal:vec3f)->f32{
  let fullDims=vec2i(textureDimensions(drySplitGeometryRead));
  let radius=dryContactVisibilityRadius();
  let viewDepth=dot(position-uniforms.cameraPosition.xyz,normalize(uniforms.cameraTarget.xyz-uniforms.cameraPosition.xyz));
  let projected=clamp(radius*uniforms.viewport.y/(2.0*max(viewDepth,.001)*cameraTanHalfFov()),2.0,160.0);
  var occlusion=0.0;
  // Six azimuths, six quadratic distance samples. A small tangent bias prevents
  // the receiver plane and fitted-normal noise from darkening themselves.
  for(var direction=0u;direction<6u;direction++){
    let angle=(f32(direction)+.5)*6.28318530718/6.0;let axis=vec2f(cos(angle),sin(angle));var horizon=0.0;
    for(var step=1u;step<=6u;step++){
      let fraction=f32(step)/6.0;let tap=vec2i(round(pixel-.5+axis*max(1.0,projected*fraction*fraction)));
      if(any(tap<vec2i(0))||any(tap>=fullDims)){continue;}
      let sample=drySplitGeometryAt(tap);if(!(sample.w<DRY_MISS)){continue;}
      let delta=rasterWorldAt(vec2f(tap)+.5,sample.w)-position;let distance=length(delta);
      if(distance<1e-5||distance>=radius){continue;}
      let cosine=max(dot(normal,delta/distance)-.08,0.0);
      horizon=max(horizon,cosine*(1.0-distance*distance/(radius*radius)));
    }occlusion+=horizon;
  }
  return pow(clamp(1.0-occlusion/6.0,0.0,1.0),2.0);
}
`;

/** Bounded world-opacity lookups, with no screen dependency or temporal history. */
export const rasterCoarseAoWGSL = /* wgsl */ `
fn rasterCoarseVisibility(position:vec3f,normal:vec3f)->f32{
  let strength=rasterLighting.sun.w;
  if(strength<=0.0||!dryNodeMipReady()){return 1.0;}
  let cell=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let minimumVoxel=cell*exp2(f32(dryNodeMipOpacityLevelFloor()));
  let radius=max(minimumVoxel*4.0,dryContactVisibilityRadius());
  let helper=select(vec3f(0,1,0),vec3f(1,0,0),abs(normal.y)>.9);
  let tangent=normalize(cross(helper,normal));let bitangent=cross(normal,tangent);
  let origin=position+normal*cell*.5;
  var visibility=0.0;
  for(var direction=0u;direction<3u;direction++){
    let angle=(f32(direction)+.5)*2.09439510239;
    let ray=normalize(normal+(tangent*cos(angle)+bitangent*sin(angle))*1.25);
    var transmission=1.0;
    var nearbyVisibility=1.0;
    var cache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);
    for(var shell=0u;shell<2u;shell++){
      // Use six of the former eight lookups: one nearby contact sample and one
      // broad canopy sample per azimuth, rather than adding another AO pass.
      let distance=radius*select(2.0,2.0/3.0,shell==0u);
      // A trilinear opacity lookup includes whole mip voxels, extending up to
      // 1.5 voxel widths from the query point on each axis. Keep that support
      // in front of the receiver plane, including on diagonal walls.
      let clearance=dot(normal,ray)*distance+cell*.5;
      let diameter=min(distance*.8,clearance*.9/(1.5*dot(abs(normal),vec3f(1))));
      let lod=max(0.0,floor(log2(diameter/cell)));
      if(minimumVoxel>diameter){continue;}
      let sample=dryNodeMipAt(origin+ray*distance,lod,&cache);
      if(sample.valid==0u){return 1.0;}
      // Mean occupancy retains the gaps between leaves. Max occupancy would
      // turn a sparse canopy into an opaque shell.
      let density=clamp(sample.sample.solidMean,0.0,1.0);
      if(shell==0u){nearbyVisibility=1.0-density*(2.0/3.0);}
      else{
        // Distant dense surfaces cannot supply a reliable horizon from one
        // volume tap. Keep the broad term for sparse canopy only.
        let sparseWeight=1.0-smoothstep(.15,.5,density);
        transmission=mix(1.0,exp(-density*distance/(cell*exp2(lod))*2.0),sparseWeight);
      }
    }
    visibility+=min(nearbyVisibility,transmission);
  }
  return mix(1.0,visibility/3.0,strength);
}
`;

export function rasterAoProducerWGSL(backdrop: boolean): string {
  return /* wgsl */ `
@group(2) @binding(4) var rasterContactOutput:texture_storage_2d<rgba16float,write>;
@group(2) @binding(6) var rasterShadowOutput:texture_storage_2d<r32float,write>;
// What this frame's shadow passes redraw, written by the cache check: the
// whole maps, or the box of finest cells the mesh re-extracted bricks in. The
// box's casters are drawn from a list, or tested a vertex at a time when
// there are too many to list.
struct RasterShadowRedraw { minimum:vec4u, maximum:vec4u }
@group(2) @binding(7) var<uniform> rasterShadowRedraw:RasterShadowRedraw;
@group(2) @binding(8) var<storage,read_write> rasterShadowCasterOutput:array<u32>;
@group(2) @binding(9) var<storage,read_write> rasterShadowCache:array<atomic<u32>>;
@group(2) @binding(10) var<storage,read> rasterShadowCasters:array<u32>;
fn rasterShadowRedrawWhole()->bool{return rasterShadowRedraw.minimum.w==${REDRAW.whole}u;}
// A rectangle of a map that holds a box: xy its centre, zw its half size. A
// sun map is an affine view, so the box's centre and half extent project to
// it. A spot map is a perspective one, bounded here through the box's sphere.
fn rasterShadowRectangle(centre:vec3f,half:vec3f,layer:u32)->vec4f{
  if(layer<2u){
    let basis=rasterSunBasis();let extent=rasterMapExtent(layer);
    return vec4f(rasterShadowPosition(centre,layer).xy,dot(abs(basis[0]),half)/extent,dot(abs(basis[1]),half)/extent);
  }
  let spot=rasterLighting.spots[layer-2u];let basis=rasterSpotBasis(spot);let delta=centre-spot.positionNear.xyz;
  let radius=length(half);let depth=dot(delta,basis[2]);
  // A sphere that reaches the light's near plane may land anywhere in the map.
  if(depth-radius<=spot.positionNear.w){return vec4f(0,0,1,1);}
  let across=vec2f(dot(delta,basis[0]),dot(delta,basis[1]));
  let near=(depth-radius)*spot.directionTan.w;let far=(depth+radius)*spot.directionTan.w;
  let low=min((across-radius)/near,(across-radius)/far);let high=max((across+radius)/near,(across+radius)/far);
  return vec4f((low+high)*.5,(high-low)*.5);
}
// The redrawn box in world space, a margin past the bricks: a smooth surface
// moves vertices of the bricks around the ones it re-extracts.
fn rasterShadowRedrawCentre()->vec3f{
  return dry.mapping.worldOrigin+(vec3f(rasterShadowRedraw.minimum.xyz)+vec3f(rasterShadowRedraw.maximum.xyz))*.5*dry.mapping.cellSize;
}
fn rasterShadowRedrawHalf()->vec3f{
  return (vec3f(rasterShadowRedraw.maximum.xyz-rasterShadowRedraw.minimum.xyz)*.5+vec3f(2.0))*dry.mapping.cellSize;
}
// Whether a quad casts into the redrawn rectangle of a map.
fn rasterShadowCasterRedrawn(quad:SurfaceQuad,layer:u32)->bool{
  // A triangle's points are offsets from its origin: up to eight extents for
  // a dual one, and half an extent past its cell for an inflated contour.
  var span=vec3f(meshQuadExtent(quad));var middle=vec3f(meshQuadOrigin(quad))+span*.5;
  if(meshDualTriangle(quad)){middle=vec3f(meshQuadOrigin(quad))+span*4.0;span*=8.0;}else if(meshTriangle(quad)){span*=2.0;}
  let caster=rasterShadowRectangle(dry.mapping.worldOrigin+middle*dry.mapping.cellSize,span*.5*dry.mapping.cellSize,layer);
  let redraw=rasterShadowRectangle(rasterShadowRedrawCentre(),rasterShadowRedrawHalf(),layer);
  return all(abs(caster.xy-redraw.xy)<=caster.zw+redraw.zw);
}
// List the casters of a partial redraw: the quads over the redrawn rectangle
// of any map. A fixed direct dispatch over the arena's capacity, so the
// passes that follow run a vertex shader for those quads and no others.
var<workgroup> rasterShadowCasterCount:atomic<u32>;
var<workgroup> rasterShadowCasterBase:u32;
@compute @workgroup_size(64)
fn rasterShadowCastersMain(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32){
  if(lane==0u){atomicStore(&rasterShadowCasterCount,0u);}
  workgroupBarrier();
  let index=id.x+id.y*65535u*64u;
  var listed=false;
  if(rasterShadowRedraw.minimum.w==${REDRAW.listed}u&&index<meshHeader[${M.frontCursor}]){
    let quad=meshDrawnQuad(index);
    if(!meshQuadDead(quad)&&meshQuadLevel(quad.face)==0u){
      for(var layer=0u;layer<2u+rasterLighting.control.z;layer+=1u){listed=listed||rasterShadowCasterRedrawn(quad,layer);}
    }
  }
  var rank=0u;if(listed){rank=atomicAdd(&rasterShadowCasterCount,1u);}
  workgroupBarrier();
  if(lane==0u){
    let count=atomicLoad(&rasterShadowCasterCount);
    rasterShadowCasterBase=0u;
    if(count!=0u){
      rasterShadowCasterBase=atomicAdd(&rasterShadowCache[${CACHE.listed}],count);
      // Past the list the redraw falls back to testing every quad's vertices.
      if(rasterShadowCasterBase+count>arrayLength(&rasterShadowCasterOutput)){
        atomicStore(&rasterShadowCache[${CACHE.overflowed}],1u);rasterShadowCasterBase=0xffffffffu;
      }
    }
  }
  workgroupBarrier();
  if(listed&&rasterShadowCasterBase!=0xffffffffu){rasterShadowCasterOutput[rasterShadowCasterBase+rank]=index;}
}
@vertex fn rasterShadowVertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->@builtin(position) vec4f{
  // A whole redraw deliberately enumerates the complete front arena, never camera-visible indices.
  let mode=rasterShadowRedraw.minimum.w;
  var drawn=instance;if(mode==${REDRAW.listed}u){drawn=rasterShadowCasters[instance];}
  let quad=meshDrawnQuad(drawn);let face=meshQuadFace(quad.face);let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  let corners=array<vec2u,4>(vec2u(1,0),vec2u(1,1),vec2u(0,0),vec2u(0,1));
  let corner=select(corners[vertex],corners[vertex].yx,(face&1u)==0u);var lattice=meshQuadOrigin(quad);lattice[u]+=corner.x*quad.extent[u];lattice[v]+=corner.y*quad.extent[v];
  var world=dry.mapping.worldOrigin+vec3f(lattice)*dry.mapping.cellSize;
  if(meshTriangle(quad)){world=dry.mapping.worldOrigin+(vec3f(meshQuadOrigin(quad))+meshTrianglePoint(quad,min(vertex,2u))*vec3f(meshQuadExtent(quad)))*dry.mapping.cellSize;}
  let layer=rasterLighting.control.y;
  var clip=vec4f(rasterShadowPosition(world,min(layer,1u)),1);
  if(layer>=2u){clip=rasterSpotClip(world,rasterLighting.spots[layer-2u]);}
  // Level zero is the complete geometric surface; other levels duplicate it.
  if(meshQuadDead(quad)||meshQuadLevel(quad.face)!=0u){return vec4f(0,0,0,1);}
  // Only the redrawn rectangle was cleared, so only casters over it are drawn
  // again. One that also reaches outside it writes there what is already there.
  if(mode!=${REDRAW.whole}u&&!rasterShadowCasterRedrawn(quad,layer)){return vec4f(0,0,0,1);}
  return clip;
}
@vertex fn rasterShadowClearVertex(@builtin(vertex_index) vertex:u32)->@builtin(position) vec4f{
  let p=array<vec2f,6>(vec2f(-1,-1),vec2f(1,-1),vec2f(-1,1),vec2f(-1,1),vec2f(1,-1),vec2f(1,1));
  let layer=rasterLighting.control.y;
  if(rasterShadowRedrawWhole()){return vec4f(p[vertex],1,1);}
  let redraw=rasterShadowRectangle(rasterShadowRedrawCentre(),rasterShadowRedrawHalf(),layer);
  return vec4f(redraw.xy+p[vertex]*(redraw.zw+vec2f(4.0/f32(${SVO_RASTER_SHADOW_SIZE}))),1,1);
}
@fragment fn rasterShadowTerrain(@builtin(position) pixel:vec4f)->@builtin(frag_depth) f32{
  ${backdrop ? `if(backdropTerrainLevels()==0u){return 1.0;}
  let layer=rasterLighting.control.y;let uv=pixel.xy/f32(${SVO_RASTER_SHADOW_SIZE})*vec2f(2,-2)+vec2f(-1,1);
  if(layer>=2u){
    let spot=rasterLighting.spots[layer-2u];let basis=rasterSpotBasis(spot);
    let ray=normalize(basis[2]+(basis[0]*uv.x+basis[1]*uv.y)*spot.directionTan.w);
    let cosine=dot(ray,basis[2]);
    let hit=backdropTerrainSmoothTrace(spot.positionNear.xyz,ray,spot.positionNear.w/cosine,spot.rangeRadiusIndex.x/cosine);
    if(hit.t>=0.0){let clip=rasterSpotClip(spot.positionNear.xyz+ray*hit.t,spot);return clamp(clip.z/clip.w,0.0,1.0);}return 1.0;
  }
  let basis=rasterSunBasis();let origin=rasterMapCenter(layer)+(basis[0]*uv.x+basis[1]*uv.y)*rasterMapExtent(layer)+basis[2]*rasterLighting.depthCell.x*.5;
  let hit=backdropTerrainSmoothTrace(origin,-basis[2],0.0,rasterLighting.depthCell.x);
  if(hit.t>=0.0){return clamp(hit.t/rasterLighting.depthCell.x,0.0,1.0);}` : ""}
  return 1.0;
}
${importedSurfaceShadowWGSL}
${rasterHorizonWGSL}
${rasterCoarseAoWGSL}
@compute @workgroup_size(8,8) fn rasterContactMain(@builtin(global_invocation_id) id:vec3u){
  let dims=textureDimensions(rasterContactOutput);if(any(id.xy>=dims)){return;}
  let fullDims=vec2i(textureDimensions(drySplitGeometryRead));
  let coord=min(vec2i(id.xy)*2+vec2i(1),fullDims-1);let g=drySplitGeometryAt(coord);let pixel=vec2f(coord)+.5;
  if(!(g.w<DRY_MISS)||dot(g.xyz,g.xyz)<.1){textureStore(rasterContactOutput,vec2i(id.xy),vec4f(1,65504,0,0));textureStore(rasterShadowOutput,vec2i(id.xy),vec4f(1));return;}
  let normal=normalize(g.xyz);let position=rasterWorldAt(pixel,g.w);
  var contact=1.0;
  if((dry.materialPublication.w&1u)!=0u){contact=min(rasterHorizonVisibility(pixel,position,normal),rasterCoarseVisibility(position,normal));}
  var shadow=1.0;
  if((dry.materialPublication.w&2u)!=0u){shadow=rasterSunVisibility(position,normal);}
  textureStore(rasterShadowOutput,vec2i(id.xy),vec4f(shadow));
  textureStore(rasterContactOutput,vec2i(id.xy),vec4f(contact,min(g.w,65504.0),rasterOctEncode(normal)));
}
`;
}

const cacheWGSL = /* wgsl */ `
struct Params { control:vec4u, farCenterExtent:vec4f, depthCell:vec4f, sun:vec4f }
struct Redraw { minimum:vec4u, maximum:vec4u }
@group(0) @binding(0) var<storage,read_write> mesh:array<u32>;
@group(0) @binding(1) var<uniform> params:Params;
@group(0) @binding(2) var<storage,read_write> cache:array<u32>;
@group(0) @binding(3) var<storage,read_write> redraw:Redraw;
fn shadowBoxMinimum(slot:u32)->vec3u{let base=${M.shadowBoxes}u+slot*6u;return ~vec3u(mesh[base],mesh[base+1u],mesh[base+2u]);}
fn shadowBoxMaximum(slot:u32)->vec3u{let base=${M.shadowBoxes}u+slot*6u;return vec3u(mesh[base+3u],mesh[base+4u],mesh[base+5u]);}
@compute @workgroup_size(1) fn main(){
  let building=mesh[${M.building}]!=0u;
  // The maps are drawn whole when what they view changed, and while and after
  // the mesh is built whole.
  let wholeWords=array<u32,3>(params.control.x,mesh[${M.wholeBuilds}],mesh[${M.usable}]);
  var whole=building&&mesh[${M.mode}]==${SVO_SURFACE_MESH_MODE.initial}u;
  for(var i=0u;i<3u;i++){whole=whole||cache[${CACHE.revisions}u+i]!=wholeWords[i];cache[${CACHE.revisions}u+i]=wholeWords[i];}
  // An edit's build redraws the box of the bricks it extracts, on every frame
  // it emits into the drawn arena and once more when it is over.
  let partWords=array<u32,5>(mesh[${M.topologyRevision}],mesh[${M.geometryRevision}],mesh[${M.frontCursor}],mesh[${M.builds}],mesh[${M.consumedMaintenanceRevision}]);
  var changed=building;
  for(var i=0u;i<5u;i++){changed=changed||cache[${CACHE.revisions + 3}u+i]!=partWords[i];cache[${CACHE.revisions + 3}u+i]=partWords[i];}
  let minimum=min(shadowBoxMinimum(0u),shadowBoxMinimum(1u));let maximum=max(shadowBoxMaximum(0u),shadowBoxMaximum(1u));
  let boxed=!whole&&changed&&all(minimum<=maximum);
  // The box of a build that is over has now been drawn. A claim made since
  // belongs to the next build and is in the other box.
  let drawn=${M.shadowBoxes}u+((mesh[${M.builds}]+select(0u,1u,building))&1u)*6u;
  for(var i=0u;i<6u;i++){mesh[drawn+i]=0u;}
  var mode=${REDRAW.nothing}u;if(whole){mode=${REDRAW.whole}u;}else if(boxed){mode=${REDRAW.listed}u;}
  redraw.minimum=vec4u(minimum,mode);redraw.maximum=vec4u(maximum,0u);
  // Clearing is an indirect draw so unchanged maps can use load. A listed
  // redraw's caster count is the list's, stored once the list is made.
  cache[${CACHE.clearDraw}]=6u;cache[${CACHE.clearDraw + 1}]=select(0u,1u,whole||boxed);cache[${CACHE.clearDraw + 2}]=0u;cache[${CACHE.clearDraw + 3}]=0u;
  cache[${CACHE.casterDraw}]=4u;cache[${CACHE.casterDraw + 1}]=select(0u,mesh[${M.frontCursor}],whole);cache[${CACHE.casterDraw + 2}]=0u;cache[${CACHE.casterDraw + 3}]=0u;
  cache[${CACHE.listed}]=0u;cache[${CACHE.overflowed}]=0u;
}
@compute @workgroup_size(1) fn resolve(){
  if(redraw.minimum.w!=${REDRAW.listed}u){return;}
  if(cache[${CACHE.overflowed}]!=0u){redraw.minimum.w=${REDRAW.tested}u;cache[${CACHE.casterDraw + 1}]=mesh[${M.frontCursor}];return;}
  cache[${CACHE.casterDraw + 1}]=cache[${CACHE.listed}];
}
`;

export class SvoRasterAo {
  readonly consumerLayout: GPUBindGroupLayout;
  readonly shadowLayout: GPUBindGroupLayout;
  readonly aoLayout: GPUBindGroupLayout;
  readonly params: GPUBuffer;
  shadow: GPUTexture;
  readonly cache: GPUBuffer;
  /** What the shadow passes of a frame redraw; written by the cache check on the GPU. */
  private readonly redraw: GPUBuffer;
  /** Arena indices of the quads a partial redraw draws, listed on the GPU. */
  private readonly casters: GPUBuffer;
  private readonly casterLayout: GPUBindGroupLayout;
  private readonly casterGroup: GPUBindGroup;
  private readonly cacheLayout: GPUBindGroupLayout;
  readonly sampler: GPUSampler;
  private cachePipeline?: GPUComputePipeline;
  private resolvePipeline?: GPUComputePipeline;
  private cacheCompile?: Promise<[GPUComputePipeline, GPUComputePipeline]>;
  private cacheGroup?: GPUBindGroup;
  private meshState?: GPUBuffer;
  private shadowGroups: GPUBindGroup[];
  private shadowViews: GPUTextureView[];
  private ao?: GPUTexture;
  private shadowVisibility?: GPUTexture;
  private aoGroup?: GPUBindGroup;
  consumer?: GPUBindGroup;
  private key = "";
  private generation = 0;
  private renderedSettledKey?: string;
  private coarseAoStrength = 0;
  private aoContrast = 1;
  private shadowLayerCount = 2;
  /** One indexed indirect record plus its immutable instance capacity per asset. */
  nativeDrawArguments?: GPUBuffer;
  private nativeArgumentGroup?: GPUBindGroup;
  private nativeArgumentPipeline?: GPUComputePipeline;
  private nativeArgumentLayout?: GPUBindGroupLayout;

  constructor(private readonly device: GPUDevice) {
    this.params=device.createBuffer({label:"Raster lighting parameters",size:(2+RASTER_MAX_SPOT_LIGHTS)*256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.cache=device.createBuffer({label:"Raster shadow cache generations and indirect draws",size:CACHE.bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    const uniform:GPUBindGroupLayoutEntry={binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT|GPUShaderStage.COMPUTE,buffer:{type:"uniform"}};
    this.consumerLayout=device.createBindGroupLayout({label:"Raster lighting consumer",entries:[uniform,
      {binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"depth",viewDimension:"2d-array"}},
      {binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"comparison"}},
      {binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"unfilterable-float"}},
      {binding:5,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"unfilterable-float"}}]});
    this.redraw=device.createBuffer({label:"Raster shadow redraw box",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.STORAGE});
    this.casters=device.createBuffer({label:"Raster shadow redraw casters",size:SVO_RASTER_SHADOW_CASTER_CAPACITY*4,usage:GPUBufferUsage.STORAGE});
    const redrawn:GPUBindGroupLayoutEntry={binding:7,visibility:GPUShaderStage.VERTEX|GPUShaderStage.COMPUTE,buffer:{type:"uniform"}};
    this.shadowLayout=device.createBindGroupLayout({entries:[uniform,redrawn,{binding:10,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]});
    this.casterLayout=device.createBindGroupLayout({entries:[uniform,redrawn,
      {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
    this.casterGroup=device.createBindGroup({layout:this.casterLayout,entries:[{binding:0,resource:{buffer:this.params,offset:0,size:256}},
      {binding:7,resource:{buffer:this.redraw}},{binding:8,resource:{buffer:this.casters}},{binding:9,resource:{buffer:this.cache}}]});
    this.cacheLayout=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
    this.aoLayout=device.createBindGroupLayout({entries:[uniform,
      {binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"depth",viewDimension:"2d-array"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,sampler:{type:"comparison"}},
      {binding:4,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba16float"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float"}}]});
    this.shadow=device.createTexture({label:"Cached world-space sun shadows",size:[SVO_RASTER_SHADOW_SIZE,SVO_RASTER_SHADOW_SIZE,2],format:"depth32float",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});
    this.shadowViews=[0,1].map(baseArrayLayer=>this.shadow.createView({dimension:"2d",baseArrayLayer,arrayLayerCount:1}));
    this.shadowGroups=Array.from({length:2+RASTER_MAX_SPOT_LIGHTS},(_,i)=>i).map(layer=>device.createBindGroup({layout:this.shadowLayout,entries:[{binding:0,resource:{buffer:this.params,offset:layer*256,size:256}},{binding:7,resource:{buffer:this.redraw}},{binding:10,resource:{buffer:this.casters}}]}));
    this.sampler=device.createSampler({compare:"less-equal",magFilter:"linear",minFilter:"linear"});
  }
  async compile(module:GPUShaderModule,scene:GPUBindGroupLayout,mesh:GPUBindGroupLayout,split:GPUBindGroupLayout,nativeShadows=true):Promise<RasterAoPipelines>{
    if(nativeShadows){
      this.nativeArgumentLayout ??= this.device.createBindGroupLayout({entries:[
        {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
        {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      ]});
      this.nativeArgumentPipeline ??= await this.device.createComputePipelineAsync({label:"Native shadow cached draw gate",
        layout:this.device.createPipelineLayout({bindGroupLayouts:[this.nativeArgumentLayout]}),compute:{entryPoint:"main",
          module:this.device.createShaderModule({code:importedSurfaceShadowGateWGSL})}});
      this.bindNativeDrawArguments();
    }
    const cacheModule=()=>this.device.createShaderModule({code:cacheWGSL});
    const cacheLayout=()=>this.device.createPipelineLayout({bindGroupLayouts:[this.cacheLayout]});
    this.cacheCompile ??= (()=>{const module=cacheModule(),layout=cacheLayout();return Promise.all([
      this.device.createComputePipelineAsync({label:"Raster shadow invalidation",layout,compute:{module,entryPoint:"main"}}),
      this.device.createComputePipelineAsync({label:"Raster shadow caster count",layout,compute:{module,entryPoint:"resolve"}})]);})();
    [this.cachePipeline,this.resolvePipeline] = await this.cacheCompile;
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[scene,mesh,this.shadowLayout]});
    const [shadow,terrain,casters,ao,nativeShadow]=await Promise.all([
      this.device.createRenderPipelineAsync({label:"Raster shadow casters",layout,vertex:{module,entryPoint:"rasterShadowVertex"},primitive:{topology:"triangle-strip",cullMode:"none"},depthStencil:{format:"depth32float",depthWriteEnabled:true,depthCompare:"less"}}),
      this.device.createRenderPipelineAsync({label:"Raster shadow terrain and clear",layout,vertex:{module,entryPoint:"rasterShadowClearVertex"},fragment:{module,entryPoint:"rasterShadowTerrain",targets:[]},primitive:{topology:"triangle-list"},depthStencil:{format:"depth32float",depthWriteEnabled:true,depthCompare:"always"}}),
      this.device.createComputePipelineAsync({label:"Raster shadow redraw casters",layout:this.device.createPipelineLayout({bindGroupLayouts:[scene,mesh,this.casterLayout]}),compute:{module,entryPoint:"rasterShadowCastersMain"}}),
      this.device.createComputePipelineAsync({label:"Horizon contact AO",layout:this.device.createPipelineLayout({bindGroupLayouts:[scene,split,this.aoLayout]}),compute:{module,entryPoint:"rasterContactMain"}}),
      nativeShadows ? this.device.createRenderPipelineAsync({label:"Native asset full-detail shadow casters",layout,
        vertex:{module,entryPoint:"rasterNativeShadowVertex",buffers:[
          {arrayStride:12,attributes:[{shaderLocation:0,offset:0,format:"float32x3"}]},
          {arrayStride:8,stepMode:"instance",attributes:[{shaderLocation:1,offset:0,format:"uint32"},{shaderLocation:2,offset:4,format:"float32"}]},
        ]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth32float",depthWriteEnabled:true,depthCompare:"less"}}) : undefined,
    ]);return {shadow,terrain,casters,ao,nativeShadow};
  }
  /** Membership/asset changes alter caster geometry even when the camera does not move. */
  invalidateNativeGeometry():void{
    this.generation++;this.renderedSettledKey=undefined;
    for(let layer=0;layer<this.shadowLayerCount;layer++)this.device.queue.writeBuffer(this.params,layer*256,new Uint32Array([this.generation]));
  }
  setNativeDrawArguments(records:Uint32Array<ArrayBuffer>):void{
    this.nativeDrawArguments?.destroy();this.nativeDrawArguments=undefined;this.nativeArgumentGroup=undefined;
    if(!records.length)return;
    this.nativeDrawArguments=this.device.createBuffer({label:"Native shadow cached indexed arguments",size:records.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT,mappedAtCreation:true});
    new Uint32Array(this.nativeDrawArguments.getMappedRange()).set(records);this.nativeDrawArguments.unmap();
    this.bindNativeDrawArguments();
  }
  private bindNativeDrawArguments():void{
    if(!this.nativeArgumentLayout||!this.nativeDrawArguments)return;
    this.nativeArgumentGroup=this.device.createBindGroup({layout:this.nativeArgumentLayout,entries:[
      {binding:0,resource:{buffer:this.redraw}},{binding:1,resource:{buffer:this.nativeDrawArguments}},
    ]});
  }
  /**
   * The views the maps are drawn from. A map is redrawn whole only when its
   * view changes: what casts into it is the mesh's to report, and a light's or
   * a material's colour casts nothing.
   */
  update(origin:readonly number[],extent:readonly number[],cellSize:readonly number[],direction:readonly number[],lights?:Uint32Array):void{
    const spots=rasterSpotLights(lights);
    const key=[...origin,...extent,...cellSize,...direction,...spots.flatMap(spot=>[...spot.position,spot.near,...spot.direction,spot.tanHalfAngle,spot.far,spot.radius,spot.index])].join(",");if(key===this.key)return;
    this.key=key;this.generation++;
    const radius=Math.max(.1,Math.hypot(...extent)*.5);
    this.shadowLayerCount=2+spots.length;
    if(this.shadow.depthOrArrayLayers!==this.shadowLayerCount){
      this.shadow.destroy();
      this.shadow=this.device.createTexture({label:"Cached sun and spot shadows",size:[SVO_RASTER_SHADOW_SIZE,SVO_RASTER_SHADOW_SIZE,this.shadowLayerCount],format:"depth32float",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});
      this.shadowViews=Array.from({length:this.shadowLayerCount},(_,baseArrayLayer)=>this.shadow.createView({dimension:"2d",baseArrayLayer,arrayLayerCount:1}));
      this.consumer=undefined;
      if(this.ao)this.ensureSize(this.ao.width*2,this.ao.height*2);
    }
    for(let layer=0;layer<this.shadowLayerCount;layer++){
      const buffer=new ArrayBuffer(256),f=new Float32Array(buffer),u=new Uint32Array(buffer);
      u.set([this.generation,layer,spots.length,0]);f.set(origin.map((v,i)=>v+extent[i]*.5),4);f[7]=radius*1.02;
      f[8]=radius*4;f[9]=Math.max(...cellSize);f[10]=this.aoContrast;f.set(direction,12);f[15]=this.coarseAoStrength;
      for(let slot=0;slot<spots.length;slot++){
        const spot=spots[slot]!,offset=16+slot*12;
        f.set([...spot.position,spot.near,...spot.direction,spot.tanHalfAngle,spot.far,spot.radius,spot.index,0],offset);
      }
      this.device.queue.writeBuffer(this.params,layer*256,buffer);
    }
  }
  setCoarseAoStrength(strength:number):void{
    if(strength===this.coarseAoStrength)return;
    this.coarseAoStrength=strength;
    for(let layer=0;layer<this.shadowLayerCount;layer++)this.device.queue.writeBuffer(this.params,layer*256+60,new Float32Array([strength]));
  }
  setAoContrast(contrast:number):void{
    if(contrast===this.aoContrast)return;
    this.aoContrast=contrast;
    for(let layer=0;layer<this.shadowLayerCount;layer++)this.device.queue.writeBuffer(this.params,layer*256+40,new Float32Array([contrast]));
  }
  ensureSize(width:number,height:number):void{
    width=Math.ceil(width/2);height=Math.ceil(height/2);
    if(this.consumer&&this.ao?.width===width&&this.ao.height===height)return;
    this.ao?.destroy();this.ao=this.device.createTexture({label:"Half-resolution sun visibility and horizon AO",size:[width,height],format:"rgba16float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});
    this.shadowVisibility?.destroy();this.shadowVisibility=this.device.createTexture({label:"Half-resolution sun visibility",size:[width,height],format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});
    const uniform={buffer:this.params,offset:0,size:256};const view=this.ao.createView();const shadowView=this.shadowVisibility.createView();
    this.aoGroup=this.device.createBindGroup({layout:this.aoLayout,entries:[{binding:0,resource:uniform},{binding:1,resource:this.shadow.createView({dimension:"2d-array"})},{binding:2,resource:this.sampler},{binding:4,resource:view},{binding:6,resource:shadowView}]});
    this.consumer=this.device.createBindGroup({layout:this.consumerLayout,entries:[{binding:0,resource:uniform},{binding:1,resource:this.shadow.createView({dimension:"2d-array"})},{binding:2,resource:this.sampler},{binding:3,resource:view},{binding:5,resource:shadowView}]});
  }
  private readonly aoPassDescriptor: GPUComputePassDescriptor = { label: "Half-resolution sun visibility and horizon AO" };
  private renderedSettledGeneration = -1;

  encode(encoder:GPUCommandEncoder,pipelines:RasterAoPipelines,scene:GPUBindGroup,mesh:GPUBindGroup,split:GPUBindGroup,state:GPUBuffer,arenaQuads:number,aoEnabled:boolean,shadowsEnabled:boolean,shadowComplete?:()=>void,aoComplete?:()=>void,settledMeshKey?:string,drawNative?:(pass:GPURenderPassEncoder)=>void):void{
    if(state!==this.meshState){this.renderedSettledKey=undefined;this.meshState=state;this.cacheGroup=this.device.createBindGroup({layout:this.cacheLayout,entries:[{binding:0,resource:{buffer:state}},{binding:1,resource:{buffer:this.params,offset:0,size:64}},{binding:2,resource:{buffer:this.cache}},{binding:3,resource:{buffer:this.redraw}}]});}
    const settledKey=settledMeshKey;
    const reuse=settledKey!==undefined&&settledKey===this.renderedSettledKey&&this.renderedSettledGeneration===this.generation;
    if(!shadowsEnabled)this.renderedSettledKey=undefined;
    if(shadowsEnabled&&!reuse){
    const invalidation=encoder.beginComputePass({label:"Raster shadow cache check"});invalidation.setPipeline(this.cachePipeline!);invalidation.setBindGroup(0,this.cacheGroup!);invalidation.dispatchWorkgroups(1);invalidation.end();
    if(this.nativeArgumentPipeline&&this.nativeArgumentGroup){
      const gate=encoder.beginComputePass({label:"Native shadow cached draw gate"});gate.setPipeline(this.nativeArgumentPipeline);gate.setBindGroup(0,this.nativeArgumentGroup);gate.dispatchWorkgroups(Math.ceil(this.nativeDrawArguments!.size/24/64));gate.end();
    }
    // The casters of a partial redraw, then their count for the draws below.
    const groups=Math.ceil(arenaQuads/64);
    const list=encoder.beginComputePass({label:"Raster shadow redraw casters"});list.setPipeline(pipelines.casters);list.setBindGroup(0,scene);list.setBindGroup(1,mesh);list.setBindGroup(2,this.casterGroup);list.dispatchWorkgroups(Math.min(groups,65535),Math.max(1,Math.ceil(groups/65535)));list.end();
    const counted=encoder.beginComputePass({label:"Raster shadow caster count"});counted.setPipeline(this.resolvePipeline!);counted.setBindGroup(0,this.cacheGroup!);counted.dispatchWorkgroups(1);counted.end();
    for(let layer=0;layer<this.shadowLayerCount;layer++){
      const pass=encoder.beginRenderPass({label:layer<2?`Cached sun shadow ${layer}`:`Cached spot shadow ${layer-2}`,colorAttachments:[],depthStencilAttachment:{view:this.shadowViews[layer],depthLoadOp:"load",depthStoreOp:"store"}});
      pass.setBindGroup(0,scene);pass.setBindGroup(1,mesh);pass.setBindGroup(2,this.shadowGroups[layer]);pass.setPipeline(pipelines.terrain);pass.drawIndirect(this.cache,0);pass.setPipeline(pipelines.shadow);pass.drawIndirect(this.cache,16);
      if(pipelines.nativeShadow&&drawNative){pass.setPipeline(pipelines.nativeShadow);drawNative(pass);}pass.end();
    }
    this.renderedSettledKey=settledKey;this.renderedSettledGeneration=this.generation;
    shadowComplete?.();
    }else if(shadowsEnabled){
      encoder.clearBuffer(this.cache,0,32);
      shadowComplete?.();
    }
    if(aoEnabled||shadowsEnabled){const ao=encoder.beginComputePass(this.aoPassDescriptor);ao.setPipeline(pipelines.ao);ao.setBindGroup(0,scene);ao.setBindGroup(1,split);ao.setBindGroup(2,this.aoGroup!);ao.dispatchWorkgroups(Math.ceil(this.ao!.width/8),Math.ceil(this.ao!.height/8));ao.end();aoComplete?.();}
  }
  destroy():void{this.nativeDrawArguments?.destroy();this.params.destroy();this.cache.destroy();this.redraw.destroy();this.casters.destroy();this.shadow.destroy();this.ao?.destroy();this.shadowVisibility?.destroy();}
}
