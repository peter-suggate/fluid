/** Optional same-owner occlusion in the native FVX scalar field. Geometry is
 * still voxel based; this supplements, rather than replaces, world AO. */
export const SAMPLED_AMBIENT_VISIBILITY_MAX_STEPS = 512;
export function sampledAmbientVisibilityWGSL(load: string, maximumSteps = SAMPLED_AMBIENT_VISIBILITY_MAX_STEPS): string {
  if (!Number.isInteger(maximumSteps) || maximumSteps < 1 || maximumSteps > 1024) throw new RangeError("Native AO step limit must be between 1 and 1024");
  return /* wgsl */ `
const SAMPLED_AO_CLEAR=0u;const SAMPLED_AO_BLOCKED=1u;
const SAMPLED_AO_EXHAUSTED=2u;const SAMPLED_AO_INVALID=3u;const SAMPLED_AO_NOT_APPLICABLE=4u;
struct SampledAoRay{status:u32,steps:u32,distance:f32}
struct SampledAmbientVisibility{visibility:f32,status:u32,steps:u32,blocked:u32}
fn sampledAoRay(base:u32,origin:vec3f,direction:vec3f,maximumDistance:f32,stepBudget:u32)->SampledAoRay{
  let lower=vec3f(bitcast<f32>(${load}(base)),bitcast<f32>(${load}(base+1u)),bitcast<f32>(${load}(base+2u)));
  let h=bitcast<f32>(${load}(base+3u));let dims=vec3u(${load}(base+4u),${load}(base+5u),${load}(base+6u));
  if(!(h>0.0)||any(dims==vec3u(0))||!(maximumDistance>0.0)||any(origin!=origin)||any(direction!=direction)){
    return SampledAoRay(SAMPLED_AO_INVALID,0u,0.0);
  }
  let upper=lower+vec3f(dims)*h;var begin=0.0;var end=maximumDistance;
  for(var axis=0u;axis<3u;axis++){
    if(abs(direction[axis])<1e-8){if(origin[axis]<lower[axis]||origin[axis]>upper[axis]){return SampledAoRay(SAMPLED_AO_CLEAR,0u,0.0);}}
    else{let a=(lower[axis]-origin[axis])/direction[axis];let b=(upper[axis]-origin[axis])/direction[axis];begin=max(begin,min(a,b));end=min(end,max(a,b));}
  }
  if(end<=begin){return SampledAoRay(SAMPLED_AO_CLEAR,0u,0.0);}
  let epsilon=max(h*1e-4,1e-7);var t=begin;
  let bd=vec3u(${load}(base+8u),${load}(base+9u),${load}(base+10u));
  // Carry the integer cell across exact boundary events instead of nudging the
  // ray by an epsilon, which could skip a thin surface in the next cell.
  var cell=clamp(vec3i(floor((origin+direction*t-lower)/h)),vec3i(0),vec3i(dims)-vec3i(1));
  for(var step=0u;step<${maximumSteps}u;step++){
    if(step>=stepBudget){return SampledAoRay(SAMPLED_AO_EXHAUSTED,step,t);}
    if(any(cell<vec3i(0))||any(cell>=vec3i(dims))){return SampledAoRay(SAMPLED_AO_CLEAR,step,t);}
    let brick=vec3u(cell)>>vec3u(3);
    let directory=base+16u+2u*(brick.x+bd.x*(brick.y+bd.y*brick.z));
    let relative=${load}(directory);var d=0.0;var minimum=1e20;
    var regionLower=cell;var regionUpper=cell+vec3i(1);
    if(relative==0u){
      // An absent brick is explicitly constant air OR constant solid. Only
      // positive finite values certify a whole brick as empty.
      d=bitcast<f32>(${load}(directory+1u));minimum=d;
      regionLower=vec3i(brick)*8;regionUpper=min(regionLower+vec3i(8),vec3i(dims));
    }else{
      let f=clamp((origin+direction*t-lower)/h-vec3f(cell),vec3f(0),vec3f(1));
      let c=vec3u(cell)&vec3u(7);
      for(var cornerIndex=0u;cornerIndex<8u;cornerIndex++){
        let corner=vec3u(cornerIndex&1u,(cornerIndex>>1u)&1u,cornerIndex>>2u);let n=c+corner;
        let value=bitcast<f32>(${load}(base+relative+2u*(n.x+9u*(n.y+9u*n.z))));
        if(!(abs(value)<1e10)){return SampledAoRay(SAMPLED_AO_INVALID,step+1u,t);}
        let w=select(vec3f(1)-f,f,corner!=vec3u(0));d+=value*w.x*w.y*w.z;minimum=min(minimum,value);
      }
    }
    if(!(abs(d)<1e10)){return SampledAoRay(SAMPLED_AO_INVALID,step+1u,t);}
    if(d<=epsilon){return SampledAoRay(SAMPLED_AO_BLOCKED,step+1u,t);}
    var boundary=vec3f(1e30);
    for(var axis=0u;axis<3u;axis++){
      if(abs(direction[axis])>=1e-8){
        let face=select(regionLower[axis],regionUpper[axis],direction[axis]>0.0);
        boundary[axis]=(lower[axis]+f32(face)*h-origin[axis])/direction[axis];
      }
    }
    let exit=min(boundary.x,min(boundary.y,boundary.z));
    // Trilinear interpolation is a convex combination of the eight corners:
    // all corners >epsilon proves the entire cell empty, even for grazing
    // rays. Mixed cells retain the production conservative Lipschitz step.
    // Keep the original safe sphere step when it reaches farther: clipping
    // every step to a cell boundary needlessly slows coarse/empty regions.
    let scalarAdvance=t+d/1.75;
    let useBoundary=minimum>epsilon&&exit>=scalarAdvance;
    let advance=select(scalarAdvance,exit,useBoundary);
    if(advance>=end){return SampledAoRay(SAMPLED_AO_CLEAR,step+1u,end);}
    if(useBoundary){
      t=max(t,exit);
      var next=vec3i(floor((origin+direction*t-lower)/h));
      for(var axis=0u;axis<3u;axis++){
        if(boundary[axis]==exit){next[axis]=select(regionLower[axis]-1,regionUpper[axis],direction[axis]>0.0);}
      }
      cell=next;
    }else{
      t=advance;cell=vec3i(floor((origin+direction*t-lower)/h));
    }
  }
  return SampledAoRay(SAMPLED_AO_EXHAUSTED,${maximumSteps}u,t);
}
fn sampledAmbientVisibility(base:u32,receiver:vec3f,maximumDistance:f32,stepBudget:u32)->SampledAmbientVisibility{
  let surface=sampledSurfaceProjection(base,receiver);
  if(surface.valid==0u){return SampledAmbientVisibility(1.0,SAMPLED_AO_INVALID,0u,0u);}
  let h=bitcast<f32>(${load}(base+3u));
  let bias=max(h*.02,2e-7*max(1.0,max(abs(receiver.x),max(abs(receiver.y),abs(receiver.z)))));
  let origin=surface.position+surface.normal*bias;
  let helper=select(vec3f(0,1,0),vec3f(1,0,0),abs(surface.normal.y)>.9);
  let tangent=normalize(cross(helper,surface.normal));let bitangent=cross(surface.normal,tangent);
  var blocked=0u;var steps=0u;
  for(var sample=0u;sample<4u;sample++){
    // Equal-weight stratified cosine hemisphere; no arbitrary visibility gain.
    let r=sqrt((f32(sample)+.5)/4.0);let angle=(f32(sample)+.5)*2.39996322973;
    let direction=tangent*(r*cos(angle))+bitangent*(r*sin(angle))+surface.normal*sqrt(1.0-r*r);
    let ray=sampledAoRay(base,origin,direction,maximumDistance,stepBudget);steps+=ray.steps;
    if(ray.status>=SAMPLED_AO_EXHAUSTED){return SampledAmbientVisibility(1.0,ray.status,steps,blocked);}
    blocked+=select(0u,1u,ray.status==SAMPLED_AO_BLOCKED);
  }
  return SampledAmbientVisibility(1.0-f32(blocked)*.25,select(SAMPLED_AO_CLEAR,SAMPLED_AO_BLOCKED,blocked>0u),steps,blocked);
}
`;
}
