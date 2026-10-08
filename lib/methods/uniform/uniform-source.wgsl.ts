import { LIQUID_EXTRUSION_MAX_EDGES } from "../../core/liquid-extrusion";

/** Bytes of the host parameter block up to the extrusion tail: seventeen vec4f. */
export const UNIFORM_PARAMS_HEAD_BYTES = 272;
/** The whole host parameter block: the head, the extrusion header and its edges. */
export const UNIFORM_PARAMS_BYTES = UNIFORM_PARAMS_HEAD_BYTES + 16 + 16 * LIQUID_EXTRUSION_MAX_EDGES;

/**
 * The tail every struct that reads a drop ends with, directly after the head.
 *
 * extrusion.x is the outline's edge count, zero unless this step's drop is an
 * extrusion; y the outward offset, z the half depth and w the edge radius, all
 * in metres. An extrusion's `drop` is its bounding ball, centred on the
 * mid-plane, so every "is a drop active" test on drop.w still reads true.
 */
export const uniformExtrusionParamsWGSL = /* wgsl */ `
  extrusion: vec4f,
  // One outline edge each: a.xy then b.xy in world metres.
  extrusionEdges: array<vec4f, ${LIQUID_EXTRUSION_MAX_EDGES}>,
`;

/** Source geometry shared by native and canonical mixed execution. */
export const uniformDropSourceWGSL = /* wgsl */ `
// Signed distance in the x/y plane to the drop's raw outline; negative inside.
// The fill is even-odd, so a contour inside another is a hole.
fn dropOutlineDistance(point:vec2f,count:u32)->f32{
  var nearest=3.0e38;var inside=false;
  for(var index=0u;index<count;index+=1u){
    let edge=params.extrusionEdges[index];let a=edge.xy;let e=edge.zw-a;let w=point-a;
    let off=w-clamp(dot(w,e)/dot(e,e),0.0,1.0)*e;nearest=min(nearest,dot(off,off));
    // The ray toward +x; the half-open test counts a shared vertex once.
    if((a.y>point.y)!=(edge.w>point.y)&&point.x<a.x+(point.y-a.y)*e.x/e.y){inside=!inside;}
  }
  return select(sqrt(nearest),-sqrt(nearest),inside);
}
// Signed distance in metres from a world point to this step's drop; negative
// is liquid. An extrusion costs an edge loop, and a caller that takes the
// minimum with a distance it already holds has no use for one at or past
// that distance: beyond the bounding ball by \`ceiling\` the ball's own
// distance, a lower bound, is returned instead. The minimum is unchanged.
fn dropDistance(point:vec3f,ceiling:f32)->f32{
  let d=point-params.drop.xyz;let count=u32(params.extrusion.x);
  if(count==0u){
    return select(length(d)-params.drop.w,
      max(length(d.xy)-params.drop.w,abs(d.z)-params.dropExtent.x),params.dropExtent.x>0.0);
  }
  let bound=length(d)-params.drop.w;if(bound>=ceiling){return bound;}
  // A rounded extrusion of the outline grown by its offset.
  let r=params.extrusion.w;
  let w=vec2f(dropOutlineDistance(point.xy,count)-params.extrusion.y,abs(d.z)-params.extrusion.z)+vec2f(r);
  return min(max(w.x,w.y),0.0)+length(max(w,vec2f(0.0)))-r;
}
fn dropSource(q:vec3i)->f32{
  if(params.drop.w<=0.0){return 0.0;}
  let h=params.cellGravity.xyz;
  let minimum=vec3f(-0.5*params.container.x,0.0,-0.5*params.container.z);
  var covered=0.0;
  for(var sample=0u;sample<8u;sample+=1u){
    let offset=vec3f(f32(sample&1u),f32((sample>>1u)&1u),f32((sample>>2u)&1u))*0.5+vec3f(0.25);
    // The smallest ceiling that is not zero: a sample on the bounding ball
    // itself is outside an extrusion, and only the sign is wanted here.
    if(dropDistance(minimum+(vec3f(q)+offset)*h,1.0e-30)<=0.0){covered+=0.125;}
  }
  return covered;
}
`;
export const uniformSourcePhiWGSL = /* wgsl */ `
fn uvSourcePhi(p:vec3f,phi:f32)->f32{
  var result=phi;
  if(params.drop.w>0.0){result=min(result,dropDistance(traceWorld(p),result));}
  let speed=length(params.inflowVelocityLength.xyz)*inflowStrength();
  if(speed>1e-6){let direction=normalize(params.inflowVelocityLength.xyz);
    let delta=traceWorld(p)-params.inflowPositionRadius.xyz;let axial=dot(delta,direction);
    let plug=max(length(delta-axial*direction)-params.inflowPositionRadius.w,
      max(-axial,axial-speed*params.dimsDt.w));result=min(result,plug);}
  return result;
}
`;
