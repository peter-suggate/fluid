import asset from './bub.generated.json';
import type { RigidBodyState } from '../../../core/rigid-body';

/** Render-only shape code. The solver continues to receive the capsule ABI. */
export const BUB_RENDER_SHAPE = 5;
export const BUB_PARTS = asset.parts;
export function robotRenderExtent(body: RigidBodyState): readonly [number, number, number] | undefined {
  if (body.description.appearance !== 'bub') return undefined;
  const d = body.description.dimensions_m;
  return [d.x, (d.y + 2 * d.x) / 2, d.x];
}
export function robotRenderRadius(body: RigidBodyState): number | undefined {
  const extent = robotRenderExtent(body);
  return extent && Math.hypot(...extent);
}
const vec = (v: readonly number[]) => `vec3f(${v.map(n => Number(n).toFixed(6)).join(',')})`;
/** Exact ray/ellipsoid intersections of Blender's exported part transforms.
 * Used by primary visibility, reflections, water and shadow rays alike. */
export const robotShaderLibrary = /* wgsl */ `
const BUB_PART_COUNT:u32=${BUB_PARTS.length}u;
const BUB_CENTERS=array<vec3f,${BUB_PARTS.length}>(${BUB_PARTS.map(p => vec(p.center)).join(',')});
const BUB_RADII=array<vec3f,${BUB_PARTS.length}>(${BUB_PARTS.map(p => vec(p.radius)).join(',')});
const BUB_COLORS=array<vec3f,${BUB_PARTS.length}>(${BUB_PARTS.map(p => vec(p.color)).join(',')});
struct BubHit { t:f32, normal:vec3f }
fn bubHit(origin:vec3f,direction:vec3f,extent:vec3f)->BubHit {
  let scale=extent/vec3f(.38,.5,.38);var best=BubHit(1e20,vec3f(0,1,0));
  for(var i=0u;i<BUB_PART_COUNT;i++){
    let radius=BUB_RADII[i]*scale;let center=BUB_CENTERS[i]*scale;
    let o=(origin-center)/radius;let d=direction/radius;
    let a=dot(d,d);let b=dot(o,d);let c=dot(o,o)-1.0;let disc=b*b-a*c;
    if(disc<0.0){continue;}let root=sqrt(disc);var t=(-b-root)/a;
    if(t<=1e-4){t=(-b+root)/a;}
    if(t>1e-4&&t<best.t){best=BubHit(t,normalize((origin+direction*t-center)/(radius*radius)));}
  }return best;
}
fn bubColor(position:vec3f,extent:vec3f)->vec3f {
  let p=position/(extent/vec3f(.38,.5,.38));var distance=1e20;var color=BUB_COLORS[0];
  for(var i=0u;i<BUB_PART_COUNT;i++){
    let radius=BUB_RADII[i];let d=abs(length((p-BUB_CENTERS[i])/radius)-1.0)*min(radius.x,min(radius.y,radius.z));
    if(d<distance){distance=d;color=BUB_COLORS[i];}
  }return color;
}
`;
