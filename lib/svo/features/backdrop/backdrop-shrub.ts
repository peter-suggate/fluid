/**
 * A cheap broadleaf shrub for stored terrain rings. Three woody shoots carry
 * five broad leaf packets each; the packets represent unresolved sprays, not
 * individual leaves. Fixed topology, no noise octaves or recursive branching.
 * Coordinates are in units of scatter radius, within the old puff's bounds.
 */
type V3 = readonly [number, number, number];
const add = (a: V3, b: V3): V3 => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
const mul = (a: V3, s: number): V3 => [a[0]*s, a[1]*s, a[2]*s];
const dot = (a: V3, b: V3) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const unit = (a: V3) => mul(a, 1 / Math.hypot(...a));
const cross = (a: V3, b: V3): V3 => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const shoots: readonly V3[] = [[.46, 1.1, .08], [-.34, 1.23, .26], [-.12, .97, -.44]];
export const BACKDROP_SHRUB_STEMS = shoots.map(tip => ({ a: [0, -.08, 0] as V3, b: tip, radius: .045 }));
export const BACKDROP_SHRUB_LEAVES = shoots.flatMap((tip, shoot) => {
  const radial = unit([tip[0], 0, tip[2]]);
  const side: V3 = [-radial[2], 0, radial[0]];
  return Array.from({ length: 5 }, (_, leaf) => {
    const terminal = leaf === 4;
    const sign = leaf % 2 === 0 ? 1 : -1;
    const t = terminal ? 1 : .49 + .24 * Math.floor(leaf / 2) + .045 * (leaf % 2);
    const u = terminal ? unit(add(mul(radial, .45), [0, .8, 0]))
      : unit(add(add(mul(side, sign), mul(radial, .22)), [0, .38 + .09 * shoot, 0]));
    const w = unit(cross(u, [0, 1, 0]));
    const v = cross(w, u);
    return { centre: add(mul(tip, t), mul(u, terminal ? .07 : .21)), u, v, w,
      radii: [terminal ? .28 : .34, .075, terminal ? .17 : .21] as V3 };
  });
});
export const BACKDROP_SHRUB_REACH = 1.3;
export const BACKDROP_SHRUB_RISE = 1.6;

/** Conservative signed distance; also the CPU oracle for the GPU field. */
export function backdropShrubDistance(p: V3): number {
  let d = Infinity;
  for (const stem of BACKDROP_SHRUB_STEMS) {
    const pa = add(p, mul(stem.a, -1)), ba = add(stem.b, mul(stem.a, -1));
    const t = Math.max(0, Math.min(1, dot(pa, ba) / dot(ba, ba)));
    d = Math.min(d, Math.hypot(...add(pa, mul(ba, -t))) - stem.radius);
  }
  for (const leaf of BACKDROP_SHRUB_LEAVES) {
    const q = add(p, mul(leaf.centre, -1));
    d = Math.min(d, (Math.hypot(dot(q, leaf.u)/leaf.radii[0], dot(q, leaf.v)/leaf.radii[1], dot(q, leaf.w)/leaf.radii[2])-1)*leaf.radii[1]);
  }
  return d;
}

const v = (p: V3) => `vec3f(${p.map(n => n.toFixed(9)).join(",")})`;
/** Both occupancy and continuous meshing call this exact same field. */
export const backdropShrubWGSL = /* wgsl */ `
struct BackdropShrubLeaf { centre:vec3f, u:vec3f, v:vec3f, w:vec3f, radii:vec3f }
const BACKDROP_SHRUB_LEAVES=array<BackdropShrubLeaf,${BACKDROP_SHRUB_LEAVES.length}>(
${BACKDROP_SHRUB_LEAVES.map(l => `BackdropShrubLeaf(${v(l.centre)},${v(l.u)},${v(l.v)},${v(l.w)},${v(l.radii)})`).join(",\n")});
const BACKDROP_SHRUB_TIPS=array<vec3f,3>(${shoots.map(v).join(",")});
fn backdropShrubDistance(p:vec3f)->f32{
  var d=1e20;
  let pa=p-vec3f(0.0,-0.08,0.0);
  for(var i=0u;i<3u;i+=1u){
    let ba=BACKDROP_SHRUB_TIPS[i]-vec3f(0.0,-0.08,0.0);
    d=min(d,length(pa-ba*clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0))-0.045);
  }
  for(var i=0u;i<${BACKDROP_SHRUB_LEAVES.length}u;i+=1u){
    let leaf=BACKDROP_SHRUB_LEAVES[i];let q=p-leaf.centre;
    let e=vec3f(dot(q,leaf.u),dot(q,leaf.v),dot(q,leaf.w))/leaf.radii;
    d=min(d,(length(e)-1.0)*leaf.radii.y);
  }
  return d;
}
`;
