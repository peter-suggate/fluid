/** Conservative distance for an axis-aligned elliptic cylinder. Axis encoded
 * as 3..5 keeps existing saved box/ellipsoid and baked-page records intact. */
export function sculptCylinderDistance(kind: number, p: readonly number[], r: readonly number[]): number {
  const axis = kind - 3, u = (axis + 1) % 3, v = (axis + 2) % 3;
  const radial = (Math.hypot(p[u]! / r[u]!, p[v]! / r[v]!) - 1) * Math.min(r[u]!, r[v]!);
  const cap = Math.abs(p[axis]!) - r[axis]!;
  return Math.hypot(Math.max(radial, 0), Math.max(cap, 0)) + Math.min(Math.max(radial, cap), 0);
}
export const sculptCylinderWGSL = /* wgsl */ `
fn sculptCylinderDistance(kind:u32,p:vec3f,r:vec3f)->f32{
  let axis=kind-3u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  let radial=(length(vec2f(p[u]/r[u],p[v]/r[v]))-1.0)*min(r[u],r[v]);
  let cap=abs(p[axis])-r[axis];let q=vec2f(radial,cap);
  return length(max(q,vec2f(0)))+min(max(q.x,q.y),0.0);
}
`;
