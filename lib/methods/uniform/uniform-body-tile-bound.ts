/** A lower bound on the distance from a box body to a tile's box, for
 * markTiles (uniform-mixed-bodies.ts), in WGSL and its TypeScript mirror
 * (tests/uniform-body-tile-bound.test.ts holds the two to one rule).
 *
 * Why it is conservative: for two convex sets A and B and any unit vector n,
 * every point of B is at least gap(n) = min_B n.x - max_A n.y from every
 * point of A along n, so dist(A, B) >= gap(n). (By duality the distance is
 * the largest gap over all n.) The bound is the largest gap over the
 * candidate axes:
 * - the body's three axes;
 * - the world's three, the tile's axes;
 * - their nine cross products, normalized and skipped when nearly parallel;
 * - the direction from the body box's closest point to the tile centre.
 * A box's projection onto n is its centre's plus or minus sum_k e_k |n.u_k|
 * (half extents e on axes u), so each gap is exact arithmetic on the poses.
 * The bound is also never below the old one: markTiles keeps the larger of
 * the two lower bounds.
 *
 * The old bound, the shape's 1-Lipschitz distance at the tile centre less
 * the tile's half diagonal, is exact only along the diagonal. Beside a face
 * it understates the gap by up to (sqrt(3) - 1) x 2h = 1.46 h: a moored
 * hull marked tiles 2-4 h from its surface. */

/** WGSL: umBodyBoxTileGap(i, middle, a), body i's box against the world-axis
 * box of centre middle and half extents a. Needs umSolidBodies, umBodyLocal
 * and umQuatRotate (uniform-mixed-solid.wgsl.ts). */
export const uniformBodyTileBoundWGSL=/* wgsl */`
// The two boxes' gap along unit n: centres c apart in the body frame, half
// extents e on the body's axes and a on the axes u0, u1, u2.
fn umBoxGap(n:vec3f,c:vec3f,e:vec3f,a:vec3f,u0:vec3f,u1:vec3f,u2:vec3f)->f32{
 return abs(dot(n,c))-dot(e,abs(n))-(a.x*abs(dot(n,u0))+a.y*abs(dot(n,u1))+a.z*abs(dot(n,u2)));
}
fn umBodyBoxTileGap(i:u32,middle:vec3f,a:vec3f)->f32{
 let b=umSolidBodies[i];let e=0.5*b.dimensions.xyz;let c=umBodyLocal(i,middle);
 let q=vec4f(b.orientation.x,-b.orientation.yzw);
 let u0=umQuatRotate(q,vec3f(1.0,0.0,0.0));let u1=umQuatRotate(q,vec3f(0.0,1.0,0.0));let u2=umQuatRotate(q,vec3f(0.0,0.0,1.0));
 var gap=max(max(umBoxGap(vec3f(1.0,0.0,0.0),c,e,a,u0,u1,u2),umBoxGap(vec3f(0.0,1.0,0.0),c,e,a,u0,u1,u2)),umBoxGap(vec3f(0.0,0.0,1.0),c,e,a,u0,u1,u2));
 gap=max(gap,max(max(umBoxGap(u0,c,e,a,u0,u1,u2),umBoxGap(u1,c,e,a,u0,u1,u2)),umBoxGap(u2,c,e,a,u0,u1,u2)));
 for(var k=0u;k<3u;k++){
  var axis=vec3f(0.0);axis[k]=1.0;
  for(var j=0u;j<3u;j++){
   let v=cross(axis,select(select(u2,u1,j==1u),u0,j==0u));let l=length(v);
   if(l>1e-3){gap=max(gap,umBoxGap(v/l,c,e,a,u0,u1,u2));}
  }
 }
 let p=c-clamp(c,-e,e);let l=length(p);
 if(l>0.0){gap=max(gap,umBoxGap(p/l,c,e,a,u0,u1,u2));}
 return gap;
}
`;

type V=readonly [number,number,number];
const dot=(a:V,b:V)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a:V,b:V):V=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const abs=(a:V):V=>[Math.abs(a[0]),Math.abs(a[1]),Math.abs(a[2])];
/** Rotate v by unit quaternion q = (w, x, y, z), as umQuatRotate. */
export function quatRotate(q:readonly [number,number,number,number],v:V):V{
 const [w,x,y,z]=q,u:V=[x,y,z],t=cross(u,v).map(c=>2*c) as unknown as V;
 const s=cross(u,t);return [v[0]+w*t[0]+s[0],v[1]+w*t[1]+s[1],v[2]+w*t[2]+s[2]];
}
/** The mirror of umBodyBoxTileGap: a box body (centre, orientation (w, x, y, z), full dimensions) against the world-axis box (middle, half extents a). */
export function bodyBoxTileGap(body:{centre:V;orientation:readonly [number,number,number,number];dimensions:V},middle:V,a:V):number{
 const e:V=[body.dimensions[0]/2,body.dimensions[1]/2,body.dimensions[2]/2];
 const q=[body.orientation[0],-body.orientation[1],-body.orientation[2],-body.orientation[3]] as const;
 const c=quatRotate(q,[middle[0]-body.centre[0],middle[1]-body.centre[1],middle[2]-body.centre[2]]);
 const u=[quatRotate(q,[1,0,0]),quatRotate(q,[0,1,0]),quatRotate(q,[0,0,1])] as const;
 const gapAlong=(n:V)=>Math.abs(dot(n,c))-dot(e,abs(n))-(a[0]*Math.abs(dot(n,u[0]))+a[1]*Math.abs(dot(n,u[1]))+a[2]*Math.abs(dot(n,u[2])));
 const axes:V[]=[[1,0,0],[0,1,0],[0,0,1]];
 let gap=-Infinity;
 for(const n of [...axes,...u])gap=Math.max(gap,gapAlong(n));
 for(const k of axes)for(const j of u){const v=cross(k,j),l=Math.hypot(...v);if(l>1e-3)gap=Math.max(gap,gapAlong([v[0]/l,v[1]/l,v[2]/l]));}
 const p:V=[c[0]-Math.min(Math.max(c[0],-e[0]),e[0]),c[1]-Math.min(Math.max(c[1],-e[1]),e[1]),c[2]-Math.min(Math.max(c[2],-e[2]),e[2])],l=Math.hypot(...p);
 if(l>0)gap=Math.max(gap,gapAlong([p[0]/l,p[1]/l,p[2]/l]));
 return gap;
}
/** The box's signed distance in its own frame (scene-shape.ts). */
export function boxDistance(d:V,p:V):number{
 const q=[Math.abs(p[0])-d[0]/2,Math.abs(p[1])-d[1]/2,Math.abs(p[2])-d[2]/2];
 return Math.hypot(Math.max(q[0]!,0),Math.max(q[1]!,0),Math.max(q[2]!,0))+Math.min(Math.max(q[0]!,Math.max(q[1]!,q[2]!)),0);
}
/** markTiles' test of one box body against one tile (low corner, h per axis),
 * old (centre distance less the half diagonal) or new (the larger of that
 * and the box gap): marked when the bound is within the margin, the
 * surface's travel over the horizon plus a cell, after the ball prefilter
 * (a spinning body stays inside its bounding ball, so the ball travels only
 * |v| over the horizon). */
export function bodyMarksTile(body:{centre:V;orientation:readonly [number,number,number,number];dimensions:V;velocity:V;spin:V},low:V,h:V,horizon:number,rule:"old"|"new"):boolean{
 const middle:V=[low[0]+2*h[0],low[1]+2*h[1],low[2]+2*h[2]],half=Math.hypot(2*h[0],2*h[1],2*h[2]),cell=Math.max(...h);
 const radius=Math.hypot(...body.dimensions)/2,high:V=[low[0]+4*h[0],low[1]+4*h[1],low[2]+4*h[2]];
 const nearest=[0,1,2].map(k=>Math.min(Math.max(body.centre[k]!,low[k]!),high[k]!)-body.centre[k]!) as unknown as V;
 if(Math.hypot(...nearest)>radius+Math.hypot(...body.velocity)*horizon+cell)return false;
 const margin=(Math.hypot(...body.velocity)+Math.hypot(...body.spin)*radius)*horizon+cell;
 const q=[body.orientation[0],-body.orientation[1],-body.orientation[2],-body.orientation[3]] as const;
 const local=quatRotate(q,[middle[0]-body.centre[0],middle[1]-body.centre[1],middle[2]-body.centre[2]]);
 const old=boxDistance(body.dimensions,local)-half;
 const bound=rule==="old"?old:Math.max(old,bodyBoxTileGap(body,middle,[2*h[0],2*h[1],2*h[2]]));
 return bound<=margin;
}
