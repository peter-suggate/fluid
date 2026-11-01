import asset from './imported-assets/paddleboats.generated.json';
import type { RigidBodyDescription } from './model';

export function paddleboatSkin(body: RigidBodyDescription) {
  const shape = body.appearance === 'paddleboat-red' ? 6 : body.appearance === 'paddleboat-teal' ? 7 : body.appearance === 'paddleboat-wheel' ? 8 : 0;
  if (!shape) return undefined;
  const half = (shape === 8 ? asset.wheel.extent : asset.hull.extent) as [number, number, number];
  return { shape, half, radius: Math.hypot(...half) };
}
const parts = [...asset.hull.parts, ...asset.wheel.parts];
const nodes = [...asset.hull.nodes, ...asset.wheel.nodes.map(n => ({ ...n, end: n.end + asset.hull.nodes.length, part: n.part < 0 ? -1 : n.part + asset.hull.parts.length }))];
const v = (a: readonly number[]) => `vec${a.length}f(${a.map(x => Number(x).toFixed(8)).join(',')})`;
/** Blender-authored analytic mesh parts, accelerated by their exported BVH.
 * The same live body transform is used for primary, reflection and shadow rays. */
export const paddleboatShaderLibrary = /* wgsl */ `
struct BoatPart { centerKind:vec4f, halfPaint:vec4f, rotation:vec4f, color:vec3f }
struct BoatNode { lo:vec3f, hi:vec3f, part:i32, escape:u32 }
const BOAT_PARTS=array<BoatPart,${parts.length}>(${parts.map(p => `BoatPart(${v([...p.p,p.kind])},${v([...p.d,p.paint?1:0])},${v(p.q)},${v(p.color)})`).join(',')});
const BOAT_NODES=array<BoatNode,${nodes.length}>(${nodes.map(n => `BoatNode(${v(n.lo)},${v(n.hi)},${n.part},${n.end}u)`).join(',')});
struct BoatHit { t:f32, normal:vec3f }
fn boatRotate(q:vec4f,p:vec3f)->vec3f {return p+2.0*cross(q.yzw,cross(q.yzw,p)+q.x*p);}
fn boatInterval(o:vec3f,d:vec3f,lo:vec3f,hi:vec3f)->vec2f {
 var near=-1e20;var far=1e20;
 for(var a=0u;a<3u;a++){if(abs(d[a])<1e-9){if(o[a]<lo[a]||o[a]>hi[a]){return vec2f(1e20,-1e20);}}else{let x=(lo[a]-o[a])/d[a];let y=(hi[a]-o[a])/d[a];near=max(near,min(x,y));far=min(far,max(x,y));}}
 return vec2f(near,far);
}
fn boatPartHit(o:vec3f,d:vec3f,p:BoatPart)->BoatHit {
 let inverse=vec4f(p.rotation.x,-p.rotation.yzw);let ro=boatRotate(inverse,o-p.centerKind.xyz);let rd=boatRotate(inverse,d);let half=p.halfPaint.xyz;
 var t=1e20;var normal=vec3f(0,1,0);let kind=i32(p.centerKind.w);
 if(kind==0){let hit=boatInterval(ro,rd,-half,half);let candidate=select(hit.x,hit.y,hit.x<=1e-5);
  if(hit.x<=hit.y&&candidate>1e-5){t=candidate;let at=ro+rd*t;let scaled=abs(at/half);if(scaled.x>=scaled.y&&scaled.x>=scaled.z){normal=vec3f(sign(at.x),0,0);}else if(scaled.y>=scaled.z){normal=vec3f(0,sign(at.y),0);}else{normal=vec3f(0,0,sign(at.z));}}
 }else if(kind==1){let a=ro/half;let b=rd/half;let aa=dot(b,b);let ab=dot(a,b);let disc=ab*ab-aa*(dot(a,a)-1.0);
  if(disc>=0.0){let root=sqrt(disc);t=(-ab-root)/aa;if(t<=1e-5){t=(-ab+root)/aa;}if(t<=1e-5){t=1e20;}normal=normalize((ro+rd*t)/(half*half));}
 }else{let a=dot(rd.xz,rd.xz);let b=dot(ro.xz,rd.xz);let disc=b*b-a*(dot(ro.xz,ro.xz)-half.x*half.x);
  if(a>1e-12&&disc>=0.0){for(var i=0u;i<2u;i++){let candidate=(-b+select(-sqrt(disc),sqrt(disc),i==1u))/a;let at=ro+candidate*rd;if(candidate>1e-5&&candidate<t&&abs(at.y)<=half.y){t=candidate;normal=normalize(vec3f(at.x,0,at.z));}}}
  if(abs(rd.y)>1e-9){for(var i=0u;i<2u;i++){let side=select(-1.0,1.0,i==1u);let candidate=(side*half.y-ro.y)/rd.y;let at=ro+rd*candidate;if(candidate>1e-5&&candidate<t&&dot(at.xz,at.xz)<=half.x*half.x){t=candidate;normal=vec3f(0,side,0);}}}
 }return BoatHit(t,boatRotate(p.rotation,normal));
}
fn paddleboatHit(o:vec3f,d:vec3f,shape:i32)->BoatHit {
 var best=BoatHit(1e20,vec3f(0,1,0));var i=select(0u,${asset.hull.nodes.length}u,shape==8);let end=select(${asset.hull.nodes.length}u,${nodes.length}u,shape==8);
 loop{if(i>=end){break;}let node=BOAT_NODES[i];let interval=boatInterval(o,d,node.lo,node.hi);
  if(interval.x>interval.y||interval.y<=1e-5||interval.x>best.t){i=node.escape;continue;}
  if(node.part>=0){let hit=boatPartHit(o,d,BOAT_PARTS[u32(node.part)]);if(hit.t<best.t){best=hit;}}i++;
 }return best;
}
fn paddleboatColor(position:vec3f,shape:i32)->vec3f {
 var nearest=1e20;var color=vec3f(.2);var i=select(0u,${asset.hull.nodes.length}u,shape==8);let end=select(${asset.hull.nodes.length}u,${nodes.length}u,shape==8);
 loop{if(i>=end){break;}let node=BOAT_NODES[i];if(any(position<node.lo-vec3f(.002))||any(position>node.hi+vec3f(.002))){i=node.escape;continue;}
  if(node.part>=0){let p=BOAT_PARTS[u32(node.part)];let local=boatRotate(vec4f(p.rotation.x,-p.rotation.yzw),position-p.centerKind.xyz);let h=p.halfPaint.xyz;var distance=0.0;
   if(p.centerKind.w<.5){let q=abs(local)-h;distance=length(max(q,vec3f(0)))+min(max(q.x,max(q.y,q.z)),0.0);}
   else if(p.centerKind.w<1.5){distance=(length(local/h)-1.0)*min(h.x,min(h.y,h.z));}
   else{let q=vec2f(length(local.xz)-h.x,abs(local.y)-h.y);distance=length(max(q,vec2f(0)))+min(max(q.x,q.y),0.0);}
   if(abs(distance)<nearest){nearest=abs(distance);color=p.color;if(p.halfPaint.w>.5&&shape==6){color=vec3f(.48,.055,.036);}}
  }i++;
 }return color;
}
`;
