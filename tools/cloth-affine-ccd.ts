/** Off-live shadow oracle for affine VT/EE motion. This does not certify OGC's
 * projected trial path or a rotating body's curved path. Inputs are the exact
 * JS numbers supplied by the caller (uploaded f32 values should be widened).
 *
 * Closest features only propose separating axes. Acceptance always checks a
 * supporting plane against EVERY primitive vertex at BOTH interval endpoints.
 * Convexity then covers the whole affine interval, including feature changes.
 * This numerical prototype is not a replacement for the production detector.
 */
import {prepareExactAffinePlaneCheck} from './cloth-exact-plane';
export type Vec3 = readonly [number, number, number];
export interface AffineVertex { readonly x: Vec3; readonly d: Vec3 }
export interface AffineCCDResult {
  /** Numerically checked prefix; never jumps over an unresolved interval.
   * The roundoff guard is not a formal interval-arithmetic proof. */
  alpha: number;
  intervals: number;
  reason: 'separated' | 'unresolved' | 'budget' | 'nonfinite';
}
const add = (a: Vec3, b: Vec3): Vec3 => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const mul = (a: Vec3, s: number): Vec3 => [a[0]*s, a[1]*s, a[2]*s];
const dot = (a: Vec3, b: Vec3) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm = (a: Vec3) => Math.hypot(...a);
const clamp = (x: number) => Math.max(0, Math.min(1, x));
const at = (p: AffineVertex, t: number) => add(p.x, mul(p.d, t));

function segmentPoint(p: Vec3, a: Vec3, b: Vec3): Vec3 {
  const e=sub(b,a), ee=dot(e,e);
  return ee > 0 ? add(a,mul(e,clamp(dot(sub(p,a),e)/ee))) : a;
}

/** Degenerate triangles fall back to their boundary. Imprecise feature choices
 * can only lose an axis: support checks below still reject invalid witnesses. */
function trianglePoint(p: Vec3, a: Vec3, b: Vec3, c: Vec3): Vec3 {
  const ab=sub(b,a), ac=sub(c,a), ap=sub(p,a), n=cross(ab,ac), nn=dot(n,n);
  if(nn > 0 && Number.isFinite(nn)) {
    const q=sub(p,mul(n,dot(ap,n)/nn));
    if(dot(cross(ab,sub(q,a)),n)>=0 && dot(cross(sub(c,b),sub(q,b)),n)>=0 && dot(cross(sub(a,c),sub(q,c)),n)>=0) return q;
  }
  const choices=[segmentPoint(p,a,b),segmentPoint(p,b,c),segmentPoint(p,c,a)];
  return choices.reduce((best,q)=>norm(sub(p,q))<norm(sub(p,best))?q:best);
}

function axes(a: readonly Vec3[], b: readonly Vec3[]): Vec3[] {
  const out:Vec3[]=[[1,0,0],[0,1,0],[0,0,1]];
  if(a.length===1) {
    out.push(sub(trianglePoint(a[0]!,b[0]!,b[1]!,b[2]!),a[0]!));
    out.push(cross(sub(b[1]!,b[0]!),sub(b[2]!,b[0]!)));
  } else {
    const u=sub(a[1]!,a[0]!),v=sub(b[1]!,b[0]!);
    out.push(cross(u,v));
    for(const p of a) out.push(sub(segmentPoint(p,b[0]!,b[1]!),p));
    for(const p of b) out.push(sub(p,segmentPoint(p,a[0]!,a[1]!)));
  }
  return out;
}

export interface AffineCCDOptions { thickness: number; maxDepth?: number; maxIntervals?: number; sameTimeEndpoints?: boolean; exactEndpoints?: boolean; maxAlpha?: number }

/** A strict supporting-plane test with a deliberately inflated roundoff guard.
 * No distance-root accuracy assumption is used to accept a path. Exhaustion is
 * fail-closed; this is a numerical prefix search, not an exact time of impact. */
function prefix(aInput: readonly AffineVertex[], bInput: readonly AffineVertex[], options: AffineCCDOptions): AffineCCDResult {
  const {thickness,maxDepth=24,maxIntervals=4096,maxAlpha=1}=options;
  if(!Number.isFinite(thickness)||thickness<0||!Number.isInteger(maxDepth)||maxDepth<0||maxDepth>48||!Number.isInteger(maxIntervals)||maxIntervals<1||!Number.isFinite(maxAlpha)||maxAlpha<0||maxAlpha>1) throw new Error('Invalid affine CCD options');
  if([...aInput,...bInput].some(p=>[...p.x,...p.d].some(x=>!Number.isFinite(x)))) return {alpha:0,intervals:0,reason:'nonfinite'};
  const exactCheck=options.exactEndpoints?prepareExactAffinePlaneCheck(aInput,bInput,thickness,options.sameTimeEndpoints):undefined;
  // Remove common translation before interpolation, preserving relative motion.
  const origin=aInput[0]!.x, velocity=aInput[0]!.d;
  const relative=(p:AffineVertex):AffineVertex=>({x:sub(p.x,origin),d:sub(p.d,velocity)});
  const a=aInput.map(relative),b=bInput.map(relative),all=[...a,...b];
  const scale=Math.max(thickness,...all.flatMap(p=>[...p.x.map(Math.abs),...p.d.map(Math.abs)]));
  if(!Number.isFinite(scale)) return {alpha:0,intervals:0,reason:'nonfinite'};
  const guard=128*Number.EPSILON*scale+Number.MIN_VALUE*128;
  const separated=(lo:number,hi:number):boolean=>{
    const mid=(lo+hi)/2, am=a.map(p=>at(p,mid)),bm=b.map(p=>at(p,mid));
    const ae=a.flatMap(p=>[at(p,lo),at(p,hi)]),be=b.flatMap(p=>[at(p,lo),at(p,hi)]);
    for(const axis of axes(am,bm)) {
      const length=norm(axis); if(!(length>0)||!Number.isFinite(length)) continue;
      const n=mul(axis,1/length),av=ae.map(p=>dot(n,p)),bv=be.map(p=>dot(n,p));
      if([...av,...bv].some(x=>!Number.isFinite(x))) continue;
      let gap=Math.max(Math.min(...bv)-Math.max(...av),Math.min(...av)-Math.max(...bv));
      if(options.sameTimeEndpoints){
        const al=av.filter((_,i)=>i%2===0),ah=av.filter((_,i)=>i%2===1),bl=bv.filter((_,i)=>i%2===0),bh=bv.filter((_,i)=>i%2===1);
        // Every oriented cross-primitive projection difference is affine.
        // The SAME orientation must separate both time endpoints.
        gap=Math.max(Math.min(Math.min(...bl)-Math.max(...al),Math.min(...bh)-Math.max(...ah)),Math.min(Math.min(...al)-Math.max(...bl),Math.min(...ah)-Math.max(...bh)));
      }
      if(gap>thickness+guard && (!exactCheck||exactCheck(axis,lo,hi))) return true;
    }
    return false;
  };
  let intervals=0,alpha=0;
  const stack:[number,number,number][]=[[0,maxAlpha,0]];
  while(stack.length) {
    if(intervals>=maxIntervals) return {alpha,intervals,reason:'budget'};
    const [lo,hi,depth]=stack.pop()!; intervals++;
    if(separated(lo,hi)) {alpha=hi;continue;}
    if(depth>=maxDepth) return {alpha,intervals,reason:'unresolved'};
    const mid=(lo+hi)/2;
    stack.push([mid,hi,depth+1],[lo,mid,depth+1]);
  }
  return {alpha:maxAlpha,intervals,reason:'separated'};
}

export function affineVertexTriangleCCD(v:AffineVertex,a:AffineVertex,b:AffineVertex,c:AffineVertex,options:AffineCCDOptions):AffineCCDResult {
  return prefix([v],[a,b,c],options);
}
export function affineEdgeEdgeCCD(a:AffineVertex,b:AffineVertex,c:AffineVertex,d:AffineVertex,options:AffineCCDOptions):AffineCCDResult {
  return prefix([a,b],[c,d],options);
}
