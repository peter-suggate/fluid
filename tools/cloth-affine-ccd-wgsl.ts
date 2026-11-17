/** M1 shadow experiment only. Affine cloth motion, no live filter integration.
 * Same endpoint support-plane construction as the CPU diagnostic, with an
 * inflated f32 guard. Unresolved/budgeted intervals return a safe-side prefix.
 * This is not yet a formally rounded predicate or a moving-solid certificate.
 */
export function clothAffineCCDShadowWGSL(workgroupSize:32|64|128,sameTimeEndpoints=false,dynamicCounts=false,projectedSupports=false,advancement=false,currentBounds=false):string {
  return /* wgsl */`
struct Params { count:u32, kind:u32, thickness:f32, budget:u32 }
@group(0) @binding(0) var<storage,read> positions:array<vec4f>;
@group(0) @binding(1) var<storage,read> directions:array<vec4f>;
@group(0) @binding(2) var<storage,read> pairs:array<vec4u>;
@group(0) @binding(3) var<storage,read_write> results:array<vec4f>;
@group(0) @binding(4) var<uniform> P:Params;
${dynamicCounts?'@group(0) @binding(5) var<storage,read_write> counts:array<atomic<u32>>;':''}
fn segmentPoint(p:vec3f,a:vec3f,b:vec3f)->vec3f {
  let e=b-a;let ee=dot(e,e);if(ee<=0.0){return a;}
  return a+e*clamp(dot(p-a,e)/ee,0.0,1.0);
}
fn trianglePoint(p:vec3f,a:vec3f,b:vec3f,c:vec3f)->vec3f {
  let ab=b-a;let ac=c-a;let n=cross(ab,ac);let nn=dot(n,n);
  if(nn>0.0){let q=p-n*(dot(p-a,n)/nn);
    if(dot(cross(ab,q-a),n)>=0.0 && dot(cross(c-b,q-b),n)>=0.0 && dot(cross(a-c,q-c),n)>=0.0){return q;}}
  let q0=segmentPoint(p,a,b);let q1=segmentPoint(p,b,c);let q2=segmentPoint(p,c,a);
  var q=q0;if(length(p-q1)<length(p-q)){q=q1;}if(length(p-q2)<length(p-q)){q=q2;}return q;
}
fn axisSeparates(axis:vec3f,x:array<vec3f,4>,d:array<vec3f,4>,lo:f32,hi:f32,guard:f32)->bool {
  let len=length(axis);if(!(len>0.0 && len<3.4e38)){return false;}let n=axis/len;
  ${projectedSupports?/* wgsl */`
  let p=vec4f(dot(n,x[0]),dot(n,x[1]),dot(n,x[2]),dot(n,x[3]));
  let w=vec4f(dot(n,d[0]),dot(n,d[1]),dot(n,d[2]),dot(n,d[3]));
  let l=p+lo*w;let h=p+hi*w;if(!all(abs(l)<vec4f(3.4e38))||!all(abs(h)<vec4f(3.4e38))){return false;}
  var alo:vec2f;var ahi:vec2f;var blo:vec2f;var bhi:vec2f;
  if(P.kind==0u){alo=vec2f(l.x,h.x);ahi=alo;blo=vec2f(min(l.y,min(l.z,l.w)),min(h.y,min(h.z,h.w)));bhi=vec2f(max(l.y,max(l.z,l.w)),max(h.y,max(h.z,h.w)));}
  else{alo=vec2f(min(l.x,l.y),min(h.x,h.y));ahi=vec2f(max(l.x,l.y),max(h.x,h.y));blo=vec2f(min(l.z,l.w),min(h.z,h.w));bhi=vec2f(max(l.z,l.w),max(h.z,h.w));}
  ${sameTimeEndpoints?'let ab=blo-ahi;let ba=alo-bhi;return max(min(ab.x,ab.y),min(ba.x,ba.y))>P.thickness+guard;':'return max(min(blo.x,blo.y)-max(ahi.x,ahi.y),min(alo.x,alo.y)-max(bhi.x,bhi.y))>P.thickness+guard;'}
  `:/* wgsl */`
  ${sameTimeEndpoints?'var alo=vec2f(3.4e38);var ahi=vec2f(-3.4e38);var blo=vec2f(3.4e38);var bhi=vec2f(-3.4e38);':'var alo=3.4e38;var ahi=-3.4e38;var blo=3.4e38;var bhi=-3.4e38;'}
  for(var i=0u;i<4u;i++){
    let l=dot(n,x[i]+lo*d[i]);let h=dot(n,x[i]+hi*d[i]);
    if(!(abs(l)<3.4e38 && abs(h)<3.4e38)){return false;}
    let isA=select(i<2u,i==0u,P.kind==0u);
    ${sameTimeEndpoints?'if(isA){alo=min(alo,vec2f(l,h));ahi=max(ahi,vec2f(l,h));}else{blo=min(blo,vec2f(l,h));bhi=max(bhi,vec2f(l,h));}':'if(isA){alo=min(alo,min(l,h));ahi=max(ahi,max(l,h));}else {blo=min(blo,min(l,h));bhi=max(bhi,max(l,h));}'}
  }
  ${sameTimeEndpoints?'let ab=blo-ahi;let ba=alo-bhi;return max(min(ab.x,ab.y),min(ba.x,ba.y))>P.thickness+guard;':'return max(blo-ahi,alo-bhi)>P.thickness+guard;'}
  `}
}
fn separated(x:array<vec3f,4>,d:array<vec3f,4>,lo:f32,hi:f32,guard:f32)->bool {
  if(axisSeparates(vec3f(1,0,0),x,d,lo,hi,guard)||axisSeparates(vec3f(0,1,0),x,d,lo,hi,guard)||axisSeparates(vec3f(0,0,1),x,d,lo,hi,guard)){return true;}
  let mid=0.5*(lo+hi);var q:array<vec3f,4>;for(var i=0u;i<4u;i++){q[i]=x[i]+mid*d[i];}
  if(P.kind==0u){
    return axisSeparates(trianglePoint(q[0],q[1],q[2],q[3])-q[0],x,d,lo,hi,guard)||axisSeparates(cross(q[2]-q[1],q[3]-q[1]),x,d,lo,hi,guard);
  }
  if(axisSeparates(cross(q[1]-q[0],q[3]-q[2]),x,d,lo,hi,guard)){return true;}
  for(var i=0u;i<2u;i++){
    if(axisSeparates(segmentPoint(q[i],q[2],q[3])-q[i],x,d,lo,hi,guard)||axisSeparates(q[i+2u]-segmentPoint(q[i+2u],q[0],q[1]),x,d,lo,hi,guard)){return true;}
  }return false;
}
${advancement?/* wgsl */`
fn axisAdvance(axis:vec3f,x:array<vec3f,4>,d:array<vec3f,4>,alpha:f32,guard:f32)->f32 {
  let len=length(axis);if(!(len>0.0&&len<3.4e38)){return alpha;}let n=axis/len;
  let p=vec4f(dot(n,x[0]),dot(n,x[1]),dot(n,x[2]),dot(n,x[3]));let w=vec4f(dot(n,d[0]),dot(n,d[1]),dot(n,d[2]),dot(n,d[3]));
  let split=select(2u,1u,P.kind==0u);let margin=P.thickness+guard;var best=alpha;
  for(var side=0u;side<2u;side++){
    let sign=select(1.0,-1.0,side==1u);var end=1.0;var valid=true;
    for(var i=0u;i<split;i++){for(var j=split;j<4u;j++){
      let gap=sign*(p[j]-p[i]);let slope=sign*(w[j]-w[i]);
      if(!(gap+alpha*slope>margin)){valid=false;}
      if(slope<0.0){let root=(margin-gap)/slope;if(!(abs(root)<3.4e38)){valid=false;}end=min(end,root);}
    }}
    if(valid&&end>alpha){let candidate=select(alpha+0.9*(end-alpha),1.0,end>=1.0);
      // The retreat is NOT a proof: recheck the entire accepted interval.
      if(candidate>best&&axisSeparates(axis,x,d,alpha,candidate,guard)){best=candidate;}
    }
  }return best;
}
fn advanceAxes(x:array<vec3f,4>,d:array<vec3f,4>,alpha:f32,guard:f32)->f32 {
  var best=alpha;best=max(best,axisAdvance(vec3f(1,0,0),x,d,alpha,guard));best=max(best,axisAdvance(vec3f(0,1,0),x,d,alpha,guard));best=max(best,axisAdvance(vec3f(0,0,1),x,d,alpha,guard));
  var q:array<vec3f,4>;for(var i=0u;i<4u;i++){q[i]=x[i]+alpha*d[i];}
  if(P.kind==0u){best=max(best,axisAdvance(trianglePoint(q[0],q[1],q[2],q[3])-q[0],x,d,alpha,guard));best=max(best,axisAdvance(cross(q[2]-q[1],q[3]-q[1]),x,d,alpha,guard));}
  else{best=max(best,axisAdvance(cross(q[1]-q[0],q[3]-q[2]),x,d,alpha,guard));for(var i=0u;i<2u;i++){best=max(best,axisAdvance(segmentPoint(q[i],q[2],q[3])-q[i],x,d,alpha,guard));best=max(best,axisAdvance(q[i+2u]-segmentPoint(q[i+2u],q[0],q[1]),x,d,alpha,guard));}}
  return best;
}
`:''}
fn evaluate(pair:u32)->vec4f {
  let ids=pairs[pair];
  let origin=positions[ids.x].xyz;let velocity=directions[ids.x].xyz;
  var x:array<vec3f,4>;var d:array<vec3f,4>;var scale=P.thickness;
  for(var i=0u;i<4u;i++){
    x[i]=positions[ids[i]].xyz-origin;d[i]=directions[ids[i]].xyz-velocity;
    if(!all(abs(x[i])<vec3f(3.4e38))||!all(abs(d[i])<vec3f(3.4e38))){return vec4f(0,0,3,0);}
    scale=max(scale,max(max(abs(x[i].x),max(abs(x[i].y),abs(x[i].z))),max(abs(d[i].x),max(abs(d[i].y),abs(d[i].z)))));
  }
  let guard=128.0*1.1920928955078125e-7*scale+1e-30;
  ${currentBounds?/* wgsl */`
  var loA=min(x[0],x[0]+d[0]);var hiA=max(x[0],x[0]+d[0]);
  var loB=min(x[2],x[2]+d[2]);var hiB=max(x[2],x[2]+d[2]);
  loB=min(loB,min(x[3],x[3]+d[3]));hiB=max(hiB,max(x[3],x[3]+d[3]));
  if(P.kind==0u){loB=min(loB,min(x[1],x[1]+d[1]));hiB=max(hiB,max(x[1],x[1]+d[1]));}
  else{loA=min(loA,min(x[1],x[1]+d[1]));hiA=max(hiA,max(x[1],x[1]+d[1]));}
  let margin=vec3f(2.0*(P.thickness+guard));
  if(any(loA-hiB>margin)||any(loB-hiA>margin)){return vec4f(1,0,0,guard);}
  `:''}
  ${advancement?/* wgsl */`
  var alpha=0.0;
  for(var step=0u;step<16u;step++){
    if(separated(x,d,alpha,1.0,guard)){return vec4f(1,f32(step+1u),0,guard);}
    let next=advanceAxes(x,d,alpha,guard);
    if(next>=1.0){return vec4f(1,f32(step+1u),0,guard);}
    if(!(next-alpha>9.5367431640625e-7)){return vec4f(alpha,f32(step+1u),1,guard);}
    alpha=next;
  }
  return vec4f(alpha,16,2,guard);
`:/* wgsl */`
  // Depth-first dyadic traversal without a per-lane stack: index's trailing
  // ones identify completed right subtrees. Never cross an unresolved gap.
  var depth=0u;var index=0u;var alpha=0.0;
  for(var count=0u;count<P.budget;count++){
    let denom=f32(1u<<depth);let lo=f32(index)/denom;let hi=f32(index+1u)/denom;
    if(separated(x,d,lo,hi,guard)){
      alpha=hi;
      loop {if(depth==0u || (index&1u)==0u){break;}index=index>>1u;depth--;}
      if(depth==0u){return vec4f(1,f32(count+1u),0,guard);}
      index++;
    }else{
      if(depth>=20u){return vec4f(alpha,f32(count+1u),1,guard);}
      depth++;index=index<<1u;
    }
  }
  return vec4f(alpha,f32(P.budget),2,guard);
`}
}
${dynamicCounts?/* wgsl */`
var<workgroup> minima:array<f32,${workgroupSize}>;
@compute @workgroup_size(${workgroupSize})fn main(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
  var alpha=1.0;
  if(gid.x<min(P.count,atomicLoad(&counts[0])) && atomicLoad(&counts[1])==0u){let r=evaluate(gid.x);results[gid.x]=r;alpha=r.x;}
  minima[lane]=alpha;workgroupBarrier();
  for(var stride=${workgroupSize/2}u;stride>0u;stride=stride>>1u){if(lane<stride){minima[lane]=min(minima[lane],minima[lane+stride]);}workgroupBarrier();}
  if(lane==0u){atomicMin(&counts[5],bitcast<u32>(minima[0]));}
}`:/* wgsl */`
@compute @workgroup_size(${workgroupSize})fn main(@builtin(global_invocation_id) gid:vec3u){if(gid.x<P.count){results[gid.x]=evaluate(gid.x);}}
`}
`;
}
