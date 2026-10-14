/** Page diagnostics read the published allocation directly: no field sampling,
 * ownership lookup or CPU readback. Shared by the slice and 3D overlay. */
export const domainPagesWGSL = /* wgsl */ `
fn domainPagesValid()->bool {
  let base=u32(layers.control.w);
  if(base==0u || arrayLength(&viewRecords)<base+8u){return false;}
  let pd=vec3u(viewRecords[base+1u],viewRecords[base+2u],viewRecords[base+3u]);
  return viewRecords[base]>0u && all(pd>vec3u(0)) && arrayLength(&viewRecords)>=base+8u+pd.x*pd.y*pd.z;
}
fn domainPageDimensions()->vec3u {
  let base=u32(layers.control.w);
  return vec3u(viewRecords[base+1u],viewRecords[base+2u],viewRecords[base+3u]);
}
fn domainPageOrigin()->vec3i {
  let base=u32(layers.control.w);
  return bitcast<vec3i>(vec3u(viewRecords[base+5u],viewRecords[base+6u],viewRecords[base+7u]));
}
fn domainPageState(q:vec3i)->u32 {
  let pd=domainPageDimensions();
  if(any(q<vec3i(0)) || any(q>=vec3i(pd))){return 0u;}
  return viewRecords[u32(layers.control.w)+8u+u32(q.x)+pd.x*(u32(q.y)+pd.y*u32(q.z))];
}
fn domainPageColor(state:u32)->vec3f {
  return select(vec3f(0.702,0.608,0.918),vec3f(0.36,0.38,0.42),state==2u);
}
// The page containing a fine cell; floor matters for signed page coordinates.
fn domainPageAt(cell:vec3i)->u32 {
  let edge=f32(viewRecords[u32(layers.control.w)]);
  return domainPageState(vec3i(floor(vec3f(cell)/edge))-domainPageOrigin());
}
`;

/** A page-sized DDA, visiting only the pages crossed by the ray. Faces have a
 * faint fill and screen-sized edges; front-to-back composition keeps dense
 * allocations legible. This is deliberately an x-ray diagnostic, like slices. */
export const domainPages3DWGSL = /* wgsl */ `
@fragment fn fragmentPages3D(input:VertexOutput)->@location(0) vec4f {
  let opacity=layers.pageView.x;
  if(opacity<=0.0 || u.gridInfo.w<=0.5 || !domainPagesValid()){discard;}
  let eye=u.cameraPosition.xyz;
  let forward=normalize(u.cameraTarget.xyz-eye);
  var right=cross(forward,vec3f(0,1,0));
  if(length(right)<1e-5){right=vec3f(1,0,0);}
  right=normalize(right);
  let up=normalize(cross(right,forward));
  let ndc=input.uv*2.0-1.0;
  let ray=normalize(forward+right*ndc.x*u.viewport.x/max(u.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());
  let edge=f32(viewRecords[u32(layers.control.w)]);
  let pageSize=edge*u.container.xyz/max(u.gridInfo.xyz,vec3f(1));
  let lower=vec3f(-0.5*u.container.x,0,-0.5*u.container.z)+vec3f(domainPageOrigin())*pageSize;
  let pd=domainPageDimensions();
  // Page coordinates make DDA independent of scene units and page dimensions.
  let ro=(eye-lower)/pageSize;let rd=ray/pageSize;
  // Keep parallel rays finite, including cameras exactly on a box face.
  let inv=1.0/select(select(vec3f(-1e-20),vec3f(1e-20),rd>=vec3f(0)),rd,abs(rd)>vec3f(1e-20));
  let near=min(-ro*inv,(vec3f(pd)-ro)*inv);
  let far=max(-ro*inv,(vec3f(pd)-ro)*inv);
  var t=max(max(near.x,max(near.y,near.z)),0.0);
  let end=min(far.x,min(far.y,far.z));
  if(end<=t){discard;}
  var q=vec3i(clamp(floor(ro+rd*(t+1e-5)),vec3f(0),vec3f(pd)-1.0));
  let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0));
  var next=(vec3f(q)+select(vec3f(0),vec3f(1),rd>=vec3f(0))-ro)*inv;
  let delta=abs(inv);
  var result=vec4f(0);
  // A ray can cross at most the sum of the box's page dimensions.
  for(var i=0u;i<pd.x+pd.y+pd.z+3u;i+=1u){
    if(any(q<vec3i(0)) || any(q>=vec3i(pd)) || t>=end || result.a>0.97){break;}
    let state=domainPageState(q);
    if(state!=0u){
      let p=clamp(ro+rd*t-vec3f(q),vec3f(0),vec3f(1));
      let distances=min(p,1.0-p)*pageSize;
      // Smallest distance is the face itself, second-smallest an edge.
      let edgeDistance=max(min(distances.x,distances.y),min(max(distances.x,distances.y),distances.z));
      let pixel=max(t*2.0*cameraTanHalfFov()/max(u.viewport.y,1.0),1e-6);
      let line=1.0-smoothstep(0.65,1.65,edgeDistance/pixel);
      let alpha=mix(select(0.055,0.025,state==2u),select(0.85,0.42,state==2u),line)*opacity;
      let color=mix(domainPageColor(state)*0.85,domainPageColor(state),line);
      let contribution=(1.0-result.a)*alpha;
      result+=vec4f(color*contribution,contribution);
    }
    let crossing=min(next.x,min(next.y,next.z));
    let crossed=next<=vec3f(crossing+1e-6);
    q+=select(vec3i(0),step,crossed);
    next+=select(vec3f(0),delta,crossed);
    t=crossing;
  }
  if(result.a<=0.001){discard;}
  return vec4f(result.rgb/result.a,result.a);
}
`;
