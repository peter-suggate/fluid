/** Page diagnostics read the published allocation directly: no field sampling,
 * ownership lookup or CPU readback. Shared by the slice and 3D overlay. */
import { uniformWindowWorldWGSL } from "../methods/uniform/uniform-window-address";
export const domainPagesWGSL = /* wgsl */ `
${uniformWindowWorldWGSL(i=>`viewRecords[u32(layers.control.w)+viewRecords[u32(layers.control.w)+3u]+viewRecords[u32(layers.control.w)+2u]*viewRecords[u32(layers.control.w)+2u]*viewRecords[u32(layers.control.w)+2u]+${i}]`,"domain")}
fn domainPagesValid()->bool {
  let base=u32(layers.control.w);
  if(base==0u || arrayLength(&viewRecords)<base+32u){return false;}
  let blocks=viewRecords[base+1u];let edge=viewRecords[base+2u];
  return viewRecords[base+6u]==2u && blocks>2u && edge>0u &&
    arrayLength(&viewRecords)>=base+32u+4u*blocks+edge*edge*edge+32u;
}
fn domainPageFrameOrigin()->vec3f {
  let b=u32(layers.control.w);
  if(bitcast<f32>(viewRecords[b+12u])>0.0){return bitcast<vec3f>(vec3u(viewRecords[b+16u],viewRecords[b+17u],viewRecords[b+18u]));}
  return vec3f(-0.5*u.container.x,0,-0.5*u.container.z);
}
fn domainPageCellSize()->vec3f {
  let b=u32(layers.control.w);
  if(bitcast<f32>(viewRecords[b+12u])>0.0){return bitcast<vec3f>(vec3u(viewRecords[b+12u],viewRecords[b+13u],viewRecords[b+14u]));}
  return u.container.xyz/max(u.gridInfo.xyz,vec3f(1));
}
fn domainPageState(q:vec3i)->u32 {
  let b=u32(layers.control.w);let edge=viewRecords[b+2u];let world=q+domainFramePage();let p=bitcast<vec3u>(world)&vec3u(edge-1u);
  if(domainPolicy()!=0u&&domainWorldSolid(world)){return 2u;}
  let block=viewRecords[b+viewRecords[b+3u]+p.x+edge*(p.y+edge*p.z)];
  if(block==0xffffffffu){return select(0u,2u,domainPolicy()==0u);}
  if(block==0u || block>=viewRecords[b+1u]-1u){return 0u;}
  let r=b+32u+4u*block;
  let page=bitcast<vec3i>(vec3u(viewRecords[r],viewRecords[r+1u],viewRecords[r+2u]));
  return select(0u,viewRecords[r+3u],all(page==world));
}
fn domainPageColor(state:u32)->vec3f {
  return select(vec3f(0.702,0.608,0.918),vec3f(0.36,0.38,0.42),state==2u);
}
// The page containing a fine cell; floor matters for signed page coordinates.
fn domainPageAt(cell:vec3i)->u32 {
  let edge=f32(viewRecords[u32(layers.control.w)]);
  return domainPageState(vec3i(floor(vec3f(cell)/edge)));
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
  let pageSize=edge*domainPageCellSize();
  // Bounds are published with the block records, once per pool commit.
  let b=u32(layers.control.w);
  // World pages; the copied directory header's F makes them lattice pages.
  let low=bitcast<vec3i>(vec3u(viewRecords[b+20u],viewRecords[b+21u],viewRecords[b+22u]))-domainFramePage();
  let high=bitcast<vec3i>(vec3u(viewRecords[b+24u],viewRecords[b+25u],viewRecords[b+26u]))-domainFramePage();
  if(any(high<=low)){discard;}
  let lower=domainPageFrameOrigin()+vec3f(low)*pageSize;
  let pd=vec3u(high-low);
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
    let state=domainPageState(q+low);
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
