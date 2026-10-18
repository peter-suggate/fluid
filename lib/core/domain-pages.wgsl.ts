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
// The pool block holding lattice page q, or 0 (AIR, SOLID, or a record of another page).
fn domainPageBlock(q:vec3i)->u32 {
  let b=u32(layers.control.w);let edge=viewRecords[b+2u];let world=q+domainFramePage();let p=bitcast<vec3u>(world)&vec3u(edge-1u);
  let block=viewRecords[b+viewRecords[b+3u]+p.x+edge*(p.y+edge*p.z)];
  if(block==0xffffffffu || block==0u || block>=viewRecords[b+1u]-1u){return 0u;}
  let r=b+32u+4u*block;
  let page=bitcast<vec3i>(vec3u(viewRecords[r],viewRecords[r+1u],viewRecords[r+2u]));
  return select(0u,block,all(page==world));
}
// The work tier of lattice region q (16h, half a page a side; the tiers the
// overlay copies at word layers.pageView.z, layers.pageView.w regions): 0
// cold, 1 warm, 2 hot, 3 where no block holds the region.
fn domainHeatTier(q:vec3i)->u32 {
  let base=u32(layers.pageView.z);
  if(base==0u){return 3u;}
  let block=domainPageBlock(q>>vec3u(1u));
  if(block==0u){return 3u;}
  let o=vec3u(q&vec3i(1));
  let region=8u*block+o.x+2u*o.y+4u*o.z;
  if(region>=u32(layers.pageView.w)){return 3u;}
  return min(viewRecords[base+region],2u);
}
fn domainHeatColor(tier:u32)->vec3f {
  return select(select(LP_heatCold,LP_heatWarm,tier==1u),LP_heatHot,tier==2u);
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

/** A box-sized DDA, visiting only the boxes crossed by the ray: pages, or
 * 16h regions (`split` 2, half a page a side). `box` paints lattice box q
 * (DomainBox: no line and no face skips it). Faces have a faint fill and
 * screen-sized edges; front-to-back composition keeps dense allocations
 * legible. This is deliberately an x-ray diagnostic, like slices. */
const domainBoxes3DWGSL = (entry: string, opacity: string, split: number, box: string) => /* wgsl */ `
@fragment fn ${entry}(input:VertexOutput)->@location(0) vec4f {
  let opacity=${opacity};
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
  let boxSize=edge/${split}.0*domainPageCellSize();
  // Bounds are published with the block records, once per pool commit.
  let b=u32(layers.control.w);
  // World pages; the copied directory header's F makes them lattice pages.
  let low=${split}*(bitcast<vec3i>(vec3u(viewRecords[b+20u],viewRecords[b+21u],viewRecords[b+22u]))-domainFramePage());
  let high=${split}*(bitcast<vec3i>(vec3u(viewRecords[b+24u],viewRecords[b+25u],viewRecords[b+26u]))-domainFramePage());
  if(any(high<=low)){discard;}
  let lower=domainPageFrameOrigin()+vec3f(low)*boxSize;
  let pd=vec3u(high-low);
  // Box coordinates make DDA independent of scene units and page dimensions.
  let ro=(eye-lower)/boxSize;let rd=ray/boxSize;
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
  // A ray can cross at most the sum of the box's dimensions.
  for(var i=0u;i<pd.x+pd.y+pd.z+3u;i+=1u){
    if(any(q<vec3i(0)) || any(q>=vec3i(pd)) || t>=end || result.a>0.97){break;}
    let paint=${box}(q+low);
    if(paint.face>0.0 || paint.line>0.0){
      let p=clamp(ro+rd*t-vec3f(q),vec3f(0),vec3f(1));
      let distances=min(p,1.0-p)*boxSize;
      // Smallest distance is the face itself, second-smallest an edge.
      let edgeDistance=max(min(distances.x,distances.y),min(max(distances.x,distances.y),distances.z));
      let pixel=max(t*2.0*cameraTanHalfFov()/max(u.viewport.y,1.0),1e-6);
      let line=1.0-smoothstep(0.65,1.65,edgeDistance/pixel);
      let alpha=mix(paint.face,paint.line,line)*opacity;
      let color=mix(paint.color*0.85,paint.color,line);
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
/** The pages and heat 3D entry points: allocated pages, and the work tier
 * of every region of an allocated page. Cold regions are faintest, so the
 * warm and hot ones read through a mostly settled pool. */
export const domainPages3DWGSL = /* wgsl */ `
struct DomainBox { color:vec3f, face:f32, line:f32 }
fn domainPageBox(q:vec3i)->DomainBox {
  let state=domainPageState(q);
  if(state==0u){return DomainBox(vec3f(0),0.0,0.0);}
  return DomainBox(domainPageColor(state),select(0.055,0.025,state==2u),select(0.85,0.42,state==2u));
}
fn domainHeatBox(q:vec3i)->DomainBox {
  let tier=domainHeatTier(q);
  if(tier>2u){return DomainBox(vec3f(0),0.0,0.0);}
  return DomainBox(domainHeatColor(tier),array<f32,3>(0.012,0.05,0.09)[tier],array<f32,3>(0.22,0.7,0.9)[tier]);
}
${domainBoxes3DWGSL("fragmentPages3D", "layers.pageView.x", 1, "domainPageBox")}
${domainBoxes3DWGSL("fragmentHeat3D", "layers.pageView.y", 2, "domainHeatBox")}
`;
