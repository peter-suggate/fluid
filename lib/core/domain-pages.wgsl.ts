/** Page diagnostics read the published allocation directly: no field sampling,
 * ownership lookup or CPU readback. Shared by the slice and 3D overlay. */
import { uniformWindowWorldWGSL } from "../methods/uniform/uniform-window-address";
import { FRACTION_READOUT_FONT, packFractionReadoutGlyph } from "./fraction-readout.wgsl";

/** The heat layer's resolution labels ("h", "4h", "16h"): the V/K readout's
 * 5×7 digits and an h, glyph code = index. */
const HEAT_LABEL_FONT = [
  ["#....", "#....", "#.##.", "##..#", "#...#", "#...#", "#...#"],
  FRACTION_READOUT_FONT["1"]!, FRACTION_READOUT_FONT["4"]!, FRACTION_READOUT_FONT["6"]!,
];
const hex = (value: number): string => `0x${value.toString(16)}u`;
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
// The resolution region q of tier \`tier\` is computed at: 0 h, 1 4h, 2 16h,
// 3 unknown. COLD is one 16h term a region when frozen (layers.heat.y), WARM
// is 4h when warm regions are computed at 4h (layers.heat.x); otherwise h
// when any of the region's 64 tiles is h in the bound topology (mixedTopology,
// the overlay's binding 17: a word a tile key, bit 31 h).
fn domainHeatResolution(q:vec3i,tier:u32)->u32 {
  if(tier==0u&&layers.heat.y>0.5){return 2u;}
  if(tier==1u&&layers.heat.x>0.5){return 1u;}
  let words=arrayLength(&mixedTopology);
  if(words<=1u){return 3u;}
  let block=domainPageBlock(q>>vec3u(1u));let o=4u*vec3u(q&vec3i(1));
  // An opaque bound (the topology is longer than 64 words when bound): a
  // literal one lets Metal unroll the scan into the overlay's slice sampler.
  for(var k=0u;k<min(words,64u);k+=1u){
    let t=512u*block+o.x+(k&3u)+8u*(o.y+((k>>2u)&3u))+64u*(o.z+(k>>4u));
    if(t<words&&(mixedTopology[t]&0x80000000u)!=0u){return 0u;}
  }
  return 1u;
}
fn domainHeatGlyph(code:u32)->vec2u {
  switch code {
${HEAT_LABEL_FONT.map((rows, code) => { const [low, high] = packFractionReadoutGlyph(rows); return `    case ${code}u: { return vec2u(${hex(low)},${hex(high)}); }`; }).join("\n")}
    default: { return vec2u(0u); }
  }
}
// A resolution's label as glyph codes four bits apiece (x, first lowest) and
// their count (y): h, 4h, 16h; none for an unknown resolution.
fn domainHeatText(resolution:u32)->vec2u {
  if(resolution==0u){return vec2u(0u,1u);}
  if(resolution==1u){return vec2u(2u,2u);}
  if(resolution==2u){return vec2u(1u|(3u<<4u),3u);}
  return vec2u(0u);
}
fn domainHeatCovers(text:vec2u,local:vec2i,scale:i32)->bool {
  if(local.x<0||local.y<0){return false;}
  let font=vec2u(local/scale);let slot=font.x/6u;let column=font.x%6u;
  if(slot>=text.y||column>=5u||font.y>=7u){return false;}
  let rows=domainHeatGlyph((text.x>>(4u*slot))&15u);
  return ((select(rows.x,rows.y,font.y>=4u)>>(5u*(font.y%4u)+4u-column))&1u)!=0u;
}
// The resolution label of region q at a framebuffer pixel (y down): ink (x),
// its one-pixel casing (y) and the label's opacity (z). It sits inside the
// region's top-left corner on screen: the box corner farthest along
// up - right, or on a slice (normal axis >= 0, at \`coordinate\`) the corner
// of the region's rectangle in the plane. A label fades in as the region
// grows from two to three label sizes on screen, so distant regions stay bare;
// a slice's labels cannot pile up behind each other, so they start at 1.25.
fn domainHeatLabel(q:vec3i,tier:u32,pixel:vec2f,normal:i32,coordinate:f32)->vec3f {
  let origin=u.cameraPosition.xyz;let forward=normalize(u.cameraTarget.xyz-origin);
  var right=cross(forward,vec3f(0,1,0));if(length(right)<1e-5){right=vec3f(1,0,0);}right=normalize(right);
  let up=normalize(cross(right,forward));
  let size=0.5*f32(viewRecords[u32(layers.control.w)])*domainPageCellSize();
  let low=domainPageFrameOrigin()+vec3f(q)*size;
  let corner=(up-right)>vec3f(0);
  var a=low+select(vec3f(0),size,corner);var b=low+select(size,vec3f(0),corner);
  if(normal>=0){a[normal]=coordinate;b[normal]=coordinate;}
  let pa=overlayFramebufferPixel(a,origin,forward,right,up);let pb=overlayFramebufferPixel(b,origin,forward,right,up);
  if(pa.z<=0.0||pb.z<=0.0){return vec3f(0);}
  // cameraTarget.w is the presentation's pixels per CSS pixel.
  let scale=max(1,i32(round(u.cameraTarget.w)));
  let room=vec2i(3*6-1,7)*scale;let inset=2.0*f32(scale);
  let extent=abs(pb.xy-pa.xy)/(vec2f(room)+2.0*inset);
  let start=select(2.0,1.25,normal>=0);
  let fade=smoothstep(start,start+1.0,min(extent.x,extent.y));
  let local=vec2i(floor(pixel))-vec2i(round(pa.xy+inset));
  if(fade<=0.0||any(local<vec2i(-1))||any(local>room)){return vec3f(0);}
  let text=domainHeatText(domainHeatResolution(q,tier));
  if(domainHeatCovers(text,local,scale)){return vec3f(1,0,fade);}
  for(var dy=-1;dy<=1;dy+=1){for(var dx=-1;dx<=1;dx+=1){
    if(domainHeatCovers(text,local+vec2i(dx,dy),scale)){return vec3f(0,1,fade);}
  }}
  return vec3f(0);
}
// A label's ink: the tier's colour lifted toward white, over a dark casing.
fn domainHeatInk(tier:u32)->vec3f {return mix(domainHeatColor(tier),vec3f(1),0.45);}
const DOMAIN_HEAT_CASING:vec3f=vec3f(0.035,0.04,0.05);
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
 * 16h regions (`split` 2, half a page a side). `box` paints lattice box q at
 * the fragment's pixel (DomainBox: no line and no face skips it; its label,
 * ink or casing at an opacity, is composited at the box's own depth). Faces have a faint fill and
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
    let paint=${box}(q+low,input.position.xy);
    if(paint.face>0.0 || paint.line>0.0){
      let p=clamp(ro+rd*t-vec3f(q),vec3f(0),vec3f(1));
      let distances=min(p,1.0-p)*boxSize;
      // Smallest distance is the face itself, second-smallest an edge.
      let edgeDistance=max(min(distances.x,distances.y),min(max(distances.x,distances.y),distances.z));
      let pixel=max(t*2.0*cameraTanHalfFov()/max(u.viewport.y,1.0),1e-6);
      let line=1.0-smoothstep(0.65,1.65,edgeDistance/pixel);
      let mark=max(paint.label.x,paint.label.y)*paint.label.z;
      let alpha=max(mix(paint.face,paint.line,line),mark)*opacity;
      let color=mix(mix(paint.color*0.85,paint.color,line),select(DOMAIN_HEAT_CASING,paint.ink,paint.label.x>0.5),mark);
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
 * of every region of an allocated page with its resolution label. Cold
 * regions are faintest, so the warm and hot ones read through a mostly
 * settled pool. */
export const domainPages3DWGSL = /* wgsl */ `
struct DomainBox { color:vec3f, face:f32, line:f32, label:vec3f, ink:vec3f }
fn domainPageBox(q:vec3i,pixel:vec2f)->DomainBox {
  let state=domainPageState(q);
  if(state==0u){return DomainBox(vec3f(0),0.0,0.0,vec3f(0),vec3f(0));}
  return DomainBox(domainPageColor(state),select(0.055,0.025,state==2u),select(0.85,0.42,state==2u),vec3f(0),vec3f(0));
}
// Cold labels at half strength, like their boxes.
fn domainHeatBox(q:vec3i,pixel:vec2f)->DomainBox {
  let tier=domainHeatTier(q);
  if(tier>2u){return DomainBox(vec3f(0),0.0,0.0,vec3f(0),vec3f(0));}
  let label=domainHeatLabel(q,tier,pixel,-1,0.0)*vec3f(1,1,select(1.0,0.5,tier==0u));
  return DomainBox(domainHeatColor(tier),array<f32,3>(0.012,0.05,0.09)[tier],array<f32,3>(0.22,0.7,0.9)[tier],label,domainHeatInk(tier));
}
${domainBoxes3DWGSL("fragmentPages3D", "layers.pageView.x", 1, "domainPageBox")}
${domainBoxes3DWGSL("fragmentHeat3D", "layers.pageView.y", 2, "domainHeatBox")}
`;
