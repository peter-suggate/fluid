import { PROCEDURAL_SOLID_MAGIC } from "../../core/procedural-solid";

/** The boundary is appended to the mask buffer; ordinary scenes have no
 * footer. No extra binding or per-frame scene callback is required. */
export const uniformSolidBoundaryWGSL = /* wgsl */ `
fn umBoundaryBase()->u32{
 let size=arrayLength(&umSolidScratch);
 if(size<2u||umSolidScratch[size-1u]!=${PROCEDURAL_SOLID_MAGIC}u){return 0u;}
 return umSolidScratch[size-2u];
}
fn umBoundaryRecord(p:vec3i)->u32{
 if(!umSolidValid(p)){return 0u;}
 let base=umBoundaryBase();if(base==0u){return 0u;}
 let b=(p+32*umFramePage())>>vec3u(3);let mask=umSolidScratch[base];
 var slot=((u32(b.x)*73856093u)^(u32(b.y)*19349663u)^(u32(b.z)*83492791u))&mask;
 for(var probe=0u;probe<=mask;probe++){
  let at=base+4u+4u*slot;let r=umSolidScratch[at+3u];if(r==0u){return 0u;}
  if(all(vec3i(bitcast<i32>(umSolidScratch[at]),bitcast<i32>(umSolidScratch[at+1u]),bitcast<i32>(umSolidScratch[at+2u]))==b)){
   let c=vec3u(p)&vec3u(7);let i=c.x+8u*(c.y+8u*c.z);
   if((umSolidScratch[base+r+(i>>5u)]&(1u<<(i&31u)))==0u){return 0u;}
   return base+r+16u;
  }
  slot=(slot+1u)&mask;
 }
 return 0u;
}
// Derivatives and field value; gradient is in lattice coordinates.
fn umBoundarySampleAt(q:vec3f,cell:vec3i,record:u32)->vec4f{
 let c=vec3u(cell)&vec3u(7);let f=q-vec3f(cell);var result=vec4f(0);
 for(var k=0u;k<8u;k++){
  let b=vec3u(k&1u,(k>>1u)&1u,k>>2u);let n=c+b;let w=select(1.0-f,f,b!=vec3u(0));
  let v=bitcast<f32>(umSolidScratch[record+n.x+9u*(n.y+9u*n.z)]);
  let sign=select(vec3f(-1),vec3f(1),b!=vec3u(0));
  result+=v*vec4f(sign.x*w.y*w.z,sign.y*w.x*w.z,sign.z*w.x*w.y,w.x*w.y*w.z);
 }
 return result;
}
fn umBoundarySample(q:vec3f)->vec4f{
 let c=vec3i(floor(q));let r=umBoundaryRecord(c);
 if(r==0u){return vec4f(0,0,0,select(1e6,-1e6,umSolidVoxel(c)));}
 return umBoundarySampleAt(q,c,r);
}
fn umBoundaryBoxOpen(q:vec3f,width:vec3f)->f32{
 var open=0.0;
 for(var k=0u;k<8u;k++){
  let bit=vec3u(k&1u,(k>>1u)&1u,k>>2u);let at=q+select(vec3f(-0.25),vec3f(0.25),bit!=vec3u(0))*width;
  let s=umBoundarySample(at);
  open+=0.125*umTerrainPlaneBoxFraction(s.xyz,s.w,width*0.5);
 }
 return open;
}
fn umBoundaryTouchesBox(q:vec3f,width:vec3f)->bool{
 if(umBoundaryBase()==0u){return false;}
 for(var k=0u;k<8u;k++){
  let b=vec3u(k&1u,(k>>1u)&1u,k>>2u);
  if(umBoundaryRecord(vec3i(floor(q+select(vec3f(-0.25),vec3f(0.25),b!=vec3u(0))*width)))!=0u){return true;}
 }
 return false;
}
`;
