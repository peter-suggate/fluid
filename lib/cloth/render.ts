import type { ClothDocument, ClothSurfacePublication, ClothVector } from './types';
import { cameraApertureShaderLibrary } from '../core/webgpu-camera';
import { SVO_DRY_SCENE_REVERSED_Z_NEAR_M } from '../svo/features/shading/program';
import { clothRodCapsuleGeometry,clothRodBoundingRadius,clothRodEllipsoidAxes } from './rod-geometry';
import { clothVertexColor } from './appearance';

/** Draws the published mechanical surface into the opaque scene consumed by
 * water optics. Alpha is ray distance in metres, not opacity. Rod fragments
 * intersect the exact capsule used by hit testing rather than a ribbon. */
export const clothSurfaceWGSL = /* wgsl */ `
struct View { viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, cameraBasisPadding:array<vec4f,23>, cameraBasisRight:vec4f }
struct Record { a:vec4f,b:vec4f,color:vec4f }
@group(0) @binding(0) var<uniform> u:View;
@group(0) @binding(1) var<storage,read> records:array<Record>;
@group(0) @binding(2) var<storage,read> clipPlanes:array<vec4f>;
${cameraApertureShaderLibrary('u')}
const NEAR:f32=${SVO_DRY_SCENE_REVERSED_Z_NEAR_M};
fn forward()->vec3f{return normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);}
fn project(p:vec3f)->vec4f{
 let f=forward();let r=cameraRightDirection(f);let up=cross(r,f);let q=p-u.cameraPosition.xyz;
 let d=dot(q,f);let aperture=cameraTanHalfFov();
 return vec4f(dot(q,r)/(u.viewport.x/u.viewport.y*aperture),dot(q,up)/aperture,NEAR,d);
}
struct Out { @builtin(position) clip:vec4f,@location(0) world:vec3f,@location(1) normal:vec3f,@location(2) color:vec3f,@location(3) @interpolate(flat) index:u32,@location(4) @interpolate(flat) lighting:f32 }
@vertex fn shellVertex(@builtin(vertex_index) i:u32)->Out {
 let v=records[i];var o:Out;o.clip=project(v.a.xyz);o.world=v.a.xyz;o.normal=v.b.xyz;o.color=v.color.xyz;o.index=0u;o.lighting=v.color.w;return o;
}
const BOX=array<u32,36>(0,2,1,1,2,3,4,5,6,5,7,6,0,1,4,1,5,4,2,6,3,3,6,7,0,4,2,2,4,6,1,3,5,3,7,5);
@vertex fn rodVertex(@builtin(vertex_index) i:u32,@builtin(instance_index) instance:u32)->Out {
 let v=records[instance];let corner=BOX[i];let lo=min(v.a.xyz,v.b.xyz)-vec3f(v.a.w);let hi=max(v.a.xyz,v.b.xyz)+vec3f(v.a.w);
 let p=vec3f(select(lo.x,hi.x,(corner&1u)!=0u),select(lo.y,hi.y,(corner&2u)!=0u),select(lo.z,hi.z,(corner&4u)!=0u));
 var o:Out;o.clip=project(p);o.world=p;o.normal=vec3f(0);o.color=v.color.xyz;o.index=instance;o.lighting=v.color.w;return o;
}
fn shade(n0:vec3f,p:vec3f,color:vec3f)->vec3f{
 let view=normalize(u.cameraPosition.xyz-p);let n=select(n0,-n0,dot(n0,view)<0.0);
 let light=normalize(vec3f(-.4,1,.6));return color*(.3+.7*max(0.0,dot(n,light)));
}
struct Fragment { @location(0) color:vec4f,@builtin(frag_depth) depth:f32 }
fn output(p:vec3f,n:vec3f,color:vec3f,lighting:f32)->Fragment {
 var o:Fragment;o.color=vec4f(select(color,shade(normalize(n),p,color),lighting>0.5),length(p-u.cameraPosition.xyz));
 o.depth=NEAR/dot(p-u.cameraPosition.xyz,forward());return o;
}
@fragment fn shellFragment(v:Out)->Fragment{return output(v.world,v.normal,v.color,v.lighting);}
fn inPlanes(p:vec3f,index:u32)->bool{
 for(var j=0u;j<2u;j++){let plane=clipPlanes[6u*index+j];if(dot(plane.xyz,p)>plane.w+1e-7){return false;}}
 return true;
}
fn rodLocal(v:vec3f,d1:vec3f,d2:vec3f,t:vec3f,r:vec3f)->vec3f{return vec3f(dot(v,d1),dot(v,d2),dot(v,t))/r;}
@fragment fn rodFragment(v:Out)->Fragment {
 let record=records[v.index];let a=record.a.xyz;let b=record.b.xyz;let base=6u*v.index;
 let frame1=clipPlanes[base+4u];let frame2=clipPlanes[base+5u];let d1=frame1.xyz;let d2=frame2.xyz;
 let tangent=normalize(b-a);let radii=vec3f(frame1.w,frame2.w,sqrt(frame1.w*frame2.w));let L=length(b-a)/radii.z;
 let ro=u.cameraPosition.xyz;let rd=normalize(v.world-ro);let o=rodLocal(ro-a,d1,d2,tangent,radii);let d=rodLocal(rd,d1,d2,tangent,radii);
 var nearest=1e30;var capNormal=vec3f(0);
 let aa=dot(d.xy,d.xy);let bb=dot(o.xy,d.xy);let cc=dot(o.xy,o.xy)-1.0;let h=bb*bb-aa*cc;
 if(aa>1e-24&&h>=0.0){for(var j=0u;j<2u;j++){let q=(-bb+select(-sqrt(h),sqrt(h),j==1u))/aa;let z=o.z+q*d.z;if(q>=0.0&&z>=0.0&&z<=L&&inPlanes(ro+q*rd,v.index)){nearest=min(nearest,q);}}}
 for(var k=0u;k<2u;k++){
  let center=select(0.0,L,k==1u);let oc=o-vec3f(0,0,center);let aa0=dot(d,d);let bb0=dot(oc,d);let h0=bb0*bb0-aa0*(dot(oc,oc)-1.0);
  if(h0>=0.0){for(var j=0u;j<2u;j++){let q=(-bb0+select(-sqrt(h0),sqrt(h0),j==1u))/aa0;let z=o.z+q*d.z;if(q>=0.0&&select(z<=0.0,z>=L,k==1u)&&inPlanes(ro+q*rd,v.index)){nearest=min(nearest,q);}}}
 }
 for(var k=0u;k<2u;k++){
  let plane=clipPlanes[base+k];let denom=dot(plane.xyz,rd);
  if(abs(denom)>1e-12){let q=(plane.w-dot(plane.xyz,ro))/denom;let p=ro+q*rd;let local=rodLocal(p-a,d1,d2,tangent,radii);let delta=local-vec3f(0,0,clamp(local.z,0.0,L));
   if(q>=0.0&&q<nearest&&dot(delta,delta)<=1.0+1e-5&&inPlanes(p,v.index)){nearest=q;capNormal=plane.xyz;}
  }
 }
 if(nearest==1e30){discard;}
 let p=ro+nearest*rd;let local=rodLocal(p-a,d1,d2,tangent,radii);let q=(local-vec3f(0,0,clamp(local.z,0.0,L)))/radii;
 let normal=d1*q.x+d2*q.y+tangent*q.z;let fraction=clamp(local.z/max(L,1e-24),0.0,1.0);
 let color=mix(clipPlanes[base+2u].xyz,clipPlanes[base+3u].xyz,fraction);
 return output(p,select(normal,capNormal,dot(capNormal,capNormal)>0.0),color,v.lighting);
}
`;

const p3=(a:ArrayLike<number>,i:number):ClothVector=>[a[3*i]!,a[3*i+1]!,a[3*i+2]!];
export function packClothSurface(document:ClothDocument,surface:ClothSurfacePublication) {
 const {topology,positions_m:p}=surface;
 const shells=new Float32Array(topology.triangles.length*12);
 let segments=0;for(let r=0;r<topology.rodOffsets.length-1;r++)segments+=topology.rodOffsets[r+1]!-topology.rodOffsets[r]!-1;
 const rods=new Float32Array(segments*12);
 // Two clip planes, endpoint colors, and material ellipse axes share each record.
 const rodPlanes=new Float32Array(segments*24),groupCount=Math.max((topology.groups?.reduce((a,b)=>Math.max(a,b),0)??0)+1,document.source?.previewParticleGroupCount??1),lighting=document.appearance==='source-preview'?0:1;
 const color=(material:number,vertex:number,kind:'shell'|'rod')=>clothVertexColor(document,surface,material,vertex,kind,groupCount);
 for(let t=0;t<topology.triangles.length;t+=3){
  const ids=topology.triangles.slice(t,t+3),[a,b,c]=ids.map(i=>p3(p,i)) as [ClothVector,ClothVector,ClothVector];
  const u=b.map((v,k)=>v-a[k]!),v=c.map((x,k)=>x-a[k]!);
  const n=[u[1]!*v[2]!-u[2]!*v[1]!,u[2]!*v[0]!-u[0]!*v[2]!,u[0]!*v[1]!-u[1]!*v[0]!],length=Math.hypot(...n);
  if(!(length>0))throw new Error('Cannot draw a degenerate cloth triangle');
  for(let k=0;k<3;k++)shells.set([...p3(p,ids[k]!),0,...n.map(x=>x/length),0,...color(topology.triangleMaterials?.[t/3]??0,ids[k]!,'shell'),lighting],12*(t+k));
 }
 let segment=0;
 for(let r=0;r<topology.rodOffsets.length-1;r++){
  const material=topology.rodMaterials?.[r]??0;
  for(let j=topology.rodOffsets[r]!;j<topology.rodOffsets[r+1]!-1;j++){
   const a=topology.rodIndices[j]!,b=topology.rodIndices[j+1]!;
   const geometry=clothRodCapsuleGeometry(document,topology,p,r,j-topology.rodOffsets[r]!,surface);
   const ca=color(material,a,'rod'),cb=color(material,b,'rod');
   rods.set([...geometry.a,clothRodBoundingRadius(geometry),...geometry.b,0,...ca,lighting],12*segment);
   geometry.planes.forEach((plane,k)=>rodPlanes.set([...plane.normal,plane.offset_m],24*segment+4*k));
   rodPlanes.set([...ca,0,...cb,0],24*segment+8);const axes=clothRodEllipsoidAxes(geometry);rodPlanes.set([...axes.d1,axes.radii[0],...axes.d2,axes.radii[1]],24*segment+16);segment++;
  }
 }
 return {shells,rods,rodPlanes};
}

interface SurfaceBuffers { document:ClothDocument; publication:ClothSurfacePublication; shell:GPUBuffer; rods:GPUBuffer; planes:GPUBuffer; shellGroup:GPUBindGroup; rodGroup:GPUBindGroup; triangles:number; segments:number }
export class ClothSurfaceRenderer {
 private layout?:GPUBindGroupLayout;
 private shellPipeline?:GPURenderPipeline;
 private rodPipeline?:GPURenderPipeline;
 private readonly surfaces=new Map<string,SurfaceBuffers>();
 constructor(private readonly device:GPUDevice,private readonly view:GPUBuffer){}
 async initialize():Promise<void>{
  this.layout=this.device.createBindGroupLayout({label:'Cloth surface bindings',entries:[
   {binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:'uniform'}},
   {binding:1,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:'read-only-storage'}},
   {binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:'read-only-storage'}}]});
  const module=this.device.createShaderModule({label:'Cloth mechanical surface',code:clothSurfaceWGSL});
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.layout]});
  const create=(rod:boolean)=>this.device.createRenderPipelineAsync({label:rod?'Cloth rod capsules':'Cloth shell triangles',layout,
   vertex:{module,entryPoint:rod?'rodVertex':'shellVertex'},fragment:{module,entryPoint:rod?'rodFragment':'shellFragment',targets:[{format:'rgba16float'}]},
   primitive:{topology:'triangle-list',cullMode:'none'},depthStencil:{format:'depth32float',depthWriteEnabled:true,depthCompare:'greater'}});
  [this.shellPipeline,this.rodPipeline]=await Promise.all([create(false),create(true)]);
 }
 setSurfaces(documents:readonly ClothDocument[],publications:readonly ClothSurfacePublication[]):void{
  if(!this.layout)throw new Error('Cloth renderer is not initialized');
  const live=new Set(publications.map(p=>p.documentId));
  for(const [id,b] of this.surfaces)if(!live.has(id)){b.shell.destroy();b.rods.destroy();b.planes.destroy();this.surfaces.delete(id);}
  for(const publication of publications){
   const document=documents.find(d=>d.id===publication.documentId);if(!document)throw new Error('Unauthored cloth surface');
   const old=this.surfaces.get(publication.documentId);if(old?.publication===publication&&old.document===document)continue;
   const data=packClothSurface(document,publication);
   const upload=(values:Float32Array<ArrayBuffer>,previous:GPUBuffer|undefined)=>{
    const size=Math.max(48,values.byteLength);let buffer=previous;
    if(!buffer||buffer.size<size){buffer?.destroy();buffer=this.device.createBuffer({label:'Cloth published geometry',size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});}
    if(values.length)this.device.queue.writeBuffer(buffer,0,values);return buffer;
   };
   const shell=upload(data.shells,old?.shell),rods=upload(data.rods,old?.rods),planes=upload(data.rodPlanes,old?.planes);
   const group=(buffer:GPUBuffer)=>this.device.createBindGroup({layout:this.layout!,entries:[{binding:0,resource:{buffer:this.view}},{binding:1,resource:{buffer}},{binding:2,resource:{buffer:planes}}]});
   this.surfaces.set(publication.documentId,{document,publication,shell,rods,planes,shellGroup:old?.shell===shell&&old.planes===planes?old.shellGroup:group(shell),rodGroup:old?.rods===rods&&old.planes===planes?old.rodGroup:group(rods),triangles:data.shells.length/12,segments:data.rods.length/12});
  }
 }
 encode(encoder:GPUCommandEncoder,target:GPUTextureView,depth:GPUTextureView):void{
  if(!this.shellPipeline||!this.rodPipeline||!this.surfaces.size)return;
  const pass=encoder.beginRenderPass({label:'Cloth physical surfaces',colorAttachments:[{view:target,loadOp:'load',storeOp:'store'}],depthStencilAttachment:{view:depth,depthLoadOp:'load',depthStoreOp:'store'}});
  for(const b of this.surfaces.values()){
   if(b.triangles){pass.setPipeline(this.shellPipeline);pass.setBindGroup(0,b.shellGroup);pass.draw(b.triangles);}
   if(b.segments){pass.setPipeline(this.rodPipeline);pass.setBindGroup(0,b.rodGroup);pass.draw(36,b.segments);}
  }
  pass.end();
 }
 destroy():void{for(const b of this.surfaces.values()){b.shell.destroy();b.rods.destroy();b.planes.destroy();}this.surfaces.clear();}
}
