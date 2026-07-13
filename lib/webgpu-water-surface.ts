/** GPU extraction and refractive rasterization of the hierarchical VOF surface. */

const MAX_SURFACE_VERTICES = 750_000;
const MAX_INTERFACE_BRICKS = 65_535;

export interface HierarchySurfaceResources {
  cells: GPUBuffer;
  metadata: GPUBuffer;
  pageTable: GPUBuffer;
  params: GPUBuffer;
  revision: number;
  simulationRevision: number;
}

const extractionShader = /* wgsl */ `
struct HierarchyCell { negAlpha:vec4f, posPressure:vec4f }
struct HierarchyParams { dimsDt:vec4f, pageBrick:vec4f, originH:vec4f, physical:vec4f, containerBodies:vec4f, topology:vec4f, boundary:vec4f }
struct SurfaceVertex { position:vec4f, normal:vec4f }

@group(0) @binding(0) var<storage,read> cells:array<HierarchyCell>;
@group(0) @binding(1) var<storage,read> metadata:array<vec4u>;
@group(0) @binding(2) var<storage,read> pages:array<u32>;
@group(0) @binding(3) var<storage,read> params:HierarchyParams;
@group(0) @binding(4) var<storage,read_write> vertices:array<SurfaceVertex>;
@group(0) @binding(5) var<storage,read_write> vertexCounter:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> interfaceBricks:array<u32>;
@group(0) @binding(7) var<storage,read_write> interfaceCounter:array<atomic<u32>>;

const CAPACITY:u32=${MAX_SURFACE_VERTICES}u;
var<workgroup> brickCounts:array<u32,64>;
var<workgroup> brickOffsets:array<u32,64>;
var<workgroup> brickBase:u32;

fn alphaAt(cell:vec3i)->f32 {
  let dims=vec3i(params.dimsDt.xyz);
  let q=vec3u(clamp(cell,vec3i(0),dims-vec3i(1)));
  let page=q/4u;
  let pd=vec3u(params.pageBrick.xyz);
  let slot=pages[page.x+pd.x*(page.y+pd.y*page.z)];
  let brick=metadata[slot*2u];
  let local=vec3u(clamp((vec3f(q)-vec3f(brick.xyz))/f32(brick.w),vec3f(0.0),vec3f(3.0)));
  return cells[slot*64u+local.x+4u*(local.y+4u*local.z)].negAlpha.w;
}

@compute @workgroup_size(64)
fn classify(@builtin(global_invocation_id) global:vec3u) {
  let slot=global.x;let leafCount=u32(params.topology.x);if(slot>=leafCount){return;}
  var minimum=1.0;var maximum=0.0;for(var local=0u;local<64u;local+=1u){let alpha=cells[slot*64u+local].negAlpha.w;minimum=min(minimum,alpha);maximum=max(maximum,alpha);}
  let brick=metadata[slot*2u];let q=vec3i(brick.xyz);let scale=i32(brick.w);
  for(var a=0i;a<4i;a+=1i){for(var b=0i;b<4i;b+=1i){
    let xFace=alphaAt(q+vec3i(4,a,b)*scale);let yFace=alphaAt(q+vec3i(a,4,b)*scale);let zFace=alphaAt(q+vec3i(a,b,4)*scale);
    minimum=min(minimum,min(xFace,min(yFace,zFace)));maximum=max(maximum,max(xFace,max(yFace,zFace)));
  }}
  if(minimum<0.5&&maximum>=0.5){let destination=atomicAdd(&interfaceCounter[0],1u);if(destination<${MAX_INTERFACE_BRICKS}u){interfaceBricks[destination]=slot;}}
}

fn edgeVertex(a:u32,b:u32,p:ptr<function,array<vec3f,8>>,v:ptr<function,array<f32,8>>,n:ptr<function,array<vec3f,8>>)->SurfaceVertex {
  let va=(*v)[a];let vb=(*v)[b];
  let amount=clamp((0.5-va)/max(abs(vb-va),1e-6)*sign(vb-va),0.0,1.0);
  let normal=mix((*n)[a],(*n)[b],amount);
  return SurfaceVertex(vec4f(mix((*p)[a],(*p)[b],amount),1.0),vec4f(normalize(select(vec3f(0.0,1.0,0.0),normal,length(normal)>1e-6)),0.0));
}

fn emitTriangle(a:SurfaceVertex,bIn:SurfaceVertex,cIn:SurfaceVertex,outward:vec3f,base:u32,count:ptr<function,u32>) {
  var first=a;var b=bIn;var c=cIn;
  if(dot(cross(b.position.xyz-first.position.xyz,c.position.xyz-first.position.xyz),outward)<0.0){let swap=b;b=c;c=swap;}
  let destination=base+*count;if(destination+2u<CAPACITY){vertices[destination]=first;vertices[destination+1u]=b;vertices[destination+2u]=c;}*count+=3u;
}

fn tetraVertexCount(ids:vec4u,v:ptr<function,array<f32,8>>)->u32 {var count=0u;for(var i=0u;i<4u;i+=1u){if((*v)[ids[i]]>=0.5){count+=1u;}}return select(select(3u,6u,count==2u),0u,count==0u||count==4u);}

fn processTetra(ids:vec4u,p:ptr<function,array<vec3f,8>>,v:ptr<function,array<f32,8>>,n:ptr<function,array<vec3f,8>>,base:u32,count:ptr<function,u32>) {
  var inside=array<u32,4>();var outside=array<u32,4>();var ni=0u;var no=0u;
  for(var i=0u;i<4u;i+=1u){let id=ids[i];if((*v)[id]>=0.5){inside[ni]=id;ni+=1u;}else{outside[no]=id;no+=1u;}}
  if(ni==0u||ni==4u){return;}
  var insideCenter=vec3f(0.0);var outsideCenter=vec3f(0.0);
  for(var i=0u;i<ni;i+=1u){insideCenter+=(*p)[inside[i]];}
  for(var i=0u;i<no;i+=1u){outsideCenter+=(*p)[outside[i]];}
  let outward=outsideCenter/f32(no)-insideCenter/f32(ni);
  if(ni==1u){
    let root=inside[0];emitTriangle(edgeVertex(root,outside[0],p,v,n),edgeVertex(root,outside[1],p,v,n),edgeVertex(root,outside[2],p,v,n),outward,base,count);return;
  }
  if(ni==3u){
    let root=outside[0];emitTriangle(edgeVertex(root,inside[0],p,v,n),edgeVertex(root,inside[1],p,v,n),edgeVertex(root,inside[2],p,v,n),outward,base,count);return;
  }
  let a=inside[0];let b=inside[1];let c=outside[0];let d=outside[1];
  let p0=edgeVertex(a,c,p,v,n);let p1=edgeVertex(a,d,p,v,n);let p2=edgeVertex(b,d,p,v,n);let p3=edgeVertex(b,c,p,v,n);
  emitTriangle(p0,p1,p2,outward,base,count);emitTriangle(p0,p2,p3,outward,base,count);
}

@compute @workgroup_size(64)
fn extract(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) localIndex:u32) {
  let slot=interfaceBricks[group.x];let brick=metadata[slot*2u];let scale=i32(brick.w);
  let local=vec3i(i32(localIndex&3u),i32((localIndex>>2u)&3u),i32(localIndex>>4u));
  let q=vec3i(brick.xyz)+local*scale;
  let physical=vec3i(params.boundary.yzw);
  var valid=all(q+vec3i(scale)<physical);
  let offsets=array<vec3i,8>(vec3i(0,0,0),vec3i(1,0,0),vec3i(0,1,0),vec3i(1,1,0),vec3i(0,0,1),vec3i(1,0,1),vec3i(0,1,1),vec3i(1,1,1));
  var values:array<f32,8>;var minimum=1.0;var maximum=0.0;
  for(var i=0u;i<8u;i+=1u){let sampleLocal=local+offsets[i];if(all(sampleLocal<vec3i(4))){let index=u32(sampleLocal.x+4*(sampleLocal.y+4*sampleLocal.z));values[i]=cells[slot*64u+index].negAlpha.w;}else{values[i]=alphaAt(q+offsets[i]*scale);}minimum=min(minimum,values[i]);maximum=max(maximum,values[i]);}
  valid=valid&&minimum<0.5&&maximum>=0.5;
  let tetrahedra=array<vec4u,6>(vec4u(0,1,3,7),vec4u(0,3,2,7),vec4u(0,2,6,7),vec4u(0,6,4,7),vec4u(0,4,5,7),vec4u(0,5,1,7));
  var outputCount=0u;if(valid){for(var i=0u;i<6u;i+=1u){outputCount+=tetraVertexCount(tetrahedra[i],&values);}}
  brickCounts[localIndex]=outputCount;workgroupBarrier();
  if(localIndex==0u){var sum=0u;for(var i=0u;i<64u;i+=1u){brickOffsets[i]=sum;sum+=brickCounts[i];}brickBase=atomicAdd(&vertexCounter[0],sum);}
  workgroupBarrier();let destination=brickBase+brickOffsets[localIndex];
  var positions:array<vec3f,8>;var normals:array<vec3f,8>;
  let origin=params.originH.xyz;let h=params.originH.w;
  if(valid&&destination+outputCount<=CAPACITY){let cubeGradient=-vec3f(values[1]+values[3]+values[5]+values[7]-values[0]-values[2]-values[4]-values[6],values[2]+values[3]+values[6]+values[7]-values[0]-values[1]-values[4]-values[5],values[4]+values[5]+values[6]+values[7]-values[0]-values[1]-values[2]-values[3]);let cubeNormal=normalize(select(vec3f(0.0,1.0,0.0),cubeGradient,length(cubeGradient)>1e-6));for(var i=0u;i<8u;i+=1u){
    let sample=q+offsets[i]*scale;
    positions[i]=origin+(vec3f(sample)+vec3f(0.5*f32(scale)))*h;
    normals[i]=cubeNormal;
  }
  var written=0u;for(var i=0u;i<6u;i+=1u){processTetra(tetrahedra[i],&positions,&values,&normals,destination,&written);}}
}
`;

const finalizeShader = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> vertexCounter:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> indirect:array<u32>;
const CAPACITY:u32=${MAX_SURFACE_VERTICES}u;
@compute @workgroup_size(1) fn finalize(){let count=min(atomicLoad(&vertexCounter[0]),CAPACITY);indirect[0]=count-count%3u;indirect[1]=1u;indirect[2]=0u;indirect[3]=0u;}
`;

const interfaceDispatchShader = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> interfaceCounter:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> dispatch:array<u32>;
@compute @workgroup_size(1) fn finalize(){dispatch[0]=min(atomicLoad(&interfaceCounter[0]),${MAX_INTERFACE_BRICKS}u);dispatch[1]=1u;dispatch[2]=1u;}
`;

const surfaceShader = /* wgsl */ `
struct Uniforms { viewport:vec4f,cameraPosition:vec4f,cameraTarget:vec4f,container:vec4f,options:vec4f,gridInfo:vec4f }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var sceneColor:texture_2d<f32>;
@group(0) @binding(2) var sceneSampler:sampler;
@group(0) @binding(3) var backDistance:texture_2d<f32>;
struct BodyGPU { positionRadius:vec4f,halfSizeShape:vec4f,orientation:vec4f,colorSelected:vec4f }
@group(0) @binding(4) var<storage,read> bodies:array<BodyGPU,12>;

struct VertexInput { @location(0) position:vec4f,@location(1) normal:vec4f }
struct VertexOutput { @builtin(position) clip:vec4f,@location(0) world:vec3f,@location(1) normal:vec3f }

fn cameraBasis()->mat3x3f {
  let forward=normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);
  let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));
  let up=normalize(cross(right,forward));return mat3x3f(right,up,forward);
}

@vertex fn surfaceVertex(input:VertexInput)->VertexOutput {
  let basis=cameraBasis();let relative=input.position.xyz-u.cameraPosition.xyz;
  let view=vec3f(dot(relative,basis[0]),dot(relative,basis[1]),dot(relative,basis[2]));
  let aspect=max(u.viewport.x/u.viewport.y,0.001);let z=max(view.z,0.001);let near=0.01;let far=100.0;
  var output:VertexOutput;output.clip=vec4f(view.x/(aspect*0.72),view.y/0.72,(far*z-near*far)/(far-near),z);output.world=input.position.xyz;output.normal=input.normal.xyz;return output;
}

fn projectedUV(world:vec3f)->vec2f {
  let basis=cameraBasis();let relative=world-u.cameraPosition.xyz;let z=max(dot(relative,basis[2]),0.001);let aspect=max(u.viewport.x/u.viewport.y,0.001);
  let ndc=vec2f(dot(relative,basis[0])/(z*aspect*0.72),dot(relative,basis[1])/(z*0.72));
  return vec2f(ndc.x*0.5+0.5,0.5-ndc.y*0.5);
}

fn boxIntersection(ro:vec3f,rd:vec3f,boundsMin:vec3f,boundsMax:vec3f)->vec2f {
  let inverse=1.0/rd;let t0=(boundsMin-ro)*inverse;let t1=(boundsMax-ro)*inverse;let near3=min(t0,t1);let far3=max(t0,t1);
  return vec2f(max(max(near3.x,near3.y),near3.z),min(min(far3.x,far3.y),far3.z));
}

fn quatRotate(q:vec4f,v:vec3f)->vec3f {let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.0*(q.x*uv+uuv);}
fn quatInverseRotate(q:vec4f,v:vec3f)->vec3f {return quatRotate(vec4f(q.x,-q.yzw),v);}
fn sphereHit(ro:vec3f,rd:vec3f,center:vec3f,radius:f32)->f32 {
  let oc=ro-center;let b=dot(oc,rd);let discriminant=b*b-dot(oc,oc)+radius*radius;if(discriminant<0.0){return 1e20;}
  let root=sqrt(discriminant);let near=-b-root;let far=-b+root;return select(select(1e20,far,far>0.0001),near,near>0.0001);
}
fn cylinderHit(ro:vec3f,rd:vec3f,radius:f32,halfHeight:f32,capped:bool)->f32 {
  var best=1e20;let a=rd.x*rd.x+rd.z*rd.z;if(a>1e-8){let b=ro.x*rd.x+ro.z*rd.z;let c=ro.x*ro.x+ro.z*ro.z-radius*radius;let discriminant=b*b-a*c;if(discriminant>=0.0){let root=sqrt(discriminant);let near=(-b-root)/a;let far=(-b+root)/a;let candidate=select(far,near,near>0.0001);let y=ro.y+rd.y*candidate;if(candidate>0.0001&&abs(y)<=halfHeight){best=candidate;}}}
  if(capped&&abs(rd.y)>1e-8){let top=(halfHeight-ro.y)/rd.y;let topPoint=ro+rd*top;if(top>0.0001&&dot(topPoint.xz,topPoint.xz)<=radius*radius){best=min(best,top);}let bottom=(-halfHeight-ro.y)/rd.y;let bottomPoint=ro+rd*bottom;if(bottom>0.0001&&dot(bottomPoint.xz,bottomPoint.xz)<=radius*radius){best=min(best,bottom);}}
  return best;
}
fn bodyHit(ro:vec3f,rd:vec3f,body:BodyGPU)->f32 {
  let localOrigin=quatInverseRotate(body.orientation,ro-body.positionRadius.xyz);let localDirection=quatInverseRotate(body.orientation,rd);let shape=i32(round(body.halfSizeShape.w));
  if(shape==0){return sphereHit(localOrigin,localDirection,vec3f(0.0),body.halfSizeShape.x);}
  if(shape==1){let hit=boxIntersection(localOrigin,localDirection,-body.halfSizeShape.xyz,body.halfSizeShape.xyz);return select(select(1e20,hit.y,hit.y>0.0001),hit.x,hit.x>0.0001&&hit.x<=hit.y);}
  if(shape==2){return min(cylinderHit(localOrigin,localDirection,body.halfSizeShape.x,body.halfSizeShape.y,false),min(sphereHit(localOrigin,localDirection,vec3f(0.0,body.halfSizeShape.y,0.0),body.halfSizeShape.x),sphereHit(localOrigin,localDirection,vec3f(0.0,-body.halfSizeShape.y,0.0),body.halfSizeShape.x)));}
  return cylinderHit(localOrigin,localDirection,body.halfSizeShape.x,body.halfSizeShape.y,true);
}
fn nearestBodyDistance(ro:vec3f,rd:vec3f)->f32 {var nearest=1e20;let count=u32(round(u.options.z));for(var i=0u;i<12u;i+=1u){if(i>=count){break;}nearest=min(nearest,bodyHit(ro,rd,bodies[i]));}return nearest;}

@fragment fn backFragment(input:VertexOutput)->@location(0) f32 {return distance(input.world,u.cameraPosition.xyz);}

@fragment fn waterFragment(input:VertexOutput)->@location(0) vec4f {
  let incident=normalize(input.world-u.cameraPosition.xyz);var normal=normalize(input.normal);
  let front=distance(input.world,u.cameraPosition.xyz);if(nearestBodyDistance(u.cameraPosition.xyz,incident)<front-0.001){discard;}
  if(dot(normal,-incident)<0.0){normal=-normal;}
  let refractedDirection=refract(incident,normal,1.0/1.333);
  let size=u.container.xyz;let boundsMin=vec3f(-size.x*0.5,0.0,-size.z*0.5);let boundsMax=vec3f(size.x*0.5,size.y,size.z*0.5);
  let boxHit=boxIntersection(input.world+refractedDirection*0.002,refractedDirection,boundsMin,boundsMax);
  let boxThickness=max(0.0,boxHit.y);
  let coords=clamp(vec2i(input.clip.xy),vec2i(0),vec2i(u.viewport.xy)-vec2i(1));let back=textureLoad(backDistance,coords,0).x;
  let layerThickness=back-front;let hasBack=back<1e19&&layerThickness>0.001;
  let thickness=clamp(select(boxThickness,min(boxThickness,layerThickness),hasBack),0.0,3.0);
  let sampleWorld=input.world+refractedDirection*max(thickness,0.02);let uv=clamp(projectedUV(sampleWorld),vec2f(0.002),vec2f(0.998));
  let background=textureSample(sceneColor,sceneSampler,uv).xyz;
  let transmission=exp(-vec3f(0.95,0.28,0.16)*thickness);let scatter=vec3f(0.018,0.34,0.29)*(vec3f(1.0)-transmission);
  let refracted=background*transmission+scatter;
  let reflected=mix(vec3f(0.025,0.07,0.07),vec3f(0.19,0.38,0.34),clamp(reflect(incident,normal).y*0.5+0.5,0.0,1.0));
  let fresnel=0.0204+0.9796*pow(1.0-max(dot(normal,-incident),0.0),5.0);
  var color=mix(refracted,reflected,fresnel);color+=vec3f(0.16,0.72,0.64)*pow(max(dot(reflect(incident,normal),normalize(vec3f(-0.5,0.8,0.25))),0.0),64.0);
  return vec4f(color,1.0);
}
`;

export class WebGPUWaterSurface {
  private readonly vertices:GPUBuffer;
  private readonly vertexCounter:GPUBuffer;
  private readonly indirect:GPUBuffer;
  private readonly interfaceBricks:GPUBuffer;
  private readonly interfaceCounter:GPUBuffer;
  private readonly interfaceDispatch:GPUBuffer;
  private readonly classifyPipeline:GPUComputePipeline;
  private readonly extractionPipeline:GPUComputePipeline;
  private readonly interfaceDispatchPipeline:GPUComputePipeline;
  private readonly finalizePipeline:GPUComputePipeline;
  private readonly backPipeline:GPURenderPipeline;
  private readonly waterPipeline:GPURenderPipeline;
  private readonly finalizeGroup:GPUBindGroup;
  private readonly interfaceDispatchGroup:GPUBindGroup;
  private readonly backGroup:GPUBindGroup;
  private readonly sampler:GPUSampler;
  private extractionGroup?:GPUBindGroup;
  private classifyGroup?:GPUBindGroup;
  private renderGroup?:GPUBindGroup;
  private resourceRevision=-1;
  private extractedSimulationRevision=-1;
  private targetKey="";
  private sceneTexture?:GPUTexture;
  private backTexture?:GPUTexture;
  private backDepth?:GPUTexture;
  private frontDepth?:GPUTexture;

  private constructor(private readonly device:GPUDevice,private readonly format:GPUTextureFormat,private readonly uniforms:GPUBuffer,private readonly bodies:GPUBuffer,modules:{extraction:GPUShaderModule;interfaceDispatch:GPUShaderModule;finalize:GPUShaderModule;surface:GPUShaderModule}) {
    this.vertices=device.createBuffer({label:"Extracted water surface vertices",size:MAX_SURFACE_VERTICES*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.VERTEX});
    this.vertexCounter=device.createBuffer({label:"Water surface vertex counter",size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.indirect=device.createBuffer({label:"Water surface indirect draw",size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST});
    this.interfaceBricks=device.createBuffer({label:"Water interface brick list",size:MAX_INTERFACE_BRICKS*4,usage:GPUBufferUsage.STORAGE});
    this.interfaceCounter=device.createBuffer({label:"Water interface brick counter",size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.interfaceDispatch=device.createBuffer({label:"Water interface indirect dispatch",size:12,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST});
    this.classifyPipeline=device.createComputePipeline({label:"Classify water interface bricks",layout:"auto",compute:{module:modules.extraction,entryPoint:"classify"}});
    this.extractionPipeline=device.createComputePipeline({label:"Extract hierarchical water surface",layout:"auto",compute:{module:modules.extraction,entryPoint:"extract"}});
    this.interfaceDispatchPipeline=device.createComputePipeline({label:"Finalize water interface dispatch",layout:"auto",compute:{module:modules.interfaceDispatch,entryPoint:"finalize"}});
    this.finalizePipeline=device.createComputePipeline({label:"Finalize water surface draw",layout:"auto",compute:{module:modules.finalize,entryPoint:"finalize"}});
    this.finalizeGroup=device.createBindGroup({layout:this.finalizePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.vertexCounter}},{binding:1,resource:{buffer:this.indirect}}]});
    this.interfaceDispatchGroup=device.createBindGroup({layout:this.interfaceDispatchPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.interfaceCounter}},{binding:1,resource:{buffer:this.interfaceDispatch}}]});
    const vertexBuffers:GPUVertexBufferLayout[]=[{arrayStride:32,attributes:[{shaderLocation:0,offset:0,format:"float32x4"},{shaderLocation:1,offset:16,format:"float32x4"}]}];
    this.backPipeline=device.createRenderPipeline({label:"Water back-face depth",layout:"auto",vertex:{module:modules.surface,entryPoint:"surfaceVertex",buffers:vertexBuffers},fragment:{module:modules.surface,entryPoint:"backFragment",targets:[{format:"r32float"}]},primitive:{topology:"triangle-list",cullMode:"front",frontFace:"ccw"},depthStencil:{format:"depth24plus",depthWriteEnabled:true,depthCompare:"less"}});
    this.waterPipeline=device.createRenderPipeline({label:"Refractive water surface",layout:"auto",vertex:{module:modules.surface,entryPoint:"surfaceVertex",buffers:vertexBuffers},fragment:{module:modules.surface,entryPoint:"waterFragment",targets:[{format}]},primitive:{topology:"triangle-list",cullMode:"back",frontFace:"ccw"},depthStencil:{format:"depth24plus",depthWriteEnabled:true,depthCompare:"less"}});
    this.sampler=device.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"});
    this.backGroup=device.createBindGroup({layout:this.backPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniforms}}]});
  }

  static async create(device:GPUDevice,format:GPUTextureFormat,uniforms:GPUBuffer,bodies:GPUBuffer):Promise<WebGPUWaterSurface> {
    const modules={extraction:device.createShaderModule({label:"Water surface extraction shader",code:extractionShader}),interfaceDispatch:device.createShaderModule({label:"Water interface dispatch shader",code:interfaceDispatchShader}),finalize:device.createShaderModule({label:"Water surface finalize shader",code:finalizeShader}),surface:device.createShaderModule({label:"Refractive water surface shader",code:surfaceShader})};
    const results=await Promise.all(Object.values(modules).map((module)=>module.getCompilationInfo()));
    const errors=results.flatMap((result)=>result.messages.filter((message)=>message.type==="error"));
    if(errors.length>0)throw new Error(errors.map((error)=>`${error.lineNum}:${error.linePos} ${error.message}`).join("\n"));
    return new WebGPUWaterSurface(device,format,uniforms,bodies,modules);
  }

  resize(width:number,height:number):void {
    const key=`${width}x${height}`;if(key===this.targetKey)return;this.destroyTargets();
    this.sceneTexture=this.device.createTexture({label:"Opaque scene for water refraction",size:[width,height],format:this.format,usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.TEXTURE_BINDING});
    this.backTexture=this.device.createTexture({label:"Water back-face distance",size:[width,height],format:"r32float",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});
    this.backDepth=this.device.createTexture({label:"Water back-face depth",size:[width,height],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT});
    this.frontDepth=this.device.createTexture({label:"Water front-face depth",size:[width,height],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT});
    this.targetKey=key;this.renderGroup=undefined;
  }

  private destroyTargets():void {this.sceneTexture?.destroy();this.backTexture?.destroy();this.backDepth?.destroy();this.frontDepth?.destroy();}

  invalidateResources():void {this.resourceRevision=-1;this.extractedSimulationRevision=-1;this.extractionGroup=undefined;this.classifyGroup=undefined;}

  updateResources(resources:HierarchySurfaceResources):void {
    if(resources.revision===this.resourceRevision)return;
    this.extractionGroup=this.device.createBindGroup({layout:this.extractionPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:resources.cells}},{binding:1,resource:{buffer:resources.metadata}},{binding:2,resource:{buffer:resources.pageTable}},{binding:3,resource:{buffer:resources.params}},{binding:4,resource:{buffer:this.vertices}},{binding:5,resource:{buffer:this.vertexCounter}},{binding:6,resource:{buffer:this.interfaceBricks}}]});
    this.classifyGroup=this.device.createBindGroup({layout:this.classifyPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:resources.cells}},{binding:1,resource:{buffer:resources.metadata}},{binding:2,resource:{buffer:resources.pageTable}},{binding:3,resource:{buffer:resources.params}},{binding:6,resource:{buffer:this.interfaceBricks}},{binding:7,resource:{buffer:this.interfaceCounter}}]});
    this.resourceRevision=resources.revision;this.extractedSimulationRevision=-1;
  }

  encode(encoder:GPUCommandEncoder,leafCount:number,presentation:GPUTexture,simulationRevision:number):void {
    if(!this.extractionGroup||!this.classifyGroup||!this.sceneTexture||!this.backTexture||!this.backDepth||!this.frontDepth)return;
    if(simulationRevision!==this.extractedSimulationRevision){
      encoder.clearBuffer(this.vertexCounter);encoder.clearBuffer(this.indirect);encoder.clearBuffer(this.interfaceCounter);encoder.clearBuffer(this.interfaceDispatch);
      let compute=encoder.beginComputePass({label:"Classify water interface bricks"});compute.setPipeline(this.classifyPipeline);compute.setBindGroup(0,this.classifyGroup);compute.dispatchWorkgroups(Math.max(1,Math.ceil(leafCount/64)));compute.end();
      compute=encoder.beginComputePass({label:"Finalize water interface dispatch"});compute.setPipeline(this.interfaceDispatchPipeline);compute.setBindGroup(0,this.interfaceDispatchGroup);compute.dispatchWorkgroups(1);compute.end();
      compute=encoder.beginComputePass({label:"Extract water interface"});compute.setPipeline(this.extractionPipeline);compute.setBindGroup(0,this.extractionGroup);compute.dispatchWorkgroupsIndirect(this.interfaceDispatch,0);compute.end();
      compute=encoder.beginComputePass({label:"Finalize water interface"});compute.setPipeline(this.finalizePipeline);compute.setBindGroup(0,this.finalizeGroup);compute.dispatchWorkgroups(1);compute.end();this.extractedSimulationRevision=simulationRevision;
    }
    encoder.copyTextureToTexture({texture:presentation},{texture:this.sceneTexture},{width:presentation.width,height:presentation.height});
    const backPass=encoder.beginRenderPass({label:"Water back faces",colorAttachments:[{view:this.backTexture.createView(),clearValue:{r:1e20,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:this.backDepth.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"discard"}});backPass.setPipeline(this.backPipeline);backPass.setBindGroup(0,this.backGroup);backPass.setVertexBuffer(0,this.vertices);backPass.drawIndirect(this.indirect,0);backPass.end();
    if(!this.renderGroup)this.renderGroup=this.device.createBindGroup({layout:this.waterPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniforms}},{binding:1,resource:this.sceneTexture.createView()},{binding:2,resource:this.sampler},{binding:3,resource:this.backTexture.createView()},{binding:4,resource:{buffer:this.bodies}}]});
    const waterPass=encoder.beginRenderPass({label:"Refractive water surface",colorAttachments:[{view:presentation.createView(),loadOp:"load",storeOp:"store"}],depthStencilAttachment:{view:this.frontDepth.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"discard"}});waterPass.setPipeline(this.waterPipeline);waterPass.setBindGroup(0,this.renderGroup);waterPass.setVertexBuffer(0,this.vertices);waterPass.drawIndirect(this.indirect,0);waterPass.end();
  }

  destroy():void {this.destroyTargets();this.vertices.destroy();this.vertexCounter.destroy();this.indirect.destroy();this.interfaceBricks.destroy();this.interfaceCounter.destroy();this.interfaceDispatch.destroy();}
}
