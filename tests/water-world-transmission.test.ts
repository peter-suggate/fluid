import assert from "node:assert/strict";
import test from "node:test";
import { createWaterWorldReflectionWGSL } from "../lib/svo/features/shading/water-world-reflections";
import { waterVerifiedLiquidIntervalWGSL } from "../lib/core/water-transmission.wgsl";

const gpu={skip:!process.env.WEBGPU_NODE_MODULE};
test("GPU world transmission traces an offscreen opaque receiver and distinguishes unresolved queries",gpu,async()=>{
 const {withUniformDevice}=await import("./helpers/uniform-geometric");
 const {drySceneVertexShader}=await import("../lib/svo/features/shading/program");
 const full=createWaterWorldReflectionWGSL(),entry=full.slice(full.indexOf("var<private> waterWorldTraversalExhausted:bool;"));
 // Camera (0,4,10), aimed at (0,2,0), vertical tanHalfFov=.1: receiver
 // (0,0,-1.5) lies below the screen. A clamped screen sample cannot locate it.
 const f=[0,-2/Math.sqrt(104),-10/Math.sqrt(104)],up=[0,-f[2]!,f[1]!],q=[0,-4,-11.5];
 const depth=q.reduce((v,x,i)=>v+x*f[i]!,0),vertical=q.reduce((v,x,i)=>v+x*up[i]!,0);
 assert.ok(.5-vertical/depth/.1*.5>1);
 await withUniformDevice("offscreen water opaque transmission",async device=>{
  const stub=`const DRY_MISS=1e20;const DRY_OWNER_NONE=0xffffffffu;var<private> dryVisibilityIgnoredBody:u32;var<private> dryDerivedPageFailure:u32;var<private> dryVisibilityFailure:u32;
   struct U{cameraPosition:vec4f}const uniforms=U(vec4f(0,4,10,0));struct Mapping{cellSize:vec3f}struct Dry{mapping:Mapping}const dry=Dry(Mapping(vec3f(.25)));
   struct Hit{t:f32}fn traceOpaqueScene(ro:vec3f,rd:vec3f)->Hit{if(ro.x>2.5){return Hit(DRY_MISS);}if(ro.x>1.5){waterWorldTraversalExhausted=true;return Hit(DRY_MISS);}return Hit(-ro.y/rd.y);}
   fn shadeDrySurface(hit:Hit,ro:vec3f,rd:vec3f)->vec3f{return vec3f(.08,.1,.12);}`;
  const module=device.createShaderModule({code:stub+entry});assert.deepEqual((await module.getCompilationInfo()).messages.filter(x=>x.type==="error"),[]);
  const pipeline=await device.createRenderPipelineAsync({layout:"auto",vertex:{module:device.createShaderModule({code:drySceneVertexShader}),entryPoint:"vertexMain"},fragment:{module,entryPoint:"waterWorldTransmissionMain",targets:[{format:"rgba32float"}]},primitive:{topology:"triangle-list"}});
  const resources:GPUTexture[]=[];
  const texture=(values:number[])=>{const t=device.createTexture({size:[4,1],format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});resources.push(t);device.queue.writeTexture({texture:t},new Float32Array(values),{bytesPerRow:64},[4,1]);return t.createView();};
  const position=texture([0,2,0,1,1,2,0,1,2,2,0,1,3,2,0,1]),request=texture([0,-.8,-.6,1,0,-.8,-.6,0,0,-.8,-.6,1,0,-.8,-.6,1]);
  const target=device.createTexture({size:[4,1],format:"rgba32float",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});resources.push(target);
  const read=device.createBuffer({size:256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(1),entries:[{binding:0,resource:position},{binding:2,resource:request}]});const e=device.createCommandEncoder(),p=e.beginRenderPass({colorAttachments:[{view:target.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:0}}]});p.setPipeline(pipeline);p.setBindGroup(1,group);p.draw(3);p.end();e.copyTextureToBuffer({texture:target},{buffer:read,bytesPerRow:256},[4,1]);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const v=new Float32Array(read.getMappedRange()).slice(0,16);read.unmap();
   [.08,.1,.12,2.5].forEach((expected,i)=>assert.ok(Math.abs(v[i]!-expected)<1e-5));assert.deepEqual([...v.slice(4,8)],[0,0,0,0]);assert.equal(v[11],-2);assert.equal(v[15],-3);
  }finally{resources.forEach(t=>t.destroy());read.destroy();}
 });
});

test("GPU actual liquid texture validates the complete opaque path and rejects gaps, domain escape and budget exhaustion",gpu,async()=>{
 const {withUniformDevice}=await import("./helpers/uniform-geometric");
 await withUniformDevice("verified liquid path",async device=>{
  const source=`struct U{container:vec4f}const u=U(vec4f(16,4,16,0));var<private> availableBudget:u32=128u;fn contactBound(n:u32)->u32{return min(n,availableBudget);}
   @group(0) @binding(0) var field:texture_3d<f32>;@group(0) @binding(1) var<storage,read_write> output:array<vec4u,3>;
   fn contactFluidValue(p:vec3f)->f32{let index=vec3i(floor((p-vec3f(-8,0,-8))/.25));return textureLoad(field,index,0).r;}
   ${waterVerifiedLiquidIntervalWGSL}
   @compute @workgroup_size(1) fn main(){output[0]=vec4u(u32(waterVerifiedLiquidInterval(vec3f(1,2,0),vec3f(0,-.8,-.6),2.5,.25)),u32(waterVerifiedLiquidInterval(vec3f(-1,2,0),vec3f(0,-.8,-.6),2.5,.25)),u32(waterVerifiedLiquidInterval(vec3f(7.9,1,0),vec3f(1,0,0),2,.25)),u32(waterVerifiedLiquidInterval(vec3f(1,2,0),vec3f(0,-1,0),100,.25)));
    output[1]=vec4u(u32(waterWorldTransmissionEligible(false,true,3,1)),u32(waterWorldTransmissionEligible(false,true,.5,1)),u32(waterWorldTransmissionEligible(true,false,3,1)),u32(waterWorldTransmissionEligible(false,false,3,1)));
    availableBudget=2u;output[2]=vec4u(u32(waterVerifiedLiquidInterval(vec3f(1,2,0),vec3f(0,-.8,-.6),2.5,.25)),0,0,0);}`;
  const module=device.createShaderModule({code:source});assert.deepEqual((await module.getCompilationInfo()).messages.filter(x=>x.type==="error"),[]);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}}),field=device.createTexture({size:[64,16,64],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
  const values=new Float32Array(64*16*64);for(let z=0;z<64;z++)for(let y=0;y<16;y++)for(let x=0;x<64;x++)values[(z*16+y)*64+x]=y<8&&!(x<32&&y>=3&&y<=4)?1:0;
  device.queue.writeTexture({texture:field},values,{bytesPerRow:256,rowsPerImage:16},[64,16,64]);const out=device.createBuffer({size:48,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:48,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:field.createView()},{binding:1,resource:{buffer:out}}]}));p.dispatchWorkgroups(1);p.end();e.copyBufferToBuffer(out,0,read,0,48);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);assert.deepEqual([...new Uint32Array(read.getMappedRange())],[1,0,0,0,0,1,0,1,0,0,0,0]);read.unmap();}finally{field.destroy();out.destroy();read.destroy();}
 });
});

test("production transmission verification initializes and samples the resident mixed nodal lattice",gpu,async()=>{
 const {withUniformDevice}=await import("./helpers/uniform-geometric");
 const {waterTransmissionWGSL}=await import("../lib/core/water-transmission.wgsl");
 const {uniformMixedPresentationWGSL}=await import("../lib/methods/uniform/uniform-mixed-presentation.wgsl");
 const {uniformLatticeAddress,uniformWideVertexTexel}=await import("../lib/methods/uniform/uniform-lattice-address");
 const {createUniformMixedLayout}=await import("../lib/methods/uniform/uniform-mixed-layout");
 const {UniformMixedOwnership}=await import("../lib/methods/uniform/uniform-mixed-ownership");
 const {UniformDetailStorage}=await import("../lib/methods/uniform/uniform-detail-fields");
 await withUniformDevice("mixed liquid transmission entry",async device=>{
  const dimensions:[number,number,number]=[64,32,64],address=uniformLatticeAddress(dimensions);
  const layout=createUniformMixedLayout({dimensions,cellSize_m:[.25,.25,.25],origin_m:{x:-8,y:0,z:-8}},[],4,undefined,address);
  const ownership=new UniformMixedOwnership(device,layout,true,0),storage=new UniformDetailStorage(device,address);storage.attach(ownership);
  const resources:GPUTexture[]=[];
  const coarse=device.createTexture({size:[...address.wideVertexTexture],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});resources.push(coarse);
  const coarseValues=new Float32Array(coarse.width*coarse.height*coarse.depthOrArrayLayers).fill(.5);
  for(let z=0;z<=16;z++)for(let y=0;y<=8;y++)for(let x=0;x<=16;x++){const q=uniformWideVertexTexel(address,x,y,z,[0,0,0]);coarseValues[q[0]+coarse.width*(q[1]+coarse.height*q[2])]=-.5;}
  device.queue.writeTexture({texture:coarse},coarseValues,{bytesPerRow:coarse.width*4,rowsPerImage:coarse.height},[coarse.width,coarse.height,coarse.depthOrArrayLayers]);
  const query=device.createTexture({size:[1,1,2],format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});resources.push(query);
  device.queue.writeTexture({texture:query,origin:[0,0,1]},new Float32Array([.08,.1,.12,2]),{bytesPerRow:16},[1,1,1]);
  const dummy=device.createTexture({size:[1,1],format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING});resources.push(dummy);
  const targets=Array.from({length:2},()=>{const t=device.createTexture({size:[1,1],format:"rgba32float",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});resources.push(t);return t;});
  const verification=waterTransmissionWGSL.slice(waterTransmissionWGSL.indexOf("fn waterVerifyTransmission("),waterTransmissionWGSL.indexOf("@fragment fn transmissionRequestMain"));
  const entry=waterTransmissionWGSL.slice(waterTransmissionWGSL.indexOf("@fragment fn transmissionVerifyMain"));
  const code=`
   @group(0) @binding(1) var coarse:texture_3d<f32>;@group(0) @binding(2) var detail:texture_3d<f32>;
   @group(0) @binding(3) var frontReflections:texture_2d_array<f32>;@group(0) @binding(4) var rearReflections:texture_2d_array<f32>;
   @group(0) @binding(5) var frontPosition:texture_2d<f32>;@group(0) @binding(6) var rearFrontPosition:texture_2d<f32>;
   @group(0) @binding(7) var frontNormal:texture_2d<f32>;@group(0) @binding(8) var rearFrontNormal:texture_2d<f32>;
   ${uniformMixedPresentationWGSL(0,"detail","coarse")}
   struct U{container:vec4f,gridInfo:vec4f,cameraPosition:vec4f}const u=U(vec4f(16,8,16,0),vec4f(64,32,64,0),vec4f(0,8,0,0));
   fn contactBound(n:u32)->u32{return n;}
   fn contactFluidValue(p:vec3f)->f32{return .5-umSampleVertex((p-vec3f(-8,0,-8))/.25)/.25;}
   fn safePositionSample(t:texture_2d<f32>,uv:vec2f)->vec4f{return vec4f(0,4,0,1);}
   fn safeInterfaceSample(t:texture_2d<f32>,uv:vec2f)->vec4f{return vec4f(0,1,0,1);}
   fn waterIndexOfRefraction()->f32{return 1.333;}
   struct VOut{@builtin(position) position:vec4f,@location(0) uv:vec2f}
   struct ReflectionOut{@location(0) front:vec4f,@location(1) rear:vec4f}
   @vertex fn vertexMain(@builtin(vertex_index)i:u32)->VOut{var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));return VOut(vec4f(p[i],0,1),p[i]*.5+.5);}
   ${waterVerifiedLiquidIntervalWGSL}${verification}${entry}`;
  const read=device.createBuffer({size:256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   // Execute the actual fragment entry twice. Removing only its initialization
   // reproduces the production all-rejected failure with the same bound field.
   for(const initialize of [false,true]){
    const module=device.createShaderModule({code:initialize?code:code.replace("  udrInit();","")});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(x=>x.type==="error"),[]);
    const pipeline=await device.createRenderPipelineAsync({layout:"auto",vertex:{module,entryPoint:"vertexMain"},fragment:{module,entryPoint:"transmissionVerifyMain",targets:[{format:"rgba32float"},{format:"rgba32float"}]},primitive:{topology:"triangle-list"}});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:ownership.presentation},{binding:1,resource:coarse.createView()},{binding:2,resource:coarse.createView()},{binding:3,resource:query.createView({dimension:"2d-array"})},{binding:4,resource:query.createView({dimension:"2d-array"})},...[5,6,7,8].map(binding=>({binding,resource:dummy.createView()}))]});
    const e=device.createCommandEncoder(),p=e.beginRenderPass({colorAttachments:targets.map(t=>({view:t.createView(),loadOp:"clear" as const,storeOp:"store" as const,clearValue:{r:0,g:0,b:0,a:0}}))});p.setPipeline(pipeline);p.setBindGroup(0,group);p.draw(3);p.end();e.copyTextureToBuffer({texture:targets[0]!},{buffer:read,bytesPerRow:256},[1,1]);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const result=new Float32Array(read.getMappedRange()).slice(0,4);read.unmap();
    assert.equal(result[3],initialize?2:-4,initialize?"initialized mixed liquid accepts measured path":"uninitialized mixed lattice rejects the same path");
   }
  }finally{read.destroy();resources.forEach(t=>t.destroy());storage.destroy();ownership.destroy();}
 });
});
