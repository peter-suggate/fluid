import assert from "node:assert/strict";
import test from "node:test";
import { createWaterWorldReflectionWGSL, SvoWaterWorldReflections } from "../lib/svo/features/shading/water-world-reflections";

test("world reflections use bounded world traversal without camera-space receiver caches",()=>{
  const shader=createWaterWorldReflectionWGSL();
  assert.match(shader,/let leafBudget=256u;/);
  assert.match(shader,/waterWorldTraversalExhausted=true/);
  assert.match(shader,/let hit=traceOpaqueScene\(ro,rd\)/);
  assert.match(shader,/if\(screen.a!=0.0\)\{return screen;\}/);
  assert.doesNotMatch(shader,/@group\(2\)/);
});

test("GPU optional world reflection pipeline fits production bindings",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  const {sparseVoxelDrySceneBindGroupLayoutEntries}=await import("../lib/svo/pipeline/webgpu-svo-dry-scene");
  await withUniformDevice("world reflection pipeline",async device=>{
    const pass=new SvoWaterWorldReflections(device,device.createBindGroupLayout({entries:sparseVoxelDrySceneBindGroupLayoutEntries("raster-primary")}),"raster-primary","bounds");
    assert.equal(pass.status.state,"idle");const first=pass.ensure();assert.equal(first,pass.ensure());await first;
    assert.equal(pass.status.state,"ready");pass.destroy();
  });
});

test("GPU reflection entry preserves SSR hits, shades world hits, and distinguishes miss/exhaustion/hidden water",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  const {drySceneVertexShader}=await import("../lib/svo/features/shading/program");
  const full=createWaterWorldReflectionWGSL();const entry=full.slice(full.indexOf("var<private> waterWorldTraversalExhausted:bool;"));
  await withUniformDevice("world reflection entry",async device=>{
    const stub=`
      const DRY_MISS=1e20;const DRY_OWNER_NONE=0xffffffffu;var<private> dryVisibilityIgnoredBody:u32;var<private> dryDerivedPageFailure:u32;var<private> dryVisibilityFailure:u32;
      struct U{cameraPosition:vec4f}const uniforms=U(vec4f(0,1,0,0));
      struct Mapping{cellSize:vec3f}struct Dry{mapping:Mapping}const dry=Dry(Mapping(vec3f(.1)));
      struct Hit{t:f32}
      fn traceOpaqueScene(ro:vec3f,rd:vec3f)->Hit{
        if(ro.x>3.5&&ro.x<4.5){waterWorldTraversalExhausted=true;return Hit(DRY_MISS);}
        if(ro.x>5.5){return Hit(3);}
        if(ro.x>2.5){return Hit(DRY_MISS);}
        // A plane beyond the source water point, unseen in the supplied SSR.
        return Hit((-4.0-ro.z)/rd.z);
      }
      fn shadeDrySurface(hit:Hit,ro:vec3f,rd:vec3f)->vec3f{if(ro.x>5.5&&ro.x<6.5){dryDerivedPageFailure=1u;return vec3f(0);}
        if(ro.x>6.5){dryVisibilityFailure=1u;return vec3f(0);}
        return vec3f(.25,.5,.75);}
    `;
    const module=device.createShaderModule({code:stub+entry});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const pipeline=await device.createRenderPipelineAsync({layout:"auto",vertex:{module:device.createShaderModule({code:drySceneVertexShader}),entryPoint:"vertexMain"},fragment:{module,entryPoint:"waterWorldReflectionMain",targets:[{format:"rgba32float"}]},primitive:{topology:"triangle-list"}});
    const textures:GPUTexture[]=[];
    function texture(values:number[]){const t=device.createTexture({size:[8,1],format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});textures.push(t);device.queue.writeTexture({texture:t},new Float32Array(values),{bytesPerRow:128},{width:8,height:1});return t.createView();}
    const positions=texture(Array.from({length:8},(_,i)=>[i,0,-2,i===5?0:1]).flat());
    const normals=texture(Array.from({length:8},()=>[0,1,0,1]).flat());
    const screens=texture([[.1,.2,.3,3],[.4,.5,.6,-1],[.7,.8,.9,0],[.2,.3,.4,0],[.3,.4,.5,0],[.6,.5,.4,0],[.2,.2,.2,0],[.4,.4,.4,0]].flat());
    const output=device.createTexture({size:[8,1],format:"rgba32float",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});textures.push(output);
    const read=device.createBuffer({size:256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(1),entries:[{binding:0,resource:positions},{binding:1,resource:normals},{binding:2,resource:screens}]});
      const encoder=device.createCommandEncoder(),pass=encoder.beginRenderPass({colorAttachments:[{view:output.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:0}}]});pass.setPipeline(pipeline);pass.setBindGroup(1,group);pass.draw(3);pass.end();encoder.copyTextureToBuffer({texture:output},{buffer:read,bytesPerRow:256},{width:8,height:1});device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const values=new Float32Array(read.getMappedRange()).slice(0,32);read.unmap();
      const near=(at:number,expected:number[])=>expected.forEach((v,i)=>assert.ok(Math.abs(values[at*4+i]!-v)<1e-5,`pixel${at} channel${i}: ${values[at*4+i]} vs ${v}`));
      near(0,[.1,.2,.3,3]);near(1,[.4,.5,.6,-1]);near(2,[.25,.5,.75,3]);near(3,[.2,.3,.4,0]);near(4,[.3,.4,.5,-2]);near(5,[.6,.5,.4,0]);near(6,[.2,.2,.2,-2]);near(7,[.4,.4,.4,-2]);
    }finally{read.destroy();textures.forEach(t=>t.destroy());}
  },["float32-filterable"]);
});
