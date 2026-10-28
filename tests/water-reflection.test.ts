import test from "node:test";
import assert from "node:assert/strict";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { waterReflectionWGSL } from "../lib/core/water-reflection.wgsl";

test("water reflection intersects actual depth at distance, skips foreground discontinuities and rejects sky and offscreen rays", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("water reflection geometric oracle", async device => {
    // Uniform camera and storage rays exercise the same dynamic shader path as
    // production; constant-folded multi-call Metal fixtures were intermittent.
    // Analytic perspective depth of a plane z=10, independent of the marcher.
    // Middle row is a sky strip, allowing a true miss through the same viewport.
    const module = device.createShaderModule({ code: `
      struct U { cameraPosition:vec4f,cameraTarget:vec4f,container:vec4f,viewport:vec4f }
      @group(0) @binding(1) var<uniform> u:U;
      const sceneTexture=0.0;
      fn cameraTanHalfFov()->f32{return u.viewport.z;}
      fn project(p:vec3f)->vec2f{let q=p-u.cameraPosition.xyz;let f=normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);let r=normalize(cross(vec3f(0,1,0),f));let up=cross(f,r);let z=dot(q,f);return vec2f(.5+.5*dot(q,r)/(z*u.viewport.z*u.viewport.w),.5-.5*dot(q,up)/(z*u.viewport.z));}
      fn resolvedDrySceneDepth(d:f32)->f32{return d;}
      fn safeSample(unused:f32,uv:vec2f)->vec4f{
        let p=vec3f((uv.x-.5)*20.0,(.5-uv.y)*20.0,10);
        var depth=select(length(p),65504.0,abs(uv.y-.5)<.02);
        // A foreground leaf projects across the march but lies in front of
        // the reflected ray. Reject that discontinuity and keep tracing.
        if(uv.x>.585&&uv.x<.59&&uv.y<.45){depth=length(p)*.3;}
        return vec4f(uv,0,depth);
      }
      fn waterEnvironmentRadiance(direction:vec3f)->vec3f{return vec3f(.1);}
      fn waterReflectionColor(uv:vec2f)->vec3f{return safeSample(0.0,uv).rgb;}
      fn waterReflectionDepth(uv:vec2f)->f32{return safeSample(0.0,uv).a;}
      ${waterReflectionWGSL}
      @group(0) @binding(0) var<storage,read_write> results:array<vec4f>;
      @group(0) @binding(2) var<storage,read> rays:array<vec4f>;
      @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
        if(id.x<4u){results[id.x]=waterReflectionHit(rays[id.x*2u].xyz,normalize(rays[id.x*2u+1u].xyz),.1);}
      }` });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(message => message.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const uniforms=device.createBuffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const rays=device.createBuffer({size:128,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(uniforms,0,new Float32Array([0,0,0,0,0,0,1,0,16,16,16,0,1000,700,1,1]));
    device.queue.writeBuffer(rays,0,new Float32Array([0,0,2,0,.3,.4,1,0, 0,0,2,0,2,.1,.1,0, 0,0,2,0,0,0,1,0, 0,0,12,0,.3,.4,-1,0]));
    const output = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } },{binding:1,resource:{buffer:uniforms}},{binding:2,resource:{buffer:rays}}] }));
      pass.dispatchWorkgroups(4); pass.end(); encoder.copyBufferToBuffer(output, 0, read, 0, 64);
      device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(read.getMappedRange()).slice(); read.unmap();
      assert.ok(Math.abs(values[0]! - .62) < .002, `unexpected reflection samples ${JSON.stringify([...values])}`);
      assert.ok(Math.abs(values[1]! - .34) < .002);
      assert.ok(Math.abs(values[2]! - 8 * Math.sqrt(1.25)) < .03, "hit the plane eight metres ahead, not a fixed near-water sample");
      assert.ok(values[3]! > .99);
      assert.equal(values[7], 0, "offscreen reflection must fall back to the environment");
      assert.equal(values[11], 0, "sky depth is not geometry");
      assert.ok(Math.abs(values[12]!-.53)<.002&&Math.abs(values[13]!-.46)<.002,"camera-facing ray retains its viewport-clipped hit");
      assert.ok(Math.abs(values[14]!-2*Math.sqrt(1.25))<.03);
      assert.ok(values[15]!>.99);
    } finally { uniforms.destroy();rays.destroy();output.destroy(); read.destroy(); }
  });
});

test("water carries the resolved scene sky including black environments and specular gain", async () => {
  const { packWaterSceneOptics, resolveWaterOptics, resolveWaterKeyLight } = await import("../lib/core/webgpu-lighting");
  const { svoEnvironmentLightingRecord } = await import("../lib/svo/features/radiance/svo-environment-lighting");
  const dark = svoEnvironmentLightingRecord("garden", 1, { lowerRadianceLinear: [0, 0, 0], upperRadianceLinear: [.01, .02, .03], accentRadianceLinear: [0, 0, 0], specularScale: .5, cloudCoverage: .75 });
  const packed = packWaterSceneOptics(resolveWaterOptics(), resolveWaterKeyLight(), undefined, dark);
  assert.deepEqual([...packed.slice(32,36)], [0, 0, 0, 1], "black authored radiance is enabled, never mistaken for missing data");
  assert.deepEqual([...packed.slice(36,40)], [...dark.upperRadianceLinear.map(Math.fround), .5]);
  assert.deepEqual([...packed.slice(40,44)], [0, 0, 0, 3]);
  assert.equal(packed[29], .75, "the water fallback uses the visible sky cloud coverage");
  const localKey = resolveWaterKeyLight({direction:[1,0,0],colorLinear:[1,0,0],intensity:100});
  const independentSky = packWaterSceneOptics(resolveWaterOptics(), localKey, undefined, dark);
  assert.deepEqual([...independentSky.slice(44,47)], dark.keyLightDirection.map(Math.fround), "local fixtures cannot reorient the reflected sky");
  assert.deepEqual([...independentSky.slice(48,51)], dark.keyLightColorLinear.map(value=>Math.fround(value*dark.sunIntensity)), "cloud lighting uses the environment sun, not the strongest nearby fixture");
  assert.equal(packWaterSceneOptics(resolveWaterOptics(), resolveWaterKeyLight())[35], 0, "callers without a rig retain legacy fallback");
});

test("reflection reconstruction rejects separate sheets and occluded interfaces", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { waterReflectionFilterWGSL } = await import("../lib/core/water-reflection-filter.wgsl");
  await withUniformDevice("reflection reconstruction", async device => {
    const module=device.createShaderModule({code:`
      const screenSpaceReflections=1u;
      struct U { cameraPosition:vec4f } const u=U(vec4f(0,2,0,0));
      fn cameraTanHalfFov()->f32{return .5;}
      fn waterRoughness()->f32{return .1;}
      fn waterEnvironmentRadiance(d:vec3f)->vec3f{return vec3f(0);}
      ${waterReflectionFilterWGSL}
      @group(0) @binding(0) var colors:texture_2d_array<f32>;
      @group(0) @binding(1) var positions:texture_2d<f32>;
      @group(0) @binding(2) var normals:texture_2d<f32>;
      @group(0) @binding(3) var<storage,read_write> result:vec4f;
      @compute @workgroup_size(1) fn main(){result=vec4f(filteredWaterReflection(colors,positions,normals,vec2f(.5),vec3f(0,0,4),vec3f(0,1,0),vec3f(0,1,0),.1),1);}
    `});
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const textures=Array.from({length:3},()=>device.createTexture({size:[3,3],format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST}));
    const pixels=new Float32Array(36),positions=new Float32Array(36),normals=new Float32Array(36);
    for(let i=0;i<9;i++){pixels.set(i===4?[1,0,0,4]:[0,0,10,4],i*4);positions.set([0,i===4?0:5,4,1],i*4);normals.set([0,1,0,1],i*4);}
    [pixels,positions,normals].forEach((data,i)=>device.queue.writeTexture({texture:textures[i]!},data,{bytesPerRow:48},[3,3]));
    const output=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const read=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{
      for (const mode of ["separate sheet", "occluded interface"]) {
        if (mode === "occluded interface") {
          for(let i=0;i<9;i++){positions[i*4+1]=0;pixels[i*4+3]=i===4?4:-1;}
          device.queue.writeTexture({texture:textures[0]!},pixels,{bytesPerRow:48},[3,3]);
          device.queue.writeTexture({texture:textures[1]!},positions,{bytesPerRow:48},[3,3]);
        }
        const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);
        p.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[...textures.map((t,binding)=>({binding,resource:t.createView({dimension:binding===0?"2d-array":"2d"})})),{binding:3,resource:{buffer:output}}]}));p.dispatchWorkgroups(1);p.end();
        e.copyBufferToBuffer(output,0,read,0,16);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
        assert.deepEqual([...new Float32Array(read.getMappedRange())],[1,0,0,1], mode);read.unmap();
      }
    }finally{textures.forEach(t=>t.destroy());output.destroy();read.destroy();}
  });
});

test("pixel-boundary reflection traversal retains isolated foliage and tilted pixel-centre depths", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("discrete reflection visibility", async device => {
    const width=1000,height=700,count=128;
    const module=device.createShaderModule({code:`
      struct U { cameraPosition:vec4f,cameraTarget:vec4f,container:vec4f,viewport:vec4f }
      @group(0) @binding(3) var<uniform> u:U;
      fn cameraTanHalfFov()->f32{return u.viewport.z;}
      fn project(p:vec3f)->vec2f{let q=p-u.cameraPosition.xyz;let f=normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);let r=normalize(cross(vec3f(0,1,0),f));let up=cross(f,r);let z=dot(q,f);return vec2f(.5+.5*dot(q,r)/(z*u.viewport.z*u.viewport.w),.5-.5*dot(q,up)/(z*u.viewport.z));}
      @group(0) @binding(0) var depths:texture_2d<f32>;
      @group(0) @binding(1) var<storage,read> rays:array<vec4f>;
      @group(0) @binding(2) var<storage,read_write> results:array<vec4f>;
      fn waterReflectionDepth(uv:vec2f)->f32{return textureLoad(depths,clamp(vec2i(uv*vec2f(1000,700)),vec2i(0),vec2i(999,699)),0).r;}
      fn waterReflectionColor(uv:vec2f)->vec3f{return vec3f(.02);}
      fn waterEnvironmentRadiance(direction:vec3f)->vec3f{return vec3f(1);}
      ${waterReflectionWGSL}
      @compute @workgroup_size(32) fn main(@builtin(global_invocation_id) id:vec3u){if(id.x<128u){results[id.x]=waterReflectionHit(rays[id.x*2u].xyz,rays[id.x*2u+1u].xyz,.1);}}
    `});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const texture=device.createTexture({size:[width,height],format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    const uniforms=device.createBuffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(uniforms,0,new Float32Array([0,0,0,0,0,0,1,0,16,16,16,0,1000,700,1,1]));
    const input=device.createBuffer({size:count*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const output=device.createBuffer({size:count*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const read=device.createBuffer({size:count*16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:texture.createView()},{binding:1,resource:{buffer:input}},{binding:2,resource:{buffer:output}},{binding:3,resource:{buffer:uniforms}}]});
    let seed=453;const random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/2**32);
    try {
      for(const plane of [[0,0,1],[.8,0,.2]] as const){
        const pixels=new Float32Array(width*height);
        for(let y=0;y<height;y++)for(let x=0;x<width;x++){
          const r=[((x+.5)/width-.5)*2,(.5-(y+.5)/height)*2,1],l=Math.hypot(...r);
          const depth=10*l/(plane[0]*r[0]!+plane[2]);
          pixels[x+y*width]=(x+y)%4===0&&depth>0?depth:65504;
        }
        const rays=new Float32Array(count*8);let j=0;
        while(j<count){
          const x=Math.floor(100+random()*800),y=Math.floor(70+random()*560);
          if((x+y)%4!==0)continue;
          const r=[((x+.5)/width-.5)*2,(.5-(y+.5)/height)*2,1],z=10/(plane[0]*r[0]!+plane[2]);
          if(z<=0||z>40)continue;
          const target=r.map(v=>v*z),origin=[(random()-.5)*1.5,(random()-.5)*1.5,2+random()*3];
          const d=target.map((v,k)=>v-origin[k]!),length=Math.hypot(...d);
          if(length>55)continue;
          rays.set([...origin,0,...d.map(v=>v/length),0],j++*8);
        }
        device.queue.writeTexture({texture},pixels,{bytesPerRow:width*4},[width,height]);device.queue.writeBuffer(input,0,rays);
        const e=device.createCommandEncoder(),pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(count/32);pass.end();e.copyBufferToBuffer(output,0,read,0,count*16);device.queue.submit([e.finish()]);
        await read.mapAsync(GPUMapMode.READ);const hits=new Float32Array(read.getMappedRange()).slice();read.unmap();
        const misses=Array.from({length:count},(_,i)=>i).filter(i=>hits[i*4+3]!<=0);
        assert.deepEqual(misses,[],`isolated one-pixel leaf hits on plane ${plane} must not become bright sky`);
      }
    }finally{uniforms.destroy();texture.destroy();input.destroy();output.destroy();read.destroy();}
  });
});
