import assert from "node:assert/strict";
import test from "node:test";
import { SVO_LIGHT_KINDS, SVO_LIGHT_RECORD_WORDS } from "../lib/svo/contracts/svo-light-abi";
import { rasterVisibilityLayers, rasterVisibilityPackingWGSL, rasterVisibilityTilesWGSL } from "../lib/svo/features/lighting-visibility/raster-light-visibility";

test("GPU cached light reconstruction preserves exact pixels and weighted atlas borders", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { withUniformDevice } = await import("./helpers/uniform-geometric");
  const { readMixedBuffer } = await import("./helpers/uniform-mixed-native-fields");
  const { rasterLightVisibilityWGSL: source, rasterAtlasLayoutWGSL } = await import("../lib/svo/features/lighting-visibility/raster-light-visibility");
  const section = (start:string,end:string) => source.slice(source.indexOf(start),source.indexOf(end,source.indexOf(start)));
  await withUniformDevice("cached light atlas coordinates", async device => {
    for (const monochrome of [false,true]) for (const [width,height] of [[7,5],[8,6]] as const) {
      const samples=8,layers=monochrome?Math.ceil(samples/3):samples;
      const textures = [
        device.createTexture({size:[width,height],format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING}),
        device.createTexture({size:[Math.ceil(width/2),Math.ceil(height/2)],format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING}),
        device.createTexture({size:[width,height],format:"rg32uint",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST}),
        device.createTexture({size:[width,height,layers],format:"r32uint",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST}),
      ];
      const output=device.createBuffer({size:width*height*samples*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      try {
        const module=device.createShaderModule({code:`
@group(0) @binding(0) var drySplitGeometryRead:texture_2d<f32>;
@group(0) @binding(1) var rasterContact:texture_2d<f32>;
@group(0) @binding(2) var rasterGuide:texture_2d<u32>;
@group(0) @binding(3) var rasterLightVisibility:texture_2d_array<u32>;
@group(0) @binding(4) var<storage,read_write> output:array<vec4f>;
@group(0) @binding(5) var rasterSunSampler:sampler;
var<private> rasterPixel:vec2f;
var<private> rasterVisibilityWeights:vec4f;
var<private> rasterVisibilityWeight:f32;
var<private> rasterVisibilityCoordinates:array<vec2i,4>;
var<private> rasterVisibilityExactCoordinate:vec2i;
var<private> rasterVisibilityGatherable:bool;
var<private> rasterVisibilityQuadUV:vec2f;
fn rasterVisibilityStride()->u32{return 2u;}
fn rasterVisibilityMonochrome()->bool{return ${monochrome};}
${rasterAtlasLayoutWGSL}${rasterVisibilityPackingWGSL}
${section("fn rasterAtlasCoordinate", "// Shared by the edge producer")}
${section("fn rasterCachedLightVisibility", "fn rasterVisibilityHit")}
${section("fn rasterLoadVisibilityGuide", "@compute @workgroup_size(8,8) fn rasterLightsMain")}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
  rasterPixel=vec2f(id.xy)+.5;let guide=rasterLoadVisibilityGuide();
  // Revisit and skip triplets, then read all samples in the production order.
  let skipped=rasterCachedLightVisibility(7u)+rasterCachedLightVisibility(3u)+rasterCachedLightVisibility(1u);
  for(var sample=0u;sample<${samples}u;sample++){output[(sample*${height}u+id.y)*${width}u+id.x]=vec4f(rasterCachedLightVisibility(sample),1);}

}`});
        assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
        const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
        const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[...textures.map((t,binding)=>({binding,resource:t.createView({dimension:binding===3?"2d-array":"2d"})})),{binding:4,resource:{buffer:output}},{binding:5,resource:device.createSampler({})}]});
        const guide=new Uint32Array(width*height*2),atlas=new Uint32Array(width*height*layers);
        const value=(x:number,y:number,layer:number)=>(layer*127+y*31+x*13)%1024;
        for(let y=0;y<height;y++)for(let x=0;x<width;x++){
          guide[(y*width+x)*2+1]=(x+y)%3===0?0:(51|(102<<8)|(102<<24))>>>0;
          for(let sample=0;sample<samples;sample++){
            const v=value(x,y,sample),layer=monochrome?Math.floor(sample/3):sample;
            if(monochrome)atlas[(layer*height+y)*width+x]!|=v<<((sample%3)*10);
            else atlas[(layer*height+y)*width+x]=v|((1023-v)<<10)|(v<<20);
          }
        }
        device.queue.writeTexture({texture:textures[2]!},guide,{bytesPerRow:width*8},[width,height]);
        device.queue.writeTexture({texture:textures[3]!},atlas,{bytesPerRow:width*4,rowsPerImage:height},[width,height,layers]);
        const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(width,height);p.end();device.queue.submit([e.finish()]);
        const actual=await readMixedBuffer(device,output);
        const compact=(c:number,d:number)=>Math.floor(c/2)+(c===Math.min(Math.floor(c/2)*2+1,d-1)?0:Math.ceil(d/2));
        for(let layer=0;layer<samples;layer++)for(let y=0;y<height;y++)for(let x=0;x<width;x++){
          const expected=(x+y)%3===0?value(compact(x,width),compact(y,height),layer):[.2,.4,0,.4].reduce((sum,w,i)=>sum+w*value(Math.max(0,Math.min(Math.ceil(width/2)-1,Math.floor((x-1)/2)+(i%2))),Math.max(0,Math.min(Math.ceil(height/2)-1,Math.floor((y-1)/2)+Math.floor(i/2))),layer),0);
          const at=((layer*height+y)*width+x)*4;
          for(const [channel,v] of [expected,monochrome?expected:1023-expected,expected].entries())assert.ok(Math.abs(actual[at+channel]!-v/1023)<2e-6,`${width}x${height}, ${x},${y}, light ${layer}, channel ${channel}`);
        }
      } finally { textures.forEach(t=>t.destroy());output.destroy(); }
    }
  });
});

test("visibility allocation covers the existing light sample budget, including GI", () => {
  const kinds = ["directional", "rectangleArea", "point", "spot", "sphereArea"] as const;
  const records = new Uint32Array(kinds.length * SVO_LIGHT_RECORD_WORDS);
  kinds.forEach((kind, i) => { records[i * SVO_LIGHT_RECORD_WORDS + 24] = SVO_LIGHT_KINDS[kind]; });
  assert.equal(rasterVisibilityLayers(records, 8, 2, false), 8);
  assert.equal(rasterVisibilityLayers(records, 8, 2, true), 8);
  assert.equal(rasterVisibilityLayers(records, 4, 2, false), 6);
  assert.equal(rasterVisibilityLayers(records.subarray(0, 56), 8, 2, false), 3);
  assert.equal(rasterVisibilityLayers(undefined, 8, 2, false), 1);
});

test("GPU visibility queue preserves thin receivers and clears after an empty frame", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { withUniformDevice } = await import("./helpers/uniform-geometric");
  await withUniformDevice("visibility tiles", async device => {
    const geometry = device.createTexture({ size: [35, 19], format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const queue = device.createBuffer({ size: 40, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const read = device.createBuffer({ size: 40, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
      const module = device.createShaderModule({ code: `
@group(1) @binding(0) var drySplitGeometryRead:texture_2d<f32>;
const DRY_MISS=1e20;
struct Lighting { control:vec4u }
const rasterLighting=Lighting(vec4u(0,0,0,3));
fn rasterVisibilityGeometryAt(c:vec2i)->vec4f{return textureLoad(drySplitGeometryRead,c,0);}
${rasterVisibilityTilesWGSL}` });
      const empty = device.createBindGroupLayout({ entries: [] });
      const geometryLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } }] });
      const queueLayout = device.createBindGroupLayout({ entries: [{ binding: 17, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
      const layout = device.createPipelineLayout({ bindGroupLayouts: [empty, geometryLayout, queueLayout] });
      const classify = await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: "rasterClassifyTiles" } });
      const finalize = await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: "rasterFinalizeTiles" } });
      const group0 = device.createBindGroup({ layout: empty, entries: [] });
      const group1 = device.createBindGroup({ layout: geometryLayout, entries: [{ binding: 0, resource: geometry.createView() }] });
      const group2 = device.createBindGroup({ layout: queueLayout, entries: [{ binding: 17, resource: { buffer: queue } }] });
      for (const occupied of [true, false]) {
        const pixels = new Float32Array(35 * 19 * 4);
        for (let i = 3; i < pixels.length; i += 4) pixels[i] = 1e20;
        // Even/even thin features never appear in the odd/odd reduced lattice.
        if (occupied) for (const [x, y] of [[0, 0], [16, 0], [34, 18]]) pixels.set([0, 1, 0, 2], (x! + 35 * y!) * 4);
        device.queue.writeTexture({ texture: geometry }, pixels, { bytesPerRow: 35 * 16 }, [35, 19]);
        const encoder = device.createCommandEncoder();encoder.clearBuffer(queue, 0, 4);
        const pass = encoder.beginComputePass();pass.setBindGroup(0, group0);pass.setBindGroup(1, group1);pass.setBindGroup(2, group2);
        pass.setPipeline(classify);pass.dispatchWorkgroups(3, 2);pass.setPipeline(finalize);pass.dispatchWorkgroups(1);pass.end();
        encoder.copyBufferToBuffer(queue, 0, read, 0, 40);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
        const words = new Uint32Array(read.getMappedRange()).slice();read.unmap();
        assert.deepEqual([...words.subarray(0, 4)], occupied ? [3, 3, 1, 1] : [0, 0, 1, 1]);
        if (occupied) assert.deepEqual([...words.subarray(4, 7)].sort(), [0, 1, 5]);
      }
    } finally { geometry.destroy();queue.destroy();read.destroy(); }
  });
});

test("GPU visibility packing retains coloured transmission within RGB10 error", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { withUniformDevice } = await import("./helpers/uniform-geometric");
  const { readMixedBuffer } = await import("./helpers/uniform-mixed-native-fields");
  await withUniformDevice("visibility RGB10", async device => {
    const output = device.createBuffer({ size: 1025 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    try {
      const module = device.createShaderModule({ code: `${rasterVisibilityPackingWGSL}
@group(0) @binding(0) var<storage,read_write> result:array<vec4f>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
  if(id.x>=1025u){return;}let x=f32(id.x)/1024.;
  result[id.x]=vec4f(rasterUnpackVisibility(rasterPackVisibility(vec3f(x,1.-x,x*x))),0);
}` });
      const layout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
      const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: "main" } });
      const group = device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: output } }] });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0, group);pass.dispatchWorkgroups(17);pass.end();device.queue.submit([encoder.finish()]);
      const values = await readMixedBuffer(device, output);
      for (let i = 0; i <= 1024; i++) {
        const x = i / 1024;
        [x, 1 - x, x * x].forEach((expected, c) => assert.ok(Math.abs(values[i * 4 + c]! - expected) <= 1 / 2046 + 1e-7));
      }
    } finally { output.destroy(); }
  });
});

test("GPU exact-edge dispatch handles zero, partial groups, and more than 65535 groups", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { withUniformDevice } = await import("./helpers/uniform-geometric");
  const { rasterExactQueueWGSL } = await import("../lib/svo/features/lighting-visibility/raster-light-visibility");
  await withUniformDevice("exact visibility queue", async device => {
    const queue = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const empty = device.createBindGroupLayout({ entries: [] });
      const layout = device.createBindGroupLayout({ entries: [{ binding: 20, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
      const module = device.createShaderModule({ code: rasterExactQueueWGSL });
      const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [empty, empty, layout] }), compute: { module, entryPoint: "rasterExactFinalize" } });
      const emptyGroup = device.createBindGroup({ layout: empty, entries: [] });
      const group = device.createBindGroup({ layout, entries: [{ binding: 20, resource: { buffer: queue } }] });
      for (const [count, x, y] of [[0, 0, 1], [1, 1, 1], [64, 1, 1], [65, 2, 1], [65535 * 64 + 1, 65535, 2], [3474 * 1882, 65535, 2], [0, 0, 1]]) {
        device.queue.writeBuffer(queue, 0, new Uint32Array([count!]));
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, emptyGroup); pass.setBindGroup(1, emptyGroup); pass.setBindGroup(2, group); pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(queue, 0, read, 0, 16); device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
        assert.deepEqual([...new Uint32Array(read.getMappedRange())], [count, x, y, 1]); read.unmap();
      }
    } finally { queue.destroy(); read.destroy(); }
  });
});

test("GPU exact tiles retain every receiver once and leave AO payloads intact", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { withUniformDevice } = await import("./helpers/uniform-geometric");
  const { rasterExactQueueWGSL, rasterExactTileQueueWGSL } = await import("../lib/svo/features/lighting-visibility/raster-light-visibility");
  await withUniformDevice("exact tile queue", async device => {
    const size=(4+768*2)*4;
    const queue=device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
    const read=device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    const mode=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    try {
      const module=device.createShaderModule({code:`${rasterExactQueueWGSL}${rasterExactTileQueueWGSL}
@group(0) @binding(0) var<uniform> testMode:vec4u;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
  rasterBeginExactTile(lane);
  for(var i=0u;i<4u;i++){
    let coordinate=group.x*256u+lane*4u+i;
    if(testMode.x!=0u&&(coordinate==511u||group.x==2u)){rasterAppendExactTile(coordinate);}
  }
  rasterFlushExactTile(lane);
}`});
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(message=>message.type==="error"),[]);
      const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
      const group0=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:mode}}]});
      const group1=device.createBindGroup({layout:pipeline.getBindGroupLayout(1),entries:[]});
      const group2=device.createBindGroup({layout:pipeline.getBindGroupLayout(2),entries:[{binding:20,resource:{buffer:queue}}]});
      const sentinel=new Uint32Array(size/4).fill(0xdeadbeef);device.queue.writeBuffer(queue,0,sentinel);
      for(const enabled of [1,0,1]){
        device.queue.writeBuffer(mode,0,new Uint32Array([enabled,0,0,0]));
        const encoder=device.createCommandEncoder();encoder.clearBuffer(queue,0,4);
        const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group0);pass.setBindGroup(1,group1);pass.setBindGroup(2,group2);pass.dispatchWorkgroups(3);pass.end();
        encoder.copyBufferToBuffer(queue,0,read,0,size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
        const words=new Uint32Array(read.getMappedRange());assert.equal(words[0],enabled?257:0);
        const coordinates=Array.from({length:words[0]!},(_,i)=>words[4+i*2]!);
        if(enabled){
          assert.deepEqual([...coordinates].sort((a,b)=>a-b),Array.from({length:257},(_,i)=>511+i));
          assert.ok(coordinates[0]===511||coordinates[256]===511,"each tile owns a contiguous range");
        }
        for(let i=0;i<768;i++)assert.equal(words[5+i*2],0xdeadbeef,"coordinate compaction preserves the separate AO word");
        read.unmap();
      }
    } finally {queue.destroy();read.destroy();mode.destroy();}
  });
});

test("GPU atlas permutation keeps representatives compact and exact receivers collision-free", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { withUniformDevice } = await import("./helpers/uniform-geometric");
  const { rasterAtlasLayoutWGSL } = await import("../lib/svo/features/lighting-visibility/raster-light-visibility");
  await withUniformDevice("compact visibility atlas", async device => {
    const count = 36 * 20, size = count * 8;
    const output = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const dimensions = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    try {
      const module = device.createShaderModule({ code: `${rasterAtlasLayoutWGSL}
@group(0) @binding(0) var<uniform> dimensions:vec4i;
@group(0) @binding(1) var<storage,read_write> output:array<vec2i>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
  let i=i32(id.x);if(i>=dimensions.x*dimensions.y){return;}
  output[i]=rasterCompactAtlasCoordinate(vec2i(i%dimensions.x,i/dimensions.x),dimensions.xy);
}` });
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: dimensions } }, { binding: 1, resource: { buffer: output } },
      ] });
      for (const [width, height] of [[1, 1], [1, 7], [6, 1], [2, 2], [5, 7], [35, 19], [36, 20]] as const) {
        device.queue.writeBuffer(dimensions, 0, new Int32Array([width, height, 0, 0]));
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(width * height / 64)); pass.end();
        encoder.copyBufferToBuffer(output, 0, read, 0, size); device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
        const values = new Int32Array(read.getMappedRange()); const seen = new Set<number>();
        for (let i = 0; i < width * height; i++) {
          const x = values[2 * i]!, y = values[2 * i + 1]!;
          assert.ok(x >= 0 && x < width && y >= 0 && y < height);
          seen.add(x + width * y);
          const sourceX = i % width, sourceY = Math.floor(i / width);
          if (sourceX === Math.min(Math.floor(sourceX / 2) * 2 + 1, width - 1)
            && sourceY === Math.min(Math.floor(sourceY / 2) * 2 + 1, height - 1)) {
            assert.ok(x < Math.ceil(width / 2) && y < Math.ceil(height / 2));
          }
        }
        assert.equal(seen.size, width * height); read.unmap();
      }
    } finally { output.destroy(); read.destroy(); dimensions.destroy(); }
  });
});

test("GPU monochrome producer packs all light samples with unchanged ten-bit precision", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { withUniformDevice } = await import("./helpers/uniform-geometric");
  const { readMixedBuffer } = await import("./helpers/uniform-mixed-native-fields");
  const { rasterLightVisibilityWGSL: source } = await import("../lib/svo/features/lighting-visibility/raster-light-visibility");
  const producer=source.slice(source.indexOf("fn rasterProduceLightVisibility("),source.indexOf("fn rasterLoadVisibilityGuide")).replaceAll("textureStore(rasterLightVisibilityOutput,","storeResult(");
  await withUniformDevice("monochrome light triplets",async device=>{
    const output=device.createBuffer({size:64*4*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    try {
      const module=device.createShaderModule({code:`
struct Hit{materialId:u32,ownerId:u32}
struct Light{identity:vec4u}
struct Lighting{metadata:vec4u,lights:array<Light,32>}
struct Dry{tuningCounts0:vec4u,tuningCounts1:vec4u,materialPublication:vec4u}
struct Uniforms{options:vec4f}
struct Sample{valid:u32,towardLight:vec3f,finiteDistance_m:f32}
var<private> dry:Dry;var<private> dryLighting:Lighting;var<private> uniforms:Uniforms;
var<private> dryVisibilityIgnoredBody:u32;var<private> drySampledReceiver:bool;var<private> rasterResolvedVisibility:vec2f;
const DRY_OWNER_NONE=65535u;const SVO_LIGHT_SPHERE_AREA=2u;const SVO_LIGHT_RECTANGLE_AREA=3u;const SVO_LIGHT_SPOT=4u;const RASTER_VISIBILITY_WATER_SHADOWS=false;
@group(0) @binding(0) var<storage,read_write> output:array<f32>;
fn rasterVisibilityHit(c:vec2i,g:vec4f)->Hit{return Hit(0u,12u);}
fn rasterWorldAt(c:vec2f,t:f32)->vec3f{return vec3f(c,0);}
fn dryGeometricNormal(h:Hit)->vec3f{return vec3f(0,0,1);}
fn drySampledMaterial(m:u32)->bool{return false;}
fn rasterSpotSlot(i:u32,p:vec3f)->i32{return -1;}
fn rasterVisibilityMonochrome()->bool{return true;}
fn dryLightSample(l:Light,i:u32,p:vec3f)->Sample{return Sample(1u,vec3f(0,0,1),f32(i));}
fn rasterPointMappedVisibility(light:Light,p:vec3f,n:vec3f,s:i32)->vec2f{return vec2f(0);}
fn rasterEvaluateLightVisibility(p:vec3f,n:vec3f,o:u32,l:vec3f,d:f32,i:u32)->vec3f{return vec3f(f32((i*79u+u32(d)*31u)%1024u)/1023.0);}
fn rasterAtlasCoordinate(c:vec2i)->vec2i{return c;}
${rasterVisibilityPackingWGSL}
fn storeResult(c:vec2i,layer:i32,value:vec4u){for(var i=0u;i<3u;i++){let slot=u32(layer)*3u+i;if(slot<64u){output[u32(c.x)*64u+slot]=f32((value.x>>(i*10u))&1023u);}}}
${producer}
@compute @workgroup_size(1) fn main(){
  dry.tuningCounts0=vec4u(0,0,32,2);dry.tuningCounts1=vec4u(2);dryLighting.metadata=vec4u(32,1,0,0);
  for(var i=0u;i<32u;i++){dryLighting.lights[i].identity=vec4u(SVO_LIGHT_SPHERE_AREA,0,0,1);}
  let counts=array<u32,4>(1,2,31,32);
  for(var i=0u;i<4u;i++){dry.tuningCounts0.z=counts[i];rasterProduceLightVisibility(vec2i(i32(i),0),vec4f(0,0,1,1),1.0);}
}`});
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
      const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([encoder.finish()]);
      const values=await readMixedBuffer(device,output);
      for(const [caseIndex,count] of [1,2,31,32].entries()) for(let light=0;light<count;light++) for(let sample=0;sample<2;sample++){
        assert.equal(values[caseIndex*64+light*2+sample],(light*79+sample*31)%1024,`case ${caseIndex}, light ${light}, sample ${sample}`);
      }
    }finally{output.destroy();}
  });
});
