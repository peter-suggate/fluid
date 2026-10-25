import assert from "node:assert/strict";
import test from "node:test";
import { SparseVoxelRenderStageOverlay } from "../lib/svo/features/diagnostics/webgpu-svo-stage-overlay";
import { diagnosticQuery } from "../lib/svo/features/diagnostics/persistence";

test("AO tap overlay survives a shared-link round trip", () => {
  const query=new URLSearchParams();diagnosticQuery.write(query,{svoStageView:"raster-ao-taps"});
  assert.equal(diagnosticQuery.read(query).svoStageView,"raster-ao-taps");
});

test("GPU AO overlay shows distant budgets, graded transition taps, disabled AO and misses", {skip:!process.env.WEBGPU_NODE_MODULE}, async () => {
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  await withUniformDevice("AO tap overlay",async device=>{
    const overlay=new SparseVoxelRenderStageOverlay(device,"rgba8unorm");
    const geometry=device.createTexture({size:[5,1],format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    const importance=device.createTexture({size:[5,1],format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    device.queue.writeTexture({texture:importance},new Float32Array([8,20,32,0,8]),{bytesPerRow:20},[5,1]);
    const target=device.createTexture({size:[5,1],format:"rgba8unorm",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
    const range=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const bits=new Uint32Array(new Float32Array([8,32]).buffer);
    device.queue.writeBuffer(range,0,new Uint32Array([~bits[0]!,bits[1]!,0,0]));
    const read=device.createBuffer({size:256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{
      await overlay.initialize();
      const data=new Float32Array(20),projected=[8,20,32,8,8];
      for(let x=0;x<5;x++){
        const rayX=((x+.5)/5*2-1)*5*.5;
        const viewDepth=.06/projected[x]!;
        data.set([0,1,0,viewDepth*Math.sqrt(1+rayX*rayX)],x*4);
      }
      data.set([0,0,0,0],12);
      device.queue.writeTexture({texture:geometry},data,{bytesPerRow:80},[5,1]);
      for(const taps of [0,12,24,36])for(const enabled of [true,false]){
        const encoder=device.createCommandEncoder();
        overlay.encode(encoder,target.createView(),"raster-ao-taps",0,5,1,10,{
          rasterAoRange:range,rasterAoImportance:importance.createView(),splitGeometry:geometry.createView(),
          rasterAoBudget:{enabled,farTaps:taps,cellSize_m:.01,radiusScale:1},
        });
        encoder.copyTextureToBuffer({texture:target},{buffer:read,bytesPerRow:256},[5,1]);
        device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
        const bytes=new Uint8Array(read.getMappedRange());
        const palette:Record<number,number[]>={0:[8,5,31],12:[0,217,255],24:[234,255,0],36:[255,23,77]};
        for(let x=0;x<5;x++){
          const count=enabled?(x===2?36:x===1?Math.max(24,(taps+36)/2):taps):0;
          // Reconstruction can put the midpoint a few ulps above an integer
          // budget, activating one tiny fractional tap. Both counts are valid.
          const counts=enabled&&x===1&&taps<36?[count,count+1]:[count];
          assert.ok(counts.some(n=>{
            const low=Math.min(Math.floor(n/12),2)*12,fraction=(n-low)/12;
            const color=palette[low]!.map((v,c)=>v+(palette[low+12]![c]!-v)*fraction);
            const expected=x===3?[14,14,19]:color;
            return expected.every((v,c)=>Math.abs(bytes[x*4+c]!-v)<=1);
          }),`taps=${taps}, enabled=${enabled}, x=${x}, rgb=${[...bytes.slice(x*4,x*4+3)]}`);
        }
        read.unmap();
      }
    }finally{overlay.destroy();geometry.destroy();importance.destroy();target.destroy();read.destroy();range.destroy();}
  });
});

test("GPU importance excludes flat background, includes detail at odd edges, and resets for flat or empty views", {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  const {rasterHorizonBudgetWGSL,rasterHorizonRangeProducerWGSL}=await import("../lib/svo/features/lighting-visibility/raster-ao-budget");
  await withUniformDevice("visible AO range",async device=>{
    const geometry=device.createTexture({size:[17,3],format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    const importance=device.createTexture({size:[9,2],format:"r32float",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC});
    const range=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    const importanceRead=device.createBuffer({size:512,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const read=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{
      const module=device.createShaderModule({code:`
${rasterHorizonBudgetWGSL}
@group(0) @binding(0) var drySplitGeometryRead:texture_2d<f32>;
struct View { viewport:vec2f }
const uniforms=View(vec2f(17,3));
const DRY_MISS:f32=3.402823e38;
fn cameraTanHalfFov()->f32{return .5;}
fn rasterVisibilityStride()->u32{return 2u;}
fn dryContactVisibilityRadius()->f32{return .1;}
fn rasterWorldAt(pixel:vec2f,depth:f32)->vec3f{return vec3f(pixel.x,0,pixel.y);}
fn drySplitGeometryAt(c:vec2i)->vec4f{return textureLoad(drySplitGeometryRead,c,0);}
${rasterHorizonRangeProducerWGSL}
`});
      const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"rasterAoRangeMain"}});
      const groups=[
        device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:geometry.createView()}]}),
        device.createBindGroup({layout:pipeline.getBindGroupLayout(1),entries:[]}),
        device.createBindGroup({layout:pipeline.getBindGroupLayout(2),entries:[{binding:22,resource:{buffer:range}},{binding:24,resource:importance.createView()}]}),
      ];
      for(const mode of ["varied","flat","empty"]){
        const data=new Float32Array(17*3*4);
        for(let y=0;y<3;y++)for(let x=0;x<17;x++){
          const rayX=((x+.5)/17*2-1)*17/3*.5,rayY=((y+.5)/3*2-1)*.5;
          const footprint=mode==="flat"?8:x===16?32:8;
          data.set([0,1,0,.3/footprint*Math.sqrt(1+rayX*rayX+rayY*rayY)],(y*17+x)*4);
        }
        // An invalid foreground depth must not distort the visible range.
        data.set([0,0,0,1e-9],(17+3)*4);
        if(mode==="varied"){data[(17+5)*4]=1;data[(17+5)*4+1]=0;data[(17+16)*4]=1;data[(17+16)*4+1]=0;}
        if(mode==="empty")data.fill(0);
        device.queue.writeTexture({texture:geometry},data,{bytesPerRow:17*16},[17,3]);
        const encoder=device.createCommandEncoder();encoder.clearBuffer(range);
        const pass=encoder.beginComputePass();pass.setPipeline(pipeline);groups.forEach((group,i)=>pass.setBindGroup(i,group));pass.dispatchWorkgroups(1,1);pass.end();
        encoder.copyBufferToBuffer(range,0,read,0,16);
        encoder.copyTextureToBuffer({texture:importance},{buffer:importanceRead,bytesPerRow:256},[9,2]);
        device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
        const words=new Uint32Array(read.getMappedRange());
        if(mode!=="varied")assert.deepEqual([...words],[0,0,0,0],"background cannot determine the detail range");
        else{
          const bounds=new Float32Array(new Uint32Array([~words[0]!,words[1]!]).buffer);
          assert.ok(Math.abs(bounds[0]!-8)<1e-4);
          assert.ok(Math.abs(bounds[1]!-32)<1e-4,"odd viewport edge participates");
        }
        read.unmap();
        await importanceRead.mapAsync(GPUMapMode.READ);
        const scores=new Float32Array(importanceRead.getMappedRange());
        assert.equal(scores[0],0,"flat ground has zero importance at any depth");
        if(mode==="varied")assert.ok(scores[8]!>0,"odd edge detail survives");
        else assert.ok([...scores].every(v=>v===0));
        importanceRead.unmap();
      }
    }finally{geometry.destroy();importance.destroy();range.destroy();read.destroy();importanceRead.destroy();}
  });
});

test("GPU low-budget importance covers nearby blockers and preserves odd mip edges", {skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  const {readMixedBuffer}=await import("./helpers/uniform-mixed-native-fields");
  const {rasterAoImportanceMipWGSL,rasterHorizonImportanceWGSL,rasterHorizonBudgetWGSL}=await import("../lib/svo/features/lighting-visibility/raster-ao-budget");
  await withUniformDevice("AO footprint protection",async device=>{
    const texture=device.createTexture({size:[9,3],mipLevelCount:4,format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST});
    const output=device.createBuffer({size:60,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    try{
      const values=new Float32Array(27);values[26]=32;
      device.queue.writeTexture({texture},values,{bytesPerRow:36},[9,3]);
      const mip=await device.createComputePipelineAsync({layout:"auto",compute:{module:device.createShaderModule({code:rasterAoImportanceMipWGSL}),entryPoint:"main"}});
      const query=await device.createComputePipelineAsync({layout:"auto",compute:{entryPoint:"main",module:device.createShaderModule({code:`
@group(0) @binding(0) var rasterAoImportance:texture_2d<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
${rasterHorizonBudgetWGSL}
${rasterHorizonImportanceWGSL}
@compute @workgroup_size(16) fn main(@builtin(local_invocation_index) lane:u32){
  if(lane>=15u){return;}
  if(lane==14u){output[lane]=f32(rasterHorizonTapCount(rasterHorizonReceiverBudget(vec2f(8.5,2.5),4.0,4u,vec2f(32)),4u));return;}
  if(lane>=4u){let minimum=(lane-4u)*4u;let budget=rasterHorizonReceiverBudget(vec2f(.5),4.0,minimum,vec2f(0,32));output[lane]=f32(rasterHorizonTapCount(budget,minimum));return;}
  let coordinate=select(vec2f(.5),vec2f(8.5,2.5),lane==3u);
  output[lane]=rasterHorizonImportanceAt(coordinate,select(0.0,4.0,lane==1u),select(4u,24u,lane==2u));
}`})}});
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(mip);
      for(let level=1;level<4;level++){
        pass.setBindGroup(0,device.createBindGroup({layout:mip.getBindGroupLayout(0),entries:[
          {binding:0,resource:texture.createView({baseMipLevel:level-1,mipLevelCount:1})},
          {binding:1,resource:texture.createView({baseMipLevel:level,mipLevelCount:1})},
        ]}));pass.dispatchWorkgroups(1);
      }
      pass.setPipeline(query);pass.setBindGroup(0,device.createBindGroup({layout:query.getBindGroupLayout(0),entries:[{binding:0,resource:texture.createView()},{binding:1,resource:{buffer:output}}]}));pass.dispatchWorkgroups(1);pass.end();
      device.queue.submit([encoder.finish()]);assert.deepEqual([...(await readMixedBuffer(device,output))],[0,32,0,32,24,24,24,24,24,24,24,28,32,36,4]);
    }finally{texture.destroy();output.destroy();}
  });
});
