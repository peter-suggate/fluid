/** Untimed census of the exact budget/importance planes shown by the overlay. */
import { rasterHorizonBudgetWGSL, rasterHorizonImportanceWGSL } from "../lib/svo/features/lighting-visibility/raster-ao-budget";
import type { SvoRenderStagePlanes } from "../lib/svo/features/diagnostics/webgpu-svo-stage-overlay";

export async function rasterAoBudgetCensus(device:GPUDevice,planes:SvoRenderStagePlanes,width:number,height:number,projection:{radius_m:number;tanHalfFov:number}){
  if(!planes.rasterAoImportance||!planes.rasterAoRange||!planes.splitGeometry||!planes.rasterAoBudget)return undefined;
  const params=device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const counts=device.createBuffer({size:74*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:74*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
    device.queue.writeBuffer(params,0,new Uint32Array([width,height,planes.rasterAoBudget.farTaps,0]));
    device.queue.writeBuffer(params,16,new Float32Array([projection.radius_m,projection.tanHalfFov,0,0]));
    const module=device.createShaderModule({code:`${rasterHorizonBudgetWGSL}
${rasterHorizonImportanceWGSL}
@group(0) @binding(0) var rasterAoImportance:texture_2d<f32>;
@group(0) @binding(1) var geometry:texture_2d<f32>;
@group(0) @binding(2) var<uniform> range:vec4u;
struct Params { extent:vec4u, projection:vec4f }
@group(0) @binding(3) var<uniform> settings:Params;
@group(0) @binding(4) var<storage,read_write> counts:array<atomic<u32>>;
var<workgroup> localCounts:array<atomic<u32>,74>;
@compute @workgroup_size(16,16) fn main(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32){
  if(lane<74u){atomicStore(&localCounts[lane],0u);}workgroupBarrier();
  let params=settings.extent;
  if(all(id.xy<params.xy)){
    let g=textureLoad(geometry,vec2i(id.xy),0);
    if(g.w>0.0&&g.w<3.402823e38&&dot(g.xyz,g.xyz)>=.1){
      let dims=textureDimensions(rasterAoImportance);let stride=(params.xy+dims-1u)/dims;
      let ndc=(vec2f(id.xy)+.5)/vec2f(params.xy)*2.0-1.0;
      let ray=vec2f(ndc.x*f32(params.x)/f32(params.y),ndc.y)*settings.projection.y;
      let depth=g.w*inverseSqrt(1.0+dot(ray,ray));
      let radius=clamp(rasterHorizonProjectedRadius(settings.projection.x,depth,f32(params.y),settings.projection.y),2.0,160.0);
      let score=rasterHorizonImportanceAt((vec2f(id.xy)+.5)/vec2f(stride),radius/f32(stride.x),params.z);
      let budget=rasterHorizonReceiverBudget((vec2f(id.xy)+.5)/vec2f(stride),radius/f32(stride.x),params.z,rasterHorizonRange(range.xy));
      let taps=rasterHorizonTapCount(budget,params.z);
      atomicAdd(&localCounts[taps+select(0u,37u,score>0.0)],1u);
    }
  }
  workgroupBarrier();
  if(lane<74u){let count=atomicLoad(&localCounts[lane]);if(count>0u){atomicAdd(&counts[lane],count);}}
}`});
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:planes.rasterAoImportance},{binding:1,resource:planes.splitGeometry},
      {binding:2,resource:{buffer:planes.rasterAoRange}},{binding:3,resource:{buffer:params}},{binding:4,resource:{buffer:counts}},
    ]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(width/16),Math.ceil(height/16));pass.end();
    encoder.copyBufferToBuffer(counts,0,read,0,74*4);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
    const data=[...new Uint32Array(read.getMappedRange())];
    const summarize=(histogram:number[])=>({pixels:histogram.reduce((a,b)=>a+b,0),tapHistogram:histogram,
      meanTaps:histogram.reduce((a,b,i)=>a+b*i,0)/Math.max(1,histogram.reduce((a,b)=>a+b,0))});
    return {flat:summarize(data.slice(0,37)),detail:summarize(data.slice(37)),all:summarize(data.slice(0,37).map((n,i)=>n+data[i+37]!))};
  }finally{if(read.mapState==="mapped")read.unmap();read.destroy();counts.destroy();params.destroy();}
}
