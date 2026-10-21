import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandNearWGSL } from "../lib/methods/uniform/uniform-narrow-band-near.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

test("particle endpoints respect distant tank walls without shortening free-axis travel",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const cases:{home:number[];end:number[];low:number[];high:number[];fixed:boolean}[]=[];
 // Include translated frame coordinates, all six walls and edges/corners,
 // exact wall hits, internal page seams, and jumps across multiple pages.
 for(const origin of [0,-256,512])for(const x of [-5000,-200,0,32,64.25,128,300,5000])for(const y of [-200,0,64.25,128,300])for(const z of [-200,0,64.25,128,300]){
  cases.push({home:[64,64,64].map(v=>v+origin),end:[x,y,z].map(v=>v+origin),low:[origin,origin,origin],high:[origin+128,origin+128,origin+128],fixed:true});
 }
 // An open top and an open world must not acquire a new travel limit.
 cases.push({home:[64,64,64],end:[64,900,64],low:[0,0,0],high:[128,1e9,128],fixed:true});
 cases.push({home:[64,64,64],end:[900,-100,-800],low:[-1e9,0,-1e9],high:[1e9,1e9,1e9],fixed:true});
 // Irregular page policies retain the existing local clamp.
 cases.push({home:[127,64,64],end:[130,65,66],low:[0,0,0],high:[128,128,128],fixed:false});
 const input=new Float32Array(cases.length*16);
 cases.forEach((c,i)=>{input.set([...c.home,+c.fixed,...c.end,0,...c.low,0,...c.high,0],16*i);});
 await withUniformDevice("particle endpoint walls",async device=>{
  const module=device.createShaderModule({code:`
struct umBounds {lo:vec3f,hi:vec3f}
var<private> walls:umBounds;var<private> fixed:bool;
fn umFixedBox()->bool{return fixed;}
fn umBoxWalls()->umBounds{return walls;}
fn umBlockOf(page:vec3i)->u32{return 1u;}
fn umBlockHeld(block:u32,page:vec3i,d:vec3i)->vec3<bool>{
 let low=vec3i(walls.lo/32.0);let high=vec3i(walls.hi/32.0);
 return (d!=vec3i(0))&((page+d<low)|(page+d>=high));
}
fn umCellNearFromPage(block:u32,page:vec3i,p:vec3i)->vec3i{return p;}
fn umTileNearFromPage(block:u32,page:vec3i,p:vec3i)->vec3i{return p;}
${narrowBandNearWGSL}
@group(0) @binding(0) var<storage,read> inputs:array<vec4f>;
@group(0) @binding(1) var<storage,read_write> outputs:array<vec4f>;
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=${cases.length}u){return;}
 let home=vec3i(inputs[4u*i].xyz);fixed=inputs[4u*i].w!=0.0;
 let p=inputs[4u*i+1u].xyz;walls=umBounds(inputs[4u*i+2u].xyz,inputs[4u*i+3u].xyz);
 outputs[2u*i]=vec4f(nbTraceEndpoint(home,p),0.0);
 outputs[2u*i+1u]=vec4f(nbPointNearIn(home,p,0.01),0.0);
}`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
  const data=device.createBuffer({size:input.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:cases.length*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   device.queue.writeBuffer(data,0,input);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"probe"}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:data}},{binding:1,resource:{buffer:output}}]});
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(cases.length/64));pass.end();device.queue.submit([encoder.finish()]);
   const result=new Float32Array((await readMixedBuffer(device,output)).buffer);let missedWalls=0;
   cases.forEach((c,i)=>{
    for(let a=0;a<3;a++){
     const expected=c.fixed?Math.min(Math.max(c.end[a]!,Math.fround(c.low[a]!+Math.fround(.01))),Math.fround(c.high[a]!-Math.fround(.01))):result[8*i+4+a];
     assert.equal(result[8*i+a],expected,`case ${i}, axis ${a}`);
     if(c.fixed&&result[8*i+4+a]!==expected)missedWalls++;
    }
   });
   assert.ok(missedWalls>0,"the original neighbour-only clamp misses distant walls in these cases");
  }finally{data.destroy();output.destroy();}
 });
});
