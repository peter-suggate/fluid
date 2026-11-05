import test from 'node:test';
import assert from 'node:assert/strict';
import {uniformRigidIntervalsWGSL} from '../lib/methods/uniform/uniform-rigid-intervals.wgsl';
import {rigidRayUnionLength,rigidRayIntervals} from '../lib/core/rigid-ray-intervals';
import {SCENE_SHAPE_NAMES,sceneShapeCode} from '../lib/core/scene-shape';
import {packRigidBodyCutAtlas,RIGID_BODY_CUT_ATLAS_WIDTH} from '../lib/core/solid-cuts';
import {withUniformDevice} from './helpers/uniform-geometric';

test('native analytic interval sweep matches authoritative CPU all shapes, rotations, cuts, inversions and unions',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('native rigid interval oracle',async device=>{
  const orientation={w:Math.cos(.31),x:Math.sin(.31)/Math.sqrt(14),y:2*Math.sin(.31)/Math.sqrt(14),z:3*Math.sin(.31)/Math.sqrt(14)};
  const bodies:Parameters<typeof rigidRayIntervals>[0][]=SCENE_SHAPE_NAMES.flatMap(shape=>[false,true].map(inverted=>({shape,dimensions_m:{x:.13,y:shape==='rounded-plate'?.003:.14,z:shape==='cup'?.012:.11},position_m:{x:.02,y:-.01,z:.03},orientation,fluidBoundary:inverted?'interior' as const:undefined,cuts:[{center_m:{x:0,y:.01,z:0},halfExtents_m:{x:.017,y:.021,z:.08},orientation:{w:Math.cos(.17),x:0,y:0,z:Math.sin(.17)}}]})));
  const bodyValues=new Float32Array(bodies.length*12);bodies.forEach((b,i)=>bodyValues.set([b.position_m.x,b.position_m.y,b.position_m.z,b.fluidBoundary==='interior'?-sceneShapeCode(b.shape)-1:sceneShapeCode(b.shape),b.orientation.w,b.orientation.x,b.orientation.y,b.orientation.z,b.dimensions_m.x,b.dimensions_m.y,b.dimensions_m.z,1],i*12));
  const bodyBuffer=device.createBuffer({size:bodyValues.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(bodyBuffer,0,bodyValues);
  const cutValues=packRigidBodyCutAtlas(bodies),cuts=device.createTexture({size:[RIGID_BODY_CUT_ATLAS_WIDTH,1],format:'rgba32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});device.queue.writeTexture({texture:cuts},cutValues,{bytesPerRow:16*RIGID_BODY_CUT_ATLAS_WIDTH},[RIGID_BODY_CUT_ATLAS_WIDTH,1]);
  let seed=19;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  const queries:{o:{x:number;y:number;z:number};d:{x:number;y:number;z:number};mask:number}[]=[];
  for(let body=0;body<bodies.length;body++)for(let k=0;k<64;k++)queries.push({o:{x:.4*(random()-.5),y:.4*(random()-.5),z:.4*(random()-.5)},d:{x:2*random()-1,y:2*random()-1,z:2*random()-1},mask:1<<body});
  for(let k=0;k<64;k++)queries.push({o:{x:.4*(random()-.5),y:.4*(random()-.5),z:.4*(random()-.5)},d:{x:2*random()-1,y:2*random()-1,z:2*random()-1},mask:0x555});
  const queryValues=new Float32Array(queries.length*8);queries.forEach((q,i)=>queryValues.set([q.o.x,q.o.y,q.o.z,q.mask,q.d.x,q.d.y,q.d.z,0],i*8));
  const input=device.createBuffer({size:queryValues.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(input,0,queryValues);
  const output=device.createBuffer({size:4*queries.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:4*queries.length,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const module=device.createShaderModule({code:`struct Body {positionShape:vec4f,orientation:vec4f,dimensions:vec4f};
   @group(0) @binding(0) var<storage,read> umSolidBodies:array<Body>;
   @group(0) @binding(1) var umSolidBodyCuts:texture_2d<f32>;
   @group(0) @binding(2) var<storage,read> input:array<vec4f>;
   @group(0) @binding(3) var<storage,read_write> output:array<f32>;
   fn umQuatRotate(q:vec4f,v:vec3f)->vec3f{return v+2.0*(q.x*cross(q.yzw,v)+cross(q.yzw,cross(q.yzw,v)));}
   ${uniformRigidIntervalsWGSL}
   @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){if(id.x>=${queries.length}u){return;}let o=input[2u*id.x];let d=input[2u*id.x+1u];output[id.x]=umRayUnionLength(u32(o.w),o.xyz,d.xyz,vec2f(-.3,.3));}`});
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:bodyBuffer}},{binding:1,resource:cuts.createView()},{binding:2,resource:{buffer:input}},{binding:3,resource:{buffer:output}}]});
  try{const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(queries.length/64));pass.end();encoder.copyBufferToBuffer(output,0,read,0,4*queries.length);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const actual=new Float32Array(read.getMappedRange());
   queries.forEach((q,i)=>{const expected=rigidRayUnionLength(bodies.filter((_,b)=>(q.mask&(1<<b))!==0),q.o,q.d,[-.3,.3]);assert.ok(Math.abs(actual[i]!-expected)<2e-6,`ray ${i}: ${actual[i]} vs ${expected}`);});
  }finally{if(read.mapState==='mapped')read.unmap();for(const b of [bodyBuffer,input,output,read])b.destroy();cuts.destroy();}
 });
});
