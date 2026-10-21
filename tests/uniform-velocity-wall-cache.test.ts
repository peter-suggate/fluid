import assert from 'node:assert/strict';
import test from 'node:test';
import {uniformVelocityWallCacheWGSL} from '../lib/methods/uniform/uniform-velocity-wall-cache.wgsl';
import {withUniformDevice} from './helpers/uniform-geometric';
import {readMixedBuffer} from './helpers/uniform-mixed-native-fields';

test('velocity ghost sources reflect all box walls and corners without changing normal components or coupling thin walls',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('wall velocity ghosts',async device=>{
  const cases:number[][]=[],expected:number[][]=[];
  for(let axis=0;axis<3;axis++)for(let z=-2;z<=5;z++)for(let y=-2;y<=5;y++)for(let x=-2;x<=5;x++){
   const p=[x,y,z];cases.push([...p,axis]);
   const reachable=p[axis]!>=0&&p[axis]!<4&&p.every(v=>v>=-1&&v<=4);
   expected.push([...(reachable?p.map(v=>Math.max(0,Math.min(3,v))):p),0]);
  }
  const count=cases.length;
  const input=device.createBuffer({size:16*count,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:16*count,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  device.queue.writeBuffer(input,0,new Int32Array(cases.flat()));
  try{
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> input:array<vec4i>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4i>;
var<private> mode:u32;
fn umCacheTileKind(p:vec3i)->u32{
 if(mode==1u){return select(1u,0u,p.x==0);}
 if(mode==2u&&p.x<0){return 2u;}
 return u32(all(p>=vec3i(0))&&all(p<vec3i(4)));
}
${uniformVelocityWallCacheWGSL}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=arrayLength(&input)){return;}
 let p=input[id.x];mode=0u;let source=umCacheWallSource(p.xyz,u32(p.w));
 // A one-tile wall between two open regions has no unique reflection.
 mode=1u;let thin=umCacheWallSource(vec3i(0,1,1),1u);
 // Cut cells and moving-body support are classified as unavailable.
 mode=2u;let cut=umCacheWallSource(vec3i(-1,1,1),1u);
 output[id.x]=vec4i(source,i32(any(thin!=vec3i(0,1,1))||any(cut!=vec3i(-1,1,1))));
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]});
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();device.queue.submit([encoder.finish()]);
   const actual=new Int32Array((await readMixedBuffer(device,output)).buffer);
   for(let i=0;i<count;i++)assert.deepEqual(Array.from(actual.slice(4*i,4*i+4)),expected[i],`tile/component ${cases[i]}`);
  }finally{input.destroy();output.destroy();}
 });
});
