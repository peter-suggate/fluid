import assert from 'node:assert/strict';
import test from 'node:test';
import { uniformAddressPlanWGSL, UNIFORM_ADDRESS_HOME_MASK } from '../lib/methods/uniform/uniform-address-plan';
import { uniformBlockNeighbourLayout, uniformBlockNeighbourWords, uniformLatticeAddress, uniformLatticeAddressWGSL, uniformVertexSlot } from '../lib/methods/uniform/uniform-lattice-address';
import { uniformWindowAddressWGSL } from '../lib/methods/uniform/uniform-window-address';
import { UniformBlockNeighbours } from '../lib/methods/uniform/uniform-block-neighbours';
import { withUniformDevice } from './helpers/uniform-geometric';

const address=()=>uniformLatticeAddress([96,96,96],true,{permute:871,air:[[0,0,0],[1,0,1],[0,1,1],[2,2,1],[1,2,2]]});
test('packed canonical homes preserve the first stored incident page for fine and coarse vertices',()=>{
 const a=address(),words=uniformBlockNeighbourWords(a.seed.table,a.seed.blocks);
 const homes=33*a.blocks+1;
 for(let b=1;b<a.blocks-1;b++){
  const record=a.seed.blocks[b]!;if(record===0xffffffff)continue;
  const page=[record&63,(record>>>6)&63,(record>>>12)&63];
  for(const bits of [3,5] as const){const edge=1<<bits;
   for(let f=0;f<27;f++){
    const side=[f%3,Math.floor(f/3)%3,Math.floor(f/9)];
    const local=side.map(v=>v===0?0:v===1?3:edge);
    const word=words[homes+27*b+f]!;
    const delta=[26,28,30].map(shift=>{const v=(word>>>shift)&3;return v===2?-1:v;});
    const expected=uniformVertexSlot(a,page[0]!*edge+local[0]!,page[1]!*edge+local[1]!,page[2]!*edge+local[2]!,bits,[0,0,0,0]);
    assert.deepEqual([...local.map((v,i)=>v-edge*delta[i]!),word&UNIFORM_ADDRESS_HOME_MASK],expected,`block ${b}, family ${f}, bits ${bits}`);
   }
  }
 }
});

test('GPU rebuilt plans match directory addressing after reseeding and window movement',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('compact address plans',async device=>{
  const a=address(),layout=uniformBlockNeighbourLayout(a.blocks);
  const tail=device.createBuffer({size:4*layout.end,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:a.blocks*2*729*48,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const builder=new UniformBlockNeighbours(device,a.blocks,[{buffer:tail,offset:0}]);await builder.initialize();
  const code=`
@group(0) @binding(0) var<storage,read> umTopology:array<u32>;
@group(0) @binding(1) var<storage,read_write> result:array<vec4u>;
const UM_BLOCK_RECORD:u32=262176u;
${uniformWindowAddressWGSL('umTopology','0u',`${a.blocks}u`).replace('fn umPageWord(','fn umDirectoryPageWord(')}
${uniformLatticeAddressWGSL({constant:true,paged:{blocks:`${a.blocks}u`},world:true,addressPlan:true})}
${uniformAddressPlanWGSL}
@compute @workgroup_size(64) fn compare(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let block=group.x;umLoadAddressPlan(block,lane);
 if(!umBlockStored(block)||umBlockWord(block)==UM_FREE){return;}
 let page=vec3i(umBlockPage(block));
 for(var width=0u;width<2u;width++){
  let bits=select(3u,5u,width!=0u);let edge=1<<bits;
  let positions=array<i32,9>(-edge-1,-1,0,1,edge/2,edge-1,edge,edge+1,2*edge+1);
  for(var i=lane;i<729u;i+=64u){
   let p=(page<<vec3u(bits))+vec3i(positions[i%9u],positions[(i/9u)%9u],positions[i/81u]);
   var slot=vec4u(0);if(width==0u){slot=umWideVertexSlot(p);}else{slot=umVertexSlot(p);}
   let at=3u*((2u*block+width)*729u+i);result[at]=slot;
   if(width==0u){result[at+1u]=vec4u(vec3u(umWideVertexTexel(p)),0u);result[at+2u]=vec4u(vec3u(umWideVertexStoreTexel(p)),0u);}
   else{result[at+1u]=vec4u(vec3u(umVertexTexel(p)),0u);result[at+2u]=vec4u(vec3u(umVertexStoreTexel(p)),0u);}
  }
 }
}`;
  const module=device.createShaderModule({code});assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);
  const pipelines=await Promise.all([0,1].map(enabled=>device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'compare',constants:{umUseAddressPlan:enabled}}})));
  try{
   for(const shift of [0,1,63]){
    const seed=a.seed.tail.slice();seed[262144]=shift;seed[262145]=shift;seed[262146]=shift;
    device.queue.writeBuffer(tail,0,seed);
    const rebuild=device.createCommandEncoder();builder.encode(rebuild);device.queue.submit([rebuild.finish()]);
    let reference:Uint32Array|undefined;
    for(const pipeline of pipelines){
     const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:tail}},{binding:1,resource:{buffer:output}}]});
     const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,bind);p.dispatchWorkgroups(a.blocks);p.end();e.copyBufferToBuffer(output,0,read,0,read.size);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
     const actual=new Uint32Array(read.getMappedRange()).slice();read.unmap();
     if(reference){for(let i=0;i<actual.length;i++)assert.equal(actual[i],reference[i],`frame residue ${shift}, word ${i}, block ${Math.floor(i/(2*729*12))}`);}else reference=actual;
    }
   }
  }finally{tail.destroy();output.destroy();read.destroy();}
 });
});
