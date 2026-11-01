import assert from "node:assert/strict";
import test from "node:test";
import {uniformSolidCutMaskWGSL,uniformSolidCutRingMasks} from "../lib/methods/uniform/uniform-solid-cut-mask.wgsl";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

test("cut ring masks cover each of the original 216 cells exactly once",()=>{
 const cells=new Set<string>();
 for(const [i,mask] of uniformSolidCutRingMasks.entries()){
  const offset=[i%3-1,Math.floor(i/3)%3-1,Math.floor(i/9)-1];
  for(let bit=0;bit<64;bit++)if((mask[bit>>>5]!>>>(bit&31))&1){
   const local=[bit%4,Math.floor(bit/4)%4,Math.floor(bit/16)];
   const cell=local.map((n,a)=>n+4*offset[a]!);
   assert.ok(cell.every(n=>n>=-1&&n<=4));
   const key=cell.join(",");assert.ok(!cells.has(key),key);cells.add(key);
  }
 }
 assert.equal(cells.size,216);
 for(let z=-1;z<=4;z++)for(let y=-1;y<=4;y++)for(let x=-1;x<=4;x++)assert.ok(cells.has(`${x},${y},${z}`));
});

test("cut mask float storage preserves every bit and all neighbor subsets on GPU",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("cut mask packing",async device=>{
  const count=4096,input=new Uint32Array(count*2);let seed=0x8ca7b31;
  for(let i=0;i<input.length;i++){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;input[i]=seed>>>0;}
  for(let bit=0;bit<64;bit++){input[2*bit]=bit<32?(1<<bit)>>>0:0;input[2*bit+1]=bit>=32?(1<<(bit-32))>>>0:0;}
  input.set([0,0,0xffffffff,0xffffffff],128);
  const source=device.createBuffer({size:input.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:count*29*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   device.queue.writeBuffer(source,0,input);
   const module=device.createShaderModule({code:`${uniformSolidCutMaskWGSL}
@group(0) @binding(0) var<storage,read> source:array<vec2u>;
@group(0) @binding(1) var<storage,read_write> result:array<u32>;
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) g:vec3u){
 let mask=umUnpackCutMask(umPackCutMask(source[g.x]));let base=29u*g.x;
 result[base]=mask.x;result[base+1u]=mask.y;
 for(var i=0u;i<27u;i++){result[base+2u+i]=u32(any((mask&umCutRingMasks[i])!=vec2u(0u)));}
}`});
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"probe"}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:source}},{binding:1,resource:{buffer:output}}]});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(count/64);p.end();device.queue.submit([e.finish()]);
   const actual=new Uint32Array((await readMixedBuffer(device,output)).buffer);
   for(let n=0;n<count;n++){
    assert.equal(actual[29*n],input[2*n]);assert.equal(actual[29*n+1],input[2*n+1]);
    for(let i=0;i<27;i++){
     const d=[i%3-1,Math.floor(i/3)%3-1,Math.floor(i/9)-1];let expected=false;
     for(let bit=0;bit<64;bit++){
      const cell=[bit%4,Math.floor(bit/4)%4,Math.floor(bit/16)].map((v,a)=>v+4*d[a]!);
      if(cell.every(v=>v>=-1&&v<=4)&&((input[2*n+(bit>>>5)]!>>>(bit&31))&1))expected=true;
     }
     assert.equal(actual[29*n+2+i],Number(expected),`mask ${n}, neighbor ${i}`);
    }
   }
  }finally{source.destroy();output.destroy();}
 });
});
