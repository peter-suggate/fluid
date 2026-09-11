import assert from 'node:assert/strict';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from '../lib/harness/webgpu-smoke-isolation';
import {createSparseCM12VelocityExtensionLayout,createSparseCM12VelocityExtensionInitialWords} from '../lib/methods/adaptive-mass/sparse-cm12-velocity-extension';
import {createSparseCM12VelocityExtensionWGSL} from '../lib/methods/adaptive-mass/sparse-cm12-velocity-extension.wgsl';
const dawnTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;
dawnTest('VEX rebuild initializes direct once, clears retired banks, and schedules only accepted sweeps',async()=>{
 await acquireWebGPUExclusiveLock('dawn-test','vex-rebuild');let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
  const gpu:GPU=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??'metal'}`]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const layout=createSparseCM12VelocityExtensionLayout({cellCapacity:1024,packetCapacity:128,brickFineResolution:8});
  const generated=createSparseCM12VelocityExtensionWGSL({layout,cacheAcceptedPackets:true,
   topologyGenerationExpression:'p.x',effectiveVelocityHookPrefix:'fixture'});
  const extract=(name:string)=>{const text=generated.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];assert.ok(text,name);return text;};
  const code=`
@group(0)@binding(0)var<storage,read_write>activity:array<atomic<u32>>;
@group(0)@binding(1)var<storage,read_write>state:array<f32>;
@group(0)@binding(2)var<storage,read>leaves:array<vec4u>;
@group(0)@binding(3)var<uniform>p:vec4u;
const cm12ExtensionInvalid=0xffffffffu;const cm12ExtensionCapacity=1024u;
const cm12ExtensionPacketCapacity=128u;const cm12ExtensionDispatchPacketsPerLeaf=8u;
const cm12ExtensionDispatchPacketCount=16u;const cm12ExtensionDispatchWidth=65535u;
const cm12ExtensionValidityA=${layout.validityABaseWords}u;
const cm12ExtensionValidityB=${layout.validityBBaseWords}u;
const cm12ExtensionAcceptedDepth=${layout.acceptedDepthBaseWords}u;
const CM12_VEX_SCHEDULE=${layout.scheduleBaseWords}u;
const CM12_VEX_PACKET_LIST=${layout.packetListBaseWords}u;
const CM12_LIQUID_ISOVALUE=0.5;
var<workgroup>cm12ExtensionDispatchPacket:u32;
fn cm12ExtensionLoad(at:u32)->u32{return atomicLoad(&activity[at]);}
fn cm12ExtensionStore(at:u32,v:u32){atomicStore(&activity[at],v);}
fn acceptedTopologySlot()->u32{return p.y;}
fn acceptedLeafInvocation(rank:u32)->u32{if(rank>=2u||leaves[rank].x==0u){return cm12ExtensionInvalid;}return rank;}
struct Leaf{flags:u32}
fn cm12TeiLoadLeaf(slot:u32,leaf:u32)->Leaf{_=slot;let r=leaves[leaf].x;return Leaf(select(0u,0x80000000u|r,r>0u));}
struct Packet{first:u32,counts:vec3u,strideY:u32,strideZ:u32}
fn cm12TeiPacket(packet:u32,slot:u32)->Packet{
 _=slot;let leaf=packet/64u;let local=packet%64u;let d=leaves[leaf];
 let axis=max(1u,(d.x+3u)/4u);
 if(d.x==0u||local>=axis*axis*axis){return Packet(cm12ExtensionInvalid,vec3u(0),0u,0u);}
 let q=4u*vec3u(local%axis,(local/axis)%axis,local/(axis*axis));
 if(any(q>=d.yzw)){return Packet(cm12ExtensionInvalid,vec3u(0),0u,0u);}
 return Packet(512u*leaf+q.x+d.y*(q.y+d.z*q.z),min(vec3u(4),d.yzw-q),d.y,d.y*d.z);
}
fn sourceDensity()->u32{return 0u;}
fn sourceCellVelocity()->u32{return 1024u;}
fn fixturePublishVexAcceptedEffectiveVelocity(cell:u32,value:vec4f){
 for(var i=0u;i<4u;i++){state[5120u+4u*cell+i]=value[i];}}
${extract('cm12ExtensionStablePacket')}
@compute @workgroup_size(1)
${extract('beginSparseCM12VelocityExtensionSchedule')}
@compute @workgroup_size(64)
${extract('compileSparseCM12VelocityExtensionSchedule')}
@compute @workgroup_size(1)
${extract('sealSparseCM12VelocityExtensionSchedule')}
@compute @workgroup_size(64)
${extract('initializeVelocityExtensionPackets')}
`;
  const shader=device.createShaderModule({code});assert.deepEqual((await shader.getCompilationInfo()).messages.filter(x=>x.type==='error').map(x=>x.message),[]);
  const bgl=device.createBindGroupLayout({entries:[0,1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===3?'uniform':binding===2?'read-only-storage':'storage'}}))});
  const pl=device.createPipelineLayout({bindGroupLayouts:[bgl]});
  const buffers:GPUBuffer[]=[];const make=(size:number,usage:number)=>{const b=device!.createBuffer({size,usage});buffers.push(b);return b;};
  const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
  const arena=make(layout.totalWords*4,storage),state=make(9216*4,storage),leaves=make(32,storage),params=make(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
  const indirect=make(24,GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST),read=make(layout.totalWords*4+9216*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
  const bg=device.createBindGroup({layout:bgl,entries:[arena,state,leaves,params].map((buffer,binding)=>({binding,resource:{buffer}}))});
  const names=['beginSparseCM12VelocityExtensionSchedule','compileSparseCM12VelocityExtensionSchedule','sealSparseCM12VelocityExtensionSchedule','initializeVelocityExtensionPackets'];
  const pipelines=await Promise.all(names.map(entryPoint=>device!.createComputePipelineAsync({layout:pl,compute:{module:shader,entryPoint}})));
  device.queue.writeBuffer(arena,0,new Uint32Array(createSparseCM12VelocityExtensionInitialWords(layout)));
  const data=new Float32Array(9216);for(let c=0;c<1024;c++){data[c]=c%3===0?1:0;data.set([c+.25,-c-.5,3.125,0],1024+4*c);}
  device.queue.writeBuffer(state,0,data);
  const run=async(label:string,descriptors:number[],generation:number,slot:number,initialGroups:number,sweepGroups:number)=>{
   device!.queue.writeBuffer(leaves,0,new Uint32Array(descriptors));device!.queue.writeBuffer(params,0,new Uint32Array([generation,slot,0,0]));
   // Both banks start dirty; retired packets must be scrubbed even without a sweep.
   device!.queue.writeBuffer(arena,4*layout.validityABaseWords,new Uint32Array(256).fill(0xffffffff));
   device!.queue.writeBuffer(arena,4*layout.validityBBaseWords,new Uint32Array(256).fill(0xffffffff));
   const e=device!.createCommandEncoder();
   for(let i=0;i<3;i++){const pass=e.beginComputePass();pass.setBindGroup(0,bg);pass.setPipeline(pipelines[i]!);pass.dispatchWorkgroups(1);pass.end();}
   e.copyBufferToBuffer(arena,4*(layout.scheduleBaseWords+4),indirect,0,24);
   const pass=e.beginComputePass();pass.setBindGroup(0,bg);pass.setPipeline(pipelines[3]!);pass.dispatchWorkgroupsIndirect(indirect,0);pass.end();
   e.copyBufferToBuffer(arena,0,read,0,arena.size);e.copyBufferToBuffer(state,0,read,arena.size,state.size);device!.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const a=new Uint32Array(read.getMappedRange()).slice();const v=new Float32Array(a.buffer,arena.size);read.unmap();
   const base=layout.scheduleBaseWords;assert.equal(a[base+4],initialGroups,label+' init');assert.equal(a[base+7],sweepGroups,label+' sweeps');
   for(let leaf=0;leaf<2;leaf++)for(let local=0;local<8;local++){
    const [r,w,h,d]=descriptors.slice(4*leaf,4*leaf+4) as [number,number,number,number];const axis=Math.max(1,Math.ceil(r/4));
    const q=[4*(local%axis),4*(Math.floor(local/axis)%axis),4*Math.floor(local/(axis*axis))];
    const valid=r>0&&local<axis**3&&q[0]!<w&&q[1]!<h&&q[2]!<d;const packet=64*leaf+local;
    if(!valid){if(initialGroups===16)for(const bank of [layout.validityABaseWords,layout.validityBBaseWords])assert.deepEqual([...a.slice(bank+2*packet,bank+2*packet+2)],[0,0],label+' retired');continue;}
    const expected=[0,0];for(let z=0;z<Math.min(4,d-q[2]!);z++)for(let y=0;y<Math.min(4,h-q[1]!);y++)for(let x=0;x<Math.min(4,w-q[0]!);x++){
     const cell=leaf*512+q[0]!+x+w*(q[1]!+y+h*(q[2]!+z)),wet=cell%3===0,lane=x+4*y+16*z;
     if(wet)expected[lane>>5]=(expected[lane>>5]!|1<<(lane&31))>>>0;
     assert.equal(a[layout.acceptedDepthBaseWords+cell],wet?0:0xffffffff,label+' depth');
     assert.deepEqual([...v.slice(5120+4*cell,5120+4*cell+4)],wet?[cell+.25,-cell-.5,3.125,1]:[0,0,0,0],label+' velocity');
    }
    assert.deepEqual([...a.slice(layout.validityABaseWords+2*packet,layout.validityABaseWords+2*packet+2)],expected,label+' clipped mask');
   }
  };
  await run('clipped bootstrap',[4,2,1,1,8,8,8,8],1,0,16,9);
  await run('stable compact',[4,2,1,1,8,8,8,8],1,0,9,9);
  await run('retire and rerung',[0,0,0,0,4,4,2,1],2,1,16,1);
  await run('recycle and dense',[8,8,8,8,8,8,8,8],3,0,16,16);
  await run('empty rebuild',[0,0,0,0,0,0,0,0],4,1,16,1);
  await run('empty stable',[0,0,0,0,0,0,0,0],4,1,1,1);
  await run('slot-only rebuild',[0,0,0,0,0,0,0,0],4,0,16,1);
  for(const b of buffers)b.destroy();
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
