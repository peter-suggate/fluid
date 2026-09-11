import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock} from '../lib/harness/webgpu-smoke-isolation';
import {createSparseCM12CanonicalMembershipLayout, createSparseCM12CanonicalMembershipInitialWords} from '../lib/methods/adaptive-mass/sparse-cm12-canonical-membership';
import {createSparseCM12CanonicalMembershipWGSL} from '../lib/methods/adaptive-mass/sparse-cm12-canonical-membership.wgsl';

const dawnTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;
dawnTest('compact pressure row tiles match full publication through dirty, retired, recycled and empty epochs', async()=>{
  await acquireWebGPUExclusiveLock('dawn-test','pressure-row-repair');
  let device:GPUDevice|undefined;
  try {
    const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis,dawn.globals);
    const gpu:GPU=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??'metal'}`]);
    const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
    const errors:string[]=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
    const capacity=131; // Final tile has one bitmap word, with only three valid bits.
    const layout=createSparseCM12CanonicalMembershipLayout({cellCapacity:1,rowCapacity:capacity});
    const resident=readFileSync(new URL('../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts',import.meta.url),'utf8');
    const extract=(name:string)=>{
      const source=resident.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
      assert.ok(source,name);return source;
    };
    const shader=device.createShaderModule({code:`
struct Params{counts:vec4u,acceleration:vec4f}
@group(0)@binding(0)var<storage,read_write>arena:array<atomic<u32>>;
@group(0)@binding(1)var<storage,read>fixture:array<vec4u>;
@group(0)@binding(2)var<storage,read_write>theta:array<u32>;
@group(0)@binding(3)var<uniform>p:Params;
const INVALID:u32=0xffffffffu;
var<workgroup>pcmRowBallot:array<u32,64>;
var<workgroup>pcmRowRepairWorkgroupTile:u32;
fn pcmCellAcceptedTopologyContains(id:u32)->bool{_=id;return true;}
fn acceptedTemplateRowInvocation(rank:u32)->u32{
 if(rank>=p.counts.x){return INVALID;}return fixture[${capacity}u+rank].x;}
fn rowAccepted(row:u32)->bool{return fixture[row].x!=0u;}
fn rowTermOffset(row:u32)->u32{return row;}
fn rowTermCount(row:u32)->u32{_=row;return 1u;}
fn rowTermRange(row:u32)->vec2u{let first=rowTermOffset(row);return vec2u(first,first+rowTermCount(row));}
fn termCell(term:u32)->u32{return term;}
fn fsm1ChangedOrFlipCell(cell:u32)->bool{return fixture[cell].z!=0u;}
fn ptrTopologyGeneration()->u32{return p.counts.z;}
fn hasStaticSolidVoxels()->bool{return p.counts.w!=0u;}
fn classifyPressureRow(row:u32)->bool{
 let enabled=rowAccepted(row)&&fixture[row].y!=0u;
 theta[row]=select(0u,fixture[row].w,enabled);return enabled;}
${createSparseCM12CanonicalMembershipWGSL({layout,arenaName:'arena'})}
${extract('publishCanonicalPressureRowTile')}
@compute @workgroup_size(64)
${extract('markCanonicalPressureRowRepairTiles')}
@compute @workgroup_size(64)
${extract('compileCanonicalPressureRows')}
@compute @workgroup_size(64)
${extract('compileDirtyCanonicalPressureRows')}
@compute @workgroup_size(1)fn begin(){_=pcmRowBegin(ptrTopologyGeneration());}
@compute @workgroup_size(1)fn finish(){_=pcmRowFinalize(ptrTopologyGeneration());}
`});
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);
    const bgl=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},
      {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
    ]});
    const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[bgl]});
    const names=['begin','markCanonicalPressureRowRepairTiles','compileCanonicalPressureRowRepairTiles',
      'sealCanonicalPressureRowRepairTiles','compileDirtyCanonicalPressureRows','compileCanonicalPressureRows','finish'];
    const pipelines=new Map<string,GPUComputePipeline>();
    for(const name of names)pipelines.set(name,await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module:shader,entryPoint:name}}));
    const buffers:GPUBuffer[]=[];
    const buffer=(size:number,usage:number)=>{const b=device!.createBuffer({size,usage});buffers.push(b);return b;};
    const fixtureBuffer=buffer(capacity*32,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
    const parameters=buffer(32,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
    const indirect=buffer(12,GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST);
    const makeArm=()=>{
      const arena=buffer(layout.totalBytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC);
      const theta=buffer(capacity*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
      device!.queue.writeBuffer(arena,0,new Uint32Array(createSparseCM12CanonicalMembershipInitialWords(layout)));
      const group=device!.createBindGroup({layout:bgl,entries:[arena,fixtureBuffer,theta,parameters].map((b,binding)=>({binding,resource:{buffer:b}}))});
      return {arena,theta,group};
    };
    const compact=makeArm(),full=makeArm();
    const readback=buffer(2*(layout.totalBytes+capacity*4),GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
    const data=new Uint32Array(capacity*8);let generation=1;
    const omittedFromStream=new Set<number>();
    const set=(row:number,accepted:number,enabled:number,changed:number,value=17)=>data.set([accepted,enabled,changed,value],row*4);
    for(const row of [0,63,64,128,130])set(row,1,Number(row!==128&&row!==63),0);
    const run=async(label:string,gravity=0,solids=0,expectedTiles?:number)=>{
      const accepted=Array.from({length:capacity},(_,i)=>i).filter(i=>data[4*i]!==0&&!omittedFromStream.has(i));
      accepted.forEach((row,rank)=>{data[4*(capacity+rank)]=row;});
      device!.queue.writeBuffer(fixtureBuffer,0,data);
      const params=new ArrayBuffer(32);new Uint32Array(params).set([accepted.length,capacity,generation,solids]);
      new Float32Array(params).set([0,-10,0,gravity],4);device!.queue.writeBuffer(parameters,0,params);
      const encoder=device!.createCommandEncoder();
      const dispatch=(arm:typeof compact,name:string,count:number|undefined)=>{
        const pass=encoder.beginComputePass();pass.setBindGroup(0,arm.group);pass.setPipeline(pipelines.get(name)!);
        if(count===undefined)pass.dispatchWorkgroupsIndirect(indirect,0);else pass.dispatchWorkgroups(count);pass.end();
      };
      for(const arm of [compact,full]){
        dispatch(arm,'begin',1);
        if(arm===compact){
          dispatch(arm,'markCanonicalPressureRowRepairTiles',Math.ceil(accepted.length/64));
          dispatch(arm,'compileCanonicalPressureRowRepairTiles',1);
          dispatch(arm,'sealCanonicalPressureRowRepairTiles',1);
          encoder.copyBufferToBuffer(arm.arena,4*(layout.row.repairControlBaseWords+1),indirect,0,12);
          dispatch(arm,'compileDirtyCanonicalPressureRows',undefined);
        }else dispatch(arm,'compileCanonicalPressureRows',Math.ceil(capacity/64));
        dispatch(arm,'finish',1);
      }
      const stride=layout.totalBytes+capacity*4;
      for(const [index,arm]of [compact,full].entries()){
        encoder.copyBufferToBuffer(arm.arena,0,readback,index*stride,layout.totalBytes);
        encoder.copyBufferToBuffer(arm.theta,0,readback,index*stride+layout.totalBytes,capacity*4);
      }
      device!.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
      const words=new Uint32Array(readback.getMappedRange()).slice();readback.unmap();
      const a=words.subarray(0,stride/4),b=words.subarray(stride/4);
      const h=layout.row.headerBaseWords,r=layout.row;
      assert.equal(a[h+6],expectedTiles??a[h+6],`${label}: tiles`);
      assert.equal(a[h+1],1,`${label}: accepted phase`);assert.equal(a[h+4],0,`${label}: fault`);
      assert.equal(a[h+11],b[h+11],`${label}: population`);
      assert.equal(a[h+3],b[h+3],`${label}: generation`);
      assert.deepEqual(a.slice(r.activeBitsBaseWords,r.activeBitsBaseWords+r.activeBitWordCount),b.slice(r.activeBitsBaseWords,r.activeBitsBaseWords+r.activeBitWordCount),`${label}: membership`);
      assert.deepEqual(a.slice(layout.totalWords,layout.totalWords+capacity),b.slice(layout.totalWords,layout.totalWords+capacity),`${label}: theta`);
      if(expectedTiles===0)assert.equal(a[h+7],0,`${label}: no rewritten words`);
      console.log(`${label}: ${a[h+6]} tiles, ${a[h+7]} words, ${a[h+11]} members`);
      for(let i=0;i<capacity;i++)data[4*i+2]=0;
    };
    await run('bootstrap',0,0,3);
    await run('unchanged',0,0,0);
    set(64,1,1,1,29);await run('theta change without membership flip',0,0,1);
    set(64,1,1,1,0);await run('active zero-response row',0,0,1);
    await run('retained active zero-response row',0,0,0);
    set(0,1,0,1);await run('liquid disappears',0,0,1);
    set(63,1,1,1);await run('liquid appears',0,0,1);
    set(130,0,0,0);set(128,0,0,0);generation++;await run('retired final tile absent from accepted stream',0,0,3);
    set(129,1,1,0,33);generation++;await run('recycled final tile',0,0,3);
    set(64,1,0,0);await run('gravity invalidation',1,0,3);
    set(64,1,1,0,42);await run('solid invalidation',0,1,3);
    for(let i=0;i<capacity;i++)set(i,0,0,0);generation++;await run('retire everything',0,0,3);
    await run('empty unchanged',0,0,0);
    for(let i=0;i<capacity;i++)set(i,1,1,1,i+1);
    generation++;await run('all lanes mark shared words',0,0,3);
    await run('dense unchanged after publication',0,0,0);
    set(31,1,0,1);set(32,1,0,1);set(127,1,0,1);set(130,1,0,1);
    await run('word boundaries and final partial word',0,0,3);
    omittedFromStream.add(65);
    set(64,1,1,1,31);set(65,1,1,1,43);
    await run('eligible row outside accepted stream in selected tile',0,0,1);
    await run('uncovered row unchanged with empty worklist',0,0,0);
    set(65,1,1,0,51);await run('uncovered row gravity invalidation',1,0,3);
    assert.deepEqual(errors,[]);for(const b of buffers)b.destroy();
  }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
