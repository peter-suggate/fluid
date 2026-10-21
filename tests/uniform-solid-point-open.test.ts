import assert from 'node:assert/strict';
import test from 'node:test';
import {uniformSolidPointCellWGSL,uniformSolidSlideStartWGSL,uniformSolidPointOpenWGSL,uniformSolidSampleClosedWGSL} from '../lib/methods/uniform/uniform-solid-point-open.wgsl';
import {withUniformDevice} from './helpers/uniform-geometric';
import {readMixedBuffer} from './helpers/uniform-mixed-native-fields';
import {readFileSync} from 'node:fs';

test('adaptive surface interpolation reads liquid heat on every reflected wall',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('wall interpolation policy',async device=>{
  const source=readFileSync(new URL('../lib/methods/uniform/uniform-mixed-surface.ts',import.meta.url),'utf8');
  const start=source.indexOf('fn umCubicAt('),raw=source.slice(start,source.indexOf('// A deep air vertex',start));
  const policy=Function('return `'+raw+'`;').call({farGate:true,solid:true,claimBytes:0,ownership:{capacity:{tiles:64}}}) as string;
  const claims=device.createBuffer({size:128*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const heat=new Float32Array(128);
  for(let z=0;z<2;z++)for(let y=0;y<2;y++)for(let x=0;x<2;x++)heat[64+x+1+4*(y+1)+16*(z+1)]=2;
  device.queue.writeBuffer(claims,0,heat);
  const output=device.createBuffer({size:27*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> umClaims:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> output:array<u32>;
struct Params {flags:vec4u};var<private> params:Params;
fn umCellIsOpen(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(8));}
fn umTileAt(p:vec3u)->u32{return p.x+1u+4u*(p.y+1u)+16u*(p.z+1u);}
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn umTileKey(p:vec3i)->u32{return u32(p.x+1+4*(p.y+1)+16*(p.z+1));}
fn umKeyStored(t:u32)->bool{return t<64u;}
${uniformSolidPointOpenWGSL}
${uniformSolidSlideStartWGSL}
${uniformSolidPointCellWGSL}
${policy}
@compute @workgroup_size(32) fn main(@builtin(local_invocation_index) i:u32){
 if(i>=27u){return;}params.flags=vec4u(0,3,0,0);
 output[i]=u32(umCubicAt(4u*vec3u(i%3u,(i/3u)%3u,i/9u),1u));
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[claims,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
   for(const hotSide of [-1,0,1]){
    heat.fill(0);
    for(let z=0;z<2;z++)for(let y=0;y<2;y++)for(let x=0;x<2;x++)
     heat[64+x+1+4*(y+1)+16*(z+1)]=hotSide<0||z===hotSide?2:0;
    device.queue.writeBuffer(claims,0,heat);
    const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();device.queue.submit([e.finish()]);
    // The middle plane touches hot particles under either reflection.
    // Only the far cold wall uses cubic interpolation.
    const expected=Array.from({length:27},(_,i)=>Number(hotSide>=0&&Math.floor(i/9)===(hotSide===0?2:0)));
    assert.deepEqual(Array.from(new Uint32Array((await readMixedBuffer(device,output)).buffer)),expected,`hot side ${hotSide}`);
   }
  }finally{claims.destroy();output.destroy();}
 });
});

test('solid point contact is reflection invariant on faces, edges and corners without admitting solid interiors',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('symmetric wall contact',async device=>{
  const points:number[][]=[];
  for(const x of [-.001,0,.25,4,7.999,8,8.001])for(const y of [0,.7,8])for(const z of [0,.8,8])points.push([x,y,z,0]);
  const input=device.createBuffer({size:16*points.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(input,0,new Float32Array(points.flat()));
  const output=device.createBuffer({size:16*points.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> points:array<vec4f>;
@group(0) @binding(1) var<storage,read_write> result:array<vec4u>;
fn umCellIsOpen(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(8));}
fn umSolidEnabled()->bool{return true;}
fn umTileKey(p:vec3i)->u32{return u32((p.x+1)+4*(p.y+1)+16*(p.z+1));}
fn umKeyStored(t:u32)->bool{return t<64u;}
var<private> fixtureWidth:u32;
fn umTileWidth(t:u32)->u32{return fixtureWidth;}
fn umTileOpenAt(p:vec3i)->f32{return f32(all(p>=vec3i(0))&&all(p<vec3i(2)));}
fn umTileOpen(t:u32)->f32{return umTileOpenAt(vec3i(i32(t%4u)-1,i32((t/4u)%4u)-1,i32(t/16u)-1));}
${uniformSolidPointOpenWGSL}
${uniformSolidSlideStartWGSL}
${uniformSolidPointCellWGSL}
${uniformSolidSampleClosedWGSL}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3u){if(g.x<arrayLength(&points)){fixtureWidth=1u;let fine=u32(!umSampleClosed(points[g.x].xyz));fixtureWidth=4u;result[g.x]=vec4u(u32(umOpenAt(points[g.x].xyz)),fine,u32(!umSampleClosed(points[g.x].xyz)),u32(umOpenAt(points[g.x].xyz)&&!umCellIsOpen(umOpenPointCell(points[g.x].xyz))));}}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(Math.ceil(points.length/64));p.end();device.queue.submit([e.finish()]);
   const actual=new Uint32Array((await readMixedBuffer(device,output)).buffer);
   assert.deepEqual(Array.from(actual),points.flatMap(p=>[...Array(3).fill(Number(p.slice(0,3).every(v=>v>=0&&v<=8))),0]));
  }finally{input.destroy();output.destroy();}
 });
});



test('wall-start traces clip only outward normal motion at every reflected face and corner',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('wall trace sliding',async device=>{
  const cases:number[][]=[],expected:number[][]=[];
  for(const x of [0,4,8])for(const y of [0,4,8])for(const z of [0,4,8])
   for(const dx of [-.125,0,.125])for(const dy of [-.125,0,.125])for(const dz of [-.125,0,.125]){
    const p=[x,y,z],d=[dx,dy,dz];cases.push([...p,0,...p.map((v,a)=>v+d[a]!),0]);
    expected.push([...p.map((v,a)=>Math.min(8,Math.max(0,v+d[a]!))),0]);
   }
  const input=device.createBuffer({size:32*cases.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:16*cases.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  device.queue.writeBuffer(input,0,new Float32Array(cases.flat()));
  try{
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> input:array<vec4f>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4f>;
fn umCellIsOpen(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(8));}
${uniformSolidPointOpenWGSL}
${uniformSolidSlideStartWGSL}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x<arrayLength(&output)){output[id.x]=vec4f(umSlideStart(input[2u*id.x].xyz,input[2u*id.x+1u].xyz),0);}
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]});
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(cases.length/64));pass.end();device.queue.submit([encoder.finish()]);
   const actual=new Float32Array((await readMixedBuffer(device,output)).buffer);
   for(let i=0;i<cases.length;i++)assert.deepEqual(Array.from(actual.slice(4*i,4*i+4)),expected[i],`trace ${cases[i]}`);
  }finally{input.destroy();output.destroy();}
 });
});
