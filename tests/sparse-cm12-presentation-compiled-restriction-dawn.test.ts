import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(new URL("../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const functionSource = (name: string) => {
  const fn = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(fn); return fn;
};
const restrict = ["presentationTeiWord", "presentationLoadLeaf", "presentationLeafMass", "presentationCompiledDensityAt"].map(functionSource).join("\n");
const live = new Set<GPU>();
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("presentation restriction matches exhaustive volume sums across empty tiles and mixed query widths", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "presentation-empty-restriction");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create(["backend=metal"]); live.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice(); assert.ok(device);
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const queries: number[] = [], expected: number[] = [];
    const density = (x: number, y: number, z: number) => {
      if (Math.min(x,y,z)<0 || Math.max(x,y,z)>=24) return 0;
      const bx=Math.floor(x/8), by=Math.floor(y/8), bz=Math.floor(z/8);
      const brick=bx+3*(by+3*bz);
      if(brick%3===1) return 0;
      const scale=1<<(Math.floor(brick/3)%4), resolution=8/scale;
      const cell=Math.floor(x%8/scale)+resolution*(Math.floor(y%8/scale)+resolution*Math.floor(z%8/scale));
      return ((brick*13+cell*7)%257)/256;
    };
    for (const width of [1,2,4,8,16]) for(let i=0;i<240;i++) {
      const x=(i%7-2)*width, y=(Math.floor(i/7)%5-1)*width, z=(Math.floor(i/35)-1)*width;
      queries.push(x,y,z,width); let sum=0;
      for(let dz=0;dz<width;dz++) for(let dy=0;dy<width;dy++) for(let dx=0;dx<width;dx++) sum+=density(x+dx,y+dy,z+dz);
      expected.push(sum/width**3);
    }
    const shader = `
const INVALID=0xffffffffu;const BRICK_FINE_RESOLUTION=8u;
struct Params{dimensions:vec4u} const p=Params(vec4u(24));
@group(0)@binding(0)var<storage,read>state:array<f32>;
@group(0)@binding(1)var<storage,read_write>activity:array<atomic<u32>>;
@group(0)@binding(2)var<storage,read>queries:array<vec4i>;
@group(0)@binding(3)var<storage,read_write>output:array<f32>;
@group(0)@binding(4)var<storage,read>partials:array<vec4f>;
const CM12_TEI_LEAF_CAPACITY=27u;const CM12_TEI_LEAF_WORDS=8u;
fn cm12TeiLeafBase(slot:u32)->u32{return slot*216u;}
fn cm12TeiScaleLog2(scale:u32,descriptor:u32)->u32{_=scale;return descriptor;}

fn cm12WorldFloorToSpan(v:i32,s:i32)->i32{return i32(floor(f32(v)/f32(s)))*s;}
fn brickDirectoryLookupAtSignedCoordinate(q:vec3i)->u32{
 if(any(q<vec3i(0))||any(q>=vec3i(3))){return INVALID;}
 let brick=u32(q.x+3*(q.y+3*q.z));if(brick%3u==1u){return INVALID;}return brick;
}
fn cm12WorldLeafCoordinate(b:u32)->vec3i{return vec3i(i32(b%3u),i32((b/3u)%3u),i32(b/9u));}
fn brickSpan(b:u32)->u32{_=b;return 1u;}
fn brickActive(b:u32)->bool{return b<27u;}
fn brickHasUnclippedWorldGeometry(b:u32)->bool{_=b;return false;}
fn cellOpenFraction(c:u32)->f32{_=c;return 1.;}
fn activityRecord(b:u32)->u32{return b*2u;}
fn templateBrickCellRange(b:u32,r:u32)->vec2u{_=r;return vec2u(b*512u,64u);}
fn compactOwnerCellAt(q:vec3i)->vec3u{
 let b=brickDirectoryLookupAtSignedCoordinate(vec3i(floor(vec3f(q)/8.)));
 if(b==INVALID){return vec3u(INVALID);}
 let local=vec3u(q-cm12WorldLeafCoordinate(b)*8)/2u;
 return vec3u(b*512u+local.x+4u*(local.y+4u*local.z),b,4u);
}
struct CM12TransportLeaf{generation:u32,flags:u32,first:u32,count:u32,
 owner:u32,valid:vec3u,scale:u32,scaleLog2:u32}
fn presentationLeafAt(q:vec3i)->CM12TransportLeaf{
 let b=brickDirectoryLookupAtSignedCoordinate(vec3i(floor(vec3f(q)/8.)));
 return presentationLoadLeaf(0u,b);
}
${restrict}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id)gid:vec3u){
 if(gid.x>=arrayLength(&queries)){return;}let q=queries[gid.x];
 output[gid.x]=presentationCompiledDensityAt(q.xyz,q.w,0u);
}`;
    const pipeline=device.createComputePipeline({layout:"auto",compute:{module:device.createShaderModule({code:shader}),entryPoint:"main"}});
    const makeBuffer=(data:ArrayBufferView,usage:number) => { const b=device!.createBuffer({size:data.byteLength,usage:usage|GPUBufferUsage.COPY_DST}); device!.queue.writeBuffer(b,0,data.buffer,data.byteOffset,data.byteLength);return b; };
    const state=new Float32Array(27*512), activity=new Uint32Array(54);
    for(let b=0;b<27;b++) {activity[b*2+1]=64;for(let c=0;c<512;c++) state[b*512+c]=((b*13+c*7)%257)/256;}
    const image=new Uint32Array(27*8);
    for(let b=0;b<27;b++) {
      const shift=Math.floor(b/3)%4, scale=1<<shift, resolution=8/scale;
      image.set([1,(0x80000000|resolution)>>>0,b*512,resolution**3,b,
        resolution|(resolution<<5)|(resolution<<10),scale,shift],b*8);
    }
    const buffers=[makeBuffer(state,GPUBufferUsage.STORAGE),makeBuffer(activity,GPUBufferUsage.STORAGE),makeBuffer(new Int32Array(queries),GPUBufferUsage.STORAGE),device.createBuffer({size:expected.length*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),makeBuffer(image,GPUBufferUsage.STORAGE)];
    const readback=device.createBuffer({size:expected.length*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
    const encoder=device.createCommandEncoder(), pass=encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(expected.length/64));pass.end();
    encoder.copyBufferToBuffer(buffers[3],0,readback,0,readback.size);device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);const actual=Array.from(new Float32Array(readback.getMappedRange()));
    assert.deepEqual(actual,expected);assert.deepEqual(errors,[]);
    readback.unmap();readback.destroy();buffers.forEach(b=>b.destroy());
  } finally {device?.destroy();await releaseWebGPUExclusiveLock();}
});
