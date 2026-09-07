import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(new URL("../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const fn = (name: string) => {
  const result = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(result, name); return result;
};
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
dawnTest("mixed-width transport stencils preserve reflections without changing uniform coarse interpolation", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "transport-stencil-symmetry");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const shader = device.createShaderModule({ code: `
const INVALID=0xffffffffu;
fn cm12RecordFailure(code:u32,cell:u32,data:vec4u){_=code;_=cell;_=data;}
struct TransportStencil{cells:array<u32,8>,weights:array<f32,8>}
struct Owner{cell:u32}
@group(0)@binding(0)var<storage,read>queries:array<vec4f>;
@group(0)@binding(1)var<storage,read_write>result:array<vec4f>;
@group(0)@binding(2)var<storage,read>state:array<f32>;
struct Params{frame:vec4f}
const p=Params(vec4f(1,0,0,0));
fn destinationCellVelocity()->u32{return 0u;}
fn transportCharacteristicStencilClearance(s:TransportStencil)->f32{_=s;return 0.02;}
${fn("transportDepartureCharacteristicClearance")}
@compute @workgroup_size(8)
fn clearance(@builtin(global_invocation_id)id:vec3u){
  var s:TransportStencil;
  for(var i=0u;i<8u;i++){s.cells[i]=(i+id.x)%8u;s.weights[i]=0.125;}
  result[id.x]=vec4f(transportDepartureCharacteristicClearance(s));
}
fn cm12ClampToResidentWorld(q:vec3f,margin:vec3f)->vec3f{return clamp(q,margin,vec3f(32,24,32)-margin);}
fn cellTransportActive(c:u32)->bool{return c!=INVALID;}
fn minimum(c:u32)->vec3u{return vec3u(c%32u,(c/32u)%32u,c/1024u);}
fn scale(q:vec3u)->u32{return select(2u,1u,all(q>=vec3u(8))&&all(q<vec3u(24)));}
fn cellWidths(c:u32)->vec3f{return vec3f(f32(scale(minimum(c))));}
fn cellMinimumWidth(c:u32)->f32{return cellWidths(c).x;}
fn cellCenter(c:u32)->vec3f{return vec3f(minimum(c))+0.5*cellWidths(c);}
fn cm12TransportOwnerAtFine(q:vec3i,direct:bool)->Owner{
  _=direct;let v=vec3u(clamp(q,vec3i(0),vec3i(31,23,31)));let w=scale(v);let m=(v/w)*w;
  return Owner(m.x+32u*(m.y+32u*m.z));
}
${fn("transportSourceSamplingSpans")}
${fn("effectiveTransportStencilAtSpansMode")}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)id:vec3u){
  if(id.x>=arrayLength(&queries)){return;}
  let q=queries[id.x];let owner=cm12TransportOwnerAtFine(vec3i(floor(q.xyz)),true).cell;
  let spans=transportSourceSamplingSpans(owner,true);
  // Signed radial displacement transforms with each reflected query.
  let departure=q.xyz+(q.xyz-vec3f(16))*q.w;
  let stencil=effectiveTransportStencilAtSpansMode(departure,spans,true);
  var value=0.0;var linear=0.0;var total=0.0;
  for(var i=0u;i<8u;i++){
    let c=stencil.cells[i];let weight=stencil.weights[i];
    if(c!=INVALID){let center=cellCenter(c);let d=center-vec3f(16);
      value+=weight*dot(d,d);linear+=weight*center.x;total+=weight;}
  }
  result[id.x]=vec4f(value,linear,total,spans.x);
}` });
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error").map(m => m.message), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shader, entryPoint: "main" } });
    const points: number[][] = [];
    for (const q of [[7, 13, 13], [7, 7, 13], [7, 7, 7], [9.5, 13.5, 13.5]]) {
      for (const motion of [0, 0.01, -0.03]) {
        points.push([...q, motion], [32-q[0]!,q[1]!,q[2]!,motion],
          [q[0]!,q[1]!,32-q[2]!,motion], [q[2]!,q[1]!,q[0]!,motion]);
      }
    }
    // Deterministic interior samples exercise the inverse geometric map at
    // seam faces, edges and corners, and both sides of its dual-cell borders.
    let seed=73;
    const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
    for(let sample=0;sample<1024;sample++){
      const q=[2+28*random(),2+20*random(),2+28*random()];const motion=0.005;
      points.push([...q,motion],[32-q[0]!,q[1]!,q[2]!,motion],
        [q[0]!,q[1]!,32-q[2]!,motion],[q[2]!,q[1]!,q[0]!,motion]);
    }
    for(const z of [7.9,8-2**-21,8,8.1]) for(const y of [23,23.9]){
      const q=[19,y,z],motion=0;
      points.push([...q,motion],[32-q[0]!,q[1]!,q[2]!,motion],
        [q[0]!,q[1]!,32-q[2]!,motion],[q[2]!,q[1]!,q[0]!,motion]);
    }
    // Uniform coarse support: motion must interpolate immediately, at its
    // physical width, rather than sticking until a finest-grid knot is crossed.
    points.push([3,3,3,0], [3,3,3,0.01]);
    const input = device.createBuffer({ size: points.length*16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: input.size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: input.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(input, 0, new Float32Array(points.flat()));
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [input,output].map((buffer,binding) => ({binding,resource:{buffer}})) });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(points.length/64));pass.end();
    encoder.copyBufferToBuffer(output,0,readback,0,output.size);device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(readback.getMappedRange());
    for(let i=0;i<points.length-2;i+=4) for(let transform=1;transform<4;transform++)
      assert.ok(Math.abs(values[4*i]!-values[4*(i+transform)]!)<1e-3,
        `query ${i}, transform ${transform}: ${values[4*i]} vs ${values[4*(i+transform)]}`);
    for(let i=0;i<points.length;i++) assert.ok(Math.abs(values[4*i+2]!-1)<1e-6);
    for(let i=0;i<points.length;i++){
      const q=points[i]!;const expected=q[0]!+(q[0]!-16)*q[3]!;
      assert.ok(Math.abs(values[4*i+1]!-expected)<1e-5,
        `affine query ${i}: ${values[4*i+1]} vs ${expected}`);
    }
    const last=4*(points.length-1);
    assert.equal(values[last+3],2);
    assert.ok(Math.abs(values[last+1]!-2.87)<1e-5);
    readback.unmap();
    const clearance = device.createComputePipeline({layout: "auto",compute:{module:shader,entryPoint:"clearance"}});
    const velocities = device.createBuffer({size:8*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const velocityData = new Float32Array(32); velocityData[4] = -0.0015; velocityData[8] = 0.0015;
    device.queue.writeBuffer(velocities,0,velocityData);
    const clearanceGroup = device.createBindGroup({layout:clearance.getBindGroupLayout(0),entries:[
      {binding:1,resource:{buffer:output}},{binding:2,resource:{buffer:velocities}}]});
    const e = device.createCommandEncoder(), cp=e.beginComputePass();
    cp.setPipeline(clearance);cp.setBindGroup(0,clearanceGroup);cp.dispatchWorkgroups(1);cp.end();
    e.copyBufferToBuffer(output,0,readback,0,8*16);device.queue.submit([e.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    assert.ok(new Float32Array(readback.getMappedRange()).slice(0,32).every(v=>v===0),
      "a 0.003 velocity diameter must reject reuse regardless of which corner comes first");
    readback.unmap();velocities.destroy();input.destroy();output.destroy();readback.destroy();
  } finally {
    device?.destroy(); await releaseWebGPUExclusiveLock();
    assert.ok(gpu);
  }
});

dawnTest("transport dual cells locate graded 1/2/4/8 corners and clipped boundary geometry", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "transport-multirung-locator");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    for (const fixture of ["graded", "clipped", "wall-wedge", "rung-corner", "ocean-junction", "ocean-apex"] as const) {
      const dimensions = fixture === "graded" || fixture.startsWith("ocean-") ? [64,64,64] : fixture !== "clipped" ? [32,32,32] : [13,10,9];
      const shader = device.createShaderModule({code:`
const INVALID=0xffffffffu;
const DIM=vec3u(${dimensions.join(",")});
struct Owner{cell:u32}
struct TransportStencil{cells:array<u32,8>,weights:array<f32,8>}
@group(0)@binding(0)var<storage,read>queries:array<vec4f>;
@group(0)@binding(1)var<storage,read_write>output:array<vec4f>;
@group(0)@binding(2)var<storage,read_write>donors:array<vec2u>;
fn cm12RecordFailure(code:u32,c:u32,data:vec4u){_=code;_=c;_=data;}
fn cm12ClampToResidentWorld(q:vec3f,m:vec3f)->vec3f{
  return clamp(q,m,vec3f(DIM)-m);
}
fn minimum(c:u32)->vec3u{return vec3u(c%64u,(c/64u)%64u,c/4096u);}
fn scale(q:vec3u)->u32{
  ${fixture === "ocean-apex" ? "if(q.x<32u){return select(8u,16u,q.y<32u);}if(q.y<32u&&q.z<32u){return 32u;}if(q.y>=32u&&q.z>=32u){return 4u;}return 16u;" : fixture === "ocean-junction" ? "if(q.y<32u&&q.z<32u){return 32u;}if(q.y<32u||q.z<32u){return 16u;}return select(4u,2u,q.x<32u);" : fixture === "clipped" ? "_=q;return 8u;" : fixture === "wall-wedge" ? "return select(1u,2u,q.x>=16u&&q.y>=16u);" : fixture === "rung-corner" ? "return select(2u,1u,q.x>=24u)*select(2u,1u,q.z>=8u);" : `
  if(all(q>=vec3u(24))&&all(q<vec3u(40))){return 1u;}
  if(all(q>=vec3u(16))&&all(q<vec3u(48))){return 2u;}
  if(all(q>=vec3u(8))&&all(q<vec3u(56))){return 4u;}
  return 8u;`}
}
fn cellWidths(c:u32)->vec3f{return vec3f(min(vec3u(scale(minimum(c))),DIM-minimum(c)));}
fn cellMinimumWidth(c:u32)->f32{return min(cellWidths(c).x,min(cellWidths(c).y,cellWidths(c).z));}
fn cellCenter(c:u32)->vec3f{return vec3f(minimum(c))+0.5*cellWidths(c);}
fn cellTransportActive(c:u32)->bool{return c!=INVALID;}
fn cm12TransportOwnerAtFine(q:vec3i,direct:bool)->Owner{
  _=direct;if(any(q<vec3i(0))||any(q>=vec3i(DIM))){return Owner(INVALID);}
  let v=vec3u(q);let w=scale(v);let m=(v/w)*w;
  return Owner(m.x+64u*(m.y+64u*m.z));
}
${fn("effectiveTransportStencilAtSpansMode")}
${fn("effectiveTransportStencilAtSpansMode").replace("effectiveTransportStencilAtSpansMode", "ordinaryTransportStencil").replace("search<2u", "search<1u")}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)id:vec3u){
  if(id.x>=arrayLength(&queries)){return;}
  let q=queries[id.x].xyz;
  let s=effectiveTransportStencilAtSpansMode(q,vec3f(1),true);
  var moment=vec3f(0);var weightSum=0.0;
  for(var k=0u;k<8u;k++){
    donors[8u*id.x+k]=vec2u(s.cells[k],bitcast<u32>(s.weights[k]));
    if(s.cells[k]==INVALID){continue;}
    moment+=s.weights[k]*cellCenter(s.cells[k]);weightSum+=s.weights[k];
    if(s.weights[k]<0.0){weightSum=-100.0;break;}
  }
  // Recovery must leave every successful ordinary stencil bit-exact.
  let ordinary=ordinaryTransportStencil(q,vec3f(1),true);
  var ordinarySum=0.0;
  for(var k=0u;k<8u;k++){ordinarySum+=ordinary.weights[k];}
  if(ordinarySum>0.0){
    for(var k=0u;k<8u;k++){
      if(s.cells[k]!=ordinary.cells[k]||s.weights[k]!=ordinary.weights[k]){weightSum=-200.0;}
    }
  }
  output[id.x]=vec4f(moment,weightSum);
}`});
      assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
      const points: number[][]=[];
      let seed=481;
      const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
      if(fixture==="graded") {
        // Coarse source centres beside successively finer face/edge/corner
        // support, including the 2/4/8 junction missed by a global minimum step.
        for(const x of [8,16,24,40,48,56]) for(const y of [4,12,20,28,36,44,52,60])
          for(const z of [4,12,20,28,36,44,52,60]) points.push([x,y,z,1],[y,x,z,1],[y,z,x,1]);
        for(let i=0;i<2048;i++) {
          const q=[4+56*random(),4+56*random(),4+56*random()];
          points.push([...q,1],[64-q[0]!,q[1]!,q[2]!,1]);
        }
      } else if(fixture==="ocean-apex") {
        // The next frame-35 departure lies beside a collapsed dual-cell
        // corner. Its inverse needs more than eight Newton iterations.
        points.push([33.99876403808594,34.0017204284668,33.99898529052734,1]);
        for(let i=0;i<8192;i++) points.push([34+.02*(random()-.5),34+.02*(random()-.5),34+.02*(random()-.5),1]);
      } else if(fixture==="ocean-junction") {
        // Frame 35 in the default ocean: a projected Newton cycle at a
        // 32/16/4/2 junction must retain its best exit and locate the pyramid.
        points.push([24,39,31,1],
          [23.251650311984122,32.09116139262915,24.09707679785788,1],
          [30.83250356093049,37.096764665097,31.990629073232412,1],
          [46.00022888183594,33.99953079223633,34.0005874633789,1]);
        for(let i=0;i<1024;i++) points.push([46+.02*(random()-.5),34+.02*(random()-.5),34+.02*(random()-.5),1]);
        for(let i=0;i<8192;i++) points.push([20+12*random(),32+16*random(),24+8*random(),1]);
      } else if(fixture==="rung-corner") {
        // A 4/2/1 corner, with a departure very close to the coarse centre's
        // transverse plane, requires two successive dual-cell crossings.
        points.push([22.34659194946289,26.006866455078125,6.153258323669434,1]);
        for(let i=0;i<8192;i++) points.push([22+2*random(),24+4*random(),6+2*random(),1]);
      } else if(fixture==="wall-wedge") {
        // The wall's dual cell meets a narrow wedge next to a coarse centre.
        // Clamping Newton onto the repeated-node edge makes its Jacobian
        // singular even though this interior query has a positive stencil.
        for(const dx of [0,1e-5,1e-4,.005,.03])
          for(const dy of [0,1e-5,1e-4,.005,.03])
            for(const dz of [0,1e-5,1e-4,.005,.03]) {
              // Below the lowest coarse centre the mirrored ghost geometry
              // retains the interior donor value: only tangential affine
              // fields obey the homogeneous Neumann boundary condition.
              points.push([17-dx,17-dy,1-dz,2]);
              points.push([17-dx,17-dy,1+dz,1]);
            }
      } else {
        // Affine reproduction is meaningful between physical cell centres.
        for(let i=0;i<512;i++) points.push([4+6.5*random(),4+5*random(),4+4.5*random(),1]);
        // Beyond the outermost centres, the Neumann continuation need only
        // preserve a constant field and positive partition, including corners.
        for(const x of [.0001,.5,4,8,10.5,12.9999])
          for(const y of [.0001,.5,4,8,9,9.9999])
            for(const z of [.0001,.5,4,8,8.5,8.9999]) points.push([x,y,z,0]);
      }
      const nodalQueries=new Map<number,number>();
      const cellWidth=(q:readonly number[])=>fixture==="ocean-apex"
        ?(q[0]!<32?(q[1]!<32?16:8):q[1]!<32&&q[2]!<32?32:q[1]!>=32&&q[2]!>=32?4:16)
        :fixture==="ocean-junction"
        ?(q[1]!<32&&q[2]!<32?32:q[1]!<32||q[2]!<32?16:q[0]!<32?2:4)
        :fixture==="clipped"?8
        :fixture==="wall-wedge"?(q[0]!>=16&&q[1]!>=16?2:1)
        :fixture==="rung-corner"?(q[0]!>=24?1:2)*(q[2]!>=8?1:2)
        :q.every(v=>v>=24&&v<40)?1:q.every(v=>v>=16&&v<48)?2
        :q.every(v=>v>=8&&v<56)?4:8;
      // The first moment alone cannot establish cardinal interpolation: tiny
      // leaked neighbour weights can satisfy its residual tolerance but still
      // perturb a zero-characteristic beta/gamma column. Check every actual
      // donor centre, including mixed-face wedges and corner pyramids.
      for(let z=0;z<dimensions[2]!;z++) for(let y=0;y<dimensions[1]!;y++)
        for(let x=0;x<dimensions[0]!;x++) {
          const minimum=[x,y,z],width=cellWidth(minimum);
          if(minimum.some(v=>v%width!==0))continue;
          const centre=minimum.map((v,axis)=>v+.5*Math.min(width,dimensions[axis]!-v));
          nodalQueries.set(points.length,x+64*(y+64*z));points.push([...centre,1]);
        }
      const covariancePairs: Array<[number,number,number]>=[];
      if(fixture==="graded") {
        const originals=points.length;
        for(let i=0;i<originals;i+=7) {
          const q=points[i]!;
          covariancePairs.push([i,points.length,0]);points.push([64-q[0]!,q[1]!,q[2]!,1]);
          covariancePairs.push([i,points.length,1]);points.push([q[2]!,q[1]!,q[0]!,1]);
        }
      }
      const input=device.createBuffer({size:points.length*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      const output=device.createBuffer({size:input.size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      const donors=device.createBuffer({size:points.length*64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      const copy=device.createBuffer({size:input.size+donors.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      device.queue.writeBuffer(input,0,new Float32Array(points.flat()));
      const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:shader,entryPoint:"main"}});
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output,donors].map((buffer,binding)=>({binding,resource:{buffer}}))});
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
      pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(points.length/64));pass.end();
      encoder.copyBufferToBuffer(output,0,copy,0,output.size);
      encoder.copyBufferToBuffer(donors,0,copy,output.size,donors.size);device.queue.submit([encoder.finish()]);
      await copy.mapAsync(GPUMapMode.READ);const mapped=copy.getMappedRange();const values=new Float32Array(mapped);
      for(let i=0;i<points.length;i++) {
        const q=points[i]!;
        assert.ok(Math.abs(values[4*i+3]!-1)<1e-6,`${fixture} partition at ${q}: ${values[4*i+3]}`);
        if(q[3]) for(let axis=0;axis<(q[3]===2?2:3);axis++) assert.ok(Math.abs(values[4*i+axis]!-q[axis]!)<2e-5,
          `${fixture} affine at ${q}, axis ${axis}: ${values[4*i+axis]}`);
      }
      const donorWords=new Uint32Array(mapped,output.size);
      const donorWeights=new Float32Array(mapped,output.size);
      const aggregate=(query:number,transform=-1)=>{
        const weights=new Map<number,number>();
        for(let k=0;k<8;k++) {
          let cell=donorWords[16*query+2*k]!;const weight=donorWeights[16*query+2*k+1]!;
          if(cell===0xffffffff||weight===0)continue;
          const q=[cell%64,Math.floor(cell/64)%64,Math.floor(cell/4096)];
          if(transform===0) {
            const width=q.every(v=>v>=24&&v<40)?1:q.every(v=>v>=16&&v<48)?2:q.every(v=>v>=8&&v<56)?4:8;
            q[0]=64-q[0]!-width;
          } else if(transform===1) [q[0],q[2]]=[q[2]!,q[0]!];
          cell=q[0]!+64*(q[1]!+64*q[2]!);weights.set(cell,(weights.get(cell)??0)+weight);
        }
        return weights;
      };
      for(const [query,cell] of nodalQueries) {
        const weights=aggregate(query);
        assert.deepEqual([...weights],[[cell,1]],
          `${fixture} exact cardinal weights at centre ${points[query]}`);
      }
      for(const [a,b,transform] of covariancePairs) {
        const expected=aggregate(a,transform),actual=aggregate(b);
        for(const cell of new Set([...expected.keys(),...actual.keys()]))
          assert.ok(Math.abs((expected.get(cell)??0)-(actual.get(cell)??0))<3e-6,
            `donor covariance at ${points[a]}, transform ${transform}, cell ${cell}: ${expected.get(cell)} vs ${actual.get(cell)}`);
      }
      copy.unmap();input.destroy();output.destroy();donors.destroy();copy.destroy();
    }
  } finally {device?.destroy();await releaseWebGPUExclusiveLock();assert.ok(gpu);}
});
