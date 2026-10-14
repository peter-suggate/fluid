import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandSparseWGSL } from "../lib/methods/uniform/uniform-narrow-band-sparse.wgsl";
import { narrowBandParticleStride, narrowBandParticleWGSL } from "../lib/methods/uniform/uniform-narrow-band-apic.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

test("GPU sparse lifetime preserves sheets and dense droplets, expires isolated samples, and resets on recovery",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("sparse particle lifetime",async device=>{
  type Sample={p:number[];name:string};const samples:Sample[]=[];
  const add=(name:string,p:number[])=>samples.push({name,p});
  add("single",[4.3,4.3,4.3]);add("pair",[10.1,4.3,4.3]);add("pair",[10.5,4.3,4.3]);
  // Eight samples including self are insufficient, even at a physical wall.
  for(let i=0;i<8;i++)add("sparse cluster at wall",[0.1+0.1*(i&1),0.1+0.1*((i>>1)&1),0.1+0.1*(i>>2)]);
  for(let y=0;y<5;y++)for(let x=0;x<5;x++)add("thin sheet",[20+0.5*x,4.5+0.5*y,4.5]);
  for(let z=0;z<3;z++)for(let y=0;y<3;y++)for(let x=0;x<3;x++)add("dense droplet",[40+0.4*x,4.5+0.4*y,4.5+0.4*z]);
  for(let i=0;i<9;i++)add("page boundary",[31.6+0.1*i,16.5,16.5]);
  // Close in cell index, outside the spherical distance cutoff.
  add("outside radius",[4.01,24.01,4.01]);
  for(let i=0;i<9;i++)add("distant neighbours",[6.9,26.9,6.9+0.001*i]);
  const key=(p:number[])=>Math.floor(p[0]!)+64*(Math.floor(p[1]!)+32*Math.floor(p[2]!));
  samples.sort((a,b)=>key(a.p)-key(b.p));
  const runs=new Uint32Array(64*32*32*2);
  samples.forEach((s,i)=>{const k=key(s.p);if(!runs[2*k+1])runs[2*k]=i;runs[2*k+1]=i+1;});
  const expectedSparse=samples.map((s,i)=>samples.filter((other,j)=>i!==j&&s.p.reduce((sum,v,k)=>sum+(v-other.p[k]!)**2,0)<=4).length<8);
  for(const apic of [false,true]){
   const stride=narrowBandParticleStride(apic),ageOffset=stride-4;
   const data=new Float32Array(stride*samples.length);samples.forEach((s,i)=>{data.set(s.p,stride*i);data[stride*i+11]=i%2;});
   const particles=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
   const bins=device.createBuffer({size:runs.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
   const params=device.createBuffer({size:48,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
   device.queue.writeBuffer(particles,0,data);device.queue.writeBuffer(bins,0,runs);
   try{
    const module=device.createShaderModule({code:`
${narrowBandParticleWGSL(apic)}
struct Params{hDt:vec4f,settings:vec4f,activity:vec4f}
@group(0) @binding(0) var<storage,read_write> particles:array<Particle>;
@group(0) @binding(1) var<storage,read> bins:array<vec2u>;
@group(0) @binding(2) var<uniform> params:Params;
fn nbCellRun(c:vec3i)->vec2u{if(any(c<vec3i(0))||any(c>=vec3i(64,32,32))){return vec2u(0);}return bins[u32(c.x+64*(c.y+32*c.z))];}
fn nbPosition(i:u32)->vec3f{return particles[i].position.xyz;}
${narrowBandSparseWGSL}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;if(i>=arrayLength(&particles)){return;}
 let age=nbSparseAge(i,particles[i].position.xyz,particles[i].lifetime.x);
 particles[i].lifetime=vec4f(age,f32(nbSparseExpired(age)),0,0);
}`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(errors.map(m=>`${m.lineNum}: ${m.message}`),[]);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:particles}},{binding:1,resource:{buffer:bins}},{binding:2,resource:{buffer:params}}]});
    const advance=async(dt:number,lifetime:number,steps:number)=>{
     const words=new Float32Array(12);words[0]=0.1;words[1]=0.4;words[2]=0.02;words[3]=dt;words[9]=lifetime;device.queue.writeBuffer(params,0,words);
     const e=device.createCommandEncoder();for(let step=0;step<steps;step++){const p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(Math.ceil(samples.length/64));p.end();}device.queue.submit([e.finish()]);
     return readMixedBuffer(device,particles);
    };
    const check=(values:Float32Array,age:number,expired:boolean)=>samples.forEach((s,i)=>{
     assert.ok(Math.abs(values[stride*i+ageOffset]!-(expectedSparse[i]?age:0))<2e-6,`${apic?"APIC":"FLIP"} ${s.name} age ${values[stride*i+ageOffset]}`);
     assert.equal(values[stride*i+ageOffset+1],Number(expectedSparse[i]&&expired),s.name);
    });
    check(await advance(0.125,2,15),1.875,false);
    check(await advance(0.125,2,1),2,true);
    check(await advance(0.1,0,1),0,false);
    check(await advance(0.0625,2,30),1.875,false);
    check(await advance(0.0625,2,2),2,true);
    // Dense samples with an old isolation age recover immediately; new ones
    // begin at zero. This also verifies the APIC affine rows aren't the age.
    samples.forEach((_,i)=>{data[stride*i+ageOffset]=1.875;});device.queue.writeBuffer(particles,0,data);
    check(await advance(0.125,2,1),2,true);
   }finally{particles.destroy();bins.destroy();params.destroy();}
  }
 });
});

test("production solver expires sparse orphans and releases their page while retaining a dense droplet",{skip:!process.env.WEBGPU_NODE_MODULE},async t=>{
 const {nbflipDomain}=await import("../lib/core/nbflip-paper-scenes");
 const {createUniformSolver,advanceUniform}=await import("./helpers/uniform-geometric");
 const {UNIFORM_PAGE_POOL_STATUS:S,UNIFORM_PAGE_POOL_RECORD:R,UNIFORM_PAGE_POOL_BLOCK:B}=await import("../lib/methods/uniform/uniform-page-pool");
 type Stage=import("../lib/methods/uniform/uniform-narrow-band-flip").UniformNarrowBandFlip;
 type Pool=import("../lib/methods/uniform/uniform-page-pool").UniformPagePool;
 await withUniformDevice("sparse particle residency",async device=>{
  for(const particleTransfer of ["flip","apic"]){
  // No gravity or sources: isolated stationary samples cannot escape the
  // page by moving, cooling, or falling deep into a resolved liquid body.
  const dt=0.0625,scene=nbflipDomain("sparse-particle-residency",[128,32,32],0.02,32,0,undefined,dt,0);
  // Seed all four pages so the injected epoch starts in resident storage.
  scene.fluid.pagePool={blocks:6,air:[]};
  const solver=await createUniformSolver(device,scene,{volumeControlSeconds:0,sparseParticleLifetime:0,particleTransfer});
  const frame=(solver as unknown as {mixedFrame:{narrowBandFlip:Stage;pagePool:Pool;ownership:{support:GPUBuffer;capacity:{tiles:number}}}}).mixedFrame;
  const stage=frame.narrowBandFlip,pool=frame.pagePool,stride=stage.particleStride;
  try{
   const points:number[][]=[];
   for(let i=0;i<3;i++)points.push([8+4*i,16,16]);
   for(let z=0;z<3;z++)for(let y=0;y<3;y++)for(let x=0;x<3;x++)points.push([108+0.4*x,16+0.4*y,16+0.4*z]);
   const words=new Float32Array(stride*points.length);
   points.forEach((p,i)=>{words.set([...p,2],stride*i);words[stride*i+11]=1;});
   device.queue.writeBuffer(stage.activeParticles,0,words);device.queue.writeBuffer(stage.state,0,new Uint32Array([points.length,points.length]));
   // Inject a completed particle epoch instead of bootstrapping a liquid
   // description. Every subsequent order, update and compaction is real.
   (stage as unknown as {started:boolean}).started=true;
   const snapshot=async()=>{
    const receipt=new Uint32Array((await readMixedBuffer(device,stage.state)).buffer),count=receipt[0]!;
    const data=await readMixedBuffer(device,stage.activeParticles),pages=[];
    const poolWords=new Uint32Array((await readMixedBuffer(device,pool.state)).buffer);
    for(let b=1;b<pool.layout.blocks-1;b++){const r=pool.layout.state.records+R.words*b;if(poolWords[r+R.state]===B.resident)pages.push(poolWords[r]!);}
    const support=new Uint32Array((await readMixedBuffer(device,frame.ownership.support)).buffer);
    assert.equal(support[9*frame.ownership.capacity.tiles+29],0,"residency support certificate");
    assert.equal(receipt[3],0);assert.equal(poolWords[S.fatal],0);
    let sparse=0,dense=0,maxAge=0;
    for(let i=0;i<count;i++){if(data[i*stride]!<32){sparse++;maxAge=Math.max(maxAge,data[i*stride+stride-4]!);}else dense++;}
    return {sparse,dense,maxAge,pages};
   };
   for(let step=1;step<=6;step++)await advanceUniform(solver,step*dt);
   const disabled=await snapshot();assert.equal(disabled.sparse,3);assert.equal(disabled.maxAge,0);assert.ok(disabled.pages.includes(0));
   solver.applyRuntimeValues({timeStep:"scene",sparseParticleLifetime:0.25});
   for(let step=7;step<=9;step++)await advanceUniform(solver,step*dt);
   const grace=await snapshot();assert.equal(grace.sparse,3);assert.ok(grace.maxAge>0&&grace.maxAge<0.25);assert.ok(grace.pages.includes(0));
   await advanceUniform(solver,10*dt);
   const expired=await snapshot();assert.equal(expired.sparse,0,"orphan samples compact out at their lifetime");assert.ok(expired.dense>=27,"dense droplet survives");
   for(let step=11;step<=16;step++)await advanceUniform(solver,step*dt);
   const retired=await snapshot();assert.equal(retired.sparse,0);assert.ok(!retired.pages.includes(0),"orphan page retires");assert.ok(retired.pages.includes(3),"droplet page stays resident");assert.ok(retired.dense>=27);
   t.diagnostic(JSON.stringify({particleTransfer,disabled,grace,expired,retired}));
  }finally{solver.destroy();}
  }
 },["subgroups"]);
});
