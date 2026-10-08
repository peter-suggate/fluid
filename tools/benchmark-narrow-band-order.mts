/** Matched order-only / full bin-maintenance microbenchmarks.
 * --baseline=/path/to/49fd3505-worktree [--candidate=/path/to/candidate]
 * --maintenance includes clear/count/order; default measures cursor-clear/order.
 * --mid selects 8M cells, --large selects 16M; default is 1M cells.
 * Both variants share identical manufactured particle inputs; sparse is forced
 * at every size to expose why production keeps the small-domain dense path.
 * The measurements include command encoding and queue completion, not a frame.
 */
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {writeFileSync,mkdirSync} from 'node:fs';
import {resolve,dirname} from 'node:path';
const argument=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.split('=').slice(1).join('=');
const baseline=argument('baseline');assert.ok(baseline,'Pass --baseline=/absolute/path/to/the/immutable/baseline');
const base=resolve(baseline),sparse=resolve(argument('candidate')??process.cwd());
const maintenance=process.argv.includes('--maintenance');
const load=(p:string)=>import(pathToFileURL(p).href);
const dawn=await load(base+'/node_modules/webgpu/index.js');Object.assign(globalThis,dawn.globals);
const {createProcessRetainedDawnGPU}=await load(base+'/lib/harness/node-dawn-provider.ts');
const {managedGPUDevice}=await load(base+'/lib/core/gpu-compilation-manager.ts');
const {requiredFluidDeviceLimits}=await load(base+'/lib/core/webgpu-device-limits.ts');
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']);const adapter=await gpu.requestAdapter();const raw=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});const device=managedGPUDevice(raw,{requireWorkerRealm:false});
const errors:string[]=[];device.addEventListener('uncapturederror',(e:any)=>{errors.push(e.error.message);console.error(e.error.message);});
const large=process.argv.includes('--large'),mid=process.argv.includes('--mid'),dims=large?[256,256,256]:mid?[256,256,128]:[128,128,64],cells=dims.reduce((n,v)=>n*v,1),tiles=cells/64,capacity=524288;
const buffer=(size:number)=>device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
const particles=[buffer(capacity*48),buffer(capacity*48)],bins=buffer((2*cells+tiles)*4),links=buffer((cells+4*capacity)*4),state=buffer(48);
const module=device.createShaderModule({code:`
struct Particle{position:vec4f,velocity:vec4f,before:vec4f}
@group(0) @binding(0) var<storage,read> particles:array<Particle>;
@group(0) @binding(1) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read> state:array<u32>;
override sparse:bool=false;
@compute @workgroup_size(64) fn count(@builtin(global_invocation_id) id:vec3u){
 for(var i=id.x;i<state[0];i+=65536u){let p=vec3u(particles[i].position.xyz);let t=p/4u;let l=p%4u;let tile=t.x+${dims[0]/4}u*(t.y+${dims[1]/4}u*t.z);let cell=64u*tile+l.x+4u*l.y+16u*l.z;atomicAdd(&bins[cell],1u);if(sparse){atomicAdd(&bins[${2*cells}u+tile],1u);}}
}`});
const countLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}}]});
const countGroup=device.createBindGroup({layout:countLayout,entries:[particles[1],bins,state].map((buffer,binding)=>({binding,resource:{buffer}}))});
const countPipelines=maintenance?await Promise.all([false,true].map(sparse=>device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[countLayout]}),compute:{module,entryPoint:'count',constants:{sparse:+sparse}}}))):[];
const variants=[];for(const [label,path] of [['base',base],['sparse',sparse]]){const {UniformNarrowBandOrder}=await load(path+'/lib/methods/uniform/uniform-narrow-band-order.ts');const order=new UniformNarrowBandOrder(device,dims,particles,bins,links,state,label==='sparse');await order.initialize();variants.push({label,order});}
const rows=[];
for(const occupancy of (large?[0.001,0.005,0.0078125]:mid?[0.002,0.01,0.015625]:[0.01,0.1,0.125])){
 const active=Math.floor(tiles*occupancy),count=Math.min(capacity,active*64*4),samples=new Float32Array(capacity*12),counts=new Uint32Array(2*cells+tiles);
 for(let i=0;i<count;i++){const order=Math.floor(i/4),tile=Math.floor(order/64),cell=order%64;const x=(tile%(dims[0]/4))*4+cell%4,y=(Math.floor(tile/(dims[0]/4))%(dims[1]/4))*4+Math.floor(cell/4)%4,z=Math.floor(tile/(dims[0]*dims[1]/16))*4+Math.floor(cell/16);samples.set([x+0.25,y+0.25,z+0.25,1,i,0,0,0,0,0,0,0],12*i);counts[order]++;counts[2*cells+tile]++;}
 device.queue.writeBuffer(particles[1],0,samples);device.queue.writeBuffer(bins,0,counts);device.queue.writeBuffer(state,0,new Uint32Array([count,0,0,0]));
 for(let round=0;round<6;round++){
  for(const v of (round%2?[...variants].reverse():variants)){
   const begin=performance.now();const e=device.createCommandEncoder();for(let i=0;i<20;i++){
    if(maintenance){
     if(v.label==='base'){e.clearBuffer(bins,0,cells*8);}else{v.order.prepare(e,0);}
     const p=e.beginComputePass();p.setPipeline(countPipelines[v.label==='base'?0:1]);p.setBindGroup(0,countGroup);p.dispatchWorkgroups(1024);p.end();
    }else{e.clearBuffer(bins,cells*4,cells*4);}
    v.order.encode(e,0);
   }device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();const ms=(performance.now()-begin)/20;
   rows.push({occupancy,count,round,variant:v.label,ms});console.log(JSON.stringify(rows.at(-1)));assert.deepEqual(errors,[]);
  }
 }
}
// Force sparse even below the production cutoff: this intentionally
// reproduces the rejected unconditional policy at one million cells.
const file=resolve(argument('output')??`docs/verification/sparse-${maintenance?'maintenance':'order'}-bench${large?'-large':mid?'-mid':''}-replay.json`);
mkdirSync(dirname(file),{recursive:true});
writeFileSync(file,JSON.stringify({dims,rows,errors,baseline:base,candidate:sparse,maintenance,
 forcedSparse:true,defaultSparse:cells>=8_388_608,repetitionsPerRow:20,discardWarmupRound:0},null,2)+'\n');
for(const v of variants)v.order.destroy();for(const b of [...particles,bins,links,state])b.destroy();device.destroy();
