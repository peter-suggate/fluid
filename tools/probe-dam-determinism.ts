/** Find the first field/particle difference between fresh, identically configured runs.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-dam-determinism.ts
 * --canonical-order is a diagnostic control: sort each cell's particle records
 * before reconstruction/P2G, without changing the particle multiset.
 */
import { damProbeOrder } from './dam-probe-order';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHighResolutionDamBreakScene } from '../lib/core/scenes';
import { uniformDetailField } from '../lib/methods/uniform/uniform-detail-fields';
import { createUniformSolver, withUniformDevice, advanceUniform } from '../tests/helpers/uniform-geometric';
import { paceDawnEventPolling } from '../lib/harness/node-dawn-provider';
const canonicalOrder=process.argv.includes('--canonical-order');
const option=(name:string)=>process.argv.find(a=>a.startsWith(`--${name}=`))?.slice(name.length+3);
const output=option('output')??`/tmp/fluid-determinism${canonicalOrder?'-canonical':''}`;
const steps=Number(option('steps')??2);
await mkdir(output,{recursive:true});paceDawnEventPolling();
const hash=(a:ArrayBufferView)=>createHash('sha256').update(new Uint8Array(a.buffer,a.byteOffset,a.byteLength)).digest('hex').slice(0,20);
await withUniformDevice('dam determinism',async device=>{
 for(let run=0;run<2;run++){
  const scene=createHighResolutionDamBreakScene();scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
  const solver=await createUniformSolver(device,scene);const f=(solver as any).mixedFrame,nb=f.narrowBandFlip;
  const canonicalize=canonicalOrder?await damProbeOrder(device,nb):undefined;
  let step=0,sequence=0;const pending:{name:string;get:()=>Promise<Float32Array>;particles?:boolean}[]=[];
  const buffer=(e:GPUCommandEncoder,b:GPUBuffer,size=b.size)=>{
   const read=device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});e.copyBufferToBuffer(b,0,read,0,size);
   return async()=>{try{await read.mapAsync(GPUMapMode.READ);return new Float32Array(read.getMappedRange()).slice();}finally{read.unmap();read.destroy();}};
  };
  const texture=(e:GPUCommandEncoder,name:string,t:GPUTexture)=>pending.push({name:`${step}-${sequence++}-${name}`,get:uniformDetailField(t)!.storage.capture(e,t)});
  const particles=(e:GPUCommandEncoder,name:string,b:GPUBuffer,countWord:number)=>{
   const count=buffer(e,nb.state,16),data=buffer(e,b);
   pending.push({name:`${step}-${sequence++}-${name}`,particles:true,get:async()=>{
    const state=new Uint32Array((await count()).buffer),values=await data();const n=state[countWord]!;
    console.log(JSON.stringify({run,step,name,particles:n,stride:nb.particleStride}));return values.slice(0,n*nb.particleStride);
   }});
  };
  const hook=(o:any,key:string,action:(e:GPUCommandEncoder)=>void,before=false)=>{const original=o[key].bind(o);o[key]=(...args:any[])=>{if(before)action(args[0]);const result=original(...args);if(!before)action(args[0]);return result;};};
  hook(nb,'bootstrap',e=>{texture(e,'pre-seed-phi',f.fields.phi);texture(e,'pre-seed-velocity',f.fields.velocityScratch);},true);
  hook(nb,'bootstrap',e=>particles(e,'seeded',nb.activeParticles,0));
  const order=nb.order.encode.bind(nb.order);nb.order.encode=(e:GPUCommandEncoder,parity:number)=>{
   order(e,parity);
   canonicalize?.(e,parity);
   particles(e,'ordered',nb.particles[parity],1);
  };
  hook(f.surface,'encode',e=>texture(e,'advected-phi',f.fields.phiScratch));
  hook(nb,'reconstruct',e=>texture(e,'reconstructed-phi',f.fields.phi));
  hook(nb,'transfer',e=>texture(e,'pre-p2g-velocity',f.fields.departure),true);
  hook(nb,'transfer',e=>texture(e,'p2g-velocity',f.fields.velocity));
  hook(f.forces,'encode',e=>texture(e,'forces-velocity',f.fields.velocityScratch));
  hook(f.split.transfer,'encodeToSimulation',e=>texture(e,'root-pressure-velocity',f.fields.velocity));
  hook(f.band,'encodeSolve',e=>texture(e,'band-pressure-velocity',f.fields.velocity));
  const compare=async(name:string,values:Float32Array)=>{
   const path=`${output}/${name}.bin`;const words=new Uint32Array(values.buffer,values.byteOffset,values.length);
   if(run===0){await writeFile(path,new Uint8Array(values.buffer,values.byteOffset,values.byteLength));console.log(JSON.stringify({run,name,hash:hash(words),length:words.length}));return;}
   const bytes=await readFile(path),old=new Uint32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4),floats=new Float32Array(bytes.buffer,bytes.byteOffset,old.length);
   let different=Math.abs(old.length-words.length),first=-1,max=0;for(let i=0;i<Math.min(old.length,words.length);i++)if(old[i]!==words[i]){different++;if(first<0)first=i;max=Math.max(max,Math.abs(floats[i]!-values[i]!));}
   console.log(JSON.stringify({run,name,hash:hash(words),length:words.length,different,first,max,before:first>=0?floats[first]:undefined,after:first>=0?values[first]:undefined}));
   if(different)await writeFile(`${output}/${name}-run1.bin`,new Uint8Array(values.buffer,values.byteOffset,values.byteLength));
  };
  const drain=async()=>{for(const item of pending.splice(0)){
   const a=await item.get();await compare(item.name,a);
   if(item.particles){
    const stride=nb.particleStride,n=a.length/stride,indices=Array.from({length:n},(_,i)=>i),u=new Uint32Array(a.buffer);
    indices.sort((i,j)=>{for(let k=0;k<stride;k++){const d=u[i*stride+k]!-u[j*stride+k]!;if(d)return d;}return 0;});
    const sorted=new Uint32Array(a.length);indices.forEach((i,j)=>sorted.set(u.subarray(i*stride,(i+1)*stride),j*stride));
    await compare(`${item.name}-canonical`,new Float32Array(sorted.buffer));
   }
  }};
  try{
   const e=device.createCommandEncoder();texture(e,'initial-phi',f.fields.phi);texture(e,'initial-velocity',f.fields.velocity);device.queue.submit([e.finish()]);await drain();
   for(step=1;step<=steps;step++){sequence=0;await advanceUniform(solver,step/60);await drain();}
  }finally{solver.destroy();}
 }
});
