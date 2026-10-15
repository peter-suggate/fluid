/** Inspect the exact cached crossing geometry immediately before metric construction. */
import {createHighResolutionDamBreakScene} from '../lib/core/scenes';
import {createUniformSolver,withUniformDevice,advanceUniform} from '../tests/helpers/uniform-geometric';
import {readMixedBuffer} from '../tests/helpers/uniform-mixed-native-fields';
import {paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
const stop=Number(process.argv.find(a=>a.startsWith('--step='))?.slice(7)??153);
paceDawnEventPolling();
await withUniformDevice('redistance geometry failure',async device=>{
 const entries=new WeakMap<GPUBindGroup,GPUBindGroupDescriptor>();
 const sources=new WeakMap<GPUShaderModule,string>(),descriptors=new WeakMap<GPUComputePipeline,GPUComputePipelineDescriptor>();
 const source=new Proxy(device,{get(target,key){
  if(key==='createBindGroup')return(d:GPUBindGroupDescriptor)=>{const g=target.createBindGroup(d);entries.set(g,d);return g;};
  if(key==='createShaderModule')return(d:GPUShaderModuleDescriptor)=>{
   if(process.argv.includes('--original')&&d.code.includes('fn nbSurfaceDistance('))d={...d,code:d.code.replace('nbCrossingVertex(tile,origin,p+umCorner(k,2u))/h','umVertexValue(p+umCorner(k,2u))/h').replace('nbCrossingVertex(tile,tileOrigin,base+umCorner(k,2u))/h','umVertexValue(base+umCorner(k,2u))/h').replace('if(low>0.0||high<0.0||low==high){return -1.0;}','').replace('var distance=length(p-origin)+sqrt(3.0);','var distance=1e20;')};
   const m=target.createShaderModule(d);sources.set(m,d.code);return m;
  };
  if(key==='createComputePipelineAsync')return async(d:GPUComputePipelineDescriptor)=>{const p=await target.createComputePipelineAsync(d);descriptors.set(p,d);return p;};
  const v=Reflect.get(target,key,target);return typeof v==='function'?v.bind(target):v;
 }});
 const scene=createHighResolutionDamBreakScene();scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
 const solver=await createUniformSolver(source,scene),f=(solver as any).mixedFrame,nb=f.narrowBandFlip;
 const desc=descriptors.get(nb.batches.redistanceFine.steps[1].pipeline)!;
 const buffer=device.createBuffer({size:4*(1+100*32),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
 const original=sources.get(desc.compute.module)!;
 const diagnostic=`

@compute @workgroup_size(64) fn inspect(@builtin(global_invocation_id) gid:vec3u){
 for(var cell=gid.x;cell<UM_CELL_KEYS;cell+=65536u){
  if(!umCellLive(cell)){continue;}let base=umCellCoord(cell);if(!nbBandCrossing(base)){continue;}
  let distance=nbSurfaceDistance(vec3f(base)+0.5,cell);if(distance>=0.0&&distance<1e19){continue;}
  let index=atomicAdd(&state[0],1u);if(index>=100u){continue;}let at=1u+32u*index;
  let tile=nbBandTile(base);let origin=4u*umTileCoord(tile);let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
  let mask=vec2u(atomicLoad(&bins[NB_BAND_MASK+2u*tile]),atomicLoad(&bins[NB_BAND_MASK+2u*tile+1u]));let local=base&vec3u(3u);
  let slot=atomicLoad(&bins[NB_BAND_CANDIDATES+tile])+nbGeometryRank(mask,local.x+4u*(local.y+4u*local.z));
  atomicStore(&state[at],base.x);atomicStore(&state[at+1u],base.y);atomicStore(&state[at+2u],base.z);atomicStore(&state[at+3u],slot);
  for(var k=0u;k<8u;k++){
   atomicStore(&state[at+4u+k],atomicLoad(&bins[UM_CELL_KEYS+1u+8u*slot+k]));
   atomicStore(&state[at+12u+k],bitcast<u32>(umVertexValue(base+umCorner(k,2u))/h));
   atomicStore(&state[at+20u+k],bitcast<u32>(nbCrossingVertex(tile,origin,base+umCorner(k,2u))/h));
  }
 }
}`;
 const module=device.createShaderModule({code:original+diagnostic.replace(/\b(?:UM_CELL_KEYS|NB_BAND_MASK|NB_BAND_CANDIDATES)\b/g,name=>original.includes(`fn ${name}(`)?`${name}()`:name)});
 const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
 const pipeline=await device.createComputePipelineAsync({...desc,compute:{...desc.compute,module,entryPoint:'inspect'}});
 let step=0;const run=nb.run.bind(nb);
 nb.run=(e:GPUCommandEncoder,b:any,shared?:GPUComputePassEncoder)=>{
  if(step!==stop||b!==nb.batches.redistanceFine)return run(e,b,shared);
  if(shared)throw new Error('Unexpected shared pass');run(e,{...b,steps:b.steps.slice(0,1)});
  const originalGroup=entries.get(nb.groups.surface[nb.parity].group)!;
  const group=device.createBindGroup({...originalGroup,entries:[...originalGroup.entries].map(b=>b.binding===4?{...b,resource:{buffer}}:b)});
  const pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,nb.ownership.bindGroup);pass.setBindGroup(1,group);pass.setBindGroup(2,nb.solid.tileGroup);pass.dispatchWorkgroups(1024);pass.end();
  run(e,{...b,steps:b.steps.slice(1)});
 };
 try{
  for(step=1;step<=stop;step++)await advanceUniform(solver,step/60);
  const data=await readMixedBuffer(device,buffer),words=new Uint32Array(data.buffer);console.log(JSON.stringify({step:stop,failures:words[0],records:Array.from({length:Math.min(100,words[0]!)},(_,i)=>{const a=1+32*i;return{cell:Array.from(words.subarray(a,a+3)),slot:words[a+3],cached:Array.from(data.subarray(a+4,a+12)),vertex:Array.from(data.subarray(a+12,a+20)),crossing:Array.from(data.subarray(a+20,a+28))};})}));
 }finally{solver.destroy();buffer.destroy();}
});
