/** Trace reflection symmetry and repeatability through the production NB-FLIP stages.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-nbflip-symmetry.ts
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { sceneDocument } from '../lib/core/scene-definition';
import { getSceneDefinition } from '../lib/core/scenes';
import { uniformTileKey } from '../lib/methods/uniform/uniform-lattice-address';
import { uniformDetailField, uniformDetailExtent } from '../lib/methods/uniform/uniform-detail-fields';
import { createUniformSolver, withUniformDevice, advanceUniform } from '../tests/helpers/uniform-geometric';
import { paceDawnEventPolling } from '../lib/harness/node-dawn-provider';

const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const steps=Number(option('steps')??3),runs=Number(option('runs')??2);
const out=option('out')??'/tmp/nbflip-symmetry';mkdirSync(out,{recursive:true});
const scene=sceneDocument(getSceneDefinition(option('scene')??'sparse-cm12-ladder-symmetric-2d'));
scene.numerics.fixedDt_s=Number(option('dt')??1/60);
const tank=[scene.container.width_m,scene.container.height_m,scene.container.depth_m].map(v=>Math.round(v/scene.voxelDomain.finestCellSize_m));
const previous=new Map<string,Float32Array>(),rows:unknown[]=[];
let repeatFailure=false;
paceDawnEventPolling();
await withUniformDevice('NB-FLIP symmetry',async device=>{
 if(option('exact-surface')==='true'||option('wall-contact')==='off'||option('linear-surface')==='true'){
  const create=device.createShaderModule.bind(device);
  device.createShaderModule=(d:GPUShaderModuleDescriptor)=>{
   if(option('exact-surface')==='true'&&d.code.includes('fn umCheapAdvected(')){
    const code=d.code.replace(/let early=umEarlyFarAdvected\(vertex,width\);/g,'let early=vec2f(0.0);').replace(/let cheap=umCheapAdvected\(vertex,width\);/g,'let cheap=vec2f(0.0);').replace(/let euler=umEulerAdvected\(vertex,width,foot\);/g,'let euler=vec2f(0.0);');
    console.log(JSON.stringify({exactSurface:code!==d.code}));return create({...d,code});
   }
   if(option('wall-contact')==='off'&&d.code.includes('fn umWallContact(')){return create({...d,code:d.code.replace('if(params.hDt.w<=0.0){return value;}var interior=p;', 'if(params.hDt.w>=0.0){return value;}var interior=p;')});}
   if(option('linear-surface')==='true'&&d.code.includes('fn umCubicPhi(')){return create({...d,code:d.code.replace('fn umCubicPhi(q:vec3f,cell:UMVertexCell)->f32{','fn umCubicPhi(q:vec3f,cell:UMVertexCell)->f32{if(q.x>=0.0){return cell.value;}')});}
   return create(d);
  };
 }
 for(let run=0;run<runs;run++){
  const solver=await createUniformSolver(device,structuredClone(scene),{...JSON.parse(option('values')??'{}'),...(option('budget')===undefined?{}:{adaptiveBudgetPercent:Number(option('budget'))})});
  const f=(solver as any).mixedFrame,nb=f.narrowBandFlip;
  if(option('census')==='true'){
   f.setLayoutViews(true);const dynamic=(solver as any).mixedDynamic,original=dynamic.encode.bind(dynamic);
   dynamic.encode=(e:GPUCommandEncoder,...args:any[])=>{const result=original(e,...args);
    const sr=f.solid.coarse.record,sb=device.createBuffer({size:sr.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});e.copyBufferToBuffer(sr,0,sb,0,sb.size);
    pendingBuffers.push(async()=>{await sb.mapAsync(GPUMapMode.READ);const a=new Float32Array(sb.getMappedRange()).slice();sb.unmap();sb.destroy();writeFileSync(`${out}/${run}-${step}-solid.bin`,new Uint8Array(a.buffer));console.log(JSON.stringify({solidCount:f.solid.coarse.count,tiles:f.ownership.capacity.tiles}));});
    const b=device.createBuffer({size:dynamic.work.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});e.copyBufferToBuffer(dynamic.work,0,b,0,b.size);
    pendingBuffers.push(async()=>{await b.mapAsync(GPUMapMode.READ);const u=new Uint32Array(b.getMappedRange()).slice();b.unmap();b.destroy();writeFileSync(`${out}/${run}-${step}-census.bin`,new Uint8Array(u.buffer));console.log(JSON.stringify({census:Array.from({length:4},(_,z)=>Array.from({length:4},(_,y)=>Array.from({length:4},(_,x)=>{const key=uniformTileKey(f.ownership.capacity.address,x,y,z);return [u[dynamic.importanceOffset+2*key],u[dynamic.importanceOffset+2*key+1]];})))}));});return result;};
  }
  const pendingBuffers:(()=>Promise<void>)[]=[];
  let step=0,sequence=0;
  const pending:{name:string;extent:readonly number[];get:()=>Promise<Float32Array>;topology:()=>Promise<Uint32Array>}[]=[];
  const capture=(e:GPUCommandEncoder,name:string,t:GPUTexture)=>{
   const read=device.createBuffer({size:f.ownership.presentation.buffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   e.copyBufferToBuffer(f.ownership.presentation.buffer,0,read,0,read.size);
   const topology=async()=>{await read.mapAsync(GPUMapMode.READ);const words=new Uint32Array(read.getMappedRange()).slice();read.unmap();read.destroy();return words;};
   pending.push({topology,name:`${step}-${name==='extended-velocity'?'extra':sequence++}-${name}`,extent:uniformDetailExtent(t),get:uniformDetailField(t)!.storage.capture(e,t)});
  };
  const hook=(o:any,key:string,name:string,texture:()=>GPUTexture,before=false)=>{
   const original=o[key].bind(o);o[key]=(...args:any[])=>{if(before)capture(args[0],name,texture());const result=original(...args);if(!before)capture(args[0],name,texture());return result;};
  };
  if(option('particles')==='true'){
   const move=nb.move.bind(nb);
   nb.move=(encoder:GPUCommandEncoder,...args:any[])=>{
    const result=move(encoder,...args),bank=nb.activeParticles;
    const read=device.createBuffer({size:bank.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const header=device.createBuffer({size:128,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    encoder.copyBufferToBuffer(bank,0,read,0,read.size);encoder.copyBufferToBuffer(nb.state,0,header,0,128);
    pendingBuffers.push(async()=>{
     await Promise.all([read.mapAsync(GPUMapMode.READ),header.mapAsync(GPUMapMode.READ)]);
     const mapped=read.getMappedRange(),words=new Uint32Array(mapped),floats=new Float32Array(mapped),state=new Uint32Array(header.getMappedRange());
     const stride=nb.apic?28:12,count=Math.min(words.length/stride,state[0]!+state[21]!),records:number[][]=[];
     for(let i=0;i<count;i++)if(floats[stride*i+3]!>0)records.push(Array.from(words.slice(stride*i,stride*(i+1))));
     records.sort((a,b)=>{for(let k=0;k<stride;k++)if(a[k]!==b[k])return a[k]!-b[k]!;return 0;});
     const canonical=new Uint32Array(records.flat()),name=`particles-${step}`,old=previous.get(name);
     const bytes=new Uint8Array(canonical.buffer);let differences=0;
     if(old){const before=new Uint32Array(old.buffer);for(let i=0;i<Math.max(before.length,canonical.length);i++)differences+=Number(before[i]!==canonical[i]);}else previous.set(name,new Float32Array(canonical.buffer));
     repeatFailure ||= differences>0;
     console.log(JSON.stringify({run,step,particles:records.length,particleDifferences:differences,particleHash:createHash('sha256').update(bytes).digest('hex')}));
     writeFileSync(`${out}/${run}-${name}.bin`,bytes);read.unmap();header.unmap();read.destroy();header.destroy();
    });return result;
   };
  }
  hook(nb,'bootstrap','pre-seed-phi',()=>f.fields.phi,true);
  if(option('capture-extended')==='true'||(option('capture-extended')==='alternate'&&run%2))hook(f.surface,'encode','extended-velocity',()=>f.fields.velocityScratch,true);
  if(option('capture-unit')==='true')hook(f.surface,'encode','unit-velocity',()=>f.hanging.unitVelocity,true);
  hook(f.surface,'encode','advected-phi',()=>f.fields.phiScratch);
  hook(nb,'reconstruct','reconstructed-phi',()=>f.fields.phi);
  if(option('capture-retired')==='true')hook(nb,'retire','retired-phi',()=>f.fields.phi);
  hook(nb,'redistance','redistanced-phi',()=>f.fields.phi);
  hook(nb,'transfer','p2g-velocity',()=>f.fields.velocity);
  hook(f.forces,'encode','forces-velocity',()=>f.fields.velocityScratch);
  hook(f.split.transfer,'encodeToSimulation','root-pressure-velocity',()=>f.fields.velocity);
  hook(f.band,'encodeProject','band-pressure-velocity',()=>f.fields.velocity);
  const drain=async()=>{for(const get of pendingBuffers.splice(0))await get();for(const item of pending.splice(0)){
   const a=await item.get(),[nx,ny,nz]=item.extent as [number,number,number],components=a.length/(nx*ny*nz);
   const topology=await item.topology();
   writeFileSync(`${out}/${run}-${item.name}-topology.bin`,new Uint8Array(topology.buffer));
   if(step===1&&sequence>0&&item.name.includes("forces")){console.log(JSON.stringify({widths:Array.from({length:4},(_,z)=>Array.from({length:4},(_,y)=>Array.from({length:8},(_,x)=>(topology[uniformTileKey(f.ownership.capacity.address,x,y,z)]!>>>31)).join("")).join("/"))}));}
   const width=(x:number,y:number,z:number)=>(topology[uniformTileKey(f.ownership.capacity.address,x>>2,y>>2,z>>2)]!&0x80000000)?1:4;
   let max=0,nearSurfaceMax=0,different=0,first:unknown,finite=true,layoutDifferent=0;
   const scalar=(x:number,y:number,z:number)=>{
    // Evaluate the represented field, including hanging wall vertices.
    // Unowned fine backing texels are not independent surface samples.
    if(((x|y|z)&3)===0)return a[x+nx*(y+ny*z)]!;
    for(let k=0;k<8;k++){
     const cell=[x-(k&1),y-((k>>1)&1),z-(k>>2)];
     if(cell.some((v,axis)=>v<0||v>=item.extent[axis]!-1)||width(cell[0]!,cell[1]!,cell[2]!)!==4)continue;
     const origin=cell.map(v=>v-v%4),fraction=[x,y,z].map((v,axis)=>(v-origin[axis]!)/4);let value=0;
     for(let j=0;j<8;j++){
      const corner=[j&1,(j>>1)&1,j>>2],weight=corner.reduce((w,b,axis)=>w*(b?fraction[axis]!:1-fraction[axis]!),1);
      if(weight)value+=weight*a[origin[0]!+4*corner[0]!+nx*(origin[1]!+4*corner[1]!+ny*(origin[2]!+4*corner[2]!))]!;
     }
     return value;
    }
    return a[x+nx*(y+ny*z)]!;
   };
   // Scalars are nodal; velocity stores the positive MAC face of each cell.
   for(let z=0;z<tank[2]!+(components===1?1:0);z++)for(let y=0;y<tank[1]!+(components===1?1:0);y++)for(let x=0;x<tank[0]!+(components===1?1:0);x++)for(let c=0;c<Math.min(components,3);c++){
    let w=1;
    if(components===1){}
    else{w=width(x,y,z);if(w===4&&[x,y,z].some((v,axis)=>v%4!==(axis===c?3:0)))continue;}
    const mz=components===1?tank[2]!-z:c===2?tank[2]!-2-z:tank[2]!-w-z;if(mz<0)continue;
    if(components!==1&&width(x,y,mz)!==w){layoutDifferent++;continue;}
    const i=components*(x+nx*(y+ny*z))+c,j=components*(x+nx*(y+ny*mz))+c;
    const left=components===1?scalar(x,y,z):a[i]!,right=components===1?scalar(x,y,mz):(c===2?-a[j]!:a[j]!);
    const error=Math.abs(left-right);finite&&=Number.isFinite(error);
    if(components===1&&Math.min(Math.abs(left),Math.abs(right))<=2*scene.voxelDomain.finestCellSize_m)nearSurfaceMax=Math.max(nearSurfaceMax,error);
    if(error>1e-6)different++;if(error>max){max=error;first={x,y,z,c,a:left,b:right};}
   }
   const old=previous.get(item.name);let repeatDifferent=0,repeatMax=0;
   if(old){const u=new Uint32Array(a.buffer),v=new Uint32Array(old.buffer);for(let i=0;i<a.length;i++)if(u[i]!==v[i]){repeatDifferent++;repeatMax=Math.max(repeatMax,Math.abs(a[i]!-old[i]!));}}
   else previous.set(item.name,a);
   repeatFailure ||= repeatDifferent>0;
   const row={run,name:item.name,extent:item.extent,components,layoutDifferent,finite,max,nearSurfaceMax,different,first,repeatDifferent,repeatMax,hash:createHash('sha256').update(new Uint8Array(a.buffer)).digest('hex')};
   rows.push(row);console.log(JSON.stringify(row));writeFileSync(`${out}/${run}-${item.name}.bin`,new Uint8Array(a.buffer));
  }};
  console.log(JSON.stringify({run,grid:[solver.info.nx,solver.info.ny,solver.info.nz],dt:scene.numerics.fixedDt_s}));
  try{
   const e=device.createCommandEncoder();capture(e,'initial-phi',f.fields.phi);capture(e,'initial-velocity',f.fields.velocity);device.queue.submit([e.finish()]);await drain();
   for(step=1;step<=steps;step++){sequence=0;await advanceUniform(solver,step*scene.numerics.fixedDt_s);await drain();}
  }finally{solver.destroy();}
 }
},['subgroups']);
writeFileSync(`${out}/report.json`,JSON.stringify(rows,null,2)+'\n');

if(option('assert-repeat')==='true'&&repeatFailure)throw new Error('Repeated runs differ; see the stage and particle reports.');
