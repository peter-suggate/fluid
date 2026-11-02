/** Production-kernel experiments, one explicit comparison for each of the ten
 * optimization sections. Includes preparation/queues/conversion in GPU timings.
 * Spatial directory access is a controlled 8^3-tile fixture, not a whole solver.
 * bash tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/benchmark-nbflip-ten-sections.ts
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdirSync,writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {narrowBandSplatWGSL} from '../lib/methods/uniform/uniform-narrow-band-splat.wgsl';
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
import {sectionVariants,studyArgsWGSL,studyExtraWGSL,type StudyStep,type StudyVariant} from './nbflip-ten-section-variants';
const option=(k:string)=>process.argv.find(a=>a.startsWith(`--${k}=`))?.slice(k.length+3);
const out=resolve(option('out')??'artifacts/nbflip-ten-sections-2026-11-02');mkdirSync(out,{recursive:true});
const repeats=Number(option('repeats')??9),inner=Number(option('inner')??32);
const sections=(option('sections')??'1,2,3,4,5,6,7,8,9,10').split(',').map(Number);
assert.ok(repeats>=3&&Number.isInteger(repeats)&&inner>=1&&Number.isInteger(inner));
assert.ok(sections.every(n=>Number.isInteger(n)&&n>=1&&n<=10));
const variants=sectionVariants.filter(v=>v.section===0||sections.includes(v.section));
const tiles=512,vertices=33**3,coverage=64*tiles,depth=coverage+tiles,hotDepth=depth+vertices,band=hotDepth+vertices,occupied=band+4,words=occupied+4+tiles;
const code=`
const UM_CELL_KEYS=${64*tiles}u;const UM_TILES=${tiles}u;const NB_LINKS=UM_CELL_KEYS;
const NB_COVERAGE=${coverage}u;const NB_DEPTH_A=${depth}u;const NB_SURFACE_HOT_NEAREST=${hotDepth}u;
const NB_BAND=${band}u;const NB_OCCUPIED=${occupied}u;const nbSparseBins=true;
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> links:array<u32>;
@group(0) @binding(2) var<storage,read> particles:array<vec4f>;
fn nbBinTile(t:u32)->u32{return 64u*t;}fn nbBinCell(c:u32)->u32{return c;}
fn umTileLive(t:u32)->bool{return t<UM_TILES;}
fn umLiveTileCount()->u32{return UM_TILES;}fn umLiveTile(j:u32)->u32{return j;}
fn nbTileCold(t:u32)->bool{return (t&292u)==0u;}
fn umCorner(j:u32,n:u32)->vec3u{return vec3u(j%n,(j/n)%n,j/(n*n));}
fn umTileCoord(t:u32)->vec3u{return umCorner(t,8u);}
fn umVertexPresent(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(33));}
fn umVertexAt(p:vec3u)->u32{return p.x+33u*(p.y+33u*p.z);}
fn umVertexSlot(p:vec3i)->vec4u{return vec4u(vec3u(p),u32(umVertexPresent(p)));}
fn umBlockStored(b:u32)->bool{return b!=0u;}
fn umVertexSlotAt(p:vec4u)->u32{return umVertexAt(p.xyz);}
fn umLoadAddressPlan(b:u32,lane:u32){}
fn nbPosition(i:u32)->vec3f{let a=NB_LINKS+4u*i;return bitcast<vec3f>(vec3u(links[a],links[a+1u],links[a+2u]));}
fn nbSurfaceParticleHot(i:u32)->bool{return bitcast<f32>(links[NB_LINKS+4u*i+3u])>1.0;}
`+narrowBandSplatWGSL+studyExtraWGSL;
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal','disable-dawn-features=timestamp_quantization']);const adapter=await gpu.requestAdapter();assert.ok(adapter);
const device=await adapter.requestDevice({requiredFeatures:['timestamp-query']});
const errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
const layout=device.createBindGroupLayout({entries:[0,1,2,3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===4?'uniform' as const:binding===2?'read-only-storage' as const:'storage' as const}}))});
const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
const argsLayout=device.createBindGroupLayout({entries:[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===2?'storage' as const:'read-only-storage' as const}}))});
const argsPipelineLayout=device.createPipelineLayout({bindGroupLayouts:[argsLayout]});
const report={completed:false,requestedCases:option('cases')??'empty,sparse,mixed-cost,dense,clustered',date:new Date().toISOString(),adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},scope:'controlled production surface kernels; includes setup; not whole-frame or DRAM-counter measurements',repeats,inner,sections,sourceSha256:createHash('sha256').update(code).digest('hex'),variants:variants.map(({name,section,claim,source,steps})=>({name,section,claim,steps,sha256:createHash('sha256').update(source(code)).digest('hex')})),results:[] as unknown[],coordination:[] as unknown[],errors};
const save=()=>writeFileSync(`${out}/results.json`,JSON.stringify(report,null,2)+'\n');
async function compile(source:string,entryPoint:string,layout=pipelineLayout,constants?:Record<string,number>){
 const module=device.createShaderModule({code:source});const failures=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>({message:m.message,line:m.lineNum,column:m.linePos}));
 if(failures.length){writeFileSync(`${out}/failed-${entryPoint}.wgsl`,source);assert.deepEqual(failures,[]);}
 return device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants}});
}
const buffer=(size:number,usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST)=>device.createBuffer({size,usage});
const median=(xs:number[])=>[...xs].sort((a,b)=>a-b)[Math.floor(xs.length/2)]!;
const cases=[{name:'empty',active:0,population:()=>0,hot:()=>false},{name:'sparse',active:32,population:()=>8,hot:(t:number,i:number)=>t%13===0&&i%3===0},{name:'mixed-cost',active:512,population:(t:number)=>[8,64,256,1025][t%4]!,hot:(t:number,i:number)=>t%13===0&&i%5===0},{name:'dense',active:512,population:()=>1025,hot:(_t:number,i:number)=>i%3===0},{name:'clustered',active:64,population:()=>4097,hot:(t:number,i:number)=>t%13===0&&i%11===0}].filter(c=>!option('cases')||option('cases')!.split(',').includes(c.name));
assert.ok(cases.length>0,'at least one known fixture must be selected');
if(option('cases'))assert.ok(option('cases')!.split(',').every(name=>cases.some(c=>c.name===name)),'unknown fixture');
save();
try{
 const compiled=new Map<string,Map<string,GPUComputePipeline>>();
 const argsP=await Promise.all([0,1,2].map(mode=>compile(studyArgsWGSL(band),'studyArgs',argsPipelineLayout,{mode})));
 for(const v of variants){const source=v.source(code),pipelines=new Map<string,GPUComputePipeline>();writeFileSync(`${out}/${v.name}.wgsl`,source);
  for(const step of v.steps)if(step.args===undefined&&!pipelines.has(step.entry))pipelines.set(step.entry,await compile(source,step.entry));
  compiled.set(v.name,pipelines);console.log(JSON.stringify({compiled:v.name,section:v.section}));
 }
 const oracleSteps:StudyStep[]=[{entry:'studyReset',groups:1024},{entry:'surfaceCells',groups:1024},{entry:'studyArgs',args:1,groups:1},{entry:'surfaceSplatReference',indirect:0}];
 const oracleP=new Map<string,GPUComputePipeline>();for(const s of oracleSteps)if(s.args===undefined)oracleP.set(s.entry,await compile(code,s.entry));
 const baseline=compiled.get('baseline')!;
 for(const fixture of cases){
  const populations=new Uint32Array(tiles);for(let k=0;k<fixture.active;k++){const t=k*37%tiles;populations[t]=fixture.population(t);}
  const total=populations.reduce((a,b)=>a+b,0),capacity=Math.max(64,Math.ceil(Math.max(total,1)*1.5/64)*64);
  const initial=new Uint32Array(words),stream=new Uint32Array(64*tiles+7*capacity),positions=new Float32Array(4*total);let at=0,jobCount=0,occupiedCount=0,hotCount=0;const expectedJobs:[number,number][]=[];
  for(let t=0;t<tiles;t++){const n=populations[t]!;for(let k=0;k<n;k+=512)expectedJobs.push([t,at+k]);initial[coverage+t]=n;jobCount+=Math.ceil(n/512);if(n)occupiedCount++;
   for(let i=0;i<n;i++){const cell=Math.floor(i*64/n),p=at++;initial[64*t+cell]!++;
    const hot=fixture.hot(t,i);hotCount+=Number(hot);
    positions.set([4*(t%8)+(cell%4)+(i*17%127)/128,4*(Math.floor(t/8)%8)+(Math.floor(cell/4)%4)+(i*31%127)/128,4*Math.floor(t/64)+Math.floor(cell/16)+(i*53%127)/128,hot?2:1],4*p);
   }
  }
  initial[occupied]=occupiedCount;let list=0;for(let k=0;k<tiles;k++){const t=k*197%tiles;if(populations[t])initial[occupied+4+list++]=t;}
  let start=0;for(let c=0;c<64*tiles;c++){stream[c]=start;start+=initial[c]!;}assert.equal(start,total);
  stream.set(new Uint32Array(positions.buffer),64*tiles);
  const bins=buffer(initial.byteLength),links=buffer(stream.byteLength),particles=buffer(16*capacity),study=buffer(16+32*capacity),meta=buffer(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),dispatchArgs=buffer(24,GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(links,0,stream);device.queue.writeBuffer(meta,0,new Uint32Array([total,0,0,0]));
  const group=device.createBindGroup({layout,entries:[bins,links,particles,study,meta].map((buffer,binding)=>({binding,resource:{buffer}}))});
  const argGroup=device.createBindGroup({layout:argsLayout,entries:[bins,study,dispatchArgs].map((buffer,binding)=>({binding,resource:{buffer}}))});
  const output=buffer(8*vertices,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
  const emit=(pass:GPUComputePassEncoder,steps:StudyStep[],pipelines:Map<string,GPUComputePipeline>)=>{for(const s of steps){if(s.args!==undefined){pass.setPipeline(argsP[s.args]!);pass.setBindGroup(0,argGroup);}else{pass.setPipeline(pipelines.get(s.entry)!);pass.setBindGroup(0,group);}if(s.indirect!==undefined)pass.dispatchWorkgroupsIndirect(dispatchArgs,s.indirect);else pass.dispatchWorkgroups(s.groups!);}};
  const readOutput=async(e:GPUCommandEncoder)=>{e.copyBufferToBuffer(bins,4*depth,output,0,output.size);device.queue.submit([e.finish()]);await output.mapAsync(GPUMapMode.READ);const a=new Uint32Array(output.getMappedRange()).slice();output.unmap();return a;};
  let oracle:Uint32Array|undefined;const worklists:Record<string,number[]>={};
  const queries=device.createQuerySet({type:'timestamp',count:2*repeats*variants.length}),resolved=buffer(16*repeats*variants.length,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC),times=buffer(resolved.size,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
  try{
   for(let v=-1;v<variants.length;v++){
    device.queue.writeBuffer(bins,0,initial);const e=device.createCommandEncoder();const pass=e.beginComputePass();emit(pass,v<0?oracleSteps:variants[v]!.steps,v<0?oracleP:compiled.get(variants[v]!.name)!);pass.end();
    const actual=await readOutput(e);if(v<0){oracle=actual;assert.equal(actual.some(n=>n!==0),total>0);}else assert.deepEqual(actual,oracle,`${fixture.name}: ${variants[v]!.name}`);
    if(v>=0&&!['fused-tile','reuse-scalar','reuse-shared'].includes(variants[v]!.name)){
     const name=variants[v]!.name,queues=name==='cost-buckets'?3:name==='split-hot'?2:1;
     const listed=buffer(16+8*Math.max(1,jobCount)*queues,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
     try{
      const check=device.createCommandEncoder();
      if(queues>1){check.copyBufferToBuffer(study,0,listed,0,16);for(let q=0;q<queues;q++)if(jobCount)check.copyBufferToBuffer(study,16+8*q*capacity,listed,16+8*q*jobCount,8*jobCount);}
      else{check.copyBufferToBuffer(bins,4*(band+3),listed,0,4);if(jobCount)check.copyBufferToBuffer(links,4*(64*tiles+4*capacity),listed,16,8*jobCount);}
      device.queue.submit([check.finish()]);await listed.mapAsync(GPUMapMode.READ);const w=new Uint32Array(listed.getMappedRange()),actualJobs:[number,number][]=[];
      for(let q=0;q<queues;q++){assert.ok(w[q]!<=jobCount,`${name}: queue bounded`);for(let j=0;j<w[q]!;j++){const at=4+2*(q*jobCount+j);actualJobs.push([w[at]!,w[at+1]!]);}}
      worklists[name]=Array.from(w.slice(0,queues));listed.unmap();actualJobs.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);assert.deepEqual(actualJobs,expectedJobs,`${fixture.name}: ${name} must preserve every job exactly once`);
     }finally{listed.destroy();}
    }
   }
   // Warm every complete chain. Result planes, planners, and local windows
   // reset inside every iteration; global reductions are not pre-saturated.
   {const e=device.createCommandEncoder();const pass=e.beginComputePass();for(const v of variants)emit(pass,v.steps,compiled.get(v.name)!);pass.end();device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();}
   const e=device.createCommandEncoder();let slot=0;
   for(let r=0;r<repeats;r++)for(let k=0;k<variants.length;k++){
    const v=variants[(k+r)%variants.length]!;const pass=e.beginComputePass({timestampWrites:{querySet:queries,beginningOfPassWriteIndex:2*slot,endOfPassWriteIndex:2*slot+1}});
    for(let i=0;i<inner;i++)emit(pass,v.steps,compiled.get(v.name)!);pass.end();slot++;
   }
   e.resolveQuerySet(queries,0,2*slot,resolved,0);e.copyBufferToBuffer(resolved,0,times,0,times.size);device.queue.submit([e.finish()]);await times.mapAsync(GPUMapMode.READ);const stamps=new BigUint64Array(times.getMappedRange());
   const rows=variants.map((v,j)=>{const ms:number[]=[];for(let s=0;s<slot;s++)if((s%variants.length+Math.floor(s/variants.length))%variants.length===j)ms.push(Number(stamps[2*s+1]!-stamps[2*s]!)/1e6/inner);return {name:v.name,section:v.section,median_ms:median(ms),ms};});times.unmap();
   const result={fixture:fixture.name,particles:total,capacity,occupiedTiles:occupiedCount,jobs:jobCount,hotParticles:hotCount,logicalStreamBytes:{aos:16*total,soa:16*total,tiled:16*Math.ceil(total/64)*64,packed:12*total+4*Math.ceil(total/32)},parity:'all distance/hot-distance words equal scalar oracle; explicit job lists preserve every job exactly once',worklists,rows};report.results.push(result);save();console.log(JSON.stringify(result));
   if(sections.includes(10)&&fixture.name!=='empty'){
    const receipts=Array.from({length:4},()=>buffer(4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST));
    const coordinationSteps:StudyStep[]=[{entry:'studyReset',groups:1024},{entry:'surfaceSplatPlan',groups:1024},{entry:'studyArgs',args:0,groups:1},{entry:'surfaceSplatOccupied',indirect:0}];
    const modes=['cpu-count','gpu-count','async-receipts'] as const,rounds=7,frames=24;
    const values=modes.map(()=>[] as number[]);
    try{
     for(let round=-1;round<rounds;round++)for(let k=0;k<modes.length;k++){
      const which=(k+Math.max(0,round))%modes.length,mode=modes[which]!;const pending:Promise<void>[]=[];const started=performance.now();
      const map=async(b:GPUBuffer)=>{await b.mapAsync(GPUMapMode.READ);assert.equal(new Uint32Array(b.getMappedRange())[0],jobCount);b.unmap();};
      for(let frame=0;frame<frames;frame++){
       const e=device.createCommandEncoder();let pass=e.beginComputePass();
       if(mode==='cpu-count'){
        emit(pass,sectionVariants[0]!.steps.slice(0,2),baseline);pass.end();e.copyBufferToBuffer(bins,4*(band+3),receipts[0]!,0,4);device.queue.submit([e.finish()]);
        await receipts[0]!.mapAsync(GPUMapMode.READ);const count=new Uint32Array(receipts[0]!.getMappedRange())[0]!;assert.equal(count,jobCount);receipts[0]!.unmap();
        const next=device.createCommandEncoder();pass=next.beginComputePass();emit(pass,[{entry:'surfaceSplatOccupied',groups:Math.min(count,1024)}],baseline);pass.end();device.queue.submit([next.finish()]);await device.queue.onSubmittedWorkDone();
       }else{
        emit(pass,coordinationSteps,baseline);pass.end();if(mode==='async-receipts')e.copyBufferToBuffer(bins,4*(band+3),receipts[frame%4]!,0,4);device.queue.submit([e.finish()]);
        if(mode==='gpu-count')await device.queue.onSubmittedWorkDone();else{pending.push(map(receipts[frame%4]!));if(pending.length===4){await Promise.all(pending);pending.length=0;}}
       }
      }
      await Promise.all(pending);await device.queue.onSubmittedWorkDone();if(round>=0)values[which]!.push((performance.now()-started)/frames);
      assert.deepEqual(await readOutput(device.createCommandEncoder()),oracle,`coordination ${mode}`);
     }
     const coordination={fixture:fixture.name,section:10,frames,rounds,readbackBytes:4,ring:4,scope:'Node/Dawn CPU wall ms per identical chain; allocation and final field comparison excluded',rows:modes.map((mode,k)=>({mode,median_ms:median(values[k]!),ms:values[k]})),parity:'field outputs and every read counter match'};report.coordination.push(coordination);save();console.log(JSON.stringify(coordination));
    }finally{receipts.forEach(b=>b.destroy());}
   }
   assert.deepEqual(errors,[]);
  }finally{for(const b of [bins,links,particles,study,meta,dispatchArgs,output,resolved,times])b.destroy();queries.destroy();}
 }
 report.completed=true;save();
}catch(error){errors.push(error instanceof Error?error.stack??error.message:String(error));save();throw error;}
finally{await device.queue.onSubmittedWorkDone();device.destroy();}
