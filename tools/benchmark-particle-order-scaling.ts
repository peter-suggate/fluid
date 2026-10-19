/** Production order stage, identical inputs, alternating legacy/merge arms.
 * Run with WEBGPU_NODE_MODULE=... node --import tsx this-file.ts.
 * GPU timestamps exclude host readback; pass-span totals include GPU gaps.
 * Default instrumentation is exactly two timestamps per complete sort in
 * each arm. --phases adds diagnostic per-pass instrumentation.
 * Other GPU workloads can still distort both metrics. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { withUniformDevice } from "../tests/helpers/uniform-geometric";
import { particleOrderFixture, particleOrderInput, type ParticleOrderPattern } from "../tests/helpers/uniform-particle-order";

const arg=(name:string)=>process.argv.find(s=>s.startsWith(`--${name}=`))?.slice(name.length+3);
const phases=process.argv.includes("--phases");
const capacity=Number(arg("capacity")??131072);
const rounds=Number(arg("rounds")??9),repetitions=Number(arg("repetitions")??3);
const grid=Number(arg("grid")??4096);
const out=resolve(arg("out")??"artifacts/particle-order-scaling/results.json");
const cases:{name:string;count:number;occupancy:number;pattern:ParticleOrderPattern;legacy:boolean}[]=[];
for(const occupancy of [4,16,32,63,64,65,96,127,128,129,255,256,257,1024,4096,8192])cases.push({name:"concentration",count:32768,occupancy,pattern:"positions",legacy:true});
for(const count of [4096,8192,16384,32768,65536,131072,262144,524288,1048576].filter(n=>n<=capacity))cases.push({name:"population",count,occupancy:128,pattern:"positions",legacy:true});
for(const pattern of ["positions","ties","identical","late"] as ParticleOrderPattern[])for(const count of [1024,4096,8192,16384,32768,65536,131072].filter(n=>n<=capacity))cases.push({name:"one-cell",count,occupancy:count,pattern,legacy:count<=8192});
for(const occupancy of [0,64,127,128,129,128,4096,128,0])cases.push({name:"activation",count:occupancy,occupancy:Math.max(1,occupancy),pattern:"ties",legacy:true});
const rows:unknown[]=[];
function workModel(data:Float32Array){
 const counts=new Map<string,number>();for(let i=0;i<data.length;i+=12){const key=[data[i],data[i+1],data[i+2]].map(Math.floor).join(",");counts.set(key,(counts.get(key)??0)+1);}
 let legacyRankComparisons=0,runRankComparisons=0,mergeKeyReads=0,mergeTiles=0,mergeStages=0,maxCellCount=0;
 for(const count of counts.values()){
  maxCellCount=Math.max(maxCellCount,count);legacyRankComparisons+=count*count;
  runRankComparisons+=Math.floor(count/128)*128*128+(count%128)**2;
  let stages=0;for(let width=128;width<count;width*=2){stages++;mergeKeyReads+=count;mergeTiles+=Math.ceil(count/256);}
  mergeStages=Math.max(mergeStages,stages);
 }
 return {occupiedCells:counts.size,maxCellCount,legacyRankComparisons,runRankComparisons,mergeKeyReads,mergeTiles,mergeStages};
}
const quantile=(a:number[],q:number)=>a.toSorted((a,b)=>a-b)[Math.min(a.length-1,Math.floor(q*a.length))];
await withUniformDevice("particle order scaling",async device=>{
 assert.ok(device.features.has("timestamp-query"),"GPU timestamps required");
 const f=await particleOrderFixture(device,capacity);f.order.workgroups=grid;
 const querySet=device.createQuerySet({type:"timestamp",count:2048});
 const resolveBuffer=device.createBuffer({size:16384,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
 const read=device.createBuffer({size:16384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
 async function sample(legacy:boolean,repeats:number){
  f.order.legacyCanonical=legacy;
  const e=device.createCommandEncoder();let next=0;const passes:{label:string;at:number;repeat:number}[]=[];let repeat=0;
  const timed=new Proxy(e,{get(target,key){if(key==="beginComputePass")return (descriptor:GPUComputePassDescriptor={})=>{
   const at=next;next+=2;passes.push({label:descriptor.label??"",at,repeat});
   // Equal instrumentation in both arms: two timestamps per complete sort.
   // Per-pass timestamps are an opt-in diagnostic, not the default timing.
   const beginning=phases||descriptor.label==="APIC bin prefix scan";
   const ending=phases||descriptor.label===(legacy?"Narrow-band FLIP canonical cell order":"Narrow-band FLIP crowded cell parallel merge");
   return target.beginComputePass(beginning||ending?{...descriptor,timestampWrites:{querySet,...(beginning?{beginningOfPassWriteIndex:at}:{}),...(ending?{endOfPassWriteIndex:at+1}:{})}}:descriptor);
  };const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;}});
  const start=performance.now();for(repeat=0;repeat<repeats;repeat++)f.encode(timed);
  e.resolveQuerySet(querySet,0,next,resolveBuffer,0);e.copyBufferToBuffer(resolveBuffer,0,read,0,8*next);device.queue.submit([e.finish()]);
  await read.mapAsync(GPUMapMode.READ);const timestamps=new BigUint64Array(read.getMappedRange()).slice();read.unmap();
  const result=[];for(let i=0;i<repeats;i++){
   const p=passes.filter(p=>p.repeat===i),first=p[0],last=p.at(-1)!;
   const begin=timestamps[first.at],end=timestamps[last.at+1];
   const validRange=(a:bigint,b:bigint)=>a>0n&&b>a;
   const valid=validRange(begin,end)&&(!phases||p.every(p=>validRange(timestamps[p.at],timestamps[p.at+1])));
   const duration=(at:number)=>Number(timestamps[at+1]-timestamps[at])/1e6;
   result.push({valid,invalidTimestampRange:valid?undefined:{begin:begin.toString(),end:end.toString()},spanMs:valid?Number(end-begin)/1e6:null,
    canonicalMs:phases&&valid?p.filter(p=>/canonical|merge/.test(p.label)).reduce((sum,p)=>sum+duration(p.at),0):undefined,
    passes:phases?p.map(p=>({label:p.label,ms:duration(p.at)})):undefined});
  }
  return {gpu:result,hostMs:(performance.now()-start)/repeats};
 }
 try{
  for(const c of cases.filter(c=>!arg("case")||c.name===arg("case"))){
   const data=particleOrderInput(c.count,c.occupancy,false,c.pattern);f.load(data);const expected=f.oracle(data);
   const arms=c.legacy?[true,false]:[false];const samples=new Map<boolean,Awaited<ReturnType<typeof sample>>[]>();
   // First dispatch is kept separately to expose activation/cold-pipeline spikes.
   for(const legacy of arms){samples.set(legacy,[]);const first=await sample(legacy,1);assert.deepEqual(await f.read(c.count),expected);rows.push({case:c,arm:legacy?"legacy":"merge",first});}
   for(let round=0;round<rounds;round++)for(const legacy of round%2?[...arms].reverse():arms)samples.get(legacy)!.push(await sample(legacy,repetitions));
   for(const legacy of arms){const raw=samples.get(legacy)!,gpu=raw.flatMap(r=>r.gpu),valid=gpu.filter(g=>g.valid),span=valid.map(g=>g.spanMs!),canonical=valid.map(g=>g.canonicalMs??0);
    assert.ok(span.length,"No valid GPU timestamp samples");
    const summary={case:c,work:workModel(data),arm:legacy?"legacy":"merge",validSamples:valid.length,invalidSamples:gpu.length-valid.length,medianMs:quantile(span,.5),p95Ms:quantile(span,.95),...(phases?{canonicalMedianMs:quantile(canonical,.5),canonicalP95Ms:quantile(canonical,.95)}:{})};
    console.log(JSON.stringify(summary));rows.push({...summary,raw});
   }
   mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify({capacity,grid,rounds,repetitions,phases,adapter:{vendor:device.adapterInfo.vendor,architecture:device.adapterInfo.architecture,description:device.adapterInfo.description},timestampQuantization:false,contention:"Other GPU jobs may be running; timings are not isolated.",allocatedSortBytes:f.order.allocatedBytes,rows},null,2)+"\n");
  }
 }finally{read.destroy();resolveBuffer.destroy();querySet.destroy();f.destroy();}
},["timestamp-query","subgroups"],["disable-dawn-features=timestamp_quantization"]);
