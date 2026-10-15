/** Export labelled empty-phi counter evidence from its Instruments trace. */
import assert from 'node:assert/strict';import {spawn} from 'node:child_process';
import {writeFileSync,createWriteStream} from 'node:fs';import {resolve} from 'node:path';
import {parseTraceTable,durationMicroseconds,timestampMicroseconds,type TraceRow} from './xctrace-trace-tables';
const dir=process.argv[2]??'/tmp/fluid-phi-counter-20261015',trace=resolve(dir,'empty-phi.trace');
async function* rows(schema:string){
 const p=spawn('xcrun',['xctrace','export','--input',trace,'--xpath',`/trace-toc/run[@number="1"]/data/table[@schema="${schema}"]`]);p.stdout.setEncoding('utf8');let error='';p.stderr.on('data',b=>error+=b);
 const done=new Promise<void>((yes,no)=>{p.once('error',no);p.once('exit',code=>code===0?yes():no(Error(error)));});
 yield* parseTraceTable(p.stdout as AsyncIterable<string>);await done;
}
const labels=new Map<string,string>(),processes=new Set<string>(),encoderRows:TraceRow[]=[];
for await(const r of rows('metal-application-encoders-list')){encoderRows.push(r);const label=String(r['encoder-label']??'');if(label.includes('QA empty phi')){labels.set(String(r['encoder-id']),label);processes.add(String(r.process));}}
writeFileSync(resolve(dir,'encoders.jsonl'),encoderRows.map(r=>JSON.stringify(r)).join('\n'));assert.ok(labels.size,'labelled phi encoders captured');
type Span={start:number;end:number;label:string;id:string};const ours:Span[]=[],foreign:Span[]=[];const intervalRows:TraceRow[]=[];
for await(const r of rows('metal-gpu-intervals')){
 intervalRows.push(r);const start=timestampMicroseconds(r.start as string),duration=durationMicroseconds(r.duration as string),id=String(r['encoder-id']),label=labels.get(id),span={start,end:start+duration,label:label??'',id};
 if(label)ours.push(span);else if(!processes.has(String(r.process)))foreign.push(span);
}
writeFileSync(resolve(dir,'intervals.jsonl'),intervalRows.map(r=>JSON.stringify(r)).join('\n'));
ours.sort((a,b)=>a.start-b.start);foreign.sort((a,b)=>a.start-b.start);const merged:Span[]=[];
for(const x of foreign){const last=merged.at(-1);if(last&&x.start<=last.end)last.end=Math.max(last.end,x.end);else merged.push({...x});}
const lookup=(spans:Span[],t:number)=>{let lo=0,hi=spans.length;while(lo<hi){const m=(lo+hi)>>1;if(spans[m]!.start<=t)lo=m+1;else hi=m;}const s=spans[lo-1];return s&&t<s.end?s:undefined;};
const names=new Map<string,string>(),info:TraceRow[]=[];for await(const r of rows('gpu-counter-info')){info.push(r);names.set(String(r['counter-id']),String(r.name));}writeFileSync(resolve(dir,'counter-info.json'),JSON.stringify(info,null,2));
const buckets=new Map<string,{intervals:number;us:number;ids:Set<string>;samples:Record<string,{sum:number;n:number}>}>();
for(const s of ours){const b=buckets.get(s.label)??{intervals:0,us:0,ids:new Set<string>(),samples:{}};b.intervals++;b.us+=s.end-s.start;b.ids.add(s.id);buckets.set(s.label,b);}
let seen=0,accepted=0,contended=0;const sink=createWriteStream(resolve(dir,'phi-counter-samples.jsonl'));
for await(const r of rows('gpu-counter-value')){
 seen++;const time=timestampMicroseconds(r.timestamp as string),span=lookup(ours,time);if(!span)continue;
 if(lookup(merged,time)){contended++;continue;}const name=names.get(String(r['counter-id']));if(!name)continue;
 const value=Number(String(r.value??'').replace(/,/g,''));if(!Number.isFinite(value))continue;
 const b=buckets.get(span.label)!,a=b.samples[name]??{sum:0,n:0};a.sum+=value;a.n++;b.samples[name]=a;accepted++;sink.write(JSON.stringify({label:span.label,name,...r})+'\n');
}
await new Promise<void>(done=>sink.end(done));
const report={processes:[...processes],seen,accepted,contended,counterNames:[...names.values()],passes:[...buckets].map(([label,b])=>({label,intervals:b.intervals,encoders:b.ids.size,total_ms:b.us/1000,mean_us:b.us/b.ids.size,counters:Object.fromEntries(Object.entries(b.samples).map(([name,a])=>[name,{mean:a.sum/a.n,readings:a.n}]))}))};
writeFileSync(resolve(dir,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
