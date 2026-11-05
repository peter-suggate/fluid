/** Run through tools/gpu/fluid-gpu-run.sh check. Each original scene receives
 * its own bounded process: a nonlinear/contact pathology cannot hold a shared
 * GPU check slot indefinitely. Timeouts remain explicit validation failures. */
import {spawn} from 'node:child_process';
import {mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {wetClothWaterSources} from '../lib/cloth/scenes/sources';
const arg=(name:string)=>process.argv.find(s=>s.startsWith(`--${name}=`))?.slice(name.length+3),timeout_s=Number(arg('timeout-s')??120),out=resolve(arg('out')??'artifacts/wetcloth/source-dry-first-step.json'),directory=`${out}.scenes`;
if(!(timeout_s>0&&Number.isFinite(timeout_s)))throw new Error('Invalid source diagnostic timeout');
await mkdir(dirname(out),{recursive:true});await mkdir(directory,{recursive:true});
const rows:Record<string,unknown>[]=[];
for(const source of wetClothWaterSources.filter(s=>s.counts.vertices&&(!arg('scene')||s.id===arg('scene')))){
 const file=`${directory}/${source.id}.json`;await rm(file,{force:true});let log='',timedOut=false;const start=performance.now();
 const status=await new Promise<{code:number|null;signal:NodeJS.Signals|null}>((resolve,reject)=>{
  const child=spawn(process.execPath,['--import','tsx','tools/probe-cloth-source-mechanics.ts',`--scene=${source.id}`,`--out=${file}`],{stdio:['ignore','pipe','pipe']});
  let force:ReturnType<typeof setTimeout>|undefined;const timeout=setTimeout(()=>{timedOut=true;child.kill('SIGTERM');force=setTimeout(()=>child.kill('SIGKILL'),5000);},timeout_s*1000);
  const append=(chunk:Buffer)=>{log+=chunk.toString();if(log.length>100000)log=log.slice(-100000);};child.stdout.on('data',append);child.stderr.on('data',append);
  child.once('error',error=>{clearTimeout(timeout);if(force)clearTimeout(force);reject(error);});child.once('exit',(code,signal)=>{clearTimeout(timeout);if(force)clearTimeout(force);resolve({code,signal});});
 });
 await writeFile(`${file}.log`,log);
 let row:Record<string,unknown>;
 if(timedOut)row={id:source.id,status:'timeout',timeout_s,elapsed_ms:performance.now()-start};
 else try{const parsed=JSON.parse(await readFile(file,'utf8'));row=parsed.rows?.[0];if(!row||row.id!==source.id)throw new Error('Missing source receipt');}
 catch(error){row={id:source.id,status:'failed',process:status,error:String(error),elapsed_ms:performance.now()-start};}
 rows.push(row);console.log(JSON.stringify(row));await writeFile(out,JSON.stringify({reproductionValidated:false,mode:'dry-mechanics-first-step-only',sourceTimestepAndTopology:true,timeout_s,rows},null,2)+'\n');
}
if(rows.some(row=>row.status!=='dry-mechanics-passed'))process.exitCode=1;
