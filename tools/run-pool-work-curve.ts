/** Explicitly invoked GPU experiment; serialize every capture with the shared lock.
 * node --import tsx tools/run-pool-work-curve.ts --prefix transport-next [--baseline /absolute/pre-feature/snapshot] [--speeds 0.1,1,8]
 * Run only after source freeze/tests and a useful pilot. No field captures.
 */
import assert from "node:assert/strict";
import {spawnSync} from "node:child_process";
import {closeSync,copyFileSync,existsSync,mkdirSync,openSync,readFileSync,readdirSync,writeFileSync} from "node:fs";
import {createHash} from "node:crypto";
import {resolve} from "node:path";
const args=process.argv.slice(2),option=(name:string)=>{const index=args.indexOf(name);return index<0?undefined:args[index+1];};
const prefix=option("--prefix");assert.ok(prefix&&/^[a-z0-9-]+$/.test(prefix),"--prefix requires a fresh artifact name");
const speeds=(option("--speeds")??"0.1,1,8").split(",").map(Number);assert.ok(speeds.length&&speeds.every(speed=>Number.isFinite(speed)&&speed>0)&&new Set(speeds).size===speeds.length,"--speeds must contain unique positive finite speeds");
const root=process.cwd(),baseline=option("--baseline"),directory=resolve(root,"artifacts/work-adaptive-investigation");mkdirSync(directory,{recursive:true});
const sourceHash=(path:string)=>{const hash=createHash("sha256");for(const file of readdirSync(resolve(path,"lib/methods/uniform")).filter(file=>file.endsWith(".ts")).sort()){hash.update(file);hash.update(readFileSync(resolve(path,"lib/methods/uniform",file)));}return hash.digest("hex");};
const sources=[{role:"candidate",path:root},...(baseline?[{role:"baseline",path:resolve(baseline)}]:[])].map(source=>({...source,hash:sourceHash(source.path)}));
if(baseline)for(const file of ["investigate-pool-work.ts","sample-pool-velocity.ts","gpu-pass-profile.ts","narrow-band-dispatch-probe.ts","measure-pool-volume.ts"])copyFileSync(resolve(root,"tools",file),resolve(baseline,"tools",file));
for(const source of sources)for(const speed of speeds)for(const counters of source.role==="candidate"?[false,true]:[false]){
 assert.equal(sourceHash(source.path),source.hash,"source changed during the curve; stop and freeze again");
 const name=`${prefix}-${source.role}-speed${String(speed).replace(".","p")}${counters?"-counts":""}`,out=resolve(directory,name+".json"),log=resolve(directory,name+".log");
 assert.ok(!existsSync(out),`refusing to overwrite ${out}`);
 const env:NodeJS.ProcessEnv={...process.env,WEBGPU_NODE_MODULE:resolve(root,"node_modules/webgpu/index.js"),SCENARIO:"pool-impact",IMPACT_SPEED:String(speed),IMPACT_X:"0",IMPACT_Z:"0",GEO_SNAPSHOTS:"1",STEPS:"30",FROM:"2",OUT:out};
 for(const key of ["FIELDS_OUT","PROFILE","SPLIT","NB_SPLIT_ALL","WORK_COUNTERS","MODEL_DIAGNOSTICS","FORCE_FULL","NO_BODY","BAND_TOLERANCE","CURRENT_VOLUME"])delete env[key];
 Object.assign(env,counters?{WORK_COUNTERS:"1"}:{PROFILE:"1"});
 console.log(JSON.stringify({phase:"curve-start",role:source.role,speed_m_s:speed,counters,sourceHash:source.hash,out}));
 const fd=openSync(log,"w");let result;try{result=spawnSync("/usr/bin/lockf",["/private/tmp/claude-501/fluid-gpu.lock","node","--import","tsx","tools/investigate-pool-work.ts"],{cwd:source.path,env,stdio:["ignore",fd,fd]});}finally{closeSync(fd);}
 assert.equal(result.status,0,`capture failed; inspect ${log}`);
 const capture=JSON.parse(readFileSync(out,"utf8"));assert.equal(capture.sourceHash,source.hash,"capture imported an unexpected source");
 capture.benchmarkRole=source.role;writeFileSync(out,JSON.stringify(capture,null,2)+"\n");
 console.log(JSON.stringify({phase:"curve-complete",out,mean_ms:capture.summary.mean_ms}));
}
