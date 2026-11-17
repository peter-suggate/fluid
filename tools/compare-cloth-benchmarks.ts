/** Strict promotion gate for collision-only experiments: same scene/tuning/poses, faster at every checkpoint.
 * node --import tsx tools/compare-cloth-benchmarks.ts baseline.json candidate.json baseline-poses.json candidate-poses.json
 * Pose files may contain full positions or per-frame SHA256 hashes from --pose-hashes.
 * Timing reports must come from exclusive M1 Max runs; this tool cannot establish that provenance itself.
 */
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';

type Step = { ms:number; spanMs:number; newton:number; pcg:number; g2:number[]; alpha:number; counters:Record<string,number|number[]> };
type Provenance={physics:Record<string,{stateSha256:string;physicsSha256:string}>;sourceSha256:string;backend:string;adapter:Record<string,string>;warmupWindows:number;warmupSteps:number};
type Report = { provenance:Provenance; nodes:number; vertices:number; newton:unknown; report:{tick:number;steps:Step[]}[] };
type Poses = {nodes:number;poses:({tick:number;positions_m:number[]}|{tick:number;sha256:string})[]};
const [baselineFile,candidateFile,baselinePoses,candidatePoses,expected="20,60,120"]=process.argv.slice(2);
const expectedCheckpoints=expected.split(",").map(Number);
assert.ok(expectedCheckpoints.length>0&&expectedCheckpoints.every(n=>Number.isInteger(n)&&n>=0),"explicit valid checkpoints required");
if(!baselineFile || !candidateFile || !baselinePoses || !candidatePoses) throw new Error('Provide baseline/candidate reports and baseline/candidate pose files');
const read=<T>(file:string):T=>JSON.parse(readFileSync(file,'utf8')) as T;
const base=read<Report>(baselineFile),candidate=read<Report>(candidateFile);
const referencePose=read<Poses>(baselinePoses),trialPose=read<Poses>(candidatePoses);
assert.ok(base.report.length>0 && candidate.report.length>0,'reports must contain checkpoints');
assert.ok(referencePose.poses.length>0 && trialPose.poses.length>0,'pose evidence must not be empty');
assert.ok(base.provenance && candidate.provenance,'checkpoint and hardware provenance is required');
assert.deepEqual(base.report.map(r=>r.tick),expectedCheckpoints,'reference must cover every requested checkpoint');
assert.ok(base.newton && typeof base.newton==='object','solver settings must be recorded');
assert.match(base.provenance.sourceSha256,/^[a-f0-9]{64}$/);
assert.match(candidate.provenance.sourceSha256,/^[a-f0-9]{64}$/);
assert.match(Object.values(base.provenance.adapter).join(' '),/m1[- ]max/i,'promotion target must be M1 Max');
assert.ok(Number.isInteger(base.provenance.warmupSteps)&&base.provenance.warmupSteps>=3);
assert.deepEqual(candidate.provenance.physics,base.provenance.physics,'checkpoint/physics hashes must match');
assert.equal(candidate.provenance.sourceSha256,base.provenance.sourceSha256,'compare flags using one frozen implementation');
assert.deepEqual(candidate.provenance.adapter,base.provenance.adapter);
assert.equal(base.provenance.backend,'metal');assert.equal(candidate.provenance.backend,'metal');
assert.ok(base.provenance.warmupWindows>=2 && candidate.provenance.warmupWindows>=2,'warm disposable instances before measurement');
assert.equal(candidate.provenance.warmupWindows,base.provenance.warmupWindows);
assert.equal(candidate.provenance.warmupSteps,base.provenance.warmupSteps);
assert.equal(base.nodes,150,'promotion reference must be the full 150×150 scene');
assert.equal(candidate.nodes,base.nodes);assert.equal(candidate.vertices,base.vertices);assert.equal(candidate.vertices,22500);
assert.deepEqual(candidate.newton,base.newton,'do not loosen solver tuning while comparing collision optimizations');
assert.deepEqual(trialPose,referencePose,'every measured pose must match exactly');
assert.equal(referencePose.nodes,150);
assert.deepEqual(candidate.report.map(r=>r.tick),base.report.map(r=>r.tick),'all reference checkpoints must be measured');
const median=(values:number[])=>{
  assert.ok(values.length>=3,'at least three steps per checkpoint');
  assert.ok(values.every(v=>Number.isFinite(v)&&v>0),'timings must be positive and finite');
  const sorted=[...values].sort((a,b)=>a-b),middle=Math.floor(sorted.length/2);
  return sorted.length%2 ? sorted[middle]! : 0.5*(sorted[middle-1]!+sorted[middle]!);
};
const expectedTicks=base.report.flatMap(r=>r.steps.map((_,i)=>r.tick+i+1));
assert.deepEqual(referencePose.poses.map(p=>p.tick),expectedTicks,'pose coverage must match every measured step');
for(const r of base.report){const p=base.provenance.physics[r.tick];assert.ok(p && /^[a-f0-9]{64}$/.test(p.stateSha256) && /^[a-f0-9]{64}$/.test(p.physicsSha256),'valid checkpoint/physics hashes required');}
for(const p of referencePose.poses){if('sha256' in p)assert.match(p.sha256,/^[a-f0-9]{64}$/);else {assert.equal(p.positions_m.length,4*22500);assert.ok(p.positions_m.every(Number.isFinite));}}
const results=[];
for(let i=0;i<base.report.length;i++){
  const a=base.report[i]!,b=candidate.report[i]!;
  assert.equal(a.steps.length,b.steps.length,'compare equal-length continuations');
  for(let j=0;j<a.steps.length;j++){
    const old=a.steps[j]!,next=b.steps[j]!;
    for(const row of [old,next]){assert.ok([row.newton,row.pcg,row.alpha,...row.g2].every(Number.isFinite));assert.equal(row.g2.length,2);}
    for(const key of ['newton','pcg','g2','alpha'] as const)assert.deepEqual(next[key],old[key],`solver change at checkpoint ${a.tick}, step ${j}`);
    for(const key of ['status','zoneVertices','zonePeak','conflicts','redos','levelFallback','intersections','solidIntersections','missingPairs','lagPairs','uncertified','uncovered'])
      {assert.ok(Number.isFinite(next.counters[key])&&Number.isFinite(old.counters[key]),`missing ${key} evidence`);assert.deepEqual(next.counters[key],old.counters[key],`${key} changed at checkpoint ${a.tick}, step ${j}`);}
  }
  const gpuBefore=median(a.steps.map(s=>s.spanMs)),gpuAfter=median(b.steps.map(s=>s.spanMs));
  const wallBefore=median(a.steps.map(s=>s.ms)),wallAfter=median(b.steps.map(s=>s.ms));
  results.push({tick:a.tick,gpuBefore,gpuAfter,wallBefore,wallAfter,speedup:gpuBefore/gpuAfter});
}
console.table(results);
for(const r of results){
  assert.ok(r.gpuAfter<=0.97*r.gpuBefore,`checkpoint ${r.tick}: GPU gain below the 3% promotion margin`);
  assert.ok(r.wallAfter<=r.wallBefore,`checkpoint ${r.tick}: fenced wall time regressed`);
}
console.log('Provisional fixed-window gate passed. Independent restored timing repeats, long trajectories, saved-state continuation, and live-render checks are still required.');
