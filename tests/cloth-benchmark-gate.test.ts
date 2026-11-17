import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {spawnSync} from 'node:child_process';

test('cloth promotion gate rejects missing fidelity evidence and performance regressions',()=>{
  const dir=mkdtempSync(join(tmpdir(),'cloth-gate-'));
  const keys=['status','zoneVertices','zonePeak','conflicts','redos','levelFallback','intersections','solidIntersections','missingPairs','lagPairs','uncertified','uncovered'];
  const steps=Array.from({length:3},()=>({ms:120,spanMs:100,newton:8,pcg:40,g2:[100,1],alpha:1,counters:Object.fromEntries(keys.map(k=>[k,0]))}));
  const baseline={nodes:150,vertices:22500,newton:{iterations:8},provenance:{physics:{20:{stateSha256:'1'.repeat(64),physicsSha256:'2'.repeat(64)}},sourceSha256:'3'.repeat(64),adapter:{description:'Apple M1 Max'},backend:'metal',warmupWindows:2,warmupSteps:3},report:[{tick:20,steps}]};
  const candidate=structuredClone(baseline);candidate.report[0]!.steps.forEach(s=>{s.ms=100;s.spanMs=90;});
  const poses={nodes:150,poses:[21,22,23].map(tick=>({tick,sha256:'4'.repeat(64)}))};
  const run=(a:unknown,b:unknown,p:unknown=poses,q:unknown=p)=>{
    const files=['base','candidate','base-poses','candidate-poses'].map(n=>join(dir,n+'.json'));
    [a,b,p,q].forEach((v,i)=>writeFileSync(files[i]!,JSON.stringify(v)));
    return spawnSync(process.execPath,['--import','tsx',resolve('tools/compare-cloth-benchmarks.ts'),...files,'20'],{encoding:'utf8'});
  };
  try{
    const ok=run(baseline,candidate);assert.equal(ok.status,0,ok.stderr);
    assert.notEqual(run({...baseline,report:[]},{...candidate,report:[]},{nodes:150,poses:[]}).status,0,'empty reports must fail');
    const absent=structuredClone(candidate);delete (absent.report[0]!.steps[0]!.counters as Record<string,unknown>).status;
    assert.notEqual(run(baseline,absent).status,0,'missing contact status must fail');
    assert.notEqual(run(baseline,candidate,{...poses,poses:poses.poses.slice(1)}).status,0,'incomplete poses must fail');
    assert.notEqual(run(baseline,{...candidate,provenance:undefined}).status,0,'missing checkpoint provenance must fail');
    const slower=structuredClone(candidate);slower.report[0]!.steps.forEach(s=>s.spanMs=110);
    assert.notEqual(run(baseline,slower).status,0,'fidelity-equivalent slowdowns must fail');
    const changed=structuredClone(poses);changed.poses[0]!.sha256='5'.repeat(64);
    assert.notEqual(run(baseline,candidate,poses,changed).status,0,'faster changed motion must fail');
  }finally{rmSync(dir,{recursive:true,force:true});}
});
