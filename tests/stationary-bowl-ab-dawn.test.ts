import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
const dawnTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;
const root=fileURLToPath(new URL('..',import.meta.url));
dawnTest('fully adaptive stationary bowl settles without generating motion or changing resolved volume', {timeout:240_000},()=>{
 const out='artifacts/stationary-bowl-2x/regression/adaptive';
 const run=spawnSync(process.execPath,['--import','tsx','tools/probe-stationary-bowl-ab-dawn.ts',
  '--arm=adaptive','--steps=480',`--output=${out}`],{cwd:root,env:process.env,encoding:'utf8',timeout:230_000,maxBuffer:8*1024*1024});
 assert.equal(run.status,0,`${run.error??''}\n${run.stdout}\n${run.stderr}`);
 const trace=JSON.parse(readFileSync(`${root}/${out}/trace.json`,'utf8')) as Array<{
  step:number;mass:number;maxSpeed:number;symmetry:number;generation:number;columnChangeRMS_mm:number;
  fullHeightMaxError_mm:number;curvatureXRMSError:number;curvatureZRMSError:number;
 }>;
 const settled=trace.find(r=>r.step===10)!;
 for(const row of trace){
  assert.ok(row.maxSpeed<1e-6,`step ${row.step} generated motion: ${row.maxSpeed}`);
  assert.ok(row.symmetry<1e-5,`step ${row.step} density reflection error: ${row.symmetry}`);
  assert.ok(Math.abs(row.mass/trace[0]!.mass-1)<1e-6,`step ${row.step} lost mass`);
  // Common control-volume comparison below is the authority. The finer column
  // check separately rejects the observed accumulating, interior lateral drift.
  assert.ok(row.columnChangeRMS_mm<.001,`step ${row.step} column drift ${row.columnChangeRMS_mm} mm`);
  assert.ok(row.fullHeightMaxError_mm<.1,`step ${row.step} surface height error ${row.fullHeightMaxError_mm} mm`);
  assert.ok(row.curvatureXRMSError<.06,`step ${row.step} x curvature error ${row.curvatureXRMSError}`);
  assert.ok(row.curvatureZRMSError<.042,`step ${row.step} z curvature error ${row.curvatureZRMSError}`);
  if(row.step>=10)assert.equal(row.generation,settled.generation,`step ${row.step} topology oscillation`);
 }
 const load=(step:number)=>{
  const b=readFileSync(`${root}/${out}/${step}-density.bin`);
  return new Float32Array(b.buffer,b.byteOffset,b.byteLength/4);
 };
 const a=load(0),b=load(480);
 // Restrict both states to identical width-8 volumes; a coarse cell average
 // is not comparable to the fine point samples produced by diagnostics.
 for(let z=0;z<40;z+=8)for(let y=0;y<32;y+=8)for(let x=0;x<48;x+=8){
  let delta=0;
  for(let dz=0;dz<8;dz++)for(let dy=0;dy<8;dy++)for(let dx=0;dx<8;dx++){
   const i=x+dx+48*(y+dy+32*(z+dz));delta+=b[i]!-a[i]!;
  }
  assert.ok(Math.abs(delta/512)<1e-6,`common volume ${x},${y},${z}: ${delta/512}`);
 }
});
