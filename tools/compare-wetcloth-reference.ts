/** Compare actual production publications to diagnostic-only upstream exports.
 * No fitted translation, time warp, or inferred vertex correspondence.
 * --probe=REPORT --reference=JSONL --manifest=JSON --out=REPORT
 */
import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import type {SceneDescription} from '../lib/core/model';
import type {ClothSurfacePublication} from '../lib/cloth/types';

const option=(key:string)=>process.argv.find(v=>v.startsWith(`--${key}=`))?.slice(key.length+3);
const required=(key:string)=>{const p=option(key);assert.ok(p,`Supply --${key}`);return resolve(p);};
const probePath=required('probe'),referencePath=required('reference'),manifestPath=required('manifest');
const probe=JSON.parse(await readFile(probePath,'utf8'));
const manifest=JSON.parse(await readFile(manifestPath,'utf8'));
assert.ok(!probe.failure&&!probe.superseded&&probe.completedSteps>0&&probe.physics,'Requires completed nonsuperseded production probe with physics provenance');
assert.equal(manifest.purpose,'offline-upstream-comparison-only');assert.equal(manifest.physicsCodeModified,false);
assert.equal(probe.upstreamCommit,manifest.upstreamCommit);assert.equal(probe.source.sha256,manifest.inputSha256,'Reference and application must use the same released XML');
assert.equal(probe.configuration,'released-input','Corrected variants need a matching independently exported reference');
const digest=(v:Uint8Array)=>createHash('sha256').update(v).digest('hex');
assert.equal(digest(await readFile(manifest.diagnosticsPatch)),manifest.diagnosticsPatchSha256,'Reference diagnostics patch changed after run');
assert.equal(digest(await readFile(referencePath)),manifest.outputSha256,'Reference state export differs from its completed-run manifest');
const scene=JSON.parse(gunzipSync(await readFile(probePath+'.scene.json.gz')).toString()) as SceneDescription;
assert.equal(scene.wetClothSource?.sourceId,probe.sourceId);
const offset=scene.wetClothSource!.sourceOffset_m;
interface Reference {step:number;time_s:number;dt_s:number;units:string;softParticles:number;elastoParticles:number;fluidParticles:number;x:number[];v:number[];waterVolume:number[]}
const reference=(await readFile(referencePath,'utf8')).trim().split('\n').map(s=>JSON.parse(s) as Reference);
assert.equal(reference.at(-1)?.step,manifest.requestedSteps,'Reference run is incomplete');
const key=(t:number)=>Math.round(t*1e10),times=new Map(reference.map(row=>[key(row.time_s),row]));
const rows=[];
for(const sample of probe.rows){
 const ref=times.get(key(sample.time_s));if(!ref)continue;
 assert.equal(ref.units,'CGS');assert.equal(ref.dt_s,manifest.dt_s);
 const frame=JSON.parse(gunzipSync(await readFile(probePath+`.frame-${String(sample.step).padStart(8,'0')}.json.gz`)).toString());
 assert.equal(frame.surfaces.length,1,'This exporter compares one source structural document');
 const surface=frame.surfaces[0] as ClothSurfacePublication,n=surface.positions_m.length/3;
 assert.equal(n,ref.softParticles,'Original authored vertex order and count are required');
 assert.ok(ref.x.length>=4*n&&ref.v.length>=4*n&&ref.waterVolume.length>=n);
 let position2=0,velocity2=0,boundL1=0,referenceBound=0;const errors=[];
 for(let v=0;v<n;v++){
  const distance=Math.hypot(...[0,1,2].map(a=>surface.positions_m[3*v+a]!-offset[a]!-.01*ref.x[4*v+a]!));
  const speed=Math.hypot(...[0,1,2].map(a=>surface.velocities_m_s![3*v+a]!-.01*ref.v[4*v+a]!));
  assert.ok(Number.isFinite(distance)&&Number.isFinite(speed));position2+=distance*distance;velocity2+=speed*speed;
  const water=ref.waterVolume[v]!*1e-6;referenceBound+=water;boundL1+=Math.abs(surface.waterVolume_m3[v]!-water);
  errors.push({vertex:v,positionError_m:distance,velocityError_m_s:speed});
 }
 errors.sort((a,b)=>b.positionError_m-a.positionError_m);
 rows.push({time_s:sample.time_s,productionStep:sample.step,referenceStep:ref.step,vertices:n,
  positionRms_m:n?Math.sqrt(position2/n):null,positionMax_m:errors[0]?.positionError_m??null,
  velocityRms_m_s:n?Math.sqrt(velocity2/n):null,boundDistributionL1_m3:boundL1,
  productionBoundVolume_m3:sample.boundVolume_m3,referenceBoundVolume_m3:referenceBound,
  referenceTotalWaterVolume_m3:ref.waterVolume.reduce((s,v)=>s+v,0)*1e-6,worstVertices:errors.slice(0,10)});
}
assert.ok(rows.some(row=>row.time_s>0),'No common positive accepted sample time');
const out=resolve(option('out')??'artifacts/wetcloth/upstream-comparison.json');await mkdir(dirname(out),{recursive:true});
await writeFile(out,JSON.stringify({comparison:'upstream-original-physics',paperReproductionValidated:false,sourceId:probe.sourceId,
 production:probePath,reference:referencePath,manifest:manifestPath,physics:probe.physics,
 coordinateTransform:{units:'CGS to SI',translation_m:offset,source:'saved scene sourceOffset_m; no fitted alignment'},
 productionDt_s:probe.dt_s,referenceDt_s:manifest.dt_s,rows},null,2)+'\n');console.log(out);
