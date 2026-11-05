/** Full released topology/timestep mechanics and prescribed-contact audit.
 * This dry subsystem probe does not certify liquid/cloth scene reproduction. */
import {readFile,writeFile} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import {performance} from 'node:perf_hooks';
import {wetClothWaterSources,sourceClothDocument} from '../lib/cloth/scenes/sources';
import {decodeClothTopology} from '../lib/cloth/source-assets';
import {ClothRuntime} from '../lib/cloth/runtime';
import {withUniformDevice} from '../tests/helpers/uniform-geometric';
const arg=(name:string)=>process.argv.find(v=>v.startsWith(`--${name}=`))?.slice(name.length+3),steps=Number(arg('steps')??1),out=arg('out')??'/tmp/fluid-wetcloth-research/source-mechanics-sweep.json',rows:unknown[]=[],mechanicsOnly=process.argv.includes('--mechanics-only');
if(!Number.isSafeInteger(steps)||steps<1)throw new Error('Invalid source mechanics step count');
const run=async(device?:GPUDevice)=>{
 for(const source of wetClothWaterSources.filter(s=>s.counts.vertices&&(!arg('scene')||s.id===arg('scene')))){
  let runtime:ClothRuntime|undefined;const start=performance.now();let completedSteps=0;
  try{
   if(source.fields.some(f=>f.kind==='file'))throw new Error('External source SDF file requires complete scene host');
   console.log(JSON.stringify({id:source.id,phase:'decode'}));const topology=decodeClothTopology(gunzipSync(await readFile(`public${source.asset.url}`)));console.log(JSON.stringify({id:source.id,phase:'prepare'}));runtime=new ClothRuntime(sourceClothDocument(source.id),topology,{sourceFields:mechanicsOnly?[]:source.fields});console.log(JSON.stringify({id:source.id,phase:'initialize-gpu'}));if(device)await runtime.initializeGPU(device);console.log(JSON.stringify({id:source.id,phase:'step'}));
   let maximumUpdate_m=0,maximumIterations=0,maximumContactIterations=0,refinementSteps=0;for(let i=0;i<steps;i++){
    const report=await runtime.stepAsync(source.dt_s,[],{gravity_m_s2:source.gravity_m_s2,...(mechanicsOnly?{selfCollision:false}:{})});completedSteps++;maximumUpdate_m=Math.max(maximumUpdate_m,report.maximumUpdate_m);maximumIterations=Math.max(maximumIterations,report.iterations);maximumContactIterations=Math.max(maximumContactIterations,report.contacts.iterations??0);if(report.refinementIterations)refinementSteps++;
   }
   rows.push({id:source.id,status:'dry-mechanics-passed',completedSteps,time_s:runtime.state.time_s,maximumUpdate_m,maximumIterations,maximumContactIterations,refinementSteps,elapsed_ms:performance.now()-start});
  }catch(error){rows.push({id:source.id,status:'failed',completedSteps,error:String(error),elapsed_ms:performance.now()-start});}
  finally{runtime?.dispose();}
  console.log(JSON.stringify(rows.at(-1)));await writeFile(out,JSON.stringify({reproductionValidated:false,sourceAttachments:'finite-spring',mode:mechanicsOnly?'dry-mechanics-without-contact-diagnostic':'dry-mechanics-subsystem-only',rows},null,2)+'\n');
 }
};
if(process.argv.includes('--cpu'))await run();else await withUniformDevice('full source mechanics audit',run);
