/** Frozen full-frame addressing experiment. Replays one accepted first frame,
 * including queue parameter writes, from restored GPU inputs. This isolates
 * GPU execution; it excludes restoration, CPU planning, rendering and churn.
 */
import assert from "node:assert/strict";
import {mkdirSync,readFileSync,readdirSync,writeFileSync} from "node:fs";
import {dirname,resolve} from "node:path";
import {pathToFileURL} from "node:url";
import {createHash} from "node:crypto";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {getSceneDefinition} from "../lib/core/scenes";
import {sceneDocument} from "../lib/core/scene-definition";
import {refinementRegionLattice} from "../lib/core/refinement-regions";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";
import {usePerformanceInstrumentationStore} from "../lib/core/stores/performance-instrumentation-store";
import {readMixedTexture,readMixedTileWords} from "../tests/helpers/uniform-mixed-native-fields";
import {UniformAtlasAddressExperiment,type AtlasExperimentMode} from "./uniform-atlas-address-experiment";
import {UniformFrozenGPUFrame} from "./uniform-frozen-gpu-frame";
const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const mode=arg("atlas","native"),sceneId=arg("scene","minimal-power-dam-break-64"),edge=Number(arg("atlas-edge","32")),samples=Number(arg("samples","24")),fine=process.argv.includes("--fine"),identity=process.argv.includes("--identity");
assert.ok(["native","dense","affine","table"].includes(mode)&&[16,32].includes(edge)&&Number.isInteger(samples)&&samples>=4);
assert.ok(!identity||mode==="table");
const out=resolve(arg("out","/tmp/frozen-uniform.json"));
const files=[...readdirSync("lib/methods/uniform",{recursive:true}).map(String).filter(p=>p.endsWith(".ts")).map(p=>`lib/methods/uniform/${p}`),"lib/core/scenes.ts","lib/core/cm12-paper-scenes.ts","tools/probe-uniform-frozen-frame-dawn.ts","tools/uniform-frozen-gpu-frame.ts","tools/uniform-atlas-address-experiment.ts"].sort();
const fingerprint=()=>createHash("sha256").update(files.map(p=>`${p}:${createHash("sha256").update(readFileSync(p)).digest("hex")}`).join("\n")).digest("hex");
const sourceBefore=fingerprint();
const report:Record<string,unknown>={sceneId,mode,edge,samples,fine,identity,sourceBefore,scope:"One accepted first frame replayed from a complete GPU checkpoint. GPU timestamps include parameter-copy replay commands; restore and CPU solver planning are excluded. No trajectory, sparse allocation or renderer claim."};
const save=()=>{mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(report,null,2)+"\n");};
let raw:GPUDevice|undefined,solver:any,freeze:UniformFrozenGPUFrame|undefined,atlasResources:{destroy:()=>void}|undefined;
try{
 const dawn=await import(pathToFileURL(resolve(process.env.WEBGPU_NODE_MODULE??"node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
 const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal","disable-dawn-features=timestamp_quantization"]).requestAdapter();assert.ok(adapter);
 raw=await adapter.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 const errors:string[]=[];raw.addEventListener("uncapturederror",(e:GPUUncapturedErrorEvent)=>{e.preventDefault();errors.push(e.error.message);});
 const scene=sceneDocument(getSceneDefinition(sceneId));scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
 assert.ok(!scene.fluid.inflow?.enabled&&!scene.rigidBodies?.length,"Frozen fixture excludes inflow and rigid bodies");
 if(fine){const c=scene.container;scene.fluid.refinementRegions=[{id:"full-h",rule:"minimum-cell-size",minimumCellSize_cells:1,maximumCellSize_cells:1,min_m:{x:-c.width_m/2,y:0,z:-c.depth_m/2},max_m:{x:c.width_m/2,y:c.height_m,z:c.depth_m/2}}];}
 const dimensions=refinementRegionLattice(scene).dimensions;
 const atlas=mode==="native"?undefined:new UniformAtlasAddressExperiment(dimensions,edge as 16|32,mode as AtlasExperimentMode);
 if(atlas){if(identity)atlas.offsets.fill(0);atlasResources=atlas.install(raw);}
 freeze=new UniformFrozenGPUFrame(raw);
 const device=managedGPUDevice(raw,{requireWorkerRealm:false});usePerformanceInstrumentationStore.getState().setEnabled(false);
 const start=performance.now();solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",{timeStep:"scene",coarsening:"dynamic"},undefined,()=>{});
 await device.queue.onSubmittedWorkDone();report.setup_ms=performance.now()-start;report.initial={...await solver.readStats()};
 report.dimensions=dimensions;report.adapter={vendor:adapter.info.vendor,architecture:adapter.info.architecture,description:adapter.info.description};
 const fields=async()=>{
  const hashes:Record<string,string>={};
  for(const name of ["volume","velocity","phi"]){const t=solver.mixedFrame.fields[name] as GPUTexture;let a=await readMixedTexture(device,t);if(atlas)a=atlas.reorder(a,[t.width,t.height,t.depthOrArrayLayers],t.format==="rgba32float"?4:1);hashes[name]=createHash("sha256").update(new Uint8Array(a.buffer,a.byteOffset,a.byteLength)).digest("hex");}
  const t=await readMixedTileWords(device,solver);hashes.tiles=createHash("sha256").update(new Uint8Array(t.buffer)).digest("hex");return hashes;
 };
 report.inputHashes=await fields();freeze.checkpoint();await device.queue.onSubmittedWorkDone();freeze.begin();
 assert.ok(solver.advanceTo(1/60,[]));await solver.awaitFrameCompletion();freeze.end();
 report.outputHashes=await fields();report.accepted={...await solver.readStats()};assert.equal((report.accepted as any).simulationPipelineError,undefined);
 console.log(JSON.stringify({mode,setup_ms:report.setup_ms,checkpoint:freeze.statistics}));
 const times=await freeze.measure(samples);report.gpu_ms=times;
 const sorted=[...times].sort((a,b)=>a-b);report.summary={mean:times.reduce((a,b)=>a+b,0)/times.length,median:sorted[Math.floor(sorted.length/2)],p90:sorted[Math.floor(.9*sorted.length)]};
 report.replayedHashes=await fields();assert.deepEqual(report.replayedHashes,report.outputHashes,"Restored replay must reproduce the accepted first frame exactly");
 assert.deepEqual(errors,[]);report.validationErrors=errors;report.checkpoint=freeze.statistics;report.sourceAfter=fingerprint();save();console.log(JSON.stringify({out,summary:report.summary}));
}catch(e){report.failure=String(e);report.sourceAfter=fingerprint();save();throw e;}finally{solver?.destroy();freeze?.destroy();atlasResources?.destroy();raw?.destroy();}
