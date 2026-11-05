import type { GPUSolverInstance } from '../core/method-contract';
import type { SceneDescription } from '../core/model';
import type { RigidBodyState } from '../core/rigid-body';
import { ClothRuntime } from './runtime';
import { clothCollidersForRigidBodies } from './contact';
import { cachedClothTopology, loadClothTopology } from './source-assets';
import { clothDocumentsKey } from './document-key';
import { rebaseClothRuntime, editedSourceFields } from './rebase';
import { clothContainerColliders, clothSceneStaticColliders } from './scene-contact';
import { composeClothFeatures } from './definition';

/** Which owner runs a cloth's nonlinear solve. The GPU sweep is Gauss-Seidel
 * over independent colors, so its latency is colors x one thread's serial
 * line-searched Newton update, whatever the vertex count. The Float64 CPU
 * owner is a global Newton-PCG solve of the same energy to the same tolerance
 * (lib/cloth/nonlinear-solve.ts), costing per vertex. Measured on the released
 * small fixtures (tools/probe-wetcloth-step-cost.ts --cpu, 1 ms steps, 40 steps
 * from rest; Newton iterations / coordinate sweeps per step):
 *   simple_cloth (52 vertices)   5.3 ms, 3.0   vs Gauss-Seidel   42 ms, 24
 *   simple_plastic (52)          4.2 ms, 2.0   vs Gauss-Seidel 2627 ms, 900
 *   cut simple_cloth (56, 120)   4.9 ms, 3.3   vs Gauss-Seidel  135 ms, 92
 *   simple_yarn (28)            25 ms,   3.9   vs Gauss-Seidel   97 ms, 12
 *   simple_yarn_loop (20)       21 ms,   4.7   vs Gauss-Seidel  289 ms, 73
 * The GPU owner took 188 ms per simple_cloth step. */
export const CLOTH_CPU_SOLVE_MAX_VERTICES=256;
export function clothSolvesOnGPU(runtime:ClothRuntime):boolean{return runtime.prepared.vertexCount>CLOTH_CPU_SOLVE_MAX_VERTICES;}

/** Attach the same mechanical plugin to a dry scene owner. No fluid fields
 * are allocated or advanced. A dynamic rigid contact needs an external
 * impulse owner; source fixtures use their prescribed SDF supports. */
export async function withDryClothPhysics(base:GPUSolverInstance,device:GPUDevice,initial:SceneDescription):Promise<GPUSolverInstance>{
 if(!initial.cloth?.length)return base;
 if(initial.cloth.length>1)throw new Error('Dry cloth requires one shared topology for cross-cloth contact');
 if(initial.systems?.fluid!==false)throw new Error('Dry cloth host requires explicit fluid:false');
 composeClothFeatures();
 let scene=initial,last=0,pending:Promise<void>|undefined,stepping=false,generation=0,failure:Error|undefined,disposed=false;
 const options=(s:SceneDescription)=>({sourceFields:s.wetClothSource?.fields,excludedSourceColliderIndices:s.wetClothSource?.bakedSolidFieldIndices,sourceOffset_m:s.wetClothSource?.sourceOffset_m,nonlinearSolver:'newton-pcg' as const});
 let runtimes:ClothRuntime[]=[];
 try {
  for(const document of initial.cloth){
   const runtime=new ClothRuntime(document,await loadClothTopology(document.geometry),options(initial));
   runtimes.push(runtime);if(clothSolvesOnGPU(runtime))await runtime.initializeGPU(device);
  }
 }catch(error){runtimes.forEach(r=>r.dispose());throw error;}
 let surfaces=runtimes.map(r=>r.publish());
 const failed=(error:unknown)=>{failure=error instanceof Error?error:new Error(String(error));base.info.simulationPipelineError=failure.message;};
 const adopt=(next:SceneDescription)=>{
  if(clothDocumentsKey(next.cloth)!==clothDocumentsKey(scene.cloth)){
   // A step in flight owns the state; an initializing solver does not, so the
   // accepted state is rebased and the newer initialization supersedes it.
   if(stepping)throw new Error('Cloth edit requires an accepted mechanical frame');
   const docs=next.cloth??[];
   if(docs.length!==runtimes.length||docs.some((d,i)=>d.id!==runtimes[i]!.document.id))throw new Error('Cloth roster change requires a new host');
   const replacements=docs.map((document,i)=>{
    const data=cachedClothTopology(document.geometry);if(!data)throw new Error('Cloth edit asset is not loaded');
    return rebaseClothRuntime(runtimes[i]!,document,data,options(next));
   });
   runtimes.forEach(r=>r.dispose());runtimes=replacements;surfaces=runtimes.map(r=>r.publish());
   const current=++generation;
   pending=Promise.all(runtimes.filter(clothSolvesOnGPU).map(r=>r.initializeGPU(device))).then(()=>{},failed).finally(()=>{if(generation===current)pending=undefined;});
  } else if(next.wetClothSource!==scene.wetClothSource) {
   for(const runtime of runtimes){runtime.setExcludedSourceColliderIndices(next.wetClothSource?.bakedSolidFieldIndices??[]);runtime.motion.restore({...runtime.motion.checkpoint(),fields:editedSourceFields(runtime,next.wetClothSource?.fields??[])});}
  }
  scene=next;
 };
 const advance=(time_s:number,bodies:RigidBodyState[]=[])=>{
  if(disposed||pending||failure||time_s<=last+1e-12)return false;
  if(bodies.some(b=>b.description.motion!=='static'&&!b.held))throw new Error('Dry cloth dynamic rigid contacts require a reciprocal rigid impulse owner');
  const dt=Math.min(scene.numerics.maxDt_s,time_s-last),end=last+dt,checkpoints=runtimes.map(r=>r.checkpoint());
  base.info.submittedTime_s=end;base.info.lastDt_s=dt;base.info.encodedSteps=(base.info.encodedSteps??0)+1;
  stepping=true;const current=++generation;
  pending=(async()=>{
   try {
    const g=scene.fluid.gravity_m_s2,colliders=[...clothCollidersForRigidBodies(bodies),...clothContainerColliders(scene),...clothSceneStaticColliders(scene)];
    for(const runtime of runtimes)await runtime.stepAsync(dt,colliders,{gravity_m_s2:[g.x,g.y,g.z]});
    if(disposed)return;
    surfaces=runtimes.map(r=>r.publish());last=end;base.info.completedTime_s=end;
   }catch(error){runtimes.forEach((r,i)=>r.restore(checkpoints[i]!));throw error;}
  })().catch(failed).finally(()=>{stepping=false;if(generation===current)pending=undefined;});
  return true;
 };
 const overrides={
  get clothSurfaces(){return surfaces;},get framePending(){return Boolean(pending);},deferredFramePublication:true,
  get presentationPending(){return Boolean(pending)||Boolean(failure);},
  advanceTo:advance,
  async awaitFrameCompletion(){await pending;if(failure)throw failure;},
  async assertSimulationHealthy(){await pending;if(failure)throw failure;},
  applySceneUniforms(next:SceneDescription){adopt(next);base.stageSceneUpdate?.(next);},
  stageSceneUpdate(next:SceneDescription){adopt(next);base.stageSceneUpdate?.(next);},
  destroy(){if(disposed)return;disposed=true;runtimes.forEach(r=>r.dispose());base.destroy();},
 };
 // Preserve all presentation APIs and instanceof identity of the actual
 // scene owner; the composed plugin replaces only mechanical lifecycle seams.
 return new Proxy(base,{get(target,key){const owner=key in overrides?overrides:target;const value=Reflect.get(owner,key,owner);return typeof value==='function'?value.bind(owner):value;}});
}
