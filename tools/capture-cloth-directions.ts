/** Off-live shadow CCD fixtures, full 150x150 scene. Run through fluid-gpu-run.sh
 * check. Copies are made before the existing filter and after nwApply; no solve
 * is skipped or modified. A second uninstrumented run must match before output.
 * These directions are AFFINE diagnostics, not the projected accepted path.
 * --ticks 231,340 --out /tmp/cloth-directions --cache <checkpoint directory>
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdirSync,readFileSync,writeFileSync,readdirSync} from 'node:fs';
import {homedir} from 'node:os';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createProcessRetainedDawnGPU} from '../lib/harness/node-dawn-provider';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {ClothSystem,CLOTH_SCENE_STEP_S} from '../lib/cloth/system';
import type {ClothGPUSolver} from '../lib/cloth/gpu-solver';
import {OGC_POSE_WORD,OGC_POSE_WORDS} from '../lib/cloth/contact/ogc-wgsl';
import {bridson2002RotatingSphere} from '../lib/cloth/scenes/bridson-2002';
import {bridson2002SceneDescription} from '../lib/cloth/scenes/bridson-2002/scene-description';

const args=process.argv.slice(2),opt=(k:string,d:string)=>{const i=args.indexOf('--'+k);return i<0?d:args[i+1]!;};
const ticks=opt('ticks','231,340').split(',').map(Number),out=resolve(opt('out','/tmp/cloth-directions'));
const cache=opt('cache',join(homedir(),'code/me/fluid-scratch/cloth-bench'));
assert.ok(ticks.every(t=>Number.isInteger(t)&&t>=0));
const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE??resolve('node_modules/webgpu/index.js')).href);
Object.assign(globalThis,dawn.globals);
const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);
const device=await adapter.requestDevice({requiredFeatures:adapter.features.has('subgroups')?['subgroups']:[],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const errors:string[]=[];
device.addEventListener('uncapturederror',(e:GPUUncapturedErrorEvent)=>{e.preventDefault();errors.push(e.error.message);});
const hash=(a:ArrayBufferView)=>createHash('sha256').update(new Uint8Array(a.buffer,a.byteOffset,a.byteLength)).digest('hex');
const sourceHash=createHash('sha256');
function sourceTree(dir:string){for(const entry of readdirSync(dir,{withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name))){const path=join(dir,entry.name);if(entry.isDirectory())sourceTree(path);else if(/\.(ts|tsx|wgsl)$/.test(path)){sourceHash.update(path);sourceHash.update(readFileSync(path));}}}
sourceTree('lib/cloth');sourceHash.update(readFileSync('tools/capture-cloth-directions.ts'));
const clothSourceSha256=sourceHash.digest('hex');
type Copy={name:string;buffer:GPUBuffer};
const create=device.createCommandEncoder.bind(device);
let active:ClothGPUSolver|undefined,copies:Copy[]=[],iteration=0;
const copy=(encoder:GPUCommandEncoder,name:string,b:GPUBufferBinding)=>{
  const size=Number(b.size);assert.ok(Number.isInteger(size)&&size>0);
  const buffer=device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  encoder.copyBufferToBuffer(b.buffer,b.offset??0,buffer,0,size);copies.push({name,buffer});
};
device.createCommandEncoder=(descriptor?:GPUCommandEncoderDescriptor)=>{
  const encoder=create(descriptor),solver=active;
  if(!solver)return encoder;
  return new Proxy(encoder,{get(target,key){
    if(key!=='beginComputePass'){const v=Reflect.get(target,key);return typeof v==='function'?v.bind(target):v;}
    return (d?:GPUComputePassDescriptor)=>{
      if(d?.label!=='Cloth step')return target.beginComputePass(d);
      let pass=target.beginComputePass(d),pipeline:GPUComputePipeline;
      const groups=new Map<number,[GPUBindGroup,number[]]>();
      const cut=(fn:()=>void)=>{pass.end();fn();pass=target.beginComputePass(d);if(pipeline)pass.setPipeline(pipeline);for(const [i,[g,o]] of groups)pass.setBindGroup(i,g,o);};
      const run=(dispatch:()=>void)=>{
        if(pipeline.label==='ogc filterBegin')cut(()=>{
          copy(target,`${iteration}-x`,solver.positionsBinding);copy(target,`${iteration}-d`,solver.newtonDirection);copy(target,`${iteration}-control`,solver.newtonControl);
        });
        dispatch();
        if(pipeline.label==='Cloth nwApply')cut(()=>{
          copy(target,`${iteration}-accepted`,solver.positionsBinding);iteration++;
        });
      };
      return {
        setPipeline(p:GPUComputePipeline){pipeline=p;pass.setPipeline(p);},
        setBindGroup(i:number,g:GPUBindGroup,offsets:Iterable<number>=[]){const o=Array.from(offsets);groups.set(i,[g,o]);pass.setBindGroup(i,g,o);},
        dispatchWorkgroups(x:number,y=1,z=1){run(()=>pass.dispatchWorkgroups(x,y,z));},
        dispatchWorkgroupsIndirect(b:GPUBuffer,o:number){run(()=>pass.dispatchWorkgroupsIndirect(b,o));},
        end(){pass.end();},
      } as unknown as GPUComputePassEncoder;
    };
  }});
};
try {
  mkdirSync(out,{recursive:true});
  for(const tick of ticks) {
    const checkpoint=readFileSync(join(cache,`sphere-150-tick${tick}.json`),'utf8');
    let baseline:unknown;
    for(const capture of [false,true]) {
      const scene=bridson2002SceneDescription(bridson2002RotatingSphere({nodesAcross:150}));
      const document=JSON.parse(checkpoint).document;scene.cloth=[document];
      const system=await ClothSystem.create(device,scene,{liveOptimizations:true,statusInterval:1});
      const solver=system.solverFor(document.id)!,contact=system.contactFor(document.id)!;
      try {
        iteration=0;active=capture?solver:undefined;
        assert.ok(system.stepTo(system.time_s+CLOTH_SCENE_STEP_S));active=undefined;
        await device.queue.onSubmittedWorkDone();assert.equal(system.failure,undefined);
        const carried=await contact.readCarriedState(),counters=await contact.readCounters(),state=await solver.readState();
        const entries=await contact.readContactEntries(),contacts=createHash('sha256');contacts.update(new Uint8Array(entries.counts.buffer));
        for(let v=0;v<entries.counts.length;v++)contacts.update(new Uint8Array(entries.entries.buffer,entries.entries.byteOffset+16*v*entries.header[1]!,16*entries.counts[v]!));
        const pairs=await contact.readPairs(),bits=new Uint32Array(pairs.planes.buffer),rows:string[]=[];
        for(let p=0;p<pairs.slots;p++)if(pairs.ids[4*p]!==0xffffffff)rows.push([...pairs.ids.subarray(4*p,4*p+4),...bits.subarray(4*p,4*p+4)].join(','));
        rows.sort();
        const accepted={pose:hash(await solver.readPositions()),control:Array.from(await solver.readNewtonControl()),held:hash(carried.held),reference:hash(carried.reference),
          velocity:hash(state.velocities_m_s),history:state.solverState,contacts:contacts.digest('hex'),dat:createHash('sha256').update(rows.join(';')).digest('hex'),
          safety:Object.fromEntries(['status','intersections','solidIntersections','missingPairs','lagPairs','uncovered','uncertified','conflicts','redos','heldConflict','zoneVertices','zonePeak','levelFallback'].map(k=>[k,counters[k as keyof typeof counters]]))};
        if(!capture){baseline=accepted;continue;}
        assert.deepEqual(accepted,baseline,'Instrumentation changed accepted solver state');
        assert.ok(iteration>0);assert.equal(copies.length,4*iteration);
        const encoder=create();copy(encoder,'triangles',solver.triangles);copy(encoder,'edges',solver.edges);
        const solidParts=[['solid-samples',contact.sampleBase,contact.solidSamples],['solid-edges',contact.edgeBase,contact.solidSampleEdges],['solid-facets',contact.facetBase,contact.solidSampleFacets],['body-samples-local',contact.bodySampleBase,contact.bodySamples],['body-edges',contact.bodyEdgeBase,contact.bodySampleEdges],['body-facets',contact.bodyFacetBase,contact.bodySampleFacets]] as const;
        for(const [name,base,count] of solidParts)if(count)copy(encoder,name,{buffer:contact.contacts,offset:16*base,size:16*count});
        if(contact.bodyBase)copy(encoder,'body-poses',{buffer:contact.contacts,offset:4*OGC_POSE_WORD,size:8*OGC_POSE_WORDS});
        device.queue.submit([encoder.finish()]);
        const files=[],slots=[];
        for(const {name,buffer} of copies){
          await buffer.mapAsync(GPUMapMode.READ);const data=new Uint8Array(buffer.getMappedRange().slice(0));
          if(name.endsWith('-control')){const control=new Float32Array(data.buffer);slots.push({iteration:Number(name.split('-')[0]),active:control[0]===0,control:Array.from(control)});}
          const filename=`tick${tick}-${name}.bin`;writeFileSync(join(out,filename),data,{flag:'wx'});files.push({name,filename,bytes:data.byteLength,sha256:hash(data)});buffer.destroy();
        }
        copies=[];
        assert.deepEqual(errors,[]);
        writeFileSync(join(out,`tick${tick}.json`),JSON.stringify({tick,nodes:150,vertices:solver.N,triangles:solver.T,edges:solver.E,stride:solver.tables.stride,iterations:iteration,
          pathContract:'unprojected affine Newton direction, cloth only; NOT a certificate for OGC projected trials or moving solids',
          checkpointSha256:createHash('sha256').update(checkpoint).digest('hex'),clothSourceSha256,dt:CLOTH_SCENE_STEP_S,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,description:adapter.info.description},
          documentSolver:document.solver,liveOptimizations:true,sepEps:contact.sepEps,gammaR:contact.gammaR,activationRadius:contact.config.r,
          solidParts:Object.fromEntries(solidParts.map(([name,,count])=>[name,count])),solidContract:'Original shared-field hull samples/facets; unsigned hull CCD alone does not certify signed-field positivity or rotating-body paths',slots,accepted,files},null,2),{flag:'wx'});
        console.log('PASS captured unchanged full-resolution directions',tick,iteration);
      }finally{active=undefined;system.destroy();for(const c of copies)c.buffer.destroy();copies=[];}
    }
  }
}finally{device.createCommandEncoder=create;device.destroy();}
