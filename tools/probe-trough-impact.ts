/** Diagnose the settled trough contact contour using the production solver.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js npx tsx tools/probe-trough-impact.ts
 * PROBE_STEPS=5 STAGES=1 captures each surface stage. ZERO_G=1 keeps the
 * detached ball suspended. FINE_CONTACT=1 and NO_CONTACT_PLANE=1 are
 * diagnostic ablations only, not proposed solver fixes. Displacements and phi are metres; sample
 * coordinates are lattice indices.
 */
import {createUniformTroughScene} from '../lib/core/uniform-trough-scenes';
import {advanceUniform,createUniformSolver,readUniformFields,withUniformDevice} from '../tests/helpers/uniform-geometric';
import {uniformDetailField} from '../lib/methods/uniform/uniform-detail-fields';
import {writeFileSync} from 'node:fs';
import {RasterWaterPipeline} from '../lib/core/webgpu-water-pipeline';
import {bathSolidDistance} from '../lib/core/voxel-bath';
const h=Number(process.env.PROBE_H??.0125),steps=Number(process.env.PROBE_STEPS??35);
await withUniformDevice('trough impact probe',async device=>{
 if(process.env.NO_CONTACT_PLANE){const create=device.createShaderModule.bind(device);device.createShaderModule=(descriptor:GPUShaderModuleDescriptor)=>{const from='if(planeWriter(corner)==tile&&oldBuried(corner))';if(descriptor.code.includes(from)){console.log('ablating buried-corner plane extrapolation');descriptor={...descriptor,code:descriptor.code.replace(from,'if(false)')};}return create(descriptor);};}
 const scene=createUniformTroughScene('settled-tank',h);const level=Math.max(...scene.fluid.initialLiquidVolumes!.map(v=>(v as any).max_m.y));
 if(process.env.ZERO_G)scene.fluid.gravity_m_s2={x:0,y:0,z:0};
 const owned:GPUBuffer[]=[];let water:RasterWaterPipeline|undefined;
 if(process.env.MESH){const u=device.createBuffer({size:416,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const bodies=device.createBuffer({size:1536,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.STORAGE});owned.push(u,bodies);
  const words=new Float32Array(104);words.set([3.2,1.2,1.2,0],12);words.set([Math.round(3.2/h),Math.round(1.2/h),Math.round(1.2/h),1],20);device.queue.writeBuffer(u,0,words);
  water=new RasterWaterPipeline(device,'rgba8unorm',u,bodies);await water.initialize();console.log('water ready');}
 const solver=await createUniformSolver(device,scene,{...(process.env.STRICT?{bandExit:'strict'}:{}),...(process.env.FINE_CONTACT?{warmResolution:'h',coldRegions:'fine',coldFrameSkip:'off',regionTiering:'motion'}:{})});
 try{
  const frame=(solver as any).mixedFrame;console.log('scene',JSON.stringify({level,h,dimensions:[solver.info.nx,solver.info.ny,solver.info.nz]}));
  for(let i=1;i<=120;i++)await advanceUniform(solver,i/60);
  const meshContacts=async()=>{
   if(!water)return undefined;water.setVolume(solver.surfaceFieldTexture,solver.denseLevelSetVolumeSource);
   const e=device.createCommandEncoder();water.encodeDenseSurfaceExtractionForQA(e,solver.info.nx,solver.info.ny,solver.info.nz,'windows');device.queue.submit([e.finish()]);
   const mesh=await water.readDenseSurfaceExtractionForQA();const points=new Map<string,number>();let finite=true;
   for(let i=0;i<mesh.vertices.length;i+=8){const [x,y,z]=mesh.vertices.subarray(i,i+3);finite&&=Number.isFinite(x+y+z);
    if(x< -1||x>1||y<.075||y>level-2*h||z<.2||z>.5)continue;
    const ix=(x+1.6)/h-.5,iy=y/h-.5;
    if(Math.abs(ix-Math.round(ix))>1e-4||Math.abs(iy-Math.round(iy))>1e-4)continue;
    const key=`${Math.round(ix)},${Math.round(iy)}`;points.set(key,Math.min(points.get(key)??Infinity,z));
   }
   return {points,vertices:mesh.vertexCount,allocated:mesh.vertexAllocator,finite};
  };
  const initialMesh=await meshContacts();console.log('initial mesh',initialMesh&&{vertices:initialMesh.vertices,contacts:initialMesh.points.size});
  const initial=await readUniformFields(device,solver),{nx,ny,nz}=solver.info;
  const samples:{x:number,y:number,z:number,a:number,b:number}[]=[];
  for(let x=Math.ceil(.6/h);x<Math.floor(2.6/h);x+=2)for(let y=Math.ceil(.075/h);y<Math.floor((level-.01)/h);y+=2){
   for(let z=Math.floor(.7/h);z<Math.floor(1.1/h);z++){
    const a=initial.vertex(x,y,z),b=initial.vertex(x,y,z+1);if(a<=0&&b>0){samples.push({x,y,z:z+a/(a-b),a,b});break;}
   }
  }
  console.log('samples',samples.length,samples.slice(0,4));
  const initialVolume=initial.density.reduce((sum:number,v:number)=>sum+v,0)*h*h*h;
  const history:any[]=[];
  const measure=(f:typeof initial)=>{
   let max=0,loc:any;let lost=0,sum=0;
   for(const s of samples){let found=false;for(let z=Math.max(0,Math.floor(s.z)-8);z<Math.min(nz,Math.ceil(s.z)+8);z++){
    const a=f.vertex(s.x,s.y,z),b=f.vertex(s.x,s.y,z+1);if(a<=0&&b>0){const zero=z+a/(a-b),delta=(zero-s.z)*h;sum+=Math.abs(delta);if(Math.abs(delta)>max){max=Math.abs(delta);loc={x:s.x,y:s.y,z:s.z,zero,a,b,delta};}found=true;break;}
   }if(!found)lost++;}
   return {max,mean:sum/samples.length,lost,loc};
  };
  solver.injectLiquidBall({centre_m:{x:Number(process.env.BALL_X??-.5),y:level+.3,z:0},radius_m:.075});
  let stage='remap';const captures:{stage:string,done:()=>Promise<Float32Array>}[]=[];
  for(const name of ['reconstruct','clearAir','retire','redistance']){const original=frame.narrowBandFlip[name].bind(frame.narrowBandFlip);frame.narrowBandFlip[name]=(...args:any[])=>{stage=name;if(process.env.OMIT_STAGE===name)return false;return original(...args);};}
  const resolve=frame.phiResolve.encode.bind(frame.phiResolve);frame.phiResolve.encode=(encoder:any,group:any,...args:any[])=>{
   resolve(encoder,group,...args);if(process.env.STAGES){const tex=group===frame.phiResolveGroups.scratch?frame.fields.phiScratch:frame.fields.phi;captures.push({stage:group===frame.phiResolveGroups.scratch?'advect':stage,done:uniformDetailField(tex)!.storage.capture(encoder,tex)});}
  };
  const gated=frame.phiResolve.encodeGated.bind(frame.phiResolve);frame.phiResolve.encodeGated=(encoder:any,...args:any[])=>{gated(encoder,...args);if(process.env.STAGES){const tex=frame.fields.phi;captures.push({stage:'clearAir',done:uniformDetailField(tex)!.storage.capture(encoder,tex)});}};
  for(let i=1;i<=steps;i++){
   stage='remap';await advanceUniform(solver,(120+i)/60);const f=await readUniformFields(device,solver);const row:any={step:i,time:i/60,wallLine:Array.from({length:13},(_,j)=>f.vertex(Math.round(.6/h),Math.round(.2/h),Math.round(.9/h)+j-4)),edge:measure(f),particles:frame.narrowBandFlip.count,fine:f.tiles.filter((w:number)=>w>>>31).length};
   if(process.env.STAGES){row.stages=[];for(const cap of captures.splice(0)){const phi=await cap.done();const saved=f.phi.slice();f.phi.set(phi);row.stages.push({stage:cap.stage,...measure(f),wallLine:Array.from({length:13},(_,j)=>f.vertex(Math.round(0.6/h),Math.round(.2/h),Math.round(.9/h)+j-4))});f.phi.set(saved);}}
   if(water){const mesh=(await meshContacts())!;let max=0,lost=0;for(const [key,z] of initialMesh!.points){const next=mesh.points.get(key);if(next===undefined)lost++;else max=Math.max(max,Math.abs(z-next));}row.mesh={vertices:mesh.vertices,allocated:mesh.allocated,finite:mesh.finite,contacts:mesh.points.size,max,lost};}
   row.volume=f.density.reduce((sum:number,v:number)=>sum+v,0)*h*h*h; if(row.edge.loc){const p=row.edge.loc;row.edge.solidDistance=bathSolidDistance(p.x*h-1.6,p.y*h,p.zero*h-.6);} history.push(row);console.log(JSON.stringify(row));
  }
  writeFileSync(`/tmp/trough-impact-${process.env.OMIT_STAGE??(process.env.FINE_CONTACT?'fine-contact':process.env.NO_CONTACT_PLANE?'no-plane':process.env.STAGES?'stages':process.env.ZERO_G?'zero':process.env.STRICT?'strict':'normal')}-${h}.json`,JSON.stringify({h,level,initialVolume,samples,history},null,2));
 }finally{solver.destroy();water?.destroy();owned.forEach(b=>b.destroy());}
});
