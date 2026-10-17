import assert from "node:assert/strict";
import test from "node:test";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";
import { uniformLatticeAddress,uniformTileCoord } from "../lib/methods/uniform/uniform-lattice-address";
import { createUniformMixedLayout } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UniformSurfaceWorkControl } from "../lib/methods/uniform/uniform-surface-work-control";
import { UniformMixedSurfaceGeometry } from "../lib/methods/uniform/uniform-mixed-surface-geometry";
import { UniformMixedPhiResolve } from "../lib/methods/uniform/uniform-mixed-phi-resolve";
import { UniformMixedForces, UNIFORM_NORMAL_ANCHOR_ERROR } from "../lib/methods/uniform/uniform-mixed-forces";
import { uniformCoarseCacheExtent } from "../lib/methods/uniform/uniform-coarse-velocity-cache";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { UniformDetailStorage,uniformDetailBindLayout,uniformDetailGroup,uniformDetailModule,uniformDetailPipeline,uniformDetailPick } from "../lib/methods/uniform/uniform-detail-fields";

for(const [fineLocation,bounded,snapshots] of [[-1,false,true],[0,false,true],[73,false,true],[-1,true,true],[-1,false,false],[73,false,false],[-1,true,false]] as const)test(`persistent geometry normals match full force stencils and survive deferred curvature (${fineLocation<0?"coarse":`fine overflow at ${fineLocation}`}, bounded=${bounded}, snapshots=${snapshots})`,{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("retained exact normal and curvature regression",async device=>{
  const mixed=fineLocation>=0;
  const dimensions:[number,number,number]=[32,32,32],address=uniformLatticeAddress(dimensions);
  const forced=mixed?new Uint8Array(address.boxTiles):undefined;if(forced)forced[fineLocation]=1;
  const layout=createUniformMixedLayout({dimensions,cellSize_m:[1,1,1],origin_m:{x:0,y:0,z:0}},[],4,forced,address);
  const ownership=new UniformMixedOwnership(device,layout,true,mixed?1:0),storage=new UniformDetailStorage(device,address);storage.attach(ownership);
  const scalar=(name:string,pin=false)=>storage.createField(name,"cell","r32float",pin);
  const phi=storage.createField("normal fixture phi","vertex","r32float"),target=scalar("target"),center=scalar("center"),volume=scalar("volume"),pressure={target:scalar("pressure target",true),centerPhi:scalar("pressure center",true)};
  const velocity=storage.createField("normal fixture velocity","face","rgba32float"),unit=storage.createField("unit velocity","face","rgba32float"),outputs=[0,1].map(i=>storage.createField(`forced ${i}`,"face","rgba32float"));
  const curves=[scalar("reference curvature"),scalar("retained curvature scratch")];
  const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
  const negatives=[0,1,2].map(()=>device.createBuffer({size:4*address.lowFaceKeys,usage}));
  const retainedScratch=device.createBuffer({size:UniformMixedForces.normalBytes(ownership.capacity.owners),usage});
  const referenceNormals=device.createBuffer({size:UniformMixedForces.normalBytes(ownership.capacity.owners),usage});
  const params=device.createBuffer({size:48,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(params,0,new Float32Array([1,1,1,.001,-8,1000,0,.072,0,0,0,0]));
  const coarse=device.createTexture({size:uniformCoarseCacheExtent(address),dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
  const resolve=new UniformMixedPhiResolve(device,ownership),resolveGroup=resolve.bind(phi);
  const control=new UniformSurfaceWorkControl(device,ownership,phi),geometry=new UniformMixedSurfaceGeometry(device,ownership,undefined,control);
  control.geometrySnapshots=snapshots;
  if(mixed)geometry.reserveNormals(0); // Deliberately force the production overflow branch with one fine tile.
  const reference=new UniformMixedForces(device,ownership,true,undefined,undefined,true),retained=new UniformMixedForces(device,ownership,true,undefined,undefined,true,true,true,bounded);
  assert.equal(geometry.normalBytesAt(1000000),16*(64*8192+ownership.capacity.tiles));
  let geometryGroup=geometry.bind(phi,target,center,pressure,volume);
  const fields={advected:velocity,phi,volume,centerPhi:center,coarseVelocity:coarse,unitVelocity:unit,negative:negatives[0]!,params};
  const referenceGroup=reference.bind({...fields,output:outputs[0]!,outputNegative:negatives[1]!,curvature:curves[0]!,normals:{buffer:referenceNormals}});
  let retainedGroup=retained.bind({...fields,output:outputs[1]!,outputNegative:negatives[2]!,curvature:curves[1]!,normals:{buffer:retainedScratch},retainedNormals:geometry.normalBinding,normalDirty:geometry.curvatureDirty});
  const writerParams=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const writerLayout=uniformDetailBindLayout(device,{entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]});
  const module=uniformDetailModule(device,{code:uniformMixedTopologyWGSL(ownership.capacity,0)+`
@group(1) @binding(0) var outputPhi:texture_storage_3d<r32float,write>;
@group(1) @binding(1) var<uniform> mode:vec4f;
@compute @workgroup_size(64) fn writePhi(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=35937u){return;}let q=umCorner(id.x,33u);let p=vec3f(q);
 var value=dot(p,vec3f(1,2,3))/16.0-3.0+mode.y+mode.z*p.x;
 if(mode.x>0.5){value=dot(p*p,vec3f(1,2,3))/1024.0-2.0+mode.y;}
 textureStore(outputPhi,vec3i(q),vec4f(value));}`});
  const writer=await uniformDetailPipeline(device,ownership,{layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,writerLayout]}),compute:{module,entryPoint:"writePhi"}});
  const writerGroup=uniformDetailGroup(device,{layout:writerLayout,entries:[{binding:0,resource:phi},{binding:1,resource:{buffer:writerParams}}]});
  let fixtureRevision=1;
  // Phi-only publication keeps exact topology invalidation independently testable.
  const publishPhi=()=>device.queue.writeBuffer(control.revisions.buffer,control.revisions.liveOffset,new Uint32Array(control.regions).fill(++fixtureRevision));
  const write=async(mode:number,offset:number,tilt=0)=>{device.queue.writeBuffer(writerParams,0,new Float32Array([mode,offset,tilt,0]));const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(uniformDetailPick(writer));p.setBindGroup(0,ownership.bindGroup);p.setBindGroup(1,writerGroup.group);p.dispatchWorkgroups(562);p.end();resolve.encode(e,resolveGroup);device.queue.submit([e.finish()]);publishPhi();};
  const geom=()=>{const e=device.createCommandEncoder();geometry.encode(e,geometryGroup,{pressure:true});device.queue.submit([e.finish()]);};
  const force=(cap=true)=>{const e=device.createCommandEncoder();reference.encode(e,referenceGroup,cap);retained.encode(e,retainedGroup,cap);device.queue.submit([e.finish()]);};
  const pending=async()=>new Uint32Array((await readMixedBuffer(device,geometry.curvatureDirty!)).buffer).slice(0,control.regions);
  const verify=async()=>{
   const normalBound=UNIFORM_NORMAL_ANCHOR_ERROR+64*2**-23,curvatureBound=12*normalBound,velocityBound=.001*(.072/1000)*2*curvatureBound+1e-8;
   const equal=(actual:number|undefined,expected:number|undefined,bound:number,label:string)=>{if(bounded)assert.ok(Math.abs(actual!-expected!)<=bound,label);else assert.equal(actual,expected,label);};
   const scratch=await readMixedBuffer(device,retainedScratch);
   const [actual,normalRef,curveRef,outA,outB]=await Promise.all([readMixedBuffer(device,geometry.normalBinding.buffer),readMixedBuffer(device,referenceNormals),storage.read(curves[0]!),storage.read(outputs[0]!),storage.read(outputs[1]!)]);
   const prefix=actual.length/4-ownership.capacity.tiles;
   // Cached owners consume persistent values directly. Only overflow owners
   // materialize borrowed normal/curvature scratch.
   for(const tile of [...layout.fineTiles,...layout.coarseTiles]){
    const fine=(layout.tiles[tile]!&0x80000000)!==0,base=layout.tiles[tile]!&0x3fffffff;
    for(let lane=0;lane<(fine?64:1);lane++){
     const owner=base+lane,index=fine?(owner<prefix?owner:-1):prefix+tile;
     const values=index<0?scratch:actual,at=4*(index<0?owner:index);
     for(let axis=0;axis<3;axis++)equal(values[at+axis],normalRef[4*owner+axis],normalBound,`effective normal ${tile}/${lane}/${axis}`);
    }
   }
   for(const tile of layout.coarseTiles){const i=layout.tiles[tile]!&0x3fffffff;const p=uniformTileCoord(address,tile,[0,0,0]).map(v=>4*v);
    for(let axis=0;axis<3;axis++)equal(actual[4*(geometry.normalBinding.buffer.size/16-ownership.capacity.tiles+tile)+axis],normalRef[4*i+axis],normalBound,`normal ${tile}/${axis}`);
    equal(actual[4*(geometry.normalBinding.buffer.size/16-ownership.capacity.tiles+tile)+3],curveRef[p[0]!+32*(p[1]!+32*p[2]!)],curvatureBound,`curvature ${tile}`);
   }
   const curvatureActual=await storage.read(curves[1]!);
   for(const tile of layout.fineTiles){const base=layout.tiles[tile]!&0x3fffffff,p=uniformTileCoord(address,tile,[0,0,0]).map(v=>4*v);
    for(let lane=0;lane<64;lane++){
     const owner=base+lane,texel=p[0]!+(lane&3)+32*(p[1]!+((lane>>2)&3)+32*(p[2]!+(lane>>4)));
     equal(owner<prefix?actual[4*owner+3]:curvatureActual[texel],curveRef[texel],curvatureBound,`effective curvature ${tile}/${lane}`);
    }
   }
   for(let i=0;i<outA.length;i++)equal(outB[i],outA[i],velocityBound,`forced velocity ${i}`);
  };
  try{
   await Promise.all([resolve.initialize(),control.initialize(),geometry.initialize(),reference.initialize(),retained.initialize()]);
   await write(1,0);geom();assert.ok((await pending()).some(Boolean));force(false);assert.ok((await pending()).some(Boolean),"tension disabled retains pending curvature");
   force();await verify();assert.ok((await pending()).every(v=>v===0),"curvature consumption retires dirty evidence");
   await write(0,0);geom();force();await verify();
   await write(0,.25);geom();force();await verify();
   const stats=new Uint32Array((await readMixedBuffer(device,geometry.curvatureDirty!)).buffer);if(!mixed)assert.equal(stats[4*control.regions],0,"translated plane launches no dirty curvature regions");else assert.ok(stats[4*control.regions]!>0,"overflow normal publication invalidates neighboring cached curvature");
   assert.equal(geometry.reserveNormals(1),true);geometryGroup=geometry.bind(phi,target,center,pressure,volume);
   retainedGroup=retained.bind({...fields,output:outputs[1]!,outputNegative:negatives[2]!,curvature:curves[1]!,normals:{buffer:retainedScratch},retainedNormals:geometry.normalBinding,normalDirty:geometry.curvatureDirty});
   geom();assert.ok((await pending()).some(Boolean),"resize forces curvature refresh even for copied unchanged normals");force();await verify();
   if(bounded){
    const anchor=await readMixedBuffer(device,geometry.normalBinding.buffer);
    for(let n=1;n<=4;n++){await write(0,.25,n*UNIFORM_NORMAL_ANCHOR_ERROR*.02);geom();force();}
    assert.deepEqual(await readMixedBuffer(device,geometry.normalBinding.buffer),anchor,"subthreshold normal changes retain the original anchor");
    let state=new Uint32Array((await readMixedBuffer(device,geometry.curvatureDirty!)).buffer);assert.equal(state[4*control.regions],0,"small moving-plane changes reuse curvature");
    const actual=await storage.read(outputs[1]!),expected=await storage.read(outputs[0]!);
    for(let i=0;i<actual.length;i++)assert.ok(Math.abs(actual[i]!-expected[i]!)<=.001*(.072/1000)*2*12*(UNIFORM_NORMAL_ANCHOR_ERROR+64*2**-23)+1e-8,`bounded forced velocity ${i}`);
    await write(0,.25,UNIFORM_NORMAL_ANCHOR_ERROR*.4);geom();force();state=new Uint32Array((await readMixedBuffer(device,geometry.curvatureDirty!)).buffer);
    assert.ok(state[4*control.regions]!>0,"accumulated drift eventually refreshes the retained anchor");
    device.queue.writeBuffer(params,0,new Float32Array([1,1,1,10,-8,1000,0,.072,0,0,0,0]));
    force();await verify(); // A tighter current displacement budget forces exact refresh even without new geometry.
    await verify(); // Persistent cache, plus overflow scratch, now holds exact refreshed normals.
    assert.deepEqual(await storage.read(outputs[1]!),await storage.read(outputs[0]!),"inadmissible current budget restores exact forced velocities");

   }
  }finally{geometry.destroy();control.destroy();storage.destroy();ownership.destroy();params.destroy();writerParams.destroy();referenceNormals.destroy();retainedScratch.destroy();coarse.destroy();negatives.forEach(b=>b.destroy());}
 });
});
