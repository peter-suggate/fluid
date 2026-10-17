import assert from "node:assert/strict";
import test from "node:test";
import { UNIFORM_SURFACE_SUPPORT_WORDS as S } from "../lib/methods/uniform/uniform-surface-transport.wgsl";
import { UniformSurfaceWorkControl } from "../lib/methods/uniform/uniform-surface-work-control";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { createUniformMixedLayout, mixedCellWidth } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedPhiResolve } from "../lib/methods/uniform/uniform-mixed-phi-resolve";
import { uniformLatticeAddress, uniformTileCoord } from "../lib/methods/uniform/uniform-lattice-address";
import { UniformDetailStorage, uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline, uniformDetailPick } from "../lib/methods/uniform/uniform-detail-fields";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { UniformPagePool } from "../lib/methods/uniform/uniform-page-pool";
import { withUniformDevice } from "./helpers/uniform-geometric";

const gpu={skip:!process.env.WEBGPU_NODE_MODULE};
async function words(device:GPUDevice,buffer:GPUBuffer,count:number,offset=0){
 const read=device.createBuffer({size:4*count,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 try{const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,offset,read,0,4*count);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);return new Uint32Array(read.getMappedRange()).slice();}
 finally{if(read.mapState==="mapped")read.unmap();read.destroy();}
}

test("production 16h current models and retained geometry handle coverage, collisions, and page claims",gpu,async()=>{
 await withUniformDevice("16h evidence regression",async device=>{
  // 75 stored blocks give 600 live regions: collisions exceed the bounded
  // 512-region cache while the all-coarse phi backing stays below 0.3 MB.
  const dimensions:[number,number,number]=[160,160,96];
  const address=uniformLatticeAddress(dimensions);
  const layout=createUniformMixedLayout({dimensions,cellSize_m:[1,1,1],origin_m:{x:0,y:0,z:0}},[],4,undefined,address);
  const ownership=new UniformMixedOwnership(device,layout,true,0);
  const storage=new UniformDetailStorage(device,address);storage.attach(ownership);
  const phi=storage.createField("work evidence fixture","vertex","r32float");
  const pool=new UniformPagePool(device,{table:16,blocks:address.blocks,idleSteps:2});
  const control=new UniformSurfaceWorkControl(device,ownership,phi,undefined,pool);
  const texture=storage.physical(phi);
  let fixtureRevision=1;
  // These fixture writers change phi alone: publish producer revisions,
  // preserving the topology-vs-field distinction exercised below.
  const publishPhi=()=>device.queue.writeBuffer(control.revisions.buffer,control.revisions.liveOffset,new Uint32Array(control.regions).fill(++fixtureRevision));
  const fill=(value:number)=>{device.queue.writeTexture({texture},new Float32Array(texture.width*texture.height*texture.depthOrArrayLayers).fill(value),{bytesPerRow:4*texture.width,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);publishPhi();};
  const capture=async(stage:"surface"|"geometry"="surface")=>{
   const e=device.createCommandEncoder();control.encode(e,stage);device.queue.submit([e.finish()]);
   return words(device,stage==="surface"?control.surface:control.geometry,control.snapshotOffset);
  };
  try{
   await control.initialize();fill(1);
   device.queue.writeBuffer(ownership.regionVelocity,0,new Uint32Array(6*control.regions).fill(0x80000000));
   const first=await capture(),r=control.regions;
   assert.equal(control.cacheRegions,512);assert.ok(control.geometry.size<18*1024*1024);
   assert.equal(control.epochOffset,undefined);assert.equal(control.surface.size,4*(control.liveEpochWordOffset+7*control.regions+3*ownership.capacity.tiles),"surface bank retains producer revisions and compact identity metadata, not phi snapshots");
   const models=await words(device,control.surface,8*control.regions,4*control.modelWordOffset);
   assert.equal(models[8*256+5],0x7ffffff,"interior current model is valid");
   assert.equal(models[8*529+5],0x7ffffff,"current model remains valid outside temporal snapshot cache");
   const floats=new Float32Array(models.buffer);
   assert.equal(floats[8*256+3],1);assert.ok(floats[8*256+4]!<.00001,"constant field has only rounding allowance");
   assert.ok((models[8*8+5]!&(1<<13))!==0,"boundary model covers its stored home region");
   assert.ok(models[8*8+5]!==0x7ffffff,"outside neighbors are omitted from coverage rather than invalidating the model");
   const stable=await capture();assert.equal(stable[3*r],first[3*r]);
   for(const value of [1.0004,1.0008,1.0012]){
    fill(value);await capture();const model=new Float32Array((await words(device,control.surface,8,4*(control.modelWordOffset+8*256))).buffer);
    assert.ok(Math.abs(model[3]!-Math.fround(value))<1e-6,"current model refreshes even for subthreshold motion");
   }
   // A real production field writer changes one canonical coarse corner.
   const resources=uniformDetailBindLayout(device,{entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}}]});
   const module=uniformDetailModule(device,{code:uniformMixedTopologyWGSL(ownership.capacity,0)+`
@group(1) @binding(0) var outputPhi:texture_storage_3d<r32float,write>;
@compute @workgroup_size(1) fn change(){textureStore(outputPhi,vec3i(8),vec4f(1.02));}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
   const pipeline=await uniformDetailPipeline(device,ownership,{layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,resources]}),compute:{module,entryPoint:"change"}});
   const group=uniformDetailGroup(device,{layout:resources,entries:[{binding:0,resource:phi}]});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(uniformDetailPick(pipeline));p.setBindGroup(0,ownership.bindGroup);p.setBindGroup(1,group.group);p.dispatchWorkgroups(1);p.end();device.queue.submit([e.finish()]);publishPhi();
   await capture();
   const localModels=new Float32Array((await words(device,control.surface,8*control.regions,4*control.modelWordOffset)).buffer);
   assert.ok(localModels[8*8+6]!>.018,"edited region current residual refreshes");
   assert.ok(localModels[8*9+4]!>.018,"neighbor halo includes edited region residual");
   assert.ok(localModels[8*80+4]!<.0001,"remote current model remains affine");
   const exactFirst=await capture("geometry"),exactStable=await capture("geometry");
   assert.equal(exactStable[8],0,"unchanged exact geometry is clean");
   assert.equal((await words(device,control.geometry,1,4*(control.topologyWordOffset+520)))[0],1,"uncached geometry conservatively invalidates topology-dependent curvature");
   fill(1.00121);const exactChanged=await capture("geometry");assert.equal(exactChanged[8],1,"geometry notices small current phi changes");
   assert.equal(exactFirst[8],1);
   // Supply the production page-claim hook a one-block new-list fixture.
   // No phi/topology change: the metadata invalidation alone must refresh.
   await capture("geometry");
   const mapped=(await words(device,control.geometry,1,4*(control.mappingWordOffset+8)))[0]!;
   const marker=control.snapshotOffset+132*(64*(mapped-1))+131;
   assert.equal((await words(device,control.geometry,1,4*marker))[0],1);
   device.queue.writeBuffer(pool.state,8,new Uint32Array([1]));
   device.queue.writeBuffer(pool.state,4*pool.layout.state.claimed,new Uint32Array([1]));
   const claim=device.createCommandEncoder();control.encodeInitBlocks(claim);device.queue.submit([claim.finish()]);
   assert.equal((await words(device,control.geometry,1,4*marker))[0],0,"new page clears cached live marker");
   const after=await capture("geometry");
   assert.equal(after[8],1,"new page metadata forces fresh geometry certificate");
   control.enabled=false;await capture();
   assert.equal((await words(device,control.surface,1,4*(control.modelWordOffset+8*256+5)))[0],0,"diagnostic full mode disables current model");
   const disabledSupports=await words(device,control.surface,64*S,4*(control.supportWordOffset+64*S*256));
   for(let box=0;box<64;box++)assert.equal(disabledSupports[S*box+7],0,"disabled control cannot certify any support");
   control.enabled=true;
   const affine=uniformDetailModule(device,{code:uniformMixedTopologyWGSL(ownership.capacity,0)+`
@group(1) @binding(0) var outputPhi:texture_storage_3d<r32float,write>;
@compute @workgroup_size(64) fn affine(@builtin(global_invocation_id) id:vec3u){
 let dims=vec3u(41u,41u,25u);if(id.x>=dims.x*dims.y*dims.z){return;}
 let q=4u*vec3u(id.x%dims.x,(id.x/dims.x)%dims.y,id.x/(dims.x*dims.y));
 textureStore(outputPhi,vec3i(q),vec4f(dot(vec3f(.1,.2,-.3),vec3f(q))+7.0));
}
@compute @workgroup_size(1) fn warp(){textureStore(outputPhi,vec3i(40),vec4f(7.01));}`});
   assert.deepEqual((await affine.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
   const affineLayout=device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,resources]});
   const apply=async(entryPoint:string,jobs:number)=>{
    const pipeline=await uniformDetailPipeline(device,ownership,{layout:affineLayout,compute:{module:affine,entryPoint}});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(uniformDetailPick(pipeline));pass.setBindGroup(0,ownership.bindGroup);pass.setBindGroup(1,group.group);pass.dispatchWorkgroups(jobs);pass.end();device.queue.submit([encoder.finish()]);publishPhi();await capture();
   };
   await apply("affine",Math.ceil(41*41*25/64));
   const planeWords=await words(device,control.surface,8,4*(control.modelWordOffset+8*256));const plane=new Float32Array(planeWords.buffer);
   assert.equal(planeWords[5],0x7ffffff);
   [.1,.2,-.3].forEach((g,a)=>assert.ok(Math.abs(plane[a]!-g)<1e-6,`gradient axis ${a}: ${plane[a]}`));
   assert.ok(Math.abs(plane[3]!-7)<1e-5);assert.ok(plane[4]!<.001,"affine halo residual is rounding only");
   const planeSlopes=new Float32Array((await words(device,control.surface,3,4*(control.slopeWordOffset+3*256))).buffer);
   [.1,.2,.3].forEach((g,a)=>{assert.ok(planeSlopes[a]!>=g);assert.ok(planeSlopes[a]!<g+.00002,`native slope axis ${a}: ${planeSlopes[a]}`);});
   await apply("warp",1);
   const warped=new Float32Array((await words(device,control.surface,8,4*(control.modelWordOffset+8*256))).buffer);
   assert.ok(warped[4]!>=.0099,"interior departure from the corner fit contributes to halo residual");
   const constant=new Float32Array([.1,-.23,4.125]);
   const order=(value:number)=>{const bits=new Uint32Array(new Float32Array([value]).buffer)[0]!;return (bits&0x80000000)?(~bits)>>>0:(bits^0x80000000)>>>0;};
   const encoded=new Uint32Array(6*control.regions);
   for(let region=0;region<control.regions;region++)for(let axis=0;axis<3;axis++){encoded[6*region+axis]=order(constant[axis]!);encoded[6*region+3+axis]=order(-constant[axis]!);}
   device.queue.writeBuffer(ownership.regionVelocity,0,encoded);await capture();
   const interval=new Float32Array((await words(device,control.surface,6,4*(control.velocityWordOffset+6*256))).buffer);
   for(let axis=0;axis<3;axis++){
    assert.ok(interval[axis]!<constant[axis]!&&interval[axis+3]!>constant[axis]!,`constant interval expands axis ${axis}`);
    const expansion=Math.max(constant[axis]!-interval[axis]!,interval[axis+3]!-constant[axis]!);
    assert.ok(expansion<=Math.abs(constant[axis]!)*.000008,"published rounding allowance stays small");
   }
   assert.deepEqual(await words(device,ownership.regionVelocity,6),encoded.slice(0,6),"raw face extrema remain exact");
   const local=new Float32Array((await words(device,control.surface,6,4*(control.localVelocityWordOffset+6*256))).buffer);
   assert.deepEqual(Array.from(local),[...constant,...constant],"local extrema do not inherit halo padding");
   // Region 257 is adjacent to 256. Its outlier must enter the broad halo
   // but remain absent from region 256's local record.
   encoded[6*257]=order(100);device.queue.writeBuffer(ownership.regionVelocity,0,encoded);await capture();
   const haloOutlier=new Float32Array((await words(device,control.surface,6,4*(control.velocityWordOffset+6*256))).buffer);
   const localWithoutOutlier=new Float32Array((await words(device,control.surface,6,4*(control.localVelocityWordOffset+6*256))).buffer);
   assert.ok(haloOutlier[3]!>=100,"broad halo includes nearby splash speed");
   assert.deepEqual(Array.from(localWithoutOutlier),Array.from(local),"unrelated neighbor does not contaminate local source bounds");
   // Independently reconstruct every rectangular support from current local
   // records and seeded physical page coordinates, including boundary boxes.
   const allSlopes=new Float32Array((await words(device,control.surface,3*r,4*control.slopeWordOffset)).buffer);
   const allModelWords=await words(device,control.surface,8*r,4*control.modelWordOffset),allModels=new Float32Array(allModelWords.buffer);
   const coordinates=new Map<string,number>(),origins=new Map<number,number[]>();
   for(let region=0;region<r;region++){
    if(allModelWords[8*region+7]!==1)continue;
    const local=region&7,tile=512*(region>>3)+4*(local&1)+32*((local>>1)&1)+256*((local>>2)&1);
    const origin=uniformTileCoord(address,tile,[0,0,0]).map(v=>4*v);origins.set(region,origin);coordinates.set(origin.join(","),region);
   }
   const contains=(code:number,delta:number)=>delta===0||(delta<0&&(code&1)!==0)||(delta>0&&(code&2)!==0);
   for(const region of [8,256,529]){
    const records=await words(device,control.surface,64*S,4*(control.supportWordOffset+64*S*region)),values=new Float32Array(records.buffer),origin=origins.get(region)!;
    for(let box=0;box<64;box++){
     const codes=[box&3,(box>>2)&3,(box>>4)&3];let valid=true,rho=0;const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity],slopes=[0,0,0];
     for(let z=-1;z<=1;z++)for(let y=-1;y<=1;y++)for(let x=-1;x<=1;x++){
      const delta=[x,y,z];if(!delta.every((d,a)=>contains(codes[a]!,d)))continue;
      const neighbor=coordinates.get(origin.map((v,a)=>v+16*delta[a]!).join(","));if(neighbor===undefined){valid=false;continue;}
      const at=8*neighbor,home=8*region,g=Array.from(allModels.slice(home,home+3)),ng=Array.from(allModels.slice(at,at+3));
      rho=Math.max(rho,allModels[at+6]!+Math.abs(allModels[at+3]!-allModels[home+3]!-16*g.reduce((sum,v,a)=>sum+v*delta[a]!,0))+8*g.reduce((sum,v,a)=>sum+Math.abs(ng[a]!-v),0));
      for(let axis=0;axis<3;axis++){const nlo=constant[axis]!,nhi=neighbor===257&&axis===0?100:constant[axis]!;lo[axis]=Math.min(lo[axis]!,nlo);hi[axis]=Math.max(hi[axis]!,nhi);slopes[axis]=Math.max(slopes[axis]!,allSlopes[3*neighbor+axis]!);}
     }
     assert.equal(records[S*box+7]!&1,Number(valid),`support validity ${region}/${box}`);if(!valid)continue;
     assert.equal(records[S*box+7]!&2,2,"a scene without solids certifies clear stored supports");
     const expectedRho=rho+32*2**-23*(Math.abs(allModels[8*region+3]!)+32*Array.from(allModels.slice(8*region,8*region+3)).reduce((sum,v)=>sum+Math.abs(v),0)+rho);
     assert.ok(values[S*box]!>=rho,`support rho encloses exact translated residual ${region}/${box}`);
     assert.ok(Math.abs(values[S*box]!-expectedRho)<=2e-5*Math.max(1,expectedRho),`support rho agrees with independent reduction ${region}/${box}`);
     for(let axis=0;axis<3;axis++){
      assert.equal(values[S*box+8+axis],slopes[axis],`support native-edge slope union ${region}/${box}/${axis}`);
      const padding=64*2**-23*Math.max(Math.abs(lo[axis]!),Math.abs(hi[axis]!));
      assert.ok(values[S*box+1+axis]!<=lo[axis]!&&values[S*box+4+axis]!>=hi[axis]!,`support interval outward ${region}/${box}/${axis}`);
      assert.ok(Math.abs(values[S*box+1+axis]!-(lo[axis]!-padding))<=2e-6*Math.max(1,Math.abs(lo[axis]!)));
      assert.ok(Math.abs(values[S*box+4+axis]!-(hi[axis]!+padding))<=2e-6*Math.max(1,Math.abs(hi[axis]!)));
     }
    }
    for(const reach of [.5,6,8.5,14])for(const px of [0,8,16])for(const py of [0,8,16])for(const pz of [0,8,16]){
     const low=[px,py,pz].map(v=>Math.floor((v-reach)/16)),high=[px,py,pz].map(v=>Math.floor((v+reach)/16));
     if(low.some(v=>v< -1)||high.some(v=>v>1))continue;
     const codes=low.map((v,a)=>(v<0?1:0)|(high[a]!>0?2:0)),box=codes[0]!+4*codes[1]!+16*codes[2]!;let needed=0;
     for(let z=low[2]!;z<=high[2]!;z++)for(let y=low[1]!;y<=high[1]!;y++)for(let x=low[0]!;x<=high[0]!;x++)needed|=1<<(x+1+3*(y+1)+9*(z+1));
     assert.equal(records[S*box+7]!&1,Number((allModelWords[8*region+5]!&needed)===needed),"footprint rectangle maps to its strict coverage certificate");
    }
   }
   encoded[6*256]=0xffffffff;device.queue.writeBuffer(ownership.regionVelocity,0,encoded);await capture();
   const poisoned=new Float32Array((await words(device,control.surface,6,4*(control.localVelocityWordOffset+6*256))).buffer);
   assert.ok(poisoned[0]!< -1e38&&poisoned[3]!>1e38,"nonfinite source cannot certify a small local interval");
   const badSupports=await words(device,control.surface,64*S,4*(control.supportWordOffset+64*S*256));
   for(let box=0;box<64;box++)assert.equal(badSupports[S*box+7],0,"poisoned home invalidates every rectangle");
   const adjacentSupports=await words(device,control.surface,64*S,4*(control.supportWordOffset+64*S*257));
   assert.equal(adjacentSupports[7]!&1,1,"unaffected neighboring home-only support remains valid");
   assert.equal(adjacentSupports[S+7],0,"support including poisoned low neighbor is invalid");
   device.queue.writeBuffer(ownership.regionVelocity,0,new Uint32Array(6*control.regions).fill(0x80000000));await capture();
   const zero=new Float32Array((await words(device,control.surface,6,4*(control.velocityWordOffset+6*256))).buffer);
   assert.ok(zero.every(value=>value===0),"zero intervals remain exactly zero");
   // Physical allocation holes must not create direct-map conflicts when
   // fewer live regions remain than the bounded cache can hold.
   const mirror=ownership.pageMirror,blockAt=mirror.offset+address.seed.layout.tail.blocks;
   for(let block=2;block<address.blocks-1;block+=2)device.queue.writeBuffer(mirror.buffer,4*(blockAt+block),new Uint32Array([0xffffffff]));
   await capture();const retiredSupports=await words(device,control.surface,64*S,4*(control.supportWordOffset+64*S*16));
   for(let box=0;box<64;box++)assert.equal(retiredSupports[S*box+7],0,"retired physical region cannot expose stale valid support");
   await capture("geometry");const compact=await capture("geometry");
   assert.equal(compact[3*r+3],0,"compact ranks cache every live sparse region below capacity");
   assert.equal(compact[520],0,"previously colliding live region now reuses exact geometry");
   assert.ok((await words(device,control.geometry,1,4*(control.mappingWordOffset+520)))[0]!>0,"sparse high block has a recorded snapshot slot");
   control.geometrySnapshots=false;await capture("geometry");const current=await capture("geometry");
   assert.equal(current[520],1,"metadata-only mode always evaluates current geometry");
   assert.equal((await words(device,control.geometry,1,4*(control.topologyWordOffset+520)))[0],0,"unchanged metadata does not force exact normal anchors");
   const slot=(await words(device,control.geometry,1,4*(control.mappingWordOffset+520)))[0]!-1;
   const snapshot=control.snapshotOffset+132*64*slot;
   const held=await words(device,control.geometry,1,4*snapshot);fill(12);await capture("geometry");
   assert.deepEqual(await words(device,control.geometry,1,4*snapshot),held,"metadata-only mode does not read or overwrite retained phi");
   assert.equal((await words(device,control.geometry,1,4*(control.topologyWordOffset+520)))[0],0,"phi changes do not masquerade as topology");
   device.queue.writeBuffer(control.geometry,4*(snapshot+131),new Uint32Array([0]));await capture("geometry");
   assert.equal((await words(device,control.geometry,1,4*(control.topologyWordOffset+520)))[0],1,"page-claim marker invalidates metadata-only retained normals");
   control.geometrySnapshots=true;await capture("geometry");assert.equal((await capture("geometry"))[520],0,"reenabling snapshots rebuilds their current phi before reuse");
  }finally{control.destroy();pool.destroy();storage.destroy();ownership.destroy();}
 });
});

test("current slope evidence encloses native trilinear derivatives through resolved h/4h seams",gpu,async()=>{
 await withUniformDevice("native edge work bounds",async device=>{
  const dimensions:[number,number,number]=[32,32,32],address=uniformLatticeAddress(dimensions),forced=new Uint8Array(address.boxTiles);
  // A slab crosses all four lower/upper16h region boundaries in y and z.
  for(let tile=0;tile<forced.length;tile++)if((tile%8)>=3&&(tile%8)<=4)forced[tile]=1;
  const layout=createUniformMixedLayout({dimensions,cellSize_m:[1,1,1],origin_m:{x:0,y:0,z:0}},[],4,forced,address);
  const ownership=new UniformMixedOwnership(device,layout,true,layout.fineTiles.length),storage=new UniformDetailStorage(device,address);storage.attach(ownership);
  const phi=storage.createField("native edge phi","vertex","r32float"),control=new UniformSurfaceWorkControl(device,ownership,phi),resolve=new UniformMixedPhiResolve(device,ownership),resolved=resolve.bind(phi);
  try{
   await Promise.all([control.initialize(),resolve.initialize()]);
   const resources=uniformDetailBindLayout(device,{entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}}]});
   const module=uniformDetailModule(device,{code:uniformMixedTopologyWGSL(ownership.capacity,0)+`
@group(1) @binding(0) var outputPhi:texture_storage_3d<r32float,write>;
@compute @workgroup_size(64) fn writePhi(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=35937u){return;}let q=umCorner(id.x,33u);let p=vec3f(q);
 let value=.003*p.x*p.y-.001*p.z*p.z+.1*p.y+select(0.0,.025,(q.x+q.y+q.z)%3u==0u);
 textureStore(outputPhi,vec3i(q),vec4f(value));}`});
   const pipeline=await uniformDetailPipeline(device,ownership,{layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,resources]}),compute:{module,entryPoint:"writePhi"}}),group=uniformDetailGroup(device,{layout:resources,entries:[{binding:0,resource:phi}]});
   device.queue.writeBuffer(ownership.regionVelocity,0,new Uint32Array(6*control.regions).fill(0x80000000));
   const e=device.createCommandEncoder(),pass=e.beginComputePass();pass.setPipeline(uniformDetailPick(pipeline));pass.setBindGroup(0,ownership.bindGroup);pass.setBindGroup(1,group.group);pass.dispatchWorkgroups(562);pass.end();resolve.encode(e,resolved);control.invalidate();control.encode(e,"surface");device.queue.submit([e.finish()]);
   const values=await storage.read(phi),slopes=new Float32Array((await words(device,control.surface,3*control.regions,4*control.slopeWordOffset)).buffer);
   const v=(q:number[])=>values[q[0]!+33*(q[1]!+33*q[2]!)]!;let checked=0;
   for(const tile of [...layout.coarseTiles,...layout.fineTiles]){
    const width=mixedCellWidth(layout.tiles[tile]!),base=uniformTileCoord(address,tile,[0,0,0]).map(n=>4*n),region=8*(tile>>>9)+(base[0]!%32>=16?1:0)+(base[1]!%32>=16?2:0)+(base[2]!%32>=16?4:0);
    for(let z=0;z<4;z+=width)for(let y=0;y<4;y+=width)for(let x=0;x<4;x+=width){
     const origin=base.map((n,a)=>n+[x,y,z][a]!);
     for(let axis=0;axis<3;axis++)for(let side=0;side<4;side++){
      const lo=origin.slice();lo[(axis+1)%3]!+=(side&1)*width;lo[(axis+2)%3]!+=(side>>1)*width;
      const hi=lo.slice();hi[axis]!+=width;const slope=Math.abs(v(hi)-v(lo))/width;
      assert.ok(Number.isFinite(slope)&&slope<=slopes[3*region+axis]!,`native edge ${tile}/${origin}/${axis}: ${slope} <= ${slopes[3*region+axis]}`);checked++;
     }
    }
   }
   assert.ok(checked>100000,"exercise coarse, fine and seam edges with nonaffine current input");
  }finally{control.destroy();storage.destroy();ownership.destroy();}
 });
});
