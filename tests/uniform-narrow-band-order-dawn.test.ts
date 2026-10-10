import assert from "node:assert/strict";
import test from "node:test";
import {UniformNarrowBandOrder} from "../lib/methods/uniform/uniform-narrow-band-order";
import {advanceUniform,withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer,readMixedTexture} from "./helpers/uniform-mixed-native-fields";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformNarrowBandMethod} from "../lib/methods/uniform/uniform-narrow-band-method";
import {UniformNarrowBandFlip} from "../lib/methods/uniform/uniform-narrow-band-flip";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";

for(const sparse of [false,true])(process.env.WEBGPU_NODE_MODULE?test:test.skip)(`NB ${sparse?"occupied-tile":"dense"} ordering clears stale cells and preserves every live sample`,async()=>{
 await withUniformDevice("NB sparse order",async device=>{
  const dims=[32,16,8],cells=4096,tiles=cells/64,capacity=1024;
  const buffer=(size:number)=>device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const particles=[buffer(capacity*48),buffer(capacity*48)] as const;
  const bins=buffer((2*cells+tiles)*4),links=buffer((cells+4*capacity)*4),state=buffer(48);
  const order=new UniformNarrowBandOrder(device,dims,particles,bins,links,state,sparse);
  const words=async(b:GPUBuffer)=>{const a=await readMixedBuffer(device,b);return new Uint32Array(a.buffer,a.byteOffset,a.length);};
  const cellOrder=(x:number,y:number,z:number)=>64*(Math.floor(x/4)+8*(Math.floor(y/4)+4*Math.floor(z/4)))+x%4+4*(y%4)+16*(z%4);
  try{
   await order.initialize();
   for(const [epoch,count] of [300,61,0,700].entries()){
    const clear=device.createCommandEncoder();order.prepare(clear,0);device.queue.submit([clear.finish()]);await device.queue.onSubmittedWorkDone();
    assert.ok((await words(bins)).slice(0,sparse?2*cells+tiles:2*cells).every(v=>v===0),`epoch ${epoch} cleared prior occupied cells and cursors`);
    const input=new Float32Array(capacity*12),counts=new Uint32Array(2*cells+tiles),expected=new Map<number,number[]>();
    for(let i=0;i<count;i++){
     const x=epoch===0?i%4:epoch===1?28+i%4:(i*13)%32;
     const y=(i*7)%16,z=(i*3)%8,live=i%11!==0;
     input.set([live?x+0.25:-1,y+0.25,z+0.25,1,i,epoch,0,0,0,0,0,0],i*12);
     if(live){const cell=cellOrder(x,y,z);counts[cell]++;counts[2*cells+Math.floor(cell/64)]++;const list=expected.get(cell)??[];list.push(i);expected.set(cell,list);}
    }
    // Transfer borrows expired cursors; each newly occupied cell must reset
    // its cursor in the prefix pass before any particle scatters.
    if(sparse)for(const cell of expected.keys())counts[cells+cell]=12345;
    device.queue.writeBuffer(particles[1],0,input);device.queue.writeBuffer(bins,0,counts);device.queue.writeBuffer(state,0,new Uint32Array([count,0,0,0]));
    const e=device.createCommandEncoder();order.encode(e,0);device.queue.submit([e.finish()]);
    const receipt=await words(state),starts=await words(links),output=await readMixedBuffer(device,particles[0]);
    assert.equal(receipt[1],[...expected.values()].reduce((n,a)=>n+a.length,0));
    let offset=0;
    for(let cell=0;cell<cells;cell++){
     const ids=expected.get(cell)??[];
     if(!sparse||counts[2*cells+Math.floor(cell/64)]!==0)assert.equal(starts[cell],offset,`epoch ${epoch} cell ${cell} offset`);
     const actual=Array.from({length:ids.length},(_,j)=>output[12*(offset+j)+4]).sort((a,b)=>a-b);
     assert.deepEqual(actual,ids.sort((a,b)=>a-b),`epoch ${epoch} cell ${cell} sample identities`);
     offset+=ids.length;
    }
   }
  }finally{order.destroy();for(const b of [...particles,bins,links,state])b.destroy();}
 });
});


/** Replace only the ordering policy before production shaders are generated.
 * Restore the prototype immediately on entry, including on creation failure. */
async function createOrderPolicySolver(device:GPUDevice,moving:boolean,sparse:boolean){
 const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
 Object.assign(scene.container,{width_m:1,height_m:1,depth_m:1,fillFraction:moving?0:0.5,top:"closed",fluidWallMode:"free-slip"});
 scene.voxelDomain.finestCellSize_m=1/32;scene.rigidBodies=[];scene.solidVoxels=[];
 // Binary velocity and dt prescribe exactly 2h per step. Independent dense
 // runs can resample different particles, so compare geometry across policies
 // and validate production bin counts exactly against each actual move input.
 scene.numerics={...scene.numerics,fixedDt_s:0.0625,maxDt_s:0.0625};
 Object.assign(scene.fluid,{initialCondition:"tank-fill",initialLiquidVolumes:moving?[{shape:"sphere",center_m:{x:-0.2,y:0.6,z:0},radius_m:0.12}]:[],
  initialVelocity_m_s:{x:moving?1:0,y:0,z:0},gravity_m_s2:{x:0,y:0,z:0},inflow:undefined,surfaceTension_N_m:0,dynamicViscosity_Pa_s:0,
  initialBrickSeeds_m:undefined,initialHeightField:undefined});
 const initialize=UniformNarrowBandFlip.prototype.initialize;let replaced=false;
 if(sparse)UniformNarrowBandFlip.prototype.initialize=function(){
  UniformNarrowBandFlip.prototype.initialize=initialize;
  const stage=this as unknown as {order:UniformNarrowBandOrder;device:GPUDevice;ownership:{capacity:{lattice:{dimensions:readonly number[]}}};bins:GPUBuffer;next:GPUBuffer};
  stage.order.destroy();
  stage.order=new UniformNarrowBandOrder(stage.device,stage.ownership.capacity.lattice.dimensions,this.particles,stage.bins,stage.next,this.state,true);
  replaced=true;return initialize.call(this);
 };
 try{
  // Transport alone: volume control answers a drop's volume error with a divergence, which is a velocity.
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{timeStep:"scene",detailPolicy:"full",volumeControlSeconds:0},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  if(sparse)assert.ok(replaced,"force sparse before compiling the real move and gather stages");
  return solver;
 }finally{UniformNarrowBandFlip.prototype.initialize=initialize;}
}

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("NB production sparse bins count real moves and preserve rest and translated geometry",{timeout:180_000},async()=>{
 await withUniformDevice("NB sparse production integration",async device=>{
  for(const moving of [false,true]){
   const results:{particles:number;volume:number;centroid:number;volumeCentroid:number}[][]=[];
   for(const sparse of [false,true]){
    const solver=await createOrderPolicySolver(device,moving,sparse);
    const snapshots:GPUBuffer[]=[];
    try{
     const frame=(solver as unknown as {mixedFrame:{narrowBandFlip:UniformNarrowBandFlip;fields:{phi:GPUTexture;volume:GPUTexture;velocity:GPUTexture}}}).mixedFrame;
     const stage=frame.narrowBandFlip,internals=stage as unknown as {order:UniformNarrowBandOrder;bins:GPUBuffer};
     assert.equal(internals.order.sparse,sparse);
     const snapshot=(size:number)=>{const b=device.createBuffer({size,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});snapshots.push(b);return b;};
     const moved=snapshot(stage.particles[0].size),moveBins=snapshot((2*32**3+512)*4),receipt=snapshot(stage.state.size);
     const encode=internals.order.encode.bind(internals.order);
     // Observe production movement and sorting before resampling mutates counts.
     // The solver still executes its original advection, ordering and gathers.
     internals.order.encode=(encoder,parity)=>{
      encoder.copyBufferToBuffer(stage.particles[1-parity]!,0,moved,0,moved.size);
      encode(encoder,parity);
      encoder.copyBufferToBuffer(internals.bins,0,moveBins,0,moveBins.size);
      encoder.copyBufferToBuffer(stage.state,0,receipt,0,receipt.size);
     };
     if(moving){const velocity=new Float32Array(32**3*4);for(let i=0;i<32**3;i++)velocity[4*i]=1;solver.initializeVelocityForQA(velocity);}
     const rows:{particles:number;volume:number;centroid:number;volumeCentroid:number}[]=[];let previousCount=0,previousTiles=new Set<number>(),vacated=0;
     for(let step=1;step<=4;step++){
      await advanceUniform(solver,step*0.0625);
      const samples=(await readMixedBuffer(device,stage.activeParticles)).subarray(0,stage.count*12);
      const [volume,phi,velocity]=await Promise.all([readMixedTexture(device,frame.fields.volume),readMixedTexture(device,frame.fields.phi),readMixedTexture(device,frame.fields.velocity)]);
      assert.ok(samples.every(Number.isFinite)&&volume.every(Number.isFinite)&&phi.every(Number.isFinite)&&velocity.every(Number.isFinite));
      assert.ok(stage.count>0);let centroid=0,speedError=0;
      for(let i=0;i<stage.count;i++){
       centroid+=samples[12*i]!;speedError=Math.max(speedError,Math.abs(samples[12*i+4]!-(moving?1:0)),Math.abs(samples[12*i+5]!),Math.abs(samples[12*i+6]!));
      }
      centroid/=stage.count;const total=volume.reduce((a,b)=>a+b,0);
      const volumeCentroid=volume.reduce((a,v,i)=>a+v*(i%32+0.5),0)/total;
      assert.ok(speedError<1e-3,`unchanged constant-velocity tolerance: ${speedError}`);
      assert.ok(stage.diagnostics.afterMaxOutside<0.5,"particles remain on the reconstructed surface");
      if(moving){assert.ok(Math.abs(centroid-(rows[0]?.centroid??centroid)-(step-1)*2)<0.25,"unchanged multi-cell translation bound");}
      else{
       assert.equal(stage.count,32*32*4*8,"all resting samples survive each epoch");
       assert.ok(Math.abs(total/16384-1)<1e-5,"unchanged resting-volume bound");
       for(let z=0;z<=32;z++)for(let x=0;x<=32;x++)assert.ok(Math.abs(phi[x+33*(16+33*z)]!)<1e-5,"unchanged resting-plane bound");
      }
      const [input,binData,receiptData]=await Promise.all([readMixedBuffer(device,moved),readMixedBuffer(device,moveBins),readMixedBuffer(device,receipt)]);
      const bins=new Uint32Array(binData.buffer,binData.byteOffset,binData.length),counts=new Uint32Array(receiptData.buffer,receiptData.byteOffset,receiptData.length);
      const expected=new Uint32Array(32**3),tileCounts=new Uint32Array(512),occupied=new Set<number>();let live=0;
      assert.ok(input.subarray(0,counts[0]!*12).every(Number.isFinite));
      for(let i=0;i<counts[0]!;i++){
       if(input[12*i]!<0)continue;
       const x=Math.floor(input[12*i]!),y=Math.floor(input[12*i+1]!),z=Math.floor(input[12*i+2]!);
       assert.ok(x<32&&y>=0&&y<32&&z>=0&&z<32,"moved samples remain in the closed domain");
       const tile=Math.floor(x/4)+8*(Math.floor(y/4)+8*Math.floor(z/4));
       expected[64*tile+x%4+4*(y%4)+16*(z%4)]++;tileCounts[tile]++;live++;
      }
      assert.equal(counts[1],live,"production sort preserves every advected live sample exactly");
      if(step>1)assert.equal(live,previousCount,"production advection counts each prior live sample once");
      for(let cell=0;cell<32**3;cell++){
       assert.equal(bins[cell],expected[cell],"production cell counter matches actual advected input");
       assert.equal(bins[32**3+cell],expected[cell],"scatter cursors match counts, including vacated cells");
      }
      for(let tile=0;tile<512;tile++){
       if(sparse)assert.equal(bins[2*32**3+tile],tileCounts[tile],"production tile atomic counts match actual input");
       if(tileCounts[tile])occupied.add(tile);
      }
      for(const tile of previousTiles)if(!occupied.has(tile))vacated++;
      previousTiles=occupied;
      rows.push({particles:stage.count,volume:total,centroid,volumeCentroid});previousCount=stage.count;
     }
     if(sparse&&moving)assert.ok(vacated>0,"translation actually leaves previously occupied tiles empty");
     results.push(rows);console.log(JSON.stringify({moving,sparse,vacated,rows}));
    }finally{for(const b of snapshots)b.destroy();solver.destroy();}
   }
   for(let step=0;step<4;step++){
    const dense=results[0]![step]!,sparse=results[1]![step]!;
    if(!moving)assert.equal(sparse.particles,dense.particles,"ordering policy preserves all resting particles");
    assert.ok(Math.abs(sparse.volume/dense.volume-1)<1e-5,"ordering policy preserves geometric volume within the existing rest bound");
    assert.ok(Math.abs(sparse.volumeCentroid-dense.volumeCentroid)<1e-5,"ordering policy preserves the geometric volume centroid");
   }
  }
 });
});
