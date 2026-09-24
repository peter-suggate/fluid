import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { WebGPUUniformReferenceSolver, type WebGPUUniformReferenceOptions } from "../lib/methods/uniform/webgpu-uniform-reference";

async function read(device:GPUDevice,texture:GPUTexture) {
  const channels=texture.format==="rgba32float"?4:1;
  const row=Math.ceil(texture.width*channels*4/256)*256;
  const staging=device.createBuffer({size:row*texture.height*texture.depthOrArrayLayers,
    usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try {
    const encoder=device.createCommandEncoder();
    encoder.copyTextureToBuffer({texture},{buffer:staging,bytesPerRow:row,rowsPerImage:texture.height},
      [texture.width,texture.height,texture.depthOrArrayLayers]);
    device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);
    const raw=new Float32Array(staging.getMappedRange()),result=new Float32Array(texture.width*texture.height*texture.depthOrArrayLayers*channels);
    for(let z=0;z<texture.depthOrArrayLayers;z++)for(let y=0;y<texture.height;y++)result.set(
      raw.subarray((z*texture.height+y)*row/4,(z*texture.height+y)*row/4+texture.width*channels),
      (z*texture.height+y)*texture.width*channels);
    return result;
  } finally {staging.unmap();staging.destroy();}
}
function write(device:GPUDevice,texture:GPUTexture,data:Float32Array) {
  const channels=texture.format==="rgba32float"?4:1;
  device.queue.writeTexture({texture},data as Float32Array<ArrayBuffer>,
    {bytesPerRow:texture.width*channels*4,rowsPerImage:texture.height},
    [texture.width,texture.height,texture.depthOrArrayLayers]);
}
process.env.FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT="1";
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("airborne momentum preserves free flight and gives overfilled air pressure support",{timeout:420_000},async t=>{
  await acquireWebGPUExclusiveLock("dawn-test","Uniform airborne momentum");
  let device:GPUDevice|undefined;
  try {
    const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
    const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);
    const adapter=await gpu.requestAdapter();assert.ok(adapter);
    device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
    const errors:string[]=[];
    device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
    async function fixture(fill:number,gravity:number,speed:number,airborne=true,originX=10,extra:Partial<WebGPUUniformReferenceOptions>={}) {
      const scene=sceneDocument(getSceneDefinition("symmetric-expansion"));
      scene.fluid.gravity_m_s2.y=gravity;
      const solver=await WebGPUUniformReferenceSolver.createAsync(device!,scene,"balanced",undefined,{
        geometricVolume:true,airborneMomentum:airborne,velocityTransport:"semi-lagrangian",
        densitySharpening:false,totalSurfaceVolume:false,volumeDustThreshold:0,
        solidExcessCorrection:false,twoLevelVelocity:false,activeRegion:false,pressureWindow:false,phiWindowForQA:false,...extra,
      },()=>{});
      const {nx,ny,nz}=solver.info;
      assert.deepEqual([nx,ny,nz],[32,16,32]);
      const volume=new Float32Array(nx*ny*nz),phi=new Float32Array((nx+1)*(ny+1)*(nz+1)).fill(.3);
      for(let z=14;z<18;z++)for(let y=8;y<12;y++)for(let x=originX;x<originX+4;x++)volume[x+nx*(y+ny*z)]=fill;
      write(device!,solver.volumeTexture,volume);write(device!,solver.vertexPhiTexture!,phi);
      const velocity=new Float32Array(volume.length*4);
      for(let i=0;i<volume.length;i++)velocity[4*i]=speed;
      solver.initializeVelocityForQA(velocity);
      return solver;
    }
    for(const fill of [.5,1])await t.test(`a cloud with fill ${fill} translates without gaining kinetic energy`,async()=>{
      const solver=await fixture(fill,0,1.5);
      try {
        for(let frame=1;frame<=6;frame++) {
          assert.ok(solver.advanceTo(frame/30));
          const volume=await read(device!,solver.volumeTexture),velocity=await read(device!,solver.velocityTexture);
          let mass=0;
          for(let z=0;z<32;z++)for(let y=0;y<16;y++)for(let x=0;x<32;x++) {
            const i=x+32*(y+16*z),expected=x>=10+frame&&x<14+frame&&y>=8&&y<12&&z>=14&&z<18 ? fill : 0;
            assert.ok(Math.abs(volume[i]!-expected)<2e-6,`frame ${frame}: translated volume at ${x},${y},${z}`);
            mass+=volume[i]!;
            if(expected>0) {
              assert.ok(Math.abs(velocity[4*i]!-1.5)<2e-5,"ballistic horizontal velocity");
              assert.ok(Math.abs(velocity[4*i+1]!)<2e-5&&Math.abs(velocity[4*i+2]!)<2e-5,"no transverse energy");
            }
          }
          assert.ok(Math.abs(mass-64*fill)<1e-5,"free flight conserves mass");
        }
      }finally{solver.destroy();}
    });
    await t.test("drained positive phi plateau retires without requiring volume",async()=>{
      const solver=await fixture(0,0,0,true,10,{phiDrain:true,volumeDustThreshold:.001});
      try{
        write(device!,solver.vertexPhiTexture!,new Float32Array(33*17*33).fill(.025));
        assert.ok(solver.advanceTo(1/30));
        const phi=await read(device!,solver.vertexPhiTexture!);
        for(const value of phi)assert.ok(value>=.2-1e-6,"unsupported positive plateau must leave the 4h seed band");
        assert.equal((await read(device!,solver.volumeTexture)).reduce((a,b)=>a+b,0),0);
      }finally{solver.destroy();}
    });
    await t.test("positive plateau beside a real contour retains its zero crossings",async()=>{
      const solver=await fixture(1,0,0,true,10,{phiDrain:true,volumeDustThreshold:.001});
      try{
        // A clipped distance field exercises zero-gradient air beside a body.
        const phi=Float32Array.from({length:33*17*33},(_,i)=>Math.min(.025,
          .05*(Math.max(Math.abs(i%33-12),Math.abs(Math.floor(i/33)%17-10),Math.abs(Math.floor(i/(33*17))-16))-1.5)));
        write(device!,solver.vertexPhiTexture!,phi);
        assert.ok(solver.advanceTo(1/30));
        const actual=await read(device!,solver.vertexPhiTexture!);
        let crossings=0;
        for(let z=0;z<32;z++)for(let y=0;y<16;y++)for(let x=0;x<32;x++){
          const i=x+33*(y+17*z);
          for(const stride of [1,33,33*17])if((phi[i]!<0)!==(phi[i+stride]!<0)){
            assert.ok(Math.abs(-phi[i]!/(phi[i+stride]!-phi[i]!)+actual[i]!/(actual[i+stride]!-actual[i]!))<1e-5,"cleanup preserves contour");crossings++;
          }
        }
        assert.ok(crossings>20);
      }finally{solver.destroy();}
    });
    for(const originX of [1,10])await t.test(`dilute orphan dust clears at x=${originX}`,async()=>{
      const solver=await fixture(.005,0,0,true,originX,{phiDrain:false,volumeDustThreshold:.001,orphanDustThreshold:.01});
      try{
        assert.ok(solver.advanceTo(1/30));
        const volume=await read(device!,solver.volumeTexture);
        assert.equal(volume.reduce((a,b)=>a+b,0),0,"remote sub-threshold haze is removed, including wall residue");
        const stats=await solver.readStats();
        assert.equal(stats.uniformVolumeOrphanDustCells,64);
        assert.ok(stats.uniformVolumeOrphanDustMass_cells!>=.31&&stats.uniformVolumeOrphanDustMass_cells!<=.32,
          "orphan mass diagnostics use their own threshold units");
      }finally{solver.destroy();}
    });
    await t.test("orphan cleanup preserves compact sub-cell droplets",async()=>{
      const solver=await fixture(.04,0,0,true,10,{phiDrain:false,volumeDustThreshold:.001,orphanDustThreshold:.05});
      try{
        assert.ok(solver.advanceTo(1/30));
        const volume=await read(device!,solver.volumeTexture);
        assert.ok(Math.abs(volume.reduce((a,b)=>a+b,0)-64*.04)<1e-5,"quarter-cell neighbourhood protects droplets below airborne threshold");
      }finally{solver.destroy();}
    });
    await t.test("orphan floor leaves a resting surface tail unchanged",async()=>{
      const outputs:Float32Array[]=[];
      for(const orphanDustThreshold of [0,.01]){
        const solver=await fixture(0,0,0,true,10,{phiDrain:false,volumeDustThreshold:.001,orphanDustThreshold});
        try{
          const phi=Float32Array.from({length:33*17*33},(_,i)=>.05*((Math.floor(i/33)%17)-7));
          const volume=Float32Array.from({length:32*16*32},(_,i)=>{
            const y=Math.floor(i/32)%16;return y<7?1:y===8?.005:0;
          });
          write(device!,solver.vertexPhiTexture!,phi);write(device!,solver.volumeTexture,volume);
          for(let frame=1;frame<=3;frame++)assert.ok(solver.advanceTo(frame/30));
          outputs.push(await read(device!,solver.volumeTexture));
        }finally{solver.destroy();}
      }
      assert.deepEqual(outputs[1],outputs[0],"stronger floor must not erode a resting interface or its near-surface tail");
    });
    await t.test("gravity accelerates the cloud once per step",async()=>{
      const solver=await fixture(.5,-9.80665,0);
      try {
        for(let frame=1;frame<=6;frame++) {
          assert.ok(solver.advanceTo(frame/30));
          const volume=await read(device!,solver.volumeTexture),velocity=await read(device!,solver.velocityTexture);
          let count=0;
          for(let i=0;i<volume.length;i++)if(volume[i]!>.1) {
            assert.ok(Math.abs(velocity[4*i+1]!+frame*9.80665/30)<2e-4,`frame ${frame}: one gravity impulse, cell ${i}, speed ${velocity[4*i+1]}`);
            count++;
          }
          assert.ok(count>=32,"cloud remains represented");
        }
      }finally{solver.destroy();}
    });
    await t.test("compressed liquid retains gravity inside ballistic wall clearance",async()=>{
      const solver=await fixture(2,-9.80665,0,true,1);
      try {
        assert.ok(solver.advanceTo(1/30));
        const forced=await read(device!,solver.symmetryStageAuditTextures!.velocityAdvection);
        const i=1+32*(9+16*15);
        assert.ok(Math.abs(forced[4*i+1]!+9.80665/30)<2e-5,"pressure-supported material gets gravity even where ballistic clearance is false");
      }finally{solver.destroy();}
    });
    for(const airborne of [false,true])await t.test(`overfilled phi-air pressure ownership, airborne ${airborne?"on":"off"}`,async()=>{
      const solver=await fixture(2,0,0,airborne);
      try {
        assert.ok(solver.advanceTo(1/30));
        const pressure=await read(device!,solver.physicsFieldsForQA.pressure);
        const [nx,ny]=solver.physicsFieldsForQA.latticeDimensions;
        const centre=pressure[12+1+nx*(10+1+ny*(16+1))]!;
        if(airborne)assert.ok(centre>1,"overfilled material must participate in pressure");
        else assert.equal(centre,0,"airborne-off pressure topology is unchanged");
        const volume=await read(device!,solver.volumeTexture);
        assert.ok(Math.abs(volume.reduce((a,b)=>a+b,0)-128)<1e-4,"pressure support must not delete excess mass");
      }finally{solver.destroy();}
    });
    await t.test("one-ulp capacity differences do not create a pressure jet",async()=>{
      const solver=await fixture(0,0,0,true,10,{phiDrain:false,geometricRedistance:false});
      try {
        const phi=Float32Array.from({length:33*17*33},(_,i)=>.042*(Math.floor(i/33)%17)-.066);
        const volume=new Float32Array(32*16*32);
        for(let z=14;z<18;z++)for(let x=4;x<9;x++)for(let y=0;y<3;y++){
          const fill=[1.2,.9,1][y]!;
          volume[x+32*(y+16*z)]=fill;
          volume[31-x+32*(y+16*z)]=fill+(y===2?2**-23:0);
        }
        write(device!,solver.vertexPhiTexture!,phi);write(device!,solver.volumeTexture,volume);
        assert.ok(solver.advanceTo(1/30));
        const velocity=await read(device!,solver.velocityTexture);
        const left=velocity[4*(6+32*(1+16*15))+1]!;
        const right=velocity[4*(25+32*(1+16*15))+1]!;
        console.log(JSON.stringify({capacityUlp:left-right,left,right}));
        assert.ok(Math.abs(left-right)<.005,"continuous ghost distance across V=1");
      }finally{solver.destroy();}
    });
    await t.test("automatic airborne redistance preserves a stationary curved interface",async()=>{
      const solver=await fixture(0,0,0,true,10,{phiDrain:false});
      try {
        const phi=Float32Array.from({length:33*17*33},(_,i)=>
          .05*Math.hypot(i%33-16,Math.floor(i/33)%17-8,Math.floor(i/(33*17))-16)-.18);
        write(device!,solver.vertexPhiTexture!,phi);
        assert.ok(solver.advanceTo(1/30));
        const actual=await read(device!,solver.vertexPhiTexture!);
        let crossings=0;
        for(let z=0;z<32;z++)for(let y=0;y<16;y++)for(let x=0;x<32;x++){
          const i=x+33*(y+17*z);
          for(const stride of [1,33,33*17])if((phi[i]!<0)!==(phi[i+stride]!<0)){
            const before=-phi[i]!/(phi[i+stride]!-phi[i]!);
            const after=-actual[i]!/(actual[i+stride]!-actual[i]!);
            assert.ok(Math.abs(after-before)<1e-5,"redistance must not move an advected zero crossing");
            crossings++;
          }
        }
        assert.ok(crossings>100,"curved interface exercises many grid edges");
      }finally{solver.destroy();}
    });
    await t.test("redistance stays continuous across the Newton residual cutoff",async()=>{
      const solver=await fixture(0,0,0,true,10,{phiDrain:false});
      try {
        assert.ok(solver.advanceTo(1/30));
        // A stretched, piecewise-linear field leaves the eight-step search
        // on either side of the old 0.005h convergence cutoff. The two flat
        // horizontal regions differ by only two micrometres of input phi.
        const phi=Float32Array.from({length:33*17*33},(_,i)=>{
          const x=i%33,y=Math.floor(i/33)%17;
          return (y<8?.08:.0002)*(y-8)-(x<16?.000259:.000261);
        });
        write(device!,solver.advectedVertexPhiTexture!,phi);
        const internals=solver as unknown as {
          volumePipelines:{uvRedistancePhi:GPUComputePipeline};phiReverseGroup:GPUBindGroup;
          runVertex(encoder:GPUCommandEncoder,label:string,pipeline:GPUComputePipeline,group:GPUBindGroup):void;
        };
        const encoder=device!.createCommandEncoder();
        internals.runVertex(encoder,"Frozen redistance continuity regression",internals.volumePipelines.uvRedistancePhi,internals.phiReverseGroup);
        device!.queue.submit([encoder.finish()]);
        const actual=await read(device!,solver.vertexPhiTexture!);
        const left=actual[8+33*(6+17*16)]!,right=actual[24+33*(6+17*16)]!;
        assert.ok(left<0&&right<0,"interior sign is retained");
        // The exact zero contour shifts by 0.000002 / (0.0002 / 0.05)
        // = 0.0005 m. Redistance must not amplify it beyond that displacement.
        assert.ok(Math.abs(left-right)<.0005,`two-micrometre input perturbation produced ${Math.abs(left-right)} m of redistance difference`);
      }finally{solver.destroy();}
    });
    assert.deepEqual(errors,[]);
  }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
