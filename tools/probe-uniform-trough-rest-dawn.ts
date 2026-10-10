import {createUniformTroughScene} from "../lib/core/uniform-trough-scenes";
import {resolveMethodValues} from "../lib/core/method-contract";
import {uniformNarrowBandMethod} from "../lib/methods/uniform/uniform-narrow-band-method";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {advanceUniform,readUniformFields} from "../tests/helpers/uniform-geometric";
import {readMixedTexture} from "../tests/helpers/uniform-mixed-native-fields";
import {withPondRestDevice} from "./uniform-pond-rest";
const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const h=Number(arg("h","0.0125")),frames=Number(arg("frames","120"));
const method=arg("method","nb-flip")==="uniform"?uniformVolumeMethod:uniformNarrowBandMethod;
await withPondRestDevice("settled trough",async device=>{
 const scene=createUniformTroughScene("settled-tank",h);
 scene.fluid.gravity_m_s2.y=Number(arg("gravity",String(scene.fluid.gravity_m_s2.y)));
 const level=Math.max(...scene.fluid.initialLiquidVolumes!.map(v=>v.shape==="box"?v.max_m.y:0));
 const values=resolveMethodValues(method,"balanced",{...method.appDefaults,timeStep:"scene",...JSON.parse(arg("values","{}"))});
 const solver=await method.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 try{
  let initial=0;const {nx,ny,nz}=solver.info;
  for(let frame=0;frame<=frames;frame++){
   if(frame)await advanceUniform(solver,frame/60);
   if(frame>3&&frame%30!==0&&frame!==frames)continue;
   const f=await readUniformFields(device,solver),velocity=await readMixedTexture(device,solver.velocityTexture);
   const volume=f.density.reduce((a,b)=>a+b,0);if(!frame)initial=volume;
   let speed=0,maxCell:number[]=[];for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
    const i=x+nx*(y+ny*z);if(f.density[i]!<.01)continue;
    const w=f.widthAt(x,y,z),at=4*(x-x%w+nx*(y-y%w+ny*(z-z%w)));
    const u=Math.hypot(...velocity.subarray(at,at+3));if(u>speed){speed=u;maxCell=[x,y,z];}
   }
   let maxSurface=0,rms=0,count=0;const row=Math.floor(level/h);
   for(let z=1;z<nz;z++)for(let x=1;x<nx;x++){
    const a=f.vertex(x,row,z),b=f.vertex(x,row+1,z);if(a>0||b<=0)continue;
    const error=(row-a/(b-a))*h-level;maxSurface=Math.max(maxSurface,Math.abs(error));rms+=error*error;count++;
   }
   const stats=await solver.readStats();
   console.log(JSON.stringify({frame,time:frame/60,h,level,method:method.id,volume,drift:volume/initial-1,speed,maxCell,maxSurface,rms:Math.sqrt(rms/Math.max(1,count)),count,residual:stats.uniformPressureAcceptedResidual,bandResidual:stats.uniformPressureBandResidual}));
  }
 }finally{solver.destroy();}
});
