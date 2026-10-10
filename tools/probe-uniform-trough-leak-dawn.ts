import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { bathInteriorContains, bathSolidContains } from "../lib/core/voxel-bath";
import { resolveMethodValues } from "../lib/core/method-contract";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { advanceUniform, readUniformFields } from "../tests/helpers/uniform-geometric";
import { readMixedBuffer } from "../tests/helpers/uniform-mixed-native-fields";
import { withPondRestDevice } from "./uniform-pond-rest";

const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const methodId=arg("method","nb-flip");
if(methodId!=="nb-flip"&&methodId!=="uniform")throw new Error("--method must be nb-flip or uniform");
const method=methodId==="nb-flip"?uniformNarrowBandMethod:uniformVolumeMethod;
const frames=Number(arg("frames","240"));
if(!Number.isInteger(frames)||frames<1)throw new Error("--frames must be a positive integer");
await withPondRestDevice("trough containment",async device=>{
 const definition=getSceneDefinition("uniform-trough-hose-fill"),scene=sceneDocument(definition);
 const values=resolveMethodValues(method,"balanced",{...method.appDefaults,...definition.methodProfile?.overrides,timeStep:"scene"});
 const solver=await method.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 try{
  const {nx,ny,nz}=solver.info,step=scene.numerics.fixedDt_s;
  const {width_m:width,height_m:height,depth_m:depth}=scene.container;
  const position=(x:number,y:number,z:number)=>[x*width/nx-width/2,y*height/ny,z*depth/nz-depth/2] as const;
  for(let frame=0;frame<=frames;frame++){
   if(frame)await advanceUniform(solver,frame*step);
   if(frame>3&&frame%30!==0&&frame!==frames)continue;
   const fields=await readUniformFields(device,solver);
   let total=0,exterior=0,exteriorWetCells=0,exteriorPhi=0;
   const worst:{cell:number[];volume:number;phi:number}[]=[];
   for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
    const p=position(x+.5,y+.5,z+.5);
    const v=fields.density[x+nx*(y+ny*z)]!;total+=v;
    if(bathInteriorContains(...p)||bathSolidContains(...p))continue;
    exterior+=v;if(v>1e-5)exteriorWetCells++;
    // A sub-cell leak can leave the centre in air: inspect every corner.
    let phi=Infinity;for(let k=0;k<8;k++)phi=Math.min(phi,fields.vertex(x+(k&1),y+((k>>1)&1),z+((k>>2)&1)));
    if(phi<0)exteriorPhi++;
    if(v>.01||phi<0)worst.push({cell:[x,y,z],volume:v,phi});
   }
   const stage=(solver as unknown as {mixedFrame:{narrowBandFlip?:{activeParticles:GPUBuffer;count:number;adaptiveSurface:boolean}}}).mixedFrame.narrowBandFlip;
   let escapedParticles=0;
   if(stage){const particles=await readMixedBuffer(device,stage.activeParticles);
    for(let i=0;i<stage.count;i++){const p=position(particles[12*i]!,particles[12*i+1]!,particles[12*i+2]!);
     if(!bathInteriorContains(...p)&&!bathSolidContains(...p))escapedParticles++;
    }
   }
   worst.sort((a,b)=>b.volume-a.volume);
   console.log(JSON.stringify({method:method.id,frame,time:frame*step,total,exterior,exteriorWetCells,exteriorPhi,escapedParticles,particleCount:stage?.count??0,adaptiveSurface:stage?.adaptiveSurface??false,worst:worst.slice(0,5)}));
  }
 }finally{solver.destroy();}
});
