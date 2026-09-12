import { productionSceneSliceSeedById } from "../lib/methods/adaptive-volume/advance-slice/production-scene-slice";
import { advanceSlice, createAdvanceSlice, type AdvanceSlice,
  type SliceStageId } from "../lib/methods/adaptive-volume/advance-slice/slice-solver";

function integerArgument(name:string,fallback:number):number {
  const value=Number(process.argv.find(argument=>argument.startsWith(`--${name}=`))?.split("=")[1]
    ?? fallback);
  if(!Number.isSafeInteger(value)||value<0)throw new RangeError(`--${name} must be a nonnegative integer`);
  return value;
}
const frames = integerArgument("frames",6);
const pressureIterations = integerArgument("pressure",64);
const sceneId = process.argv.find(value => value.startsWith("--scene="))?.split("=")[1]
  ?? "coarse-first-pool-impact-half";

function maximumPairError(slice:AdvanceSlice, field:ArrayLike<number>):number {
  let maximum=0;
  for(let y=0;y<slice.ny;y++)for(let x=0;x<slice.nx/2;x++)maximum=Math.max(maximum,
    Math.abs(field[y*slice.nx+x]!-field[y*slice.nx+slice.nx-1-x]!));
  return maximum;
}

function maximumVelocityError(slice:AdvanceSlice):readonly[number,number] {
  let xError=0,yError=0;
  for(let y=0;y<slice.ny;y++)for(let x=0;x<=slice.nx;x++)xError=Math.max(xError,
    Math.abs(slice.u[y*(slice.nx+1)+x]!+slice.u[y*(slice.nx+1)+slice.nx-x]!));
  for(let y=0;y<=slice.ny;y++)for(let x=0;x<slice.nx;x++)yError=Math.max(yError,
    Math.abs(slice.v[y*slice.nx+x]!-slice.v[y*slice.nx+slice.nx-1-x]!));
  return [xError,yError];
}

function topologyErrors(slice:AdvanceSlice):number {
  let errors=0;
  for(const brick of slice.topology.accepted.bricks){
    const span=brick.spanBricks??1;
    const mirror=slice.topology.accepted.bricks.find(candidate =>
      candidate.coordinate[0]===slice.bx-span-brick.coordinate[0]
      &&candidate.coordinate[1]===brick.coordinate[1]&&(candidate.spanBricks??1)===span);
    if(!mirror||mirror.resolution!==brick.resolution||mirror.active!==brick.active)errors++;
  }
  return errors;
}

const slice=createAdvanceSlice(productionSceneSliceSeedById(sceneId));
const report=(frame:number,stage:SliceStageId|"initial")=>{
  const [u,v]=maximumVelocityError(slice);
  const record={frame,completedFrame:slice.frame,time_s:slice.time_s,stage,
    density:maximumPairError(slice,slice.V),
    pressure:maximumPairError(slice,slice.p),xVelocity:u,yVelocity:v,
    topologyErrors:topologyErrors(slice),fault:slice.fault};
  console.log(JSON.stringify(record,(_key,value)=>typeof value==="number"&&!Number.isFinite(value)
    ?String(value):value));
};
report(0,"initial");
for(let frame=1;frame<=frames;frame++)advanceSlice(slice,{pressureIterations,
  onStageComplete:stage=>report(frame,stage)});
