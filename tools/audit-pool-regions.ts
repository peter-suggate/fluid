import {UNIFORM_REGION_HEAT_STATE_WORDS as S,UNIFORM_REGION_HEAT_BUDGET_H as budget,UNIFORM_REGION_HEAT_SUBMERGED_BUDGET_H as submergedBudget} from "../lib/methods/uniform/uniform-region-heat";
/** Post-frame diagnostics only: no runtime readback or scheduling policy. */
import {readMixedBuffer} from "../tests/helpers/uniform-mixed-native-fields";
import type {UniformMixedFrame} from "../lib/methods/uniform/uniform-mixed-frame";
import {UNIFORM_STAGE_IMPORTANCE as I} from "../lib/methods/uniform/uniform-stage-grids";

export async function auditPoolRegions(device:GPUDevice,frame:UniformMixedFrame){
 const f=frame as unknown as {heat:{state:GPUBuffer;tiers:GPUBuffer;regions:number};ownership:{regionVelocity:GPUBuffer;capacity:{tiles:number}};solid:{bodyRegions:{buffer:GPUBuffer}};dynamics:{record:GPUBuffer};plan:{updateErrors:GPUBuffer};narrowBandFlip:{airGateBuffer:GPUBuffer;activityHeatWord:number}};
 const [state,tier,velocity,dynamics,errors,mask,bins]=await Promise.all([f.heat.state,f.heat.tiers,f.ownership.regionVelocity,f.dynamics.record,f.plan.updateErrors,f.solid.bodyRegions.buffer,f.narrowBandFlip.airGateBuffer].map(b=>readMixedBuffer(device,b)));
 const words=(a:Float32Array)=>new Uint32Array(a.buffer,a.byteOffset,a.length),s=words(state!),tiers=words(tier!),v=words(velocity!),m=words(mask!),b=words(bins!);
 const names=['host','identity','claim','body','source','unknown','freezeBudget','velocitySpan','remap','heldHot','heldWarm','tileError'];
 const totals=Array.from({length:3},()=>({count:0,reasons:{} as Record<string,number>,particle:0,cooling:0,halo:0}));
 const flip=f.narrowBandFlip,N=f.ownership.capacity.tiles,decode=(w:number)=>new Float32Array(new Uint32Array([w&0x80000000?w^0x80000000:~w]).buffer)[0]!;
 const regions=[];
 for(let r=0;r<f.heat.regions;r++){
  if(s[S*r+2]===0xffffffff)continue;
  const t=tiers[r]!,bits=s[S*r+7]!&4095,heat=state![S*r]!,c=Array.from(s.slice(S*r+3,S*r+6));
  let speed=0,span=0,error=0,particle=0;const criteria=[0,0,0,0,0,0];let fine=0,normal=0,acceleration=0,physicalTiles=0;
  for(let a=0;a<3;a++){const hi=v[6*r+a]!,lo=v[6*r+3+a]!;if(hi===0||lo===0)continue;speed=Math.max(speed,Math.abs(decode(hi)),Math.abs(decode(lo)));span+=Math.max(0,decode(hi)+decode(lo));}
  for(let l=0;l<64;l++){
   const x=4*(r&1)+(l&3),y=4*((r>>1)&1)+((l>>2)&3),z=4*((r>>2)&1)+(l>>4),tile=512*(r>>3)+x+8*y+64*z;
   normal=Math.max(normal,errors![12*N+4*tile]??0);acceleration=Math.max(acceleration,errors![12*N+4*tile+1]??0);physicalTiles+=+(errors![12*N+4*tile+2]!>0);
   error=Math.max(error,errors![8*tile+7]!);particle=Math.max(particle,bins![flip.activityHeatWord+tile]!);fine+=+(errors![8*tile+6]===1);
   const flags=b[flip.activityHeatWord-2*N+2*tile+1]!;
   if(!(flags&I.dropped))for(let k=0;k<6;k++)criteria[k]!+=+!!(flags&(1<<(I.triggeredShift+k)));
  }
  const total=totals[t]!;total.count++;
  const reasons=names.filter((_,k)=>bits&(1<<k));for(const name of reasons)total.reasons[name]=(total.reasons[name]??0)+1;
  if(particle>=1)total.particle++;if(heat>0&&!bits&&particle===0)total.cooling++;if(t===1&&!bits&&heat===0)total.halo++;
  const submerged=(s[S*r+7]!&8192)!==0;
  regions.push({decision:{submerged,budget:submerged?submergedBudget:budget,surface:state![S*r+9],momentum:state![S*r+10],pressure:state![S*r+11],frozen:state![S*r+12],hotError:state![S*r+13],physicalTiles:s[S*r+14],debt:state![S*r+15]},r,tier:t,c,bits,reasons,heat,pending:s[S*r+6],held:s[S*r+8],displacement:state![S*r+1],speed,span,error,normal,acceleration,physicalTiles,particle,criteria,fine,pressureDelta:dynamics![10*r+6],pressureResidual:dynamics![10*r+7],swept:m[r]!&255,pose:(m[r]!>>8)&255});
 }
 const byCoordinate=new Map(regions.map(r=>[r.c.join(","),r]));
 const unexplained:number[]=[],invalidBudget:number[]=[];
 const justificationCounts:Record<string,number>={};
 const justified=regions.map(r=>{
  const reasons:string[]=[];
  if(r.tier>0){
   if(r.bits&0x227||r.pending!>=2)reasons.push("invalidation-or-hot-hold");
   if(r.bits&8)reasons.push("moving-boundary");
   if(r.bits&16)reasons.push("source");
   if(r.bits&2048)reasons.push("tile-update-error");
   if(r.bits&64){
    reasons.push("freeze-error");const d=r.decision;
    if(d.debt!+d.surface!+d.momentum!+d.pressure!+d.frozen!<=d.budget)invalidBudget.push(r.r);
   }
   if(r.bits&1024)reasons.push("warm-hold");
   if(r.bits&256||r.pending===1)reasons.push("spatial-remap");
   if(r.heat>0)reasons.push("bounded-detail-hysteresis");
   if(!reasons.length&&r.tier===1&&r.decision.physicalTiles!>0){
    for(let z=-1;z<=1;z++)for(let y=-1;y<=1;y++)for(let x=-1;x<=1;x++){
     const n=byCoordinate.get([r.c[0]!+x,r.c[1]!+y,r.c[2]!+z].join(","));
     if(n&&(n.heat>=1||(n.bits&0x227)!==0))reasons.push("physical-dependency");
    }
   }
   if(!reasons.length)unexplained.push(r.r);
   for(const reason of new Set(reasons))justificationCounts[reason]=(justificationCounts[reason]??0)+1;
  }
  return {...r,justification:[...new Set(reasons)]};
 });
 return {totals,justificationCounts,unexplained,invalidBudget,regions:justified};
}
