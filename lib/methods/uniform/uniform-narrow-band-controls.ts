import type { MethodParamValues, NumberParamSpec } from "../../core/method-contract";

export const NARROW_BAND_ACTIVITY_CONTROLS = [
 {key:"adaptiveBudgetPercent",label:"Adaptive budget",unit:"%",min:0,max:100,step:1,digits:0,default:50,
  hint:"Share of tiles requesting automatic fine detail to admit. 100 admits all; 0 admits none. Cooling particles, source liquid and required contact can keep extra tiles fine."},
 {key:"adaptiveFadeSeconds",label:"Retirement time",unit:"s",min:0.05,max:2,step:0.05,digits:2,default:0.5,
  hint:"Time after activity stops before particles retire. The first half holds full detail; the second blends back to the grid. Shorter releases fine tiles sooner."},
] as const;
export const narrowBandActivityValues = (values:MethodParamValues={}) => Object.fromEntries(NARROW_BAND_ACTIVITY_CONTROLS.map(c=>{
 const raw=Number(values[c.key]);
 return [c.key,Number.isFinite(raw)?Math.min(c.max,Math.max(c.min,raw)):c.default];
})) as Record<(typeof NARROW_BAND_ACTIVITY_CONTROLS)[number]["key"],number>;
export const narrowBandActivityParams:readonly NumberParamSpec[]=NARROW_BAND_ACTIVITY_CONTROLS.map(c=>({...c,kind:"number",tier:"coarse",update:"runtime",dedicated:true}));
