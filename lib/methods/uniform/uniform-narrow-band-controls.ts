import type { MethodParamValues, NumberParamSpec } from "../../core/method-contract";

export const NARROW_BAND_ACTIVITY_CONTROLS = [
 {key:"adaptiveBudgetPercent",label:"Adaptive budget",unit:"%",min:0,max:100,step:1,digits:0,default:50,
  hint:"Share of tiles requesting automatic fine detail to admit. 100 admits all; 0 admits none. Cooling particles, source liquid and required contact can keep extra tiles fine."},
 {key:"adaptiveFadeSeconds",label:"Retirement time",unit:"s",min:0.05,max:2,step:0.05,digits:2,default:0.5,
  hint:"Time after activity stops before particles retire. Surface influence fades before the remaining particle support retires. Shorter releases fine tiles sooner."},
] as const;
/** Volume control: the pressure solve closes the liquid's gap to its volume budget over this time. */
export const NARROW_BAND_VOLUME_CONTROL = {key:"volumeControlSeconds",label:"Volume control time",unit:"s",min:0,max:2,step:0.05,digits:2,default:0.1,
  hint:"Time over which the liquid's measured gap to its volume budget is closed, as a uniform divergence in the pressure solve that carries the particles and the surface together. Shorter holds the volume tighter and moves the liquid more. 0 turns volume control off."} as const;
const controls=[...NARROW_BAND_ACTIVITY_CONTROLS,NARROW_BAND_VOLUME_CONTROL] as const;
export const narrowBandControlValues = (values:MethodParamValues={}) => Object.fromEntries(controls.map(c=>{
 const raw=Number(values[c.key]);
 return [c.key,Number.isFinite(raw)?Math.min(c.max,Math.max(c.min,raw)):c.default];
})) as Record<(typeof controls)[number]["key"],number>;
export const narrowBandControlParams:readonly NumberParamSpec[]=controls.map(c=>({...c,kind:"number",tier:"coarse",update:"runtime",dedicated:true}));
