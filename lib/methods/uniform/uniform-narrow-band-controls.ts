import type { MethodParamValues, NumberParamSpec, SelectParamSpec } from "../../core/method-contract";

export const NARROW_BAND_ACTIVITY_CONTROLS = [
 {key:"adaptiveBudgetPercent",label:"Adaptive budget",unit:"%",min:0,max:100,step:1,digits:0,default:100,
  hint:"Target share of tiles requesting automatic fine detail. Equal-score tiles stay together, so admission can exceed this share. 100 admits all; 0 admits none. Cooling particles, source liquid and required contact can keep extra tiles fine."},
 {key:"adaptiveFadeSeconds",label:"Retirement time",unit:"s",min:0.05,max:2,step:0.05,digits:2,default:0.5,
  hint:"Time after activity stops before particles retire. Surface influence fades before the remaining particle support retires. Shorter releases fine tiles sooner."},
] as const;
/** Volume control: the pressure solve closes the liquid's gap to its volume budget over this time. */
export const NARROW_BAND_VOLUME_CONTROL = {key:"volumeControlSeconds",label:"Volume control time",unit:"s",min:0,max:2,step:0.05,digits:2,default:0.1,
  hint:"Time over which the liquid's measured gap to its volume budget is closed, as a uniform divergence in the pressure solve that carries the particles and the surface together. Shorter holds the volume tighter and moves the liquid more. A gap under 1% of the budget is left alone, so liquid at rest stays at rest. 0 turns volume control off."} as const;
export const NARROW_BAND_SPARSE_CONTROL = {key:"sparseParticleLifetime",label:"Sparse particle lifetime",unit:"s",min:0,max:10,step:0.1,digits:1,default:2,
  hint:"Retire a particle after this many continuous seconds with fewer than eight neighbours in the 3×3×3 grid cells around it. Neighbours returning reset its timer. Applies to coupled and ballistic particles; dense sheets and droplets retain their support. 0 disables sparse retirement."} as const;
export const NARROW_BAND_PARTICLE_CAP_CONTROL = {key:"particleCellCap",label:"Particles per cell",unit:"",min:0,max:128,step:1,digits:0,default:20,
  hint:"Limits crowding more than one cell beneath the surface. Lower values reduce particle work. Surface particles are preserved; newly seeded particles may temporarily exceed this limit. 0 disables the cap."} as const;
export function narrowBandParticleCellCap(value:unknown,fallback=20):number{
 const n=Number(value);return Number.isFinite(n)?Math.floor(Math.min(128,Math.max(0,n))):fallback;
}
const controls=[...NARROW_BAND_ACTIVITY_CONTROLS,NARROW_BAND_VOLUME_CONTROL,NARROW_BAND_SPARSE_CONTROL,NARROW_BAND_PARTICLE_CAP_CONTROL] as const;
export const narrowBandControlValues = (values:MethodParamValues={}) => Object.fromEntries(controls.map(c=>{
 const raw=Number(values[c.key]);
 return [c.key,c.key==="particleCellCap"?narrowBandParticleCellCap(raw):Number.isFinite(raw)?Math.min(c.max,Math.max(c.min,raw)):c.default];
})) as Record<(typeof controls)[number]["key"],number>;
/** The level-set advect at h vertices. The particles carry the surface there and the advected one is blended in wherever heat is below 1.
 * Adaptive: trilinear where incident open tiles have particles that pin the surface (surface heat over 1.5, the couple's theta over 0.5), cubic elsewhere. */
export const NARROW_BAND_FINE_ADVECTION:SelectParamSpec={kind:"select",key:"phiFineAdvection",label:"h-tile phi advection",default:"adaptive",tier:"fine",update:"runtime",
  options:[{value:"adaptive",label:"Adaptive"},{value:"cubic",label:"Cubic"},{value:"trilinear",label:"Trilinear"}],
  hint:"Trilinear moves the level set at h vertices with eight taps in place of the cubic's 64, 1 to 3 ms of a figure-9 frame; 4h vertices keep the cubic. Curved surfaces that the particles do not fully carry shrink a little more each step: the figure-9 take then holds 5 to 9% fewer particles. Adaptive keeps the cubic wherever no hot particles hold the surface (spray, the letters' first frames) and takes trilinear under them. No effect with cubic phi advection off."};
export const narrowBandFineAdvection=(value:unknown):"cubic"|"trilinear"|"adaptive"=>value==="cubic"||value==="trilinear"?value:"adaptive";
export const narrowBandControlParams:readonly NumberParamSpec[]=controls.map(c=>({...c,kind:"number",tier:"coarse",update:"runtime",dedicated:true}));
