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
  hint:"Retire a particle after this many continuous seconds with fewer than Sparse neighbours in the 3×3×3 grid cells around it. Neighbours returning reset its timer. Applies to coupled and ballistic particles; dense sheets and droplets retain their support. 0 disables sparse retirement."} as const;
/** Sparse retirement's threshold: a full h cell seeds eight samples, so 8 is about one cell's worth across the 27-cell block. */
export const NARROW_BAND_SPARSE_NEIGHBOURS_CONTROL = {key:"sparseNeighbours",label:"Sparse neighbours",unit:"",min:0,max:64,step:1,digits:0,default:16,
  hint:"A particle with fewer than this many other particles in the 3×3×3 grid cells around it is sparse and retires after Sparse particle lifetime. A full cell seeds eight, so 8 keeps any clump of about a cell; 16 or 32 also retires small clumps of spray. 0 disables sparse retirement."} as const;
export const NARROW_BAND_PARTICLE_CAP_CONTROL = {key:"particleCellCap",label:"Particles per cell",unit:"",min:0,max:128,step:1,digits:0,default:20,
  hint:"Limits crowding more than one cell beneath the surface. Lower values reduce particle work. Surface particles are preserved; newly seeded particles may temporarily exceed this limit. 0 disables the cap."} as const;
export function narrowBandParticleCellCap(value:unknown,fallback=20):number{
 const n=Number(value);return Number.isFinite(n)?Math.floor(Math.min(128,Math.max(0,n))):fallback;
}
export function narrowBandSparseNeighbours(value:unknown,fallback=16):number{
 const n=Number(value);return Number.isFinite(n)?Math.round(Math.min(64,Math.max(0,n))):fallback;
}
const controls=[...NARROW_BAND_ACTIVITY_CONTROLS,NARROW_BAND_VOLUME_CONTROL,NARROW_BAND_SPARSE_CONTROL,NARROW_BAND_SPARSE_NEIGHBOURS_CONTROL,NARROW_BAND_PARTICLE_CAP_CONTROL] as const;
export const narrowBandControlValues = (values:MethodParamValues={}) => Object.fromEntries(controls.map(c=>{
 const raw=Number(values[c.key]);
 return [c.key,c.key==="particleCellCap"?narrowBandParticleCellCap(raw):c.key==="sparseNeighbours"?narrowBandSparseNeighbours(raw):Number.isFinite(raw)?Math.min(c.max,Math.max(c.min,raw)):c.default];
})) as Record<(typeof controls)[number]["key"],number>;
/** The level-set advect at h vertices. The particles carry the surface there and the advected one is blended in wherever heat is below 1.
 * Adaptive: trilinear where incident open tiles have particles that pin the surface (surface heat over 1.5, the couple's theta over 0.5), cubic elsewhere. */
export const NARROW_BAND_FINE_ADVECTION:SelectParamSpec={kind:"select",key:"phiFineAdvection",label:"h-tile phi advection",default:"adaptive",tier:"fine",update:"runtime",
  options:[{value:"adaptive",label:"Adaptive"},{value:"cubic",label:"Cubic"},{value:"trilinear",label:"Trilinear"}],
  hint:"Trilinear moves the level set at h vertices with eight taps in place of the cubic's 64, 1 to 3 ms of a figure-9 frame (about 1.9 ms of the pour's late frame); 4h vertices follow the 4h-tile phi advection. Curved surfaces that the particles do not fully carry shrink a little more each step: the figure-9 take then holds 5 to 9% fewer particles. Adaptive keeps the cubic wherever no hot particles hold the surface (spray, the letters' first frames) and takes trilinear under them. With cubic phi advection off, h vertices take trilinear. A falling or translating block, kept at h at its corners, loses 3% in 10 frames under trilinear and under 1% under the cubic."};
export const narrowBandFineAdvection=(value:unknown):"cubic"|"trilinear"|"adaptive"=>value==="cubic"||value==="trilinear"?value:"adaptive";
/** The level-set advect at 4h vertices. No particles carry a 4h surface: the
 * census keeps a moving surface at h where 4h would round its corners or
 * curves, so a 4h surface is flat or gently curved. */
export const NARROW_BAND_COARSE_ADVECTION:SelectParamSpec={kind:"select",key:"phiCoarseAdvection",label:"4h-tile phi advection",default:"cubic",tier:"fine",update:"runtime",
  options:[{value:"cubic",label:"Cubic"},{value:"trilinear",label:"Trilinear"}],
  hint:"How 4h vertices move the level set. The census keeps a moving corner, curve, thin sheet or droplet at h, so a 4h surface is flat or gently curved: trilinear moves a flat surface exactly with eight taps in place of the cubic's 64, and pulls a curved one in by a twelfth of its 4h second difference a step. Cubic holds a curved 4h surface. In the analytic suite trilinear 4h costs 0 to 0.5% more volume in 10 frames (translation, dam-break) and saves about 0.4 ms of the pour's late frame."};
export const narrowBandCoarseAdvection=(value:unknown):"cubic"|"trilinear"=>value==="trilinear"?"trilinear":"cubic";
export const narrowBandControlParams:readonly NumberParamSpec[]=controls.map(c=>({...c,kind:"number",tier:"coarse",update:"runtime",dedicated:true}));
