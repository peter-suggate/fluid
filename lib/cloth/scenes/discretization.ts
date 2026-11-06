import type {WetClothSourceDefinition,WetClothSourceDiscretization,WetClothSourceVariant} from '../source-types';

const source='https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/';

/** Authored discretizations that depart from a released XML's dx and dt.
 * libWetCloth carries its liquid on particles, so a feature a few buckets
 * wide keeps its volume there. The narrow-band FLIP level set does not: a
 * pure-fluid sweep of the capillary tube and drop at the released dt lost
 * 9.5%, 5.8% and 0.08% of the liquid's phi volume in 30 steps at radii of
 * 2h, 4h and 8h. Each entry resolves the scene's liquid-bounding features at
 * about 8h or finer, keeps the geometry exact, and keeps the released
 * Courant number (dt scales with h). The released values stay available as
 * an exploratory variant (wetClothUpstreamDiscretizationVariants). */
export const wetClothSourceDiscretizations:ReadonlyMap<string,WetClothSourceDiscretization>=new Map([
 ['wetcloth-source-capillary-suction',{
  sourceCellSize_m:.00288,sourceDt_s:.001,cellSize_m:.00072,dt_s:.00025,
  reason:'The capillary tube and its drop have radius 0.576 cm, 2h at the released dx 0.288 cm (bucket 1.152 cm over 4 cells). At 2h the NB-FLIP drop loses 9.5% of its volume in 30 steps; at dx 0.072 cm both are 8h. dt 0.25 ms keeps the released Courant number: at dx 0.072 cm and the released 1 ms the loss was 0.51% in 30 steps, against 0.042% over the same 30 ms at 0.25 ms.',
  evidence:[source+'assets/unit_tests/capillary_suction.xml'],
 }],
]);

const cm=(m:number)=>+(m*100).toPrecision(6),ms=(s:number)=>+(s*1000).toPrecision(6);
const close=(a:number,b:number)=>Math.abs(a-b)<=1e-12*Math.max(Math.abs(a),Math.abs(b));

/** The authored discretization of a released source; refuses a source whose
 * released dx or dt differs from the one the entry was measured against. */
export function applyWetClothSourceDiscretization(base:WetClothSourceDefinition):WetClothSourceDefinition{
 const d=wetClothSourceDiscretizations.get(base.id);if(!d)return base;
 if(base.variant||base.discretization||!close(base.cellSize_m,d.sourceCellSize_m)||!close(base.dt_s,d.sourceDt_s))throw new Error(`Discretization of ${base.id} does not match the released dx and dt`);
 if(!(d.cellSize_m>0&&d.cellSize_m<base.cellSize_m&&d.dt_s>0&&d.dt_s<=base.dt_s))throw new Error(`Discretization of ${base.id} must refine the released dx and dt`);
 return {...base,cellSize_m:d.cellSize_m,dt_s:d.dt_s,discretization:d,
  auditNotes:[...base.auditNotes,`Discretization: dx ${cm(d.cellSize_m)} cm and dt ${ms(d.dt_s)} ms replace the released ${cm(d.sourceCellSize_m)} cm and ${ms(d.sourceDt_s)} ms. ${d.reason}`]};
}

/** The released dx and dt of each refined source, as exploratory inputs. */
export const wetClothUpstreamDiscretizationVariants:readonly WetClothSourceVariant[]=[...wetClothSourceDiscretizations].map(([id,d])=>({
 id:`${id}-upstream-dx`,sourceId:id,kind:'exploratory',label:`Released dx ${cm(d.sourceCellSize_m)} cm, dt ${ms(d.sourceDt_s)} ms`,
 reason:`The released XML's discretization, under-resolved for NB-FLIP. ${d.reason}`,evidence:d.evidence,
 overrides:[{path:'cellSize_m',sourceValue:d.cellSize_m,value:d.sourceCellSize_m},{path:'dt_s',sourceValue:d.dt_s,value:d.sourceDt_s}],reproductionValidated:false,
}));
