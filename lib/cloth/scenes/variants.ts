import type {ClothTopologyData} from '../types';
import type {WetClothSourceDefinition,WetClothSourceVariant} from '../source-types';

const paper='https://www.cs.columbia.edu/cg/wetcloth/figs/wetcloth_reduced.pdf#page=12';
const supplement='https://www.cs.columbia.edu/cg/wetcloth/supplemental.pdf#page=2';
const source='https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/';

/** Distinct reviewable inputs, never silent repairs to the released XML.
 * The ring correction is deliberately exploratory: the paper does not give
 * a ring-specific uniform fraction, so its material default is not evidence
 * of the published experiment's value. */
export const wetClothSourceVariants:readonly WetClothSourceVariant[]=[{
 id:'wetcloth-paper-figure10b-25um',sourceId:'wetcloth-source-pore-test-low',kind:'paper-corrected',label:'Figure 10b: 25 µm fibers',
 reason:'Figure 10b specifies 25 µm fibers at solid fraction 0.4; the released low-pore XML specifies 20 µm. Pore radius and thread count derive from these inputs (supplement Table 2).',
 evidence:[paper,supplement],overrides:[{path:'liquid.fiberDiameter_m',sourceValue:20e-6,value:25e-6}],reproductionValidated:false,
},{
 id:'wetcloth-paper-figure10j-90deg',sourceId:'wetcloth-source-pore-test-zero',kind:'paper-corrected',label:'Figure 10j: 90° contact angle with water surface tension',
 reason:'Figure 10j changes contact angle to 90° to remove pore suction. The released XML also zeros surface tension; restore reference water surface tension 72 dyn/cm while retaining water viscosity and an exact π/2 contact angle.',
 evidence:[paper,supplement,source+'assets/parameter_tests/pore_test_mid.xml'],
 overrides:[{path:'liquid.contactAngle_rad',sourceValue:1.5707963268,value:Math.PI/2},{path:'liquid.surfaceTension_N_m',sourceValue:0,value:.072}],reproductionValidated:false,
},{
 id:'wetcloth-exploratory-ring-material-packing',sourceId:'wetcloth-source-ring-test',kind:'exploratory',label:'Exploratory ring: uniform material packing 0.4',
 reason:'The released XML enables nonuniform fractions but supplies no vf, leaving effective vertex vf=1 and zero pore capacity. This exploratory comparison selects its material restVolumeFraction=0.4; that value is not confirmed as the paper ring parameter.',
 evidence:[source+'assets/ring_tests/ring_test.xml',source+'libWetCloth/App/TwoDSceneXMLParser.cpp#L1472',source+'libWetCloth/Core/TwoDScene.cpp#L5885'],
 overrides:[{path:'geometry.solidFraction',sourceValue:1,value:.4},{path:'sourceSettings.liquid.initNonuniformFraction',sourceValue:1,value:0}],reproductionValidated:false,
}];

export function applyWetClothSourceVariant(base:WetClothSourceDefinition,variant:WetClothSourceVariant):WetClothSourceDefinition{
 if(base.id!==variant.sourceId||base.variant)throw new Error('Invalid released source for variant');
 const liquid={...base.liquid},settings={...base.sourceSettings.liquid};
 for(const change of variant.overrides){
  const close=(actual:unknown)=>{if(typeof actual!=='number'||Math.abs(actual-change.sourceValue)>1e-12*Math.max(1,Math.abs(change.sourceValue)))throw new Error(`Variant source value mismatch ${change.path}`);};
  if(change.path==='liquid.fiberDiameter_m'){close(liquid.fiberDiameter_m);liquid.fiberDiameter_m=change.value;settings.yarnDiameter=change.value*100;}
  else if(change.path==='liquid.surfaceTension_N_m'){close(liquid.surfaceTension_N_m);liquid.surfaceTension_N_m=change.value;settings.surfTensionCoeff=change.value*1000;}
  else if(change.path==='liquid.contactAngle_rad'){close(liquid.contactAngle_rad);liquid.contactAngle_rad=change.value;settings.restContactAngle=change.value;}
  else if(change.path==='sourceSettings.liquid.initNonuniformFraction'){close(settings.initNonuniformFraction);settings.initNonuniformFraction=change.value;}
  else if(change.path!=='geometry.solidFraction')throw new Error(`Unsupported source variant override ${change.path}`);
 }
 return {...base,id:variant.id,name:variant.label,variant,liquid,sourceSettings:{...base.sourceSettings,liquid:settings},reproductionStatus:variant.kind==='paper-corrected'?'paper-corrected-unverified':'exploratory-unverified',
  auditNotes:[...base.auditNotes,`${variant.kind}: ${variant.reason}`]};
}

export function wetClothVariantTopology(data:ClothTopologyData,variant:WetClothSourceVariant|undefined):ClothTopologyData{
 const override=variant?.overrides.find(o=>o.path==='geometry.solidFraction');if(!override)return data;
 if(!data.solidFraction?.every(v=>v===override.sourceValue))throw new Error('Exploratory packing source fractions do not match the audited release');
 return {...data,solidFraction:data.solidFraction.map(()=>override.value)};
}
