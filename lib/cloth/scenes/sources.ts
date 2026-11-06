import catalogData from './source-catalog.json';
import type { ClothDocument, ClothMaterial, ClothSupportScript } from '../types';
import type { WetClothSourceCatalog, WetClothSourceDefinition } from '../source-types';
import {SourceEmissionScheduler} from '../source-fields';
import {applyWetClothSourceVariant,wetClothSourceVariants,wetClothVariantTopology} from './variants';
import {cachedClothTopology} from '../source-assets';
import {applyWetClothSourceDiscretization,wetClothUpstreamDiscretizationVariants} from './discretization';

const catalog = catalogData as unknown as WetClothSourceCatalog;
if (catalog.schemaVersion !== 1 || catalog.runtimeUnits !== 'SI' || catalog.scenes.length !== 36) throw new Error('Unsupported wet-cloth source catalog');
// Released inputs at their authored discretization (scenes/discretization.ts).
const released = catalog.scenes.map(applyWetClothSourceDiscretization);
const sources = new Map(released.map(source => [source.id, source]));
if (sources.size !== catalog.scenes.length) throw new Error('Duplicate wet-cloth source identity');
export const wetClothVariantSources:readonly WetClothSourceDefinition[]=wetClothSourceVariants.map(variant=>{
 const base=sources.get(variant.sourceId);if(!base)throw new Error('Unknown variant released source');return applyWetClothSourceVariant(base,variant);
});
/** The released dx and dt of each refined source. Resolvable by id, not yet
 * on the validation shelf. */
export const wetClothUpstreamDiscretizationSources:readonly WetClothSourceDefinition[]=wetClothUpstreamDiscretizationVariants.map(variant=>applyWetClothSourceVariant(sources.get(variant.sourceId)!,variant));
for(const variant of [...wetClothVariantSources,...wetClothUpstreamDiscretizationSources]){if(sources.has(variant.id))throw new Error('Duplicate source variant identity');sources.set(variant.id,variant);}

/** These are imported inputs, not certification that the simulations match. */
export const wetClothWaterSources: readonly WetClothSourceDefinition[] = released;
export const wetClothValidationSources:readonly WetClothSourceDefinition[]=[...wetClothWaterSources,...wetClothVariantSources];
export const WETCLOTH_UPSTREAM_COMMIT = catalog.upstreamCommit;

export function wetClothSource(id: string): WetClothSourceDefinition {
  const source = sources.get(id);
  if (!source) throw new Error(`Unknown water/cloth source ${id}`);
  return source;
}

/** Source-compatible defaults. Source discrepancies remain visible in the
 * catalog and must not be corrected by silently changing this constructor. */
export function sourceClothDocument(id: string): ClothDocument {
  const source = wetClothSource(id), liquid = source.liquid;
  if (!source.counts.vertices) throw new Error(`Source ${id} contains no cloth or rods`);
  const materials: ClothMaterial[] = source.materials.map(material => {
    const viscous=Number(material.sourceSettings.accumulateWithViscous??0)!==0;
    const bendingOnly=Number(material.sourceSettings.accumulateViscousOnlyForBendingModes??1)!==0;
    const relaxation=material.solidViscosity_Pa_s/material.youngModulus_Pa;
    return {
    id: material.id, density_kg_m3: material.density_kg_m3, youngModulus_Pa: material.youngModulus_Pa,
    poissonRatio: material.poissonRatio, thickness_m: material.thickness_m, rodRadius_m: material.radius_m,rodRadiusB_m:material.radiusB_m,
    solidFraction: liquid.solidFraction, solidFractionExponent: 2,
    bendingMode: source.sourceSettings.liquid.bendingScheme === 'tantheta' ? 'tan-half'
      : source.sourceSettings.liquid.bendingScheme === 'sintheta' ? 'sin-half' : 'theta',
    fiberDiameter_m: liquid.fiberDiameter_m, contactAngle_rad: liquid.contactAngle_rad,
    surfaceTension_N_m: liquid.surfaceTension_N_m, waterDensity_kg_m3: liquid.density_kg_m3,
    waterViscosity_Pa_s: liquid.dynamicViscosity_Pa_s,
    // The pinned parser reads frictionAngle, not the legacy friction_angle.
    friction: Math.tan(Number(material.sourceSettings.frictionAngle ?? 0) * Math.PI / 180),
    mechanicalDamping_s: material.solidViscosity_Pa_s / material.youngModulus_Pa,
    membraneDamping_s:viscous&&!bendingOnly?3*relaxation:0,
    bendingDamping_s:viscous?3*relaxation:0,
    rodStretchDamping_s:viscous&&!bendingOnly?3*relaxation:0,
    rodBendingDamping_s:viscous?3*relaxation:0,
    rodTwistDamping_s:viscous?2*(1+material.poissonRatio)*relaxation:0,
    attachmentMultiplier:Number(material.sourceSettings.attachMultiplier??1e-3),
  };});
  const scripts: ClothSupportScript[] = source.scripts.map(script => ({
    kind: script.kind, group: script.group, start_s: script.start_s, end_s: script.end_s,
    vector: script.kind === 'translate' ? script.displacement_m! : script.axis!,
    ...(script.angle_rad === undefined ? {} : { angle_rad: script.angle_rad }),
    ...(script.origin_m === undefined ? {} : { origin_m: script.origin_m }),
    global: script.global, easing: script.easing, easeIn_s: script.easeStart_s, easeOut_s: script.easeEnd_s,
  }));
  const initialSources=new Set(new SourceEmissionScheduler(source.fields).advance(0,source.dt_s).filter(e=>e.start_s===0).map(e=>e.fieldIndex));
  const previewParticleGroupCount=source.fields.filter(f=>f.root&&f.sampled&&((f.usage==='solid')||initialSources.has(f.index))).reduce((n,f)=>Math.max(n,f.group+1),1);
  const changesTopology=source.variant?.overrides.some(o=>o.path==='geometry.solidFraction'),data=changesTopology?cachedClothTopology(source.asset):undefined;
  if(changesTopology&&!data)throw new Error(`Load ${id} with prepareWetClothSource before applying its explicit topology override`);
  return {
    schemaVersion: '1.0.0', id: source.id, geometry: data?{kind:'inline',data:wetClothVariantTopology(data,source.variant)}:source.asset, materials, scripts, appearance: 'source-preview',
    physics: {
      nonlinearDrag: Number(source.sourceSettings.liquid.useNonlinearDrag ?? 1) !== 0,
      dragExponent: Number(source.sourceSettings.liquid.yazdchiPower ?? 1.6),
      pressureOnSolid: Number(source.sourceSettings.liquid.applyPressureSolid ?? 1) !== 0,
      varyingSolidFraction: Number(source.sourceSettings.liquid.useVaryingFraction ?? 0) !== 0,
      porePressure: Number(source.sourceSettings.liquid.applyPorePressureSolid ?? 0) !== 0,
      manifoldSubsteps: Number(source.sourceSettings.integrator.manifoldsubsteps ?? 4),
      // Original nonlinear manifold floor compares mass/time in CGS units.
      manifoldDragFloorReferenceVolume_m3:1e-6,
      captureRate: Number(source.sourceSettings.liquid.elastoCaptureRate ?? 1),
      manifoldTransport: true,
      pressureOnManifold: Number(source.sourceSettings.liquid.applyPressureManifold ?? 1) !== 0,
      absorption: Number(source.sourceSettings.liquid.elastoCaptureRate ?? 1) > 0,
      solveSolid: Number(source.sourceSettings.liquid.solveSolid ?? 1) !== 0,
    },
    source: { url: source.source.url, sha256: source.source.sha256, configuration:source.variant?.kind==='paper-corrected'?'paper-corrected':source.variant?'interactive':'source',previewParticleGroupCount },
  };
}

/** Source audit only, never a runtime capability or reproduction gate.
 * Never hide unavailable external geometry behind a substitute. */
export function wetClothSourceAudit(id: string): { readonly ready: boolean; readonly reasons: readonly string[] } {
  const source = wetClothSource(id);
  const reasons = [
    ...source.externalDependencies.filter(d => !d.available_in_release && !d.recovered_from).map(d => `Missing source asset: ${d.path}`),
    ...source.auditNotes,
  ];
  return { ready: reasons.length === 0, reasons };
}
