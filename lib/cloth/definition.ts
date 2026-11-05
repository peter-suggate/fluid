import { composeFeatures, type FeatureDefinition } from "../framework/composition";
import { publicationPort } from "../framework/ports";
import type { ClothDocument, ClothState, ClothSurfacePublication } from "./types";
import type { ClothImpulsePublication } from "./coupling";
import type { UniformClothGridFields } from "../methods/uniform/uniform-cloth-grid-coupling";

export const CLOTH_DOCUMENT_PORT = publicationPort<ClothDocument>({ id: "cloth.document", representation: "cloth-document-v1", lifetime: "scene", unit: "SI" });
export const CLOTH_STATE_PORT = publicationPort<ClothState>({ id: "cloth.state", representation: "cloth-state-v1", lifetime: "generation", unit: "SI" });
export const CLOTH_SURFACE_PORT = publicationPort<ClothSurfacePublication>({ id: "cloth.surface", representation: "cloth-surface-v1", lifetime: "frame", unit: "m" });
export const CLOTH_NBFLIP_GRID_PORT = publicationPort<UniformClothGridFields>({ id: "fluid.cloth-grid", representation: "nbflip-forced-mac-mixture-v1", lifetime: "generation", unit: "SI" });
export const CLOTH_FLUID_IMPULSE_PORT = publicationPort<ClothImpulsePublication>({ id: "cloth.fluid-impulse", representation: "cloth-impulse-v1", lifetime: "frame", unit: "N s" });
/** A ledger is a paired committed-volume publication, not a capture request.
 * Transport cannot be composed until an actual free-water adapter provides it. */
export interface ClothWaterLedger { readonly generation:number; readonly tick:number; readonly freeDelta_m3:number; readonly boundDelta_m3:number; readonly externalDelta_m3:number }
export interface ClothLiquidSampler {sampleLiquidPhi(positions_m:ArrayLike<number>):Promise<Float64Array>;readFreeWaterVolume():Promise<number>}
export const CLOTH_LIQUID_SAMPLE_PORT=publicationPort<ClothLiquidSampler>({id:"fluid.cloth-liquid-sampler",representation:"native-nbflip-liquid-sampler-v1",lifetime:"generation",unit:"SI"});
export const CLOTH_WATER_LEDGER_PORT = publicationPort<ClothWaterLedger>({ id: "cloth.water-ledger", representation: "paired-water-ledger-v1", lifetime: "frame", unit: "m3" });

export const clothDocumentFeature: FeatureDefinition = { id: "cloth.document", provides: ["cloth.authored"], outputs: [CLOTH_DOCUMENT_PORT] };
export const clothMechanicsFeature: FeatureDefinition = { id: "cloth.mechanics", requires: ["cloth.authored"], provides: ["cloth.mechanical-state"], inputs: [{ port: CLOTH_DOCUMENT_PORT, provider: "cloth.document" }], outputs: [CLOTH_STATE_PORT] };
export const clothGeometryFeature: FeatureDefinition = { id: "cloth.geometry", requires: ["cloth.mechanical-state"], provides: ["cloth.editable-surface"], inputs: [{ port: CLOTH_STATE_PORT, provider: "cloth.mechanics" }], outputs: [CLOTH_SURFACE_PORT] };
export const clothContactFeature: FeatureDefinition = { id: "cloth.contact", requires: ["cloth.editable-surface", "cloth.mechanical-state"], provides: ["cloth.physical-contact"], inputs: [{ port: CLOTH_SURFACE_PORT, provider: "cloth.geometry" }] };

export const narrowBandClothGridFeature:FeatureDefinition={id:"fluid.narrowband-cloth-grid",provides:["fluid.narrowband-flip","fluid.forced-mac","fluid.mixture-pressure","fluid.local-support"],outputs:[CLOTH_NBFLIP_GRID_PORT]};
export const clothPorousCouplingFeature:FeatureDefinition={id:"cloth.porous-coupling",requires:["cloth.mechanical-state","fluid.narrowband-flip","fluid.forced-mac","fluid.mixture-pressure","fluid.local-support"],provides:["cloth.reciprocal-fluid-momentum","cloth.projected-fluid-pressure","cloth.eulerian-pore-pressure"],inputs:[{port:CLOTH_STATE_PORT,provider:"cloth.mechanics"},{port:CLOTH_NBFLIP_GRID_PORT,provider:"fluid.narrowband-cloth-grid"}],outputs:[CLOTH_FLUID_IMPULSE_PORT]};
export const clothPorousTransportFeature:FeatureDefinition={id:"cloth.porous-transport",requires:["cloth.mechanical-state","fluid.local-volume-exchange","cloth.projected-fluid-pressure"],provides:["cloth.conservative-water-transport"],inputs:[{port:CLOTH_STATE_PORT,provider:"cloth.mechanics"},{port:CLOTH_WATER_LEDGER_PORT,provider:"fluid.cloth-water-exchange"},{port:CLOTH_FLUID_IMPULSE_PORT,provider:"cloth.porous-coupling"}]};
export const clothLocalWaterExchangeFeature:FeatureDefinition={id:"fluid.cloth-water-exchange",requires:["fluid.narrowband-flip","fluid.local-support","cloth.mechanical-state"],provides:["fluid.local-volume-exchange","fluid.native-liquid-sampling"],inputs:[{port:CLOTH_NBFLIP_GRID_PORT,provider:"fluid.narrowband-cloth-grid"},{port:CLOTH_STATE_PORT,provider:"cloth.mechanics"}],outputs:[CLOTH_WATER_LEDGER_PORT,CLOTH_LIQUID_SAMPLE_PORT]};
/** Cohesion requires free-liquid evidence from the real grid in addition to
 * stored pore water. A saturation-only mechanics plugin cannot provide it. */
export const clothWetCohesionFeature:FeatureDefinition={id:"cloth.wet-cohesion",requires:["cloth.mechanical-state","cloth.physical-contact","fluid.narrowband-flip","fluid.local-volume-exchange","fluid.native-liquid-sampling"],provides:["cloth.liquid-bridge-cohesion"],inputs:[{port:CLOTH_LIQUID_SAMPLE_PORT,provider:"fluid.cloth-water-exchange"},{port:CLOTH_STATE_PORT,provider:"cloth.mechanics"},{port:CLOTH_NBFLIP_GRID_PORT,provider:"fluid.narrowband-cloth-grid"},{port:CLOTH_WATER_LEDGER_PORT,provider:"fluid.cloth-water-exchange"}]};

export const CLOTH_FEATURES = [clothDocumentFeature, clothMechanicsFeature, clothGeometryFeature, clothContactFeature] as const;
export function composeClothFeatures(additional: readonly FeatureDefinition[] = []) {
  return composeFeatures({ features: [...CLOTH_FEATURES, ...additional] });
}
