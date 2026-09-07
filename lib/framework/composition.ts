import type { ControlMetadata } from "./controls";
import { portsMatch, type PublicationPort, type PortRequirement } from "./ports";

/** Data-only feature composition. Domain hosts retain execution and resource ownership. */
export type UpdateImpact = "live" | "rebuild" | "reset";
export interface FeatureControl extends ControlMetadata {
  readonly id: string;
  readonly label: string;
  readonly kind: "number" | "choice" | "toggle" | "action" | "readout";
  readonly setting?: string;
  readonly unit?: string;
  readonly hint?: string;
  readonly min?: number;
  readonly max?: number;
  readonly step?: number;
  readonly options?: readonly { readonly value: string; readonly label: string; readonly hint?: string }[];
  readonly update?: UpdateImpact;
}
export interface FeaturePlacement {
  readonly slot: string;
  readonly control: string;
  readonly order?: number;
  readonly presentation?: "compact" | "expanded";
}
export interface VariantDefinition {
  readonly id: string;
  readonly point: string;
  readonly label?: string;
  readonly inputs?: readonly PortRequirement[];
  readonly outputs?: readonly PublicationPort<any>[];
  readonly requires?: readonly string[];
  readonly provides?: readonly string[];
  readonly update: UpdateImpact;
  readonly default?: boolean;
}
export interface FeatureDefinition {
  readonly id: string;
  readonly label?: string;
  readonly inputs?: readonly PortRequirement[];
  readonly outputs?: readonly PublicationPort<any>[];
  readonly requires?: readonly string[];
  readonly provides?: readonly string[];
  readonly controls?: readonly FeatureControl[];
  readonly placements?: readonly FeaturePlacement[];
  readonly variants?: readonly VariantDefinition[];
}
export interface FeatureCompositionInput {
  readonly features: readonly FeatureDefinition[];
  readonly selections?: Readonly<Record<string, string>>;
  readonly capabilities?: readonly string[];
}
export interface ResolvedControl extends FeatureControl { readonly feature: string }
export interface ResolvedPlacement extends FeaturePlacement { readonly feature: string }
export interface FeatureComposition {
  readonly features: readonly FeatureDefinition[];
  readonly variants: readonly VariantDefinition[];
  readonly controls: readonly ResolvedControl[];
  readonly placements: readonly ResolvedPlacement[];
  readonly capabilities: readonly string[];
}

/** Definitions are data. Snapshot them so later caller mutation cannot bypass validation. */
function immutableCopy<T>(value: T): T {
  if (Array.isArray(value)) return Object.freeze(value.map(item => immutableCopy(item))) as T;
  if (value !== null && typeof value === "object") {
    return Object.freeze(Object.fromEntries(Object.entries(value).map(([key, item]) => [key, immutableCopy(item)]))) as T;
  }
  return value;
}

/** Validate the complete configuration before a host mutates state or allocates resources. */
export function composeFeatures(input: FeatureCompositionInput): FeatureComposition {
  const features = immutableCopy(input.features);
  const ids = new Set<string>();
  const points = new Map<string, VariantDefinition[]>();
  const controls: ResolvedControl[] = [];
  const placements: ResolvedPlacement[] = [];
  const owners = new Map<string, string>();
  const provide = (capability: string, owner: string) => {
    const previous = owners.get(capability);
    if (previous !== undefined && previous !== owner) {
      throw new Error(`Ambiguous capability ${capability}: ${previous} and ${owner}`);
    }
    owners.set(capability, owner);
  };
  for (const capability of input.capabilities ?? []) provide(capability, "host");
  for (const feature of features) {
    if (!feature.id || ids.has(feature.id)) throw new Error(`Duplicate or empty feature ID: ${feature.id}`);
    ids.add(feature.id);
    for (const capability of feature.provides ?? []) provide(capability, feature.id);
    const controlIds = new Set<string>();
    for (const control of feature.controls ?? []) {
      if (!control.id || controlIds.has(control.id)) throw new Error(`Duplicate or empty control ${feature.id}/${control.id}`);
      controlIds.add(control.id);
      if (control.min !== undefined && control.max !== undefined && control.min > control.max) {
        throw new Error(`Invalid range ${feature.id}/${control.id}`);
      }
      controls.push(Object.freeze({ ...control, feature: feature.id }));
    }
    for (const placement of feature.placements ?? []) {
      if (!controlIds.has(placement.control)) throw new Error(`Unknown control ${feature.id}/${placement.control}`);
      if (!placement.slot) throw new Error(`Empty slot for ${feature.id}/${placement.control}`);
      if (placements.some(p => p.feature === feature.id && p.control === placement.control && p.slot === placement.slot)) {
        throw new Error(`Duplicate placement ${feature.id}/${placement.control} in ${placement.slot}`);
      }
      placements.push(Object.freeze({ ...placement, feature: feature.id }));
    }
    for (const variant of feature.variants ?? []) {
      const candidates = points.get(variant.point) ?? [];
      if (candidates.some(candidate => candidate.id === variant.id)) {
        throw new Error(`Duplicate variant ${variant.point}/${variant.id}`);
      }
      candidates.push(variant);
      points.set(variant.point, candidates);
    }
  }
  for (const point of Object.keys(input.selections ?? {})) {
    if (!points.has(point)) throw new Error(`Unknown variation point ${point}`);
  }
  const variants: VariantDefinition[] = [];
  for (const [point, candidates] of points) {
    const selectedId = input.selections?.[point];
    const defaults = candidates.filter(candidate => candidate.default);
    if (defaults.length > 1) throw new Error(`Multiple defaults for ${point}`);
    const selected = selectedId === undefined
      ? defaults[0] ?? (candidates.length === 1 ? candidates[0] : undefined)
      : candidates.find(candidate => candidate.id === selectedId);
    if (!selected) throw new Error(`Select a supported variant for ${point}${selectedId === undefined ? "" : `: ${selectedId}`}`);
    variants.push(selected);
    for (const capability of selected.provides ?? []) provide(capability, `${point}/${selected.id}`);
  }
  for (const owner of [...features, ...variants]) {
    for (const capability of owner.requires ?? []) {
      if (!owners.has(capability)) throw new Error(`${owner.id} requires missing capability ${capability}`);
    }
  }
  const connected = [...features, ...variants.map(variant => ({ ...variant, id: `${variant.point}/${variant.id}` }))];
  const providers = new Map(connected.map(feature => [feature.id, feature]));
  if (providers.size !== connected.length) throw new Error("Feature and variant provider IDs collide");
  for (const feature of connected) {
    const outputIds = (feature.outputs ?? []).map(port => port.id);
    if (new Set(outputIds).size !== outputIds.length) throw new Error(`Duplicate output port in ${feature.id}`);
    for (const requirement of feature.inputs ?? []) {
      const provider = providers.get(requirement.provider);
      if (!provider?.outputs?.some(output => portsMatch(requirement.port, output))) {
        throw new Error(`${feature.id} requires port ${requirement.port.id} from ${requirement.provider} with matching representation and lifetime`);
      }
    }
  }
  return Object.freeze({
    features: Object.freeze([...features]),
    variants: Object.freeze(variants),
    controls: Object.freeze(controls),
    placements: Object.freeze(placements.sort((a, b) => (a.order ?? 0) - (b.order ?? 0))),
    capabilities: Object.freeze([...owners.keys()]),
  });
}

/** Configuration identity is independent of control placement and presentation. */
export function compositionUpdateImpact(before: FeatureComposition, after: FeatureComposition): UpdateImpact {
  let impact: UpdateImpact = "live";
  const prior = new Map(before.variants.map(variant => [variant.point, variant]));
  for (const variant of after.variants) {
    const previous = prior.get(variant.point);
    if (previous?.id !== variant.id) {
      for (const change of [previous?.update, variant.update]) {
        if (change === "reset") return "reset";
        if (change === "rebuild") impact = "rebuild";
      }
    }
    prior.delete(variant.point);
  }
  for (const variant of prior.values()) {
    if (variant.update === "reset") return "reset";
    if (variant.update === "rebuild") impact = "rebuild";
  }
  return impact;
}
