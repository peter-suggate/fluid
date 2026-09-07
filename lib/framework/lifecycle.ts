import { compositionUpdateImpact, type FeatureComposition, type UpdateImpact } from "./composition";

export function strongestImpact(...impacts: readonly UpdateImpact[]): UpdateImpact {
  return impacts.includes("reset") ? "reset" : impacts.includes("rebuild") ? "rebuild" : "live";
}

/** Resolve changes before mutating a store or announcing GPU work. */
export function configurationChangeImpact(
  before: FeatureComposition, after: FeatureComposition,
  previous: Readonly<Record<string, unknown>>, next: Readonly<Record<string, unknown>>,
  settings: readonly { readonly key: string; readonly impact: UpdateImpact }[],
): UpdateImpact | "none" {
  const changed = settings.filter(setting => !Object.is(previous[setting.key], next[setting.key]));
  const variantChanged = before.variants.length !== after.variants.length || before.variants.some(
    prior => !after.variants.some(candidate => candidate.point === prior.point && candidate.id === prior.id));
  if (!changed.length && !variantChanged) return "none";
  return strongestImpact(compositionUpdateImpact(before, after), ...changed.map(setting => setting.impact));
}
