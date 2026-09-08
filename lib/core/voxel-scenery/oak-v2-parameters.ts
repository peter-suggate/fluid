import type { SceneryMaterial } from "../scenery-graph";

/** Saved alongside editable geometry. Only an explicit growth edit regenerates it. */
export interface OakV2Recipe {
  readonly version: 1;
  readonly parameters: OakV2Parameters;
  readonly bark: SceneryMaterial;
  readonly foliage: SceneryMaterial;
}

export const OAK_CONTROL_GROUPS = ["Specimen", "Crown", "Branches", "Twigs", "Foliage"] as const;
type Group = typeof OAK_CONTROL_GROUPS[number];
const control = (group: Group, label: string, initial: number, min: number, max: number, step: number, hint: string, unit?: string) =>
  ({ group, label, initial, min, max, step, hint, unit });

/** One set of bounds for document validation, the planner, and both editor surfaces. */
export const OAK_V2_CONTROLS = {
  seed: control("Specimen", "Seed", 4258, 0, 4294967295, 1, "Repeatable variation. The same seed and settings reproduce the same tree."),
  scale_m: control("Specimen", "Tree size", 1, .1, 8, .05, "Scale the generated tree about its planted base.", "m"),
  crownWidth: control("Crown", "Crown width", 1, .55, 1.5, .05, "Spread the boughs and shoot sites horizontally.", "×"),
  crownHeight: control("Crown", "Crown rise", 1, .65, 1.4, .05, "Vertical spread of the crown above the trunk.", "×"),
  trunkHeight: control("Crown", "Trunk height", 1, .65, 1.4, .05, "Raise the fork sites and crown together.", "×"),
  boughsPerTier: control("Crown", "Boughs per tier", 2, 1, 3, 1, "Four fork tiers; fewer boughs leave larger gaps."),
  sitesPerBough: control("Crown", "Shoots per bough", 12, 4, 12, 1, "Primary shoot sites before recursive twig splitting."),
  branchBend: control("Branches", "Branch bend", 1, 0, 1.8, .05, "Bow the centreline between connected fork endpoints.", "×"),
  woodScale: control("Branches", "Wood thickness", 1, .6, 1.6, .05, "Scale trunk and branch radii while conserving area at forks.", "×"),
  twigDepth: control("Twigs", "Fork generations", 3, 0, 3, 1, "Binary twig generations. Independent of voxel refinement depth."),
  twigLength: control("Twigs", "Twig reach", 1, .5, 1.6, .05, "Length of the first twig generation.", "×"),
  twigDecay: control("Twigs", "Length retention", .62, .45, .8, .01, "Each generation retains this fraction of its parent's length."),
  forkAngle: control("Twigs", "Fork angle", Math.atan2(.68, .72) * 180 / Math.PI, 25, 65, 1, "Angle of each child away from the parent direction.", "°"),
  upwardBias: control("Twigs", "Upward growth", .16, 0, .4, .02, "Upward bias added to each new twig direction."),
  leafScale: control("Foliage", "Leaf spray size", 1, .45, 1.5, .05, "Size of the small foliage fields attached to terminal twigs.", "×"),
  leafFlatten: control("Foliage", "Leaf spray height", .85, .4, 1, .05, "Flatten each spray vertically without changing its attachment."),
  leafThreshold: control("Foliage", "Leaf openness", .5, .35, .65, .01, "Higher values remove leaf mass and reveal more gaps."),
  clumpScale: control("Foliage", "Clump spacing", 1, .5, 1.8, .05, "Spacing of the broad noise features within each spray.", "×"),
  detailWeight: control("Foliage", "Fine breakup", .7, .35, .8, .01, "Weight of fine detail relative to broader leaf clusters."),
  interiorBias: control("Foliage", "Interior fill", .04, 0, .12, .01, "Add leaf density inside each spray."),
  showFoliage: control("Foliage", "Leaves", 1, 0, 1, 1, "Hide leaves to inspect the connected branch skeleton."),
} as const;
export type OakV2Parameter = keyof typeof OAK_V2_CONTROLS;
export type OakV2Parameters = Readonly<Record<OakV2Parameter, number>>;
export const OAK_V2_DEFAULTS = Object.freeze(Object.fromEntries(Object.entries(OAK_V2_CONTROLS)
  .map(([key, spec]) => [key, spec.initial])) as Record<OakV2Parameter, number>);

export function oakParameterErrors(input: unknown): string[] {
  if (!input || typeof input !== "object") return ["Oak v2 parameters must be an object"];
  const values = input as Record<string, unknown>;
  return Object.entries(OAK_V2_CONTROLS).flatMap(([key, spec]) => {
    const value = values[key];
    return typeof value !== "number" || !Number.isFinite(value) || value < spec.min || value > spec.max
      || (spec.step === 1 && key !== "forkAngle" && !Number.isInteger(value))
      ? [`Oak v2 ${key} must be ${spec.step === 1 && key !== "forkAngle" ? "an integer " : ""}in ${spec.min}..${spec.max}`] : [];
  });
}
export function oakParameters(patch: Partial<OakV2Parameters> = {}): OakV2Parameters {
  const values = { ...OAK_V2_DEFAULTS, ...patch };
  // Older callers used signed 32-bit seeds; retain their exact hash sequence.
  if (Number.isInteger(values.seed) && values.seed < 0 && values.seed >= -2147483648) values.seed >>>= 0;
  const errors = oakParameterErrors(values);
  if (!Number.isInteger(values.seed) || values.seed < 0 || values.seed > 0xffffffff) errors.push("Oak v2 needs a 32-bit integer seed");
  if (errors.length) throw new RangeError(errors.join("; "));
  return values;
}

export const OAK_V2_PRESETS = [
  { id: "fractal", label: "Fractal oak", hint: "The original v2 oak with three twig generations.", values: {} },
  { id: "open", label: "Open crown", hint: "Separated boughs, smaller leaf sprays, visible forks.", values: { sitesPerBough: 8, crownWidth: 1.2, leafScale: .75, leafThreshold: .56, twigLength: 1.2 } },
  { id: "spreading", label: "Spreading oak", hint: "A lower, wider crown with sweeping branches.", values: { crownWidth: 1.4, crownHeight: .75, trunkHeight: .8, branchBend: 1.4, twigLength: 1.2, forkAngle: 55 } },
  { id: "fine", label: "Fine tracery", hint: "Small foliage sprays and long, finely divided twigs.", values: { sitesPerBough: 8, leafScale: .55, leafThreshold: .58, twigLength: 1.4, twigDecay: .72, woodScale: .8 } },
] as const;

export function oakPrimitiveCount(p: OakV2Parameters): { branches: number; sprays: number } {
  const tips = 4 * p.boughsPerTier * p.sitesPerBough * 2 ** p.twigDepth;
  return { branches: 4 + 2 * tips - 4 * p.boughsPerTier, sprays: p.showFoliage ? tips : 0 };
}
