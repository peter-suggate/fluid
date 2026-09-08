import type { EditorControlGroup } from "./editor-entity";
import { normalizeControlNumber } from "../framework/controls";
import type { SceneDescription } from "./model";
import { findSceneryNode, withSceneryNode } from "./scenery-edit";
import type { SceneryGroupNode, SceneryNode, SceneryMaterial } from "./scenery-graph";
import { planOakV2 } from "./voxel-scenery/oak-v2";
import { OAK_CONTROL_GROUPS, OAK_V2_CONTROLS, OAK_V2_DEFAULTS, OAK_V2_PRESETS, oakParameters, oakPrimitiveCount,
  type OakV2Parameter, type OakV2Parameters, type OakV2Recipe } from "./voxel-scenery/oak-v2-parameters";

export function isEditableOak(node: SceneryNode | undefined): node is SceneryGroupNode & { oak: OakV2Recipe } {
  return node?.kind === "group" && node.oak?.version === 1;
}

/** Compare the SAME authored tree at different sample depths, without reloading a preset. */
export function withTreePreviewDepth(scene: SceneDescription, depth: number): SceneDescription {
  if (!Number.isInteger(depth) || depth < 0 || depth > 3) throw new RangeError("Tree preview depth must be in 0..3");
  if (scene.systems?.fluid !== false) throw new Error("Turn water off before comparing scenery voxel depths");
  return { ...scene, voxelDomain: { ...scene.voxelDomain,
    detailCellSize_m: scene.voxelDomain.finestCellSize_m / 2 ** depth,
    environmentRefinementBaseCellSize_m: undefined,
  } };
}

/** Explicit regeneration preserves the object's transform, identity and material choices. */
export function withOakParameters(scene: SceneDescription, id: string, patch: Partial<OakV2Parameters>): SceneDescription {
  return withSceneryNode(scene, id, node => {
    if (!isEditableOak(node)) return node;
    const parameters = oakParameters({ ...node.oak.parameters, ...patch });
    const generated = planOakV2({ key: node.id, ...parameters, bark: node.oak.bark, foliage: node.oak.foliage }).node;
    return { ...node, children: generated.children, oak: generated.oak };
  });
}

export const OAK_MATERIALS = [
  { id: "natural", label: "Natural", bark: { colorLinear: [.16, .09, .045], surface: "architectural" }, foliage: { colorLinear: [.10, .22, .035], surface: "foliage" } },
  { id: "clay", label: "Clay", bark: { colorLinear: [.78, .76, .70], surface: "architectural" }, foliage: { colorLinear: [.88, .87, .82], surface: "foliage" } },
] as const satisfies readonly { id: string; label: string; bark: SceneryMaterial; foliage: SceneryMaterial }[];

export function withOakMaterial(scene: SceneDescription, id: string, style: typeof OAK_MATERIALS[number]): SceneDescription {
  return withSceneryNode(scene, id, node => {
    if (!isEditableOak(node)) return node;
    // Recolour stored geometry directly; material edits must not erase hand-sculpted leaves.
    const recolour = (part: SceneryNode): SceneryNode => {
      if (part.kind === "group") return { ...part, children: part.children.map(recolour) };
      const material = part.tags?.includes("foliage") ? style.foliage : style.bark;
      if (part.kind === "recursive-shape") return { ...part, material, children: part.children?.map(child => recolour(child) as typeof child) };
      return "material" in part ? { ...part, material } : part;
    };
    return { ...node, children: node.children.map(recolour), oak: { ...node.oak, bark: style.bark, foliage: style.foliage } };
  });
}

/** Declarations consumed by the regular object inspector and voxel editing shelf. */
export function oakTreeControlGroups(scene: SceneDescription, id: string): readonly EditorControlGroup[] {
  const node = findSceneryNode(scene, id);
  if (!isEditableOak(node)) return [];
  const parameters = node.oak.parameters;
  const count = oakPrimitiveCount(parameters);
  return OAK_CONTROL_GROUPS.map(label => ({
    id: `oak-${label.toLowerCase()}`, label,
    fields: (Object.entries(OAK_V2_CONTROLS) as [OakV2Parameter, typeof OAK_V2_CONTROLS[OakV2Parameter]][])
      .filter(([key, control]) => control.group === label && key !== "showFoliage")
      .map(([key, control]) => ({ ...control, id: `oak-${key}`, value: parameters[key],
        // A 32-bit seed needs exact entry, not a four-billion-stop scrub.
        max: key === "seed" ? undefined : control.max,
        apply: (value: number) => withOakParameters(scene, id, {
          [key]: normalizeControlNumber(value, parameters[key], control, control.step === 1 && key !== "forkAngle"),
        }) })),
    choices: label === "Specimen" ? [
      { id: "oak-preset", label: "Growth preset", value: OAK_V2_PRESETS.find(preset => {
        const candidate = { ...OAK_V2_DEFAULTS, ...preset.values };
        return (Object.keys(candidate) as OakV2Parameter[]).every(key => ["seed", "scale_m", "showFoliage"].includes(key) || candidate[key] === parameters[key]);
      })?.id ?? "custom", options: OAK_V2_PRESETS.map(preset => ({ ...preset,
        apply: () => withOakParameters(scene, id, { ...OAK_V2_DEFAULTS, ...preset.values, seed: parameters.seed, scale_m: parameters.scale_m, showFoliage: parameters.showFoliage }) })) },
      { id: "oak-seed-action", label: "Variation", value: "current", options: [
        { id: "next", label: "Next seed", hint: "Generate a repeatable new variation.", apply: () => withOakParameters(scene, id, { seed: (parameters.seed + 1) >>> 0 }) },
      ] },
      { id: "oak-material", label: "Colour", value: OAK_MATERIALS.find(style => JSON.stringify(node.oak.foliage) === JSON.stringify(style.foliage))?.id ?? "custom",
        options: OAK_MATERIALS.map(style => ({ id: style.id, label: style.label, apply: () => withOakMaterial(scene, id, style) })) },
    ] : label === "Foliage" ? [{ id: "oak-leaves", label: "Show", value: parameters.showFoliage ? "leaves" : "bare", options: [
      { id: "leaves", label: "Leaves", apply: () => withOakParameters(scene, id, { showFoliage: 1 }) },
      { id: "bare", label: "Branches", apply: () => withOakParameters(scene, id, { showFoliage: 0 }) },
    ] }] : [],
    summary: label === "Specimen" ? `${count.branches.toLocaleString()} branches · ${count.sprays.toLocaleString()} leaf sprays. Growth edits regenerate the tree; undo restores previous geometry. Position and colour are kept.`
      : label === "Twigs" ? "Fork generations change tree geometry. Use scene refinement to change voxel resolution."
      : label === "Foliage" ? "Small leaf sprays attach to the tips. Trees are decorative scenery and do not block water." : undefined,
  }));
}
