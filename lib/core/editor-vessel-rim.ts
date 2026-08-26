import {
  pickExcluded,
  type EditorEntity,
  type EditorEntityDefinition,
} from "./editor-entity";
import { rayProbeAction } from "./editor-probe-actions";
import type { PondVesselSpec } from "./voxel-scenery/pond-vessel";
import { pondVesselPlanCurve } from "./voxel-scenery/pond-vessel";
import { rimBandReach_m, sceneVessel, sceneVessels, vesselRimAt } from "./vessel-rim-controls";
import { intersectAuthoredTerrain } from "./terrain";

/**
 * Selecting the pond's coping.
 *
 * The rim is not a scenery node: it is the vessel's own crest, a section of the
 * procedural terrain. So the click target is the coping *band* — the strip of
 * ground within the crest's half-width of the vessel's plan curve — and the
 * pick is a terrain ray-march followed by a plan-distance test, which makes
 * what is clickable exactly the raised ring that is drawn.
 *
 * No handles: the rim is not movable — the stones, the path and the terraces
 * are all laid against its plan curve — and its editable properties are the
 * crest section's, which live on the rim flyout's dials rather than on a gizmo.
 */

export const VESSEL_RIM_SELECTION_PREFIX = "vessel-rim-";

export function vesselRimSelectionId(name: string): string {
  return `${VESSEL_RIM_SELECTION_PREFIX}${name}`;
}

export function vesselNameFromSelection(selectionId: string): string | undefined {
  return selectionId.startsWith(VESSEL_RIM_SELECTION_PREFIX)
    ? selectionId.slice(VESSEL_RIM_SELECTION_PREFIX.length) : undefined;
}

function vesselRimEntityFor(name: string, spec: PondVesselSpec): EditorEntity {
  // The outline is the band's world bounds: the wandered plan curve, padded by
  // the band's own reach, from the plaster to the crest's tallest swell.
  const reach = rimBandReach_m(spec);
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const [x, z] of pondVesselPlanCurve(spec)) {
    minX = Math.min(minX, x - reach); maxX = Math.max(maxX, x + reach);
    minZ = Math.min(minZ, z - reach); maxZ = Math.max(maxZ, z + reach);
  }
  const baseY = spec.groundHeight_m;
  const topY = spec.groundHeight_m + spec.rimHeight_m * (1 + spec.sectionHeightVariation);
  const origin_m = { x: 0.5 * (minX + maxX), y: 0.5 * (baseY + topY), z: 0.5 * (minZ + maxZ) };
  return {
    selection: { kind: "vessel-rim", id: vesselRimSelectionId(name) },
    label: `RIM · ${name}`,
    tone: "prop",
    frame: { origin_m, orientation: { w: 1, x: 0, y: 0, z: 0 } },
    box: {
      min: { x: minX - origin_m.x, y: baseY - origin_m.y, z: minZ - origin_m.z },
      max: { x: maxX - origin_m.x, y: topY - origin_m.y, z: maxZ - origin_m.z },
    },
    sizeLabel: `crest ${(spec.rimHeight_m * 1000).toFixed(0)} mm · band ${(2 * spec.rimHalfWidth_m * 1000).toFixed(0)} mm`,
    handles: [],
    draftSubject: "terrain",
    editLabel: () => `Adjusted ${name} rim`,
  };
}

export const vesselRimEntity: EditorEntityDefinition = {
  kind: "vessel-rim",
  instances: (context) => sceneVessels(context.scene)
    .map(([name, spec]) => vesselRimEntityFor(name, spec)),
  /**
   * The rim has no verb of its own — its dials are on the flyout `Edit` opens —
   * but it is the only way the ring reaches the *terrain*, and terrain is the
   * solid a hero frame spends most of its primary visibility on. So the one
   * thing declared here is the ray probe.
   */
  actions: (_context, target) => [rayProbeAction(target)],
  find: (context, id) => {
    const name = vesselNameFromSelection(id);
    const spec = name === undefined ? undefined : sceneVessel(context.scene, name);
    return name !== undefined && spec ? vesselRimEntityFor(name, spec) : undefined;
  },
  pick: (context, ray, exclude) => {
    const terrain = context.scene.terrain;
    if (!terrain) return undefined;
    const c = context.scene.container;
    const hit = intersectAuthoredTerrain(
      terrain, ray.origin, ray.direction, Math.max(c.width_m, c.height_m, c.depth_m));
    if (!hit || !(hit.t_m > 0)) return undefined;
    const name = vesselRimAt(context.scene, hit.position_m.x, hit.position_m.z);
    if (name === undefined) return undefined;
    if (pickExcluded(exclude, "vessel-rim", vesselRimSelectionId(name))) return undefined;
    return {
      selection: { kind: "vessel-rim", id: vesselRimSelectionId(name) },
      distance_m: hit.t_m,
    };
  },
};
