/**
 * The pond's rim, two ways, at a distance that can actually settle the question.
 *
 *     FLUID_SVO_DRY_FRAME_SCENE_MODULE=tools/preview/coping.ts
 *
 * Three switches, all environment variables so one build renders the whole
 * comparison:
 *
 *   FLUID_COPING_FORM=heightfield|swept   the shipped vessel, or a swept solid
 *   FLUID_COPING_VIEW=hero|rim|macro      the hero framing, or 26 cm, or 13 cm
 *   FLUID_COPING_SET=full|bare            the whole set, or vessel and hose only
 *
 * `bare` exists because the stone set is seated on the ground the *shipped*
 * vessel bakes, and the swept form rebakes it: leaving the pebble courses in
 * would compare two beddings as well as two rims, and they sit exactly where the
 * meeting being judged is. The default view is `rim`, because the whole question
 * is the edge and a wide shot cannot answer it.
 *
 * Everything goes through `heroPreviewScene` — see `tools/preview/README.md`.
 * The document's terrain is then replaced, which is a preview's business and not
 * an edit to the hero scene: the preset builds a fresh document per call.
 */
import type { CameraState, SceneDescription } from "../../lib/model";
import { cameraPosition } from "../../lib/math";
import type { SceneryNode } from "../../lib/scenery-graph";
import { terrainHeightAt } from "../../lib/terrain";
import {
  HERO_GARDEN_CONTAINER,
  HERO_GARDEN_TERRAIN_SAMPLE_M,
  HERO_GARDEN_VESSEL,
} from "../../lib/hero-garden-scene";
import { bakePondVesselTerrain, pondVesselPlanCurve } from "../../lib/voxel-scenery/pond-vessel";
import {
  SWEPT_COPING_POND_BULLNOSE,
  sweptCopingNodes,
  sweptCopingSection,
} from "../../lib/voxel-scenery/swept-coping";
import { heroPreviewCamera, heroPreviewScene } from "./hero-still";

const FORM = process.env.FLUID_COPING_FORM === "heightfield" ? "heightfield" : "swept";
const VIEW = process.env.FLUID_COPING_VIEW ?? "rim";
const BARE = process.env.FLUID_COPING_SET !== "full";

const number = (name: string, fallback: number) => {
  const value = Number(process.env[name]);
  return Number.isFinite(value) ? value : fallback;
};

/**
 * The form under test, with every shape number overridable.
 *
 * Not tuning for its own sake. The first swept render came back banded like a
 * caterpillar and there were two candidate causes — a section modulating faster
 * than the rail can carry, or the joints themselves — which no amount of looking
 * at one frame separates. `FLUID_COPING_VARIATION=0 FLUID_COPING_RELIEF=0`
 * renders a perfectly uniform tube, and it banded identically: the modulation was
 * innocent and the joints were not. `FLUID_COPING_SAMPLES` then settled what to
 * do about it.
 */
const FORM_UNDER_TEST = {
  ...SWEPT_COPING_POND_BULLNOSE,
  crestHeight_m: number("FLUID_COPING_CREST", SWEPT_COPING_POND_BULLNOSE.crestHeight_m),
  undercut: number("FLUID_COPING_UNDERCUT", SWEPT_COPING_POND_BULLNOSE.undercut),
  sectionVariation: number("FLUID_COPING_VARIATION", SWEPT_COPING_POND_BULLNOSE.sectionVariation),
  crestVariation: number("FLUID_COPING_VARIATION", SWEPT_COPING_POND_BULLNOSE.crestVariation),
  relief_m: number("FLUID_COPING_RELIEF", SWEPT_COPING_POND_BULLNOSE.relief_m),
  segmentStride: Math.max(1, Math.round(number("FLUID_COPING_STRIDE", 1))),
};

/**
 * How finely the rail is polylined, in samples per plan lobe.
 *
 * The knob that settles the chain's real question. A cone chain's *silhouette*
 * is exact at any resolution — the sagitta of a 25 mm chord on this plan is
 * 0.18 mm — but its shading normal steps by the whole turn angle at every joint,
 * and 112 points around this pond is 3.2 degrees a joint. Sixteen is the bake's
 * resolution; four times that is 0.8 degrees and 896 records.
 */
const RAIL_SAMPLES_PER_LOBE = Math.max(2, Math.round(number("FLUID_COPING_SAMPLES", 16)));

const SECTION = sweptCopingSection(FORM_UNDER_TEST);

/**
 * The vessel the swept form bakes: the same pond with its crest omitted.
 *
 * `rimHalfWidth_m` comes down to the solid's own footprint at the same time,
 * because with no crest that number no longer describes a coping — it is purely
 * where the inner face starts falling. Leaving it at the bullnose's 55 mm would
 * put a 23 mm shelf of dry plaster between the rim's inner foot and the top of
 * the wall, which the reference does not have.
 */
const SWEPT_VESSEL = {
  ...HERO_GARDEN_VESSEL,
  crest: "flat" as const,
  rimHalfWidth_m: SECTION.groundHalfWidth_m,
};

function copingScene(): SceneDescription {
  const scene = heroPreviewScene();
  const graph = scene.scenery;
  if (!graph) throw new Error("The hero preset must carry a scenery graph");

  let nodes: readonly SceneryNode[] = graph.nodes;
  let terrain = scene.terrain;
  if (!terrain) throw new Error("The hero preset must carry a terrain description");

  if (FORM === "swept") {
    terrain = {
      ...terrain,
      grid: bakePondVesselTerrain(SWEPT_VESSEL, HERO_GARDEN_CONTAINER, HERO_GARDEN_TERRAIN_SAMPLE_M),
    };
    const ground = (x_m: number, z_m: number) => terrainHeightAt(terrain, x_m, z_m);
    nodes = [
      ...nodes,
      ...sweptCopingNodes({
        ...FORM_UNDER_TEST,
        key: "coping",
        rail: pondVesselPlanCurve(SWEPT_VESSEL, RAIL_SAMPLES_PER_LOBE),
        groundHeightAt: ground,
        material: { palette: "stone", value: 0.92 },
        seed: HERO_GARDEN_VESSEL.seed ^ 0x00c0_9179,
      }),
    ];
  }

  if (BARE) {
    nodes = nodes.filter((node) => node.kind === "terrain-shell"
      || node.id.startsWith("hose/")
      || node.id.startsWith("coping/"));
  }

  return { ...scene, terrain, scenery: { ...graph, nodes } };
}

/**
 * Where the near rim is, found rather than authored.
 *
 * The rail is the pond's own plan curve, so the point of it closest to the hero
 * eye is the arc the hero framing shows nearest the camera — which is the one
 * whose meeting with the plaster this preview exists to look at. Deriving it
 * means the crops still land on the rim if the vessel's control points move.
 */
function nearestRimPoint(): { x: number; y: number; z: number } {
  const hero = { ...heroPreviewCamera() } as CameraState;
  const eye = cameraPosition(hero);
  const rail = pondVesselPlanCurve(SWEPT_VESSEL);
  let best = rail[0], bestDistance = Infinity;
  for (const point of rail) {
    const distance = Math.hypot(point[0] - eye.x, point[1] - eye.z);
    if (distance < bestDistance) { bestDistance = distance; best = point; }
  }
  return { x: best[0], y: HERO_GARDEN_VESSEL.groundHeight_m + FORM_UNDER_TEST.crestHeight_m, z: best[1] };
}

/**
 * Close enough that the rim is the frame.
 *
 * The dry shader builds its primary rays with a fixed vertical half-tangent of
 * 0.72 — about 72 degrees vertical and past 100 horizontal at this aspect — so
 * "close up" is the only zoom there is, and at 0.26 m that lens still shows the
 * far bank and the horizon behind it. 0.15 m puts a 190 mm-tall frame on a
 * 55 mm rim, which is a crop rather than a view of the pond.
 *
 * The elevation is a compromise and worth naming as one. The undercut reads
 * best from below the crest and the meeting line reads best from above it;
 * twenty degrees shows the flank turning under while still looking down onto the
 * plaster the rim is set into.
 */
function copingCamera(): Partial<CameraState> {
  const hero = heroPreviewCamera();
  if (VIEW === "hero") return hero;
  return {
    ...hero,
    elevation_rad: number("FLUID_COPING_ELEVATION", 0.35),
    distance_m: number("FLUID_COPING_DISTANCE", VIEW === "macro" ? 0.09 : 0.15),
    target_m: nearestRimPoint(),
  };
}

export const createScene = copingScene;
export const camera = copingCamera();
