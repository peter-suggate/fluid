/**
 * How the Bridson 2002 scenes look: the 2002 video's own set, not a studio.
 * Saturated magenta cloth, a pale stone ground under a black sky, near-black frustum and legs, and one key light whose shadows read
 * on the ground. Cameras are the video's framings (frames_cloth{0,1,2}).
 *
 * Every visible surface here is ordinary scene geometry: the ground is the
 * scene's voxel slab (the surface the cloth lands on), the frustum and legs
 * its procedural bake, so all of it hit-tests, edits and collides as one.
 */
import type { CameraState, SceneDescription, Vec3 } from "../../../core/model";
import type { SceneryGraph } from "../../../core/scenery-graph";
import { VOXEL_MATERIAL_IDS } from "../../../core/voxel-scene";
import type { ClothVector } from "../../types";

/** sRGB #FF00FF: the cloth of every 2002 frame. */
export const BRIDSON_2002_CLOTH_COLOR_LINEAR: ClothVector = [1, 0, 1];
/** The voxel material of the ground slab. */
export const BRIDSON_2002_GROUND_MATERIAL = VOXEL_MATERIAL_IDS.marble;
/** The voxel material of every other static solid (frustum, legs). */
export const BRIDSON_2002_SOLID_MATERIAL = VOXEL_MATERIAL_IDS.obsidian;

export interface Bridson2002Look {
  /** Camera in the paper's frame (scene = paper + layout translation). */
  readonly camera: { readonly azimuth_rad: number; readonly elevation_rad: number; readonly distance_m: number; readonly target_m: Vec3; readonly tanHalfFov: number };
  /** Unit direction toward the key light, scene axes. */
  readonly keyLight: readonly [number, number, number];
}

const unit = (v: readonly [number, number, number]): [number, number, number] => {
  const n = Math.hypot(...v);
  return [v[0] / n, v[1] / n, v[2] / n];
};

/**
 * Framings read off the 2002 video (horizon height, subject scale and side):
 * the curtain at cloth0 f040, low and to the +x+z side of the hanging wall;
 * the sphere at cloth1 f018, at mound height in front (+z) with the frustum to
 * the left; the table at cloth2 f010, a little above leg mid-height and turned
 * about 35 degrees. The key light is behind and to the right of each camera,
 * high, as the shadows in those frames fall to the front-left.
 */
export const BRIDSON_2002_LOOKS: Readonly<Record<string, Bridson2002Look>> = {
  "bridson-2002-curtain": {
    camera: { azimuth_rad: 0.62, elevation_rad: 0.14, distance_m: 2.3, target_m: { x: 0.25, y: 0.22, z: 0.5 }, tanHalfFov: 0.5 },
    keyLight: unit([0.55, 0.8, -0.3]),
  },
  "bridson-2002-rotating-sphere": {
    camera: { azimuth_rad: 0.1, elevation_rad: 0.02, distance_m: 2.6, target_m: { x: -0.3, y: 0.42, z: 0 }, tanHalfFov: 0.5 },
    keyLight: unit([0.5, 0.85, -0.25]),
  },
  "bridson-2002-tablecloth": {
    camera: { azimuth_rad: 0.6, elevation_rad: 0.06, distance_m: 2.4, target_m: { x: 0.05, y: 0.4, z: 0 }, tanHalfFov: 0.5 },
    keyLight: unit([0.6, 0.8, -0.1]),
  },
};

export function bridson2002Look(id: string): Bridson2002Look {
  const look = BRIDSON_2002_LOOKS[id];
  if (!look) throw new Error(`Bridson 2002: no look for scene "${id}"`);
  return look;
}

/** The look's camera in scene coordinates. */
export function bridson2002LookCamera(look: Bridson2002Look, translation_m: Vec3): Partial<CameraState> {
  const c = look.camera, t = c.target_m;
  return { azimuth_rad: c.azimuth_rad, elevation_rad: c.elevation_rad, distance_m: c.distance_m, tanHalfFov: c.tanHalfFov,
    target_m: { x: t.x + translation_m.x, y: t.y + translation_m.y, z: t.z + translation_m.z } };
}

/**
 * A black sky with one key light. The backdrop radiance is near zero, so the
 * sky and its reflections are black; the diffuse environment is scaled back up
 * so the shadowed ground keeps the grey fill the 2002 frames show.
 */
export function bridson2002Lighting(look: Bridson2002Look): NonNullable<SceneDescription["lighting"]> {
  const sky: [number, number, number] = [0.002, 0.002, 0.002];
  return {
    directional: { direction: look.keyLight, colorLinear: [1, 1, 1], intensity: 3.2 },
    environment: { cloudCoverage: 0, diffuseScale: 120, specularScale: 1, lowerRadianceLinear: sky, upperRadianceLinear: sky, accentRadianceLinear: [0, 0, 0] },
    // Reinhard keeps the cloth the saturated magenta of the frames (AgX desaturates it toward pink).
    grade: { exposure: 1, toneCurve: "reinhard" },
  };
}

/**
 * No enclosure: the black sky shows wherever the ground slab does not.
 *
 * The 2002 ground runs to the horizon. Continuing the slab's stone past the
 * container as scenery was tried and does not fit at this lattice: at 1 cm
 * cells a 6 m ring needs 4.2M surface-mesh vertices against a 2.1M arena (a
 * 60 m one exhausts the 4 GB JS heap), so the ground ends at the slab edge.
 */
export function bridson2002Scenery(): SceneryGraph {
  return {
    palettes: { stage: { tint: [1, 1, 1] } },
    nodes: [{ kind: "room-shell", id: "shell", materialModel: "room", faces: [],
      floor: { palette: "stage", value: 0.62 }, wall: { palette: "stage", value: 0.62 }, ceiling: { palette: "stage", value: 0.62 } }],
  };
}
