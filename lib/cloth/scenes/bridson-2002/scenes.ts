import type { Quaternion, RigidPoseTrack, Vec3 } from "../../../core/model";
import type { ClothDocument, ClothMaterial } from "../../types";
import { equalNodeMass_kg, herringBoneSheet } from "./mesh";
import { BRIDSON_2002_STEP_S, BRIDSON_2002_VIDEO_FRAME_RATE, CLOTH, CURTAIN, SPHERE, TABLE, VOXEL } from "./parameters";
import { assertBridsonSolidResolved, yawQuaternion, type BridsonSolid } from "./solids";

/** A moment to capture and compare with the 2002 frames. */
export interface Bridson2002KeyMoment {
  readonly time_s: number;
  /** Frame of the 2002 video (24 fps) shown at this scene time. */
  readonly videoFrame: number;
  /** Reference image, relative to ~/code/me/fluid-scratch/bridson2002/ (not in git). */
  readonly reference: string;
  /** What the 2002 reference shows here. */
  readonly expect: string;
}

export interface Bridson2002Scene {
  readonly id: string;
  readonly figure: 1 | 2 | 3;
  /** The paper's caption, verbatim. */
  readonly caption: string;
  readonly duration_s: number;
  /** One implicit cloth step per frame; never substepped. */
  readonly step_s: number;
  /** Scene time at which the 2002 video's frame 0 was taken. */
  readonly videoStart_s: number;
  readonly cloth: ClothDocument;
  readonly sheet: { readonly columns: number; readonly rows: number; readonly width_m: number; readonly depth_m: number };
  /**
   * The paper's equal particle mass. The solver lumps mass by area today, so
   * this is carried beside the document until C1 accepts a per-vertex mass
   * (progress-lane-s.md, COORDINATION).
   */
  readonly nodeMass_kg: number;
  readonly selfFriction: number;
  readonly solids: readonly BridsonSolid[];
  /** Voxel lattice spacing every solid is baked at. */
  readonly voxelCell_m: number;
  readonly keyMoments: readonly Bridson2002KeyMoment[];
}

/** Fig 4 is a production robe on a digital character: reference only. Its
 * character, garment and solver were never published, so nothing stands in
 * for it. */
export const BRIDSON_2002_FIGURE_4 = {
  figure: 4,
  caption: "Frames from a production animation of a robe draped over a digital character.",
  referenceOnly: true,
  reason: "The character, robe pattern and the 'highly sophisticated model from an industrial production system' are unpublished; the 2002 PDF itself prints 'Not available in this version'.",
} as const;

const IDENTITY: Quaternion = { w: 1, x: 0, y: 0, z: 0 };
const v3 = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
const frame = (videoFrame: number, start_s = 0) => start_s + videoFrame / BRIDSON_2002_VIDEO_FRAME_RATE.value;

function material(id: string): ClothMaterial {
  return {
    id, arealDensity_kg_m2: CLOTH.arealDensity.value,
    stretchWarp_N_m: CLOTH.stretch.value, stretchWeft_N_m: CLOTH.stretch.value, shear_N_m: CLOTH.shear.value,
    poisson: CLOTH.poisson.value, compressionRatio: CLOTH.compressionRatio.value, bending_N_m: CLOTH.bending.value,
    damping_s: CLOTH.damping.value, thickness_m: CLOTH.thickness.value, strainLimit: CLOTH.strainLimit.value,
  };
}

/** Square-celled sheet: `nodesAcross` nodes over the short side, PhysBAM's
 * m = int(aspect * (n - 1)) + 1 over the long side. */
function sheetSize(short_m: number, long_m: number) {
  const rows = CLOTH.nodesAcross.value, columns = Math.floor((long_m / short_m) * (rows - 1) + 1e-9) + 1;
  return { columns, rows };
}

function groundSlab(id: string, friction: number, xMin: number, xMax: number, zMin: number, zMax: number): BridsonSolid {
  const d = VOXEL.groundDepth.value / 2;
  return { id, friction, shape: { kind: "box", halfExtents_m: v3((xMax - xMin) / 2, d, (zMax - zMin) / 2) },
    motion: { kind: "static", position_m: v3((xMin + xMax) / 2, -d, (zMin + zMax) / 2) } };
}

function finish(scene: Bridson2002Scene): Bridson2002Scene {
  for (const solid of scene.solids) assertBridsonSolidResolved(solid, scene.voxelCell_m, VOXEL.minimumFeatureCells.value);
  return scene;
}

function track(keyframes: readonly (readonly [number, Vec3, Quaternion?])[]): RigidPoseTrack {
  return { keyframes: keyframes.map(([time_s, position_m, orientation]) => ({ time_s, position_m, orientation: orientation ?? IDENTITY })) };
}

/**
 * Fig 1. The paper's caption has a second ball enter; section 9 has the one
 * sphere "move up and away". PhysBAM test 5 and the video agree with both:
 * one sphere lifts the curtain away (2-3.5 s), drops behind it (3.5-4 s),
 * and returns along the ground (4 s on) as the caption's "another ball".
 */
export function bridson2002Curtain(): Bridson2002Scene {
  const width = CURTAIN.side.value * CURTAIN.aspect.value, depth = CURTAIN.side.value;
  const { columns, rows } = sheetSize(depth, width), y = CURTAIN.height.value, z = CURTAIN.ballCentreZ.value;
  const geometry = herringBoneSheet({ columns, rows, width_m: width, depth_m: depth, origin_m: [0, y, 0],
    group: (i, j) => i === 0 && (j === 0 || j === rows - 1) ? 1 : 0 });
  const t1 = CURTAIN.ballStart.value, t2 = CURTAIN.ballRiseEnd.value, t3 = CURTAIN.ballDropEnd.value, end = CURTAIN.duration.value;
  // PhysBAM test 5 positions: (t-2, (t-2)/2, z), then down at 1.5 m/s, then -x at 1.5 m/s.
  const top = v3(t2 - t1, 0.5 * (t2 - t1), z), down = v3(top.x, top.y - 1.5 * (t3 - t2), z);
  const ball: BridsonSolid = { id: "ball", friction: CURTAIN.ballFriction.value, shape: { kind: "sphere", radius_m: CURTAIN.ballRadius.value },
    motion: { kind: "pose-track", track: track([[0, v3(0, 0, z)], [t1, v3(0, 0, z)], [t2, top], [t3, down],
      [end, v3(down.x - CURTAIN.ballReturnSpeed.value * (end - t3), 0, z)]]) } };
  const k = (f: number, expect: string, reference = `video/frames_cloth0/f${String(f).padStart(3, "0")}.png`) =>
    ({ time_s: frame(f), videoFrame: f, reference, expect });
  return finish({
    id: "bridson-2002-curtain", figure: 1,
    caption: "Initially, a curtain is draped over a ball on the ground. The ball moves, flipping the curtain over on top of itself producing stable folds and wrinkles using our static friction model. Another ball enters the scene and pushes through the complicated structure of folds before slipping underneath unraveling the curtain.",
    duration_s: end, step_s: BRIDSON_2002_STEP_S, videoStart_s: 0,
    cloth: { schemaVersion: "2.0.0", id: "bridson-2002-curtain-cloth", geometry: { kind: "inline", data: geometry },
      materials: [material("bridson-2002-cloth")], solver: { integrator: "bdf2", iterations: CLOTH.iterations.value } },
    sheet: { columns, rows, width_m: width, depth_m: depth },
    nodeMass_kg: equalNodeMass_kg(columns, rows, width, depth, CLOTH.arealDensity.value),
    selfFriction: CLOTH.selfFriction.value,
    solids: [groundSlab("ground", CURTAIN.groundFriction.value, -3.2, 2.4, -0.8, 1.8), ball],
    voxelCell_m: VOXEL.cell.value,
    keyMoments: [
      k(0, "flat sheet at the pins' height"),
      k(12, "sheet has fallen over the dome onto the ground"),
      k(48, "settled drape, ball about to move (Fig 1a)", "figs-lossless/fig-000.png"),
      k(72, "ball lifting the curtain up and away (Fig 1b; the panel's own frame is not identified)", "figs-lossless/fig-001.png"),
      k(96, "curtain flipped back over itself in a pile, back side up"),
      k(114, "ball re-enters along the ground and meets the pile (Fig 1c = Stanford test0_frame114)", "web/test0_frame114.jpg"),
      k(159, "unravelled onto the -x side, back side up (Fig 1d)", "figs-lossless/fig-003.png"),
    ],
  });
}

/** Fig 2. The black frustum is in the video and the figure but not the text. */
export function bridson2002RotatingSphere(): Bridson2002Scene {
  const side = SPHERE.side.value, { columns, rows } = sheetSize(side, side), y = SPHERE.height.value;
  const geometry = herringBoneSheet({ columns, rows, width_m: side, depth_m: side, origin_m: [-side / 2, y, -side / 2] });
  const centre = v3(0, SPHERE.sphereCentreY.value, 0), start = SPHERE.spinStart.value, end = SPHERE.duration.value;
  // Constant rate from spinStart; keyframes 1 s (pi/4) apart keep the slerp on its shortest arc.
  const keys: [number, Vec3, Quaternion][] = [[0, centre, IDENTITY]];
  for (let t = start; t < end + 1; t += 1) { const s = Math.min(t, end); keys.push([s, centre, yawQuaternion(SPHERE.spinRate.value * (s - start))]); if (s === end) break; }
  const sphere: BridsonSolid = { id: "sphere", friction: SPHERE.sphereFriction.value, shape: { kind: "sphere", radius_m: SPHERE.sphereRadius.value },
    motion: { kind: "pose-track", track: track(keys) } };
  const frustum: BridsonSolid = { id: "frustum", friction: SPHERE.frustumFriction.value,
    shape: { kind: "square-frustum", base_m: SPHERE.frustumBase.value, top_m: SPHERE.frustumTop.value, height_m: SPHERE.frustumHeight.value,
      footing_m: SPHERE.frustumFooting.value },
    motion: { kind: "static", position_m: v3(SPHERE.frustumX.value, 0, SPHERE.frustumZ.value) } };
  const k = (f: number, expect: string, reference = `video/frames_cloth1/f${String(f).padStart(3, "0")}.png`) =>
    ({ time_s: frame(f), videoFrame: f, reference, expect });
  return finish({
    id: "bridson-2002-rotating-sphere", figure: 2,
    caption: "The friction between a rotating sphere and a piece of cloth draped over it creates a complex structure of wrinkles and folds.",
    duration_s: end, step_s: BRIDSON_2002_STEP_S, videoStart_s: 0,
    cloth: { schemaVersion: "2.0.0", id: "bridson-2002-rotating-sphere-cloth", geometry: { kind: "inline", data: geometry },
      materials: [material("bridson-2002-cloth")], solver: { integrator: "bdf2", iterations: CLOTH.iterations.value } },
    sheet: { columns, rows, width_m: side, depth_m: side },
    nodeMass_kg: equalNodeMass_kg(columns, rows, side, side, CLOTH.arealDensity.value),
    selfFriction: CLOTH.selfFriction.value,
    solids: [groundSlab("ground", SPHERE.groundFriction.value, -1.6, 1.6, -1.6, 1.6), sphere, frustum],
    voxelCell_m: VOXEL.cell.value,
    keyMoments: [
      k(0, "flat sheet just above the sphere"),
      k(12, "draped over the sphere and the frustum corner, before the spin"),
      k(48, "the corner on the frustum has been pulled off"),
      k(53, "skirt wound about the sphere (Fig 2a = Stanford test1_frame053)", "web/test1_frame053.jpg"),
      k(120, "winding folds travelling about the vertical"),
      k(212, "last frame: tightly wound folds"),
    ],
  });
}

/** Fig 3. Video geometry scaled by the PhysBAM sphere radius; the sharp leg
 * tops carry the thesis's rounding as a 45 degree bevel. */
export function bridson2002Tablecloth(): Bridson2002Scene {
  const side = TABLE.side.value, { columns, rows } = sheetSize(side, side), y = TABLE.height.value;
  const geometry = herringBoneSheet({ columns, rows, width_m: side, depth_m: side, origin_m: [-side / 2, y, -side / 2] });
  const t0 = TABLE.videoStart.value, r = TABLE.sphereRadius.value, y0 = TABLE.sphereYAtVideoStart.value, v = TABLE.descentSpeed.value;
  const land = t0 + (y0 - r) / v, end = t0 + TABLE.duration.value;
  const sphere: BridsonSolid = { id: "sphere", friction: TABLE.sphereFriction.value, shape: { kind: "sphere", radius_m: r },
    motion: { kind: "pose-track", track: track([[0, v3(0, y0 + v * t0, 0)], [land, v3(0, r, 0)], [end, v3(TABLE.rollSpeed.value * (end - land), r, 0)]]) } };
  const s = TABLE.legSpacing.value / 2;
  const legs: BridsonSolid[] = [[1, 1], [1, -1], [-1, 1], [-1, -1]].map(([a, b], n) => ({
    id: `leg-${n}`, friction: TABLE.legFriction.value,
    shape: { kind: "chamfered-post", width_m: TABLE.legWidth.value, height_m: TABLE.legHeight.value, chamfer_m: TABLE.legChamfer.value },
    motion: { kind: "static", position_m: v3(a! * s, 0, b! * s) } }));
  const k = (f: number, expect: string, reference = `video/frames_cloth2/f${String(f).padStart(3, "0")}.png`) =>
    ({ time_s: frame(f, t0), videoFrame: f, reference, expect });
  return finish({
    id: "bridson-2002-tablecloth", figure: 3,
    caption: "A tablecloth draped over table legs with no tabletop is dragged to the ground by a descending sphere.",
    duration_s: end, step_s: BRIDSON_2002_STEP_S, videoStart_s: t0,
    cloth: { schemaVersion: "2.0.0", id: "bridson-2002-tablecloth-cloth", geometry: { kind: "inline", data: geometry },
      materials: [material("bridson-2002-cloth")], solver: { integrator: "bdf2", iterations: CLOTH.iterations.value } },
    sheet: { columns, rows, width_m: side, depth_m: side },
    nodeMass_kg: equalNodeMass_kg(columns, rows, side, side, CLOTH.arealDensity.value),
    selfFriction: CLOTH.selfFriction.value,
    solids: [groundSlab("ground", TABLE.groundFriction.value, -1.4, 2.4, -1.4, 1.4), ...legs, sphere],
    voxelCell_m: VOXEL.cell.value,
    keyMoments: [
      k(0, "settled drape over the four leg tops, corners on the ground", "video/full_t00.png"),
      k(30, "sphere descending above the cloth"),
      k(43, "sphere meets the cloth"),
      k(56, "cloth dragged down between the legs (Fig 3b = Stanford test2_frame056)", "web/test2_frame056.jpg"),
      k(75, "sphere on the ground, leaving along +x between two legs"),
      k(92, "last frame: cloth pulled off the far legs (Fig 3d)", "figs-lossless/fig-010.png"),
    ],
  });
}

export const BRIDSON_2002_SCENES = [bridson2002Curtain, bridson2002RotatingSphere, bridson2002Tablecloth] as const;
