/**
 * Every number the Bridson, Fedkiw and Anderson (SIGGRAPH 2002) scenes use,
 * each with where it came from. The scene builders read only this table; the
 * provenance document (docs/plans/gpu-cloth-2026-11-09/bridson2002-scenes.md)
 * carries the quotes and measurement methods behind each `ref`.
 *
 * Sources, strongest first:
 * - paper:    the SIGGRAPH 2002 paper text and captions;
 * - thesis:   R. Bridson, PhD thesis, Stanford 2003, chapters 1-2;
 * - video:    the 2002 result videos cloth0/1/2.avi (24 fps simulation frames,
 *             see `BRIDSON_2002_VIDEO_FRAME_RATE`), measured as described in the doc;
 * - physbam:  PhysBAM `Projects/solids_3d/Standard_Tests/STANDARD_TESTS.h`
 *             tests 5, 12 and 13, a 2006-2008 descendant of the 2002 code;
 *             later tuning, so it ranks below the paper and the video;
 * - assumption: none of the above states it; the reason is in `ref`.
 */

export type Bridson2002Source = "paper" | "thesis" | "video" | "physbam" | "assumption";

export interface Bridson2002Parameter {
  readonly value: number;
  readonly unit: string;
  readonly source: Bridson2002Source;
  /** Short pointer into the source (section, line, frame, method or reason). */
  readonly ref: string;
}

const P = (value: number, unit: string, source: Bridson2002Source, ref: string): Bridson2002Parameter => ({ value, unit, source, ref });

/** The videos hold 24 fps simulation frames, played back at 15 fps. */
export const BRIDSON_2002_VIDEO_FRAME_RATE = P(24, "1/s", "video",
  "curtain ball onset at video frame 52-56 = PhysBAM test 5 t = 2 s at 24 fps; ball meets the pile at f114 = 4.75 s; PhysBAM default frame_rate = 24");

/** One implicit cloth step per rendered frame. Never substepped. */
export const BRIDSON_2002_STEP_S = 1 / 60;

/** Shared cloth model (section 3 and 4 of the paper). */
export const CLOTH = {
  nodesAcross: P(150, "nodes", "paper", "section 9: 'a piece of cloth with 150 x 150 nodes'"),
  thickness: P(0.001, "m", "paper", "section 6: 'Our cloth is given a realistic thickness, e.g. 1mm'"),
  strainLimit: P(0.1, "1", "paper", "section 4: 'a triangle edge should not change length by more than 10%'"),
  arealDensity: P(1, "kg/m^2", "physbam", "Create_Cloth_Panel: mass_node = aspect*side^2/(m*n), i.e. 1 kg per m^2 shared equally"),
  stretch: P(200, "N/m", "assumption",
    "paper gives no spring constant; PhysBAM edge springs 10/(1+sqrt 2) N per unit strain at 40 panels/m give ~170-220 N/m; soft below the 10% limit (biphasic, section 4)"),
  shear: P(100, "N/m", "assumption", "isotropic lattice with equal diagonal springs: G = E/2 at Poisson 0"),
  poisson: P(0, "1", "assumption", "mass-spring sheets carry no designed Poisson coupling"),
  compressionRatio: P(1, "1", "paper", "section 3: linear structural springs resist compression as they resist extension"),
  bending: P(1.5e-4, "N m", "assumption",
    "bending springs between every other node have no linear bending modulus; chosen for a 2.5 cm bending length (B = rho g c^3) at 1 kg/m^2, to be checked against Fig 1 fold spacing"),
  damping: P(0.002, "s", "assumption", "paper gives none; critically damps edge modes near 1000 rad/s"),
  newtonIterations: P(8, "iterations", "assumption",
    "our solver's choice, not the paper's (C1 RESULT 21): Newton-PCG at 8 Newton x 200 PCG iterations is within 1.5 mm of 20 x 400 on the centre-held 150 x 150 sheet at tick 30, where VBD K 128 is 155 mm off. A cap: a step stops earlier once its residual is below newtonResidualTolerance"),
  newtonPcgIterations: P(200, "iterations", "assumption", "PCG cap per Newton iteration (C1 RESULT 21, as newtonIterations)"),
  newtonPcgTolerance: P(1e-4, "1", "assumption", "PCG stops at r.z <= tol^2 r0.z0 (C1 RESULT 21's converged runs)"),
  newtonResidualTolerance: P(1e-5, "1", "assumption", "Newton stops at |g| <= tol |m g| over free vertices (C1 RESULT 21's converged runs; f32 stalls near 2e-2 |m g| on 150 x 150, where the line search stops it instead)"),
  selfFriction: P(0.3, "1", "assumption",
    "paper gives one mu for static and kinetic (section 7.3) but no value; PhysBAM later uses 3.2 for self contact, rejected as tuned for its repulsion scheme"),
} as const;

/** Fig 1: curtain over a ball (PhysBAM test 5 "Cloth Curtain and Ball"). */
export const CURTAIN = {
  side: P(1, "m", "physbam", "test 5: side_length = 1"),
  aspect: P(1.7, "1", "physbam", "test 5: aspect_ratio = 1.7; Fig 1d ground run ~2.5x the hanging height agrees"),
  height: P(0.5, "m", "physbam", "Create_Cloth_Panel places the sheet at y = 0.5"),
  pinCorners: P(2, "corners", "physbam", "test 5 pins the two x = 0 corners; video top edge sags 0.19-0.23 of its width, a two-point catenary"),
  groundFriction: P(0.3, "1", "physbam", "Add_Ground() default friction 0.3"),
  ballRadius: P(0.25, "m", "physbam", "test 5: Add_Rigid_Body(\"sphere\", 0.25, 0)"),
  ballFriction: P(0, "1", "physbam", "test 5: sphere friction 0"),
  ballCentreZ: P(0.5, "m", "physbam", "test 5: frame.t.z = 0.5, centred on the sheet width"),
  ballStart: P(2, "s", "physbam", "Set_Kinematic_Positions test 5: at rest until t = 2; video onset f52-56"),
  ballRiseEnd: P(3.5, "s", "physbam", "velocity (1, 0.5, 0) for 2 <= t < 3.5: 'the sphere moves up and away' (section 9)"),
  ballDropEnd: P(4, "s", "physbam", "velocity (0, -1.5, 0) for 3.5 <= t < 4"),
  ballReturnSpeed: P(1.5, "m/s", "physbam", "velocity (-1.5, 0, 0) for t >= 4: the caption's 'another ball' re-entering; video f104-150"),
  duration: P(160 / 24, "s", "video", "cloth0.avi: 160 frames at 24 fps"),
} as const;

/** Fig 2: cloth over a rotating sphere (PhysBAM test 13 "Cloth and spinning sphere"). */
export const SPHERE = {
  side: P(1.9, "m", "physbam", "test 13: Create_Cloth_Panel(n, 1.9*side_length, aspect 1)"),
  height: P(0.56, "m", "physbam", "test 13: panel frame t = (0, 0.56, 0); 1 cm above the sphere top; video f0 flat at the sphere top"),
  groundFriction: P(0.1, "1", "physbam", "test 13: Add_Ground(0.1)"),
  sphereRadius: P(0.25, "m", "physbam", "test 13: Add_Rigid_Body(\"sphere\", 0.25, 1)"),
  sphereFriction: P(1, "1", "physbam", "test 13: sphere friction 1"),
  sphereCentreY: P(0.3, "m", "physbam", "test 13: frame.t = (0, 0.30, 0), 5 cm clear of the ground"),
  spinStart: P(0.75, "s", "physbam", "Set_Kinematic_Velocities test 13: twist.angular only for t > 0.75"),
  spinRate: P(-Math.PI / 4, "rad/s", "physbam", "twist.angular = (0, -pi/4, 0); video skirt flaps travel about the vertical"),
  frustumBase: P(0.4, "m", "video", "black frustum in cloth1.avi and Fig 2 (not in the paper text); +-25% from perspective"),
  frustumTop: P(0.2, "m", "video", "frustum top square, +-25%"),
  frustumHeight: P(0.24, "m", "video", "frustum height, +-25%"),
  frustumX: P(-0.65, "m", "physbam", "test 13 cut_pyramid frame t = (-0.65, 0.05, 0.65)"),
  frustumZ: P(0.65, "m", "physbam", "test 13 cut_pyramid frame t.z = 0.65"),
  frustumFriction: P(0.1, "1", "physbam", "test 13: cut_pyramid friction 0.1"),
  duration: P(213 / 24, "s", "video", "cloth1.avi: 213 frames at 24 fps"),
  planeRecords: P(1000, "records/vertex", "assumption",
    "interim: demand bug, C2, to be removed with the relative-motion test; measured peak 911 per vertex as the sheet lands on the sphere (t = 0.333 s), on 8b58f374"),
} as const;

/**
 * Fig 3: tablecloth over four legs without a top. PhysBAM test 12 is a later,
 * different table (1 m cloth, hex-link legs), so the geometry is the 2002
 * video's, reconstructed through its camera and scaled by one anchor.
 */
export const TABLE = {
  sphereRadius: P(0.15, "m", "physbam", "scale anchor: test 12 sphere radius 0.15; every other length is a video ratio times this"),
  legHeight: P(0.75, "m", "video", "sphere diameter = 0.40 leg heights (camera reconstruction, f ~ 798 px)"),
  legWidth: P(0.11, "m", "video", "square posts, height/width ~ 6.7"),
  legSpacing: P(0.48, "m", "video", "square table, leg centres 0.65 leg heights apart (4.3 leg widths)"),
  legChamfer: P(0.022, "m", "video", "tops bevelled ~0.2 leg widths; thesis: the tips 'were actually rounded off' for figure 2.2"),
  side: P(1.59, "m", "video", "cloth side ~2.1 leg heights; its corners reach the ground"),
  height: P(0.76, "m", "assumption", "cloth starts flat 1 cm above the leg tops, as PhysBAM test 13 starts 1 cm above its sphere"),
  sphereFriction: P(0.5, "1", "physbam", "test 12: sphere friction 0.5"),
  legFriction: P(0, "1", "physbam", "test 12: legs friction 0"),
  groundFriction: P(0.3, "1", "physbam", "test 12: Add_Ground() default 0.3"),
  videoStart: P(2, "s", "assumption",
    "cloth2.avi f0 already shows the settled drape (the 2002 run started from a draped state: Stanford's test2_frame056 matches video f56), so a 2 s pre-roll drapes and settles the 0.55 m overhang first"),
  sphereYAtVideoStart: P(2.1, "m", "video", "tracked highlight, y = 2.10 - 0.82 (t - t0); constant 9.6 px/frame from f27 to f42"),
  descentSpeed: P(0.82, "m/s", "video", "tracked highlight; PhysBAM test 12 uses 1.0 m/s"),
  landVideoTime: P(57 / 24, "s", "video", "sphere reaches the ground at f57"),
  rollSpeed: P(0.62, "m/s", "video", "after landing it leaves along +x between two legs to the last frame; pure translation assumed"),
  duration: P(93 / 24, "s", "video", "cloth2.avi: 93 frames at 24 fps after t0"),
} as const;

/** Voxel lattice shared by every Bridson solid. */
export const VOXEL = {
  cell: P(0.01, "m", "assumption",
    "smallest feature is the 22 mm leg chamfer: 2.2 cells; leg width 11 cells, sphere radii 15-25 cells, frustum top 20 cells"),
  minimumFeatureCells: P(2, "cells", "assumption", "a feature narrower than two cells is lost to the trilinear field; authoring refuses it"),
  groundDepth: P(0.08, "m", "assumption", "one 8-cell brick of solid under the floor plane"),
} as const;

export const BRIDSON_2002_PARAMETERS = { CLOTH, CURTAIN, SPHERE, TABLE, VOXEL } as const;
