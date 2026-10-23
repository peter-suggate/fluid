import type { CameraState, SceneDescription } from "./model";
import type { SceneryNode } from "./scenery-graph";
import { nbflipDomain } from "./nbflip-paper-scenes";
import { interactionCatalog } from "../interaction/installed";

/** Ballistic Simulations, "FLIP-FLUIDS apic test with blender":
 * https://www.youtube.com/watch?v=PRrQysfQOdc
 *
 * Reconstruction from the 0, .8, 1.6, 2.4, 4 and 7.2 second images. The
 * square liquid sides persist while the whole volume tips: an invisible
 * vessel is inferred, not a free water cube. The original .blend, dimensions,
 * APIC settings and animation curves are not published. Lengths, feed and
 * initial conditions below are authored estimates; the water is simulated by NB-FLIP.
 * The vessel is sealed, with the water initialized inside its closed lid.
 */
export const NBFLIP_STAIR_CUBE_ID = "nbflip-stair-cube";
export const NBFLIP_STAIR_CUBE_GRID = [192, 128, 192] as const;
export const NBFLIP_STAIR_CUBE_CELL_M = 0.04;
export const NBFLIP_STAIR_BEAMS = [
  { z: -1.6, height: 1.8 }, { z: -0.4, height: 1.2 }, { z: 0.8, height: 0.6 },
] as const;

/** The empty shell balances behind the top beam's downhill edge (-1.24 m). */
export const NBFLIP_STAIR_VESSEL_POSITION = { x: 0.48, y: 2.54, z: -1.27 } as const;

export function createNbflipStairCube(): SceneDescription {
  const scene = nbflipDomain(NBFLIP_STAIR_CUBE_ID, NBFLIP_STAIR_CUBE_GRID, NBFLIP_STAIR_CUBE_CELL_M, 480, 0,
    NBFLIP_STAIR_CUBE_GRID, 1 / 60);
  // A distant, invisible catastrophe boundary keeps escaped spray finite.
  scene.container.vessel = "none";
  // Release water above the floor and toward the downhill wall. Its fall and
  // pressure torque initiate the tip; the shell has no initial kick or track.
  // Preserve the original four 0.64 m cubes' total volume (1.048576 m^3).
  const waterHeight = 4 * 0.64 ** 3 / (1.28 * 0.96);
  scene.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -0.16, y: 2.14, z: NBFLIP_STAIR_VESSEL_POSITION.z - 0.32 }, max_m: { x: 1.12, y: 2.14 + waterHeight, z: NBFLIP_STAIR_VESSEL_POSITION.z + 0.64 } }];
  delete scene.fluid.scheduledDrops;
  scene.fluid.pagePool = { blocks: 160 };
  scene.fluid.optics = { absorption_mInv: [0.12, 0.055, 0.035], scatter: [0.045, 0.055, 0.065], roughness: 0.04, indexOfRefraction: 1.333 };
  scene.rigidBodies = [];
  scene.interaction = { schemaVersion: 1, connections: [], entities: [interactionCatalog.createEntity({
    id: "stair-vessel", name: "Invisible sealed vessel", kind: "fluid.sealed-vessel",
    transform: { position_m: { ...NBFLIP_STAIR_VESSEL_POSITION }, orientation: { w: 1, x: 0, y: 0, z: 0 } },
  })] };
  scene.surfaceStyle = "smooth";
  scene.lighting = {
    directional: { direction: [-0.4, 0.82, 0.35], colorLinear: [1, 0.94, 0.84], intensity: 3 },
    environment: { diffuseScale: 0.75, specularScale: 1,
      lowerRadianceLinear: [0.12, 0.15, 0.18], upperRadianceLinear: [0.48, 0.57, 0.64], accentRadianceLinear: [0.9, 0.92, 0.95] },
    grade: { exposure: 1, toneCurve: "agx" },
  };
  const nodes: SceneryNode[] = [
    { kind: "room-shell", id: "shell", materialModel: "room", faces: [],
      floor: { colorLinear: [0.15, 0.19, 0.23] }, wall: { colorLinear: [0.15, 0.19, 0.23] }, ceiling: { colorLinear: [0.15, 0.19, 0.23] } },
    { kind: "box", id: "stair-floor", tags: ["floor", "fluid-collider"],
      place: { units: "metres", position: { x: 0, y: -0.08, z: 0 } }, halfSize: { x: 24, y: 0.08, z: 24 },
      material: { colorLinear: [0.13, 0.17, 0.21], surface: "architectural" } },
    ...NBFLIP_STAIR_BEAMS.map(({ z, height }, i): SceneryNode => ({ kind: "box", id: `stair-beam-${i + 1}`,
      tags: ["fluid-collider"], place: { units: "metres", position: { x: 0, y: height / 2, z } },
      halfSize: { x: 2.8, y: height / 2, z: 0.36 },
      material: { colorLinear: [0.055, 0.071, 0.09], surface: "brushed-metal" } })),
  ];
  // Grid lines are scenery, not ridges in the collision floor.
  for (let line = -24; line <= 24; line++) for (const axis of ["x", "z"] as const) {
    nodes.push({ kind: "box", id: `stair-grid-${axis}-${line}`,
      place: { units: "metres", position: { x: axis === "x" ? line * 0.5 : 0, y: 0.02, z: axis === "z" ? line * 0.5 : 0 } },
      halfSize: { x: axis === "x" ? 0.02 : 12, y: 0.02, z: axis === "z" ? 0.02 : 12 },
      material: { colorLinear: [0.24, 0.29, 0.33], surface: "architectural" } });
  }
  scene.scenery = { palettes: {}, nodes };
  return scene;
}

export const nbflipStairCubeCamera: Partial<CameraState> = {
  azimuth_rad: -0.8423, elevation_rad: 0.5104, distance_m: 17.9607,
  target_m: { x: 3.2384, y: 0.1878, z: 0.55 }, tanHalfFov: 0.25,
};
