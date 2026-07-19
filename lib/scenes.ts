import { cloneScene, defaultCamera, defaultScene, DEFAULT_GPU_CPU_TIMESTEP_RATIO, type CameraState, type SceneDescription } from "./model";
import { createPaperScenario } from "./paper-scenarios";
import { applyGardenPool, GARDEN_DAM_BRICK_SEED_M, GARDEN_WATERLINE_M, gardenPoolTerrain } from "./garden-scene";
import { terrainHeightAt } from "./terrain";
import type { EnvironmentId } from "./environments";

export interface ScenePreset {
  id: string;
  name: string;
  group: "Interactive" | "Garden" | "Paper figures" | "Comparisons";
  description: string;
  create(): SceneDescription;
  camera?: Partial<CameraState>;
  /** Art-directed background that is part of this preset's presentation. */
  background: EnvironmentId;
}

/** World-space centre of the single seeded 8-cubed fluid brick (the -x/-z quadrant). */
export const BRICK_QUAD_DAM_SEED_M = { x: -0.2, y: 0.2, z: -0.2 };

/**
 * A tank sized so the finest solver grid is exactly 16x8x16 cells: a 2x2 x/z
 * arrangement of 8-cubed fluid bricks at one brick of height. Water starts as
 * a full-height column filling exactly one brick quadrant and dam-breaks
 * across the brick boundaries into the other three, which makes it the
 * minimal watchable scenario for brick residency activation, the sparse brick
 * atlas, and seam quality.
 */
export function createBrickQuadDamBreakScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = "brick-quad-dam-break";
  scene.rigidBodies = [];
  scene.container = { ...scene.container, width_m: 0.8, height_m: 0.4, depth_m: 0.8, fillFraction: 0.25, top: "open", fluidWallMode: "no-slip" };
  scene.fluid.initialCondition = "dam-break";
  scene.fluid.initialBrickSeeds_m = [{ ...BRICK_QUAD_DAM_SEED_M }];
  delete scene.fluid.inflow;
  // 256 columns over the 0.8 m square footprint give 16x16 columns of 0.05 m
  // cells; the 0.4 m height then yields exactly 8 fine layers (one brick).
  scene.numerics.surfaceColumnsOverride = 256;
  return scene;
}

/**
 * A wide ocean tank sized so the finest solver grid is exactly 384x96x64
 * cells of 0.025 m (48x12x8 fluid bricks). The pool fills to 72 cells
 * (1.8 m); a 2x1x8-brick slab of extra water (0.4 m wide, 0.2 m tall, full
 * depth extent) rests on the surface along the -x wall. Releasing it launches
 * a long gravity wave (~sqrt(gH) = 4.2 m/s) that crosses the tank in ~2.3 s
 * and reflects. The calm deep interior is exactly what large octree leaves
 * coarsen best: below the graded surface band the water collapses into
 * 16-cubed and 32-cubed pressure cells when the octree method's maximum leaf
 * is raised to 32.
 */
export function createOceanSeicheScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = "ocean-seiche";
  scene.rigidBodies = [];
  scene.container = { ...scene.container, width_m: 9.6, height_m: 2.4, depth_m: 1.6, fillFraction: 0.75, top: "open", fluidWallMode: "no-slip" };
  scene.fluid.initialCondition = "tank-fill";
  // A long gravity wave has no meaningful capillary scale; keep the scene in
  // the same physical scope as the deep-water A/B preset.
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.inflow;
  // 24576 columns over the 9.6 x 1.6 m footprint give 384x64 columns of
  // 0.025 m cells; the 2.4 m height then yields exactly 96 fine layers.
  scene.numerics.surfaceColumnsOverride = 24576;
  // The raised slab: brick tiers x {0,1}, y tier 9 (cells 72..79 — directly
  // on the 72-cell pool surface), and every z tier. Seeds are the world-space
  // centres of those 8-cubed bricks at the exact grid above.
  const h = 0.025, brick = 8 * h;
  const seeds: { x: number; y: number; z: number }[] = [];
  for (let zTier = 0; zTier < 8; zTier += 1) {
    const z = -0.8 + (zTier + 0.5) * brick;
    seeds.push({ x: -4.8 + 0.5 * brick, y: 9.5 * brick, z }, { x: -4.8 + 1.5 * brick, y: 9.5 * brick, z });
  }
  scene.fluid.initialBrickSeeds_m = seeds;
  scene.fluid.initialBrickSeedsAdditive = true;
  return scene;
}

const paperCamera: Partial<CameraState> = { distance_m: 2.45, target_m: { x: 0, y: 0.42, z: 0 } };
// Close and low: the pond fills the frame with the cloud trees and mushrooms
// crowding its banks, while the banks still occlude the far waterline so the
// water reads as inset into the ground.
const gardenCamera: Partial<CameraState> = { azimuth_rad: 0.58, elevation_rad: 0.38, distance_m: 2.95, target_m: { x: 0, y: 0.26, z: 0 } };

const authoredScenePresets: ReadonlyArray<ScenePreset> = [
  {
    id: "water-box-dam-break",
    name: "Water box · dam break",
    group: "Interactive",
    description: "A collapsing water column. Drag bodies in from the viewport tray.",
    background: "default",
    create: () => {
      const scene = cloneScene(defaultScene);
      scene.rigidBodies = [];
      return scene;
    }
  },
  {
    id: "water-box-tank-fill",
    name: "Water box · settled tank",
    group: "Interactive",
    description: "The same container starting from a settled fill; drop bodies into calm water.",
    background: "bathhouse",
    create: () => {
      const scene = cloneScene(defaultScene);
      scene.sceneId = "interactive-water-box-settled";
      scene.fluid.initialCondition = "tank-fill";
      return scene;
    }
  },
  {
    id: "garden-pond",
    name: "Garden pond · still water",
    group: "Garden",
    description: "A white-clay pond settled to its waterline, ringed by cloud trees and oversized mushrooms. A cork ball bobs over the deep end; stepping stones cross the beach shelf.",
    create: () => {
      const scene = applyGardenPool(cloneScene(defaultScene));
      scene.sceneId = "garden-pond-still";
      scene.fluid.initialCondition = "tank-fill";
      const terrain = gardenPoolTerrain();
      const beach = { x: 0.25, z: -0.55 };
      const stone = (index: number, x: number, z: number) => ({
        id: `garden-stone-${index}`, name: `Stepping stone ${index}`, shape: "cylinder" as const,
        dimensions_m: { x: 0.13, y: 0.06, z: 0.13 }, density_kg_m3: 2600,
        position_m: { x, y: terrainHeightAt(terrain, x, z) + 0.03, z },
        orientation: { w: 1, x: 0, y: 0, z: 0 },
        linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
        restitution: 0.05, friction: 0.9, motion: "static" as const
      });
      scene.rigidBodies = [
        {
          id: "garden-cork-ball", name: "Cork ball", shape: "sphere",
          dimensions_m: { x: 0.09, y: 0.09, z: 0.09 }, density_kg_m3: 240,
          position_m: { x: -0.35, y: GARDEN_WATERLINE_M + 0.25, z: -0.12 },
          orientation: { w: 1, x: 0, y: 0, z: 0 },
          linearVelocity_m_s: { x: 0.1, y: 0, z: 0.05 }, angularVelocity_rad_s: { x: 0, y: 1.2, z: 0 },
          restitution: 0.4, friction: 0.35
        },
        stone(1, beach.x - 0.18, beach.z - 0.1), stone(2, beach.x + 0.08, beach.z + 0.04), stone(3, beach.x + 0.34, beach.z + 0.16)
      ];
      return scene;
    },
    camera: gardenCamera,
    background: "garden"
  },
  {
    id: "garden-dam-break",
    name: "Garden pond · dam break",
    group: "Garden",
    description: "One resident fluid brick releases on the upper lawn, vacates its source region, and activates neighbouring bricks as it washes into the pond.",
    create: () => {
      const scene = applyGardenPool(createPaperScenario("dam-break-boxes"), { fillFraction: 0.16 });
      scene.sceneId = "garden-pond-dam-break";
      scene.fluid.initialBrickSeeds_m = [{ ...GARDEN_DAM_BRICK_SEED_M }];
      return scene;
    },
    camera: gardenCamera,
    background: "garden"
  },
  {
    id: "garden-hose",
    name: "Garden pond · hose fill",
    group: "Garden",
    description: "The pond starts as a puddle in the deep end and a hose arcs water in until the banks fill to the waterline.",
    create: () => {
      const scene = applyGardenPool(createPaperScenario("hose-tank"), { fillFraction: 0.08 });
      scene.sceneId = "garden-pond-hose-fill";
      if (scene.fluid.inflow) scene.fluid.inflow.end_s = 30;
      return scene;
    },
    camera: gardenCamera,
    background: "garden"
  },
  {
    id: "hose-tank",
    name: "Hose-filled tank",
    group: "Paper figures",
    description: "Figure 3 · a continuous jet fills a shallow tank.",
    background: "conservatory",
    create: () => createPaperScenario("hose-tank"),
    camera: paperCamera
  },
  {
    id: "dam-break-boxes",
    name: "Dam break + box stack",
    group: "Paper figures",
    description: "Figure 4 · a dam break strikes a stack of rigid boxes.",
    background: "concrete-gallery",
    create: () => createPaperScenario("dam-break-boxes"),
    camera: paperCamera
  },
  {
    id: "sphere-jet",
    name: "Jet past sphere",
    group: "Paper figures",
    description: "Figure 6 · an inlet jet flows past a static sphere.",
    background: "night-lab",
    create: () => createPaperScenario("sphere-jet"),
    camera: paperCamera
  },
  {
    id: "brick-quad-dam-break",
    name: "Brick quad · dam break",
    group: "Comparisons",
    description: "A 2x2 four-brick tank: one brick quadrant of water releases and crosses every brick boundary, exercising cross-brick transport, residency activation, and seam quality.",
    background: "default",
    create: createBrickQuadDamBreakScene,
    camera: { distance_m: 1.9, target_m: { x: 0, y: 0.2, z: 0 } }
  },
  {
    id: "ocean-seiche",
    name: "Ocean · rolling wave",
    group: "Comparisons",
    description: "A wide 9.6 m tank of deep calm water; a raised slab along one wall releases a long wave that ripples across and reflects. With the octree method, set Maximum leaf to 32³ to watch the deep interior coarsen into 16³/32³ pressure cells.",
    background: "research-station",
    create: createOceanSeicheScene,
    camera: { azimuth_rad: 0.35, elevation_rad: 0.32, distance_m: 10.0, target_m: { x: 0, y: 1.1, z: 0 } }
  },
  {
    id: "deep-water-ab",
    name: "Deep-water A/B",
    group: "Comparisons",
    description: "20 m tank at 80% fill, 1/30 s paper step, σ = 0 · isolates the grid method.",
    background: "research-station",
    create: () => {
      const scene = cloneScene(defaultScene);
      scene.sceneId = "deep-water-grid-comparison";
      scene.container.height_m = 20;
      scene.container.fillFraction = 0.8;
      scene.fluid.initialCondition = "tank-fill";
      // The tall-cell paper does not include a capillary-force discretization.
      // Keep the A/B preset within that shared physical scope so the grid and
      // pressure methods are the only variables in the comparison.
      scene.fluid.surfaceTension_N_m = 0;
      scene.numerics.fixedDt_s = 1 / 30;
      scene.numerics.maxDt_s = scene.numerics.fixedDt_s * DEFAULT_GPU_CPU_TIMESTEP_RATIO;
      scene.rigidBodies = [];
      return scene;
    }
  }
];

/** Attach the art-directed environment to the scene consumed by GPU solvers. */
export const scenePresets: ReadonlyArray<ScenePreset> = authoredScenePresets.map((preset) => ({
  ...preset,
  create: () => {
    const scene = preset.create();
    scene.environment = preset.background;
    return scene;
  }
}));

export const defaultScenePresetId = scenePresets[0].id;

export function getScenePreset(id: string): ScenePreset {
  return scenePresets.find((preset) => preset.id === id) ?? scenePresets[0];
}

export function cameraForPreset(preset: ScenePreset): CameraState {
  return { ...defaultCamera, ...preset.camera };
}
