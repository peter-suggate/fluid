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

const paperCamera: Partial<CameraState> = { distance_m: 2.45, target_m: { x: 0, y: 0.42, z: 0 } };
// Low enough that the pond reads as inset into the lawn (banks occlude the
// far waterline) while the whole garden still fits the frame.
const gardenCamera: Partial<CameraState> = { azimuth_rad: 0.58, elevation_rad: 0.3, distance_m: 3.7, target_m: { x: 0, y: 0.3, z: 0 } };

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
    description: "An organic pool inset into the lawn, settled to its waterline. A cork ball bobs over the deep end; stepping stones cross the beach shelf.",
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
