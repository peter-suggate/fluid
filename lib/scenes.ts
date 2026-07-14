import { cloneScene, defaultCamera, defaultScene, type CameraState, type SceneDescription } from "./model";
import { createPaperScenario } from "./paper-scenarios";

export interface ScenePreset {
  id: string;
  name: string;
  group: "Interactive" | "Paper figures" | "Comparisons";
  description: string;
  create(): SceneDescription;
  camera?: Partial<CameraState>;
}

const paperCamera: Partial<CameraState> = { distance_m: 2.45, target_m: { x: 0, y: 0.42, z: 0 } };

export const scenePresets: ReadonlyArray<ScenePreset> = [
  {
    id: "water-box-dam-break",
    name: "Water box · dam break",
    group: "Interactive",
    description: "A collapsing water column. Drag bodies in from the viewport tray.",
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
    create: () => {
      const scene = cloneScene(defaultScene);
      scene.sceneId = "interactive-water-box-settled";
      scene.fluid.initialCondition = "tank-fill";
      return scene;
    }
  },
  {
    id: "hose-tank",
    name: "Hose-filled tank",
    group: "Paper figures",
    description: "Figure 3 · a continuous jet fills a shallow tank.",
    create: () => createPaperScenario("hose-tank"),
    camera: paperCamera
  },
  {
    id: "dam-break-boxes",
    name: "Dam break + box stack",
    group: "Paper figures",
    description: "Figure 4 · a dam break strikes a stack of rigid boxes.",
    create: () => createPaperScenario("dam-break-boxes"),
    camera: paperCamera
  },
  {
    id: "sphere-jet",
    name: "Jet past sphere",
    group: "Paper figures",
    description: "Figure 6 · an inlet jet flows past a static sphere.",
    create: () => createPaperScenario("sphere-jet"),
    camera: paperCamera
  },
  {
    id: "deep-water-ab",
    name: "Deep-water A/B",
    group: "Comparisons",
    description: "20 m tank at 80% fill, 1/30 s paper step, σ = 0 · isolates the grid method.",
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
      scene.numerics.maxDt_s = 1 / 30;
      scene.rigidBodies = [];
      return scene;
    }
  }
];

export const defaultScenePresetId = scenePresets[0].id;

export function getScenePreset(id: string): ScenePreset {
  return scenePresets.find((preset) => preset.id === id) ?? scenePresets[0];
}

export function cameraForPreset(preset: ScenePreset): CameraState {
  return { ...defaultCamera, ...preset.camera };
}
