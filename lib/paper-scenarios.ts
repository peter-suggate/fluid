import { cloneScene, defaultScene, type RigidBodyDescription, type SceneDescription } from "./model";

export type PaperScenarioId = "hose-tank" | "dam-break-boxes" | "sphere-jet";

export const paperScenarios: ReadonlyArray<{ id: PaperScenarioId; name: string; paperFigure: string; description: string }> = [
  { id: "hose-tank", name: "Hose-filled tank", paperFigure: "Figure 3", description: "A continuous jet fills a shallow tank." },
  { id: "dam-break-boxes", name: "Dam break + boxes", paperFigure: "Figure 4", description: "A dam break strikes a stack of rigid boxes." },
  { id: "sphere-jet", name: "Jet past sphere", paperFigure: "Figure 6", description: "An inlet jet flows past a sphere into a tank." }
];

function box(id: number, x: number, y: number, z: number, angle = 0): RigidBodyDescription {
  return {
    id: `paper-box-${id}`, name: `Paper box ${id}`, shape: "box",
    dimensions_m: { x: 0.13, y: 0.11, z: 0.12 }, density_kg_m3: 720,
    position_m: { x, y, z }, orientation: { w: Math.cos(angle / 2), x: 0, y: 0, z: Math.sin(angle / 2) },
    linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    restitution: 0.18, friction: 0.62
  };
}

export function createPaperScenario(id: PaperScenarioId, source: SceneDescription = defaultScene): SceneDescription {
  const scene = cloneScene(source);
  scene.randomSeed = 2011;
  scene.duration_s = 15;
  scene.container.width_m = 1.2;
  scene.container.height_m = 0.9;
  scene.container.depth_m = 0.8;
  scene.container.top = "open";
  scene.container.fluidWallMode = "free-slip";
  scene.fluid.surfaceTension_N_m = 0;
  // The paper reports 1/30 s. Conservative surface-density transport tolerates
  // large CFL, but the collocated projection and rigid proxy contacts still use
  // a smaller step in these impact-heavy validation scenes.
  scene.numerics.fixedDt_s = 1 / 180;
  scene.numerics.maxDt_s = 1 / 180;
  scene.nominalResolution.length_m = 0.025;
  scene.voxelDomain.finestCellSize_m = 0.02;

  if (id === "hose-tank") {
    scene.sceneId = "paper-figure-3-hose-filled-tank";
    scene.container.fillFraction = 0.06;
    scene.fluid.initialCondition = "tank-fill";
    // Horizontal hose: the jet enters from the left wall and arcs into the
    // pool under gravity. Direction is carried entirely by velocity_m_s (the
    // injection cylinder is oriented along it), adjustable in scene config.
    scene.fluid.inflow = {
      center_m: { x: -0.40, y: 0.55, z: 0 }, radius_m: 0.08, length_m: 0.12,
      velocity_m_s: { x: 0.80, y: 0, z: 0 }, start_s: 0, end_s: 14, ramp_s: 0.35
    };
    scene.rigidBodies = [{
      id: "paper-hose-nozzle", name: "Hose nozzle", shape: "cylinder",
      dimensions_m: { x: 0.10, y: 0.26, z: 0.10 }, density_kg_m3: 5000,
      // Cylinder axis defaults to +y; rotate 90 degrees about z to lie along x.
      position_m: { x: -0.47, y: 0.55, z: 0 }, orientation: { w: Math.SQRT1_2, x: 0, y: 0, z: Math.SQRT1_2 },
      linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
      restitution: 0.05, friction: 0.8, motion: "static"
    }];
  } else if (id === "dam-break-boxes") {
    scene.sceneId = "paper-figure-4-dam-break-box-stack";
    scene.container.fillFraction = 0.26;
    scene.fluid.initialCondition = "dam-break";
    delete scene.fluid.inflow;
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 360;
    scene.rigidBodies = [
      box(1, 0.08, 0.06, 0.08), box(2, 0.08, 0.28, 0.08, 0.06), box(3, 0.08, 0.50, 0.08, -0.04),
      box(4, 0.31, 0.06, 0.08, -0.05), box(5, 0.31, 0.28, 0.08, 0.04), box(6, 0.50, 0.06, 0.08)
    ];
  } else {
    scene.sceneId = "paper-figure-6-jet-past-sphere";
    scene.container.fillFraction = 0.14;
    scene.fluid.initialCondition = "tank-fill";
    scene.fluid.inflow = {
      center_m: { x: -0.50, y: 0.58, z: 0 }, radius_m: 0.075, length_m: 0.12,
      velocity_m_s: { x: 1.20, y: -0.05, z: 0 }, start_s: 0, end_s: 12, ramp_s: 0.25
    };
    scene.rigidBodies = [{
      id: "paper-sphere", name: "Flow obstacle", shape: "sphere",
      dimensions_m: { x: 0.14, y: 0.14, z: 0.14 }, density_kg_m3: 5000,
      position_m: { x: -0.10, y: 0.55, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
      linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 }, restitution: 0.05, friction: 0.8
      , motion: "static"
    }];
  }
  return scene;
}
