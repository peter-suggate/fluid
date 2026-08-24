import { cloneScene, defaultScene, type RigidBodyDescription, type SceneDescription } from "./model";

export type PaperScenarioId = "hose-tank" | "dam-break-boxes" | "sphere-jet";

export const paperScenarios: ReadonlyArray<{ id: PaperScenarioId; name: string; paperFigure: string; description: string }> = [
  { id: "hose-tank", name: "Hose-filled tank", paperFigure: "Paper-inspired legacy demo", description: "A continuous jet fills a shallow tank." },
  { id: "dam-break-boxes", name: "Dam break + boxes", paperFigure: "Paper-inspired legacy demo", description: "A dam break strikes a stack of rigid boxes." },
  { id: "sphere-jet", name: "Jet past sphere", paperFigure: "Paper-inspired legacy demo", description: "An inlet jet flows past a sphere into a tank." }
];

/**
 * Published-parameter reconstruction of the initial dam phase of CM12 Figure
 * 9. The paper specifies the 128x128x64 lattice, dx=.05 m, dt=1/30 s, gravity
 * 10 m/s2, and D=2.1, but not the reservoir dimensions. The 40x96x64-cell
 * reservoir below is therefore explicit harness geometry, not a claim that
 * the unpublished production asset has been recovered. Later injected balls
 * are intentionally outside this initial-condition conformance case.
 */
export function createMassConservingFigure9DamBreak(
  source: SceneDescription = defaultScene,
): SceneDescription {
  const scene = cloneScene(source);
  scene.sceneId = "mass-conserving-figure-9-dam-break";
  scene.randomSeed = 2012;
  scene.duration_s = 4;
  scene.container = {
    ...scene.container,
    width_m: 6.4,
    height_m: 6.4,
    depth_m: 3.2,
    fillFraction: 0.234375,
    top: "closed",
    fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { ...scene.voxelDomain, finestCellSize_m: 0.05 };
  scene.nominalResolution = { length_m: 0.05 };
  scene.numerics.fixedDt_s = 1 / 30;
  scene.numerics.maxDt_s = 1 / 30;
  scene.fluid.initialCondition = "dam-break";
  scene.fluid.initialDamBreakDimensions_m = { x: 2, y: 4.8, z: 3.2 };
  delete scene.fluid.initialDamBreakOrigin_m;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.inflow;
  scene.fluid.gravity_m_s2 = { x: 0, y: -10, z: 0 };
  scene.fluid.surfaceTension_N_m = 0;
  scene.rigidBodies = [];
  return scene;
}

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
  scene.container.top = "closed";
  scene.container.fluidWallMode = "free-slip";
  scene.fluid.surfaceTension_N_m = 0;
  // These are retained paper-inspired product demos, not reconstructions of
  // CM12 Figures 3, 4, or 6. Their smaller step and 20 mm lattice are authored
  // for the interactive scenes rather than for numerical paper validation.
  scene.numerics.fixedDt_s = 1 / 180;
  scene.numerics.maxDt_s = 1 / 180;
  scene.nominalResolution.length_m = 0.025;
  scene.voxelDomain.finestCellSize_m = 0.02;

  if (id === "hose-tank") {
    scene.sceneId = "paper-figure-3-hose-filled-tank";
    // Keep the 20 mm finest lattice on an exact sparse-CM12 brick tiling:
    // 64 x 48 x 40 admits complete size-8 brick pages.
    scene.container.width_m = 1.28;
    scene.container.height_m = 0.96;
    // Retain a small authored scene-step lane; sparse CM12's paper-step lane
    // injects the corresponding nozzle-swept volume once per simulation step.
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 360;
    scene.container.fillFraction = 0.06;
    scene.fluid.initialCondition = "tank-fill";
    // Horizontal hose: the jet enters from the left wall and arcs into the
    // pool under gravity. Direction is carried entirely by velocity_m_s (the
    // injection cylinder is oriented along it), adjustable in scene config.
    scene.fluid.inflow = {
      center_m: { x: -0.44, y: 0.55, z: 0 }, radius_m: 0.08, length_m: 0.12,
      velocity_m_s: { x: 2.60, y: 0, z: 0 }, start_s: 0, end_s: 14, ramp_s: 0
    };
    scene.rigidBodies = [{
      id: "paper-hose-nozzle", name: "Hose nozzle", shape: "cylinder",
      dimensions_m: { x: 0.16, y: 0.26, z: 0.16 }, density_kg_m3: 5000,
      // Cylinder axis defaults to +y; rotate 90 degrees about z to lie along x.
      position_m: { x: -0.51, y: 0.55, z: 0 }, orientation: { w: Math.SQRT1_2, x: 0, y: 0, z: Math.SQRT1_2 },
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
