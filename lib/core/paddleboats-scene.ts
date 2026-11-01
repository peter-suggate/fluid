import { studioStageSceneryGraph } from "./studio-stage-scene";
import assets from "./imported-assets/paddleboats.generated.json";
import type { SceneryNode } from "./scenery-graph";
import { cloneScene, defaultScene, type RigidBodyDescription, type SceneDescription } from './model';
import { rigidMotorPose } from './rigid-motor';
import type { WidgetPlacementOverrides } from '../framework/widgets';

export const PADDLEBOATS_SCENE_ID = 'paddleboats';
export const paddleboatsCamera = { distance_m: 11.5, tanHalfFov: .55, target_m: { x: 0, y: .65, z: 0 }, elevation_rad: .62, azimuth_rad: .7 };
/** Each boat's throttle on its own side of the screen: from this camera Ember (-x) is on the left. */
export const paddleboatsWidgets = {
  'motor:paddleboat-0': { anchor: 'bottom-left' }, 'motor:paddleboat-1': { anchor: 'bottom-right' },
} as const satisfies WidgetPlacementOverrides;

/** Two free assemblies. Only wheel angle is prescribed: the NB-FLIP pressure
 * solve supplies buoyancy, wakes and propulsion. Hull mass includes the cabin,
 * axle and motor; the four paddle fixtures carry no independent rigid mass. */
export function createPaddleboatsScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = PADDLEBOATS_SCENE_ID;
  scene.duration_s = 30;
  scene.container = { width_m: 12.8, height_m: .8, depth_m: 2.4, fillFraction: .75, top: 'open', fluidWallMode: 'no-slip' };
  scene.voxelDomain = { finestCellSize_m: .025, brickSize_cells: 8 };
  scene.nominalResolution = { length_m: .025 };
  scene.solidVoxels = [];
  delete scene.terrain;
  scene.fluid = { density_kg_m3: 998.2, dynamicViscosity_Pa_s: 0, surfaceTension_N_m: 0,
    optics: { indexOfRefraction: 1.333, absorption_mInv: [.22, .055, .035], roughness: .04 },
    gravity_m_s2: { x: 0, y: -9.80665, z: 0 }, initialCondition: 'tank-fill', initialVelocity_m_s: { x: 0, y: 0, z: 0 } };
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
  scene.rigidBodies = [];
  const zero = () => ({ x: 0, y: 0, z: 0 });
  // Preserve each sternwheel's clearance from its end wall as the tank grows.
  const startX = scene.container.width_m / 2 - 1.45;
  for (const [index, x] of [-startX, startX].entries()) {
    const hull: RigidBodyDescription = { id: `paddleboat-${index}`, name: index ? 'Jade · free hull' : 'Ember · free hull',
      shape: 'box', motion: 'dynamic', dimensions_m: { x: 1.4, y: .28, z: .54 }, density_kg_m3: 499.1,
      position_m: { x, y: .6, z: 0 }, orientation: index ? { w: 0, x: 0, y: 1, z: 0 } : { w: 1, x: 0, y: 0, z: 0 },
      linearVelocity_m_s: zero(), angularVelocity_rad_s: zero(), restitution: .05, friction: .4,
      appearance: index ? 'paddleboat-teal' : 'paddleboat-red' };
    // The wheels are rated for three times their cruising speed: the boats open
    // at one third throttle, and the fixtures below carry no drive of their own.
    scene.rigidBodies.push({ ...hull, motorStrength: 1 / 3 });
    for (let blade = 0; blade < 4; blade++) {
      const motorAttachment = { parentId: hull.id, pivot_m: { x: -1.0, y: .075, z: 0 }, offset_m: { x: .235, y: 0, z: 0 },
        axis: { x: 0, y: 0, z: 1 }, orientation: { w: 1, x: 0, y: 0, z: 0 }, phase_rad: blade * Math.PI / 2,
        angularSpeed_rad_s: -11.4, startTime_s: 0, rampUp_s: .25 };
      scene.rigidBodies.push({ ...hull, id: `${hull.id}-paddle-${blade}`, name: `${hull.name.split(' ·')[0]} · paddle ${blade + 1}`,
        motion: 'static', dimensions_m: { x: .13, y: .05, z: .5 }, density_kg_m3: 650,
        ...rigidMotorPose(hull, motorAttachment, 0), motorAttachment, appearance: 'paddleboat-wheel' });
    }
  }
  const stage = studioStageSceneryGraph(scene);
  scene.scenery = { ...stage, nodes: [...stage.nodes.filter(node => node.kind === "room-shell" || node.id === "stage/floor"), ...assets.basin.map((part, index): SceneryNode => ({
    kind: 'box', id: `paddleboat-basin-${index}`, place: { units: 'metres', position: { x: part.p[0]!, y: part.p[1]!, z: part.p[2]! } },
    halfSize: { x: part.d[0]!, y: part.d[1]!, z: part.d[2]! }, material: { colorLinear: part.color as [number, number, number], surface: 'ceramic' },
  }))] };
  scene.surfaceStyle = 'smooth';
  scene.lighting = {
    directional: { direction: [-.45, .82, .35], colorLinear: [1, .94, .82], intensity: 3 },
    environment: { diffuseScale: 1, specularScale: 1, lowerRadianceLinear: [.3, .29, .25], upperRadianceLinear: [.7, .85, 1.05], accentRadianceLinear: [.6, .7, .8] },
    grade: { toneCurve: 'agx', exposure: 1.15, whiteBalance: [1, 1, 1] },
  };
  return scene;
}
