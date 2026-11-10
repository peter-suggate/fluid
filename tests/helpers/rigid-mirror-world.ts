import type { SceneDescription } from '../../lib/core/model';
import { initializeRigidBodies, type RigidBodyState } from '../../lib/core/rigid-body';
import { rigidMotorStrength, updateRigidMotorDrive } from '../../lib/core/rigid-motor';
import { rigidMotorRecords, rigidStateRecord } from '../../lib/core/webgpu-rigid-body';
import {
  containerMirrorPlanes, decodeMirrorBodies, decodeMirrorMotors, mirrorShapeSupportRadius,
  type MirrorPlane, type MirrorSolid, type MirrorWorld, type V3,
} from '../../lib/core/rigid-assembly-contact';
import { SolidOccupancyMask, type SolidOccupancyMaskWalls } from '../../lib/core/solid-occupancy-mask';
import { solidWorldForFluidLattice } from '../../lib/core/solid-world';
import { sceneFluidLatticeDimensions, sceneLatticeDimensions } from '../../lib/core/scene-lattice';
import { packSvoSculptCuts, sampleSvoSculptCuts } from '../../lib/svo/features/scene-publication/svo-sculpt-cuts';
import { planSculptBake, type SculptBakePlan } from '../../lib/svo/features/scene-publication/svo-sculpt-substrate';

/**
 * The CPU mirror of the rigid kernel, started from exactly the records
 * `WebGPURigidBodySystem.syncBodies` uploads: `rigidStateRecord` per body and
 * `rigidMotorRecords` for the roster (drives at motor time zero). Static
 * surfaces default to what the uniform reference's kernel sees in a scene
 * without terrain relief: the container's walls and floor, then the flat
 * terrain plane at y = 0.
 */
export function rigidMirrorWorld(scene: SceneDescription, options: {
  bodies?: RigidBodyState[];
  planes?: MirrorPlane[];
  gravity?: V3;
  /** The voxel solid assemblies contact, as the uniform reference's kernel binds it (`rigidMirrorSolid`). */
  solid?: MirrorSolid;
} = {}): MirrorWorld {
  const states = options.bodies ?? initializeRigidBodies(scene.rigidBodies);
  const rho = scene.fluid.density_kg_m3, descriptions = states.map(state => state.description);
  const records = states.flatMap(state => rigidStateRecord(state, rho, 0));
  const drives = states.map(state => {
    const m = state.description.motorAttachment;
    if (!m) return undefined;
    const parent = states.find(candidate => candidate.description.id === m.parentId);
    return updateRigidMotorDrive(m, rigidMotorStrength(parent ? parent.description : state.description, m.channel), 0, undefined);
  });
  const g = scene.fluid.gravity_m_s2;
  return {
    bodies: decodeMirrorBodies(Float32Array.from(records), states.length),
    motors: decodeMirrorMotors(rigidMotorRecords(descriptions, drives), states.length),
    rho, gravity: options.gravity ?? [g.x, g.y, g.z], time_s: 0, step_s: 0,
    planes: options.planes ?? [...containerMirrorPlanes(scene.container), { normal: [0, 1, 0], offset: 0 }],
    supportRadius: mirrorShapeSupportRadius,
    solid: options.solid,
  };
}

/**
 * What the uniform reference's rigid kernel reads for assembly contact
 * (`assemblySolid`): the fluid's solid mask image, folded from the scene as
 * the solver folds it, and the sculpt cut field's pages and samples planned
 * from the scene's cuts — incrementally from `previous`, as the GPU
 * substrate replans on an edit or an undo. A terrain plane is the floor.
 */
export function rigidMirrorSolid(scene: SceneDescription, previous?: SculptBakePlan): MirrorSolid & { readonly plan: SculptBakePlan } {
  if (scene.container.shape === 'sphere') throw new Error('An assembly contacts a box shell: a spherical container has no assembly solid');
  const cells = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER) as [number, number, number];
  const pages = sceneFluidLatticeDimensions(scene, Number.MAX_SAFE_INTEGER).map(n => Math.ceil(n / 32)) as [number, number, number];
  const origin: [number, number, number] = [1, 1, 1];
  const walls: SolidOccupancyMaskWalls = { policy: scene.container.top === 'closed' ? 1 : 2, origin, pages };
  const mask = new SolidOccupancyMask({ walls, tileKeys: 0, separateTerrain: true, origin, halfSolid: scene.solidBoundary?.collision === 'voxels',
    slots: pages[0] * pages[1] * pages[2] + (walls.policy === 2 ? pages[0] * pages[2] : 0) });
  mask.update(solidWorldForFluidLattice(scene));
  const c = scene.container, h: [number, number, number] = [c.width_m / cells[0], c.height_m / cells[1], c.depth_m / cells[2]];
  const o: [number, number, number] = [-c.width_m / 2, 0, -c.depth_m / 2];
  const words = packSvoSculptCuts(scene.solidVoxels, { origin_m: o, cellSize_m: h });
  const plan = planSculptBake(previous, words, false);
  return {
    layout: { cells, origin_m: o, cellSize_m: h, walls, floor: false },
    mask: mask.words, cutActive: plan.pages.size > 0,
    cutPage: page => plan.pages.has(page.join(',')),
    cutDistance: cell => sampleSvoSculptCuts(words, [0, 1, 2].map(a => o[a]! + (cell[a]! + .5) * h[a]!) as [number, number, number]).distance,
    containerPlanes: containerMirrorPlanes(c).length,
    plan,
  };
}
