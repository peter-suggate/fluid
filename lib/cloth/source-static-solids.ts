import type { SceneDescription } from '../core/model';
import { sceneLatticeDimensions } from '../core/scene-lattice-dimensions';
import { bakeProceduralSolid } from '../core/procedural-solid';
import { sourceFieldDistance } from './source-fields';
import type { ClothVector } from './types';

/** Sample the actual source SDF at the original fluid lattice nodes. The
 * existing procedural-solid representation is one authority for rendering,
 * hit testing, cell/face collision and local edits; there is no proxy AABB. */
export function withStaticSourceSolids(scene: SceneDescription): SceneDescription {
  const program = scene.wetClothSource;
  if (!program) return scene;
  const moving = new Set(program.scripts.map(s => s.group));
  const roots = program.fields.filter(f => f.root && f.usage === 'solid' && !moving.has(f.group));
  if (!roots.length) return scene;
  if (scene.solidBoundary || scene.solidVoxels.length) throw new Error('Source solid compilation needs an empty authored solid layer');
  const dimensions = sceneLatticeDimensions(scene), h = scene.voxelDomain.finestCellSize_m;
  if (dimensions.some(n => n % 8) || [scene.container.width_m,scene.container.height_m,scene.container.depth_m].some((v,i) => Math.abs(v / dimensions[i]! - h) > 1e-12)) throw new Error('Source solids require their unchanged isotropic lattice spacing');
  const origin: ClothVector = [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2];
  const baked = bakeProceduralSolid(dimensions, [h,h,h], p => {
    const world: ClothVector = [origin[0]+p[0]*h, origin[1]+p[1]*h, origin[2]+p[2]*h];
    return Math.min(...roots.map(f => sourceFieldDistance(program.fields, f.index, world))) / h;
  });
  const hasMovingSolids = program.fields.some(f => f.root && f.usage === 'solid' && moving.has(f.group));
  const requiredCapabilities = [...program.requiredCapabilities.filter(c => c !== 'source-solid-collision' || hasMovingSolids), 'source-static-sdf'];
  return {...scene, ...baked, wetClothSource: {...program, requiredCapabilities, bakedSolidFieldIndices: roots.map(f => f.index)}};
}
