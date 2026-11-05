import type { SceneDescription } from '../core/model';
import type { ClothStrokeCut } from './edit';
import type { WetClothSourceField } from './source-types';
import { sourceFieldBounds } from './source-motion';

/** Static source collision consumes the same exact box subtraction as the
 * shared procedural layer. Retaining the old source SDF after a cut would
 * leave an invisible collider behind the edited visible surface. */
export function sceneWithSourceSolidCuts(scene: SceneDescription, cuts: readonly ClothStrokeCut[]): SceneDescription {
  const program = scene.wetClothSource;
  if (!program?.bakedSolidFieldIndices?.length || !cuts.length) return scene;
  let fields = [...program.fields], roots = [...program.bakedSolidFieldIndices], changed = false;
  for (const {box, kind} of cuts) {
    for (let r = 0; r < roots.length; r++) {
      const old = fields.find(f => f.index === roots[r])!;
      if (!old) throw new Error('Missing baked source solid');
      if (!old.inside && old.kind !== 'intersect') {
        const bounds = sourceFieldBounds(fields, old.index);
        if ([0,1,2].some(i => bounds.max[i]! <= box.minimum_m[i]! || bounds.min[i]! >= box.maximum_m[i]!)) continue;
      }
      if (kind !== 'box') throw new Error(`Source solid ${kind} cuts require matching collision geometry; refusing a bounding-box substitute`);
      const cutIndex = Math.max(...fields.map(f => f.index)) + 1;
      const cut: WetClothSourceField = {index: cutIndex, kind: 'box', usage: 'solid', group: old.group, inside: true, root: false, sampled: false,
        center_m: [0,1,2].map(i => (box.minimum_m[i]!+box.maximum_m[i]!)/2) as unknown as readonly [number,number,number],
        halfExtents_m: [0,1,2].map(i => (box.maximum_m[i]!-box.minimum_m[i]!)/2) as unknown as readonly [number,number,number],
        axis: [0,1,0], angle_rad: 0, roundingRadius_m: 0, sourceAttributes: {edit: 'box-subtraction'}};
      fields = fields.map(f => f.index === old.index ? {...f, root: false} : f);
      fields.push(cut, {...old, index: cutIndex+1, kind: 'intersect', inside: false, children: [old.index,cutIndex], root: true,
        sourceAttributes: {edit: 'box-subtraction', sourceField: String(old.index)}});
      roots[r] = cutIndex+1; changed = true;
    }
  }
  return changed ? {...scene, wetClothSource: {...program, fields, bakedSolidFieldIndices: roots}} : scene;
}
