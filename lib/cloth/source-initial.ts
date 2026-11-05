import type { SceneDescription, Vec3 } from '../core/model';
import type { WetClothSourceProgram } from './source-types';
import { sourceFieldDistance } from './source-fields';
import { sourceFieldBounds, type SourceBounds } from './source-motion';

/** Source CSG/rounded shapes seed the real initial phi and cell-volume path.
 * These are analytic fields, never their bounding boxes as liquid. */
export function sourceInitialSignedDistance(scene: Pick<SceneDescription,'wetClothSource'>, point: Vec3): number | undefined {
  const p = scene.wetClothSource, indices = p?.initialSdfFieldIndices;
  if (!p || !indices?.length) return undefined;
  return Math.min(...indices.map(i => sourceFieldDistance(p.fields,i,[point.x,point.y,point.z])));
}
export function sourceInitialBounds(scene: Pick<SceneDescription,'wetClothSource'>): readonly SourceBounds[] {
  const p = scene.wetClothSource;
  return p?.initialSdfFieldIndices?.map(i => sourceFieldBounds(p.fields,i)) ?? [];
}
const keys = new WeakMap<WetClothSourceProgram,string>();
/** Solid-only edits must not reseed the free water. Hash source descendants only. */
export function sourceInitialLiquidKey(scene: Pick<SceneDescription,'wetClothSource'>): string {
  const p = scene.wetClothSource;
  if (!p?.initialSdfFieldIndices?.length) return '';
  let key = keys.get(p);
  if (key !== undefined) return key;
  const used = new Set<number>();
  const visit = (id:number) => {if (used.has(id)) return;used.add(id);for (const child of p.fields.find(f=>f.index===id)?.children??[]) visit(child);};
  p.initialSdfFieldIndices.forEach(visit);
  key = JSON.stringify([p.initialSdfFieldIndices,p.fields.filter(f=>used.has(f.index))]);keys.set(p,key);return key;
}
