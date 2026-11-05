import type { SceneDescription } from '../core/model';
import type { ClothDocument } from './types';
import type { WetClothSourceProgram } from './source-types';

const documents = new WeakMap<ClothDocument, string>();
const programs = new WeakMap<WetClothSourceProgram, string>();
function stamp(value: unknown): string {
  const encoded = JSON.stringify(value);
  let a = 2166136261, b = 2246822519;
  for (let i = 0; i < encoded.length; i++) {
    const c = encoded.charCodeAt(i);
    a = Math.imul(a ^ c, 16777619); b = Math.imul(b ^ c, 3266489917);
  }
  return `${encoded.length.toString(36)}-${(a >>> 0).toString(16)}${(b >>> 0).toString(16)}`;
}
/** Immutable documents make hashing proportional to edits, not render frames.
 * This belongs to live uniform adoption, never the fluid seed/restart key. */
export function clothDocumentsKey(cloth: readonly ClothDocument[] | undefined): string {
  const parts = (cloth ?? []).map(document => {
    let value = documents.get(document);
    if (!value) { value = stamp(document); documents.set(document, value); }
    return `${document.id}:${value}`;
  });
  return parts.length ? parts.join('|') : 'no-cloth';
}

export function clothSceneContentStamp(scene: Pick<SceneDescription, 'cloth' | 'wetClothSource'>): string {
  const parts = [clothDocumentsKey(scene.cloth)];
  if (scene.wetClothSource) {
    let value = programs.get(scene.wetClothSource);
    if (!value) { value = stamp(scene.wetClothSource); programs.set(scene.wetClothSource, value); }
    parts.push(`source:${value}`);
  }
  return parts.length ? parts.join('|') : 'no-cloth';
}
