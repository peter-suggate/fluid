import type { ClothDocument } from './types';

const documents = new WeakMap<ClothDocument, string>();
function stamp(value: unknown): string {
  const encoded = JSON.stringify(value);
  let a = 2166136261, b = 2246822519;
  for (let i = 0; i < encoded.length; i++) {
    const c = encoded.charCodeAt(i);
    a = Math.imul(a ^ c, 16777619); b = Math.imul(b ^ c, 3266489917);
  }
  return `${encoded.length.toString(36)}-${(a >>> 0).toString(16)}${(b >>> 0).toString(16)}`;
}
/** Content identity of the authored cloth roster. Immutable documents make
 * hashing proportional to edits, not frames; never part of the fluid seed. */
export function clothDocumentsKey(cloth: readonly ClothDocument[] | undefined): string {
  const parts = (cloth ?? []).map(document => {
    let value = documents.get(document);
    if (!value) { value = stamp(document); documents.set(document, value); }
    return `${document.id}:${value}`;
  });
  return parts.length ? parts.join('|') : 'no-cloth';
}
