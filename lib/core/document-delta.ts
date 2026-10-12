/**
 * What separates two immutable documents, small enough to cross a worker
 * boundary in place of the document.
 *
 * Documents here are replaced, never edited: a change makes new objects along
 * the path to what changed and leaves every other part the object it was. So
 * a part that is the same object on both sides is skipped without being
 * entered, and the cost of a delta follows the change, not the document.
 *
 * A part that is another object is compared by content, because a document
 * that was copied whole is still the same document. Only what differs becomes
 * an entry; an equal document under a new identity has none.
 *
 * The receiver applies the entries with the same discipline: new objects along
 * each path, every other part the object it already held. Whatever it keys on
 * the identity of a part therefore stays valid across an update.
 */

/** One part that changed: where it is, and what stands there now. */
export interface DocumentDeltaEntry {
  path: (string | number)[];
  value?: unknown;
  /** The key is gone, rather than holding `value`. */
  removed?: true;
}

/** The entries name a part the held document does not have: it is not the document they were made against. */
export class DocumentDeltaError extends Error {}

/** Parts are named this far down; a part that differs below is sent whole at this depth. */
const MAXIMUM_DEPTH = 12;
/** Past this many entries the two are different documents, and the caller sends the new one whole. */
const MAXIMUM_ENTRIES = 16_384;

const ARRAY = 1, RECORD = 2, VIEW = 3;
function kindOf(value: unknown): number {
  if (typeof value !== "object" || value === null) return 0;
  if (Array.isArray(value)) return ARRAY;
  if (ArrayBuffer.isView(value)) return VIEW;
  const prototype = Object.getPrototypeOf(value) as unknown;
  return prototype === Object.prototype || prototype === null ? RECORD : 0;
}

function sameView(a: ArrayBufferView, b: ArrayBufferView): boolean {
  if (a.constructor !== b.constructor || a.byteLength !== b.byteLength) return false;
  const left = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  const right = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  for (let index = 0; index < left.length; index += 1) if (left[index] !== right[index]) return false;
  return true;
}

/** Whether two parts hold the same content, entered only where they are not the same object. */
function sameContent(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  const kind = kindOf(a);
  if (kind !== kindOf(b)) return false;
  if (kind === 0) return Object.is(a, b);
  if (kind === VIEW) return sameView(a as ArrayBufferView, b as ArrayBufferView);
  if (kind === ARRAY) {
    const left = a as unknown[], right = b as unknown[];
    if (left.length !== right.length) return false;
    for (let index = 0; index < left.length; index += 1) {
      const x = left[index], y = right[index];
      if (x !== y && !sameContent(x, y)) return false;
    }
    return true;
  }
  const left = a as Record<string, unknown>, right = b as Record<string, unknown>;
  let unmatched = 0;
  for (const key in left) unmatched += 1;
  for (const key in right) {
    const x = left[key], y = right[key];
    // A value that is there and the same is held; nothing a document can hold is inherited.
    if ((x !== y || x === undefined) && (!Object.hasOwn(left, key) || !sameContent(x, y))) return false;
    unmatched -= 1;
  }
  return unmatched === 0;
}

/**
 * The parts of `next` that differ from `base`, or `undefined` when the two are
 * not versions of one document — another kind of thing at the root, or more
 * differences than are worth naming — and `next` should cross whole.
 *
 * Arrays are compared element by element when they are the same length, and
 * sent whole when they are not.
 */
export function documentDelta(base: unknown, next: unknown): DocumentDeltaEntry[] | undefined {
  const entries: DocumentDeltaEntry[] = [];
  const path: (string | number)[] = [];
  const emit = (value: unknown, removed: boolean): boolean => {
    if (path.length === 0 || entries.length >= MAXIMUM_ENTRIES) return false;
    entries.push(removed ? { path: path.slice(), removed: true } : { path: path.slice(), value });
    return true;
  };
  /** False once the delta is abandoned. */
  const walk = (a: unknown, b: unknown): boolean => {
    if (a === b) return true;
    const kind = kindOf(a);
    if (kind !== kindOf(b)) return emit(b, false);
    if (kind === 0) return Object.is(a, b) || emit(b, false);
    if (kind === VIEW) return sameView(a as ArrayBufferView, b as ArrayBufferView) || emit(b, false);
    if (path.length >= MAXIMUM_DEPTH) return sameContent(a, b) || emit(b, false);
    if (kind === ARRAY) {
      const left = a as unknown[], right = b as unknown[];
      if (left.length !== right.length) return emit(b, false);
      for (let index = 0; index < right.length; index += 1) {
        const x = left[index], y = right[index];
        if (x === y) continue;
        path.push(index);
        const kept = walk(x, y);
        path.pop();
        if (!kept) return false;
      }
      return true;
    }
    const left = a as Record<string, unknown>, right = b as Record<string, unknown>;
    let unmatched = 0;
    for (const key in left) unmatched += 1;
    for (const key in right) {
      const x = left[key], y = right[key];
      // A value that is there and the same is held; nothing a document can hold is inherited.
      if (x === y && x !== undefined) { unmatched -= 1; continue; }
      const held = Object.hasOwn(left, key);
      if (held) unmatched -= 1;
      path.push(key);
      const kept = held ? walk(x, y) : emit(y, false);
      path.pop();
      if (!kept) return false;
    }
    if (unmatched === 0) return true;
    for (const key in left) {
      if (Object.hasOwn(right, key)) continue;
      path.push(key);
      const kept = emit(undefined, true);
      path.pop();
      if (!kept) return false;
    }
    return true;
  };
  return walk(base, next) ? entries : undefined;
}

/**
 * `base` with the entries applied: a new object along each entry's path, and
 * every other part the object `base` holds. No entries is `base` itself.
 */
export function applyDocumentDelta<T>(base: T, entries: readonly DocumentDeltaEntry[]): T {
  if (entries.length === 0) return base;
  const fresh = new Set<unknown>();
  const own = (part: unknown, entry: DocumentDeltaEntry, depth: number): Record<string | number, unknown> => {
    if (fresh.has(part)) return part as Record<string | number, unknown>;
    const kind = kindOf(part);
    if (kind !== ARRAY && kind !== RECORD) {
      throw new DocumentDeltaError(`Document delta names ${entry.path.join(".")}, and the held document has nothing to hold `
        + `${entry.path.slice(0, depth).join(".") || "its root"}.`);
    }
    const copy = kind === ARRAY ? (part as unknown[]).slice() : { ...(part as object) };
    fresh.add(copy);
    return copy as Record<string | number, unknown>;
  };
  const root = own(base, entries[0]!, 0);
  for (const entry of entries) {
    const last = entry.path.length - 1;
    if (last < 0) throw new DocumentDeltaError("Document delta names the root, which crosses whole.");
    let part = root;
    for (let depth = 0; depth < last; depth += 1) {
      const key = entry.path[depth]!;
      const child = own(part[key], entry, depth + 1);
      part[key] = child;
      part = child;
    }
    if (entry.removed) delete part[entry.path[last]!];
    else part[entry.path[last]!] = entry.value;
  }
  return root as T;
}
