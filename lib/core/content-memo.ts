/**
 * A few most-recent derived tables, keyed by the content they were derived from.
 *
 * A world build derives several tables that depend on a small part of the
 * scene — the backdrop's tiles on its description and the lattice, the render
 * terrain on the terrain's stamp and the cell — while the scene around them
 * changes. Keyed on exactly that content, a rebuild that did not change it
 * takes the table it already has. The values are shared between builds, so a
 * holder must never write to one.
 */
export interface ContentMemo<T> {
  get(key: string): T | undefined;
  set(key: string, value: T): T;
}

export function createContentMemo<T>(capacity = 4): ContentMemo<T> {
  if (!Number.isInteger(capacity) || capacity < 1) throw new RangeError("Content memo capacity must be a positive integer");
  const entries = new Map<string, T>();
  return {
    get(key) {
      const value = entries.get(key);
      if (value === undefined) return undefined;
      // Most recent last: the oldest key is the first the map iterates.
      entries.delete(key);
      entries.set(key, value);
      return value;
    },
    set(key, value) {
      entries.delete(key);
      entries.set(key, value);
      while (entries.size > capacity) entries.delete(entries.keys().next().value!);
      return value;
    },
  };
}
