/** Small bounded cache shared by immutable CPU-side SVO publications. */
export const SVO_STATIC_PUBLICATION_CACHE_MAXIMUM_ENTRIES = 64;

function fnvStep(hash: number, value: number): number {
  return Math.imul((hash ^ value) >>> 0, 0x01000193) >>> 0;
}

export function hashSvoStaticPublication(
  words: Uint32Array,
  metadata = "",
): string {
  let hash = 0x811c9dc5;
  for (const word of words) {
    hash = fnvStep(hash, word & 0xff);
    hash = fnvStep(hash, (word >>> 8) & 0xff);
    hash = fnvStep(hash, (word >>> 16) & 0xff);
    hash = fnvStep(hash, word >>> 24);
  }
  for (let index = 0; index < metadata.length; index += 1) {
    const code = metadata.charCodeAt(index);
    hash = fnvStep(hash, code & 0xff);
    hash = fnvStep(hash, code >>> 8);
  }
  return hash.toString(16).padStart(8, "0");
}

/**
 * Return the first immutable publication built for a content key. Typed-array
 * records are shared by identity; callers must treat them as read-only GPU
 * upload sources. The bounded FIFO policy prevents editor scene churn from
 * retaining an unbounded number of publications.
 */
export function internSvoStaticPublication<T extends object>(
  cache: Map<string, T>,
  cacheKey: string,
  publication: T,
): T {
  const cached = cache.get(cacheKey);
  if (cached) return cached;
  if (cache.size >= SVO_STATIC_PUBLICATION_CACHE_MAXIMUM_ENTRIES) {
    const oldest = cache.keys().next().value as string | undefined;
    if (oldest !== undefined) cache.delete(oldest);
  }
  cache.set(cacheKey, publication);
  return publication;
}

export function cachedSvoStaticPublication<T extends object>(
  cache: ReadonlyMap<string, T>,
  cacheKey: string,
): T | undefined {
  return cache.get(cacheKey);
}
