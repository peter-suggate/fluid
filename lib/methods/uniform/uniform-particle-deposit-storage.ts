/** DEPB and FIN run after SORT consumes the source particle bank, and before
 * the paired tail extension borrows it. Preserve the previous deposit capacity
 * and gather fallback on small adapters; only change the backing store. */
export function particleDepositStorage(prefixWords: number, poolTiles: number, maxBytes: number, particleBankBytes: number) {
  const wordsPerTile = 6 * 64 + 1;
  const tiles = Math.max(0, Math.min(poolTiles, Math.floor((Math.floor(maxBytes / 4) - prefixWords) / wordsPerTile)));
  const depositBytes = 4 * wordsPerTile * tiles;
  const shared = depositBytes <= particleBankBytes;
  return { tiles, shared, offsetWords: shared ? 0 : prefixWords,
    binsBytes: 4 * prefixWords + (shared ? 0 : depositBytes), depositBytes };
}
