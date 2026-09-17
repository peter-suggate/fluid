/**
 * The address bar as a mirror, without knowing what is being mirrored.
 *
 * This is the loop `startQueryStateSync` has always run for the studio, with
 * everything studio-shaped lifted out into the three things a host supplies:
 * the path it owns, how to turn its stores into a query string, and how to read
 * one back. Nothing here names a scene, a preset, a method or a store — which
 * is the whole reason it can be extracted at all, and the reason the 2-D lab
 * can mirror *its* state through the same three flags rather than a second
 * hand-written copy of them.
 *
 * The three flags are the interesting part, and they are the ones a second copy
 * would get subtly wrong:
 *
 * - `active` survives teardown. A microtask already queued when a route
 *   unmounts must not write the address of the page that replaced it.
 * - `applyingUrl` is up for the whole of a hydration. Hydration writes every
 *   store, each write notifies subscribers, and a subscriber that scheduled a
 *   mirror would race the rest of the hydration and canonicalise half a page.
 * - `queued` coalesces. One reader gesture moves several stores; the address is
 *   written once, at the end of the microtask, from whatever they all settled
 *   on. This is why anything that *reads* `location.search` must force a write
 *   first — the mirror is deliberately a microtask behind.
 *
 * The path gate is what keeps a retained-but-hidden route from fighting for the
 * address bar. `AppShell` keeps the studio mounted while the library is up, and
 * a lab tab does the same to the studio; whoever owns the current path writes.
 */

export interface HostQueryStateSync {
  /** The route this host owns. `/x` and `/x/…` are both this host. */
  readonly path: string;
  /** The canonical query for the host's current state, preserving foreign keys. */
  serialize(search: string): string;
  /** Write a whole query onto the host's stores. */
  hydrate(search: string): void;
  /**
   * The stores to mirror, each as its own subscribe-with-teardown.
   *
   * A closure rather than a store, because *which* changes are worth an address
   * write is the host's judgement: the studio mirrors every scene edit but only
   * the shell's view and compare record, and the lab mirrors a lens change but
   * not a playback tick.
   */
  readonly sources: readonly ((onChange: () => void) => () => void)[];
  /** False when a client navigation or Fast Refresh already retained the stores. */
  readonly hydrateFromUrl?: boolean;
}

/**
 * Put a query string in the address bar, in place.
 *
 * Replace and never push: the URL describes what is on screen, and a back
 * button that stepped through every panel toggle would never reach the page the
 * reader actually arrived from. Compared before writing, so a mirror that
 * changed nothing costs no history entry at all.
 */
export function replaceLocationSearch(search: string): void {
  const next = `${window.location.pathname}${search ? `?${search}` : ""}${window.location.hash}`;
  const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (next !== current) window.history.replaceState(window.history.state, "", next);
}

/**
 * Hydrate the host's stores once, then mirror them to `history.replaceState`.
 *
 * Popstate follows the same path, so back/forward, reloads and development
 * module replacement all rebuild the page from one coherent store snapshot
 * rather than from whichever half of it a partial write left behind.
 */
export function startHostQueryStateSync(sync: HostQueryStateSync): () => void {
  let active = true;
  let queued = false;
  let applyingUrl = false;
  const onPath = () => window.location.pathname === sync.path
    || window.location.pathname.startsWith(`${sync.path}/`);

  const writeUrl = () => {
    if (!active || applyingUrl || !onPath()) return;
    replaceLocationSearch(sync.serialize(window.location.search));
  };

  const scheduleWrite = () => {
    if (queued || applyingUrl) return;
    queued = true;
    queueMicrotask(() => { queued = false; writeUrl(); });
  };

  const hydrate = () => {
    applyingUrl = true;
    sync.hydrate(window.location.search);
    applyingUrl = false;
    // Immediately, not on a microtask: what a reader typed is not yet canonical,
    // and a link carrying a retired key has to lose it before anything reads the
    // address back.
    writeUrl();
  };

  if (sync.hydrateFromUrl === false) writeUrl();
  else hydrate();
  const stops = sync.sources.map((source) => source(scheduleWrite));
  const hydrateOwnPath = () => { if (onPath()) hydrate(); };
  window.addEventListener("popstate", hydrateOwnPath);

  return () => {
    active = false;
    for (const stop of stops) stop();
    window.removeEventListener("popstate", hydrateOwnPath);
  };
}
