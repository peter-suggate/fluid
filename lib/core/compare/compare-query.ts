import type { PaneId } from "../session/session";

/**
 * The `b.*` half of the address: pane B, as a diff over pane A.
 *
 * Deliberately free of every other module in the compare feature — the URL
 * writer has to know these key names and this group table, and the writer is
 * imported by half the studio. Anything that needs a store, a session or a
 * simulation lives in `compare-model.ts` instead, which imports this.
 *
 * The contract in one line: **B is A plus a diff.** A key absent from the diff
 * means B reads A's value for it; a key present carries B's own. Nothing here
 * is a second document — see `docs/ab-compare-handoff.md`.
 */

/**
 * The four things a reader looks *with* rather than *at*.
 *
 * Each is linked by default and carries a padlock on the diff strip, because
 * the experiment is about the solver: if the two panes are framed differently,
 * or sliced differently, or shaded differently, the eye compares two viewpoints
 * instead of two simulations. Unlinking a group is the deliberate act of
 * comparing *that* thing instead.
 */
export type CompareLinkGroup = "view" | "cut" | "instrument" | "look";

/**
 * `config` is everything else — the solver, its parameters and the scene
 * document. It has no padlock because it is the whole point of the mode: a
 * config key always records a diff rather than moving both panes.
 */
export type CompareGroup = CompareLinkGroup | "config";

export const COMPARE_LINK_GROUPS: readonly CompareLinkGroup[] = ["view", "cut", "instrument", "look"];

export const COMPARE_GROUP_LABELS: Readonly<Record<CompareLinkGroup, string>> = {
  view: "View",
  cut: "Cut",
  instrument: "Instrument",
  look: "Look",
};

export const COMPARE_GROUP_HINTS: Readonly<Record<CompareLinkGroup, string>> = {
  view: "One camera fed to both draws — orbit either pane and both move",
  cut: "The same slice of the same axis in both panes",
  instrument: "An instrument raised on one pane is raised on both, each reading its own session",
  look: "Render settings identical, so a pixel difference is a physics difference",
};

/** Compare mode with an empty diff still has to survive a reload: `?b=1`. */
export const COMPARE_ACTIVE_KEY = "b";
/** Every diff entry: `b.<managed key>=<value>`. */
export const COMPARE_KEY_PREFIX = "b.";
/** The groups that are *not* linked, listed; omitted when everything is. */
export const COMPARE_LINK_KEY = "b.link";

/**
 * B does not carry this key at all.
 *
 * A diff value has to be able to say "absent", or a key A carries and B does
 * not — A slicing on Z while B slices on nothing — would be unrepresentable and
 * would silently re-inherit A's value on the next mirror. Spelled like
 * `url-state`'s own `~delete` marker so the two read as one convention.
 */
export const COMPARE_ABSENT = "~absent";

export interface CompareLinks {
  readonly view: boolean;
  readonly cut: boolean;
  readonly instrument: boolean;
  readonly look: boolean;
}

export const COMPARE_ALL_LINKED: CompareLinks = Object.freeze({
  view: true, cut: true, instrument: true, look: true,
});

export interface CompareState {
  /** Whether the second pane is mounted at all. */
  readonly active: boolean;
  /** Managed URL key → the value pane B carries instead of pane A's. */
  readonly diff: Readonly<Record<string, string>>;
  readonly links: CompareLinks;
  /** Which pane the keyboard and the ring belong to: last pointer-down or hover. */
  readonly focusedPane: PaneId;
}

export const INITIAL_COMPARE_STATE: CompareState = Object.freeze({
  active: false,
  diff: Object.freeze({}),
  links: COMPARE_ALL_LINKED,
  focusedPane: "a" as PaneId,
});

/**
 * Keys grouped by what they are a property *of*, not by which store holds them.
 *
 * `quality` sits under Look beside the render arms rather than under the solver
 * it is stored with, because it is a statement about how hard the frame is
 * drawn — a difference in it makes every pixel differ for a reason that is not
 * the physics.
 */
const CUT_KEYS: Readonly<Record<string, true>> = {
  grid: true, gridSlice: true, gridMode: true, lensPhase: true,
};

const LOOK_KEYS: Readonly<Record<string, true>> = {
  svoStage: true, svoShadows: true, svoAO: true, svoPrimarySeamClosure: true,
  svoCones: true, svoPrimary: true, svoFlatExempt: true, svoLodPixels: true,
  quality: true,
};

/** Which padlock, if any, governs a managed key. */
export function compareGroupForKey(key: string): CompareGroup {
  if (key.startsWith("camera.")) return "view";
  if (Object.hasOwn(CUT_KEYS, key)) return "cut";
  if (key === "overlay") return "instrument";
  if (Object.hasOwn(LOOK_KEYS, key)) return "look";
  return "config";
}

/**
 * Keys that may never diverge, whatever the padlocks say.
 *
 * `view` is the shell layer — a property of the page, not of a pane, and there
 * is only one page. `scene.randomSeed` is the more interesting one: an A/B
 * whose two runs were seeded differently is not an A/B, so the seed follows A
 * even when the diff has forked the rest of the document.
 */
const NEVER_DIVERGES: Readonly<Record<string, true>> = {
  view: true,
  "scene.randomSeed": true,
};

export function compareKeyIsDivergeable(key: string): boolean {
  if (Object.hasOwn(NEVER_DIVERGES, key)) return false;
  return key !== COMPARE_ACTIVE_KEY && !key.startsWith(COMPARE_KEY_PREFIX);
}

/** True for the address keys this module owns, so the URL writer can clear them. */
export function isCompareQueryKey(key: string): boolean {
  return key === COMPARE_ACTIVE_KEY || key.startsWith(COMPARE_KEY_PREFIX);
}

/** Groups a reader has deliberately unlinked, in a stable order. */
export function unlinkedCompareGroups(links: CompareLinks): readonly CompareLinkGroup[] {
  return COMPARE_LINK_GROUPS.filter((group) => !links[group]);
}

/**
 * The `b.*` block for the address bar.
 *
 * Never a second scene: the whole of pane B is these few keys, which is what
 * makes a compare link readable and what makes a reload restore both panes.
 */
export function compareQueryEntries(
  state: CompareState | undefined,
): readonly (readonly [key: string, value: string])[] {
  if (!state?.active) return [];
  const entries: (readonly [string, string])[] = [];
  const keys = Object.keys(state.diff).sort();
  // The bare flag exists so "compare is open and nothing differs yet" survives
  // a reload; with any diff key present the mode is already implied.
  if (keys.length === 0) entries.push([COMPARE_ACTIVE_KEY, "1"]);
  for (const key of keys) entries.push([`${COMPARE_KEY_PREFIX}${key}`, state.diff[key]!]);
  const unlinked = unlinkedCompareGroups(state.links);
  if (unlinked.length > 0) entries.push([COMPARE_LINK_KEY, unlinked.join(",")]);
  return entries;
}

/** Read the `b.*` block back. Anything unrecognised resolves to "not compare". */
export function parseCompareQuery(search: string): CompareState {
  const query = new URLSearchParams(search);
  const diff: Record<string, string> = {};
  let sawCompareKey = query.get(COMPARE_ACTIVE_KEY) !== null;
  const links: { -readonly [Key in keyof CompareLinks]: boolean } = { ...COMPARE_ALL_LINKED };
  for (const [key, value] of query) {
    if (!key.startsWith(COMPARE_KEY_PREFIX)) continue;
    sawCompareKey = true;
    if (key === COMPARE_LINK_KEY) {
      for (const group of value.split(",")) {
        if ((COMPARE_LINK_GROUPS as readonly string[]).includes(group)) links[group as CompareLinkGroup] = false;
      }
      continue;
    }
    const managed = key.slice(COMPARE_KEY_PREFIX.length);
    if (!compareKeyIsDivergeable(managed)) continue;
    diff[managed] = value;
  }
  return { active: sawCompareKey, diff, links, focusedPane: "a" };
}
