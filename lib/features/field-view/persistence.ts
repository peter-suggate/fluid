import type { QueryCodec } from "../../framework/persistence";

/**
 * A field view in the address bar: which lens is over the water, and which
 * annotations are on it.
 *
 * The two keys are here rather than in either host because they are the same
 * two facts on both: `FieldViewRows` and `FieldOverlayRows` are one row each,
 * mounted at the 3-D tank's corner and at the 2-D slice's, and a link that
 * restored the lens on one page and not the other would be the row's own state
 * having two different amounts of permanence.
 *
 * What the *roster* is, on the other hand, is entirely the host's — which is
 * why the acceptance test is a predicate rather than a list. The studio's lens
 * vocabulary is open-ended by construction: `parseGridOverlayMode` is five
 * independent predicates over a dense-grid record, the octree technique modes,
 * the sparse dirty modes, the pressure journal and the stage lenses, and any
 * attempt to flatten that into an array here would either narrow what a shipped
 * link may say or force this file to import four solvers. The lab's roster is a
 * closed sixteen-entry table and its predicate is `includes`. One codec, two
 * honest answers.
 *
 * The two hosts never share an address — `/scene` and `/advance-lab` are
 * different pages and `startHostQueryStateSync`'s path gate means exactly one of
 * them ever writes — so `gridMode` means "the field view of the page this link
 * names", which is always one roster.
 */

export const FIELD_VIEW_MODE_QUERY_KEY = "gridMode";
export const FIELD_VIEW_OVERLAY_QUERY_KEY = "overlays";
/**
 * `_` and not `,`, so the value survives `URLSearchParams` unescaped: the
 * urlencoded serializer keeps alphanumerics and `*-._` and turns the obvious
 * comma into `%2C`. The same reason `regions=` separates its fields this way.
 */
const FIELD_VIEW_OVERLAY_SEPARATOR = "_";

export interface FieldViewQueryState {
  readonly mode: string;
  /** In roster order, whatever order the link listed them in. */
  readonly overlays: readonly string[];
}

export interface FieldViewRoster {
  /** The lens this host opens on, and the one a link does not have to name. */
  readonly initialMode: string;
  /** Whether this host can draw the named lens at all. */
  accepts(raw: string): boolean;
  /** Every annotation this host offers, in the order its rows list them. */
  readonly overlays: readonly string[];
  readonly modeKey?: string;
  readonly overlayKey?: string;
}

export function fieldViewQuery(roster: FieldViewRoster): QueryCodec<FieldViewQueryState> {
  const modeKey = roster.modeKey ?? FIELD_VIEW_MODE_QUERY_KEY;
  const overlayKey = roster.overlayKey ?? FIELD_VIEW_OVERLAY_QUERY_KEY;
  return {
    keys: [modeKey, overlayKey],
    read: (query) => {
      const mode = query.get(modeKey);
      const asked = new Set((query.get(overlayKey) ?? "")
        .split(FIELD_VIEW_OVERLAY_SEPARATOR).filter((id) => id.length > 0));
      return {
        mode: mode !== null && roster.accepts(mode) ? mode : roster.initialMode,
        // Roster order rather than the link's: the set is what is on, and a
        // canonical write has to produce one string per set or the mirror would
        // rewrite the address on every read.
        overlays: roster.overlays.filter((id) => asked.has(id)),
      };
    },
    write: (query, state) => {
      query.delete(modeKey);
      query.delete(overlayKey);
      // An untouched row is the absence of its key, so an ordinary link does not
      // have to say which annotations it is not drawing.
      if (state.mode !== roster.initialMode) query.set(modeKey, state.mode);
      const on = roster.overlays.filter((id) => state.overlays.includes(id));
      if (on.length > 0) query.set(overlayKey, on.join(FIELD_VIEW_OVERLAY_SEPARATOR));
    },
  };
}
