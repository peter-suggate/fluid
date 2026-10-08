import type { PaneId } from "../gpu-startup";

export type { PaneId };

/** Compare mode is two panes; the wipe stretch is still two draws of the same pair. */
export const MAXIMUM_PANE_LEASES = 2;

export type PaneLeaseResult =
  | { readonly status: "acquired"; readonly release: () => void }
  | { readonly status: "exhausted"; readonly message: string };

export interface PaneLeaseBroker {
  /**
   * Take this pane's slot in the page. A pane that already holds one, or a
   * pane beyond the cap, is refused rather than given a second GPU session.
   */
  acquire(paneId: PaneId): PaneLeaseResult;
  /** Panes currently holding a lease, for diagnostics and tests. */
  heldPanes(): readonly PaneId[];
}

/** Hand out at most `maximumLeases` in-page leases, one per pane. */
export function createPaneLeaseBroker(maximumLeases: number = MAXIMUM_PANE_LEASES): PaneLeaseBroker {
  // Token per pane, so a stale release closure from a previous lease cannot
  // drop the lease a re-mounted pane has since taken.
  const held = new Map<PaneId, number>();
  let nextToken = 0;

  return {
    heldPanes: () => [...held.keys()],
    acquire(paneId: PaneId): PaneLeaseResult {
      if (held.has(paneId)) {
        return { status: "exhausted", message: `Pane ${paneId} already holds a WebGPU pane lease` };
      }
      if (held.size >= maximumLeases) {
        return {
          status: "exhausted",
          message: `All ${maximumLeases} in-page WebGPU pane leases are held (${[...held.keys()].join(", ")})`,
        };
      }
      const token = ++nextToken;
      held.set(paneId, token);
      return { status: "acquired", release: () => { if (held.get(paneId) === token) held.delete(paneId); } };
    },
  };
}

type PaneLeaseWindow = Window & {
  /**
   * Survives Fast Refresh and Vinext RSC program reloads for the same reason
   * the viewport lifecycle does: a fresh module instance with an empty ledger
   * would hand a pane a second lease while its first session is still live.
   */
  __fluidLabGPUPaneLeases?: PaneLeaseBroker;
};

/** The page's one broker; every viewport pane leases through it. */
export function gpuPaneLeaseBroker(): PaneLeaseBroker {
  if (typeof window === "undefined") return createPaneLeaseBroker();
  const host = window as PaneLeaseWindow;
  host.__fluidLabGPUPaneLeases ??= createPaneLeaseBroker();
  return host.__fluidLabGPUPaneLeases;
}

/** Take one pane's lease for the lifetime of its GPU session. */
export function acquirePaneGPULease(paneId: PaneId): PaneLeaseResult {
  return gpuPaneLeaseBroker().acquire(paneId);
}
