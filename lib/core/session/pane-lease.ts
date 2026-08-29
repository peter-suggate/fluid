import {
  acquireBrowserGPULease,
  type BrowserGPULeaseResult,
  type PaneId,
} from "../gpu-startup";

export type { PaneId };

/** Compare mode is two panes; the wipe stretch is still two draws of the same pair. */
export const MAXIMUM_PANE_LEASES = 2;

export type PaneLeaseResult = BrowserGPULeaseResult;

export interface PaneLeaseBroker {
  /**
   * Take this pane's share of the page's one WebGPU lock.
   *
   * The first caller acquires the cross-tab Web Lock; later callers ride it.
   * A refusal is returned verbatim, so a second *tab* still reads "Another
   * Fluid Lab tab owns the WebGPU safety lock" exactly as it did with one pane.
   */
  acquire(paneId: PaneId): Promise<PaneLeaseResult>;
  /** Panes currently holding a lease, for diagnostics and tests. */
  heldPanes(): readonly PaneId[];
  /** Whether the page-exclusive Web Lock is currently held by this page. */
  lockHeld(): boolean;
}

/** The lock request the broker makes once per page, not once per pane. */
export type PaneLockRequest = () => Promise<BrowserGPULeaseResult>;

type LockOutcome = { readonly ok: true } | { readonly ok: false; readonly refusal: PaneLeaseResult };

/**
 * Hand out at most `maximumLeases` in-page leases against one Web Lock.
 *
 * Cross-tab exclusivity is unchanged: the lock is still an `ifAvailable`
 * exclusive request, still made once, and still held until the *last* pane
 * releases. What grows is only the number of holders inside this page.
 */
export function createPaneLeaseBroker(
  requestLock: PaneLockRequest,
  maximumLeases: number = MAXIMUM_PANE_LEASES,
): PaneLeaseBroker {
  // Token per pane, so a stale release closure from a previous lease cannot
  // drop the lease a re-mounted pane has since taken.
  const held = new Map<PaneId, number>();
  let nextToken = 0;
  let releaseLock: (() => void) | undefined;
  let pendingLock: Promise<LockOutcome> | undefined;

  const releasePane = (paneId: PaneId, token: number) => {
    if (held.get(paneId) !== token) return;
    held.delete(paneId);
    if (held.size > 0) return;
    const release = releaseLock;
    releaseLock = undefined;
    release?.();
  };

  const ensureLock = (): Promise<LockOutcome> => {
    if (releaseLock !== undefined) return Promise.resolve({ ok: true });
    // One request even when both panes start in the same tick: the second
    // pane awaits the first pane's acquisition rather than racing it into a
    // self-refusal against a lock this very page already owns.
    pendingLock ??= (async (): Promise<LockOutcome> => {
      const lock = await requestLock();
      if (lock.status !== "acquired") return { ok: false, refusal: lock };
      releaseLock = lock.release;
      return { ok: true };
    })().finally(() => { pendingLock = undefined; });
    return pendingLock;
  };

  return {
    heldPanes: () => [...held.keys()],
    lockHeld: () => releaseLock !== undefined,
    async acquire(paneId: PaneId): Promise<PaneLeaseResult> {
      if (held.has(paneId)) {
        return { status: "exhausted", message: `Pane ${paneId} already holds a WebGPU pane lease` };
      }
      if (held.size >= maximumLeases) {
        return {
          status: "exhausted",
          message: `All ${maximumLeases} in-page WebGPU pane leases are held (${[...held.keys()].join(", ")})`,
        };
      }
      // Claim the slot before awaiting, so a third pane starting in the same
      // tick is refused rather than admitted behind the first two.
      const token = ++nextToken;
      held.set(paneId, token);
      let outcome: LockOutcome;
      try {
        outcome = await ensureLock();
      } catch {
        releasePane(paneId, token);
        return { status: "error", message: "The browser WebGPU safety lock failed" };
      }
      if (!outcome.ok) {
        // Includes `unsupported`: a browser without Web Locks refuses every
        // pane exactly as it refused the single pane, and the viewport starts
        // anyway. Nothing is recorded, so nothing has to be released.
        releasePane(paneId, token);
        return outcome.refusal;
      }
      return { status: "acquired", release: () => releasePane(paneId, token) };
    },
  };
}

type PaneLeaseWindow = Window & {
  /**
   * Survives Fast Refresh and Vinext RSC program reloads for the same reason
   * the viewport lifecycle does: a fresh module instance with an empty ledger
   * would ask for a lock this page already holds, and refuse itself.
   */
  __fluidLabGPUPaneLeases?: PaneLeaseBroker;
};

function browserLockManager(): Parameters<typeof acquireBrowserGPULease>[0] {
  if (typeof navigator === "undefined" || !("locks" in navigator)) return undefined;
  return navigator.locks as Parameters<typeof acquireBrowserGPULease>[0];
}

/** The page's one broker; every viewport pane leases through it. */
export function gpuPaneLeaseBroker(): PaneLeaseBroker {
  const request: PaneLockRequest = () => acquireBrowserGPULease(browserLockManager());
  if (typeof window === "undefined") return createPaneLeaseBroker(request);
  const host = window as PaneLeaseWindow;
  host.__fluidLabGPUPaneLeases ??= createPaneLeaseBroker(request);
  return host.__fluidLabGPUPaneLeases;
}

/** Take one pane's lease against the page-exclusive WebGPU lock. */
export function acquirePaneGPULease(paneId: PaneId): Promise<PaneLeaseResult> {
  return gpuPaneLeaseBroker().acquire(paneId);
}
