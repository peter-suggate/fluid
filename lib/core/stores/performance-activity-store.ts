import { create, type StoreApi, type UseBoundStore } from "zustand";
import {
  mergePerformanceActivityFrame,
  performanceActivityAdditionMatchesIdentity,
  type ActivityFrameIdentity,
  type PerformanceActivityFrame,
  type PerformanceActivityFrameAddition,
  type PerformanceActivityEvidenceIngestResult,
} from "../performance-activity";

export const DEFAULT_PERFORMANCE_ACTIVITY_HISTORY_LIMIT = 8;

export interface PendingPerformanceActivityEvidence {
  identity: ActivityFrameIdentity;
  additions: PerformanceActivityFrameAddition[];
}

export interface PerformanceActivityStore {
  enabled: boolean;
  generation: number;
  /** Epoch-aligned enable instant, safe across CPU performance contexts. */
  enabledAt_epoch_ms: number;
  historyLimit: number;
  history: PerformanceActivityFrame[];
  /** Evidence whose base frame has not been published yet. Cleared on generations. */
  pendingEvidence: PendingPerformanceActivityEvidence[];
  latest?: PerformanceActivityFrame;
  selectedFrameId?: string;
  referenceFrameId?: string;
  setEnabled(enabled: boolean): void;
  /** Explicitly invalidate pending producers without changing enable state. */
  beginGeneration(): number;
  /** Returns false when disabled, stale, pre-enable, or malformed. */
  publish(frame: PerformanceActivityFrame): boolean;
  /** Merge now when retained, otherwise buffer until this exact frame is published. */
  ingestEvidence(
    identity: ActivityFrameIdentity,
    addition: PerformanceActivityFrameAddition,
  ): PerformanceActivityEvidenceIngestResult;
  /** Merge asynchronous CPU/worker/GPU readback evidence into a retained frame. */
  mergeEvidence(frameId: string, addition: PerformanceActivityFrameAddition): boolean;
  selectFrame(frameId?: string): void;
  /** Pin a retained frame as the comparison reference; omit to clear it. */
  pinReference(frameId?: string): void;
  clear(): void;
}

const epochNow_ms = () => performance.timeOrigin + performance.now();

export function createPerformanceActivityStore(
  historyLimit = DEFAULT_PERFORMANCE_ACTIVITY_HISTORY_LIMIT,
  initialEnabled = false,
): UseBoundStore<StoreApi<PerformanceActivityStore>> {
  const boundedLimit = Math.max(1, Math.floor(historyLimit));
  return create<PerformanceActivityStore>((set, get) => ({
    enabled: initialEnabled,
    generation: initialEnabled ? 1 : 0,
    enabledAt_epoch_ms: initialEnabled ? epochNow_ms() : Infinity,
    historyLimit: boundedLimit,
    history: [],
    pendingEvidence: [],
    setEnabled: (enabled) => set((state) => {
      if (enabled === state.enabled) return state;
      const generation = state.generation + 1;
      return {
        enabled,
        generation,
        enabledAt_epoch_ms: enabled ? epochNow_ms() : Infinity,
        // No capture generation may silently mix with held samples.
        history: [], pendingEvidence: [], latest: undefined, selectedFrameId: undefined, referenceFrameId: undefined,
      };
    }),
    beginGeneration: () => {
      const generation = get().generation + 1;
      set((state) => ({
        generation,
        enabledAt_epoch_ms: state.enabled ? epochNow_ms() : Infinity,
        history: [],
        pendingEvidence: [],
        latest: undefined,
        selectedFrameId: undefined,
        referenceFrameId: undefined,
      }));
      return generation;
    },
    publish: (frame) => {
      const state = get();
      if (!state.enabled
        || frame.identity.generation !== state.generation
        || frame.capturedAt_epoch_ms < state.enabledAt_epoch_ms
        || !Number.isFinite(frame.capturedAt_epoch_ms)) return false;
      // A base frame is immutable once retained. Controller report cadence can
      // observe the same completed GPU trace twice; replacing it would erase
      // heartbeat evidence that already merged asynchronously.
      if (state.history.some((candidate) =>
        candidate.identity.frameId === frame.identity.frameId
        && candidate.identity.generation === frame.identity.generation)) return true;
      const pending = state.pendingEvidence.find((candidate) =>
        candidate.identity.frameId === frame.identity.frameId
        && candidate.identity.generation === frame.identity.generation);
      let completedFrame = frame;
      try {
        for (const addition of pending?.additions ?? []) {
          completedFrame = mergePerformanceActivityFrame(completedFrame, addition);
        }
      } catch {
        return false;
      }
      set((current) => {
        // Re-check after producer work in case an enable transition raced it.
        if (!current.enabled || current.generation !== frame.identity.generation) return current;
        const history = appendBoundedFrame(
          current.history.filter((candidate) => candidate.identity.frameId !== frame.identity.frameId),
          completedFrame,
          current.historyLimit,
          current.referenceFrameId,
        );
        const referenceFrameId = current.referenceFrameId
          && history.some((candidate) => candidate.identity.frameId === current.referenceFrameId)
          ? current.referenceFrameId : undefined;
        const followedLatest = current.selectedFrameId === undefined
          || current.selectedFrameId === current.latest?.identity.frameId;
        const retainedSelection = current.selectedFrameId
          && history.some((candidate) => candidate.identity.frameId === current.selectedFrameId)
          ? current.selectedFrameId : undefined;
        return {
          history,
          pendingEvidence: current.pendingEvidence.filter((candidate) =>
            candidate.identity.frameId !== frame.identity.frameId
            || candidate.identity.generation !== frame.identity.generation),
          latest: completedFrame,
          selectedFrameId: followedLatest ? frame.identity.frameId : retainedSelection ?? frame.identity.frameId,
          referenceFrameId,
        };
      });
      return true;
    },
    ingestEvidence: (identity, addition) => {
      const state = get();
      if (!state.enabled
        || identity.generation !== state.generation
        || !performanceActivityAdditionMatchesIdentity(identity, addition)) return "rejected";
      const frame = state.history.find((candidate) =>
        candidate.identity.frameId === identity.frameId
        && candidate.identity.generation === identity.generation);
      if (!frame) {
        set((current) => {
          if (!current.enabled || current.generation !== identity.generation) return current;
          const existing = current.pendingEvidence.find((candidate) =>
            candidate.identity.frameId === identity.frameId
            && candidate.identity.generation === identity.generation);
          const next = existing
            ? current.pendingEvidence.map((candidate) => candidate === existing
              ? { ...candidate, additions: [...candidate.additions, addition] }
              : candidate)
            : [...current.pendingEvidence, { identity: { ...identity }, additions: [addition] }];
          // Readbacks can beat publication, but cannot grow an unbounded queue
          // if a producer never publishes its base frame.
          return { pendingEvidence: next.slice(-Math.max(2, current.historyLimit * 2)) };
        });
        return "buffered";
      }
      let merged: PerformanceActivityFrame;
      try {
        merged = mergePerformanceActivityFrame(frame, addition);
      } catch {
        return "rejected";
      }
      set((current) => {
        if (!current.enabled || current.generation !== identity.generation) return current;
        return {
          history: current.history.map((candidate) =>
            candidate.identity.frameId === identity.frameId
              && candidate.identity.generation === identity.generation ? merged : candidate),
          latest: current.latest?.identity.frameId === identity.frameId
            && current.latest.identity.generation === identity.generation ? merged : current.latest,
        };
      });
      return "merged";
    },
    mergeEvidence: (frameId, addition) => {
      const state = get();
      const frame = state.history.find((candidate) => candidate.identity.frameId === frameId);
      if (!frame) return false;
      return get().ingestEvidence(frame.identity, addition) === "merged";
    },
    selectFrame: (frameId) => set((state) => frameId === undefined
      ? { selectedFrameId: undefined }
      : state.history.some((frame) => frame.identity.frameId === frameId)
        ? { selectedFrameId: frameId }
        : state),
    pinReference: (frameId) => set((state) => frameId === undefined
      ? { referenceFrameId: undefined }
      : state.history.some((frame) => frame.identity.frameId === frameId)
        ? { referenceFrameId: frameId }
        : state),
    clear: () => set({
      history: [], pendingEvidence: [], latest: undefined, selectedFrameId: undefined, referenceFrameId: undefined,
    }),
  }));
}

function appendBoundedFrame(
  history: readonly PerformanceActivityFrame[],
  frame: PerformanceActivityFrame,
  limit: number,
  pinnedFrameId?: string,
): PerformanceActivityFrame[] {
  if (limit === 1) return [frame];
  const next = [...history, frame];
  while (next.length > limit) {
    const removable = next.findIndex((candidate) => candidate.identity.frameId !== pinnedFrameId);
    // The new frame and one pinned reference fit for comparison-sized rings.
    // This fallback is defensive against duplicate IDs.
    next.splice(removable >= 0 ? removable : 0, 1);
  }
  return next;
}

export const usePerformanceActivityStore = createPerformanceActivityStore(
  DEFAULT_PERFORMANCE_ACTIVITY_HISTORY_LIMIT,
  true,
);
