import { getMethod, interactiveMethodId } from "../method-registry";
import { isOctreeTechniqueOverlayMode } from "../octree-technique-debug";
import { isSparseCM12DirtyOverlayMode } from "../sparse-cm12-dirty-visualizations";
import { isStageLensOverlayMode, stageLensOverlayMode } from "../stage-lens";
import { isPressureJournalOverlayMode } from "../webgpu-pressure-journal-overlay";
import type { PaneSession } from "../session/session";
import { useShellStore } from "../stores/shell-store";
import type { TracePinRequest } from "../stores/ui-store";
import {
  createSceneQueryLayerCache,
  parseMethodQueryState,
  parseQueryState,
  parseUIQueryState,
  serializeQueryState,
} from "../url-state";
import {
  COMPARE_ABSENT,
  COMPARE_LINK_GROUPS,
  compareGroupForKey,
  compareKeyIsDivergeable,
  type CompareGroup,
  type CompareLinkGroup,
  type CompareState,
} from "./compare-query";

/**
 * The mirror that makes "B is A plus a diff" true at every instant.
 *
 * There is no second document anywhere in this file. Both panes are read
 * through the one serializer the address bar already uses, which means the
 * definition of "these two panes differ" is exactly the definition of "this
 * link carries an override" — one comparison, one vocabulary, and a diff that
 * can never disagree with the URL it writes.
 *
 * Three events, three rules:
 *
 *  - **A changed.** Every key the diff does not claim is pushed onto B. With an
 *    empty diff that makes editing A an edit to both panes, which is what makes
 *    the mode worth opening: set the experiment up once, then fork one key.
 *  - **B changed.** Each key B moved is either *linked* — a padlocked View, Cut,
 *    Instrument or Look key, which is pushed back onto A so both panes move and
 *    the diff stays clean — or it is not, and it is recorded as an override.
 *    A key B moved back onto A's value leaves the diff.
 *  - **The diff or a padlock changed.** Re-linking a group drops that group's
 *    overrides (linked means *both panes read A*), then A is mirrored down.
 */

/** The scene layers that ride their own key; see `url-state`'s `REGIONS_QUERY_KEY`. */
const SCENE_LAYER_KEYS: ReadonlySet<string> = new Set(["regions", "canopy", "stones", "rim", "seeds"]);

/** The document one pane holds, spelled through the store this module already reads. */
type PaneScene = ReturnType<PaneSession["scene"]["getState"]>["scene"];

/** A key whose value only a rebuilt document can carry. */
function isDocumentKey(key: string): boolean {
  return key === "scene" || key.startsWith("scene.") || SCENE_LAYER_KEYS.has(key);
}

/** A key held by the method store. */
function isMethodKey(key: string): boolean {
  return key === "method" || key === "quality" || key.startsWith("param.");
}

/** Everything else the serializer writes lives in the UI store. */
function isUIKey(key: string): boolean {
  return !isDocumentKey(key) && !isMethodKey(key);
}

/** The canonical query for one pane, with no address and no shell layer in it. */
function paneSearch(session: PaneSession, layer: ReturnType<typeof createSceneQueryLayerCache>): URLSearchParams {
  const scene = session.scene.getState();
  return new URLSearchParams(serializeQueryState(
    "",
    { presetId: scene.presetId, scene: scene.scene },
    session.method.getState(),
    session.ui.getState(),
    { view: "studio" },
    layer({ presetId: scene.presetId, scene: scene.scene }),
  ));
}

function valueOf(query: URLSearchParams, key: string): string {
  return query.get(key) ?? COMPARE_ABSENT;
}

function withDiff(base: URLSearchParams, diff: Readonly<Record<string, string>>): URLSearchParams {
  const next = new URLSearchParams(base);
  for (const [key, value] of Object.entries(diff)) {
    if (value === COMPARE_ABSENT) next.delete(key);
    else next.set(key, value);
  }
  return next;
}

/** Managed keys where two canonical queries disagree. */
function changedKeys(desired: URLSearchParams, current: URLSearchParams): readonly string[] {
  const keys = new Set([...desired.keys(), ...current.keys()]);
  return [...keys].filter((key) => valueOf(desired, key) !== valueOf(current, key));
}

/**
 * Write only the layers a change actually touches.
 *
 * The document layer is the expensive one — `parseQueryState` runs the preset
 * factory — so a camera orbit in one pane must never reach it. When the diff
 * has not forked the document, `exactScene` skips the parse entirely and hands
 * over the very object the other pane holds, which also carries across the
 * seam the authoring a URL cannot express: a sculpted terrain grid, a scenery
 * graph, a painted volume too large to serialize.
 */
function applyLayers(
  session: PaneSession,
  desired: URLSearchParams,
  changed: readonly string[],
  exactScene?: { readonly presetId: string; readonly scene: PaneScene },
  adopt: CompareAdoptions = {},
): void {
  const search = `?${desired.toString()}`;
  /** The document this pane held before the write, once the write moved it. */
  let replaced: PaneScene | undefined;
  if (changed.some(isDocumentKey)) {
    const previous = session.scene.getState().scene;
    if (exactScene) session.scene.getState().setScene(exactScene.scene, exactScene.presetId);
    else {
      const parsed = parseQueryState(search);
      session.scene.getState().setScene(parsed.scene, parsed.presetId);
    }
    if (session.scene.getState().scene !== previous) replaced = previous;
  }
  let switchedMethod = false;
  if (changed.some(isMethodKey)) {
    const method = parseMethodQueryState(search);
    const methodId = interactiveMethodId(method.methodId);
    // A solver is not a value: adopting one has to rebuild this pane's GPU work
    // and put its clock back to a defined t=0, which is what the controller's
    // own switch does and what a bare store write does not. Only when the id
    // actually moves — quality and a parameter override are settings *of* the
    // running solver and the viewport picks those up on its own.
    switchedMethod = methodId !== session.method.getState().methodId;
    session.method.setState({ methodId, quality: method.quality, overrides: method.overrides });
    if (switchedMethod) adopt.onMethodSwitched?.(session, methodId);
  }
  // After the document, so a camera that falls back to a preset framing falls
  // back to the preset this pane is now actually holding.
  if (changed.some(isUIKey)) {
    session.ui.setState(parseUIQueryState(search, session.scene.getState().presetId));
  }
  // A document is not a value either, and for the same reason: the pane that
  // receives one has a *running* rigid roster and a *running* solver built from
  // the document it used to hold, and a store write moves neither. Last, so the
  // adoption sees every layer this pass wrote; and never behind a solver
  // switch, whose own rebuild starts this pane from the new document anyway.
  if (replaced && !switchedMethod) adopt.onSceneAdopted?.(session, replaced);
}

/** The compare record, as a store the mirror can read, write and watch. */
export interface CompareStateStore {
  getState(): CompareState;
  setState(next: CompareState): void;
  subscribe(listener: () => void): () => void;
}

/** The page's own record. Single-pane mode simply never starts a mirror over it. */
export function shellCompareStore(): CompareStateStore {
  return {
    getState: () => useShellStore.getState().compare,
    setState: (next) => useShellStore.getState().setCompare(next),
    subscribe: (listener) => useShellStore.subscribe((state, previous) => {
      if (state.compare !== previous.compare) listener();
    }),
  };
}

function sameDiff(left: Readonly<Record<string, string>>, right: Readonly<Record<string, string>>): boolean {
  const leftKeys = Object.keys(left);
  if (leftKeys.length !== Object.keys(right).length) return false;
  return leftKeys.every((key) => left[key] === right[key]);
}

export interface CompareSync {
  /** Re-run the mirror now. The subscriptions do this; callers rarely need to. */
  refresh(): void;
  stop(): void;
}

/**
 * What to do when the mirror hands a pane a different solver.
 *
 * A callback rather than a controller import, so this module — and the CPU test
 * over it — stays clear of the simulation singleton and of WebGPU. The host
 * passes `simulation.setMethod`'s rebuild; a test passes nothing and reads the
 * stores.
 */
export type CompareMethodSwitch = (session: PaneSession, methodId: string) => void;

/**
 * What to do when the mirror hands a pane a different *document*.
 *
 * The same argument as the solver, one tier down. A pane's rigid roster and its
 * GPU solver are built from the document it was last started on, and writing
 * the scene store moves the drawing without moving either: pane B ends up
 * holding a sphere that its own `advanceTo` has never heard of. Only the
 * controller can reconcile that, so — again — a callback rather than an import.
 *
 * The *previous* document is the argument because the receiving pane's store
 * already holds the new one by the time this runs, and the question the
 * controller has to answer ("is this a re-seed or a roster adoption?") is a
 * comparison of the two.
 */
export type CompareSceneAdopt = (session: PaneSession, previous: PaneScene) => void;

/**
 * The two writes the mirror cannot make on its own, as one object.
 *
 * Every surface that mirrors passes the same pair; a CPU test passes neither
 * and reads the stores, or passes a spy and counts the calls.
 */
export interface CompareAdoptions {
  readonly onMethodSwitched?: CompareMethodSwitch;
  readonly onSceneAdopted?: CompareSceneAdopt;
}

/**
 * Bind pane B to pane A through the diff, for as long as compare mode is open.
 *
 * Re-entrancy is handled by a single running flag and a re-run loop rather than
 * by unsubscribing: every write below lands in a store whose own subscription
 * would otherwise call straight back into here, and a mirror that suppressed
 * those events would miss the second half of a two-store edit. The loop instead
 * lets a pass observe what the previous pass wrote and settles when nothing
 * moves — which for a round-tripping serializer is always the second pass.
 */
export function startCompareSync(
  a: PaneSession,
  b: PaneSession,
  store: CompareStateStore,
  adopt: CompareAdoptions = {},
): CompareSync {
  const layerA = createSceneQueryLayerCache();
  const layerB = createSceneQueryLayerCache();
  let running = false;
  let pending = false;
  let pendingFromB = false;
  let stopped = false;
  /**
   * Pane B as the last pass left it.
   *
   * Without it, any key whose parse does not round-trip its own serialization —
   * a camera elevation outside the parser's clamp, say — would read as "B
   * diverged" on the pass immediately after the mirror wrote it, and the two
   * panes would trade the key forever. The rule this makes precise is the one
   * the mode is described by: a diff is recorded when *B moved*, not when the
   * two happen to disagree.
   */
  let lastB = new URLSearchParams();

  const mirrorAtoB = () => {
    const state = store.getState();
    const searchA = paneSearch(a, layerA);
    const desired = withDiff(searchA, state.diff);
    const changed = changedKeys(desired, paneSearch(b, layerB));
    if (changed.length === 0) return;
    const sceneA = a.scene.getState();
    const forked = Object.keys(state.diff).some(isDocumentKey);
    applyLayers(b, desired, changed,
      forked ? undefined : { presetId: sceneA.presetId, scene: sceneA.scene },
      adopt);
  };

  const reconcileFromB = () => {
    const state = store.getState();
    const searchA = paneSearch(a, layerA);
    const searchB = paneSearch(b, layerB);
    const nextDiff: Record<string, string> = {};
    const promoted = new Map<string, string>();
    const keys = new Set([...searchA.keys(), ...searchB.keys(), ...Object.keys(state.diff)]);
    for (const key of keys) {
      if (!compareKeyIsDivergeable(key)) continue;
      const inA = valueOf(searchA, key);
      const inB = valueOf(searchB, key);
      // Back on A's value: the override is gone, however it got there.
      if (inA === inB) continue;
      const moved = inB !== valueOf(lastB, key);
      // Neither B's doing nor an override already standing: this is a value the
      // mirror could not express, not an edit. Leaving it alone is what stops
      // the two panes trading it.
      if (!moved && !Object.hasOwn(state.diff, key)) continue;
      const group = compareGroupForKey(key);
      if (group !== "config" && state.links[group]) { promoted.set(key, inB); continue; }
      nextDiff[key] = inB;
    }
    if (promoted.size > 0) {
      // A linked group moved in B, so it moves in A: both panes, one value, and
      // nothing recorded. This is the whole meaning of the padlock.
      const desiredA = new URLSearchParams(searchA);
      for (const [key, value] of promoted) {
        if (value === COMPARE_ABSENT) desiredA.delete(key);
        else desiredA.set(key, value);
      }
      applyLayers(a, desiredA, [...promoted.keys()], undefined, adopt);
    }
    if (!sameDiff(state.diff, nextDiff)) store.setState({ ...state, diff: nextDiff });
  };

  const pass = (fromB: boolean) => {
    pending = true;
    pendingFromB ||= fromB;
    if (running) return;
    running = true;
    try {
      let guard = 0;
      while (pending && guard++ < 8) {
        pending = false;
        const reconcile = pendingFromB;
        pendingFromB = false;
        if (reconcile) reconcileFromB();
        mirrorAtoB();
      }
      lastB = paneSearch(b, layerB);
    } finally {
      running = false;
    }
  };

  const onA = () => { if (!stopped) pass(false); };
  const onB = () => { if (!stopped) pass(true); };

  // ---- cross-pane pins ---------------------------------------------------
  //
  // A pinned trace is about *a place*, not about a pane, so pinning in one pane
  // aims the same probe in the other. The aim is a viewport fraction, which
  // under a linked View is the same ray and therefore the same world cell —
  // the probes resolve the cell themselves, each against its own solver, which
  // is exactly the reading a compare wants. TODO(world-cell): with View
  // unlinked the two rays differ; pinning by resolved world cell needs the
  // viewport's pick, which lands in `components/WebGPUViewport.tsx:1596`
  // (`openRadialMenuAt`) and in the probe's own gather, both owned elsewhere.
  let mirroringProbe = false;
  const probeMirror = (from: PaneSession, to: PaneSession) => {
    let lastPixelRequest: TracePinRequest | null = from.ui.getState().pixelTracePinRequest;
    let lastCellRequest: TracePinRequest | null = from.ui.getState().fluidCellTracePinRequest;
    let lastPixelEnabled = from.ui.getState().pixelTraceEnabled;
    let lastCellEnabled = from.ui.getState().fluidCellTraceEnabled;
    return from.ui.subscribe(() => {
      if (stopped || mirroringProbe) return;
      const source = from.ui.getState();
      const target = to.ui.getState();
      const pixelRequest = source.pixelTracePinRequest;
      const cellRequest = source.fluidCellTracePinRequest;
      const pixelEnabled = source.pixelTraceEnabled;
      const cellEnabled = source.fluidCellTraceEnabled;
      const changedPixelRequest = pixelRequest !== lastPixelRequest && pixelRequest !== null;
      const changedCellRequest = cellRequest !== lastCellRequest && cellRequest !== null;
      const changedPixelEnabled = pixelEnabled !== lastPixelEnabled;
      const changedCellEnabled = cellEnabled !== lastCellEnabled;
      lastPixelRequest = pixelRequest;
      lastCellRequest = cellRequest;
      lastPixelEnabled = pixelEnabled;
      lastCellEnabled = cellEnabled;
      if (!changedPixelRequest && !changedCellRequest && !changedPixelEnabled && !changedCellEnabled) return;
      mirroringProbe = true;
      try {
        if (changedPixelEnabled && !changedPixelRequest) target.setPixelTraceEnabled(pixelEnabled);
        if (changedCellEnabled && !changedCellRequest) target.setFluidCellTraceEnabled(cellEnabled);
        // Same three calls, same order, as the ring's own probe effect: enabling
        // clears the held pin, so the ask has to follow it.
        if (changedPixelRequest) {
          target.setPixelTraceEnabled(true);
          target.setPixelTracePinned(false);
          target.requestPixelTracePin({ aim: pixelRequest?.aim });
        }
        if (changedCellRequest) {
          target.setFluidCellTraceEnabled(true);
          target.setFluidCellTracePinned(false);
          target.requestFluidCellTracePin({ aim: cellRequest?.aim });
        }
      } finally {
        mirroringProbe = false;
      }
    });
  };

  // B starts as A: the first frame of the second pane is the sanity check that
  // the mode works, so the mirror runs before anything is subscribed.
  pass(false);

  const unsubscribe = [
    a.scene.subscribe(onA), a.method.subscribe(onA), a.ui.subscribe(onA),
    b.scene.subscribe(onB), b.method.subscribe(onB), b.ui.subscribe(onB),
    store.subscribe(() => { if (!stopped) pass(false); }),
    probeMirror(a, b),
    probeMirror(b, a),
  ];

  return {
    refresh: () => pass(false),
    stop: () => {
      stopped = true;
      for (const off of unsubscribe) off();
    },
  };
}

// ---- diff operations -------------------------------------------------------

/** Drop one override: B falls back to A for that key. */
export function dropCompareOverride(store: CompareStateStore, key: string): void {
  const state = store.getState();
  if (!Object.hasOwn(state.diff, key)) return;
  const diff = { ...state.diff };
  delete diff[key];
  store.setState({ ...state, diff });
}

/**
 * Move one override to A instead.
 *
 * Both panes end on B's value, and the diff loses the row — the `⇄` on the diff
 * strip, and the way an experiment that turned out well gets adopted one key at
 * a time rather than all at once.
 */
export function moveCompareOverrideToA(
  store: CompareStateStore,
  a: PaneSession,
  key: string,
  adopt: CompareAdoptions = {},
): void {
  const state = store.getState();
  const value = state.diff[key];
  if (value === undefined) return;
  const layer = createSceneQueryLayerCache();
  const desired = withDiff(paneSearch(a, layer), { [key]: value });
  applyLayers(a, desired, [key], undefined, adopt);
  dropCompareOverride(store, key);
}

/** A ← A ⊕ diff, and the diff empties. "Keep this one", and half of a swap. */
export function promoteCompareDiff(
  store: CompareStateStore,
  a: PaneSession,
  adopt: CompareAdoptions = {},
): void {
  const state = store.getState();
  if (Object.keys(state.diff).length === 0) return;
  const layer = createSceneQueryLayerCache();
  const desired = withDiff(paneSearch(a, layer), state.diff);
  applyLayers(a, desired, Object.keys(state.diff), undefined, adopt);
  store.setState({ ...state, diff: {} });
}

/**
 * Exchange the panes: B's values become A's, and A's old values become the diff.
 *
 * Ordering matters. A's values have to be read *before* the promotion, or the
 * inverted diff would be a copy of the diff it is inverting.
 */
export function swapComparePanes(
  store: CompareStateStore,
  a: PaneSession,
  adopt: CompareAdoptions = {},
): void {
  const state = store.getState();
  const keys = Object.keys(state.diff);
  if (keys.length === 0) return;
  const layer = createSceneQueryLayerCache();
  const searchA = paneSearch(a, layer);
  const inverted: Record<string, string> = {};
  for (const key of keys) inverted[key] = valueOf(searchA, key);
  const desired = withDiff(searchA, state.diff);
  applyLayers(a, desired, keys, undefined, adopt);
  store.setState({ ...state, diff: inverted });
}

// ---- readout ---------------------------------------------------------------

export interface CompareDiffRow {
  readonly key: string;
  readonly group: CompareGroup;
  /** The value pane A holds, or `COMPARE_ABSENT` when it carries no such key. */
  readonly valueA: string;
  readonly valueB: string;
}

/** Every row the diff strip draws, in group order then key order. */
export function compareDiffRows(
  state: CompareState,
  a: PaneSession,
  b: PaneSession,
): readonly CompareDiffRow[] {
  const layerA = createSceneQueryLayerCache();
  const layerB = createSceneQueryLayerCache();
  const searchA = paneSearch(a, layerA);
  const searchB = paneSearch(b, layerB);
  const order: readonly CompareGroup[] = [...COMPARE_LINK_GROUPS, "config"];
  return Object.keys(state.diff)
    .map((key) => ({
      key,
      group: compareGroupForKey(key),
      valueA: valueOf(searchA, key),
      // The diff is authoritative for B even a microtask before the mirror has
      // written it, so a row never blinks through "identical" while settling.
      valueB: state.diff[key] ?? valueOf(searchB, key),
    }))
    .sort((left, right) => order.indexOf(left.group) - order.indexOf(right.group)
      || left.key.localeCompare(right.key));
}

/**
 * Whether a pane's method can actually draw a field view.
 *
 * A method-only overlay stays legal in a *linked* Cut — the two panes hold the
 * same key and one of them simply draws nothing — so the strip has to be able
 * to say "n/a in B" instead of the key being silently dropped, which would make
 * the two panes disagree about a value neither of them changed. The generic
 * dense views are drawable by everything and are the common case, so they fall
 * straight through.
 */
export function compareOverlayModeDrawable(mode: string, methodId: string): boolean {
  const method = getMethod(methodId);
  if (isStageLensOverlayMode(mode)) {
    return (method.stageLenses ?? []).some((lens) => stageLensOverlayMode(lens.stage) === mode);
  }
  if (isSparseCM12DirtyOverlayMode(mode) || isOctreeTechniqueOverlayMode(mode)
    || isPressureJournalOverlayMode(mode)) {
    return (method.supportedFieldModes ?? []).includes(mode);
  }
  return true;
}

/** Groups a reader has unlinked, for the strip's own warning line. */
export function unlinkedGroupLabels(links: CompareState["links"]): readonly CompareLinkGroup[] {
  return COMPARE_LINK_GROUPS.filter((group) => !links[group]);
}
