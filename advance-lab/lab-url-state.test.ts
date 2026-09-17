import assert from "node:assert/strict";
import test from "node:test";

// `createPaneSession` reaches the method registry while building its stores.
import "../lib/methods";

import { createPaneSession } from "../lib/core/session/session";
import { DEFAULT_LAB_SCENE_ID } from "./lab-scenes";
import { createLabStore } from "./lab-store";
import {
  LAB_PATH, isLabQueryKey, parseLabQuery, serializeLabQuery, startLabQueryStateSync,
} from "./lab-url-state";

/**
 * The 2-D lab's address bar: what a link says, and what it means.
 *
 * The invariant a mirroring loop actually rests on is idempotence — hydrate a
 * link, write the address back, and get the same link — because the write runs
 * a microtask after every store change, including the ones hydration itself
 * caused. A page that canonicalised a link into a *different* link would rewrite
 * its own address forever, and a reader's back button would walk through the
 * difference.
 */

/** A window with an address bar and nothing else. */
function fakeWindow(url: string) {
  const [pathname, search = ""] = url.split("?");
  const listeners = new Map<string, Set<() => void>>();
  const location = { pathname: pathname!, search: search ? `?${search}` : "", hash: "" };
  const entries: string[] = [];
  return {
    location,
    entries,
    history: {
      state: null as unknown,
      replaceState(_state: unknown, _title: string, next: string) {
        entries.push(next);
        const [path, query = ""] = next.split("?");
        location.pathname = path!;
        location.search = query ? `?${query}` : "";
      },
    },
    addEventListener(type: string, handler: () => void) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type)!.add(handler);
    },
    removeEventListener(type: string, handler: () => void) {
      listeners.get(type)?.delete(handler);
    },
  };
}

async function onPage(url: string, body: (win: ReturnType<typeof fakeWindow>) =>
  Promise<void> | void) {
  const win = fakeWindow(url);
  const globals = globalThis as unknown as { window?: unknown };
  const had = "window" in globals;
  const previous = globals.window;
  globals.window = win;
  try { await body(win); } finally {
    if (had) globals.window = previous; else delete globals.window;
  }
}

const tick = () => new Promise<void>((resolve) => queueMicrotask(resolve));

/** One link naming every key §9.3 gives this page, in the writer's own order. */
const MAXIMAL = "scene=high-resolution-dam-break&freezeTopology=1"
  + "&transport=baseline&solve=48&surface=plic"
  + "&gridMode=pressure-solve&overlays=fraction_normal"
  + "&view.zoom=4&view.x=0.25&view.y=0.75"
  + "&regions=25_25_50_50_8";

test("the page claims its own keys and nobody else's", () => {
  for (const key of ["scene", "transport", "solve", "surface", "gridMode",
    "overlays", "view.zoom", "view.x", "view.y", "regions", "freezeTopology"]) {
    assert.equal(isLabQueryKey(key), true, key);
  }
  // The studio's families are not this page's: it has no patch layer and no
  // camera prefix, so a `view.something` belonging to somebody else is safe.
  for (const key of ["utm_source", "camera.azimuth", "param.uniform.x",
    "scene.container.width_m", "quality", "method", "view.tilt", "panel"]) {
    assert.equal(isLabQueryKey(key), false, key);
  }
});

test("a link naming everything hydrates and re-writes to itself", async () => {
  await onPage(`${LAB_PATH}?${MAXIMAL}`, async (win) => {
    const session = createPaneSession("a");
    const lab = createLabStore();
    const stop = startLabQueryStateSync(session, lab);
    const hydrated = win.location.search.slice(1);
    // Every value survived the canonical write, including the two that are
    // still waiting on a lattice the controller has not published yet.
    assert.deepEqual([...new URLSearchParams(hydrated).entries()].sort(),
      [...new URLSearchParams(MAXIMAL).entries()].sort());

    const state = lab.getState();
    assert.equal(state.sceneId, "high-resolution-dam-break");
    assert.equal(state.transport, "baseline");
    assert.equal(state.budget, 48);
    assert.equal(state.surface, "plic");
    assert.equal(state.lens, "pressure-solve");
    assert.deepEqual([...state.overlays].sort(), ["fraction", "normal"]);
    assert.equal(state.linkedRegions, "25_25_50_50_8");
    assert.deepEqual(state.linkedView, { zoom: 4, x: 0.25, y: 0.75 });
    assert.equal(session.runtime.getState().topologyFrozen, true);

    // A second write changes nothing, so the loop settles rather than chasing
    // its own output.
    const before = win.entries.length;
    lab.setState({ ...lab.getState() });
    await tick();
    assert.equal(win.entries.length, before);
    stop();
  });
});

test("an empty link is the page's defaults, written as an empty address", async () => {
  await onPage(LAB_PATH, async (win) => {
    const session = createPaneSession("a");
    const lab = createLabStore();
    const stop = startLabQueryStateSync(session, lab);
    // Nothing is written, because nothing differs from the defaults: an
    // ordinary visit does not have to say which arm it is not running.
    assert.equal(win.location.search, "");
    assert.equal(lab.getState().sceneId, DEFAULT_LAB_SCENE_ID);
    stop();
  });
});

test("a scene nobody ships falls back rather than loading nothing", () => {
  assert.equal(parseLabQuery("?scene=not-a-scene").sceneId, DEFAULT_LAB_SCENE_ID);
  assert.equal(parseLabQuery("?scene=").sceneId, DEFAULT_LAB_SCENE_ID);
  assert.equal(parseLabQuery("?scene=high-resolution-dam-break").sceneId,
    "high-resolution-dam-break");
});

test("a value the page cannot honour reads as the default", () => {
  // Each of these is a different owner's clamp, read through the one codec the
  // page mounts: an arm nobody ships, a budget off the declared range, a lens
  // that is a scrub position rather than a stage, an overlay that does not
  // exist, and a camera outside the slice.
  const query = parseLabQuery("?transport=not-an-arm&solve=99999&gridMode=represent"
    + "&overlays=fraction_nonsense&view.zoom=1000&view.x=7");
  assert.equal(query.transport, "level-set-volume");
  assert.equal(query.lens, "conservative-transport");
  assert.deepEqual([...query.overlays], ["fraction"]);
  assert.deepEqual(query.view, { zoom: 1, x: 0.5, y: 0.5 });
});

test("a foreign key is carried through untouched", () => {
  const out = new URLSearchParams(serializeLabQuery("?utm_source=mail&ref=x",
    createLabStore().getState(), { topologyFrozen: false }));
  assert.equal(out.get("utm_source"), "mail");
  assert.equal(out.get("ref"), "x");
});

test("the lab writes nothing while the studio owns the address", async () => {
  await onPage("/scene?scene=water-box-dam-break&camera.azimuth=1", async (win) => {
    const session = createPaneSession("a");
    const lab = createLabStore();
    const stop = startLabQueryStateSync(session, lab);
    lab.getState().setTransport("baseline");
    await tick();
    // Both hosts mount `scene`, so a lab that ignored the path gate would have
    // rewritten the studio's link with its own default.
    assert.deepEqual(win.entries, []);
    assert.equal(win.location.search, "?scene=water-box-dam-break&camera.azimuth=1");
    stop();
  });
});

test("an edited box writes the key, and an emptied list writes it empty", async () => {
  await onPage(LAB_PATH, async (win) => {
    const session = createPaneSession("a");
    const lab = createLabStore();
    const stop = startLabQueryStateSync(session, lab);
    const store = lab.getState();
    // A loaded run: a lattice, and the scene's own authored box as the baseline.
    store.setLattice(128, 64);
    store.setRegionBaseline("0_0_50_50_8");
    store.setRegions([{
      id: "advance-region-1", minimumFine: [0, 0], maximumFine: [64, 32],
      minimumCellWidth: 8,
    }]);
    await tick();
    // Unedited: the authored box stays out of the address.
    assert.equal(new URLSearchParams(win.location.search).has("regions"), false);

    lab.getState().setRegions([]);
    await tick();
    // Deleted: the key is present and empty, or a reload would bring it back.
    assert.equal(new URLSearchParams(win.location.search).get("regions"), "");
    stop();
  });
});
