import assert from "node:assert/strict";
import test from "node:test";

import {
  replaceLocationSearch, startHostQueryStateSync, type HostQueryStateSync,
} from "../query-state-sync";

/**
 * The mirror loop, on a window that is three objects and a listener list.
 *
 * Every claim here is about the *loop* rather than about either host: what it
 * writes, when, and what it stops writing. That is the whole point of the
 * extraction — the studio and the 2-D lab run this same code, so a rule broken
 * here is broken on both pages, and neither page's stores are needed to see it.
 */

interface FakeWindow {
  readonly location: { pathname: string; search: string; hash: string };
  readonly history: { state: unknown; readonly entries: string[];
    replaceState(state: unknown, title: string, url: string): void };
  readonly listeners: Map<string, Set<() => void>>;
  addEventListener(type: string, handler: () => void): void;
  removeEventListener(type: string, handler: () => void): void;
  fire(type: string): void;
}

/** A window with an address bar and nothing else. */
function fakeWindow(url = "/scene?a=1"): FakeWindow {
  const [pathname, search = ""] = url.split("?");
  const listeners = new Map<string, Set<() => void>>();
  const location = { pathname: pathname!, search: search ? `?${search}` : "", hash: "" };
  const win: FakeWindow = {
    location,
    history: {
      state: null,
      entries: [],
      replaceState(state, _title, next) {
        win.history.state = state;
        win.history.entries.push(next);
        const [address, fragment = ""] = next.split("#");
        const [path, query = ""] = address!.split("?");
        location.pathname = path!;
        location.search = query ? `?${query}` : "";
        location.hash = fragment ? `#${fragment}` : "";
      },
    },
    listeners,
    addEventListener: (type, handler) => {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type)!.add(handler);
    },
    removeEventListener: (type, handler) => { listeners.get(type)?.delete(handler); },
    fire: (type) => { for (const handler of [...(listeners.get(type) ?? [])]) handler(); },
  };
  return win;
}

/** Install a fake window for the duration of `body`. */
async function withWindow(win: FakeWindow, body: () => Promise<void> | void) {
  const globals = globalThis as unknown as { window?: unknown };
  const had = "window" in globals;
  const previous = globals.window;
  globals.window = win;
  try { await body(); } finally {
    if (had) globals.window = previous; else delete globals.window;
  }
}

/** A host whose state is one string, and whose sources are hand-fired. */
function host(path = "/scene") {
  const notify: (() => void)[] = [];
  const stopped: number[] = [];
  const state = { value: "one" };
  const hydrated: string[] = [];
  const source = (index: number) =>
    (onChange: () => void) => { notify[index] = onChange; return () => stopped.push(index); };
  const sync: HostQueryStateSync = {
    path,
    serialize: (search) => {
      const query = new URLSearchParams(search);
      query.delete("v");
      query.set("v", state.value);
      return query.toString();
    },
    hydrate: (search) => {
      hydrated.push(search);
      state.value = new URLSearchParams(search).get("v") ?? "one";
    },
    sources: [source(0), source(1)],
  };
  return { sync, state, hydrated, stopped, change: (index: number) => notify[index]!() };
}

const tick = () => new Promise<void>((resolve) => queueMicrotask(resolve));

test("hydration happens once, before any source is subscribed", async () => {
  const win = fakeWindow("/scene?v=two&foreign=keep");
  const page = host();
  await withWindow(win, async () => {
    const stop = startHostQueryStateSync(page.sync);
    // The link's value won, and the canonical form was written immediately
    // rather than on a microtask: anything reading the address back after a
    // hydrate must see the canonical answer, not the typed one.
    assert.deepEqual(page.hydrated, ["?v=two&foreign=keep"]);
    assert.equal(page.state.value, "two");
    assert.equal(win.location.search, "?foreign=keep&v=two");
    stop();
  });
});

test("a foreign key survives every write", async () => {
  const win = fakeWindow("/scene?utm_source=mail");
  const page = host();
  await withWindow(win, async () => {
    const stop = startHostQueryStateSync(page.sync);
    page.state.value = "three";
    page.change(0);
    await tick();
    assert.equal(new URLSearchParams(win.location.search).get("utm_source"), "mail");
    stop();
  });
});

test("many changes in one tick are one write", async () => {
  const win = fakeWindow("/scene?v=one");
  const page = host();
  await withWindow(win, async () => {
    const stop = startHostQueryStateSync(page.sync);
    const before = win.history.entries.length;
    page.state.value = "a";
    page.change(0);
    page.change(1);
    page.change(0);
    page.state.value = "settled";
    await tick();
    // One entry, carrying what the stores settled on rather than the first
    // value seen — the coalescer is what makes a pointer-rate source affordable.
    assert.equal(win.history.entries.length, before + 1);
    assert.equal(new URLSearchParams(win.location.search).get("v"), "settled");
    stop();
  });
});

test("a write that would change nothing costs no history entry", async () => {
  const win = fakeWindow("/scene?v=one");
  const page = host();
  await withWindow(win, async () => {
    const stop = startHostQueryStateSync(page.sync);
    const before = win.history.entries.length;
    page.change(0);
    await tick();
    assert.equal(win.history.entries.length, before);
    stop();
  });
});

test("a host on another route stays out of the address bar", async () => {
  const win = fakeWindow("/advance-lab?v=one");
  const page = host("/scene");
  await withWindow(win, async () => {
    const stop = startHostQueryStateSync(page.sync);
    // Hydration still ran — the stores are the host's own — but nothing was
    // written, because the page on screen is somebody else's.
    assert.equal(win.history.entries.length, 0);
    page.state.value = "four";
    page.change(0);
    await tick();
    assert.equal(win.history.entries.length, 0);
    assert.equal(win.location.search, "?v=one");
    stop();
  });
});

test("a nested route is still this host's", async () => {
  const win = fakeWindow("/scene/water-box?v=one");
  const page = host("/scene");
  await withWindow(win, async () => {
    const stop = startHostQueryStateSync(page.sync);
    page.state.value = "five";
    page.change(0);
    await tick();
    assert.equal(new URLSearchParams(win.location.search).get("v"), "five");
    assert.equal(win.location.pathname, "/scene/water-box");
    stop();
  });
});

test("popstate re-hydrates, and only on this host's route", async () => {
  const win = fakeWindow("/scene?v=one");
  const page = host("/scene");
  await withWindow(win, async () => {
    const stop = startHostQueryStateSync(page.sync);
    win.location.search = "?v=back";
    win.fire("popstate");
    assert.equal(page.state.value, "back");
    assert.equal(page.hydrated.length, 2);

    win.location.pathname = "/advance-lab";
    win.location.search = "?v=elsewhere";
    win.fire("popstate");
    assert.equal(page.hydrated.length, 2);
    assert.equal(page.state.value, "back");
    stop();
  });
});

test("a retained page mirrors without reading the address", async () => {
  // `hydrateFromUrl: false` is the client-navigation case: the stores already
  // hold the state, so the address is written *from* them rather than read.
  const win = fakeWindow("/scene?v=stale");
  const page = host();
  await withWindow(win, async () => {
    const stop = startHostQueryStateSync({ ...page.sync, hydrateFromUrl: false });
    assert.deepEqual(page.hydrated, []);
    assert.equal(new URLSearchParams(win.location.search).get("v"), "one");
    stop();
  });
});

test("stop unsubscribes every source and the listener, and silences a queued write", async () => {
  const win = fakeWindow("/scene?v=one");
  const page = host();
  await withWindow(win, async () => {
    const stop = startHostQueryStateSync(page.sync);
    page.state.value = "six";
    page.change(0);
    // Teardown between the schedule and the microtask: the route that replaced
    // this one owns the address bar now.
    stop();
    await tick();
    assert.deepEqual(page.stopped.sort(), [0, 1]);
    assert.equal(win.listeners.get("popstate")?.size ?? 0, 0);
    assert.equal(new URLSearchParams(win.location.search).get("v"), "one");

    win.location.search = "?v=ignored";
    win.fire("popstate");
    assert.equal(page.state.value, "six");
  });
});

test("replaceLocationSearch keeps the path and the hash", async () => {
  const win = fakeWindow("/scene?v=one");
  await withWindow(win, () => {
    win.location.hash = "#water";
    replaceLocationSearch("v=two");
    assert.deepEqual(win.history.entries, ["/scene?v=two#water"]);
    replaceLocationSearch("");
    assert.equal(win.history.entries[1], "/scene#water");
  });
});
