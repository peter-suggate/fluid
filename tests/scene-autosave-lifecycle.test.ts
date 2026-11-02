import assert from "node:assert/strict";
import test from "node:test";
import { createEmptyScene } from "../lib/core/empty-scene";
import { createSceneAutosave } from "../lib/core/scene-autosave";

/**
 * The autosave writer's lifecycle against a store it does not own. The
 * regressions: a version change from another tab closed the connection and the
 * next write into it stopped autosaving for the session, and a cancelled
 * request could never be asked again because the writer still remembered it.
 */
function fakeIndexedDB() {
  const data = new Map<string, unknown>();
  const connections: { closed: boolean; onversionchange: (() => void) | null }[] = [];
  const transaction = (connection: { closed: boolean }) => {
    if (connection.closed) throw new DOMException("The database connection is closing.", "InvalidStateError");
    const tx = {
      oncomplete: null as (() => void) | null, onabort: null as (() => void) | null, error: null,
      abort: () => { tx.aborted = true; queueMicrotask(() => tx.onabort?.()); }, aborted: false,
      objectStore: () => ({
        put: (value: unknown, key: string) => { data.set(key, value); },
        delete: (key: string) => { data.delete(key); },
        getAllKeys: () => {
          const request = { result: [...data.keys()], onsuccess: null as (() => void) | null };
          queueMicrotask(() => request.onsuccess?.());
          return request;
        },
      }),
    };
    setTimeout(() => { if (!tx.aborted) tx.oncomplete?.(); });
    return tx;
  };
  const factory = {
    open: () => {
      const connection = { closed: false, onversionchange: null as (() => void) | null,
        transaction: () => transaction(connection), close: () => { connection.closed = true; },
        createObjectStore: () => undefined };
      connections.push(connection);
      const request = { result: connection, onupgradeneeded: null as (() => void) | null,
        onsuccess: null as (() => void) | null, onerror: null };
      setTimeout(() => { request.onupgradeneeded?.(); request.onsuccess?.(); });
      return request;
    },
  } as unknown as IDBFactory;
  return { data, connections, factory };
}

async function until(predicate: () => boolean) {
  for (let i = 0; i < 200 && !predicate(); i++) await new Promise(resolve => setTimeout(resolve, 1));
  assert.ok(predicate());
}

const headPreset = (data: Map<string, unknown>) => (data.get("head") as { presetId?: string } | undefined)?.presetId;

test("a version change from another tab closes this connection and the next save reopens it", async () => {
  const store = fakeIndexedDB();
  const failures: Error[] = [];
  const autosave = createSceneAutosave({ database: store.factory, delay_ms: 0, failed: error => failures.push(error) });
  const scene = createEmptyScene();
  autosave.request({ scene, presetId: "first" });
  await until(() => headPreset(store.data) === "first");
  store.connections[0]!.onversionchange?.();
  autosave.request({ scene, presetId: "second" });
  await until(() => headPreset(store.data) === "second" || failures.length > 0);
  assert.deepEqual(failures, []);
  assert.equal(store.connections.length, 2);
});

test("a cancelled request can be asked for again", async () => {
  const store = fakeIndexedDB();
  const autosave = createSceneAutosave({ database: store.factory, delay_ms: 5 });
  const working = { scene: createEmptyScene(), presetId: "asked" };
  autosave.request(working);
  autosave.cancel();
  autosave.request(working);
  await until(() => headPreset(store.data) === "asked");
});
