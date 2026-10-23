import assert from "node:assert/strict";
import { createServer, get } from "node:http";
import test from "node:test";
import { queryLocationUrl, readLocationQuery } from "../lib/core/location-query";

const largeBodies = JSON.stringify(Array.from({ length: 120 }, (_, i) => ({
  id: `body-${i}`, poseTrack: { keyframes: [{ time_s: 1.2, position_m: { x: .48, y: 2.52, z: -1.6 } }] },
})));

test("large scene links survive a real HTTP round trip without raising header limits", async () => {
  const query = new URLSearchParams({ scene: "nbflip-stair-cube", gpu: "off", "scene.rigidBodies": largeBodies });
  const server = createServer({ maxHeaderSize: 16384 }, (_request, response) => response.end("ok"));
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  try {
    const address = server.address();
    assert.ok(address && typeof address !== "string");
    const origin = `http://127.0.0.1:${address.port}`;
    const status = (url: URL) => new Promise<number | undefined>((resolve, reject) => {
      get(url, response => { response.resume(); response.on("end", () => resolve(response.statusCode)); }).on("error", reject);
    });
    assert.equal(await status(new URL(`/scene?${query}`, origin)), 431);
    const link = new URL(queryLocationUrl("/scene", query.toString(), "#water"), origin);
    assert.ok(link.search.length < 4096);
    assert.equal(link.searchParams.get("scene"), "nbflip-stair-cube");
    assert.equal(link.searchParams.get("gpu"), "off");
    assert.equal(await status(link), 200);
    const restored = new URLSearchParams(readLocationQuery(link));
    assert.equal(restored.get("scene.rigidBodies"), largeBodies);
    assert.equal(queryLocationUrl("/scene", restored.toString(), link.hash), link.pathname + link.search + link.hash);
    restored.delete("scene.rigidBodies");
    assert.equal(queryLocationUrl("/scene", restored.toString(), link.hash), "/scene?scene=nbflip-stair-cube&gpu=off#water");
  } finally {
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  }
});

test("fragment state preserves duplicate values and explicit query edits win", () => {
  const query = new URLSearchParams({ scene: "test" });
  query.append("b.scene.rigidBodies", largeBodies);
  query.append("b.scene.rigidBodies", "second");
  const link = new URL(queryLocationUrl("/scene", query.toString()), "http://localhost");
  assert.deepEqual(new URLSearchParams(readLocationQuery(link)).getAll("b.scene.rigidBodies"), [largeBodies, "second"]);
  link.searchParams.set("b.scene.rigidBodies", "[]");
  assert.deepEqual(new URLSearchParams(readLocationQuery(link)).getAll("b.scene.rigidBodies"), ["[]"]);
});

test("short URLs and unrelated or malformed fragments are preserved", () => {
  for (const hash of ["", "#water", "#fluid-state=%", "#fluid-state=null"]) {
    assert.equal(readLocationQuery({ search: "?scene=test", hash }), "?scene=test");
    assert.equal(queryLocationUrl("/scene", "scene=test", hash), `/scene?scene=test${hash}`);
  }
});
