import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import test from "node:test";
import { blenderAssets, selectedBlenderAsset } from "../shape-lab/assets";
import { decodeMeshVolume } from "../lib/core/imported-assets/mesh-volume";
import { MESH_ASSETS, meshAssetUrl } from "../lib/core/imported-assets/asset-library";
import { prepareShapeLabAsset, traceShapeLabTile, type ShapeLabTraceOptions } from "../shape-lab/trace";
import type { ShapeLabRequest, ShapeLabResponse } from "../shape-lab/worker";

const paverBytes = gunzipSync(readFileSync(new URL("../public/assets/pool-party/honed-travertine-paver.fvx.gz", import.meta.url)));
const volume = decodeMeshVolume(paverBytes);
const scene = prepareShapeLabAsset(volume);
const view: ShapeLabTraceOptions = {
  width: 32, height: 24, shading: "material", showGround: false,
  camera: { target_m: { x: 0, y: 0, z: 0 }, azimuth_rad: .7, elevation_rad: .65, distance_m: 2, reach_m: .5 },
};

test("Blender lab discovers every shipped mesh and distinguishes assets without a bake", () => {
  const meshes = new Set(blenderAssets.flatMap(asset => asset.meshUrl ? [asset.meshUrl] : []));
  for (const file of readdirSync(new URL("../public/assets", import.meta.url), { recursive: true })) {
    if (typeof file === "string" && file.endsWith(".glb")) {
      assert.ok(meshes.has(`/assets/${file}`), `Unregistered exported mesh: ${file}`);
    }
  }
  for (const asset of blenderAssets) {
    if (asset.meshUrl) assert.ok(existsSync(new URL(`../public${asset.meshUrl}`, import.meta.url)), asset.meshUrl);
    if (asset.volumeUrl) assert.ok(existsSync(new URL(`../public${asset.volumeUrl}`, import.meta.url)), asset.volumeUrl);
    if (asset.size) assert.ok(asset.size.every(value => value > 0));
    assert.ok(asset.collectionLabel.length > 0);
  }
  assert.equal(selectedBlenderAsset("hero-oak").id, "hero-oak");
  assert.equal(selectedBlenderAsset("bub").id, "bub");
  assert.equal(selectedBlenderAsset("bub").header, undefined);
  assert.equal(selectedBlenderAsset("removed-asset").id, "teak-lounger");
  const baked = blenderAssets.filter(asset => asset.volumeUrl);
  assert.deepEqual(new Set(Object.keys(MESH_ASSETS)), new Set(baked.map(asset => asset.id)),
    "the renderer and lab must consume the same published volume registrations");
  for (const asset of baked) assert.equal(meshAssetUrl(asset.id), asset.volumeUrl);
  assert.ok(!Object.hasOwn(MESH_ASSETS, "bub"), "mesh-only assets cannot become sampled fields");
});

test("Blender lab renders the shipped native field and authored colours independently of geometry", () => {
  const material = traceShapeLabTile(scene, view);
  const clay = traceShapeLabTile(scene, { ...view, shading: "clay" });
  const normal = traceShapeLabTile(scene, { ...view, shading: "normal" });
  assert.ok(material.hits > 100 && material.hits < material.pixels, "the real slab is framed with background around it");
  assert.equal(material.hits, clay.hits);
  assert.equal(material.hits, normal.hits);
  assert.notDeepEqual(material.rgba, clay.rgba, "material mode must use the baked source colours");
  assert.notDeepEqual(normal.rgba, clay.rgba, "normal inspection must expose source normals");

  const recoloured = prepareShapeLabAsset({ ...volume, header: { ...volume.header,
    materials: volume.header.materials.map(material => ({ ...material, baseColorLinear: [1, 0, 0] as const })),
  } });
  const red = traceShapeLabTile(recoloured, view);
  assert.equal(red.hits, material.hits, "changing the palette must not move the surface");
  assert.notDeepEqual(red.rgba, material.rgba);
});

test("Blender lab compares occupied voxels at the native bake and a coarser spacing", () => {
  const native = traceShapeLabTile(scene, view);
  for (const scale of [1, 4]) {
    const voxels = traceShapeLabTile(scene, { ...view, leaf_m: volume.header.cellSize * scale });
    assert.ok(voxels.hits > native.hits * .8 && voxels.hits < native.hits * 1.2,
      "the voxel view retains the asset silhouette");
    assert.notDeepEqual(voxels.rgba, native.rgba, "voxel faces must remain visibly distinct from the native field");
  }
});

test("empty narrow-band rays do not become surfaces when the pixel footprint is large", () => {
  // This ray crosses the volume's empty upper band. Its pixel footprint is
  // larger than the band width, which used to make a generic march report a hit.
  const empty = traceShapeLabTile(scene, { ...view, width: 1, height: 1,
    camera: { ...view.camera, target_m: { x: 0, y: .04, z: 0 }, azimuth_rad: 0, elevation_rad: 0 },
  });
  assert.equal(empty.hits, 0);
  const outside = traceShapeLabTile(scene, { ...view, width: 1, height: 1,
    camera: { ...view.camera, target_m: { x: 2, y: 0, z: 0 }, azimuth_rad: 0, elevation_rad: 0 },
  });
  assert.equal(outside.hits, 0);
});

test("an obsolete failed asset load cannot poison a newer selection of the same asset", async () => {
  const selfDescriptor = Object.getOwnPropertyDescriptor(globalThis, "self");
  const originalFetch = globalThis.fetch;
  let receive!: (event: MessageEvent<ShapeLabRequest>) => Promise<void>;
  const replies: ShapeLabResponse[] = [];
  const loads: { resolve: (response: Response) => void; reject: (error: Error) => void }[] = [];
  Object.defineProperty(globalThis, "self", { configurable: true, value: {
    addEventListener: (_type: string, callback: typeof receive) => { receive = callback; },
    postMessage: (message: ShapeLabResponse) => { replies.push(message); },
  } });
  globalThis.fetch = () => new Promise<Response>((resolve, reject) => loads.push({ resolve, reject }));
  const send = (data: ShapeLabRequest) => receive({ data } as MessageEvent<ShapeLabRequest>);
  try {
    await import("../shape-lab/worker");
    for (const sceneId of ["a", "b", "a"]) {
      await send({ type: "scene", sceneId, assetUrl: `/assets/${sceneId}.fvx.gz`, descriptors: [], colors: [] });
    }
    loads[2].resolve(new Response(new Uint8Array(paverBytes)));
    await send({ type: "tile", sceneId: "a", generation: 1, pass: 1, options: { ...view, width: 1, height: 1 } });
    assert.equal(replies.at(-1)?.type, "tile");
    loads[0].reject(new Error("obsolete request failed"));
    loads[1].reject(new Error("other obsolete request failed"));
    await new Promise(resolve => setImmediate(resolve));
    await send({ type: "tile", sceneId: "a", generation: 2, pass: 1, options: { ...view, width: 1, height: 1 } });
    assert.equal(replies.at(-1)?.type, "tile");
  } finally {
    globalThis.fetch = originalFetch;
    if (selfDescriptor) Object.defineProperty(globalThis, "self", selfDescriptor);
    else Reflect.deleteProperty(globalThis, "self");
  }
});
