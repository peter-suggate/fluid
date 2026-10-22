import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { createAssetRegistry, type AssetPlugin } from "../lib/core/imported-assets/asset-registry";
import type { MeshVolumeHeader } from "../lib/core/imported-assets/mesh-volume";
import { discoverAssetPlugins, generateAssetCatalog } from "../tools/asset-catalog";

const header = (asset: string): MeshVolumeHeader => ({
  version: 1, units: "metres", upAxis: "Y", asset,
  origin: [0, 0, 0], dimensions: [8, 8, 8], cellSize: .01,
  bandWidth: .03, brickCount: 0, brickCells: 8, nodeStrideBytes: 16,
  materials: [{ name: "clay", baseColorLinear: [.5, .5, .5], roughness: .5, metallic: 0 }],
});
const plugin = (id: string, assetId: string = id): AssetPlugin => ({
  id, label: `Collection ${id}`, sources: [],
  assets: [{ id: assetId, name: `Asset ${assetId}`, meshUrl: `/assets/${id}/${assetId}.glb` }],
});

test("the bundled installation includes every currently discoverable asset", async () => {
  await generateAssetCatalog(fileURLToPath(new URL("../", import.meta.url)), true);
});

async function fixture(run: (root: string, put: (path: string, value?: string) => Promise<void>) => Promise<void>) {
  const root = await mkdtemp(join(tmpdir(), "fluid-asset-plugins-"));
  const put = async (path: string, value = "") => {
    const file = join(root, path);
    await mkdir(dirname(file), { recursive: true });
    await writeFile(file, value);
  };
  try { await run(root, put); } finally { await rm(root, { recursive: true, force: true }); }
}

test("an independently registered collection is discoverable without garden/pool branches", () => {
  const third: AssetPlugin = { id: "observatory", label: "Night observatory", sources: ["assets/observatory/night.blend"],
    assets: [{ id: "telescope", name: "Brass telescope", meshUrl: "/assets/observatory/optics.glb",
      volume: { url: "/assets/observatory/optics.fvx.gz", header: header("telescope") } }],
  };
  const registry = createAssetRegistry([plugin("garden"), plugin("pool"), third]);
  assert.equal(registry.plugins.length, 3);
  assert.deepEqual(registry.assets.map(asset => asset.id), ["garden", "pool", "telescope"]);
  const asset = registry.get("telescope");
  assert.equal(asset.collection, "observatory");
  assert.equal(asset.collectionLabel, "Night observatory");
  assert.equal(asset.volume?.url, "/assets/observatory/optics.fvx.gz");
  assert.deepEqual(asset.sourceFiles, ["assets/observatory/night.blend"]);
  assert.ok(Object.isFrozen(registry.assets));
  assert.ok(Object.isFrozen(asset));
  assert.throws(() => registry.get("not-installed"), /unknown/i);
});

test("registration rejects conflicting plugin/asset identities and mislabelled bake metadata", () => {
  assert.throws(() => createAssetRegistry([plugin("duplicate", "a"), plugin("duplicate", "b")]), /duplicate/i);
  assert.throws(() => createAssetRegistry([plugin("one", "shared"), plugin("two", "shared")]), /duplicate/i);
  assert.throws(() => createAssetRegistry([{ ...plugin("one"), assets: [{
    id: "one", name: "One", volume: { url: "/assets/one.fvx.gz", header: header("different") },
  }] }]), /metadata|header|match/i);
});

test("registry lookup does not inherit object prototype entries", () => {
  const registry = createAssetRegistry([plugin("custom", "constructor")]);
  assert.equal(registry.get("constructor").name, "Asset constructor");
  assert.throws(() => registry.get("toString"), /unknown/i);
  assert.throws(() => registry.get("__proto__"), /unknown/i);
});

test("registered metadata is a snapshot of the provider rather than a mutable alias", () => {
  const provider = { id: "independent", label: "Independent", sources: ["assets/independent/source.blend"], assets: [{
    id: "prop", name: "Original prop", volume: { url: "/assets/independent/prop.fvx.gz", header: header("prop") },
  }] };
  const registry = createAssetRegistry([provider]);
  provider.label = "Changed collection";
  provider.sources.push("assets/independent/new-source.blend");
  provider.assets[0].name = "Changed prop";
  provider.assets[0].volume.header.cellSize = 99;
  const registered = registry.get("prop");
  assert.equal(registered.collectionLabel, "Independent");
  assert.equal(registered.name, "Original prop");
  assert.equal(registered.volume?.header.cellSize, .01);
  assert.equal(registered.sourceFiles.length, 1);
  assert.ok(Object.isFrozen(registered.volume?.header));
});

test("discovery finds mesh-only, source-only and custom manifest exports with optional bakes", async () => {
  await fixture(async (root, put) => {
    await put("public/assets/characters/bub.glb");
    await put("assets/characters/bub.blend");
    await put("assets/waterfall/stillwater.blend");
    await put("assets/waterfall/stillwater-preview.blend");
    await put("assets/waterfall/stillwater.blend1");
    await put("public/assets/observatory/manifest.json", JSON.stringify({ collection: { id: "night", label: "Night sky" },
      source: "observatory.blend", assets: [{ id: "telescope", name: "Brass telescope", mesh: "optics.glb", volume: "optics.fvx.gz" }] }));
    await put("assets/observatory/observatory.blend");
    await put("public/assets/observatory/optics.glb");
    await put("public/assets/observatory/optics.fvx.gz");
    await put("public/assets/observatory/unlisted.glb");
    await put("lib/core/imported-assets/metadata/telescope-volume.json", JSON.stringify(header("telescope")));

    const discovered = await discoverAssetPlugins(root);
    const registry = createAssetRegistry(discovered);
    assert.equal(registry.get("bub").meshUrl, "/assets/characters/bub.glb");
    assert.equal(registry.get("bub").volume, undefined);
    assert.equal(registry.get("stillwater").meshUrl, undefined);
    assert.deepEqual(registry.get("stillwater").sourceFiles, ["assets/waterfall/stillwater.blend"]);
    assert.ok(!registry.assets.some(asset => asset.id.includes("preview")));
    const telescope = registry.get("telescope");
    assert.equal(telescope.collection, "night");
    assert.equal(telescope.collectionLabel, "Night sky");
    assert.equal(telescope.name, "Brass telescope");
    assert.equal(telescope.volume?.header.asset, "telescope");
    assert.equal(telescope.volume?.url, "/assets/observatory/optics.fvx.gz");
    assert.equal(registry.get("unlisted").collection, "night", "new exports need no manifest list edit");

    const generated = await generateAssetCatalog(root);
    const first = await readFile(generated.outputPath, "utf8");
    assert.equal((await generateAssetCatalog(root)).changed, false);
    assert.equal((await generateAssetCatalog(root, true)).changed, false);
    assert.equal(await readFile(generated.outputPath, "utf8"), first);
    await put("public/assets/characters/new-character.glb");
    await assert.rejects(generateAssetCatalog(root, true), /out of date/i);
    await generateAssetCatalog(root);
    assert.equal((await discoverAssetPlugins(root)).flatMap(collection => collection.assets).some(asset => asset.id === "new-character"), true);
  });
});

test("an unbaked manifest export remains registered without advertising a missing volume", async () => {
  await fixture(async (root, put) => {
    await put("public/assets/new-set/manifest.json", JSON.stringify({ assets: [{ id: "new-prop", mesh: "new-prop.glb", volume: "new-prop.fvx.gz" }] }));
    await put("public/assets/new-set/new-prop.glb");
    const registry = createAssetRegistry(await discoverAssetPlugins(root));
    assert.equal(registry.get("new-prop").volume, undefined);
  });
});

test("discovery includes root exports and standalone volumes without requiring a GLB", async () => {
  await fixture(async (root, put) => {
    await put("public/assets/root-prop.glb");
    await put("assets/root-prop.blend");
    await put("public/assets/fields/rock.fvx.gz");
    await put("lib/core/imported-assets/metadata/rock-volume.json", JSON.stringify(header("rock")));
    const registry = createAssetRegistry(await discoverAssetPlugins(root));
    const rootProp = registry.get("root-prop");
    assert.equal(rootProp.meshUrl, "/assets/root-prop.glb");
    assert.deepEqual(rootProp.sourceFiles, ["assets/root-prop.blend"]);
    const rock = registry.get("rock");
    assert.equal(rock.meshUrl, undefined);
    assert.equal(rock.volume?.url, "/assets/fields/rock.fvx.gz");
  });
});

test("discovery refuses broken references and unpublished or mismatched volume metadata", async () => {
  await fixture(async (root, put) => {
    await put("public/assets/broken/manifest.json", JSON.stringify({ assets: [{ id: "prop", mesh: "missing.glb" }] }));
    await assert.rejects(discoverAssetPlugins(root), /missing|exist|ENOENT/i);
    await put("public/assets/broken/manifest.json", JSON.stringify({ assets: [{ id: "prop", mesh: "../escaped.glb" }] }));
    await assert.rejects(discoverAssetPlugins(root), /path|traversal|relative|outside|escape/i);
    await put("public/assets/broken/manifest.json", JSON.stringify({ assets: [{ id: "prop", mesh: "prop.glb", volume: "prop.fvx.gz" }] }));
    await put("public/assets/broken/prop.glb");
    await put("public/assets/broken/prop.fvx.gz");
    await assert.rejects(discoverAssetPlugins(root), /metadata|ENOENT/i);
    await put("lib/core/imported-assets/metadata/prop-volume.json", JSON.stringify(header("other")));
    await assert.rejects(discoverAssetPlugins(root), /metadata|header|match/i);
  });
});
