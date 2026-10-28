/** Process-local asset review, without publishing staged files to the app.
 * FLUID_ASSET_VARIANTS=/absolute/variants.json node --import tsx
 *   --import ./tools/blender/preload-asset-variants.ts tools/capture-forest-waterfall.ts ...
 * Manifest: [{"id":"known-asset-id","volume":"/absolute/staged.fvx.gz"}]
 * Run before importing the scene/renderer: their registry snapshots this data.
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import { bundledAssetPlugins } from "../../lib/core/imported-assets/bundled-asset-plugins";
import type { AssetPlugin } from "../../lib/core/imported-assets/asset-registry";
import type { MeshVolumeHeader } from "../../lib/core/imported-assets/mesh-volume";

const manifestPath = process.env.FLUID_ASSET_VARIANTS;
if (manifestPath) {
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as {id:string;volume:string}[];
  assert.ok(Array.isArray(manifest) && manifest.length > 0, "Asset variants must be a nonempty array");
  const publicRoot = fileURLToPath(new URL("../../public/", import.meta.url));
  const plugins: readonly AssetPlugin[] = bundledAssetPlugins;
  const seen = new Set<string>(), proof = [];
  for (const variant of manifest) {
    assert.ok(typeof variant.id === "string" && typeof variant.volume === "string");
    assert.ok(!seen.has(variant.id), `Duplicate asset variant ${variant.id}`); seen.add(variant.id);
    const entry = plugins.flatMap(plugin => plugin.assets).find(asset => asset.id === variant.id);
    assert.ok(entry?.volume, `Unknown sampled asset ${variant.id}`);
    const path = resolve(dirname(resolve(manifestPath)), variant.volume);
    const compressed = readFileSync(path);
    const bytes = compressed[0] === 0x1f && compressed[1] === 0x8b ? gunzipSync(compressed) : compressed;
    assert.ok(bytes.length >= 8 && bytes.toString("utf8", 0, 4) === "FVX1", "Invalid staged FVX");
    const length = bytes.readUInt32LE(4);
    assert.ok(length <= bytes.length - 8, "Truncated staged FVX header");
    const header = JSON.parse(bytes.toString("utf8", 8, 8 + length)) as MeshVolumeHeader;
    assert.equal(header.asset, variant.id, "Staged identity must match its catalog entry");
    proof.push({ id: variant.id, path, sha256: createHash("sha256").update(bytes).digest("hex"),
      original: entry.volume.header, variant: header });
    // Only this Node process sees the replacement. The ordinary registry,
    // decoder and registerMeshAsset validation still validate the staged bake.
    Object.assign(entry.volume, { header, url: `/${relative(publicRoot, path).split("\\").join("/")}` });
  }
  process.env.FLUID_ASSET_VARIANT_PROOF = JSON.stringify(proof);
  console.log(JSON.stringify({phase:"staged-asset-variants",assets:proof.map(({id,path,sha256})=>({id,path,sha256}))}));
}
