/** Node harness preload; browser/render workers load the same files via fetch. */
import { readFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import { meshAssetUrl, registerMeshAsset, sceneMeshAssetIds } from "../../lib/core/imported-assets/asset-library";
import { decodeMeshVolume } from "../../lib/core/imported-assets/mesh-volume";
import type { SceneDescription } from "../../lib/core/model";
/** Load only this scene's assets, so unrelated collections cost no memory. */
export async function registerSceneMeshAssets(scene: SceneDescription): Promise<void> {
  await Promise.all(sceneMeshAssetIds(scene).map(async id => {
    const file = await readFile(new URL(`../../public${meshAssetUrl(id)}`, import.meta.url));
    registerMeshAsset(id, decodeMeshVolume(gunzipSync(file)));
  }));
}
