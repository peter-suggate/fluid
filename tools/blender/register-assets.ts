/** Node harness preload; browser/render workers load the same files via fetch. */
import { readFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import { MESH_ASSETS, registerMeshAsset } from "../../lib/core/imported-assets/asset-library";
import { decodeMeshVolume } from "../../lib/core/imported-assets/mesh-volume";
await Promise.all(Object.keys(MESH_ASSETS).map(async id => {
  const file = await readFile(new URL(`../../public/assets/pool-party/${id}.fvx.gz`, import.meta.url));
  registerMeshAsset(id, decodeMeshVolume(gunzipSync(file)));
}));
