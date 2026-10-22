import { assetRegistry } from "../lib/core/imported-assets/asset-catalog";
import type { MeshVolumeHeader } from "../lib/core/imported-assets/mesh-volume";

/** Sources, meshes and baked fields all remain discoverable in the same library. */
export const blenderAssets = assetRegistry.assets.map((asset) => {
  const header = asset.volume?.header;
  return {
    ...asset,
    header,
    volumeUrl: asset.volume?.url,
    size: header?.dimensions.map((cells) => cells * header.cellSize),
  };
});

export type BlenderAsset = typeof blenderAssets[number];
export type BakedBlenderAsset = BlenderAsset & { header: MeshVolumeHeader; volumeUrl: string; size: number[] };

export function isBakedBlenderAsset(asset: BlenderAsset): asset is BakedBlenderAsset {
  return asset.header !== undefined && asset.volumeUrl !== undefined && asset.size !== undefined;
}

export const assetCollections = [
  { value: "", label: "All assets" },
  ...assetRegistry.plugins.map((plugin) => ({ value: plugin.id, label: plugin.label })),
];

export function selectedBlenderAsset(id: string | null): BlenderAsset {
  const asset = blenderAssets.find((entry) => entry.id === id)
    ?? blenderAssets.find((entry) => entry.id === "teak-lounger") ?? blenderAssets[0];
  if (!asset) throw new Error("The asset registry contains no assets");
  return asset;
}

export function assetColor(color: readonly number[]): string {
  return `rgb(${color.map((value) => Math.round(255 * (value <= 0.0031308 ? 12.92 * value : 1.055 * value ** (1 / 2.4) - 0.055))).join(" ")})`;
}
