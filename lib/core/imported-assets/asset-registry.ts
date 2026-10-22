import type { MeshVolumeHeader } from "./mesh-volume";

/** A collection contributes data, not renderer or lab-specific branching. */
export interface AssetDefinition {
  readonly id: string;
  readonly name: string;
  readonly meshUrl?: string;
  readonly volume?: { readonly url: string; readonly header: MeshVolumeHeader };
  readonly sourceFiles?: readonly string[];
}

export interface AssetPlugin {
  readonly id: string;
  readonly label: string;
  readonly sources: readonly string[];
  readonly assets: readonly AssetDefinition[];
}

export interface RegisteredAsset extends AssetDefinition {
  readonly collection: string;
  readonly collectionLabel: string;
  readonly sourceFiles: readonly string[];
}

export interface AssetRegistry {
  readonly plugins: readonly AssetPlugin[];
  readonly assets: readonly RegisteredAsset[];
  get(id: string): RegisteredAsset;
}

function nonempty(value: string, label: string): void {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${label} must be nonempty`);
}

function freeze<T>(value: T): T {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}

/** Resolve plugins once at the composition root. Snapshotting prevents a later
 * provider mutation from changing runtime asset IDs or their published headers. */
export function createAssetRegistry(input: readonly AssetPlugin[]): AssetRegistry {
  const plugins = freeze(structuredClone(input));
  const pluginIds = new Set<string>();
  const byId = new Map<string, RegisteredAsset>();
  for (const plugin of plugins) {
    nonempty(plugin.id, "Asset plugin ID");
    nonempty(plugin.label, `Asset plugin ${plugin.id} label`);
    if (pluginIds.has(plugin.id)) throw new Error(`Duplicate asset plugin ID: ${plugin.id}`);
    pluginIds.add(plugin.id);
    for (const source of plugin.sources) nonempty(source, `Asset plugin ${plugin.id} source path`);
    for (const entry of plugin.assets) {
      nonempty(entry.id, `Asset ID in ${plugin.id}`);
      nonempty(entry.name, `Asset ${entry.id} name`);
      if (byId.has(entry.id)) throw new Error(`Duplicate asset ID: ${entry.id}`);
      const sourceFiles = entry.sourceFiles ?? plugin.sources;
      if (!entry.meshUrl && !entry.volume && sourceFiles.length === 0) {
        throw new Error(`Asset ${entry.id} has no source, mesh or volume`);
      }
      for (const source of sourceFiles) nonempty(source, `Asset ${entry.id} source path`);
      if (entry.meshUrl !== undefined) nonempty(entry.meshUrl, `Asset ${entry.id} mesh URL`);
      if (entry.volume) {
        nonempty(entry.volume.url, `Asset ${entry.id} volume URL`);
        const header = entry.volume.header;
        if (header.asset !== entry.id) throw new Error(`Asset ${entry.id} does not match volume metadata ${header.asset}`);
        if (header.version !== 1 || header.units !== "metres" || header.upAxis !== "Y"
          || !Number.isFinite(header.cellSize) || header.cellSize <= 0
          || header.dimensions.length !== 3 || !header.dimensions.every((n) => Number.isSafeInteger(n) && n > 0 && n % 8 === 0)) {
          throw new Error(`Invalid volume metadata for asset ${entry.id}`);
        }
      }
      byId.set(entry.id, freeze({ ...entry, collection: plugin.id, collectionLabel: plugin.label, sourceFiles }));
    }
  }
  return Object.freeze({
    plugins,
    assets: Object.freeze([...byId.values()]),
    get(id: string): RegisteredAsset {
      const asset = byId.get(id);
      if (!asset) throw new Error(`Unknown asset: ${id}`);
      return asset;
    },
  });
}
