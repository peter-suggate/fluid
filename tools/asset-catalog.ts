/** Build-time discovery. Browser and worker entry points consume only the
 * generated plugin installation, never the filesystem or public JSON imports. */
import { readFile, readdir, mkdir, rename, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createAssetRegistry, type AssetDefinition, type AssetPlugin } from "../lib/core/imported-assets/asset-registry";
import type { MeshVolumeHeader } from "../lib/core/imported-assets/mesh-volume";

const GENERATED = "lib/core/imported-assets/bundled-asset-plugins.ts";
const METADATA = "lib/core/imported-assets/metadata";
const title = (value: string) => value.replace(/[-_]/g, " ").replace(/\b\w/g, letter => letter.toUpperCase());
const safeId = (value: unknown, context: string): string => {
  if (typeof value !== "string" || !/^[a-zA-Z0-9][a-zA-Z0-9_.-]*$/.test(value)) throw new Error(`Invalid asset ID in ${context}`);
  return value;
};

async function readOptional(path: string): Promise<string | undefined> {
  try { return await readFile(path, "utf8"); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; }
}
async function files(directory: string, prefix = ""): Promise<string[]> {
  let entries;
  try { entries = await readdir(directory, { withFileTypes: true }); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return []; throw error; }
  const found: string[] = [];
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name, "en"))) {
    if (entry.name.startsWith(".")) continue;
    const name = prefix + entry.name;
    if (entry.isDirectory()) found.push(...await files(resolve(directory, entry.name), `${name}/`));
    else if (entry.isFile()) found.push(name);
  }
  return found;
}

/** Manifest references are collection-relative files, never arbitrary paths. */
function manifestFile(value: unknown, context: string, extension: string): string {
  if (typeof value !== "string" || value.includes("\\") || value.startsWith("/")
    || value.split("/").some(part => !part || part === "." || part === "..") || !value.endsWith(extension)) {
    throw new Error(`Invalid ${extension} path in ${context}`);
  }
  return value;
}
const urlFor = (folder: string, file: string) => `/assets/${folder ? `${folder}/` : ""}${file}`.split("/").map(encodeURIComponent).join("/");

interface ManifestEntry { id?: unknown; name?: unknown; mesh?: unknown; volume?: unknown }
interface Manifest { collection?: { id?: unknown; label?: unknown }; source?: unknown; assets?: ManifestEntry[] }

export async function discoverAssetPlugins(root: string): Promise<AssetPlugin[]> {
  const exported = await files(resolve(root, "public/assets"));
  const authored = (await files(resolve(root, "assets"))).filter(file => file.endsWith(".blend") && !/preview\.blend$/i.test(file));
  const folders = [...new Set([...exported.filter(file => /(?:\.(glb|fvx\.gz)|(?:^|\/)manifest\.json)$/.test(file)), ...authored]
    .map(file => file.includes("/") ? file.split("/")[0]! : ""))].sort();
  const plugins: AssetPlugin[] = [];
  const assetOwners = new Map<string, string>(), pluginIds = new Set<string>();
  for (const folder of folders) {
    const prefix = folder ? `${folder}/` : "";
    const belongs = (file: string) => folder ? file.startsWith(prefix) : !file.includes("/");
    const publicFiles = new Set(exported.filter(belongs).map(file => file.slice(prefix.length)));
    const sourceFiles = authored.filter(belongs).map(file => `assets/${file}`);
    const manifestPath = `public/assets/${prefix}manifest.json`;
    const raw = await readOptional(resolve(root, manifestPath));
    const manifest: Manifest = raw === undefined ? {} : JSON.parse(raw);
    if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)
      || (manifest.collection !== undefined && (!manifest.collection || typeof manifest.collection !== "object" || Array.isArray(manifest.collection)))
      || (manifest.assets !== undefined && !Array.isArray(manifest.assets))) {
      throw new Error(`Invalid asset manifest ${manifestPath}`);
    }
    const id = safeId(manifest.collection?.id ?? (folder || "assets"), manifestPath);
    if (pluginIds.has(id)) throw new Error(`Duplicate asset plugin ID: ${id}`);
    pluginIds.add(id);
    const label = manifest.collection?.label ?? title(folder || "assets");
    if (typeof label !== "string" || !label.trim()) throw new Error(`Invalid collection label in ${manifestPath}`);
    let commonSources = sourceFiles;
    if (manifest.source !== undefined) {
      const source = `assets/${prefix}${manifestFile(manifest.source, manifestPath, ".blend")}`;
      if (!sourceFiles.includes(source)) throw new Error(`Missing Blender source ${source}`);
      commonSources = [source];
    }
    const assets: AssetDefinition[] = [];
    const consumed = new Set<string>();
    const add = async (entry: ManifestEntry, inferredMesh?: string, inferredVolume?: string) => {
      const mesh = entry.mesh !== undefined ? manifestFile(entry.mesh, manifestPath, ".glb") : inferredMesh;
      const volume = entry.volume !== undefined ? manifestFile(entry.volume, manifestPath, ".fvx.gz")
        : inferredVolume ?? mesh?.replace(/\.glb$/, ".fvx.gz");
      if (mesh && !publicFiles.has(mesh)) throw new Error(`Missing asset mesh public/assets/${prefix}${mesh}`);
      if (!mesh && (!volume || !publicFiles.has(volume))) throw new Error(`Asset in ${manifestPath} has no mesh or baked volume`);
      const assetId = safeId(entry.id ?? (mesh ?? volume)!.replace(/\.(glb|fvx\.gz)$/, "").replaceAll("/", "-"), manifestPath);
      const owner = assetOwners.get(assetId);
      if (owner !== undefined) throw new Error(`Duplicate asset ID ${assetId} in ${owner} and ${folder}`);
      assetOwners.set(assetId, folder);
      if (mesh) consumed.add(mesh);
      if (volume) consumed.add(volume);
      const name = entry.name ?? title(assetId);
      if (typeof name !== "string" || !name.trim()) throw new Error(`Invalid asset name ${assetId}`);
      let bake: AssetDefinition["volume"];
      if (volume && publicFiles.has(volume)) {
        const headerPath = `${METADATA}/${assetId}-volume.json`;
        const metadata = await readOptional(resolve(root, headerPath));
        if (metadata === undefined) throw new Error(`Missing volume metadata ${headerPath}`);
        const header = JSON.parse(metadata) as MeshVolumeHeader;
        if (!header || header.asset !== assetId) throw new Error(`Volume metadata does not match asset ${assetId}`);
        bake = { url: urlFor(folder, volume), header };
      }
      const matchingSource = sourceFiles.find(source => source === `assets/${prefix}${assetId}.blend`);
      assets.push({ id: assetId, name, ...(mesh ? { meshUrl: urlFor(folder, mesh) } : {}),
        ...(bake ? { volume: bake } : {}), sourceFiles: matchingSource ? [matchingSource] : commonSources });
    };
    for (const entry of manifest.assets ?? []) {
      if (!entry || typeof entry !== "object") throw new Error(`Invalid asset entry in ${manifestPath}`);
      await add(entry);
    }
    for (const mesh of [...publicFiles].filter(file => file.endsWith(".glb")).sort()) {
      if (!consumed.has(mesh)) await add({}, mesh);
    }
    for (const volume of [...publicFiles].filter(file => file.endsWith(".fvx.gz")).sort()) {
      if (!consumed.has(volume)) await add({}, undefined, volume);
    }
    if (assets.length === 0) for (const source of sourceFiles) {
      const assetId = safeId(source.slice(`assets/${prefix}`.length).replace(/\.blend$/, "").replaceAll("/", "-"), source);
      const owner = assetOwners.get(assetId);
      if (owner !== undefined) throw new Error(`Duplicate asset ID ${assetId} in ${owner} and ${folder}`);
      assetOwners.set(assetId, folder);
      assets.push({ id: assetId, name: title(assetId), sourceFiles: [source] });
    }
    if (assets.length) plugins.push({ id, label, sources: sourceFiles, assets });
  }
  // Validate the same capabilities that every runtime entry point installs,
  // before publishing generated code that would prevent the app from starting.
  createAssetRegistry(plugins);
  return plugins;
}

function catalogSource(plugins: readonly AssetPlugin[]): string {
  const headers = plugins.flatMap(plugin => plugin.assets.filter(asset => asset.volume));
  const imports = headers.map((asset, index) => `import header${index} from ${JSON.stringify(`./metadata/${asset.id}-volume.json`)};`);
  const headerIndex = new Map(headers.map((asset, index) => [asset.id, index]));
  const lines = plugins.map(plugin => {
    const assets = plugin.assets.map(asset => {
      const { volume, ...rest } = asset;
      const properties = JSON.stringify(rest).slice(1, -1);
      return `      { ${properties}${volume ? `, volume: { url: ${JSON.stringify(volume.url)}, header: header${headerIndex.get(asset.id)} as unknown as MeshVolumeHeader }` : ""} },`;
    });
    return `  { id: ${JSON.stringify(plugin.id)}, label: ${JSON.stringify(plugin.label)}, sources: ${JSON.stringify(plugin.sources)}, assets: [\n${assets.join("\n")}\n    ] },`;
  });
  return `// Generated by tools/asset-catalog.ts. Run npm run generate:assets; do not edit.\nimport type { AssetPlugin } from "./asset-registry";\nimport type { MeshVolumeHeader } from "./mesh-volume";\n${imports.join("\n")}\n\nexport const bundledAssetPlugins = [\n${lines.join("\n")}\n] satisfies readonly AssetPlugin[];\n`;
}

export async function generateAssetCatalog(root: string, check = false) {
  const plugins = await discoverAssetPlugins(root);
  const outputPath = resolve(root, GENERATED);
  const source = catalogSource(plugins);
  const changed = await readOptional(outputPath) !== source;
  if (changed && check) throw new Error("Asset catalog is out of date. Run npm run generate:assets.");
  if (changed) {
    await mkdir(dirname(outputPath), { recursive: true });
    const temporary = `${outputPath}.${process.pid}.tmp`;
    await writeFile(temporary, source);
    await rename(temporary, outputPath);
  }
  return { changed, plugins, outputPath };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const result = await generateAssetCatalog(process.cwd(), process.argv.includes("--check"));
  console.log(`${result.plugins.length} asset plugins, ${result.plugins.reduce((count, plugin) => count + plugin.assets.length, 0)} assets${result.changed ? " — catalog updated" : " — catalog current"}`);
}
