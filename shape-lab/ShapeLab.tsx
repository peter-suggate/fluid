"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, Box, Download, Search } from "lucide-react";
import { ThemeSwitch } from "../components/ThemeSwitch";
import { Button, Choice, Facts, FieldList, Readout, SelectField, Swatch } from "../components/ui";
import { AssetPreview } from "./AssetPreview";
import { assetCollections, assetColor, blenderAssets, isBakedBlenderAsset, selectedBlenderAsset } from "./assets";
import { SHAPE_LAB_STYLES } from "./shape-lab-styles";
import type { ShapeLabShading } from "./trace";

type Collection = typeof assetCollections[number]["value"];
type Surface = "surface" | "voxel";
const shadings = [
  { value: "material", label: "Material" },
  { value: "clay", label: "Clay" },
  { value: "normal", label: "Normals" },
] as const;
const surfaces = [
  { value: "surface", label: "Baked surface" },
  { value: "voxel", label: "Voxels" },
] as const;

function readSelection() {
  const query = new URLSearchParams(window.location.search);
  return {
    asset: selectedBlenderAsset(query.get("asset")),
    shading: shadings.find(({ value }) => value === query.get("shading"))?.value ?? "material",
    mode: query.get("mode") === "voxel" ? "voxel" as const : "surface" as const,
    voxelScale: [1, 2, 4].find((scale) => String(scale) === query.get("scale")) ?? 1,
  };
}

/** The lab is a browser-only inspection tool; volume decoding stays in its worker. */
export function ShapeLab() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted ? <AssetWorkbench /> : null;
}

function AssetWorkbench() {
  const [initial] = useState(readSelection);
  const [asset, setAsset] = useState(initial.asset);
  const [shading, setShading] = useState<ShapeLabShading>(initial.shading);
  const [mode, setMode] = useState<Surface>(initial.mode);
  const [voxelScale, setVoxelScale] = useState(initial.voxelScale);
  const [collection, setCollection] = useState<Collection>("");
  const [search, setSearch] = useState("");
  const [revision, setRevision] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const list = listRef.current;
    const selected = list?.querySelector<HTMLElement>(".is-selected");
    if (!list || !selected) return;
    if (selected.offsetTop < list.scrollTop || selected.offsetTop + selected.offsetHeight > list.scrollTop + list.clientHeight) {
      list.scrollTop = selected.offsetTop - (list.clientHeight - selected.offsetHeight) / 2;
    }
  }, [asset.id, collection, search]);

  useEffect(() => {
    const url = new URL(window.location.href);
    // Retire the procedural lab's query state while preserving unrelated keys.
    for (const key of ["specimen", "depth", "ground"]) url.searchParams.delete(key);
    url.searchParams.set("asset", asset.id);
    url.searchParams.set("shading", shading);
    url.searchParams.set("mode", mode);
    if (voxelScale === 1) url.searchParams.delete("scale");
    else url.searchParams.set("scale", String(voxelScale));
    window.history.replaceState(window.history.state, "", url.pathname + url.search + url.hash);
  }, [asset.id, shading, mode, voxelScale]);

  const filtered = useMemo(() => {
    const terms = search.trim().toLowerCase().split(/\s+/);
    return blenderAssets.filter((entry) => (collection === "" || entry.collection === collection)
      && terms.every((term) => `${entry.name} ${entry.id} ${entry.collection} ${entry.collectionLabel} ${entry.sourceFiles.join(" ")}`.toLowerCase().includes(term)));
  }, [collection, search]);
  const materialFamilies = useMemo(() => [...new Map((asset.header?.materials ?? []).map((material) => [material.name.split("/")[0], { ...material, name: material.name.split("/")[0] }])).values()], [asset]);
  const baked = isBakedBlenderAsset(asset) ? asset : undefined;
  const assetState = baked ? "Baked volume" : asset.meshUrl ? "Mesh exported" : "Blender source";

  return <main className="sl-root" aria-label="Asset lab">
    <style>{SHAPE_LAB_STYLES}</style>
    <header className="library-bar sl-bar">
      <a className="library-brand" href="/" aria-label="Fluid Lab scene library">
        <span className="brand-mark" aria-hidden="true">FL</span>
        <span className="library-wordmark">Fluid Lab</span>
      </a>
      <span className="sl-bar-divider" aria-hidden="true" />
      <span className="sl-bar-title">Shape lab <span> / Assets</span></span>
      <a className="pill sl-back" href="/"><ArrowLeft size={14} aria-hidden="true" />Scenes</a>
      <ThemeSwitch />
    </header>

    <div className="sl-workspace">
      <aside className="sl-library" aria-label="Asset library">
        <div className="sl-library-heading">
          <div><span className="sl-eyebrow">Library</span><h1>Asset library</h1></div>
          <span className="chip">{blenderAssets.length}</span>
        </div>
        <p className="sl-description">Browse sources, exported meshes and baked volumes.</p>
        <label className="pill library-search sl-search">
          <Search size={15} aria-hidden="true" />
          <input type="search" placeholder="Find an asset…" aria-label="Search assets" value={search}
            onChange={(event) => setSearch(event.target.value)} />
        </label>
        <FieldList>
          <SelectField label="Collection" value={collection} options={assetCollections} onChange={setCollection} />
        </FieldList>
        <div className="sl-results" role="status">{filtered.length} {filtered.length === 1 ? "asset" : "assets"}</div>
        <div className="sl-asset-list" ref={listRef}>
          {assetCollections.filter(({ value }) => value !== "").map(({ value, label }) => {
            const entries = filtered.filter((entry) => entry.collection === value);
            if (!entries.length) return null;
            return <section key={value} aria-label={label}>
              <h2 className="sl-list-heading">{label}<span>{entries.length}</span></h2>
              {entries.map((entry) => <Button key={entry.id} className={`sl-asset${entry.id === asset.id ? " is-selected" : ""}`}
                ariaLabel={`${entry.name}${entry.id === asset.id ? ", selected" : ""}`} onClick={() => setAsset(entry)}>
                <span className="sl-asset-icon" aria-hidden="true"><Box size={19} /></span>
                <span className="sl-asset-text"><strong>{entry.name}</strong><small>{entry.header ? `${entry.header.cellSize * 1000} mm bake` : entry.meshUrl ? "Mesh exported" : "Blender source"}</small></span>
                <span className="sl-selected-dot" aria-hidden="true" />
              </Button>)}
            </section>;
          })}
          {filtered.length === 0 && <div className="sl-empty">
            <Box size={26} aria-hidden="true" /><strong>No matching assets</strong>
            <p>Try another name or collection.</p>
            <Button onClick={() => { setSearch(""); setCollection(""); }}>Clear filters</Button>
          </div>}
        </div>
        <p className="sl-library-foot">{assetCollections.length - 1} collections · source and exported assets</p>
      </aside>

      <section className="sl-viewer" aria-labelledby="sl-asset-title">
        <header className="sl-viewer-heading">
          <div><span className="sl-eyebrow">{asset.collectionLabel}</span><h2 id="sl-asset-title">{asset.name}</h2></div>
          <span className="chip" data-tone={baked ? "accent" : undefined}>{baked ? `${baked.header.cellSize * 1000} mm source` : assetState}</span>
        </header>
        {baked ? <>
        <div className="sl-view-options">
          <Choice ariaLabel="Preview surface" value={mode} options={surfaces} onChange={setMode} />
          <Choice ariaLabel="Preview shading" value={shading} options={shadings} onChange={setShading} />
        </div>
        {mode === "voxel" && <FieldList className="sl-voxel-options">
          <SelectField label="Voxel size" value={String(voxelScale)} onChange={(value) => setVoxelScale(Number(value))}
            options={[1, 2, 4].map((scale) => ({ value: String(scale), label: `${baked.header.cellSize * 1000 * scale} mm${scale === 1 ? " · source spacing" : ""}` }))} />
        </FieldList>}
        <AssetPreview key={`${asset.id}:${revision}`} asset={baked} mode={mode} shading={shading} voxelScale={voxelScale} />
        <p className="sl-preview-caption">{mode === "surface" ? "Surface reconstructed from the published Blender bake." : "Solid cells sampled from the published Blender bake."} Neutral inspection lighting.</p>
        </> : <div className="sl-unbaked-preview">
          <Box size={38} strokeWidth={1.25} aria-hidden="true" />
          <h3>{assetState}</h3>
          <p>{asset.meshUrl
            ? "The mesh is available to download. Bake it to a volume to inspect its surface, materials and voxel sampling here."
            : "This collection has editable Blender source files. Export a mesh and bake a volume to enable the interactive preview."}</p>
          {asset.meshUrl && <a className="pill" href={asset.meshUrl} download><Download size={15} aria-hidden="true" />Download mesh</a>}
        </div>}
      </section>

      <aside className="sl-inspector" aria-label="Asset details">
        <section className="sl-detail-section">
          <span className="sl-eyebrow">Asset details</span>
          <h2>{assetState}</h2>
          <p className="sl-description">{baked ? "Inspect the exported asset at its authored scale and sampling resolution." : "Source and export availability for this asset."}</p>
          <Facts items={[
            { label: "Collection", value: asset.collectionLabel },
            { label: "Asset ID", value: <code>{asset.id}</code> },
          ]} />
        </section>
        {baked && <section className="sl-detail-section">
          <h3>Dimensions</h3>
          <FieldList>
            {baked.size.map((size, axis) => <Readout key={axis} label={["Width · X", "Height · Y", "Depth · Z"][axis]} value={`${size.toFixed(3)} m`} />)}
            <Readout label="Source spacing" value={`${baked.header.cellSize * 1000} mm`} />
          </FieldList>
          <p className="sl-small">Sampled bounds, including the bake’s distance band.</p>
        </section>}
        {baked && <section className="sl-detail-section">
          <div className="sl-section-heading"><h3>Materials</h3><span className="chip">{materialFamilies.length}</span></div>
          <div className="sl-materials">
            {materialFamilies.map((material) => <div key={material.name} className="sl-material" title={`${material.name} · roughness ${material.roughness} · metallic ${material.metallic}`}>
              <Swatch color={assetColor(material.baseColorLinear)} /><span>{material.name.replace(/-/g, " ")}</span>
            </div>)}
          </div>
          <p className="sl-small">{baked.header.materials.length} baked colour samples. Materials and source normals are retained in the volume.</p>
        </section>}
        {(asset.meshUrl || baked) && <section className="sl-detail-section">
          <h3>Asset files</h3>
          {asset.meshUrl && <a className="sl-file" href={asset.meshUrl} download><Box size={17} aria-hidden="true" /><span><strong>Exported mesh</strong><small>GLB · geometry & textures</small></span><Download size={15} aria-hidden="true" /></a>}
          {baked && <>
            <a className="sl-file" href={baked.volumeUrl} download><Box size={17} aria-hidden="true" /><span><strong>Baked volume</strong><small>FVX · {baked.header.brickCount.toLocaleString()} bricks</small></span><Download size={15} aria-hidden="true" /></a>
            <p className="sl-small">Edit the source in Blender, export and re-bake, then reload the preview.</p>
            <Button onClick={() => setRevision((value) => value + 1)}>Reload asset</Button>
          </>}
        </section>}
        {asset.sourceFiles.length > 0 && <section className="sl-detail-section">
          <h3>Blender source</h3>
          <ul className="sl-source-files">{asset.sourceFiles.map((path) => <li key={path}><code>{path}</code></li>)}</ul>
          <p className="sl-small">Workspace-relative paths. Open these files in Blender to edit the asset.</p>
        </section>}
      </aside>
    </div>
  </main>;
}
