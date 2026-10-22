# Asset plugins and discovery

The asset catalog is shared by the Shape Lab and imported-volume runtime. A
collection contributes an `AssetPlugin`: an ID, display label, source files and
asset definitions. Each asset declares its available files independently, so a
Blender source, a mesh export and a baked volume are distinct capabilities.
The lab lists all registered assets; only assets with a published volume offer
the bake preview. Mesh-only assets still expose their GLB download, and
source-only assets identify their editable Blender files.

## Add a collection

Put exported GLBs under `public/assets/<collection>/` and editable sources under
`assets/<collection>/`. Discovery includes every GLB, including exports that
are not listed in a manifest. It also registers `.blend` files from collections
without exports; generated `*preview.blend` files and numbered Blender backups
are excluded. Files directly under `public/assets/` or `assets/` use a generic
Assets collection. A standalone FVX can register without a GLB when its matching
metadata is present.

A collection can optionally add `public/assets/<collection>/manifest.json` to
supply stable IDs, readable names and nonstandard export filenames:

```json
{
  "collection": { "id": "observatory", "label": "Night observatory" },
  "source": "observatory.blend",
  "assets": [
    {
      "id": "brass-telescope",
      "name": "Brass telescope",
      "mesh": "optics.glb",
      "volume": "optics.fvx.gz"
    }
  ]
}
```

Manifest paths are relative to the collection directory. IDs must be unique
across the catalog. Without an override, the directory and filename determine
the collection and asset IDs and readable display names. No garden/pool branch,
URL switch, or asset-name list needs to be changed in the lab or renderer.

## Publish a bake

The existing Blender baker writes a compressed FVX payload beside the GLB and
a matching header to
`lib/core/imported-assets/metadata/<asset-id>-volume.json`. Discovery attaches a
volume capability only when its payload exists. A declared but not-yet-baked
volume does not hide the mesh or create a broken preview link. An existing
volume without its matching metadata is an error, as is metadata naming a
different asset.

The catalog contains metadata and URLs, not decoded meshes or volumes. Binary
asset loading remains lazy. The runtime's `MESH_ASSETS` compatibility view is
derived from registered volume capabilities, so mesh-only and source-only
assets cannot accidentally enter the sampled-field renderer.

## Generate and verify

```sh
npm run generate:assets
npm run check:assets
npm run check:types
npm run test:unit
```

The generator writes the bundled plugin module in the source tree. Commit that
module with the asset exports and metadata so Node tools and tests use the same
catalog as the browser. Vite regenerates it at startup and watches asset,
manifest, source and metadata changes during development. Catalog and metadata
changes reload the page so the renderer and UI keep matching asset indices.
The check command fails when regeneration would change the committed module.

Discovery is deterministic and validates missing manifest meshes, unsafe paths,
duplicate identities, and volume metadata. Regression tests install an
independent third collection and exercise source-only, mesh-only and baked
assets, custom filenames, unlisted exports, and stale generated catalogs.

## Registry contract

`createAssetRegistry(plugins)` composes collection plugins into a validated,
immutable catalog. Consumers enumerate `registry.assets` or look up a definition
with `registry.get(id)`; registered entries include their collection label and
source files. Unknown IDs fail explicitly. Plugins can also be composed directly
for another host or a test without modifying or mutating the bundled catalog.
