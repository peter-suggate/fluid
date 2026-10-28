import { meshAssetVolume } from "./asset-library";
import type { SurfaceMapping } from "./surface-mapping";

export interface SurfaceTextureAtlas {
  readonly mappings: readonly SurfaceMapping[];
  readonly width: number; readonly height: number; readonly layers: number; readonly mipLevels: number;
}
/** Deduplicated in the same first-use order as field-program packing. Images
 * keep their native dimensions in the upper-left of each layer; periodic
 * sampling uses those dimensions, so differently sized images never bleed. */
export function meshSurfaceTextureAtlas(assets: Iterable<string>): SurfaceTextureAtlas | undefined {
  const mappings=[...new Set(assets)].flatMap(id=>{
    const mapping=meshAssetVolume(id).surfaceMapping; return mapping ? [mapping] : [];
  });
  if(!mappings.length) return undefined;
  return {mappings,width:Math.max(...mappings.map(m=>m.header.textureWidth)),
    height:Math.max(...mappings.map(m=>m.header.textureHeight)),
    layers:mappings.reduce((sum,m)=>sum+m.header.textureLayerCount,0),
    mipLevels:Math.max(...mappings.map(m=>m.header.mipLevelCount))};
}
