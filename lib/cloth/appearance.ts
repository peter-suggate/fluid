import type {ClothDocument,ClothSurfacePublication,ClothVector} from './types';

/** Pinned application's diagnostic colors, not the paper's offline materials.
 * TwoDSceneRenderer.cpp: face_color(.75), fluid_color(blue), group grayscale,
 * and fluid volume / total material control volume (not pore saturation). */
export const SOURCE_CLOTH_PREVIEW=Object.freeze({
 source:'https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/libWetCloth/App/TwoDSceneRenderer.cpp',
 dryShell:[.75,.75,.75] as ClothVector,water:[0,0,1] as ClothVector,
 description:'Released source preview palette; paper Houdini appearance remains unverified',
});

export function clothVertexColor(document:ClothDocument,surface:ClothSurfacePublication,material:number,vertex:number,kind:'shell'|'rod',groupCount=1):ClothVector{
 if(document.appearance==='source-preview'){
  const water=surface.waterVolume_m3[vertex]??0,volume=surface.controlVolume_m3?.[vertex];
  if(water>0&&volume===undefined)throw new Error('Source cloth preview requires published control volume');
  // Source epsilon is 1e-12 cm³, converted to SI cubic metres.
  const fraction=Math.max(0,Math.min(1,water/Math.max(1e-18,volume??0)));
  const dry=kind==='shell'?.75:((surface.topology.groups?.[vertex]??0)%groupCount)/groupCount;
  return [dry*(1-fraction),dry*(1-fraction),dry*(1-fraction)+fraction];
 }
 const base=document.materials[material]!.baseColor??[.72,.66,.51],saturation=surface.saturation?.[vertex]??0;
 const darkening=1-.42*Math.max(0,Math.min(1,saturation));return [base[0]*darkening,base[1]*darkening,base[2]*darkening];
}
