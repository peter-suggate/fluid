import { meshVolumeMaterialWeightsWGSL } from "../../../core/imported-assets/mesh-volume-material.wgsl";

/** Reconstruct the authored optical fields in linear space. Stable discrete
 * material IDs still govern ownership, picking, visibility and sculpting.
 * This only samples PBR attributes after a surface has already been accepted. */
export function sampledMaterialFilterWGSL(loadWord: string, loadMaterial: string, validMaterial: string): string {
  return /* wgsl */ `
${meshVolumeMaterialWeightsWGSL("sampledMaterialWeights", loadWord)}
struct SampledMaterialPbr {
  baseColor:vec3f,roughness:f32,emissive:vec3f,metallic:f32,
  specularF0:vec3f,specularWeight:f32,weight:f32,
}
fn sampledMaterialPbr(volumeBase:u32,paletteBase:u32,point:vec3f)->SampledMaterialPbr{
  var result:SampledMaterialPbr;
  if(paletteBase==0u){return result;}
  let samples=sampledMaterialWeights(volumeBase,point);
  for(var corner=0u;corner<8u;corner++){
    let w=samples.weights[corner];let localId=samples.ids[corner];
    if(localId==0u||w<=0.0){continue;}
    let id=paletteBase+localId-1u;let material=${loadMaterial}(id);
    if(!${validMaterial}(material,id)){continue;}
    result.baseColor+=w*material.baseColorOpacity.xyz;
    result.roughness+=w*material.emissiveRoughness.w;
    result.emissive+=w*material.emissiveRoughness.xyz;
    result.metallic+=w*material.surface.x;
    result.specularF0+=w*vec3f(svoMaterialDielectricF0(material));
    result.specularWeight+=w*material.surface.y;
    result.weight+=w;
  }
  if(result.weight>0.0){
    result.baseColor/=result.weight;result.roughness/=result.weight;
    result.emissive/=result.weight;result.metallic/=result.weight;
    result.specularF0/=result.weight;result.specularWeight/=result.weight;
  }
  return result;
}
`;
}
