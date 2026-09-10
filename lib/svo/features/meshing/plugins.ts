import { svoDualMarchingCubesFitWGSL } from "./dual-marching-cubes";
import { svoDualMarchingCubesMeshWGSL } from "./dual-marching-cubes-mesh";
export { svoDualMarchingCubesFitWGSL, svoDualMarchingCubesMeshWGSL };
import { svoVoxelMeshWGSL } from "./voxels";
export { svoVoxelMeshWGSL };
import { svoContourMeshWGSL } from "./contours";
import { svoDualContouringMeshWGSL } from "./dual-contouring-mesh";
import { svoDualContouringFitWGSL } from "./dual-contouring";
export { svoContourMeshWGSL, svoDualContouringMeshWGSL, svoDualContouringFitWGSL };
/** Renderer meshing strategies, not application/plugin installations. */
export const SVO_MESHING_PLUGINS = [
  {id:"voxels",label:"Voxel faces",construction:"occupancy", extractionWGSL:svoVoxelMeshWGSL,uniform:false, attachmentBytesPerVoxel:0},
  {id:"contours",label:"Clipped contours",construction:"support-planes", extractionWGSL:svoContourMeshWGSL,uniform:false, attachmentBytesPerVoxel:0},
  {id:"dual-contouring",label:"Hermite DC",construction:"hermite-qef", fitWGSL:svoDualContouringFitWGSL, extractionWGSL:svoDualContouringMeshWGSL,uniform:true, attachmentBytesPerVoxel:16},
  {id:"dual-marching-cubes",label:"Dual Marching Cubes",construction:"function-graph-qef", fitWGSL:svoDualMarchingCubesFitWGSL, extractionWGSL:svoDualMarchingCubesMeshWGSL,uniform:true, attachmentBytesPerVoxel:16},
] as const;
export type SvoMeshingPluginId=typeof SVO_MESHING_PLUGINS[number]["id"];
export function svoMeshingPlugin(id:SvoMeshingPluginId){return SVO_MESHING_PLUGINS.find(p=>p.id===id)!;}
