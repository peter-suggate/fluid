import { publicationPort } from "../../framework/ports";
import type { SparseVoxelDrySceneData } from "./scene-publication";
import type { SparseVoxelGBufferTextures } from "../features/primary-visibility/webgpu-svo-gbuffer-targets";

export const SVO_SCENE_PUBLICATION_PORT = publicationPort<SparseVoxelDrySceneData>({
  id: "svo.scene", representation: "svo.scene-payload.v1", lifetime: "generation",
});
export const SVO_GBUFFER_PORT = publicationPort<SparseVoxelGBufferTextures>({
  id: "svo.gbuffer", representation: "svo.compact-gbuffer.v1", lifetime: "frame",
});
