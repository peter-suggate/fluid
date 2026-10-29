import { booleanQuery, numberQuery, queryRecord } from "../../framework/persistence";
import { DEFAULT_LENS_BLOOM_OPTIONS } from "../../core/webgpu-lens-bloom";

export interface LensBloomState {
  lensBloomEnabled: boolean;
  lensBloomStrength: number;
  lensBloomThreshold: number;
  lensBloomRadius: number;
}
export const lensBloomQuery=queryRecord<LensBloomState>({
  lensBloomEnabled:booleanQuery("lensBloom",DEFAULT_LENS_BLOOM_OPTIONS.enabled),
  lensBloomStrength:numberQuery("lensBloomStrength",DEFAULT_LENS_BLOOM_OPTIONS.strength,0,1),
  lensBloomThreshold:numberQuery("lensBloomThreshold",DEFAULT_LENS_BLOOM_OPTIONS.threshold,0,64),
  lensBloomRadius:numberQuery("lensBloomRadius",DEFAULT_LENS_BLOOM_OPTIONS.radius,.002,.15),
});
