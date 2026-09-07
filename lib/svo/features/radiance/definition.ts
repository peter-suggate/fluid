import type { FeatureDefinition } from "../../../framework/composition";
import { SVO_GBUFFER_PORT } from "../../contracts/ports";

/** Wire values are shared by CPU packing and the fused reconstruction shader. */
export const SVO_CONE_RADIANCE_RECONSTRUCTION_CODES = Object.freeze({
  nearest: 0,
  "gated-linear": 1,
  "joint-bilateral": 2,
  "wide-relight": 3,
  "full-res-relight": 4,
});
export type SvoConeRadianceReconstruction = keyof typeof SVO_CONE_RADIANCE_RECONSTRUCTION_CODES;
export const SVO_RADIANCE_RECONSTRUCTION_OPTIONS = [
  { value: "full-res-relight", label: "RELIGHT", hint: "Full-rate material and BRDF over the reduced visibility cache. Preserves material and edge detail at either reduced rate." },
  { value: "wide-relight", label: "WIDE", hint: "Relight with an unguided wide gather of the reduced cache." },
  { value: "joint-bilateral", label: "BILAT", hint: "Guided upsample weighted by depth, normal and identity." },
  { value: "gated-linear", label: "LINEAR", hint: "Bilinear upsample gated off wherever the guide disagrees." },
  { value: "nearest", label: "EXACT", hint: "Nearest reduced texel; also used when a reconstruction guide fails." },
] as const;

export const SVO_RADIANCE_RECONSTRUCTION_FEATURE = {
  id: "svo.radiance-reconstruction",
  label: "Radiance reconstruction",
  inputs: [{ port: SVO_GBUFFER_PORT, provider: "svo.primary-visibility" }],
  variants: SVO_RADIANCE_RECONSTRUCTION_OPTIONS.map(option => ({
    id: option.value, point: "svo.radiance-reconstruction", label: option.label,
    default: option.value === "full-res-relight", update: "live" as const,
    requires: ["svo.lighting-visibility"], provides: ["svo.reconstructed-radiance"],
  })),
  controls: [{ id: "mode", kind: "choice", label: "Reconstruction", setting: "svoRenderTuning.coneRadianceReconstruction", options: SVO_RADIANCE_RECONSTRUCTION_OPTIONS, update: "live" }],
  placements: [{ slot: "frame.reconstruction", control: "mode", presentation: "expanded" }],
} as const satisfies FeatureDefinition;
