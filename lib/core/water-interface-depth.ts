/** Water, particle skin, spray and wireframe share the same depth attachment.
 * Match the SVO camera's infinite reversed-Z convention: no far-plane clamp,
 * and floating-point precision remains useful as the camera pulls back. */
export const WATER_INTERFACE_DEPTH = Object.freeze({
  near_m: .01,
  format: "depth32float" as GPUTextureFormat,
  clearValue: 0,
  compare: "greater" as GPUCompareFunction,
});

export const waterInterfaceDepthWGSL = /* wgsl */ `
const WATER_INTERFACE_NEAR_M:f32=${WATER_INTERFACE_DEPTH.near_m};
`;
