export interface FineLevelSetDispatch2D {
  readonly x: number;
  readonly y: number;
  readonly z: 1;
  readonly workgroups: number;
}

/**
 * Tiles a logically one-dimensional fine-level-set workload over WebGPU's x/y
 * dispatch dimensions. Kernels must flatten workgroup IDs as
 * `wid.x + wid.y * numWorkgroups.x`.
 */
export function planFineLevelSetDispatch2D(
  workgroups: number,
  maximumWorkgroupsPerDimension: number,
): FineLevelSetDispatch2D {
  if (!Number.isSafeInteger(workgroups) || workgroups < 0) {
    throw new RangeError("Fine level-set workgroup count must be a non-negative integer");
  }
  if (!Number.isSafeInteger(maximumWorkgroupsPerDimension) || maximumWorkgroupsPerDimension < 1) {
    throw new RangeError("Fine level-set dispatch limit must be a positive integer");
  }
  if (workgroups === 0) return { x: 0, y: 0, z: 1, workgroups };
  const x = Math.min(workgroups, maximumWorkgroupsPerDimension);
  const y = Math.ceil(workgroups / x);
  if (y > maximumWorkgroupsPerDimension) {
    throw new RangeError(`Fine level-set workload ${workgroups} exceeds the two-dimensional WebGPU dispatch limit`);
  }
  return { x, y, z: 1, workgroups };
}

export const fineLevelSetLinearWorkgroupWGSL = /* wgsl */ `
fn fineLinearWorkgroup(wid:vec3u,numWorkgroups:vec3u)->u32{return wid.x+wid.y*numWorkgroups.x;}
`;
