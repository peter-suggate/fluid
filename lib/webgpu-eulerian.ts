/** Shared contracts for the single hierarchical WebGPU Eulerian implementation. */
export type GPUQuality = "balanced" | "high" | "ultra";

export interface GPUStageTimings {
  advection_ms: number;
  control_ms: number;
  pressure_ms: number;
  projection_ms: number;
  rigidCoupling_ms: number;
  diagnostics_ms: number;
  overhead_ms: number;
  total_ms: number;
}

export interface GPUAdvancePlan {
  elapsed_s: number;
  substeps: number;
  dt_s: number;
}

/** Plan one bounded catch-up submission without allowing an individual CFL step to exceed maxDt. */
export function planGPUAdvance(pending_s: number, maxDt_s: number, stableDt_s: number, maxSubsteps: number): GPUAdvancePlan {
  const stepLimit = Math.max(1e-9, Math.min(maxDt_s, stableDt_s));
  const elapsed_s = Math.max(0, Math.min(pending_s, stepLimit * Math.max(1, maxSubsteps)));
  const substeps = elapsed_s > 0 ? Math.max(1, Math.min(maxSubsteps, Math.ceil(elapsed_s / stepLimit))) : 0;
  return { elapsed_s, substeps, dt_s: substeps > 0 ? elapsed_s / substeps : 0 };
}

/** Convert disjoint WebGPU timestamp pairs without allowing bigint underflow. */
export function decodeGPUStageTimings(times: BigUint64Array, substeps: number): GPUStageTimings | undefined {
  const milliseconds = (begin: bigint, end: bigint) => end >= begin ? Number(end - begin) / 1e6 : 0;
  const stage = [0, 0, 0, 0, 0];
  const timestampsPerSubstep=10,count = Math.max(0, Math.min(substeps, Math.floor((times.length - 2) / timestampsPerSubstep)));
  if (count === 0) return undefined;
  for (let substep = 0; substep < count; substep += 1) {
    const base = substep * timestampsPerSubstep;
    for (let index = 0; index < 5; index += 1) stage[index] += milliseconds(times[base + index * 2], times[base + index * 2 + 1]);
  }
  const diagnostics = milliseconds(times[count * timestampsPerSubstep], times[count * timestampsPerSubstep + 1]);
  const measured = stage.reduce((sum, value) => sum + value, diagnostics);
  const span = milliseconds(times[0], times[count * timestampsPerSubstep + 1]);
  if (!(span > 0) || !Number.isFinite(measured)) return undefined;
  return {
    advection_ms: stage[0], control_ms:stage[1],pressure_ms: stage[2], projection_ms: stage[3], rigidCoupling_ms: stage[4],
    diagnostics_ms: diagnostics, overhead_ms: Math.max(0, span - measured), total_ms: Math.max(span, measured)
  };
}

export interface GPUEulerianInfo {
  nx: number;
  ny: number;
  nz: number;
  cellCount: number;
  cellSize_m: number;
  pressureIterations: number;
  pressureMethod?: "jacobi" | "pcg";
  pressureIterationsExecuted?: number;
  allocatedBytes: number;
  quality: GPUQuality;
  hierarchyLevels?: number;
  activeBrickCount?: number;
  equivalentUniformCells?: number;
  compressionRatio?: number;
  topologySaturated?: boolean;
  divergenceMax_s?: number;
  divergenceBefore_s?: number;
  pressureResidual?: number;
  pressureMax_Pa?: number;
  nanCount?: number;
  topologyRevision?: number;
  regridCount?: number;
  cpuRegrid_ms?: number;
  regridReadbackBytes?: number;
  volumeCellSum?: number;
  front_m?: number;
  maxSpeed_m_s?: number;
  encodedSteps?: number;
  simulatedTime_s?: number;
  queuedSubmissions?: number;
  queueLatency_ms?: number;
  completedSimulationTime_s?: number;
  simulationLag_s?: number;
  simulationThroughput_x?: number;
  blockedFrames?: number;
  cpuCommandEncode_ms?: number;
  cpuQueueSubmit_ms?: number;
  simulationRevision?: number;
  timestampSamplingEnabled?: boolean;
  substepsLast?: number;
  gpuStep_ms?: number;
  initialVolumeCellSum?: number;
  volumeDrift?: number;
  rawVolumeDrift?: number;
  gpuTimings?: GPUStageTimings;
}

export interface GPURigidLoad {
  bodyId: string;
  impulse_N_s: { x: number; y: number; z: number };
  angularImpulse_N_m_s: { x: number; y: number; z: number };
  couplingInterval_s: number;
  displacedVolume_m3: number;
}

const addLoadVector = (a: GPURigidLoad["impulse_N_s"], b: GPURigidLoad["impulse_N_s"]) => ({
  x: a.x + b.x,
  y: a.y + b.y,
  z: a.z + b.z
});

export function mergeGPURigidLoads(current: GPURigidLoad[], incoming: GPURigidLoad[]): GPURigidLoad[] {
  const pending = new Map(current.map((load) => [load.bodyId, load]));
  for (const load of incoming) {
    const previous = pending.get(load.bodyId);
    pending.set(load.bodyId, previous ? {
      ...load,
      impulse_N_s: addLoadVector(previous.impulse_N_s, load.impulse_N_s),
      angularImpulse_N_m_s: addLoadVector(previous.angularImpulse_N_m_s, load.angularImpulse_N_m_s),
      couplingInterval_s: previous.couplingInterval_s + load.couplingInterval_s
    } : load);
  }
  return [...pending.values()];
}

export function consumeGPURigidLoad(load: GPURigidLoad, dt: number) {
  const deliveryTime = Math.max(load.couplingInterval_s, dt);
  const fraction = Math.min(1, dt / deliveryTime);
  const impulse_N_s = {
    x: load.impulse_N_s.x * fraction,
    y: load.impulse_N_s.y * fraction,
    z: load.impulse_N_s.z * fraction
  };
  const angularImpulse_N_m_s = {
    x: load.angularImpulse_N_m_s.x * fraction,
    y: load.angularImpulse_N_m_s.y * fraction,
    z: load.angularImpulse_N_m_s.z * fraction
  };
  load.impulse_N_s = {
    x: load.impulse_N_s.x - impulse_N_s.x,
    y: load.impulse_N_s.y - impulse_N_s.y,
    z: load.impulse_N_s.z - impulse_N_s.z
  };
  load.angularImpulse_N_m_s = {
    x: load.angularImpulse_N_m_s.x - angularImpulse_N_m_s.x,
    y: load.angularImpulse_N_m_s.y - angularImpulse_N_m_s.y,
    z: load.angularImpulse_N_m_s.z - angularImpulse_N_m_s.z
  };
  load.couplingInterval_s = Math.max(0, load.couplingInterval_s - dt);
  return { impulse_N_s, angularImpulse_N_m_s };
}
