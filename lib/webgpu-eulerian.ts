/** Shared contracts for the single hierarchical WebGPU Eulerian implementation. */
export type GPUQuality = "balanced" | "high" | "ultra";

export interface GPUEulerianInfo {
  nx: number;
  ny: number;
  nz: number;
  cellCount: number;
  cellSize_m: number;
  pressureIterations: number;
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
  substepsLast?: number;
  gpuStep_ms?: number;
  initialVolumeCellSum?: number;
  volumeDrift?: number;
  rawVolumeDrift?: number;
  gpuTimings?: {
    advection_ms: number;
    pressure_ms: number;
    projection_ms: number;
    rigidCoupling_ms: number;
    diagnostics_ms: number;
    overhead_ms: number;
    total_ms: number;
  };
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
