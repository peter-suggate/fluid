import type { EulerianRenderState } from "./eulerian-solver";
import type { SceneDescription } from "./model";
import { cloneRigidBodies, type RigidBodyState } from "./rigid-body";

export const RECORDING_FRAME_INTERVAL_S = 1 / 60;

export interface SimulationRecordingFrame {
  time_s: number;
  bodies: RigidBodyState[];
  fluid: EulerianRenderState;
}

export interface SimulationRecording {
  scene: SceneDescription;
  backend: "webgpu" | "cpu-reference";
  quality: "balanced" | "high" | "ultra";
  frames: SimulationRecordingFrame[];
  duration_s: number;
}

function cloneFluidState(fluid: EulerianRenderState): EulerianRenderState {
  return { ...fluid, occupancy: fluid.occupancy.slice() };
}

function snapshot(time_s: number, bodies: RigidBodyState[], fluid: EulerianRenderState): SimulationRecordingFrame {
  return { time_s, bodies: cloneRigidBodies(bodies), fluid: cloneFluidState(fluid) };
}

export function createSimulationRecording(
  scene: SceneDescription,
  backend: SimulationRecording["backend"],
  quality: SimulationRecording["quality"],
  bodies: RigidBodyState[],
  fluid: EulerianRenderState
): SimulationRecording {
  return {
    scene: structuredClone(scene),
    backend,
    quality,
    frames: [snapshot(0, bodies, fluid)],
    duration_s: 0
  };
}

export function appendSimulationFrame(
  recording: SimulationRecording,
  time_s: number,
  bodies: RigidBodyState[],
  fluid: EulerianRenderState,
  force = false
): boolean {
  const last = recording.frames.at(-1);
  if (!force && last && time_s - last.time_s < RECORDING_FRAME_INTERVAL_S) return false;
  if (last && Math.abs(time_s - last.time_s) < 1e-9) {
    recording.frames[recording.frames.length - 1] = snapshot(time_s, bodies, fluid);
  } else {
    recording.frames.push(snapshot(time_s, bodies, fluid));
  }
  recording.duration_s = Math.max(recording.duration_s, time_s);
  return true;
}

export function simulationFrameAt(recording: SimulationRecording, time_s: number): SimulationRecordingFrame {
  const frames = recording.frames;
  let low = 0;
  let high = frames.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (frames[middle].time_s <= time_s) low = middle;
    else high = middle - 1;
  }
  return frames[low];
}
