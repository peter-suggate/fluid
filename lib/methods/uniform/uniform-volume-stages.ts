import type { GPUTimestampPhase } from "../../core/performance-trace";
export const UNIFORM_VOLUME_PHASE = {
  transportReach: {id:"fine-sdf-advection",label:"4h post-extension transport reach"},
  phi: {id:"fine-sdf-advection",label:"Vertex phi transport and redistance"},
  coupling: {id:"fine-sdf-advection",label:"Geometric volume coupling"},
  gather: {id:"fine-sdf-advection",label:"Conservative volume gather"},
  sharpen: {id:"fine-sdf-redistance",label:"Conservative volume sharpening"},
  surface: {id:"surface-extraction",label:"Phi surface publication"},
} as const satisfies Record<string,GPUTimestampPhase>;
