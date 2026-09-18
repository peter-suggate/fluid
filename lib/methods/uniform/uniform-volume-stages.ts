import type { GPUTimestampPhase } from "../../core/performance-trace";
export const UNIFORM_VOLUME_PHASE = {
  phi: {id:"fine-sdf-advection",label:"Dense vertex phi transport and redistance"},
  coupling: {id:"fine-sdf-advection",label:"Dense geometric volume coupling"},
  balance: {id:"fine-sdf-advection",label:"Dense liquid capacity balancing"},
  sharpen: {id:"fine-sdf-redistance",label:"Dense conservative volume sharpening"},
  surface: {id:"surface-extraction",label:"Dense phi surface publication"},
} as const satisfies Record<string,GPUTimestampPhase>;
