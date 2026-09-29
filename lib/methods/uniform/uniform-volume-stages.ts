import type { GPUTimestampPhase } from "../../core/performance-trace";
/** The Geometric (mixed h/4h) frame's own seams, in encode order, beside the
 * shared UNIFORM_ADVANCE_PHASE seams it also emits (relayout, advection,
 * pressure setup, cycles, projection, rigid coupling, census). Every label is
 * owned by exactly one stage of UNIFORM_VOLUME_PIPELINE. */
export const UNIFORM_VOLUME_PHASE = {
  solids: {id:"other",label:"Solid record + liquid displacement"},
  support: {id:"velocity-extrapolation",label:"Support plan + interface phase"},
  extension: {id:"velocity-extrapolation",label:"Mixed extension sweeps + hierarchy fill"},
  transportReach: {id:"velocity-extrapolation",label:"4h transport reach + sampling caches"},
  phi: {id:"fine-sdf-advection",label:"Vertex phi transport and redistance"},
  coupling: {id:"fine-sdf-advection",label:"Geometric volume coupling"},
  gather: {id:"fine-sdf-advection",label:"Surface volume constraint + geometry"},
  sharpen: {id:"fine-sdf-redistance",label:"Conservative volume sharpening"},
  band: {id:"velocity-projection",label:"h pressure band solve + projection"},
  surface: {id:"surface-extraction",label:"Phi surface publication"},
} as const satisfies Record<string,GPUTimestampPhase>;
