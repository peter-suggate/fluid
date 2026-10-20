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

/** Exclusive parts of the NB-FLIP surface stage, in execution order. */
export const NARROW_BAND_SURFACE_PHASE = {
  move: {id:"fine-sdf-advection",label:"Particle movement + volume budget"},
  advect: {id:"fine-sdf-advection",label:"Grid surface advection"},
  reconstruct: {id:"fine-sdf-advection",label:"Particle surface union"},
  retire: {id:"fine-sdf-redistance",label:"Crossing search + surface retirement"},
  distance: {id:"fine-sdf-redistance",label:"Surface distance + revision publication"},
} as const satisfies Record<string,GPUTimestampPhase>;
