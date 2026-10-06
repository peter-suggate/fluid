/**
 * Transient solver detail input and its accepted diagnostics: the typed seam
 * beside applyRuntimeValues/applySceneUniforms
 * (docs/plans/uniform-4h-first-implementation-handoff-2026-10-03.md, "Runtime wiring and persistence").
 *
 * The focus is per pane (each pane's renderer owns its solver), carried with
 * that pane's frame request, coalesced to its latest revision and consumed at
 * an accepted frame boundary. It is never written to the scene document or URL.
 */
import type { Vec3 } from "./model";

export interface SolverDetailFocus {
  /** Monotonic per pane; a solver ignores a revision it has already seen. */
  readonly revision: number;
  /** The orbit/interaction target in world metres, not the camera eye. */
  readonly position_m: Vec3;
}
export interface SolverDetailInput { readonly focus?: SolverDetailFocus }

/** Coalesce a per-frame focus position into revisions: equal positions keep the revision. */
export function createDetailFocusCoalescer() {
  let last: SolverDetailFocus | undefined;
  return (position_m: Vec3 | undefined): SolverDetailFocus | undefined => {
    if (!position_m || ![position_m.x, position_m.y, position_m.z].every(Number.isFinite)) return last;
    if (last && last.position_m.x === position_m.x && last.position_m.y === position_m.y && last.position_m.z === position_m.z) return last;
    return last = { revision: (last?.revision ?? 0) + 1, position_m: { x: position_m.x, y: position_m.y, z: position_m.z } };
  };
}

/** What the accepted generation's detail plan holds; plain data, posted from the worker. */
export interface SolverDetailDiagnostics {
  readonly policy: string;
  /** uniformDetailRequestKey of the inputs the accepted layout answered. */
  readonly requestKey: string;
  readonly focusRevision?: number;
  readonly tiles: number;
  readonly requestedTiles: number;
  readonly admittedTiles: number;
  readonly supportTiles: number;
  readonly deferredTiles: number;
  readonly budgetClippedTiles: number;
  readonly budgetTiles: number;
  readonly automaticCostTiles: number;
  /** h tiles the existing solver adds for solid/body contact, outside any request. */
  readonly contactTiles: number;
  readonly patchCells: number;
  readonly residentPatches: number;
  readonly allocatedCells: number;
  readonly activeCells: number;
  readonly supportCells: number;
  readonly wastedCells: number;
  readonly reasons: Readonly<Record<"region" | "full" | "focus" | "activity", number>>;
  /** Fine regions whose bounds enforce nothing (lower bound 1, unbounded ceiling). */
  readonly unconstrainedRegions: number;
  /** How this build realises the policy, in a sentence. */
  readonly mapping: string;
  /** The latest request was refused before adoption (the device cannot hold
   * its h tiles): these diagnostics are the generation still running. */
  readonly rejected?: string;
  /** The latest change waits for pipelines no earlier state needed (a first
   * voxel, body or policy): what is being compiled. These diagnostics are
   * the state still advancing; the change is applied when they exist. */
  readonly preparing?: string;
}
