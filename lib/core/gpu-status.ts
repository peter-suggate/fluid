import type { ResourcePluginDefinition } from "./resource-plugin";

/**
 * One Dawn command that reproduces a device failure exactly.
 *
 * The identity lives here rather than beside the catalog of reproductions
 * because status is published by code that must not reach the scene library:
 * a status consumer needs to name the reproduction, not construct one.
 */
export interface GPUFailureReproduction {
  readonly caseId:
    | "dam-ui-t0-two-step"
    | "dam-ui-runtime-190-step"
    | "minimal-power-dam-32-t0-two-step"
    | "minimal-power-dam-32-runtime-69-step";
  readonly command:
    | "npm run test:webgpu:dam-ui-two-step"
    | "npm run test:webgpu:dam-ui-runtime"
    | "npm run test:webgpu:minimal-power-dam-32-two-step"
    | "npm run test:webgpu:minimal-power-dam-32-runtime";
  readonly scene: "dam-break-ui" | "minimal-power-dam-break-32";
  readonly grid: readonly [number, number, number];
  readonly steps: 2 | 69 | 190;
  readonly targetTime_s: 0.008 | 0.016 | 0.276 | 1.52;
  readonly maxDt_s: 0.004 | 0.008;
  readonly validated: true;
  readonly isolated: true;
}

/**
 * Device-lifecycle status published by whichever resource currently owns the
 * GPU. Every published arm carries the resource that produced it so the UI can
 * attribute a phase to its owner without re-deriving one from the label text,
 * and readiness refuses a status it cannot attribute. The field is optional
 * only for the pre-start placeholder the diagnostics store boots with, which
 * no owner published and readiness never reduces.
 */
export type GPUStatus =
  | {
      state: "initializing";
      label: string;
      phase?: string;
      completed?: number;
      total?: number;
      startedAt_ms?: number;
      /** Startup has no previous frame; rebuild keeps the prior GPU state visible. */
      kind?: "startup" | "rebuild";
      /** User-facing description captured synchronously when a setting changes. */
      operation?: string;
      retainingPrevious?: boolean;
      resource: ResourcePluginDefinition;
    }
  | { state: "ready"; label: string; adapter: string; resource: ResourcePluginDefinition }
  | { state: "blocked"; label: string; resource: ResourcePluginDefinition }
  | { state: "manual"; label: string; resource: ResourcePluginDefinition }
  | { state: "stopping"; label: string; resource: ResourcePluginDefinition }
  | { state: "unavailable"; label: string; reproduction?: GPUFailureReproduction; resource: ResourcePluginDefinition }
  | { state: "lost"; label: string; resource: ResourcePluginDefinition };
