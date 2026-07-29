import type { SceneDiagnosticRuntimeRegistry } from "./scene-diagnostic-runtime";
import {
  sceneCustomHookImplementations,
} from "./scene-custom-diagnostic-implementations";
import { sceneDiagnosticPackImplementations } from "./scene-diagnostic-pack-implementations";

export const sceneDiagnosticRuntimeRegistry: SceneDiagnosticRuntimeRegistry = Object.freeze({
  packs: Object.freeze({
    ...sceneDiagnosticPackImplementations,
  }),
  hooks: sceneCustomHookImplementations,
});
