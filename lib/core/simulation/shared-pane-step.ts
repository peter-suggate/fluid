import type { PaneSession } from "../session/session";
import { effectiveSimulationStep_s } from "../simulation-step";

/** The comparison owns one dt, independent of scene and solver defaults. */
export function sharePaneStep(panes: readonly PaneSession[]) {
  let step_s = effectiveSimulationStep_s(panes[0].scene.getState().scene, panes[0].method.getState());
  let applying = false;
  const apply = () => {
    if (applying) return;
    applying = true;
    try {
      for (const pane of panes) {
        const scene = pane.scene.getState().scene;
        if (scene.numerics.fixedDt_s !== step_s || scene.numerics.maxDt_s !== step_s) {
          pane.scene.getState().patchNumerics({ fixedDt_s: step_s, maxDt_s: step_s });
        }
        const method = pane.method.getState();
        if (effectiveSimulationStep_s(pane.scene.getState().scene, method) !== step_s) {
          method.setParam(method.methodId, "timeStep", "scene");
        }
      }
    } finally {
      applying = false;
    }
  };
  const unsubscribe = panes.flatMap((pane) => [pane.scene.subscribe(apply), pane.method.subscribe(apply)]);
  apply();
  return {
    setStepSize(next: number) { step_s = next; apply(); },
    stop() { unsubscribe.forEach((off) => off()); },
  };
}
