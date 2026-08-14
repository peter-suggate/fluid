"use client";

import { useRouter } from "next/navigation";
import { SceneOverridesChip } from "./SceneOverridesChip";
import { findSceneDefinition } from "../lib/core/scenes";
import { useSceneStore } from "../lib/core/stores/scene-store";
import { useUIStore } from "../lib/core/stores/ui-store";
import { planSceneRuntime } from "../lib/core/scene-runtime";

/**
 * The only always-visible scene affordances: which scene is loaded, the way back
 * to the library, and the way into this scene's configuration.
 *
 * This was a dropdown over every preset — twenty-five options, of which
 * thirteen were analytic oracles, in a 214 px pill. Choosing what to explore is
 * not a form control, so it moved to the library and this became a chip that
 * names where you are and opens it. That also retires a real hazard: a focused
 * dropdown reads as text editing to the shortcut chassis, so choosing a scene
 * used to silently disable every single-key shortcut until something else took
 * focus, and the fix was a `blur()` someone had to remember. A button cannot
 * hold that state at all.
 */
export function SceneOverlay() {
  const router = useRouter();
  const presetId = useSceneStore((state) => state.presetId);
  const scene = useSceneStore((state) => state.scene);
  const sceneModalOpen = useUIStore((state) => state.sceneModalOpen);
  const setSceneModalOpen = useUIStore((state) => state.setSceneModalOpen);
  const definition = findSceneDefinition(presetId);
  const runtime = planSceneRuntime(scene).fluidSolver ? `seed ${scene.randomSeed}` : "live SVO · no fluid";
  return (
    <div className="scene-overlay" data-testid="scene-panel">
      <span className="brand-mark" title="Fluid Lab · WebGPU CFD workbench">FL</span>
      <button
        type="button"
        className="scene-overlay-chip"
        onClick={() => router.push("/")}
        data-testid="open-scene-library"
        title={definition ? `${definition.blurb}\n\nBrowse all scenes` : "Browse all scenes"}
      >
        <strong>{definition?.name ?? scene.sceneId}</strong>
        <small>{scene.sceneId} · {runtime}</small>
      </button>
      {/* Between the name and CONFIGURE deliberately: it qualifies the scene
          you are looking at, and it is the reason you would open the panel it
          sits next to. Renders nothing at all when the link is clean. */}
      <SceneOverridesChip />
      <button
        className={`scene-overlay-configure${sceneModalOpen ? " active" : ""}`}
        onClick={() => setSceneModalOpen(!sceneModalOpen)}
        data-testid="configure-scene"
        aria-expanded={sceneModalOpen}
        title="Scene, method, container, fluid, numerics, and saved scenes"
      >
        <i aria-hidden="true" />CONFIGURE
      </button>
    </div>
  );
}
