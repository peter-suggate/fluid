"use client";

import { scenePresets } from "@/lib/scenes";
import { simulation } from "@/lib/simulation/controller";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { planSceneRuntime } from "@/lib/scene-runtime";

const groups = [...new Set(scenePresets.map((preset) => preset.group))];

/**
 * The only always-visible scene affordances: which preset is loaded, and the
 * way into its configuration. Everything else authored per-scene lives in the
 * configuration panel, so this rides over the viewport instead of costing a
 * sidebar column.
 */
export function SceneOverlay() {
  const presetId = useSceneStore((state) => state.presetId);
  const scene = useSceneStore((state) => state.scene);
  const sceneModalOpen = useUIStore((state) => state.sceneModalOpen);
  const setSceneModalOpen = useUIStore((state) => state.setSceneModalOpen);
  const active = scenePresets.find((preset) => preset.id === presetId);
  const runtime = planSceneRuntime(scene).fluidSolver ? `seed ${scene.randomSeed}` : "static SVO · no fluid";
  return (
    <div className="scene-overlay" data-testid="scene-panel">
      <span className="brand-mark" title="Fluid Lab · WebGPU CFD workbench">FL</span>
      <label className="scene-overlay-preset" title={active?.description}>
        <span className="visually-hidden">Scene preset</span>
        {/* Release focus once the choice is made. A focused <select> counts as
            text editing to the shortcut chassis, so leaving it focused silently
            kills every single-key shortcut — you pick a scene, press a tool
            key, and nothing happens for no visible reason. */}
        <select
          aria-label="Scene preset"
          value={presetId}
          onChange={(event) => { simulation.loadPreset(event.target.value); event.target.blur(); }}
        >
          {groups.map((group) => (
            <optgroup key={group} label={group}>
              {scenePresets.filter((preset) => preset.group === group).map((preset) => (
                <option key={preset.id} value={preset.id}>{preset.name}</option>
              ))}
            </optgroup>
          ))}
        </select>
        <small>{scene.sceneId} · {runtime}</small>
      </label>
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
