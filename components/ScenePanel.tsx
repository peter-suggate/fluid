"use client";

import { scenePresets } from "@/lib/scenes";
import { simulation } from "@/lib/simulation/controller";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useUIStore } from "@/lib/stores/ui-store";

const groups = [...new Set(scenePresets.map((preset) => preset.group))];

export function ScenePanel() {
  const presetId = useSceneStore((state) => state.presetId);
  const scene = useSceneStore((state) => state.scene);
  const sceneModalOpen = useUIStore((state) => state.sceneModalOpen);
  const setSceneModalOpen = useUIStore((state) => state.setSceneModalOpen);
  const active = scenePresets.find((preset) => preset.id === presetId);
  return (
    <section className="panel-section scene-title" data-testid="scene-panel">
      <p className="eyebrow">SCENE · SI</p>
      <label className="select-control" title={active?.description}>
        <span className="visually-hidden">Scene preset</span>
        <select aria-label="Scene preset" value={presetId} onChange={(event) => simulation.loadPreset(event.target.value)}>
          {groups.map((group) => (
            <optgroup key={group} label={group}>
              {scenePresets.filter((preset) => preset.group === group).map((preset) => (
                <option key={preset.id} value={preset.id}>{preset.name}</option>
              ))}
            </optgroup>
          ))}
        </select>
      </label>
      <div className="scene-meta"><span>{scene.sceneId}</span><span>seed {scene.randomSeed}</span></div>
      <button className={`drop-button${sceneModalOpen ? " active" : ""}`} onClick={() => setSceneModalOpen(!sceneModalOpen)} data-testid="configure-scene" aria-expanded={sceneModalOpen}>Configure scene…</button>
    </section>
  );
}
